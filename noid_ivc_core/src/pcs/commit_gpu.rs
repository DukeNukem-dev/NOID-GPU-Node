//! amun: the PCS commit (interleaved additive NTT + Merkle tree) on the GPU (see `crate::gpu`).
//!
//! The message is uploaded once; the GPU replicates it, runs the NTT layers `log_inv_rate..log_d`
//! (the zero-padded oracle's first layers only copy the message into each replica), builds the Merkle tree
//! straight from the device codeword, and the codeword is copied into the caller's (pre-faulted) host buffer
//! for the later PCS steps. Checks before returning: one full codeword lane is recomputed on the CPU (in
//! parallel with the GPU work) and compared at 64 random positions, and the tree gets the leaf/parent spot check.
//! Any error or mismatch falls back to the CPU commit and disables this path. `NOID_GPU_COMMIT=0` disables it.

use super::{Commitment, PcsParams, ProverData};
use crate::field::F128;
use crate::gpu::{self, Breaker, DBuf, arg};
use crate::merkle::{self, Hash};
use crate::ntt::AdditiveNttF128;
use std::sync::atomic::{AtomicU64, Ordering};

static BREAKER: Breaker = Breaker::new();
static COMMITS: AtomicU64 = AtomicU64::new(0);
const BLOCK: u32 = 256;

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NOID_GPU_COMMIT").is_ok_and(|v| v == "0"))
}

fn grid(n: u64) -> u64 {
    n.div_ceil(BLOCK as u64).clamp(1, 1 << 16)
}

/// The commit on the GPU when possible; `codeword` is the caller's buffer (len = codeword_len_f128).
pub(super) fn try_commit(z_packed: &[F128], mut codeword: Vec<F128>, params: &PcsParams) -> Result<(Commitment, ProverData), Vec<F128>> {
    let num_ntts = params.num_ntts();
    let total = codeword.len();
    if !enabled() || BREAKER.broken() || total < (1 << 16) || total % num_ntts != 0 || z_packed.len() == 0
        || total % z_packed.len() != 0 || !(total / z_packed.len()).is_power_of_two()
    {
        return Err(codeword);
    }
    let log_d = (total / num_ntts).trailing_zeros() as usize;
    let log_rep = (total / z_packed.len()).trailing_zeros() as usize;
    let n_leaves = params.n_leaves();
    let leaf_size = total * 16 / n_leaves;
    if leaf_size % 32 != 0 || 1usize << log_d != total / num_ntts {
        return Err(codeword);
    }
    let ntt = AdditiveNttF128::standard(params.k_code());
    if ntt.log_domain_size() != log_d {
        return Err(codeword);
    }
    let Some(mut g) = gpu::lock() else { return Err(codeword) };
    let mut tree: Vec<Hash> = vec![[0u8; 32]; 2 * n_leaves - 1];
    let k = COMMITS.fetch_add(1, Ordering::Relaxed);
    let mut x = (k.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ total as u64) | 1;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x as usize
    };
    let lane = next() % num_ntts;
    let n = 1usize << log_d;
    let mut bufs = std::mem::take(&mut g.merkle);
    bufs.resize_with(3, DBuf::default);
    // The GPU commit and one CPU reference column (for the check) run at the same time.
    let (run, reference) = std::thread::scope(|scope| {
        let helper = scope.spawn(|| reference_value_column(&ntt, z_packed, num_ntts, log_d, lane));
        let run = (|| -> Result<(), String> {
            let k_rep = g.function(c"ntt_replicate")?;
            let k_layer = g.function(c"ntt_layer")?;
            let mut d = g.ensure(&mut bufs[0], total * 16)?;
            g.htod(d, z_packed)?;
            let (mut ml, mut tt) = (z_packed.len() as u64, total as u64);
            g.launch(k_rep, grid(tt), BLOCK, &mut [arg!(d), arg!(ml), arg!(tt)])?;
            // Twiddle bases per layer: row[j] = twiddle(layer, 1 << j), j < layer.
            let mut rows: Vec<F128> = Vec::new();
            let mut offsets = Vec::new();
            for layer in 0..log_d {
                offsets.push(rows.len());
                for j in 0..layer {
                    rows.push(ntt.twiddle(layer, 1 << j));
                }
            }
            rows.push(F128::ZERO);
            let d_rows = g.ensure(&mut bufs[2], rows.len() * 16)?;
            g.htod(d_rows, &rows)?;
            for layer in log_rep..log_d {
                let (mut pairs, mut nn, mut hl, mut row) = (
                    (total / num_ntts / 2) as u64,
                    num_ntts as u64,
                    (log_d - layer - 1) as u64,
                    d_rows + (offsets[layer] * 16) as u64,
                );
                g.launch(k_layer, grid(pairs * nn), BLOCK, &mut [arg!(d), arg!(pairs), arg!(nn), arg!(hl), arg!(row)])?;
            }
            merkle::gpu::tree_from_device(&g, &mut bufs[1], d, n_leaves, leaf_size, &mut tree)?;
            g.dtoh(&mut codeword, d)
        })();
        (run, helper.join().expect("reference column thread"))
    });
    g.merkle = bufs;
    drop(g);
    if let Err(e) = run {
        if e.contains("CUresult 2)") {
            eprintln!("amun gpu commit: out of GPU memory, this commit on the CPU");
        } else {
            eprintln!("amun gpu commit: error, using CPU from now on ({e})");
            BREAKER.trip();
        }
        return Err(codeword);
    }
    let mut ok = true;
    for _ in 0..64 {
        let pos = next() % n;
        if reference[pos] != codeword[pos * num_ntts + lane] {
            ok = false;
        }
    }
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(codeword.as_ptr() as *const u8, total * 16) };
    if ok && !merkle::gpu::spot_check(bytes, n_leaves, leaf_size, &tree, (k as u64) ^ 0x5eed) {
        ok = false;
    }
    if !ok {
        eprintln!("amun gpu commit: SELF-CHECK MISMATCH, GPU commit disabled");
        BREAKER.trip();
        return Err(codeword);
    }
    let root = *tree.last().expect("tree");
    Ok((Commitment { root, params: params.clone() }, ProverData { codeword, merkle_tree: tree }))
}

/// One lane of the zero-padded forward NTT, computed on the CPU.
fn reference_value_column(ntt: &AdditiveNttF128, msg: &[F128], num_ntts: usize, log_d: usize, lane: usize) -> Vec<F128> {
    let n = 1usize << log_d;
    let msg_pos = msg.len() / num_ntts;
    let mut col: Vec<F128> = (0..n).map(|p| if p < msg_pos { msg[p * num_ntts + lane] } else { F128::ZERO }).collect();
    for layer in 0..log_d {
        let half = 1usize << (log_d - layer - 1);
        for block in 0..(1usize << layer) {
            let t = ntt.twiddle(layer, block);
            let base = block * 2 * half;
            for r in 0..half {
                let (u, v) = (col[base + r], col[base + r + half]);
                let nu = u + v * t;
                col[base + r] = nu;
                col[base + r + half] = v + nu;
            }
        }
    }
    col
}
