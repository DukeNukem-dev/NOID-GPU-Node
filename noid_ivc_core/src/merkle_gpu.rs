//! amun: Merkle trees on the GPU (see `crate::gpu`). Only the fixed-length no-pad leaf mode is offloaded
//! (`leaf_size % 32 == 0`, the production PCS shape). Every GPU tree is spot-checked on the CPU
//! (random leaves and parents); a mismatch or error disables this path for the rest of the process.
//!
//! `NOID_GPU_MIN_LEAVES` (default 1024) sets the smallest offloaded tree.

use super::{Hash, POSEIDON_MERKLE_NODE_TAG, capacity_iv_flat, hash_leaf, hash_pair, leaf_fixed_iv_flat};
use crate::gpu::{self, Breaker, DBuf, Gpu, arg};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

static BREAKER: Breaker = Breaker::new();
static TREES: AtomicU64 = AtomicU64::new(0);

fn min_leaves() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("NOID_GPU_MIN_LEAVES").ok().and_then(|v| v.parse().ok()).unwrap_or(1024)
    })
}

fn split(v: u128) -> (u64, u64) {
    (v as u64, (v >> 64) as u64)
}

fn tree_on(g: &mut Gpu, data: &[u8], num_leaves: usize, leaf_size: usize, tree: &mut [Hash]) -> Result<(), String> {
    let mut bufs = std::mem::take(&mut g.merkle);
    bufs.resize_with(2, DBuf::default);
    let result = (|| {
        let d_data = g.ensure(&mut bufs[0], data.len())?;
        g.htod(d_data, data)?;
        tree_from_device(g, &mut bufs[1], d_data, num_leaves, leaf_size, tree)
    })();
    g.merkle = bufs;
    result
}

/// The tree of `num_leaves` leaves of `leaf_size` bytes already on the device at `d_data`.
pub(crate) fn tree_from_device(
    g: &Gpu,
    tree_buf: &mut DBuf,
    mut d_data: crate::gpu::DevPtr,
    num_leaves: usize,
    leaf_size: usize,
    tree: &mut [Hash],
) -> Result<(), String> {
    const BLOCK: u32 = 128;
    let tree_bytes = tree.len() * 32;
    {
        let d_tree = g.ensure(tree_buf, tree_bytes)?;
        let k_leaves = g.function(c"merkle_leaves")?;
        let k_level = g.function(c"merkle_level")?;

        let [ivh, ivl] = leaf_fixed_iv_flat(leaf_size);
        let ((mut a, mut b), (mut c, mut d)) = (split(ivh), split(ivl));
        let mut ls = leaf_size as u64;
        let mut n = num_leaves as u64;
        let mut d_out = d_tree;
        g.launch(
            k_leaves,
            n.div_ceil(BLOCK as u64),
            BLOCK,
            &mut [arg!(d_data), arg!(ls), arg!(n), arg!(a), arg!(b), arg!(c), arg!(d), arg!(d_out)],
        )?;

        let [nh, nl] = capacity_iv_flat(POSEIDON_MERKLE_NODE_TAG);
        let ((mut a, mut b), (mut c, mut d)) = (split(nh), split(nl));
        let (mut read_start, mut read_len) = (0u64, num_leaves as u64);
        while read_len > 1 {
            let mut n_out = read_len >> 1;
            let mut d_in = d_tree + read_start * 32;
            let mut d_out = d_tree + (read_start + read_len) * 32;
            g.launch(
                k_level,
                n_out.div_ceil(BLOCK as u64),
                BLOCK,
                &mut [arg!(d_in), arg!(n_out), arg!(a), arg!(b), arg!(c), arg!(d), arg!(d_out)],
            )?;
            read_start += read_len;
            read_len = n_out;
        }
        g.sync()?;
        g.dtoh(tree, d_tree)
    }
}

/// Spot-check a GPU tree on the CPU: 8 leaves and 8 parents at pseudo-random positions per call.
pub(crate) fn spot_check(data: &[u8], num_leaves: usize, leaf_size: usize, tree: &[Hash], seed: u64) -> bool {
    let mut x = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..8 {
        let i = (next() as usize) % num_leaves;
        if hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]) != tree[i] {
            return false;
        }
    }
    let total = tree.len();
    for _ in 0..8 {
        let p = num_leaves + (next() as usize) % (total - num_leaves);
        let (mut start, mut len) = (0usize, num_leaves);
        while p >= start + len {
            start += len;
            len >>= 1;
        }
        let below = start - 2 * len;
        let k = p - start;
        if hash_pair(&tree[below + 2 * k], &tree[below + 2 * k + 1]) != tree[p] {
            return false;
        }
    }
    true
}

/// Build the tree on the GPU when possible. `None` means: use the CPU path.
pub(super) fn try_merkle_tree(data: &[u8], num_leaves: usize, tree: &mut [Hash]) -> Option<()> {
    let leaf_size = data.len() / num_leaves;
    if num_leaves < min_leaves() || leaf_size == 0 || leaf_size % 32 != 0 || BREAKER.broken() {
        return None;
    }
    let mut g = gpu::lock()?;
    if let Err(e) = tree_on(&mut g, data, num_leaves, leaf_size, tree) {
        eprintln!("amun gpu merkle: error, using CPU from now on ({e})");
        BREAKER.trip();
        return None;
    }
    drop(g);
    let k = TREES.fetch_add(1, Ordering::Relaxed);
    if !spot_check(data, num_leaves, leaf_size, tree, k.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ data.len() as u64) {
        eprintln!("amun gpu merkle: SELF-CHECK MISMATCH (n_leaves={num_leaves} leaf_size={leaf_size}), GPU disabled");
        BREAKER.trip();
        return None;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::super::{merkle_tree, merkle_tree_sequential};

    fn data(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    /// Needs a GPU; without one the CPU path runs and the comparison is trivially equal.
    #[test]
    fn gpu_tree_matches_sequential() {
        for &(n, leaf) in &[(1024usize, 32usize), (1024, 512), (4096, 64), (65536, 512), (2048, 4096)] {
            let d = data(n * leaf, (n * 31 + leaf) as u64);
            let mut t = vec![[0u8; 32]; 2 * n - 1];
            let used = super::try_merkle_tree(&d, n, &mut t).is_some();
            let cpu = merkle_tree_sequential(&d, n);
            if used {
                assert_eq!(t, cpu, "gpu tree differs: n={n} leaf={leaf}");
            }
            assert_eq!(merkle_tree(&d, n), cpu, "merkle_tree differs: n={n} leaf={leaf}");
            eprintln!("n={n} leaf={leaf} gpu={used}");
        }
    }
}
