//! amun: the C1 ragged deep-chain walk (`prove_ragged_deep_chain_walk`) on the GPU (see `crate::gpu`).
//!
//! Same transcript, same proof. The GPU holds each child's layer states (checkpoints every 8 rounds plus one
//! replayed window, as `RaggedDescendingLayerStates`), the equality and folded state tables, and computes
//! every pair sum of the sumcheck rounds (the generic relation formula; the CPU's shortcut paths give the
//! same field values). The host keeps the transcript and the per-child bookkeeping, using the CPU helpers.
//! Children whose tables are exhausted (one entry left) continue on the host exactly as on the CPU.
//!
//! GPU setup (upload, checkpoints) happens before the transcript is touched, so setup failures fall back to
//! the CPU. A failure after that disables the GPU walk and panics this proof.
//!
//! `NOID_GPU_WALK=0` disables it. `NOIDH_C1_DEEP_CHAIN_TIMING` prints a timing line.

use super::{
    C1LaneClaimGroup, C1MultiDeepChainWalkProof, C1MultiWalkLayerProof, absorb_groups, column_weights,
    compress, convolve_equality_factor, exhausted_base_round_contribution, horner, lane_weight_table,
    scale_contribution, wide_relation_coefficients,
};
use super::{flat_mds, flat_round_constant, is_full_round, WALK_DEGREE};
use crate::challenger::Challenger;
use crate::field::{F128, F256};
use crate::gpu::{self, Breaker, CuFunction, DBuf, DevPtr, Gpu, arg};
use noid_poseidon2b::native::permutation::{N_ROUNDS, STATE_SIZE};
use std::time::Instant;

static BREAKER: Breaker = Breaker::new();
const BLOCK: u32 = 128;
const MAX_REL_BLOCKS: u64 = 128;
const SPACING: usize = 8;

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NOID_GPU_WALK").is_ok_and(|v| v == "0"))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct C4([F256; STATE_SIZE]);
#[repr(C)]
#[derive(Clone, Copy)]
struct Rc4([F128; STATE_SIZE]);

struct Kernels {
    checkpoints: CuFunction,
    window: CuFunction,
    rel_first: CuFunction,
    rel_wide: CuFunction,
    reduce: CuFunction,
    fold_base: CuFunction,
    fold4: CuFunction,
    fold: CuFunction,
    eq_step: CuFunction,
}

fn kernels(g: &Gpu) -> Result<Kernels, String> {
    Ok(Kernels {
        checkpoints: g.function(c"dw_checkpoints")?,
        window: g.function(c"dw_window")?,
        rel_first: g.function(c"dw_rel_first")?,
        rel_wide: g.function(c"dw_rel_wide")?,
        reduce: g.function(c"dw_reduce_partials")?,
        fold_base: g.function(c"dw_fold_base")?,
        fold4: g.function(c"dw_fold4")?,
        fold: g.function(c"fold")?,
        eq_step: g.function(c"eq_step")?,
    })
}

const ROW: usize = 4 * 16; // [F128; 4]
const FROW: usize = 4 * 32; // [F256; 4]
const F: usize = 32;

#[derive(Default)]
struct Child {
    ckpt: DBuf,
    window: DBuf,
    eq: [DBuf; 2],
    st: [DBuf; 2],
    partials: DBuf,
}

#[derive(Default)]
struct State {
    consts: DBuf,
    children: Vec<Child>,
    results: DBuf,
}

fn grid(n: u64) -> u64 {
    n.div_ceil(BLOCK as u64).clamp(1, 1 << 16)
}

fn rel_blocks(pairs: u64) -> u64 {
    pairs.div_ceil(BLOCK as u64).clamp(1, MAX_REL_BLOCKS)
}

fn read<T: Copy + Default>(g: &Gpu, p: DevPtr) -> Result<T, String> {
    let mut v = [T::default()];
    g.dtoh(&mut v, p)?;
    Ok(v[0])
}

fn eq_table(g: &Gpu, k: &Kernels, buf: DevPtr, point: &[F256]) -> Result<(), String> {
    g.htod(buf, &[F256::ONE])?;
    for (j, &r) in point.iter().enumerate() {
        let (mut out, mut len, mut r) = (buf, 1u64 << j, r);
        g.launch(k.eq_step, grid(len), BLOCK, &mut [arg!(out), arg!(len), arg!(r)])?;
    }
    Ok(())
}

type Output = (C1MultiDeepChainWalkProof, Vec<C1LaneClaimGroup>);

/// Upload the S_0 rows and build the checkpoints (no transcript use).
fn prepare(g: &mut Gpu, st: &mut State, k: &Kernels, base_states: &[&[Vec<F128>; STATE_SIZE]]) -> Result<(), String> {
    // Constants: rc[lane][q] then MDS full and partial, row-major.
    let mut consts: Vec<F128> = Vec::with_capacity(STATE_SIZE * N_ROUNDS + 32);
    for lane in 0..STATE_SIZE {
        for q in 0..N_ROUNDS {
            consts.push(flat_round_constant(lane, q));
        }
    }
    for full in [true, false] {
        for row in flat_mds(full) {
            consts.extend_from_slice(row);
        }
    }
    let cp = g.ensure(&mut st.consts, consts.len() * 16)?;
    g.htod(cp, &consts)?;
    st.children.resize_with(base_states.len(), Child::default);
    let n_ckpt = (N_ROUNDS - 1) / SPACING + 1;
    for (child, states) in base_states.iter().enumerate() {
        let n = states[0].len();
        let rows: Vec<[F128; STATE_SIZE]> = (0..n).map(|i| std::array::from_fn(|l| states[l][i])).collect();
        let c = &mut st.children[child];
        let ckpt = g.ensure(&mut c.ckpt, n_ckpt * n * ROW)?;
        let window = g.ensure(&mut c.window, SPACING * n * ROW)?;
        // Stage S_0 in the window buffer, then expand into the checkpoints.
        g.htod(window, &rows)?;
        let half = (n / 2).max(1);
        g.ensure(&mut c.eq[0], half * F)?;
        g.ensure(&mut c.eq[1], (half / 2).max(1) * F)?;
        g.ensure(&mut c.st[0], half * FROW)?;
        g.ensure(&mut c.st[1], (half / 2).max(1) * FROW)?;
        g.ensure(&mut c.partials, MAX_REL_BLOCKS as usize * 8 * F)?;
        let (mut s0, mut nn, mut nc, mut rc, mut mds, mut out) =
            (window, n as u64, n_ckpt as u64, cp, cp + (STATE_SIZE * N_ROUNDS * 16) as u64, ckpt);
        g.launch(k.checkpoints, grid(nn), BLOCK, &mut [arg!(s0), arg!(nn), arg!(nc), arg!(rc), arg!(mds), arg!(out)])?;
    }
    g.ensure(&mut st.results, base_states.len() * 8 * F)?;
    g.sync()
}

/// Per-child host view: either GPU tables (len > 1) or the exhausted single entry.
struct Track {
    w: usize,
    n: usize,
    eq_cur: usize,
    st_cur: usize,
    len: usize,
    exhausted: Option<(F256, [F256; STATE_SIZE])>,
}

#[allow(clippy::too_many_lines)]
fn transcript<C: Challenger>(
    g: &mut Gpu,
    st: &mut State,
    k: &Kernels,
    base_states: &[&[Vec<F128>; STATE_SIZE]],
    w_logs: &[usize],
    output_groups: &[C1LaneClaimGroup],
    channel: &mut C,
) -> Result<Output, String> {
    let n_children = base_states.len();
    let max_w_log = *w_logs.iter().max().unwrap();
    let consts = st.consts.ptr;
    let mds = consts + (STATE_SIZE * N_ROUNDS * 16) as u64;
    let results = st.results.ptr;

    absorb_groups(channel, w_logs, output_groups);
    let mut groups = output_groups.to_vec();
    let mut layers = Vec::with_capacity(N_ROUNDS);
    let mut window_base = usize::MAX;

    for layer in (1..=N_ROUNDS).rev() {
        let round = layer - 1;
        let full = is_full_round(round);
        let rc = Rc4(std::array::from_fn(|l| flat_round_constant(l, round)));
        let alpha = channel.sample_f256();
        let weights = lane_weight_table(alpha, groups.len());
        let columns = weights.iter().map(|w| column_weights(round, w)).collect::<Vec<_>>();
        let mut claim = groups.iter().zip(&weights).fold(F256::ZERO, |sum, (group, weights)| {
            sum + (0..STATE_SIZE).fold(F256::ZERO, |inner, lane| inner + weights[lane] * group.values[lane])
        });

        // Layer states S_round: replay the window from its checkpoint when the window changes.
        let base = (round / SPACING) * SPACING;
        if window_base != base {
            let len = SPACING.min(N_ROUNDS - base);
            for (child, states) in base_states.iter().enumerate() {
                let n = states[0].len();
                let c = &st.children[child];
                let (mut ck, mut nn, mut b, mut l, mut rcp, mut m, mut win) =
                    (c.ckpt.ptr + ((base / SPACING) * n * ROW) as u64, n as u64, base as u64, len as u64, consts, mds, c.window.ptr);
                g.launch(k.window, grid(nn), BLOCK, &mut [arg!(ck), arg!(nn), arg!(b), arg!(l), arg!(rcp), arg!(m), arg!(win)])?;
            }
            window_base = base;
        }

        let mut tracks = Vec::with_capacity(n_children);
        let mut prefix = vec![F256::ONE; n_children];
        for (child, states) in base_states.iter().enumerate() {
            let n = states[0].len();
            let w = w_logs[child];
            let c = &st.children[child];
            let s_ptr = c.window.ptr + ((round - base) * n * ROW) as u64;
            if w == 0 {
                let s0: [F128; STATE_SIZE] = read(g, s_ptr)?;
                tracks.push(Track { w, n, eq_cur: 0, st_cur: 0, len: 1, exhausted: Some((F256::ONE, s0.map(F256::from_base))) });
                // Coordinate 0 uses the base-field state; keep it via the exhausted slot's F128 image.
            } else {
                eq_table(g, k, c.eq[0].ptr, &groups[child].point[1..])?;
                tracks.push(Track { w, n, eq_cur: 0, st_cur: 0, len: n, exhausted: None });
            }
        }

        let mut round_coeffs = Vec::with_capacity(max_w_log);
        let mut point = Vec::with_capacity(max_w_log);
        for coordinate in 0..max_w_log {
            // GPU pair sums for every child that still has tables.
            let mut launched = vec![false; n_children];
            for (child, t) in tracks.iter().enumerate() {
                if t.exhausted.is_some() {
                    continue;
                }
                let c = &st.children[child];
                let mut cols = C4(columns[child]);
                let mut rcv = rc;
                let mut fl: i32 = i32::from(full);
                let mut parts = c.partials.ptr;
                let pairs = (t.len / 2) as u64;
                let blocks = rel_blocks(pairs);
                let mut pr = pairs;
                if coordinate == 0 {
                    let (mut eqp, mut sp) = (c.eq[0].ptr, c.window.ptr + ((round - base) * t.n * ROW) as u64);
                    g.launch(k.rel_first, blocks, BLOCK, &mut [arg!(eqp), arg!(sp), arg!(pr), arg!(cols), arg!(rcv), arg!(fl), arg!(parts)])?;
                } else {
                    let (mut eqp, mut sp) = (c.eq[t.eq_cur].ptr, c.st[t.st_cur].ptr);
                    g.launch(k.rel_wide, blocks, BLOCK, &mut [arg!(eqp), arg!(sp), arg!(pr), arg!(cols), arg!(rcv), arg!(fl), arg!(parts)])?;
                }
                let (mut pp, mut nb, mut out) = (parts, blocks, results + (child * 8 * F) as u64);
                g.launch(k.reduce, 1, BLOCK, &mut [arg!(pp), arg!(nb), arg!(out)])?;
                launched[child] = true;
            }
            let mut sums = vec![[F256::ZERO; WALK_DEGREE]; n_children];
            if launched.iter().any(|&x| x) {
                g.dtoh(&mut sums, results)?;
            }

            let mut full_poly = [F256::ZERO; WALK_DEGREE + 1];
            for (child, t) in tracks.iter().enumerate() {
                let contribution = if coordinate == 0 {
                    if t.w > 0 {
                        convolve_equality_factor(sums[child], F256::ONE + groups[child].point[0])
                    } else {
                        let (eq0, s) = t.exhausted.unwrap();
                        let base_state: [F128; STATE_SIZE] = s.map(|v| v.lo);
                        exhausted_base_round_contribution(round, eq0, &columns[child], &base_state)
                    }
                } else if t.exhausted.is_none() {
                    scale_contribution(
                        convolve_equality_factor(sums[child], F256::ONE + groups[child].point[coordinate]),
                        prefix[child],
                    )
                } else {
                    let (eq0, s) = t.exhausted.unwrap();
                    let relation = wide_relation_coefficients(round, &columns[child], &s, &[F256::ZERO; STATE_SIZE]);
                    let relation_sum: [F256; WALK_DEGREE] = std::array::from_fn(|d| eq0 * relation[d]);
                    scale_contribution(convolve_equality_factor(relation_sum, F256::ONE), prefix[child])
                };
                for (coefficient, c) in full_poly.iter_mut().zip(contribution) {
                    *coefficient += c;
                }
            }
            let wire = compress(&full_poly);
            channel.observe_f256_slice(&wire);
            let challenge = channel.sample_f256();
            claim = horner(&full_poly, challenge);
            point.push(challenge);
            round_coeffs.push(wire);

            // Folds.
            for (child, t) in tracks.iter_mut().enumerate() {
                let c = &st.children[child];
                if let Some((eq0, s)) = t.exhausted.as_mut() {
                    *eq0 *= F256::ONE + challenge;
                    let _ = s;
                    continue;
                }
                let mut ch = challenge;
                let half = (t.len / 2) as u64;
                let mut h = half;
                if coordinate == 0 {
                    let (mut dst, mut src) = (c.st[0].ptr, c.window.ptr + ((round - base) * t.n * ROW) as u64);
                    g.launch(k.fold_base, grid(half), BLOCK, &mut [arg!(dst), arg!(src), arg!(h), arg!(ch)])?;
                    t.st_cur = 0;
                    prefix[child] = F256::ONE + groups[child].point[0] + challenge;
                } else {
                    let (mut dst, mut src) = (c.eq[1 - t.eq_cur].ptr, c.eq[t.eq_cur].ptr);
                    g.launch(k.fold, grid(half), BLOCK, &mut [arg!(dst), arg!(src), arg!(h), arg!(ch)])?;
                    let (mut dst, mut src) = (c.st[1 - t.st_cur].ptr, c.st[t.st_cur].ptr);
                    g.launch(k.fold4, grid(half), BLOCK, &mut [arg!(dst), arg!(src), arg!(h), arg!(ch)])?;
                    t.eq_cur = 1 - t.eq_cur;
                    t.st_cur = 1 - t.st_cur;
                }
                t.len = half as usize;
                if t.len == 1 {
                    let eq0: F256 = read(g, c.eq[t.eq_cur].ptr)?;
                    let s0: [F256; STATE_SIZE] = read(g, c.st[t.st_cur].ptr)?;
                    t.exhausted = Some((eq0, s0));
                }
            }
        }
        let _ = claim;

        let next_values = tracks.iter().map(|t| t.exhausted.expect("every child exhausted").1).collect::<Vec<_>>();
        channel.observe_f256_slice(&next_values.iter().flat_map(|v| v.iter().copied()).collect::<Vec<_>>());
        layers.push(C1MultiWalkLayerProof { round_coeffs, next_values: next_values.clone() });
        groups = next_values
            .into_iter()
            .enumerate()
            .map(|(child, values)| C1LaneClaimGroup { point: point[..w_logs[child]].to_vec(), values })
            .collect();
    }
    Ok((C1MultiDeepChainWalkProof { layers }, groups))
}

/// The walk on the GPU when possible. `None`: use the CPU path (the transcript was not touched).
pub(super) fn try_prove<C: Challenger>(
    base_states: &[&[Vec<F128>; STATE_SIZE]],
    output_groups: &[C1LaneClaimGroup],
    channel: &mut C,
    min_w_log: usize,
) -> Option<Output> {
    if !enabled() || BREAKER.broken() || base_states.is_empty() || base_states.len() != output_groups.len() {
        return None;
    }
    if N_ROUNDS != 66 || (0..N_ROUNDS).any(|q| is_full_round(q) != !(4..62).contains(&q)) {
        return None; // the kernels hard-code the 8 + 58 round schedule
    }
    let mut w_logs = Vec::with_capacity(base_states.len());
    for (states, group) in base_states.iter().zip(output_groups) {
        let width = states[0].len();
        if !width.is_power_of_two() || states.iter().any(|c| c.len() != width) {
            return None;
        }
        let w = width.trailing_zeros() as usize;
        if group.point.len() != w {
            return None;
        }
        w_logs.push(w);
    }
    if *w_logs.iter().max().unwrap() < min_w_log {
        return None;
    }
    let mut g = gpu::lock()?;
    let timing = std::env::var_os("NOIDH_C1_DEEP_CHAIN_TIMING").is_some();
    let t0 = Instant::now();
    let mut st: Box<State> = match g.walk.take().and_then(|b| b.downcast::<State>().ok()) {
        Some(s) => s,
        None => Box::default(),
    };
    let k = match kernels(&g) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("amun gpu walk: disabled ({e})");
            BREAKER.trip();
            g.walk = Some(st);
            return None;
        }
    };
    if let Err(e) = prepare(&mut g, &mut st, &k, base_states) {
        eprintln!("amun gpu walk: setup error, using CPU from now on ({e})");
        BREAKER.trip();
        g.walk = Some(st);
        return None;
    }
    let t1 = Instant::now();
    let out = transcript(&mut g, &mut st, &k, base_states, &w_logs, output_groups, channel);
    g.walk = Some(st);
    match out {
        Ok(out) => {
            if timing {
                eprintln!(
                    "NOIDH C1 deep-chain-ragged-gpu widths={w_logs:?} prepare_ms={:.0} transcript_ms={:.0} total_ms={:.0}",
                    (t1 - t0).as_secs_f64() * 1e3,
                    t1.elapsed().as_secs_f64() * 1e3,
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            Some(out)
        }
        Err(e) => {
            BREAKER.trip();
            panic!("amun gpu walk: failed after the transcript started ({e}); GPU walk disabled");
        }
    }
}
