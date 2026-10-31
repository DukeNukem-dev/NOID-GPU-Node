//! amun: large C1 column-relation sumchecks (`prove_column_relation`, w_log >= 15) on the GPU (see `crate::gpu`).
//!
//! Same transcript, same proof: column resolution, round interpolation and the transcript stay on the host
//! with the CPU helpers; the GPU evaluates every pair's degree-4 round polynomial at the five nodes (the
//! generic `coeff * prod(factors)` form — the CPU's pivot grouping is only a reordering of the same field sum),
//! and folds the equality and column tables. Resolution and uploads happen before the transcript is touched;
//! failures there fall back to the CPU. A failure after that disables this path and panics the proof.
//! `NOID_GPU_RELATION=0` disables it.

use super::{
    C1ColumnRelationProof, C1RelationTerm, ColRef, RELATION_DEGREE, RELATION_NODES, RelationColumns,
    absorb_relation_header, build_eq_table, claimed_refs, distinct_refs, horner, interpolate_round,
};
use crate::challenger::Challenger;
use crate::field::{F128, F256};
use crate::gpu::{self, Breaker, CuFunction, DBuf, Gpu, arg};
use rayon::prelude::*;

static BREAKER: Breaker = Breaker::new();
const BLOCK: u32 = 128;
const MAX_BLOCKS: u64 = 128;
const F: usize = 32;

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NOID_GPU_RELATION").is_ok_and(|v| v == "0"))
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RTerm {
    coeff: [u64; 4],
    n: u32,
    f: [u32; 3],
}
const _: () = assert!(std::mem::size_of::<RTerm>() == 48);

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FDesc {
    src: u64,
    dst: u64,
}

fn raw(v: F256) -> [u64; 4] {
    [v.lo.lo, v.lo.hi, v.hi.lo, v.hi.hi]
}

#[derive(Default)]
struct State {
    base: Vec<DBuf>,
    a: Vec<DBuf>,
    b: Vec<DBuf>,
    ptrs: DBuf,
    terms: DBuf,
    eq: [DBuf; 2],
    eqf: DBuf,
    partials: DBuf,
    out: DBuf,
    descs: DBuf,
}

struct Kernels {
    first: CuFunction,
    wide: CuFunction,
    reduce: CuFunction,
    fold_first: CuFunction,
    fold_wide: CuFunction,
    fold: CuFunction,
    gather: CuFunction,
}

fn kernels(g: &Gpu) -> Result<Kernels, String> {
    Ok(Kernels {
        first: g.function(c"rel_first")?,
        wide: g.function(c"rel_wide")?,
        reduce: g.function(c"rel_reduce")?,
        fold_first: g.function(c"rel_fold_first")?,
        fold_wide: g.function(c"rel_fold_wide")?,
        fold: g.function(c"fold")?,
        gather: g.function(c"rel_gather0")?,
    })
}

fn grid(n: u64) -> u64 {
    n.div_ceil(BLOCK as u64).clamp(1, 1 << 16)
}

type Output = (C1ColumnRelationProof, Vec<F256>, Vec<F256>);

pub(super) fn try_prove<C: Challenger>(
    target: F256,
    eq_point: &[F256],
    terms: &[C1RelationTerm],
    columns: &RelationColumns<'_>,
    channel: &mut C,
    min_w_log: usize,
) -> Option<Output> {
    let w_log = eq_point.len();
    if !enabled() || BREAKER.broken() || w_log < min_w_log.max(2) || terms.is_empty() {
        return None;
    }
    if terms.iter().any(|t| t.factors.len() > 3) {
        return None;
    }
    let width = 1usize << w_log;
    let references = distinct_refs(terms);
    let claimed = claimed_refs(terms);
    let mut g = gpu::lock()?;
    let mut st: Box<State> = match g.rel.take().and_then(|b| b.downcast::<State>().ok()) {
        Some(s) => s,
        None => Box::default(),
    };
    let r = references.len();
    // Setup (no transcript): resolve, upload, kernels.
    let setup = (|| -> Result<Kernels, String> {
        let k = kernels(&g)?;
        let base_tables: Vec<Vec<F128>> = references.par_iter().map(|&reference| columns.resolve(reference, width)).collect();
        st.base.resize_with(r, DBuf::default);
        st.a.resize_with(r, DBuf::default);
        st.b.resize_with(r, DBuf::default);
        for (i, table) in base_tables.iter().enumerate() {
            let p = g.ensure(&mut st.base[i], width * 16)?;
            g.htod(p, table)?;
            g.ensure(&mut st.a[i], width / 2 * F)?;
            g.ensure(&mut st.b[i], (width / 4).max(1) * F)?;
        }
        let rterms: Vec<RTerm> = terms
            .iter()
            .map(|t| {
                let mut f = [0u32; 3];
                for (j, factor) in t.factors.iter().enumerate() {
                    f[j] = references.iter().position(|x| x == factor).expect("distinct reference") as u32;
                }
                RTerm { coeff: raw(t.coeff), n: t.factors.len() as u32, f }
            })
            .collect();
        let tp = g.ensure(&mut st.terms, rterms.len() * 48)?;
        g.htod(tp, &rterms)?;
        let equality = build_eq_table(&eq_point[1..]);
        let e0 = g.ensure(&mut st.eq[0], equality.len() * F)?;
        g.htod(e0, &equality)?;
        g.ensure(&mut st.eq[1], (equality.len() / 2).max(1) * F)?;
        let eqf: [F256; RELATION_NODES] =
            std::array::from_fn(|node| F256::ONE + eq_point[0] + F256::from_base(F128::new(node as u64, 0)));
        let ef = g.ensure(&mut st.eqf, RELATION_NODES * F)?;
        g.htod(ef, &eqf)?;
        g.ensure(&mut st.ptrs, r * 8)?;
        g.ensure(&mut st.partials, MAX_BLOCKS as usize * RELATION_NODES * F)?;
        g.ensure(&mut st.out, r.max(RELATION_NODES) * F)?;
        g.ensure(&mut st.descs, r * 16)?;
        Ok(k)
    })();
    let k = match setup {
        Ok(k) => k,
        Err(e) => {
            eprintln!("amun gpu relation: setup error, using CPU from now on ({e})");
            BREAKER.trip();
            g.rel = Some(st);
            return None;
        }
    };
    let out = transcript(&g, &mut st, &k, target, eq_point, terms, &references, &claimed, width, channel);
    g.rel = Some(st);
    match out {
        Ok(v) => Some(v),
        Err(e) => {
            BREAKER.trip();
            panic!("amun gpu relation: failed after the transcript started ({e}); GPU relation prover disabled");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn transcript<C: Challenger>(
    g: &Gpu,
    st: &mut State,
    k: &Kernels,
    target: F256,
    eq_point: &[F256],
    terms: &[C1RelationTerm],
    references: &[ColRef],
    claimed: &[ColRef],
    width: usize,
    channel: &mut C,
) -> Result<Output, String> {
    let w_log = eq_point.len();
    let r = references.len();
    absorb_relation_header(channel, target, eq_point, terms);
    let mut claim = target;
    let mut rounds = Vec::with_capacity(w_log);
    let mut point = Vec::with_capacity(w_log);

    let read_evals = |g: &Gpu, blocks: u64| -> Result<[F256; RELATION_NODES], String> {
        let (mut pp, mut nb, mut out) = (st.partials.ptr, blocks, st.out.ptr);
        g.launch(k.reduce, 1, BLOCK, &mut [arg!(pp), arg!(nb), arg!(out)])?;
        let mut ev = [F256::ZERO; RELATION_NODES];
        g.dtoh(&mut ev, st.out.ptr)?;
        Ok(ev)
    };

    // Round 1 on the base tables.
    let ptrs: Vec<u64> = st.base[..r].iter().map(|b| b.ptr).collect();
    g.htod(st.ptrs.ptr, &ptrs)?;
    let pairs = (width / 2) as u64;
    let blocks = pairs.div_ceil(BLOCK as u64).clamp(1, MAX_BLOCKS);
    {
        let (mut tabs, mut tp, mut nt, mut eq, mut pr, mut ef, mut parts) =
            (st.ptrs.ptr, st.terms.ptr, terms.len() as u32, st.eq[0].ptr, pairs, st.eqf.ptr, st.partials.ptr);
        g.launch(k.first, blocks, BLOCK, &mut [arg!(tabs), arg!(tp), arg!(nt), arg!(eq), arg!(pr), arg!(ef), arg!(parts)])?;
    }
    let evaluations = read_evals(g, blocks)?;
    let full = interpolate_round(&evaluations);
    let mut wire = [F256::ZERO; RELATION_DEGREE];
    wire[0] = full[0];
    wire[1..].copy_from_slice(&full[2..]);
    channel.observe_f256_slice(&wire);
    let challenge = channel.sample_f256();
    claim = horner(&full, challenge);
    point.push(challenge);
    rounds.push(wire);
    let equality_prefix = F256::ONE + eq_point[0] + challenge;
    {
        let descs: Vec<FDesc> = (0..r).map(|i| FDesc { src: st.base[i].ptr, dst: st.a[i].ptr }).collect();
        g.htod(st.descs.ptr, &descs)?;
        let (mut dp, mut h, mut ch) = (st.descs.ptr, pairs, challenge);
        g.launch2(k.fold_first, grid(pairs), r as u64, BLOCK, &mut [arg!(dp), arg!(h), arg!(ch)])?;
    }
    let mut in_a = true;
    let mut eq_cur = 0usize;
    let mut len = width / 2;
    for _ in 1..w_log {
        let half = (len / 2) as u64;
        let cur: Vec<u64> = (0..r).map(|i| if in_a { st.a[i].ptr } else { st.b[i].ptr }).collect();
        g.htod(st.ptrs.ptr, &cur)?;
        let blocks = half.div_ceil(BLOCK as u64).clamp(1, MAX_BLOCKS);
        {
            let (mut tabs, mut tp, mut nt, mut eq, mut pr, mut parts) =
                (st.ptrs.ptr, st.terms.ptr, terms.len() as u32, st.eq[eq_cur].ptr, half, st.partials.ptr);
            g.launch(k.wide, blocks, BLOCK, &mut [arg!(tabs), arg!(tp), arg!(nt), arg!(eq), arg!(pr), arg!(parts)])?;
        }
        let mut evaluations = read_evals(g, blocks)?;
        for evaluation in &mut evaluations {
            *evaluation *= equality_prefix;
        }
        let full = interpolate_round(&evaluations);
        let mut wire = [F256::ZERO; RELATION_DEGREE];
        wire[0] = full[0];
        wire[1..].copy_from_slice(&full[2..]);
        channel.observe_f256_slice(&wire);
        let challenge = channel.sample_f256();
        claim = horner(&full, challenge);
        point.push(challenge);
        rounds.push(wire);
        {
            let (mut dst, mut src, mut h, mut ch) = (st.eq[1 - eq_cur].ptr, st.eq[eq_cur].ptr, half, challenge);
            g.launch(k.fold, grid(half), BLOCK, &mut [arg!(dst), arg!(src), arg!(h), arg!(ch)])?;
            let descs: Vec<FDesc> = (0..r)
                .map(|i| {
                    let (s, d) = if in_a { (st.a[i].ptr, st.b[i].ptr) } else { (st.b[i].ptr, st.a[i].ptr) };
                    FDesc { src: s, dst: d }
                })
                .collect();
            g.htod(st.descs.ptr, &descs)?;
            let (mut dp, mut h, mut ch) = (st.descs.ptr, half, challenge);
            g.launch2(k.fold_wide, grid(half), r as u64, BLOCK, &mut [arg!(dp), arg!(h), arg!(ch)])?;
        }
        in_a = !in_a;
        eq_cur = 1 - eq_cur;
        len /= 2;
    }
    let _ = claim;
    let cur: Vec<u64> = (0..r).map(|i| if in_a { st.a[i].ptr } else { st.b[i].ptr }).collect();
    g.htod(st.ptrs.ptr, &cur)?;
    {
        let (mut tabs, mut n, mut out) = (st.ptrs.ptr, r as u64, st.out.ptr);
        g.launch(k.gather, 1, BLOCK, &mut [arg!(tabs), arg!(n), arg!(out)])?;
    }
    let mut values = vec![F256::ZERO; r];
    g.dtoh(&mut values, st.out.ptr)?;
    let final_values = references
        .iter()
        .zip(&values)
        .filter(|(reference, _)| !matches!(reference, ColRef::Fixed(_)))
        .map(|(_, &v)| v)
        .collect::<Vec<_>>();
    debug_assert_eq!(final_values.len(), claimed.len());
    channel.observe_f256_slice(&final_values);
    Ok((C1ColumnRelationProof { rounds, final_values: final_values.clone() }, point, final_values))
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::challenger::FsLaneChallenger;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f128(&mut self) -> F128 {
            F128::new(self.next(), self.next())
        }
        fn f256(&mut self) -> F256 {
            F256::new(self.f128(), self.f128())
        }
    }

    /// Needs a GPU; without one the test only exercises the CPU path.
    #[test]
    fn gpu_relation_matches_cpu() {
        for (w_log, n_terms, seed) in [(10usize, 12usize, 3u64), (15, 20, 5), (16, 40, 7)] {
            let mut rng = Rng(0x5eed ^ seed);
            let width = 1usize << w_log;
            let committed: Vec<Vec<F128>> = (0..6).map(|_| (0..width).map(|_| rng.f128()).collect()).collect();
            let internal: Vec<Vec<F128>> = (0..2).map(|_| (0..width).map(|_| rng.f128()).collect()).collect();
            let committed_refs: Vec<&[F128]> = committed.iter().map(|c| c.as_slice()).collect();
            let internal_refs: Vec<&[F128]> = internal.iter().map(|c| c.as_slice()).collect();
            let columns = RelationColumns { committed: &committed_refs, internal: &internal_refs, fixed: &[] };
            let pick = |rng: &mut Rng| match rng.next() % 4 {
                0 | 1 => ColRef::Committed((rng.next() % 6) as usize),
                2 => ColRef::CommittedShift((rng.next() % 2) as usize),
                _ => ColRef::Internal((rng.next() % 2) as usize),
            };
            let terms: Vec<C1RelationTerm> = (0..n_terms)
                .map(|_| {
                    let n = (rng.next() % 4) as usize;
                    C1RelationTerm { coeff: rng.f256(), factors: (0..n).map(|_| pick(&mut rng)).collect() }
                })
                .collect();
            let eq_point: Vec<F256> = (0..w_log).map(|_| rng.f256()).collect();
            let target = rng.f256();
            let mut a = FsLaneChallenger::new_c1(b"amun-gpu-relation-test");
            let cpu = prove_column_relation_cpu(target, &eq_point, &terms, &columns, &mut a);
            let mut b = FsLaneChallenger::new_c1(b"amun-gpu-relation-test");
            match super::try_prove(target, &eq_point, &terms, &columns, &mut b, 0) {
                Some(gpu) => {
                    assert_eq!(gpu.0, cpu.0, "proof differs at w_log={w_log}");
                    assert_eq!((gpu.1, gpu.2), (cpu.1, cpu.2), "point/values differ at w_log={w_log}");
                    assert_eq!(a.sample_f256(), b.sample_f256(), "transcript differs");
                    eprintln!("w_log={w_log} terms={n_terms}: gpu relation == cpu");
                }
                None => eprintln!("w_log={w_log}: no GPU, skipped"),
            }
        }
    }
}
