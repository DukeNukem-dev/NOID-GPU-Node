//! amun: the C1 matrix-claim fold (`prove_from_rows`) on the GPU (see `crate::gpu`).
//!
//! Same transcript, same proof: every table, round coefficient and final value is computed exactly as on
//! the CPU (GF(2^256) arithmetic is exact and additions are XORs, so evaluation order cannot change a
//! value). Only the compact (frozen protocol) matrices are offloaded; their CSR and stacked CSC are
//! decoded once and kept on the GPU, keyed by the statement digest.
//!
//! All GPU work that does not depend on the transcript (equality tables, row images, phase-1 weights)
//! runs before the transcript is touched, so any setup failure falls back to the CPU path. A failure
//! after the first transcript write cannot fall back; it disables the GPU fold and panics this proof.
//!
//! `NOID_GPU_FOLD=0` disables it. `NOIDH_MATRIX_FOLD_TIMING` prints a timing line.

use super::{
    C1FreshLincheckClaim, C1MatrixAccClaim, C1MatrixFoldProof, MatrixFoldRows, absorb_fold_header,
    build_eq_table, final_stacked_round_coefficients, lagrange_weights,
};
use crate::challenger::Challenger;
use crate::field::{F128, F256};
use crate::field_r1cs::{CompactFieldR1cs, FieldR1csArtifactMatrix};
use crate::gpu::{self, Breaker, CuFunction, DBuf, DevPtr, Gpu, arg};
use std::collections::HashMap;
use std::time::Instant;

static BREAKER: Breaker = Breaker::new();
const BLOCK: u32 = 256;
const REDUCE_GRID: u64 = 1024;

fn timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NOIDH_MATRIX_FOLD_TIMING").is_some())
}

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NOID_GPU_FOLD").is_ok_and(|v| v == "0"))
}

/// The frozen matrix on the GPU: CSR per side (A, B) and the stacked CSC, sharing one coefficient dictionary.
struct Matrix {
    key: ([u8; 32], usize),
    width: usize,
    dict: DBuf,
    rp: [DBuf; 2],
    col: [DBuf; 2],
    ci: [DBuf; 2],
    cp: DBuf,
    ecol: DBuf,
    srow: DBuf,
    csc_ci: DBuf,
    nnz: usize,
}

struct Kernels {
    eq_step: CuFunction,
    outer: CuFunction,
    spmv_rows: CuFunction,
    colimage: CuFunction,
    colimage_bal: CuFunction,
    fold: CuFunction,
    round_stacked: CuFunction,
    mix_round: CuFunction,
    round_one: CuFunction,
    scale: CuFunction,
}

#[derive(Default)]
struct Work {
    rest: DBuf,
    v: DBuf,
    e: DBuf,
    gv: [DBuf; 2],
    ge: [DBuf; 2],
    inner: DBuf,
    lambda: DBuf,
    zp: DBuf,
    u: [DBuf; 2],
    inc: [DBuf; 2],
    h: [DBuf; 2],
    v2: DBuf,
    eqlo: DBuf,
    eqhi: DBuf,
    partials: DBuf,
}

struct State {
    matrix: Option<Matrix>,
    work: Work,
}

const F: usize = std::mem::size_of::<F256>();

/// Release the large per-proof work tables (the cached matrix stays); other GPU stages need the memory.
fn free_work(g: &Gpu, w: &mut Work) {
    for b in [&mut w.rest, &mut w.v, &mut w.e, &mut w.inner, &mut w.lambda, &mut w.zp, &mut w.v2, &mut w.eqlo, &mut w.eqhi, &mut w.partials] {
        g.free(b);
    }
    for arr in [&mut w.gv, &mut w.ge, &mut w.u, &mut w.inc, &mut w.h] {
        for b in arr.iter_mut() {
            g.free(b);
        }
    }
}

fn kernels(g: &Gpu) -> Result<Kernels, String> {
    Ok(Kernels {
        eq_step: g.function(c"eq_step")?,
        outer: g.function(c"outer")?,
        spmv_rows: g.function(c"spmv_rows")?,
        colimage: g.function(c"colimage")?,
        colimage_bal: g.function(c"colimage_bal")?,
        fold: g.function(c"fold")?,
        round_stacked: g.function(c"round_stacked")?,
        mix_round: g.function(c"mix_round")?,
        round_one: g.function(c"round_one")?,
        scale: g.function(c"scale")?,
    })
}

fn grid(n: u64) -> u64 {
    n.div_ceil(BLOCK as u64).min(1 << 16)
}

/// Decode the compact matrix into CSR (per side) and stacked CSC, upload, free the host copies.
fn upload_matrix(g: &Gpu, r1cs: &CompactFieldR1cs, width: usize) -> Result<Matrix, String> {
    let t0 = Instant::now();
    let mut dict: Vec<F128> = Vec::new();
    let mut dict_index: HashMap<(u64, u64), u32> = HashMap::new();
    let mut triples: [Vec<(u32, u32, u32)>; 2] = [Vec::new(), Vec::new()];
    for (s, side) in [FieldR1csArtifactMatrix::A, FieldR1csArtifactMatrix::B].into_iter().enumerate() {
        for group in 0..r1cs.matrix_group_count(side) {
            let mut bad = false;
            let visited = r1cs.for_each_matrix_group_entry(side, group, |row, column, coefficient| {
                if row >= width || column as usize >= width {
                    bad = true;
                    return;
                }
                let next = dict.len() as u32;
                let ci = *dict_index.entry((coefficient.lo, coefficient.hi)).or_insert_with(|| {
                    dict.push(coefficient);
                    next
                });
                triples[s].push((row as u32, column, ci));
            });
            if !visited || bad {
                return Err(format!("matrix group {group} could not be decoded"));
            }
        }
    }
    let mut m = Matrix {
        key: (r1cs.amun_statement_digest(), width),
        width,
        dict: DBuf::default(),
        rp: Default::default(),
        col: Default::default(),
        ci: Default::default(),
        cp: DBuf::default(),
        ecol: DBuf::default(),
        srow: DBuf::default(),
        csc_ci: DBuf::default(),
        nnz: triples[0].len() + triples[1].len(),
    };
    let d = g.ensure(&mut m.dict, dict.len() * 16)?;
    g.htod(d, &dict)?;
    // CSR per side (counting sort by row; the order within a row does not matter for an XOR sum).
    for s in 0..2 {
        let t = &triples[s];
        let mut rp = vec![0u64; width + 1];
        for &(r, _, _) in t {
            rp[r as usize + 1] += 1;
        }
        for i in 0..width {
            rp[i + 1] += rp[i];
        }
        let mut fill = rp.clone();
        let mut col = vec![0u32; t.len()];
        let mut ci = vec![0u32; t.len()];
        for &(r, c, k) in t {
            let at = fill[r as usize] as usize;
            col[at] = c;
            ci[at] = k;
            fill[r as usize] += 1;
        }
        let p = g.ensure(&mut m.rp[s], rp.len() * 8)?;
        g.htod(p, &rp)?;
        let p = g.ensure(&mut m.col[s], col.len() * 4)?;
        g.htod(p, &col)?;
        let p = g.ensure(&mut m.ci[s], ci.len() * 4)?;
        g.htod(p, &ci)?;
    }
    // Stacked CSC: rows of B are offset by `width`.
    let mut cp = vec![0u64; width + 1];
    for t in &triples {
        for &(_, c, _) in t {
            cp[c as usize + 1] += 1;
        }
    }
    for i in 0..width {
        cp[i + 1] += cp[i];
    }
    let mut fill = cp.clone();
    let mut srow = vec![0u32; m.nnz];
    let mut ci = vec![0u32; m.nnz];
    for (s, t) in triples.iter().enumerate() {
        for &(r, c, k) in t {
            let at = fill[c as usize] as usize;
            srow[at] = r + (s * width) as u32;
            ci[at] = k;
            fill[c as usize] += 1;
        }
    }
    let p = g.ensure(&mut m.cp, cp.len() * 8)?;
    g.htod(p, &cp)?;
    let mut ecol = vec![0u32; m.nnz];
    let mut longest = 0u64;
    for c in 0..width {
        ecol[cp[c] as usize..cp[c + 1] as usize].fill(c as u32);
        longest = longest.max(cp[c + 1] - cp[c]);
    }
    let p = g.ensure(&mut m.ecol, ecol.len() * 4)?;
    g.htod(p, &ecol)?;
    eprintln!("amun gpu fold: longest column {longest} entries");
    let p = g.ensure(&mut m.srow, srow.len() * 4)?;
    g.htod(p, &srow)?;
    let p = g.ensure(&mut m.csc_ci, ci.len() * 4)?;
    g.htod(p, &ci)?;
    g.sync()?;
    eprintln!(
        "amun gpu fold: matrix uploaded (width 2^{}, nnz {}, {} distinct coefficients, {:.0} ms)",
        width.trailing_zeros(),
        m.nnz,
        dict.len(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    Ok(m)
}

/// eq table of `point` into `buf` (2^len entries), same step order as `build_eq_table`.
fn eq_table(g: &Gpu, k: &Kernels, buf: DevPtr, point: &[F256]) -> Result<(), String> {
    g.htod(buf, &[F256::ONE])?;
    for (j, &r) in point.iter().enumerate() {
        let (mut out, mut len, mut r) = (buf, 1u64 << j, r);
        g.launch(k.eq_step, grid(len), BLOCK, &mut [arg!(out), arg!(len), arg!(r)])?;
    }
    Ok(())
}

fn outer(g: &Gpu, k: &Kernels, out: DevPtr, low: DevPtr, low_bits: usize, rest: DevPtr, n: usize) -> Result<(), String> {
    let (mut out, mut low, mut lb, mut rest, mut n) = (out, low, low_bits as u64, rest, n as u64);
    g.launch(k.outer, grid(n), BLOCK, &mut [arg!(out), arg!(low), arg!(lb), arg!(rest), arg!(n)])
}

fn fold(g: &Gpu, k: &Kernels, dst: DevPtr, src: DevPtr, half: usize, ch: F256) -> Result<(), String> {
    let (mut dst, mut src, mut half, mut ch) = (dst, src, half as u64, ch);
    g.launch(k.fold, grid(half), BLOCK, &mut [arg!(dst), arg!(src), arg!(half), arg!(ch)])
}

fn read_partials(g: &Gpu, partials: DevPtr) -> Result<[F256; 2], String> {
    let mut host = vec![F256::ZERO; 2 * REDUCE_GRID as usize];
    g.dtoh(&mut host, partials)?;
    let mut out = [F256::ZERO; 2];
    for pair in host.chunks_exact(2) {
        out[0] += pair[0];
        out[1] += pair[1];
    }
    Ok(out)
}

fn read_one(g: &Gpu, p: DevPtr, index: usize) -> Result<F256, String> {
    let mut v = [F256::ZERO];
    g.dtoh(&mut v, p + (index * F) as u64)?;
    Ok(v[0])
}

/// Everything before the transcript: tables, row images, phase-1 weights (incoming base unscaled).
fn prepare(
    g: &mut Gpu,
    st: &mut State,
    k: &Kernels,
    r1cs: &CompactFieldR1cs,
    k_log: usize,
    k_skip: usize,
    fresh: &C1FreshLincheckClaim,
    incoming: &C1MatrixAccClaim,
) -> Result<(), String> {
    let width = 1usize << k_log;
    let key = (r1cs.amun_statement_digest(), width);
    if st.matrix.as_ref().is_none_or(|m| m.key != key) {
        if let Some(mut old) = st.matrix.take() {
            for b in [&mut old.dict, &mut old.cp, &mut old.ecol, &mut old.srow, &mut old.csc_ci] {
                g.free(b);
            }
            for s in 0..2 {
                g.free(&mut old.rp[s]);
                g.free(&mut old.col[s]);
                g.free(&mut old.ci[s]);
            }
        }
        st.matrix = Some(upload_matrix(g, r1cs, width)?);
    }
    let m = st.matrix.as_ref().unwrap();
    let w = &mut st.work;
    let rest_len = width >> k_skip;
    let (incoming_row, incoming_column) = incoming.point.split_at(k_log + 1);

    let rest = g.ensure(&mut w.rest, rest_len * F)?;
    eq_table(g, k, rest, &fresh.r_inner_rest)?;
    let zp = g.ensure(&mut w.zp, fresh.z_partial.len() * F)?;
    g.htod(zp, &fresh.z_partial)?;
    let v = g.ensure(&mut w.v, width * F)?;
    outer(g, k, v, zp, k_skip, rest, width)?;
    let e = g.ensure(&mut w.e, width * F)?;
    eq_table(g, k, e, incoming_column)?;

    let gv = g.ensure(&mut w.gv[0], 2 * width * F)?;
    let ge = g.ensure(&mut w.ge[0], 2 * width * F)?;
    for s in 0..2 {
        let (mut rp, mut col, mut ci, mut dict) = (m.rp[s].ptr, m.col[s].ptr, m.ci[s].ptr, m.dict.ptr);
        let (mut nrows, mut vv, mut ee) = (width as u64, v, e);
        let (mut gvs, mut ges) = (gv + (s * width * F) as u64, ge + (s * width * F) as u64);
        g.launch(
            k.spmv_rows,
            grid(width as u64),
            BLOCK,
            &mut [arg!(rp), arg!(col), arg!(ci), arg!(dict), arg!(nrows), arg!(vv), arg!(ee), arg!(gvs), arg!(ges)],
        )?;
    }

    let lambda = lagrange_weights(k_skip, fresh.z_skip, 0);
    let lp = g.ensure(&mut w.lambda, lambda.len() * F)?;
    g.htod(lp, &lambda)?;
    let inner = g.ensure(&mut w.inner, rest_len * F)?;
    eq_table(g, k, inner, &fresh.x_inner_rest)?;
    let u = g.ensure(&mut w.u[0], width * F)?;
    outer(g, k, u, lp, k_skip, inner, width)?;
    let inc = g.ensure(&mut w.inc[0], width * F)?;
    eq_table(g, k, inc, &incoming_row[..k_log])?;
    // Ping-pong partners for the folds, and the column image.
    g.ensure(&mut w.gv[1], width * F)?;
    g.ensure(&mut w.ge[1], width * F)?;
    g.ensure(&mut w.u[1], width / 2 * F)?;
    g.ensure(&mut w.inc[1], width / 2 * F)?;
    g.ensure(&mut w.h[0], width * F)?;
    g.ensure(&mut w.h[1], width / 2 * F)?;
    g.ensure(&mut w.v2, width / 2 * F)?;
    g.ensure(&mut w.partials, 2 * REDUCE_GRID as usize * F)?;
    let _ = m.nnz;
    g.sync()
}

type Output = (C1MatrixFoldProof, C1MatrixAccClaim);

/// The transcript part; mirrors `prove_from_rows` step by step.
#[allow(clippy::too_many_arguments)]
fn transcript<Ch: Challenger>(
    g: &mut Gpu,
    st: &mut State,
    k: &Kernels,
    k_log: usize,
    fresh: &C1FreshLincheckClaim,
    incoming: &C1MatrixAccClaim,
    gate: bool,
    channel: &mut Ch,
) -> Result<Output, String> {
    let width = 1usize << k_log;
    let m = st.matrix.as_ref().unwrap();
    let w = &mut st.work;
    let partials = w.partials.ptr;

    let t_start = Instant::now();
    let gate = if gate { F256::ONE } else { F256::ZERO };
    absorb_fold_header(channel, fresh, incoming, gate);
    let gamma = channel.sample_f256();
    let (incoming_row, _) = incoming.point.split_at(k_log + 1);
    let gamma_gate = gamma * gate;
    {
        let (mut out, mut n, mut s) = (w.inc[0].ptr, width as u64, gamma_gate);
        g.launch(k.scale, grid(n), BLOCK, &mut [arg!(out), arg!(n), arg!(s)])?;
    }
    let incoming_stack_point = incoming_row[k_log];
    let alpha = fresh.alpha;

    // Phase 1 (run_phase_stacked).
    let mut claim = fresh.value + gamma_gate * incoming.value;
    let mut phase1_rounds = Vec::with_capacity(k_log + 1);
    let mut rho = Vec::with_capacity(k_log + 1);
    let (mut u, mut u2) = (w.u[0].ptr, w.u[1].ptr);
    let (mut inc, mut inc2) = (w.inc[0].ptr, w.inc[1].ptr);
    let (mut gv, mut gv2) = (w.gv[0].ptr, w.gv[1].ptr);
    let (mut ge, mut ge2) = (w.ge[0].ptr, w.ge[1].ptr);
    let mut side = width;
    while side > 1 {
        {
            let (mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) =
                (u, alpha, gv, inc, incoming_stack_point, ge, side as u64, partials);
            g.launch(
                k.round_stacked,
                REDUCE_GRID,
                BLOCK,
                &mut [arg!(a1), arg!(a2), arg!(a3), arg!(a4), arg!(a5), arg!(a6), arg!(a7), arg!(a8)],
            )?;
        }
        let [constant, quadratic] = read_partials(g, partials)?;
        let linear = claim + quadratic;
        let wire = [constant, quadratic];
        channel.observe_f256_slice(&wire);
        let challenge = channel.sample_f256();
        claim = (quadratic * challenge + linear) * challenge + constant;
        phase1_rounds.push(wire);
        rho.push(challenge);
        fold(g, k, u2, u, side / 2, challenge)?;
        fold(g, k, gv2, gv, side, challenge)?;
        fold(g, k, inc2, inc, side / 2, challenge)?;
        fold(g, k, ge2, ge, side, challenge)?;
        (u, u2, gv, gv2, inc, inc2, ge, ge2) = (u2, u, gv2, gv, inc2, inc, ge2, ge);
        side /= 2;
    }
    let u0 = read_one(g, u, 0)?;
    let inc0 = read_one(g, inc, 0)?;
    let gvf = [read_one(g, gv, 0)?, read_one(g, gv, 1)?];
    let gef = [read_one(g, ge, 0)?, read_one(g, ge, 1)?];
    let [constant, quadratic] = final_stacked_round_coefficients(u0, alpha, &gvf, inc0, incoming_stack_point, &gef);
    let linear = claim + quadratic;
    let wire = [constant, quadratic];
    channel.observe_f256_slice(&wire);
    let challenge = channel.sample_f256();
    claim = (quadratic * challenge + linear) * challenge + constant;
    phase1_rounds.push(wire);
    rho.push(challenge);
    let _ = claim;
    let g_v = gvf[0] + challenge * (gvf[0] + gvf[1]);
    let g_e = gef[0] + challenge * (gef[0] + gef[1]);
    channel.observe_f256(g_v);
    channel.observe_f256(g_e);
    let delta = channel.sample_f256();

    // Column image with the factored equality table of rho.
    let low_bits = rho.len() / 2;
    let eqlo = build_eq_table(&rho[..low_bits]);
    let eqhi = build_eq_table(&rho[low_bits..]);
    let lo = g.ensure(&mut w.eqlo, eqlo.len() * F)?;
    g.htod(lo, &eqlo)?;
    let hi = g.ensure(&mut w.eqhi, eqhi.len() * F)?;
    g.htod(hi, &eqhi)?;
    let (mut h, mut h2) = (w.h[0].ptr, w.h[1].ptr);
    let t_p1 = Instant::now();
    {
        g.zero(h, width * F)?;
        let (mut ecol, mut srow, mut ci, mut dict) = (m.ecol.ptr, m.srow.ptr, m.csc_ci.ptr, m.dict.ptr);
        let (mut nnz, mut lo, mut lb, mut hi, mut out) = (m.nnz as u64, lo, low_bits as u64, hi, h);
        g.launch(
            k.colimage_bal,
            grid((m.nnz as u64).div_ceil(32)),
            BLOCK,
            &mut [arg!(ecol), arg!(srow), arg!(ci), arg!(dict), arg!(nnz), arg!(lo), arg!(lb), arg!(hi), arg!(out)],
        )?;
        let _ = k.colimage;
    }
    if timing() {
        g.sync()?;
        eprintln!("[matrix-fold-c1-gpu] phase1={:.0}ms column-image={:.0}ms", (t_p1 - t_start).as_secs_f64() * 1e3, t_p1.elapsed().as_secs_f64() * 1e3);
    }
    let t_p2 = Instant::now();

    // Phase 2 (mix, then run_phase_one_product); weights = v_table, mixed in place.
    let delta_gate = delta * gate;
    let (mut wts, mut wts2) = (w.v.ptr, w.v2.ptr);
    {
        let (mut a1, mut a2, mut a3, mut a4, mut a5, mut a6) = (wts, w.e.ptr, h, delta_gate, (width / 2) as u64, partials);
        g.launch(
            k.mix_round,
            REDUCE_GRID,
            BLOCK,
            &mut [arg!(a1), arg!(a2), arg!(a3), arg!(a4), arg!(a5), arg!(a6)],
        )?;
    }
    let mut first_round = Some(read_partials(g, partials)?);
    let mut claim = g_v + delta_gate * g_e;
    let mut phase2_rounds = Vec::with_capacity(k_log);
    let mut sigma = Vec::with_capacity(k_log);
    let mut len = width;
    for _ in 0..k_log {
        let [constant, quadratic] = match first_round.take() {
            Some(c) => c,
            None => {
                let (mut a1, mut a2, mut a3, mut a4) = (wts, h, (len / 2) as u64, partials);
                g.launch(k.round_one, REDUCE_GRID, BLOCK, &mut [arg!(a1), arg!(a2), arg!(a3), arg!(a4)])?;
                read_partials(g, partials)?
            }
        };
        let linear = claim + quadratic;
        let wire = [constant, quadratic];
        channel.observe_f256_slice(&wire);
        let challenge = channel.sample_f256();
        claim = (quadratic * challenge + linear) * challenge + constant;
        phase2_rounds.push(wire);
        sigma.push(challenge);
        fold(g, k, wts2, wts, len / 2, challenge)?;
        fold(g, k, h2, h, len / 2, challenge)?;
        (wts, wts2, h, h2) = (wts2, wts, h2, h);
        len /= 2;
    }
    let _ = claim;
    let final_matrix_eval = read_one(g, h, 0)?;
    if timing() {
        eprintln!("[matrix-fold-c1-gpu] phase2={:.0}ms", t_p2.elapsed().as_secs_f64() * 1e3);
    }
    channel.observe_f256(final_matrix_eval);

    let mut point = rho;
    point.extend(sigma);
    Ok((
        C1MatrixFoldProof { phase1_rounds, g_v, g_e, phase2_rounds, final_matrix_eval },
        C1MatrixAccClaim { point, value: final_matrix_eval },
    ))
}

/// The fold on the GPU when possible. `None`: use the CPU path (the transcript was not touched).
pub(super) fn try_prove_from_rows<Ch: Challenger>(
    rows: &MatrixFoldRows<'_>,
    fresh: &C1FreshLincheckClaim,
    incoming: &C1MatrixAccClaim,
    gate: bool,
    channel: &mut Ch,
) -> Option<Output> {
    try_prove_min(rows, fresh, incoming, gate, channel, 16)
}

fn try_prove_min<Ch: Challenger>(
    rows: &MatrixFoldRows<'_>,
    fresh: &C1FreshLincheckClaim,
    incoming: &C1MatrixAccClaim,
    gate: bool,
    channel: &mut Ch,
    min_k_log: usize,
) -> Option<Output> {
    let MatrixFoldRows::Compact(r1cs) = rows else { return None };
    if !enabled() || BREAKER.broken() {
        return None;
    }
    let k_log = rows.k_log();
    let k_skip = rows.k_skip();
    if k_log < min_k_log.max(2)
        || fresh.x_inner_rest.len() != k_log - k_skip
        || fresh.r_inner_rest.len() != k_log - k_skip
        || fresh.z_partial.len() != 1usize << k_skip
        || incoming.point.len() != 2 * k_log + 1
    {
        return None;
    }
    let mut g = gpu::lock()?;
    let timing = std::env::var_os("NOIDH_MATRIX_FOLD_TIMING").is_some();
    let t0 = Instant::now();
    let mut st: Box<State> = match g.fold.take().and_then(|b| b.downcast::<State>().ok()) {
        Some(s) => s,
        None => Box::new(State { matrix: None, work: Work::default() }),
    };
    let k = match kernels(&g) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("amun gpu fold: disabled ({e})");
            BREAKER.trip();
            g.fold = Some(st);
            return None;
        }
    };
    if let Err(e) = prepare(&mut g, &mut st, &k, r1cs, k_log, k_skip, fresh, incoming) {
        eprintln!("amun gpu fold: setup error, using CPU from now on ({e})");
        BREAKER.trip();
        g.fold = Some(st);
        return None;
    }
    let t1 = Instant::now();
    let out = transcript(&mut g, &mut st, &k, k_log, fresh, incoming, gate, channel);
    free_work(&g, &mut st.work);
    g.fold = Some(st);
    match out {
        Ok(out) => {
            if timing {
                eprintln!(
                    "[matrix-fold-c1-gpu] k_log={k_log} prepare={:.0}ms transcript={:.0}ms total={:.0}ms",
                    (t1 - t0).as_secs_f64() * 1e3,
                    t1.elapsed().as_secs_f64() * 1e3,
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            Some(out)
        }
        Err(e) => {
            BREAKER.trip();
            panic!("amun gpu fold: failed after the transcript started ({e}); GPU fold disabled");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::challenger::FsLaneChallenger;
    use crate::field_r1cs::synthetic_satisfiable;

    fn value(seed: u64) -> F256 {
        F256::new(F128::new(seed, seed.rotate_left(7)), F128::new(!seed, seed ^ 0xC1))
    }

    /// Needs a GPU; without one the test only checks the CPU path against itself.
    #[test]
    fn gpu_fold_matches_cpu() {
        for (k_log, gate) in [(6usize, true), (8, true), (8, false), (12, true), (14, true)] {
            let (relation, _) = synthetic_satisfiable(k_log, k_log, 0xC1_6F0 + k_log as u64);
            let rest = k_log - relation.k_skip;
            let mut fresh = C1FreshLincheckClaim {
                alpha: value(1),
                z_skip: value(2),
                x_inner_rest: (0..rest).map(|i| value(10 + i as u64)).collect(),
                r_inner_rest: (0..rest).map(|i| value(20 + i as u64)).collect(),
                z_partial: (0..1usize << relation.k_skip).map(|i| value(100 + i as u64)).collect(),
                value: value(3),
            };
            fresh.value = fresh_claim_value_c1(&relation, &fresh);
            let mut incoming = C1MatrixAccClaim {
                point: (0..2 * k_log + 1).map(|i| value(200 + i as u64)).collect(),
                value: value(4),
            };
            incoming.value = stacked_matrix_mle_eval_c1(&relation, &incoming);
            let shape = FieldShape::of(&relation);
            let digest = relation.structural_statement_digest();
            let mut artifact = Vec::new();
            relation.write_artifact(&mut artifact).unwrap();
            let compact = CompactFieldR1cs::open_packed(artifact.into_boxed_slice(), shape, digest).unwrap();

            let mut cpu_channel = FsLaneChallenger::new_c1(b"amun-gpu-fold-test");
            let cpu = prove_from_rows_cpu(MatrixFoldRows::Compact(&compact), &fresh, &incoming, gate, &mut cpu_channel);
            let mut gpu_channel = FsLaneChallenger::new_c1(b"amun-gpu-fold-test");
            let rows = MatrixFoldRows::Compact(&compact);
            match super::try_prove_min(&rows, &fresh, &incoming, gate, &mut gpu_channel, 0) {
                Some(gpu) => {
                    assert_eq!(gpu.0, cpu.0, "proof differs at k_log={k_log} gate={gate}");
                    assert_eq!(gpu.1, cpu.1, "claim differs at k_log={k_log} gate={gate}");
                    assert_eq!(gpu_channel.sample_f256(), cpu_channel.sample_f256(), "transcript differs");
                    eprintln!("k_log={k_log} gate={gate}: gpu == cpu ({} + {} rounds)", gpu.0.phase1_rounds.len(), gpu.0.phase2_rounds.len());
                }
                None => eprintln!("k_log={k_log}: no GPU, skipped"),
            }
        }
    }
}

/// amun: the lincheck alpha-batched column fold `fold_alpha_batched_quirky_c1` of a compact matrix on the GPU
/// (same matrix copy as the fold). `None`: use the CPU. `NOID_GPU_LINCHECK=0` disables it.
pub(crate) fn try_fold_alpha_quirky(
    r1cs: &CompactFieldR1cs,
    alpha: F256,
    z_skip: F256,
    x_inner_rest: &[F256],
    k_skip: usize,
) -> Option<Vec<F256>> {
    try_fold_alpha_quirky_min(r1cs, alpha, z_skip, x_inner_rest, k_skip, 16)
}

fn lincheck_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NOID_GPU_LINCHECK").is_ok_and(|v| v == "0"))
}

static LINCHECK_BREAKER: Breaker = Breaker::new();

fn try_fold_alpha_quirky_min(
    r1cs: &CompactFieldR1cs,
    alpha: F256,
    z_skip: F256,
    x_inner_rest: &[F256],
    k_skip: usize,
    min_k_log: usize,
) -> Option<Vec<F256>> {
    let width = r1cs.k();
    let k_log = width.trailing_zeros() as usize;
    if !lincheck_enabled() || LINCHECK_BREAKER.broken() || k_log < min_k_log.max(2) || k_skip > k_log
        || x_inner_rest.len() != k_log - k_skip
    {
        return None;
    }
    let mut g = gpu::lock()?;
    let mut st: Box<State> = match g.fold.take().and_then(|b| b.downcast::<State>().ok()) {
        Some(s) => s,
        None => Box::new(State { matrix: None, work: Work::default() }),
    };
    let result = (|| -> Result<Vec<F256>, String> {
        let k = kernels(&g)?;
        let f = g.function(c"lincheck_fold_bal")?;
        let key = (r1cs.amun_statement_digest(), width);
        if st.matrix.as_ref().is_none_or(|m| m.key != key) {
            st.matrix = None;
            st.matrix = Some(upload_matrix(&g, r1cs, width)?);
        }
        let m = st.matrix.as_ref().unwrap();
        let w = &mut st.work;
        let skip = lagrange_weights(k_skip, z_skip, 0);
        let sp = g.ensure(&mut w.lambda, skip.len() * F)?;
        g.htod(sp, &skip)?;
        let rest = g.ensure(&mut w.inner, (width >> k_skip) * F)?;
        eq_table(&g, &k, rest, x_inner_rest)?;
        let comb = g.ensure(&mut w.h[0], width * F)?;
        g.zero(comb, width * F)?;
        let (mut ecol, mut srow, mut ci, mut dict) = (m.ecol.ptr, m.srow.ptr, m.csc_ci.ptr, m.dict.ptr);
        let (mut nnz, mut kl, mut skp, mut ks, mut rst, mut al, mut out) =
            (m.nnz as u64, k_log as u64, sp, k_skip as u64, rest, alpha, comb);
        g.launch(
            f,
            grid((m.nnz as u64).div_ceil(32)),
            BLOCK,
            &mut [arg!(ecol), arg!(srow), arg!(ci), arg!(dict), arg!(nnz), arg!(kl), arg!(skp), arg!(ks), arg!(rst), arg!(al), arg!(out)],
        )?;
        let mut host = vec![F256::ZERO; width];
        g.dtoh(&mut host, comb)?;
        Ok(host)
    })();
    free_work(&g, &mut st.work);
    g.fold = Some(st);
    match result {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("amun gpu lincheck: error, using CPU from now on ({e})");
            LINCHECK_BREAKER.trip();
            None
        }
    }
}

#[cfg(test)]
mod lincheck_tests {
    use crate::field::{F128, F256};
    use crate::field_r1cs::{CompactFieldR1cs, synthetic_satisfiable};
    use crate::lincheck::LincheckCircuit;
    use crate::proof::FieldShape;

    fn value(seed: u64) -> F256 {
        F256::new(F128::new(seed, seed.rotate_left(7)), F128::new(!seed, seed ^ 0xC1))
    }

    /// Needs a GPU; without one only the CPU path runs.
    #[test]
    fn gpu_lincheck_fold_matches_resident() {
        for k_log in [8usize, 12, 14] {
            let (relation, _) = synthetic_satisfiable(k_log, k_log, 0x11C + k_log as u64);
            let mut artifact = Vec::new();
            relation.write_artifact(&mut artifact).unwrap();
            let compact = CompactFieldR1cs::open_packed(
                artifact.into_boxed_slice(),
                FieldShape::of(&relation),
                relation.structural_statement_digest(),
            )
            .unwrap();
            let k_skip = relation.k_skip;
            let rest: Vec<F256> = (0..k_log - k_skip).map(|i| value(40 + i as u64)).collect();
            let (alpha, z) = (value(7), value(9));
            let cpu = relation.fold_alpha_batched_quirky_c1(alpha, z, &rest, k_skip);
            match super::try_fold_alpha_quirky_min(&compact, alpha, z, &rest, k_skip, 0) {
                Some(gpu) => {
                    assert_eq!(gpu, cpu, "lincheck fold differs at k_log={k_log}");
                    eprintln!("k_log={k_log}: gpu lincheck fold == resident cpu fold");
                }
                None => eprintln!("k_log={k_log}: no GPU, skipped"),
            }
        }
    }
}
