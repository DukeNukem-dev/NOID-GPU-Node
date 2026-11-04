// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Paranoid Zero.

use super::*;
use parent::{prepare_parent, PreparedParent};

pub struct Built {
    class: Class,
    matrix: HistoryStepMatrixLease,
    witness: Vec<F128>,
    io: Vec<F128>,
    preparations: Preparations,
}

struct Assembly {
    class: Class,
    builder: FieldR1csBuilder,
    block: SelectedZkBlockSlotsAssembly,
    parent: HistoryStepParentRegionPreparation,
    io_seal: DeferredHistoryStepIo,
    io: Vec<F128>,
    matrix: Option<HistoryStepMatrixLease>,
}

fn parent_selector(
    builder: &mut FieldR1csBuilder,
    authenticated: &LinExpr,
    class: Class,
) -> [LinExpr; 3] {
    let selectors = Class::ALL.map(|arm| LinExpr::from_wire(builder.alloc_bool(class == arm)));
    let mut total = LinExpr::zero();
    let mut code = LinExpr::zero();
    for (index, selector) in selectors.iter().enumerate() {
        total = total.add(selector);
        code = code.add(&selector.scale(f128_from_u128(index as u128)));
        for other in &selectors[..index] {
            let both = mul(builder, selector, other);
            pin_eq(builder, &both, &LinExpr::zero());
        }
    }
    pin_eq(builder, &total, &LinExpr::constant(F128::ONE));
    pin_eq(builder, authenticated, &code);
    selectors
}

/// Exact carry is part of the relation, including a previously used large
/// matrix after returning to small blocks. The inactive fold has no authority.
fn bind_lane(
    builder: &mut FieldR1csBuilder,
    cells: &[LinExpr],
    prev: &[LinExpr],
    lane: Lane,
    gate: &LinExpr,
    folded: &C1MatrixAccClaimTrace,
) {
    let outputs = folded
        .point
        .iter()
        .flat_map(|p| [&p.lo, &p.hi])
        .chain([&folded.value.lo, &folded.value.hi]);
    for (offset, value) in (lane.point..lane.live).zip(outputs) {
        let delta = mul(builder, gate, &value.add(&prev[offset]));
        pin_eq(builder, &cells[offset], &prev[offset].add(&delta));
    }
    let delta = mul(builder, gate, &prev[lane.live].add_const(F128::ONE));
    pin_eq(builder, &cells[lane.live], &prev[lane.live].add(&delta));
}

fn prepare_assembly<const TIER: usize>(
    runtime: &Runtime,
    prepared: PreparedParent<'_>,
    current: HistoryStepBlockInput<TIER>,
    frozen: bool,
    record_matrix_rows: bool,
) -> Result<Assembly, V2Error> {
    let PreparedParent {
        selected,
        current: class,
        envelopes,
        scratch,
        folds,
        io,
    } = prepared;
    let config = runtime.bank.config().class(class);
    if config.pages() != TIER {
        return Err(V2Error::Runtime);
    }
    let matrix = if frozen {
        None
    } else {
        Some(runtime.load_matrix(class)?)
    };
    let mut builder = if record_matrix_rows {
        FieldR1csBuilder::new()
    } else {
        FieldR1csBuilder::new_witness_only()
    };
    if std::env::var_os("NOID_PRODUCT_OBSERVATIONS").is_some() {
        builder.observe_symbolic_products(65536);
    }
    if runtime.bank.config().product_aliases() {
        builder.enable_symbolic_product_aliases();
    }
    if runtime.bank.config().reuses_linear_rows() {
        builder.enable_reused_linear_rows();
    }
    if runtime.bank.config().fused_pins() {
        builder.enable_fused_boolean_pins();
    }
    let mut row_ledger = 0;
    crate::acceptance::section_timer_start();
    let (cells, io_seal) =
        allocate_deferred_history_step_io(&mut builder, &runtime.bank.config().io_spec(), &io, ACC);
    if runtime.bank.config().activation_height() == 1 {
        let genesis = noid_chain::consensus::genesis_header();
        let genesis_id = digest_lanes(&noid_chain::hash_block_header(&genesis));
        for (index, value) in genesis_id.into_iter().enumerate() {
            pin_eq(
                &mut builder,
                &cells[ORIGIN_ID + index],
                &LinExpr::constant(flat_of(value)),
            );
        }
        for (index, value) in crate::accumulator::genesis_accumulator()
            .to_lanes()
            .into_iter()
            .enumerate()
        {
            pin_eq(
                &mut builder,
                &cells[ORIGIN_ACC + index],
                &LinExpr::constant(flat_of(value)),
            );
        }
    }
    let gate = cells[BASE].add_const(F128::ONE);
    let boolean = mul(&mut builder, &cells[BASE], &gate);
    pin_eq(&mut builder, &boolean, &LinExpr::zero());
    pin_eq(
        &mut builder,
        &cells[TIP_CLASS],
        &LinExpr::constant(f128_from_u128(class.wire_id() as u128)),
    );
    let params = Class::ALL.map(|c| runtime.bank.config().class(c).pcs_params());
    let r_proofs = Class::ALL.map(|c| RPcsProof {
        native: &envelopes[c.index()].proof().field.pcs_open,
        params: &params[c.index()],
        commitment_root: flat_digest_lanes(&envelopes[c.index()].proof().commitment.root),
    });
    let (children, parents): (Vec<_>, Vec<_>) =
        scratch.into_iter().map(|s| (s.child, s.parent)).unzip();
    let amun_diff = std::env::var_os("NOID_FORK_DIFF").is_some();
    let amun_diff_inputs = amun_diff.then(|| (children.clone(), parents.clone()));
    // amun: witness assembly fork (v2.1). Sections: A = parent columns, S = parent seal, B = block slots, P = pins,
    // pre = roots/prev/selectors, C1 = arm Small, C2 = arms Large + Native (shared replay). A, S+B, C1, C2 only read
    // wires below A, so after one sequential assembly per class they run on witness-only forks at the recorded
    // offsets, merged in order; align + finalize then run on the merged builder.
    let fork_wa = builder.num_wires();
    let section_a = |builder: &mut FieldR1csBuilder| {
        prepare_history_step_parent_columns(
            builder,
            &r_proofs,
            selected.index(),
            &runtime.parts.geometry,
            children,
            parents,
        )
    };
    let HistoryStepBlockInput {
        start_accumulator,
        end_accumulator,
        components,
        authorization,
        sealed_header,
        parent_header,
        ..
    } = current;
    let section_sb = |builder: &mut FieldR1csBuilder| {
        let seal_start = builder.num_wires();
        let parent_seal = ParentSealTrace::alloc(&mut *builder, &parent_header);
        let seal_size = builder.num_wires() - seal_start;
        let block = build_block_slots_selected_zk_prefix(
            &mut *builder,
            &start_accumulator,
            &end_accumulator,
            &components,
            &sealed_header,
            TIER,
            authorization,
            &parent_header,
            &parent_seal.block_id,
            BlockRelationProfile::ScheduledV2(config),
        );
        (parent_seal, block, seal_size)
    };
    let c_pre = |builder: &mut FieldR1csBuilder| {
        let roots = envelopes
            .iter()
            .map(|e| alloc_flat_digest(&mut *builder, &e.proof().commitment.root))
            .collect::<Vec<_>>();
        let prev = envelopes[selected.index()]
            .proof()
            .io
            .iter()
            .map(|v| LinExpr::from_wire(builder.alloc_f128(*v)))
            .collect::<Vec<_>>();
        let selectors = parent_selector(&mut *builder, &prev[TIP_CLASS], selected);
        // The entire ordered bank and sealed legacy boundary are inherited.
        for index in BANK..POINT {
            pin_eq(&mut *builder, &cells[index], &prev[index]);
        }
        (roots, prev, selectors)
    };
    let c_arms = |builder: &mut FieldR1csBuilder,
                  arms: &[Class],
                  roots: &[_],
                  prev: &[LinExpr],
                  selectors: &[LinExpr; 3]|
     -> Result<Vec<_>, V2Error> {
        let mut out = Vec::new();
        let mut shared = None;
        let mut amun_ledger = 0;
        for &arm in arms {
            let slot = arm.index();
            let entry = runtime.bank.config().class(arm);
            let envelope = envelopes[slot].proof();
            let arm_gate = mul(&mut *builder, &gate, &selectors[slot]);
            let replay = if runtime.bank.config().shared_m24() && arm != Class::Small {
                if arm == Class::Large {
                    shared = Some(
                        shared_parent::replay_shared(
                            &mut *builder,
                            runtime,
                            selected,
                            &gate,
                            &selectors,
                            &prev,
                            &roots,
                            &envelopes,
                        )?
                        .map(Some),
                    );
                }
                shared.as_mut().ok_or(V2Error::Runtime)?[slot - 1]
                    .take()
                    .ok_or(V2Error::Runtime)?
            } else {
                let proof = C1FieldR1csProofTrace::alloc_shape_mode(
                    &mut *builder,
                    &envelope.field,
                    &entry.shape(),
                    &params[slot],
                    false,
                );
                let mut arm_obligations = PcsWalkObligations::default();
                let mut recorder = BaseSelectableParentRecorder::new_c1(PROOF_DOMAIN);
                let mut child = None;
                let mut error = None;
                let (_, fresh) = with_pin_gate(&arm_gate, || {
                    verify_field_c1_trace_deferred_region_with_post_commit_context_expr(
                        &mut *builder,
                        &mut recorder,
                        &entry.shape(),
                        &params[slot],
                        &[
                            prev[MATRIX + 2 * slot].clone(),
                            prev[MATRIX + 2 * slot + 1].clone(),
                        ],
                        &roots[slot],
                        &proof,
                        &runtime.bank.config().io_spec(),
                        &prev,
                        &[
                            prev[POST + 2 * slot].clone(),
                            prev[POST + 2 * slot + 1].clone(),
                        ],
                        Some(&mut arm_obligations),
                        |builder, context| match verify_joint_c1_region_sidecar_trace_post_commit(
                            builder,
                            context,
                            &runtime.parts.parent_vk,
                            runtime.parts.block_vk(arm),
                            &envelope.sidecar,
                        ) {
                            Ok(recording) => child = Some(recording),
                            Err(e) => error = Some(e),
                        },
                    )
                });
                if let Some(error) = error {
                    return Err(error.into());
                }
                shared_parent::ArmReplay {
                    fresh,
                    obligations: arm_obligations,
                    child: child.ok_or(V2Error::Layout)?,
                    parent: recorder.finish(),
                }
            };
            let shared_parent::ArmReplay {
                fresh,
                obligations: arm_obligations,
                child,
                parent,
            } = replay;
            out.push((arm_obligations, child, parent));
            let lane = Lane::for_class(arm);
            let incoming = C1MatrixAccClaimTrace {
                point: prev[lane.point..lane.value]
                    .chunks_exact(2)
                    .map(|x| ExtExpr::new(x[0].clone(), x[1].clone()))
                    .collect(),
                value: ExtExpr::new(prev[lane.value].clone(), prev[lane.value + 1].clone()),
            };
            let fold = C1MatrixFoldProofTrace::alloc(&mut *builder, &folds[slot], entry.outer_m());
            let mut channel = FsChannelTrace::new_c1(&mut *builder, FOLD_DOMAIN);
            channel.observe_label(&mut *builder, ROUTE_DOMAIN);
            channel.observe_lanes(&mut *builder, 1, std::slice::from_ref(&prev[TIP_CLASS]));
            let outgoing = with_pin_gate(&arm_gate, || {
                verify_matrix_claim_fold_c1_trace(
                    &mut *builder,
                    &mut channel,
                    entry.shape().k_log,
                    entry.shape().k_skip,
                    &fresh,
                    &incoming,
                    &prev[lane.live],
                    &fold,
                )
            });
            bind_lane(&mut *builder, &cells, &prev, lane, &arm_gate, &outgoing);
            crate::acceptance::row_ledger_mark(
                &*builder,
                &mut amun_ledger,
                &format!("bank: parent verifier and fold {arm:?}"),
            );
            // A new origin cannot import an unchecked v2 matrix claim.
            with_pin_gate(&cells[BASE], || {
                for cell in &cells[lane.point..=lane.live] {
                    pin_eq(&mut *builder, cell, &LinExpr::zero());
                }
            });
        }
        Ok(out)
    };
    enum AmunStep<C, A, P> {
        Later(C, A),
        Early(P),
    }
    let (parent_seal, block, prev, selectors, step) = match super::amun_fork::plan(class.wire_id(), TIER, frozen) {
        Some(((a_size, seal_size, b_size, p_size, pre_size, small_size, ln_size), snaps)) if amun_diff => {
            let fail = super::amun_fork::fail;
            let w_seal = fork_wa + a_size;
            let w_block = w_seal + seal_size;
            let start2 = w_block + b_size + p_size;
            let w_ln = start2 + pre_size + small_size;
            // Trial forks, taken and seeded exactly as the fork path does.
            let mut fa = builder.amun_fork_witness_cached(fork_wa, &builder.amun_cache_snapshot()).map_err(fail)?;
            let mut fc0 = builder.amun_fork_witness_cached(start2, &snaps.pre).map_err(fail)?;
            let (t_roots, t_prev, t_selectors) = c_pre(&mut fc0);
            let mut fc1 = fc0.amun_fork_witness_cached(w_ln, &snaps.ln).map_err(fail)?;
            let columns = section_a(&mut builder)?;
            let a_end = builder.num_wires();
            crate::acceptance::row_ledger_mark(&builder, &mut row_ledger, "bank: IO and parent columns");
            let snap_sb = builder.amun_cache_snapshot();
            let (parent_seal, block, seal_size) = section_sb(&mut builder);
            let snap_p = builder.amun_cache_snapshot();
            let seal_end = a_end + seal_size;
            let b_end = builder.num_wires();
            pin_eq(&mut builder, &parent_seal.height, &block.slots().start_acc.height);
            pin_eq(
                &mut builder,
                &cells[ORIGIN_ACC],
                &LinExpr::constant(flat_of(noid_core::Block128((config.activation_height() - 1) as u128))),
            );
            let p_end = builder.num_wires();
            crate::acceptance::row_ledger_mark(&builder, &mut row_ledger, "bank: complete block prefix");
            let snap_pre = builder.amun_cache_snapshot();
            let (roots, prev, selectors) = c_pre(&mut builder);
            let pre_end = builder.num_wires();
            let mut arms = c_arms(&mut builder, &[Class::Small], &roots, &prev, &selectors)?;
            let small_end = builder.num_wires();
            let snap_ln = builder.amun_cache_snapshot();
            arms.extend(c_arms(&mut builder, &[Class::Large, Class::Native], &roots, &prev, &selectors)?);
            let ln_end = builder.num_wires();
            let snap_end = builder.amun_cache_snapshot();
            // Trials (results discarded), then compare with the sequential witness.
            let (ch, pa) = amun_diff_inputs.ok_or(V2Error::Layout)?;
            let _ = prepare_history_step_parent_columns(&mut fa, &r_proofs, selected.index(), &runtime.parts.geometry, ch, pa)?;
            let _ = c_arms(&mut fc0, &[Class::Small], &t_roots, &t_prev, &t_selectors)?;
            let _ = c_arms(&mut fc1, &[Class::Large, Class::Native], &t_roots, &t_prev, &t_selectors)?;
            let (sv, sf) = builder.amun_witness_view();
            for (name, fork, lo, hi) in [
                ("A parent columns", &fa, fork_wa, w_seal),
                ("pre + C1 arm Small", &fc0, start2, w_ln),
                ("C2 arms Large+Native", &fc1, w_ln, w_ln + ln_size),
            ] {
                let (fv, ff) = fork.amun_witness_view();
                let mut count = 0usize;
                let mut shown = 0;
                for w in lo..hi {
                    let (a, b) = (fv.get(w), sv.get(w));
                    let (x, y) = (ff.get(w), sf.get(w));
                    if a != b || x != y {
                        count += 1;
                        if shown < 8 {
                            eprintln!("amun fork diff [{name}] wire {w} (+{}): fork {:?} flags {:?} / sequential {:?} flags {:?}", w - lo, a, x, b, y);
                            shown += 1;
                        }
                    }
                }
                // flags below the section that the fork changed (a sequential section may clear flags of earlier wires)
                let mut below = 0usize;
                for w in 0..lo.min(ff.len()).min(sf.len()) {
                    if ff[w] != sf[w] && w >= fork_wa {
                        below += 1;
                    }
                }
                eprintln!("amun fork diff [{name}]: wires {lo}..{hi} (fork len {}), {count} differ; earlier-wire flag differences {below}", fv.len());
            }
            let _ = fail("diff run finished (diagnostics only; this proof is sequential)");
            (parent_seal, block, prev, selectors, AmunStep::Later(columns, arms))
        }
        Some(((a_size, seal_size, b_size, p_size, pre_size, small_size, ln_size), snaps)) => {
            let w_seal = fork_wa + a_size;
            let w_block = w_seal + seal_size;
            let start2 = w_block + b_size + p_size;
            let w_ln = start2 + pre_size + small_size;
            let fail = super::amun_fork::fail;
            // S+B is the long section. Start it, then take the other forks and run c_pre on this
            // thread (it must see this thread's pin-gate stack) while S+B is already running.
            let mut fs = builder.amun_fork_witness_cached(w_seal, &snaps.sb).map_err(fail)?;
            let amun_t = std::time::Instant::now();
            let overlapped = std::thread::scope(|scope| -> Result<_, V2Error> {
                let sb = scope.spawn(|| {
                    let t = std::time::Instant::now();
                    let r = crate::acceptance::trace::with_clean_pin_gates(|| section_sb(&mut fs));
                    (r, t.elapsed().as_secs_f64() * 1e3)
                });
                let setup_t = std::time::Instant::now();
                let mut fa = builder.amun_fork_witness_cached(fork_wa, &builder.amun_cache_snapshot()).map_err(fail)?;
                let mut fc0 = builder.amun_fork_witness_cached(start2, &snaps.pre).map_err(fail)?;
                let (roots, prev, selectors) = c_pre(&mut fc0);
                if fc0.num_wires() != start2 + pre_size {
                    return Err(fail("pre-arm section size changed"));
                }
                let mut fc1 = fc0.amun_fork_witness_cached(w_ln, &snaps.ln).map_err(fail)?;
                let setup_ms = setup_t.elapsed().as_secs_f64() * 1e3;
                let ((columns, ms_a), ((small, ms_c1), (large_native, ms_c2))) = rayon::join(
                    || { let t = std::time::Instant::now(); let r = section_a(&mut fa); (r, t.elapsed().as_secs_f64() * 1e3) },
                    || {
                        rayon::join(
                            || {
                                crate::acceptance::trace::with_clean_pin_gates(|| {
                                    { let t = std::time::Instant::now(); let r = c_arms(&mut fc0, &[Class::Small], &roots, &prev, &selectors); (r, t.elapsed().as_secs_f64() * 1e3) }
                                })
                            },
                            || {
                                crate::acceptance::trace::with_clean_pin_gates(|| {
                                    { let t = std::time::Instant::now(); let r = c_arms(&mut fc1, &[Class::Large, Class::Native], &roots, &prev, &selectors); (r, t.elapsed().as_secs_f64() * 1e3) }
                                })
                            },
                        )
                    },
                );
                let mut columns = columns?;
                let mut arms = small?;
                arms.extend(large_native?);
                if fa.num_wires() != w_seal || fc0.num_wires() != w_ln || fc1.num_wires() != w_ln + ln_size {
                    let _ = sb.join();
                    return Err(fail("a forked section changed size"));
                }
                // Finalize reads the parent columns and the arm wires, not the block section.
                // Build that witness (block range left zero) and run it while S+B is still going.
                let end = w_ln + ln_size;
                let mut early = fa.amun_clone_witness().map_err(fail)?;
                early.amun_pad_witness(end).map_err(fail)?;
                early.amun_copy_witness_range(&fc0, start2, w_ln).map_err(fail)?;
                early.amun_copy_witness_range(&fc1, w_ln, end).map_err(fail)?;
                early.amun_restore_caches(&snaps.end).map_err(fail)?;
                early.amun_apply_flags(&snaps.flag_changes).map_err(fail)?;
                early.amun_recount_pending();
                let flags_before = early.amun_witness_view().1.to_vec();
                let pending_before = early.amun_pending_counts();
                let publics_before = early.amun_publics_len();
                let patches_before = early.amun_patch_lens();
                let mut obligations = Vec::new();
                let mut recorded_children = Vec::new();
                let mut recorded_parents = Vec::new();
                for (obligation, child, parent) in arms {
                    obligations.push(obligation);
                    recorded_children.push(child);
                    recorded_parents.push(parent);
                }
                if runtime.bank.config().shared_m24() {
                    columns.align_shared_replay_hints(&early, &recorded_children, &recorded_parents)?;
                }
                let fin_t = std::time::Instant::now();
                let fin_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    with_pin_gate(&gate, || {
                        let call = if runtime.bank.config().shared_m24() {
                            crate::acceptance::trace::r_pcs_region::finalize_shared_history_step_parent_region
                        } else {
                            finalize_history_step_parent_region
                        };
                        call(
                            &mut early,
                            columns,
                            &obligations,
                            &selectors,
                            &recorded_children,
                            &recorded_parents,
                        )
                    })
                }));
                let finalize_ms = fin_t.elapsed().as_secs_f64() * 1e3;
                let prepared = match fin_result {
                    Ok(result) => result?,
                    Err(_) => {
                        let _ = sb.join();
                        return Err(fail("finalize overlap panicked"));
                    }
                };
                let parent_seal_timed = sb.join().map_err(|_| fail("S+B section panicked"))?;
                Ok((
                    fa,
                    fc0,
                    fc1,
                    early,
                    prepared,
                    flags_before,
                    pending_before,
                    publics_before,
                    patches_before,
                    prev,
                    selectors,
                    setup_ms,
                    ms_a,
                    ms_c1,
                    ms_c2,
                    finalize_ms,
                    parent_seal_timed,
                ))
            })?;
            let (
                fa,
                fc0,
                fc1,
                early,
                prepared,
                flags_before,
                pending_before,
                publics_before,
                patches_before,
                prev,
                selectors,
                setup_ms,
                ms_a,
                ms_c1,
                ms_c2,
                finalize_ms,
                parent_seal_block_timed,
            ) = overlapped;
            let ((parent_seal, block, _), ms_sb) = parent_seal_block_timed;
            let parallel_ms = amun_t.elapsed().as_secs_f64() * 1e3;
            let join_t = std::time::Instant::now();
            if fs.num_wires() != w_block + b_size {
                return Err(fail("a forked section changed size"));
            }
            builder.amun_join_witness(fa, fork_wa).map_err(fail)?;
            builder.amun_join_witness(fs, w_seal).map_err(fail)?;
            builder.amun_restore_caches(&snaps.p).map_err(fail)?;
            pin_eq(&mut builder, &parent_seal.height, &block.slots().start_acc.height);
            pin_eq(
                &mut builder,
                &cells[ORIGIN_ACC],
                &LinExpr::constant(flat_of(noid_core::Block128((config.activation_height() - 1) as u128))),
            );
            if builder.num_wires() != start2 {
                return Err(fail("pin section size changed"));
            }
            builder.amun_join_witness(fc0, start2).map_err(fail)?;
            builder.amun_join_witness(fc1, w_ln).map_err(fail)?;
            builder.amun_restore_caches(&snaps.end).map_err(fail)?;
            builder.amun_apply_flags(&snaps.flag_changes).map_err(fail)?;
            builder
                .amun_install_finalize_tail(
                    &early,
                    w_seal,
                    start2,
                    &flags_before,
                    pending_before,
                    publics_before,
                    patches_before,
                )
                .map_err(fail)?;
            if super::amun_fork::trace() {
                let join_ms = join_t.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "amun fork v2.1 trace: parallel part {parallel_ms:.0} ms (setup {setup_ms:.0} beside S+B, A {ms_a:.0}, S+B {ms_sb:.0}, C1 {ms_c1:.0}, C2 {ms_c2:.0}, join {join_ms:.0}, finalize {finalize_ms:.0} beside S+B)"
                );
            }
            (parent_seal, block, prev, selectors, AmunStep::Early(prepared))
        }
        None => {
            let columns = section_a(&mut builder)?;
            let a_end = builder.num_wires();
            crate::acceptance::row_ledger_mark(&builder, &mut row_ledger, "bank: IO and parent columns");
            let snap_sb = builder.amun_cache_snapshot();
            let flags_a = builder.amun_witness_view().1.to_vec();
            let (parent_seal, block, seal_size) = section_sb(&mut builder);
            let snap_p = builder.amun_cache_snapshot();
            let mut flag_changes = builder.amun_flag_changes(&flags_a, a_end);
            let seal_end = a_end + seal_size;
            let b_end = builder.num_wires();
            pin_eq(&mut builder, &parent_seal.height, &block.slots().start_acc.height);
            pin_eq(
                &mut builder,
                &cells[ORIGIN_ACC],
                &LinExpr::constant(flat_of(noid_core::Block128((config.activation_height() - 1) as u128))),
            );
            let p_end = builder.num_wires();
            crate::acceptance::row_ledger_mark(&builder, &mut row_ledger, "bank: complete block prefix");
            let snap_pre = builder.amun_cache_snapshot();
            let flags_p = builder.amun_witness_view().1.to_vec();
            let (roots, prev, selectors) = c_pre(&mut builder);
            let pre_end = builder.num_wires();
            let mut arms = c_arms(&mut builder, &[Class::Small], &roots, &prev, &selectors)?;
            let small_end = builder.num_wires();
            let snap_ln = builder.amun_cache_snapshot();
            flag_changes.extend(builder.amun_flag_changes(&flags_p, p_end));
            let flags_s = builder.amun_witness_view().1.to_vec();
            arms.extend(c_arms(&mut builder, &[Class::Large, Class::Native], &roots, &prev, &selectors)?);
            let ln_end = builder.num_wires();
            let snap_end = builder.amun_cache_snapshot();
            flag_changes.extend(builder.amun_flag_changes(&flags_s, small_end));
            super::amun_fork::record(
                class.wire_id(),
                TIER,
                frozen,
                (a_end - fork_wa, seal_end - a_end, b_end - seal_end, p_end - b_end, pre_end - p_end, small_end - pre_end, ln_end - small_end),
                super::amun_fork::Snaps { sb: snap_sb, p: snap_p, pre: snap_pre, ln: snap_ln, end: snap_end, flag_changes },
            );
            (parent_seal, block, prev, selectors, AmunStep::Later(columns, arms))
        }
    };
    let parent = match step {
        AmunStep::Early(prepared) => prepared,
        AmunStep::Later(mut columns, arms) => {
            let mut obligations = Vec::new();
            let mut recorded_children = Vec::new();
            let mut recorded_parents = Vec::new();
            for (o, c, p) in arms {
                obligations.push(o);
                recorded_children.push(c);
                recorded_parents.push(p);
            }
            if runtime.bank.config().shared_m24() {
                columns.align_shared_replay_hints(&builder, &recorded_children, &recorded_parents)?;
            }
            let finalize = if runtime.bank.config().shared_m24() {
                crate::acceptance::trace::r_pcs_region::finalize_shared_history_step_parent_region
            } else {
                finalize_history_step_parent_region
            };
            let fin_t = std::time::Instant::now();
            let parent = with_pin_gate(&gate, || {
                finalize(
                    &mut builder,
                    columns,
                    &obligations,
                    &selectors,
                    &recorded_children,
                    &recorded_parents,
                )
            })?;
            if super::amun_fork::trace() {
                eprintln!("amun fork v2.1 trace: finalize {:.0} ms", fin_t.elapsed().as_secs_f64() * 1e3);
            }
            parent
        }
    };
    crate::acceptance::row_ledger_mark(
        &builder,
        &mut row_ledger,
        "bank: parent column finalization",
    );
    for lane in 0..2 {
        with_pin_gate(&gate, || {
            pin_eq(
                &mut builder,
                &parent_seal.semantic_id[lane],
                &prev[ACC + 1 + lane],
            )
        });
        with_pin_gate(&cells[BASE], || {
            pin_eq(
                &mut builder,
                &parent_seal.block_id[lane],
                &cells[ORIGIN_ID + lane],
            );
            pin_eq(
                &mut builder,
                &parent_seal.semantic_id[lane],
                &cells[ORIGIN_ACC + 1 + lane],
            );
        });
    }
    for (index, start) in block.slots().start_acc.ordered_lanes().iter().enumerate() {
        with_pin_gate(&gate, || pin_eq(&mut builder, start, &prev[ACC + index]));
        with_pin_gate(&cells[BASE], || {
            pin_eq(&mut builder, start, &cells[ORIGIN_ACC + index])
        });
    }
    for (index, end) in block.slots().end_acc.ordered_lanes().iter().enumerate() {
        pin_eq(&mut builder, end, &cells[ACC + index]);
    }
    Ok(Assembly {
        class,
        builder,
        block,
        parent,
        io_seal,
        io,
        matrix,
    })
}

enum AssemblyOutput {
    Frozen(FrozenV2),
    Built(Built),
}

fn finish_assembly(
    runtime: &Runtime,
    assembly: Assembly,
    header: &BlockHeader,
    end: &ChainAccumulator,
) -> Result<AssemblyOutput, V2Error> {
    let Assembly {
        class,
        mut builder,
        mut block,
        parent,
        io_seal,
        mut io,
        matrix,
    } = assembly;
    let config = runtime.bank.config().class(class);
    io_seal.seal(&mut builder, &mut io, ACC, end)?;
    block
        .seal_direct_tail(&mut builder, header, end)
        .map_err(|_| V2Error::Boundary)?;
    if let Some(stats) = builder.product_observation() {
        eprintln!("[products] class={class:?} products={} eligible={} repeated={} clears={} maximum_entries={} nested_bytes={}",stats.products,stats.eligible,stats.repeated,stats.clear_count,stats.maximum_entries,stats.nested_bytes);
        if builder.uses_fused_boolean_pins() {
            eprintln!(
                "[fused-pins] class={class:?} stats={:?}",
                builder.boolean_pin_observation()
            );
        }
        eprintln!(
            "[linear-pins] class={class:?} reused={}",
            builder.reused_linear_pins()
        );
        eprintln!(
            "[product-aliases] class={class:?} stats={:?}",
            builder.product_alias_observation()
        );
    }
    let used = builder.num_wires();
    let limit = 1usize << config.outer_m();
    if used > limit {
        return Err(V2Error::Shape { used, limit });
    }
    let block = finalize_selected_zk_block_region(block, config.outer_m())?;
    let preparations = Preparations { parent, block };
    match matrix {
        None => {
            let (matrix, witness) = builder.build();
            let (matrix, witness) =
                crate::acceptance::expand_empty_field_tail(matrix, witness, config.shape());
            Ok(AssemblyOutput::Frozen(FrozenV2 {
                matrix,
                witness,
                preparations,
            }))
        }
        Some(matrix) => {
            let (_, mut witness) = builder.build_witness_only();
            witness.resize(limit, F128::ZERO);
            if used != matrix.useful_rows()
                || preparations.parent.vk() != runtime.parts.parent_vk()
                || preparations.block.vk() != runtime.parts.block_vk(class)
            {
                return Err(V2Error::Runtime);
            }
            runtime.bank.parse(&io)?;
            Ok(AssemblyOutput::Built(Built {
                class,
                matrix,
                witness,
                io,
                preparations,
            }))
        }
    }
}

pub fn assemble_frozen<const TIER: usize>(
    runtime: &Runtime,
    class: Class,
    origin: &Origin,
    parent: Option<&Terminal>,
    current: HistoryStepBlockInput<TIER>,
) -> Result<FrozenV2, V2Error> {
    let prepared = prepare_parent(runtime, class, origin, parent, &current)?;
    let header = current.sealed_header;
    let end = current.end_accumulator.clone();
    let assembly = prepare_assembly(runtime, prepared, current, true, true)?;
    match finish_assembly(runtime, assembly, &header, &end)? {
        AssemblyOutput::Frozen(frozen) => Ok(frozen),
        _ => unreachable!("frozen assembly"),
    }
}

pub fn derive_assembled_block_vk<const TIER: usize>(
    runtime: &Runtime,
    class: Class,
    origin: &Origin,
    parent: Option<&Terminal>,
    current: HistoryStepBlockInput<TIER>,
) -> Result<(BlockRegionSidecarVk, usize), V2Error> {
    let prepared = prepare_parent(runtime, class, origin, parent, &current)?;
    let header = current.sealed_header;
    let end = current.end_accumulator.clone();
    let Assembly {
        mut builder,
        mut block,
        parent,
        io_seal,
        mut io,
        matrix,
        ..
    } = prepare_assembly(runtime, prepared, current, true, false)?;
    if matrix.is_some() || parent.vk() != runtime.parts.parent_vk() {
        return Err(V2Error::Runtime);
    }
    io_seal.seal(&mut builder, &mut io, ACC, &end)?;
    block
        .seal_direct_tail(&mut builder, &header, &end)
        .map_err(|_| V2Error::Boundary)?;
    let used = builder.num_wires();
    let limit = 1usize << runtime.bank.config().class(class).outer_m();
    if used > limit {
        return Err(V2Error::Shape { used, limit });
    }
    let block =
        finalize_selected_zk_block_region(block, runtime.bank.config().class(class).outer_m())?;
    let vk = block.vk().clone();
    let (count, _) = builder.build_witness_only();
    if count != used {
        return Err(V2Error::Runtime);
    }
    Ok((vk, used))
}

#[must_use = "dropping this value cancels the unsealed block attempt"]
pub struct PreparedForPow {
    assembly: Assembly,
    header: BlockHeader,
    parent_header: BlockHeader,
    start: ChainAccumulator,
}

pub fn prepare_for_pow<const TIER: usize>(
    runtime: &Runtime,
    class: Class,
    origin: &Origin,
    parent: Option<&Terminal>,
    current: HistoryStepBlockInput<TIER>,
) -> Result<PreparedForPow, V2Error> {
    if current.sealed_header.nonce != 0 {
        return Err(V2Error::Boundary);
    }
    super::amun_fork::stage("prepare_parent start");
    let prepared = prepare_parent(runtime, class, origin, parent, &current)?;
    super::amun_fork::stage("prepare_parent done");
    let header = current.sealed_header;
    let parent_header = current.parent_header;
    let start = current.start_accumulator.clone();
    let amun_assembly = prepare_assembly(runtime, prepared, current, false, false)?;
    super::amun_fork::stage("assembly done");
    Ok(PreparedForPow {
        assembly: amun_assembly,
        header,
        parent_header,
        start,
    })
}
impl PreparedForPow {
    pub fn retained_witness_bytes(&self) -> usize {
        self.assembly.builder.retained_witness_bytes()
    }
    pub fn seal_nonce(mut self, runtime: &Runtime, nonce: u128) -> Result<Built, V2Error> {
        self.header.nonce = nonce;
        let end = self
            .start
            .advance(&self.parent_header, &self.header)
            .map_err(|_| V2Error::Boundary)?;
        super::amun_fork::stage("seal_nonce start");
        match finish_assembly(runtime, self.assembly, &self.header, &end)? {
            AssemblyOutput::Built(built) => { super::amun_fork::stage("finish done");
 Ok(built) }
            _ => unreachable!("witness assembly"),
        }
    }
}

pub fn prove_built(
    runtime: &Runtime,
    built: &Built,
    cancellation: &std::sync::atomic::AtomicBool,
) -> Result<Terminal, V2Error> {
    if cancellation.load(std::sync::atomic::Ordering::Acquire) {
        return Err(V2Error::Cancelled);
    }
    super::amun_fork::stage("prove_built start");
    let class = built.class;
    let config = runtime.bank.config().class(class);
    runtime.bank.authenticate(class, &built.matrix)?;
    super::amun_fork::stage("authenticated");
    if runtime.bank.parse(&built.io)?.class != class {
        return Err(V2Error::Io);
    }
    if built.preparations.parent.vk() != runtime.parts.parent_vk()
        || built.preparations.block.vk() != runtime.parts.block_vk(class)
    {
        return Err(V2Error::Runtime);
    }
    let params = config.pcs_params();
    let spec = runtime.bank.config().io_spec();
    macro_rules! prove {
        ($function:path, $matrix:expr) => {{
            let parent = built.preparations.parent.certified_c1_prover_plan()?;
            let block = built.preparations.block.certified_c1_prover_plan()?;
            let mut channel = FsLaneChallenger::new_c1(PROOF_DOMAIN);
            super::amun_fork::stage("plans ready");
            $function(
                $matrix,
                &built.witness,
                &params,
                &spec,
                &built.io,
                &runtime.bank.post_commit_digest(class),
                cancellation,
                &mut channel,
                |context| -> Result<JointC1RegionSidecarProof, RegionSidecarError> {
                    let (proof, claims) = crate::region_sidecar::prove_joint_c1_region_sidecar(
                        &parent,
                        &block,
                        context.witness(),
                        context,
                    )?;
                    context.append_c1_claims(claims);
                    Ok(proof)
                },
            )
            .map_err(|_| V2Error::Cancelled)
        }};
    }
    let (field, sidecar, commitment, _) = match &built.matrix {
        HistoryStepMatrixLease::Resident(m) => prove!(
            prove_field_c1_with_public_io_and_post_commit_context_cancellable,
            m.as_ref()
        ),
        HistoryStepMatrixLease::Compact(m) => prove!(
            prove_field_compact_c1_with_public_io_and_post_commit_context_cancellable,
            m.as_ref()
        ),
    }?;
    if cancellation.load(std::sync::atomic::Ordering::Acquire) {
        return Err(V2Error::Cancelled);
    }
    Ok(Terminal {
        class,
        proof: V2Proof {
            field,
            commitment,
            io: built.io.clone(),
            sidecar: sidecar?,
        },
    })
}
