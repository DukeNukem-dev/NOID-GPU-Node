// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Paranoid Zero.

use super::*;
use noid_ivc_core::challenger::Challenger;


fn io_key(io: &[F128]) -> [u8; 32] {
    let mut state = [0x243f6a8885a308d3u64, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89];
    for (i, value) in io.iter().enumerate() {
        let lane = i % 4;
        state[lane] = state[lane]
            .wrapping_mul(0x9e3779b185ebca87)
            .wrapping_add(value.lo)
            .wrapping_add(value.hi.rotate_left((i % 29) as u32));
    }
    let mut out = [0u8; 32];
    for (i, lane) in state.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&lane.to_le_bytes());
    }
    out
}

pub(super) struct GhostPair {
    arms: [Class; 2],
    proofs: [(V2Proof, Scratch); 2],
}

pub(super) struct GhostInflight {
    key: [u8; 32],
    handle: Option<std::thread::JoinHandle<Result<GhostPair, V2Error>>>,
}

/// Compute the two shape-only arms for `class` off the history-step clock.
/// `io` is the parent terminal the next template will extend. The next prepare
/// consumes a matching result. A key miss leaves the thread running and proves
/// inline. `replace` false does not overwrite a prefetch for a different terminal.
pub(super) fn start_ghost_prefetch(
    runtime: std::sync::Arc<Runtime>,
    class: Class,
    io: Vec<F128>,
    replace: bool,
) {
    let key = io_key(&io);
    if let Ok(slot) = runtime.ghost_prefetch.lock() {
        if let Some(inflight) = slot.as_ref() {
            if inflight.key == key || !replace {
                return;
            }
        }
    }
    let worker = std::sync::Arc::clone(&runtime);
    let spawned = std::thread::Builder::new()
        .name("v21-ghost-prefetch".into())
        .spawn(move || -> Result<GhostPair, V2Error> {
            let arms: Vec<Class> = Class::ALL.into_iter().filter(|arm| *arm != class).collect();
            if arms.len() != 2 {
                return Err(V2Error::Runtime);
            }
            let arm0 = arms[0];
            let arm1 = arms[1];
            let io1 = io.clone();
            let worker1 = std::sync::Arc::clone(&worker);
            let (left, right) = rayon::join(
                move || shape_only_arm(&worker, arm0, io),
                move || shape_only_arm(&worker1, arm1, io1),
            );
            let (proof0, scratch0) = left?;
            let (proof1, scratch1) = right?;
            Ok(GhostPair {
                arms: [arm0, arm1],
                proofs: [(proof0, scratch0), (proof1, scratch1)],
            })
        });
    let Ok(handle) = spawned else {
        return;
    };
    if let Ok(mut slot) = runtime.ghost_prefetch.lock() {
        if let Some(inflight) = slot.as_ref() {
            if inflight.key == key || !replace {
                return;
            }
        }
        *slot = Some(GhostInflight {
            key,
            handle: Some(handle),
        });
    }
}

fn take_prefetched(runtime: &Runtime, io: &[F128]) -> Option<Result<GhostPair, V2Error>> {
    let key = io_key(io);
    let handle = {
        let mut slot = runtime.ghost_prefetch.lock().ok()?;
        let inflight = slot.as_mut()?;
        if inflight.key != key {
            return None;
        }
        inflight.handle.take()?
    };
    if let Ok(mut slot) = runtime.ghost_prefetch.lock() {
        if slot.as_ref().is_some_and(|inflight| inflight.key == key) {
            *slot = None;
        }
    }
    handle.join().ok()
}

pub(super) fn shape_only_arm(
    runtime: &Runtime,
    class: Class,
    io: Vec<F128>,
) -> Result<(V2Proof, Scratch), V2Error> {
    let mut envelope = shape_only_envelope(runtime, class, io)?;
    let scratch = scratch_replay(runtime, class, &envelope)?;
    crate::acceptance::trace::self_verify::patch_shape_only_query_positions_c1(
        &mut envelope.field,
        &runtime.bank.config().class(class).pcs_params(),
        &scratch.query_values,
    );
    Ok((envelope, scratch))
}

pub(super) struct PreparedParent<'a> {
    pub selected: Class,
    pub current: Class,
    pub envelopes: [ParentEnvelope<'a>; 3],
    pub scratch: [Scratch; 3],
    pub folds: [C1MatrixFoldProof; 3],
    pub io: Vec<F128>,
}

pub(super) fn prepare_parent<'a, const TIER: usize>(
    runtime: &Runtime,
    current_class: Class,
    origin: &Origin,
    parent: Option<&'a Terminal>,
    current: &HistoryStepBlockInput<TIER>,
) -> Result<PreparedParent<'a>, V2Error> {
    origin.check(&runtime.bank)?;
    if runtime.bank.config().class(current_class).pages() != TIER {
        return Err(V2Error::Runtime);
    }
    if current.sealed_header.height < origin.activation_height()? {
        return Err(V2Error::Boundary);
    }
    let mut folds = Class::ALL.map(|c| zero_fold_proof(runtime.bank.config().class(c).outer_m()));
    match parent {
        None => {
            if current.start_accumulator != origin.boundary
                || current.end_accumulator.height != origin.activation_height()?
                || noid_chain::hash_block_header(&current.parent_header) != origin.parent_id
            {
                return Err(V2Error::Boundary);
            }
            let prior_io = runtime
                .bank
                .initial_io(origin, Class::Small, &origin.boundary);
            let mut envelopes = Vec::new();
            let mut scratches = Vec::new();
            for arm in Class::ALL {
                let (ghost, scratch) = shape_only_arm(runtime, arm, prior_io.clone())?;
                envelopes.push(ParentEnvelope::Local(ghost));
                scratches.push(scratch);
            }
            Ok(PreparedParent {
                selected: Class::Small,
                current: current_class,
                envelopes: envelopes.try_into().map_err(|_| V2Error::Runtime)?,
                scratch: scratches.try_into().map_err(|_| V2Error::Runtime)?,
                folds,
                io: runtime
                    .bank
                    .initial_io(origin, current_class, &current.end_accumulator),
            })
        }
        Some(parent) => {
            let parsed = runtime.bank.parse(&parent.proof.io)?;
            if parsed.origin != *origin
                || parsed.accumulator != current.start_accumulator
                || parsed.class != parent.class
            {
                return Err(V2Error::Boundary);
            }
            let selected = parsed.class;
            // amun: the selected arm's replay + matrix fold and the two shape-only ghost arms are independent;
            // run them in parallel. Errors keep the sequential order (selected arm first, then ghosts by class).
            let amun_t = std::time::Instant::now();
            let prefetch_hit = std::sync::atomic::AtomicBool::new(false);
            let ghost_arms: Vec<Class> = Class::ALL.into_iter().filter(|arm| *arm != selected).collect();
            let (selected_part, (ghost_0, ghost_1)) = rayon::join(
                || -> Result<_, V2Error> {
            // Admission already replayed this parent. Reuse it when the proof bytes match.
            let replay_started = std::time::Instant::now();
            let (fresh, scratch, replay_hit) = if let Some(hit) = cached_replay(runtime, parent) {
                (hit.0, hit.1, true)
            } else {
                let (fresh, scratch) = replay(runtime, selected, &parent.proof)?;
                remember_replay(runtime, parent, &fresh, &scratch);
                (fresh, scratch, false)
            };
            let replay_ms = replay_started.elapsed().as_secs_f64() * 1e3;
            let incoming = parsed.claims[selected.index()].clone().unwrap_or_else(|| {
                C1MatrixAccClaim::zero(runtime.bank.config().class(selected).outer_m())
            });
            let matrix = runtime.load_matrix(selected)?;
            let mut channel = FsLaneChallenger::new_c1(FOLD_DOMAIN);
            channel.observe_label(ROUTE_DOMAIN);
            channel.observe_bytes(&[selected.wire_id()]);
            let (fold, claim) = match &matrix {
                HistoryStepMatrixLease::Resident(m) => prove_matrix_claim_fold_c1(
                    m,
                    &fresh,
                    &incoming,
                    parsed.claims[selected.index()].is_some(),
                    &mut channel,
                ),
                HistoryStepMatrixLease::Compact(m) => prove_matrix_claim_fold_compact_c1(
                    m,
                    &fresh,
                    &incoming,
                    parsed.claims[selected.index()].is_some(),
                    &mut channel,
                ),
            };
            drop(matrix);
            Ok((scratch, fold, claim, amun_t.elapsed().as_secs_f64() * 1e3, replay_ms, replay_hit))
                },
                || {
                    if let Some(Ok(pair)) = take_prefetched(runtime, &parent.proof.io) {
                        if pair.arms == [ghost_arms[0], ghost_arms[1]] {
                            prefetch_hit.store(true, std::sync::atomic::Ordering::Relaxed);
                            let [(proof0, scratch0), (proof1, scratch1)] = pair.proofs;
                            return (
                                (Ok((proof0, scratch0)), 0.0),
                                (Ok((proof1, scratch1)), 0.0),
                            );
                        }
                    }
                    rayon::join(
                        || {
                            let t = std::time::Instant::now();
                            (shape_only_arm(runtime, ghost_arms[0], parent.proof.io.clone()), t.elapsed().as_secs_f64() * 1e3)
                        },
                        || {
                            let t = std::time::Instant::now();
                            (shape_only_arm(runtime, ghost_arms[1], parent.proof.io.clone()), t.elapsed().as_secs_f64() * 1e3)
                        },
                    )
                },
            );
            let (scratch, fold, claim, selected_ms, replay_ms, replay_hit) = selected_part?;
            if super::amun_fork::trace() {
                eprintln!(
                    "amun prepare_parent: replay {} {replay_ms:.0} ms, selected replay+fold {selected_ms:.0} ms, ghosts {:.0} / {:.0} ms {}, wall {:.0} ms",
                    if replay_hit { "hit" } else { "miss" },
                    ghost_0.1, ghost_1.1,
                    if prefetch_hit.load(std::sync::atomic::Ordering::Relaxed) { "prefetched" } else { "computed" },
                    amun_t.elapsed().as_secs_f64() * 1e3
                );
            }
            let mut ghosts = vec![Some(ghost_0.0?), Some(ghost_1.0?)];
            folds[selected.index()] = fold;
            let mut io = parent.proof.io.clone();
            install_claim(selected, &mut io, &claim)?;
            io[TIP_CLASS] = f128_from_u128(current_class.wire_id() as u128);
            io[ACC..ACC + 10].copy_from_slice(&block_acc_lanes(&current.end_accumulator));
            runtime.bank.parse(&io)?;
            let mut envelopes = Vec::new();
            let mut scratches = Vec::new();
            let mut selected_scratch = Some(scratch);
            for arm in Class::ALL {
                if arm == selected {
                    envelopes.push(ParentEnvelope::Persisted(&parent.proof));
                    scratches.push(selected_scratch.take().ok_or(V2Error::Runtime)?);
                } else {
                    let slot = ghost_arms.iter().position(|a| *a == arm).ok_or(V2Error::Runtime)?;
                    let (ghost, scratch) = ghosts[slot].take().ok_or(V2Error::Runtime)?;
                    envelopes.push(ParentEnvelope::Local(ghost));
                    scratches.push(scratch);
                }
            }
            Ok(PreparedParent {
                selected,
                current: current_class,
                envelopes: envelopes.try_into().map_err(|_| V2Error::Runtime)?,
                scratch: scratches.try_into().map_err(|_| V2Error::Runtime)?,
                folds,
                io,
            })
        }
    }
}

fn shape_only_envelope(runtime: &Runtime, class: Class, io: Vec<F128>) -> Result<V2Proof, V2Error> {
    let (field, root) = shape_only_field_r1cs_proof_c1(
        &runtime.bank.config().class(class).shape(),
        &runtime.bank.config().class(class).pcs_params(),
    );
    Ok(V2Proof {
        field,
        commitment: Commitment {
            root,
            params: runtime.bank.config().class(class).pcs_params(),
        },
        io,
        sidecar: shape_only_joint_c1_region_sidecar_proof(
            &runtime.parts.parent_vk,
            runtime.parts.block_vk(class),
            runtime.bank.config().class(class).outer_m(),
        )?,
    })
}

fn scratch_replay(runtime: &Runtime, class: Class, envelope: &V2Proof) -> Result<Scratch, V2Error> {
    let mut builder = FieldR1csBuilder::new_witness_only();
    let statement = alloc_pinned_flat_digest(&mut builder, &runtime.bank.matrix_digest(class));
    let post = alloc_pinned_flat_digest(&mut builder, &runtime.bank.post_commit_digest(class));
    let root = alloc_flat_digest(&mut builder, &envelope.commitment.root);
    let io = envelope
        .io
        .iter()
        .map(|v| LinExpr::from_wire(builder.alloc_f128(*v)))
        .collect::<Vec<_>>();
    let proof = C1FieldR1csProofTrace::alloc_shape_mode(
        &mut builder,
        &envelope.field,
        &runtime.bank.config().class(class).shape(),
        &runtime.bank.config().class(class).pcs_params(),
        false,
    );
    let mut channel = FsChannelUnionRecorder::new_c1(PROOF_DOMAIN);
    let mut child = None;
    let mut error = None;
    verify_field_c1_trace_deferred_region_with_post_commit_context_expr(
        &mut builder,
        &mut channel,
        &runtime.bank.config().class(class).shape(),
        &runtime.bank.config().class(class).pcs_params(),
        &statement,
        &root,
        &proof,
        &runtime.bank.config().io_spec(),
        &io,
        &post,
        Some(&mut PcsWalkObligations::default()),
        |builder, context| match verify_joint_c1_region_sidecar_trace_post_commit(
            builder,
            context,
            &runtime.parts.parent_vk,
            runtime.parts.block_vk(class),
            &envelope.sidecar,
        ) {
            Ok(recording) => child = Some(recording),
            Err(e) => error = Some(e),
        },
    );
    if let Some(error) = error {
        return Err(error.into());
    }
    let parent = channel.finish();
    let start = parent
        .challenge_wires
        .len()
        .checked_sub(history_step_query_lane_count(
            &runtime.bank.config().class(class).pcs_params(),
        ))
        .ok_or(V2Error::Layout)?;
    Ok(Scratch {
        child: capture_scratch_recording(&child.ok_or(V2Error::Layout)?, &builder),
        query_values: parent.challenge_wires[start..]
            .iter()
            .map(|v| v.eval(builder.values()))
            .collect(),
        parent: capture_scratch_recording(&parent, &builder),
    })
}


pub(super) struct ReplaySlot {
    key: [u8; 32],
    fresh: C1FreshLincheckClaim,
    scratch: Scratch,
}

fn replay_cache_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NOID_REPLAY_CACHE").is_ok_and(|value| value == "0"))
}

pub(super) fn remember_replay(
    runtime: &Runtime,
    terminal: &Terminal,
    fresh: &C1FreshLincheckClaim,
    scratch: &Scratch,
) {
    if !replay_cache_on() {
        return;
    }
    let Ok(key) = super::wire::terminal_replay_key(runtime, terminal) else {
        return;
    };
    if let Ok(mut slot) = runtime.replay_cache.lock() {
        *slot = Some(ReplaySlot {
            key,
            fresh: fresh.clone(),
            scratch: scratch.clone(),
        });
    }
}

fn cached_replay(runtime: &Runtime, terminal: &Terminal) -> Option<(C1FreshLincheckClaim, Scratch)> {
    if !replay_cache_on() {
        return None;
    }
    {
        let Ok(slot) = runtime.replay_cache.lock() else {
            return None;
        };
        if slot.is_none() {
            return None;
        }
    }
    let key = super::wire::terminal_replay_key(runtime, terminal).ok()?;
    let Ok(slot) = runtime.replay_cache.lock() else {
        return None;
    };
    let found = slot.as_ref()?;
    (found.key == key).then(|| (found.fresh.clone(), found.scratch.clone()))
}

pub(super) fn replay(
    runtime: &Runtime,
    class: Class,
    envelope: &V2Proof,
) -> Result<(C1FreshLincheckClaim, Scratch), V2Error> {
    if pcs_params_statement_bytes(&envelope.commitment.params)
        != pcs_params_statement_bytes(&runtime.bank.config().class(class).pcs_params())
    {
        return Err(V2Error::Runtime);
    }
    if runtime.bank.parse(&envelope.io)?.class != class {
        return Err(V2Error::Io);
    }
    let mut channel =
        LayoutRecordingChallenger::new_c1(PROOF_DOMAIN, runtime.parts.parent_layout(class).clone());
    let mut child = None;
    let (_, fresh) = verify_field_c1_deferred_matrix_with_post_commit_context(
        &runtime.bank.config().class(class).shape(),
        &runtime.bank.matrix_digest(class),
        &envelope.commitment,
        &envelope.field,
        &runtime.bank.config().io_spec(),
        &envelope.io,
        &runtime.bank.post_commit_digest(class),
        &envelope.sidecar,
        &mut channel,
        |sidecar, context| {
            child = Some(
                verify_joint_c1_region_sidecar_post_commit_layout_captured(
                    &runtime.parts.parent_vk,
                    runtime.parts.block_vk(class),
                    sidecar,
                    context,
                    runtime.parts.child_layout(class).clone(),
                )
                .map_err(|_| VerifyError::Auxiliary)?,
            );
            Ok(())
        },
    )?;
    Ok((
        fresh,
        Scratch {
            child: child.ok_or(V2Error::Layout)?,
            parent: channel.finish().map_err(|_| V2Error::Layout)?,
            query_values: Vec::new(),
        },
    ))
}

#[test]
#[ignore = "amun: split v2 shape-only ghost arms. Set NOID_V21_META to v21-runtime-metadata.bin"]
fn amun_v2_ghost_arm_phases() {
    let bytes = std::fs::read(std::env::var("NOID_V21_META").expect("NOID_V21_META")).unwrap();
    assert!(bytes.starts_with(b"O1V21BN1"), "metadata magic");
    const PREFIX: usize = 8 + 3 * 32 + 4;
    let length = u32::from_le_bytes(bytes[PREFIX - 4..PREFIX].try_into().unwrap()) as usize;
    assert_eq!(length, bytes.len() - PREFIX, "metadata length");
    let parts = RuntimeParts::decode_compact(&bytes[PREFIX..]).unwrap();
    let digests: [[u8; 32]; 3] = [
        bytes[8..40].try_into().unwrap(),
        bytes[40..72].try_into().unwrap(),
        bytes[72..104].try_into().unwrap(),
    ];
    let bank = Bank::pin(digests, &parts);
    let runtime = std::sync::Arc::new(Runtime::new(bank, parts, Box::new(super::NoMatrix)).unwrap());
    let io = vec![F128::ZERO; IO_LEN];

    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
    for class in Class::ALL {
        for pass in 0..2 {
            let cfg = runtime.bank.config().class(class);
            let t = std::time::Instant::now();
            let (field, root) = shape_only_field_r1cs_proof_c1(&cfg.shape(), &cfg.pcs_params());
            let field_synth = ms(t);
            let t = std::time::Instant::now();
            let sidecar = shape_only_joint_c1_region_sidecar_proof(
                &runtime.parts.parent_vk,
                runtime.parts.block_vk(class),
                cfg.outer_m(),
            )
            .unwrap();
            let side_synth = ms(t);
            let envelope = V2Proof {
                field,
                commitment: Commitment {
                    root,
                    params: cfg.pcs_params(),
                },
                io: io.clone(),
                sidecar,
            };
            let mut builder = FieldR1csBuilder::new_witness_only();
            let statement =
                alloc_pinned_flat_digest(&mut builder, &runtime.bank.matrix_digest(class));
            let post =
                alloc_pinned_flat_digest(&mut builder, &runtime.bank.post_commit_digest(class));
            let root_e = alloc_flat_digest(&mut builder, &envelope.commitment.root);
            let io_e = envelope
                .io
                .iter()
                .map(|v| LinExpr::from_wire(builder.alloc_f128(*v)))
                .collect::<Vec<_>>();
            let t = std::time::Instant::now();
            let proof = C1FieldR1csProofTrace::alloc_shape_mode(
                &mut builder,
                &envelope.field,
                &cfg.shape(),
                &cfg.pcs_params(),
                false,
            );
            let alloc_ms = ms(t);
            let wires_before = builder.num_wires();
            let mut channel = FsChannelUnionRecorder::new_c1(PROOF_DOMAIN);
            let mut child = None;
            let mut error = None;
            let mut side_ms = 0.0;
            let t = std::time::Instant::now();
            verify_field_c1_trace_deferred_region_with_post_commit_context_expr(
                &mut builder,
                &mut channel,
                &cfg.shape(),
                &cfg.pcs_params(),
                &statement,
                &root_e,
                &proof,
                &runtime.bank.config().io_spec(),
                &io_e,
                &post,
                Some(&mut PcsWalkObligations::default()),
                |builder, context| {
                    let ts = std::time::Instant::now();
                    match verify_joint_c1_region_sidecar_trace_post_commit(
                        builder,
                        context,
                        &runtime.parts.parent_vk,
                        runtime.parts.block_vk(class),
                        &envelope.sidecar,
                    ) {
                        Ok(recording) => {
                            side_ms = ts.elapsed().as_secs_f64() * 1e3;
                            child = Some(recording);
                        }
                        Err(e) => {
                            side_ms = ts.elapsed().as_secs_f64() * 1e3;
                            error = Some(e);
                        }
                    }
                },
            );
            let verify_ms = ms(t);
            assert!(error.is_none(), "sidecar recording failed");
            let t = std::time::Instant::now();
            let parent = channel.finish();
            let start = parent
                .challenge_wires
                .len()
                .checked_sub(history_step_query_lane_count(&cfg.pcs_params()))
                .unwrap();
            let queries: Vec<_> = parent.challenge_wires[start..]
                .iter()
                .map(|v| v.eval(builder.values()))
                .collect();
            let _child = capture_scratch_recording(&child.unwrap(), &builder);
            let _parent_rec = capture_scratch_recording(&parent, &builder);
            let finish_ms = ms(t);
            eprintln!(
                "[ghost-phase] {class:?} pass={pass} m={} pages={} field_synth={field_synth:.1} side_synth={side_synth:.1} alloc={alloc_ms:.1} verify={verify_ms:.1} sidecar_cb={side_ms:.1} field_verify={:.1} finish={finish_ms:.1} wires={wires_before}/{} queries={}",
                cfg.outer_m(),
                cfg.pages(),
                verify_ms - side_ms,
                builder.num_wires(),
                queries.len(),
            );
        }
    }

    let (fresh_large, scratch_large) = shape_only_arm(&runtime, Class::Large, io.clone()).unwrap();
    let (fresh_native, scratch_native) = shape_only_arm(&runtime, Class::Native, io.clone()).unwrap();
    start_ghost_prefetch(std::sync::Arc::clone(&runtime), Class::Small, io.clone(), true);
    std::thread::sleep(std::time::Duration::from_millis(900));
    let t_hit = std::time::Instant::now();
    let hit = take_prefetched(&runtime, &io).expect("prefetch present").expect("prefetch ok");
    let hit_ms = t_hit.elapsed().as_secs_f64() * 1e3;
    assert_eq!(hit.arms, [Class::Large, Class::Native]);
    assert_eq!(hit.proofs[0].1.query_values, scratch_large.query_values);
    assert_eq!(hit.proofs[1].1.query_values, scratch_native.query_values);
    assert_eq!(hit.proofs[0].0.commitment.root, fresh_large.commitment.root);
    assert_eq!(hit.proofs[1].0.commitment.root, fresh_native.commitment.root);
    eprintln!("[ghost-phase] prefetch hit {hit_ms:.1} ms");

    let t = std::time::Instant::now();
    let ((large_ok, large_ms), (native_ok, native_ms)) = rayon::join(
        || {
            let t = std::time::Instant::now();
            let ok = shape_only_arm(&runtime, Class::Large, io.clone()).is_ok();
            (ok, t.elapsed().as_secs_f64() * 1e3)
        },
        || {
            let t = std::time::Instant::now();
            let ok = shape_only_arm(&runtime, Class::Native, io.clone()).is_ok();
            (ok, t.elapsed().as_secs_f64() * 1e3)
        },
    );
    eprintln!(
        "[ghost-phase] join Large={large_ms:.1} ok={large_ok} Native={native_ms:.1} ok={native_ok} wall={:.1}",
        t.elapsed().as_secs_f64() * 1e3
    );
}
