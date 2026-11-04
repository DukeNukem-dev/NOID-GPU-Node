//! amun: wire-offset plan for the forked v2.1 witness assembly (see `assembly::prepare_assembly`).
//! The first witness-only assembly per (class, tier) runs sequentially and records the section sizes; later
//! assemblies fork at those offsets. Any mismatch disables forking for the process. `NOID_FORK=0` disables it.

use super::V2Error;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub(super) use noid_ivc_core::field_circuit::AmunCacheSnap;

/// (parent columns, seal, block slots, pins, pre-arms, arm Small, arms Large+Native) in wires.
pub(super) type Sizes = (usize, usize, usize, usize, usize, usize, usize);

/// Cache states at the section starts of a sequential assembly: S+B, pins, pre-arms, arms Large+Native, end.
pub(super) struct Snaps {
    pub sb: AmunCacheSnap,
    pub p: AmunCacheSnap,
    pub pre: AmunCacheSnap,
    pub ln: AmunCacheSnap,
    pub end: AmunCacheSnap,
    /// Flag changes on earlier wires made by S+B, pre+Small and Large+Native, in that order.
    pub flag_changes: Vec<(u32, u8)>,
}

static PLANS: Mutex<Option<HashMap<(u8, usize), (Sizes, Arc<Snaps>)>>> = Mutex::new(None);
static DISABLED: AtomicBool = AtomicBool::new(false);

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NOID_FORK").is_ok_and(|v| v == "0")) && !DISABLED.load(Ordering::Relaxed)
        && !crate::amun_fork_disabled()
}

pub(super) fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NOID_FORK_TRACE").is_ok_and(|v| v != "0"))
}

pub(super) fn plan(class: u8, tier: usize, frozen: bool) -> Option<(Sizes, Arc<Snaps>)> {
    if frozen || !enabled() {
        return None;
    }
    PLANS.lock().ok()?.as_ref()?.get(&(class, tier)).cloned()
}

pub(super) fn record(class: u8, tier: usize, frozen: bool, sizes: Sizes, snaps: Snaps) {
    if frozen || !enabled() {
        return;
    }
    if let Ok(mut g) = PLANS.lock() {
        let map = g.get_or_insert_with(HashMap::new);
        if map.insert((class, tier), (sizes, Arc::new(snaps))).is_none() {
            eprintln!(
                "amun fork v2.1: plan for class {class} tier {tier}: parent columns {} wires, seal {}, block slots {}, \
                 pins {}, pre-arms {}, arm Small {}, arms Large+Native {}",
                sizes.0, sizes.1, sizes.2, sizes.3, sizes.4, sizes.5, sizes.6
            );
        }
    }
}

pub(super) fn fail(reason: &str) -> V2Error {
    DISABLED.store(true, Ordering::Relaxed);
    eprintln!("amun fork v2.1: {reason}; forking disabled, sequential assembly from now on");
    V2Error::Layout
}

/// amun: stage marker for the proof timeline (journal timestamps give the gaps).
pub(super) fn stage(label: &str) {
    if trace() {
        eprintln!("amun stage: {label}");
    }
}
