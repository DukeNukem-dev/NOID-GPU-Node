//! amun: wire-offset plan for the forked witness assembly (see `assembly::prepare_assembly`).
//! The first witness-only assembly per (class, tier) runs sequentially and records the block-slots and pin
//! section sizes; later assemblies fork at those offsets. Any mismatch disables forking for the process.
//! `NOID_FORK=0` disables it.

use super::V2Error;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

static PLANS: Mutex<Option<HashMap<(u8, usize), (usize, usize)>>> = Mutex::new(None);
static DISABLED: AtomicBool = AtomicBool::new(false);

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NOID_FORK").is_ok_and(|v| v == "0")) && !DISABLED.load(Ordering::Relaxed)
}

pub(super) fn plan(class: u8, tier: usize, frozen: bool) -> Option<(usize, usize)> {
    if frozen || !enabled() {
        return None;
    }
    PLANS.lock().ok()?.as_ref()?.get(&(class, tier)).copied()
}

pub(super) fn record(class: u8, tier: usize, frozen: bool, block: usize, pin: usize) {
    if frozen {
        return;
    }
    if let Ok(mut g) = PLANS.lock() {
        let map = g.get_or_insert_with(HashMap::new);
        if map.insert((class, tier), (block, pin)).is_none() {
            eprintln!("amun fork: plan for class {class} tier {tier}: block-slots {block} wires, pin {pin} wires");
        }
    }
}

pub(super) fn fail(reason: &str) -> V2Error {
    DISABLED.store(true, Ordering::Relaxed);
    eprintln!("amun fork: {reason}; forking disabled, sequential assembly from now on");
    V2Error::Layout
}
