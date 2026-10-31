#!/usr/bin/env bash
# Build the NOID GPU node (x86-64-v3) with the official v2.1.0 HistoryStep packs embedded.
# Usage: ./build-gpu-node.sh /path/to/packs   (contains v1/history-step-pack-v1, v2/history-step-pack-v2, v21/history-step-pack-v21)
#   PRECHECKED=1   trust the packs' matrix-checks.json instead of re-checking every matrix (~35 min)
#   TARGET_CPU=... default x86-64-v3 (+pclmulqdq,+aes); e.g. znver4 for a Zen 4/5-only build
set -euo pipefail
P=$(cd "${1:?usage: $0 /path/to/packs}" && pwd)
cd "$(dirname "$0")"
set -a; . "$P/v1/history-step-pack-v1/pins.env"; . "$P/v2/history-step-pack-v2/pins.env"; . "$P/v21/history-step-pack-v21/pins.env"; set +a
export NOID_HISTORY_STEP_PACK_DIR=$P/v1/history-step-pack-v1
export NOID_V2_PACK_DIR=$P/v2/history-step-pack-v2
export NOID_V21_PACK_DIR=$P/v21/history-step-pack-v21
export NOID_RETIREMENT_KEYS_DIR=$P/v2/history-step-pack-v2/retirement-keys
# x86-64-v3 does not include PCLMULQDQ/AES, but every AVX2 CPU has them; noid_core picks its carry-less multiply at
# compile time, so without them a v3 build is ~13% slower. AVX-512 paths are still chosen at run time.
export RUSTFLAGS="-C target-cpu=${TARGET_CPU:-x86-64-v3} -C target-feature=+pclmulqdq,+aes"
if [[ ${PRECHECKED:-0} == 1 ]]; then
  export NOID_PRECHECKED_MATRIX_PACKS=1
  export NOID_V2_MATRIX_CHECKS_SHA256=$(sha256sum "$NOID_V2_PACK_DIR/matrix-checks.json" | cut -d" " -f1)
  export NOID_V21_MATRIX_CHECKS_SHA256=$(sha256sum "$NOID_V21_PACK_DIR/matrix-checks.json" | cut -d" " -f1)
fi
cargo build --locked --release -p noid_node --bins --features retired-history
ls -la target/release/parano1d
