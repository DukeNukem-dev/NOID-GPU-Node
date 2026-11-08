#!/usr/bin/env bash
# Build the NOID GPU node (default x86-64-v3) with the official v2.1.0 HistoryStep packs embedded.
# Usage: ./build-gpu-node.sh /path/to/packs   (contains v1/history-step-pack-v1, v2/history-step-pack-v2, v21/history-step-pack-v21)
#   PRECHECKED=1   trust the packs' matrix-checks.json instead of re-checking every matrix (~35 min)
#   TARGET_CPU=... default x86-64-v3 (+pclmulqdq,+aes); x86-64 for CPUs without AVX2 (+sse4.1,+pclmulqdq,
#                  the node's own minimum); e.g. znver4 for a Zen 4/5-only build
set -euo pipefail
P=$(cd "${1:?usage: $0 /path/to/packs}" && pwd)
cd "$(dirname "$0")"
set -a; . "$P/v1/history-step-pack-v1/pins.env"; . "$P/v2/history-step-pack-v2/pins.env"; . "$P/v21/history-step-pack-v21/pins.env"; set +a
export NOID_HISTORY_STEP_PACK_DIR=$P/v1/history-step-pack-v1
export NOID_V2_PACK_DIR=$P/v2/history-step-pack-v2
export NOID_V21_PACK_DIR=$P/v21/history-step-pack-v21
export NOID_RETIREMENT_KEYS_DIR=$P/v2/history-step-pack-v2/retirement-keys
# x86-64 and x86-64-v3 do not include PCLMULQDQ. noid_core detects it at run time anyway, but only a build with the
# feature enabled can inline the carry-less multiply (13-16% faster proofs, upstream issue #66). The node refuses to
# start without SSE4.1 + PCLMULQDQ, so enabling them drops no supported CPU. AVX-512/VPCLMULQDQ paths stay run time.
TARGET_CPU=${TARGET_CPU:-x86-64-v3}
if [[ $TARGET_CPU == x86-64 ]]; then FEATURES=+sse4.1,+pclmulqdq; else FEATURES=+pclmulqdq,+aes; fi
export RUSTFLAGS="-C target-cpu=$TARGET_CPU -C target-feature=$FEATURES"
if [[ ${PRECHECKED:-0} == 1 ]]; then
  export NOID_PRECHECKED_MATRIX_PACKS=1
  export NOID_V2_MATRIX_CHECKS_SHA256=$(sha256sum "$NOID_V2_PACK_DIR/matrix-checks.json" | cut -d" " -f1)
  export NOID_V21_MATRIX_CHECKS_SHA256=$(sha256sum "$NOID_V21_PACK_DIR/matrix-checks.json" | cut -d" " -f1)
fi
cargo build --locked --release -p noid_node --bins --features retired-history
ls -la "${CARGO_TARGET_DIR:-target}/release/parano1d"
