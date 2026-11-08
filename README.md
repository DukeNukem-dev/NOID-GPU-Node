# NOID GPU Node

A build of the [Parano1d](https://git.parano1d.org/ignotusnemo/parano1d) (NOID) v2.1.0 node with
GPU-accelerated HistoryStep proving and a faster mining pipeline. It is the node behind
[pqpool.com](https://pqpool.com).

| v2.1 HistoryStep proof, `V21(Small)` | Stock v2.1.0 release | This build |
| --- | --- | --- |
| Ryzen 9 3900X + RTX 3070 (PCIe 3.0 x16) | 31.2 s | **3.2 s** (GPU) |
| Ryzen 9 3900X, CPU only | 31.2 s | **12.1 s** |
| Intel i7-7700T, CPU only | 89.3 s | **27.2 s** |
| New block → miners hashing the new job (extminer) | after the proof | **~8 ms** |

GPU row: median of 213 live proofs over 19.6 h (8 Nov 2026). CPU rows: median of 3 from a fixed chain snapshot
(method and full tables in upstream issue
[#66](https://github.com/ignotusnemo/parano1d/issues/66)). Every live proof is also checked with the node's own
verifier.

**Status: experimental.** Runs v2.1 (`Small`, `Large`, `Native` classes) on mainnet.

## What is changed

Proving (`noid_ivc_core`, `noid_ivc_prover`, `noid_recursive`, `noid_poseidon2b`):
- GPU kernels (CUDA, `noid_ivc_core/cuda/noid_gpu.cu`) for the PCS commitment (additive NTT and Merkle tree straight
  from GPU memory), the C1 matrix-claim fold, the deep-chain walk, the lincheck alpha fold, and the large column
  relations and shift proofs. GF(2^128) arithmetic uses the GPU's carry-less multiply (`clmad`).
  Every GPU result is spot-checked against the CPU; any error or mismatch switches that stage back to the CPU.
- Witness assembly runs as four parallel streams (witness-only builder forks, merged in order with exact checks).
- A·z and B·z are computed while the GPU commit runs; parallel parent replay; a Poseidon2b kernel for CPUs with
  PCLMULQDQ but no VPCLMULQDQ (Intel before Ice Lake, AMD Zen 1/2) and a memo for repeated large hashes. These two
  give most of the CPU-only gain and are proposed upstream in #66.

Mining and networking (`noid_rpc`, `noid_miner`, `noid_node`, `noid_p2p`):
- **Early job hand-out** (extminer): `getBlockTemplate` returns as soon as the block header is built. The HistoryStep
  proof does not depend on the nonce, so it finishes in the background; `submitBlock` waits for it only if a nonce is
  found first. `NOID_EARLY_JOB=0` restores the stock order.
- The first template after a new block is coinbase-only (no transaction proving on the critical path); transactions
  follow in the refresh template.
- 16 automatic peers instead of 8; exact-object hedging after 0.8 s instead of 4 s (faster block download).
- A shadow verify of every template proof (`NOID_SHADOW_VERIFY=0` disables).

`GPU-CHANGES.md` lists every modified file. All changes are commits on top of the unmodified upstream source.

## Requirements

- Linux x86-64. The default build (`x86-64-v3`) needs AVX2 (Intel Haswell or newer, AMD Zen or newer); check with
  `grep -c avx2 /proc/cpuinfo` (0 = no AVX2). For CPUs without AVX2 (many Celeron/Pentium/Atom parts) build with
  `TARGET_CPU=x86-64` (see Build): the node itself needs SSE4.1 + PCLMULQDQ on any build. AVX-512 and VPCLMULQDQ code
  paths are selected at run time. A build for a different CPU stops with "illegal instruction".
- Optional: NVIDIA GPU, compute capability 8.6 or newer (RTX 30 and RTX 40 series), driver 580 or newer. GPU memory:
  ~6 GB during a proof. Newer architectures load the PTX (untested). Without a usable GPU the node proves on the CPU.
- A full PCIe slot matters. On a PCIe 2.0 x1 riser with a 4-core i7-7700T the same proof took 11.7 s (RTX 3070) and
  10.4 s (RTX 3090): the commitment step copies ~512 MB between CPU and GPU, and the CPU stages scale with cores.
- 16 GB RAM or more (node ~2.5–5 GB).

## Build

You need Rust (rustup), a C compiler, and the three HistoryStep proof packs from the official v2.1.0 release
(`history-step-pack-v1.tar.gz`, `-v2.tar.gz`, `-v21.tar.gz`, checked against that release's `SHA256SUMS`).
The CUDA kernels are prebuilt (`noid_ivc_core/src/noid_gpu.fatbin`); rebuilding them needs CUDA 13.x (`nvcc`).

```
./build-gpu-node.sh /path/to/packs                     # x86-64-v3 + PCLMULQDQ/AES (AVX2 CPUs)
TARGET_CPU=x86-64 ./build-gpu-node.sh /path/to/packs   # portable: x86-64 + SSE4.1/PCLMULQDQ (CPUs without AVX2)
```

The binary is `target/release/parano1d`. On a 3900X, CPU only, the portable build proves `V21(Small)` in 15.2 s and
the v3 build in 12.1 s.

The packs directory must contain `v1/history-step-pack-v1`, `v2/history-step-pack-v2` and `v21/history-step-pack-v21`
(the unpacked archives). The first build checks every proof matrix (~35 min); `PRECHECKED=1` trusts the packs'
`matrix-checks.json` like the upstream release script does.

## Run

Use it exactly like the stock `parano1d` (same config, data directory and CLI). For solo mining with an external
miner, run `--mode extminer` with a mining key and point your stratum bridge or pool at its RPC.

Useful environment variables: `NOID_GPU=0` (CPU only, below), `NOID_EARLY_JOB=0`, `NOID_SHADOW_VERIFY=0`,
`NOIDH_FIELD_PROVE_TIMING=1` (per-stage proof timings in the log). Recommended allocator setting:
`_RJEM_MALLOC_CONF=thp:always,dirty_decay_ms:30000,muzzy_decay_ms:30000,oversize_threshold:0`.

## CPU-only proving

- No NVIDIA GPU or driver: nothing to do. The node looks for `libcuda.so.1` at start and proves on the CPU when it is
  missing, with all CPU improvements active.
- A GPU is present but should not be used (for example it mines): set `NOID_GPU=0`. The GPU is then never opened.
  With systemd, add a drop-in such as `/etc/systemd/system/parano1d.service.d/gpu-off.conf`:

  ```
  [Service]
  Environment=NOID_GPU=0
  ```

  then `sudo systemctl daemon-reload && sudo systemctl restart parano1d`.
- Check: `nvidia-smi` lists no `parano1d` process, and the log's `history_step_ms` values are CPU times
  (seconds, not ~3 s).

GPU sharing with a miner: if `/usr/local/bin/gpu-turn.py` exists and the GPU is in use, the node asks it for a turn
before each proof (we use this to pause a GPU miner during the proof). Without that file, or with `NOID_GPU=0`, the
node does not ask.

## Licence

Apache-2.0, like upstream (`LICENSE`, `NOTICE`). The upstream README is in `UPSTREAM-README.md`.
This is an independent build, not an official Parano1d release.
