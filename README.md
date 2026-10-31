# NOID GPU Node

A build of the [Parano1d](https://git.parano1d.org/ignotusnemo/parano1d) (NOID) v2.1.0 node with
GPU-accelerated HistoryStep proving and a faster mining pipeline. It is the node behind
[pqpool.com](https://pqpool.com).

| | Stock v2.1.0 | This build |
| --- | --- | --- |
| HistoryStep proof, Ryzen 9 3900X + RTX 3070 (PCIe 3.0 x16) | 30.0 s | **3.05 s** |
| New block → miners hashing the new job (extminer) | after the proof (~3–30 s) | **~8 ms** |

All numbers are medians of 20+ proofs on one machine (Ryzen 9 3900X, 32 GB DDR4-3200, RTX 3070 on PCIe 3.0 x16),
v2 proof class (`V2(Small)`, 63 pages). Every proof was also checked with the node's own verifier.

**Status: experimental. v2 only.** NOID v2.1 activates at block #279,370 (about 4 November 2026); from then on this
build falls back to slower (sequential) witness assembly for v2.1 proofs until the v2.1 work is done. Watch this
repository for an updated release.

## What is changed

Proving (`noid_ivc_core`, `noid_ivc_prover`, `noid_recursive`, `noid_poseidon2b`):
- GPU kernels (CUDA, `noid_ivc_core/cuda/noid_gpu.cu`) for the PCS commitment (additive NTT and Merkle tree straight
  from GPU memory), the C1 matrix-claim fold, the deep-chain walk, the lincheck alpha fold, and the large column
  relations and shift proofs. GF(2^128) arithmetic uses the GPU's carry-less multiply (`clmad`).
  Every GPU result is spot-checked against the CPU; any error or mismatch switches that stage back to the CPU.
- Witness assembly runs as four parallel streams (witness-only builder forks, merged in order with exact checks).
- A·z and B·z are computed while the GPU commit runs; parallel parent replay; faster Poseidon2b (PCLMUL) kernel.

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

- Linux x86-64 with AVX2, PCLMULQDQ and AES (Intel Haswell or newer, AMD Zen or newer). AVX-512 code paths are
  selected at run time where the CPU has them. Tested on AMD Ryzen 9 3900X (3.16 s proofs with this portable build,
  the same as a Zen 2-tuned build) and Intel i7-7700T. A build tuned for one CPU (`-C target-cpu=znver2`) can crash
  on other CPUs with an illegal instruction; use the default portable build unless you know what you are doing.
- NVIDIA GPU, compute capability 8.6 or newer (RTX 30 and RTX 40 series), driver 580 or newer. GPU memory: ~6 GB
  during a proof. Newer architectures load the PTX (untested). Without a usable GPU the node proves on the CPU.
- A full PCIe slot matters. On a PCIe 2.0 x1 riser with a 4-core i7-7700T the same proof took 11.7 s (RTX 3070) and
  10.4 s (RTX 3090): the commitment step copies ~512 MB between CPU and GPU, and the CPU stages scale with cores.
- 16 GB RAM or more (node ~2.5–5 GB).

## Build

You need Rust (rustup), a C compiler, and the three HistoryStep proof packs from the official v2.1.0 release
(`history-step-pack-v1.tar.gz`, `-v2.tar.gz`, `-v21.tar.gz`, checked against that release's `SHA256SUMS`).
The CUDA kernels are prebuilt (`noid_ivc_core/src/noid_gpu.fatbin`); rebuilding them needs CUDA 13.x (`nvcc`).

```
./build-gpu-node.sh /path/to/packs        # x86-64-v3 + PCLMULQDQ/AES build; binary in target/release/parano1d
```

The packs directory must contain `v1/history-step-pack-v1`, `v2/history-step-pack-v2` and `v21/history-step-pack-v21`
(the unpacked archives). The first build checks every proof matrix (~35 min); `PRECHECKED=1` trusts the packs'
`matrix-checks.json` like the upstream release script does.

## Run

Use it exactly like the stock `parano1d` (same config, data directory and CLI). For solo mining with an external
miner, run `--mode extminer` with a mining key and point your stratum bridge or pool at its RPC.

Useful environment variables: `NOID_GPU=0` (CPU only), `NOID_EARLY_JOB=0`, `NOID_SHADOW_VERIFY=0`,
`NOIDH_FIELD_PROVE_TIMING=1` (per-stage proof timings in the log). Recommended allocator setting:
`_RJEM_MALLOC_CONF=thp:always,dirty_decay_ms:30000,muzzy_decay_ms:30000,oversize_threshold:0`.

## Licence

Apache-2.0, like upstream (`LICENSE`, `NOTICE`). The upstream README is in `UPSTREAM-README.md`.
This is an independent build, not an official Parano1d release.
