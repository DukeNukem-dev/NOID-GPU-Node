// amun: Poseidon2b (flat/GCM basis) Merkle tree kernels for noid_ivc_core::merkle (leaf sponge no-pad, node compress).
// Loaded at runtime with the CUDA driver API (cuModuleLoadData); entry points are extern "C".
#include <cstdint>
struct u128 { uint64_t lo, hi; };
#include "p2b_consts.h"

__device__ __forceinline__ void cl(uint64_t a, uint64_t b, uint64_t &lo, uint64_t &hi) {
  asm("clmad.lo.u64 %0, %1, %2, %3;" : "=l"(lo) : "l"(a), "l"(b), "l"(0ull));
  asm("clmad.hi.u64 %0, %1, %2, %3;" : "=l"(hi) : "l"(a), "l"(b), "l"(0ull));
}
struct w256 { uint64_t r0, r1, r2, r3; };
__device__ __forceinline__ w256 mulw(u128 a, u128 b) {
  uint64_t p0l, p0h, p1l, p1h, ml, mh;
  cl(a.lo, b.lo, p0l, p0h); cl(a.hi, b.hi, p1l, p1h); cl(a.lo ^ a.hi, b.lo ^ b.hi, ml, mh);
  ml ^= p0l ^ p1l; mh ^= p0h ^ p1h;
  return {p0l, p0h ^ ml, p1l ^ mh, p1h};
}
__device__ __forceinline__ w256 sqw(u128 a) {
  uint64_t l0, h0, l1, h1; cl(a.lo, a.lo, l0, h0); cl(a.hi, a.hi, l1, h1); return {l0, h0, l1, h1};
}
__device__ __forceinline__ u128 red(w256 w) {
  uint64_t r0 = w.r0, r1 = w.r1, r2 = w.r2, r3 = w.r3;
  r1 ^= r3 ^ (r3 << 1) ^ (r3 << 2) ^ (r3 << 7); r2 ^= (r3 >> 63) ^ (r3 >> 62) ^ (r3 >> 57);
  r0 ^= r2 ^ (r2 << 1) ^ (r2 << 2) ^ (r2 << 7); r1 ^= (r2 >> 63) ^ (r2 >> 62) ^ (r2 >> 57);
  return {r0, r1};
}
__device__ __forceinline__ u128 x(u128 a, u128 b) { return {a.lo ^ b.lo, a.hi ^ b.hi}; }
__device__ __forceinline__ u128 mul(u128 a, u128 b) { return red(mulw(a, b)); }
__device__ __forceinline__ u128 sq(u128 a) { return red(sqw(a)); }
__device__ __forceinline__ u128 sbox(u128 v) { u128 v2 = sq(v), v4 = sq(v2); return mul(mul(v, v2), v4); }
__device__ __forceinline__ bool is_one(u128 c) { return c.lo == P2B_ONE.lo && c.hi == P2B_ONE.hi; }

__device__ __forceinline__ void mds_full(u128 s[4]) {
  u128 in[4] = {s[0], s[1], s[2], s[3]};
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    w256 acc = {0, 0, 0, 0}; u128 plain = {0, 0};
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      u128 c = P2B_FULL[i][j];
      if (is_one(c)) plain = x(plain, in[j]);
      else { w256 p = mulw(in[j], c); acc.r0 ^= p.r0; acc.r1 ^= p.r1; acc.r2 ^= p.r2; acc.r3 ^= p.r3; }
    }
    s[i] = x(red(acc), plain);
  }
}
__device__ __forceinline__ void permute(u128 s[4]) {
  mds_full(s);
  #pragma unroll 1
  for (int r = 0; r < P2B_R; r++) {
    if (r < P2B_RF / 2 || r >= P2B_RF / 2 + P2B_RP) {
      #pragma unroll
      for (int i = 0; i < 4; i++) s[i] = sbox(x(s[i], P2B_RC[r][i]));
      mds_full(s);
    } else {
      s[0] = sbox(x(s[0], P2B_RC[r][0]));
      u128 sum = x(x(s[0], s[1]), x(s[2], s[3]));
      #pragma unroll
      for (int i = 0; i < 4; i++) s[i] = x(sum, mul(s[i], P2B_DM1[i]));
    }
  }
}

__device__ __forceinline__ u128 ld(const uint8_t *p) {
  ulonglong2 v = *reinterpret_cast<const ulonglong2 *>(p); return {v.x, v.y};
}
__device__ __forceinline__ void st(uint8_t *p, u128 a) {
  *reinterpret_cast<ulonglong2 *>(p) = make_ulonglong2(a.lo, a.hi);
}

// One thread per leaf: fixed-length no-pad sponge, state [0, 0, iv_hi, iv_lo], rate words 0/1.
extern "C" __global__ void merkle_leaves(const uint8_t *data, uint64_t leaf_size, uint64_t n,
                                         uint64_t iv_hi_lo, uint64_t iv_hi_hi, uint64_t iv_lo_lo, uint64_t iv_lo_hi,
                                         uint8_t *out) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u128 s[4] = {{0, 0}, {0, 0}, {iv_hi_lo, iv_hi_hi}, {iv_lo_lo, iv_lo_hi}};
  const uint8_t *p = data + i * leaf_size;
  for (uint64_t b = 0; b < leaf_size; b += 32) {
    s[0] = x(s[0], ld(p + b)); s[1] = x(s[1], ld(p + b + 16));
    permute(s);
  }
  st(out + i * 32, s[0]); st(out + i * 32 + 16, s[1]);
}

// One thread per parent: compress_flat_feed_forward: state [a0, a1, b0^iv_hi, b1^iv_lo], out = s[0..2] ^ a.
extern "C" __global__ void merkle_level(const uint8_t *in, uint64_t n_out,
                                        uint64_t iv_hi_lo, uint64_t iv_hi_hi, uint64_t iv_lo_lo, uint64_t iv_lo_hi,
                                        uint8_t *out) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  const uint8_t *p = in + i * 64;
  u128 a0 = ld(p), a1 = ld(p + 16), b0 = ld(p + 32), b1 = ld(p + 48);
  u128 s[4] = {a0, a1, x(b0, {iv_hi_lo, iv_hi_hi}), x(b1, {iv_lo_lo, iv_lo_hi})};
  permute(s);
  st(out + i * 32, x(s[0], a0)); st(out + i * 32 + 16, x(s[1], a1));
}
