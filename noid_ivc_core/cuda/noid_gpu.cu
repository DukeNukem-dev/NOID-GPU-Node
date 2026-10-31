// amun: GPU kernels for noid_ivc_core: Poseidon2b Merkle trees (merkle_gpu.rs) and the C1 matrix-claim fold (matrix_claim/c1_gpu.rs).
// GF(2^128) is GHASH (x^128 + x^7 + x^2 + x + 1, lo = x^0..x^63); GF(2^256) = GF(2^128)[X]/(X^2 + X + tau).
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

// ---------------------------------------------------------------- C1 matrix-claim fold (GF(2^256))
struct f256 { u128 lo, hi; };
__constant__ u128 F256_TAU = {0xFAAFC4143960829Aull, 0xB9E80FCC7CA8FE72ull};
__device__ __forceinline__ f256 fx(f256 a, f256 b) { return {x(a.lo, b.lo), x(a.hi, b.hi)}; }
__device__ __forceinline__ f256 fmul(f256 a, f256 b) {
  u128 p0 = mul(a.lo, b.lo), p1 = mul(a.hi, b.hi), m = mul(x(a.lo, a.hi), x(b.lo, b.hi));
  return {x(p0, mul(p1, F256_TAU)), x(m, p0)};
}
__device__ __forceinline__ f256 fscale(f256 a, u128 s) { return {mul(a.lo, s), mul(a.hi, s)}; }
__device__ __forceinline__ f256 f1p(f256 a) { a.lo.lo ^= 1; return a; }  // 1 + a
#define FZ f256{{0, 0}, {0, 0}}
#define GID ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x)
#define GSTRIDE ((uint64_t)gridDim.x * blockDim.x)

// Block-level XOR reduction of two accumulators; thread 0 writes partials[2*block], [2*block+1].
__device__ __forceinline__ void reduce2(f256 a, f256 b, f256 *partials) {
  __shared__ f256 s0[256], s1[256];
  s0[threadIdx.x] = a; s1[threadIdx.x] = b;
  __syncthreads();
  for (unsigned k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) { s0[threadIdx.x] = fx(s0[threadIdx.x], s0[threadIdx.x + k]); s1[threadIdx.x] = fx(s1[threadIdx.x], s1[threadIdx.x + k]); }
    __syncthreads();
  }
  if (threadIdx.x == 0) { partials[2 * blockIdx.x] = s0[0]; partials[2 * blockIdx.x + 1] = s1[0]; }
}

// out[i + len] = out[i] * r; out[i] *= (1 + r)   (one build_eq_table step)
extern "C" __global__ void eq_step(f256 *out, uint64_t len, f256 r) {
  for (uint64_t i = GID; i < len; i += GSTRIDE) { f256 v = out[i]; out[i + len] = fmul(v, r); out[i] = fmul(v, f1p(r)); }
}
// out[i] = low[i & (2^low_bits - 1)] * rest[i >> low_bits]
extern "C" __global__ void outer(f256 *out, const f256 *low, uint64_t low_bits, const f256 *rest, uint64_t n) {
  uint64_t mask = (1ull << low_bits) - 1;
  for (uint64_t i = GID; i < n; i += GSTRIDE) out[i] = fmul(low[i & mask], rest[i >> low_bits]);
}
// Row images: gv[r] = sum_c M[r,c] * v[c], ge[r] = sum_c M[r,c] * e[c]   (CSR, coefficient dictionary)
extern "C" __global__ void spmv_rows(const uint64_t *rp, const uint32_t *col, const uint32_t *ci, const u128 *dict,
                                     uint64_t nrows, const f256 *v, const f256 *e, f256 *gv, f256 *ge) {
  for (uint64_t r = GID; r < nrows; r += GSTRIDE) {
    f256 av = FZ, ae = FZ;
    for (uint64_t k = rp[r]; k < rp[r + 1]; k++) {
      u128 c = dict[ci[k]]; uint32_t j = col[k];
      av = fx(av, fscale(v[j], c)); ae = fx(ae, fscale(e[j], c));
    }
    gv[r] = av; ge[r] = ae;
  }
}
// Column image: h[c] = sum over stacked rows s of eq(s) * M_hat[s,c], eq(s) = lo[s & mask] * hi[s >> lo_bits]  (CSC)
extern "C" __global__ void colimage(const uint64_t *cp, const uint32_t *srow, const uint32_t *ci, const u128 *dict,
                                    uint64_t ncols, const f256 *eqlo, uint64_t lo_bits, const f256 *eqhi, f256 *h) {
  uint64_t mask = (1ull << lo_bits) - 1;
  for (uint64_t c = GID; c < ncols; c += GSTRIDE) {
    f256 acc = FZ;
    for (uint64_t k = cp[c]; k < cp[c + 1]; k++) {
      uint32_t s = srow[k];
      acc = fx(acc, fscale(fmul(eqlo[s & mask], eqhi[s >> lo_bits]), dict[ci[k]]));
    }
    h[c] = acc;
  }
}
// Column image, load-balanced: thread t takes CSC entries [t*CH, (t+1)*CH), accumulates per column run and
// XORs each run into h with 64-bit atomics (XOR is field addition, so the result is order-independent).
// h must be zeroed first. col_of_entry[k] is the column of entry k.
#define CH 32
__device__ __forceinline__ void xor_into(f256 *dst, f256 v) {
  unsigned long long *d = reinterpret_cast<unsigned long long *>(dst);
  if (v.lo.lo) atomicXor(d + 0, v.lo.lo); if (v.lo.hi) atomicXor(d + 1, v.lo.hi);
  if (v.hi.lo) atomicXor(d + 2, v.hi.lo); if (v.hi.hi) atomicXor(d + 3, v.hi.hi);
}
extern "C" __global__ void colimage_bal(const uint32_t *ecol, const uint32_t *srow, const uint32_t *ci, const u128 *dict,
                                        uint64_t nnz, const f256 *eqlo, uint64_t lo_bits, const f256 *eqhi, f256 *h) {
  uint64_t mask = (1ull << lo_bits) - 1;
  for (uint64_t t = GID; t * CH < nnz; t += GSTRIDE) {
    uint64_t k0 = t * CH, k1 = min(k0 + CH, nnz);
    uint32_t c = ecol[k0]; f256 acc = FZ;
    for (uint64_t k = k0; k < k1; k++) {
      uint32_t cc = ecol[k];
      if (cc != c) { xor_into(&h[c], acc); acc = FZ; c = cc; }
      uint32_t s = srow[k];
      acc = fx(acc, fscale(fmul(eqlo[s & mask], eqhi[s >> lo_bits]), dict[ci[k]]));
    }
    xor_into(&h[c], acc);
  }
}
// dst[i] = lo + ch * (lo + hi)
extern "C" __global__ void fold(f256 *dst, const f256 *src, uint64_t half, f256 ch) {
  for (uint64_t i = GID; i < half; i += GSTRIDE) { f256 lo = src[2 * i], hi = src[2 * i + 1]; dst[i] = fx(lo, fmul(ch, fx(lo, hi))); }
}
// round_coefficients_two_stacked_products, per-block partial sums
extern "C" __global__ void round_stacked(const f256 *u, f256 alpha, const f256 *gv, const f256 *inc, f256 sp,
                                         const f256 *ge, uint64_t side, f256 *partials) {
  f256 a0 = FZ, a1 = FZ;
  for (uint64_t i = GID; i < side / 2; i += GSTRIDE) {
    uint64_t lo = 2 * i, hi = lo + 1;
    f256 fvl = fx(fmul(alpha, gv[lo]), gv[side + lo]);
    f256 fvd = fx(fx(fmul(alpha, fx(gv[lo], gv[hi])), gv[side + lo]), gv[side + hi]);
    f256 ivl = fx(ge[lo], fmul(sp, fx(ge[lo], ge[side + lo])));
    f256 ivd = fx(fx(ge[lo], ge[hi]), fmul(sp, fx(fx(ge[lo], ge[hi]), fx(ge[side + lo], ge[side + hi]))));
    a0 = fx(a0, fx(fmul(u[lo], fvl), fmul(inc[lo], ivl)));
    a1 = fx(a1, fx(fmul(fx(u[lo], u[hi]), fvd), fmul(fx(inc[lo], inc[hi]), ivd)));
  }
  reduce2(a0, a1, partials);
}
// mix_and_round_coefficients_one_product: w += scale * eq (in place), partial sums
extern "C" __global__ void mix_round(f256 *w, const f256 *eq, const f256 *vals, f256 scale, uint64_t pairs, f256 *partials) {
  f256 a0 = FZ, a1 = FZ;
  for (uint64_t i = GID; i < pairs; i += GSTRIDE) {
    f256 lo = fx(w[2 * i], fmul(scale, eq[2 * i])), hi = fx(w[2 * i + 1], fmul(scale, eq[2 * i + 1]));
    w[2 * i] = lo; w[2 * i + 1] = hi;
    a0 = fx(a0, fmul(lo, vals[2 * i])); a1 = fx(a1, fmul(fx(lo, hi), fx(vals[2 * i], vals[2 * i + 1])));
  }
  reduce2(a0, a1, partials);
}
// round_coefficients_one_product, partial sums
extern "C" __global__ void round_one(const f256 *w, const f256 *vals, uint64_t half, f256 *partials) {
  f256 a0 = FZ, a1 = FZ;
  for (uint64_t i = GID; i < half; i += GSTRIDE) {
    f256 wl = w[2 * i], wh = w[2 * i + 1], vl = vals[2 * i], vh = vals[2 * i + 1];
    a0 = fx(a0, fmul(wl, vl)); a1 = fx(a1, fmul(fx(wl, wh), fx(vl, vh)));
  }
  reduce2(a0, a1, partials);
}
// out[i] *= s
extern "C" __global__ void scale(f256 *out, uint64_t n, f256 s) {
  for (uint64_t i = GID; i < n; i += GSTRIDE) out[i] = fmul(out[i], s);
}
// Test hook: out[i] = a[i] * b[i]
extern "C" __global__ void f256_mul_test(const f256 *a, const f256 *b, f256 *out, uint64_t n) {
  for (uint64_t i = GID; i < n; i += GSTRIDE) out[i] = fmul(a[i], b[i]);
}
