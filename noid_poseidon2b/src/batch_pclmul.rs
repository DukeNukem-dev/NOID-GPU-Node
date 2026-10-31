// amun/proving: Poseidon2b flat-basis kernel for x86-64 PCLMULQDQ without VPCLMULQDQ (e.g. Zen 2).
// Bit-exact with `native::permutation::permute_flat_u128` (checked in tests and in ~/noid/p2bfast).
// Values stay in XMM registers; products use Karatsuba (3 PCLMUL) and a PCLMUL-by-0x87 reduction;
// squares use 2 PCLMUL; the full MDS sums unreduced products and reduces once per output.
// Disable with NOID_P2B_FAST=0.

use core::arch::x86_64::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

use noid_core::hardware::tower_to_flat_u128;
use noid_core::packed::PackedBlock128;
use noid_core::Block128;

use crate::native::permutation::{
    F_ROUNDS, MDS_FULL, MDS_PARTIAL, N_ROUNDS, P_ROUNDS, ROUND_CONSTANTS, STATE_SIZE,
};

type V = __m128i;

struct Tables {
    rc: [[u128; STATE_SIZE]; N_ROUNDS],
    full: [[u128; STATE_SIZE]; STATE_SIZE],
    full_one: [[bool; STATE_SIZE]; STATE_SIZE],
    dm1: [u128; STATE_SIZE],
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut rc = [[0u128; STATE_SIZE]; N_ROUNDS];
        for r in 0..N_ROUNDS {
            for i in 0..STATE_SIZE {
                rc[r][i] = tower_to_flat_u128(ROUND_CONSTANTS[i][r]);
            }
        }
        let mut full = [[0u128; STATE_SIZE]; STATE_SIZE];
        let mut full_one = [[false; STATE_SIZE]; STATE_SIZE];
        let mut dm1 = [0u128; STATE_SIZE];
        for i in 0..STATE_SIZE {
            for j in 0..STATE_SIZE {
                full[i][j] = tower_to_flat_u128(MDS_FULL[i][j]);
                full_one[i][j] = MDS_FULL[i][j] == 1;
                if i != j {
                    assert_eq!(MDS_PARTIAL[i][j], 1, "partial MDS must be J + diag");
                }
            }
            dm1[i] = tower_to_flat_u128(MDS_PARTIAL[i][i]) ^ tower_to_flat_u128(1);
        }
        Tables { rc, full, full_one, dm1 }
    })
}

static BROKEN: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);

/// True when this kernel should run: PCLMUL backend selected, not disabled, no self-check failure.
#[inline]
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        noid_core::cpu::pclmul_available()
            && std::env::var("NOID_P2B_FAST").map_or(true, |v| v != "0")
    }) && !BROKEN.load(Ordering::Relaxed)
}

/// Sampled self-check: every 4096th call also runs the generic schedule on one state.
/// A mismatch disables this kernel for the rest of the process (generic code takes over).
#[inline(always)]
fn sample() -> bool {
    CALLS.fetch_add(1, Ordering::Relaxed) & 4095 == 0
}

fn check(input: &[u128; STATE_SIZE], output: &[u128; STATE_SIZE]) {
    let mut reference = *input;
    crate::native::permutation::permute_flat_u128_generic(&mut reference);
    if &reference != output {
        BROKEN.store(true, Ordering::Relaxed);
        eprintln!("noid-p2b-fast: SELF-CHECK MISMATCH, fast Poseidon2b kernel disabled");
    }
}

#[inline(always)]
unsafe fn ld(x: u128) -> V {
    core::mem::transmute(x)
}
#[inline(always)]
unsafe fn st(v: V) -> u128 {
    core::mem::transmute(v)
}

#[inline(always)]
unsafe fn mul_wide(a: V, b: V) -> (V, V) {
    let p0 = _mm_clmulepi64_si128(a, b, 0x00);
    let p1 = _mm_clmulepi64_si128(a, b, 0x11);
    let am = _mm_xor_si128(a, _mm_shuffle_epi32(a, 0x4e));
    let bm = _mm_xor_si128(b, _mm_shuffle_epi32(b, 0x4e));
    let m = _mm_xor_si128(_mm_clmulepi64_si128(am, bm, 0x00), _mm_xor_si128(p0, p1));
    (
        _mm_xor_si128(p0, _mm_slli_si128(m, 8)),
        _mm_xor_si128(p1, _mm_srli_si128(m, 8)),
    )
}

#[inline(always)]
unsafe fn reduce(lo: V, hi: V) -> V {
    let r = _mm_set_epi64x(0, 0x87);
    let t1 = _mm_clmulepi64_si128(hi, r, 0x01);
    let lo = _mm_xor_si128(lo, _mm_slli_si128(t1, 8));
    let h0 = _mm_xor_si128(_mm_move_epi64(hi), _mm_srli_si128(t1, 8));
    let t0 = _mm_clmulepi64_si128(h0, r, 0x00);
    let t2 = _mm_clmulepi64_si128(h0, r, 0x01);
    _mm_xor_si128(_mm_xor_si128(lo, t0), _mm_slli_si128(t2, 8))
}

#[inline(always)]
unsafe fn mul(a: V, b: V) -> V {
    let (l, h) = mul_wide(a, b);
    reduce(l, h)
}

#[inline(always)]
unsafe fn sq(a: V) -> V {
    reduce(_mm_clmulepi64_si128(a, a, 0x00), _mm_clmulepi64_si128(a, a, 0x11))
}

#[inline(always)]
unsafe fn sbox(x: V) -> V {
    let x2 = sq(x);
    let x4 = sq(x2);
    mul(mul(x, x2), x4)
}

#[inline(always)]
unsafe fn mds_full(s: &mut [V; STATE_SIZE], t: &Tables) {
    let x = *s;
    for i in 0..STATE_SIZE {
        let (mut lo, mut hi, mut plain) = (_mm_setzero_si128(), _mm_setzero_si128(), _mm_setzero_si128());
        for j in 0..STATE_SIZE {
            if t.full_one[i][j] {
                plain = _mm_xor_si128(plain, x[j]);
            } else {
                let (l, h) = mul_wide(x[j], ld(t.full[i][j]));
                lo = _mm_xor_si128(lo, l);
                hi = _mm_xor_si128(hi, h);
            }
        }
        s[i] = _mm_xor_si128(reduce(lo, hi), plain);
    }
}

#[inline(always)]
unsafe fn mds_partial(s: &mut [V; STATE_SIZE], t: &Tables) {
    let sum = _mm_xor_si128(_mm_xor_si128(s[0], s[1]), _mm_xor_si128(s[2], s[3]));
    for i in 0..STATE_SIZE {
        s[i] = _mm_xor_si128(sum, mul(s[i], ld(t.dm1[i])));
    }
}

/// N independent permutations, interleaved round by round.
#[inline(always)]
unsafe fn permute_n<const N: usize>(s: &mut [[V; STATE_SIZE]; N], t: &Tables) {
    for k in 0..N {
        mds_full(&mut s[k], t);
    }
    for r in 0..N_ROUNDS {
        if !(F_ROUNDS / 2..F_ROUNDS / 2 + P_ROUNDS).contains(&r) {
            for k in 0..N {
                for i in 0..STATE_SIZE {
                    s[k][i] = sbox(_mm_xor_si128(s[k][i], ld(t.rc[r][i])));
                }
            }
            for k in 0..N {
                mds_full(&mut s[k], t);
            }
        } else {
            for k in 0..N {
                s[k][0] = sbox(_mm_xor_si128(s[k][0], ld(t.rc[r][0])));
            }
            for k in 0..N {
                mds_partial(&mut s[k], t);
            }
        }
    }
}

/// Scalar flat permutation (transcript chains).
#[target_feature(enable = "pclmulqdq")]
pub(crate) unsafe fn permute_flat_single_u128(flat: &mut [u128; STATE_SIZE]) {
    let t = tables();
    let input = *flat;
    let mut s = [core::array::from_fn(|i| ld(flat[i]))];
    permute_n::<1>(&mut s, t);
    for i in 0..STATE_SIZE {
        flat[i] = st(s[0][i]);
    }
    if sample() {
        check(&input, flat);
    }
}

#[inline(always)]
fn lanes(p: PackedBlock128) -> [u128; 2] {
    let a = p.to_array();
    [a[0].0, a[1].0]
}

#[inline(always)]
fn pack(a: u128, b: u128) -> PackedBlock128 {
    PackedBlock128::from_array([Block128(a), Block128(b)])
}

/// One packed state = two permutations.
#[target_feature(enable = "pclmulqdq")]
pub(crate) unsafe fn permute_flat_one(states: &mut [PackedBlock128; STATE_SIZE]) {
    let t = tables();
    let l: [[u128; 2]; STATE_SIZE] = core::array::from_fn(|i| lanes(states[i]));
    let mut s: [[V; STATE_SIZE]; 2] = core::array::from_fn(|k| core::array::from_fn(|i| ld(l[i][k])));
    permute_n::<2>(&mut s, t);
    for i in 0..STATE_SIZE {
        states[i] = pack(st(s[0][i]), st(s[1][i]));
    }
}

/// Many packed states: two packed states (four permutations) per interleaved group.
#[target_feature(enable = "pclmulqdq")]
pub(crate) unsafe fn permute_flat_groups(states: &mut [[PackedBlock128; STATE_SIZE]]) {
    let t = tables();
    let probe: Option<[u128; STATE_SIZE]> = (sample() && !states.is_empty())
        .then(|| core::array::from_fn(|i| lanes(states[0][i])[0]));
    let mut chunks = states.chunks_exact_mut(2);
    for pair in &mut chunks {
        let la: [[u128; 2]; STATE_SIZE] = core::array::from_fn(|i| lanes(pair[0][i]));
        let lb: [[u128; 2]; STATE_SIZE] = core::array::from_fn(|i| lanes(pair[1][i]));
        let mut s: [[V; STATE_SIZE]; 4] = core::array::from_fn(|k| {
            core::array::from_fn(|i| ld(if k < 2 { la[i][k] } else { lb[i][k - 2] }))
        });
        permute_n::<4>(&mut s, t);
        for i in 0..STATE_SIZE {
            pair[0][i] = pack(st(s[0][i]), st(s[1][i]));
            pair[1][i] = pack(st(s[2][i]), st(s[3][i]));
        }
    }
    for one in chunks.into_remainder() {
        permute_flat_one(one);
    }
    if let Some(input) = probe {
        let out: [u128; STATE_SIZE] = core::array::from_fn(|i| lanes(states[0][i])[0]);
        check(&input, &out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix(x: &mut u64) -> u64 {
        *x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    #[test]
    fn fast_kernel_matches_generic_permutation() {
        if !std::arch::is_x86_feature_detected!("pclmulqdq") {
            return;
        }
        let mut seed = 99u64;
        let mut r = || ((splitmix(&mut seed) as u128) << 64) | splitmix(&mut seed) as u128;
        for n in 0..20_000 {
            let mut a: [u128; STATE_SIZE] = core::array::from_fn(|_| r());
            if n % 5 == 0 {
                a[n % STATE_SIZE] = 0;
            }
            if n % 13 == 0 {
                a = [u128::MAX; STATE_SIZE];
            }
            let mut b = a;
            crate::native::permutation::permute_flat_u128_generic(&mut a);
            unsafe { permute_flat_single_u128(&mut b) };
            assert_eq!(a, b, "scalar mismatch at {n}");
        }
        // packed group path: 5 packed states (two groups + remainder) = 10 permutations
        let mut packed: Vec<[PackedBlock128; STATE_SIZE]> =
            (0..5).map(|_| core::array::from_fn(|_| pack(r(), r()))).collect();
        let expect: Vec<[[u128; STATE_SIZE]; 2]> = packed
            .iter()
            .map(|s| {
                core::array::from_fn(|k| {
                    let mut x: [u128; STATE_SIZE] = core::array::from_fn(|i| lanes(s[i])[k]);
                    crate::native::permutation::permute_flat_u128_generic(&mut x);
                    x
                })
            })
            .collect();
        unsafe { permute_flat_groups(&mut packed) };
        for (s, e) in packed.iter().zip(expect.iter()) {
            for k in 0..2 {
                let got: [u128; STATE_SIZE] = core::array::from_fn(|i| lanes(s[i])[k]);
                assert_eq!(got, e[k]);
            }
        }
    }
}
