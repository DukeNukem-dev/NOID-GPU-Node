//! amun: Merkle trees on an NVIDIA GPU (sm_86+, CLMAD) through the CUDA driver API.
//!
//! libcuda is opened at runtime (dlopen), so the node builds and runs without a CUDA toolkit or GPU;
//! any failure falls back to the CPU tree. Only the fixed-length no-pad leaf mode is offloaded
//! (`leaf_size % 32 == 0`, the production PCS shape). Every GPU tree is spot-checked on the CPU
//! (random leaves and parents); a mismatch disables the GPU for the rest of the process.
//!
//! `NOID_GPU=0` disables it; `NOID_GPU_MIN_LEAVES` (default 1024) sets the smallest offloaded tree.
//! Kernel source: noid_merkle.cu (fatbin for sm_86 plus compute_86 PTX, built with CUDA 13.4).

use super::{Hash, POSEIDON_MERKLE_NODE_TAG, capacity_iv_flat, hash_leaf, hash_pair, leaf_fixed_iv_flat};
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static FATBIN: &[u8] = include_bytes!("noid_merkle.fatbin");

type CuResult = c_int;
type CuDevice = c_int;
type CuCtx = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevPtr = u64;

struct Api {
    ctx_set_current: unsafe extern "C" fn(CuCtx) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut CuDevPtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuDevPtr) -> CuResult,
    htod: unsafe extern "C" fn(CuDevPtr, *const c_void, usize) -> CuResult,
    dtoh: unsafe extern "C" fn(*mut c_void, CuDevPtr, usize) -> CuResult,
    launch: unsafe extern "C" fn(
        CuFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void,
        *mut *mut c_void, *mut *mut c_void,
    ) -> CuResult,
    sync: unsafe extern "C" fn() -> CuResult,
}

struct Gpu {
    api: Api,
    ctx: CuCtx,
    k_leaves: CuFunction,
    k_level: CuFunction,
    // Grow-only device buffers: leaf data and the whole tree.
    d_data: CuDevPtr,
    cap_data: usize,
    d_tree: CuDevPtr,
    cap_tree: usize,
}
// The raw CUDA handles are only used under the Mutex below.
unsafe impl Send for Gpu {}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();
static BROKEN: AtomicBool = AtomicBool::new(false);
static TREES: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

fn min_leaves() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("NOID_GPU_MIN_LEAVES").ok().and_then(|v| v.parse().ok()).unwrap_or(1024)
    })
}

fn check(r: CuResult, what: &str) -> Result<(), String> {
    if r == 0 { Ok(()) } else { Err(format!("{what} failed (CUresult {r})")) }
}

unsafe fn sym<T>(lib: *mut c_void, name: &CStr) -> Result<T, String> {
    let p = unsafe { dlsym(lib, name.as_ptr()) };
    if p.is_null() {
        return Err(format!("libcuda has no {name:?}"));
    }
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
}

fn init() -> Result<Gpu, String> {
    unsafe {
        let lib = dlopen(c"libcuda.so.1".as_ptr(), 2 /* RTLD_NOW */);
        if lib.is_null() {
            return Err("libcuda.so.1 not found".into());
        }
        let cu_init: unsafe extern "C" fn(c_uint) -> CuResult = sym(lib, c"cuInit")?;
        let device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult = sym(lib, c"cuDeviceGet")?;
        let device_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult =
            sym(lib, c"cuDeviceGetName")?;
        let primary_retain: unsafe extern "C" fn(*mut CuCtx, CuDevice) -> CuResult =
            sym(lib, c"cuDevicePrimaryCtxRetain")?;
        let module_load: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult =
            sym(lib, c"cuModuleLoadData")?;
        let get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult =
            sym(lib, c"cuModuleGetFunction")?;
        let api = Api {
            ctx_set_current: sym(lib, c"cuCtxSetCurrent")?,
            mem_alloc: sym(lib, c"cuMemAlloc_v2")?,
            mem_free: sym(lib, c"cuMemFree_v2")?,
            htod: sym(lib, c"cuMemcpyHtoD_v2")?,
            dtoh: sym(lib, c"cuMemcpyDtoH_v2")?,
            launch: sym(lib, c"cuLaunchKernel")?,
            sync: sym(lib, c"cuCtxSynchronize")?,
        };
        check(cu_init(0), "cuInit")?;
        let mut dev: CuDevice = 0;
        check(device_get(&mut dev, 0), "cuDeviceGet")?;
        let mut name = [0 as c_char; 128];
        check(device_name(name.as_mut_ptr(), 127, dev), "cuDeviceGetName")?;
        let mut ctx: CuCtx = std::ptr::null_mut();
        check(primary_retain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
        check((api.ctx_set_current)(ctx), "cuCtxSetCurrent")?;
        let mut module: CuModule = std::ptr::null_mut();
        check(module_load(&mut module, FATBIN.as_ptr().cast()), "cuModuleLoadData")?;
        let mut k_leaves: CuFunction = std::ptr::null_mut();
        let mut k_level: CuFunction = std::ptr::null_mut();
        check(get_function(&mut k_leaves, module, c"merkle_leaves".as_ptr()), "merkle_leaves")?;
        check(get_function(&mut k_level, module, c"merkle_level".as_ptr()), "merkle_level")?;
        let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
        eprintln!("amun gpu merkle: enabled on {name}");
        Ok(Gpu { api, ctx, k_leaves, k_level, d_data: 0, cap_data: 0, d_tree: 0, cap_tree: 0 })
    }
}

fn gpu() -> Option<&'static Mutex<Gpu>> {
    GPU.get_or_init(|| {
        if std::env::var("NOID_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        match init() {
            Ok(g) => Some(Mutex::new(g)),
            Err(e) => {
                eprintln!("amun gpu merkle: disabled ({e})");
                None
            }
        }
    })
    .as_ref()
}

fn split(v: u128) -> (u64, u64) {
    (v as u64, (v >> 64) as u64)
}

impl Gpu {
    fn ensure(&mut self, data_len: usize, tree_len: usize) -> Result<(), String> {
        unsafe {
            if self.cap_data < data_len {
                if self.d_data != 0 {
                    (self.api.mem_free)(self.d_data);
                    self.d_data = 0;
                    self.cap_data = 0;
                }
                check((self.api.mem_alloc)(&mut self.d_data, data_len), "cuMemAlloc data")?;
                self.cap_data = data_len;
            }
            if self.cap_tree < tree_len {
                if self.d_tree != 0 {
                    (self.api.mem_free)(self.d_tree);
                    self.d_tree = 0;
                    self.cap_tree = 0;
                }
                check((self.api.mem_alloc)(&mut self.d_tree, tree_len), "cuMemAlloc tree")?;
                self.cap_tree = tree_len;
            }
        }
        Ok(())
    }

    fn launch(&self, f: CuFunction, n: u64, args: &mut [*mut c_void]) -> Result<(), String> {
        const BLOCK: u64 = 128;
        let grid = n.div_ceil(BLOCK);
        let r = unsafe {
            (self.api.launch)(
                f, grid as c_uint, 1, 1, BLOCK as c_uint, 1, 1, 0, std::ptr::null_mut(),
                args.as_mut_ptr(), std::ptr::null_mut(),
            )
        };
        check(r, "cuLaunchKernel")
    }

    fn tree(&mut self, data: &[u8], num_leaves: usize, leaf_size: usize, tree: &mut [Hash]) -> Result<(), String> {
        let tree_bytes = tree.len() * 32;
        unsafe {
            check((self.api.ctx_set_current)(self.ctx), "cuCtxSetCurrent")?;
        }
        self.ensure(data.len(), tree_bytes)?;
        unsafe {
            check((self.api.htod)(self.d_data, data.as_ptr().cast(), data.len()), "cuMemcpyHtoD")?;
        }
        let (ivh_lo, ivh_hi) = split(leaf_fixed_iv_flat(leaf_size)[0]);
        let (ivl_lo, ivl_hi) = split(leaf_fixed_iv_flat(leaf_size)[1]);
        let mut d_data = self.d_data;
        let mut ls = leaf_size as u64;
        let mut n = num_leaves as u64;
        let (mut a, mut b, mut c, mut d) = (ivh_lo, ivh_hi, ivl_lo, ivl_hi);
        let mut d_out = self.d_tree;
        let mut args: [*mut c_void; 8] = [
            (&raw mut d_data).cast(), (&raw mut ls).cast(), (&raw mut n).cast(), (&raw mut a).cast(),
            (&raw mut b).cast(), (&raw mut c).cast(), (&raw mut d).cast(), (&raw mut d_out).cast(),
        ];
        self.launch(self.k_leaves, n, &mut args)?;

        let [nh, nl] = capacity_iv_flat(POSEIDON_MERKLE_NODE_TAG);
        let (mut a, mut b) = split(nh);
        let (mut c, mut d) = split(nl);
        let mut read_start = 0u64;
        let mut read_len = num_leaves as u64;
        while read_len > 1 {
            let mut n_out = read_len >> 1;
            let mut d_in = self.d_tree + read_start * 32;
            let mut d_out = self.d_tree + (read_start + read_len) * 32;
            let mut args: [*mut c_void; 7] = [
                (&raw mut d_in).cast(), (&raw mut n_out).cast(), (&raw mut a).cast(), (&raw mut b).cast(),
                (&raw mut c).cast(), (&raw mut d).cast(), (&raw mut d_out).cast(),
            ];
            self.launch(self.k_level, n_out, &mut args)?;
            read_start += read_len;
            read_len = n_out;
        }
        unsafe {
            check((self.api.sync)(), "kernels")?;
            check((self.api.dtoh)(tree.as_mut_ptr().cast(), self.d_tree, tree_bytes), "cuMemcpyDtoH")?;
        }
        Ok(())
    }
}

/// Spot-check a GPU tree on the CPU: 8 leaves and 8 parents at pseudo-random positions per call.
fn spot_check(data: &[u8], num_leaves: usize, leaf_size: usize, tree: &[Hash], seed: u64) -> bool {
    let mut x = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..8 {
        let i = (next() as usize) % num_leaves;
        if hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]) != tree[i] {
            return false;
        }
    }
    // Parent p (any level) sits at index >= num_leaves; its children are found by walking the levels.
    let total = tree.len();
    for _ in 0..8 {
        let p = num_leaves + (next() as usize) % (total - num_leaves);
        let (mut start, mut len) = (0usize, num_leaves);
        while p >= start + len {
            start += len;
            len >>= 1;
        }
        // p is in the level starting at `start`; its children are in the level below.
        let below = start - 2 * len;
        let k = p - start;
        if hash_pair(&tree[below + 2 * k], &tree[below + 2 * k + 1]) != tree[p] {
            return false;
        }
    }
    true
}

/// Build the tree on the GPU when possible. `None` means: use the CPU path.
pub(super) fn try_merkle_tree(data: &[u8], num_leaves: usize, tree: &mut [Hash]) -> Option<()> {
    let leaf_size = data.len() / num_leaves;
    if num_leaves < min_leaves() || leaf_size == 0 || leaf_size % 32 != 0 || BROKEN.load(Ordering::Relaxed) {
        return None;
    }
    // Never wait for the GPU: if another thread is using it, this tree goes to the CPU.
    let mut g = gpu()?.try_lock().ok()?;
    if let Err(e) = g.tree(data, num_leaves, leaf_size, tree) {
        eprintln!("amun gpu merkle: error, using CPU from now on ({e})");
        BROKEN.store(true, Ordering::Relaxed);
        return None;
    }
    drop(g);
    let k = TREES.fetch_add(1, Ordering::Relaxed);
    if !spot_check(data, num_leaves, leaf_size, tree, k.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ data.len() as u64) {
        eprintln!("amun gpu merkle: SELF-CHECK MISMATCH (n_leaves={num_leaves} leaf_size={leaf_size}), GPU disabled");
        BROKEN.store(true, Ordering::Relaxed);
        return None;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::super::{merkle_tree, merkle_tree_sequential};

    fn data(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    /// Needs a GPU; without one the CPU path runs and the comparison is trivially equal.
    #[test]
    fn gpu_tree_matches_sequential() {
        for &(n, leaf) in &[(1024usize, 32usize), (1024, 512), (4096, 64), (65536, 512), (2048, 4096)] {
            let d = data(n * leaf, (n * 31 + leaf) as u64);
            let mut t = vec![[0u8; 32]; 2 * n - 1];
            let used = super::try_merkle_tree(&d, n, &mut t).is_some();
            let cpu = merkle_tree_sequential(&d, n);
            if used {
                assert_eq!(t, cpu, "gpu tree differs: n={n} leaf={leaf}");
            }
            assert_eq!(merkle_tree(&d, n), cpu, "merkle_tree differs: n={n} leaf={leaf}");
            eprintln!("n={n} leaf={leaf} gpu={used}");
        }
    }
}
