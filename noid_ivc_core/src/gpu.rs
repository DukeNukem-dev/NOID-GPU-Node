//! amun: shared NVIDIA GPU access for noid_ivc_core (sm_86+, CLMAD), through the CUDA driver API.
//!
//! libcuda is opened at runtime (dlopen), so the node builds and runs without a CUDA toolkit or a GPU;
//! callers fall back to their CPU paths whenever [`lock`] returns `None`. One process-wide context,
//! one module (`noid_gpu.fatbin`: sm_86 SASS + compute_86 PTX, built with CUDA 13.4 from
//! `noid_ivc_core/cuda/noid_gpu.cu`), one mutex. Users never wait for the GPU: `lock` is a try-lock.
//!
//! `NOID_GPU=0` disables every GPU path.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

static FATBIN: &[u8] = include_bytes!("noid_gpu.fatbin");

type CuResult = c_int;
type CuDevice = c_int;
type CuCtx = *mut c_void;
type CuModule = *mut c_void;
pub(crate) type CuFunction = *mut c_void;
pub(crate) type DevPtr = u64;

pub(crate) struct Api {
    ctx_set_current: unsafe extern "C" fn(CuCtx) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut DevPtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(DevPtr) -> CuResult,
    htod: unsafe extern "C" fn(DevPtr, *const c_void, usize) -> CuResult,
    dtoh: unsafe extern "C" fn(*mut c_void, DevPtr, usize) -> CuResult,
    memset: unsafe extern "C" fn(DevPtr, u8, usize) -> CuResult,
    launch: unsafe extern "C" fn(
        CuFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void,
        *mut *mut c_void, *mut *mut c_void,
    ) -> CuResult,
    sync: unsafe extern "C" fn() -> CuResult,
    get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
}

/// Grow-only device buffer.
#[derive(Default)]
pub(crate) struct DBuf {
    pub ptr: DevPtr,
    pub cap: usize,
}

pub(crate) struct Gpu {
    api: Api,
    ctx: CuCtx,
    module: CuModule,
    pub name: String,
    /// Per-user state, owned here so it lives under the same lock.
    pub merkle: Vec<DBuf>,
    pub fold: Option<Box<dyn std::any::Any + Send>>,
}
// The raw CUDA handles are only used while holding the mutex.
unsafe impl Send for Gpu {}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

pub(crate) fn check(r: CuResult, what: &str) -> Result<(), String> {
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
        let api = Api {
            ctx_set_current: sym(lib, c"cuCtxSetCurrent")?,
            mem_alloc: sym(lib, c"cuMemAlloc_v2")?,
            mem_free: sym(lib, c"cuMemFree_v2")?,
            htod: sym(lib, c"cuMemcpyHtoD_v2")?,
            dtoh: sym(lib, c"cuMemcpyDtoH_v2")?,
            memset: sym(lib, c"cuMemsetD8_v2")?,
            launch: sym(lib, c"cuLaunchKernel")?,
            sync: sym(lib, c"cuCtxSynchronize")?,
            get_function: sym(lib, c"cuModuleGetFunction")?,
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
        let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
        eprintln!("amun gpu: enabled on {name}");
        Ok(Gpu { api, ctx, module, name, merkle: Vec::new(), fold: None })
    }
}

/// Try to take the GPU. `None`: no GPU, disabled, or busy (use the CPU path).
pub(crate) fn lock() -> Option<MutexGuard<'static, Gpu>> {
    let gpu = GPU
        .get_or_init(|| {
            if std::env::var("NOID_GPU").is_ok_and(|v| v == "0") {
                return None;
            }
            match init() {
                Ok(g) => Some(Mutex::new(g)),
                Err(e) => {
                    eprintln!("amun gpu: disabled ({e})");
                    None
                }
            }
        })
        .as_ref()?;
    let g = gpu.try_lock().ok()?;
    // The context must be current on the calling thread.
    if unsafe { (g.api.ctx_set_current)(g.ctx) } != 0 {
        return None;
    }
    Some(g)
}

/// A disable switch per GPU user, set on the first error or self-check mismatch.
pub(crate) struct Breaker(AtomicBool);
impl Breaker {
    pub const fn new() -> Self {
        Self(AtomicBool::new(false))
    }
    pub fn broken(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
    pub fn trip(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl Gpu {
    pub fn function(&self, name: &CStr) -> Result<CuFunction, String> {
        let mut f: CuFunction = std::ptr::null_mut();
        check(unsafe { (self.api.get_function)(&mut f, self.module, name.as_ptr()) }, "cuModuleGetFunction")?;
        Ok(f)
    }

    pub fn ensure(&self, buf: &mut DBuf, bytes: usize) -> Result<DevPtr, String> {
        if buf.cap < bytes {
            if buf.ptr != 0 {
                unsafe { (self.api.mem_free)(buf.ptr) };
                *buf = DBuf::default();
            }
            check(unsafe { (self.api.mem_alloc)(&mut buf.ptr, bytes.max(256)) }, "cuMemAlloc")?;
            buf.cap = bytes.max(256);
        }
        Ok(buf.ptr)
    }

    pub fn free(&self, buf: &mut DBuf) {
        if buf.ptr != 0 {
            unsafe { (self.api.mem_free)(buf.ptr) };
        }
        *buf = DBuf::default();
    }

    pub fn htod<T: Copy>(&self, dst: DevPtr, src: &[T]) -> Result<(), String> {
        check(
            unsafe { (self.api.htod)(dst, src.as_ptr().cast(), std::mem::size_of_val(src)) },
            "cuMemcpyHtoD",
        )
    }

    pub fn dtoh<T: Copy>(&self, dst: &mut [T], src: DevPtr) -> Result<(), String> {
        check(
            unsafe { (self.api.dtoh)(dst.as_mut_ptr().cast(), src, std::mem::size_of_val(dst)) },
            "cuMemcpyDtoH",
        )
    }

    pub fn zero(&self, dst: DevPtr, bytes: usize) -> Result<(), String> {
        check(unsafe { (self.api.memset)(dst, 0, bytes) }, "cuMemsetD8")
    }

    pub fn sync(&self) -> Result<(), String> {
        check(unsafe { (self.api.sync)() }, "cuCtxSynchronize")
    }

    /// Launch `f` on `grid` blocks of `block` threads. `args` holds a pointer to each kernel argument.
    pub fn launch(&self, f: CuFunction, grid: u64, block: u32, args: &mut [*mut c_void]) -> Result<(), String> {
        let grid = grid.clamp(1, 1 << 30) as c_uint;
        check(
            unsafe {
                (self.api.launch)(
                    f, grid, 1, 1, block, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

/// Kernel argument pointer helper: `arg!(x)` = `&raw mut x` as `*mut c_void`.
macro_rules! arg {
    ($x:ident) => {
        (&raw mut $x).cast::<std::ffi::c_void>()
    };
}
pub(crate) use arg;
