//! G2 at the readout level: [`MemorylessContext::read_last`] against a
//! minimal native harness, bit for bit, on LiquidAI's d1-2B (artifact
//! `d1-2b-q4km`).
//!
//! The harness is written directly on llama.cpp's C API. Its types and the
//! native archives come from `llama-cpp-sys`, its functions are declared
//! below, and it uses none of xybrid-llama's wrappers nor `wrapper.cpp`'s
//! model, context, batch or readout shims (only the `-sys` crate's one-time
//! backend initialization, which both sides share in one process). It loads
//! its own instance of the same GGUF file on the CPU (no offload device, no
//! GPU layer), and builds its context as `MemorylessContext`'s is specified:
//! context, batch and microbatch size equal to the token budget, one
//! sequence, one output row, no embeddings, the model's own attention type,
//! Flash Attention on or off, one thread count for prompts and single
//! tokens, no op offload, and llama.cpp's default KV types and `swa_full`.
//! These are built independently and then compared, field by field, with the
//! parameters `wrapper.cpp` gives the readout context
//! (`llama_xybrid_readout_params_c`). Its batch is the server's: the whole
//! prompt as sequence 0 at positions `0..n`, only the last token flagged for
//! output, in `llama_batch_init` storage filled as `common_batch_add` fills
//! it.
//!
//! What must be bit-identical:
//!
//! - every distinct prepared pass of P3a's decision cases whose upstream
//!   evaluation ran the model (`integration-tests/fixtures/decision/cases`),
//!   including the 4,096-token `budget-at-limit` pass, at settings
//!   `cpu-t4-fa-off` and `cpu-t4-fa-on` (four threads, a 4,096-token budget),
//!   and the short ones at one thread too;
//! - a context reused for every pass against a fresh context per pass, on
//!   both sides;
//! - repeats (A/A) and interleaving (A/B/A) in one context;
//! - reordered and repeated selectors, with an earlier readout left intact by
//!   later decodes;
//! - a valid readout after each refusal (over the budget, an empty prompt,
//!   nothing selected, ids outside the vocabulary) against a fresh context's;
//! - readouts after a cancel, before a readout and in the middle of a
//!   2,048-token one that llama.cpp stops between graph nodes, once the
//!   next request has started, with stale handles of earlier requests used
//!   again;
//! - readouts after an injected native failure in the decode shim and in the
//!   logits shim (P2's fault hook throws inside the shim's exception boundary
//!   before its body runs: an exception and status path, not a partially
//!   executed decode, which the cancel case covers);
//! - prompts of 1, budget − 1 and budget tokens at a 512-token budget.
//!
//! `MemorylessContext` registers llama.cpp's abort callback for its cancel
//! handle; the harness registers none. Every comparison therefore also shows
//! that a registered callback that never fires leaves the logits alone.
//!
//! This proves the readout's wiring, and that no state survives a readout,
//! under these pinned conditions: one llama.cpp build, the CPU, these
//! settings. It does not prove prompts or answer arithmetic (gate G0), and
//! says nothing about other builds, backends or devices, whose logits may
//! differ (gate G3).
//!
//! The conditions are checked, not assumed: every ggml backend device must be
//! a CPU device, the staged GGUF must have the pinned size and sha256, and the
//! test prints the build identity: the Cargo profile, target and features,
//! `llama_print_system_info`, the devices, and the llama.cpp archives cargo
//! linked (their sha256, `llama.h`'s, and the CMake build type, compiler and
//! flags that produced them, read from the build script's record). Release
//! builds only: llama.cpp's logits differ between its debug and release
//! builds. A debug build, a missing model or a build with accelerator devices
//! skips each test; `XYBRID_REQUIRE_MODELS` turns a skip into a failure and
//! also requires llama.cpp built from source (`XYBRID_NATIVES_FORCE_SOURCE=1`),
//! whose producer this test can identify. Stage the GGUF as for
//! `d1_readout_smoke` (`integration-tests/fixtures/models/d1-2b-q4km/d1-2B-Q4_K_M.gguf`,
//! or `XYBRID_TEST_MODELS`), then run:
//!
//! ```sh
//! cargo test --release -p xybrid-llama --features bindings --test d1_native_parity -- --nocapture --test-threads=1
//! ```
//!
//! The tests take a lock, so they run one at a time whatever the thread
//! count, and two budget-sized contexts never compete for memory. The whole
//! file takes about fifty minutes on four cores (peak RSS about 5 GB), most
//! of it in the full-corpus sweeps.

#![cfg(feature = "bindings")]

use std::ffi::{c_char, c_int, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use llama_cpp_sys::bindings as sys;
use sha2::{Digest, Sha256};
use xybrid_llama::{backend_init, LlamaError, LlamaModel, MemorylessContext, MemorylessParams};

const MODEL_ID: &str = "d1-2b-q4km";
const MODEL_FILE: &str = "d1-2B-Q4_K_M.gguf";
/// The pinned size and sha256 of `d1-2b-q4km` (`artifacts.json`).
const MODEL_BYTES: u64 = 1_099_629_888;
const MODEL_SHA256: &str = "adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b";

/// A context's settings: the token budget (context, batch and microbatch
/// size), threads, and Flash Attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Setting {
    budget: u32,
    threads: u32,
    flash_attn: bool,
}

/// Settings `cpu-t4-fa-off` of `artifacts.json`.
const CPU_T4_FA_OFF: Setting = Setting {
    budget: 4096,
    threads: 4,
    flash_attn: false,
};
/// Settings `cpu-t4-fa-on`, the product default.
const CPU_T4_FA_ON: Setting = Setting {
    flash_attn: true,
    ..CPU_T4_FA_OFF
};

impl Setting {
    fn with_threads(self, threads: u32) -> Self {
        Self { threads, ..self }
    }

    fn with_budget(self, budget: u32) -> Self {
        Self { budget, ..self }
    }

    fn label(self) -> String {
        format!(
            "budget {}, {} thread{}, flash attention {}",
            self.budget,
            self.threads,
            if self.threads == 1 { "" } else { "s" },
            if self.flash_attn { "on" } else { "off" }
        )
    }
}

// =========================================================================
// The harness: llama.cpp's C API, nothing else
// =========================================================================

// The upstream functions the harness calls. `llama-cpp-sys` generates the
// types from the same `llama.h` and links the archives that define these
// symbols; it does not generate these declarations, because xybrid-llama only
// calls the `_c` shims.
extern "C" {
    fn llama_model_default_params() -> sys::llama_model_params;
    fn llama_model_load_from_file(
        path_model: *const c_char,
        params: sys::llama_model_params,
    ) -> *mut sys::llama_model;
    fn llama_model_free(model: *mut sys::llama_model);
    fn llama_model_get_vocab(model: *const sys::llama_model) -> *const sys::llama_vocab;
    fn llama_vocab_n_tokens(vocab: *const sys::llama_vocab) -> i32;
    fn llama_context_default_params() -> sys::llama_context_params;
    fn llama_init_from_model(
        model: *mut sys::llama_model,
        params: sys::llama_context_params,
    ) -> *mut sys::llama_context;
    fn llama_free(ctx: *mut sys::llama_context);
    fn llama_n_ctx(ctx: *const sys::llama_context) -> u32;
    fn llama_n_batch(ctx: *const sys::llama_context) -> u32;
    fn llama_n_ubatch(ctx: *const sys::llama_context) -> u32;
    fn llama_n_seq_max(ctx: *const sys::llama_context) -> u32;
    fn llama_n_threads(ctx: *mut sys::llama_context) -> i32;
    fn llama_n_threads_batch(ctx: *mut sys::llama_context) -> i32;
    fn llama_get_memory(ctx: *const sys::llama_context) -> sys::llama_memory_t;
    fn llama_get_causal_attn(ctx: *const sys::llama_context) -> bool;
    fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> sys::llama_batch;
    fn llama_batch_free(batch: sys::llama_batch);
    fn llama_decode(ctx: *mut sys::llama_context, batch: sys::llama_batch) -> i32;
    fn llama_get_logits_ith(ctx: *mut sys::llama_context, i: i32) -> *mut f32;
    fn llama_print_system_info() -> *const c_char;
    fn ggml_backend_dev_count() -> usize;
    fn ggml_backend_dev_get(index: usize) -> sys::ggml_backend_dev_t;
    fn ggml_backend_dev_type(device: sys::ggml_backend_dev_t) -> c_int;
    fn ggml_backend_dev_name(device: sys::ggml_backend_dev_t) -> *const c_char;
}

/// `GGML_BACKEND_DEVICE_TYPE_CPU` of `ggml-backend.h`.
const GGML_BACKEND_DEVICE_TYPE_CPU: c_int = 0;

/// The harness's own instance of the GGUF: llama.cpp's default model
/// parameters, no GPU layer and no offload device.
struct HarnessModel {
    ptr: NonNull<sys::llama_model>,
    n_vocab: usize,
}

// SAFETY: llama.cpp lets contexts on any thread read one model; the model is
// only freed by Drop, and every test that uses it holds the serial lock.
unsafe impl Send for HarnessModel {}
// SAFETY: as above; nothing mutates the model after loading.
unsafe impl Sync for HarnessModel {}

impl HarnessModel {
    fn load(path: &Path) -> Self {
        let path = CString::new(path.to_str().expect("a UTF-8 path")).expect("no NUL in the path");
        // An empty, null-terminated device list: nothing is offloaded.
        let mut no_devices: [sys::ggml_backend_dev_t; 1] = [std::ptr::null_mut()];
        // SAFETY: plain value constructors; the path and the device list are
        // live for the call, and llama.cpp copies what it keeps.
        let ptr = unsafe {
            let mut params = llama_model_default_params();
            params.n_gpu_layers = 0;
            params.devices = no_devices.as_mut_ptr();
            llama_model_load_from_file(path.as_ptr(), params)
        };
        let ptr = NonNull::new(ptr).expect("llama.cpp loads the GGUF");
        // SAFETY: the model is live; its vocabulary lives as long as it.
        let n_vocab = unsafe { llama_vocab_n_tokens(llama_model_get_vocab(ptr.as_ptr())) };
        Self {
            ptr,
            n_vocab: usize::try_from(n_vocab).expect("a positive vocabulary size"),
        }
    }
}

impl Drop for HarnessModel {
    fn drop(&mut self) {
        // SAFETY: loaded by `load`, freed once; contexts borrow the model and
        // are dropped first.
        unsafe { llama_model_free(self.ptr.as_ptr()) };
    }
}

/// What llama.cpp built: the same facts `MemorylessContext::facts` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Facts {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    has_memory: bool,
    causal_attn: bool,
}

/// A harness context on [`HarnessModel`].
struct HarnessContext<'m> {
    ptr: NonNull<sys::llama_context>,
    model: &'m HarnessModel,
    budget: usize,
}

impl<'m> HarnessContext<'m> {
    fn new(model: &'m HarnessModel, setting: Setting) -> Self {
        // SAFETY: a plain value constructor.
        let mut params = unsafe { llama_context_default_params() };
        // The KV types and swa_full the readout context inherits from these
        // defaults (wrapper.cpp sets neither). A pin that changes them must
        // be reviewed here, not followed silently.
        assert_eq!(
            (params.type_k, params.type_v, params.swa_full),
            (
                sys::ggml_type_GGML_TYPE_F16,
                sys::ggml_type_GGML_TYPE_F16,
                true
            ),
            "llama.cpp's default KV types and swa_full"
        );
        params.n_ctx = setting.budget;
        params.n_batch = setting.budget;
        params.n_ubatch = setting.budget;
        params.n_seq_max = 1;
        params.n_outputs_max = 1;
        params.n_outputs_max_per_seq = 1;
        params.embeddings = false;
        params.attention_type = sys::llama_attention_type_LLAMA_ATTENTION_TYPE_UNSPECIFIED;
        params.flash_attn_type = if setting.flash_attn {
            sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED
        } else {
            sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED
        };
        let threads = i32::try_from(setting.threads).expect("threads fit an i32");
        params.n_threads = threads;
        params.n_threads_batch = threads;
        // Every operation runs where its weights are, as in the readout context.
        params.op_offload = false;
        // The readout context's parameters, as wrapper.cpp builds them: the
        // harness built its own above, and they must agree field by field.
        // SAFETY: the shim only writes the struct it is given.
        let readout = unsafe {
            let mut readout = llama_context_default_params();
            sys::llama_xybrid_readout_params_c(
                setting.budget,
                setting.budget,
                setting.budget,
                1,
                threads,
                setting.flash_attn,
                &mut readout,
            );
            readout
        };
        assert_eq!(
            format!("{params:?}"),
            format!("{readout:?}"),
            "the harness asks llama.cpp for the readout context's parameters, every field"
        );
        // SAFETY: the model is live and outlives the context ('m).
        let ptr = unsafe { llama_init_from_model(model.ptr.as_ptr(), params) };
        let context = Self {
            ptr: NonNull::new(ptr).expect("llama.cpp creates the harness context"),
            model,
            budget: usize::try_from(setting.budget).expect("a u32 fits usize"),
        };
        let facts = context.facts();
        assert!(
            !facts.has_memory && !facts.causal_attn && facts.n_seq_max == 1,
            "the harness context reads d1 as the readout context must: {facts:?}"
        );
        assert_eq!(
            (facts.n_batch, facts.n_ubatch),
            (setting.budget, setting.budget),
            "batch and microbatch are the budget"
        );
        // SAFETY: the context is live; plain getters.
        let read_back = unsafe {
            (
                llama_n_threads(context.ptr.as_ptr()),
                llama_n_threads_batch(context.ptr.as_ptr()),
            )
        };
        assert_eq!(read_back, (threads, threads), "both thread counts");
        context
    }

    fn facts(&self) -> Facts {
        let ctx = self.ptr.as_ptr();
        // SAFETY: the context is live; these are plain getters.
        unsafe {
            Facts {
                n_ctx: llama_n_ctx(ctx),
                n_batch: llama_n_batch(ctx),
                n_ubatch: llama_n_ubatch(ctx),
                n_seq_max: llama_n_seq_max(ctx),
                has_memory: !llama_get_memory(ctx).is_null(),
                causal_attn: llama_get_causal_attn(ctx),
            }
        }
    }

    /// The logits of `select` at the last of `tokens`, read the way
    /// llama.cpp's server reads a decision prompt.
    fn read_last(&mut self, tokens: &[i32], select: &[i32]) -> Vec<f32> {
        assert!(
            !tokens.is_empty() && tokens.len() <= self.budget,
            "the harness only evaluates prompts inside the budget"
        );
        let in_vocab = |id: &i32| usize::try_from(*id).is_ok_and(|id| id < self.model.n_vocab);
        assert!(
            tokens.iter().chain(select).all(in_vocab),
            "ids in the vocabulary"
        );
        let n = i32::try_from(tokens.len()).expect("the budget fits an i32");
        // SAFETY: llama_batch_init allocates n entries for every array and
        // one sequence id each; the loop writes exactly those entries.
        // llama_decode reads the batch during the call; the logits row lives
        // in the context until its next decode, and is copied out before
        // anything else runs on it.
        unsafe {
            let mut batch = llama_batch_init(n, 0, 1);
            for (i, &token) in tokens.iter().enumerate() {
                *batch.token.add(i) = token;
                *batch.pos.add(i) = i32::try_from(i).expect("positions fit an i32");
                *batch.n_seq_id.add(i) = 1;
                **batch.seq_id.add(i) = 0;
                *batch.logits.add(i) = i8::from(i + 1 == tokens.len());
            }
            batch.n_tokens = n;
            let status = llama_decode(self.ptr.as_ptr(), batch);
            llama_batch_free(batch);
            assert_eq!(status, 0, "llama_decode on a {n}-token prompt");
            let row = llama_get_logits_ith(self.ptr.as_ptr(), n - 1);
            assert!(!row.is_null(), "logits for the last token");
            let row = std::slice::from_raw_parts(row, self.model.n_vocab);
            select
                .iter()
                .map(|&id| row[usize::try_from(id).expect("checked above")])
                .collect()
        }
    }
}

impl Drop for HarnessContext<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `new`, freed once.
        unsafe { llama_free(self.ptr.as_ptr()) };
    }
}

// =========================================================================
// The conditions: model bytes, devices, build identity
// =========================================================================

/// Skips a test with `reason`, or fails it under `XYBRID_REQUIRE_MODELS`.
fn skip<T>(reason: &str) -> Option<T> {
    if std::env::var_os("XYBRID_REQUIRE_MODELS").is_some() {
        panic!("XYBRID_REQUIRE_MODELS is set but {reason}");
    }
    eprintln!("Skipping test: {reason}");
    None
}

fn strict() -> bool {
    std::env::var_os("XYBRID_REQUIRE_MODELS").is_some()
}

fn sha256_file(path: &Path) -> String {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("open a file to hash");
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer).expect("read a file to hash");
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The staged GGUF, checked against its pin, or `None` (with a note) when it
/// is absent or this is a debug build.
fn staged_model_path() -> Option<PathBuf> {
    if cfg!(debug_assertions) {
        return skip("this is a debug build: G2 compares release natives (cargo test --release)");
    }
    let models = std::env::var_os("XYBRID_TEST_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../integration-tests/fixtures/models")
        });
    let path = models.join(MODEL_ID).join(MODEL_FILE);
    if !path.is_file() {
        return skip(&format!(
            "'{MODEL_ID}/{MODEL_FILE}' is not staged under {}",
            models.display()
        ));
    }
    let bytes = std::fs::metadata(&path).expect("stat the GGUF").len();
    assert_eq!(bytes, MODEL_BYTES, "{} is not {MODEL_ID}", path.display());
    assert_eq!(
        sha256_file(&path),
        MODEL_SHA256,
        "{} is not {MODEL_ID}",
        path.display()
    );
    Some(path)
}

/// Every ggml backend device: (name, type).
fn backend_devices() -> Vec<(String, c_int)> {
    // SAFETY: the backend is initialized; the registry outlives the process,
    // and names are static strings.
    unsafe {
        (0..ggml_backend_dev_count())
            .map(|i| {
                let device = ggml_backend_dev_get(i);
                let name = CStr::from_ptr(ggml_backend_dev_name(device))
                    .to_string_lossy()
                    .into_owned();
                (name, ggml_backend_dev_type(device))
            })
            .collect()
    }
}

/// The llama.cpp archives cargo linked into this test, as the build script of
/// `llama-cpp-sys` recorded them (`target/<profile>/build/xybrid-llama-sys-*/output`),
/// and what produced them.
struct NativeIdentity {
    /// `source build`, `prebuilt or staged slice`, or why it is unknown.
    kind: String,
    lines: Vec<String>,
}

fn native_identity() -> NativeIdentity {
    let unknown = |why: &str| NativeIdentity {
        kind: format!("unknown ({why})"),
        lines: Vec::new(),
    };
    let Ok(exe) = std::env::current_exe() else {
        return unknown("no executable path");
    };
    let Some(build_dir) = exe.parent().and_then(Path::parent).map(|p| p.join("build")) else {
        return unknown("no cargo target layout");
    };
    let Ok(entries) = std::fs::read_dir(&build_dir) else {
        return unknown("no cargo build-script records (not a cargo build)");
    };
    let mut candidates = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        let is_sys = dir
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("xybrid-llama-sys-"));
        let Ok(output) = std::fs::read_to_string(dir.join("output")) else {
            continue;
        };
        if !is_sys {
            continue;
        }
        let search: Vec<PathBuf> = output
            .lines()
            .filter_map(|l| l.strip_prefix("cargo:rustc-link-search=native="))
            .map(PathBuf::from)
            .collect();
        let libs: Vec<String> = output
            .lines()
            .filter_map(|l| l.strip_prefix("cargo:rustc-link-lib=static="))
            .map(str::to_string)
            .collect();
        let vision = libs.iter().any(|lib| lib == "mtmd");
        if vision == cfg!(feature = "vision") && !libs.is_empty() {
            candidates.push((dir, search, libs));
        }
    }
    if candidates.len() != 1 {
        return unknown(&format!(
            "{} llama-cpp-sys build records match this test's features",
            candidates.len()
        ));
    }
    let (dir, search, libs) = candidates.remove(0);
    let out = dir.join("out");
    let mut lines = Vec::new();
    let mut all_under_out = true;
    for lib in &libs {
        let name = format!("lib{lib}.a");
        match search.iter().map(|d| d.join(&name)).find(|p| p.is_file()) {
            Some(path) => {
                all_under_out &= path.starts_with(&out);
                lines.push(format!("{name} sha256 {}", sha256_file(&path)));
            }
            None => lines.push(format!("{name} not found in the link search path")),
        }
    }
    let header = out.join("include").join("llama.h");
    if header.is_file() {
        lines.push(format!("llama.h sha256 {}", sha256_file(&header)));
    }
    let cache = out.join("build").join("CMakeCache.txt");
    let source_build = all_under_out && cache.is_file();
    if source_build {
        let text = std::fs::read_to_string(&cache).unwrap_or_default();
        const KEYS: [&str; 22] = [
            "CMAKE_BUILD_TYPE",
            "CMAKE_C_COMPILER",
            "CMAKE_CXX_COMPILER",
            "CMAKE_C_FLAGS",
            "CMAKE_C_FLAGS_RELEASE",
            "CMAKE_CXX_FLAGS",
            "CMAKE_CXX_FLAGS_RELEASE",
            "GGML_NATIVE",
            "GGML_SSE42",
            "GGML_AVX",
            "GGML_AVX2",
            "GGML_AVX512",
            "GGML_BMI2",
            "GGML_FMA",
            "GGML_F16C",
            "GGML_OPENMP",
            "GGML_BLAS",
            "GGML_LLAMAFILE",
            "GGML_CUDA",
            "GGML_METAL",
            "GGML_VULKAN",
            "LLAMA_BUILD_MTMD",
        ];
        for key in KEYS {
            if let Some(value) = text.lines().find_map(|l| {
                l.strip_prefix(key)
                    .filter(|rest| rest.starts_with(':'))
                    .and_then(|rest| rest.split_once('='))
                    .map(|(_, v)| v.to_string())
            }) {
                lines.push(format!("{key}={value}"));
            }
        }
        if let Ok(files) = std::fs::read_dir(out.join("build").join("CMakeFiles")) {
            for compiler in files
                .flatten()
                .map(|e| e.path().join("CMakeCXXCompiler.cmake"))
            {
                if let Ok(text) = std::fs::read_to_string(&compiler) {
                    for key in ["CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION"] {
                        let prefix = format!("set({key} \"");
                        if let Some(v) = text.lines().find_map(|l| l.strip_prefix(prefix.as_str()))
                        {
                            lines.push(format!("{key}={}", v.trim_end_matches("\")")));
                        }
                    }
                }
            }
        }
    }
    NativeIdentity {
        kind: if source_build {
            format!("source build ({})", out.display())
        } else {
            "prebuilt or staged slice: its producer is recorded by the slice, not here".to_string()
        },
        lines,
    }
}

fn print_build_identity(model: &Path, devices: &[(String, c_int)], native: &NativeIdentity) {
    // SAFETY: llama.cpp returns a static, NUL-terminated string.
    let system = unsafe { CStr::from_ptr(llama_print_system_info()) };
    println!(
        "G2 build: {} profile (debug assertions {}), {}-{}-{}, features: bindings{}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        if cfg!(debug_assertions) { "on" } else { "off" },
        std::env::consts::ARCH,
        std::env::consts::OS,
        if cfg!(target_env = "gnu") {
            "gnu"
        } else if cfg!(target_env = "msvc") {
            "msvc"
        } else {
            "other"
        },
        if cfg!(feature = "vision") {
            ", vision"
        } else {
            ""
        },
    );
    println!("G2 llama.cpp: {}", system.to_string_lossy().trim());
    println!("G2 devices: {devices:?} (type 0 = CPU)");
    println!("G2 natives: {}", native.kind);
    for line in &native.lines {
        println!("G2 natives:   {line}");
    }
    println!(
        "G2 model: {MODEL_ID} ({MODEL_BYTES} bytes, sha256 {MODEL_SHA256}, both verified) at {}",
        model.display()
    );
}

// =========================================================================
// The fixture: both model instances, P3a's passes
// =========================================================================

/// One prepared d1 pass of a P3a case.
struct Pass {
    /// `<case id>#<pass index>`.
    name: String,
    tokens: Vec<i32>,
    select: Vec<i32>,
    /// The oracle's logits for this pass (`d1-2b-q4km`, `cpu-t4-fa-off`), as
    /// f32 bits. Another build of the same sources: compared for the record,
    /// not asserted.
    oracle: Vec<u32>,
}

struct Fixture {
    xybrid: LlamaModel,
    harness: HarnessModel,
    passes: Vec<Pass>,
}

impl Fixture {
    fn readout(&self, setting: Setting) -> MemorylessContext {
        MemorylessContext::new(
            &self.xybrid,
            MemorylessParams::new(setting.budget, setting.threads, setting.flash_attn),
        )
        .expect("create a readout context")
    }

    fn harness(&self, setting: Setting) -> HarnessContext<'_> {
        HarnessContext::new(&self.harness, setting)
    }

    fn pass(&self, name: &str) -> &Pass {
        self.passes
            .iter()
            .find(|pass| pass.name == name)
            .unwrap_or_else(|| panic!("P3a's cases have no pass {name}"))
    }
}

/// Both model instances and P3a's passes, loaded once, after the conditions
/// are checked and the build identity printed.
fn fixture() -> Option<&'static Fixture> {
    static FIXTURE: OnceLock<Option<Fixture>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let path = staged_model_path()?;
            backend_init();
            llama_cpp_sys::backend_init();
            let devices = backend_devices();
            let native = native_identity();
            print_build_identity(&path, &devices, &native);
            if devices.is_empty()
                || devices
                    .iter()
                    .any(|(_, kind)| *kind != GGML_BACKEND_DEVICE_TYPE_CPU)
            {
                return skip(&format!(
                    "the native build registers devices other than the CPU ({devices:?}); \
                     G2's cpu-* settings need a CPU-only build"
                ));
            }
            if strict() && !native.kind.starts_with("source build") {
                panic!(
                    "XYBRID_REQUIRE_MODELS needs llama.cpp built from source \
                     (XYBRID_NATIVES_FORCE_SOURCE=1), whose producer this test identifies: {}",
                    native.kind
                );
            }
            let xybrid =
                LlamaModel::load(path.to_str().expect("a UTF-8 path"), 0).expect("load the GGUF");
            let harness = HarnessModel::load(&path);
            assert_eq!(xybrid.n_vocab(), harness.n_vocab, "one vocabulary");
            Some(Fixture {
                passes: model_backed_passes()?,
                xybrid,
                harness,
            })
        })
        .as_ref()
}

/// What this test reads of a P3a case file. Everything else is skipped
/// unparsed, which also keeps the readout's fixtures clear of serde_json's
/// nesting limit (a depth case's legend nests 131 levels).
#[derive(serde::Deserialize)]
struct CaseFile {
    id: String,
    artifact: String,
    settings: Option<String>,
    upstream: CaseUpstream,
}

#[derive(serde::Deserialize)]
struct CaseUpstream {
    #[serde(default)]
    prepare: Option<CaseStage>,
    #[serde(default)]
    evaluate: Option<CaseStage>,
}

/// A stage, or the refusal that replaced it (every field absent).
#[derive(serde::Deserialize)]
struct CaseStage {
    #[serde(default)]
    passes: Option<Vec<CasePass>>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    raw_scores: Option<Vec<Vec<String>>>,
}

#[derive(serde::Deserialize)]
struct CasePass {
    token_ids: Vec<i32>,
    select: Vec<i32>,
    output_index: usize,
}

/// The prepared passes of P3a's cases whose upstream evaluation ran the
/// model, each distinct (tokens, select) pair once, in case order; `None`
/// (with a note) when this tree has no P3a corpus.
fn model_backed_passes() -> Option<Vec<Pass>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../integration-tests/fixtures/decision/cases");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return skip(&format!(
            "P3a's decision cases are not in this tree ({}); the readout comparisons replay \
             their prepared passes",
            dir.display()
        ));
    };
    let mut files: Vec<PathBuf> = entries
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();

    let mut passes: Vec<Pass> = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(&file).expect("read a case");
        let case: CaseFile = serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("{}: {error}", file.display()));
        let id = case.id.as_str();
        let (Some(prepare), Some(evaluate)) = (case.upstream.prepare, case.upstream.evaluate)
        else {
            continue;
        };
        if evaluate.source.as_deref() != Some("model") {
            continue;
        }
        assert_eq!(
            (case.artifact.as_str(), case.settings.as_deref()),
            (MODEL_ID, Some("cpu-t4-fa-off")),
            "{id}: the oracle's logits come from {MODEL_ID} at cpu-t4-fa-off"
        );
        let prepared = prepare.passes.expect("prepared passes");
        let raw_scores = evaluate.raw_scores.expect("raw scores");
        assert_eq!(prepared.len(), raw_scores.len(), "{id}: a readout per pass");
        for (index, (pass, raw)) in prepared.into_iter().zip(raw_scores).enumerate() {
            assert_eq!(
                pass.output_index + 1,
                pass.token_ids.len(),
                "{id}: upstream reads the last token"
            );
            let oracle: Vec<u32> = raw
                .iter()
                .map(|bits| {
                    u32::from_str_radix(bits.trim_start_matches("0x"), 16).expect("hex f32 bits")
                })
                .collect();
            assert_eq!(oracle.len(), pass.select.len(), "{id}: a logit per label");
            if passes
                .iter()
                .any(|seen| seen.tokens == pass.token_ids && seen.select == pass.select)
            {
                continue;
            }
            passes.push(Pass {
                name: format!("{id}#{index}"),
                tokens: pass.token_ids,
                select: pass.select,
                oracle,
            });
        }
    }
    Some(passes)
}

/// The tests share both models and run one at a time.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Where two readouts of one pass differ, for the failure message.
fn difference(name: &str, what: &str, left: &[u32], right: &[u32]) -> Option<String> {
    if left == right {
        return None;
    }
    let first = left
        .iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .unwrap_or(left.len().min(right.len()));
    Some(format!(
        "{name} ({what}): {} vs {} values, first difference at {first}: {:?} vs {:?}",
        left.len(),
        right.len(),
        left.get(first).map(|b| f32::from_bits(*b)),
        right.get(first).map(|b| f32::from_bits(*b)),
    ))
}

fn assert_no_differences(differences: &[String]) {
    assert!(
        differences.is_empty(),
        "{} readouts differ:\n{}",
        differences.len(),
        differences.join("\n")
    );
}

fn verdict(differences: &[String]) -> &'static str {
    if differences.is_empty() {
        "bit-identical"
    } else {
        "DIFFERENT"
    }
}

/// The passes of at most `max_tokens` tokens.
fn short(passes: &[Pass], max_tokens: usize) -> Vec<&Pass> {
    passes
        .iter()
        .filter(|pass| pass.tokens.len() <= max_tokens)
        .collect()
}

/// This process's CPU time (user + system), from `/proc/self/stat` on Linux.
fn process_cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Fields after the parenthesized command name; utime and stime are the
    // 14th and 15th fields, in clock ticks (USER_HZ, 100 on Linux).
    let rest = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    Some((utime + stime) / 100.0)
}

/// Makes the next call of the guarded shim `shim` throw inside its exception
/// boundary, before its body runs (P2's test hook in `wrapper.cpp`).
fn arm_native_fault(shim: &str) {
    let name = CString::new(shim).expect("a shim name");
    // SAFETY: the hook reads the name during the call and keeps no pointer.
    let armed = unsafe { sys::llama_xybrid_fail_next_c(name.as_ptr(), 0) };
    assert!(armed, "{shim} is one of wrapper.cpp's guarded shims");
}

// =========================================================================
// The comparisons
// =========================================================================

#[test]
fn both_sides_build_the_same_context() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    for setting in [
        CPU_T4_FA_OFF,
        CPU_T4_FA_ON,
        CPU_T4_FA_OFF.with_threads(1).with_budget(512),
    ] {
        let started = Instant::now();
        let readout = fx.readout(setting).facts();
        let readout_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let harness = fx.harness(setting).facts();
        let harness_seconds = started.elapsed().as_secs_f64();
        assert_eq!(
            Facts {
                n_ctx: readout.n_ctx,
                n_batch: readout.n_batch,
                n_ubatch: readout.n_ubatch,
                n_seq_max: readout.n_seq_max,
                has_memory: readout.has_memory,
                causal_attn: readout.causal_attn,
            },
            harness,
            "{}",
            setting.label()
        );
        println!(
            "{}: the same facts; a context takes {readout_seconds:.2} s (readout) and \
             {harness_seconds:.2} s (harness) to create and free",
            setting.label()
        );
    }

    // The corpus holds what the comparisons below are meant to cover.
    let longest = fx.passes.iter().map(|pass| pass.tokens.len()).max();
    assert_eq!(longest, Some(4096), "the budget-at-limit pass");
    assert!(
        fx.passes.iter().any(|pass| pass.select.len() > 400),
        "a pass with hundreds of labels (255 options)"
    );
    assert!(
        fx.passes.iter().any(|pass| pass.name.ends_with("#1")),
        "a case with several passes"
    );
    println!(
        "P3a's model-backed cases: {} distinct passes, {} tokens",
        fx.passes.len(),
        fx.passes
            .iter()
            .map(|pass| pass.tokens.len())
            .sum::<usize>()
    );
}

/// Every pass at `setting`: a fresh harness context per pass is the
/// reference; one readout context reused across all passes, one fresh
/// readout context per short pass, and one harness context reused across the
/// short passes must equal it.
fn sweep(fx: &Fixture, setting: Setting, passes: &[&Pass], short_tokens: usize) {
    let started = Instant::now();
    let reference: Vec<Vec<u32>> = passes
        .iter()
        .map(|pass| bits(&fx.harness(setting).read_last(&pass.tokens, &pass.select)))
        .collect();
    let mut differences = Vec::new();

    let mut reused = fx.readout(setting);
    for (pass, expected) in passes.iter().zip(&reference) {
        let got = bits(
            &reused
                .read_last(&pass.tokens, &pass.select)
                .expect("a readout"),
        );
        differences.extend(difference(
            &pass.name,
            "reused readout context",
            &got,
            expected,
        ));
    }
    drop(reused);

    let mut harness_reused = fx.harness(setting);
    let mut short_count = 0;
    for (pass, expected) in passes.iter().zip(&reference) {
        if pass.tokens.len() > short_tokens {
            continue;
        }
        short_count += 1;
        let fresh = bits(
            &fx.readout(setting)
                .read_last(&pass.tokens, &pass.select)
                .expect("a readout"),
        );
        differences.extend(difference(
            &pass.name,
            "fresh readout context",
            &fresh,
            expected,
        ));
        let again = bits(&harness_reused.read_last(&pass.tokens, &pass.select));
        differences.extend(difference(
            &pass.name,
            "reused harness context",
            &again,
            expected,
        ));
    }

    let tokens: usize = passes.iter().map(|pass| pass.tokens.len()).sum();
    println!(
        "{}: {} passes ({tokens} tokens) reused, {short_count} of them also fresh: {} [{:.1} s]",
        setting.label(),
        passes.len(),
        if differences.is_empty() {
            "bit-identical".to_string()
        } else {
            format!("{} differences", differences.len())
        },
        started.elapsed().as_secs_f64()
    );
    if setting == CPU_T4_FA_OFF {
        let same = passes
            .iter()
            .zip(&reference)
            .filter(|(pass, got)| &&pass.oracle == got)
            .count();
        println!(
            "  for the record: {same} of {} passes equal the oracle's committed logits \
             (another build of the same sources)",
            passes.len()
        );
    }
    assert_no_differences(&differences);
}

#[test]
fn p3a_passes_match_the_harness_at_cpu_t4_fa_off() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    let passes: Vec<&Pass> = fx.passes.iter().collect();
    sweep(fx, CPU_T4_FA_OFF, &passes, 512);
}

#[test]
fn p3a_passes_match_the_harness_at_cpu_t4_fa_on() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    let passes: Vec<&Pass> = fx.passes.iter().collect();
    sweep(fx, CPU_T4_FA_ON, &passes, 512);
}

#[test]
fn short_p3a_passes_match_the_harness_at_one_thread() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    let passes = short(&fx.passes, 128);
    for setting in [CPU_T4_FA_OFF, CPU_T4_FA_ON] {
        sweep(fx, setting.with_threads(1), &passes, 128);
    }
}

#[test]
fn repeats_and_interleaving_leave_no_state() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    // A: a short choice; B: a pass three times longer with other labels.
    let a = fx.pass("choice-letter-keys#0");
    let b = fx.pass("choice-36-options#0");
    for setting in [CPU_T4_FA_OFF, CPU_T4_FA_ON] {
        let expected_a = bits(&fx.harness(setting).read_last(&a.tokens, &a.select));
        let expected_b = bits(&fx.harness(setting).read_last(&b.tokens, &b.select));
        let mut differences = Vec::new();

        let mut ctx = fx.readout(setting);
        let mut read = |pass: &Pass| bits(&ctx.read_last(&pass.tokens, &pass.select).unwrap());
        for (step, pass, expected) in [
            ("A", a, &expected_a),
            ("A/A", a, &expected_a),
            ("A/A/B", b, &expected_b),
            ("A/A/B/A", a, &expected_a),
            ("A/A/B/A/B", b, &expected_b),
        ] {
            differences.extend(difference(&pass.name, step, &read(pass), expected));
        }

        let mut harness = fx.harness(setting);
        for (step, pass, expected) in [
            ("harness A", a, &expected_a),
            ("harness A/B", b, &expected_b),
            ("harness A/B/A", a, &expected_a),
        ] {
            let got = bits(&harness.read_last(&pass.tokens, &pass.select));
            differences.extend(difference(&pass.name, step, &got, expected));
        }
        println!(
            "{}: A/A and A/B/A on both sides: {}",
            setting.label(),
            verdict(&differences)
        );
        assert_no_differences(&differences);
    }
}

#[test]
fn selectors_keep_their_order_repeats_and_values() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    let a = fx.pass("choice-letter-keys#0");
    let b = fx.pass("choice-36-options#0");
    let (x, y) = (a.select[0], a.select[2]);
    let reversed: Vec<i32> = a.select.iter().rev().copied().collect();
    let picks: [(&str, Vec<i32>); 4] = [
        ("[a, b, a]", vec![x, y, x]),
        ("[b, a]", vec![y, x]),
        ("[a, a, a, a]", vec![x; 4]),
        ("reversed", reversed),
    ];
    for setting in [CPU_T4_FA_OFF, CPU_T4_FA_ON] {
        let full = bits(&fx.harness(setting).read_last(&a.tokens, &a.select));
        let value_of = |id: i32| {
            full[a
                .select
                .iter()
                .position(|&s| s == id)
                .expect("a selected id")]
        };
        let mut differences = Vec::new();
        let mut ctx = fx.readout(setting);
        let first = ctx.read_last(&a.tokens, &a.select).expect("a readout");
        let kept = bits(&first);
        differences.extend(difference(&a.name, "full selection", &kept, &full));
        for (what, pick) in &picks {
            let got = bits(&ctx.read_last(&a.tokens, pick).expect("a readout"));
            let want: Vec<u32> = pick.iter().map(|&id| value_of(id)).collect();
            differences.extend(difference(&a.name, what, &got, &want));
            let harness = bits(&fx.harness(setting).read_last(&a.tokens, pick));
            differences.extend(difference(
                &a.name,
                &format!("harness {what}"),
                &harness,
                &want,
            ));
        }
        ctx.read_last(&b.tokens, &b.select).expect("a readout of B");
        differences.extend(difference(
            &a.name,
            "an earlier readout after later decodes",
            &bits(&first),
            &kept,
        ));
        println!(
            "{}: reordered and repeated selectors; an earlier readout after later decodes: {}",
            setting.label(),
            verdict(&differences)
        );
        assert_no_differences(&differences);
    }
}

#[test]
fn a_refused_readout_leaves_the_context_as_new() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    let a = fx.pass("choice-letter-keys#0");
    let b = fx.pass("choice-36-options#0");
    let n_vocab = i32::try_from(fx.harness.n_vocab).expect("the vocabulary fits an i32");
    for setting in [CPU_T4_FA_OFF, CPU_T4_FA_ON] {
        let budget = usize::try_from(setting.budget).unwrap();
        let over_budget: Vec<i32> = a.tokens.iter().copied().cycle().take(budget + 1).collect();
        let mut bad_prompt = a.tokens.clone();
        let mut negative_prompt = a.tokens.clone();
        bad_prompt[1] = n_vocab;
        negative_prompt[1] = -1;
        let past_vocab = [n_vocab];
        let negative_select = [a.select[0], -1];
        let refusals: [(&str, &[i32], &[i32]); 7] = [
            ("a prompt over the budget", &over_budget, &a.select),
            ("an empty prompt", &[], &a.select),
            ("nothing selected", &a.tokens, &[]),
            ("a prompt id past the vocabulary", &bad_prompt, &a.select),
            ("a negative prompt id", &negative_prompt, &a.select),
            ("a selected id past the vocabulary", &a.tokens, &past_vocab),
            ("a negative selected id", &a.tokens, &negative_select),
        ];

        let expected = bits(&fx.harness(setting).read_last(&a.tokens, &a.select));
        let fresh = bits(
            &fx.readout(setting)
                .read_last(&a.tokens, &a.select)
                .expect("a readout"),
        );
        let mut differences = Vec::new();
        differences.extend(difference(
            &a.name,
            "fresh readout context",
            &fresh,
            &expected,
        ));

        let mut ctx = fx.readout(setting);
        for (what, tokens, select) in refusals {
            // A readout of B first, so a refusal that left B's row behind, or
            // state of B, would show in the next valid readout of A.
            ctx.read_last(&b.tokens, &b.select).expect("a readout of B");
            let refused = ctx.read_last(tokens, select);
            assert!(
                matches!(refused, Err(LlamaError::InvalidInput(_))),
                "{what}: {:?}",
                refused.map(|values| values.len())
            );
            let after = bits(&ctx.read_last(&a.tokens, &a.select).expect("a readout"));
            differences.extend(difference(&a.name, what, &after, &expected));
        }
        println!(
            "{}: a valid readout after each of 7 refusals: {}",
            setting.label(),
            verdict(&differences)
        );
        assert_no_differences(&differences);
    }
}

#[test]
fn a_cancelled_readout_leaves_the_context_as_new() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    let a = fx.pass("choice-letter-keys#0");
    let source = fx.pass("choice-36-options#0");
    // Long enough (tens of seconds on four threads) for a cancel 300 ms in to
    // land between two of llama.cpp's graph nodes.
    let long: Vec<i32> = source.tokens.iter().copied().cycle().take(2048).collect();
    for setting in [CPU_T4_FA_OFF, CPU_T4_FA_ON] {
        let started = Instant::now();
        let expected_a = bits(&fx.harness(setting).read_last(&a.tokens, &a.select));
        let expected_long = bits(&fx.harness(setting).read_last(&long, &source.select));
        let mut differences = Vec::new();
        let mut ctx = fx.readout(setting);

        // Before a readout: refused without evaluating anything. A handle
        // binds to the current request; clear_cancel starts the next.
        let first_request = ctx.cancel_handle();
        first_request.cancel();
        assert!(matches!(
            ctx.read_last(&a.tokens, &a.select),
            Err(LlamaError::Cancelled)
        ));
        ctx.clear_cancel();
        let after = bits(&ctx.read_last(&a.tokens, &a.select).expect("a readout"));
        differences.extend(difference(
            &a.name,
            "cancelled, then cleared",
            &after,
            &expected_a,
        ));

        // In the middle of one: the worker announces the call, the cancel
        // comes 300 ms later, when only llama.cpp's abort callback can stop
        // the evaluation.
        let cancel = ctx.cancel_handle();
        let calling = AtomicBool::new(false);
        let cpu_before = process_cpu_seconds();
        let (result, call_time) = std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                calling.store(true, Ordering::SeqCst);
                let started = Instant::now();
                let result = ctx.read_last(&long, &source.select);
                (result, started.elapsed())
            });
            while !calling.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            std::thread::sleep(Duration::from_millis(300));
            cancel.cancel();
            worker.join().expect("the readout thread")
        });
        let cpu_spent = cpu_before
            .zip(process_cpu_seconds())
            .map(|(before, after)| after - before);
        assert!(
            matches!(result, Err(LlamaError::Cancelled)),
            "a 2,048-token readout cancelled after 300 ms: {:?} after {call_time:?}",
            result.map(|values| values.len())
        );
        assert!(matches!(
            ctx.read_last(&a.tokens, &a.select),
            Err(LlamaError::Cancelled)
        ));
        ctx.clear_cancel();
        let after = bits(&ctx.read_last(&a.tokens, &a.select).expect("a readout"));
        differences.extend(difference(
            &a.name,
            "stopped mid-graph, then cleared",
            &after,
            &expected_a,
        ));
        // Handles of earlier requests cannot cancel the current one.
        first_request.cancel();
        cancel.cancel();
        let full = Instant::now();
        let again = bits(&ctx.read_last(&long, &source.select).expect("a readout"));
        let full_time = full.elapsed();
        differences.extend(difference(
            "2,048 tokens",
            "stopped mid-graph, then read in full",
            &again,
            &expected_long,
        ));
        // The evaluation had started (CPU time was spent in it) and ended
        // long before a full one could have: llama.cpp aborted it.
        assert!(
            call_time * 4 < full_time,
            "the cancelled readout took {call_time:?}, a full one {full_time:?}"
        );
        if let Some(cpu) = cpu_spent {
            assert!(
                cpu > 0.1,
                "the cancelled readout used {cpu:.2} s of CPU: it never evaluated"
            );
        }
        println!(
            "{}: readouts after a cancel (before a readout; 300 ms into a 2,048-token one, \
             stopped after {call_time:.2?} and {} s of CPU, where a full readout takes \
             {full_time:.2?}), and with stale handles cancelled: {} [{:.1} s]",
            setting.label(),
            cpu_spent.map_or("?".to_string(), |cpu| format!("{cpu:.2}")),
            verdict(&differences),
            started.elapsed().as_secs_f64()
        );
        assert_no_differences(&differences);
    }
}

#[test]
fn a_native_failure_leaves_the_context_as_new() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    let a = fx.pass("choice-letter-keys#0");
    let b = fx.pass("choice-36-options#0");
    for setting in [CPU_T4_FA_OFF, CPU_T4_FA_ON] {
        let expected_a = bits(&fx.harness(setting).read_last(&a.tokens, &a.select));
        let expected_b = bits(&fx.harness(setting).read_last(&b.tokens, &b.select));
        let mut differences = Vec::new();
        let mut ctx = fx.readout(setting);

        // The decode shim throws before decoding: a status, no evaluation.
        ctx.read_last(&b.tokens, &b.select).expect("a readout of B");
        arm_native_fault("llama_decode_c");
        let failed = ctx.read_last(&a.tokens, &a.select);
        assert!(
            matches!(failed, Err(LlamaError::DecodeFailed { .. })),
            "{failed:?}"
        );
        let after = bits(&ctx.read_last(&a.tokens, &a.select).expect("a readout"));
        differences.extend(difference(
            &a.name,
            "after a failed decode",
            &after,
            &expected_a,
        ));

        // The logits shim throws after a successful decode: no row escapes,
        // and the next readouts are a fresh context's.
        ctx.read_last(&b.tokens, &b.select).expect("a readout of B");
        arm_native_fault("llama_get_logits_ith_c");
        let failed = ctx.read_last(&a.tokens, &a.select);
        assert!(matches!(failed, Err(LlamaError::Internal(_))), "{failed:?}");
        let after = bits(&ctx.read_last(&a.tokens, &a.select).expect("a readout"));
        differences.extend(difference(
            &a.name,
            "after a failed logits read",
            &after,
            &expected_a,
        ));
        let after = bits(&ctx.read_last(&b.tokens, &b.select).expect("a readout"));
        differences.extend(difference(
            &b.name,
            "after a failed logits read",
            &after,
            &expected_b,
        ));
        println!(
            "{}: readouts after an injected failure in the decode shim and in the logits shim: {}",
            setting.label(),
            verdict(&differences)
        );
        assert_no_differences(&differences);
    }
}

#[test]
fn prompts_at_the_budget_boundary_match_the_harness() {
    let Some(fx) = fixture() else { return };
    let _serial = serial();
    // Real d1 tokens, cycled to the lengths that matter at a 512-token budget.
    let source = fx.pass("choice-36-options#0");
    for setting in [
        CPU_T4_FA_OFF.with_budget(512),
        CPU_T4_FA_ON.with_budget(512),
        CPU_T4_FA_OFF.with_budget(512).with_threads(1),
        CPU_T4_FA_ON.with_budget(512).with_threads(1),
    ] {
        let budget = usize::try_from(setting.budget).unwrap();
        let mut ctx = fx.readout(setting);
        let mut differences = Vec::new();
        for length in [1, budget - 1, budget] {
            let tokens: Vec<i32> = source.tokens.iter().copied().cycle().take(length).collect();
            let expected = bits(&fx.harness(setting).read_last(&tokens, &source.select));
            let got = bits(&ctx.read_last(&tokens, &source.select).expect("a readout"));
            differences.extend(difference(
                &format!("{length} tokens"),
                "budget boundary",
                &got,
                &expected,
            ));
        }
        let over: Vec<i32> = source
            .tokens
            .iter()
            .copied()
            .cycle()
            .take(budget + 1)
            .collect();
        assert!(matches!(
            ctx.read_last(&over, &source.select),
            Err(LlamaError::InvalidInput(_))
        ));
        println!(
            "{}: 1, {} and {budget} tokens: {}",
            setting.label(),
            budget - 1,
            verdict(&differences)
        );
        assert_no_differences(&differences);
    }
}
