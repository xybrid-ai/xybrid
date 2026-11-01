//! G2 at the readout level: [`MemorylessContext::read_last`] against a
//! minimal native harness, bit for bit, on LiquidAI's d1-2B (artifact
//! `d1-2b-q4km`).
//!
//! The harness is written directly on llama.cpp's C API. Its types and the
//! native archives come from `llama-cpp-sys`, its functions are declared
//! below, and it uses none of xybrid-llama's wrappers nor `wrapper.cpp`'s
//! model, context, batch or readout shims (only the `-sys` crate's one-time
//! backend initialization, which both sides share in one process). It loads
//! its own instance of the same GGUF file with llama.cpp's default model
//! parameters on the CPU, and builds its context as `MemorylessContext`'s is
//! specified: context, batch and microbatch size equal to the token budget,
//! one sequence, one output row, no embeddings, the model's own attention
//! type, Flash Attention on or off, one thread count for prompts and single
//! tokens, and llama.cpp's default KV types and `swa_full`. Its batch is the
//! server's: the whole prompt as sequence 0 at positions `0..n`, only the
//! last token flagged for output, in `llama_batch_init` storage filled as
//! `common_batch_add` fills it.
//!
//! What must be bit-identical:
//!
//! - every prepared pass of P3a's decision cases whose upstream evaluation
//!   ran the model (`integration-tests/fixtures/decision/cases`), including
//!   the 4,096-token `budget-at-limit` pass, at settings `cpu-t4-fa-off` and
//!   `cpu-t4-fa-on` (four threads, a 4,096-token budget), and the short ones
//!   at one thread too;
//! - a context reused for every pass against a fresh context per pass, on
//!   both sides;
//! - repeats (A/A) and interleaving (A/B/A) in one context;
//! - a valid readout after each refusal (over the budget, an empty prompt,
//!   nothing selected, ids outside the vocabulary) against a fresh context's;
//! - prompts of 1, budget − 1 and budget tokens at a 512-token budget.
//!
//! This proves the readout's wiring, and that no state survives a readout,
//! under these pinned conditions: one llama.cpp build, the CPU, these
//! settings. It does not prove prompts or answer arithmetic (gate G0), and
//! says nothing about other builds, backends or devices, whose logits may
//! differ (gate G3).
//!
//! Release builds only: llama.cpp's logits differ between its debug and
//! release builds, and the settings name release natives. A debug build
//! skips each test, as a missing model does; `XYBRID_REQUIRE_MODELS` turns
//! either skip into a failure. Stage the GGUF as for `d1_readout_smoke`
//! (`integration-tests/fixtures/models/d1-2b-q4km/d1-2B-Q4_K_M.gguf`, or
//! `XYBRID_TEST_MODELS`), then run:
//!
//! ```sh
//! cargo test --release -p xybrid-llama --features bindings --test d1_native_parity -- --nocapture
//! ```
//!
//! The whole file takes about a quarter of an hour on four cores, most of it
//! in the 4,096-token pass; the tests run one at a time whatever the test
//! harness's thread count, so that two budget-sized contexts never compete
//! for memory.

#![cfg(feature = "bindings")]

use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use llama_cpp_sys::bindings as sys;
use xybrid_llama::{backend_init, LlamaError, LlamaModel, MemorylessContext, MemorylessParams};

const MODEL_ID: &str = "d1-2b-q4km";
const MODEL_FILE: &str = "d1-2B-Q4_K_M.gguf";
/// The pinned size of `d1-2b-q4km` (`artifacts.json`).
const MODEL_BYTES: u64 = 1_099_629_888;

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
    fn llama_get_memory(ctx: *const sys::llama_context) -> sys::llama_memory_t;
    fn llama_get_causal_attn(ctx: *const sys::llama_context) -> bool;
    fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> sys::llama_batch;
    fn llama_batch_free(batch: sys::llama_batch);
    fn llama_decode(ctx: *mut sys::llama_context, batch: sys::llama_batch) -> i32;
    fn llama_get_logits_ith(ctx: *mut sys::llama_context, i: i32) -> *mut f32;
    fn llama_print_system_info() -> *const c_char;
}

/// The harness's own instance of the GGUF, loaded with llama.cpp's default
/// model parameters and no GPU layers, as `LlamaModel::load(path, 0)` asks.
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
        // SAFETY: plain value constructors; the path is a live C string for
        // the call, and llama.cpp copies what it keeps.
        let ptr = unsafe {
            let mut params = llama_model_default_params();
            params.n_gpu_layers = 0;
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
            (sys::ggml_type_GGML_TYPE_F16, sys::ggml_type_GGML_TYPE_F16, true),
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
        assert!(tokens.iter().chain(select).all(in_vocab), "ids in the vocabulary");
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
// The fixture: both model instances, P3a's passes, the build identity
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

/// The staged GGUF, or `None` (with a note) when it is absent or this is a
/// debug build.
///
/// # Panics
///
/// When `XYBRID_REQUIRE_MODELS` is set and the test would skip, and when the
/// staged file is not the pinned size.
fn staged_model_path() -> Option<PathBuf> {
    let skip = |reason: String| {
        if std::env::var_os("XYBRID_REQUIRE_MODELS").is_some() {
            panic!("XYBRID_REQUIRE_MODELS is set but {reason}");
        }
        eprintln!("Skipping test: {reason}");
        None
    };
    if cfg!(debug_assertions) {
        return skip(
            "this is a debug build: G2 compares release natives (cargo test --release)".to_string(),
        );
    }
    let models = std::env::var_os("XYBRID_TEST_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../integration-tests/fixtures/models")
        });
    let path = models.join(MODEL_ID).join(MODEL_FILE);
    if !path.is_file() {
        return skip(format!(
            "'{MODEL_ID}/{MODEL_FILE}' is not staged under {}",
            models.display()
        ));
    }
    let bytes = std::fs::metadata(&path).expect("stat the GGUF").len();
    assert_eq!(
        bytes, MODEL_BYTES,
        "{} is not the pinned {MODEL_ID} artifact",
        path.display()
    );
    Some(path)
}

/// Both model instances and P3a's passes, loaded once, after printing the
/// build identity.
fn fixture() -> Option<&'static Fixture> {
    static FIXTURE: OnceLock<Option<Fixture>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let path = staged_model_path()?;
            backend_init();
            llama_cpp_sys::backend_init();
            print_build_identity(&path);
            let xybrid =
                LlamaModel::load(path.to_str().expect("a UTF-8 path"), 0).expect("load the GGUF");
            let harness = HarnessModel::load(&path);
            assert_eq!(xybrid.n_vocab(), harness.n_vocab, "one vocabulary");
            Some(Fixture {
                passes: model_backed_passes(),
                xybrid,
                harness,
            })
        })
        .as_ref()
}

fn print_build_identity(model: &Path) {
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
    println!(
        "G2 model: {MODEL_ID} ({MODEL_BYTES} bytes, the pinned size) at {}",
        model.display()
    );
}

/// The prepared passes of P3a's cases whose upstream evaluation ran the
/// model, each distinct (tokens, select) pair once, in case order.
fn model_backed_passes() -> Vec<Pass> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../integration-tests/fixtures/decision/cases");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("read P3a's cases at {}: {error}", dir.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();

    let mut passes: Vec<Pass> = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(&file).expect("read a case");
        let case: serde_json::Value = serde_json::from_str(&text).expect("a JSON case");
        let upstream = &case["upstream"];
        if upstream["evaluate"]["source"] != "model" {
            continue;
        }
        let id = case["id"].as_str().expect("a case id");
        assert_eq!(
            (&case["artifact"], &case["settings"]),
            (
                &serde_json::Value::from(MODEL_ID),
                &serde_json::Value::from("cpu-t4-fa-off")
            ),
            "{id}: the oracle's logits come from {MODEL_ID} at cpu-t4-fa-off"
        );
        let prepared = upstream["prepare"]["passes"]
            .as_array()
            .expect("prepared passes");
        let raw_scores = upstream["evaluate"]["raw_scores"]
            .as_array()
            .expect("raw scores");
        assert_eq!(prepared.len(), raw_scores.len(), "{id}: a readout per pass");
        for (index, (pass, raw)) in prepared.iter().zip(raw_scores).enumerate() {
            let ids = |key: &str| -> Vec<i32> {
                pass[key]
                    .as_array()
                    .unwrap_or_else(|| panic!("{id}: {key}"))
                    .iter()
                    .map(|id| {
                        i32::try_from(id.as_i64().expect("an integer id")).expect("an i32 id")
                    })
                    .collect()
            };
            let tokens = ids("token_ids");
            let select = ids("select");
            assert_eq!(
                pass["output_index"].as_u64(),
                Some(tokens.len() as u64 - 1),
                "{id}: upstream reads the last token"
            );
            let oracle: Vec<u32> = raw
                .as_array()
                .expect("a pass's raw scores")
                .iter()
                .map(|bits| {
                    let bits = bits.as_str().expect("an f32 bit pattern");
                    u32::from_str_radix(bits.trim_start_matches("0x"), 16).expect("hex bits")
                })
                .collect();
            assert_eq!(oracle.len(), select.len(), "{id}: a logit per label");
            if passes
                .iter()
                .any(|seen| seen.tokens == tokens && seen.select == select)
            {
                continue;
            }
            passes.push(Pass {
                name: format!("{id}#{index}"),
                tokens,
                select,
                oracle,
            });
        }
    }
    passes
}

/// The tests share both models and run one at a time.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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

/// The passes of at most `max_tokens` tokens.
fn short(passes: &[Pass], max_tokens: usize) -> Vec<&Pass> {
    passes
        .iter()
        .filter(|pass| pass.tokens.len() <= max_tokens)
        .collect()
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
        let readout = fx.readout(setting).facts();
        let harness = fx.harness(setting).facts();
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
        assert_eq!(
            (harness.n_batch, harness.n_ubatch, harness.n_seq_max),
            (setting.budget, setting.budget, 1),
            "{}",
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
        fx.passes.iter().map(|pass| pass.tokens.len()).sum::<usize>()
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
        let got = bits(&reused.read_last(&pass.tokens, &pass.select).expect("a readout"));
        differences.extend(difference(&pass.name, "reused readout context", &got, expected));
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
        differences.extend(difference(&pass.name, "fresh readout context", &fresh, expected));
        let again = bits(&harness_reused.read_last(&pass.tokens, &pass.select));
        differences.extend(difference(&pass.name, "reused harness context", &again, expected));
    }

    let tokens: usize = passes.iter().map(|pass| pass.tokens.len()).sum();
    println!(
        "{}: {} passes ({tokens} tokens) reused, {short_count} of them also fresh: {} \
         [{:.1} s]",
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
            if differences.is_empty() {
                "bit-identical"
            } else {
                "DIFFERENT"
            }
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
        let refusals: [(&str, &[i32], &[i32]); 7] = [
            ("a prompt over the budget", &over_budget, &a.select),
            ("an empty prompt", &[], &a.select),
            ("nothing selected", &a.tokens, &[]),
            ("a prompt id past the vocabulary", &bad_prompt, &a.select),
            ("a negative prompt id", &negative_prompt, &a.select),
            ("a selected id past the vocabulary", &a.tokens, &[n_vocab]),
            ("a negative selected id", &a.tokens, &[a.select[0], -1]),
        ];

        let expected = bits(&fx.harness(setting).read_last(&a.tokens, &a.select));
        let fresh = bits(
            &fx.readout(setting)
                .read_last(&a.tokens, &a.select)
                .expect("a readout"),
        );
        let mut differences = Vec::new();
        differences.extend(difference(&a.name, "fresh readout context", &fresh, &expected));

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
            if differences.is_empty() {
                "bit-identical to a fresh context's and the harness's"
            } else {
                "DIFFERENT"
            }
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
        let over: Vec<i32> = source.tokens.iter().copied().cycle().take(budget + 1).collect();
        assert!(matches!(
            ctx.read_last(&over, &source.select),
            Err(LlamaError::InvalidInput(_))
        ));
        println!(
            "{}: 1, {} and {budget} tokens: {}",
            setting.label(),
            budget - 1,
            if differences.is_empty() {
                "bit-identical"
            } else {
                "DIFFERENT"
            }
        );
        assert_no_differences(&differences);
    }
}
