//! One-shot readouts on a context without memory, for decision models.
//!
//! llama.cpp runs a model that has no memory (no KV cache, no recurrent
//! state) as an encoder: a decode evaluates the whole batch in one
//! microbatch, with attention as the model defines it, and produces outputs
//! only for the tokens the batch flags. LFM3 d1 is such a model: its GGUF
//! turns causal attention off, so llama.cpp creates no memory for it.
//!
//! [`MemorylessContext`] is that context with every limit explicit: one
//! sequence, at most one output row, and a token budget that is the
//! context, batch and microbatch size at once.
//! [`MemorylessContext::read_last`] evaluates one prompt and copies out the
//! logits of the requested tokens at its last position.
//!
//! # Why one output
//!
//! llama.cpp reserves the prompt graph for as many output rows as the
//! context allows, each a full vocabulary of logits, and allows one per
//! batch token unless told otherwise. With d1-2B's 128,000-token vocabulary
//! and an 8,192-token microbatch, that logits tensor alone is 8,192 × 128,000
//! `f32`, 4.2 GB. A readout needs one row, so the context allows one.
//!
//! # What the context accepts
//!
//! Only the `lfm3-d1/text/v1` profile: a model whose GGUF reads
//! `general.architecture = lfm3`, `lfm3.decision.type = lfm3-d1`,
//! `lfm3.attention.causal = false` and has a `systemone` chat template, and
//! no encoder. [`MemorylessContext::new`] asserts that natively, through
//! llama.cpp's view of the loaded model, and then checks what llama.cpp
//! actually built. It complements, and does not replace, classifying the
//! GGUF file from its typed header before loading it (llama.cpp's metadata
//! accessors do not see array-valued keys, so they cannot prove a key
//! absent).

use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::context::ContextFacts;
use crate::error::{LlamaError, LlamaResult};
use crate::ffi::{self, ReadoutSizes, TokenBatch};
use crate::model::{LlamaModel, ModelHandle};

/// The output rows a readout context allows. One readout needs one row.
const OUTPUT_CAP: u32 = 1;

/// Facts before llama.cpp has reported any.
const EMPTY_FACTS: ContextFacts = ContextFacts {
    n_ctx: 0,
    n_batch: 0,
    n_ubatch: 0,
    n_seq_max: 0,
    has_memory: false,
    causal_attn: false,
};

/// The profile a model must assert before a context is created for it.
const ARCHITECTURE: &str = "lfm3";
const DECISION_TYPE_KEY: &str = "lfm3.decision.type";
const DECISION_TYPE: &str = "lfm3-d1";
const CAUSAL_KEY: &str = "lfm3.attention.causal";
const TEMPLATE_NAME: &str = "systemone";

/// What [`MemorylessContext::new`] builds.
///
/// Whatever the parameters, the context runs every operation where its
/// weights are: it turns llama.cpp's op offload off. With it on (llama.cpp's
/// default, which the chat contexts keep), ggml's scheduler moves a
/// large-batch operation on weights in host memory to a GPU backend that asks
/// for it, as Metal, CUDA and Vulkan do, even for a model loaded with no GPU
/// layers: a whole-prompt readout is exactly such a batch. So a model loaded
/// with `n_gpu_layers = 0` is read on the CPU, as far as llama.cpp's
/// scheduler goes: no other pass moves an operation from host memory to a
/// device. That is necessary, not shown sufficient, on an accelerator build:
/// the context still creates the backends of the model's devices (llama.cpp
/// does for every context), and placement there has not been measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorylessParams {
    token_budget: u32,
    n_threads: u32,
    flash_attn: bool,
}

impl MemorylessParams {
    /// The largest token budget: token positions, and the context size
    /// llama.cpp pads up to a multiple of 256, must fit a C `int`.
    pub const MAX_TOKEN_BUDGET: u32 = i32::MAX as u32 - 255;

    /// A context that evaluates prompts of at most `token_budget` tokens,
    /// on `n_threads` threads (`0` uses every core), with Flash Attention
    /// when `flash_attn` is set.
    ///
    /// The budget is the context, batch and microbatch size at once, and
    /// what llama.cpp reserves its graph for, so it is a resource choice:
    /// memory and latency grow with it.
    pub const fn new(token_budget: u32, n_threads: u32, flash_attn: bool) -> Self {
        Self {
            token_budget,
            n_threads,
            flash_attn,
        }
    }

    /// The most tokens one readout may evaluate.
    pub const fn token_budget(&self) -> u32 {
        self.token_budget
    }

    /// Threads llama.cpp evaluates with; `0` means every core.
    pub const fn n_threads(&self) -> u32 {
        self.n_threads
    }

    /// Whether Flash Attention is on.
    pub const fn flash_attn(&self) -> bool {
        self.flash_attn
    }
}

/// A llama.cpp context without memory that reads one output row per
/// evaluation. See the [module documentation](self).
///
/// # Threading
///
/// `Send` but not `Sync`: a readout writes the context's buffers, and
/// [`Self::read_last`] takes `&mut self` so that no two run at once.
pub struct MemorylessContext {
    ptr: *mut c_void,
    budget: usize,
    n_vocab: usize,
    facts: ContextFacts,
    /// Shared with every [`ReadoutCancel`]; llama.cpp's abort callback reads
    /// it, so it must outlive the native context (fields drop after
    /// [`Drop::drop`]).
    cancel: Arc<CancelState>,
    /// The model this context reads. Fields drop after [`Drop::drop`] frees
    /// the native context, so the model outlives it.
    _model: Arc<ModelHandle>,
}

impl MemorylessContext {
    /// Create a readout context for `model`.
    ///
    /// Everything that can be checked without a context is checked first:
    /// the parameters, then the model's profile (see the
    /// [module documentation](self)). Then the context is created and must
    /// have what was asked for: no memory, non-causal attention, one
    /// sequence, and context, batch and microbatch sizes of at least the
    /// budget. The output cap cannot be read back from llama.cpp; it holds
    /// because the model has no encoder, the one case where llama.cpp
    /// overrides it.
    ///
    /// # Errors
    ///
    /// [`LlamaError::InvalidInput`] for a budget of zero or above
    /// [`MemorylessParams::MAX_TOKEN_BUDGET`], a thread count beyond a C
    /// `int`, a model outside the profile, or a context without the
    /// properties above;
    /// [`LlamaError::ContextCreationFailed`] when llama.cpp cannot create
    /// it (commonly: not enough memory for the budget).
    pub fn new(model: &LlamaModel, params: MemorylessParams) -> LlamaResult<Self> {
        Self::create(model, params, OUTPUT_CAP)
    }

    /// [`Self::new`] with an explicit output cap; `0` is llama.cpp's
    /// default of one row per batch token, which measurements compare
    /// against.
    fn create(
        model: &LlamaModel,
        params: MemorylessParams,
        n_outputs_max: u32,
    ) -> LlamaResult<Self> {
        let (sizes, n_threads) = readout_sizes(params, n_outputs_max)?;
        ensure_profile(model)?;
        let n_vocab = model.n_vocab();
        if n_vocab == 0 || i32::try_from(n_vocab).is_err() {
            return Err(LlamaError::InvalidInput(format!(
                "a {n_vocab}-token vocabulary cannot be read out"
            )));
        }
        // SAFETY: the model pointer is live (the handle owns it) and stays
        // live until the context is freed: the context holds the handle.
        let ptr = unsafe {
            ffi::new_readout_context(
                model.as_ptr() as *mut c_void,
                sizes,
                n_threads,
                params.flash_attn,
            )
        };
        if ptr.is_null() {
            return Err(LlamaError::ContextCreationFailed(format!(
                "llama.cpp could not create a readout context with a {}-token budget",
                params.token_budget
            )));
        }
        // Owned from here, so an error below frees the context.
        let mut context = Self {
            ptr,
            budget: usize::try_from(params.token_budget).expect("a u32 fits usize"),
            n_vocab,
            facts: EMPTY_FACTS,
            cancel: Arc::new(CancelState::default()),
            _model: Arc::clone(model.handle()),
        };
        // SAFETY: ptr is the live context created above.
        context.facts = unsafe { ContextFacts::read(ptr) }?;
        check_facts(&context.facts, params.token_budget)?;
        let state = Arc::as_ptr(&context.cancel).cast_mut().cast::<c_void>();
        // SAFETY: the context is live; the state lives in `context.cancel`,
        // which outlives the native context, and the callback only loads it.
        if !unsafe { ffi::set_abort_callback(ptr, cancel_requested, state) } {
            return Err(ffi::native_failure(
                "registering the readout's cancel check",
            ));
        }
        Ok(context)
    }

    /// A context with no native handle, for tests. Only calls that refuse
    /// their input before any native call may receive it.
    #[cfg(test)]
    pub(crate) fn test_stub(model: &LlamaModel, budget: usize, n_vocab: usize) -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            budget,
            n_vocab,
            facts: EMPTY_FACTS,
            cancel: Arc::new(CancelState::default()),
            _model: Arc::clone(model.handle()),
        }
    }

    /// Evaluate `tokens` and return the logits of each `select` id at the
    /// last token.
    ///
    /// The prompt is one sequence at positions `0..tokens.len()`, evaluated
    /// in one pass with only its last token flagged for output. The result
    /// has one value per `select` entry, in order, duplicates included,
    /// copied as llama.cpp computed them: no normalization, and non-finite
    /// values are returned, not refused (a caller reduces its groups first
    /// and judges the result).
    ///
    /// Nothing survives the call: the context has no memory, and a value is
    /// only ever read from the evaluation this call made, so a failure
    /// never returns an earlier row.
    ///
    /// A [`ReadoutCancel`] (see [`Self::cancel_handle`]) stops it. Once the
    /// arguments are checked, a readout ends at the first of:
    ///
    /// - a failure, returned as its own error, even when a cancel was
    ///   requested meanwhile;
    /// - a cancel request seen at a checkpoint: before the pass starts, when
    ///   llama.cpp returns from it, and after the result is copied, the
    ///   readout's completion point. During the pass, llama.cpp's CPU
    ///   backend also checks between two graph nodes and stops (other
    ///   backends finish the pass first; its result is still not returned).
    ///
    /// A cancel requested after the completion point leaves the result as
    /// it is.
    ///
    /// How soon a cancel during the pass returns has no fixed bound: the
    /// node running at that moment finishes first, and one node is a
    /// whole-batch matrix product, so the wait grows with the prompt's
    /// length and shrinks with more threads (see `measure::cancel_latency`).
    ///
    /// # Errors
    ///
    /// [`LlamaError::InvalidInput`], before any native call, when `tokens`
    /// is empty or longer than the budget, `select` is empty, or an id in
    /// either is outside the vocabulary. [`LlamaError::Cancelled`] when the
    /// context's cancel handle was used before the readout completed (before
    /// any native call when it was used before this call started).
    /// [`LlamaError::DecodeFailed`] when llama.cpp fails to evaluate the
    /// prompt, and [`LlamaError::Internal`] when it returns no logits for
    /// the last token.
    pub fn read_last(&mut self, tokens: &[i32], select: &[i32]) -> LlamaResult<Vec<f32>> {
        check_readout(tokens, select, self.budget, self.n_vocab)?;
        let (ptr, n_vocab) = (self.ptr, self.n_vocab);
        let last = i32::try_from(tokens.len() - 1).expect("the budget fits an i32");
        readout(
            || self.cancel.current_is_cancelled(),
            tokens.len(),
            select,
            || {
                let mut batch = TokenBatch::new(tokens.len());
                batch.set(tokens, 0, true);
                // SAFETY: the context is live. The batch holds at most the
                // budget of tokens, which the microbatch fits (checked at
                // creation), ids inside the vocabulary, positions from 0, and
                // exactly one output, which the context's cap allows.
                unsafe { ffi::decode(ptr, &mut batch) }
            },
            || {
                // SAFETY: the pass succeeded and flagged entry `last`, the
                // prompt's last token, for output.
                let row = unsafe { ffi::logits_ith(ptr, last) };
                // SAFETY: a row holds n_vocab floats, the vocabulary of the
                // model this context reads, and stays valid until the next
                // decode; `readout` copies from it before returning.
                (!row.is_null()).then(|| unsafe { std::slice::from_raw_parts(row, n_vocab) })
            },
        )
    }

    /// The most tokens one readout may evaluate.
    pub fn token_budget(&self) -> usize {
        self.budget
    }

    /// Vocabulary size of the model, the range of valid token ids.
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// The sizes, memory and attention llama.cpp gave the context.
    pub fn facts(&self) -> ContextFacts {
        self.facts
    }

    /// A handle that cancels the context's current request from any thread.
    ///
    /// A request is every readout from one [`Self::clear_cancel`] (or the
    /// context's creation) to the next. Using the handle stops the running
    /// readout (on the CPU backend, once the graph node running then
    /// finishes: see [`Self::read_last`]) and refuses every later one of the
    /// same request with [`LlamaError::Cancelled`].
    /// Once the context starts its next request, the handle (and every
    /// clone) does nothing to it: take a new one.
    pub fn cancel_handle(&self) -> ReadoutCancel {
        ReadoutCancel {
            state: Arc::clone(&self.cancel),
            request: self.cancel.current.load(Ordering::Relaxed),
        }
    }

    /// Starts the next request: a cancel of an earlier one no longer
    /// refuses readouts, and handles taken before cannot cancel the new
    /// one. Call it between requests, cancelled or not.
    pub fn clear_cancel(&mut self) {
        self.cancel.current.fetch_add(1, Ordering::Relaxed);
    }
}

/// The cancel state a context shares with its handles, by request number.
#[derive(Debug, Default)]
struct CancelState {
    /// The current request; [`MemorylessContext::clear_cancel`] starts the
    /// next.
    current: AtomicU64,
    /// One more than the newest cancelled request (0: none). Only grows, so
    /// a handle of an earlier request cannot reach a later one, and two
    /// cancels never undo each other.
    cancelled_below: AtomicU64,
}

impl CancelState {
    fn current_is_cancelled(&self) -> bool {
        self.is_cancelled(self.current.load(Ordering::Relaxed))
    }

    fn is_cancelled(&self, request: u64) -> bool {
        self.cancelled_below.load(Ordering::Relaxed) > request
    }

    fn cancel(&self, request: u64) {
        self.cancelled_below
            .fetch_max(request + 1, Ordering::Relaxed);
    }
}

/// Cancels one request of a [`MemorylessContext`], from any thread: the one
/// current when the handle was taken ([`MemorylessContext::cancel_handle`]).
///
/// Cheap to clone; every clone cancels the same request. Using it once the
/// context has moved to a later request, or is gone, does nothing.
#[derive(Debug, Clone)]
pub struct ReadoutCancel {
    state: Arc<CancelState>,
    request: u64,
}

impl ReadoutCancel {
    /// Stops the context's running readout, if it belongs to this handle's
    /// request, and refuses that request's later readouts.
    pub fn cancel(&self) {
        self.state.cancel(self.request);
    }

    /// Whether this handle's request is over by a cancel: its own, or one
    /// of a later request of the context (which only starts once this one
    /// is over).
    pub fn is_cancelled(&self) -> bool {
        self.state.is_cancelled(self.request)
    }
}

/// One readout's control flow around its two native steps: `pass` evaluates
/// the prompt and returns llama.cpp's status, `row` reads the last token's
/// logits. A failure ends it as it happens; `cancelled` is checked before the
/// pass, when it returns, and after the result is copied (the completion
/// point). See [`MemorylessContext::read_last`].
fn readout<'row>(
    cancelled: impl Fn() -> bool,
    n_tokens: usize,
    select: &[i32],
    pass: impl FnOnce() -> c_int,
    row: impl FnOnce() -> Option<&'row [f32]>,
) -> LlamaResult<Vec<f32>> {
    if cancelled() {
        return Err(LlamaError::Cancelled);
    }
    let code = pass();
    // llama.cpp reports a pass its abort callback stopped as 2: the cancel
    // itself, not a failure.
    if code == 2 && cancelled() {
        return Err(LlamaError::Cancelled);
    }
    if code != 0 {
        return Err(LlamaError::DecodeFailed {
            code,
            n_past_in: 0,
            detail: format!(
                "{} on a {n_tokens}-token readout",
                ffi::decode_failure(code)
            ),
        });
    }
    // A backend that does not check for a cancel during the pass finishes
    // it; a cancel requested meanwhile still wins.
    if cancelled() {
        return Err(LlamaError::Cancelled);
    }
    let row = row().ok_or_else(|| ffi::native_failure("reading the last token's logits"))?;
    let logits = select_logits(row, select);
    if cancelled() {
        return Err(LlamaError::Cancelled);
    }
    Ok(logits)
}

/// llama.cpp's abort callback: whether the current request is cancelled.
///
/// # Safety
///
/// `data` must point at the context's live [`CancelState`].
unsafe extern "C" fn cancel_requested(data: *mut c_void) -> bool {
    (*data.cast::<CancelState>()).current_is_cancelled()
}

/// For tests, an abort callback that cancels the current request whenever
/// llama.cpp calls it and never stops the pass: a backend that does not
/// stop.
///
/// # Safety
///
/// As [`cancel_requested`].
#[cfg(test)]
unsafe extern "C" fn cancel_without_stopping(data: *mut c_void) -> bool {
    let state = &*data.cast::<CancelState>();
    state.cancel(state.current.load(Ordering::Relaxed));
    false
}

#[cfg(test)]
impl MemorylessContext {
    /// Registers [`cancel_without_stopping`] in place of the real check.
    fn cancel_during_the_pass_without_stopping(&mut self) {
        let state = Arc::as_ptr(&self.cancel).cast_mut().cast::<c_void>();
        // SAFETY: as for the real callback in `create`.
        assert!(unsafe { ffi::set_abort_callback(self.ptr, cancel_without_stopping, state) });
    }
}

impl Drop for MemorylessContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from `ffi::new_readout_context`, checked
            // non-null on construction. Drop runs at most once, and
            // `self._model` drops after it, so the model outlives the context.
            unsafe { ffi::free_context(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// SAFETY: llama.cpp accepts a context handed to another thread as long as no
// two threads call into it at once. NOT Sync: `read_last` takes `&mut self`,
// and nothing else writes the context.
unsafe impl Send for MemorylessContext {}

/// The native sizes for `params`, checked before any native call.
fn readout_sizes(
    params: MemorylessParams,
    n_outputs_max: u32,
) -> LlamaResult<(ReadoutSizes, c_int)> {
    let budget = params.token_budget;
    let max = MemorylessParams::MAX_TOKEN_BUDGET;
    if budget == 0 || budget > max {
        return Err(LlamaError::InvalidInput(format!(
            "a token budget must be between 1 and {max}, not {budget}"
        )));
    }
    let n_threads = c_int::try_from(params.n_threads).map_err(|_| {
        LlamaError::InvalidInput(format!("{} threads do not fit a C int", params.n_threads))
    })?;
    let sizes = ReadoutSizes {
        n_ctx: budget,
        n_batch: budget,
        n_ubatch: budget,
        n_outputs_max,
    };
    Ok((sizes, n_threads))
}

/// Asserts, through llama.cpp's view of the loaded model, the profile a
/// readout context serves.
fn ensure_profile(model: &LlamaModel) -> LlamaResult<()> {
    if model.has_encoder()? {
        return Err(LlamaError::InvalidInput(
            "the model has an encoder, for which llama.cpp ignores the output cap".to_string(),
        ));
    }
    let refuse = |what: &str| {
        Err(LlamaError::InvalidInput(format!(
            "not an {DECISION_TYPE} decision model: {what}"
        )))
    };
    if model.meta_str("general.architecture")?.as_deref() != Some(ARCHITECTURE) {
        return refuse("the architecture is not lfm3");
    }
    if model.meta_str(DECISION_TYPE_KEY)?.as_deref() != Some(DECISION_TYPE) {
        return refuse("lfm3.decision.type is not lfm3-d1");
    }
    if model.meta_str(CAUSAL_KEY)?.as_deref() != Some("false") {
        return refuse("lfm3.attention.causal is not false");
    }
    if model.named_chat_template(TEMPLATE_NAME)?.is_none() {
        return refuse("there is no systemone template");
    }
    Ok(())
}

/// Refuses a context llama.cpp did not build as asked.
fn check_facts(facts: &ContextFacts, budget: u32) -> LlamaResult<()> {
    let refuse = |what: String| Err(LlamaError::InvalidInput(what));
    if facts.has_memory {
        return refuse("llama.cpp gave the readout context memory".to_string());
    }
    if facts.causal_attn {
        return refuse("llama.cpp gave the readout context causal attention".to_string());
    }
    if facts.n_seq_max != 1 {
        return refuse(format!(
            "the readout context has {} sequences, not one",
            facts.n_seq_max
        ));
    }
    for (name, size) in [
        ("context", facts.n_ctx),
        ("batch", facts.n_batch),
        ("microbatch", facts.n_ubatch),
    ] {
        if size < budget {
            return refuse(format!(
                "the readout context's {name} size is {size}, below the {budget}-token budget"
            ));
        }
    }
    Ok(())
}

/// Refuses a readout before any native call reads it.
fn check_readout(tokens: &[i32], select: &[i32], budget: usize, n_vocab: usize) -> LlamaResult<()> {
    if tokens.is_empty() {
        return Err(LlamaError::InvalidInput(
            "cannot read out an empty prompt".to_string(),
        ));
    }
    if tokens.len() > budget {
        return Err(LlamaError::InvalidInput(format!(
            "a {}-token prompt exceeds the {budget}-token budget",
            tokens.len()
        )));
    }
    if select.is_empty() {
        return Err(LlamaError::InvalidInput(
            "no logits selected to read out".to_string(),
        ));
    }
    let in_vocab = |id: &i32| usize::try_from(*id).is_ok_and(|id| id < n_vocab);
    if let Some(index) = tokens.iter().position(|id| !in_vocab(id)) {
        return Err(LlamaError::InvalidInput(format!(
            "prompt token {index} is outside the {n_vocab}-token vocabulary"
        )));
    }
    if let Some(index) = select.iter().position(|id| !in_vocab(id)) {
        return Err(LlamaError::InvalidInput(format!(
            "selected token {index} is outside the {n_vocab}-token vocabulary"
        )));
    }
    Ok(())
}

/// The `select` entries of `row`, in order. Every id must index `row`.
fn select_logits(row: &[f32], select: &[i32]) -> Vec<f32> {
    select
        .iter()
        .map(|&id| row[usize::try_from(id).expect("selected ids are validated")])
        .collect()
}

#[cfg(all(test, target_os = "linux"))]
mod measure;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_models::staged_d1;
    use std::cell::Cell;
    use std::sync::atomic::AtomicBool;

    const ROW: [f32; 4] = [0.5, -1.0, 2.0, 0.0];

    /// Runs the readout's control flow with its native steps injected, no
    /// model needed: `pass` gets the cancel flag and returns llama.cpp's
    /// status, and the row step requests a cancel when `cancel_in_row`.
    /// Returns the result and whether the row was read.
    fn injected(
        cancelled_first: bool,
        pass: impl FnOnce(&AtomicBool) -> c_int,
        cancel_in_row: bool,
    ) -> (LlamaResult<Vec<f32>>, bool) {
        let cancel = AtomicBool::new(cancelled_first);
        let row_read = Cell::new(false);
        let result = readout(
            || cancel.load(Ordering::Relaxed),
            3,
            &[2, 0],
            || pass(&cancel),
            || {
                row_read.set(true);
                if cancel_in_row {
                    cancel.store(true, Ordering::Relaxed);
                }
                Some(&ROW[..])
            },
        );
        (result, row_read.get())
    }

    fn cancel_then(status: c_int) -> impl FnOnce(&AtomicBool) -> c_int {
        move |cancel| {
            cancel.store(true, Ordering::Relaxed);
            status
        }
    }

    #[test]
    fn an_uncancelled_readout_returns_the_selection() {
        let (result, row_read) = injected(false, |_| 0, false);
        assert_eq!(result.unwrap(), [2.0, 0.5]);
        assert!(row_read);
    }

    #[test]
    fn a_cancel_before_the_pass_skips_it() {
        let (result, row_read) = injected(true, |_| panic!("the pass must not run"), false);
        assert!(matches!(result, Err(LlamaError::Cancelled)));
        assert!(!row_read);
    }

    /// A backend that does not stop between nodes finishes the pass; its
    /// result is not returned.
    #[test]
    fn a_pass_finished_after_a_cancel_returns_no_result() {
        let (result, row_read) = injected(false, cancel_then(0), false);
        assert!(matches!(result, Err(LlamaError::Cancelled)));
        assert!(!row_read, "nothing is read after the cancel");
    }

    #[test]
    fn a_pass_llama_cpp_stopped_for_a_cancel_is_cancelled() {
        let (result, _) = injected(false, cancel_then(2), false);
        assert!(matches!(result, Err(LlamaError::Cancelled)));
        let (result, _) = injected(false, |_| 2, false);
        assert!(
            matches!(result, Err(LlamaError::DecodeFailed { code: 2, .. })),
            "a stop nobody requested is a failure"
        );
    }

    #[test]
    fn a_failure_during_a_cancel_is_still_the_failure() {
        let (result, _) = injected(false, cancel_then(-3), false);
        assert!(matches!(
            result,
            Err(LlamaError::DecodeFailed { code: -3, .. })
        ));
    }

    #[test]
    fn a_cancel_while_the_output_is_read_returns_no_result() {
        let (result, row_read) = injected(false, |_| 0, true);
        assert!(matches!(result, Err(LlamaError::Cancelled)));
        assert!(row_read);
    }

    #[test]
    fn a_missing_row_is_an_error() {
        let result = readout(|| false, 3, &[0], || 0, || None);
        assert!(matches!(result, Err(LlamaError::Internal(_))));
    }

    /// On the real model with a backend that requests a cancel during the
    /// pass and never stops it: llama.cpp finishes (status 0), and the
    /// readout still returns no result.
    #[test]
    fn a_real_pass_that_does_not_stop_for_a_cancel_returns_no_result() {
        let Some(path) = staged_d1() else { return };
        crate::backend_init();
        let model = LlamaModel::load(path.to_str().unwrap(), 0).expect("load d1-2B");
        let mut ctx = MemorylessContext::new(&model, MemorylessParams::new(256, 4, false))
            .expect("a context");
        let prompt = model
            .tokenize_special("<|startoftext|><|im_start|>user\nHello", false)
            .expect("tokenize");
        assert!(ctx.read_last(&prompt, &[0]).is_ok());
        ctx.cancel_during_the_pass_without_stopping();
        assert!(matches!(
            ctx.read_last(&prompt, &[0]),
            Err(LlamaError::Cancelled)
        ));
    }

    fn facts(n: u32) -> ContextFacts {
        ContextFacts {
            n_ctx: n,
            n_batch: n,
            n_ubatch: n,
            n_seq_max: 1,
            has_memory: false,
            causal_attn: false,
        }
    }

    #[test]
    fn readouts_are_refused_before_any_native_call() {
        let model = LlamaModel::test_stub();
        let mut ctx = MemorylessContext::test_stub(&model, 4, 10);
        let calls = ffi::native_calls();
        let refused =
            |result: LlamaResult<Vec<f32>>| matches!(result, Err(LlamaError::InvalidInput(_)));

        assert!(refused(ctx.read_last(&[], &[1])), "empty prompt");
        assert!(refused(ctx.read_last(&[1; 5], &[1])), "past the budget");
        assert!(refused(ctx.read_last(&[1, 2], &[])), "nothing selected");
        assert!(
            refused(ctx.read_last(&[1, 10], &[1])),
            "prompt id past the vocabulary"
        );
        assert!(refused(ctx.read_last(&[1, -1], &[1])), "negative prompt id");
        assert!(
            refused(ctx.read_last(&[1], &[10])),
            "selected id past the vocabulary"
        );
        assert!(
            refused(ctx.read_last(&[1], &[3, -1])),
            "negative selected id"
        );
        assert_eq!(ffi::native_calls(), calls, "no native call was made");
    }

    #[test]
    fn the_budget_is_the_last_accepted_length() {
        assert!(
            check_readout(&[1; 4], &[0], 4, 10).is_ok(),
            "exactly the budget"
        );
        assert!(
            check_readout(&[9], &[9, 9, 0], 4, 10).is_ok(),
            "the last id"
        );
        assert!(check_readout(&[1; 5], &[0], 4, 10).is_err());
    }

    #[test]
    fn refusals_name_positions_not_ids() {
        let message = check_readout(&[3, 1234567], &[0], 4, 10)
            .unwrap_err()
            .to_string();
        assert!(message.contains("prompt token 1 "), "{message}");
        assert!(!message.contains("1234567"), "the id stays out: {message}");
    }

    #[test]
    fn the_selection_keeps_order_and_duplicates_and_raw_values() {
        let row = [0.5, f32::NEG_INFINITY, 2.0, f32::NAN, -0.0];
        let selected = select_logits(&row, &[2, 0, 2, 1, 4, 3]);
        assert_eq!(selected[..5], [2.0, 0.5, 2.0, f32::NEG_INFINITY, -0.0]);
        assert!(selected[4].is_sign_negative(), "copied bit for bit");
        assert!(
            selected[5].is_nan(),
            "non-finite values are returned, not refused"
        );
    }

    #[test]
    fn budgets_and_threads_are_checked_before_any_native_call() {
        let calls = ffi::native_calls();
        let refused = |budget, threads| {
            matches!(
                readout_sizes(MemorylessParams::new(budget, threads, false), OUTPUT_CAP),
                Err(LlamaError::InvalidInput(_))
            )
        };
        assert!(refused(0, 4), "an empty budget");
        assert!(
            refused(MemorylessParams::MAX_TOKEN_BUDGET + 1, 4),
            "positions would overflow"
        );
        assert!(refused(4096, u32::MAX), "threads past a C int");
        let (sizes, threads) = readout_sizes(MemorylessParams::new(4096, 4, true), OUTPUT_CAP)
            .expect("a valid budget");
        assert_eq!(
            sizes,
            ReadoutSizes {
                n_ctx: 4096,
                n_batch: 4096,
                n_ubatch: 4096,
                n_outputs_max: 1,
            }
        );
        assert_eq!(threads, 4);
        assert!(readout_sizes(
            MemorylessParams::new(MemorylessParams::MAX_TOKEN_BUDGET, 0, false),
            1
        )
        .is_ok());
        assert_eq!(ffi::native_calls(), calls);
    }

    #[test]
    fn a_context_must_be_memoryless_noncausal_single_sequence_and_large_enough() {
        assert!(check_facts(&facts(4096), 4096).is_ok());
        assert!(
            check_facts(&facts(4352), 4100).is_ok(),
            "padded sizes are larger"
        );
        let refused = |f: ContextFacts| check_facts(&f, 4096).is_err();
        assert!(refused(ContextFacts {
            has_memory: true,
            ..facts(4096)
        }));
        assert!(refused(ContextFacts {
            causal_attn: true,
            ..facts(4096)
        }));
        assert!(refused(ContextFacts {
            n_seq_max: 2,
            ..facts(4096)
        }));
        assert!(refused(ContextFacts {
            n_ctx: 4095,
            ..facts(4096)
        }));
        assert!(refused(ContextFacts {
            n_batch: 512,
            ..facts(4096)
        }));
        assert!(refused(ContextFacts {
            n_ubatch: 512,
            ..facts(4096)
        }));
    }

    #[test]
    fn a_context_keeps_its_model_loaded_until_both_are_dropped() {
        let model = LlamaModel::test_stub();
        let handle = Arc::downgrade(model.handle());
        let ctx = MemorylessContext::test_stub(&model, 4, 10);

        drop(model);
        assert!(
            handle.upgrade().is_some(),
            "the context must keep the model"
        );
        drop(ctx);
        assert!(handle.upgrade().is_none(), "the last owner frees the model");
    }

    #[test]
    fn a_cancelled_request_refuses_before_any_native_call() {
        let model = LlamaModel::test_stub();
        let mut ctx = MemorylessContext::test_stub(&model, 4, 10);
        let cancel = ctx.cancel_handle();
        assert!(!cancel.is_cancelled());
        cancel.clone().cancel();
        assert!(
            cancel.is_cancelled(),
            "every clone cancels the same request"
        );

        let calls = ffi::native_calls();
        for _ in 0..2 {
            assert!(matches!(
                ctx.read_last(&[1, 2], &[0]),
                Err(LlamaError::Cancelled)
            ));
        }
        assert!(
            matches!(ctx.read_last(&[], &[0]), Err(LlamaError::InvalidInput(_))),
            "input checks come first"
        );
        assert_eq!(ffi::native_calls(), calls, "no native call was made");
        drop(ctx);
        cancel.cancel(); // the context is gone: nothing to stop
    }

    /// llama.cpp has no accessor for `op_offload`, and on a CPU build it
    /// changes nothing (there is no device to offload to), so this reads the
    /// parameters the readout context is created with.
    #[test]
    fn a_readout_context_offloads_no_operation() {
        let (sizes, n_threads) =
            readout_sizes(MemorylessParams::new(4096, 4, true), OUTPUT_CAP).unwrap();
        let params = ffi::readout_params(sizes, n_threads, true);
        assert!(!params.op_offload, "no operation moves to a GPU backend");
        assert_eq!(
            (params.n_ctx, params.n_batch, params.n_ubatch),
            (4096, 4096, 4096)
        );
        assert_eq!(
            (
                params.n_seq_max,
                params.n_outputs_max,
                params.n_outputs_max_per_seq
            ),
            (1, 1, 1)
        );
        assert!(!params.embeddings);
        assert_eq!(
            params.flash_attn_type,
            llama_cpp_sys::bindings::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED
        );
        assert_eq!((params.n_threads, params.n_threads_batch), (4, 4));
    }

    /// The stub has no native context: a readout that is not refused as
    /// cancelled reaches the decode shim, which fails it (-1).
    #[test]
    fn a_handle_of_an_earlier_request_cannot_cancel_the_next() {
        let model = LlamaModel::test_stub();
        let mut ctx = MemorylessContext::test_stub(&model, 4, 10);
        let reached_the_pass = |ctx: &mut MemorylessContext| match ctx.read_last(&[1, 2], &[0]) {
            Err(LlamaError::DecodeFailed { code: -1, .. }) => true,
            Err(LlamaError::Cancelled) => false,
            other => panic!("{other:?}"),
        };
        let first = ctx.cancel_handle();
        first.cancel();
        assert!(!reached_the_pass(&mut ctx));

        ctx.clear_cancel();
        assert!(reached_the_pass(&mut ctx), "the next request reads");
        first.cancel();
        first.clone().cancel();
        assert!(
            reached_the_pass(&mut ctx),
            "a stale handle and its clones do nothing"
        );

        let second = ctx.cancel_handle();
        assert!(!second.is_cancelled());
        second.cancel();
        assert!(!reached_the_pass(&mut ctx));
        first.cancel();
        assert!(second.is_cancelled(), "nothing undoes a cancel");

        ctx.clear_cancel();
        let third = ctx.cancel_handle();
        second.cancel();
        assert!(!third.is_cancelled());
        assert!(reached_the_pass(&mut ctx));
    }

    #[test]
    fn the_cancel_handle_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ReadoutCancel>();
    }

    #[test]
    fn the_context_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MemorylessContext>();
    }

    /// Compiles only while the context is not `Sync`: for a `Sync` type both
    /// impls below apply and the call is ambiguous.
    #[test]
    fn the_context_is_not_sync() {
        trait AmbiguousIfSync<A> {
            fn check() {}
        }
        impl<T: ?Sized> AmbiguousIfSync<()> for T {}
        struct IsSync;
        impl<T: ?Sized + Sync> AmbiguousIfSync<IsSync> for T {}
        <MemorylessContext as AmbiguousIfSync<_>>::check();
    }
}
