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
use std::sync::Arc;

use crate::context::ContextFacts;
use crate::error::{LlamaError, LlamaResult};
use crate::ffi::{self, ReadoutSizes, TokenBatch};
use crate::model::{LlamaModel, ModelHandle};

/// The output rows a readout context allows. One readout needs one row.
const OUTPUT_CAP: u32 = 1;

/// The profile a model must assert before a context is created for it.
const ARCHITECTURE: &str = "lfm3";
const DECISION_TYPE_KEY: &str = "lfm3.decision.type";
const DECISION_TYPE: &str = "lfm3-d1";
const CAUSAL_KEY: &str = "lfm3.attention.causal";
const TEMPLATE_NAME: &str = "systemone";

/// What [`MemorylessContext::new`] builds.
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
        // SAFETY: ptr is the live context created above.
        let facts = unsafe { ContextFacts::read(ptr) };
        // Owned from here, so an error below frees the context.
        let context = Self {
            ptr,
            budget: usize::try_from(params.token_budget).expect("a u32 fits usize"),
            n_vocab,
            facts,
            _model: Arc::clone(model.handle()),
        };
        check_facts(&facts, params.token_budget)?;
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
            facts: ContextFacts {
                n_ctx: 0,
                n_batch: 0,
                n_ubatch: 0,
                n_seq_max: 0,
                has_memory: false,
                causal_attn: false,
            },
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
    /// # Errors
    ///
    /// [`LlamaError::InvalidInput`], before any native call, when `tokens`
    /// is empty or longer than the budget, `select` is empty, or an id in
    /// either is outside the vocabulary. [`LlamaError::DecodeFailed`] when
    /// llama.cpp fails to evaluate the prompt, and [`LlamaError::Internal`]
    /// when it returns no logits for the last token.
    pub fn read_last(&mut self, tokens: &[i32], select: &[i32]) -> LlamaResult<Vec<f32>> {
        check_readout(tokens, select, self.budget, self.n_vocab)?;
        let mut batch = TokenBatch::new(tokens.len());
        batch.set(tokens, 0, true);
        // SAFETY: the context is live. The batch holds at most the budget of
        // tokens, which the microbatch fits (checked at creation), ids inside
        // the vocabulary, positions from 0, and exactly one output, which
        // the context's cap allows.
        let code = unsafe { ffi::decode(self.ptr, &mut batch) };
        if code != 0 {
            return Err(LlamaError::DecodeFailed {
                code,
                n_past_in: 0,
                detail: format!("llama_decode failed on a {}-token readout", tokens.len()),
            });
        }
        let last = i32::try_from(tokens.len() - 1).expect("the budget fits an i32");
        // SAFETY: the decode above succeeded and flagged entry `last`, the
        // prompt's last token, for output.
        let row = unsafe { ffi::logits_ith(self.ptr, last) };
        if row.is_null() {
            return Err(LlamaError::Internal(
                "llama.cpp returned no logits for the last token".to_string(),
            ));
        }
        // SAFETY: the row holds n_vocab floats, the vocabulary of the model
        // this context reads, and stays valid until the next decode; the
        // slice does not outlive this call.
        let row = unsafe { std::slice::from_raw_parts(row, self.n_vocab) };
        Ok(select_logits(row, select))
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
    if model.has_encoder() {
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

#[cfg(test)]
mod tests {
    use super::*;

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
