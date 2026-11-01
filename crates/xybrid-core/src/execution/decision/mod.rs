//! The LFM3 d1 decision protocol: from a [`DecisionRequest`] to the passes a
//! model evaluates, and from their readouts to [`Decisions`].
//!
//! # Contents
//!
//! | Item | Role |
//! |------|------|
//! | [`DecisionModelFacts`] | What a model's GGUF header says the protocol needs: its profile, temperatures and [`DecisionLimits`] |
//! | [`ProfileRules`] | The profile's ceilings (options, levels, JSON depths), for a capability view |
//! | [`DecisionVocab`] | The model's tokenizer and control-token spellings (`llama::LlamaVocab` with `llm-llamacpp`) |
//! | `prepare` (`llm-llamacpp`) | Checks a whole request, renders and tokenizes one pass per question, allocates labels: a [`DecisionPlan`] |
//! | [`finish`] | Reads each question's answer from its pass's readout, in `f64` |
//!
//! A run is `prepare`, one evaluation per [`DecisionPass`] (the logits of
//! its [`select`](DecisionPass::select) tokens at its last token), then
//! [`finish`]. `prepare` refuses anything the protocol cannot read before
//! any pass runs, so a refused request costs no inference; `finish` is pure
//! arithmetic. Nothing here touches native code except through the
//! [`DecisionVocab`] it is given (tokenization).
//!
//! The API is internal (`pub(crate)`); the `test-util` feature makes it
//! public for cross-crate conformance tests.
//!
//! # Parity
//!
//! Everything here reproduces llama.cpp's own decision code
//! (`vendor/llama-cpp`, pinned at v0.8.0), and the conformance cases in
//! `integration-tests/fixtures/decision/` hold its observations; the tests
//! in `tests/` check every case. Where upstream behaves in a way a caller
//! must not rely on, xybrid refuses instead (the cases' `xybrid`
//! expectations). The sources, by step:
//!
//! | Step | Upstream |
//! |------|----------|
//! | Temperatures: `<type>.<bucket>`, then `<type>`, then 1 | `tools/server/server-decision.cpp:58-72` (`init`), `:1572-1591` (`get_temperature`); the value is the metadata string read back with `strtof` (`src/llama-impl.cpp:161`) |
//! | Control spellings: CONTROL tokens of 2 bytes or more, longest first | `server-decision.cpp:170`, `:213-230` (`init_quoted_spellings`) |
//! | `d1_break` on every string and object key of the template input | `server-decision.cpp:606-661` (`decision_escape`, `d1_break`), `:1152-1159` (`render`) |
//! | Labels: codes, the fallback pool, groups | `server-decision.cpp:968-1065` (`d1_labels`) |
//! | The template input and rendering | `server-decision.cpp:1067-1188` (`render_options`, `render`); `common/jinja/value.cpp:250-277` (`tojson`), `:1444-1476` (`from_json`), `:1555-1623` (`float_repr`, `float_json`), `:1627-1706` (`value_to_json_internal`) |
//! | Tokenization: the prompt with `parse_special`, labels without | `server-decision.cpp:1250`, `:974` |
//! | Readout: each group's maximum at the last prompt token | `tools/server/server-context.cpp:2526-2546` (`send_decision`) |
//! | Answers: softmax in `f64`, first maximum, median, confidence | `server-decision.cpp:1640-1642` (confidence), `:1651-1747` (`format_answer`) |
//! | Usage: every pass's tokens | `server-context.cpp:6656-6686` (`post_systemone`) |
//!
//! # Where xybrid is stricter than upstream
//!
//! - Only the pinned `systemone` template renders
//!   ([`SYSTEMONE_TEMPLATE_SHA256`]); another one is refused
//!   ([`DecisionError::UnsupportedTemplate`]) until its prompts are checked.
//! - Two object keys that `d1_break` makes equal are refused
//!   ([`DecisionError::EscapedKeyCollision`]); upstream keeps the last value.
//! - An integer above `i64::MAX` is refused
//!   ([`DecisionError::NumberOutOfRange`]); upstream's template engine
//!   wraps it to a negative number.
//! - A pass over the token budget, or passes over the request's total, are
//!   refused before any pass runs; upstream fails mid-request. A prompt is
//!   also bounded while it renders: no longer than the budget times the
//!   vocabulary's longest token, which no prompt that fits the budget
//!   exceeds ([`DecisionError::RenderedTooLong`]).
//! - NaN anywhere among a pass's label logits is refused
//!   ([`DecisionError::NonFiniteScores`]); upstream's answer then depends on
//!   label order.
//! - A vocabulary whose control spellings `d1_break` cannot terminate on
//!   (not UTF-8, a single character, or holding U+200B) is refused
//!   ([`DecisionError::UnsupportedVocabulary`]); upstream would loop. So is
//!   one past 4,096 spellings, 256 bytes each or 64 KiB in all (d1-2B has
//!   124, at most 21 bytes, 2,238 in all), and escaping that would take more
//!   than 64 trie steps per byte of a string
//!   ([`DecisionError::EscapeWorkExceeded`]): upstream compares every
//!   spelling at every candidate position, with no bound.
//!
//! As upstream, only CONTROL spellings are broken: USER_DEFINED ones in the
//! caller's text (`<think>`, `</think>`, `<tool_call>`, …) still tokenize as
//! special tokens.
//!
//! # Content
//!
//! Nothing here logs, and every `Debug` prints counts and lengths only:
//! plans hold token ids, label codes and option keys derived from the
//! caller's content.

mod escape;
mod facts;
mod finish;
mod json;
mod labels;
#[cfg(feature = "llm-llamacpp")]
pub mod llama;
mod plan;
#[cfg(feature = "llm-llamacpp")]
mod prepare;
mod protocol;
#[cfg(feature = "llm-llamacpp")]
mod render;

#[cfg(test)]
mod tests;

use std::fmt;

use crate::gguf::{DecisionProfileId, GgufHeader};
use crate::ir::decision::{DecisionError, DecisionRequest, DecisionUsage, Decisions};

use facts::Temperatures;
use plan::AnswerPlan;

/// The sha256 of the one `tokenizer.chat_template.systemone` template this
/// build renders: 1,380 bytes, shared by d1-2B and d1-omni-2B, ending in
/// `<think></think>`.
pub const SYSTEMONE_TEMPLATE_SHA256: &str =
    "ee5fc62461a9bbbcb8389c57f4cd2cd694ccb0dec19521576bd1419049490d8b";

/// A token id of the model's vocabulary (llama.cpp's `llama_token`).
pub type Token = i32;

/// The vocabulary a decision protocol tokenizes with.
///
/// Implemented over a loaded llama.cpp model by `llama::LlamaVocab`
/// (feature `llm-llamacpp`); tests use fakes.
pub trait DecisionVocab: Send + Sync {
    /// Tokenizes `text` as llama.cpp's `common_tokenize` does: with
    /// `add_special` the model's BOS (and EOS) are added if its vocabulary
    /// says so; with `parse_special` the spellings of special tokens in the
    /// text become those tokens. A NUL byte is text like any other.
    ///
    /// # Errors
    ///
    /// [`DecisionError::TokenizationFailed`]; never one that carries the
    /// text or the tokenizer's own message.
    fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Token>, DecisionError>;

    /// The texts of the vocabulary's CONTROL tokens that are 2 bytes or
    /// longer, in token-id order (llama.cpp's `llama_vocab_get_text` of
    /// every token whose attributes include `LLAMA_TOKEN_ATTR_CONTROL`).
    fn control_spellings(&self) -> &[Vec<u8>];

    /// The most bytes of text one token covers: the longest token text of
    /// the vocabulary (at least 1).
    ///
    /// A token's text is at least as long as the bytes it covers (a
    /// byte-level BPE writes a byte as one or two, SentencePiece a space as
    /// three, a byte-fallback token one byte as six), so a text of `n`
    /// bytes takes at least `n / max_token_bytes` tokens, given a tokenizer
    /// that covers every byte of its text (no normalization that drops
    /// bytes; d1-2B's byte-level BPE covers them all). `prepare` refuses a
    /// prompt longer than the token budget times this while it renders.
    fn max_token_bytes(&self) -> usize;
}

/// Default [`DecisionLimits::token_budget`]: 4,096 tokens.
pub const DEFAULT_TOKEN_BUDGET: u32 = 4096;
/// Default [`DecisionLimits::max_questions`]: 16.
pub const DEFAULT_MAX_QUESTIONS: usize = 16;
/// Default [`DecisionLimits::max_options`]: 255, LFM3 d1's own ceiling.
pub const DEFAULT_MAX_OPTIONS: usize = 255;
/// Default [`DecisionLimits::max_total_tokens`]: 16,384 tokens.
pub const DEFAULT_MAX_TOTAL_TOKENS: u64 = 16_384;

/// How much one request may make a decision model do.
///
/// The defaults are provisional, pending device measurements: a 4,096-token
/// pass takes about 0.6 GiB of compute buffer with flash attention, and
/// minutes on a 4-vCPU CPU (contract notes C10, C12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecisionLimits {
    /// Most tokens one pass may hold, inclusive: the context the model's
    /// passes run in.
    pub token_budget: u32,
    /// Most questions (so passes) in one request.
    pub max_questions: usize,
    /// Most options of one choice question. LFM3 d1 reads at most 255
    /// whatever this says.
    pub max_options: usize,
    /// Most tokens all passes of one request may hold together, inclusive.
    pub max_total_tokens: u64,
}

impl Default for DecisionLimits {
    fn default() -> Self {
        Self {
            token_budget: DEFAULT_TOKEN_BUDGET,
            max_questions: DEFAULT_MAX_QUESTIONS,
            max_options: DEFAULT_MAX_OPTIONS,
            max_total_tokens: DEFAULT_MAX_TOTAL_TOKENS,
        }
    }
}

impl DecisionLimits {
    /// Override the per-pass token budget.
    pub fn with_token_budget(mut self, token_budget: u32) -> Self {
        self.token_budget = token_budget;
        self
    }

    /// Override the question ceiling.
    pub fn with_max_questions(mut self, max_questions: usize) -> Self {
        self.max_questions = max_questions;
        self
    }

    /// Override the option ceiling.
    pub fn with_max_options(mut self, max_options: usize) -> Self {
        self.max_options = max_options;
        self
    }

    /// Override the request's token total.
    pub fn with_max_total_tokens(mut self, max_total_tokens: u64) -> Self {
        self.max_total_tokens = max_total_tokens;
        self
    }
}

/// The ceilings a decision profile reads requests under, as `prepare`
/// checks them: what a caller describing a model's capabilities reads,
/// rather than copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileRules {
    /// Most options of a choice question.
    pub max_options: usize,
    /// Fewest levels of a score question.
    pub min_levels: usize,
    /// Most levels of a score question.
    pub max_levels: usize,
    /// Deepest JSON state, counting a value itself
    /// ([`OrderedJson::depth`](crate::ir::decision::OrderedJson::depth)).
    pub max_state_depth: usize,
    /// Deepest JSON instructions.
    pub max_instructions_depth: usize,
    /// Deepest JSON option, yes/no or level description.
    pub max_description_depth: usize,
}

impl ProfileRules {
    /// The rules of `profile`.
    pub const fn of(profile: DecisionProfileId) -> Self {
        match profile {
            DecisionProfileId::Lfm3D1Text => Self {
                max_options: protocol::MAX_OPTIONS,
                min_levels: protocol::MIN_LEVELS,
                max_levels: protocol::MAX_LEVELS,
                max_state_depth: protocol::MAX_STATE_DEPTH,
                max_instructions_depth: protocol::MAX_INSTRUCTIONS_DEPTH,
                max_description_depth: protocol::MAX_DESCRIPTION_DEPTH,
            },
        }
    }
}

/// What the protocol needs from a decision model's GGUF: its profile, its
/// temperatures, and the limits it runs under.
///
/// The scanned header is the single authority: the template is checked
/// against [`SYSTEMONE_TEMPLATE_SHA256`], and each temperature is the value
/// llama.cpp reads (its six-decimal metadata string read back, see
/// [`effective_f32`](crate::gguf::effective_f32)). The caller checks that
/// the loaded model matches the header (`gguf::verify_native_load`).
#[derive(Debug, Clone)]
pub struct DecisionModelFacts {
    profile: DecisionProfileId,
    temperatures: Temperatures,
    limits: DecisionLimits,
}

impl DecisionModelFacts {
    /// The facts of a model whose header classified as `profile`.
    ///
    /// # Errors
    ///
    /// - [`DecisionError::InvalidMetadata`] when the header does not
    ///   classify (a temperature that reads as zero or less, a decision key
    ///   of the wrong type, …);
    /// - [`DecisionError::UnsupportedMetadata`] when it carries a decision
    ///   key the profile does not implement (a score bias);
    /// - [`DecisionError::ProfileMismatch`] when it classifies as anything
    ///   but `profile`;
    /// - [`DecisionError::UnsupportedTemplate`] when its `systemone`
    ///   template is not the pinned one.
    pub fn new(
        profile: DecisionProfileId,
        header: &GgufHeader,
        limits: DecisionLimits,
    ) -> Result<Self, DecisionError> {
        let temperatures = facts::read(profile, header)?;
        Ok(Self {
            profile,
            temperatures,
            limits,
        })
    }

    /// The protocol profile.
    pub fn profile(&self) -> DecisionProfileId {
        self.profile
    }

    /// The limits requests run under.
    pub fn limits(&self) -> &DecisionLimits {
        &self.limits
    }

    /// The ceilings the profile reads requests under.
    pub fn rules(&self) -> ProfileRules {
        ProfileRules::of(self.profile)
    }
}

/// One model evaluation: a prompt's tokens, and the label tokens whose
/// logits are read at its last token.
///
/// `Debug` prints counts: token ids spell the caller's content.
#[derive(Clone, PartialEq, Eq)]
pub struct DecisionPass {
    tokens: Vec<Token>,
    select: Vec<Token>,
}

impl DecisionPass {
    /// The prompt's tokens: sequence 0, positions `0..len`, evaluated as one
    /// batch.
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// The tokens whose logits the readout returns, in this order,
    /// duplicates kept: every label group's tokens, concatenated.
    pub fn select(&self) -> &[Token] {
        &self.select
    }

    /// The position whose logits are read: the last token.
    pub fn output_index(&self) -> usize {
        self.tokens.len() - 1
    }
}

impl fmt::Debug for DecisionPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionPass")
            .field("tokens", &self.tokens.len())
            .field("select", &self.select.len())
            .finish()
    }
}

/// A prepared request: the passes to evaluate, in order, and how each
/// question's answer is read from them.
///
/// `Debug` prints counts only.
#[derive(Clone)]
pub struct DecisionPlan {
    passes: Vec<DecisionPass>,
    answers: Vec<AnswerPlan>,
}

impl DecisionPlan {
    /// The passes, in question order (LFM3 d1: one per question).
    pub fn passes(&self) -> &[DecisionPass] {
        &self.passes
    }

    /// What the passes take: their number and their tokens together.
    pub fn usage(&self) -> DecisionUsage {
        let passes = u32::try_from(self.passes.len())
            .expect("prepare refuses more questions than a u32 counts");
        let tokens = self
            .passes
            .iter()
            .map(|pass| pass.tokens.len() as u64)
            .sum();
        DecisionUsage::new(passes, tokens)
    }
}

impl fmt::Debug for DecisionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionPlan")
            .field("passes", &self.passes)
            .field("answers", &self.answers.len())
            .finish()
    }
}

/// Prepares `request` for a model with these `facts` and this `vocab`: checks
/// everything the protocol reads, then renders, tokenizes and labels one pass
/// per question.
///
/// Nothing is evaluated: the whole request is checked and prepared first, so
/// a refusal costs no inference. Only `vocab` is called (to tokenize).
///
/// # Ceilings
///
/// A pass's prompt renders to at most `token_budget ×`
/// [`DecisionVocab::max_token_bytes`] bytes: no prompt that fits the budget
/// is longer, since no token covers more bytes than the vocabulary's
/// longest token text. Rendering writes into a buffer that refuses to grow
/// past it, `tojson`'s own text included, and a longer prompt is refused
/// as [`DecisionError::RenderedTooLong`] (kind `budget_exceeded`) before it
/// is tokenized; only its label forms (one short code each) are tokenized
/// before that, because the prompt shows the codes. That is the effective
/// per-pass ceiling (d1-2B: 4,096 × 386 = 1,581,056 bytes), and a request
/// renders at most `max_questions` (16 by default) of them. Escaping is
/// bounded on its own (`escape.rs`: capped CONTROL spellings, a trie walk per
/// position, a work ceiling per string), not by the prompt's cap: some
/// escaped fields (the question id) are never rendered.
///
/// # Errors
///
/// The first problem found, as a content-free [`DecisionError`] whose
/// [`kind`](DecisionError::kind) names it: a structural problem
/// ([`validate_structure`](crate::ir::decision::validate_structure) under the
/// facts' limits), missing instructions, a score with other than 2 to 10
/// levels, more than 255 options, media, JSON deeper than LFM3 d1 reads
/// (127 levels for the state and instructions, 125 for descriptions), an
/// integer above `i64::MAX`, an escaped-key collision, no single-token label
/// left, a prompt longer than the token budget can hold (refused while it
/// renders, at the budget times [`DecisionVocab::max_token_bytes`]), a pass
/// over the token budget or passes over the request's total;
/// [`DecisionError::UnsupportedVocabulary`] for control spellings `d1_break`
/// cannot handle, and [`DecisionError::RenderFailed`] or
/// [`DecisionError::TokenizationFailed`] when a step fails.
#[cfg(feature = "llm-llamacpp")]
pub fn prepare(
    request: &DecisionRequest,
    facts: &DecisionModelFacts,
    vocab: &dyn DecisionVocab,
) -> Result<DecisionPlan, DecisionError> {
    prepare::prepare(request, facts, vocab, None)
}

/// Reads the answers of `request` from the readouts of the passes `plan`
/// prepared for it: `readouts[i]` holds the logits of
/// `plan.passes()[i].select()`, in order.
///
/// Each label group scores its highest logit; a question's distribution is
/// `exp((s - max) / T)` over its options, normalized, in `f64` and in option
/// order, at the model's temperature for the question. A choice is the first
/// most probable option, with that probability as confidence; yes/no is the
/// probability of `"true"`, with no confidence; a score is the first level
/// whose cumulative probability reaches one half, with the highest
/// probability as confidence.
///
/// # Errors
///
/// [`DecisionError::ReadoutCount`] or [`DecisionError::ReadoutLength`] for
/// readouts that do not match the plan (no partial result);
/// [`DecisionError::NonFiniteScores`] for a NaN or `+inf` logit in a pass, or
/// a question whose scores are all `-inf`.
///
/// # Panics
///
/// When `plan` was not prepared for `request` (a different number of
/// questions or options): a bug in the caller.
pub fn finish(
    request: &DecisionRequest,
    plan: &DecisionPlan,
    readouts: Vec<Vec<f32>>,
) -> Result<Decisions, DecisionError> {
    finish::finish(request, plan, readouts)
}
