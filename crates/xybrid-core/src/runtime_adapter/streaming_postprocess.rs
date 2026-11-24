//! Streaming-text post-processing shared across LLM backends.
//!
//! Pulls stop-pattern detection, `<think>...</think>` stripping, and
//! the safe-prefix buffer logic out of the individual backends so any
//! future engine (MLX, CoreML, ...) can reuse them. These concerns
//! are intrinsic to **chat-style model output**, not to any specific
//! engine — they belong above the engine layer.
//!
//! # Usage
//!
//! - Non-streaming path: call [`truncate_at_first_stop`],
//!   [`trim_partial_stop_suffix`], and [`strip_thinking_tags`] on the
//!   full output text.
//! - Streaming path: instantiate [`StreamingTextFilter`] once, feed
//!   raw chunk text in via [`StreamingTextFilter::push`], emit
//!   whatever it returns to the user callback.
//!
//! Engines that strip these internally (e.g. mistralrs) don't use
//! this module.
//!
//! Output framed differently (Inkling writes every part of its turn as a
//! message block) is split by its own blocks instead; see [`OutputFraming`].

use std::borrow::Cow;

use crate::gateway::{Tool, ToolCall};
use crate::runtime_adapter::tool_call::{
    decode_inkling_blocks, is_inkling_prompt, parse_tool_calls, parse_tool_calls_in,
    parse_tool_calls_with_tools, InklingBlockDecoder, ToolCallProtocol, INKLING_TOOL_BLOCK_MARKERS,
    INKLING_TURN_END_MARKERS, TOOL_BLOCK_MARKERS,
};

/// Stop markers emitted by common chat templates:
/// - `<|im_end|>` / `<|im_start|>` / `<|endoftext|>`: ChatML (Qwen, Phi)
/// - `</s>`: Llama 2 style
/// - `<end_of_turn>`: Gemma 3 / Gemma 3n
/// - `<turn|>`: Gemma 4. The Gemma 4 GGUFs published as
///   `ggml-org/gemma-4-E2B-it-GGUF` decode the chat end-of-turn
///   special token to the literal string `<turn|>` rather than
///   `<end_of_turn>`. Confirmed against llama.cpp's own vocab table —
///   `vendor/llama.cpp/src/llama-vocab.cpp` lists `<turn|>` as a
///   `gemma4` EOG marker. Without this entry the marker leaks as the
///   trailing tail of caption text.
/// - `<end_of_utterance>`: SmolVLM / Idefics-style VLM chat templates
///   use this marker for user and assistant message boundaries. llama.cpp
///   detects SmolVLM templates from this marker and also treats it as an
///   EOT token, so Xybrid must filter the decoded literal as well.
///
/// Always check for these in addition to user-supplied stop sequences —
/// they're emitted by the chat template, not by the user. A framing adds its
/// own ([`OutputFraming::stop_patterns`]).
pub(crate) const CHAT_STOP_PATTERNS: &[&str] = &[
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "</s>",
    "<end_of_turn>",
    "<start_function_response>",
    "<turn|>",
    "<end_of_utterance>",
];

/// Fallback variants without the leading `<`, for models whose
/// tokenizer breaks the angle bracket off from the marker body.
///
/// Only safe to use in **final-text cleanup** — during streaming
/// these would false-positive on legitimate text that happens to
/// start with `|`. [`StreamingTextFilter`] does NOT include them.
/// Only a complete match counts: [`trim_partial_stop_suffix`] skips
/// them, because without the `<` their prefixes are ordinary text
/// (`end_of_turn>` starts with `e`).
///
/// NOTE: deliberately omits a `turn|>` entry. The marker body starts
/// with the very common letter `t`, and `trim_partial_stop_suffix`
/// would then trim any final answer ending in `t`, `tu`, `tur`, or
/// `turn` — see the test
/// `trim_partial_stop_suffix_does_not_chop_short_words_with_turn_marker`
/// below for the regression guard.
pub(crate) const CHAT_STOP_PATTERNS_BROKEN: &[&str] =
    &["|im_end|>", "|im_start|>", "|endoftext|>", "end_of_turn>"];

/// How a chat turn's raw output is framed, read off the prompt it follows.
///
/// Every family but Inkling writes its turn as answer text, with reasoning in
/// a [`REASONING_BLOCK_MARKERS`] block and tool calls inline in any supported
/// notation. Inkling writes each part as its own message block: its turn is
/// split by those blocks ([`InklingBlockDecoder`]), so `<think>` or another
/// family's call notation inside a block is plain text, and only its own call
/// blocks are calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputFraming {
    /// Answer text with inline reasoning and tool-call blocks.
    Plain,
    /// Inkling message blocks.
    Inkling,
}

impl OutputFraming {
    /// The framing of the output generated after `prompt`.
    pub(crate) fn of_prompt(prompt: &str) -> Self {
        if is_inkling_prompt(prompt) {
            Self::Inkling
        } else {
            Self::Plain
        }
    }

    /// Whether output in this framing can start inside a primed `<think>`.
    /// Inkling opens its own thinking blocks and is never primed.
    pub(crate) fn allows_think_priming(self) -> bool {
        self == Self::Plain
    }

    /// Stop patterns this framing adds to [`CHAT_STOP_PATTERNS`]: Inkling's
    /// end-of-turn markers, which other families' output may quote.
    pub(crate) fn stop_patterns(self) -> &'static [&'static str] {
        match self {
            Self::Plain => &[],
            Self::Inkling => INKLING_TURN_END_MARKERS,
        }
    }

    /// The streaming filter for a turn in this framing. Feed it the output of
    /// [`Self::decoder`].
    ///
    /// Plain output is filtered for every recognized reasoning block,
    /// starting inside one when `primed`, and, with `suppress_tools`, for
    /// every family's tool-call blocks. Inkling's reasoning never reaches the
    /// filter (the decoder keeps it apart), and only its own call blocks are
    /// suppressed.
    pub(crate) fn streaming_filter(
        self,
        stop_patterns: Vec<String>,
        primed: bool,
        suppress_tools: bool,
    ) -> StreamingTextFilter {
        let filter = match self {
            Self::Plain if primed => StreamingTextFilter::new_reasoning_primed(stop_patterns),
            Self::Plain => StreamingTextFilter::new(stop_patterns),
            Self::Inkling => StreamingTextFilter::new(stop_patterns).for_inkling_blocks(),
        };
        if suppress_tools {
            filter.with_tool_call_suppression()
        } else {
            filter
        }
    }

    /// Split a finished turn, already cut at its stop patterns, into the
    /// answer and the reasoning.
    ///
    /// Plain output goes through [`strip_and_capture_thinking_tags_primed`].
    /// Inkling output is split by its blocks: the answer is its text and
    /// tool-call blocks, the reasoning its thinking blocks.
    pub(crate) fn split_reasoning(
        self,
        text: &str,
        primed: bool,
        truncated_by_budget: bool,
    ) -> (String, Option<String>) {
        match self {
            Self::Plain => {
                strip_and_capture_thinking_tags_primed(text, primed, truncated_by_budget)
            }
            Self::Inkling => {
                let turn = decode_inkling_blocks(text);
                let reasoning = turn.reasoning.trim();
                let reasoning = (!reasoning.is_empty()).then(|| reasoning.to_string());
                (turn.visible, reasoning)
            }
        }
    }

    /// The tool calls in an answer from [`Self::split_reasoning`]: every
    /// family's for plain output, only Inkling's call blocks for Inkling's.
    pub(crate) fn parse_tool_calls(self, text: &str, tools: &[Tool]) -> Vec<ToolCall> {
        match self {
            Self::Plain => parse_tool_calls_with_tools(text, tools),
            Self::Inkling => parse_tool_calls_in(ToolCallProtocol::Inkling, text, tools),
        }
    }

    /// The calls to record on the turn's output for callers that only see
    /// its text (`GenerationOutput::tool_calls`). `None` for plain output,
    /// whose text any caller can parse; Inkling's are parsed here, where the
    /// framing is known.
    pub(crate) fn output_tool_calls(self, text: &str, tools: &[Tool]) -> Option<Vec<ToolCall>> {
        match self {
            Self::Plain => None,
            Self::Inkling => Some(self.parse_tool_calls(text, tools)),
        }
    }

    /// A decoder for one streamed turn.
    pub(crate) fn decoder(self) -> FramingDecoder {
        FramingDecoder(match self {
            Self::Plain => None,
            Self::Inkling => Some(InklingBlockDecoder::new()),
        })
    }
}

/// Turns a streamed turn's raw chunks into the text its
/// [`OutputFraming::streaming_filter`] reads: plain output unchanged, an
/// Inkling turn's visible text (its reasoning is kept apart).
pub(crate) struct FramingDecoder(Option<InklingBlockDecoder>);

impl FramingDecoder {
    /// Decode the next raw chunk.
    pub(crate) fn push<'a>(&mut self, chunk: &'a str) -> Cow<'a, str> {
        match &mut self.0 {
            None => Cow::Borrowed(chunk),
            Some(decoder) => Cow::Owned(decoder.push(chunk).visible),
        }
    }

    /// The text still held back once the turn has ended.
    pub(crate) fn finish(&mut self) -> String {
        self.0
            .as_mut()
            .map(|decoder| decoder.finish().visible)
            .unwrap_or_default()
    }
}

/// Merge caller-supplied patterns with defaults, de-duplicated while
/// preserving caller order first.
pub(crate) fn merge_stop_patterns<S: AsRef<str>>(user: &[S], extras: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = user.iter().map(|s| s.as_ref().to_string()).collect();
    for e in extras {
        if !out.iter().any(|s| s == e) {
            out.push((*e).to_string());
        }
    }
    out
}

/// The reasoning opener xybrid primes onto the assistant turn (see the
/// llama.cpp adapter's `THINK_PRIME`). The primed / dangling-close capture path
/// is keyed specifically on this pair — `<think>` is the only form xybrid primes.
const PRIMED_OPEN: &str = "<think>";
const PRIMED_CLOSE: &str = "</think>";

/// `(open, close)` delimiter pairs recognized as chain-of-thought / reasoning
/// blocks. A balanced block in any of these forms is captured into
/// `reasoning_content` and stripped from the answer, in both the non-streaming
/// final-cleanup pass ([`strip_and_capture_thinking_tags`]) and the streaming
/// filter ([`StreamingTextFilter`]). This is the single place to add a format.
///
/// - `<think>` — Qwen3 / DeepSeek-R1 / LFM2.5-thinking (the form xybrid primes).
/// - `<thinking>` — Anthropic-style reasoning.
/// - `<|channel>thought` … `<channel|>` — gemma-4's thought channel. Forward
///   coverage: xybrid runs gemma-4 with thinking disabled today (the raw-decode
///   backend has no channel parser), so these are derived from gemma-4's
///   template shape and are not yet exercised against a real reasoning-emitting
///   gemma-4 run. They won't false-positive on ordinary text; correct them here
///   once verified against a real channel-emitting model.
pub(crate) const REASONING_BLOCK_MARKERS: &[(&str, &str)] = &[
    (PRIMED_OPEN, PRIMED_CLOSE),
    ("<thinking>", "</thinking>"),
    ("<|channel>thought", "<channel|>"),
];

/// Strip every recognized reasoning block (see [`REASONING_BLOCK_MARKERS`])
/// from `text`.
///
/// An unclosed opening tag strips from the opener to end of string —
/// this is the partial-stream safety case for Qwen 3.5 and similar
/// models that emit reasoning blocks before the final answer.
///
/// This discards the reasoning text. Use [`strip_and_capture_thinking_tags`]
/// when the caller wants to surface the chain-of-thought (e.g. populate
/// `GenerationOutput::reasoning_content`).
pub(crate) fn strip_thinking_tags(text: &str) -> String {
    strip_and_capture_thinking_tags(text).0
}

/// Like [`strip_thinking_tags`], but also returns the captured reasoning
/// text instead of discarding it.
///
/// Returns `(clean, reasoning)` where `clean` is `text` with every
/// `<think>...</think>` block removed and `reasoning` is the concatenated
/// inner text of those blocks (multiple blocks joined by `\n`), or `None`
/// when the input contained no reasoning block. The `<think>` / `</think>`
/// delimiters themselves are not included in `reasoning`.
///
/// An unclosed opening tag contributes everything after `<think>` to
/// `reasoning` and strips it from `clean` — matching the partial-stream
/// truncation behavior of [`strip_thinking_tags`].
///
/// A **dangling close** — a `</think>` with no `<think>` before it — is also
/// captured: everything up to that `</think>` becomes reasoning. This is the
/// shape a thinking model produces when the opening `<think>` was primed into
/// the prompt (see the llama.cpp adapter's `THINK_PRIME`), so the model's
/// output is `reasoning</think>answer`. Handling it here means every caller —
/// streaming and non-streaming — recovers reasoning without knowing whether the
/// tag was primed.
pub(crate) fn strip_and_capture_thinking_tags(text: &str) -> (String, Option<String>) {
    let mut clean = text.to_string();
    let mut reasoning = String::new();

    // Dangling close(es): a primed `</think>` with no `<think>` before it (the
    // opening tag was primed into the prompt). Keyed specifically on the primed
    // pair — `<think>` is the only form xybrid primes. Peel each leading
    // reasoning segment before the balanced-block pass. Looping handles a model
    // that emits more than one dangling close (repetition / multiple primed
    // turns) — each would otherwise leak into the answer.
    while let Some(close) = clean.find(PRIMED_CLOSE) {
        // Stop once a `<think>` opens before the next close: that's a balanced
        // block, handled by the pass below.
        if clean.find(PRIMED_OPEN).is_some_and(|open| open < close) {
            break;
        }
        push_reasoning(&mut reasoning, &clean[..close]);
        clean.replace_range(..close + PRIMED_CLOSE.len(), "");
    }

    // Balanced blocks in any recognized marker form: repeatedly strip the
    // earliest opener across all markers together with its matching close.
    loop {
        let Some((start, open, close)) = REASONING_BLOCK_MARKERS
            .iter()
            .filter_map(|&(open, close)| clean.find(open).map(|pos| (pos, open, close)))
            .min_by_key(|&(pos, ..)| pos)
        else {
            break;
        };

        let inner_start = start + open.len();
        if let Some(end_rel) = clean[inner_start..].find(close) {
            let inner_end = inner_start + end_rel;
            push_reasoning(&mut reasoning, &clean[inner_start..inner_end]);
            clean.replace_range(start..inner_end + close.len(), "");
        } else {
            // Unclosed opener — everything after it is reasoning (partial-stream
            // safety; the model is still mid-thought).
            push_reasoning(&mut reasoning, &clean[inner_start..]);
            clean.truncate(start);
            break;
        }
    }

    // A whitespace-only block (e.g. a thinking model that opened and closed with
    // nothing in between) is not reasoning — surface `None` so callers don't show
    // an empty reasoning section.
    let reasoning = reasoning.trim();
    let reasoning = (!reasoning.is_empty()).then(|| reasoning.to_string());
    (clean, reasoning)
}

/// Whether output generated after `prompt` starts inside a reasoning block:
/// the prompt ends in the `<think>` opener, primed by the llama.cpp adapter
/// (`THINK_PRIME`) or written by the chat template itself (LFM3 with thinking
/// on, including its tool continuations). The model's reasoning then carries
/// no opener of its own.
///
/// The chat paths' double-prime guard applies the same test before appending
/// `THINK_PRIME`. LFM3's thinking-off state (`<think></think>`) is closed, and
/// raw codec-TTS prompts never end in `<think>`.
pub(crate) fn is_reasoning_primed(prompt: &str) -> bool {
    prompt.trim_end().ends_with(PRIMED_OPEN)
}

/// Like [`strip_and_capture_thinking_tags`], for output generated from a
/// reasoning-primed prompt (the llama.cpp adapter's `THINK_PRIME`). The opener
/// lives in the prompt, so output whose budget ran out before `</think>` is
/// mid-thought prose with no marker at all — indistinguishable from an answer
/// without the priming bit. Reconstructing the opener only for budget-truncated
/// output keeps marker-bearing behavior identical to the unprimed pass while
/// preserving a naturally terminated markerless answer. A rare true
/// mid-thought EOS is therefore surfaced as answer text rather than silently
/// hiding it, which is the safer failure mode for callers.
pub(crate) fn strip_and_capture_thinking_tags_primed(
    text: &str,
    primed: bool,
    truncated_by_budget: bool,
) -> (String, Option<String>) {
    if !primed || !truncated_by_budget {
        return strip_and_capture_thinking_tags(text);
    }
    let mut reconstructed = String::with_capacity(PRIMED_OPEN.len() + text.len());
    reconstructed.push_str(PRIMED_OPEN);
    reconstructed.push_str(text);
    strip_and_capture_thinking_tags(&reconstructed)
}

/// Append one reasoning block, separating consecutive blocks with a newline.
fn push_reasoning(buf: &mut String, block: &str) {
    if !buf.is_empty() {
        buf.push('\n');
    }
    buf.push_str(block);
}

/// Truncate `text` at the earliest complete occurrence of any pattern
/// in `patterns`. Returns `true` if a stop was found and truncated.
///
/// An empty pattern is skipped, as the native llama.cpp loop skips one:
/// it would match at offset 0 and erase the whole text.
pub(crate) fn truncate_at_first_stop<S: AsRef<str>>(text: &mut String, patterns: &[S]) -> bool {
    let mut earliest: Option<usize> = None;
    for p in patterns.iter().map(AsRef::as_ref).filter(|p| !p.is_empty()) {
        if let Some(pos) = text.find(p) {
            earliest = Some(match earliest {
                None => pos,
                Some(cur) => cur.min(pos),
            });
        }
    }
    if let Some(pos) = earliest {
        text.truncate(pos);
        true
    } else {
        false
    }
}

/// Trim a trailing partial-stop suffix from `text` — e.g. remove
/// `<|im_` when the full pattern is `<|im_end|>`. Only strict
/// prefixes (length `1..pattern.len()`) are trimmed; complete
/// matches are the job of [`truncate_at_first_stop`].
///
/// [`CHAT_STOP_PATTERNS_BROKEN`] entries are skipped: their prefixes are
/// ordinary text, and `end_of_turn>` trimmed the `e` off every answer
/// ending in one (`"true"` came back as `"tru"`).
///
/// Returns `true` on first trim and stops (matches the prior
/// llama_cpp behavior).
pub(crate) fn trim_partial_stop_suffix<S: AsRef<str>>(text: &mut String, patterns: &[S]) -> bool {
    for pattern in patterns {
        let p = pattern.as_ref();
        if CHAT_STOP_PATTERNS_BROKEN.contains(&p) {
            continue;
        }
        for prefix_len in 1..p.len() {
            let prefix = &p[..prefix_len];
            if text.ends_with(prefix) {
                text.truncate(text.len() - prefix_len);
                return true;
            }
        }
    }
    false
}

/// Stateful text filter for a streaming generation loop.
///
/// Feed raw chunk text in via [`Self::push`]; the filter returns the
/// portion that's safe to emit to the user callback, holding back
/// any trailing bytes that could turn into a stop pattern once more
/// chunks arrive. It also transparently suppresses reasoning blocks in any
/// recognized form (see [`REASONING_BLOCK_MARKERS`]) and stops emitting once a
/// complete stop pattern is observed (see [`Self::is_stopped`]).
///
/// After [`Self::is_stopped`] returns `true`, further `push` calls
/// are no-ops and return `None`. The backend should still drain the
/// engine's stream, but nothing further will be emitted upward.
pub(crate) struct StreamingTextFilter {
    stop_patterns: Vec<String>,
    cumulative_text: String,
    last_emitted_len: usize,
    /// The close marker of the reasoning block currently being suppressed, or
    /// `None` when not inside a block. Holds the specific closer to wait for so
    /// mixed marker forms (`</think>`, `</thinking>`, `<channel|>`) don't confuse
    /// each other.
    active_reasoning_close: Option<&'static str>,
    hit_stop_pattern: bool,
    reasoning: String,
    suppress_tool_call_blocks: bool,
    saw_tool_call_block: bool,
    /// The reasoning delimiters recognized in the text:
    /// [`REASONING_BLOCK_MARKERS`], or none for Inkling, whose reasoning
    /// never reaches the filter.
    reasoning_markers: &'static [(&'static str, &'static str)],
    /// The tool-call blocks suppressed, and the one protocol whose calls they
    /// must parse as (`None`: every family's).
    tool_block_markers: &'static [(&'static str, &'static str, bool)],
    tool_protocol: Option<ToolCallProtocol>,
}

impl StreamingTextFilter {
    pub fn new(mut stop_patterns: Vec<String>) -> Self {
        // An empty pattern is not a stop (the native llama.cpp loop drops one
        // too); every text contains it, so it would end the stream before the
        // first chunk.
        stop_patterns.retain(|pattern| !pattern.is_empty());
        Self {
            stop_patterns,
            cumulative_text: String::new(),
            last_emitted_len: 0,
            active_reasoning_close: None,
            hit_stop_pattern: false,
            reasoning: String::new(),
            suppress_tool_call_blocks: false,
            saw_tool_call_block: false,
            reasoning_markers: REASONING_BLOCK_MARKERS,
            tool_block_markers: TOOL_BLOCK_MARKERS,
            tool_protocol: None,
        }
    }

    /// Construct a filter for a turn whose `<think>` channel was primed into
    /// the prompt (thinking models — see the llama.cpp adapter's `THINK_PRIME`).
    ///
    /// The model's output begins with reasoning and *no* opening tag, so the
    /// filter starts already inside the think block: it suppresses the reasoning
    /// from the emit stream and captures it, resuming emission after the model's
    /// `</think>`.
    pub fn new_reasoning_primed(stop_patterns: Vec<String>) -> Self {
        Self {
            // Start inside the primed `<think>` block, waiting for its close.
            active_reasoning_close: Some(PRIMED_CLOSE),
            ..Self::new(stop_patterns)
        }
    }

    /// Suppress complete tool-protocol blocks (and hold back potential
    /// marker prefixes) from the emitted stream. Enabled only for requests
    /// that offered tools.
    pub fn with_tool_call_suppression(mut self) -> Self {
        self.suppress_tool_call_blocks = true;
        self
    }

    /// Filter an Inkling turn's visible text: recognize no reasoning
    /// delimiters (`<think>` there is answer text) and, with tool
    /// suppression, only Inkling's own call blocks.
    pub fn for_inkling_blocks(mut self) -> Self {
        self.reasoning_markers = &[];
        self.tool_block_markers = INKLING_TOOL_BLOCK_MARKERS;
        self.tool_protocol = Some(ToolCallProtocol::Inkling);
        self
    }

    /// Whether a complete stop pattern has been observed.
    pub fn is_stopped(&self) -> bool {
        self.hit_stop_pattern
    }

    /// Whether a complete tool-*call* block was suppressed. Response blocks
    /// don't count: they are protocol scaffolding, not calls.
    pub fn saw_tool_call_block(&self) -> bool {
        self.saw_tool_call_block
    }

    /// The reasoning text captured from recognized reasoning blocks (see
    /// [`REASONING_BLOCK_MARKERS`]) seen so far, or `None` if the stream emitted
    /// no closed reasoning block.
    ///
    /// Only blocks that have been *closed* (their close marker arrived) are
    /// captured here — an unclosed trailing block is suppressed from the
    /// emit stream but not surfaced as reasoning, since the streaming filter
    /// can't tell a still-open block from a malformed one. Backends that run
    /// a final [`strip_and_capture_thinking_tags`] pass over the full decoded
    /// text recover the unclosed tail there.
    pub fn reasoning(&self) -> Option<&str> {
        (!self.reasoning.is_empty()).then_some(self.reasoning.as_str())
    }

    /// Cumulative text up to the last emission point. Use this to
    /// populate `PartialToken.cumulative_text` on user callbacks —
    /// it always ends where the last emitted chunk ended.
    pub fn cumulative_emitted(&self) -> &str {
        &self.cumulative_text[..self.last_emitted_len]
    }

    /// Push a raw chunk. Returns `Some(safe_text)` if new content is
    /// ready for the user callback; `None` if the chunk is fully
    /// held back (partial stop prefix, inside a reasoning block, or
    /// after a stop has been observed).
    pub fn push(&mut self, chunk: &str) -> Option<String> {
        if self.hit_stop_pattern {
            return None;
        }

        self.cumulative_text.push_str(chunk);

        // Enter a reasoning block? Detect the earliest opener among the
        // recognized marker forms and remember its matching close.
        if self.active_reasoning_close.is_none() {
            let scan_start = self.last_emitted_len;
            if let Some((pos, close)) = self
                .reasoning_markers
                .iter()
                .filter_map(|&(open, close)| {
                    self.cumulative_text[scan_start..]
                        .find(open)
                        .map(|rel| (scan_start + rel, close))
                })
                .min_by_key(|&(pos, _)| pos)
            {
                self.active_reasoning_close = Some(close);
                self.last_emitted_len = pos;
            }
        }

        if let Some(close) = self.active_reasoning_close {
            if self.cumulative_text.contains(close) {
                self.active_reasoning_close = None;
                let (clean, reasoning) = strip_and_capture_thinking_tags(&self.cumulative_text);
                if let Some(block) = reasoning {
                    push_reasoning(&mut self.reasoning, &block);
                }
                self.cumulative_text = clean;
                // After stripping, last_emitted_len may point past end.
                self.last_emitted_len = self.last_emitted_len.min(self.cumulative_text.len());
            }
            return None;
        }

        if self.suppress_tool_call_blocks {
            self.suppress_complete_tool_blocks();
        }

        // Complete stop pattern observed? Cut at the earliest one, as the
        // final-text cleanup does, and fall through: the chunk that completes
        // a stop can carry text ahead of it ("The" completes the stop "he"),
        // and that text is emitted below like any other. Later pushes return
        // nothing.
        if truncate_at_first_stop(&mut self.cumulative_text, &self.stop_patterns) {
            self.hit_stop_pattern = true;
            // The stop can complete at a position below text that was
            // already emitted (a partial prefix under-held in an earlier
            // chunk). Re-establish the boundary invariant
            // (`last_emitted_len <= cumulative_text.len()`) exactly like the
            // think-strip branch above, or `cumulative_emitted()` slices out
            // of bounds and panics.
            self.last_emitted_len = self.last_emitted_len.min(self.cumulative_text.len());
        }

        // Find the safe emission boundary (exclude potential partial
        // stop prefixes hanging off the tail).
        let mut safe_end = find_potential_stop_start(&self.cumulative_text, &self.stop_patterns)
            .unwrap_or(self.cumulative_text.len());
        if self.suppress_tool_call_blocks {
            if let Some(start) = self.open_tool_block_start() {
                safe_end = safe_end.min(start);
            }
            if let Some(start) =
                find_potential_tool_start(&self.cumulative_text, self.tool_block_markers)
            {
                safe_end = safe_end.min(start);
            }
        }
        // Hold back a partial reasoning-opener prefix at the tail so a marker
        // split across chunks isn't emitted before the block is recognized.
        if let Some(start) =
            find_potential_reasoning_start(&self.cumulative_text, self.reasoning_markers)
        {
            safe_end = safe_end.min(start);
        }

        if safe_end > self.last_emitted_len {
            let safe = self.cumulative_text[self.last_emitted_len..safe_end].to_string();
            self.last_emitted_len = safe_end;
            Some(safe)
        } else {
            None
        }
    }

    fn suppress_complete_tool_blocks(&mut self) {
        let mut scan_start = self.last_emitted_len;
        loop {
            let search = &self.cumulative_text[scan_start..];
            let Some((relative_start, start, end, is_call_block)) =
                find_next_tool_block(search, self.tool_block_markers)
            else {
                break;
            };
            let start_index = scan_start + relative_start;
            let after_start_index = start_index + start.len();
            let Some(relative_end) = self.cumulative_text[after_start_index..].find(end) else {
                break;
            };
            let remove_end = after_start_index + relative_end + end.len();

            if is_call_block && !self.parses_as_call(&self.cumulative_text[start_index..remove_end])
            {
                scan_start = remove_end;
                continue;
            }

            self.cumulative_text
                .replace_range(start_index..remove_end, "");
            // `last_emitted_len <= start_index` holds by construction here
            // (`scan_start` begins at the emit boundary and only advances),
            // so the removal can't strand the boundary past the end — and it
            // must NOT advance to `start_index` either: text between the
            // boundary and the removed block is still pending emission.
            if is_call_block {
                self.saw_tool_call_block = true;
            }
            scan_start = start_index;
        }
    }

    /// Whether a complete call block holds a call this filter suppresses.
    fn parses_as_call(&self, block: &str) -> bool {
        match self.tool_protocol {
            Some(protocol) => !parse_tool_calls_in(protocol, block, &[]).is_empty(),
            None => !parse_tool_calls(block).is_empty(),
        }
    }

    fn open_tool_block_start(&self) -> Option<usize> {
        let mut scan_start = self.last_emitted_len;
        loop {
            let search = &self.cumulative_text[scan_start..];
            let (relative_start, start, end, _) =
                find_next_tool_block(search, self.tool_block_markers)?;
            let start_index = scan_start + relative_start;
            let after_start_index = start_index + start.len();
            let Some(relative_end) = self.cumulative_text[after_start_index..].find(end) else {
                return Some(start_index);
            };
            scan_start = after_start_index + relative_end + end.len();
        }
    }
}

/// Find the position where a potential partial-stop prefix begins at
/// the tail of `text`, if any. Kept internal — callers use
/// [`StreamingTextFilter`] instead.
///
/// Per pattern the LONGEST matching tail prefix wins, and across patterns the
/// earliest hold position wins: holding less than the longest match emits
/// text that a later chunk can complete into a stop pattern *behind* the
/// emission boundary (the user sees text past the stop, and the completion
/// truncates `cumulative_text` below `last_emitted_len`). Comparison is
/// byte-wise so a multi-byte UTF-8 stop pattern can't slice mid-char; the
/// returned hold is then rounded down to a char boundary (holding more is
/// always safe, and callers slice `text` at this position).
fn find_potential_stop_start(text: &str, patterns: &[String]) -> Option<usize> {
    let text_bytes = text.as_bytes();
    patterns
        .iter()
        .filter_map(|pattern| {
            let pattern = pattern.as_bytes();
            (1..=pattern.len())
                .rev()
                .find(|&prefix_len| text_bytes.ends_with(&pattern[..prefix_len]))
                .map(|prefix_len| text.len() - prefix_len)
        })
        .min()
        .map(|mut start| {
            while start > 0 && !text.is_char_boundary(start) {
                start -= 1;
            }
            start
        })
}

fn find_next_tool_block(
    text: &str,
    markers: &'static [(&'static str, &'static str, bool)],
) -> Option<(usize, &'static str, &'static str, bool)> {
    markers
        .iter()
        .filter_map(|(start, end, is_call_block)| {
            text.find(start)
                .map(|position| (position, *start, *end, *is_call_block))
        })
        .min_by_key(|(position, _, _, _)| *position)
}

fn find_potential_tool_start(text: &str, markers: &[(&str, &str, bool)]) -> Option<usize> {
    markers
        .iter()
        .filter_map(|(start, _, _)| potential_suffix_start(text, start))
        .min()
}

/// Position where a partial reasoning-opener prefix begins at the tail of
/// `text`, if any — so a reasoning marker split across chunks is held back
/// rather than leaked to the emit stream. Mirrors [`find_potential_tool_start`].
fn find_potential_reasoning_start(text: &str, markers: &[(&str, &str)]) -> Option<usize> {
    markers
        .iter()
        .filter_map(|(open, _)| potential_suffix_start(text, open))
        .min()
}

fn potential_suffix_start(text: &str, pattern: &str) -> Option<usize> {
    let mut start = None;
    for prefix_len in 1..=pattern.len() {
        let prefix = &pattern[..prefix_len];
        if text.ends_with(prefix) {
            start = Some(text.len() - prefix_len);
        }
    }
    start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primed_strip_captures_markerless_mid_thought_as_reasoning() {
        // Budget exhausted before `</think>`: the whole output is still inside
        // the primed think channel, so none of it is answer text.
        let (clean, reasoning) =
            strip_and_capture_thinking_tags_primed("Okay, so I need to count rooks", true, true);
        assert_eq!(clean, "");
        assert_eq!(reasoning.as_deref(), Some("Okay, so I need to count rooks"));
    }

    #[test]
    fn primed_strip_matches_dangling_close_behavior_for_completed_thoughts() {
        let (clean, reasoning) =
            strip_and_capture_thinking_tags_primed("first the plan</think>the answer", true, true);
        assert_eq!(clean, "the answer");
        assert_eq!(reasoning.as_deref(), Some("first the plan"));
    }

    #[test]
    fn primed_strip_matches_dangling_close_behavior_after_natural_termination() {
        let (clean, reasoning) =
            strip_and_capture_thinking_tags_primed("first the plan</think>the answer", true, false);
        assert_eq!(clean, "the answer");
        assert_eq!(reasoning.as_deref(), Some("first the plan"));
    }

    #[test]
    fn primed_strip_whitespace_only_thought_is_not_reasoning() {
        let (clean, reasoning) = strip_and_capture_thinking_tags_primed("   \n ", true, true);
        assert_eq!(clean, "");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn prompt_ending_in_an_open_think_is_reasoning_primed() {
        // xybrid's own prime, and the channel an LFM3 continuation with
        // thinking on reopens.
        assert!(is_reasoning_primed(
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        ));
        assert!(is_reasoning_primed(
            "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>"
        ));
    }

    #[test]
    fn prompt_without_an_open_think_is_not_reasoning_primed() {
        for prompt in [
            // LFM3 with thinking off pre-fills a closed channel.
            "<|im_start|>assistant\n<think></think>",
            // A ChatML continuation opens a fresh turn.
            "<|im_start|>tool\n<|tool_response_start|>{}<|tool_response_end|><|im_end|>\n<|im_start|>assistant\n",
            // A replayed thought earlier in the prompt does not count.
            "<|im_start|>assistant\n<think>plan</think>answer<|im_end|>\n<|im_start|>user\n",
            // A codec-TTS raw prompt.
            "user: Convert the text to speech:<|TEXT_PROMPT_START|>hi<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>",
            "",
        ] {
            assert!(!is_reasoning_primed(prompt), "{prompt:?}");
        }
    }

    #[test]
    fn unprimed_strip_leaves_markerless_text_as_answer() {
        let (clean, reasoning) =
            strip_and_capture_thinking_tags_primed("just an answer", false, false);
        assert_eq!(clean, "just an answer");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn strip_thinking_tags_removes_closed_blocks() {
        assert_eq!(
            strip_thinking_tags("before<think>hidden</think>after"),
            "beforeafter"
        );
    }

    #[test]
    fn strip_thinking_tags_removes_multiple_blocks() {
        assert_eq!(
            strip_thinking_tags("a<think>x</think>b<think>y</think>c"),
            "abc"
        );
    }

    #[test]
    fn strip_thinking_tags_truncates_unclosed_block() {
        assert_eq!(
            strip_thinking_tags("visible<think>still reasoning"),
            "visible"
        );
    }

    #[test]
    fn strip_thinking_tags_passthrough_no_tags() {
        assert_eq!(strip_thinking_tags("nothing to see"), "nothing to see");
    }

    #[test]
    fn capture_thinking_tags_returns_inner_reasoning() {
        let (clean, reasoning) =
            strip_and_capture_thinking_tags("before<think>hidden</think>after");
        assert_eq!(clean, "beforeafter");
        assert_eq!(reasoning.as_deref(), Some("hidden"));
    }

    #[test]
    fn capture_thinking_tags_joins_multiple_blocks() {
        let (clean, reasoning) =
            strip_and_capture_thinking_tags("a<think>x</think>b<think>y</think>c");
        assert_eq!(clean, "abc");
        assert_eq!(reasoning.as_deref(), Some("x\ny"));
    }

    #[test]
    fn capture_thinking_tags_unclosed_block_captured() {
        let (clean, reasoning) = strip_and_capture_thinking_tags("visible<think>still reasoning");
        assert_eq!(clean, "visible");
        assert_eq!(reasoning.as_deref(), Some("still reasoning"));
    }

    #[test]
    fn capture_thinking_tags_none_when_absent() {
        let (clean, reasoning) = strip_and_capture_thinking_tags("plain answer");
        assert_eq!(clean, "plain answer");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn capture_anthropic_thinking_block() {
        // Anthropic-style `<thinking>...</thinking>`.
        let (clean, reasoning) =
            strip_and_capture_thinking_tags("<thinking>step by step</thinking>The answer is 4.");
        assert_eq!(clean, "The answer is 4.");
        assert_eq!(reasoning.as_deref(), Some("step by step"));
    }

    #[test]
    fn capture_gemma4_thought_channel() {
        // gemma-4's flipped-pipe thought channel `<|channel>thought ... <channel|>`.
        let (clean, reasoning) = strip_and_capture_thinking_tags(
            "<|channel>thought\nchecking divisors<channel|>97 is prime.",
        );
        assert_eq!(clean, "97 is prime.");
        assert_eq!(reasoning.as_deref(), Some("checking divisors"));
    }

    #[test]
    fn capture_think_and_thinking_are_distinct_markers() {
        // `<think>` is not a false prefix of `<thinking>` (the `>` disambiguates),
        // so a `<thinking>` block is captured whole, not mis-split.
        let (clean, reasoning) =
            strip_and_capture_thinking_tags("<thinking>outer <think> mention</thinking>done");
        assert_eq!(clean, "done");
        assert_eq!(reasoning.as_deref(), Some("outer <think> mention"));
    }

    #[test]
    fn capture_thinking_tags_dangling_close_is_primed_reasoning() {
        // Shape a thinking model produces when `<think>` was primed into the
        // prompt: reasoning, a closing tag, then the answer — no opening tag.
        let (clean, reasoning) =
            strip_and_capture_thinking_tags("let me work it out</think>The answer is 8.");
        assert_eq!(clean, "The answer is 8.");
        assert_eq!(reasoning.as_deref(), Some("let me work it out"));
    }

    #[test]
    fn capture_thinking_tags_multiple_dangling_closes_all_stripped() {
        // A model that repeats `</think>` must not leak the later ones into the
        // answer — every leading dangling close is peeled as reasoning.
        let (clean, reasoning) =
            strip_and_capture_thinking_tags("first</think>second</think>answer");
        assert_eq!(clean, "answer");
        assert_eq!(reasoning.as_deref(), Some("first\nsecond"));
    }

    #[test]
    fn capture_thinking_tags_empty_block_is_none() {
        // A thinking model can open and immediately close the block with only
        // whitespace inside — that's not reasoning worth surfacing.
        let (clean, reasoning) = strip_and_capture_thinking_tags("<think>\n\n</think>\n\nHello!");
        assert_eq!(clean.trim(), "Hello!");
        assert_eq!(reasoning, None);

        // Same, primed/dangling shape.
        let (clean, reasoning) = strip_and_capture_thinking_tags("\n</think>\nHello!");
        assert_eq!(clean.trim(), "Hello!");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn capture_thinking_tags_no_close_keeps_answer() {
        // Primed but the model never closed `</think>` (e.g. answered directly).
        // The answer must NOT be eaten as reasoning.
        let (clean, reasoning) =
            strip_and_capture_thinking_tags_primed("The answer is 8.", true, false);
        assert_eq!(clean, "The answer is 8.");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn streaming_filter_reasoning_primed_suppresses_and_captures() {
        let mut f = StreamingTextFilter::new_reasoning_primed(vec![]);
        // Reasoning streams in with no opening tag — all suppressed.
        assert_eq!(f.push("checking divisors "), None);
        assert_eq!(f.push("none divide it"), None);
        // Closing tag ends the reasoning; the answer streams normally after.
        assert_eq!(f.push("</think>"), None);
        assert_eq!(f.push("97 is prime."), Some("97 is prime.".to_string()));
        assert_eq!(f.reasoning(), Some("checking divisors none divide it"));
    }

    #[test]
    fn streaming_filter_suppresses_complete_lfm2_tool_block_mid_stream() {
        let (start, end, _) = TOOL_BLOCK_MARKERS[0];
        let mut f = StreamingTextFilter::new(vec![]).with_tool_call_suppression();

        assert_eq!(f.push("I will check "), Some("I will check ".to_string()));
        assert_eq!(f.push(start), None);
        assert_eq!(f.push("[get_weather(city=\"Paris\")]"), None);
        assert_eq!(f.push(end), None);
        assert_eq!(f.push(" now."), Some(" now.".to_string()));

        assert_eq!(f.cumulative_emitted(), "I will check  now.");
        assert!(f.saw_tool_call_block());
    }

    #[test]
    fn streaming_filter_suppresses_complete_valid_tool_call_block() {
        let (start, end, _) = TOOL_BLOCK_MARKERS[0];
        let mut f = StreamingTextFilter::new(vec![]).with_tool_call_suppression();

        assert_eq!(
            f.push(&format!(
                "before {start}[get_weather(city=\"Paris\")]{end} after"
            )),
            Some("before  after".to_string())
        );

        assert_eq!(f.cumulative_emitted(), "before  after");
        assert!(f.saw_tool_call_block());
    }

    #[test]
    fn streaming_filter_emits_complete_invalid_tool_call_block() {
        let (start, end, _) = TOOL_BLOCK_MARKERS[0];
        let mut f = StreamingTextFilter::new(vec![]).with_tool_call_suppression();
        let block = format!("{start}not a real call{end}");

        assert_eq!(
            f.push(&format!("before {block} after")),
            Some(format!("before {block} after"))
        );

        assert_eq!(f.cumulative_emitted(), format!("before {block} after"));
        assert!(!f.saw_tool_call_block());
    }

    #[test]
    fn streaming_filter_suppresses_complete_gemma_tool_call_block() {
        let (start, end, _) = TOOL_BLOCK_MARKERS[1];
        let mut f = StreamingTextFilter::new(vec![]).with_tool_call_suppression();

        assert_eq!(f.push("Checking "), Some("Checking ".to_string()));
        assert_eq!(
            f.push(&format!(
                "{start}call:get_weather{{city:<|\"|>Paris<|\"|>}}{end}"
            )),
            None
        );
        assert_eq!(f.push("done"), Some("done".to_string()));

        assert_eq!(f.cumulative_emitted(), "Checking done");
        assert!(f.saw_tool_call_block());
    }

    /// LFM3's call block, streamed in the token pieces the model produces:
    /// nothing of it reaches the callback, prose on either side does.
    #[test]
    fn streaming_filter_suppresses_lfm3_tool_block_split_across_tokens() {
        let mut f = StreamingTextFilter::new(vec![]).with_tool_call_suppression();

        assert_eq!(f.push("Checking."), Some("Checking.".to_string()));
        for piece in [
            "<tool_call>",
            "\n",
            "<function=",
            "get",
            "_weather",
            ">\n",
            "<parameter=",
            "city",
            ">\nParis\n",
            "</parameter>",
            "\n",
            "</function>",
            "\n",
            "</tool_call>",
        ] {
            assert_eq!(f.push(piece), None, "{piece:?} must not reach the callback");
        }
        assert_eq!(f.push(" Done."), Some(" Done.".to_string()));

        assert_eq!(f.cumulative_emitted(), "Checking. Done.");
        assert!(f.saw_tool_call_block());
    }

    /// A reasoning-primed LFM3 stream closes its thinking before the call:
    /// neither the reasoning nor the call block is emitted.
    #[test]
    fn streaming_filter_hides_lfm3_reasoning_and_tool_block() {
        let mut f = StreamingTextFilter::new_reasoning_primed(vec![]).with_tool_call_suppression();

        assert_eq!(f.push("The user wants the weather."), None);
        assert_eq!(f.push("</think>"), None);
        assert_eq!(
            f.push("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
            None
        );

        assert_eq!(f.cumulative_emitted(), "");
        assert_eq!(f.reasoning(), Some("The user wants the weather."));
        assert!(f.saw_tool_call_block());
    }

    #[test]
    fn streaming_filter_holds_back_split_tool_start_marker() {
        let (start, end, _) = TOOL_BLOCK_MARKERS[0];
        let mut f = StreamingTextFilter::new(vec![]).with_tool_call_suppression();

        assert_eq!(f.push("before "), Some("before ".to_string()));
        assert_eq!(f.push(&start[..5]), None);
        assert_eq!(f.cumulative_emitted(), "before ");
        assert_eq!(
            f.push(&format!("{}[f()]{}", &start[5..], end)),
            None,
            "held marker head and complete block must stay suppressed"
        );
        assert_eq!(f.push(" after"), Some(" after".to_string()));

        assert_eq!(f.cumulative_emitted(), "before  after");
    }

    #[test]
    fn streaming_filter_drops_partial_tool_start_marker_at_stream_end() {
        let (start, _, _) = TOOL_BLOCK_MARKERS[0];
        let mut f = StreamingTextFilter::new(vec![]).with_tool_call_suppression();

        assert_eq!(f.push("before "), Some("before ".to_string()));
        assert_eq!(f.push(&start[..8]), None);

        assert_eq!(f.cumulative_emitted(), "before ");
        assert!(!f.saw_tool_call_block());
    }

    #[test]
    fn streaming_filter_drops_unclosed_gemma_tool_response_opener() {
        let (start, _, _) = TOOL_BLOCK_MARKERS[2];
        let mut f = StreamingTextFilter::new(vec![]).with_tool_call_suppression();

        assert_eq!(f.push("answer"), Some("answer".to_string()));
        assert_eq!(f.push(start), None);
        assert_eq!(f.push("response:get_weather{"), None);

        assert_eq!(f.cumulative_emitted(), "answer");
        assert!(!f.saw_tool_call_block());
    }

    #[test]
    fn streaming_filter_without_tool_suppression_emits_call_block_verbatim() {
        let (start, end, _) = TOOL_BLOCK_MARKERS[0];
        let mut f = StreamingTextFilter::new(vec![]);

        let block = format!("{start}[get_weather(city=\"Paris\")]{end}");
        assert_eq!(f.push("before "), Some("before ".to_string()));
        assert_eq!(f.push(&block), Some(block.clone()));
        assert_eq!(f.push(" after"), Some(" after".to_string()));

        assert_eq!(f.cumulative_emitted(), format!("before {block} after"));
        assert!(!f.saw_tool_call_block());
    }

    #[test]
    fn streaming_filter_saw_tool_call_block_excludes_response_blocks() {
        let (call_start, call_end, _) = TOOL_BLOCK_MARKERS[1];
        let (response_start, response_end, _) = TOOL_BLOCK_MARKERS[2];

        let mut response_only = StreamingTextFilter::new(vec![]).with_tool_call_suppression();
        assert_eq!(
            response_only.push(&format!(
                "{response_start}response:get_weather{{ok:true}}{response_end}"
            )),
            None
        );
        assert!(!response_only.saw_tool_call_block());

        let mut with_call = StreamingTextFilter::new(vec![]).with_tool_call_suppression();
        assert_eq!(
            with_call.push(&format!("{call_start}call:get_weather{{}}{call_end}")),
            None
        );
        assert!(with_call.saw_tool_call_block());
    }

    #[test]
    fn streaming_filter_reasoning_primed_and_tool_suppression_compose() {
        let (start, end, _) = TOOL_BLOCK_MARKERS[0];
        let mut f = StreamingTextFilter::new_reasoning_primed(vec![]).with_tool_call_suppression();

        assert_eq!(f.push("thinking"), None);
        assert_eq!(f.push("</think>"), None);
        assert_eq!(f.push(" visible "), Some(" visible ".to_string()));
        assert_eq!(f.push(&format!("{start}[f()]{end}")), None);
        assert_eq!(f.push("done"), Some("done".to_string()));

        assert_eq!(f.reasoning(), Some("thinking"));
        assert_eq!(f.cumulative_emitted(), " visible done");
        assert!(f.saw_tool_call_block());
    }

    #[test]
    fn merge_stop_patterns_deduplicates() {
        let user = ["<|im_end|>".to_string(), "CUSTOM".to_string()];
        let got = merge_stop_patterns(&user, CHAT_STOP_PATTERNS);
        assert_eq!(got[0], "<|im_end|>");
        assert_eq!(got[1], "CUSTOM");
        assert!(got.contains(&"<end_of_turn>".to_string()));
        assert_eq!(got.iter().filter(|s| *s == "<|im_end|>").count(), 1);
    }

    #[test]
    fn truncate_at_first_stop_picks_earliest() {
        let mut text = String::from("hello <end_of_turn> world <|im_end|>");
        let patterns = ["<|im_end|>", "<end_of_turn>"];
        assert!(truncate_at_first_stop(&mut text, &patterns));
        assert_eq!(text, "hello ");
    }

    #[test]
    fn truncate_at_first_stop_no_match() {
        let mut text = String::from("no stops here");
        assert!(!truncate_at_first_stop(&mut text, &["<|im_end|>"]));
        assert_eq!(text, "no stops here");
    }

    /// `""` occurs at offset 0 of every text. Counting it erased the whole
    /// answer and reported a stop.
    #[test]
    fn truncate_at_first_stop_ignores_an_empty_pattern() {
        let mut text = String::from("The rest");
        assert!(!truncate_at_first_stop(&mut text, &[""]));
        assert_eq!(text, "The rest");

        // A real stop in the same list still applies.
        assert!(truncate_at_first_stop(&mut text, &["", " rest"]));
        assert_eq!(text, "The");
    }

    #[test]
    fn trim_partial_stop_suffix_removes_partial_prefix() {
        let mut text = String::from("response tail <|im_");
        let patterns = ["<|im_end|>"];
        assert!(trim_partial_stop_suffix(&mut text, &patterns));
        assert_eq!(text, "response tail ");
    }

    #[test]
    fn trim_partial_stop_suffix_ignores_clean_end() {
        let mut text = String::from("clean response");
        assert!(!trim_partial_stop_suffix(&mut text, &["<|im_end|>"]));
        assert_eq!(text, "clean response");
    }

    #[test]
    fn streaming_filter_emits_safe_chunks() {
        let mut f = StreamingTextFilter::new(vec!["<|im_end|>".to_string()]);
        assert_eq!(f.push("Hello "), Some("Hello ".to_string()));
        assert_eq!(f.push("world"), Some("world".to_string()));
        assert_eq!(f.cumulative_emitted(), "Hello world");
        assert!(!f.is_stopped());
    }

    #[test]
    fn streaming_filter_holds_back_partial_stop_prefix() {
        let mut f = StreamingTextFilter::new(vec!["<|im_end|>".to_string()]);
        assert_eq!(f.push("Hello "), Some("Hello ".to_string()));
        // `<|im_` is a prefix of the stop pattern — must be held back.
        assert_eq!(f.push("<|im_"), None);
        // Non-matching continuation releases the held bytes. Cumulative is
        // now "Hello <|im_portant!", nothing matches as a suffix prefix, so
        // the whole held segment + new bytes are safe to emit.
        assert_eq!(f.push("portant!"), Some("<|im_portant!".to_string()));
    }

    #[test]
    fn streaming_filter_stops_on_complete_pattern() {
        let mut f = StreamingTextFilter::new(vec!["<|im_end|>".to_string()]);
        f.push("hello ");
        f.push("<|im_end|>");
        assert!(f.is_stopped());
        // Further pushes are no-ops.
        assert_eq!(f.push(" ignored"), None);
    }

    /// Every text contains `""`, so counting it stopped the stream before
    /// its first chunk and nothing was ever emitted.
    #[test]
    fn streaming_filter_ignores_an_empty_stop_pattern() {
        let mut f = StreamingTextFilter::new(vec![String::new(), "<|im_end|>".to_string()]);
        assert_eq!(f.push("The"), Some("The".to_string()));
        assert_eq!(f.push(" rest"), Some(" rest".to_string()));
        assert!(!f.is_stopped());

        f.push("<|im_end|>");
        assert!(f.is_stopped(), "the real stop in the list still applies");

        let mut primed = StreamingTextFilter::new_reasoning_primed(vec![String::new()]);
        primed.push("thought</think>");
        assert_eq!(primed.push("answer"), Some("answer".to_string()));
    }

    #[test]
    fn streaming_filter_suppresses_think_block() {
        let mut f = StreamingTextFilter::new(vec![]);
        assert_eq!(f.push("visible "), Some("visible ".to_string()));
        assert_eq!(f.push("<think>"), None);
        assert_eq!(f.push("reasoning"), None);
        assert_eq!(f.push("</think>"), None);
        // After closing </think>, emission resumes on next chunk.
        assert_eq!(f.push("answer"), Some("answer".to_string()));
    }

    #[test]
    fn streaming_filter_captures_closed_think_reasoning() {
        let mut f = StreamingTextFilter::new(vec![]);
        assert_eq!(f.reasoning(), None);
        assert_eq!(f.push("<think>"), None);
        assert_eq!(f.push("step one "), None);
        assert_eq!(f.push("step two</think>"), None);
        assert_eq!(f.push("answer"), Some("answer".to_string()));
        assert_eq!(f.reasoning(), Some("step one step two"));
    }

    #[test]
    fn streaming_filter_suppresses_and_captures_anthropic_thinking() {
        // The filter recognizes `<thinking>` too, waiting for its own close.
        let mut f = StreamingTextFilter::new(vec![]);
        assert_eq!(f.push("visible "), Some("visible ".to_string()));
        assert_eq!(f.push("<thinking>"), None);
        assert_eq!(f.push("reasoning</thinking>"), None);
        assert_eq!(f.push("answer"), Some("answer".to_string()));
        assert_eq!(f.reasoning(), Some("reasoning"));
    }

    #[test]
    fn streaming_filter_holds_back_split_reasoning_marker() {
        // A `<thinking>` opener split across chunks must not leak its prefix.
        let mut f = StreamingTextFilter::new(vec![]);
        assert_eq!(f.push("answer<thin"), Some("answer".to_string()));
        // The held-back `<thin` completes into `<thinking>` → suppressed.
        assert_eq!(f.push("king>hidden</thinking>"), None);
        assert_eq!(f.push("done"), Some("done".to_string()));
        assert_eq!(f.reasoning(), Some("hidden"));
    }

    #[test]
    fn streaming_filter_unclosed_think_not_captured() {
        // An unclosed block is suppressed from emission but NOT surfaced as
        // reasoning — the backend's final-text pass recovers it instead.
        let mut f = StreamingTextFilter::new(vec![]);
        assert_eq!(f.push("<think>"), None);
        assert_eq!(f.push("still going"), None);
        assert_eq!(f.reasoning(), None);
    }

    /// An unclosed `<think>` must never leak its body upward. The final
    /// cleanup's `strip_thinking_tags` would also handle this, but the
    /// streaming contract promises the user never sees reasoning text —
    /// so the filter must withhold it even if `</think>` never arrives.
    #[test]
    fn streaming_filter_unclosed_think_stays_suppressed() {
        let mut f = StreamingTextFilter::new(vec![]);
        assert_eq!(f.push("<think>"), None);
        assert_eq!(f.push("still reasoning"), None);
        assert_eq!(f.push(" forever"), None);
        assert!(!f.is_stopped());
        assert_eq!(f.cumulative_emitted(), "");
    }

    /// `<think>` can arrive mid-chunk with user-visible text preceding
    /// it in the same chunk (e.g. a model that emits
    /// `scratch<think>hidden</think>final`). The `push` return stream
    /// correctly withholds `scratch` — it was never safe to emit
    /// before the block opened, so the consumer never sees it as a
    /// delta.
    ///
    /// Known quirk: `cumulative_emitted()` still reports `"scratchfinal"`
    /// here, because it returns `cumulative_text[..last_emitted_len]`
    /// and `strip_thinking_tags` only removes the `<think>...</think>`
    /// span — the pre-block bytes remain in the buffer. The emitted
    /// **delta** stream is what consumers actually see, and that is
    /// correct. If a future change tightens `cumulative_emitted()` to
    /// reflect only actually-emitted bytes, this assertion should
    /// change to `"final"`.
    #[test]
    fn streaming_filter_think_block_swallows_preceding_unemitted_text() {
        let mut f = StreamingTextFilter::new(vec![]);
        assert_eq!(f.push("scratch<think>"), None);
        assert_eq!(f.push("hidden</think>"), None);
        // The consumer-visible delta stream is just `final` — `scratch`
        // was never emitted.
        assert_eq!(f.push("final"), Some("final".to_string()));
        // Documents the cumulative_emitted() leakage described above.
        assert_eq!(f.cumulative_emitted(), "scratchfinal");
    }

    /// Regression guard: stop-pattern prefix math computes byte offsets
    /// like `text.len() - pattern.len()`. With multi-byte UTF-8 content
    /// in the cumulative buffer, a naive implementation could slice on
    /// a non-char boundary and panic. Reaching the assertion without a
    /// panic IS the assertion.
    #[test]
    fn streaming_filter_utf8_text_does_not_panic_on_ascii_stop_patterns() {
        let mut f = StreamingTextFilter::new(vec!["<|im_end|>".to_string()]);
        let _ = f.push("héllo ");
        let _ = f.push("wörld");
        let _ = f.push("<|im_");
        let _ = f.push("end|>");
        assert!(f.is_stopped());
    }

    /// Gemma 4 E2B VLM regression: the Q8 GGUF at
    /// `ggml-org/gemma-4-E2B-it-GGUF` decodes its chat end-of-turn
    /// special token to the literal string `<turn|>` (rather than the
    /// expected `<end_of_turn>` from Gemma 3 / 3n). Without this
    /// pattern, the marker leaks as the trailing tail of the VLM
    /// caption. Confirmed against `vendor/llama.cpp/src/llama-vocab.cpp`
    /// which lists `<turn|>` as a `gemma4` EOG marker.
    #[test]
    fn truncate_at_first_stop_handles_gemma4_turn_marker() {
        let mut text =
            String::from("A simple graphic is composed of three colored squares.<turn|>");
        assert!(truncate_at_first_stop(&mut text, CHAT_STOP_PATTERNS));
        assert_eq!(
            text,
            "A simple graphic is composed of three colored squares."
        );
    }

    /// Gemma 4 streaming regression: the `<turn|>` marker should hold
    /// back the `<` byte until the next chunk arrives, then suppress
    /// the rest of the stream once the complete marker is seen.
    #[test]
    fn streaming_filter_stops_on_gemma4_turn_marker() {
        let mut f = StreamingTextFilter::new(vec!["<turn|>".to_string()]);
        assert_eq!(f.push("Three colors. "), Some("Three colors. ".to_string()));
        // `<turn` is a prefix of `<turn|>` — must be held back.
        assert_eq!(f.push("<turn"), None);
        // Completing the marker stops the stream; nothing further emits.
        assert_eq!(f.push("|>"), None);
        assert!(f.is_stopped());
        assert_eq!(f.cumulative_emitted(), "Three colors. ");
    }

    /// Vision-model catalog regression: each Studio VLM family has a
    /// chat-template stop marker that must be removed in final text and
    /// withheld from streaming callbacks.
    #[test]
    fn vision_chat_stop_patterns_do_not_leak() {
        let cases = [
            ("lfm2-vl-450m", "<|im_end|>"),
            ("qwen3-vl-2b-instruct", "<|im_end|>"),
            ("qwen3-vl-4b-instruct", "<|im_end|>"),
            ("qwen3.5-2b", "<|im_end|>"),
            ("qwen2.5-vl-3b-instruct", "<|im_end|>"),
            ("internvl3-2b-instruct", "<|im_end|>"),
            ("gemma-4-e2b", "<turn|>"),
            ("gemma-4-e4b", "<turn|>"),
            ("smolvlm-500m-instruct", "<end_of_utterance>"),
            ("smolvlm-instruct", "<end_of_utterance>"),
        ];

        for (model_id, marker) in cases {
            assert!(
                CHAT_STOP_PATTERNS.contains(&marker),
                "{model_id} stop marker {marker:?} must be registered"
            );

            let mut text = format!("A concise image description.{marker} trailing");
            assert!(
                truncate_at_first_stop(&mut text, CHAT_STOP_PATTERNS),
                "{model_id} final output should stop at {marker:?}"
            );
            assert_eq!(text, "A concise image description.");

            let split_at = marker.len() / 2;
            let mut f = StreamingTextFilter::new(
                CHAT_STOP_PATTERNS.iter().map(|s| s.to_string()).collect(),
            );
            assert_eq!(
                f.push("A concise image description."),
                Some("A concise image description.".to_string())
            );
            assert_eq!(&f.push(&marker[..split_at]), &None);
            assert_eq!(&f.push(&marker[split_at..]), &None);
            assert!(
                f.is_stopped(),
                "{model_id} streaming output should stop at {marker:?}"
            );
            assert_eq!(f.cumulative_emitted(), "A concise image description.");
        }
    }

    /// Inkling's raw turn, the way the native loop streams it: one special
    /// token per chunk, text in pieces.
    const INKLING_TURN: &[&str] = &[
        "<|message_model|>",
        "<|content_thinking|>",
        "84 * 3 = 252",
        ", halved: 126.",
        "<|end_message|>",
        "<|message_model|>",
        "<|content_text|>",
        "The answer",
        " is 126.",
        "<|end_message|>",
    ];

    /// LFM3's call notation, which an Inkling answer may quote.
    const LFM3_EXAMPLE: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    /// The stop patterns a chat turn in `framing` runs with.
    fn chat_stops(framing: OutputFraming) -> Vec<String> {
        CHAT_STOP_PATTERNS
            .iter()
            .chain(framing.stop_patterns())
            .map(|s| s.to_string())
            .collect()
    }

    /// Stream `chunks` through `framing`'s decoder and filter, the way the
    /// llama.cpp backend's streaming paths do, returning the emitted deltas
    /// and the filter.
    fn stream(
        framing: OutputFraming,
        suppress_tools: bool,
        chunks: &[&str],
    ) -> (Vec<String>, StreamingTextFilter) {
        let mut filter = framing.streaming_filter(chat_stops(framing), false, suppress_tools);
        let mut decoder = framing.decoder();
        let mut emitted: Vec<String> = chunks
            .iter()
            .filter_map(|chunk| filter.push(&decoder.push(chunk)))
            .collect();
        emitted.extend(filter.push(&decoder.finish()));
        (emitted, filter)
    }

    /// Finish a raw Inkling turn the way the backend does: cut it at its
    /// stops, then split it by its blocks.
    fn finish_inkling(raw: &str, truncated_by_budget: bool) -> (String, Option<String>) {
        let mut text = raw.to_string();
        truncate_at_first_stop(&mut text, &chat_stops(OutputFraming::Inkling));
        OutputFraming::Inkling.split_reasoning(&text, false, truncated_by_budget)
    }

    #[test]
    fn output_framing_is_read_off_the_prompt() {
        let inkling = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>";
        let chatml = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";

        assert_eq!(OutputFraming::of_prompt(inkling), OutputFraming::Inkling);
        assert!(!OutputFraming::Inkling.allows_think_priming());
        assert_eq!(
            OutputFraming::Inkling.stop_patterns(),
            INKLING_TURN_END_MARKERS
        );
        assert_eq!(OutputFraming::of_prompt(chatml), OutputFraming::Plain);
        assert!(OutputFraming::Plain.allows_think_priming());
        assert!(OutputFraming::Plain.stop_patterns().is_empty());
        assert_eq!(
            OutputFraming::Plain.split_reasoning("<think>a</think>b", false, false),
            ("b".to_string(), Some("a".to_string()))
        );
        assert_eq!(
            OutputFraming::Plain
                .output_tool_calls(LFM3_EXAMPLE, &[])
                .map(|calls| calls.len()),
            None,
            "plain output is parsed by whoever reads its text"
        );
    }

    /// Other families' output may quote Inkling's role markers as plain
    /// text; only an Inkling turn stops on them.
    #[test]
    fn inkling_role_markers_stop_only_inkling_turns() {
        for marker in INKLING_TURN_END_MARKERS {
            assert!(!CHAT_STOP_PATTERNS.contains(marker), "{marker}");
        }
        let quoted = "The placeholder is <|message_user|> in this example.";

        let (emitted, filter) = stream(OutputFraming::Plain, false, &[quoted]);
        assert_eq!(emitted.concat(), quoted);
        assert!(!filter.is_stopped());
        let mut text = quoted.to_string();
        assert!(!truncate_at_first_stop(&mut text, CHAT_STOP_PATTERNS));
    }

    #[test]
    fn inkling_stream_hides_the_thought_and_emits_only_the_answer() {
        let (emitted, filter) = stream(OutputFraming::Inkling, false, INKLING_TURN);

        assert_eq!(emitted.concat(), "The answer is 126.");
        assert!(!filter.is_stopped());
        assert_eq!(
            finish_inkling(&INKLING_TURN.concat(), false),
            (
                "The answer is 126.".to_string(),
                Some("84 * 3 = 252, halved: 126.".to_string())
            )
        );
    }

    /// `<think>` and `</think>` inside Inkling blocks are payload: a literal
    /// close in the thought does not leak it, and a literal opener in the
    /// answer does not hide it.
    #[test]
    fn inkling_think_tags_are_payload_in_the_stream_and_the_final_text() {
        let chunks = [
            "<|message_model|>",
            "<|content_thinking|>",
            "Mention the ",
            "</think>",
            " tag, then more.",
            "<|end_message|>",
            "<|message_model|>",
            "<|content_text|>",
            "Print ",
            "<think>",
            " exactly.",
            "<|end_message|>",
        ];

        let (emitted, _) = stream(OutputFraming::Inkling, false, &chunks);
        assert_eq!(emitted.concat(), "Print <think> exactly.");
        assert_eq!(
            finish_inkling(&chunks.concat(), false),
            (
                "Print <think> exactly.".to_string(),
                Some("Mention the </think> tag, then more.".to_string())
            )
        );
    }

    /// An Inkling answer that quotes another family's call notation is
    /// answer text: not suppressed from the stream, and not a call.
    #[test]
    fn inkling_quoted_call_notation_is_answer_text_not_a_call() {
        let chunks = [
            "<|message_model|>",
            "<|content_text|>",
            "LFM3 writes ",
            LFM3_EXAMPLE,
            "<|end_message|>",
        ];

        let (emitted, filter) = stream(OutputFraming::Inkling, true, &chunks);
        assert_eq!(emitted.concat(), format!("LFM3 writes {LFM3_EXAMPLE}"));
        assert!(!filter.saw_tool_call_block());
        let (answer, _) = finish_inkling(&chunks.concat(), false);
        assert!(OutputFraming::Inkling
            .parse_tool_calls(&answer, &[])
            .is_empty());
        assert_eq!(
            OutputFraming::Inkling
                .output_tool_calls(&answer, &[])
                .map(|calls| calls.len()),
            Some(0)
        );
        // The cross-family scan reads it as a call; Inkling's must not.
        assert_eq!(OutputFraming::Plain.parse_tool_calls(&answer, &[]).len(), 1);
    }

    #[test]
    fn inkling_stream_suppresses_its_own_tool_call_block() {
        let chunks = [
            "<|message_model|>",
            "<|content_text|>",
            "Let me check.",
            "<|end_message|>",
            "<|message_model|>",
            "get",
            "_weather",
            "<|content_invoke_tool_json|>",
            "{\"name\":\"get_weather\",",
            "\"args\":{\"city\":\"Paris\"}}",
            "<|end_message|>",
        ];

        let (emitted, filter) = stream(OutputFraming::Inkling, true, &chunks);

        assert_eq!(emitted.concat(), "Let me check.");
        assert!(filter.saw_tool_call_block());
        let (answer, _) = finish_inkling(&chunks.concat(), false);
        let calls = OutputFraming::Inkling.parse_tool_calls(&answer, &[]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn inkling_turn_end_markers_stop_the_stream() {
        let chunks = [
            "<|message_model|>",
            "<|content_text|>",
            "Hi.",
            "<|end_message|>",
            "<|message_user|>",
            "<|content_text|>",
            "and another question",
        ];

        let (emitted, filter) = stream(OutputFraming::Inkling, false, &chunks);

        assert_eq!(emitted.concat(), "Hi.");
        assert!(filter.is_stopped());
        assert_eq!(
            finish_inkling(&chunks.concat(), false),
            ("Hi.".to_string(), None)
        );
    }

    /// Split raw Inkling output the way the native loop streams it: every
    /// special token alone, text in three-char pieces.
    fn inkling_chunks(raw: &str) -> Vec<String> {
        const SPECIALS: &[&str] = &[
            "<|message_model|>",
            "<|content_thinking|>",
            "<|content_text|>",
            "<|content_invoke_tool_json|>",
            "<|end_message|>",
        ];
        let mut chunks = Vec::new();
        let mut rest = raw;
        while !rest.is_empty() {
            let next = SPECIALS
                .iter()
                .filter_map(|special| rest.find(special).map(|at| (at, *special)))
                .min_by_key(|(at, _)| *at);
            let text_end = next.map_or(rest.len(), |(at, _)| at);
            let text: Vec<char> = rest[..text_end].chars().collect();
            chunks.extend(text.chunks(3).map(|piece| piece.iter().collect::<String>()));
            match next {
                Some((at, special)) => {
                    chunks.push(special.to_string());
                    rest = &rest[at + special.len()..];
                }
                None => rest = "",
            }
        }
        chunks
    }

    /// llama.cpp's own Inkling 2 parser cases (`tests/test-chat.cpp` at
    /// master, 2026-11-23): a finished turn must split into the same answer,
    /// reasoning and tool calls, and the stream must show the same answer.
    /// llama.cpp keeps a call's argument text as written; xybrid re-encodes
    /// it compactly, so arguments compare as JSON.
    #[test]
    fn inkling_output_splits_like_llama_cpp_parser() {
        use crate::runtime_adapter::tool_call::strip_tool_calls;

        const H: &str = "<|message_model|>";
        const E: &str = "<|end_message|>";
        let call = format!(
            "special_function<|content_invoke_tool_json|>{{\"name\":\"special_function\",\"args\":{{\"arg1\":1}}}}{E}"
        );
        struct Case {
            name: &'static str,
            raw: String,
            content: &'static str,
            reasoning: Option<&'static str>,
            calls: Vec<(&'static str, serde_json::Value)>,
        }
        let case = |name, raw, content, reasoning, calls| Case {
            name,
            raw,
            content,
            reasoning,
            calls,
        };
        let special = |args: serde_json::Value| ("special_function", args);
        let cases = vec![
            case(
                "content only",
                format!("{H}<|content_text|>Hello, world!\nWhat's up?{E}"),
                "Hello, world!\nWhat's up?",
                None,
                vec![],
            ),
            case(
                "thinking, then content",
                format!("{H}<|content_thinking|>I'm thinking{E}{H}<|content_text|>Hello, world!\nWhat's up?{E}"),
                "Hello, world!\nWhat's up?",
                Some("I'm thinking"),
                vec![],
            ),
            case(
                "cut off in thinking",
                format!("{H}<|content_thinking|>I was still thinking"),
                "",
                Some("I was still thinking"),
                vec![],
            ),
            case(
                "cut off in content",
                format!("{H}<|content_thinking|>I'm thinking{E}{H}<|content_text|>Hello, wor"),
                "Hello, wor",
                Some("I'm thinking"),
                vec![],
            ),
            case(
                "single tool call",
                format!("{H}{call}"),
                "",
                None,
                vec![special(serde_json::json!({"arg1": 1}))],
            ),
            case(
                "thinking, content, then a tool call",
                format!("{H}<|content_thinking|>I should call it{E}{H}<|content_text|>On it.{E}{H}{call}"),
                "On it.",
                Some("I should call it"),
                vec![special(serde_json::json!({"arg1": 1}))],
            ),
            case(
                "parallel tool calls",
                format!(
                    "{H}{call}{H}special_function_with_opt<|content_invoke_tool_json|>{{\"name\":\"special_function_with_opt\",\"args\":{{\"arg1\":1,\"arg2\":2}}}}{E}"
                ),
                "",
                None,
                vec![
                    special(serde_json::json!({"arg1": 1})),
                    ("special_function_with_opt", serde_json::json!({"arg1": 1, "arg2": 2})),
                ],
            ),
            case(
                "no name before the tool token",
                format!(
                    "{H}<|content_invoke_tool_json|>{{\"name\":\"special_function\",\"args\":{{\"arg1\":1}}}}{E}"
                ),
                "",
                None,
                vec![special(serde_json::json!({"arg1": 1}))],
            ),
            case(
                "unknown tool, spaces in the JSON",
                format!(
                    "{H}unknown_tool<|content_invoke_tool_json|>{{\"name\": \"unknown_tool\", \"args\": {{\"arg1\": 1}}}}{E}"
                ),
                "",
                None,
                vec![("unknown_tool", serde_json::json!({"arg1": 1}))],
            ),
            case(
                "the JSON names the tool, not the hint",
                format!(
                    "{H}special<|content_invoke_tool_json|>{{\"name\":\"special_function\",\"args\":{{\"arg1\":1}}}}{E}"
                ),
                "",
                None,
                vec![special(serde_json::json!({"arg1": 1}))],
            ),
            case(
                "response format",
                format!("{H}<|content_thinking|>I'm thinking{E}{H}<|content_text|>{{\"amount\": 123.45}}{E}"),
                "{\"amount\": 123.45}",
                Some("I'm thinking"),
                vec![],
            ),
        ];

        for Case {
            name,
            raw,
            content,
            reasoning,
            calls,
        } in cases
        {
            let (answer, captured) = finish_inkling(&raw, false);
            assert_eq!(strip_tool_calls(&answer), content, "{name}: answer");
            assert_eq!(captured.as_deref(), reasoning, "{name}: reasoning");
            let parsed: Vec<(String, serde_json::Value)> = OutputFraming::Inkling
                .parse_tool_calls(&answer, &[])
                .into_iter()
                .map(|call| {
                    let arguments = serde_json::from_str(&call.function.arguments)
                        .unwrap_or_else(|e| panic!("{name}: arguments: {e}"));
                    (call.function.name, arguments)
                })
                .collect();
            let expected: Vec<(String, serde_json::Value)> = calls
                .into_iter()
                .map(|(tool, arguments)| (tool.to_string(), arguments))
                .collect();
            assert_eq!(parsed, expected, "{name}: tool calls");

            let chunks = inkling_chunks(&raw);
            let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let (emitted, _) = stream(OutputFraming::Inkling, true, &chunks);
            assert_eq!(emitted.concat(), content, "{name}: streamed answer");
        }
    }

    /// A budget that runs out mid-thought leaves the reasoning and no answer.
    #[test]
    fn inkling_thought_cut_off_by_the_budget_is_reasoning() {
        let chunks = [
            "<|message_model|>",
            "<|content_thinking|>",
            "Let me work through",
        ];

        let (emitted, _) = stream(OutputFraming::Inkling, false, &chunks);

        assert!(emitted.concat().is_empty());
        assert_eq!(
            finish_inkling(&chunks.concat(), true),
            (String::new(), Some("Let me work through".to_string()))
        );
    }

    /// The chat paths clean their final text with every pattern, the broken
    /// variants included. `end_of_turn>` is one, so a partial match on its
    /// `e` cut the last letter off any answer ending in `e`, even one the
    /// model ended itself. The real markers' prefixes are still trimmed.
    #[test]
    fn trim_partial_stop_suffix_keeps_words_ending_in_broken_marker_prefixes() {
        let mut patterns: Vec<&str> = CHAT_STOP_PATTERNS.to_vec();
        patterns.extend_from_slice(CHAT_STOP_PATTERNS_BROKEN);
        for answer in ["true", "The", "the end", "Ready to send", "| a | b |"] {
            let mut text = answer.to_string();
            assert!(
                !trim_partial_stop_suffix(&mut text, &patterns),
                "{answer:?} was trimmed to {text:?}"
            );
            assert_eq!(text, answer);
        }

        let mut text = String::from("done <|im_");
        assert!(trim_partial_stop_suffix(&mut text, &patterns));
        assert_eq!(text, "done ");
    }

    /// Regression guard: `CHAT_STOP_PATTERNS_BROKEN` must NOT contain
    /// `turn|>`. The body starts with the common letter `t`, and
    /// `trim_partial_stop_suffix` would then trim any final answer
    /// ending in `t`, `tu`, `tur`, or `turn`. The full marker is in
    /// `CHAT_STOP_PATTERNS` (`<turn|>`) which is enough — the broken
    /// variant is too dangerous to ship.
    #[test]
    fn trim_partial_stop_suffix_does_not_chop_short_words_with_turn_marker() {
        for tail in ["yes it is t", "the next turn", "tu", "the answer is tur"] {
            let mut text = tail.to_string();
            let trimmed = trim_partial_stop_suffix(&mut text, CHAT_STOP_PATTERNS_BROKEN);
            assert!(
                !trimmed,
                "broken-variant trim must not fire on benign suffix {tail:?} \
                 — got trimmed text {text:?}"
            );
            assert_eq!(text, tail, "broken-variant trim must leave {tail:?} intact");
        }
    }

    /// Regression: a multi-char stop pattern completing across chunks used to
    /// (a) over-emit — only the SHORTEST matching tail prefix was held, so
    /// `"x``"` with stop `"```"` emitted `"x`"` — and then (b) panic: the
    /// completion truncated `cumulative_text` below `last_emitted_len` and
    /// `cumulative_emitted()` sliced out of bounds.
    #[test]
    fn streaming_filter_stop_completing_across_chunks_holds_longest_prefix() {
        let mut f = StreamingTextFilter::new(vec!["```".to_string()]);
        // Longest-prefix hold: both trailing backticks are withheld.
        assert_eq!(f.push("x``").as_deref(), Some("x"));
        // Completing the stop truncates cleanly at the held boundary.
        assert_eq!(f.push("`"), None);
        assert!(f.is_stopped());
        // Pre-fix this call panicked (byte index past the truncated text).
        assert_eq!(f.cumulative_emitted(), "x");
    }

    /// Tool suppression can splice a stop sequence together behind the emit
    /// boundary. The boundary must follow the resulting truncation.
    #[test]
    fn streaming_filter_clamps_emit_boundary_after_suppression_splices_stop() {
        let (start, end, _) = TOOL_BLOCK_MARKERS[2];
        let mut f = StreamingTextFilter::new(vec!["ab".to_string()]);
        f = f.with_tool_call_suppression();

        assert_eq!(f.push(&format!("a{start}")).as_deref(), Some("a"));
        assert_eq!(f.push(&format!("payload{end}b")), None);
        assert!(f.is_stopped());
        assert_eq!(f.cumulative_emitted(), "");
    }

    /// Across patterns the EARLIEST hold position must win — holding for the
    /// first pattern only can under-hold a longer match of another pattern.
    #[test]
    fn streaming_filter_multi_pattern_hold_takes_earliest() {
        let mut f = StreamingTextFilter::new(vec!["b!".to_string(), "abc".to_string()]);
        // Tail "ab": "b" is a 1-byte prefix of "b!" (hold at 1), but "ab" is
        // a 2-byte prefix of "abc" (hold at 0) — the earlier hold must win,
        // so nothing is emitted yet.
        assert_eq!(f.push("ab"), None);
        assert_eq!(f.push("c"), None); // completes "abc" at pos 0
        assert!(f.is_stopped());
        assert_eq!(f.cumulative_emitted(), "");
    }

    /// Multi-byte UTF-8 stop patterns must not panic: the old prefix loop
    /// sliced the pattern at byte offsets (`&pattern[..1]` on `"→x"` panics
    /// on a non-char boundary). Byte-wise comparison handles it.
    #[test]
    fn streaming_filter_multibyte_stop_pattern_does_not_panic() {
        let mut f = StreamingTextFilter::new(vec!["→x".to_string()]);
        assert_eq!(f.push("hello ").as_deref(), Some("hello "));
        // Tail ends with the full first char of the pattern: held.
        assert_eq!(f.push("→").as_deref(), None);
        assert_eq!(f.push("x"), None); // completes the stop
        assert!(f.is_stopped());
        assert_eq!(f.cumulative_emitted(), "hello ");
    }

    /// A chunk can carry text ahead of the stop it completes: the token "The"
    /// completes the stop "he". That text is part of the answer, which the
    /// final text keeps, so the stream emits it before it stops.
    #[test]
    fn streaming_filter_emits_the_text_before_a_stop_inside_a_chunk() {
        let mut f = StreamingTextFilter::new(vec!["he".to_string()]);
        assert_eq!(f.push("The"), Some("T".to_string()));
        assert!(f.is_stopped());
        assert_eq!(f.cumulative_emitted(), "T");

        let mut f = StreamingTextFilter::new(vec!["<|im_end|>".to_string()]);
        assert_eq!(f.push("Hello wor"), Some("Hello wor".to_string()));
        assert_eq!(f.push("ld.<|im_end|>tail"), Some("ld.".to_string()));
        assert_eq!(f.push(" more"), None);
        assert_eq!(f.cumulative_emitted(), "Hello world.");
    }

    /// Two stops in one chunk: the stream cuts at the earlier one, wherever
    /// it sits in the pattern list, as `truncate_at_first_stop` cuts the
    /// final text.
    #[test]
    fn streaming_filter_cuts_at_the_earliest_stop_in_a_chunk() {
        let mut f = StreamingTextFilter::new(vec!["B".to_string(), "A".to_string()]);
        assert_eq!(f.push("xAyBz"), Some("x".to_string()));
        assert_eq!(f.cumulative_emitted(), "x");
    }

    /// Text before the stop that could be the start of a stop stays held
    /// back, as the final-text cleanup trims it.
    #[test]
    fn streaming_filter_holds_a_partial_stop_left_before_the_stop() {
        let mut f = StreamingTextFilter::new(vec!["ab".to_string()]);
        assert_eq!(f.push("xaab"), Some("x".to_string()));
        assert_eq!(f.cumulative_emitted(), "x");

        let mut text = String::from("xaab");
        assert!(truncate_at_first_stop(&mut text, &["ab"]));
        assert!(trim_partial_stop_suffix(&mut text, &["ab"]));
        assert_eq!(text, "x");
    }
}
