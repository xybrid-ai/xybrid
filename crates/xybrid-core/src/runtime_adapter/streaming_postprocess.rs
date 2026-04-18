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

/// Stop markers emitted by common chat templates:
/// - `<|im_end|>` / `<|im_start|>` / `<|endoftext|>`: ChatML (Qwen, Phi)
/// - `</s>`: Llama 2 style
/// - `<end_of_turn>`: Gemma
///
/// Always check for these in addition to user-supplied stop sequences —
/// they're emitted by the chat template, not by the user.
pub(crate) const CHAT_STOP_PATTERNS: &[&str] = &[
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "</s>",
    "<end_of_turn>",
];

/// Fallback variants without the leading `<`, for models whose
/// tokenizer breaks the angle bracket off from the marker body.
///
/// Only safe to use in **final-text cleanup** — during streaming
/// these would false-positive on legitimate text that happens to
/// start with `|`. [`StreamingTextFilter`] does NOT include them.
pub(crate) const CHAT_STOP_PATTERNS_BROKEN: &[&str] =
    &["|im_end|>", "|im_start|>", "|endoftext|>", "end_of_turn>"];

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

/// Strip every `<think>...</think>` block from `text`.
///
/// An unclosed opening tag strips from `<think>` to end of string —
/// this is the partial-stream safety case for Qwen 3.5 and similar
/// models that emit reasoning blocks before the final answer.
pub(crate) fn strip_thinking_tags(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            let end_absolute = start + end + "</think>".len();
            result.replace_range(start..end_absolute, "");
        } else {
            result.truncate(start);
            break;
        }
    }
    result
}

/// Truncate `text` at the earliest complete occurrence of any pattern
/// in `patterns`. Returns `true` if a stop was found and truncated.
pub(crate) fn truncate_at_first_stop<S: AsRef<str>>(text: &mut String, patterns: &[S]) -> bool {
    let mut earliest: Option<usize> = None;
    for p in patterns {
        if let Some(pos) = text.find(p.as_ref()) {
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
/// Returns `true` on first trim and stops (matches the prior
/// llama_cpp behavior).
pub(crate) fn trim_partial_stop_suffix<S: AsRef<str>>(text: &mut String, patterns: &[S]) -> bool {
    for pattern in patterns {
        let p = pattern.as_ref();
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
/// chunks arrive. It also transparently suppresses
/// `<think>...</think>` blocks and stops emitting once a complete
/// stop pattern is observed (see [`Self::is_stopped`]).
///
/// After [`Self::is_stopped`] returns `true`, further `push` calls
/// are no-ops and return `None`. The backend should still drain the
/// engine's stream, but nothing further will be emitted upward.
pub(crate) struct StreamingTextFilter {
    stop_patterns: Vec<String>,
    cumulative_text: String,
    last_emitted_len: usize,
    inside_think_block: bool,
    hit_stop_pattern: bool,
}

impl StreamingTextFilter {
    pub fn new(stop_patterns: Vec<String>) -> Self {
        Self {
            stop_patterns,
            cumulative_text: String::new(),
            last_emitted_len: 0,
            inside_think_block: false,
            hit_stop_pattern: false,
        }
    }

    /// Whether a complete stop pattern has been observed.
    pub fn is_stopped(&self) -> bool {
        self.hit_stop_pattern
    }

    /// Cumulative text up to the last emission point. Use this to
    /// populate `PartialToken.cumulative_text` on user callbacks —
    /// it always ends where the last emitted chunk ended.
    pub fn cumulative_emitted(&self) -> &str {
        &self.cumulative_text[..self.last_emitted_len]
    }

    /// Push a raw chunk. Returns `Some(safe_text)` if new content is
    /// ready for the user callback; `None` if the chunk is fully
    /// held back (partial stop prefix, inside a `<think>` block, or
    /// after a stop has been observed).
    pub fn push(&mut self, chunk: &str) -> Option<String> {
        if self.hit_stop_pattern {
            return None;
        }

        self.cumulative_text.push_str(chunk);

        // Enter a <think> block?
        if !self.inside_think_block
            && self.cumulative_text[self.last_emitted_len..].contains("<think>")
        {
            self.inside_think_block = true;
            if let Some(pos) = self.cumulative_text.find("<think>") {
                self.last_emitted_len = pos;
            }
        }

        if self.inside_think_block {
            if self.cumulative_text.contains("</think>") {
                self.inside_think_block = false;
                self.cumulative_text = strip_thinking_tags(&self.cumulative_text);
                // After stripping, last_emitted_len may point past end.
                self.last_emitted_len = self.last_emitted_len.min(self.cumulative_text.len());
            }
            return None;
        }

        // Complete stop pattern observed?
        for pattern in &self.stop_patterns {
            if self.cumulative_text.contains(pattern.as_str()) {
                self.hit_stop_pattern = true;
                if let Some(pos) = self.cumulative_text.find(pattern.as_str()) {
                    self.cumulative_text.truncate(pos);
                }
                return None;
            }
        }

        // Find the safe emission boundary (exclude potential partial
        // stop prefixes hanging off the tail).
        let safe_end = find_potential_stop_start(&self.cumulative_text, &self.stop_patterns)
            .unwrap_or(self.cumulative_text.len());

        if safe_end > self.last_emitted_len {
            let safe = self.cumulative_text[self.last_emitted_len..safe_end].to_string();
            self.last_emitted_len = safe_end;
            Some(safe)
        } else {
            None
        }
    }
}

/// Find the position where a potential partial-stop prefix begins at
/// the tail of `text`, if any. Kept internal — callers use
/// [`StreamingTextFilter`] instead.
fn find_potential_stop_start(text: &str, patterns: &[String]) -> Option<usize> {
    for pattern in patterns {
        for prefix_len in 1..=pattern.len() {
            let prefix = &pattern[..prefix_len];
            if text.ends_with(prefix) {
                return Some(text.len() - prefix_len);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
