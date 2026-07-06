//! LFM2-family and gemma-4-family tool-call text protocol support.
//!
//! LFM2-family GGUF models emit pythonic function-call text between special
//! markers. Gemma-4-family GGUF models emit `call:name{args}` blocks inside
//! their model turn. This module is always compiled because both protocols are
//! plain text: parsing and continuation composition do not depend on a specific
//! runtime, feature flag, or I/O surface.
//!
//! Model output is untrusted. Malformed candidates are skipped instead of
//! surfaced as fatal errors so callers can still use surrounding prose or
//! sibling tool calls. LFM2 nested dictionaries, nested calls, and nested lists
//! are deliberately unsupported in v1 to keep the accepted grammar predictable;
//! gemma-4 supports nested maps and lists with an explicit depth cap.
//!
//! The cross-protocol scanner and continuation dispatcher live here.
//! Grammar-specific parsing and response formatting live in `lfm2` and
//! `gemma`, with shared cursor primitives isolated in `cursor`.

mod cursor;
mod gemma;
mod lfm2;

use crate::gateway::{FunctionCall, ToolCall};
use crate::runtime_adapter::AdapterError;
use serde_json::Value;

const TOOL_CALL_START: &str = "<|tool_call_start|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";
const TOOL_RESPONSE_START: &str = "<|tool_response_start|>";
const TOOL_RESPONSE_END: &str = "<|tool_response_end|>";
const GEMMA_TOOL_CALL_START: &str = "<|tool_call>";
const GEMMA_TOOL_CALL_END: &str = "<tool_call|>";
const GEMMA_TOOL_RESPONSE_START: &str = "<|tool_response>";
const GEMMA_TOOL_RESPONSE_END: &str = "<tool_response|>";

/// Shared turn markers used by continuation stop configuration and
/// [`truncate_at_turn_marker`].
pub(crate) const TURN_MARKERS: &[&str] = &["<|im_end|>", "<|im_start|>", "<turn|>", "<|turn>"];

/// Complete tool-protocol block delimiters as `(start, end, is_call_block)`,
/// for streaming suppression and any consumer that must recognize whole
/// blocks. Includes both families' call blocks and gemma-4's tool-response
/// block (gemma models emit a response opener as trained scaffolding after
/// their calls); a response block is suppressible protocol traffic but not a
/// tool call, hence the flag.
pub(crate) const TOOL_BLOCK_MARKERS: &[(&str, &str, bool)] = &[
    (TOOL_CALL_START, TOOL_CALL_END, true),
    (GEMMA_TOOL_CALL_START, GEMMA_TOOL_CALL_END, true),
    (GEMMA_TOOL_RESPONSE_START, GEMMA_TOOL_RESPONSE_END, false),
];

/// Which tool-call text protocol a model speaks. Owned here so every
/// dispatch site (tool shape at render time, continuation framing, output
/// scanning) derives it the same way instead of re-sniffing markers ad hoc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolCallProtocol {
    /// LFM2-family: ChatML turns, pythonic calls, bare-function tool lists.
    /// Also the default for unrecognized templates (HF-generic `tojson`
    /// tool lists take the same bare-function shape).
    Lfm2,
    /// Gemma-4-family: `<|turn>` framing, `call:name{...}` notation,
    /// wrapper-shaped tool declarations.
    Gemma,
}

impl ToolCallProtocol {
    /// Detect from an embedded chat template's source text.
    /// Gemma-4 templates render `<|tool>` declaration blocks; nothing in the
    /// LFM2/ChatML marker family contains that substring.
    pub(crate) fn detect_from_template(template: &str) -> Self {
        if template.contains("<|tool>") {
            Self::Gemma
        } else {
            Self::Lfm2
        }
    }

    /// Detect from a rendered prompt. Gemma-4 prompts frame turns with
    /// `<|turn>`. Invariant: a template that detects as Gemma renders a
    /// prompt that detects as Gemma (the framing comes from the template).
    pub(crate) fn detect_from_prompt(prompt: &str) -> Self {
        if prompt.contains("<|turn>") {
            Self::Gemma
        } else {
            Self::Lfm2
        }
    }

    fn call_start(self) -> &'static str {
        match self {
            Self::Lfm2 => TOOL_CALL_START,
            Self::Gemma => GEMMA_TOOL_CALL_START,
        }
    }

    fn call_end(self) -> &'static str {
        match self {
            Self::Lfm2 => TOOL_CALL_END,
            Self::Gemma => GEMMA_TOOL_CALL_END,
        }
    }
}

/// Parses LFM2-family and gemma-4-family tool-call blocks from model output.
///
/// Complete marker blocks are scanned in left-to-right order across both
/// protocols. Malformed call candidates are skipped silently; successfully
/// parsed calls are numbered across the full response as `call_0`, `call_1`,
/// and so on.
///
/// # Examples
///
/// ```
/// use xybrid_core::runtime_adapter::tool_call::parse_tool_calls;
///
/// let calls = parse_tool_calls(
///     r#"<|tool_call_start|>[weather(city="Paris")]<|tool_call_end|>"#,
/// );
///
/// assert_eq!(calls.len(), 1);
/// assert_eq!(calls[0].function.name, "weather");
/// assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
///
/// let gemma = parse_tool_calls(
///     r#"<|tool_call>call:get_temperature{room:<|"|>bedroom<|"|>}<tool_call|>"#,
/// );
///
/// assert_eq!(gemma[0].function.name, "get_temperature");
/// assert_eq!(gemma[0].function.arguments, r#"{"room":"bedroom"}"#);
/// ```
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut parsed = Vec::new();
    let mut rest = text;

    while let Some((protocol, start)) = find_next_call_marker(rest) {
        let after_start = &rest[start + protocol.call_start().len()..];
        let Some(end) = after_start.find(protocol.call_end()) else {
            break;
        };

        let calls = match protocol {
            ToolCallProtocol::Lfm2 => lfm2::parse_lfm2_block(after_start[..end].trim()),
            ToolCallProtocol::Gemma => gemma::parse_gemma_block(after_start[..end].trim())
                .into_iter()
                .collect(),
        };

        for call in calls {
            parsed.push(ToolCall {
                id: format!("call_{}", parsed.len()),
                tool_type: "function".into(),
                function: FunctionCall {
                    name: call.name,
                    arguments: call.arguments,
                },
            });
        }

        rest = &after_start[end + protocol.call_end().len()..];
    }

    parsed
}

/// Strips LFM2/gemma-4 tool-call AND tool-response protocol blocks.
///
/// Complete blocks are removed with both markers. A trailing start marker
/// with no end marker is display garbage, so everything from that marker to
/// the end of the text is removed. Response blocks are stripped too because
/// models only emit response markers as protocol scaffolding — gemma-4 in
/// particular is trained to end a call turn with a bare `<|tool_response>`
/// opener for the runtime to fill in — never as answer text.
///
/// # Examples
///
/// ```
/// use xybrid_core::runtime_adapter::tool_call::strip_tool_calls;
///
/// let visible = strip_tool_calls(
///     "answer <|tool_call_start|>[weather(city='Paris')]<|tool_call_end|> done",
/// );
///
/// assert_eq!(visible, "answer  done");
///
/// let visible = strip_tool_calls(
///     r#"answer <|tool_call>call:weather{city:<|"|>Paris<|"|>}<tool_call|><|tool_response>"#,
/// );
///
/// assert_eq!(visible, "answer");
/// ```
pub fn strip_tool_calls(text: &str) -> String {
    // Call blocks of both families plus both families' response blocks.
    const BLOCK_MARKERS: &[(&str, &str)] = &[
        (TOOL_CALL_START, TOOL_CALL_END),
        (TOOL_RESPONSE_START, TOOL_RESPONSE_END),
        (GEMMA_TOOL_CALL_START, GEMMA_TOOL_CALL_END),
        (GEMMA_TOOL_RESPONSE_START, GEMMA_TOOL_RESPONSE_END),
    ];

    let mut output = String::new();
    let mut rest = text;

    loop {
        let next = BLOCK_MARKERS
            .iter()
            .filter_map(|(start, end)| rest.find(start).map(|index| (index, *start, *end)))
            .min_by_key(|(index, _, _)| *index);
        let Some((index, start, end)) = next else {
            break;
        };
        output.push_str(&rest[..index]);
        let after_start = &rest[index + start.len()..];
        let Some(end_index) = after_start.find(end) else {
            // Unclosed opener: everything from here on is protocol garbage.
            return output.trim().to_string();
        };
        rest = &after_start[end_index + end.len()..];
    }

    output.push_str(rest);
    output.trim().to_string()
}

/// Composes a raw tool-call continuation prompt with tool responses.
///
/// ChatML/LFM2 prompts keep the legacy scaffold. Gemma-4 prompts are detected
/// by `<|turn>` in `base_prompt`; their tool responses are appended directly
/// after `prior_assistant_text` because gemma-4 keeps tool response blocks
/// inside the still-open model turn and its template emits no new turn header
/// after tool responses.
///
/// Crate-internal on purpose: callers drive continuations through
/// [`Envelope::tool_results`](crate::ir::Envelope::tool_results), not by
/// hand-composing raw prompts.
///
/// # Errors
///
/// Returns [`AdapterError::InvalidInput`] when `tool_responses_json` is not a
/// non-empty JSON array of objects containing a `content` field.
pub(crate) fn compose_tool_continuation(
    base_prompt: &str,
    prior_assistant_text: &str,
    tool_responses_json: &str,
) -> Result<String, AdapterError> {
    let value: Value = serde_json::from_str(tool_responses_json).map_err(|err| {
        AdapterError::InvalidInput(format!("tool_responses_json must be valid JSON: {err}"))
    })?;
    let responses = validate_tool_responses(&value)?;

    match ToolCallProtocol::detect_from_prompt(base_prompt) {
        ToolCallProtocol::Lfm2 => {
            lfm2::compose_chatml_continuation(base_prompt, prior_assistant_text, responses)
        }
        ToolCallProtocol::Gemma => {
            gemma::compose_gemma_continuation(base_prompt, prior_assistant_text, responses)
        }
    }
}

fn validate_tool_responses(value: &Value) -> Result<&Vec<Value>, AdapterError> {
    let responses = value.as_array().ok_or_else(|| {
        AdapterError::InvalidInput("tool_responses_json must be a JSON array".to_string())
    })?;

    if responses.is_empty() {
        return Err(AdapterError::InvalidInput(
            "tool_responses_json must contain at least one response".to_string(),
        ));
    }

    Ok(responses)
}

fn tool_response_content(response: &Value, index: usize) -> Result<&Value, AdapterError> {
    let object = response.as_object().ok_or_else(|| {
        AdapterError::InvalidInput(format!("tool response at index {index} must be an object"))
    })?;
    object.get("content").ok_or_else(|| {
        AdapterError::InvalidInput(format!(
            "tool response at index {index} must include content"
        ))
    })
}

/// Truncates `text` at the first supported turn marker and trims the tail.
///
/// The raw continuation path stops generation on protocol turn markers but,
/// unlike the chat path's stop-pattern cleanup, the backend's raw output keeps
/// the marker text. Cutting it here keeps response text clean and keeps the
/// next continuation protocol-faithful. Neither ChatML/LFM2 nor gemma-4
/// legitimately emits the other's turn markers as assistant content, so the
/// combined marker set is safe.
pub(crate) fn truncate_at_turn_marker(text: &mut String) {
    let cut = TURN_MARKERS
        .iter()
        .filter_map(|marker| text.find(marker))
        .min();
    if let Some(pos) = cut {
        text.truncate(pos);
        text.truncate(text.trim_end().len());
    }
}

fn find_next_call_marker(text: &str) -> Option<(ToolCallProtocol, usize)> {
    let lfm2 = text
        .find(TOOL_CALL_START)
        .map(|index| (ToolCallProtocol::Lfm2, index));
    let gemma = text
        .find(GEMMA_TOOL_CALL_START)
        .map(|index| (ToolCallProtocol::Gemma, index));

    match (lfm2, gemma) {
        (Some(left), Some(right)) => Some(if left.1 <= right.1 { left } else { right }),
        (Some(found), None) | (None, Some(found)) => Some(found),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrapped(content: &str) -> String {
        format!("{TOOL_CALL_START}{content}{TOOL_CALL_END}")
    }

    fn gemma_wrapped(content: &str) -> String {
        format!("{GEMMA_TOOL_CALL_START}{content}{GEMMA_TOOL_CALL_END}")
    }

    #[test]
    fn detect_from_template_returns_gemma_when_template_has_tool_block() {
        let template = "{% for tool in tools %}<|tool>{{ tool['function']['name'] }}";

        assert_eq!(
            ToolCallProtocol::detect_from_template(template),
            ToolCallProtocol::Gemma
        );
    }

    #[test]
    fn detect_from_template_returns_lfm2_for_chatml_tool_list_template() {
        let template = "List of tools: {{ tools | tojson }}";

        assert_eq!(
            ToolCallProtocol::detect_from_template(template),
            ToolCallProtocol::Lfm2
        );
    }

    #[test]
    fn detect_from_prompt_returns_lfm2_for_chatml_prompt() {
        let prompt = "<|im_start|>user
hi<|im_end|>
<|im_start|>assistant
";

        assert_eq!(
            ToolCallProtocol::detect_from_prompt(prompt),
            ToolCallProtocol::Lfm2
        );
    }

    #[test]
    fn detect_from_prompt_returns_gemma_for_turn_prompt() {
        let prompt = "<|turn>user
hi<turn|>
<|turn>model
";

        assert_eq!(
            ToolCallProtocol::detect_from_prompt(prompt),
            ToolCallProtocol::Gemma
        );
    }

    #[test]
    fn parse_mixed_protocols_number_calls_by_position() {
        let text = format!(
            "{}{}",
            wrapped("[first(x=1)]"),
            gemma_wrapped("call:second{y:2}")
        );

        let calls = parse_tool_calls(&text);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].function.name, "first");
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(calls[1].function.name, "second");
    }

    #[test]
    fn strip_removes_lfm2_and_gemma_blocks_in_one_text() {
        let text = format!(
            "before {} middle {} after",
            wrapped("[first(x=1)]"),
            gemma_wrapped("call:second{y:2}")
        );

        assert_eq!(strip_tool_calls(&text), "before  middle  after");
    }

    #[test]
    fn compose_returns_exact_prompt_when_happy_path() {
        let result = compose_tool_continuation(
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
            "I'll check",
            r#"[{"content":{"weather":"sunny"}}]"#,
        )
        .unwrap();

        assert_eq!(
            result,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nI'll check<|im_end|>\n<|im_start|>tool\n<|tool_response_start|>{\"weather\":\"sunny\"}<|tool_response_end|><|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn compose_preserves_response_order_when_two_responses() {
        let result =
            compose_tool_continuation("base", "prior", r#"[{"content":"one"},{"content":2}]"#)
                .unwrap();

        assert!(result.contains(
            "<|tool_response_start|>\"one\"<|tool_response_end|><|tool_response_start|>2<|tool_response_end|>"
        ));
    }

    #[test]
    fn compose_errors_when_json_is_malformed() {
        assert!(matches!(
            compose_tool_continuation("base", "prior", "not-json"),
            Err(AdapterError::InvalidInput(_))
        ));
    }

    #[test]
    fn compose_errors_when_array_is_empty() {
        assert!(matches!(
            compose_tool_continuation("base", "prior", "[]"),
            Err(AdapterError::InvalidInput(_))
        ));
    }

    #[test]
    fn compose_errors_when_element_missing_content() {
        assert!(matches!(
            compose_tool_continuation("base", "prior", r#"[{"other":1}]"#),
            Err(AdapterError::InvalidInput(_))
        ));
    }

    #[test]
    fn truncate_cuts_at_first_turn_marker_and_trims() {
        let mut text = "the answer is 21 <|im_end|>\n<|im_start|>user\nmore".to_string();
        truncate_at_turn_marker(&mut text);
        assert_eq!(text, "the answer is 21");
    }

    #[test]
    fn truncate_leaves_marker_free_text_untouched() {
        let mut text = "plain answer".to_string();
        truncate_at_turn_marker(&mut text);
        assert_eq!(text, "plain answer");
    }

    #[test]
    fn truncate_keeps_tool_call_block_before_marker() {
        let mut text = format!("{}<|im_end|>", wrapped("[f(x=1)]"));
        truncate_at_turn_marker(&mut text);
        assert_eq!(text, wrapped("[f(x=1)]"));
        assert_eq!(parse_tool_calls(&text).len(), 1);
    }

    #[test]
    fn truncate_cuts_at_gemma_turn_markers() {
        let mut end_marker = "answer <turn|>\n<|turn>user\nnext".to_string();
        truncate_at_turn_marker(&mut end_marker);
        assert_eq!(end_marker, "answer");

        let mut start_marker = "answer <|turn>user\nnext".to_string();
        truncate_at_turn_marker(&mut start_marker);
        assert_eq!(start_marker, "answer");
    }
}
