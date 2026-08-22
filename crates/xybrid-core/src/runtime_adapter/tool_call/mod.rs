//! LFM2-family, gemma-4-family, and FunctionGemma tool-call protocols.
//!
//! LFM2-family GGUF models emit pythonic function-call text between special
//! markers. Gemma-4-family GGUF models emit `call:name{args}` blocks inside
//! their model turn. FunctionGemma uses the same structured call grammar with
//! different markers and a different string delimiter. This module is always
//! compiled because every protocol is plain text: parsing and continuation
//! composition do not depend on a specific runtime, feature flag, or I/O surface.
//!
//! Model output is untrusted. Malformed candidates are skipped instead of
//! surfaced as fatal errors so callers can still use surrounding prose or
//! sibling tool calls. LFM2 nested dictionaries, nested calls, and nested lists
//! are deliberately unsupported in v1 to keep the accepted grammar predictable;
//! both structured Gemma dialects support nested maps and lists with an
//! explicit depth cap.
//!
//! The cross-protocol scanner and continuation dispatcher live here.
//! Grammar-specific parsing and response formatting live in `lfm2` and
//! `gemma`, with shared cursor primitives isolated in `cursor`.

mod cursor;
#[cfg(test)]
mod functiongemma_tests;
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
const FUNCTION_GEMMA_DECLARATION_START: &str = "<start_function_declaration>";
const FUNCTION_GEMMA_DECLARATION_END: &str = "<end_function_declaration>";
const FUNCTION_GEMMA_TOOL_CALL_START: &str = "<start_function_call>";
const FUNCTION_GEMMA_TOOL_CALL_END: &str = "<end_function_call>";
const FUNCTION_GEMMA_TOOL_RESPONSE_START: &str = "<start_function_response>";
const FUNCTION_GEMMA_TOOL_RESPONSE_END: &str = "<end_function_response>";
const FUNCTION_GEMMA_STRING_DELIMITER: &str = "<escape>";
const FAMILY_MARKERS: &[&str] = &[
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESPONSE_START,
    TOOL_RESPONSE_END,
    GEMMA_TOOL_CALL_START,
    GEMMA_TOOL_CALL_END,
    GEMMA_TOOL_RESPONSE_START,
    GEMMA_TOOL_RESPONSE_END,
    FUNCTION_GEMMA_DECLARATION_START,
    FUNCTION_GEMMA_DECLARATION_END,
    FUNCTION_GEMMA_TOOL_CALL_START,
    FUNCTION_GEMMA_TOOL_CALL_END,
    FUNCTION_GEMMA_TOOL_RESPONSE_START,
    FUNCTION_GEMMA_TOOL_RESPONSE_END,
];
const RESERVED_PROTOCOL_MARKERS: &[&str] = &[
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESPONSE_START,
    TOOL_RESPONSE_END,
    GEMMA_TOOL_CALL_START,
    GEMMA_TOOL_CALL_END,
    GEMMA_TOOL_RESPONSE_START,
    GEMMA_TOOL_RESPONSE_END,
    FUNCTION_GEMMA_DECLARATION_START,
    FUNCTION_GEMMA_DECLARATION_END,
    FUNCTION_GEMMA_TOOL_CALL_START,
    FUNCTION_GEMMA_TOOL_CALL_END,
    FUNCTION_GEMMA_TOOL_RESPONSE_START,
    FUNCTION_GEMMA_TOOL_RESPONSE_END,
    FUNCTION_GEMMA_STRING_DELIMITER,
    "<|\"|>",
    "<|im_end|>",
    "<|im_start|>",
    "<turn|>",
    "<|turn>",
    "<start_of_turn>",
    "<end_of_turn>",
];

/// Shared turn markers used by continuation stop configuration and
/// [`truncate_at_turn_marker`].
pub(crate) const TURN_MARKERS: &[&str] = &[
    "<|im_end|>",
    "<|im_start|>",
    "<turn|>",
    "<|turn>",
    "<start_of_turn>",
    "<end_of_turn>",
    FUNCTION_GEMMA_TOOL_RESPONSE_START,
];

/// Complete tool-protocol block delimiters as `(start, end, is_call_block)`,
/// for streaming suppression and any consumer that must recognize whole
/// blocks. Includes each family's calls plus structured Gemma declaration and
/// response blocks. Declarations and responses are suppressible protocol
/// traffic but not tool calls, hence the flag.
pub(crate) const TOOL_BLOCK_MARKERS: &[(&str, &str, bool)] = &[
    (TOOL_CALL_START, TOOL_CALL_END, true),
    (GEMMA_TOOL_CALL_START, GEMMA_TOOL_CALL_END, true),
    (GEMMA_TOOL_RESPONSE_START, GEMMA_TOOL_RESPONSE_END, false),
    (
        FUNCTION_GEMMA_TOOL_CALL_START,
        FUNCTION_GEMMA_TOOL_CALL_END,
        true,
    ),
    (
        FUNCTION_GEMMA_TOOL_RESPONSE_START,
        FUNCTION_GEMMA_TOOL_RESPONSE_END,
        false,
    ),
    (
        FUNCTION_GEMMA_DECLARATION_START,
        FUNCTION_GEMMA_DECLARATION_END,
        false,
    ),
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
    /// FunctionGemma: Gemma turns, `call:name{...}` notation with
    /// `<escape>`-delimited strings, wrapper-shaped tool declarations.
    FunctionGemma,
}

impl ToolCallProtocol {
    /// Detect from an embedded chat template's source text.
    /// Structured Gemma templates carry distinct declaration markers; the
    /// LFM2/ChatML marker family contains neither marker.
    pub(crate) fn detect_from_template(template: &str) -> Self {
        if template.contains(FUNCTION_GEMMA_DECLARATION_START) {
            Self::FunctionGemma
        } else if template.contains("<|tool>") {
            Self::Gemma
        } else {
            Self::Lfm2
        }
    }

    pub(crate) fn detect_from_prompt(prompt: &str) -> Option<Self> {
        if prompt.contains(FUNCTION_GEMMA_DECLARATION_START)
            || prompt.contains(FUNCTION_GEMMA_DECLARATION_END)
        {
            Some(Self::FunctionGemma)
        } else if prompt.contains("<|turn>") {
            Some(Self::Gemma)
        } else if prompt.contains("<|im_start|>") {
            Some(Self::Lfm2)
        } else {
            None
        }
    }

    fn call_start(self) -> &'static str {
        match self {
            Self::Lfm2 => TOOL_CALL_START,
            Self::Gemma => GEMMA_TOOL_CALL_START,
            Self::FunctionGemma => FUNCTION_GEMMA_TOOL_CALL_START,
        }
    }

    fn call_end(self) -> &'static str {
        match self {
            Self::Lfm2 => TOOL_CALL_END,
            Self::Gemma => GEMMA_TOOL_CALL_END,
            Self::FunctionGemma => FUNCTION_GEMMA_TOOL_CALL_END,
        }
    }
}

/// Parses supported tool-call blocks from model output.
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
            ToolCallProtocol::FunctionGemma => {
                gemma::parse_function_gemma_block(after_start[..end].trim())
                    .into_iter()
                    .collect()
            }
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

/// Strips supported tool-call and tool-response protocol blocks.
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
    const BLOCK_MARKERS: &[(&str, &str)] = &[
        (TOOL_CALL_START, TOOL_CALL_END),
        (TOOL_RESPONSE_START, TOOL_RESPONSE_END),
        (GEMMA_TOOL_CALL_START, GEMMA_TOOL_CALL_END),
        (GEMMA_TOOL_RESPONSE_START, GEMMA_TOOL_RESPONSE_END),
        (
            FUNCTION_GEMMA_DECLARATION_START,
            FUNCTION_GEMMA_DECLARATION_END,
        ),
        (FUNCTION_GEMMA_TOOL_CALL_START, FUNCTION_GEMMA_TOOL_CALL_END),
        (
            FUNCTION_GEMMA_TOOL_RESPONSE_START,
            FUNCTION_GEMMA_TOOL_RESPONSE_END,
        ),
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

/// Returns whether text contains any tool-family protocol marker.
///
/// Use this to distinguish "the model made no tool call" from "the model
/// attempted a tool call that failed to parse" when [`parse_tool_calls`]
/// returns an empty vector. Generic turn markers such as `<|im_end|>` are not
/// counted.
///
/// # Examples
///
/// ```
/// use xybrid_core::runtime_adapter::tool_call::has_tool_markers;
///
/// assert!(has_tool_markers("<|tool_call_start|>[weather()]"));
/// assert!(!has_tool_markers("plain assistant prose"));
/// ```
pub fn has_tool_markers(text: &str) -> bool {
    FAMILY_MARKERS.iter().any(|marker| text.contains(marker))
}

/// Composes a raw tool-call continuation prompt with tool responses.
///
/// ChatML/LFM2 prompts keep the legacy scaffold. Structured Gemma prompts are
/// detected from their declaration or turn markers; their tool responses are
/// appended directly after `prior_assistant_text` because their templates emit
/// no new model-turn header after a tool response.
///
/// Crate-internal on purpose: callers drive continuations through
/// [`Envelope::tool_results`](crate::ir::Envelope::tool_results), not by
/// hand-composing raw prompts.
///
/// # Errors
///
/// Returns [`AdapterError::InvalidInput`] when responses do not exactly answer
/// the prior calls or `base_prompt` has no supported protocol markers.
pub(crate) fn compose_tool_continuation(
    base_prompt: &str,
    prior_assistant_text: &str,
    tool_responses_json: &str,
) -> Result<String, AdapterError> {
    let value: Value = serde_json::from_str(tool_responses_json).map_err(|err| {
        AdapterError::InvalidInput(format!("tool_responses_json must be valid json: {err}"))
    })?;
    let calls = parse_tool_calls(prior_assistant_text);
    let responses = validate_tool_responses(&value, &calls)?;

    let protocol = ToolCallProtocol::detect_from_prompt(base_prompt).ok_or_else(|| {
        AdapterError::InvalidInput(
            "could not detect a supported tool-call protocol in the rendered prompt; refusing \
             to substitute a ChatML continuation"
                .to_string(),
        )
    })?;

    match protocol {
        ToolCallProtocol::Lfm2 => {
            lfm2::compose_chatml_continuation(base_prompt, prior_assistant_text, &responses)
        }
        ToolCallProtocol::Gemma => {
            gemma::compose_gemma_continuation(base_prompt, prior_assistant_text, &responses)
        }
        ToolCallProtocol::FunctionGemma => gemma::compose_function_gemma_continuation(
            base_prompt,
            prior_assistant_text,
            &responses,
        ),
    }
}

fn validate_tool_responses(value: &Value, calls: &[ToolCall]) -> Result<Vec<Value>, AdapterError> {
    let responses = value.as_array().ok_or_else(|| {
        AdapterError::InvalidInput("tool_responses_json must be a json array".to_string())
    })?;

    if responses.is_empty() {
        return Err(AdapterError::InvalidInput(
            "tool_responses_json must contain at least one response".to_string(),
        ));
    }
    if calls.is_empty() {
        return Err(AdapterError::InvalidInput(
            "prior assistant text must contain tool calls".to_string(),
        ));
    }
    if responses.len() != calls.len() {
        return Err(AdapterError::InvalidInput(format!(
            "tool response count {} must match tool call count {}",
            responses.len(),
            calls.len()
        )));
    }

    let mut ordered = vec![None; calls.len()];
    for (response_index, response) in responses.iter().enumerate() {
        let object = response.as_object().ok_or_else(|| {
            AdapterError::InvalidInput(format!(
                "tool response at index {response_index} must be an object"
            ))
        })?;
        if !object.contains_key("content") {
            return Err(AdapterError::InvalidInput(format!(
                "tool response at index {response_index} must include content"
            )));
        }

        let explicit_call_id = object
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        let call_index = if let Some(call_id) = explicit_call_id {
            calls
                .iter()
                .position(|call| call.id == call_id)
                .ok_or_else(|| {
                    AdapterError::InvalidInput(format!(
                        "tool response at index {response_index} references unknown call id {call_id}"
                    ))
                })?
        } else {
            response_index
        };
        let call = &calls[call_index];

        if ordered[call_index].is_some() {
            return Err(AdapterError::InvalidInput(format!(
                "tool call {} has duplicate responses",
                call.id
            )));
        }
        if let Some(name) = object
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        {
            if name != call.function.name {
                return Err(AdapterError::InvalidInput(format!(
                    "tool response at index {response_index} names {name} but call {} names {}",
                    call.id, call.function.name
                )));
            }
        }

        let mut resolved = object.clone();
        resolved.insert("call_id".to_string(), Value::String(call.id.clone()));
        resolved.insert(
            "name".to_string(),
            Value::String(call.function.name.clone()),
        );
        ordered[call_index] = Some(Value::Object(resolved));
    }

    let mut resolved = Vec::with_capacity(ordered.len());
    for (index, response) in ordered.into_iter().enumerate() {
        let Some(response) = response else {
            return Err(AdapterError::InvalidInput(format!(
                "tool call {} has no response",
                calls[index].id
            )));
        };
        resolved.push(response);
    }

    Ok(resolved)
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

pub(super) fn sanitize_tool_result_content(value: &Value) -> Value {
    match value {
        Value::String(value) => Value::String(neutralize_reserved_markers(value)),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(sanitize_tool_result_content)
                .collect::<Vec<_>>(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| {
                    (
                        neutralize_reserved_markers(key),
                        sanitize_tool_result_content(value),
                    )
                })
                .collect(),
        ),
        Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
    }
}

fn neutralize_reserved_markers(value: &str) -> String {
    let mut sanitized = value.to_string();
    for marker in RESERVED_PROTOCOL_MARKERS {
        sanitized = sanitized.replace(marker, neutralized_marker(marker));
    }
    sanitized
}

fn neutralized_marker(marker: &str) -> &'static str {
    if marker == TOOL_CALL_START {
        "tool_call_start"
    } else if marker == TOOL_CALL_END {
        "tool_call_end"
    } else if marker == TOOL_RESPONSE_START {
        "tool_response_start"
    } else if marker == TOOL_RESPONSE_END {
        "tool_response_end"
    } else if marker == GEMMA_TOOL_CALL_START || marker == GEMMA_TOOL_CALL_END {
        "tool_call"
    } else if marker == GEMMA_TOOL_RESPONSE_START || marker == GEMMA_TOOL_RESPONSE_END {
        "tool_response"
    } else if marker == FUNCTION_GEMMA_DECLARATION_START || marker == FUNCTION_GEMMA_DECLARATION_END
    {
        "function_declaration"
    } else if marker == FUNCTION_GEMMA_TOOL_CALL_START || marker == FUNCTION_GEMMA_TOOL_CALL_END {
        "function_call"
    } else if marker == FUNCTION_GEMMA_TOOL_RESPONSE_START
        || marker == FUNCTION_GEMMA_TOOL_RESPONSE_END
    {
        "function_response"
    } else if marker == FUNCTION_GEMMA_STRING_DELIMITER || marker == "<|\"|>" {
        "string_delimiter"
    } else if marker == "<|im_end|>" {
        "im_end"
    } else if marker == "<|im_start|>" {
        "im_start"
    } else if marker == "<turn|>"
        || marker == "<|turn>"
        || marker == "<start_of_turn>"
        || marker == "<end_of_turn>"
    {
        "turn"
    } else {
        "protocol_marker"
    }
}

/// Truncates `text` at the first supported turn marker and trims the tail.
///
/// The raw continuation path stops generation on protocol turn markers but,
/// unlike the chat path's stop-pattern cleanup, the backend's raw output keeps
/// the marker text. Cutting it here keeps response text clean and keeps the
/// next continuation protocol-faithful. Supported families do not legitimately
/// emit each other's turn markers as assistant content, so the combined marker
/// set is safe.
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
    [
        (ToolCallProtocol::Lfm2, TOOL_CALL_START),
        (ToolCallProtocol::Gemma, GEMMA_TOOL_CALL_START),
        (
            ToolCallProtocol::FunctionGemma,
            FUNCTION_GEMMA_TOOL_CALL_START,
        ),
    ]
    .into_iter()
    .filter_map(|(protocol, marker)| text.find(marker).map(|index| (protocol, index)))
    .min_by_key(|(_, index)| *index)
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

    fn assert_invalid_input(result: Result<String, AdapterError>) {
        assert!(matches!(result, Err(AdapterError::InvalidInput(_))));
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
            Some(ToolCallProtocol::Lfm2)
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
            Some(ToolCallProtocol::Gemma)
        );
    }

    #[test]
    fn continuation_rejects_an_unrecognized_prompt_protocol() {
        let prior = wrapped("[weather(city='Paris')]");
        let responses = r#"[{"call_id":"call_0","name":"weather","content":{"temperature_c":21}}]"#;

        let result = compose_tool_continuation("plain prompt", &prior, responses);

        assert_invalid_input(result);
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
    fn has_tool_markers_returns_true_for_lfm2_call_block() {
        let text = wrapped("[weather(city='Paris')]");

        assert!(has_tool_markers(&text));
    }

    #[test]
    fn has_tool_markers_returns_true_for_gemma_call_block() {
        let text = gemma_wrapped("call:weather{city:<|\"|>Paris<|\"|>}");

        assert!(has_tool_markers(&text));
    }

    #[test]
    fn has_tool_markers_returns_true_for_lone_start_marker() {
        let text = format!("assistant text {TOOL_CALL_START}");

        assert!(has_tool_markers(&text));
    }

    #[test]
    fn has_tool_markers_returns_true_for_gemma_response_block() {
        let text = format!("{GEMMA_TOOL_RESPONSE_START}sunny{GEMMA_TOOL_RESPONSE_END}");

        assert!(has_tool_markers(&text));
    }

    #[test]
    fn has_tool_markers_returns_false_for_plain_prose() {
        assert!(!has_tool_markers("plain assistant prose"));
    }

    #[test]
    fn has_tool_markers_returns_false_for_turn_markers() {
        assert!(!has_tool_markers("<|im_start|>assistant\nhi<|im_end|>"));
    }

    #[test]
    fn compose_returns_exact_prompt_when_happy_path() {
        let prior = format!("I'll check{}", wrapped("[weather()]"));
        let result = compose_tool_continuation(
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
            &prior,
            r#"[{"content":{"weather":"sunny"}}]"#,
        )
        .unwrap();

        assert_eq!(
            result,
            format!(
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n{prior}<|im_end|>\n<|im_start|>tool\n<|tool_response_start|>{{\"weather\":\"sunny\"}}<|tool_response_end|><|im_end|>\n<|im_start|>assistant\n"
            )
        );
    }

    #[test]
    fn compose_preserves_response_order_when_two_responses() {
        let prior = format!("{}{}", wrapped("[first()]"), wrapped("[second()]"));
        let result = compose_tool_continuation(
            "<|im_start|>assistant\n",
            &prior,
            r#"[{"content":"one"},{"content":2}]"#,
        )
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
        assert_invalid_input(compose_tool_continuation("base", "prior", "[]"));
    }

    #[test]
    fn compose_errors_when_element_missing_content() {
        let prior = wrapped("[weather()]");
        assert_invalid_input(compose_tool_continuation(
            "base",
            &prior,
            r#"[{"other":1}]"#,
        ));
    }

    #[test]
    fn compose_errors_when_prior_has_no_tool_calls() {
        assert_invalid_input(compose_tool_continuation(
            "base",
            "plain prior",
            r#"[{"content":1}]"#,
        ));
    }

    #[test]
    fn compose_errors_when_response_count_mismatches_calls() {
        let prior = format!("{}{}", wrapped("[first()]"), wrapped("[second()]"));

        assert_invalid_input(compose_tool_continuation(
            "base",
            &prior,
            r#"[{"content":"one"}]"#,
        ));
    }

    #[test]
    fn compose_errors_when_response_call_id_is_unknown() {
        let prior = wrapped("[weather()]");

        assert_invalid_input(compose_tool_continuation(
            "base",
            &prior,
            r#"[{"call_id":"call_99","content":"sunny"}]"#,
        ));
    }

    #[test]
    fn compose_errors_when_response_call_id_is_duplicate() {
        let prior = format!("{}{}", wrapped("[first()]"), wrapped("[second()]"));

        assert_invalid_input(compose_tool_continuation(
            "base",
            &prior,
            r#"[
                {"call_id":"call_0","content":"one"},
                {"call_id":"call_0","content":"two"}
            ]"#,
        ));
    }

    #[test]
    fn compose_errors_when_response_name_mismatches_call() {
        let prior = wrapped("[weather()]");

        assert_invalid_input(compose_tool_continuation(
            "base",
            &prior,
            r#"[{"call_id":"call_0","name":"wrong","content":"sunny"}]"#,
        ));
    }

    #[test]
    fn compose_orders_explicit_call_ids_by_prior_call_order() {
        let prior = format!("{}{}", wrapped("[first()]"), wrapped("[second()]"));
        let result = compose_tool_continuation(
            "<|im_start|>assistant\n",
            &prior,
            r#"[
                {"call_id":"call_1","name":"second","content":"two"},
                {"call_id":"call_0","name":"first","content":"one"}
            ]"#,
        )
        .unwrap();

        assert!(result.contains(
            "<|tool_response_start|>\"one\"<|tool_response_end|><|tool_response_start|>\"two\"<|tool_response_end|>"
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
