//! Conversation stream chunks back to the model's own text.
//!
//! A conversation chunk is one JSON message carrying the newest part of the
//! reply (observed on v0.19.1 with Gemma 4):
//!
//! - answer text: `{"role":"assistant","content":[{"type":"text","text":"Hi"}]}`
//! - reasoning, when thinking is on:
//!   `{"role":"assistant","channels":{"thought":"Hm"},"reasoning_content":"Hm"}`
//! - the raw text of a tool call while the model writes it, when its channel
//!   is enabled: `{"role":"assistant","channels":{"tool_call":"call"}}`
//! - the call once the runtime has parsed it:
//!   `{"role":"assistant","tool_calls":[{"type":"function","function":{...}}]}`
//!
//! Everything above this backend works on the text a model emits: tool calls
//! are parsed from it, the streaming filter hides them, and an application
//! hands the text back to continue after tool results. So the chunks are put
//! back together as that text.
//!
//! # The runtime's parse is the call
//!
//! A call enters the text only when the runtime reports it parsed, and it is
//! written from that parsed form, between the call markers of the model's
//! protocol. The raw text on the tool-call channel is not used:
//!
//! - It is not a call yet. A reply cut off inside a call block leaves raw
//!   text and no parsed call; the runtime then hands the unfinished block
//!   over as answer text, opening marker included. That text passes through
//!   unchanged, so the reply ends in an unterminated block, which nothing
//!   downstream reads as a call.
//! - Read by xybrid's parser, it can mean something else than what the
//!   runtime parsed (the runtime keeps the first of two equal keys, xybrid's
//!   parser the last).
//!
//! Every call written is read back with xybrid's parser and must come out as
//! the name and arguments the runtime reported. A call that does not is an
//! error, never a different call.

use serde_json::{Map, Value};

use crate::runtime_adapter::tool_call::{parse_tool_calls, ToolCallProtocol};
use crate::runtime_adapter::AdapterError;

/// The channel a conversation is asked to stream raw tool-call text on.
///
/// Its chunks keep the stream alive and timed while a call is being written;
/// their text is not part of the reply (see the module docs).
pub(super) const TOOL_CALL_CHANNEL: &str = "tool_call";

/// One decoded part of a chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Piece {
    /// Text the model emitted, tool-call blocks included.
    Raw(String),
    /// Reasoning the runtime reported on a channel of its own.
    Reasoning(String),
}

/// Decoder for the chunks of one reply.
pub(super) struct ChunkDecoder {
    /// The model's tool-call protocol; `None` when the request offered no
    /// tools.
    protocol: Option<ToolCallProtocol>,
}

impl ChunkDecoder {
    pub(super) fn new(protocol: Option<ToolCallProtocol>) -> Self {
        Self { protocol }
    }

    /// Decode one chunk payload, in the order its parts were generated.
    pub(super) fn decode(&self, json: &str) -> Result<Vec<Piece>, AdapterError> {
        let message: Value = serde_json::from_str(json).map_err(|err| {
            AdapterError::InferenceFailed(format!(
                "LiteRT-LM returned a chunk that is not JSON: {err}"
            ))
        })?;
        let mut pieces = Vec::new();

        if let Some(content) = message.get("content") {
            let text = content_text(content);
            if !text.is_empty() {
                pieces.push(Piece::Raw(text));
            }
        }

        // The runtime repeats reasoning in a field of its own beside the
        // channel it arrived on; that field is read when present, so the
        // text is counted once.
        let reasoning_field = message
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty());
        if let Some(text) = reasoning_field {
            pieces.push(Piece::Reasoning(text.to_string()));
        }

        if let Some(channels) = message.get("channels").and_then(Value::as_object) {
            for (name, value) in channels {
                let Some(text) = value.as_str().filter(|text| !text.is_empty()) else {
                    continue;
                };
                if name != TOOL_CALL_CHANNEL && reasoning_field.is_none() {
                    pieces.push(Piece::Reasoning(text.to_string()));
                }
            }
        }

        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                pieces.push(Piece::Raw(self.call_block(call)?));
            }
        }

        Ok(pieces)
    }

    /// One parsed call as a block of the model's protocol.
    fn call_block(&self, call: &Value) -> Result<String, AdapterError> {
        let protocol = self.protocol.ok_or_else(|| {
            unrepresentable("the request offered no tools, so no tool protocol is known")
        })?;
        let function = call.get("function").unwrap_or(call);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| unrepresentable("it has no function name"))?;
        let arguments = call_arguments(function.get("arguments"))?;

        let body = protocol
            .format_call(name, &arguments)
            .ok_or_else(|| unrepresentable("xybrid has no writer for the model's tool protocol"))?;
        let block = format!("{}{}{}", protocol.call_start(), body, protocol.call_end());

        // Read back what was written: the text is what the executor parses
        // and what a continuation replays, so it has to say what the runtime
        // said.
        let read_back = parse_tool_calls(&block);
        let same = matches!(
            read_back.as_slice(),
            [only] if only.function.name == name
                && serde_json::from_str::<Value>(&only.function.arguments).ok().as_ref()
                    == Some(&arguments)
        );
        if !same {
            return Err(unrepresentable(
                "its name or arguments do not survive the model's call notation",
            ));
        }
        Ok(block)
    }
}

/// A call's arguments as a JSON object.
///
/// The runtime reports an object. A JSON string holding one is accepted too
/// (the OpenAI wire shape), and absent or null arguments are none.
fn call_arguments(arguments: Option<&Value>) -> Result<Value, AdapterError> {
    let not_an_object = || unrepresentable("its arguments are not a JSON object");
    match arguments {
        None | Some(Value::Null) => Ok(Value::Object(Map::new())),
        Some(Value::Object(map)) => Ok(Value::Object(map.clone())),
        Some(Value::String(text)) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(map)) => Ok(Value::Object(map)),
            _ => Err(not_an_object()),
        },
        Some(_) => Err(not_an_object()),
    }
}

/// The error for a parsed call that cannot be put into the reply's text.
/// The call itself is left out: its arguments are model output.
fn unrepresentable(why: &str) -> AdapterError {
    AdapterError::InferenceFailed(format!(
        "LiteRT-LM reported a tool call that cannot be written as text: {why}"
    ))
}

/// The text of a message's `content`: a string, or the text parts of an
/// array.
fn content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| match part {
                Value::String(text) => Some(text.as_str()),
                other => other.get("text").and_then(Value::as_str),
            })
            .collect(),
        Value::Object(part) => part
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn raw(pieces: &[Piece]) -> String {
        pieces
            .iter()
            .filter_map(|piece| match piece {
                Piece::Raw(text) => Some(text.as_str()),
                Piece::Reasoning(_) => None,
            })
            .collect()
    }

    fn decode_all(decoder: &ChunkDecoder, chunks: &[&str]) -> Vec<Piece> {
        chunks
            .iter()
            .flat_map(|chunk| decoder.decode(chunk).expect("a valid chunk"))
            .collect()
    }

    fn gemma() -> ChunkDecoder {
        ChunkDecoder::new(Some(ToolCallProtocol::Gemma))
    }

    fn parsed_call(name: &str, arguments: Value) -> String {
        json!({
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
        })
        .to_string()
    }

    fn channel(text: &str) -> String {
        json!({"role": "assistant", "channels": {TOOL_CALL_CHANNEL: text}}).to_string()
    }

    fn content(text: &str) -> String {
        json!({"role": "assistant", "content": [{"type": "text", "text": text}]}).to_string()
    }

    /// The chunks Gemma 4 E2B produced on LiteRT-LM v0.19.1 for one tool
    /// call, with the tool-call channel enabled.
    const GEMMA4_CALL_CHUNKS: &[&str] = &[
        r#"{"role":"assistant","channels":{"tool_call":"call"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":":"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":"get"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":"_"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":"temperature"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":"{"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":"room"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":":"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":"<|\"|>"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":"bedroom"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":"<|\"|>"}}"#,
        r#"{"role":"assistant","channels":{"tool_call":"}"}}"#,
        r#"{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"get_temperature","arguments":{"room":"bedroom"}}}]}"#,
    ];

    #[test]
    fn answer_text_passes_through() {
        let decoder = ChunkDecoder::new(None);
        let pieces = decode_all(
            &decoder,
            &[
                r#"{"role":"assistant","content":[{"type":"text","text":"hello"}]}"#,
                r#"{"role":"assistant","content":[{"type":"text","text":" world"}]}"#,
            ],
        );
        assert_eq!(raw(&pieces), "hello world");
    }

    #[test]
    fn string_content_is_accepted_too() {
        let decoder = ChunkDecoder::new(None);
        let pieces = decoder
            .decode(r#"{"role":"assistant","content":"hi"}"#)
            .expect("a valid chunk");
        assert_eq!(raw(&pieces), "hi");
    }

    #[test]
    fn a_streamed_call_enters_the_text_once_when_the_runtime_has_parsed_it() {
        let decoder = gemma();
        let (last, streamed) = GEMMA4_CALL_CHUNKS.split_last().expect("chunks");

        let before = decode_all(&decoder, streamed);
        assert!(
            before.is_empty(),
            "raw call text is not part of the reply: {before:?}"
        );

        let text = raw(&decoder.decode(last).expect("a valid chunk"));
        assert_eq!(
            text,
            r#"<|tool_call>call:get_temperature{room:<|"|>bedroom<|"|>}<tool_call|>"#
        );
        let calls = parse_tool_calls(&text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_temperature");
        assert_eq!(calls[0].function.arguments, r#"{"room":"bedroom"}"#);
    }

    #[test]
    fn two_calls_are_two_blocks() {
        let decoder = gemma();
        let text = raw(&decode_all(
            &decoder,
            &[
                &channel("call:a{}"),
                &parsed_call("a", json!({})),
                &channel("call:b{}"),
                &parsed_call("b", json!({})),
            ],
        ));
        assert_eq!(
            text,
            "<|tool_call>call:a{}<tool_call|><|tool_call>call:b{}<tool_call|>"
        );
        assert_eq!(parse_tool_calls(&text).len(), 2);
    }

    #[test]
    fn several_calls_in_one_chunk_each_get_a_block() {
        let decoder = gemma();
        let chunk = json!({
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": "a", "arguments": {"n": 1}}},
                {"type": "function", "function": {"name": "b", "arguments": {"n": 2}}},
            ],
        })
        .to_string();
        let text = raw(&decoder.decode(&chunk).expect("a valid chunk"));
        let calls = parse_tool_calls(&text);
        assert_eq!(calls.len(), 2, "text was {text:?}");
        assert_eq!(calls[0].function.arguments, r#"{"n":1}"#);
        assert_eq!(calls[1].function.arguments, r#"{"n":2}"#);
    }

    #[test]
    fn a_call_reported_without_its_text_is_written_from_the_parsed_form() {
        let decoder = gemma();
        let text = raw(&decode_all(
            &decoder,
            &[
                &content("Checking. "),
                &parsed_call(
                    "get_temperature",
                    json!({"room": "bedroom", "floor": 2, "exact": true, "offset": -0.5,
                           "tags": ["a", "b"], "where": {"wing": "east"}, "note": null}),
                ),
            ],
        ));
        let calls = parse_tool_calls(&text);
        assert_eq!(calls.len(), 1, "text was {text:?}");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).expect("json"),
            json!({"room": "bedroom", "floor": 2, "exact": true, "offset": -0.5,
                   "tags": ["a", "b"], "where": {"wing": "east"}, "note": null})
        );
        assert!(text.starts_with("Checking. <|tool_call>"));
    }

    /// The runtime ran out of output budget inside a call block. It never
    /// reports a parsed call; it hands the unfinished block over as answer
    /// text, opening marker included (upstream `SendCompleteMessage`).
    #[test]
    fn a_call_cut_off_by_the_budget_is_not_a_call() {
        let decoder = gemma();
        let text = raw(&decode_all(
            &decoder,
            &[
                &content("Checking. "),
                &channel("call:get_temperature{room:"),
                &channel(r#"<|"|>bed"#),
                &content(r#"<|tool_call>call:get_temperature{room:<|"|>bed"#),
            ],
        ));
        assert_eq!(
            text, r#"Checking. <|tool_call>call:get_temperature{room:<|"|>bed"#,
            "the unfinished block appears once, and unterminated"
        );
        assert!(parse_tool_calls(&text).is_empty());
    }

    /// The same cut, one token later: the call's text is complete but its
    /// closing marker was never generated. It is still not a call.
    #[test]
    fn a_complete_looking_call_without_its_closing_marker_is_not_a_call() {
        let decoder = gemma();
        let text = raw(&decode_all(
            &decoder,
            &[&channel("call:f{}"), &content("<|tool_call>call:f{}")],
        ));
        assert_eq!(text, "<|tool_call>call:f{}");
        assert!(parse_tool_calls(&text).is_empty());
    }

    /// The runtime keeps the first of two equal keys; xybrid's parser, given
    /// the raw text, would keep the last. The caller must get the runtime's.
    #[test]
    fn the_runtimes_parse_wins_over_the_raw_call_text() {
        let decoder = gemma();
        let text = raw(&decode_all(
            &decoder,
            &[
                &channel("call:f{x:1,x:2}"),
                &parsed_call("f", json!({"x": 1})),
            ],
        ));
        assert_eq!(text, "<|tool_call>call:f{x:1}<tool_call|>");
        let calls = parse_tool_calls(&text);
        assert_eq!(calls[0].function.arguments, r#"{"x":1}"#);
    }

    #[test]
    fn a_string_that_would_end_itself_in_the_models_notation_is_refused() {
        let decoder = gemma();
        let err = decoder
            .decode(&parsed_call("f", json!({"s": r#"a<|"|>,evil:1"#})))
            .expect_err("the text would read as other arguments");
        assert!(matches!(err, AdapterError::InferenceFailed(_)));
        assert!(
            !err.to_string().contains("evil"),
            "the arguments are model output and stay out of the error: {err}"
        );
    }

    #[test]
    fn a_string_holding_the_closing_marker_is_refused() {
        let decoder = gemma();
        let err = decoder
            .decode(&parsed_call("f", json!({"s": "x<tool_call|>y"})))
            .expect_err("the block would end early");
        assert!(matches!(err, AdapterError::InferenceFailed(_)));
    }

    #[test]
    fn a_name_the_notation_cannot_carry_is_refused() {
        let decoder = gemma();
        for name in ["", "two words", "f{}"] {
            let err = decoder
                .decode(&parsed_call(name, json!({})))
                .expect_err("not an identifier");
            assert!(matches!(err, AdapterError::InferenceFailed(_)), "{name:?}");
        }
    }

    #[test]
    fn arguments_as_a_json_string_are_read_and_anything_else_is_refused() {
        let decoder = gemma();
        let text = raw(&decoder
            .decode(&parsed_call("f", json!(r#"{"x":1}"#)))
            .expect("a JSON object in a string"));
        assert_eq!(text, "<|tool_call>call:f{x:1}<tool_call|>");

        for arguments in [json!("not json"), json!([1, 2]), json!(7)] {
            let err = decoder
                .decode(&parsed_call("f", arguments.clone()))
                .expect_err("not an object");
            assert!(
                matches!(err, AdapterError::InferenceFailed(_)),
                "{arguments}"
            );
        }
    }

    #[test]
    fn function_gemma_calls_use_its_own_markers() {
        let decoder = ChunkDecoder::new(Some(ToolCallProtocol::FunctionGemma));
        let text = raw(&decoder
            .decode(&parsed_call("plant", json!({"seed": "tomato"})))
            .expect("a valid chunk"));
        assert_eq!(
            text,
            "<start_function_call>call:plant{seed:<escape>tomato<escape>}<end_function_call>"
        );
        assert_eq!(parse_tool_calls(&text).len(), 1);
    }

    #[test]
    fn other_channels_are_reasoning_not_answer() {
        let decoder = ChunkDecoder::new(None);
        let pieces = decode_all(
            &decoder,
            &[
                r#"{"role":"assistant","channels":{"thought":"hmm"}}"#,
                r#"{"role":"assistant","content":[{"type":"text","text":"42"}]}"#,
            ],
        );
        assert_eq!(pieces[0], Piece::Reasoning("hmm".to_string()));
        assert_eq!(raw(&pieces), "42");
    }

    /// The shape Gemma 4 E2B produced on v0.19.1 with thinking on: the same
    /// text on the `thought` channel and in `reasoning_content`.
    #[test]
    fn reasoning_reported_twice_in_one_chunk_is_counted_once() {
        let decoder = ChunkDecoder::new(None);
        let pieces = decoder
            .decode(
                r#"{"role":"assistant","channels":{"thought":" Process"},"reasoning_content":" Process"}"#,
            )
            .expect("a valid chunk");
        assert_eq!(pieces, vec![Piece::Reasoning(" Process".to_string())]);
    }

    #[test]
    fn a_call_without_a_known_protocol_is_an_error_not_a_dropped_call() {
        let decoder = ChunkDecoder::new(None);
        let err = decoder
            .decode(&parsed_call("a", json!({})))
            .expect_err("no protocol to write it in");
        assert!(matches!(err, AdapterError::InferenceFailed(_)));
    }

    #[test]
    fn a_chunk_that_is_not_json_is_an_error() {
        let decoder = ChunkDecoder::new(None);
        assert!(decoder.decode("not json").is_err());
    }
}
