//! The backend against a scripted runtime (`xybrid_litert_lm::testing`): no
//! model and no native library, so the paths a real model takes only by
//! accident can be played on purpose.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use super::testing::{FakeModel, FakeRuntime, Script, Step};
use super::*;
use crate::abort::{AbortReason, CloudFallbackAbort};
use crate::gateway::Tool;
use crate::ir::{ImageDimensions, ImageFormat, ImageSource, MessageRole};
use crate::runtime_adapter::types::StreamingError;
use crate::runtime_adapter::{MultimodalImagePart, MultimodalMessagePart};

/// What Gemma 4's template opens a turn with; the backend reads the tool
/// protocol off the rendered prompt.
const GEMMA_PROMPT_PREFIX: &str = "<|turn>user\n";

fn fake() -> FakeRuntime {
    let fake = FakeRuntime::new();
    fake.set_render_prefix(GEMMA_PROMPT_PREFIX);
    fake
}

fn backend(fake: &FakeRuntime) -> LiteRtLmBackend {
    let mut backend = LiteRtLmBackend::new().expect("a backend");
    backend
        .load(&LlmConfig::new(
            fake.model_path().to_string_lossy().into_owned(),
        ))
        .expect("the fake model loads");
    backend
}

fn content(text: &str) -> Step {
    Step::text(
        json!({"role": "assistant", "content": [{"type": "text", "text": text}]}).to_string(),
    )
}

fn thought(text: &str) -> Step {
    Step::text(
        json!({"role": "assistant", "channels": {"thought": text}, "reasoning_content": text})
            .to_string(),
    )
}

fn call_text(text: &str) -> Step {
    Step::text(json!({"role": "assistant", "channels": {"tool_call": text}}).to_string())
}

fn parsed_call(name: &str, arguments: Value) -> Step {
    Step::text(
        json!({
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
        })
        .to_string(),
    )
}

fn ask(script: &str) -> Vec<ChatMessage> {
    vec![ChatMessage::user(format!("script:{script}"))]
}

fn weather_tool() -> Tool {
    Tool::function(
        "get_temperature",
        "Get the temperature in a room.",
        json!({
            "type": "object",
            "properties": { "room": { "type": "string" } },
            "required": ["room"],
        }),
    )
}

fn with_tools() -> GenerationConfig {
    GenerationConfig::greedy().with_tools([weather_tool()])
}

/// Tokens a streaming run delivered, in order.
#[derive(Default)]
struct Seen(Mutex<Vec<PartialToken>>);

impl Seen {
    fn callback(self: &Arc<Self>) -> StreamingCallback<'static> {
        let seen = Arc::clone(self);
        Box::new(move |token| {
            seen.0.lock().expect("the token log").push(token);
            Ok(())
        })
    }

    fn tokens(&self) -> Vec<PartialToken> {
        self.0.lock().expect("the token log").clone()
    }

    fn visible(&self) -> String {
        self.tokens().iter().map(|t| t.token.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

#[test]
fn a_streamed_reply_reaches_the_callback_and_ends_with_a_terminal_token() {
    let fake = fake();
    fake.set_usage(11, 2).script(
        "hi",
        Script::new([content("Hello"), content(" there"), Step::End]),
    );
    let backend = backend(&fake);
    let seen = Arc::new(Seen::default());

    let output = backend
        .generate_streaming(
            &[
                ChatMessage::system("Be brief."),
                ChatMessage::user("script:hi"),
            ],
            &GenerationConfig::greedy(),
            seen.callback(),
        )
        .expect("the turn runs");

    assert_eq!(output.text, "Hello there");
    assert_eq!(output.finish_reason, "stop");
    assert_eq!(output.tokens_generated, 2);
    let tokens = seen.tokens();
    assert_eq!(seen.visible(), "Hello there");
    let terminal = tokens.last().expect("a terminal token");
    assert_eq!(terminal.finish_reason.as_deref(), Some("stop"));
    assert!(terminal.token.is_empty());
    assert_eq!(terminal.cumulative_text, "Hello there");

    // What the runtime was given.
    let turn = &fake.turns()[0];
    assert_eq!(
        serde_json::from_str::<Value>(turn.history_json.as_deref().expect("history"))
            .expect("json"),
        json!([{"role": "system", "content": "Be brief."}])
    );
    assert_eq!(
        serde_json::from_str::<Value>(&turn.message_json).expect("json"),
        json!({"role": "user", "content": "script:hi"})
    );
    assert_eq!(turn.thinking, Some(false), "thinking is set, and off");
    assert_eq!(turn.tools_json, None);
    assert_eq!(turn.tool_call_channel, None);
    assert_eq!(turn.sampler.expect("a sampler").top_k, Some(1), "greedy");
    assert_eq!(
        turn.max_output_tokens,
        Some(i32::try_from(GenerationConfig::greedy().max_tokens).expect("fits"))
    );
    assert_eq!(fake.events(), ["wait", "conversation_delete"]);
}

#[test]
fn a_reply_that_used_its_whole_budget_was_cut_off() {
    let fake = fake();
    fake.set_usage(10, 8)
        .script("long", Script::new([content("and so on"), Step::End]));
    let backend = backend(&fake);
    let config = GenerationConfig {
        max_tokens: 8,
        ..GenerationConfig::greedy()
    };

    let output = backend
        .generate(&ask("long"), &config)
        .expect("the turn runs");

    assert_eq!(output.finish_reason, "length");
    assert_eq!(fake.turns()[0].max_output_tokens, Some(8));
}

#[test]
fn a_context_larger_than_the_file_allows_is_capped() {
    let fake = fake();
    fake.set_model(FakeModel {
        max_context_tokens: 4096,
        ..FakeModel::default()
    });
    let mut backend = LiteRtLmBackend::new().expect("a backend");
    backend
        .load(
            &LlmConfig::new(fake.model_path().to_string_lossy().into_owned())
                .with_context_length(32_768),
        )
        .expect("the fake model loads");

    assert_eq!(backend.context_length(), Some(4096));
    let engine = &fake.engines()[0];
    assert_eq!(engine.max_num_tokens, Some(4096));
    assert_eq!(engine.backend, "cpu");
    assert_eq!(
        engine.vision_backend, None,
        "no vision encoder is loaded for a text-only file"
    );
    assert_eq!(engine.activation, None, "CPU keeps the runtime's precision");
}

// ---------------------------------------------------------------------------
// Tool calls
// ---------------------------------------------------------------------------

#[test]
fn a_tool_call_is_reported_once_parsed_and_hidden_from_the_stream() {
    let fake = fake();
    fake.set_usage(40, 12).script(
        "call",
        Script::new([
            content("Checking. "),
            call_text("call:get_temperature{room:"),
            call_text(r#"<|"|>bedroom<|"|>}"#),
            parsed_call("get_temperature", json!({"room": "bedroom"})),
            Step::End,
        ]),
    );
    let backend = backend(&fake);
    let seen = Arc::new(Seen::default());

    let output = backend
        .generate_streaming(&ask("call"), &with_tools(), seen.callback())
        .expect("the turn runs");

    assert_eq!(output.finish_reason, "tool_calls");
    assert_eq!(
        output.text,
        r#"Checking. <|tool_call>call:get_temperature{room:<|"|>bedroom<|"|>}<tool_call|>"#
    );
    assert_eq!(seen.visible(), "Checking. ", "the call block is not shown");
    let tokens = seen.tokens();
    let terminal = tokens.last().expect("a terminal token");
    assert_eq!(terminal.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(terminal.tool_calls.len(), 1);
    assert_eq!(terminal.tool_calls[0].function.name, "get_temperature");
    assert_eq!(
        terminal.tool_calls[0].function.arguments,
        r#"{"room":"bedroom"}"#
    );
    assert_eq!(terminal.raw_text.as_deref(), Some(output.text.as_str()));

    let turn = &fake.turns()[0];
    assert_eq!(turn.tool_call_channel.as_deref(), Some("tool_call"));
    let tools: Value =
        serde_json::from_str(turn.tools_json.as_deref().expect("tools")).expect("json");
    assert_eq!(tools[0]["function"]["name"], "get_temperature");
}

/// The output budget ran out inside the call block. Upstream reports no
/// parsed call and hands the unfinished block over as answer text.
#[test]
fn a_tool_call_cut_off_by_the_budget_is_not_reported_as_a_call() {
    let fake = fake();
    fake.set_usage(40, 6).script(
        "cut",
        Script::new([
            call_text("call:get_temperature{}"),
            content("<|tool_call>call:get_temperature{}"),
            Step::End,
        ]),
    );
    let backend = backend(&fake);
    let seen = Arc::new(Seen::default());
    let config = GenerationConfig {
        max_tokens: 6,
        ..with_tools()
    };

    let output = backend
        .generate_streaming(&ask("cut"), &config, seen.callback())
        .expect("the turn runs");

    assert_eq!(
        output.finish_reason, "length",
        "the turn was cut off; it did not end on a call"
    );
    assert_eq!(output.text, "<|tool_call>call:get_temperature{}");
    assert_eq!(seen.visible(), "");
    let tokens = seen.tokens();
    let terminal = tokens.last().expect("a terminal token");
    assert_eq!(terminal.finish_reason.as_deref(), Some("length"));
    assert!(terminal.tool_calls.is_empty());
}

/// The runtime keeps the first of two equal keys. The caller is told what
/// the runtime parsed, not what xybrid's parser would make of the raw text.
#[test]
fn the_reported_arguments_are_the_runtimes() {
    let fake = fake();
    fake.script(
        "twice",
        Script::new([
            call_text("call:get_temperature{room:1,room:2}"),
            parsed_call("get_temperature", json!({"room": 1})),
            Step::End,
        ]),
    );
    let backend = backend(&fake);

    let output = backend
        .generate(&ask("twice"), &with_tools())
        .expect("the turn runs");

    let calls = parse_tool_calls_with_tools(&output.text, &[weather_tool()]);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments, r#"{"room":1}"#);
}

#[test]
fn a_tool_call_the_runtime_could_not_parse_fails_the_turn_as_an_inference_failure() {
    let fake = fake();
    fake.script(
        "malformed",
        Script::new([
            call_text("call:get_temperature{"),
            Step::error("INVALID_ARGUMENT: Failed to parse tool calls from code block: call:get_temperature{"),
            Step::UntilCancel(
                json!({"role": "assistant", "content": [{"type": "text", "text": "x"}]})
                    .to_string(),
            ),
            Step::cancelled(),
        ]),
    );
    let backend = backend(&fake);

    let err = backend
        .generate(&ask("malformed"), &with_tools())
        .expect_err("the runtime reported an error");

    assert!(
        matches!(err, AdapterError::InferenceFailed(_)),
        "not the caller's input: {err:?}"
    );
    // The runtime was still decoding; it is stopped before the delete.
    assert_eq!(fake.events(), ["cancel", "conversation_delete"]);
}

#[test]
fn tool_results_continue_as_structured_messages() {
    let fake = fake();
    fake.script(
        "answer",
        Script::new([content("It is 17.5 degrees."), Step::End]),
    );
    let backend = backend(&fake);
    assert!(backend.has_tool_continuation());
    let prior = r#"<|tool_call>call:get_temperature{room:<|"|>bedroom<|"|>}<tool_call|>"#;
    // The message sent is the tool result, so it names the fake's script.
    let responses = json!([{
        "call_id": "call_0",
        "content": {"temperature_c": 17.5, "source": "script:answer"},
    }])
    .to_string();
    let turn = ToolContinuationTurn {
        messages: &[ChatMessage::user("How warm is the bedroom?")],
        prior_assistant_text: prior,
        tool_responses_json: &responses,
    };

    let output = backend
        .generate_tool_continuation(&turn, &with_tools(), None)
        .expect("the turn runs");

    assert_eq!(output.text, "It is 17.5 degrees.");
    assert_eq!(output.finish_reason, "stop");
    let sent = fake.turns();
    let turn = sent.last().expect("a turn");
    assert_eq!(
        serde_json::from_str::<Value>(&turn.message_json).expect("json"),
        json!({"role": "tool", "content": [{
            "name": "get_temperature",
            "response": {"temperature_c": 17.5, "source": "script:answer"},
        }]})
    );
    assert_eq!(
        serde_json::from_str::<Value>(turn.history_json.as_deref().expect("history"))
            .expect("json"),
        json!([
            {"role": "user", "content": "How warm is the bedroom?"},
            {"role": "assistant", "tool_calls": [{
                "type": "function",
                "function": {"name": "get_temperature", "arguments": {"room": "bedroom"}},
            }]},
        ])
    );
}

// ---------------------------------------------------------------------------
// Reasoning and stop sequences
// ---------------------------------------------------------------------------

#[test]
fn reasoning_is_returned_apart_from_the_answer() {
    let fake = fake();
    fake.script(
        "think",
        Script::new([
            thought("17 times 23"),
            thought(" is 391."),
            content("391"),
            Step::End,
        ]),
    );
    let backend = backend(&fake);
    let seen = Arc::new(Seen::default());

    let output = backend
        .generate_streaming(&ask("think"), &GenerationConfig::greedy(), seen.callback())
        .expect("the turn runs");

    assert_eq!(output.text, "391");
    assert_eq!(
        output.reasoning_content.as_deref(),
        Some("17 times 23 is 391.")
    );
    assert_eq!(seen.visible(), "391");
}

/// On llama.cpp reasoning is part of the text stop sequences are matched
/// against. A stop the model writes while reasoning ends the turn here too.
#[test]
fn a_stop_sequence_inside_reasoning_ends_the_turn() {
    let fake = fake();
    fake.script(
        "think-stop",
        Script::new([
            thought("First, HA"),
            thought("LT now. More thinking."),
            Step::UntilCancel(
                json!({"role": "assistant", "content": [{"type": "text", "text": "answer"}]})
                    .to_string(),
            ),
            Step::cancelled(),
        ]),
    );
    let backend = backend(&fake);
    let seen = Arc::new(Seen::default());
    let config = GenerationConfig {
        stop_sequences: vec!["HALT".to_string()],
        ..GenerationConfig::greedy()
    };

    let output = backend
        .generate_streaming(&ask("think-stop"), &config, seen.callback())
        .expect("the turn runs");

    assert_eq!(output.text, "");
    assert_eq!(seen.visible(), "", "nothing after the stop is shown");
    assert_eq!(output.reasoning_content.as_deref(), Some("First,"));
    assert_eq!(output.finish_reason, "stop");
    assert_eq!(fake.count("cancel"), 1, "generation was stopped");
}

#[test]
fn a_stop_sequence_in_the_answer_ends_the_turn() {
    let fake = fake();
    fake.script(
        "count",
        Script::new([
            content("1 2 3"),
            content(" 4 5 6"),
            Step::UntilCancel(
                json!({"role": "assistant", "content": [{"type": "text", "text": " 7"}]})
                    .to_string(),
            ),
            Step::cancelled(),
        ]),
    );
    let backend = backend(&fake);
    let config = GenerationConfig {
        stop_sequences: vec![" 6".to_string()],
        ..GenerationConfig::greedy()
    };

    let output = backend
        .generate(&ask("count"), &config)
        .expect("the turn runs");

    assert_eq!(output.text, "1 2 3 4 5");
    assert_eq!(output.finish_reason, "stop");
}

// ---------------------------------------------------------------------------
// Errors, aborts and teardown
// ---------------------------------------------------------------------------

/// The caller's abort is the turn's error even when the runtime then fails
/// to stop: a cloud fallback must still happen.
#[test]
fn a_cloud_fallback_abort_survives_a_teardown_that_fails() {
    let fake = fake();
    fake.script(
        "abort",
        Script::new([content("partial"), Step::WaitForRelease]),
    );
    let backend = backend(&fake).with_timeouts(Duration::from_secs(10), Duration::from_millis(100));
    let abort: StreamingCallback<'_> = Box::new(|_token| {
        Err(Box::new(CloudFallbackAbort::new(AbortReason::StressMemory)) as StreamingError)
    });

    let err = backend
        .generate_streaming(&ask("abort"), &GenerationConfig::greedy(), abort)
        .expect_err("the callback aborted");

    assert!(
        matches!(
            err,
            AdapterError::AbortedForCloudFallback {
                reason: AbortReason::StressMemory
            }
        ),
        "the abort came back as itself: {err:?}"
    );
    assert_eq!(
        fake.events(),
        ["cancel"],
        "the runtime never stopped, so nothing was freed"
    );

    // The engine is retired; the next turn says so instead of entering it.
    let err = backend
        .generate(&ask("abort"), &GenerationConfig::greedy())
        .expect_err("the engine is unusable");
    assert!(matches!(err, AdapterError::InferenceFailed(_)), "{err:?}");
    assert_eq!(fake.events(), ["cancel"]);
    fake.release();
}

#[test]
fn a_callback_error_stops_the_turn_and_comes_back_as_itself() {
    let fake = fake();
    fake.script(
        "talk",
        Script::new([
            Step::UntilCancel(
                json!({"role": "assistant", "content": [{"type": "text", "text": "x"}]})
                    .to_string(),
            ),
            Step::cancelled(),
        ]),
    );
    let backend = backend(&fake);
    let stop: StreamingCallback<'_> = Box::new(|_token| Err("the consumer went away".into()));

    let err = backend
        .generate_streaming(&ask("talk"), &GenerationConfig::greedy(), stop)
        .expect_err("the callback stopped the turn");

    assert!(err.to_string().contains("the consumer went away"), "{err}");
    assert_eq!(fake.events(), ["cancel", "conversation_delete"]);

    // The engine is fine; the next turn runs.
    fake.script("ok", Script::new([content("fine"), Step::End]));
    let output = backend
        .generate(&ask("ok"), &GenerationConfig::greedy())
        .expect("the next turn runs");
    assert_eq!(output.text, "fine");
}

#[test]
fn a_turn_that_goes_quiet_fails_and_retires_the_engine() {
    let fake = fake();
    fake.script("quiet", Script::new([content("one"), Step::WaitForRelease]));
    let backend =
        backend(&fake).with_timeouts(Duration::from_millis(100), Duration::from_millis(100));

    let err = backend
        .generate(&ask("quiet"), &GenerationConfig::greedy())
        .expect_err("the runtime stopped answering");

    match err {
        AdapterError::InferenceFailed(message) => {
            assert!(message.contains("produced no output"), "{message}");
        }
        other => panic!("expected InferenceFailed, got {other:?}"),
    }
    assert_eq!(fake.events(), ["cancel"]);
    fake.release();
}

#[test]
fn an_oversize_prompt_is_the_callers_error_and_the_engine_stays_usable() {
    let fake = fake();
    fake.script(
        "too-long",
        Script::new([Step::error(
            "INVALID_ARGUMENT: Input token ids are too long. Exceeding the maximum number of \
             tokens allowed: 13904 >= 4096",
        )]),
    )
    .script("ok", Script::new([content("fine"), Step::End]));
    let backend = backend(&fake);

    let err = backend
        .generate(&ask("too-long"), &GenerationConfig::greedy())
        .expect_err("the prompt does not fit");
    assert!(matches!(err, AdapterError::InvalidInput(_)), "{err:?}");

    let output = backend
        .generate(&ask("ok"), &GenerationConfig::greedy())
        .expect("the next turn runs");
    assert_eq!(output.text, "fine");
}

#[test]
fn what_the_runtime_cannot_honour_is_refused_before_it_runs_streamed_or_not() {
    let fake = fake();
    let backend = backend(&fake);
    let grammar = GenerationConfig::greedy().with_grammar("root ::= \"a\"");

    let batch = backend
        .generate(&ask("anything"), &grammar)
        .expect_err("grammars are not supported");
    let seen = Arc::new(Seen::default());
    let streamed = backend
        .generate_streaming(&ask("anything"), &grammar, seen.callback())
        .expect_err("grammars are not supported");
    let raw = backend
        .generate_raw("a raw prompt", &GenerationConfig::greedy())
        .expect_err("raw prompts are not supported");

    for err in [batch, streamed, raw] {
        assert!(
            matches!(
                err,
                AdapterError::UnsupportedBackendCapability { ref backend, .. } if backend == BACKEND_NAME
            ),
            "{err:?}"
        );
    }
    assert!(fake.turns().is_empty(), "nothing reached the runtime");
    assert!(seen.tokens().is_empty());
}

// ---------------------------------------------------------------------------
// Re-entry
// ---------------------------------------------------------------------------

/// A token callback may render a prompt on the model that is generating.
#[test]
fn a_token_callback_can_render_a_prompt() {
    let fake = fake();
    fake.script(
        "render",
        Script::new([content("one"), content(" two"), Step::End]),
    );
    let backend = Arc::new(backend(&fake));
    let (done, finished) = mpsc::channel();

    let worker = {
        let backend = Arc::clone(&backend);
        std::thread::spawn(move || {
            let renderer = Arc::clone(&backend);
            let rendered = Arc::new(Mutex::new(Vec::new()));
            let sink = Arc::clone(&rendered);
            let callback: StreamingCallback<'_> = Box::new(move |token| {
                if token.finish_reason.is_none() {
                    let prompt = renderer
                        .render_chat_prompt(
                            &[ChatMessage::user("What next?")],
                            &GenerationConfig::greedy(),
                        )
                        .map_err(|err| Box::new(err) as StreamingError)?;
                    sink.lock().expect("the prompt log").push(prompt);
                }
                Ok(())
            });
            let output =
                backend.generate_streaming(&ask("render"), &GenerationConfig::greedy(), callback);
            let prompts = rendered.lock().expect("the prompt log").clone();
            let _ = done.send((output.map(|o| o.text), prompts));
        })
    };

    let (text, prompts) = finished
        .recv_timeout(Duration::from_secs(20))
        .expect("the turn finished: rendering from a callback must not block it");
    worker.join().expect("the worker ends");
    assert_eq!(text.expect("the turn runs"), "one two");
    assert_eq!(prompts.len(), 2);
    assert!(prompts[0].starts_with(GEMMA_PROMPT_PREFIX));
    assert!(prompts[0].contains("What next?"));
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

fn image_message(text: &str) -> MultimodalChatMessage {
    let source = ImageSource::Encoded {
        bytes: Arc::from(b"\x89PNG\r\n\x1a\nnot-a-real-image".to_vec()),
        format: ImageFormat::Png,
        dimensions: ImageDimensions {
            width: 1,
            height: 1,
        },
    };
    MultimodalChatMessage {
        role: MessageRole::User,
        parts: vec![
            MultimodalMessagePart::Image(MultimodalImagePart::new(source, "img-0")),
            MultimodalMessagePart::Text(text.to_string()),
        ],
    }
}

#[test]
fn an_image_goes_to_a_model_with_a_vision_encoder() {
    let fake = fake();
    fake.set_model(FakeModel {
        vision: true,
        ..FakeModel::default()
    })
    .script("see", Script::new([content("A red circle."), Step::End]));
    let mut backend = LiteRtLmBackend::new().expect("a backend");
    backend
        .load(
            &LlmConfig::new(fake.model_path().to_string_lossy().into_owned())
                .with_max_image_tokens(280),
        )
        .expect("the fake model loads");
    assert!(backend.supports_vision());

    let output = backend
        .generate_multimodal(
            &[
                MultimodalChatMessage {
                    role: MessageRole::System,
                    parts: vec![MultimodalMessagePart::Text("Answer in French.".into())],
                },
                image_message("script:see What is this?"),
            ],
            &GenerationConfig::greedy(),
        )
        .expect("the turn runs");

    assert_eq!(output.text, "A red circle.");
    assert_eq!(fake.engines()[0].vision_backend.as_deref(), Some("cpu"));
    let turn = &fake.turns()[0];
    assert_eq!(turn.visual_token_budget, Some(280));
    let sent: Value = serde_json::from_str(&turn.message_json).expect("json");
    assert_eq!(sent["content"][0]["type"], "image");
    assert_eq!(sent["content"][1]["text"], "script:see What is this?");
    assert_eq!(
        serde_json::from_str::<Value>(turn.history_json.as_deref().expect("history"))
            .expect("json"),
        json!([{"role": "system", "content": [{"type": "text", "text": "Answer in French."}]}])
    );
}

#[test]
fn an_image_for_a_file_without_a_vision_encoder_is_refused() {
    let fake = fake();
    let backend = backend(&fake);
    assert!(!backend.supports_vision());

    let err = backend
        .generate_multimodal(
            &[image_message("What is this?")],
            &GenerationConfig::greedy(),
        )
        .expect_err("the file has no vision encoder");

    assert!(
        matches!(err, AdapterError::UnsupportedModelCapability { .. }),
        "{err:?}"
    );
    assert!(fake.turns().is_empty(), "nothing reached the runtime");
}
