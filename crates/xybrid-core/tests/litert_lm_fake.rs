//! `.litertlm` models through the executor, on a scripted LiteRT-LM runtime.
//!
//! No model and no native library: `FakeRuntime::at` puts a fake behind the
//! model path the executor derives, and its record of what the runtime was
//! given is what these tests read. They cover what the executor decides
//! before the backend runs: which messages a request becomes, and which
//! requests it refuses. `litert_lm_gemma4.rs` covers the real model.

#![cfg(all(feature = "llm-litert", feature = "test-util"))]

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use xybrid_core::conversation::ConversationContext;
use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::gateway::Tool;
use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole, ToolCallResult};
use xybrid_core::runtime_adapter::litert_lm::testing::{FakeModel, FakeRuntime, Script, Step};
use xybrid_core::runtime_adapter::{
    AdapterError, GenerationConfig, PartialToken, StreamingCallback,
};

const MODEL_FILE: &str = "model.litertlm";
/// What Gemma 4's template opens a turn with; the backend reads the tool
/// protocol off the rendered prompt.
const GEMMA_PROMPT_PREFIX: &str = "<|turn>user\n";

/// A model directory whose `.litertlm` file is a scripted runtime.
struct Model {
    fake: FakeRuntime,
    metadata: ModelMetadata,
    executor: TemplateExecutor,
    _dir: tempfile::TempDir,
}

fn model(fake_model: FakeModel) -> Model {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let model_path = dir.path().join(MODEL_FILE);
    std::fs::write(&model_path, b"not a model: a scripted runtime stands in").expect("a file");
    let fake = FakeRuntime::at(&model_path);
    fake.set_model(fake_model)
        .set_render_prefix(GEMMA_PROMPT_PREFIX)
        .set_usage(24, 6);
    let metadata: ModelMetadata = serde_json::from_value(json!({
        "model_id": "scripted-litert",
        "version": "1.0",
        "execution_template": {
            "type": "LiteRtLm",
            "model_file": MODEL_FILE,
            "context_length": 4096,
        },
        "preprocessing": [],
        "postprocessing": [],
        "files": [MODEL_FILE],
        "metadata": { "task": "vlm" },
    }))
    .expect("valid metadata");
    metadata.validate().expect("the metadata validates");
    let executor = TemplateExecutor::with_base_path(
        dir.path()
            .to_str()
            .expect("the temporary directory is valid UTF-8"),
    );
    Model {
        fake,
        metadata,
        executor,
        _dir: dir,
    }
}

fn vision_model() -> Model {
    model(FakeModel {
        vision: true,
        ..FakeModel::default()
    })
}

fn content(text: &str) -> Step {
    Step::text(
        json!({"role": "assistant", "content": [{"type": "text", "text": text}]}).to_string(),
    )
}

fn text(text: &str) -> Envelope {
    Envelope::new(EnvelopeKind::Text(text.to_string()))
}

fn text_of(envelope: &Envelope) -> &str {
    match &envelope.kind {
        EnvelopeKind::Text(text) => text,
        other => panic!("expected text output, got {other:?}"),
    }
}

fn png() -> Vec<u8> {
    let image = image::RgbImage::from_pixel(16, 16, image::Rgb([220, 20, 20]));
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("the test image encodes");
    encoded.into_inner()
}

fn with_image(text: &str) -> Envelope {
    let image = Envelope::image(png(), "png").expect("a valid image");
    Envelope::user_message(text, vec![image]).expect("a multipart input")
}

fn temperature_tools() -> Vec<Tool> {
    vec![Tool::function(
        "get_temperature",
        "Get the current temperature in a room, in Celsius.",
        json!({
            "type": "object",
            "properties": { "room": { "type": "string" } },
            "required": ["room"],
        }),
    )]
}

fn collector() -> (Arc<Mutex<Vec<PartialToken>>>, StreamingCallback<'static>) {
    let collected = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&collected);
    let callback = Box::new(move |token: PartialToken| {
        sink.lock().expect("the collector lock").push(token);
        Ok(())
    });
    (collected, callback)
}

fn history(fake: &FakeRuntime, turn: usize) -> Value {
    let turns = fake.turns();
    serde_json::from_str(turns[turn].history_json.as_deref().expect("history")).expect("json")
}

fn sent(fake: &FakeRuntime, turn: usize) -> Value {
    serde_json::from_str(&fake.turns()[turn].message_json).expect("json")
}

const CALL: &str = r#"<|tool_call>call:get_temperature{room:<|"|>bedroom<|"|>}<tool_call|>"#;

/// Tool results that name the fake's script, since they are the message a
/// continuation sends.
fn results(script: &str) -> Vec<ToolCallResult> {
    vec![ToolCallResult {
        call_id: "call_0".to_string(),
        name: "get_temperature".to_string(),
        content: json!({"temperature_c": 17.5, "source": format!("script:{script}")}),
    }]
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

#[test]
fn a_text_request_runs_on_the_litert_backend() {
    let mut model = model(FakeModel::default());
    model
        .fake
        .script("hi", Script::new([content("Hello."), Step::End]));
    let mut input = text("script:hi Say hello.");
    input
        .metadata
        .insert("system_prompt".to_string(), "Be brief.".to_string());

    let output = model
        .executor
        .execute(&model.metadata, &input, Some(&GenerationConfig::greedy()))
        .expect("the request runs");

    assert_eq!(text_of(&output), "Hello.");
    assert_eq!(
        output.metadata.get("finish_reason").map(String::as_str),
        Some("stop")
    );
    assert_eq!(
        history(&model.fake, 0),
        json!([{"role": "system", "content": "Be brief."}])
    );
    assert_eq!(
        sent(&model.fake, 0),
        json!({"role": "user", "content": "script:hi Say hello."})
    );
    assert_eq!(model.fake.engines()[0].max_num_tokens, Some(4096));
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

/// A system prompt rides on the request envelope. Attaching an image must
/// not drop it.
#[test]
fn an_image_request_keeps_its_system_prompt() {
    let mut model = vision_model();
    model
        .fake
        .script("see", Script::new([content("Un cercle rouge."), Step::End]));
    let mut input = with_image("script:see What is this?");
    input.metadata.insert(
        "system_prompt".to_string(),
        "Answer only in French.".to_string(),
    );

    let output = model
        .executor
        .execute(&model.metadata, &input, Some(&GenerationConfig::greedy()))
        .expect("the request runs");

    assert_eq!(text_of(&output), "Un cercle rouge.");
    assert_eq!(
        history(&model.fake, 0),
        json!([{"role": "system", "content": [{"type": "text", "text": "Answer only in French."}]}])
    );
    let message = sent(&model.fake, 0);
    assert_eq!(message["role"], "user");
    assert!(
        message["content"]
            .as_array()
            .expect("parts")
            .iter()
            .any(|part| part["type"] == "image"),
        "the image is sent: {message}"
    );
}

#[test]
fn a_streamed_image_request_keeps_its_system_prompt() {
    let mut model = vision_model();
    model
        .fake
        .script("see", Script::new([content("Un cercle rouge."), Step::End]));
    let mut input = with_image("script:see What is this?");
    input.metadata.insert(
        "system_prompt".to_string(),
        "Answer only in French.".to_string(),
    );
    let (tokens, on_token) = collector();

    let output = model
        .executor
        .execute_streaming(
            &model.metadata,
            &input,
            on_token,
            Some(&GenerationConfig::greedy()),
        )
        .expect("the request runs");

    assert_eq!(text_of(&output), "Un cercle rouge.");
    assert_eq!(
        history(&model.fake, 0),
        json!([{"role": "system", "content": [{"type": "text", "text": "Answer only in French."}]}])
    );
    let emitted: String = tokens
        .lock()
        .expect("the collector lock")
        .iter()
        .map(|token| token.token.as_str())
        .collect();
    assert_eq!(emitted, "Un cercle rouge.");
}

#[test]
fn an_image_request_without_a_system_prompt_sends_no_system_message() {
    let mut model = vision_model();
    model
        .fake
        .script("see", Script::new([content("A red square."), Step::End]));

    model
        .executor
        .execute(
            &model.metadata,
            &with_image("script:see What is this?"),
            Some(&GenerationConfig::greedy()),
        )
        .expect("the request runs");

    assert_eq!(model.fake.turns()[0].history_json, None);
}

#[test]
fn an_image_for_a_file_without_a_vision_encoder_is_refused() {
    let mut model = model(FakeModel::default());

    let err = model
        .executor
        .execute(
            &model.metadata,
            &with_image("What is this?"),
            Some(&GenerationConfig::greedy()),
        )
        .expect_err("the file has no vision encoder");

    assert!(
        matches!(err, AdapterError::UnsupportedModelCapability { .. }),
        "{err:?}"
    );
    assert!(model.fake.turns().is_empty(), "nothing reached the runtime");
}

// ---------------------------------------------------------------------------
// Tool results
// ---------------------------------------------------------------------------

#[test]
fn tool_results_continue_a_text_conversation() {
    let mut model = model(FakeModel::default());
    model.fake.script(
        "answer",
        Script::new([content("It is 17.5 degrees."), Step::End]),
    );
    let mut context = ConversationContext::new()
        .with_system(text("You control a house.").with_role(MessageRole::System));
    context.push(text("Hello.").with_role(MessageRole::User));
    context.push(text("Hello! How can I help?").with_role(MessageRole::Assistant));
    let continuation = Envelope::tool_results("How warm is the bedroom?", CALL, &results("answer"));
    let config = GenerationConfig::greedy().with_tools(temperature_tools());

    let output = model
        .executor
        .execute_with_context(&model.metadata, &continuation, &context, Some(&config))
        .expect("the continuation runs");

    assert_eq!(text_of(&output), "It is 17.5 degrees.");
    let message = sent(&model.fake, 0);
    assert_eq!(message["role"], "tool");
    assert_eq!(message["content"][0]["name"], "get_temperature");
    assert_eq!(message["content"][0]["response"]["temperature_c"], 17.5);
    let history = history(&model.fake, 0);
    let roles: Vec<&str> = history
        .as_array()
        .expect("messages")
        .iter()
        .map(|message| message["role"].as_str().expect("a role"))
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "user", "assistant"]);
    assert_eq!(
        history[4]["tool_calls"][0]["function"],
        json!({"name": "get_temperature", "arguments": {"room": "bedroom"}})
    );
}

/// An image earlier in the conversation sends the turn down the multimodal
/// path, which takes messages only. Tool results cannot ride it, and running
/// the turn without them would answer the original question again.
#[test]
fn tool_results_after_an_image_are_refused_not_dropped() {
    let mut model = vision_model();
    model
        .fake
        .script("answer", Script::new([content("ungrounded"), Step::End]));
    let mut context = ConversationContext::new();
    context.push(with_image("Remember this picture."));
    context.push(text("I will.").with_role(MessageRole::Assistant));
    let continuation = Envelope::tool_results("How warm is the bedroom?", CALL, &results("answer"));
    let config = GenerationConfig::greedy().with_tools(temperature_tools());

    let batch = model
        .executor
        .execute_with_context(&model.metadata, &continuation, &context, Some(&config))
        .expect_err("tool results cannot follow an image");
    let (tokens, on_token) = collector();
    let streamed = model
        .executor
        .execute_streaming_with_context(
            &model.metadata,
            &continuation,
            &context,
            on_token,
            Some(&config),
        )
        .expect_err("tool results cannot follow an image");

    for err in [batch, streamed] {
        match err {
            AdapterError::InvalidInput(message) => {
                assert!(message.contains("text-only conversation"), "{message}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }
    assert!(
        model.fake.turns().is_empty(),
        "the turn did not run without its tool results"
    );
    assert!(tokens.lock().expect("the collector lock").is_empty());
}

/// The same conversation without tool results still runs: the refusal is
/// about the combination, not about images in history.
#[test]
fn a_question_after_an_image_still_runs() {
    let mut model = vision_model();
    model
        .fake
        .script("colour", Script::new([content("Red."), Step::End]));
    let mut context = ConversationContext::new();
    context.push(with_image("Remember this picture."));
    context.push(text("I will.").with_role(MessageRole::Assistant));

    let output = model
        .executor
        .execute_with_context(
            &model.metadata,
            &text("script:colour What colour was it?"),
            &context,
            Some(&GenerationConfig::greedy()),
        )
        .expect("the request runs");

    assert_eq!(text_of(&output), "Red.");
    let history = history(&model.fake, 0);
    assert!(
        history[0]["content"]
            .as_array()
            .expect("parts")
            .iter()
            .any(|part| part["type"] == "image"),
        "the earlier image is replayed: {history}"
    );
}
