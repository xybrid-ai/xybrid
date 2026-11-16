//! Gemma 4 E2B on the LiteRT-LM backend: text, image input and tool calling,
//! through the same executor calls an application makes.
//!
//! The tests that run the model need two things this repository does not
//! contain, and skip (passing, with a message) without them:
//!
//! - the model: `./integration-tests/download.sh gemma-4-e2b-litert`
//!   (about 2.6 GB), or `XYBRID_TEST_MODELS` pointing at a directory that
//!   holds `gemma-4-e2b-litert/`;
//! - the LiteRT-LM runtime library for the host, named by
//!   `XYBRID_LITERT_LM_LIBRARY` or findable by the platform loader.
//!
//! Set `XYBRID_REQUIRE_MODELS` to turn a skip into a failure, for a lane
//! that claims to have run them.
//!
//! They share one loaded model, so they run one at a time whatever the
//! harness's thread count.

#![cfg(feature = "llm-litert")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use xybrid_core::conversation::ConversationContext;
use xybrid_core::execution::{ExecutionTemplate, ModelMetadata, TemplateExecutor};
use xybrid_core::gateway::{Tool, ToolCall};
use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole, ToolCallResult};
use xybrid_core::runtime_adapter::tool_call::{has_tool_markers, strip_tool_calls};
use xybrid_core::runtime_adapter::{AdapterError, GenerationConfig, PartialToken};
use xybrid_core::testing::model_fixtures;

const MODEL_ID: &str = "gemma-4-e2b-litert";
const MODEL_FILE: &str = "gemma-4-E2B-it.litertlm";
/// What the fake sensor reports. Distinctive, so an answer that contains it
/// can only have come from the tool result.
const BEDROOM_TEMP_C: f64 = 17.5;

fn load_metadata(model_dir: &Path) -> ModelMetadata {
    let metadata_path = model_dir.join("model_metadata.json");
    let content = std::fs::read_to_string(&metadata_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", metadata_path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", metadata_path.display()))
}

/// The loaded model, shared by every test that runs it.
struct Loaded {
    metadata: ModelMetadata,
    executor: TemplateExecutor,
}

fn skip_or_fail(reason: &str) -> Option<PathBuf> {
    if model_fixtures::models_required() {
        panic!("{reason} (XYBRID_REQUIRE_MODELS is set)");
    }
    eprintln!("SKIP: {reason}");
    None
}

/// The model directory, when both the model and the runtime are present.
fn model_dir() -> Option<PathBuf> {
    let Some(dir) = model_fixtures::model_path(MODEL_ID) else {
        return skip_or_fail("the gemma-4-e2b-litert fixture directory was not found");
    };
    if !dir.join(MODEL_FILE).exists() {
        return skip_or_fail(&format!(
            "{MODEL_FILE} is not downloaded; run ./integration-tests/download.sh {MODEL_ID}"
        ));
    }
    if let Err(err) = xybrid_core::runtime_adapter::litert_lm::runtime_available() {
        return skip_or_fail(&format!("the LiteRT-LM runtime is not available: {err}"));
    }
    Some(dir)
}

fn loaded() -> Option<MutexGuard<'static, Loaded>> {
    static LOADED: OnceLock<Option<Mutex<Loaded>>> = OnceLock::new();
    LOADED
        .get_or_init(|| {
            let dir = model_dir()?;
            Some(Mutex::new(Loaded {
                metadata: load_metadata(&dir),
                executor: TemplateExecutor::with_base_path(
                    dir.to_str().expect("the model directory is valid UTF-8"),
                ),
            }))
        })
        .as_ref()
        .map(|loaded| {
            loaded
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        })
}

fn text_of(envelope: &Envelope) -> &str {
    match &envelope.kind {
        EnvelopeKind::Text(text) => text,
        other => panic!("expected text output, got {other:?}"),
    }
}

fn finish_reason(envelope: &Envelope) -> &str {
    envelope
        .metadata
        .get("finish_reason")
        .map(String::as_str)
        .expect("an LLM response carries finish_reason")
}

fn tool_calls(envelope: &Envelope) -> Vec<ToolCall> {
    envelope
        .metadata
        .get(Envelope::TOOL_CALLS_METADATA_KEY)
        .map(|raw| serde_json::from_str(raw).expect("tool_calls metadata is valid JSON"))
        .unwrap_or_default()
}

fn temperature_tools() -> Vec<Tool> {
    vec![Tool::function(
        "get_temperature",
        "Get the current temperature in a room, in Celsius.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "room": { "type": "string", "description": "Name of the room." }
            },
            "required": ["room"]
        }),
    )]
}

fn run_tool(call: &ToolCall) -> ToolCallResult {
    let arguments: serde_json::Value =
        serde_json::from_str(&call.function.arguments).expect("arguments are a JSON object");
    ToolCallResult {
        call_id: call.id.clone(),
        name: call.function.name.clone(),
        content: serde_json::json!({
            "room": arguments.get("room").cloned().unwrap_or_else(|| "bedroom".into()),
            "temperature_c": BEDROOM_TEMP_C,
        }),
    }
}

/// A 256x256 PNG: a filled red circle on a white background.
fn red_circle_png() -> Vec<u8> {
    let image = image::RgbImage::from_fn(256, 256, |x, y| {
        let (dx, dy) = (i64::from(x) - 128, i64::from(y) - 128);
        if dx * dx + dy * dy <= 80 * 80 {
            image::Rgb([220, 20, 20])
        } else {
            image::Rgb([255, 255, 255])
        }
    });
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("the test image encodes");
    encoded.into_inner()
}

/// A 256x256 PNG of one flat colour.
fn solid_png(rgb: [u8; 3]) -> Vec<u8> {
    let image = image::RgbImage::from_pixel(256, 256, image::Rgb(rgb));
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("the test image encodes");
    encoded.into_inner()
}

/// Collects what a streaming callback receives.
#[derive(Default)]
struct Collected {
    tokens: Vec<PartialToken>,
}

impl Collected {
    fn emitted(&self) -> String {
        self.tokens.iter().map(|t| t.token.as_str()).collect()
    }

    fn terminal(&self) -> &PartialToken {
        let last = self.tokens.last().expect("the stream produced tokens");
        assert!(last.is_final(), "the last token is the terminal one");
        last
    }
}

fn collector() -> (
    Arc<Mutex<Collected>>,
    xybrid_core::runtime_adapter::StreamingCallback<'static>,
) {
    let collected = Arc::new(Mutex::new(Collected::default()));
    let sink = Arc::clone(&collected);
    let callback = Box::new(move |token: PartialToken| {
        sink.lock().expect("the collector lock").tokens.push(token);
        Ok(())
    });
    (collected, callback)
}

#[test]
fn fixture_metadata_selects_the_litert_lm_format() {
    let dir = model_fixtures::model_path(MODEL_ID)
        .unwrap_or_else(|| panic!("the {MODEL_ID} fixture metadata directory must exist"));
    let metadata = load_metadata(&dir);
    metadata.validate().expect("the metadata validates");
    assert_eq!(metadata.model_id, MODEL_ID);
    match &metadata.execution_template {
        ExecutionTemplate::LiteRtLm {
            model_file,
            context_length,
        } => {
            assert_eq!(model_file, MODEL_FILE);
            assert_eq!(context_length.map(|n| n.get()), Some(4096));
        }
        other => panic!("expected a LiteRtLm execution template, got {other:?}"),
    }
    // A .litertlm file carries its own vision encoder.
    assert!(metadata.vision_encoder.is_none());
}

#[test]
fn text_chat_answers() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let input = Envelope::new(EnvelopeKind::Text(
        "Reply with exactly these two words: hello world".to_string(),
    ));
    let config = GenerationConfig::greedy().with_max_tokens(24);

    let output = executor
        .execute(metadata, &input, Some(&config))
        .expect("text chat succeeds");

    let text = text_of(&output);
    assert!(
        text.to_lowercase().contains("hello world"),
        "unexpected answer: {text:?}"
    );
    assert_eq!(finish_reason(&output), "stop");
    let tokens: usize = output.metadata["tokens_generated"]
        .parse()
        .expect("tokens_generated is a number");
    assert!((1..=24).contains(&tokens), "tokens_generated = {tokens}");
}

#[test]
fn text_chat_is_repeatable_under_greedy_decoding() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let input = Envelope::new(EnvelopeKind::Text(
        "Name three primary colours, separated by commas.".to_string(),
    ));
    let config = GenerationConfig::greedy().with_max_tokens(24);
    let first = executor
        .execute(metadata, &input, Some(&config))
        .expect("first run succeeds");
    let second = executor
        .execute(metadata, &input, Some(&config))
        .expect("second run succeeds");
    assert_eq!(text_of(&first), text_of(&second));
}

#[test]
fn text_chat_streams_what_it_returns() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let input = Envelope::new(EnvelopeKind::Text(
        "Count from one to five in words, separated by spaces.".to_string(),
    ));
    let config = GenerationConfig::greedy().with_max_tokens(32);
    let (collected, callback) = collector();

    let output = executor
        .execute_streaming(metadata, &input, callback, Some(&config))
        .expect("streaming chat succeeds");

    let collected = collected.lock().expect("the collector lock");
    assert!(
        collected.tokens.len() > 2,
        "the answer arrived in several tokens, not one block"
    );
    let terminal = collected.terminal();
    assert_eq!(terminal.finish_reason.as_deref(), Some("stop"));
    assert!(terminal.tool_calls.is_empty());
    assert_eq!(collected.emitted().trim(), text_of(&output));
    assert_eq!(terminal.cumulative_text, text_of(&output));
    assert!(text_of(&output).to_lowercase().contains("three"));
}

#[test]
fn the_output_budget_ends_a_turn_as_length() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let input = Envelope::new(EnvelopeKind::Text(
        "Count from 1 to 50, separated by spaces.".to_string(),
    ));
    let config = GenerationConfig::greedy().with_max_tokens(8);
    let output = executor
        .execute(metadata, &input, Some(&config))
        .expect("a budgeted run succeeds");
    assert_eq!(finish_reason(&output), "length");
    assert_eq!(output.metadata["tokens_generated"], "8");
}

#[test]
fn a_stop_sequence_ends_the_turn_and_is_not_returned() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let input = Envelope::new(EnvelopeKind::Text(
        "Count from 1 to 20, separated by spaces.".to_string(),
    ));
    let config = GenerationConfig::greedy()
        .with_max_tokens(64)
        .with_stop(" 6");
    let (collected, callback) = collector();
    let output = executor
        .execute_streaming(metadata, &input, callback, Some(&config))
        .expect("a run with a stop sequence succeeds");

    let text = text_of(&output);
    assert!(
        text.contains('5'),
        "the text before the stop is kept: {text:?}"
    );
    assert!(
        !text.contains('6'),
        "the stop sequence is not returned: {text:?}"
    );
    assert_eq!(finish_reason(&output), "stop");
    let collected = collected.lock().expect("the collector lock");
    assert!(!collected.emitted().contains('6'));
}

#[test]
fn a_system_prompt_and_earlier_turns_reach_the_model() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let mut context = ConversationContext::new().with_system(
        Envelope::new(EnvelopeKind::Text(
            "You are a terse assistant. The user's dog is called Biscuit.".to_string(),
        ))
        .with_role(MessageRole::System),
    );
    context.push(
        Envelope::new(EnvelopeKind::Text("My favourite number is 41.".to_string()))
            .with_role(MessageRole::User),
    );
    context.push(
        Envelope::new(EnvelopeKind::Text("Noted.".to_string())).with_role(MessageRole::Assistant),
    );
    let input = Envelope::new(EnvelopeKind::Text(
        "What is my dog's name, and what is my favourite number?".to_string(),
    ))
    .with_role(MessageRole::User);
    let config = GenerationConfig::greedy().with_max_tokens(32);

    let output = executor
        .execute_with_context(metadata, &input, &context, Some(&config))
        .expect("a turn with context succeeds");

    let text = text_of(&output);
    assert!(
        text.contains("Biscuit"),
        "the system prompt was used: {text:?}"
    );
    assert!(text.contains("41"), "the earlier turn was used: {text:?}");
}

#[test]
fn an_image_is_described() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let image = Envelope::image(red_circle_png(), "png").expect("a valid image");
    let input = Envelope::user_message(
        "What shape is in this image, and what colour is it? Answer in one short sentence.",
        vec![image],
    )
    .expect("a multipart input");
    let config = GenerationConfig::greedy().with_max_tokens(32);

    let output = executor
        .execute(metadata, &input, Some(&config))
        .expect("image input succeeds");

    let text = text_of(&output).to_lowercase();
    assert!(text.contains("red"), "the colour was seen: {text:?}");
    assert!(text.contains("circle"), "the shape was seen: {text:?}");
}

#[test]
fn an_image_is_described_while_streaming() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let image = Envelope::image(red_circle_png(), "png").expect("a valid image");
    let input = Envelope::user_message("What colour is the shape? One word.", vec![image])
        .expect("a multipart input");
    let config = GenerationConfig::greedy().with_max_tokens(16);
    let (collected, callback) = collector();

    let output = executor
        .execute_streaming(metadata, &input, callback, Some(&config))
        .expect("streaming image input succeeds");

    assert!(text_of(&output).to_lowercase().contains("red"));
    let collected = collected.lock().expect("the collector lock");
    assert_eq!(collected.terminal().finish_reason.as_deref(), Some("stop"));
    assert_eq!(collected.emitted().trim(), text_of(&output));
}

#[test]
fn an_earlier_image_stays_visible_in_a_later_turn() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let image = Envelope::image(red_circle_png(), "png").expect("a valid image");
    let mut context = ConversationContext::new();
    context.push(Envelope::user_message("Remember this picture.", vec![image]).expect("a turn"));
    context.push(
        Envelope::new(EnvelopeKind::Text("I will remember it.".to_string()))
            .with_role(MessageRole::Assistant),
    );
    let input = Envelope::new(EnvelopeKind::Text(
        "What colour was the shape in the picture I showed you? One word.".to_string(),
    ))
    .with_role(MessageRole::User);
    let config = GenerationConfig::greedy().with_max_tokens(16);

    let output = executor
        .execute_with_context(metadata, &input, &context, Some(&config))
        .expect("a text turn after an image turn succeeds");

    assert!(
        text_of(&output).to_lowercase().contains("red"),
        "the earlier image was used: {:?}",
        text_of(&output)
    );
}

#[test]
fn two_images_in_one_message_are_both_seen() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let first = Envelope::image(red_circle_png(), "png").expect("a valid image");
    let second = Envelope::image(solid_png([20, 20, 220]), "png").expect("a valid image");
    let input = Envelope::user_message(
        "Name the main colour of the shape in the first picture and the colour of the second \
         picture, in that order.",
        vec![first, second],
    )
    .expect("a multipart input");
    let config = GenerationConfig::greedy().with_max_tokens(48);

    let output = executor
        .execute(metadata, &input, Some(&config))
        .expect("two images in one message succeed");

    let text = text_of(&output).to_lowercase();
    assert!(text.contains("red"), "the first image was seen: {text:?}");
    assert!(text.contains("blue"), "the second image was seen: {text:?}");
}

#[test]
fn a_tool_is_called_and_its_result_is_used() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let question =
        "Use the get_temperature tool to check the bedroom temperature, then tell me what it is.";
    let config = GenerationConfig::greedy()
        .with_max_tokens(96)
        .with_tools(temperature_tools());

    // Turn 1: the model asks for the tool.
    let first = executor
        .execute(
            metadata,
            &Envelope::new(EnvelopeKind::Text(question.to_string())),
            Some(&config),
        )
        .expect("the tool-offering turn succeeds");
    let calls = tool_calls(&first);
    assert_eq!(
        calls.len(),
        1,
        "one call was made; text: {:?}",
        text_of(&first)
    );
    assert_eq!(calls[0].function.name, "get_temperature");
    let arguments: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("arguments are JSON");
    assert_eq!(arguments["room"], "bedroom");
    assert_eq!(finish_reason(&first), "tool_calls");
    // The text is the model's own, call block included, as with llama.cpp.
    assert!(has_tool_markers(text_of(&first)));

    // Turn 2: the results go back as data; the answer uses them.
    let results: Vec<ToolCallResult> = calls.iter().map(run_tool).collect();
    let continuation = Envelope::tool_results(question, text_of(&first), &results);
    let second = executor
        .execute(metadata, &continuation, Some(&config))
        .expect("the continuation turn succeeds");
    assert!(tool_calls(&second).is_empty(), "no further call was needed");
    let answer = strip_tool_calls(text_of(&second));
    assert!(
        answer.contains("17.5"),
        "the answer uses the tool result: {answer:?}"
    );
    assert_eq!(finish_reason(&second), "stop");
}

#[test]
fn a_streamed_tool_call_is_hidden_from_the_text_and_carried_on_the_last_token() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let question =
        "Use the get_temperature tool to check the bedroom temperature, then tell me what it is.";
    let config = GenerationConfig::greedy()
        .with_max_tokens(96)
        .with_tools(temperature_tools());

    let (collected, callback) = collector();
    let first = executor
        .execute_streaming(
            metadata,
            &Envelope::new(EnvelopeKind::Text(question.to_string())),
            callback,
            Some(&config),
        )
        .expect("the streamed tool-offering turn succeeds");
    let (calls, raw_text) = {
        let collected = collected.lock().expect("the collector lock");
        assert!(
            !has_tool_markers(&collected.emitted()),
            "no protocol text reached the stream: {:?}",
            collected.emitted()
        );
        let terminal = collected.terminal();
        assert_eq!(terminal.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(terminal.tool_calls.len(), 1);
        (
            terminal.tool_calls.clone(),
            terminal
                .raw_text
                .clone()
                .expect("the raw turn text rides the terminal token"),
        )
    };
    // The stream and the envelope agree.
    assert_eq!(raw_text, text_of(&first));
    assert_eq!(tool_calls(&first).len(), 1);

    let results: Vec<ToolCallResult> = calls.iter().map(run_tool).collect();
    let continuation = Envelope::tool_results(question, raw_text, &results);
    let (collected, callback) = collector();
    let second = executor
        .execute_streaming(metadata, &continuation, callback, Some(&config))
        .expect("the streamed continuation succeeds");
    let collected = collected.lock().expect("the collector lock");
    assert!(
        collected.emitted().contains("17.5"),
        "{:?}",
        collected.emitted()
    );
    assert_eq!(collected.emitted().trim(), text_of(&second));
    assert_eq!(collected.terminal().finish_reason.as_deref(), Some("stop"));
}

#[test]
fn an_oversize_prompt_is_refused_and_the_model_stays_usable() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    // About 14,000 tokens against a 4,096-token window.
    let long: String = (0..3000).map(|i| format!("w{i} ")).collect();
    let err = executor
        .execute(
            metadata,
            &Envelope::new(EnvelopeKind::Text(long)),
            Some(&GenerationConfig::greedy().with_max_tokens(8)),
        )
        .expect_err("a prompt larger than the context window is refused");
    match &err {
        AdapterError::InvalidInput(message) => {
            assert!(message.starts_with("Input too long"), "{message}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    let output = executor
        .execute(
            metadata,
            &Envelope::new(EnvelopeKind::Text(
                "Reply with exactly these two words: hello world".to_string(),
            )),
            Some(&GenerationConfig::greedy().with_max_tokens(16)),
        )
        .expect("the model still answers");
    assert!(text_of(&output).to_lowercase().contains("hello world"));
}

#[test]
fn a_reply_that_fills_the_context_window_ends_as_length() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    // About 3,900 prompt tokens, which leaves about 200 of the 4,096-token
    // window: far less than the budget below.
    let filler: String = (0..1000).map(|i| format!("w{i} ")).collect();
    let input = Envelope::new(EnvelopeKind::Text(format!(
        "{filler}\nNow count from 1 to 3000, separated by spaces."
    )));
    let config = GenerationConfig::greedy().with_max_tokens(1000);

    let output = executor
        .execute(metadata, &input, Some(&config))
        .expect("a run that fills the window succeeds");

    let tokens: usize = output.metadata["tokens_generated"]
        .parse()
        .expect("tokens_generated is a number");
    assert!(
        (1..1000).contains(&tokens),
        "the window, not the budget, ended the turn: {tokens} tokens"
    );
    assert_eq!(finish_reason(&output), "length");
}

#[test]
fn the_repetition_penalty_reaches_the_runtime() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let input = Envelope::new(EnvelopeKind::Text(
        "Write the word cat forty times, separated by single spaces, and nothing else.".to_string(),
    ));
    let mut run = |penalty: f32| {
        let config = GenerationConfig {
            repetition_penalty: penalty,
            ..GenerationConfig::greedy().with_max_tokens(48)
        };
        let output = executor
            .execute(metadata, &input, Some(&config))
            .expect("the run succeeds");
        text_of(&output).matches("cat").count()
    };

    let plain = run(1.0);
    let penalized = run(1.8);

    assert!(
        plain >= 30,
        "without a penalty the word is repeated: {plain}"
    );
    assert!(
        penalized < plain,
        "a strong penalty cuts the repetition short: {penalized} against {plain}"
    );
}

/// Thinking is switched by the model's `reasoning` metadata. The thought is
/// reported beside the answer, never inside it.
#[test]
fn thinking_is_captured_as_reasoning_and_kept_out_of_the_answer() {
    // Held for the whole test: one generation at a time on the host.
    let Some(model) = loaded() else { return };
    let dir = model_fixtures::model_path(MODEL_ID).expect("the fixture directory");
    let mut metadata = model.metadata.clone();
    metadata
        .metadata
        .insert("reasoning".to_string(), serde_json::json!(true));
    // A model's reasoning setting is fixed when it loads, so this needs an
    // executor (and an engine) of its own.
    let mut executor =
        TemplateExecutor::with_base_path(dir.to_str().expect("the directory is valid UTF-8"));
    let config = GenerationConfig::greedy().with_max_tokens(512);

    let output = executor
        .execute(
            &metadata,
            // Hard enough that the model thinks; it answers "2 + 2" outright.
            &Envelope::new(EnvelopeKind::Text(
                "What is 17 * 23? Think briefly, then answer with the number.".to_string(),
            )),
            Some(&config),
        )
        .expect("a thinking turn succeeds");

    let reasoning = output
        .metadata
        .get("reasoning_content")
        .expect("the thought is reported as reasoning_content");
    assert!(!reasoning.trim().is_empty());
    let answer = text_of(&output);
    assert!(
        answer.contains("391"),
        "the answer follows the thought: {answer:?}"
    );
    assert!(
        answer.len() < reasoning.len(),
        "the thought is not part of the answer: {answer:?}"
    );
}

#[test]
fn a_grammar_is_refused_before_anything_is_generated() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let config = GenerationConfig::greedy()
        .with_max_tokens(8)
        .with_grammar("root ::= \"yes\" | \"no\"");
    let err = executor
        .execute(
            metadata,
            &Envelope::new(EnvelopeKind::Text("Is water wet?".to_string())),
            Some(&config),
        )
        .expect_err("a grammar is not supported on this backend");
    assert!(
        matches!(err, AdapterError::UnsupportedBackendCapability { .. }),
        "got {err:?}"
    );
}

#[test]
fn a_callback_error_stops_generation_and_the_model_stays_usable() {
    let Some(mut model) = loaded() else { return };
    let Loaded { metadata, executor } = &mut *model;
    let config = GenerationConfig::greedy().with_max_tokens(128);
    let mut seen = 0usize;
    let err = executor
        .execute_streaming(
            metadata,
            &Envelope::new(EnvelopeKind::Text(
                "Count from 1 to 100, separated by spaces.".to_string(),
            )),
            Box::new(|_token| {
                seen += 1;
                if seen == 3 {
                    Err("stop here".into())
                } else {
                    Ok(())
                }
            }),
            Some(&config),
        )
        .expect_err("the callback's error ends the run");
    assert!(err.to_string().contains("stop here"), "got {err}");
    assert_eq!(seen, 3, "no token was delivered after the error");

    // The cancelled turn left nothing behind.
    let output = executor
        .execute(
            metadata,
            &Envelope::new(EnvelopeKind::Text(
                "Reply with exactly these two words: hello world".to_string(),
            )),
            Some(&GenerationConfig::greedy().with_max_tokens(16)),
        )
        .expect("the model still answers");
    assert!(text_of(&output).to_lowercase().contains("hello world"));
}
