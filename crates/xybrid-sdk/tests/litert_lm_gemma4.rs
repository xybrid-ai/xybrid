//! Gemma 4 E2B on the LiteRT-LM backend through the public SDK surface, the
//! one every binding (Kotlin, Swift, Flutter, React Native, Python) calls.
//!
//! The tests that run the model skip (passing, with a message) unless both
//! are present:
//!
//! - the model: `./integration-tests/download.sh gemma-4-e2b-litert`
//!   (about 2.6 GB), or `XYBRID_TEST_MODELS` pointing at a directory that
//!   holds `gemma-4-e2b-litert/`;
//! - the LiteRT-LM runtime library for the host, named by
//!   `XYBRID_LITERT_LM_LIBRARY` or findable by the platform loader.
//!
//! Set `XYBRID_REQUIRE_MODELS` to turn a skip into a failure.

#![cfg(feature = "llm-litert")]

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

use xybrid_core::ir::{Envelope, EnvelopeKind, ToolCallResult};
use xybrid_core::runtime_adapter::tool_call::strip_tool_calls;
use xybrid_core::runtime_adapter::GenerationConfig;
use xybrid_core::testing::model_fixtures;
use xybrid_sdk::model::{ModelLoader, XybridModel};
use xybrid_sdk::Tool;

const MODEL_ID: &str = "gemma-4-e2b-litert";
const MODEL_FILE: &str = "gemma-4-E2B-it.litertlm";

fn skip_or_fail(reason: &str) -> Option<PathBuf> {
    if model_fixtures::models_required() {
        panic!("{reason} (XYBRID_REQUIRE_MODELS is set)");
    }
    eprintln!("SKIP: {reason}");
    None
}

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

/// One loaded model for the whole test binary.
fn model() -> Option<MutexGuard<'static, XybridModel>> {
    static MODEL: OnceLock<Option<Mutex<XybridModel>>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let dir = model_dir()?;
            let model = ModelLoader::from_directory(&dir)
                .expect("a loader is built from the model directory")
                .load()
                .expect("the LiteRT-LM model loads");
            Some(Mutex::new(model))
        })
        .as_ref()
        .map(|model| {
            model
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        })
}

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

#[test]
fn the_model_is_reported_as_a_streaming_chat_model() {
    let Some(model) = model() else { return };
    assert!(model.is_llm());
    assert!(model.supports_streaming());
}

#[test]
fn text_chat_through_the_sdk() {
    let Some(model) = model() else { return };
    let input = Envelope::new(EnvelopeKind::Text(
        "Reply with exactly these two words: hello world".to_string(),
    ));
    let config = GenerationConfig::greedy().with_max_tokens(24);

    let result = model.run(&input, Some(&config)).expect("the run succeeds");

    let text = result.text().expect("a text result");
    assert!(
        text.to_lowercase().contains("hello world"),
        "unexpected answer: {text:?}"
    );
}

#[test]
fn streaming_through_the_sdk_delivers_the_answer_token_by_token() {
    let Some(model) = model() else { return };
    let input = Envelope::new(EnvelopeKind::Text(
        "Count from one to five in words, separated by spaces.".to_string(),
    ));
    let config = GenerationConfig::greedy().with_max_tokens(32);
    let mut emitted = String::new();
    let mut tokens = 0usize;

    let result = model
        .run_streaming(&input, Some(&config), |token| {
            emitted.push_str(&token.token);
            tokens += 1;
            Ok(())
        })
        .expect("the streaming run succeeds");

    assert!(tokens > 2, "the answer arrived in several tokens");
    assert_eq!(emitted.trim(), result.text().expect("a text result"));
}

#[test]
fn image_input_through_the_sdk() {
    let Some(model) = model() else { return };
    let image = Envelope::image(red_circle_png(), "png").expect("a valid image");
    let input = Envelope::user_message(
        "What shape is in this image, and what colour is it? Answer in one short sentence.",
        vec![image],
    )
    .expect("a multipart input");
    let config = GenerationConfig::greedy().with_max_tokens(32);

    let result = model.run(&input, Some(&config)).expect("the run succeeds");

    let text = result.text().expect("a text result").to_lowercase();
    assert!(text.contains("red"), "the colour was seen: {text:?}");
    assert!(text.contains("circle"), "the shape was seen: {text:?}");
}

#[test]
fn a_tool_round_trip_through_the_sdk() {
    let Some(model) = model() else { return };
    let question =
        "Use the get_temperature tool to check the bedroom temperature, then tell me what it is.";
    let tools = vec![Tool::function(
        "get_temperature",
        "Get the current temperature in a room, in Celsius.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "room": { "type": "string", "description": "Name of the room." }
            },
            "required": ["room"]
        }),
    )];
    let config = GenerationConfig::greedy()
        .with_max_tokens(96)
        .with_tools(tools);

    let first = model
        .run(
            &Envelope::new(EnvelopeKind::Text(question.to_string())),
            Some(&config),
        )
        .expect("the tool-offering turn succeeds");
    let calls = first.tool_calls();
    assert_eq!(calls.len(), 1, "one call was made: {:?}", first.text());
    assert_eq!(calls[0].function.name, "get_temperature");

    let results: Vec<ToolCallResult> = calls
        .iter()
        .map(|call| ToolCallResult {
            call_id: call.id.clone(),
            name: call.function.name.clone(),
            content: serde_json::json!({ "room": "bedroom", "temperature_c": 17.5 }),
        })
        .collect();
    let continuation = Envelope::tool_results(
        question,
        first.text().expect("the first turn's text"),
        &results,
    );
    let second = model
        .run(&continuation, Some(&config))
        .expect("the continuation turn succeeds");

    assert!(second.tool_calls().is_empty());
    let answer = strip_tool_calls(second.text().expect("a text result"));
    assert!(
        answer.contains("17.5"),
        "the answer uses the tool result: {answer:?}"
    );
}
