//! LFM3 smoke tests on the real models (`integration-tests/download.sh
//! lfm3-1.7b lfm3-vl-2b`). The runtime tests skip when the fixture is not
//! downloaded; the metadata tests always run.

#![cfg(feature = "llm-llamacpp")]

use std::path::Path;
use xybrid_core::{
    execution::{ExecutionTemplate, ModelMetadata, TemplateExecutor},
    gateway::{Tool, ToolCall},
    ir::{Envelope, EnvelopeKind, ToolCallResult},
    runtime_adapter::GenerationConfig,
    testing::model_fixtures,
};

const TEXT_MODEL_ID: &str = "lfm3-1.7b";
const TEXT_MODEL_FILE: &str = "LFM3-1.7B-QAD-Q4_0.gguf";
const VL_MODEL_ID: &str = "lfm3-vl-2b";
const VL_MODEL_FILE: &str = "LFM3-VL-2B-QAD-Q4_0.gguf";
const VL_MMPROJ_FILE: &str = "mmproj-LFM3-VL-2B-Q8_0.gguf";
const VL_MAX_IMAGE_TOKENS: usize = 1024;

fn load_metadata(model_dir: &Path) -> ModelMetadata {
    let metadata_path = model_dir.join("model_metadata.json");
    let content = std::fs::read_to_string(&metadata_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", metadata_path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", metadata_path.display()))
}

fn text(output: &Envelope) -> &str {
    match &output.kind {
        EnvelopeKind::Text(text) => text,
        other => panic!("expected text output, got {other:?}"),
    }
}

fn tool_calls(output: &Envelope) -> Vec<ToolCall> {
    output
        .metadata
        .get(Envelope::TOOL_CALLS_METADATA_KEY)
        .map(|raw| serde_json::from_str(raw).expect("tool_calls metadata is a ToolCall array"))
        .unwrap_or_default()
}

fn weather_tool() -> Tool {
    Tool::function(
        "get_weather",
        "Get the current weather for a city",
        serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"]
        }),
    )
}

#[test]
fn lfm3_fixture_metadata_is_ready_for_runtime_smoke() {
    let model_dir = model_fixtures::model_path(TEXT_MODEL_ID)
        .unwrap_or_else(|| panic!("{TEXT_MODEL_ID} fixture metadata directory must exist"));
    let metadata = load_metadata(&model_dir);

    metadata.validate().expect("metadata validates");
    assert_eq!(metadata.model_id, TEXT_MODEL_ID);
    assert_eq!(metadata.supports_tool_calling(), Some(true));
    match &metadata.execution_template {
        ExecutionTemplate::Gguf { model_file, .. } => assert_eq!(model_file, TEXT_MODEL_FILE),
        other => panic!("expected Gguf execution template, got {other:?}"),
    }
}

#[test]
fn lfm3_vl_fixture_metadata_caps_image_tokens() {
    let model_dir = model_fixtures::model_path(VL_MODEL_ID)
        .unwrap_or_else(|| panic!("{VL_MODEL_ID} fixture metadata directory must exist"));
    let metadata = load_metadata(&model_dir);

    metadata.validate().expect("metadata validates");
    match &metadata.execution_template {
        ExecutionTemplate::VisionLanguage { model_file, .. } => {
            assert_eq!(model_file, VL_MODEL_FILE)
        }
        other => panic!("expected VisionLanguage execution template, got {other:?}"),
    }
    let vision = metadata
        .vision_encoder
        .as_ref()
        .expect("VisionLanguage fixture declares vision_encoder");
    assert_eq!(vision.file, VL_MMPROJ_FILE);
    assert_eq!(vision.max_image_tokens, Some(VL_MAX_IMAGE_TOKENS));
}

/// Without a thinking state the model reasoned with no opening tag and
/// answered a question nobody asked; with thinking off it answers directly.
#[test]
fn lfm3_answers_directly_with_thinking_off() {
    let Some(model_dir) = model_fixtures::model_or_skip(TEXT_MODEL_ID) else {
        return;
    };
    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let input = Envelope::new(EnvelopeKind::Text(
        "What is the capital of France? Answer in one sentence.".to_string(),
    ));
    let config = GenerationConfig::greedy().with_max_tokens(128);

    let output = executor
        .execute(&metadata, &input, Some(&config))
        .expect("LFM3 generation succeeds");
    let answer = text(&output);
    eprintln!(
        "thinking off: {answer:?} ({:?})",
        output.metadata.get("tokens_generated")
    );

    assert!(answer.contains("Paris"), "{answer:?}");
    assert!(!answer.contains("think>"), "{answer:?}");
    let tokens: usize = output.metadata["tokens_generated"].parse().unwrap();
    assert!(
        tokens < 48,
        "a direct answer, not a reasoning run: {tokens} tokens"
    );
}

#[test]
fn lfm3_reasons_with_thinking_on() {
    let Some(model_dir) = model_fixtures::model_or_skip(TEXT_MODEL_ID) else {
        return;
    };
    let mut metadata = load_metadata(&model_dir);
    metadata
        .metadata
        .insert("reasoning".to_string(), serde_json::Value::Bool(true));
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let input = Envelope::new(EnvelopeKind::Text(
        "What is the capital of France? Answer in one sentence.".to_string(),
    ));
    let config = GenerationConfig::greedy().with_max_tokens(512);

    let output = executor
        .execute(&metadata, &input, Some(&config))
        .expect("LFM3 generation succeeds");
    let answer = text(&output);
    let reasoning = output
        .metadata
        .get("reasoning_content")
        .expect("a reasoning run surfaces its chain-of-thought");
    eprintln!("thinking on: {answer:?}\nreasoning: {reasoning:?}");

    assert!(answer.contains("Paris"), "{answer:?}");
    assert!(!answer.contains("think>"), "{answer:?}");
    assert!(!reasoning.trim().is_empty());
}

/// One full tool turn: the call parses (typed against the schema), and the
/// continuation answers from the result.
#[test]
fn lfm3_calls_a_tool_and_answers_from_its_result() {
    let Some(model_dir) = model_fixtures::model_or_skip(TEXT_MODEL_ID) else {
        return;
    };
    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let question = "What's the weather in Paris?";
    let config = GenerationConfig::greedy()
        .with_max_tokens(256)
        .with_tools([weather_tool()]);

    let first = executor
        .execute(
            &metadata,
            &Envelope::new(EnvelopeKind::Text(question.to_string())),
            Some(&config),
        )
        .expect("tool-bearing LFM3 turn succeeds");
    let calls = tool_calls(&first);
    eprintln!("call turn: {:?} -> {calls:?}", text(&first));

    assert_eq!(calls.len(), 1, "{:?}", text(&first));
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
        serde_json::json!({"city": "Paris"})
    );
    assert_eq!(
        first.metadata.get("finish_reason").map(String::as_str),
        Some("tool_calls")
    );

    let continuation = Envelope::tool_results(
        question,
        text(&first),
        &[ToolCallResult {
            call_id: calls[0].id.clone(),
            name: calls[0].function.name.clone(),
            content: serde_json::json!({"temperature_c": 21, "condition": "sunny"}),
        }],
    );
    let answer = executor
        .execute(&metadata, &continuation, Some(&config))
        .expect("LFM3 tool continuation succeeds");
    eprintln!("answer turn: {:?}", text(&answer));

    assert!(tool_calls(&answer).is_empty(), "{:?}", text(&answer));
    assert!(text(&answer).contains("21"), "{:?}", text(&answer));
}

/// The same tool turn with thinking on, as the registry entries ship: the
/// model reasons before it calls, and again before it answers from the result.
#[test]
fn lfm3_calls_a_tool_with_thinking_on() {
    let Some(model_dir) = model_fixtures::model_or_skip(TEXT_MODEL_ID) else {
        return;
    };
    let mut metadata = load_metadata(&model_dir);
    metadata
        .metadata
        .insert("reasoning".to_string(), serde_json::Value::Bool(true));
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let question = "What's the weather in Paris?";
    let config = GenerationConfig::greedy()
        .with_max_tokens(1024)
        .with_tools([weather_tool()]);

    let first = executor
        .execute(
            &metadata,
            &Envelope::new(EnvelopeKind::Text(question.to_string())),
            Some(&config),
        )
        .expect("tool-bearing LFM3 turn succeeds");
    let calls = tool_calls(&first);
    eprintln!(
        "thinking-on call turn: {:?} -> {calls:?}\nreasoning: {:?}",
        text(&first),
        first.metadata.get("reasoning_content")
    );

    assert_eq!(calls.len(), 1, "{:?}", text(&first));
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
        serde_json::json!({"city": "Paris"})
    );
    assert_eq!(
        first.metadata.get("finish_reason").map(String::as_str),
        Some("tool_calls")
    );
    assert!(!text(&first).contains("think>"), "{:?}", text(&first));

    let continuation = Envelope::tool_results(
        question,
        text(&first),
        &[ToolCallResult {
            call_id: calls[0].id.clone(),
            name: calls[0].function.name.clone(),
            content: serde_json::json!({"temperature_c": 21, "condition": "sunny"}),
        }],
    );
    let answer = executor
        .execute(&metadata, &continuation, Some(&config))
        .expect("LFM3 tool continuation succeeds");
    eprintln!(
        "thinking-on answer turn: {:?}\nreasoning: {:?}",
        text(&answer),
        answer.metadata.get("reasoning_content")
    );

    assert!(tool_calls(&answer).is_empty(), "{:?}", text(&answer));
    assert!(text(&answer).contains("21"), "{:?}", text(&answer));
    assert!(!text(&answer).contains("think>"), "{:?}", text(&answer));
}

/// LFM3 writes argument values as text; the schema types them.
#[test]
fn lfm3_tool_arguments_follow_their_schema_types() {
    let Some(model_dir) = model_fixtures::model_or_skip(TEXT_MODEL_ID) else {
        return;
    };
    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let alarm = Tool::function(
        "set_alarm",
        "Set an alarm",
        serde_json::json!({
            "type": "object",
            "properties": {
                "hour": {"type": "integer"},
                "minute": {"type": "integer"},
                "vibrate": {"type": "boolean"},
                "label": {"type": "string"}
            },
            "required": ["hour", "minute"]
        }),
    );
    let config = GenerationConfig::greedy()
        .with_max_tokens(256)
        .with_tools([alarm]);
    let input = Envelope::new(EnvelopeKind::Text(
        "Set an alarm for 6:45 with vibration, label it gym.".to_string(),
    ));

    let output = executor
        .execute(&metadata, &input, Some(&config))
        .expect("tool-bearing LFM3 turn succeeds");
    let calls = tool_calls(&output);
    eprintln!("alarm turn: {:?} -> {calls:?}", text(&output));

    assert_eq!(calls.len(), 1, "{:?}", text(&output));
    let arguments: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(arguments["hour"].is_i64(), "{arguments}");
    assert!(arguments["minute"].is_i64(), "{arguments}");
    if let Some(vibrate) = arguments.get("vibrate") {
        assert!(vibrate.is_boolean(), "{arguments}");
    }
    if let Some(label) = arguments.get("label") {
        assert!(label.is_string(), "{arguments}");
    }
}

/// A 2048x1536 image would be 3,072 tokens; the bundle caps it at 1,024.
#[cfg(feature = "llm-llamacpp-vision")]
#[test]
fn lfm3_vl_caption_respects_the_image_token_cap() {
    let Some(model_dir) = model_fixtures::model_or_skip(VL_MODEL_ID) else {
        return;
    };
    let metadata = load_metadata(&model_dir);
    let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(2048, 1536, |x, y| {
        let (cx, cy) = (x as i64 - 1434, y as i64 - 768);
        if (205..819).contains(&x) && (461..1075).contains(&y) {
            image::Rgb([220, 30, 30])
        } else if cx * cx + cy * cy <= 276 * 276 {
            image::Rgb([240, 210, 20])
        } else {
            image::Rgb([30, 60, 200])
        }
    }));
    let mut encoded = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("test image encodes");
    let image = Envelope::image(encoded.into_inner(), "png").expect("valid image");
    let input = Envelope::user_message(
        "Describe the shapes and colors in this image in one sentence.",
        vec![image],
    )
    .expect("multipart VLM input");
    let config = GenerationConfig::greedy().with_max_tokens(128);

    xybrid_core::tracing::init_tracing(true);
    xybrid_core::tracing::reset_tracing();
    let output = executor
        .execute(&metadata, &input, Some(&config))
        .expect("LFM3-VL caption succeeds");
    let spans = xybrid_core::tracing::get_stages_json();
    xybrid_core::tracing::reset_tracing();
    let caption = text(&output);
    eprintln!("caption: {caption:?}");

    assert!(!caption.trim().is_empty());
    assert!(!caption.contains("think>"), "{caption:?}");
    let tokens_in = spans["spans"]
        .as_array()
        .expect("tracing spans")
        .iter()
        .filter_map(|span| span["metadata"].get("tokens_in"))
        .filter_map(|value| value.as_str()?.parse::<usize>().ok())
        .max()
        .expect("the VLM span reports its prompt tokens");
    eprintln!("prompt tokens: {tokens_in}");
    assert!(
        tokens_in <= VL_MAX_IMAGE_TOKENS + 128,
        "the image must be scaled to the cap: {tokens_in} prompt tokens"
    );
}
