use std::path::PathBuf;

use xybrid_core::execution::{ExecutionTemplate, ModelMetadata, TemplateExecutor};
use xybrid_core::gateway::{Tool, ToolCall};
use xybrid_core::ir::{Envelope, EnvelopeKind, ToolCallResult};
use xybrid_core::runtime_adapter::tool_call::strip_tool_calls;
use xybrid_core::runtime_adapter::types::GenerationConfig;
use xybrid_core::testing::model_fixtures;

const MODEL_ID: &str = "functiongemma-270m-it";

fn weather_tool() -> Tool {
    Tool::function(
        "get_current_temperature",
        "Gets the current temperature for a given location.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco"
                }
            },
            "required": ["location"]
        }),
    )
}

fn parse_calls(response: &Envelope) -> Vec<ToolCall> {
    response
        .metadata
        .get(Envelope::TOOL_CALLS_METADATA_KEY)
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default()
}

fn model_dir() -> PathBuf {
    model_fixtures::model_path(MODEL_ID)
        .unwrap_or_else(|| PathBuf::from("integration-tests/fixtures/models").join(MODEL_ID))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = model_dir();
    let metadata_path = model_dir.join("model_metadata.json");
    if !metadata_path.exists() {
        return Err(format!(
            "metadata not found at {}; run ./integration-tests/download.sh {MODEL_ID}",
            metadata_path.display()
        )
        .into());
    }

    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    let model_file = match &metadata.execution_template {
        ExecutionTemplate::Gguf { model_file, .. } => model_file,
        other => return Err(format!("expected Gguf metadata, got {other:?}").into()),
    };
    if !model_dir.join(model_file).exists() {
        return Err(format!(
            "model not found at {}; run ./integration-tests/download.sh {MODEL_ID}",
            model_dir.join(model_file).display()
        )
        .into());
    }

    let mut executor = TemplateExecutor::with_base_path(
        model_dir
            .to_str()
            .ok_or("model directory path is not valid UTF-8")?,
    );
    let user_prompt = "What's the current temperature in London?";
    let config = GenerationConfig::greedy()
        .with_max_tokens(128)
        .with_tools(vec![weather_tool()]);
    let request = Envelope::new(EnvelopeKind::Text(user_prompt.to_string()));

    let response = executor.execute(&metadata, &request, Some(&config))?;
    let raw_text = match &response.kind {
        EnvelopeKind::Text(text) => text,
        other => return Err(format!("expected Text output, got {other:?}").into()),
    };
    let calls = parse_calls(&response);
    println!("raw model output: {raw_text}");
    println!("parsed tool calls: {calls:#?}");

    let call = calls
        .first()
        .ok_or("model emitted no parseable FunctionGemma tool call")?;
    if call.function.name != "get_current_temperature" {
        return Err(format!("unexpected tool name: {}", call.function.name).into());
    }
    let arguments: serde_json::Value = serde_json::from_str(&call.function.arguments)?;
    let location = arguments
        .get("location")
        .and_then(serde_json::Value::as_str)
        .ok_or("tool call did not include a string location")?;
    if !location.to_ascii_lowercase().contains("london") {
        return Err(format!("unexpected tool location: {location}").into());
    }

    let result = ToolCallResult {
        call_id: call.id.clone(),
        name: call.function.name.clone(),
        content: serde_json::json!({"location": location, "temperature_c": 21}),
    };
    let continuation = Envelope::tool_results(user_prompt, raw_text, &[result]);
    let final_response = executor.execute(&metadata, &continuation, Some(&config))?;
    let final_text = match &final_response.kind {
        EnvelopeKind::Text(text) => strip_tool_calls(text),
        other => return Err(format!("expected Text output, got {other:?}").into()),
    };
    println!("tool-result continuation: {final_text}");
    println!("Status: PASS - FunctionGemma emitted a parsed call and accepted its result");
    Ok(())
}
