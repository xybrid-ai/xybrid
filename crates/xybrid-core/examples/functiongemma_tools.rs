use std::path::PathBuf;

use xybrid_core::execution::{ExecutionTemplate, ModelMetadata, TemplateExecutor};
use xybrid_core::gateway::{Tool, ToolCall};
use xybrid_core::ir::{Envelope, EnvelopeKind, ToolCallResult};
use xybrid_core::runtime_adapter::tool_call::strip_tool_calls;
use xybrid_core::runtime_adapter::types::GenerationConfig;
use xybrid_core::testing::model_fixtures;

const MODEL_ID: &str = "functiongemma-270m-it";
const SYSTEM_PROMPT: &str = "Use the available function and report its result clearly.";

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

fn tool_results_for_calls(
    calls: &[ToolCall],
) -> Result<Vec<ToolCallResult>, Box<dyn std::error::Error>> {
    if calls.is_empty() {
        return Err("model emitted no parseable FunctionGemma tool call".into());
    }

    calls
        .iter()
        .map(|call| {
            if call.function.name != "get_current_temperature" {
                return Err(format!("unexpected tool name: {}", call.function.name).into());
            }
            let arguments: serde_json::Value = serde_json::from_str(&call.function.arguments)?;
            let location = arguments
                .get("location")
                .and_then(serde_json::Value::as_str)
                .filter(|location| !location.trim().is_empty())
                .ok_or("tool call did not include a non-empty string location")?;

            Ok(ToolCallResult {
                call_id: call.id.clone(),
                name: call.function.name.clone(),
                content: serde_json::json!({"location": location, "temperature_c": 21}),
            })
        })
        .collect()
}

fn validated_final_text(response: &Envelope) -> Result<String, Box<dyn std::error::Error>> {
    if !parse_calls(response).is_empty() {
        return Err("tool-result continuation emitted another tool call".into());
    }
    let text = match &response.kind {
        EnvelopeKind::Text(text) => strip_tool_calls(text),
        other => return Err(format!("expected Text output, got {other:?}").into()),
    };
    if text.trim().is_empty() {
        return Err("tool-result continuation produced no final answer".into());
    }
    Ok(text)
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
    let mut request = Envelope::new(EnvelopeKind::Text(user_prompt.to_string()));
    request
        .metadata
        .insert("system_prompt".to_string(), SYSTEM_PROMPT.to_string());

    let response = executor.execute(&metadata, &request, Some(&config))?;
    let raw_text = match &response.kind {
        EnvelopeKind::Text(text) => text,
        other => return Err(format!("expected Text output, got {other:?}").into()),
    };
    let calls = parse_calls(&response);
    println!("raw model output: {raw_text}");
    println!("parsed tool calls: {calls:#?}");

    let results = tool_results_for_calls(&calls)?;
    if !results.iter().any(|result| {
        result
            .content
            .get("location")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|location| location.to_ascii_lowercase().contains("london"))
    }) {
        return Err("model did not request the London temperature".into());
    }

    let mut continuation = Envelope::tool_results(user_prompt, raw_text, &results);
    continuation
        .metadata
        .insert("system_prompt".to_string(), SYSTEM_PROMPT.to_string());
    let final_response = executor.execute(&metadata, &continuation, Some(&config))?;
    let final_text = validated_final_text(&final_response)?;
    println!("tool-result continuation: {final_text}");
    println!(
        "Status: PASS - FunctionGemma emitted {} parsed call(s) and accepted every result",
        results.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_core::gateway::FunctionCall;

    fn weather_call(id: &str, location: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "get_current_temperature".to_string(),
                arguments: serde_json::json!({"location": location}).to_string(),
            },
        }
    }

    #[test]
    fn builds_one_result_for_every_call_in_a_multi_call_turn() {
        let calls = [
            weather_call("call_0", "London"),
            weather_call("call_1", "Paris"),
        ];

        let results = tool_results_for_calls(&calls).expect("valid calls should produce results");

        assert_eq!(results.len(), calls.len());
        assert_eq!(results[0].call_id, calls[0].id);
        assert_eq!(results[1].call_id, calls[1].id);
    }

    #[test]
    fn rejects_an_empty_final_answer_before_reporting_pass() {
        assert!(
            validated_final_text(&Envelope::new(EnvelopeKind::Text("   ".to_string()))).is_err()
        );
    }
}
