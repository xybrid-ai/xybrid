//! Local tool calling end-to-end on gemma-4 E2B (text-only turns).
//!
//! Proves the gemma-4-family tool protocol on a real model through the
//! `VisionLanguage` execution path: tool definitions ride
//! `GenerationConfig::with_tools` into the model's embedded chat template
//! (`<|tool>declaration:...<tool|>` blocks), the model's
//! `<|tool_call>call:name{...}<tool_call|>` blocks come back parsed in the
//! response envelope's `tool_calls` metadata, and tool results feed back as
//! pure data through `Envelope::tool_results` — one run per model turn,
//! with the responses appended inside the still-open model turn per the
//! gemma-4 protocol.
//!
//! The turn loop below is deliberately plain app code: chaining turns works
//! the same way in Rust, Swift, Kotlin, and Dart, so xybrid ships no loop
//! API. This file is the reference usage pattern.
//!
//! Run after downloading the fixture (~5.5 GB, not checked into git):
//!   ./integration-tests/download.sh gemma-4-e2b
//!   cargo run -p xybrid-core --example gemma4_e2b_tools \
//!     --features llm-llamacpp-vision

#[cfg(feature = "llm-llamacpp-vision")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
    use xybrid_core::gateway::{Tool, ToolCall};
    use xybrid_core::ir::{Envelope, EnvelopeKind, ToolCallResult};
    use xybrid_core::runtime_adapter::tool_call::strip_tool_calls;
    use xybrid_core::runtime_adapter::types::GenerationConfig;
    use xybrid_core::testing::model_fixtures;

    const MODEL_ID: &str = "gemma-4-e2b";
    /// Fake temperature the "sensor" reports. Distinctive on purpose: the
    /// final assertion checks the answer actually references a tool result.
    const BEDROOM_TEMP_C: f64 = 17.5;

    fn tools() -> Vec<Tool> {
        let get_temperature = Tool::function(
            "get_temperature",
            "Get the current temperature in a room, in Celsius.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "room": { "type": "string", "description": "Name of the room." }
                },
                "required": ["room"]
            }),
        );
        let set_thermostat = Tool::function(
            "set_thermostat",
            "Set the thermostat target temperature for a room, in Celsius.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "room": { "type": "string", "description": "Name of the room." },
                    "temperature_c": { "type": "number", "description": "Target temperature." }
                },
                "required": ["room", "temperature_c"]
            }),
        );
        vec![get_temperature, set_thermostat]
    }

    // The app-side tool implementations. Real apps call sensors / device
    // APIs here; the example fakes deterministic results.
    fn execute_tool(call: &ToolCall, bedroom_temp_c: f64) -> ToolCallResult {
        let args: serde_json::Value =
            serde_json::from_str(&call.function.arguments).unwrap_or(serde_json::Value::Null);
        let room = args
            .get("room")
            .and_then(|v| v.as_str())
            .unwrap_or("bedroom")
            .to_string();

        let content = match call.function.name.as_str() {
            "get_temperature" => serde_json::json!({
                "room": room,
                "temperature_c": bedroom_temp_c,
            }),
            "set_thermostat" => serde_json::json!({
                "room": room,
                "status": "ok",
                "target_c": args
                    .get("temperature_c")
                    .cloned()
                    .unwrap_or(serde_json::json!(21)),
            }),
            other => serde_json::json!({ "error": format!("unknown tool: {other}") }),
        };

        ToolCallResult {
            call_id: call.id.clone(),
            name: call.function.name.clone(),
            content,
        }
    }

    fn parse_calls(response: &Envelope) -> Vec<ToolCall> {
        response
            .metadata
            .get(Envelope::TOOL_CALLS_METADATA_KEY)
            .and_then(|raw| serde_json::from_str(raw).ok())
            .unwrap_or_default()
    }

    println!("═══════════════════════════════════════════════════════");
    println!("  gemma-4 E2B — local tool calling (llama.cpp, text-only)");
    println!("═══════════════════════════════════════════════════════\n");

    let model_dir = model_fixtures::model_path(MODEL_ID)
        .unwrap_or_else(|| PathBuf::from("integration-tests/fixtures/models").join(MODEL_ID));
    println!("Model directory: {}", model_dir.display());

    let metadata_path = model_dir.join("model_metadata.json");
    if !metadata_path.exists() {
        println!("SKIP: metadata not found at {}", metadata_path.display());
        println!("Run: ./integration-tests/download.sh {MODEL_ID}");
        return Ok(());
    }
    let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    let mut required_files = Vec::new();
    match &metadata.execution_template {
        xybrid_core::execution::ExecutionTemplate::VisionLanguage { model_file, .. } => {
            required_files.push(model_file.clone());
        }
        other => return Err(format!("expected VisionLanguage metadata, got {other:?}").into()),
    }
    if let Some(vision_encoder) = &metadata.vision_encoder {
        required_files.push(vision_encoder.file.clone());
    }
    let missing: Vec<_> = required_files
        .iter()
        .filter(|file| !model_dir.join(file.as_str()).exists())
        .collect();
    if !missing.is_empty() {
        println!("SKIP: gemma-4 E2B artifacts are not downloaded.");
        for file in missing {
            println!("Missing: {}", model_dir.join(file).display());
        }
        println!("Run: ./integration-tests/download.sh {MODEL_ID}");
        return Ok(());
    }

    let mut executor = TemplateExecutor::with_base_path(
        model_dir
            .to_str()
            .ok_or("model directory path is not valid UTF-8")?,
    );

    // NOTE: the VisionLanguage path renders only the envelope's own
    // text/image parts — `system_prompt` metadata is not threaded into
    // multimodal messages today — so all steering lives in the user
    // message. The tool declarations still land in the system turn via the
    // template's own `tools` handling.
    let user_msg = "Use the get_temperature tool to check the bedroom temperature. \
        If it is below 20 degrees, use the set_thermostat tool to set the bedroom to 21 degrees. \
        Then tell me what you did.";
    let config = GenerationConfig::greedy()
        .with_max_tokens(256)
        .with_tools(tools());

    // ── The tool loop: one `execute` call per model turn ─────────────────────
    // When a turn requests tool calls, the app runs them and packs the
    // results into the next envelope with `Envelope::tool_results`.
    const MAX_TURNS: usize = 4;
    let mut envelope = Envelope::new(EnvelopeKind::Text(user_msg.to_string()));
    let mut turn = 0usize;
    let mut first_turn_call_count = 0usize;
    let final_answer: String;

    loop {
        turn += 1;
        if turn > MAX_TURNS {
            return Err(format!("model kept requesting tools after {MAX_TURNS} turns").into());
        }

        let response = executor.execute(&metadata, &envelope, Some(&config))?;
        let text = match &response.kind {
            EnvelopeKind::Text(t) => t.clone(),
            other => return Err(format!("expected Text output, got {other:?}").into()),
        };
        let calls = parse_calls(&response);
        println!("\n── turn {turn} ──");
        println!("  raw text : {text:?}");
        println!("  tool calls: {}", calls.len());

        if turn == 1 {
            first_turn_call_count = calls.len();
        }

        if calls.is_empty() {
            final_answer = text;
            break;
        }

        let mut results = Vec::with_capacity(calls.len());
        for call in &calls {
            let result = execute_tool(call, BEDROOM_TEMP_C);
            println!(
                "  → {}({}) = {}",
                call.function.name, call.function.arguments, result.content
            );
            results.push(result);
        }
        envelope = Envelope::tool_results(user_msg, &text, &results);
    }

    // ── Assertions: the three facts this example exists to prove ─────────────
    if first_turn_call_count == 0 {
        return Err("turn 1 emitted no parseable tool call".into());
    }

    let display = strip_tool_calls(&final_answer);
    println!("\nFinal answer (display): {display}");

    // The fake sensor reported 17.5°C; a grounded answer references it (or
    // the 21°C it set).
    let references_result = ["17.5", "17,5", "21"]
        .iter()
        .any(|needle| display.contains(needle));
    if !references_result {
        return Err(
            format!("final answer does not reference the tool results: {display:?}").into(),
        );
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!(
        "Status: PASS — {first_turn_call_count} call(s) on turn 1, answer references tool results"
    );
    println!("═══════════════════════════════════════════════════════");
    Ok(())
}

#[cfg(not(feature = "llm-llamacpp-vision"))]
fn main() {
    eprintln!("This example requires the `llm-llamacpp-vision` feature.");
    eprintln!(
        "Run: cargo run -p xybrid-core --example gemma4_e2b_tools --features llm-llamacpp-vision"
    );
}
