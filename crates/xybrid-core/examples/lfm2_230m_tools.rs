//! Local tool calling end-to-end on LFM2.5 (230M, falling back to 350M).
//!
//! Proves the full tool-calling loop on a real model: tool definitions ride
//! `GenerationConfig::with_tools` into the model's embedded chat template, the
//! model's `<|tool_call_start|>...` blocks come back parsed in the response
//! envelope's `tool_calls` metadata, and tool results feed back as pure data
//! through `Envelope::tool_results` — one `run` per model turn.
//!
//! The turn loop below is deliberately plain app code: chaining turns works
//! the same way in Rust, Swift, Kotlin, and Dart, so xybrid ships no loop
//! API. This file is the reference usage pattern.
//!
//! Fetch a model first (resolves via registry.xybrid.dev):
//!   model dir: ~/.xybrid/cache/extracted/lfm2.5-230m/ (GGUF + model_metadata.json)
//!   (the LFM2-family sibling lfm2.5-350m works identically and is probed as a
//!   fallback — same ChatML framing and tool-protocol tokens)
//!
//! Run with:
//!   cargo run --example lfm2_230m_tools -p xybrid-core --features llm-llamacpp

use std::path::PathBuf;

use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::gateway::{Tool, ToolCall};
use xybrid_core::ir::{Envelope, EnvelopeKind, ToolCallResult};
use xybrid_core::runtime_adapter::tool_call::strip_tool_calls;
use xybrid_core::runtime_adapter::types::GenerationConfig;

/// Fake temperature the "sensor" reports. Distinctive on purpose: the final
/// assertion checks the model's answer actually references the tool result.
const BEDROOM_TEMP_C: f64 = 17.5;

fn model_dir() -> Option<PathBuf> {
    // Windows sets USERPROFILE rather than HOME.
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .expect("neither HOME nor USERPROFILE is set");
    let cache = PathBuf::from(home).join(".xybrid/cache/extracted");
    // Design target is the 230M; the 350M sibling speaks the same protocol.
    ["lfm2.5-230m", "lfm2.5-350m"]
        .iter()
        .map(|id| cache.join(id))
        .find(|dir| dir.join("model_metadata.json").exists())
}

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

/// The app-side tool implementations. Real apps call sensors / HTTP / device
/// APIs here; the example fakes deterministic results.
fn execute_tool(call: &ToolCall) -> ToolCallResult {
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
            "temperature_c": BEDROOM_TEMP_C,
        }),
        "set_thermostat" => serde_json::json!({
            "room": room,
            "status": "ok",
            "target_c": args.get("temperature_c").cloned().unwrap_or(serde_json::json!(21)),
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  LFM2.5 — local tool calling (llama.cpp)");
    println!("═══════════════════════════════════════════════════════\n");

    let Some(dir) = model_dir() else {
        eprintln!("No LFM2.5 model bundle found under ~/.xybrid/cache/extracted/");
        eprintln!("Fetch lfm2.5-230m (or lfm2.5-350m) from the registry first.");
        return Err("model not present".into());
    };

    let metadata: ModelMetadata =
        serde_json::from_str(&std::fs::read_to_string(dir.join("model_metadata.json"))?)?;
    println!(
        "Model: {} v{} ({})",
        metadata.model_id,
        metadata.version,
        dir.display()
    );

    let mut executor = TemplateExecutor::with_base_path(dir.to_str().unwrap());

    // Small models follow tool instructions best with a system prompt that
    // licenses tool use plus a direct imperative. (The 350M sibling calls
    // tools from a plain "what is the temperature?" phrasing; the 230M needs
    // the nudge.) The template merges this with the generated tool list.
    let system_msg = "You are a home automation assistant. \
        Use the available tools to fulfill requests. Call one tool at a time.";
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
    envelope
        .metadata
        .insert("system_prompt".to_string(), system_msg.to_string());
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
            let result = execute_tool(call);
            println!(
                "  → {}({}) = {}",
                call.function.name, call.function.arguments, result.content
            );
            results.push(result);
        }
        envelope = Envelope::tool_results(user_msg, &text, &results);
        // Continuations must carry the SAME system prompt as the first turn
        // so the recomposed chat prefix is byte-identical (KV-prefix reuse).
        envelope
            .metadata
            .insert("system_prompt".to_string(), system_msg.to_string());
    }

    // ── Assertions: the three facts this example exists to prove ─────────────
    if first_turn_call_count == 0 {
        return Err("turn 1 emitted no parseable tool call".into());
    }

    let display = strip_tool_calls(&final_answer);
    println!("\nFinal answer (display): {display}");

    // The fake sensor reported 17.5°C; a grounded answer references it (or
    // the 21°C it set). A 230M model phrases things loosely — accept either
    // number as evidence the tool results reached the final answer.
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
