//! Streaming tool calling on LFM2.5: the token feed stays protocol-clean.
//!
//! Proves the streaming contract for tool-bearing requests: raw
//! `<|tool_call_start|>...` blocks are suppressed from the `on_token` feed
//! (they are protocol traffic, not answer text), while the final envelope
//! still carries the parsed `tool_calls` metadata and a canonical
//! `finish_reason` of `"tool_calls"`. The full loop pattern lives in
//! `lfm2_230m_tools.rs`; this example is the streaming half of the contract.
//!
//! Fetch a model first (resolves via registry.xybrid.dev):
//!   model dir: ~/.xybrid/cache/extracted/lfm2.5-230m/ (or the 350m sibling)
//!
//! Run with:
//!   cargo run --example lfm2_230m_tools_streaming -p xybrid-core --features llm-llamacpp

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::gateway::{Tool, ToolCall};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::types::GenerationConfig;

fn model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .expect("neither HOME nor USERPROFILE is set");
    let cache = PathBuf::from(home).join(".xybrid/cache/extracted");
    ["lfm2.5-230m", "lfm2.5-350m"]
        .iter()
        .map(|id| cache.join(id))
        .find(|dir| dir.join("model_metadata.json").exists())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  LFM2.5 — STREAMING tool calling (clean token feed)");
    println!("═══════════════════════════════════════════════════════\n");

    let Some(dir) = model_dir() else {
        eprintln!("No LFM2.5 model bundle found under ~/.xybrid/cache/extracted/");
        return Err("model not present".into());
    };

    let metadata: ModelMetadata =
        serde_json::from_str(&std::fs::read_to_string(dir.join("model_metadata.json"))?)?;
    println!("Model: {} ({})", metadata.model_id, dir.display());

    let mut executor = TemplateExecutor::with_base_path(dir.to_str().unwrap());

    let tool = Tool::function(
        "get_temperature",
        "Get the current temperature in a room, in Celsius.",
        serde_json::json!({
            "type": "object",
            "properties": { "room": { "type": "string" } },
            "required": ["room"]
        }),
    );
    let config = GenerationConfig::greedy()
        .with_max_tokens(128)
        .with_tools([tool]);

    let mut envelope = Envelope::new(EnvelopeKind::Text(
        "Use the get_temperature tool to check the bedroom temperature.".to_string(),
    ));
    envelope.metadata.insert(
        "system_prompt".to_string(),
        "You are a home automation assistant. Use the available tools.".to_string(),
    );

    // Collect everything the stream emits: the assertion below is that NONE
    // of it is tool-protocol text.
    let streamed = Arc::new(Mutex::new(String::new()));
    let terminal_finish = Arc::new(Mutex::new(None::<String>));
    let streamed_cb = Arc::clone(&streamed);
    let finish_cb = Arc::clone(&terminal_finish);

    let response = executor.execute_streaming(
        &metadata,
        &envelope,
        Box::new(move |token| {
            streamed_cb.lock().unwrap().push_str(&token.token);
            if let Some(reason) = &token.finish_reason {
                *finish_cb.lock().unwrap() = Some(reason.clone());
            }
            Ok(())
        }),
        Some(&config),
    )?;

    let streamed = streamed.lock().unwrap().clone();
    let terminal_finish = terminal_finish.lock().unwrap().clone();
    println!("streamed text  : {streamed:?}");
    println!("terminal finish: {terminal_finish:?}");

    // ── Assertions: the streaming contract ──────────────────────────────────
    for marker in ["<|tool_call_start|>", "<|tool_call>", "<|tool_response>"] {
        if streamed.contains(marker) {
            return Err(format!("protocol marker {marker:?} leaked into the token feed").into());
        }
    }

    let calls: Vec<ToolCall> = response
        .metadata
        .get(Envelope::TOOL_CALLS_METADATA_KEY)
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default();
    if calls.is_empty() {
        return Err("no tool call parsed from the streamed turn".into());
    }
    println!(
        "parsed call    : {}({})",
        calls[0].function.name, calls[0].function.arguments
    );

    let envelope_finish = response.metadata.get("finish_reason").cloned();
    if envelope_finish.as_deref() != Some("tool_calls") {
        return Err(format!("envelope finish_reason not canonical: {envelope_finish:?}").into());
    }
    if terminal_finish.as_deref() != Some("tool_calls") {
        return Err(
            format!("terminal token finish_reason not canonical: {terminal_finish:?}").into(),
        );
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("Status: PASS — clean feed, parsed call, finish_reason=tool_calls");
    println!("═══════════════════════════════════════════════════════");
    Ok(())
}
