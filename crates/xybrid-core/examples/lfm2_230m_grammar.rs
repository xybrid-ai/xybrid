//! Constrained decoding (GBNF / JSON-Schema) end-to-end on LFM2.5-230M.
//!
//! Proves the new `GenerationConfig::with_json_schema` path actually constrains
//! a real model's output to schema-valid JSON — the on-device data-extraction
//! use case LFM2.5-230M is built for.
//!
//! Fetch the model first (resolves via registry.xybrid.dev):
//!   model dir: ~/.xybrid/cache/extracted/lfm2.5-230m/ (GGUF + model_metadata.json)
//!
//! Run with:
//!   cargo run --example lfm2_230m_grammar -p xybrid-core --features llm-llamacpp

use std::collections::HashMap;
use std::path::PathBuf;

use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::llm::GenerationConfig;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(home).join(".xybrid/cache/extracted/lfm2.5-230m")
}

fn run(
    executor: &mut TemplateExecutor,
    metadata: &ModelMetadata,
    prompt: &str,
    config: Option<&GenerationConfig>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut meta = HashMap::new();
    meta.insert(
        "system_prompt".to_string(),
        "You extract structured data. Respond with a single JSON object, nothing else.".to_string(),
    );
    let envelope = Envelope {
        kind: EnvelopeKind::Text(prompt.to_string()),
        metadata: meta,
    };
    let out = executor.execute(metadata, &envelope, config)?;
    // Diagnostic: surface generation telemetry (finish_reason, token counts).
    let mut diag: Vec<_> = out
        .metadata
        .iter()
        .filter(|(k, _)| {
            matches!(
                k.as_str(),
                "finish_reason" | "tokens_generated" | "tokens_per_second" | "stop_reason"
            )
        })
        .collect();
    diag.sort();
    if !diag.is_empty() {
        println!("  [telemetry] {diag:?}");
    }
    match out.kind {
        EnvelopeKind::Text(t) => Ok(t),
        other => Err(format!("expected Text output, got {other:?}").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  LFM2.5-230M — GBNF constrained decoding (data extraction)");
    println!("═══════════════════════════════════════════════════════\n");

    let dir = model_dir();
    let gguf = dir.join("LFM2.5-230M-Q4_K_M.gguf");
    if !gguf.exists() {
        eprintln!("Model GGUF not found at {}", gguf.display());
        eprintln!("Fetch lfm2.5-230m from the registry first.");
        return Err("model not present".into());
    }

    let metadata: ModelMetadata =
        serde_json::from_str(&std::fs::read_to_string(dir.join("model_metadata.json"))?)?;
    println!("Model: {} v{}", metadata.model_id, metadata.version);

    let mut executor = TemplateExecutor::with_base_path(dir.to_str().unwrap());

    // The extraction target: a messy receipt line → structured record.
    let prompt = "Extract fields from this receipt:\n\
        STARBUCKS STORE #1123\n\
        2x Latte         9.00\n\
        1x Croissant     3.50\n\
        TOTAL           12.50 USD\n\
        03/15/2026";

    // JSON Schema → GBNF. The grammar forces every generated token to keep the
    // output on a path that completes a schema-valid JSON object.
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "merchant": { "type": "string" },
            "total":    { "type": "number" },
            "currency": { "enum": ["USD", "EUR", "GBP"] },
            "date":     { "type": "string" },
            "items":    { "type": "array", "items": { "type": "string" } }
        }
    });

    // ── Unconstrained baseline (greedy) ──────────────────────────────────────
    println!("\n── Unconstrained (greedy) ──");
    let base_cfg = GenerationConfig::greedy().with_max_tokens(160);
    let baseline = run(&mut executor, &metadata, prompt, Some(&base_cfg))?;
    println!("{baseline}");
    let base_valid = serde_json::from_str::<serde_json::Value>(baseline.trim()).is_ok();
    println!("→ parses as JSON: {base_valid}");

    // ── Grammar-constrained (the feature under test) ─────────────────────────
    println!("\n── Constrained (with_json_schema) ──");
    let grammar_cfg = GenerationConfig::greedy()
        .with_max_tokens(160)
        .with_json_schema(&schema)?;
    let constrained = run(&mut executor, &metadata, prompt, Some(&grammar_cfg))?;
    println!("{constrained}");

    // Hard assertions: the constrained output MUST be valid JSON of the right shape.
    let value: serde_json::Value = serde_json::from_str(constrained.trim())
        .map_err(|e| format!("constrained output is not valid JSON: {e}\nraw: {constrained:?}"))?;
    let obj = value
        .as_object()
        .ok_or("constrained output is valid JSON but not an object")?;

    let mut missing = Vec::new();
    for key in ["merchant", "total", "currency", "date", "items"] {
        if !obj.contains_key(key) {
            missing.push(key);
        }
    }
    if let Some(c) = obj.get("currency").and_then(|v| v.as_str()) {
        if !["USD", "EUR", "GBP"].contains(&c) {
            return Err(format!("currency '{c}' violates the enum constraint").into());
        }
    }
    if !obj.get("total").map(|v| v.is_number()).unwrap_or(false) {
        return Err("'total' is not a JSON number (grammar should force it)".into());
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("Status: PASS — constrained output is schema-valid JSON");
    println!("  merchant : {:?}", obj.get("merchant"));
    println!("  total    : {:?}", obj.get("total"));
    println!("  currency : {:?}", obj.get("currency"));
    println!("  items    : {:?}", obj.get("items"));
    if !missing.is_empty() {
        println!("  (note: keys not emitted by the model: {missing:?})");
    }
    println!("═══════════════════════════════════════════════════════");
    Ok(())
}
