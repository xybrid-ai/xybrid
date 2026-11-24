//! Analyze explicit OTLP/JSON artifacts; never reads an implicit session log.
use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

fn scalar(value: &Value) -> Value {
    for key in ["stringValue", "boolValue", "doubleValue", "intValue"] {
        if let Some(value) = value.get(key) {
            return value.clone();
        }
    }
    Value::Null
}

fn attributes(span: &Value) -> BTreeMap<String, Value> {
    span["attributes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|kv| Some((kv["key"].as_str()?.to_owned(), scalar(&kv["value"]))))
        .collect()
}

fn integer(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_str()?.parse().ok())
}

pub(crate) fn summarize(trace: &Value) -> Result<Value> {
    let Some(resources) = trace["resourceSpans"].as_array() else {
        bail!("Expected an OTLP/JSON artifact. Create one with xybrid run --trace-export FILE; legacy session logs and Chrome traces are unsupported.");
    };
    let mut operations = Vec::new();
    let mut kinds = BTreeMap::<String, usize>::new();
    let mut input_tokens = None::<u64>;
    let mut output_tokens = None::<u64>;
    for resource in resources {
        for scope in resource["scopeSpans"].as_array().into_iter().flatten() {
            for span in scope["spans"].as_array().into_iter().flatten() {
                let attrs = attributes(span);
                let get = |key: &str| attrs.get(key).cloned().unwrap_or(Value::Null);
                let kind = get("gen_ai.operation.name");
                *kinds
                    .entry(kind.as_str().unwrap_or("internal").to_owned())
                    .or_default() += 1;
                let attempt = attrs.contains_key("xybrid.genai.attempt.reason")
                    || attrs.contains_key("xybrid.genai.attempt.index");
                // Attempts retain their own usage, but are never summed with requests.
                if !attempt
                    && matches!(
                        kind.as_str(),
                        Some("chat" | "generate_content" | "embeddings" | "predict")
                    )
                {
                    if let Some(tokens) = integer(&get("gen_ai.usage.input_tokens")) {
                        input_tokens = Some(input_tokens.unwrap_or(0).saturating_add(tokens));
                    }
                    if let Some(tokens) = integer(&get("gen_ai.usage.output_tokens")) {
                        output_tokens = Some(output_tokens.unwrap_or(0).saturating_add(tokens));
                    }
                }
                let elapsed = integer(&span["endTimeUnixNano"])
                    .zip(integer(&span["startTimeUnixNano"]))
                    .and_then(|(end, start)| end.checked_sub(start))
                    .map(|ns| ns as f64 / 1e6);
                operations.push(json!({
                    "name": span["name"], "trace_id": span["traceId"], "span_id": span["spanId"],
                    "parent_span_id": span["parentSpanId"], "kind": kind, "duration_ms": elapsed,
                    "model": get("gen_ai.request.model"), "location": get("xybrid.genai.execution.location"),
                    "attempt_reason": get("xybrid.genai.attempt.reason"), "route_target": get("xybrid.routing.target"),
                    "load_trigger": get("xybrid.load.trigger"), "error_type": get("error.type"),
                    "input_tokens": get("gen_ai.usage.input_tokens"), "output_tokens": get("gen_ai.usage.output_tokens"),
                    "first_chunk_seconds": get("gen_ai.response.time_to_first_chunk"),
                    "first_token_seconds": get("xybrid.genai.response.time_to_first_token"),
                }));
            }
        }
    }
    Ok(
        json!({"operation_counts": kinds, "logical_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}, "operations": operations}),
    )
}

pub(crate) fn analyze(file: &Path, export: Option<&Path>) -> Result<()> {
    let trace: Value = serde_json::from_slice(
        &std::fs::read(file).with_context(|| format!("Failed to read {}", file.display()))?,
    )?;
    let summary = summarize(&trace)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if let Some(export) = export {
        std::fs::write(export, serde_json::to_vec_pretty(&summary)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn summary_excludes_attempt_usage_and_governed_content() {
        let span = |attempt: bool| {
            json!({"name": "chat model", "startTimeUnixNano": "1000000", "endTimeUnixNano": "3000000", "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "7"}},
                {"key": if attempt {"xybrid.genai.attempt.reason"} else {"unused"}, "value": {"stringValue": "initial"}},
                {"key": "gen_ai.input.messages", "value": {"stringValue": "private prompt"}}
            ]})
        };
        let summary = summarize(
            &json!({"resourceSpans": [{"scopeSpans": [{"spans": [span(false), span(true)]}]}]}),
        )
        .unwrap();
        assert_eq!(summary["logical_usage"]["input_tokens"], 7);
        assert_eq!(summary["logical_usage"]["output_tokens"], Value::Null);
        assert_eq!(summary["operations"][0]["duration_ms"], 2.0);
        assert!(!summary.to_string().contains("private prompt"));
    }
}
