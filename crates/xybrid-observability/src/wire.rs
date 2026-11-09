//! Wire representation of the contract's value types (CONTRACT.md "Wire
//! types" and "Paired observer and exporter adapters").
//!
//! - Every catalog attribute has a value type ([`attribute_type`]). The
//!   package's exporters encode `double` attributes as OTLP `doubleValue`
//!   (also for whole numbers) and `int` attributes as `intValue`
//!   ([`enforce_attribute_type`], [`enforce_any_value_type`]) for telemetry of
//!   the scope [`crate::semconv::SCOPE_NAME`] only.
//! - The OpenTelemetry Rust logs API has no null [`AnyValue`]. Between the
//!   observer and the paired exporter, JSON `null` in log event attributes
//!   (top level, map values and list elements) is carried as
//!   `AnyValue::Bytes(empty)` ([`null_value`]); the package never emits bytes
//!   otherwise. Log processors placed before the exporter must preserve such
//!   values. [`decode`] is the exporter-side view that turns the carrier back
//!   into a null; foreign exporters encode it as `bytesValue: ""`, which stays
//!   distinct from `stringValue: ""`.

use opentelemetry::logs::AnyValue;
use opentelemetry::{Array, Key, StringValue, Value as OtelValue};
use serde_json::Value;

use crate::semconv::attr::PROMPT_VARIABLE_PREFIX;

/// Value type of a catalog attribute (`catalog.attributes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CatalogType {
    /// `int`: OTLP `intValue`.
    Int,
    /// `double`: OTLP `doubleValue`, also for whole numbers.
    Double,
    /// `boolean`: OTLP `boolValue`.
    Boolean,
    /// `string`: OTLP `stringValue`.
    String,
    /// `string[]`: OTLP `arrayValue` of `stringValue`.
    StringArray,
    /// `json`: a compact JSON string on spans, a structured value on events.
    Json,
}

impl CatalogType {
    /// The catalog spelling (`int`, `double`, `boolean`, `string`,
    /// `string[]`, `json`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::Double => "double",
            Self::Boolean => "boolean",
            Self::String => "string",
            Self::StringArray => "string[]",
            Self::Json => "json",
        }
    }
}

/// The catalog value type of an attribute key, `None` for keys outside the
/// catalog (adapter extensions keep their own types).
#[must_use]
pub fn attribute_type(key: &str) -> Option<CatalogType> {
    if key.starts_with(PROMPT_VARIABLE_PREFIX) {
        return Some(CatalogType::String);
    }
    match key {
        "gen_ai.request.max_tokens"
        | "gen_ai.request.top_k"
        | "gen_ai.request.seed"
        | "gen_ai.request.choice.count"
        | "gen_ai.usage.input_tokens"
        | "gen_ai.usage.output_tokens"
        | "gen_ai.usage.cache_read.input_tokens"
        | "gen_ai.usage.cache_write.input_tokens"
        | "gen_ai.usage.reasoning.output_tokens"
        | "gen_ai.usage.text.input_tokens"
        | "gen_ai.usage.image.input_tokens"
        | "gen_ai.usage.audio.input_tokens"
        | "gen_ai.usage.text.output_tokens"
        | "gen_ai.usage.image.output_tokens"
        | "gen_ai.usage.audio.output_tokens"
        | "gen_ai.embeddings.dimension.count"
        | "server.port"
        | "xybrid.genai.response.chunk_count"
        | "xybrid.genai.attempt.index"
        | "xybrid.genai.attempt.count"
        | "xybrid.genai.model.size_bytes"
        | "xybrid.genai.resource.memory.used_bytes"
        | "xybrid.genai.resource.memory.min_available_bytes"
        | "xybrid.genai.resource.process.rss_bytes"
        | "xybrid.genai.resource.process.peak_rss_bytes"
        | "xybrid.genai.resource.accelerator.memory.used_bytes" => Some(CatalogType::Int),
        "gen_ai.request.temperature"
        | "gen_ai.request.top_p"
        | "gen_ai.request.frequency_penalty"
        | "gen_ai.request.presence_penalty"
        | "gen_ai.response.time_to_first_chunk"
        | "gen_ai.evaluation.score.value"
        | "xybrid.genai.response.time_to_first_token"
        | "xybrid.genai.response.time_per_output_token"
        | "xybrid.genai.resource.cpu.utilization"
        | "xybrid.genai.resource.cpu.peak_utilization"
        | "xybrid.genai.resource.battery.level"
        | "xybrid.genai.resource.accelerator.utilization" => Some(CatalogType::Double),
        "gen_ai.request.stream"
        | "xybrid.genai.model.cache_hit"
        | "xybrid.genai.resource.battery.charging" => Some(CatalogType::Boolean),
        "gen_ai.operation.name"
        | "gen_ai.provider.name"
        | "gen_ai.request.model"
        | "gen_ai.output.type"
        | "gen_ai.conversation.id"
        | "gen_ai.response.id"
        | "gen_ai.response.model"
        | "gen_ai.prompt.name"
        | "gen_ai.prompt.version"
        | "gen_ai.workflow.name"
        | "gen_ai.tool.name"
        | "gen_ai.tool.call.id"
        | "gen_ai.tool.type"
        | "gen_ai.tool.description"
        | "gen_ai.token.modality"
        | "gen_ai.evaluation.name"
        | "gen_ai.evaluation.score.label"
        | "gen_ai.evaluation.explanation"
        | "error.type"
        | "server.address"
        | "exception.type"
        | "exception.message"
        | "xybrid.genai.execution.location"
        | "xybrid.genai.runtime.name"
        | "xybrid.genai.runtime.version"
        | "xybrid.genai.accelerator"
        | "xybrid.genai.task"
        | "xybrid.genai.attempt.reason"
        | "xybrid.genai.model.format"
        | "xybrid.genai.resource.thermal.state"
        | "xybrid.genai.resource.memory.pressure"
        | "xybrid.genai.feedback.comment" => Some(CatalogType::String),
        "gen_ai.request.stop_sequences"
        | "gen_ai.response.finish_reasons"
        | "xybrid.genai.input.modalities"
        | "xybrid.genai.output.modalities"
        | "xybrid.genai.content.truncated_attributes" => Some(CatalogType::StringArray),
        "gen_ai.input.messages"
        | "gen_ai.output.messages"
        | "gen_ai.system_instructions"
        | "gen_ai.tool.call.arguments"
        | "gen_ai.tool.call.result"
        | "gen_ai.tool.definitions" => Some(CatalogType::Json),
        _ => None,
    }
}

/// Rewrites a span or metric attribute value to its catalog numeric type:
/// integers of `double` keys become doubles, integral doubles of `int` keys
/// become integers. Other values are left as they are.
pub fn enforce_attribute_type(key: &str, value: &mut OtelValue) {
    match (attribute_type(key), &*value) {
        (Some(CatalogType::Double), OtelValue::I64(n)) => *value = OtelValue::F64(*n as f64),
        (Some(CatalogType::Int), OtelValue::F64(f)) => {
            if let Some(n) = integral(*f) {
                *value = OtelValue::I64(n);
            }
        }
        _ => {}
    }
}

/// [`enforce_attribute_type`] for top-level log record attributes.
pub fn enforce_any_value_type(key: &str, value: &mut AnyValue) {
    match (attribute_type(key), &*value) {
        (Some(CatalogType::Double), AnyValue::Int(n)) => *value = AnyValue::Double(*n as f64),
        (Some(CatalogType::Int), AnyValue::Double(f)) => {
            if let Some(n) = integral(*f) {
                *value = AnyValue::Int(n);
            }
        }
        _ => {}
    }
}

pub(crate) fn integral(f: f64) -> Option<i64> {
    // `i64::MAX as f64` rounds up to 2^63, which is out of range.
    (f.is_finite() && f.fract() == 0.0 && f >= i64::MIN as f64 && f < i64::MAX as f64)
        .then_some(f as i64)
}

/// The carrier of JSON `null` in log attributes: `AnyValue::Bytes(empty)`.
#[must_use]
pub fn null_value() -> AnyValue {
    AnyValue::Bytes(Box::default())
}

/// Whether `value` is the null carrier.
#[must_use]
pub fn is_null(value: &AnyValue) -> bool {
    matches!(value, AnyValue::Bytes(b) if b.is_empty())
}

/// One level of a log attribute value as the paired exporter encodes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decoded<'a> {
    /// JSON null (the carrier): an OTLP `AnyValue` with no value set.
    Null,
    /// `boolValue`.
    Bool(bool),
    /// `intValue`.
    Int(i64),
    /// `doubleValue`.
    Double(f64),
    /// `stringValue`.
    String(&'a str),
    /// `bytesValue` (non-empty; never produced by this package).
    Bytes(&'a [u8]),
    /// `arrayValue`; decode each element in turn.
    List(&'a [AnyValue]),
    /// `kvlistValue`; decode each value in turn.
    Map(&'a std::collections::HashMap<Key, AnyValue>),
}

/// Decodes one level of a log attribute value, mapping the null carrier to
/// [`Decoded::Null`]. Pure and allocation-free; exporters recurse into
/// lists and maps.
#[must_use]
pub fn decode(value: &AnyValue) -> Decoded<'_> {
    match value {
        AnyValue::Int(n) => Decoded::Int(*n),
        AnyValue::Double(f) => Decoded::Double(*f),
        AnyValue::String(s) => Decoded::String(s.as_str()),
        AnyValue::Boolean(b) => Decoded::Bool(*b),
        AnyValue::Bytes(b) if b.is_empty() => Decoded::Null,
        AnyValue::Bytes(b) => Decoded::Bytes(b),
        AnyValue::ListAny(items) => Decoded::List(items),
        AnyValue::Map(map) => Decoded::Map(map),
        // `AnyValue` is non-exhaustive; no other variant exists in 0.33.
        _ => Decoded::Null,
    }
}

/// Converts a log attribute value to JSON, decoding the null carrier to
/// `null`. Non-empty bytes become an array of byte values.
#[must_use]
pub fn any_value_to_json(value: &AnyValue) -> Value {
    match decode(value) {
        Decoded::Null => Value::Null,
        Decoded::Bool(b) => Value::Bool(b),
        Decoded::Int(n) => Value::from(n),
        Decoded::Double(f) => serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number),
        Decoded::String(s) => Value::String(s.to_owned()),
        Decoded::Bytes(b) => Value::Array(b.iter().map(|x| Value::from(*x)).collect()),
        Decoded::List(items) => Value::Array(items.iter().map(any_value_to_json).collect()),
        Decoded::Map(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.as_str().to_owned(), any_value_to_json(v)))
                .collect(),
        ),
    }
}

/// Structured JSON → log `AnyValue`, with the null carrier for `null`.
pub(crate) fn json_to_any(value: Value) -> AnyValue {
    match value {
        Value::Null => null_value(),
        Value::Bool(b) => AnyValue::Boolean(b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => AnyValue::Int(i),
            None => AnyValue::Double(n.as_f64().unwrap_or_default()),
        },
        Value::String(s) => AnyValue::String(s.into()),
        Value::Array(items) => {
            AnyValue::ListAny(Box::new(items.into_iter().map(json_to_any).collect()))
        }
        Value::Object(map) => AnyValue::Map(Box::new(
            map.into_iter()
                .map(|(k, v)| (Key::new(k), json_to_any(v)))
                .collect(),
        )),
    }
}

/// A span/metric attribute value as a log `AnyValue` (same type).
pub(crate) fn otel_to_any(value: &OtelValue) -> AnyValue {
    match value {
        OtelValue::Bool(b) => AnyValue::Boolean(*b),
        OtelValue::I64(n) => AnyValue::Int(*n),
        OtelValue::F64(f) => AnyValue::Double(*f),
        OtelValue::String(s) => AnyValue::String(s.clone()),
        OtelValue::Array(Array::Bool(v)) => {
            AnyValue::ListAny(Box::new(v.iter().map(|b| AnyValue::Boolean(*b)).collect()))
        }
        OtelValue::Array(Array::I64(v)) => {
            AnyValue::ListAny(Box::new(v.iter().map(|n| AnyValue::Int(*n)).collect()))
        }
        OtelValue::Array(Array::F64(v)) => {
            AnyValue::ListAny(Box::new(v.iter().map(|f| AnyValue::Double(*f)).collect()))
        }
        OtelValue::Array(Array::String(v)) => {
            AnyValue::ListAny(Box::new(v.iter().cloned().map(AnyValue::String).collect()))
        }
        other => AnyValue::String(StringValue::from(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn null_carrier_round_trips_and_differs_from_empty_strings() {
        let v = json_to_any(json!({"a": null, "b": "", "c": [null, ""]}));
        assert_eq!(
            any_value_to_json(&v),
            json!({"a": null, "b": "", "c": [null, ""]})
        );
        assert!(is_null(&json_to_any(Value::Null)));
        assert!(!is_null(&json_to_any(json!(""))));
        assert_eq!(decode(&null_value()), Decoded::Null);
        assert_eq!(decode(&AnyValue::from("")), Decoded::String(""));
    }

    #[test]
    fn catalog_numeric_types_are_enforced() {
        let mut t = OtelValue::I64(0);
        enforce_attribute_type("gen_ai.request.temperature", &mut t);
        assert_eq!(t, OtelValue::F64(0.0));
        let mut p = OtelValue::F64(443.0);
        enforce_attribute_type("server.port", &mut p);
        assert_eq!(p, OtelValue::I64(443));
        let mut frac = OtelValue::F64(1.5);
        enforce_attribute_type("server.port", &mut frac);
        assert_eq!(frac, OtelValue::F64(1.5));
        let mut other = OtelValue::I64(1);
        enforce_attribute_type("app.custom", &mut other);
        assert_eq!(other, OtelValue::I64(1));
        let mut score = AnyValue::Int(1);
        enforce_any_value_type("gen_ai.evaluation.score.value", &mut score);
        assert_eq!(score, AnyValue::Double(1.0));
        assert_eq!(
            attribute_type("gen_ai.prompt.variable.topic"),
            Some(CatalogType::String)
        );
    }
}
