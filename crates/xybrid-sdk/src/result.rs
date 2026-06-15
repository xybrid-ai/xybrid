//! Inference result types for xybrid-sdk.
//!
//! This module provides `InferenceResult` - the output of model inference
//! with convenient accessors for different output types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xybrid_core::ir::{Envelope, EnvelopeKind};

/// Per-stage latency entry for pipeline runs.
///
/// One entry per executed stage; the `stage_id` matches the stage name in the
/// pipeline definition.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StageLatency {
    pub stage_id: String,
    pub latency_ms: u32,
}

/// Typed inference metrics surfaced on every `InferenceResult`.
///
/// LLM-specific fields (`ttft_ms`, `tokens_per_second`, `prefill_tps`,
/// `decode_tps`, `tokens_out`) are `None` for ASR/TTS/embedding runs.
/// `image_preprocess_ms` is populated only for vision-language runs that
/// process one or more images.
/// `stage_latencies_ms` is empty for `model.run()` and populated for
/// `pipeline.run()`.
///
/// Population is best-effort: fields parse from the `Envelope.metadata`
/// string map written by `runtime_adapter::llm` and `execution::executor`.
/// Local LLM runs populate the LLM fields; cloud LLM runs currently surface
/// only `total_ms` (the cloud adapter writes `backend` to envelope metadata
/// but not the per-run scalars — those ride on span metadata today).
/// Unparseable values become `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct InferenceMetrics {
    /// Wall-clock latency in ms (mirrors `InferenceResult.latency_ms`).
    pub total_ms: u32,
    /// Time to first token, ms. LLM streaming only.
    pub ttft_ms: Option<u32>,
    /// Generation throughput, tokens/sec. LLM only.
    pub tokens_per_second: Option<f32>,
    /// Prefill phase tok/s. LLM only.
    pub prefill_tps: Option<f32>,
    /// Decode phase tok/s. LLM only.
    pub decode_tps: Option<f32>,
    /// Completion tokens produced. LLM only.
    pub tokens_out: Option<u32>,
    /// Image preprocessing latency in ms. Vision-language runs only.
    pub image_preprocess_ms: Option<u32>,
    /// Per-stage wall-clock latencies. Empty for single-model runs.
    pub stage_latencies_ms: Vec<StageLatency>,
}

impl InferenceMetrics {
    /// Build metrics from an envelope's metadata map.
    ///
    /// `total_ms` is passed in from the caller's outer latency measurement
    /// (envelope metadata doesn't carry it). LLM keys that are absent or
    /// fail to parse become `None`. `stage_latencies_ms` is left empty —
    /// pipeline call sites populate it from their `FfiStageExecutionResult`
    /// list.
    pub fn from_metadata(metadata: &HashMap<String, String>, total_ms: u32) -> Self {
        Self {
            total_ms,
            ttft_ms: parse_u32(metadata, "ttft_ms"),
            tokens_per_second: parse_f32(metadata, "tokens_per_second"),
            prefill_tps: parse_f32(metadata, "prefill_tps"),
            decode_tps: parse_f32(metadata, "decode_tps"),
            tokens_out: parse_u32(metadata, "tokens_out")
                .or_else(|| parse_u32(metadata, "tokens_generated")),
            image_preprocess_ms: parse_u32(metadata, "image_preprocess_ms"),
            stage_latencies_ms: Vec::new(),
        }
    }
}

fn parse_u32(metadata: &HashMap<String, String>, key: &str) -> Option<u32> {
    metadata.get(key).and_then(|v| v.parse::<u32>().ok())
}

fn parse_f32(metadata: &HashMap<String, String>, key: &str) -> Option<f32> {
    metadata.get(key).and_then(|v| v.parse::<f32>().ok())
}

/// Output type enumeration for model inference results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Text output (ASR transcription, NLP results)
    Text,
    /// Audio output (TTS synthesis, audio processing)
    Audio,
    /// Embedding output (vector representation)
    Embedding,
    /// Unknown or custom output type
    Unknown,
}

impl std::fmt::Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputType::Text => write!(f, "text"),
            OutputType::Audio => write!(f, "audio"),
            OutputType::Embedding => write!(f, "embedding"),
            OutputType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Result from model.run() or pipeline.run().
///
/// Provides type-safe accessors for different output types with both
/// safe (Option-returning) and panicking (unwrap) variants.
///
/// # Example
///
/// ```no_run
/// # use xybrid_sdk::{XybridModel, ir::Envelope, result::OutputType};
/// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
/// # let model: XybridModel = unimplemented!();
/// # let envelope: Envelope = unimplemented!();
/// let result = model.run(&envelope, None)?;
///
/// // Check output type
/// match result.output_type() {
///     OutputType::Text => println!("Text: {}", result.unwrap_text()),
///     OutputType::Audio => println!("Audio: {} bytes", result.unwrap_audio().len()),
///     OutputType::Embedding => println!("Embedding: {} dims", result.unwrap_embedding().len()),
///     OutputType::Unknown => println!("Unknown output"),
/// }
///
/// // Or use safe accessors
/// if let Some(text) = result.text() {
///     println!("Transcription: {}", text);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The underlying envelope containing the result
    envelope: Envelope,
    /// Inferred output type
    output_type: OutputType,
    /// Inference latency in milliseconds
    latency_ms: u32,
    /// Model ID that produced this result
    model_id: String,
    /// Typed metrics parsed from `envelope.metadata`
    metrics: InferenceMetrics,
}

impl InferenceResult {
    /// Create a new inference result from an envelope.
    pub fn new(envelope: Envelope, model_id: impl Into<String>, latency_ms: u32) -> Self {
        let output_type = output_type_for_envelope(&envelope);
        let metrics = InferenceMetrics::from_metadata(&envelope.metadata, latency_ms);

        Self {
            envelope,
            output_type,
            latency_ms,
            model_id: model_id.into(),
            metrics,
        }
    }

    /// Create from envelope with pre-computed output type.
    pub fn with_output_type(
        envelope: Envelope,
        output_type: OutputType,
        model_id: impl Into<String>,
        latency_ms: u32,
    ) -> Self {
        let metrics = InferenceMetrics::from_metadata(&envelope.metadata, latency_ms);
        Self {
            envelope,
            output_type,
            latency_ms,
            model_id: model_id.into(),
            metrics,
        }
    }

    // ========================================================================
    // Properties
    // ========================================================================

    /// Get the output type of this result.
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }

    /// Get the inference latency in milliseconds.
    pub fn latency_ms(&self) -> u32 {
        self.latency_ms
    }

    /// Get the model ID that produced this result.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Typed metrics for this run (TTFT, tok/s, per-stage latencies, etc.).
    pub fn metrics(&self) -> &InferenceMetrics {
        &self.metrics
    }

    /// Get a reference to the underlying envelope.
    pub fn envelope(&self) -> &Envelope {
        &self.envelope
    }

    /// Consume self and return the underlying envelope.
    pub fn into_envelope(self) -> Envelope {
        self.envelope
    }

    // ========================================================================
    // Safe Accessors (return Option)
    // ========================================================================

    /// Get text output if available.
    ///
    /// Returns `None` if the output is not text.
    pub fn text(&self) -> Option<&str> {
        match &self.envelope.kind {
            EnvelopeKind::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Get audio bytes if available.
    ///
    /// Returns `None` if the output is not audio.
    pub fn audio_bytes(&self) -> Option<&[u8]> {
        match &self.envelope.kind {
            EnvelopeKind::Audio(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Get embedding vector if available.
    ///
    /// Returns `None` if the output is not an embedding.
    pub fn embedding(&self) -> Option<&[f32]> {
        match &self.envelope.kind {
            EnvelopeKind::Embedding(vec) => Some(vec),
            _ => None,
        }
    }

    // ========================================================================
    // Unwrap Accessors (panic on wrong type)
    // ========================================================================

    /// Get text output, panicking if not text.
    ///
    /// # Panics
    ///
    /// Panics if the output type is not `Text`.
    pub fn unwrap_text(&self) -> &str {
        self.text().expect("InferenceResult is not Text type")
    }

    /// Get audio bytes, panicking if not audio.
    ///
    /// # Panics
    ///
    /// Panics if the output type is not `Audio`.
    pub fn unwrap_audio(&self) -> &[u8] {
        self.audio_bytes()
            .expect("InferenceResult is not Audio type")
    }

    /// Get embedding vector, panicking if not embedding.
    ///
    /// # Panics
    ///
    /// Panics if the output type is not `Embedding`.
    pub fn unwrap_embedding(&self) -> &[f32] {
        self.embedding()
            .expect("InferenceResult is not Embedding type")
    }

    // ========================================================================
    // Convenience Methods
    // ========================================================================

    /// Check if this result contains text.
    pub fn is_text(&self) -> bool {
        self.output_type == OutputType::Text
    }

    /// Check if this result contains audio.
    pub fn is_audio(&self) -> bool {
        self.output_type == OutputType::Audio
    }

    /// Check if this result contains an embedding.
    pub fn is_embedding(&self) -> bool {
        self.output_type == OutputType::Embedding
    }

    /// Get metadata value from the envelope.
    pub fn metadata(&self, key: &str) -> Option<&String> {
        self.envelope.metadata.get(key)
    }

    /// Get all metadata.
    pub fn all_metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.envelope.metadata
    }
}

pub(crate) fn output_type_for_envelope(envelope: &Envelope) -> OutputType {
    match &envelope.kind {
        EnvelopeKind::Text(_) => OutputType::Text,
        EnvelopeKind::Audio(_) => OutputType::Audio,
        EnvelopeKind::Embedding(_) => OutputType::Embedding,
        EnvelopeKind::Image { .. } | EnvelopeKind::MultiPart(_) => OutputType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_result() {
        let envelope = Envelope {
            kind: EnvelopeKind::Text("hello world".to_string()),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "test-model", 100);

        assert_eq!(result.output_type(), OutputType::Text);
        assert!(result.is_text());
        assert!(!result.is_audio());
        assert_eq!(result.text(), Some("hello world"));
        assert_eq!(result.unwrap_text(), "hello world");
        assert_eq!(result.audio_bytes(), None);
        assert_eq!(result.latency_ms(), 100);
        assert_eq!(result.model_id(), "test-model");
    }

    #[test]
    fn test_audio_result() {
        let envelope = Envelope {
            kind: EnvelopeKind::Audio(vec![1, 2, 3, 4]),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "tts-model", 50);

        assert_eq!(result.output_type(), OutputType::Audio);
        assert!(result.is_audio());
        assert!(!result.is_text());
        assert_eq!(result.audio_bytes(), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(result.text(), None);
    }

    #[test]
    fn test_embedding_result() {
        let envelope = Envelope {
            kind: EnvelopeKind::Embedding(vec![0.1, 0.2, 0.3]),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "embed-model", 25);

        assert_eq!(result.output_type(), OutputType::Embedding);
        assert!(result.is_embedding());
        assert_eq!(result.embedding(), Some(&[0.1f32, 0.2, 0.3][..]));
        assert_eq!(result.unwrap_embedding().len(), 3);
    }

    #[test]
    #[should_panic(expected = "InferenceResult is not Text type")]
    fn test_unwrap_wrong_type() {
        let envelope = Envelope {
            kind: EnvelopeKind::Audio(vec![1, 2, 3]),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "model", 0);
        result.unwrap_text(); // Should panic
    }

    #[test]
    fn test_metrics_parsed_from_envelope_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("ttft_ms".to_string(), "120".to_string());
        metadata.insert("tokens_per_second".to_string(), "42.50".to_string());
        metadata.insert("prefill_tps".to_string(), "180.0".to_string());
        metadata.insert("decode_tps".to_string(), "42.5".to_string());
        metadata.insert("tokens_generated".to_string(), "256".to_string());
        metadata.insert("image_preprocess_ms".to_string(), "17".to_string());

        let envelope = Envelope {
            kind: EnvelopeKind::Text("hi".to_string()),
            metadata,
        };
        let result = InferenceResult::new(envelope, "llm-model", 500);

        let m = result.metrics();
        assert_eq!(m.total_ms, 500);
        assert_eq!(m.ttft_ms, Some(120));
        assert_eq!(m.tokens_per_second, Some(42.5));
        assert_eq!(m.prefill_tps, Some(180.0));
        assert_eq!(m.decode_tps, Some(42.5));
        assert_eq!(m.tokens_out, Some(256));
        assert_eq!(m.image_preprocess_ms, Some(17));
        assert!(m.stage_latencies_ms.is_empty());
    }

    #[test]
    fn test_metrics_missing_keys_default_to_none() {
        let envelope = Envelope {
            kind: EnvelopeKind::Audio(vec![1, 2]),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "tts-model", 50);
        let m = result.metrics();
        assert_eq!(m.total_ms, 50);
        assert_eq!(m.ttft_ms, None);
        assert_eq!(m.tokens_per_second, None);
        assert_eq!(m.tokens_out, None);
        assert_eq!(m.image_preprocess_ms, None);
    }

    #[test]
    fn test_metrics_unparseable_values_become_none() {
        let mut metadata = HashMap::new();
        metadata.insert("ttft_ms".to_string(), "not-a-number".to_string());
        metadata.insert("tokens_per_second".to_string(), "nan-ish".to_string());

        let envelope = Envelope {
            kind: EnvelopeKind::Text("x".to_string()),
            metadata,
        };
        let result = InferenceResult::new(envelope, "m", 10);
        let m = result.metrics();
        assert_eq!(m.ttft_ms, None);
        assert_eq!(m.tokens_per_second, None);
    }

    #[test]
    fn test_metrics_tokens_out_canonical_key_wins_over_alias() {
        let mut metadata = HashMap::new();
        metadata.insert("tokens_out".to_string(), "64".to_string());
        metadata.insert("tokens_generated".to_string(), "999".to_string());

        let envelope = Envelope {
            kind: EnvelopeKind::Text("x".to_string()),
            metadata,
        };
        let result = InferenceResult::new(envelope, "m", 10);
        assert_eq!(result.metrics().tokens_out, Some(64));
    }

    #[test]
    fn test_into_envelope() {
        let envelope = Envelope {
            kind: EnvelopeKind::Text("test".to_string()),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "model", 0);
        let recovered = result.into_envelope();

        match recovered.kind {
            EnvelopeKind::Text(text) => assert_eq!(text, "test"),
            _ => panic!("Expected Text"),
        }
    }
}
