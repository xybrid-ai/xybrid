//! Inference result types for xybrid-sdk.
//!
//! This module provides `InferenceResult` - the output of model inference
//! with convenient accessors for different output types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xybrid_core::ir::{Envelope, EnvelopeKind};

use crate::model::SdkError;

/// A thumbs rating on a result (result flagging).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Rating {
    /// Keep this good result as a regression anchor.
    Up,
    /// "That's wrong" — mint a failure case.
    Down,
}

impl Rating {
    /// Wire label (`"up"` / `"down"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Rating::Up => "up",
            Rating::Down => "down",
        }
    }
}

/// Developer/user feedback on a result — the "flag" verb of the eval harness.
///
/// **Privacy default:** the `expected` correction and `note` are payload and are
/// only emitted with an explicit per-call opt-in ([`Feedback::capture`]). Without
/// it, [`InferenceResult::report`] emits a metadata-only event (trace id, model,
/// task, rating).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Feedback {
    /// Thumbs rating, if any.
    pub rating: Option<Rating>,
    /// A correction — the output that should have happened (payload).
    pub expected: Option<String>,
    /// A free-text note (payload).
    pub note: Option<String>,
    /// Per-call opt-in to capture `expected` / `note`. Default `false`
    /// (metadata-only).
    pub capture_payload: bool,
}

impl Feedback {
    /// A bare "that's wrong" flag (thumbs down, metadata-only).
    pub fn down() -> Self {
        Self {
            rating: Some(Rating::Down),
            ..Self::default()
        }
    }

    /// A "that's good" flag (thumbs up — keep as a regression anchor).
    pub fn up() -> Self {
        Self {
            rating: Some(Rating::Up),
            ..Self::default()
        }
    }

    /// A correction: thumbs down plus the expected output. The correction is a
    /// payload — it is only captured if [`Feedback::capture`] is also set.
    pub fn correction(expected: impl Into<String>) -> Self {
        Self {
            rating: Some(Rating::Down),
            expected: Some(expected.into()),
            ..Self::default()
        }
    }

    /// Attach a free-text note (payload — capture-gated).
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Opt in to capturing the payload (`expected` / `note`) on this call.
    pub fn capture(mut self) -> Self {
        self.capture_payload = true;
        self
    }
}

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
    /// Originating inference `trace_id`, when known. Lets `report()` join a
    /// `Feedback` event back to the trace that produced this result.
    trace_id: Option<String>,
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
            trace_id: None,
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
            trace_id: None,
        }
    }

    /// Attach the originating inference `trace_id` (builder). Used by the run
    /// paths so `report()` can join a `Feedback` event to the trace.
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
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

    /// The originating inference `trace_id`, if known.
    pub fn trace_id(&self) -> Option<&str> {
        self.trace_id.as_deref()
    }

    /// Flag this result for the eval harness — the "flag" verb of
    /// flag → collect → compare → gate → ship.
    ///
    /// Emits a `Feedback` telemetry event on the existing exporter (batching /
    /// circuit-breaker / retry for free). The event is **metadata-only**
    /// (`trace_id`, `model_id`, `task`, `rating`) unless `feedback` opts in to
    /// payload capture ([`Feedback::capture`]), and is suppressed entirely when
    /// telemetry is opted out / anonymous (no exporter → no emission).
    ///
    /// ```no_run
    /// # use xybrid_sdk::{result::InferenceResult, Feedback};
    /// # fn _ex(result: InferenceResult) -> Result<(), xybrid_sdk::SdkError> {
    /// result.report(Feedback::down())?;                   // "that's wrong"
    /// result.report(Feedback::correction("refund"))?;     // wrong + the fix
    /// result.report(Feedback::up())?;                     // keep good cases
    /// # Ok(())
    /// # }
    /// ```
    pub fn report(&self, feedback: Feedback) -> Result<(), SdkError> {
        let task = self.envelope.metadata.get("task").map(String::as_str);
        crate::telemetry::publish_feedback_event(
            self.trace_id.as_deref(),
            &self.model_id,
            task,
            feedback.rating.map(Rating::as_str),
            feedback.expected.as_deref(),
            feedback.note.as_deref(),
            feedback.capture_payload,
        );
        Ok(())
    }

    // ========================================================================
    // Continuous monitoring (continuous quality monitoring)
    // ========================================================================

    /// Compute the cheap structural quality guards (Tier A) for this result:
    /// empty, truncated, repetition loop, refusal, format validity. Reads only
    /// the output text + the backend finish reason — no judge, no user input.
    /// Non-text results yield the default (no issue).
    pub fn structural_signals(&self) -> crate::eval::monitor::StructuralSignals {
        match self.text() {
            Some(text) => {
                let finish_reason = self
                    .envelope
                    .metadata
                    .get("finish_reason")
                    .map(String::as_str);
                crate::eval::monitor::structural_signals(text, finish_reason, false)
            }
            None => crate::eval::monitor::StructuralSignals::default(),
        }
    }

    /// Auto-flag this result if any structural guard tripped — emits a
    /// metadata-only `Signal` telemetry event (the proactive complement to
    /// `report()`). Returns whether an issue was flagged. No-op when clean.
    pub fn flag_structural(&self) -> Result<bool, SdkError> {
        let signals = self.structural_signals();
        if !signals.has_issue() {
            return Ok(false);
        }
        let name = if signals.empty {
            "empty"
        } else if signals.truncated {
            "truncated"
        } else if signals.refusal_suspected {
            "refusal_suspected"
        } else if signals.format_valid == Some(false) {
            "format_invalid"
        } else {
            "repetition"
        };
        let extra = serde_json::json!({
            "empty": signals.empty,
            "truncated": signals.truncated,
            "repetition_score": signals.repetition_score,
            "refusal_suspected": signals.refusal_suspected,
        });
        let task = self.envelope.metadata.get("task").map(String::as_str);
        crate::telemetry::publish_signal_event(
            self.trace_id.as_deref(),
            &self.model_id,
            task,
            "structural",
            name,
            Some(extra),
        );
        Ok(true)
    }

    /// Record an implicit behavioral signal against this result (Tier B) — emits
    /// a metadata-only, `trace_id`-joinable `Signal` event. A `Regenerated`
    /// signal is treated as a soft 👎. See [`mark_regenerated`](Self::mark_regenerated)
    /// etc. for the per-signal shorthands.
    pub fn mark(&self, signal: crate::eval::monitor::BehavioralSignal) -> Result<(), SdkError> {
        let task = self.envelope.metadata.get("task").map(String::as_str);
        crate::telemetry::publish_signal_event(
            self.trace_id.as_deref(),
            &self.model_id,
            task,
            "behavioral",
            signal.as_str(),
            None,
        );
        Ok(())
    }

    /// The user accepted/consumed this result (a soft positive).
    pub fn mark_used(&self) -> Result<(), SdkError> {
        self.mark(crate::eval::monitor::BehavioralSignal::Used)
    }

    /// The user asked to regenerate (a soft negative — feeds the inbox).
    pub fn mark_regenerated(&self) -> Result<(), SdkError> {
        self.mark(crate::eval::monitor::BehavioralSignal::Regenerated)
    }

    /// The user edited this result before using it.
    pub fn mark_edited(&self) -> Result<(), SdkError> {
        self.mark(crate::eval::monitor::BehavioralSignal::Edited)
    }

    /// The user copied this result.
    pub fn mark_copied(&self) -> Result<(), SdkError> {
        self.mark(crate::eval::monitor::BehavioralSignal::Copied)
    }

    /// The user dismissed this result (a soft negative).
    pub fn mark_dismissed(&self) -> Result<(), SdkError> {
        self.mark(crate::eval::monitor::BehavioralSignal::Dismissed)
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

    #[test]
    fn feedback_constructors_set_expected_fields() {
        assert_eq!(Feedback::down().rating, Some(Rating::Down));
        assert_eq!(Feedback::up().rating, Some(Rating::Up));
        let c = Feedback::correction("refund");
        assert_eq!(c.rating, Some(Rating::Down));
        assert_eq!(c.expected.as_deref(), Some("refund"));
        // Corrections are metadata-only until explicitly captured.
        assert!(!c.capture_payload);
        assert!(c.clone().capture().capture_payload);
        assert_eq!(c.with_note("n").note.as_deref(), Some("n"));
    }

    #[test]
    fn with_trace_id_threads_onto_result() {
        let envelope = Envelope {
            kind: EnvelopeKind::Text("hi".to_string()),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "m", 10).with_trace_id("tr_42");
        assert_eq!(result.trace_id(), Some("tr_42"));
    }

    #[test]
    fn report_returns_ok_without_an_exporter() {
        // No exporter is registered in unit tests, so publish is a no-op; report
        // must still succeed for every feedback shape.
        let envelope = Envelope {
            kind: EnvelopeKind::Text("hi".to_string()),
            metadata: HashMap::new(),
        };
        let result = InferenceResult::new(envelope, "m", 10).with_trace_id("tr_1");
        assert!(result.report(Feedback::down()).is_ok());
        assert!(result.report(Feedback::up()).is_ok());
        assert!(result
            .report(Feedback::correction("refund").capture())
            .is_ok());
    }

    #[test]
    fn rating_wire_labels() {
        assert_eq!(Rating::Up.as_str(), "up");
        assert_eq!(Rating::Down.as_str(), "down");
    }

    fn text_result(text: &str, finish_reason: Option<&str>) -> InferenceResult {
        let mut metadata = HashMap::new();
        if let Some(fr) = finish_reason {
            metadata.insert("finish_reason".to_string(), fr.to_string());
        }
        let envelope = Envelope {
            kind: EnvelopeKind::Text(text.to_string()),
            metadata,
        };
        InferenceResult::new(envelope, "m", 10).with_trace_id("tr_1")
    }

    #[test]
    fn structural_signals_detect_truncation_and_repetition() {
        let truncated = text_result("a partial answer", Some("length"));
        assert!(truncated.structural_signals().truncated);

        let looping = text_result("go go go go go go", Some("stop"));
        assert!(looping.structural_signals().has_issue());

        let clean = text_result("The capital of France is Paris.", Some("stop"));
        assert!(!clean.structural_signals().has_issue());
    }

    #[test]
    fn flag_structural_returns_true_only_on_issue() {
        // No exporter in tests → publish is a no-op; we assert the detection
        // decision (the boolean), which is what drives the auto-flag.
        assert!(text_result("go go go go go go", Some("stop"))
            .flag_structural()
            .unwrap());
        assert!(!text_result("All good here.", Some("stop"))
            .flag_structural()
            .unwrap());
    }

    #[test]
    fn behavioral_marks_return_ok() {
        let r = text_result("hello", Some("stop"));
        assert!(r.mark_regenerated().is_ok());
        assert!(r.mark_used().is_ok());
        assert!(r.mark_edited().is_ok());
        assert!(r.mark_copied().is_ok());
        assert!(r.mark_dismissed().is_ok());
    }

    #[test]
    fn structural_signals_default_for_non_text() {
        let envelope = Envelope {
            kind: EnvelopeKind::Audio(vec![1, 2, 3]),
            metadata: HashMap::new(),
        };
        let r = InferenceResult::new(envelope, "tts", 5);
        assert!(!r.structural_signals().has_issue());
        assert!(!r.flag_structural().unwrap());
    }
}
