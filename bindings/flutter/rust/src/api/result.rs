//! Inference result FFI wrappers for Flutter.
use xybrid_sdk::{InferenceMetrics, InferenceResult, StageLatency};

/// Per-stage latency entry for pipeline runs.
///
/// Mirrors `xybrid_sdk::StageLatency`. One entry per executed stage; the
/// `stage_id` matches the stage name in the pipeline definition.
#[derive(Clone)]
pub struct FfiStageLatency {
    pub stage_id: String,
    pub latency_ms: u32,
}

impl FfiStageLatency {
    pub(crate) fn from_core(s: &StageLatency) -> Self {
        Self {
            stage_id: s.stage_id.clone(),
            latency_ms: s.latency_ms,
        }
    }
}

/// Typed inference metrics.
///
/// Mirrors `xybrid_sdk::InferenceMetrics`. LLM-specific fields are `None`
/// for ASR/TTS/embedding runs. `stage_latencies_ms` is empty for
/// `model.run()` and populated for `pipeline.run()`.
#[derive(Clone)]
pub struct FfiInferenceMetrics {
    pub total_ms: u32,
    pub ttft_ms: Option<u32>,
    pub tokens_per_second: Option<f32>,
    pub prefill_tps: Option<f32>,
    pub decode_tps: Option<f32>,
    pub tokens_out: Option<u32>,
    pub stage_latencies_ms: Vec<FfiStageLatency>,
}

impl FfiInferenceMetrics {
    pub(crate) fn from_core(m: &InferenceMetrics) -> Self {
        Self {
            total_ms: m.total_ms,
            ttft_ms: m.ttft_ms,
            tokens_per_second: m.tokens_per_second,
            prefill_tps: m.prefill_tps,
            decode_tps: m.decode_tps,
            tokens_out: m.tokens_out,
            stage_latencies_ms: m
                .stage_latencies_ms
                .iter()
                .map(FfiStageLatency::from_core)
                .collect(),
        }
    }
}

/// FFI wrapper for inference results.
/// Fields are public and accessible directly via FRB-generated bindings.
#[derive(Clone)]
pub struct FfiResult {
    pub success: bool,
    pub text: Option<String>,
    pub audio_bytes: Option<Vec<u8>>,
    pub embedding: Option<Vec<f32>>,
    pub latency_ms: u32,
    pub metrics: FfiInferenceMetrics,
}

impl FfiResult {
    pub(crate) fn from_inference_result(r: &InferenceResult) -> Self {
        Self {
            success: true,
            text: r.text().map(|s| s.to_string()),
            audio_bytes: r.audio_bytes().map(|b| b.to_vec()),
            embedding: r.embedding().map(|e| e.to_vec()),
            latency_ms: r.latency_ms(),
            metrics: FfiInferenceMetrics::from_core(r.metrics()),
        }
    }
}
