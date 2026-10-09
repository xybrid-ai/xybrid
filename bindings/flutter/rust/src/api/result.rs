//! Inference result FFI wrappers for Flutter.
use xybrid_ffi_facade as facade;
use xybrid_sdk::ir::EnvelopeKind;
use xybrid_sdk::{InferenceMetrics, InferenceResult, StageLatency};

use super::model::FfiToolCall;

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
    pub image_preprocess_ms: Option<u32>,
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
            image_preprocess_ms: m.image_preprocess_ms,
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
    /// Model chain-of-thought / reasoning (`<think>` blocks), surfaced
    /// separately from `text`, which always excludes it. `None` when the
    /// model emitted no reasoning.
    pub reasoning_content: Option<String>,
    pub audio_bytes: Option<Vec<u8>>,
    pub embedding: Option<Vec<f32>>,
    pub latency_ms: u32,
    /// Whether this answer came from the device or the cloud gateway.
    pub execution_target: FfiExecutionTarget,
    pub metrics: FfiInferenceMetrics,
    /// Tool calls the model asked for this turn.
    ///
    /// Empty unless the request offered tools via
    /// `FfiGenerationConfig.tools`. Run each call yourself, then feed the
    /// outcomes back with `FfiEnvelope::tool_results` — one run is one model
    /// turn. The raw tool-call block stays in `text` untouched, and malformed
    /// model output yields an empty list rather than an error.
    pub tool_calls: Vec<FfiToolCall>,
}

/// Check that `FfiResult` has a field for an output of this kind.
///
/// `FfiResult` carries text, audio or an embedding. Any other output would
/// come back as `success: true` with all three empty, so the run fails
/// instead, with the error the other bindings report.
pub(crate) fn ensure_ffi_payload(kind: &EnvelopeKind) -> Result<(), String> {
    let payload = match kind {
        EnvelopeKind::Text(_) | EnvelopeKind::Audio(_) | EnvelopeKind::Embedding(_) => {
            return Ok(())
        }
        EnvelopeKind::Image { .. } => "an image",
        EnvelopeKind::MultiPart(_) => "a multi-part message",
        EnvelopeKind::ChoiceRequest(_) => "a choice request",
        EnvelopeKind::ChoiceScores(_) => "choice scores",
    };
    Err(facade::Error::UnsupportedModelCapability {
        message: format!("{payload} cannot be returned through the Flutter bindings yet"),
    }
    .to_string())
}

impl FfiResult {
    /// Convert an SDK result, failing if its output has no `FfiResult` field
    /// (see [`ensure_ffi_payload`]).
    pub(crate) fn try_from_inference_result(r: &InferenceResult) -> Result<Self, String> {
        ensure_ffi_payload(&r.envelope().kind)?;
        Ok(Self {
            success: true,
            text: r.text().map(|s| s.to_string()),
            reasoning_content: r.reasoning_content().map(|s| s.to_string()),
            audio_bytes: r.audio_bytes().map(|b| b.to_vec()),
            embedding: r.embedding().map(|e| e.to_vec()),
            latency_ms: r.latency_ms(),
            execution_target: FfiExecutionTarget::from_sdk(r.provenance()),
            metrics: FfiInferenceMetrics::from_core(r.metrics()),
            tool_calls: r
                .tool_calls()
                .into_iter()
                .map(|call| FfiToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments_json: call.function.arguments,
                })
                .collect(),
        })
    }
}

/// Where a result was produced — the observed fact, not a routing preference.
///
/// Cloud fallback (speculative or reactive) keeps the model id identical on
/// both legs by design, so this is the only way to tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiExecutionTarget {
    Local,
    Cloud,
}

impl FfiExecutionTarget {
    fn from_sdk(provenance: xybrid_sdk::ExecutionProvenance) -> Self {
        match provenance {
            xybrid_sdk::ExecutionProvenance::Local => Self::Local,
            xybrid_sdk::ExecutionProvenance::Cloud => Self::Cloud,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_sdk::ir::{Envelope, ImagePlane, PixelFormat};

    fn result_of(kind: EnvelopeKind) -> InferenceResult {
        InferenceResult::new(Envelope::new(kind), "m", 7)
    }

    fn raw_pixel_image() -> Envelope {
        Envelope::image_raw(
            vec![17, 34, 51],
            PixelFormat::Rgb8,
            1,
            1,
            vec![ImagePlane {
                offset: 0,
                row_stride: 3,
                pixel_stride: 3,
                width: 1,
                height: 1,
            }],
            None,
        )
        .expect("a 1x1 RGB frame is valid")
    }

    /// `FfiResult` has no `Debug`, so `Result::expect` is out.
    fn converted(kind: EnvelopeKind) -> FfiResult {
        let Ok(result) = FfiResult::try_from_inference_result(&result_of(kind)) else {
            panic!("text, audio and embeddings convert");
        };
        result
    }

    #[test]
    fn text_audio_and_embedding_results_convert_as_before() {
        let text = converted(EnvelopeKind::Text("hello".into()));
        assert_eq!(text.text.as_deref(), Some("hello"));
        assert!(text.success && text.audio_bytes.is_none() && text.embedding.is_none());
        assert_eq!(text.latency_ms, 7);

        let audio = converted(EnvelopeKind::Audio(vec![1, 2]));
        assert_eq!(audio.audio_bytes.as_deref(), Some([1u8, 2].as_slice()));

        let embedding = converted(EnvelopeKind::Embedding(vec![0.5]));
        assert_eq!(embedding.embedding.as_deref(), Some([0.5f32].as_slice()));
    }

    /// Before, these came back as `success: true` with every payload field
    /// empty.
    #[test]
    fn a_result_with_no_ffi_field_for_its_output_fails() {
        let unsupported = [
            (
                "a multi-part message",
                result_of(EnvelopeKind::MultiPart(vec![])),
            ),
            ("an image", InferenceResult::new(raw_pixel_image(), "m", 0)),
            (
                "a choice request",
                InferenceResult::new(
                    Envelope::choice_request(Envelope::new(EnvelopeKind::Text("c".into())), vec![]),
                    "m",
                    0,
                ),
            ),
            (
                "choice scores",
                result_of(EnvelopeKind::ChoiceScores(xybrid_sdk::ChoiceScores {
                    entries: vec![],
                    label_mass: None,
                })),
            ),
        ];
        for (payload, result) in unsupported {
            let Err(error) = FfiResult::try_from_inference_result(&result) else {
                panic!("an output FfiResult cannot carry must fail");
            };
            assert!(
                error.starts_with("Unsupported model capability:") && error.contains(payload),
                "{error}"
            );
        }
    }
}
