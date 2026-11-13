//! Pipeline FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::sync::Arc;
use xybrid_sdk::{InferenceMetrics, Pipeline, PipelineExecutionResult, PipelineRef};

use super::envelope::FfiEnvelope;
use super::observability::FfiObservationOptions;
use super::result::{ensure_ffi_payload, FfiExecutionTarget, FfiResult, FfiRunResult};

fn final_execution_target(target: Option<&str>) -> FfiExecutionTarget {
    match target {
        Some("device" | "local") | None => FfiExecutionTarget::Local,
        Some(_) => FfiExecutionTarget::Cloud,
    }
}

/// Build the Dart result of a pipeline run from its final output, failing
/// when `FfiResult` has no field for that output (see [`ensure_ffi_payload`]).
fn pipeline_ffi_result(result: &PipelineExecutionResult) -> Result<FfiResult, String> {
    ensure_ffi_payload(&result.output.kind)?;

    let mut metrics =
        InferenceMetrics::from_metadata(&result.output.metadata, result.total_latency_ms);
    metrics.stage_latencies_ms = result
        .stages
        .iter()
        .map(|s| xybrid_sdk::StageLatency {
            stage_id: s.name.clone(),
            latency_ms: s.latency_ms,
        })
        .collect();

    let mut ffi_metrics = crate::api::result::FfiInferenceMetrics::from_core(&metrics);
    for (entry, stage) in ffi_metrics
        .stage_latencies_ms
        .iter_mut()
        .zip(&result.stages)
    {
        entry.observation = stage
            .observation_ref
            .as_ref()
            .map(super::observability::FfiObservationRef::from_sdk);
    }

    Ok(FfiResult {
        observation: result
            .observation_ref
            .as_ref()
            .map(super::observability::FfiObservationRef::from_sdk),
        success: true,
        text: result.text().map(|s| s.to_string()),
        reasoning_content: result.output.metadata.get("reasoning_content").cloned(),
        audio_bytes: result.audio_bytes().map(|b| b.to_vec()),
        embedding: result.embedding().map(|e| e.to_vec()),
        latency_ms: result.total_latency_ms,
        // A pipeline can mix local and remote stages, so there is no single
        // provenance for the run: report the final stage's target, which is
        // what produced `output`. The stage records it directly; the
        // envelope carries no `execution_target` key on this path, since
        // pipeline stages do not go through `InferenceResult`.
        //
        // The orchestrator records `local`, `cloud` or `fallback:<id>` (a
        // xybrid-hosted server). Only the explicitly local spellings may
        // be reported as on-device; unknown targets fail closed to cloud.
        execution_target: final_execution_target(
            result.stages.last().map(|stage| stage.target.as_str()),
        ),
        metrics: ffi_metrics,
        // Pipelines don't offer tools, so a pipeline run never produces
        // tool calls.
        tool_calls: Vec::new(),
    })
}

fn pipeline_options(
    observation: Option<&FfiObservationOptions>,
) -> Result<xybrid_sdk::RunOptions, String> {
    Ok(xybrid_sdk::RunOptions {
        observation: observation
            .map(FfiObservationOptions::to_sdk)
            .transpose()?
            .flatten(),
        ..Default::default()
    })
}

/// FFI wrapper for a loaded Pipeline ready for execution.
#[frb(opaque)]
pub struct FfiPipeline(Arc<Pipeline>);

impl FfiPipeline {
    /// Load a pipeline from a YAML string.
    ///
    /// Parses the YAML and resolves all model references via the registry.
    #[frb(sync)]
    pub fn from_yaml(yaml: String) -> Result<FfiPipeline, String> {
        let pipeline_ref = PipelineRef::from_yaml(&yaml).map_err(|e| e.to_string())?;
        let pipeline = pipeline_ref.load().map_err(|e| e.to_string())?;
        Ok(FfiPipeline(Arc::new(pipeline)))
    }

    /// Load a pipeline from a YAML file path.
    ///
    /// Reads the file, parses the YAML, and resolves all model references.
    #[frb(sync)]
    pub fn from_file(path: String) -> Result<FfiPipeline, String> {
        let pipeline_ref = PipelineRef::from_file(&path).map_err(|e| e.to_string())?;
        let pipeline = pipeline_ref.load().map_err(|e| e.to_string())?;
        Ok(FfiPipeline(Arc::new(pipeline)))
    }

    /// Load a pipeline from a bundle path.
    ///
    /// Note: Currently delegates to from_file as bundles are YAML-based.
    /// In future, this may support additional bundle formats (e.g., .xpkg).
    #[frb(sync)]
    pub fn from_bundle(path: String) -> Result<FfiPipeline, String> {
        // Bundle format is currently just a YAML file
        Self::from_file(path)
    }

    /// Execute the pipeline with the given input envelope.
    ///
    /// Returns the final output with the workflow reference and each stage inference reference.
    /// Observation controls apply to the entire pipeline and are validated before execution.
    ///
    /// LLM-specific metric fields (`ttft_ms`, `tokens_per_second`,
    /// `prefill_tps`, `decode_tps`, `tokens_out`) are parsed from the
    /// **final** stage envelope's metadata, so they are `None` whenever the
    /// final stage isn't the LLM — e.g. an `ASR → LLM → TTS` pipeline
    /// produces a TTS envelope and the LLM metrics are not surfaced here.
    /// Hoisting them from intermediate stages is a follow-up at the SDK
    /// layer.
    pub fn run(
        &self,
        envelope: FfiEnvelope,
        observation: Option<FfiObservationOptions>,
    ) -> Result<FfiRunResult, String> {
        let options = pipeline_options(observation.as_ref())?;
        let input = envelope.into_envelope();
        let result = self
            .0
            .run_with_options(&input, &options)
            .map_err(|e| e.to_string())?;
        pipeline_ffi_result(&result).map(FfiRunResult::Legacy)
    }

    /// Get the pipeline name (if specified in YAML).
    #[frb(sync)]
    pub fn name(&self) -> Option<String> {
        self.0.name().map(|s| s.to_string())
    }

    /// Get the stage names/identifiers in execution order.
    #[frb(sync)]
    pub fn stage_names(&self) -> Vec<String> {
        self.0.stage_names()
    }

    /// Get the number of stages in the pipeline.
    #[frb(sync)]
    pub fn stage_count(&self) -> usize {
        self.0.stage_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_controls_are_validated_and_preserved_for_the_pipeline() {
        let controls = FfiObservationOptions {
            traceparent: Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00".into()),
            tracestate: Some("vendor=value".into()),
            conversation_id: Some("pipeline-turn".into()),
            suppress_trace: true,
        };
        let options = pipeline_options(Some(&controls)).unwrap();
        let observation = options.observation.unwrap();
        let parent = observation.parent.unwrap();
        assert_eq!(parent.to_traceparent(), controls.traceparent.unwrap());
        assert_eq!(parent.tracestate(), Some("vendor=value"));
        assert_eq!(
            observation.conversation_id.as_deref(),
            Some("pipeline-turn")
        );
        assert_eq!(
            observation.trace,
            xybrid_sdk::observability::TraceMode::Suppress
        );
        assert!(pipeline_options(None).unwrap().observation.is_none());
        let invalid = FfiObservationOptions {
            traceparent: Some("invalid".into()),
            ..Default::default()
        };
        assert!(pipeline_options(Some(&invalid)).is_err());
    }

    #[test]
    fn pipeline_conversion_retains_distinct_workflow_and_stage_references() {
        let mut output = pipeline_ending_in(xybrid_sdk::ir::EnvelopeKind::Text("answer".into()));
        let workflow = xybrid_sdk::ObservationRef {
            trace_id: "4bf92f3577b34da6a3ce929d0e0e4736".into(),
            span_id: "00f067aa0ba902b7".into(),
        };
        let stage = xybrid_sdk::ObservationRef {
            span_id: "11f067aa0ba902b7".into(),
            ..workflow.clone()
        };
        output.observation_ref = Some(workflow.clone());
        output.stages[0].observation_ref = Some(stage.clone());
        let result = pipeline_ffi_result(&output).unwrap();
        let reference = result.observation.unwrap();
        assert_eq!(reference.trace_id, workflow.trace_id);
        assert_eq!(reference.span_id, workflow.span_id);
        let reference = result.metrics.stage_latencies_ms[0]
            .observation
            .as_ref()
            .unwrap();
        assert_eq!(reference.trace_id, stage.trace_id);
        assert_eq!(reference.span_id, stage.span_id);
        assert_eq!(result.text.as_deref(), Some("answer"));
    }

    #[test]
    fn final_stage_provenance_recognizes_current_runner_targets() {
        assert_eq!(
            final_execution_target(Some("local")),
            FfiExecutionTarget::Local
        );
        assert_eq!(
            final_execution_target(Some("cloud")),
            FfiExecutionTarget::Cloud
        );
        assert_eq!(
            final_execution_target(Some("fallback:xybrid-edge")),
            FfiExecutionTarget::Cloud
        );
        assert_eq!(
            final_execution_target(Some("unexpected-remote-target")),
            FfiExecutionTarget::Cloud
        );
    }

    fn pipeline_ending_in(kind: xybrid_sdk::ir::EnvelopeKind) -> PipelineExecutionResult {
        let output = xybrid_sdk::ir::Envelope::new(kind);
        PipelineExecutionResult {
            name: None,
            stages: vec![xybrid_sdk::PipelineStageTiming {
                name: "tts".into(),
                latency_ms: 30,
                target: "cloud".into(),
                reason: "test".into(),
                output: output.clone(),
                observation_ref: None,
            }],
            total_latency_ms: 30,
            output_type: xybrid_sdk::OutputType::Unknown,
            output,
            observation_ref: None,
        }
    }

    #[test]
    fn a_pipeline_result_converts_its_final_output_as_before() {
        let output = pipeline_ending_in(xybrid_sdk::ir::EnvelopeKind::Audio(vec![9, 8]));
        let Ok(result) = pipeline_ffi_result(&output) else {
            panic!("audio converts");
        };

        assert_eq!(result.audio_bytes.as_deref(), Some([9u8, 8].as_slice()));
        assert_eq!(result.latency_ms, 30);
        assert_eq!(result.execution_target, FfiExecutionTarget::Cloud);
        assert_eq!(result.metrics.stage_latencies_ms.len(), 1);
    }

    /// Before, this came back as `success: true` with every payload field
    /// empty.
    #[test]
    fn a_pipeline_whose_final_output_has_no_ffi_field_fails() {
        let output = pipeline_ending_in(xybrid_sdk::ir::EnvelopeKind::MultiPart(vec![]));
        let Err(error) = pipeline_ffi_result(&output) else {
            panic!("an output FfiResult cannot carry must fail");
        };

        assert!(error.contains("a multi-part message"), "{error}");
    }
}
