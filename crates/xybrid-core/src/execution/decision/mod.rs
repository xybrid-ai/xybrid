//! Local, versioned Liquid decision execution without generated text.
//!
//! Wire decoding and numerical finalization work without a native backend.
//! The `llm-llamacpp` feature adds the audited renderer and whole-prompt CPU
//! evaluator. Requests and answers never enter generation, cloud or pipelines.

pub mod liquid;
pub mod math;
pub mod prepare;
#[cfg(feature = "llm-llamacpp")]
pub(crate) mod runtime;
pub mod wire;

use crate::execution::choice::EntryPoint;
use crate::execution::{ExecutionTemplate, ModelMetadata};
use crate::ir::{Envelope, EnvelopeKind};
use crate::runtime_adapter::AdapterError;

/// A content-free refusal for an unsupported decision entry point.
pub fn unsupported(model_id: &str, capability: &str) -> AdapterError {
    AdapterError::UnsupportedModelCapability {
        model_id: model_id.into(),
        capability: capability.into(),
        hint: "decision models accept local batch DecisionRequest inputs only".into(),
    }
}

/// Rejects decision payloads recursively before dispatch or serialization.
pub fn ensure_no_decision_kinds<'a>(
    model_id: &str,
    envelopes: impl IntoIterator<Item = &'a Envelope>,
) -> Result<(), AdapterError> {
    if envelopes
        .into_iter()
        .any(|value| value.kind.contains_decision_kind())
    {
        return Err(unsupported(model_id, "decision requests"));
    }
    Ok(())
}

/// Checks the versioned template, IO shape and structure before execution.
pub fn validate_decision_request(
    metadata: &ModelMetadata,
    input: &Envelope,
    entry: EntryPoint,
) -> Result<(), AdapterError> {
    let ExecutionTemplate::Decision(spec) = &metadata.execution_template else {
        return ensure_no_decision_kinds(&metadata.model_id, std::iter::once(input));
    };
    spec.validate()?;
    if entry != EntryPoint::Batch {
        return Err(unsupported(
            &metadata.model_id,
            "decision streaming, history or pipelines",
        ));
    }
    let EnvelopeKind::DecisionRequest(request) = &input.kind else {
        return Err(unsupported(&metadata.model_id, "non-decision input"));
    };
    // These metadata keys are consumed by platform/foreign routing paths.
    // Fail before any backend can see them, including when callers bypass SDKs.
    if input.metadata.keys().any(|key| {
        matches!(
            key.as_str(),
            "target"
                | "execution_target"
                | "cloud"
                | "cloud_model"
                | "cloud_provider"
                | "cloud_fallback"
                | "fallback_to_cloud"
                | "speculative"
                | "xybrid.execution_target"
                | "xybrid.cloud_provider"
                | "xybrid.cloud_model"
                | "backend"
                | "provider"
                | "model"
                | "gateway_url"
                | "system_prompt"
                | "role"
                | "tool_calls"
                | "tool_results"
                | "tool_call_id"
                | "reasoning"
                | "temperature"
                | "top_p"
                | "top_k"
                | "max_tokens"
                | "max_new_tokens"
                | "repetition_penalty"
                | "stop_sequences"
                | "grammar"
                | "tools"
                | "tool_choice"
        )
    }) {
        return Err(unsupported(
            &metadata.model_id,
            "decision routing overrides",
        ));
    }
    if !metadata.preprocessing.is_empty() || !metadata.postprocessing.is_empty() {
        return Err(unsupported(
            &metadata.model_id,
            "decision preprocessing or postprocessing",
        ));
    }
    liquid::validate_request(request)?;
    Ok(())
}

/// Last-line refusal for adapters that consume generation/tensor inputs.
pub(crate) fn decision_kind_not_runtime_input() -> AdapterError {
    AdapterError::InvalidInput("decision payloads require a decision runtime".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::template::DecisionTemplate;
    use crate::execution::TemplateExecutor;

    fn metadata(template: ExecutionTemplate) -> ModelMetadata {
        ModelMetadata {
            model_id: "decision-test".into(),
            version: "1".into(),
            execution_template: template,
            preprocessing: vec![],
            postprocessing: vec![],
            files: vec![],
            vision_encoder: None,
            description: None,
            metadata: Default::default(),
            voices: None,
            max_chunk_chars: None,
            trim_trailing_samples: None,
        }
    }
    fn request() -> Envelope {
        Envelope::decision_request(wire::decode(
            r#"{"state":"private-state","questions":{"private-id":{"type":"noul","instructions":"private-question"}}}"#
        ).unwrap())
    }

    #[test]
    fn unsupported_shapes_and_routing_fail_before_model_io() {
        let model = metadata(ExecutionTemplate::Decision(DecisionTemplate::new(
            "missing.gguf",
        )));
        let valid = request();
        assert!(validate_decision_request(&model, &valid, EntryPoint::Batch).is_ok());
        for entry in [
            EntryPoint::Conversation,
            EntryPoint::Streaming,
            EntryPoint::Pipeline,
        ] {
            assert!(matches!(
                validate_decision_request(&model, &valid, entry),
                Err(AdapterError::UnsupportedModelCapability { .. })
            ));
        }
        let mut executor = TemplateExecutor::new(".");
        for key in [
            "cloud_model",
            "target",
            "tools",
            "temperature",
            "fallback_to_cloud",
        ] {
            let mut input = valid.clone();
            input.set_metadata(key.into(), "private-value".into());
            let error = executor.execute(&model, &input, None).unwrap_err();
            assert!(matches!(
                error,
                AdapterError::UnsupportedModelCapability { .. }
            ));
            assert!(!format!("{error:?}").contains("private-"));
        }
        assert!(executor
            .execute(&model, &valid, Some(&Default::default()))
            .is_err());
    }

    #[test]
    fn unrelated_models_refuse_decisions_at_any_depth() {
        let model = metadata(ExecutionTemplate::Onnx {
            model_file: "missing.onnx".into(),
        });
        let direct = request();
        let nested = Envelope::choice_request(
            Envelope::new(EnvelopeKind::MultiPart(vec![direct.clone()])),
            vec![],
        );
        for input in [
            direct,
            nested,
            Envelope::decisions(crate::ir::Decisions::new(
                vec![],
                crate::ir::DecisionUsage::default(),
            )),
        ] {
            assert!(matches!(
                validate_decision_request(&model, &input, EntryPoint::Batch),
                Err(AdapterError::UnsupportedModelCapability { .. })
            ));
        }
    }

    #[cfg(not(feature = "llm-llamacpp"))]
    #[test]
    fn missing_native_feature_is_a_capability_error_at_load() {
        let model = metadata(ExecutionTemplate::Decision(DecisionTemplate::new(
            "missing.gguf",
        )));
        let mut executor = TemplateExecutor::new(".");
        assert!(matches!(
            executor.load_decision(&model),
            Err(AdapterError::UnsupportedModelCapability { .. })
        ));
    }
}
