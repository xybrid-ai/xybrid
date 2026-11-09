//! Manual operations shared by every foreign binding.
use super::{sdk, Error, ObservationControls, ObservationRef, Result};
use sdk::observability as obs;
use std::sync::Arc;

/// Normalized W3C context for parenting subsequent operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservationContext {
    pub traceparent: String,
    pub tracestate: Option<String>,
}

/// The execution location determines the inference span kind.
#[derive(Clone, Copy, Debug, Default)]
pub enum ExecutionLocation {
    #[default]
    Local,
    Remote,
}

/// Describes one manually observed inference; unknown values remain absent.
#[derive(Clone, Debug, Default)]
pub struct InferenceRequest {
    pub operation: String,
    pub provider: String,
    pub model: Option<String>,
    pub location: ExecutionLocation,
    pub runtime: Option<String>,
    pub task: Option<String>,
    pub stream: bool,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub server_address: Option<String>,
    pub server_port: Option<i64>,
}

/// A tool invocation; arguments are structured JSON content.
#[derive(Clone, Debug, Default)]
pub struct ToolCallInfo {
    pub name: String,
    pub call_id: Option<String>,
    pub tool_type: Option<String>,
    pub description: Option<String>,
    pub arguments_json: Option<String>,
}

/// Measurements reported by the runtime, including measured zeroes.
#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_write_input_tokens: Option<u64>,
    pub reasoning_output_tokens: Option<u64>,
}

/// One model load, including loads performed by an application-owned runtime.
#[derive(Clone, Debug, Default)]
pub struct ModelLoadInfo {
    pub model: String,
    pub provider: Option<String>,
    pub runtime: Option<String>,
    pub format: Option<String>,
    pub size_bytes: Option<u64>,
}

fn invalid() -> Error {
    Error::ConfigError {
        message: "invalid observation content JSON or shape".into(),
    }
}

fn json(raw: &str) -> Result<serde_json::Value> {
    // Bound the foreign-language payload before deserializing/cloning it. The
    // portable content pipeline applies its own traversal/redaction budgets.
    if raw.len() > 480 * 1024 {
        return Err(invalid());
    }
    serde_json::from_str(raw).map_err(|_| invalid())
}

fn error(error_type: String, message: Option<String>) -> obs::ErrorInfo {
    let valid = !error_type.is_empty()
        && error_type.len() <= 64
        && error_type
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b));
    obs::ErrorInfo {
        error_type: if valid { error_type } else { "_OTHER".into() },
        message,
    }
}

macro_rules! operation {
    ($name:ident, $op:ty) => {
        /// Shared, thread-safe manual observation. Last-owner drop cancels without flushing.
        pub struct $name {
            op: obs::ManualObservation<$op>,
        }
        impl $name {
            pub fn context(&self) -> Option<ObservationContext> {
                self.op.context().map(|c| ObservationContext {
                    traceparent: c.to_traceparent(),
                    tracestate: c.tracestate().map(str::to_owned),
                })
            }
            pub fn observation_ref(&self) -> Option<ObservationRef> {
                self.op
                    .observation_ref()
                    .as_ref()
                    .map(ObservationRef::from_sdk)
            }
            pub fn complete(&self) {
                self.op.end();
            }
            pub fn fail(&self, error_type: String, message: Option<String>) {
                if !self.op.is_finished() {
                    self.op.end_with_error(&error(error_type, message));
                }
            }
            pub fn cancel(&self) {
                self.op.cancel();
            }
            pub fn is_finished(&self) -> bool {
                self.op.is_finished()
            }
        }
    };
}
operation!(WorkflowObservation, obs::WorkflowOperation);
operation!(ToolObservation, obs::ToolOperation);
operation!(InferenceObservation, obs::InferenceOperation);
operation!(ModelLoadObservation, obs::ModelLoadOperation);

impl WorkflowObservation {
    pub fn start(name: String, options: ObservationControls) -> Result<Arc<Self>> {
        let options = options.to_sdk()?.unwrap_or_default();
        Ok(Arc::new(Self {
            op: obs::start_workflow(&name, &options),
        }))
    }
}

impl ToolObservation {
    pub fn start(call: ToolCallInfo, options: ObservationControls) -> Result<Arc<Self>> {
        let options = options.to_sdk()?.unwrap_or_default();
        let arguments = call.arguments_json.as_deref().map(json).transpose()?;
        Ok(Arc::new(Self {
            op: obs::start_tool(
                obs::ToolCallInfo {
                    name: call.name,
                    call_id: call.call_id,
                    tool_type: call.tool_type,
                    description: call.description,
                    arguments,
                },
                &options,
            ),
        }))
    }
    pub fn record_result(&self, result_json: String) -> Result<()> {
        if !self.is_finished() {
            self.op.record_result(&json(&result_json)?);
        }
        Ok(())
    }
}

impl InferenceObservation {
    pub fn start(request: InferenceRequest, options: ObservationControls) -> Result<Arc<Self>> {
        let options = options.to_sdk()?.unwrap_or_default();
        if [request.temperature, request.top_p]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err(Error::ConfigError {
                message: "inference parameters must be finite".into(),
            });
        }
        let request = obs::InferenceRequest {
            operation: request.operation,
            provider: request.provider,
            model: request.model,
            location: Some(match request.location {
                ExecutionLocation::Local => obs::Location::Local,
                ExecutionLocation::Remote => obs::Location::Remote,
            }),
            runtime: request.runtime,
            task: request.task,
            stream: request.stream,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            server_address: request.server_address,
            server_port: request.server_port,
            ..Default::default()
        };
        Ok(Arc::new(Self {
            op: obs::start_inference(request, &options),
        }))
    }
    pub fn record_system_instructions(&self, parts_json: String) -> Result<()> {
        if !self.is_finished() {
            let parts: Vec<obs::Part> =
                serde_json::from_value(json(&parts_json)?).map_err(|_| invalid())?;
            self.op.record_system_instructions(&parts);
        }
        Ok(())
    }
    pub fn record_input_messages(&self, messages_json: String) -> Result<()> {
        if !self.is_finished() {
            let messages: Vec<obs::InputMessage> =
                serde_json::from_value(json(&messages_json)?).map_err(|_| invalid())?;
            self.op.record_input_messages(&messages);
        }
        Ok(())
    }
    pub fn record_output_messages(&self, messages_json: String) -> Result<()> {
        if !self.is_finished() {
            let messages: Vec<obs::OutputMessage> =
                serde_json::from_value(json(&messages_json)?).map_err(|_| invalid())?;
            self.op.record_output_messages(&messages);
        }
        Ok(())
    }
    pub fn record_tool_definitions(&self, definitions_json: String) -> Result<()> {
        if !self.is_finished() {
            self.op.record_tool_definitions(&json(&definitions_json)?);
        }
        Ok(())
    }
    pub fn record_chunk(&self) {
        self.op.record_chunk();
    }
    pub fn record_first_token(&self, at_epoch_ms: Option<u64>) -> Result<()> {
        if self.is_finished() {
            return Ok(());
        }
        match at_epoch_ms {
            Some(ms) => {
                let at = std::time::UNIX_EPOCH
                    .checked_add(std::time::Duration::from_millis(ms))
                    .ok_or_else(invalid)?;
                self.op.record_first_token_at(at);
            }
            None => self.op.record_first_token(),
        }
        Ok(())
    }
    pub fn record_usage(&self, usage: Usage) {
        self.op.record_usage(&obs::Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            cache_write_input_tokens: usage.cache_write_input_tokens,
            reasoning_output_tokens: usage.reasoning_output_tokens,
            ..Default::default()
        });
    }
    pub fn record_finish_reasons(&self, reasons: Vec<String>) {
        self.op.record_finish_reasons(reasons);
    }
    pub fn record_response(&self, response_id: Option<String>, model: Option<String>) {
        self.op
            .record_response(response_id.as_deref(), model.as_deref());
    }
}

impl ModelLoadObservation {
    pub fn start(info: ModelLoadInfo, options: ObservationControls) -> Result<Arc<Self>> {
        let options = options.to_sdk()?.unwrap_or_default();
        Ok(Arc::new(Self {
            op: obs::start_model_load(
                obs::ModelLoadInfo {
                    model: info.model,
                    provider: info.provider,
                    runtime: info.runtime,
                    format: info.format,
                    size_bytes: info.size_bytes,
                },
                &options,
            ),
        }))
    }
    pub fn record_cache_hit(&self, hit: bool) {
        self.op.record_cache_hit(hit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terminal_handles_ignore_later_content_and_invalid_error_labels() {
        let op = ToolObservation::start(ToolCallInfo::default(), ObservationControls::default())
            .unwrap();
        assert!(!op.is_finished());
        let clone = op.clone();
        op.complete();
        clone.fail("private invalid label".into(), Some("secret".into()));
        assert!(op.is_finished());
        assert!(op.record_result("invalid JSON".into()).is_ok());
        let inference = InferenceObservation::start(
            InferenceRequest::default(),
            ObservationControls::default(),
        )
        .unwrap();
        assert!(inference.record_input_messages("{}".into()).is_err());
        inference.cancel();
        assert!(inference.record_input_messages("{}".into()).is_ok());
        let load =
            ModelLoadObservation::start(ModelLoadInfo::default(), ObservationControls::default())
                .unwrap();
        load.cancel();
        load.complete();
        assert!(load.is_finished());
    }
    #[test]
    fn invalid_parent_and_unbounded_json_are_rejected_before_start() {
        assert!(WorkflowObservation::start(
            "test".into(),
            ObservationControls {
                traceparent: Some("bad".into()),
                ..Default::default()
            }
        )
        .is_err());
        assert!(ToolObservation::start(
            ToolCallInfo {
                arguments_json: Some("x".repeat(480 * 1024 + 1)),
                ..Default::default()
            },
            ObservationControls::default()
        )
        .is_err());
        assert_eq!(
            error("secret with spaces".into(), None).error_type,
            "_OTHER"
        );
        fn send_sync<T: Send + Sync>() {}
        send_sync::<WorkflowObservation>();
        send_sync::<InferenceObservation>();
        send_sync::<ToolObservation>();
        send_sync::<ModelLoadObservation>();
    }
}
