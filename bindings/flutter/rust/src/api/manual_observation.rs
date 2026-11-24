//! Typed manual handles for application-owned runtimes.
use super::observability::{FfiObservationOptions, FfiObservationRef};
use flutter_rust_bridge::frb;
use xybrid_ffi_facade as facade;

impl From<FfiObservationOptions> for facade::ObservationControls {
    fn from(options: FfiObservationOptions) -> Self {
        Self {
            traceparent: options.traceparent,
            tracestate: options.tracestate,
            conversation_id: options.conversation_id,
            suppress_trace: options.suppress_trace,
        }
    }
}
impl From<facade::ObservationRef> for FfiObservationRef {
    fn from(reference: facade::ObservationRef) -> Self {
        Self {
            trace_id: reference.trace_id,
            span_id: reference.span_id,
        }
    }
}
// Translation stays in the facade.
#[derive(Clone, Copy, Debug)]
pub enum FfiExecutionLocation {
    Local,
    Remote,
}
impl From<FfiExecutionLocation> for facade::ExecutionLocation {
    fn from(value: FfiExecutionLocation) -> Self {
        match value {
            FfiExecutionLocation::Local => Self::Local,
            FfiExecutionLocation::Remote => Self::Remote,
        }
    }
}

/// Typed metadata for manual observations.
#[derive(Clone, Debug)]
pub struct FfiObservationContext {
    pub traceparent: String,
    pub tracestate: Option<String>,
}
impl From<facade::ObservationContext> for FfiObservationContext {
    fn from(value: facade::ObservationContext) -> Self {
        Self {
            traceparent: value.traceparent,
            tracestate: value.tracestate,
        }
    }
}

/// Typed metadata for manual observations.
#[derive(Clone, Debug)]
pub struct FfiInferenceRequest {
    pub operation: String,
    pub provider: String,
    pub model: Option<String>,
    pub location: FfiExecutionLocation,
    pub runtime: Option<String>,
    pub task: Option<String>,
    pub stream: bool,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub server_address: Option<String>,
    pub server_port: Option<i64>,
}
impl From<FfiInferenceRequest> for facade::InferenceRequest {
    fn from(value: FfiInferenceRequest) -> Self {
        Self {
            operation: value.operation,
            provider: value.provider,
            model: value.model,
            location: value.location.into(),
            runtime: value.runtime,
            task: value.task,
            stream: value.stream,
            max_tokens: value.max_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            server_address: value.server_address,
            server_port: value.server_port,
        }
    }
}

/// Typed metadata for manual observations.
#[derive(Clone, Debug)]
pub struct FfiToolCallInfo {
    pub name: String,
    pub call_id: Option<String>,
    pub tool_type: Option<String>,
    pub description: Option<String>,
    pub arguments_json: Option<String>,
}
impl From<FfiToolCallInfo> for facade::ToolCallInfo {
    fn from(value: FfiToolCallInfo) -> Self {
        Self {
            name: value.name,
            call_id: value.call_id,
            tool_type: value.tool_type,
            description: value.description,
            arguments_json: value.arguments_json,
        }
    }
}

/// Typed metadata for manual observations.
#[derive(Clone, Debug)]
pub struct FfiUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_write_input_tokens: Option<u64>,
    pub reasoning_output_tokens: Option<u64>,
}
impl From<FfiUsage> for facade::Usage {
    fn from(value: FfiUsage) -> Self {
        Self {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            cache_read_input_tokens: value.cache_read_input_tokens,
            cache_write_input_tokens: value.cache_write_input_tokens,
            reasoning_output_tokens: value.reasoning_output_tokens,
        }
    }
}

/// Typed metadata for manual observations.
#[derive(Clone, Debug)]
pub struct FfiModelLoadInfo {
    pub model: String,
    pub provider: Option<String>,
    pub runtime: Option<String>,
    pub format: Option<String>,
    pub size_bytes: Option<u64>,
}
impl From<FfiModelLoadInfo> for facade::ModelLoadInfo {
    fn from(value: FfiModelLoadInfo) -> Self {
        Self {
            model: value.model,
            provider: value.provider,
            runtime: value.runtime,
            format: value.format,
            size_bytes: value.size_bytes,
        }
    }
}

/// Manual workflow operation; closing the final handle cancels unfinished work.
#[frb(opaque)]
pub struct FfiWorkflowObservation {
    inner: std::sync::Arc<facade::WorkflowObservation>,
}
impl FfiWorkflowObservation {
    /// Starts an operation using the configured SDK session and explicit parent controls.
    pub fn new(name: String, options: Option<FfiObservationOptions>) -> Result<Self, String> {
        facade::WorkflowObservation::start(name, options.map(Into::into).unwrap_or_default())
            .map(|inner| Self { inner })
            .map_err(|e| e.to_string())
    }
    /// Context for child operations, including valid unsampled spans.
    pub fn context(&self) -> Option<FfiObservationContext> {
        self.inner.context().map(Into::into)
    }
    /// Reference to this operation's own span.
    pub fn observation_ref(&self) -> Option<FfiObservationRef> {
        self.inner.observation_ref().map(Into::into)
    }
    /// Ends successfully. The first terminal call wins.
    pub fn complete(&self) {
        self.inner.complete();
    }
    /// Ends with a governed error message.
    pub fn fail(&self, error_type: String, message: Option<String>) {
        self.inner.fail(error_type, message);
    }
    /// Cancels without flushing.
    pub fn cancel(&self) {
        self.inner.cancel();
    }
    /// Whether any terminal transition has happened.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
}

/// Manual tool operation; closing the final handle cancels unfinished work.
#[frb(opaque)]
pub struct FfiToolObservation {
    inner: std::sync::Arc<facade::ToolObservation>,
}
impl FfiToolObservation {
    /// Starts an operation using the configured SDK session and explicit parent controls.
    pub fn new(
        call: FfiToolCallInfo,
        options: Option<FfiObservationOptions>,
    ) -> Result<Self, String> {
        facade::ToolObservation::start(call.into(), options.map(Into::into).unwrap_or_default())
            .map(|inner| Self { inner })
            .map_err(|e| e.to_string())
    }
    /// Context for child operations, including valid unsampled spans.
    pub fn context(&self) -> Option<FfiObservationContext> {
        self.inner.context().map(Into::into)
    }
    /// Reference to this operation's own span.
    pub fn observation_ref(&self) -> Option<FfiObservationRef> {
        self.inner.observation_ref().map(Into::into)
    }
    /// Ends successfully. The first terminal call wins.
    pub fn complete(&self) {
        self.inner.complete();
    }
    /// Ends with a governed error message.
    pub fn fail(&self, error_type: String, message: Option<String>) {
        self.inner.fail(error_type, message);
    }
    /// Cancels without flushing.
    pub fn cancel(&self) {
        self.inner.cancel();
    }
    /// Whether any terminal transition has happened.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
    /// Records structured tool output through the content policy.
    pub fn record_result(&self, result_json: String) -> Result<(), String> {
        self.inner
            .record_result(result_json)
            .map_err(|e| e.to_string())
    }
}

/// Manual inference operation; closing the final handle cancels unfinished work.
#[frb(opaque)]
pub struct FfiInferenceObservation {
    inner: std::sync::Arc<facade::InferenceObservation>,
}
impl FfiInferenceObservation {
    /// Starts an operation using the configured SDK session and explicit parent controls.
    pub fn new(
        request: FfiInferenceRequest,
        options: Option<FfiObservationOptions>,
    ) -> Result<Self, String> {
        facade::InferenceObservation::start(
            request.into(),
            options.map(Into::into).unwrap_or_default(),
        )
        .map(|inner| Self { inner })
        .map_err(|e| e.to_string())
    }
    /// Context for child operations, including valid unsampled spans.
    pub fn context(&self) -> Option<FfiObservationContext> {
        self.inner.context().map(Into::into)
    }
    /// Reference to this operation's own span.
    pub fn observation_ref(&self) -> Option<FfiObservationRef> {
        self.inner.observation_ref().map(Into::into)
    }
    /// Ends successfully. The first terminal call wins.
    pub fn complete(&self) {
        self.inner.complete();
    }
    /// Ends with a governed error message.
    pub fn fail(&self, error_type: String, message: Option<String>) {
        self.inner.fail(error_type, message);
    }
    /// Cancels without flushing.
    pub fn cancel(&self) {
        self.inner.cancel();
    }
    /// Whether any terminal transition has happened.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
    /// Records governed structured content.
    pub fn record_system_instructions(&self, parts_json: String) -> Result<(), String> {
        self.inner
            .record_system_instructions(parts_json)
            .map_err(|e| e.to_string())
    }
    /// Records governed structured content.
    pub fn record_input_messages(&self, messages_json: String) -> Result<(), String> {
        self.inner
            .record_input_messages(messages_json)
            .map_err(|e| e.to_string())
    }
    /// Records governed structured content.
    pub fn record_output_messages(&self, messages_json: String) -> Result<(), String> {
        self.inner
            .record_output_messages(messages_json)
            .map_err(|e| e.to_string())
    }
    /// Records governed structured content.
    pub fn record_tool_definitions(&self, definitions_json: String) -> Result<(), String> {
        self.inner
            .record_tool_definitions(definitions_json)
            .map_err(|e| e.to_string())
    }
    /// Records a chunk arrival; this does not estimate first-token time.
    pub fn record_chunk(&self) {
        self.inner.record_chunk();
    }
    /// Records a runtime-measured first token (epoch milliseconds, or now).
    pub fn record_first_token(&self, at_epoch_ms: Option<u64>) -> Result<(), String> {
        self.inner
            .record_first_token(at_epoch_ms)
            .map_err(|e| e.to_string())
    }
    /// Replaces the cumulative usage snapshot, preserving unknown fields.
    pub fn record_usage(&self, usage: FfiUsage) {
        self.inner.record_usage(usage.into());
    }
    /// Records measured finish reasons.
    pub fn record_finish_reasons(&self, reasons: Vec<String>) {
        self.inner.record_finish_reasons(reasons);
    }
    /// Records provider-assigned response identifiers.
    pub fn record_response(&self, response_id: Option<String>, model: Option<String>) {
        self.inner.record_response(response_id, model);
    }
}

/// Manual modelload operation; closing the final handle cancels unfinished work.
#[frb(opaque)]
pub struct FfiModelLoadObservation {
    inner: std::sync::Arc<facade::ModelLoadObservation>,
}
impl FfiModelLoadObservation {
    /// Starts an operation using the configured SDK session and explicit parent controls.
    pub fn new(
        info: FfiModelLoadInfo,
        options: Option<FfiObservationOptions>,
    ) -> Result<Self, String> {
        facade::ModelLoadObservation::start(
            info.into(),
            options.map(Into::into).unwrap_or_default(),
        )
        .map(|inner| Self { inner })
        .map_err(|e| e.to_string())
    }
    /// Context for child operations, including valid unsampled spans.
    pub fn context(&self) -> Option<FfiObservationContext> {
        self.inner.context().map(Into::into)
    }
    /// Reference to this operation's own span.
    pub fn observation_ref(&self) -> Option<FfiObservationRef> {
        self.inner.observation_ref().map(Into::into)
    }
    /// Ends successfully. The first terminal call wins.
    pub fn complete(&self) {
        self.inner.complete();
    }
    /// Ends with a governed error message.
    pub fn fail(&self, error_type: String, message: Option<String>) {
        self.inner.fail(error_type, message);
    }
    /// Cancels without flushing.
    pub fn cancel(&self) {
        self.inner.cancel();
    }
    /// Whether any terminal transition has happened.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
    /// Records the measured cache outcome.
    pub fn record_cache_hit(&self, hit: bool) {
        self.inner.record_cache_hit(hit);
    }
}
