//! Inputs to the observer: requests, usage, resources, errors, and
//! evaluation results.

use std::collections::BTreeMap;
use std::time::SystemTime;

use opentelemetry::KeyValue;
use serde::{Deserialize, Serialize};

use crate::content::Modality;
use crate::context::ObservationContext;

/// Where an operation executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Location {
    /// In the calling process or device (span kind `INTERNAL`).
    Local,
    /// Over the network (span kind `CLIENT`).
    Remote,
}

impl Location {
    /// `local` or `remote`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Remote => "remote",
        }
    }
}

/// Whether an operation produces a trace span.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceMode {
    /// Follow the tracer's sampler.
    #[default]
    Default,
    /// Record metrics only; create no span. Used for unsampled live frames.
    Suppress,
}

/// Options common to every `start_*` call.
#[derive(Clone, Debug, Default)]
pub struct StartOptions {
    /// Explicit parent. Wins over the ambient context.
    pub parent: Option<ObservationContext>,
    /// Trace hint.
    pub trace: TraceMode,
    /// Explicit start time (defaults to the observer clock).
    pub start_time: Option<SystemTime>,
    /// Extra span attributes (adapter extensions such as routing data).
    pub attributes: Vec<KeyValue>,
}

impl StartOptions {
    /// Options with an explicit parent.
    #[must_use]
    pub fn child_of(parent: ObservationContext) -> Self {
        Self {
            parent: Some(parent),
            ..Self::default()
        }
    }

    /// Sets the parent.
    #[must_use]
    pub fn with_parent(mut self, parent: Option<ObservationContext>) -> Self {
        self.parent = parent;
        self
    }

    /// Sets the parent from W3C headers. An invalid `traceparent` is rejected
    /// here, so it can never fall back to the ambient context or start a new
    /// root; an invalid `tracestate` is discarded.
    ///
    /// # Errors
    ///
    /// [`crate::Error::InvalidTraceparent`] when `traceparent` is invalid.
    pub fn with_traceparent(
        mut self,
        traceparent: &str,
        tracestate: Option<&str>,
    ) -> crate::Result<Self> {
        self.parent = Some(ObservationContext::from_traceparent(
            traceparent,
            tracestate,
        )?);
        Ok(self)
    }

    /// Sets the trace hint.
    #[must_use]
    pub fn with_trace(mut self, trace: TraceMode) -> Self {
        self.trace = trace;
        self
    }

    /// Sets an explicit start time.
    #[must_use]
    pub fn with_start_time(mut self, at: SystemTime) -> Self {
        self.start_time = Some(at);
        self
    }

    /// Adds a span attribute.
    #[must_use]
    pub fn with_attribute(mut self, kv: KeyValue) -> Self {
        self.attributes.push(kv);
        self
    }
}

/// A named prompt template and its variables.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptRef {
    /// Template name.
    pub name: String,
    /// Template version.
    #[serde(default)]
    pub version: Option<String>,
    /// Template variables (content: governed by the capture policy).
    #[serde(default)]
    pub variables: BTreeMap<String, String>,
}

/// Describes an inference request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InferenceRequest {
    /// `chat`, `text_completion`, `generate_content`, `embeddings`, `predict`,
    /// or a provider-specific name.
    pub operation: String,
    /// Who serves the request: upstream provider for remote calls, runtime
    /// name for local execution.
    pub provider: String,
    /// Requested model.
    pub model: Option<String>,
    /// Execution location; drives the span kind.
    pub location: Option<Location>,
    /// Runtime name (`llama.cpp`, `onnxruntime`, `transformers.js`, …).
    pub runtime: Option<String>,
    /// Runtime version.
    pub runtime_version: Option<String>,
    /// Compute device (`cpu`, `metal`, `webgpu`, …).
    pub accelerator: Option<String>,
    /// Task (`text_generation`, `speech_to_text`, `image_classification`, …).
    pub task: Option<String>,
    /// Streaming request.
    pub stream: bool,
    /// `gen_ai.request.max_tokens`.
    pub max_tokens: Option<i64>,
    /// `gen_ai.request.temperature`.
    pub temperature: Option<f64>,
    /// `gen_ai.request.top_p`.
    pub top_p: Option<f64>,
    /// `gen_ai.request.top_k`.
    pub top_k: Option<i64>,
    /// `gen_ai.request.seed`.
    pub seed: Option<i64>,
    /// `gen_ai.request.stop_sequences`.
    pub stop_sequences: Vec<String>,
    /// `gen_ai.request.frequency_penalty`.
    pub frequency_penalty: Option<f64>,
    /// `gen_ai.request.presence_penalty`.
    pub presence_penalty: Option<f64>,
    /// `gen_ai.output.type` (`text`, `json`, `image`, `speech`).
    pub output_type: Option<String>,
    /// Caller-provided conversation id (never synthesized).
    pub conversation_id: Option<String>,
    /// Prompt template.
    pub prompt: Option<PromptRef>,
    /// Input modalities.
    pub input_modalities: Vec<Modality>,
    /// Output modalities.
    pub output_modalities: Vec<Modality>,
    /// Remote server address.
    pub server_address: Option<String>,
    /// Remote server port.
    pub server_port: Option<i64>,
    /// Embedding dimensions.
    pub embedding_dimensions: Option<i64>,
}

impl InferenceRequest {
    /// A request for `operation` served by `provider`.
    pub fn new(operation: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            provider: provider.into(),
            ..Self::default()
        }
    }

    /// Sets the requested model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the location.
    #[must_use]
    pub fn with_location(mut self, location: Location) -> Self {
        self.location = Some(location);
        self
    }

    /// Sets the runtime name.
    #[must_use]
    pub fn with_runtime(mut self, runtime: impl Into<String>) -> Self {
        self.runtime = Some(runtime.into());
        self
    }

    /// Marks the request as streaming.
    #[must_use]
    pub fn streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }
}

/// Token usage. Absent fields are unknown, never zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Total input tokens, including cached tokens and all modalities.
    #[serde(default)]
    pub input_tokens: Option<u64>,
    /// Total output tokens, including reasoning tokens.
    #[serde(default)]
    pub output_tokens: Option<u64>,
    /// Input tokens served from cache.
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    /// Input tokens written to cache.
    #[serde(default)]
    pub cache_write_input_tokens: Option<u64>,
    /// Output tokens spent on reasoning.
    #[serde(default)]
    pub reasoning_output_tokens: Option<u64>,
    /// Text input tokens.
    #[serde(default)]
    pub text_input_tokens: Option<u64>,
    /// Image input tokens.
    #[serde(default)]
    pub image_input_tokens: Option<u64>,
    /// Audio input tokens.
    #[serde(default)]
    pub audio_input_tokens: Option<u64>,
    /// Text output tokens.
    #[serde(default)]
    pub text_output_tokens: Option<u64>,
    /// Image output tokens.
    #[serde(default)]
    pub image_output_tokens: Option<u64>,
    /// Audio output tokens.
    #[serde(default)]
    pub audio_output_tokens: Option<u64>,
}

impl Usage {
    /// Usage with input and output totals.
    #[must_use]
    pub fn tokens(input: Option<u64>, output: Option<u64>) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            ..Self::default()
        }
    }

    /// Merges `other` into `self`; later known values win.
    pub fn merge(&mut self, other: &Usage) {
        macro_rules! take {
            ($($f:ident),*) => { $( if other.$f.is_some() { self.$f = other.$f; } )* };
        }
        take!(
            input_tokens,
            output_tokens,
            cache_read_input_tokens,
            cache_write_input_tokens,
            reasoning_output_tokens,
            text_input_tokens,
            image_input_tokens,
            audio_input_tokens,
            text_output_tokens,
            image_output_tokens,
            audio_output_tokens
        );
    }

    /// Whether any field is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// A measured resource sample supplied by the host (CPU, memory, thermal,
/// battery, accelerator). The observer never probes the device itself.
///
/// Every field is optional: leave a field `None` when it was not measured,
/// never fill it with a default. Peak, minimum and state fields describe the
/// operation's measurement window (see `CONTRACT.md` "Resource samples").
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceSample {
    /// CPU utilization, 0.0–1.0.
    #[serde(default)]
    pub cpu_utilization: Option<f64>,
    /// Highest CPU utilization observed in the window, 0.0–1.0.
    #[serde(default)]
    pub cpu_peak_utilization: Option<f64>,
    /// System memory in use, bytes.
    #[serde(default)]
    pub memory_used_bytes: Option<u64>,
    /// Lowest system memory available for allocation in the window, bytes.
    #[serde(default)]
    pub memory_min_available_bytes: Option<u64>,
    /// Most severe system memory pressure in the window: `normal`, `warn` or
    /// `critical` ([`crate::semconv::MEMORY_PRESSURE_LEVELS`]). Any other
    /// value is ignored.
    #[serde(default)]
    pub memory_pressure: Option<String>,
    /// Process resident set size, bytes.
    #[serde(default)]
    pub process_rss_bytes: Option<u64>,
    /// Peak process resident set size, bytes.
    #[serde(default)]
    pub process_peak_rss_bytes: Option<u64>,
    /// `nominal`, `fair`, `serious`, `critical`.
    #[serde(default)]
    pub thermal_state: Option<String>,
    /// Battery level, 0.0–1.0.
    #[serde(default)]
    pub battery_level: Option<f64>,
    /// Whether the device is charging.
    #[serde(default)]
    pub battery_charging: Option<bool>,
    /// Accelerator utilization, 0.0–1.0.
    #[serde(default)]
    pub accelerator_utilization: Option<f64>,
    /// Accelerator memory in use, bytes.
    #[serde(default)]
    pub accelerator_memory_used_bytes: Option<u64>,
}

/// Why an attempt ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptReason {
    /// First attempt.
    Initial,
    /// Retry of the same target.
    Retry,
    /// Fallback to another target.
    Fallback,
}

impl AttemptReason {
    /// Lowercase name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Retry => "retry",
            Self::Fallback => "fallback",
        }
    }
}

/// One concrete execution attempt of a logical inference request.
#[derive(Clone, Debug, PartialEq)]
pub struct AttemptInfo {
    /// 0-based attempt index.
    pub index: u32,
    /// Why this attempt ran.
    pub reason: AttemptReason,
    /// Provider serving this attempt.
    pub provider: String,
    /// Model used by this attempt.
    pub model: Option<String>,
    /// Location of this attempt.
    pub location: Location,
    /// Runtime of this attempt.
    pub runtime: Option<String>,
    /// Remote server address.
    pub server_address: Option<String>,
    /// Remote server port.
    pub server_port: Option<i64>,
}

/// A tool call being executed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCallInfo {
    /// Tool name.
    pub name: String,
    /// Tool call id.
    pub call_id: Option<String>,
    /// `function`, `extension`, `datastore`, …
    pub tool_type: Option<String>,
    /// Tool description.
    pub description: Option<String>,
    /// Call arguments (content).
    pub arguments: Option<serde_json::Value>,
}

impl ToolCallInfo {
    /// A tool call by name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }
}

/// A model being loaded.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelLoadInfo {
    /// Model id.
    pub model: String,
    /// Provider name (defaults to the runtime).
    pub provider: Option<String>,
    /// Runtime loading the model.
    pub runtime: Option<String>,
    /// Model format (`gguf`, `onnx`, `safetensors`, …).
    pub format: Option<String>,
    /// Model size in bytes.
    pub size_bytes: Option<u64>,
}

/// Why an operation failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorInfo {
    /// Low-cardinality error class (`timeout`, `rate_limited`, provider code).
    pub error_type: String,
    /// Human-readable message; redacted and bounded before recording.
    pub message: Option<String>,
}

impl ErrorInfo {
    /// An error of `error_type` without a message.
    pub fn new(error_type: impl Into<String>) -> Self {
        Self {
            error_type: error_type.into(),
            message: None,
        }
    }

    /// Adds a message.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }
}

/// An evaluation of a recorded operation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evaluation {
    /// Evaluation name (metric).
    pub name: String,
    /// Numeric score.
    pub score_value: Option<f64>,
    /// Score label.
    pub score_label: Option<String>,
    /// Explanation.
    pub explanation: Option<String>,
    /// Response id of the evaluated completion.
    pub response_id: Option<String>,
}

/// End-user feedback on a recorded operation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Feedback {
    /// Numeric score.
    pub score: Option<f64>,
    /// Label (`thumbs_up`, …).
    pub label: Option<String>,
    /// Free-text comment (content).
    pub comment: Option<String>,
}
