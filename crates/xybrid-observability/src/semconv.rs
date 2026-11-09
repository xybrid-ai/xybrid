//! Semantic catalog: attribute keys, operation names, metric names, and
//! defaults shared with `@xybrid/observability`.
//!
//! Standard keys follow the OpenTelemetry GenAI semantic conventions pinned at
//! [`GENAI_SEMCONV_REVISION`]. Keys under `xybrid.genai.*` are this package's
//! portable extensions for measurements the conventions do not define yet.
//! `contract/catalog.json` is the source of truth; tests assert these
//! constants against it.

/// Revision of `open-telemetry/semantic-conventions-genai` this catalog follows.
pub const GENAI_SEMCONV_REVISION: &str = "95aff9cdf605bbc7d7e8f7c4745a693bd0c4eea8";

/// Version of the shared contract (`contract/catalog.json`).
pub const CONTRACT_VERSION: &str = "2";

/// Instrumentation scope name used for traces, metrics, and logs.
pub const SCOPE_NAME: &str = "xybrid-observability";

/// Operation names (`gen_ai.operation.name`).
pub mod op {
    /// Chat completion.
    pub const CHAT: &str = "chat";
    /// Text completion.
    pub const TEXT_COMPLETION: &str = "text_completion";
    /// Multimodal content generation (speech-to-text, text-to-speech, …).
    pub const GENERATE_CONTENT: &str = "generate_content";
    /// Embeddings.
    pub const EMBEDDINGS: &str = "embeddings";
    /// Non-generative model inference (classification, detection, VAD).
    pub const PREDICT: &str = "predict";
    /// Loading a model into a runtime.
    pub const LOAD_MODEL: &str = "load_model";
    /// Workflow invocation.
    pub const INVOKE_WORKFLOW: &str = "invoke_workflow";
    /// Tool execution.
    pub const EXECUTE_TOOL: &str = "execute_tool";

    /// Names `start_inference` resolves through the catalog
    /// (`catalog.inference_operations`). Every other name is a custom
    /// inference operation that uses the generative inference metrics.
    pub const INFERENCE_OPERATIONS: [&str; 5] =
        [CHAT, TEXT_COMPLETION, GENERATE_CONTENT, EMBEDDINGS, PREDICT];
}

/// Attribute keys.
pub mod attr {
    // Standard GenAI keys.
    /// `gen_ai.operation.name`
    pub const OPERATION_NAME: &str = "gen_ai.operation.name";
    /// `gen_ai.provider.name`
    pub const PROVIDER_NAME: &str = "gen_ai.provider.name";
    /// `gen_ai.request.model`
    pub const REQUEST_MODEL: &str = "gen_ai.request.model";
    /// `gen_ai.request.max_tokens`
    pub const REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
    /// `gen_ai.request.temperature`
    pub const REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
    /// `gen_ai.request.top_p`
    pub const REQUEST_TOP_P: &str = "gen_ai.request.top_p";
    /// `gen_ai.request.top_k`
    pub const REQUEST_TOP_K: &str = "gen_ai.request.top_k";
    /// `gen_ai.request.seed`
    pub const REQUEST_SEED: &str = "gen_ai.request.seed";
    /// `gen_ai.request.stop_sequences`
    pub const REQUEST_STOP_SEQUENCES: &str = "gen_ai.request.stop_sequences";
    /// `gen_ai.request.frequency_penalty`
    pub const REQUEST_FREQUENCY_PENALTY: &str = "gen_ai.request.frequency_penalty";
    /// `gen_ai.request.presence_penalty`
    pub const REQUEST_PRESENCE_PENALTY: &str = "gen_ai.request.presence_penalty";
    /// `gen_ai.request.choice.count`
    pub const REQUEST_CHOICE_COUNT: &str = "gen_ai.request.choice.count";
    /// `gen_ai.request.stream`
    pub const REQUEST_STREAM: &str = "gen_ai.request.stream";
    /// `gen_ai.output.type`
    pub const OUTPUT_TYPE: &str = "gen_ai.output.type";
    /// `gen_ai.conversation.id`
    pub const CONVERSATION_ID: &str = "gen_ai.conversation.id";
    /// `gen_ai.response.id`
    pub const RESPONSE_ID: &str = "gen_ai.response.id";
    /// `gen_ai.response.model`
    pub const RESPONSE_MODEL: &str = "gen_ai.response.model";
    /// `gen_ai.response.finish_reasons`
    pub const RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
    /// `gen_ai.response.time_to_first_chunk`
    pub const RESPONSE_TIME_TO_FIRST_CHUNK: &str = "gen_ai.response.time_to_first_chunk";
    /// `gen_ai.usage.input_tokens`
    pub const USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    /// `gen_ai.usage.output_tokens`
    pub const USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    /// `gen_ai.usage.cache_read.input_tokens`
    pub const USAGE_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";
    /// `gen_ai.usage.cache_write.input_tokens`
    pub const USAGE_CACHE_WRITE_INPUT_TOKENS: &str = "gen_ai.usage.cache_write.input_tokens";
    /// `gen_ai.usage.reasoning.output_tokens`
    pub const USAGE_REASONING_OUTPUT_TOKENS: &str = "gen_ai.usage.reasoning.output_tokens";
    /// `gen_ai.usage.text.input_tokens`
    pub const USAGE_TEXT_INPUT_TOKENS: &str = "gen_ai.usage.text.input_tokens";
    /// `gen_ai.usage.image.input_tokens`
    pub const USAGE_IMAGE_INPUT_TOKENS: &str = "gen_ai.usage.image.input_tokens";
    /// `gen_ai.usage.audio.input_tokens`
    pub const USAGE_AUDIO_INPUT_TOKENS: &str = "gen_ai.usage.audio.input_tokens";
    /// `gen_ai.usage.text.output_tokens`
    pub const USAGE_TEXT_OUTPUT_TOKENS: &str = "gen_ai.usage.text.output_tokens";
    /// `gen_ai.usage.image.output_tokens`
    pub const USAGE_IMAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.image.output_tokens";
    /// `gen_ai.usage.audio.output_tokens`
    pub const USAGE_AUDIO_OUTPUT_TOKENS: &str = "gen_ai.usage.audio.output_tokens";
    /// `gen_ai.embeddings.dimension.count`
    pub const EMBEDDINGS_DIMENSION_COUNT: &str = "gen_ai.embeddings.dimension.count";
    /// `gen_ai.prompt.name`
    pub const PROMPT_NAME: &str = "gen_ai.prompt.name";
    /// `gen_ai.prompt.version`
    pub const PROMPT_VERSION: &str = "gen_ai.prompt.version";
    /// Prefix of `gen_ai.prompt.variable.<key>`.
    pub const PROMPT_VARIABLE_PREFIX: &str = "gen_ai.prompt.variable.";
    /// `gen_ai.workflow.name`
    pub const WORKFLOW_NAME: &str = "gen_ai.workflow.name";
    /// `gen_ai.tool.name`
    pub const TOOL_NAME: &str = "gen_ai.tool.name";
    /// `gen_ai.tool.call.id`
    pub const TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
    /// `gen_ai.tool.type`
    pub const TOOL_TYPE: &str = "gen_ai.tool.type";
    /// `gen_ai.tool.description`
    pub const TOOL_DESCRIPTION: &str = "gen_ai.tool.description";
    /// `gen_ai.token.modality`
    pub const TOKEN_MODALITY: &str = "gen_ai.token.modality";
    /// `gen_ai.evaluation.name`
    pub const EVALUATION_NAME: &str = "gen_ai.evaluation.name";
    /// `gen_ai.evaluation.score.value`
    pub const EVALUATION_SCORE_VALUE: &str = "gen_ai.evaluation.score.value";
    /// `gen_ai.evaluation.score.label`
    pub const EVALUATION_SCORE_LABEL: &str = "gen_ai.evaluation.score.label";
    /// `gen_ai.evaluation.explanation`
    pub const EVALUATION_EXPLANATION: &str = "gen_ai.evaluation.explanation";
    /// `error.type`
    pub const ERROR_TYPE: &str = "error.type";
    /// `server.address`
    pub const SERVER_ADDRESS: &str = "server.address";
    /// `server.port`
    pub const SERVER_PORT: &str = "server.port";
    /// `exception.type`
    pub const EXCEPTION_TYPE: &str = "exception.type";
    /// `exception.message`
    pub const EXCEPTION_MESSAGE: &str = "exception.message";

    // Content (governed by the capture policy).
    /// `gen_ai.input.messages`
    pub const INPUT_MESSAGES: &str = "gen_ai.input.messages";
    /// `gen_ai.output.messages`
    pub const OUTPUT_MESSAGES: &str = "gen_ai.output.messages";
    /// `gen_ai.system_instructions`
    pub const SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";
    /// `gen_ai.tool.call.arguments`
    pub const TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
    /// `gen_ai.tool.call.result`
    pub const TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";
    /// `gen_ai.tool.definitions` (schema content: recorded in every mode).
    pub const TOOL_DEFINITIONS: &str = "gen_ai.tool.definitions";

    // Portable extensions.
    /// `xybrid.genai.execution.location` (`local` or `remote`).
    pub const EXECUTION_LOCATION: &str = "xybrid.genai.execution.location";
    /// `xybrid.genai.runtime.name`
    pub const RUNTIME_NAME: &str = "xybrid.genai.runtime.name";
    /// `xybrid.genai.runtime.version`
    pub const RUNTIME_VERSION: &str = "xybrid.genai.runtime.version";
    /// `xybrid.genai.accelerator`
    pub const ACCELERATOR: &str = "xybrid.genai.accelerator";
    /// `xybrid.genai.task`
    pub const TASK: &str = "xybrid.genai.task";
    /// `xybrid.genai.input.modalities`
    pub const INPUT_MODALITIES: &str = "xybrid.genai.input.modalities";
    /// `xybrid.genai.output.modalities`
    pub const OUTPUT_MODALITIES: &str = "xybrid.genai.output.modalities";
    /// `xybrid.genai.response.time_to_first_token`
    pub const RESPONSE_TIME_TO_FIRST_TOKEN: &str = "xybrid.genai.response.time_to_first_token";
    /// `xybrid.genai.response.time_per_output_token`
    pub const RESPONSE_TIME_PER_OUTPUT_TOKEN: &str = "xybrid.genai.response.time_per_output_token";
    /// `xybrid.genai.response.chunk_count`
    pub const RESPONSE_CHUNK_COUNT: &str = "xybrid.genai.response.chunk_count";
    /// `xybrid.genai.attempt.index`
    pub const ATTEMPT_INDEX: &str = "xybrid.genai.attempt.index";
    /// `xybrid.genai.attempt.reason`
    pub const ATTEMPT_REASON: &str = "xybrid.genai.attempt.reason";
    /// `xybrid.genai.attempt.count`
    pub const ATTEMPT_COUNT: &str = "xybrid.genai.attempt.count";
    /// `xybrid.genai.content.truncated_attributes`
    pub const CONTENT_TRUNCATED_ATTRIBUTES: &str = "xybrid.genai.content.truncated_attributes";
    /// `xybrid.genai.model.size_bytes`
    pub const MODEL_SIZE_BYTES: &str = "xybrid.genai.model.size_bytes";
    /// `xybrid.genai.model.cache_hit`
    pub const MODEL_CACHE_HIT: &str = "xybrid.genai.model.cache_hit";
    /// `xybrid.genai.model.format`
    pub const MODEL_FORMAT: &str = "xybrid.genai.model.format";
    /// `xybrid.genai.resource.cpu.utilization`
    pub const RESOURCE_CPU_UTILIZATION: &str = "xybrid.genai.resource.cpu.utilization";
    /// `xybrid.genai.resource.memory.used_bytes`
    pub const RESOURCE_MEMORY_USED_BYTES: &str = "xybrid.genai.resource.memory.used_bytes";
    /// `xybrid.genai.resource.process.rss_bytes`
    pub const RESOURCE_PROCESS_RSS_BYTES: &str = "xybrid.genai.resource.process.rss_bytes";
    /// `xybrid.genai.resource.process.peak_rss_bytes`
    pub const RESOURCE_PROCESS_PEAK_RSS_BYTES: &str =
        "xybrid.genai.resource.process.peak_rss_bytes";
    /// `xybrid.genai.resource.thermal.state`
    pub const RESOURCE_THERMAL_STATE: &str = "xybrid.genai.resource.thermal.state";
    /// `xybrid.genai.resource.battery.level`
    pub const RESOURCE_BATTERY_LEVEL: &str = "xybrid.genai.resource.battery.level";
    /// `xybrid.genai.resource.battery.charging`
    pub const RESOURCE_BATTERY_CHARGING: &str = "xybrid.genai.resource.battery.charging";
    /// `xybrid.genai.resource.accelerator.utilization`
    pub const RESOURCE_ACCELERATOR_UTILIZATION: &str =
        "xybrid.genai.resource.accelerator.utilization";
    /// `xybrid.genai.resource.accelerator.memory.used_bytes`
    pub const RESOURCE_ACCELERATOR_MEMORY_USED_BYTES: &str =
        "xybrid.genai.resource.accelerator.memory.used_bytes";
    /// `xybrid.genai.feedback.comment` (content).
    pub const FEEDBACK_COMMENT: &str = "xybrid.genai.feedback.comment";
}

/// Metric instrument names.
pub mod metric {
    /// `gen_ai.client.inference.duration` (s)
    pub const INFERENCE_DURATION: &str = "gen_ai.client.inference.duration";
    /// `gen_ai.client.embeddings.duration` (s)
    pub const EMBEDDINGS_DURATION: &str = "gen_ai.client.embeddings.duration";
    /// `gen_ai.client.operation.duration` (s), for `predict` and `load_model`.
    pub const OPERATION_DURATION: &str = "gen_ai.client.operation.duration";
    /// `gen_ai.client.inference.time_to_first_chunk` (s)
    pub const TIME_TO_FIRST_CHUNK: &str = "gen_ai.client.inference.time_to_first_chunk";
    /// `gen_ai.client.inference.time_per_output_chunk` (s)
    pub const TIME_PER_OUTPUT_CHUNK: &str = "gen_ai.client.inference.time_per_output_chunk";
    /// `gen_ai.client.inference.usage.input_tokens`
    pub const USAGE_INPUT_TOKENS: &str = "gen_ai.client.inference.usage.input_tokens";
    /// `gen_ai.client.inference.usage.output_tokens`
    pub const USAGE_OUTPUT_TOKENS: &str = "gen_ai.client.inference.usage.output_tokens";
    /// `gen_ai.client.inference.usage.cache_read.input_tokens`
    pub const USAGE_CACHE_READ_INPUT_TOKENS: &str =
        "gen_ai.client.inference.usage.cache_read.input_tokens";
    /// `gen_ai.client.inference.usage.cache_write.input_tokens`
    pub const USAGE_CACHE_WRITE_INPUT_TOKENS: &str =
        "gen_ai.client.inference.usage.cache_write.input_tokens";
    /// `gen_ai.client.inference.usage.reasoning.output_tokens`
    pub const USAGE_REASONING_OUTPUT_TOKENS: &str =
        "gen_ai.client.inference.usage.reasoning.output_tokens";
    /// `gen_ai.client.inference.operation.input_tokens`
    pub const OPERATION_INPUT_TOKENS: &str = "gen_ai.client.inference.operation.input_tokens";
    /// `gen_ai.client.inference.operation.output_tokens`
    pub const OPERATION_OUTPUT_TOKENS: &str = "gen_ai.client.inference.operation.output_tokens";
    /// `gen_ai.client.embeddings.operation.input_tokens`
    pub const EMBEDDINGS_INPUT_TOKENS: &str = "gen_ai.client.embeddings.operation.input_tokens";
    /// `gen_ai.invoke_workflow.duration` (s)
    pub const WORKFLOW_DURATION: &str = "gen_ai.invoke_workflow.duration";
    /// `gen_ai.execute_tool.duration` (s)
    pub const TOOL_DURATION: &str = "gen_ai.execute_tool.duration";
    /// `xybrid.genai.client.inference.time_to_first_token` (s)
    pub const TIME_TO_FIRST_TOKEN: &str = "xybrid.genai.client.inference.time_to_first_token";
    /// `xybrid.genai.client.inference.attempts`
    pub const ATTEMPTS: &str = "xybrid.genai.client.inference.attempts";
}

/// Log event names.
pub mod event {
    /// Inference details with structured content (severity DEBUG).
    pub const INFERENCE_DETAILS: &str = "gen_ai.client.inference.operation.details";
    /// Tool details with governed arguments and results (severity DEBUG).
    pub const TOOL_DETAILS: &str = "gen_ai.execute_tool.operation.details";
    /// Client operation exception (severity WARN).
    pub const OPERATION_EXCEPTION: &str = "gen_ai.client.operation.exception";
    /// Evaluation result (severity INFO).
    pub const EVALUATION_RESULT: &str = "gen_ai.evaluation.result";
    /// `gen_ai.evaluation.name` used for end-user feedback.
    pub const USER_FEEDBACK_EVALUATION: &str = "user_feedback";
}

/// Explicit histogram boundaries for durations, in seconds.
pub const DURATION_BUCKETS: [f64; 14] = [
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
];

/// Explicit histogram boundaries for token counts.
pub const TOKEN_BUCKETS: [f64; 14] = [
    1.0,
    4.0,
    16.0,
    64.0,
    256.0,
    1024.0,
    4096.0,
    16384.0,
    65536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
];

/// Default per-attribute byte budget for content attributes.
pub const DEFAULT_MAX_CONTENT_ATTRIBUTE_BYTES: usize = 16_384;
/// Redaction kind of error messages (`Redactor` and `omit_attribute`); not
/// an attribute key.
pub const ERROR_MESSAGE_KIND: &str = "error.message";
/// Maximum characters of an error message recorded as span status.
pub const MAX_ERROR_MESSAGE_CHARS: usize = 256;
/// Marker appended to truncated strings.
pub const TRUNCATION_MARKER: &str = "…[truncated]";
/// Strings are not halved below this many bytes.
pub const TRUNCATION_MIN_STRING_BYTES: usize = 32;
/// Replacement used by [`crate::SubstringRedactor`].
pub const REDACTION_REPLACEMENT: &str = "[REDACTED]";
/// Environment variable selecting the content capture mode.
pub const CAPTURE_CONTENT_ENV: &str = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT";
/// Legacy capture variable, honoured only when [`CAPTURE_CONTENT_ENV`] is
/// unset (`true` means `span_only`; `false` and unknown values `no_content`).
pub const CAPTURE_CONTENT_LEGACY_ENV: &str = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";
/// Default budget of content retained by one operation, in bytes (480 KiB).
pub const DEFAULT_MAX_RETAINED_OPERATION_BYTES: usize = 491_520;
/// Default preflight traversal budget per content value, in bytes (480 KiB).
pub const DEFAULT_MAX_TRAVERSAL_BYTES: usize = 491_520;
/// Maximum JSON nesting depth of a content value.
pub const MAX_JSON_DEPTH: usize = 128;
/// Bytes charged per JSON node by the conservative preflight cost.
pub const PREFLIGHT_NODE_BYTES: usize = 16;
/// Maximum `tracestate` list members (W3C Trace Context).
pub const MAX_TRACESTATE_MEMBERS: usize = 32;
/// Maximum normalized `tracestate` length in characters (W3C Trace Context).
pub const MAX_TRACESTATE_CHARS: usize = 512;
