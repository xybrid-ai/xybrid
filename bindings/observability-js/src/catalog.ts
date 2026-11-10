/**
 * Semantic constants of observability contract v2.
 *
 * `test/catalog.test.ts` asserts every value here against
 * `crates/xybrid-observability/contract/catalog.json`, the source of truth shared with the Rust
 * crate. Change both together.
 */

export const CONTRACT_VERSION = "2";
export const GENAI_SEMCONV_REVISION = "95aff9cdf605bbc7d7e8f7c4745a693bd0c4eea8";
export const INSTRUMENTATION_SCOPE_NAME = "xybrid-observability";
export const EXTENSION_NAMESPACE = "xybrid.genai";
export const ATTEMPT_SPAN_NAME = "attempt {index}";

export type MetricName = keyof typeof METRICS;

export const OPERATIONS = {
  chat: {
    span_name: "chat {model}",
    duration_metric: "gen_ai.client.inference.duration",
    category: "inference",
  },
  text_completion: {
    span_name: "text_completion {model}",
    duration_metric: "gen_ai.client.inference.duration",
    category: "inference",
  },
  generate_content: {
    span_name: "generate_content {model}",
    duration_metric: "gen_ai.client.inference.duration",
    category: "inference",
  },
  embeddings: {
    span_name: "embeddings {model}",
    duration_metric: "gen_ai.client.embeddings.duration",
    category: "embeddings",
  },
  predict: {
    span_name: "predict {model}",
    duration_metric: "gen_ai.client.operation.duration",
    category: "custom",
  },
  load_model: {
    span_name: "load_model {model}",
    duration_metric: "gen_ai.client.operation.duration",
    category: "custom",
  },
  invoke_workflow: {
    span_name: "invoke_workflow {workflow}",
    duration_metric: "gen_ai.invoke_workflow.duration",
    category: "workflow",
  },
  execute_tool: {
    span_name: "execute_tool {tool}",
    duration_metric: "gen_ai.execute_tool.duration",
    category: "tool",
  },
} as const satisfies Record<
  string,
  { span_name: string; duration_metric: MetricName; category: string }
>;

/** `start_inference` names that use their own `OPERATIONS` entry. */
export const INFERENCE_OPERATIONS = [
  "chat",
  "text_completion",
  "generate_content",
  "embeddings",
  "predict",
] as const;

/**
 * Any other `start_inference` name (provider-specific, or a catalog name of another API). The
 * span name is built literally as `{operation} {model}`, never by template substitution.
 */
export const CUSTOM_INFERENCE_OPERATION = {
  span_name: "{operation} {model}",
  duration_metric: "gen_ai.client.inference.duration",
  category: "inference",
} as const satisfies { span_name: string; duration_metric: MetricName; category: string };

/** Attribute keys, grouped as in the catalog. */
export const ATTR = {
  OPERATION_NAME: "gen_ai.operation.name",
  PROVIDER_NAME: "gen_ai.provider.name",
  REQUEST_MODEL: "gen_ai.request.model",
  REQUEST_MAX_TOKENS: "gen_ai.request.max_tokens",
  REQUEST_TEMPERATURE: "gen_ai.request.temperature",
  REQUEST_TOP_P: "gen_ai.request.top_p",
  REQUEST_TOP_K: "gen_ai.request.top_k",
  REQUEST_SEED: "gen_ai.request.seed",
  REQUEST_STOP_SEQUENCES: "gen_ai.request.stop_sequences",
  REQUEST_FREQUENCY_PENALTY: "gen_ai.request.frequency_penalty",
  REQUEST_PRESENCE_PENALTY: "gen_ai.request.presence_penalty",
  REQUEST_CHOICE_COUNT: "gen_ai.request.choice.count",
  REQUEST_STREAM: "gen_ai.request.stream",
  OUTPUT_TYPE: "gen_ai.output.type",
  CONVERSATION_ID: "gen_ai.conversation.id",
  RESPONSE_ID: "gen_ai.response.id",
  RESPONSE_MODEL: "gen_ai.response.model",
  RESPONSE_FINISH_REASONS: "gen_ai.response.finish_reasons",
  RESPONSE_TIME_TO_FIRST_CHUNK: "gen_ai.response.time_to_first_chunk",
  USAGE_INPUT_TOKENS: "gen_ai.usage.input_tokens",
  USAGE_OUTPUT_TOKENS: "gen_ai.usage.output_tokens",
  USAGE_CACHE_READ_INPUT_TOKENS: "gen_ai.usage.cache_read.input_tokens",
  USAGE_CACHE_WRITE_INPUT_TOKENS: "gen_ai.usage.cache_write.input_tokens",
  USAGE_REASONING_OUTPUT_TOKENS: "gen_ai.usage.reasoning.output_tokens",
  USAGE_TEXT_INPUT_TOKENS: "gen_ai.usage.text.input_tokens",
  USAGE_IMAGE_INPUT_TOKENS: "gen_ai.usage.image.input_tokens",
  USAGE_AUDIO_INPUT_TOKENS: "gen_ai.usage.audio.input_tokens",
  USAGE_TEXT_OUTPUT_TOKENS: "gen_ai.usage.text.output_tokens",
  USAGE_IMAGE_OUTPUT_TOKENS: "gen_ai.usage.image.output_tokens",
  USAGE_AUDIO_OUTPUT_TOKENS: "gen_ai.usage.audio.output_tokens",
  EMBEDDINGS_DIMENSION_COUNT: "gen_ai.embeddings.dimension.count",
  PROMPT_NAME: "gen_ai.prompt.name",
  PROMPT_VERSION: "gen_ai.prompt.version",
  WORKFLOW_NAME: "gen_ai.workflow.name",
  TOOL_NAME: "gen_ai.tool.name",
  TOOL_CALL_ID: "gen_ai.tool.call.id",
  TOOL_TYPE: "gen_ai.tool.type",
  TOOL_DESCRIPTION: "gen_ai.tool.description",
  TOKEN_MODALITY: "gen_ai.token.modality",
  EVALUATION_NAME: "gen_ai.evaluation.name",
  EVALUATION_SCORE_VALUE: "gen_ai.evaluation.score.value",
  EVALUATION_SCORE_LABEL: "gen_ai.evaluation.score.label",
  EVALUATION_EXPLANATION: "gen_ai.evaluation.explanation",
  ERROR_TYPE: "error.type",
  SERVER_ADDRESS: "server.address",
  SERVER_PORT: "server.port",
  EXCEPTION_TYPE: "exception.type",
  EXCEPTION_MESSAGE: "exception.message",

  INPUT_MESSAGES: "gen_ai.input.messages",
  OUTPUT_MESSAGES: "gen_ai.output.messages",
  SYSTEM_INSTRUCTIONS: "gen_ai.system_instructions",
  TOOL_CALL_ARGUMENTS: "gen_ai.tool.call.arguments",
  TOOL_CALL_RESULT: "gen_ai.tool.call.result",
  PROMPT_VARIABLE_PREFIX: "gen_ai.prompt.variable.",
  TOOL_DEFINITIONS: "gen_ai.tool.definitions",

  EXECUTION_LOCATION: "xybrid.genai.execution.location",
  RUNTIME_NAME: "xybrid.genai.runtime.name",
  RUNTIME_VERSION: "xybrid.genai.runtime.version",
  ACCELERATOR: "xybrid.genai.accelerator",
  TASK: "xybrid.genai.task",
  INPUT_MODALITIES: "xybrid.genai.input.modalities",
  OUTPUT_MODALITIES: "xybrid.genai.output.modalities",
  RESPONSE_TIME_TO_FIRST_TOKEN: "xybrid.genai.response.time_to_first_token",
  RESPONSE_TIME_PER_OUTPUT_TOKEN: "xybrid.genai.response.time_per_output_token",
  RESPONSE_CHUNK_COUNT: "xybrid.genai.response.chunk_count",
  ATTEMPT_INDEX: "xybrid.genai.attempt.index",
  ATTEMPT_REASON: "xybrid.genai.attempt.reason",
  ATTEMPT_COUNT: "xybrid.genai.attempt.count",
  CONTENT_TRUNCATED_ATTRIBUTES: "xybrid.genai.content.truncated_attributes",
  MODEL_SIZE_BYTES: "xybrid.genai.model.size_bytes",
  MODEL_CACHE_HIT: "xybrid.genai.model.cache_hit",
  MODEL_FORMAT: "xybrid.genai.model.format",
  RESOURCE_CPU_UTILIZATION: "xybrid.genai.resource.cpu.utilization",
  RESOURCE_CPU_PEAK_UTILIZATION: "xybrid.genai.resource.cpu.peak_utilization",
  RESOURCE_MEMORY_USED_BYTES: "xybrid.genai.resource.memory.used_bytes",
  RESOURCE_MEMORY_MIN_AVAILABLE_BYTES: "xybrid.genai.resource.memory.min_available_bytes",
  RESOURCE_MEMORY_PRESSURE: "xybrid.genai.resource.memory.pressure",
  RESOURCE_PROCESS_RSS_BYTES: "xybrid.genai.resource.process.rss_bytes",
  RESOURCE_PROCESS_PEAK_RSS_BYTES: "xybrid.genai.resource.process.peak_rss_bytes",
  RESOURCE_THERMAL_STATE: "xybrid.genai.resource.thermal.state",
  RESOURCE_BATTERY_LEVEL: "xybrid.genai.resource.battery.level",
  RESOURCE_BATTERY_CHARGING: "xybrid.genai.resource.battery.charging",
  RESOURCE_ACCELERATOR_UTILIZATION: "xybrid.genai.resource.accelerator.utilization",
  RESOURCE_ACCELERATOR_MEMORY_USED_BYTES: "xybrid.genai.resource.accelerator.memory.used_bytes",
  FEEDBACK_COMMENT: "xybrid.genai.feedback.comment",
} as const;

/** Attributes whose span value is a JSON string (compared after parsing in fixtures). */
export const JSON_ATTRIBUTES: ReadonlySet<string> = new Set([
  ATTR.INPUT_MESSAGES,
  ATTR.OUTPUT_MESSAGES,
  ATTR.SYSTEM_INSTRUCTIONS,
  ATTR.TOOL_CALL_ARGUMENTS,
  ATTR.TOOL_CALL_RESULT,
  ATTR.TOOL_DEFINITIONS,
]);

export const ENUMS = {
  "xybrid.genai.execution.location": ["local", "remote"],
  "xybrid.genai.attempt.reason": ["initial", "retry", "fallback"],
  "gen_ai.token.modality": ["text", "image", "audio", "unknown"],
  modality: ["text", "image", "audio", "video"],
  finish_reason: ["stop", "length", "content_filter", "tool_call", "compaction", "error"],
  content_capture: ["no_content", "span_only", "event_only", "span_and_event"],
  "xybrid.genai.resource.thermal.state": ["nominal", "fair", "serious", "critical"],
  "xybrid.genai.resource.memory.pressure": ["normal", "warn", "critical"],
  trace_mode: ["default", "suppress"],
  redaction_rule: ["replace_literal", "omit_attribute"],
  signal: ["traces", "logs", "metrics"],
  export_counter: ["accepted", "exported", "dropped", "failed", "pending"],
  header_policy: ["merge_environment", "isolated"],
} as const;

export const BUCKET_BOUNDARIES = {
  duration: [
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
  ],
  tokens: [
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
  ],
} as const;

type MetricSpec =
  | {
      instrument: "histogram";
      unit: string;
      value_type: "double" | "int";
      buckets: keyof typeof BUCKET_BOUNDARIES;
    }
  | { instrument: "counter"; unit: string; value_type: "int" };

const duration = {
  instrument: "histogram",
  unit: "s",
  value_type: "double",
  buckets: "duration",
} as const;
const tokenCounter = { instrument: "counter", unit: "{token}", value_type: "int" } as const;
const tokenHistogram = {
  instrument: "histogram",
  unit: "{token}",
  value_type: "int",
  buckets: "tokens",
} as const;

export const METRICS = {
  "gen_ai.client.inference.duration": duration,
  "gen_ai.client.embeddings.duration": duration,
  "gen_ai.client.operation.duration": duration,
  "gen_ai.client.inference.time_to_first_chunk": duration,
  "gen_ai.client.inference.time_per_output_chunk": duration,
  "gen_ai.client.inference.usage.input_tokens": tokenCounter,
  "gen_ai.client.inference.usage.output_tokens": tokenCounter,
  "gen_ai.client.inference.usage.cache_read.input_tokens": tokenCounter,
  "gen_ai.client.inference.usage.cache_write.input_tokens": tokenCounter,
  "gen_ai.client.inference.usage.reasoning.output_tokens": tokenCounter,
  "gen_ai.client.inference.operation.input_tokens": tokenHistogram,
  "gen_ai.client.inference.operation.output_tokens": tokenHistogram,
  "gen_ai.client.embeddings.operation.input_tokens": tokenHistogram,
  "gen_ai.invoke_workflow.duration": duration,
  "gen_ai.execute_tool.duration": duration,
  "xybrid.genai.client.inference.time_to_first_token": duration,
  "xybrid.genai.client.inference.attempts": {
    instrument: "counter",
    unit: "{attempt}",
    value_type: "int",
  },
} as const satisfies Record<string, MetricSpec>;

export const EVENTS = {
  "gen_ai.client.inference.operation.details": { severity_number: 5, severity_text: "DEBUG" },
  "gen_ai.execute_tool.operation.details": { severity_number: 5, severity_text: "DEBUG" },
  "gen_ai.client.operation.exception": { severity_number: 13, severity_text: "WARN" },
  "gen_ai.evaluation.result": { severity_number: 9, severity_text: "INFO" },
} as const;

export const METRIC_DIMENSIONS = {
  common: [
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "server.address",
    "server.port",
    "xybrid.genai.execution.location",
  ],
  error: ["error.type"],
  usage_counter_extra: ["gen_ai.token.modality"],
  workflow: ["gen_ai.operation.name", "gen_ai.workflow.name", "error.type"],
  tool: ["gen_ai.operation.name", "gen_ai.tool.name", "gen_ai.tool.type", "error.type"],
  attempt: [
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "xybrid.genai.execution.location",
    "xybrid.genai.attempt.reason",
    "error.type",
  ],
  model_load: [
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "error.type",
  ],
} as const;

export type ContentCapture = (typeof ENUMS.content_capture)[number];

/** `xybrid.genai.resource.memory.pressure` levels, least to most severe. */
export type MemoryPressure = (typeof ENUMS)["xybrid.genai.resource.memory.pressure"][number];

export const DEFAULTS = {
  content_capture: "span_only" as ContentCapture,
  max_content_attribute_bytes: 16384,
  max_error_message_chars: 256,
  truncation_marker: "…[truncated]",
  truncation_min_string_bytes: 32,
  redaction_replacement: "[REDACTED]",
  max_retained_operation_bytes: 212992,
  max_traversal_bytes: 491520,
  span_envelope_allowance_bytes: 65536,
  span_content_encoding_growth: 2,
  max_json_depth: 128,
  preflight_node_bytes: 16,
  batch: {
    max_queue_size: 2048,
    max_export_batch_size: 256,
    scheduled_delay_ms: 2000,
    export_timeout_ms: 10000,
  },
  retry: {
    max_retries: 2,
    initial_delay_ms: 200,
    max_delay_ms: 2000,
    retryable_status: [429, 502, 503, 504],
  },
  max_request_body_bytes: 491520,
  max_buffered_bytes_per_signal: 33554432,
  max_response_body_bytes: 65536,
  shutdown_timeout_ms: 5000,
  flush_timeout_ms: 5000,
  metric_export_interval_ms: 30000,
  max_tracestate_members: 32,
  max_tracestate_chars: 512,
} as const;

export const ENV = {
  content_capture: "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT",
  content_capture_legacy: "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
  sdk_disabled: "OTEL_SDK_DISABLED",
  otlp_endpoint: "OTEL_EXPORTER_OTLP_ENDPOINT",
  otlp_signal_endpoint: "OTEL_EXPORTER_OTLP_{SIGNAL}_ENDPOINT",
  otlp_headers: "OTEL_EXPORTER_OTLP_HEADERS",
  otlp_signal_headers: "OTEL_EXPORTER_OTLP_{SIGNAL}_HEADERS",
  otlp_protocol: "OTEL_EXPORTER_OTLP_PROTOCOL",
  otlp_signal_protocol: "OTEL_EXPORTER_OTLP_{SIGNAL}_PROTOCOL",
  otlp_timeout: "OTEL_EXPORTER_OTLP_TIMEOUT",
  otlp_signal_timeout: "OTEL_EXPORTER_OTLP_{SIGNAL}_TIMEOUT",
  signal_exporter: "OTEL_{SIGNAL}_EXPORTER",
  bsp_max_queue_size: "OTEL_BSP_MAX_QUEUE_SIZE",
  bsp_max_export_batch_size: "OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
  bsp_schedule_delay: "OTEL_BSP_SCHEDULE_DELAY",
  bsp_export_timeout: "OTEL_BSP_EXPORT_TIMEOUT",
  metric_export_interval: "OTEL_METRIC_EXPORT_INTERVAL",
  service_name: "OTEL_SERVICE_NAME",
  resource_attributes: "OTEL_RESOURCE_ATTRIBUTES",
} as const;

export const PRESETS = {
  xybrid_traces: {
    traces_endpoint: "https://ingest.xybrid.dev/v1/traces",
    auth_header: "Authorization",
    auth_scheme: "Bearer",
    signals: ["traces"],
    protocol: "http/protobuf",
    header_policy: "isolated",
  },
} as const;

/** Catalog value type of an attribute. */
export type AttributeType = "int" | "double" | "boolean" | "string" | "string[]" | "json";

/**
 * Value type of every catalog attribute (all groups merged). `gen_ai.prompt.variable.{key}`
 * stands for every prompt variable key; use {@link catalogAttributeType} for lookups.
 */
export const ATTRIBUTE_TYPES: Readonly<Record<string, AttributeType>> = {
  "gen_ai.operation.name": "string",
  "gen_ai.provider.name": "string",
  "gen_ai.request.model": "string",
  "gen_ai.request.max_tokens": "int",
  "gen_ai.request.temperature": "double",
  "gen_ai.request.top_p": "double",
  "gen_ai.request.top_k": "int",
  "gen_ai.request.seed": "int",
  "gen_ai.request.stop_sequences": "string[]",
  "gen_ai.request.frequency_penalty": "double",
  "gen_ai.request.presence_penalty": "double",
  "gen_ai.request.choice.count": "int",
  "gen_ai.request.stream": "boolean",
  "gen_ai.output.type": "string",
  "gen_ai.conversation.id": "string",
  "gen_ai.response.id": "string",
  "gen_ai.response.model": "string",
  "gen_ai.response.finish_reasons": "string[]",
  "gen_ai.response.time_to_first_chunk": "double",
  "gen_ai.usage.input_tokens": "int",
  "gen_ai.usage.output_tokens": "int",
  "gen_ai.usage.cache_read.input_tokens": "int",
  "gen_ai.usage.cache_write.input_tokens": "int",
  "gen_ai.usage.reasoning.output_tokens": "int",
  "gen_ai.usage.text.input_tokens": "int",
  "gen_ai.usage.image.input_tokens": "int",
  "gen_ai.usage.audio.input_tokens": "int",
  "gen_ai.usage.text.output_tokens": "int",
  "gen_ai.usage.image.output_tokens": "int",
  "gen_ai.usage.audio.output_tokens": "int",
  "gen_ai.embeddings.dimension.count": "int",
  "gen_ai.prompt.name": "string",
  "gen_ai.prompt.version": "string",
  "gen_ai.workflow.name": "string",
  "gen_ai.tool.name": "string",
  "gen_ai.tool.call.id": "string",
  "gen_ai.tool.type": "string",
  "gen_ai.tool.description": "string",
  "gen_ai.token.modality": "string",
  "gen_ai.evaluation.name": "string",
  "gen_ai.evaluation.score.value": "double",
  "gen_ai.evaluation.score.label": "string",
  "gen_ai.evaluation.explanation": "string",
  "error.type": "string",
  "server.address": "string",
  "server.port": "int",
  "exception.type": "string",
  "exception.message": "string",
  "gen_ai.input.messages": "json",
  "gen_ai.output.messages": "json",
  "gen_ai.system_instructions": "json",
  "gen_ai.tool.call.arguments": "json",
  "gen_ai.tool.call.result": "json",
  "gen_ai.prompt.variable.{key}": "string",
  "gen_ai.tool.definitions": "json",
  "xybrid.genai.execution.location": "string",
  "xybrid.genai.runtime.name": "string",
  "xybrid.genai.runtime.version": "string",
  "xybrid.genai.accelerator": "string",
  "xybrid.genai.task": "string",
  "xybrid.genai.input.modalities": "string[]",
  "xybrid.genai.output.modalities": "string[]",
  "xybrid.genai.response.time_to_first_token": "double",
  "xybrid.genai.response.time_per_output_token": "double",
  "xybrid.genai.response.chunk_count": "int",
  "xybrid.genai.attempt.index": "int",
  "xybrid.genai.attempt.reason": "string",
  "xybrid.genai.attempt.count": "int",
  "xybrid.genai.content.truncated_attributes": "string[]",
  "xybrid.genai.model.size_bytes": "int",
  "xybrid.genai.model.cache_hit": "boolean",
  "xybrid.genai.model.format": "string",
  "xybrid.genai.resource.cpu.utilization": "double",
  "xybrid.genai.resource.cpu.peak_utilization": "double",
  "xybrid.genai.resource.memory.used_bytes": "int",
  "xybrid.genai.resource.memory.min_available_bytes": "int",
  "xybrid.genai.resource.memory.pressure": "string",
  "xybrid.genai.resource.process.rss_bytes": "int",
  "xybrid.genai.resource.process.peak_rss_bytes": "int",
  "xybrid.genai.resource.thermal.state": "string",
  "xybrid.genai.resource.battery.level": "double",
  "xybrid.genai.resource.battery.charging": "boolean",
  "xybrid.genai.resource.accelerator.utilization": "double",
  "xybrid.genai.resource.accelerator.memory.used_bytes": "int",
  "xybrid.genai.feedback.comment": "string",
};

/** The catalog type of `key`, covering every `gen_ai.prompt.variable.<name>`. */
export const catalogAttributeType = (key: string): AttributeType | undefined =>
  Object.hasOwn(ATTRIBUTE_TYPES, key)
    ? ATTRIBUTE_TYPES[key]
    : key.startsWith("gen_ai.prompt.variable.")
      ? "string"
      : undefined;

/** OTLP encoding the package's exporters enforce for telemetry of `enforced_scope`. */
export const WIRE = {
  enforced_scope: "xybrid-observability",
  attribute_value_types: {
    int: "intValue",
    double: "doubleValue",
    boolean: "boolValue",
    string: "stringValue",
    "string[]": "arrayValue of stringValue",
    json: "stringValue holding compact JSON on spans; structured AnyValue on log events",
  },
  metric_value_types: { int: "asInt", double: "asDouble" },
  json_null: "AnyValue with no value set (OTLP/JSON {})",
  rust_null_carrier:
    "opentelemetry::logs::AnyValue::Bytes(empty), seen only between the Rust observer and the paired exporter",
} as const;
