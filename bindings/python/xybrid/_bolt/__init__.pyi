from __future__ import annotations

import uuid


from dataclasses import dataclass



from enum import IntEnum



from collections.abc import Sequence



MODULE_NAME: str
PACKAGE_NAME: str
PACKAGE_VERSION: str | None

@dataclass(frozen=True, slots=True)
class XybridMetadataEntry:
    key: str
    value: str



@dataclass(frozen=True, slots=True)
class XybridEnvelope:
    kind: XybridEnvelopeKind
    metadata: list[XybridMetadataEntry]



@dataclass(frozen=True, slots=True)
class XybridToolDefinition:
    name: str
    description: str
    parameters_json: str



@dataclass(frozen=True, slots=True)
class XybridToolCall:
    id: str
    name: str
    arguments_json: str



@dataclass(frozen=True, slots=True)
class XybridToolResult:
    call_id: str
    name: str
    content_json: str



@dataclass(frozen=True, slots=True)
class XybridGenerationConfig:
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    min_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    stop_sequences: list[str]
    grammar: str | None
    tools: list[XybridToolDefinition]



@dataclass(frozen=True, slots=True)
class XybridRunOptions:
    generation_config: XybridGenerationConfig | None
    abort_on: list[XybridAbortSignal]
    fallback_to_cloud: bool
    max_grace_tokens: int
    correlation_id: str | None



@dataclass(frozen=True, slots=True)
class XybridStageLatency:
    stage_id: str
    latency_ms: int



@dataclass(frozen=True, slots=True)
class XybridInferenceMetrics:
    total_ms: int
    ttft_ms: int | None
    tokens_per_second: float | None
    prefill_tps: float | None
    decode_tps: float | None
    tokens_out: int | None
    stage_latencies_ms: list[XybridStageLatency]



@dataclass(frozen=True, slots=True)
class XybridResult:
    envelope: XybridEnvelope
    output_type: XybridOutputType
    model_id: str
    latency_ms: int
    execution_target: XybridExecutionTarget
    metrics: XybridInferenceMetrics
    tool_calls: list[XybridToolCall]
    reasoning_content: str | None = None



@dataclass(frozen=True, slots=True)
class XybridDownloadStatus:
    state: XybridDownloadState
    progress: float



@dataclass(frozen=True, slots=True)
class XybridStreamToken:
    token: str
    token_id: int | None
    index: int
    cumulative_text: str
    finish_reason: str | None
    tool_calls: list[XybridToolCall]
    raw_text: str | None



@dataclass(frozen=True, slots=True)
class XybridStreamEvent:
    kind: XybridStreamEventKind
    token: XybridStreamToken | None



@dataclass(frozen=True, slots=True)
class XybridAsrStreamConfig:
    sample_rate: int
    enable_vad: bool
    vad_threshold: float
    vad_model_dir: str | None
    language: str | None
    audio_ctx: int | None



@dataclass(frozen=True, slots=True)
class XybridAsrPartialResult:
    text: str
    is_stable: bool
    chunk_index: int
    audio_duration_ms: int



@dataclass(frozen=True, slots=True)
class XybridAsrTranscriptionResult:
    text: str
    duration_ms: int
    chunks_processed: int



@dataclass(frozen=True, slots=True)
class XybridVoiceInfo:
    id: str
    name: str
    gender: str | None
    language: str | None
    style: str | None




class XybridError:
    pass


@dataclass(frozen=True, slots=True)
class XybridErrorModelNotFound(XybridError):
    id: str


@dataclass(frozen=True, slots=True)
class XybridErrorDirectoryNotFound(XybridError):
    path: str


@dataclass(frozen=True, slots=True)
class XybridErrorMetadataNotFound(XybridError):
    path: str


@dataclass(frozen=True, slots=True)
class XybridErrorMetadataInvalid(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorLoadError(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorInferenceError(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorAbortedForCloudFallback(XybridError):
    reason: str


@dataclass(frozen=True, slots=True)
class XybridErrorStreamingNotSupported(XybridError):
    pass


@dataclass(frozen=True, slots=True)
class XybridErrorNotLoaded(XybridError):
    pass


@dataclass(frozen=True, slots=True)
class XybridErrorConfigError(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorNetworkError(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorOffline(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorIoError(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorCacheError(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorPipelineError(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorCircuitOpen(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorRateLimited(XybridError):
    retry_after_secs: int


@dataclass(frozen=True, slots=True)
class XybridErrorTimeout(XybridError):
    timeout_ms: int


@dataclass(frozen=True, slots=True)
class XybridErrorMissingArtifact(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorUnsupportedModelCapability(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorUnsupportedBackendCapability(XybridError):
    message: str


@dataclass(frozen=True, slots=True)
class XybridErrorInvalidImage(XybridError):
    message: str



class XybridErrorException(RuntimeError):
    error: XybridError
    def __init__(self, error: XybridError) -> None: ...



class XybridEnvelopeKind:
    pass


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindText(XybridEnvelopeKind):
    text: str


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindAudio(XybridEnvelopeKind):
    bytes: bytes


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindEmbedding(XybridEnvelopeKind):
    values: list[float]


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindImage(XybridEnvelopeKind):
    bytes: bytes
    format: str


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindMultiPart(XybridEnvelopeKind):
    parts: list[XybridEnvelope]




class XybridMessageRole(IntEnum):
    SYSTEM = 0
    USER = 1
    ASSISTANT = 2


class XybridAbortSignal(IntEnum):
    MEMORY_PRESSURE_WARN = 0
    MEMORY_PRESSURE_CRITICAL = 1
    THERMAL_HOT = 2
    THERMAL_CRITICAL = 3


class XybridOutputType(IntEnum):
    TEXT = 0
    AUDIO = 1
    EMBEDDING = 2
    UNKNOWN = 3


class XybridExecutionTarget(IntEnum):
    LOCAL = 0
    CLOUD = 1


class XybridDownloadState(IntEnum):
    DOWNLOADING = 0
    READY = 1
    FAILED = 2


class XybridStreamEventKind(IntEnum):
    TOKEN = 0
    COMPLETE = 1


class XybridThermalState(IntEnum):
    NORMAL = 0
    WARM = 1
    HOT = 2
    CRITICAL = 3



class XybridModel:
    _handle: int

    def __init__(self) -> None: ...

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridModel": ...
    def __del__(self) -> None: ...
    @classmethod
    def from_registry(cls, id: str) -> "XybridModel": ...
    @classmethod
    def from_registry_speculative(cls, id: str) -> "XybridModel": ...
    @classmethod
    def from_directory(cls, path: str) -> "XybridModel": ...
    @classmethod
    def from_bundle(cls, path: str) -> "XybridModel": ...
    @classmethod
    def from_huggingface(cls, repo: str) -> "XybridModel": ...
    @classmethod
    def from_huggingface_with_revision(cls, repo: str, revision: str) -> "XybridModel": ...
    @classmethod
    def from_model_file(cls, path: str) -> "XybridModel": ...
    def model_id(self) -> str: ...
    def version(self) -> str: ...
    def output_type(self) -> XybridOutputType: ...
    def is_loaded(self) -> bool: ...
    def is_cloud_serving(self) -> bool: ...
    def download_status(self) -> XybridDownloadStatus: ...
    def await_download(self, timeout_ms: int) -> XybridDownloadStatus: ...
    def supports_streaming(self) -> bool: ...
    def supports_token_streaming(self) -> bool: ...
    def default_generation_config(self) -> XybridGenerationConfig: ...
    def is_llm(self) -> bool: ...
    def supports_tool_calling(self) -> bool | None: ...
    def has_voices(self) -> bool: ...
    def voices(self) -> list[XybridVoiceInfo]: ...
    def default_voice(self) -> XybridVoiceInfo | None: ...
    def voice(self, voice_id: str) -> XybridVoiceInfo | None: ...
    def stream(self, config: XybridAsrStreamConfig) -> XybridAsrStreamSession: ...
    def run(self, envelope: XybridEnvelope, options: XybridRunOptions | None) -> XybridResult: ...
    def run_stream(self, envelope: XybridEnvelope, options: XybridRunOptions | None) -> int: ...
    def stream_next(self, stream_id: int) -> XybridStreamEvent: ...
    def stream_result(self, stream_id: int) -> XybridResult: ...
    def stream_close(self, stream_id: int) -> None: ...
    def run_with_context(self, envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions | None) -> XybridResult: ...
    def run_stream_with_context(self, envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions | None) -> int: ...
    def warmup(self) -> None: ...
    def unload(self) -> None: ...



class XybridAsrStreamSession:
    _handle: int

    def __init__(self) -> None: ...

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridAsrStreamSession": ...
    def __del__(self) -> None: ...
    def feed(self, samples: Sequence[float]) -> None: ...
    def next(self) -> XybridAsrPartialResult | None: ...
    def flush(self) -> XybridAsrTranscriptionResult: ...
    def reset(self) -> None: ...
    def stop(self) -> None: ...



class XybridConversationContext:
    _handle: int


    def __init__(self) -> None: ...


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridConversationContext": ...
    def __del__(self) -> None: ...
    @classmethod
    def with_id(cls, id: str) -> "XybridConversationContext": ...
    def push(self, envelope: XybridEnvelope) -> None: ...
    def set_system(self, envelope: XybridEnvelope) -> None: ...
    def clear(self) -> None: ...
    def id(self) -> str: ...
    def history_len(self) -> int: ...
    def history(self) -> list[XybridEnvelope]: ...
    def has_system(self) -> bool: ...
    def set_max_history_len(self, len: int) -> None: ...



class XybridTelemetryConfig:
    _handle: int


    def __init__(self, api_key: str) -> None: ...


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridTelemetryConfig": ...
    def __del__(self) -> None: ...
    def set_endpoint(self, endpoint: str) -> None: ...
    def set_app_version(self, version: str) -> None: ...
    def set_device_label(self, label: str) -> None: ...
    def set_device_attribute(self, key: str, value: str) -> None: ...
    def set_batch_size(self, batch_size: int) -> None: ...
    def set_flush_interval_secs(self, secs: int) -> None: ...
    def init(self) -> None: ...



class XybridBundle:
    _handle: int

    def __init__(self) -> None: ...

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridBundle": ...
    def __del__(self) -> None: ...
    @classmethod
    def open(cls, path: str) -> "XybridBundle": ...
    def model_id(self) -> str: ...
    def version(self) -> str: ...
    def target(self) -> str: ...
    def hash(self) -> str: ...
    def has_metadata(self) -> bool: ...
    def file_count(self) -> int: ...
    def file_name(self, index: int) -> str | None: ...
    def manifest_json(self) -> str: ...
    def metadata_json(self) -> str | None: ...
    def extract(self, output_dir: str) -> None: ...





def tool_results_envelope(user_text: str, prior_assistant_text: str, results: Sequence[XybridToolResult]) -> XybridEnvelope: ...
def json_schema_to_gbnf(schema_json: str) -> str: ...
def set_thermal_state(state: XybridThermalState) -> None: ...
def clear_thermal_state() -> None: ...
def set_battery_level(percent: int) -> None: ...
def clear_battery_level() -> None: ...
def configure_runtime(api_key: str | None, gateway_url: str | None, ingest_url: str | None) -> None: ...
def init_sdk_cache_dir(cache_dir: str) -> None: ...
def set_binding(binding: str) -> None: ...
def set_api_key(api_key: str) -> None: ...
def set_provider_api_key(provider: str, api_key: str) -> None: ...
def set_platform_url(url: str) -> None: ...
def set_speculative_cloud(enabled: bool) -> None: ...
def has_api_key() -> bool: ...
def is_speculative_cloud_enabled() -> bool: ...
def will_speculate_for_model(model_id: str) -> bool: ...
def version() -> str: ...
def release_memory() -> int: ...
def set_auto_release(enabled: bool) -> None: ...
def is_auto_release_enabled() -> bool: ...
def telemetry_default_endpoint() -> str: ...
def telemetry_flush() -> None: ...
def telemetry_shutdown() -> None: ...
