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
    """Single metadata key/value entry. BoltFFI doesn't auto-derive
    `WireEncode` for `HashMap<String, String>`, so we expose metadata as
    `Vec<XybridMetadataEntry>`. The conversion back to `HashMap` happens
    at the facade boundary inside [`XybridEnvelope::into`].
    """
    key: str
    value: str



@dataclass(frozen=True, slots=True)
class XybridEnvelope:
    kind: XybridEnvelopeKind
    metadata: list[XybridMetadataEntry]



@dataclass(frozen=True, slots=True)
class XybridToolDefinition:
    """A tool (function) the model may ask to call.

    `parameters_json` is the JSON Schema for the arguments, carried as a JSON
    string because no binding generator can describe an arbitrary JSON tree.
    """
    name: str
    description: str
    parameters_json: str



@dataclass(frozen=True, slots=True)
class XybridToolCall:
    """One tool call the model emitted, from [`XybridResult::tool_calls`]."""
    id: str
    name: str
    arguments_json: str



@dataclass(frozen=True, slots=True)
class XybridToolResult:
    """The outcome of running one tool, fed back with [`tool_results_envelope`]."""
    call_id: str
    """The [`XybridToolCall::id`] this answers."""
    name: str
    content_json: str
    """The tool's output as a JSON string."""



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
    """Optional GBNF grammar constraining generation to structured output
    (local llama backend only). Produce one from a JSON Schema with
    [`json_schema_to_gbnf`], or pass raw GBNF. Appended last: `#[data]`
    PODs serialize by field order across the FFI boundary.
    """
    tools: list[XybridToolDefinition]
    """Tools the model may call this turn. Empty means no tool calling —
    existing behavior, unchanged. Appended after `grammar` for the same
    field-order reason.

    Tool calling is llama.cpp-only today; unsupported paths (no embedded
    chat template, the mistralrs backend, the cloud fallback leg) reject
    tool-bearing requests rather than quietly generating without them.
    """



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
    """Inference output. Named `XybridResult` (not `XybridInferenceResult`)
    to match the existing uniffi-generated Kotlin/Swift name — the iOS
    example references `XybridResult` directly.
    """
    envelope: XybridEnvelope
    output_type: XybridOutputType
    model_id: str
    latency_ms: int
    execution_target: XybridExecutionTarget
    """Where the answer actually came from. Cloud fallback keeps `model_id`
    identical on both legs, so this is the only way to tell them apart.
    """
    metrics: XybridInferenceMetrics
    tool_calls: list[XybridToolCall]
    """Tool calls the model emitted this turn. Empty unless the request
    offered tools via [`XybridGenerationConfig::tools`].
    `#[data]` PODs serialize by field order across the FFI boundary.
    """
    reasoning_content: str | None = None
    """Model reasoning emitted separately from the final answer text.
    Appended last because `#[data]` fields serialize in declaration order.
    """



@dataclass(frozen=True, slots=True)
class XybridDownloadStatus:
    """Download progress, bytes and state in one consistent read.

    `progress` is aggregated across every artifact the model needs (weights
    plus companions such as a vision projector), never moves backwards, and
    reaches 1.0 only alongside `Ready`. `totalBytes` is null when the source
    declares no size — a Hugging Face repo, or a registry entry without one —
    in which case `downloadedBytes` is still exact and `progress` is coarser.

    Derives `Copy` because it is carried as a stream item.
    """
    state: XybridDownloadState
    progress: float
    """0.0..=1.0."""
    downloaded_bytes: int
    """Bytes written so far, across every artifact."""
    total_bytes: int | None
    """Declared total across every artifact, or null when unknown."""



@dataclass(frozen=True, slots=True)
class XybridStreamToken:
    token: str
    token_id: int | None
    index: int
    cumulative_text: str
    finish_reason: str | None
    """`"tool_calls"` when the turn ended on a parseable tool-call block."""
    tool_calls: list[XybridToolCall]
    """Tool calls parsed from the completed turn — populated on the
    **terminal** token only (the one carrying `finish_reason`).

    Tool-call blocks are suppressed from the emitted stream, so there is
    nothing in the token text to parse: a streaming caller halts here,
    runs the tools, then continues the turn by streaming a
    [`tool_results_envelope`] through the same call. Empty on every
    mid-stream token and on turns that emitted no call.
    """
    raw_text: str | None
    """The completed turn's raw output text, tool-call block included — pass
    it to [`tool_results_envelope`] as `prior_assistant_text`.

    Present only alongside a non-empty [`Self::tool_calls`]. Not the same
    as `cumulative_text`, which reports the *emitted* text with the
    protocol blocks suppressed — which is why this field exists at all.
    """



@dataclass(frozen=True, slots=True)
class XybridStreamEvent:
    """One pull from a streaming inference session.

    This is a flat record instead of a data-carrying enum because the pinned
    C# generator cannot lower that enum shape reliably. `kind` selects the one
    populated payload: `token` for `Token`, none for `Complete`. A `Complete`
    event is followed by [`XybridModel::stream_result`] to retrieve the final
    result. Inference failures are returned as typed [`XybridError`] values by
    [`XybridModel::stream_next`].
    """
    kind: XybridStreamEventKind
    token: XybridStreamToken | None



@dataclass(frozen=True, slots=True)
class XybridVoiceInfo:
    id: str
    name: str
    gender: str | None
    language: str | None
    style: str | None



@dataclass(frozen=True, slots=True)
class XybridStreamingConfig:
    """Configuration for a live ASR session.

    The model is not named here — it comes from the loaded `XybridModel` the
    session is opened on. This only configures *how* the audio is chunked.
    """
    sample_rate: int
    """Sample rate of the audio you will feed. Must be 16000; the ASR
    backends are fixed there, so anything else is rejected rather than
    silently resampled.
    """
    vad: XybridVadMode
    """Voice-activity-detection mode."""
    vad_threshold: float
    """VAD sensitivity, 0.0–1.0. Ignored when `vad` is `Off`."""
    language: str | None
    """Language hint (e.g. `"en"`); null uses the model default."""
    audio_ctx: int | None
    """Whisper encoder context in mel frames; null uses the model default."""



@dataclass(frozen=True, slots=True)
class XybridPartialResult:
    """A partial transcript emitted while audio is streaming."""
    text: str
    """Best-effort transcript so far. Cumulative, not a delta — render it in
    place of the previous partial rather than appending.
    """
    is_stable: bool
    """`true` once this span is committed and will not change."""
    chunk_sequence: int
    """Monotonic chunk sequence number this result corresponds to."""
    audio_duration_ms: int
    """Audio covered so far, in milliseconds."""




class XybridError:
    """Errors surfaced across the FFI boundary. Variants mirror
    [`facade::Error`] — the facade owns the SDK→FFI translation; this enum
    only re-decorates it for the BoltFFI generator (proc macros must live
    on the type definition).

    Named `XybridError` (not `Error`) so the emitted Swift type doesn't
    shadow / collide with Swift's stdlib `Error` protocol, and so the
    Kotlin sealed-hierarchy name matches the existing uniffi consumer
    expectations.

    **Variant order is part of the wire contract.** BoltFFI encodes `#[error]`
    (and `#[data]`) enums by ordinal tag, so reordering or inserting a variant
    renumbers every variant after it and breaks already-built foreign clients.
    Only ever append at the tail, and keep this order in lockstep with
    [`facade::Error`] and its `code()` table.
    """
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


@dataclass(frozen=True, slots=True)
class XybridErrorCancelled(XybridError):
    """The host called `cancel` — today, on a model download."""
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
    """Where a result was produced — observed fact, not a routing preference."""
    LOCAL = 0
    CLOUD = 1


class XybridDownloadState(IntEnum):
    """Lifecycle of a model download — a standalone [`XybridDownload`] or
    the background download behind a speculative load.
    """
    DOWNLOADING = 0
    READY = 1
    FAILED = 2
    CANCELLED = 3


class XybridStreamEventKind(IntEnum):
    TOKEN = 0
    COMPLETE = 1


class XybridThermalState(IntEnum):
    NORMAL = 0
    WARM = 1
    HOT = 2
    CRITICAL = 3


class XybridVadMode:
    """How voice-activity detection (VAD) chunking is resolved for a session."""
    pass


@dataclass(frozen=True, slots=True)
class XybridVadModeOff(XybridVadMode):
    """Fixed time-window chunking; no voice-activity detection."""
    pass


@dataclass(frozen=True, slots=True)
class XybridVadModeDefault(XybridVadMode):
    """VAD on, using the bundled default Silero model."""
    pass


@dataclass(frozen=True, slots=True)
class XybridVadModeCustom(XybridVadMode):
    """VAD on, using a Silero model from this directory."""
    model_dir: str





class XybridDownload:
    _handle: int

    def __init__(self) -> None: ...

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridDownload": ...
    def __del__(self) -> None: ...
    @classmethod
    def from_registry(cls, id: str) -> "XybridDownload":
        """Start downloading a registry model. Returns immediately."""
    @classmethod
    def from_registry_with_platform(cls, id: str, platform: str) -> "XybridDownload":
        """Start downloading a registry model resolved for a specific platform."""
    def status(self) -> XybridDownloadStatus:
        """Current snapshot. Never blocks — safe from a UI thread or a per-frame
        render loop.
        """
    def is_finished(self) -> bool:
        """Whether the download reached a terminal state."""
    def error(self) -> str | None:
        """The failure message once the download ended in `Failed` or
        `Cancelled`; null otherwise. The stream carries the terminal *state*,
        this carries the reason.
        """
    def cancel(self) -> None:
        """Ask the download to stop. Takes effect within one chunk read, discards
        the partial file, and moves the status to `Cancelled`. Idempotent, and
        a no-op once the download is terminal.
        """
    def progress(self) -> "XybridDownloadProgressSubscription":
        """Pushed progress updates, closing once the download is terminal.

        Generated as an `AsyncStream` in Swift, a `Flow` in Kotlin, an
        `IAsyncEnumerable` in C# and a subscription object in Python.
        Cancelling the consuming task / scope / token unsubscribes; it does
        **not** cancel the download itself — call [`Self::cancel`] for that.

        The current snapshot is delivered first, so subscribing late still
        yields a frame, and a download that already finished closes at once
        instead of hanging.
        """


class XybridDownloadProgressSubscription:
    _handle: int | None
    def __init__(self) -> None: ...
    @classmethod
    def _from_handle(cls, handle: int) -> "XybridDownloadProgressSubscription": ...
    def __del__(self) -> None: ...
    def pop_batch(self, max_count: int = 16) -> list[XybridDownloadStatus]: ...
    def wait(self, timeout_milliseconds: int) -> int: ...
    def unsubscribe(self) -> None: ...



class XybridStreamingSession:
    _handle: int

    def __init__(self) -> None: ...

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridStreamingSession": ...
    def __del__(self) -> None: ...
    @classmethod
    def for_model(cls, model: XybridModel, config: XybridStreamingConfig) -> "XybridStreamingSession":
        """Open a session on an already-loaded ASR model.

        Starts a worker thread and warms the weights, so the first spoken
        words do not pay the cold-start cost. Returns an error for a model
        that does not support streaming, or a sample rate other than 16000.
        """
    def feed(self, samples: Sequence[float]) -> None:
        """Feed PCM f32 mono 16 kHz samples.

        Hands the buffer to the worker and returns; transcription happens
        there, never on the caller's thread. Blocks only when the queue is
        full, which back-pressures a producer feeding faster than the model
        can keep up.
        """
    def flush(self) -> str:
        """Finalize: drain buffered audio and return the complete transcript.

        The session is over afterwards — `feed` fails and the partial stream
        closes. Blocks until the last chunk is transcribed, so call it off the
        UI thread.
        """
    def reset(self) -> None:
        """Reset to transcribe fresh audio without reloading the model."""
    def cancel(self) -> None:
        """Stop the session and release the model, discarding buffered audio.

        Idempotent. Use [`Self::flush`] when you want the transcript — this is
        the "user walked away" path. Named `cancel` rather than `close`
        because BoltFFI already gives every handle a generated `close()` for
        the host's disposal idiom.
        """
    def is_running(self) -> bool:
        """Whether the session is still accepting audio."""
    def partials(self) -> "XybridStreamingSessionPartialsSubscription":
        """Pushed partial transcripts, closing once the session ends.

        Generated as an `AsyncStream` in Swift, a `Flow` in Kotlin, an
        `IAsyncEnumerable` in C# and an iterable subscription in Python.

        A partial produced before subscribing is delivered immediately, so
        audio fed before the stream is attached is never silently lost, and
        subscribing to a finished session closes at once instead of hanging.
        """


class XybridStreamingSessionPartialsSubscription:
    _handle: int | None
    def __init__(self) -> None: ...
    @classmethod
    def _from_handle(cls, handle: int) -> "XybridStreamingSessionPartialsSubscription": ...
    def __del__(self) -> None: ...
    def pop_batch(self, max_count: int = 16) -> list[XybridPartialResult]: ...
    def wait(self, timeout_milliseconds: int) -> int: ...
    def unsubscribe(self) -> None: ...



class XybridCancellationToken:
    _handle: int


    def __init__(self) -> None:
        """Create a fresh, un-cancelled token."""


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridCancellationToken": ...
    def __del__(self) -> None: ...
    def cancel(self) -> None:
        """Request cancellation. Idempotent, and safe to call from any thread."""
    def is_cancelled(self) -> bool:
        """Whether [`Self::cancel`] has been called on this token."""



class XybridModel:
    _handle: int

    def __init__(self) -> None: ...

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridModel": ...
    def __del__(self) -> None: ...
    @classmethod
    def from_registry(cls, id: str) -> "XybridModel":
        """Load from the xybrid registry. Recommended path."""
    @classmethod
    def from_registry_speculative(cls, id: str) -> "XybridModel":
        """Load from the registry, serving from the cloud gateway while the weights
        download in the background.

        Returns almost immediately instead of blocking on the download. Requires
        a resolvable API key and an uncached model; otherwise it behaves exactly
        like `from_registry`. Poll `download_status` for progress and
        `is_cloud_serving` to know which leg is answering. LLM/chat models only.
        """
    @classmethod
    def from_directory(cls, path: str) -> "XybridModel":
        """Load from a local model directory (must contain `model_metadata.json`)."""
    @classmethod
    def from_bundle(cls, path: str) -> "XybridModel":
        """Load from a local `.xyb` bundle."""
    @classmethod
    def from_huggingface(cls, repo: str) -> "XybridModel":
        """Resolve and load from a HuggingFace repo (`org/repo` or `org/repo:variant`)."""
    @classmethod
    def from_huggingface_with_revision(cls, repo: str, revision: str) -> "XybridModel":
        """Resolve and load a HuggingFace repository pinned to a revision."""
    @classmethod
    def from_model_file(cls, path: str) -> "XybridModel":
        """Load from a raw GGUF file, auto-generating `model_metadata.json` from the
        GGUF header (written next to the file if absent).
        """
    def model_id(self) -> str: ...
    def version(self) -> str: ...
    def output_type(self) -> XybridOutputType: ...
    def is_loaded(self) -> bool: ...
    def is_cloud_serving(self) -> bool:
        """Whether runs are currently answered by the cloud because the local
        weights are not ready yet. `false` for ordinary local models.
        """
    def download_status(self) -> XybridDownloadStatus:
        """Download progress + state in one read — poll this to drive a progress
        bar. Reports `Ready` at 1.0 for an ordinary local model, so hosts need
        no special case.
        """
    def await_download(self, timeout_ms: int) -> XybridDownloadStatus:
        """Block until the download finishes or `timeout_ms` elapses, then report
        the status. Call it off the UI thread (the same place `from_registry` is
        already called). `timeout_ms = 0` makes it a non-blocking read.
        """
    def supports_streaming(self) -> bool: ...
    def supports_token_streaming(self) -> bool:
        """Whether this model emits true token-by-token output."""
    def default_generation_config(self) -> XybridGenerationConfig:
        """Return the model's resolved generation defaults."""
    def is_llm(self) -> bool: ...
    def supports_tool_calling(self) -> bool | None:
        """Whether the model bundle declares local tool-calling support.

        Advisory tri-state: `null` means the bundle says nothing, so the host
        cannot tell. Gate tool UI on it; enforcement stays at run time — a
        tools-bearing request against a model whose chat template has no tool
        support fails as invalid input regardless of what this reports.
        """
    def has_voices(self) -> bool: ...
    def voices(self) -> list[XybridVoiceInfo]: ...
    def default_voice(self) -> XybridVoiceInfo | None: ...
    def voice(self, voice_id: str) -> XybridVoiceInfo | None: ...
    def run(self, envelope: XybridEnvelope, options: XybridRunOptions | None, cancel: XybridCancellationToken) -> XybridResult:
        """Run inference, optionally with [`XybridRunOptions`] (generation config,
        abort signals, cloud-fallback). Pass `None` for the model's defaults.

        The hand-written wrappers add a one-arg `run(envelope)` convenience that
        forwards `None`, so simple call sites stay ergonomic.
        Pass a [`XybridCancellationToken`] to keep a stop button on the run;
        `None` means the run cannot be cancelled.
        """
    def run_stream(self, envelope: XybridEnvelope, options: XybridRunOptions | None, cancel: XybridCancellationToken) -> int:
        """Start token streaming and return a model-scoped session identifier.

        The identifier remains valid until the final result is taken, an error
        is returned, or [`Self::stream_close`] is called.
        Pass a [`XybridCancellationToken`] to keep a stop button on the run;
        `None` means the run cannot be cancelled.
        """
    def stream_next(self, stream_id: int) -> XybridStreamEvent:
        """Block until the next item for `stream_id` is ready."""
    def stream_result(self, stream_id: int) -> XybridResult:
        """Take the final result after receiving a `Complete` event."""
    def stream_close(self, stream_id: int) -> None:
        """Forget a streaming session."""
    def run_with_context(self, envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions | None, cancel: XybridCancellationToken) -> XybridResult:
        """Run inference seeded with a conversation `context` (multi-turn chat).

        Only the generation config from `options` is applied — abort signals and
        cloud fallback are not wired on the context path (matches the facade's
        `run_with_context`).
        Pass a [`XybridCancellationToken`] to keep a stop button on the run;
        `None` means the run cannot be cancelled.

        Routes through the facade's options path, so abort signals and cloud
        fallback on `options` are honoured rather than dropped.
        """
    def run_stream_with_context(self, envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions | None, cancel: XybridCancellationToken) -> int:
        """Start context-aware token streaming; returns a model-scoped session id.
        The pull protocol is identical to [`Self::run_stream`]
        (`stream_next` / `stream_result` / `stream_close`).
        Pass a [`XybridCancellationToken`] to keep a stop button on the run;
        `None` means the run cannot be cancelled.
        """
    def warmup(self) -> None: ...
    def unload(self) -> None: ...
    def download_progress(self) -> "XybridModelDownloadProgressSubscription":
        """Pushed download updates for a speculatively-loaded model — the stream
        counterpart of [`Self::await_download`], and what issue #504 asks for.

        Emits the current snapshot first, then every update, then closes on
        the terminal state. An ordinary local model is already `Ready`, so its
        stream yields one frame and ends.
        """


class XybridModelDownloadProgressSubscription:
    _handle: int | None
    def __init__(self) -> None: ...
    @classmethod
    def _from_handle(cls, handle: int) -> "XybridModelDownloadProgressSubscription": ...
    def __del__(self) -> None: ...
    def pop_batch(self, max_count: int = 16) -> list[XybridDownloadStatus]: ...
    def wait(self, timeout_milliseconds: int) -> int: ...
    def unsubscribe(self) -> None: ...



class XybridConversationContext:
    _handle: int


    def __init__(self) -> None:
        """Create an empty conversation context (fresh id)."""


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridConversationContext": ...
    def __del__(self) -> None: ...
    @classmethod
    def with_id(cls, id: str) -> "XybridConversationContext":
        """Create a context with a caller-supplied id (for telemetry correlation
        across turns).
        """
    def push(self, envelope: XybridEnvelope) -> None:
        """Append a turn — typically a user or assistant message envelope."""
    def set_system(self, envelope: XybridEnvelope) -> None:
        """Set the persistent system-prompt envelope (survives [`clear`](Self::clear))."""
    def clear(self) -> None:
        """Drop the history; the system envelope (if any) is preserved."""
    def id(self) -> str:
        """The context id."""
    def history_len(self) -> int:
        """Number of history turns (excludes the system envelope)."""
    def history(self) -> list[XybridEnvelope]:
        """Return history turns, excluding the persistent system envelope."""
    def has_system(self) -> bool:
        """Whether a persistent system-prompt envelope is set."""
    def set_max_history_len(self, len: int) -> None:
        """Set the max history length before FIFO pruning."""



class XybridTelemetryConfig:
    _handle: int


    def __init__(self, api_key: str) -> None:
        """A new config bound to the default ingest endpoint and the given API key."""


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridTelemetryConfig": ...
    def __del__(self) -> None: ...
    def set_endpoint(self, endpoint: str) -> None:
        """Override the ingest endpoint (self-hosted collector / non-prod)."""
    def set_app_version(self, version: str) -> None:
        """Set the app version reported with every event."""
    def set_device_label(self, label: str) -> None:
        """Set the human-friendly device label reported with every event."""
    def set_device_attribute(self, key: str, value: str) -> None:
        """Attach an app-provided device attribute (stored under `device.custom`)."""
    def set_batch_size(self, batch_size: int) -> None:
        """Set the number of events buffered before a flush."""
    def set_flush_interval_secs(self, secs: int) -> None:
        """Set the background flush interval, in seconds."""
    def init(self) -> None:
        """Start the process-global telemetry exporter from this config.

        Consumes the config: subsequent setters no-op and a second `init` on the
        same handle errors. Modeled as a method (not a free `telemetry_init`)
        because boltffi 0.25.3 drops free functions that take a handle
        parameter, but lowers a handle self-method fine (same reason the
        generated `run` lives on `XybridModel`).

        # Errors
        Errors if this config was already consumed, or if telemetry is already
        initialized without an intervening [`telemetry_shutdown`].
        """



class XybridBundle:
    _handle: int

    def __init__(self) -> None: ...

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridBundle": ...
    def __del__(self) -> None: ...
    @classmethod
    def open(cls, path: str) -> "XybridBundle":
        """Open and parse a `.xyb` bundle (decompress zstd, parse tar, validate the
        manifest).
        """
    def model_id(self) -> str:
        """The model identifier from the manifest."""
    def version(self) -> str:
        """The version string from the manifest."""
    def target(self) -> str:
        """The target platform from the manifest."""
    def hash(self) -> str:
        """The SHA-256 hash from the manifest."""
    def has_metadata(self) -> bool:
        """Whether the bundle carries a `model_metadata.json`."""
    def file_count(self) -> int:
        """Number of files in the bundle (excludes `manifest.json`)."""
    def file_name(self, index: int) -> str | None:
        """The file name at `index`, or `None` if out of bounds."""
    def manifest_json(self) -> str:
        """The full bundle manifest serialized as JSON."""
    def metadata_json(self) -> str | None:
        """The `model_metadata.json` contents, or `None` if the bundle has none."""
    def extract(self, output_dir: str) -> None:
        """Extract every bundle file to `output_dir` (created if absent)."""





def tool_results_envelope(user_text: str, prior_assistant_text: str, results: Sequence[XybridToolResult]) -> XybridEnvelope:
    """Build the continuation envelope for the turn after the model asked for
    tools.

    One `run` is one model turn, so the loop lives in your code: run a
    tools-bearing request, execute every [`XybridToolCall`] it returns, then
    run this envelope to feed the outcomes back. Pass the same tools on the
    continuation's [`XybridGenerationConfig`] as on the original turn.

    A free function rather than a constructor because `XybridEnvelope` is a
    `#[data]` record, not a handle type — records carry no methods across the
    generated bindings.
    """
def json_schema_to_gbnf(schema_json: str) -> str:
    """Convert a JSON Schema (as a JSON string) into a GBNF grammar for
    [`XybridGenerationConfig::grammar`]. Fails on invalid JSON or schema
    constructs outside the supported subset.
    """
def set_thermal_state(state: XybridThermalState) -> None: ...
def clear_thermal_state() -> None: ...
def set_battery_level(percent: int) -> None: ...
def clear_battery_level() -> None: ...
def configure_runtime(api_key: str | None, gateway_url: str | None, ingest_url: str | None) -> None:
    """One-stop SDK initialization: API key + gateway/ingest URL overrides in
    one call. Delegates to [`facade::configure_runtime`]; blank strings are
    treated as absent. This is the canonical init the Swift
    `Xybrid.initialize(apiKey:gatewayUrl:ingestUrl:)` and Kotlin
    `Xybrid.init(context, apiKey, gatewayUrl, ingestUrl)` wrappers call.
    """
def init_sdk_cache_dir(cache_dir: str) -> None: ...
def set_binding(binding: str) -> None: ...
def set_api_key(api_key: str) -> None: ...
def set_provider_api_key(provider: str, api_key: str) -> None: ...
def set_platform_url(url: str) -> None:
    """Point the cloud gateway at a platform base URL (staging, self-hosted).
    Pass a bare base URL — the `/v1` suffix is applied internally.
    """
def set_speculative_cloud(enabled: bool) -> None:
    """Enable speculative cloud fallback globally: a registry model that isn't
    downloaded yet is served from the gateway while the weights download.

    LLM/chat only — prefer `XybridModel.fromRegistrySpeculative` when the app
    also loads ASR/TTS models, which cannot be served this way.
    """
def has_api_key() -> bool:
    """Whether a Xybrid gateway API key is resolvable (in-memory or env)."""
def is_speculative_cloud_enabled() -> bool:
    """Whether the global speculative-cloud default is on."""
def will_speculate_for_model(model_id: str) -> bool:
    """Whether `XybridModel::from_registry_speculative(model_id)` would actually
    speculate: an API key resolves and the model is not already cached.

    Lets the hand-written Swift/Kotlin loader facades answer "will this
    speculate?" before loading. Never touches the network.
    """
def version() -> str:
    """The SDK version string (tracks `CARGO_PKG_VERSION`)."""
def release_memory() -> int:
    """Release every idle loaded model's memory; returns how many were released.

    Call this from the platform's low-memory hook (`didReceiveMemoryWarning`
    on iOS, `onTrimMemory` on Android). Models with a run in flight are
    skipped, and a released model reloads itself on next use — no reload call,
    no new error to handle.
    """
def set_auto_release(enabled: bool) -> None:
    """Enable or disable automatic model release for subsequent loads.

    When enabled, loading a model under device memory pressure first releases
    least-recently-used idle models. Off by default; [`release_memory`] works
    either way.
    """
def is_auto_release_enabled() -> bool:
    """Whether automatic model release is enabled process-wide."""
def telemetry_default_endpoint() -> str:
    """The SDK's default telemetry ingest endpoint (for display alongside a config)."""
def telemetry_flush() -> None:
    """Flush pending telemetry events. Safe before init / after shutdown."""
def telemetry_shutdown() -> None:
    """Shut down the telemetry exporter. Idempotent."""
