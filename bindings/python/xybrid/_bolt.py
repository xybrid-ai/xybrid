# Hand-port of the BoltFFI 0.25.3 wire ABI for xybrid's Python SDK.
# Reference implementation: bindings/apple/Sources/Xybrid/xybrid_bolt.swift.
# Enum variant order is the wire contract; append only, and keep it in
# lockstep with crates/xybrid-bolt/src/lib.rs. This file should be replaced by
# generated output once the workspace migrates to boltffi >= 0.26.

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import Final, TypeAlias, TypeVar

_T = TypeVar("_T")

_LIBRARY_ENV: Final = "XYBRID_BOLT_LIBRARY"
_SYMBOLS: Final = frozenset(
    {
        "boltffi_clear_battery_level",
        "boltffi_clear_last_error",
        "boltffi_clear_thermal_state",
        "boltffi_configure_runtime",
        "boltffi_free_buf",
        "boltffi_free_string",
        "boltffi_init_sdk_cache_dir",
        "boltffi_is_speculative_cloud_enabled",
        "boltffi_json_schema_to_gbnf",
        "boltffi_last_error_message",
        "boltffi_set_api_key",
        "boltffi_set_platform_url",
        "boltffi_set_speculative_cloud",
        "boltffi_set_battery_level",
        "boltffi_set_binding",
        "boltffi_set_provider_api_key",
        "boltffi_set_thermal_state",
        "boltffi_xybrid_model_await_download",
        "boltffi_xybrid_model_default_voice",
        "boltffi_xybrid_model_download_status",
        "boltffi_xybrid_model_free",
        "boltffi_xybrid_model_from_bundle",
        "boltffi_xybrid_model_from_directory",
        "boltffi_xybrid_model_from_huggingface",
        "boltffi_xybrid_model_from_registry",
        "boltffi_xybrid_model_from_registry_speculative",
        "boltffi_xybrid_model_has_voices",
        "boltffi_xybrid_model_is_cloud_serving",
        "boltffi_xybrid_model_is_llm",
        "boltffi_xybrid_model_is_loaded",
        "boltffi_xybrid_model_model_id",
        "boltffi_xybrid_model_output_type",
        "boltffi_xybrid_model_run",
        "boltffi_xybrid_model_supports_streaming",
        "boltffi_xybrid_model_unload",
        "boltffi_xybrid_model_version",
        "boltffi_xybrid_model_voice",
        "boltffi_xybrid_model_voices",
        "boltffi_xybrid_model_warmup",
    }
)


class _FfiBuf(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("cap", ctypes.c_size_t),
        ("align", ctypes.c_size_t),
    ]


class _FfiString(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("cap", ctypes.c_size_t),
    ]


class _FfiStatus(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int32)]


_LIB: ctypes.CDLL | None = None
_LIB_LOCK: Final = threading.Lock()


class XybridMessageRole(IntEnum):
    """Message role wire ordinals."""

    SYSTEM = 0
    USER = 1
    ASSISTANT = 2


class XybridAbortSignal(IntEnum):
    """Abort signal wire ordinals."""

    MEMORY_PRESSURE_WARN = 0
    MEMORY_PRESSURE_CRITICAL = 1
    THERMAL_HOT = 2
    THERMAL_CRITICAL = 3


class XybridOutputType(IntEnum):
    """Inference output type wire ordinals."""

    TEXT = 0
    AUDIO = 1
    EMBEDDING = 2
    UNKNOWN = 3


class XybridExecutionTarget(IntEnum):
    """Where a result was actually produced.

    Cloud fallback keeps the model id identical on both legs, so this is the
    only way to tell a device answer from a gateway answer.
    """

    LOCAL = 0
    CLOUD = 1


class XybridDownloadState(IntEnum):
    """Lifecycle of the background download behind a speculative load."""

    DOWNLOADING = 0
    READY = 1
    #: Download failed; the cloud keeps serving and ``is_loaded`` never flips.
    FAILED = 2


class XybridThermalState(IntEnum):
    """Thermal state wire ordinals."""

    NORMAL = 0
    WARM = 1
    HOT = 2
    CRITICAL = 3


class _WireReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def _read(self, fmt: str, size: int) -> int | float:
        if self._pos + size > len(self._data):
            raise _WireDecodeError("wire buffer ended early")
        value = struct.unpack_from(fmt, self._data, self._pos)[0]
        self._pos += size
        return value

    def read_u8(self) -> int:
        return int(self._read("<B", 1))

    def read_i32(self) -> int:
        return int(self._read("<i", 4))

    def read_u32(self) -> int:
        return int(self._read("<I", 4))

    def read_u64(self) -> int:
        return int(self._read("<Q", 8))

    def read_f32(self) -> float:
        return float(self._read("<f", 4))

    def read_bool(self) -> bool:
        return self.read_u8() != 0

    def read_string(self) -> str:
        byte_len = self.read_u32()
        if self._pos + byte_len > len(self._data):
            raise _WireDecodeError("wire string ended early")
        # Lossy decode matches the Swift reference (String(decoding:as:), which
        # never throws); strict decoding would leak UnicodeDecodeError past the
        # typed XybridError surface.
        value = self._data[self._pos : self._pos + byte_len].decode("utf-8", errors="replace")
        self._pos += byte_len
        return value

    def read_bytes(self) -> bytes:
        byte_len = self.read_u32()
        if self._pos + byte_len > len(self._data):
            raise _WireDecodeError("wire bytes ended early")
        value = self._data[self._pos : self._pos + byte_len]
        self._pos += byte_len
        return value

    def read_optional(self, reader: Callable[["_WireReader"], _T]) -> _T | None:
        tag = self.read_u8()
        if tag == 0:
            return None
        return reader(self)

    def read_array(self, reader: Callable[["_WireReader"], _T]) -> list[_T]:
        count = self.read_u32()
        return [reader(self) for _ in range(count)]

    def read_f32_array(self) -> list[float]:
        count = self.read_u32()
        byte_len = 4 * count
        if self._pos + byte_len > len(self._data):
            raise _WireDecodeError("wire f32 array ended early")
        values = list(struct.unpack_from(f"<{count}f", self._data, self._pos))
        self._pos += byte_len
        return values


class _WireWriter:
    def __init__(self) -> None:
        self._data = bytearray()

    def write_u8(self, value: int) -> None:
        self._data.extend(struct.pack("<B", value))

    def write_i32(self, value: int) -> None:
        self._data.extend(struct.pack("<i", value))

    def write_u32(self, value: int) -> None:
        self._data.extend(struct.pack("<I", value))

    def write_u64(self, value: int) -> None:
        self._data.extend(struct.pack("<Q", value))

    def write_f32(self, value: float) -> None:
        self._data.extend(struct.pack("<f", value))

    def write_bool(self, value: bool) -> None:
        self.write_u8(1 if value else 0)

    def write_string(self, value: str) -> None:
        data = value.encode("utf-8")
        self.write_u32(len(data))
        self._data.extend(data)

    def write_bytes(self, value: bytes) -> None:
        self.write_u32(len(value))
        self._data.extend(value)

    def write_optional(
        self,
        value: _T | None,
        writer: Callable[["_WireWriter", _T], None],
    ) -> None:
        if value is None:
            self.write_u8(0)
            return
        self.write_u8(1)
        writer(self, value)

    def write_array(
        self,
        values: list[_T],
        writer: Callable[["_WireWriter", _T], None],
    ) -> None:
        self.write_u32(len(values))
        for value in values:
            writer(self, value)

    def write_f32_array(self, values: list[float]) -> None:
        self.write_u32(len(values))
        self._data.extend(struct.pack(f"<{len(values)}f", *values))

    def finalize(self) -> bytes:
        return bytes(self._data)


class XybridError(Exception):
    """Base class for typed xybrid FFI errors."""

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridError":
        tag = reader.read_i32()
        match tag:
            case 0:
                return ModelNotFound(reader.read_string())
            case 1:
                return DirectoryNotFound(reader.read_string())
            case 2:
                return MetadataNotFound(reader.read_string())
            case 3:
                return MetadataInvalid(reader.read_string())
            case 4:
                return LoadError(reader.read_string())
            case 5:
                return InferenceError(reader.read_string())
            case 6:
                return AbortedForCloudFallback(reader.read_string())
            case 7:
                return StreamingNotSupported()
            case 8:
                return NotLoaded()
            case 9:
                return ConfigError(reader.read_string())
            case 10:
                return NetworkError(reader.read_string())
            case 11:
                return Offline(reader.read_string())
            case 12:
                return IoError(reader.read_string())
            case 13:
                return CacheError(reader.read_string())
            case 14:
                return PipelineError(reader.read_string())
            case 15:
                return CircuitOpen(reader.read_string())
            case 16:
                return RateLimited(reader.read_u64())
            case 17:
                return Timeout(reader.read_u64())
            case 18:
                return MissingArtifact(reader.read_string())
            case 19:
                return UnsupportedModelCapability(reader.read_string())
            case 20:
                return UnsupportedBackendCapability(reader.read_string())
            case 21:
                return InvalidImage(reader.read_string())
            case _:
                return LoadError(f"unknown XybridError tag: {tag}")


class _WireDecodeError(XybridError):
    """Raised when a native wire buffer is truncated or malformed."""


@dataclass(frozen=True, slots=True)
class ModelNotFound(XybridError):
    id: str

    def __str__(self) -> str:
        return f"model not found: {self.id}"


@dataclass(frozen=True, slots=True)
class DirectoryNotFound(XybridError):
    path: str

    def __str__(self) -> str:
        return f"directory not found: {self.path}"


@dataclass(frozen=True, slots=True)
class MetadataNotFound(XybridError):
    path: str

    def __str__(self) -> str:
        return f"metadata not found: {self.path}"


@dataclass(frozen=True, slots=True)
class MetadataInvalid(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class LoadError(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class InferenceError(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class AbortedForCloudFallback(XybridError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class StreamingNotSupported(XybridError):
    def __str__(self) -> str:
        return "streaming not supported"


@dataclass(frozen=True, slots=True)
class NotLoaded(XybridError):
    def __str__(self) -> str:
        return "model is not loaded"


@dataclass(frozen=True, slots=True)
class ConfigError(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class NetworkError(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class Offline(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class IoError(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class CacheError(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class PipelineError(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class CircuitOpen(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class RateLimited(XybridError):
    retry_after_secs: int

    def __str__(self) -> str:
        return f"rate limited; retry after {self.retry_after_secs}s"


@dataclass(frozen=True, slots=True)
class Timeout(XybridError):
    timeout_ms: int

    def __str__(self) -> str:
        return f"timed out after {self.timeout_ms}ms"


@dataclass(frozen=True, slots=True)
class MissingArtifact(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class UnsupportedModelCapability(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class UnsupportedBackendCapability(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class InvalidImage(XybridError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class XybridMetadataEntry:
    """Single metadata key/value entry."""

    key: str
    value: str

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridMetadataEntry":
        return XybridMetadataEntry(key=reader.read_string(), value=reader.read_string())

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_string(self.key)
        writer.write_string(self.value)


class XybridEnvelopeKind:
    """Tagged envelope payload."""

    @staticmethod
    def text(text: str) -> "XybridEnvelopeKind":
        return _EnvelopeText(text=text)

    @staticmethod
    def audio(data: bytes) -> "XybridEnvelopeKind":
        return _EnvelopeAudio(bytes=data)

    @staticmethod
    def embedding(values: list[float]) -> "XybridEnvelopeKind":
        return _EnvelopeEmbedding(values=values)

    @staticmethod
    def image(data: bytes, format: str) -> "XybridEnvelopeKind":
        return _EnvelopeImage(bytes=data, format=format)

    @staticmethod
    def multi_part(parts: list["XybridEnvelope"]) -> "XybridEnvelopeKind":
        return _EnvelopeMultiPart(parts=parts)

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridEnvelopeKind":
        tag = reader.read_i32()
        match tag:
            case 0:
                return _EnvelopeText(text=reader.read_string())
            case 1:
                return _EnvelopeAudio(bytes=reader.read_bytes())
            case 2:
                return _EnvelopeEmbedding(values=reader.read_f32_array())
            case 3:
                return _EnvelopeImage(bytes=reader.read_bytes(), format=reader.read_string())
            case 4:
                return _EnvelopeMultiPart(parts=reader.read_array(XybridEnvelope._decode))
            case _:
                raise _WireDecodeError(f"unknown XybridEnvelopeKind tag: {tag}")

    def _encode(self, writer: _WireWriter) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _EnvelopeText(XybridEnvelopeKind):
    text: str

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_i32(0)
        writer.write_string(self.text)


@dataclass(frozen=True, slots=True)
class _EnvelopeAudio(XybridEnvelopeKind):
    bytes: bytes

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_i32(1)
        writer.write_bytes(self.bytes)


@dataclass(frozen=True, slots=True)
class _EnvelopeEmbedding(XybridEnvelopeKind):
    values: list[float]

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_i32(2)
        writer.write_f32_array(self.values)


@dataclass(frozen=True, slots=True)
class _EnvelopeImage(XybridEnvelopeKind):
    bytes: bytes
    format: str

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_i32(3)
        writer.write_bytes(self.bytes)
        writer.write_string(self.format)


@dataclass(frozen=True, slots=True)
class _EnvelopeMultiPart(XybridEnvelopeKind):
    parts: list["XybridEnvelope"]

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_i32(4)
        writer.write_array(self.parts, lambda nested_writer, item: item._encode(nested_writer))


@dataclass(frozen=True, slots=True)
class XybridEnvelope:
    """Inference envelope plus metadata."""

    kind: XybridEnvelopeKind
    metadata: list[XybridMetadataEntry] = field(default_factory=list)

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridEnvelope":
        return XybridEnvelope(
            kind=XybridEnvelopeKind._decode(reader),
            metadata=reader.read_array(XybridMetadataEntry._decode),
        )

    def _encode(self, writer: _WireWriter) -> None:
        self.kind._encode(writer)
        writer.write_array(self.metadata, lambda nested_writer, item: item._encode(nested_writer))

    # -- Factories mirroring the hand-written Swift/Kotlin wrappers --

    @staticmethod
    def text(
        content: str,
        voice: str | None = None,
        speed: float | None = None,
        *,
        voice_id: str | None = None,
    ) -> "XybridEnvelope":
        """Create a text envelope, optionally carrying TTS voice metadata."""

        selected_voice = voice if voice is not None else voice_id
        metadata: list[XybridMetadataEntry] = []
        if selected_voice is not None:
            metadata.append(XybridMetadataEntry(key="voice_id", value=selected_voice))
            metadata.append(XybridMetadataEntry(key="speed", value=str(float(1.0 if speed is None else speed))))
        elif speed is not None:
            metadata.append(XybridMetadataEntry(key="speed", value=str(float(speed))))
        return XybridEnvelope(kind=XybridEnvelopeKind.text(content), metadata=metadata)

    @staticmethod
    def audio(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> "XybridEnvelope":
        """Create an audio envelope with sample-rate and channel metadata."""

        return XybridEnvelope(
            kind=XybridEnvelopeKind.audio(pcm_data),
            metadata=[
                XybridMetadataEntry(key="sample_rate", value=str(sample_rate)),
                XybridMetadataEntry(key="channels", value=str(channels)),
            ],
        )

    @staticmethod
    def embedding(data: list[float]) -> "XybridEnvelope":
        """Create an embedding envelope from a float vector."""

        return XybridEnvelope(kind=XybridEnvelopeKind.embedding(data), metadata=[])

    @staticmethod
    def image(data: bytes, format: str) -> "XybridEnvelope":
        """Create an encoded image envelope for vision-language models.

        Raises:
            ConfigError: If ``format`` is not png, jpeg, jpg, or webp.
        """

        return XybridEnvelope(kind=XybridEnvelopeKind.image(data, _normalize_image_format(format)), metadata=[])

    @staticmethod
    def user_message(text: str, images: "list[XybridEnvelope] | None" = None) -> "XybridEnvelope":
        """Create a multi-part user message from prompt text and image envelopes.

        Raises:
            ConfigError: If any entry in ``images`` is not an image envelope.
        """

        image_parts = [] if images is None else images
        if not all(isinstance(envelope.kind, _EnvelopeImage) for envelope in image_parts):
            raise ConfigError("Envelope.user_message accepts only image envelopes")
        parts = [XybridEnvelope(kind=XybridEnvelopeKind.text(text), metadata=[])]
        parts.extend(image_parts)
        return XybridEnvelope(
            kind=XybridEnvelopeKind.multi_part(parts),
            metadata=[XybridMetadataEntry(key="xybrid.role", value="user")],
        )


def _normalize_image_format(format: str) -> str:
    # Mirrors the Rust-side allowlist (xybrid-core ir/envelope.rs); keep in
    # sync when the ImageFormat enum grows.
    normalized = format.strip().lower()
    match normalized:
        case "jpg":
            return "jpeg"
        case "jpeg" | "png" | "webp":
            return normalized
        case _:
            raise ConfigError(f"Unsupported image format '{format}'. Supported formats: png, jpeg, jpg, webp")


@dataclass(frozen=True, slots=True)
class XybridGenerationConfig:
    """Generation options carried inside run options."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    grammar: str | None = None

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridGenerationConfig":
        return XybridGenerationConfig(
            max_tokens=reader.read_optional(lambda nested_reader: nested_reader.read_u32()),
            temperature=reader.read_optional(lambda nested_reader: nested_reader.read_f32()),
            top_p=reader.read_optional(lambda nested_reader: nested_reader.read_f32()),
            min_p=reader.read_optional(lambda nested_reader: nested_reader.read_f32()),
            top_k=reader.read_optional(lambda nested_reader: nested_reader.read_u32()),
            repetition_penalty=reader.read_optional(lambda nested_reader: nested_reader.read_f32()),
            stop_sequences=reader.read_array(lambda nested_reader: nested_reader.read_string()),
            grammar=reader.read_optional(lambda nested_reader: nested_reader.read_string()),
        )

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_optional(self.max_tokens, lambda nested_writer, value: nested_writer.write_u32(value))
        writer.write_optional(self.temperature, lambda nested_writer, value: nested_writer.write_f32(value))
        writer.write_optional(self.top_p, lambda nested_writer, value: nested_writer.write_f32(value))
        writer.write_optional(self.min_p, lambda nested_writer, value: nested_writer.write_f32(value))
        writer.write_optional(self.top_k, lambda nested_writer, value: nested_writer.write_u32(value))
        writer.write_optional(
            self.repetition_penalty,
            lambda nested_writer, value: nested_writer.write_f32(value),
        )
        writer.write_array(self.stop_sequences, lambda nested_writer, value: nested_writer.write_string(value))
        writer.write_optional(self.grammar, lambda nested_writer, value: nested_writer.write_string(value))


@dataclass(frozen=True, slots=True)
class XybridRunOptions:
    """Options for a model run."""

    generation_config: XybridGenerationConfig | None = None
    abort_on: list[XybridAbortSignal] = field(default_factory=list)
    fallback_to_cloud: bool = False
    max_grace_tokens: int = 0
    correlation_id: str | None = None

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridRunOptions":
        return XybridRunOptions(
            generation_config=reader.read_optional(XybridGenerationConfig._decode),
            abort_on=reader.read_array(lambda nested_reader: XybridAbortSignal(nested_reader.read_i32())),
            fallback_to_cloud=reader.read_bool(),
            max_grace_tokens=reader.read_u32(),
            correlation_id=reader.read_optional(lambda nested_reader: nested_reader.read_string()),
        )

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_optional(
            self.generation_config,
            lambda nested_writer, value: value._encode(nested_writer),
        )
        writer.write_array(self.abort_on, lambda nested_writer, value: nested_writer.write_i32(value.value))
        writer.write_bool(self.fallback_to_cloud)
        writer.write_u32(self.max_grace_tokens)
        writer.write_optional(self.correlation_id, lambda nested_writer, value: nested_writer.write_string(value))


@dataclass(frozen=True, slots=True)
class XybridStageLatency:
    """Per-stage latency metric."""

    stage_id: str
    latency_ms: int

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridStageLatency":
        return XybridStageLatency(stage_id=reader.read_string(), latency_ms=reader.read_u32())

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_string(self.stage_id)
        writer.write_u32(self.latency_ms)


@dataclass(frozen=True, slots=True)
class XybridInferenceMetrics:
    """Inference timing and token metrics."""

    total_ms: int
    ttft_ms: int | None = None
    tokens_per_second: float | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    tokens_out: int | None = None
    stage_latencies_ms: list[XybridStageLatency] = field(default_factory=list)

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridInferenceMetrics":
        return XybridInferenceMetrics(
            total_ms=reader.read_u32(),
            ttft_ms=reader.read_optional(lambda nested_reader: nested_reader.read_u32()),
            tokens_per_second=reader.read_optional(lambda nested_reader: nested_reader.read_f32()),
            prefill_tps=reader.read_optional(lambda nested_reader: nested_reader.read_f32()),
            decode_tps=reader.read_optional(lambda nested_reader: nested_reader.read_f32()),
            tokens_out=reader.read_optional(lambda nested_reader: nested_reader.read_u32()),
            stage_latencies_ms=reader.read_array(XybridStageLatency._decode),
        )

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_u32(self.total_ms)
        writer.write_optional(self.ttft_ms, lambda nested_writer, value: nested_writer.write_u32(value))
        writer.write_optional(self.tokens_per_second, lambda nested_writer, value: nested_writer.write_f32(value))
        writer.write_optional(self.prefill_tps, lambda nested_writer, value: nested_writer.write_f32(value))
        writer.write_optional(self.decode_tps, lambda nested_writer, value: nested_writer.write_f32(value))
        writer.write_optional(self.tokens_out, lambda nested_writer, value: nested_writer.write_u32(value))
        writer.write_array(
            self.stage_latencies_ms,
            lambda nested_writer, value: value._encode(nested_writer),
        )


@dataclass(frozen=True, slots=True)
class XybridDownloadStatus:
    """Download progress and state in one consistent read.

    Taken as a snapshot so a polling caller never sees a torn pair (for
    example ``READY`` alongside a stale ``0.34`` progress).
    """

    state: XybridDownloadState
    #: 0.0 to 1.0.
    progress: float

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridDownloadStatus":
        return XybridDownloadStatus(
            state=XybridDownloadState(reader.read_i32()),
            progress=reader.read_f32(),
        )

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_i32(self.state.value)
        writer.write_f32(self.progress)


@dataclass(frozen=True, slots=True)
class XybridResult:
    """Inference output returned from a model run."""

    envelope: XybridEnvelope
    output_type: XybridOutputType
    model_id: str
    latency_ms: int
    execution_target: XybridExecutionTarget
    metrics: XybridInferenceMetrics

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridResult":
        return XybridResult(
            envelope=XybridEnvelope._decode(reader),
            output_type=XybridOutputType(reader.read_i32()),
            model_id=reader.read_string(),
            latency_ms=reader.read_u32(),
            execution_target=XybridExecutionTarget(reader.read_i32()),
            metrics=XybridInferenceMetrics._decode(reader),
        )

    def _encode(self, writer: _WireWriter) -> None:
        self.envelope._encode(writer)
        writer.write_i32(self.output_type.value)
        writer.write_string(self.model_id)
        writer.write_u32(self.latency_ms)
        writer.write_i32(self.execution_target.value)
        self.metrics._encode(writer)

    # -- Accessors mirroring the hand-written Swift/Kotlin wrappers; payload
    # -- presence is decided by the envelope kind, not output_type.

    @property
    def text(self) -> str | None:
        """Text payload, or ``None`` when the result is not text."""

        kind = self.envelope.kind
        return kind.text if isinstance(kind, _EnvelopeText) else None

    @property
    def audio_bytes(self) -> bytes | None:
        """Audio payload, or ``None`` when the result is not audio."""

        kind = self.envelope.kind
        return kind.bytes if isinstance(kind, _EnvelopeAudio) else None

    @property
    def embedding(self) -> list[float] | None:
        """Embedding vector, or ``None`` when the result is not an embedding."""

        kind = self.envelope.kind
        return kind.values if isinstance(kind, _EnvelopeEmbedding) else None

    @property
    def reasoning_content(self) -> str | None:
        """Chain-of-thought text carried on the ``reasoning_content`` metadata key."""

        for entry in self.envelope.metadata:
            if entry.key == "reasoning_content":
                return entry.value
        return None

    @property
    def success(self) -> bool:
        """Always ``True``; failures raise instead. Shape-compat with Swift/Kotlin."""

        return True

    @property
    def is_failure(self) -> bool:
        """``True`` when the result carries no output."""

        return self.output_type == XybridOutputType.UNKNOWN

    @property
    def latency_seconds(self) -> float:
        """Latency in seconds."""

        return self.latency_ms / 1000.0


@dataclass(frozen=True, slots=True)
class XybridVoiceInfo:
    """Voice metadata for speech-capable models."""

    id: str
    name: str
    gender: str | None = None
    language: str | None = None
    style: str | None = None

    @staticmethod
    def _decode(reader: _WireReader) -> "XybridVoiceInfo":
        return XybridVoiceInfo(
            id=reader.read_string(),
            name=reader.read_string(),
            gender=reader.read_optional(lambda nested_reader: nested_reader.read_string()),
            language=reader.read_optional(lambda nested_reader: nested_reader.read_string()),
            style=reader.read_optional(lambda nested_reader: nested_reader.read_string()),
        )

    def _encode(self, writer: _WireWriter) -> None:
        writer.write_string(self.id)
        writer.write_string(self.name)
        writer.write_optional(self.gender, lambda nested_writer, value: nested_writer.write_string(value))
        writer.write_optional(self.language, lambda nested_writer, value: nested_writer.write_string(value))
        writer.write_optional(self.style, lambda nested_writer, value: nested_writer.write_string(value))

    @property
    def is_male(self) -> bool:
        """``True`` when the voice gender is male."""

        return self.gender == "male"

    @property
    def is_female(self) -> bool:
        """``True`` when the voice gender is female."""

        return self.gender == "female"


_PathOrNone: TypeAlias = Path | None


def _platform_library_name() -> str:
    system = platform.system()
    match system:
        case "Darwin":
            return "libxybrid_bolt.dylib"
        case "Windows":
            return "xybrid_bolt.dll"
        case _:
            return "libxybrid_bolt.so"


def _repo_root_from_package() -> _PathOrNone:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "Cargo.toml").is_file() and (candidate / "target").is_dir():
            return candidate
    return None


def _resolve_library_path() -> Path:
    env_path = os.environ.get(_LIBRARY_ENV)
    if env_path:
        path = Path(env_path)
        if path.is_absolute() and path.is_file():
            return path
        raise ImportError(f"{_LIBRARY_ENV} must point to an existing absolute native library path")

    name = _platform_library_name()
    bundled = Path(__file__).resolve().parent / "_native" / name
    if bundled.is_file():
        return bundled

    root = _repo_root_from_package()
    if root is not None:
        for profile in ("release", "debug"):
            candidate = root / "target" / profile / name
            if candidate.is_file():
                return candidate

    features = "platform-macos" if platform.system() == "Darwin" else "platform-desktop"
    raise ImportError(
        "Could not locate xybrid BoltFFI native library. Set XYBRID_BOLT_LIBRARY "
        "to an absolute path, run tools/scripts/build-python-bolt.sh, or run: "
        f"cargo build -p xybrid-bolt --release --features {features}"
    )


def _load_library() -> ctypes.CDLL:
    if _LIB is not None:
        return _LIB

    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        return _load_library_locked()


def _load_library_locked() -> ctypes.CDLL:
    global _LIB

    path = _resolve_library_path()
    lib = ctypes.CDLL(str(path))
    for symbol in _SYMBOLS:
        getattr(lib, symbol)

    lib.boltffi_free_buf.argtypes = [_FfiBuf]
    lib.boltffi_free_buf.restype = None
    lib.boltffi_free_string.argtypes = [_FfiString]
    lib.boltffi_free_string.restype = None
    lib.boltffi_last_error_message.argtypes = [ctypes.POINTER(_FfiString)]
    lib.boltffi_last_error_message.restype = _FfiStatus
    lib.boltffi_clear_last_error.argtypes = []
    lib.boltffi_clear_last_error.restype = None

    lib.boltffi_json_schema_to_gbnf.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_json_schema_to_gbnf.restype = _FfiBuf
    lib.boltffi_set_thermal_state.argtypes = [ctypes.c_int32]
    lib.boltffi_set_thermal_state.restype = None
    lib.boltffi_clear_thermal_state.argtypes = []
    lib.boltffi_clear_thermal_state.restype = None
    lib.boltffi_set_battery_level.argtypes = [ctypes.c_uint8]
    lib.boltffi_set_battery_level.restype = None
    lib.boltffi_clear_battery_level.argtypes = []
    lib.boltffi_clear_battery_level.restype = None
    lib.boltffi_configure_runtime.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.boltffi_configure_runtime.restype = None
    lib.boltffi_init_sdk_cache_dir.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_init_sdk_cache_dir.restype = None
    lib.boltffi_set_binding.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_set_binding.restype = None
    lib.boltffi_set_api_key.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_set_api_key.restype = None
    lib.boltffi_set_provider_api_key.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_set_provider_api_key.restype = None
    lib.boltffi_set_platform_url.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_set_platform_url.restype = None
    lib.boltffi_set_speculative_cloud.argtypes = [ctypes.c_bool]
    lib.boltffi_set_speculative_cloud.restype = None
    lib.boltffi_is_speculative_cloud_enabled.argtypes = []
    lib.boltffi_is_speculative_cloud_enabled.restype = ctypes.c_bool

    lib.boltffi_xybrid_model_from_registry.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_xybrid_model_from_registry.restype = ctypes.c_void_p
    lib.boltffi_xybrid_model_from_registry_speculative.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_xybrid_model_from_registry_speculative.restype = ctypes.c_void_p
    lib.boltffi_xybrid_model_is_cloud_serving.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_is_cloud_serving.restype = ctypes.c_bool
    lib.boltffi_xybrid_model_download_status.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_download_status.restype = _FfiBuf
    lib.boltffi_xybrid_model_await_download.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.boltffi_xybrid_model_await_download.restype = _FfiBuf
    lib.boltffi_xybrid_model_from_directory.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_xybrid_model_from_directory.restype = ctypes.c_void_p
    lib.boltffi_xybrid_model_from_bundle.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_xybrid_model_from_bundle.restype = ctypes.c_void_p
    lib.boltffi_xybrid_model_from_huggingface.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_xybrid_model_from_huggingface.restype = ctypes.c_void_p
    lib.boltffi_xybrid_model_free.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_free.restype = None
    lib.boltffi_xybrid_model_model_id.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_model_id.restype = _FfiBuf
    lib.boltffi_xybrid_model_version.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_version.restype = _FfiBuf
    lib.boltffi_xybrid_model_output_type.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_output_type.restype = ctypes.c_int32
    lib.boltffi_xybrid_model_is_loaded.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_is_loaded.restype = ctypes.c_bool
    lib.boltffi_xybrid_model_supports_streaming.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_supports_streaming.restype = ctypes.c_bool
    lib.boltffi_xybrid_model_is_llm.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_is_llm.restype = ctypes.c_bool
    lib.boltffi_xybrid_model_has_voices.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_has_voices.restype = ctypes.c_bool
    lib.boltffi_xybrid_model_voices.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_voices.restype = _FfiBuf
    lib.boltffi_xybrid_model_default_voice.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_default_voice.restype = _FfiBuf
    lib.boltffi_xybrid_model_voice.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.boltffi_xybrid_model_voice.restype = _FfiBuf
    lib.boltffi_xybrid_model_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.boltffi_xybrid_model_run.restype = _FfiBuf
    lib.boltffi_xybrid_model_warmup.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_warmup.restype = _FfiBuf
    lib.boltffi_xybrid_model_unload.argtypes = [ctypes.c_void_p]
    lib.boltffi_xybrid_model_unload.restype = _FfiBuf

    _LIB = lib
    return lib


def _bytes_arg(data: bytes) -> ctypes.c_char_p:
    return ctypes.c_char_p(data)


def _string_arg(value: str) -> tuple[ctypes.c_char_p, int]:
    data = value.encode("utf-8")
    return _bytes_arg(data), len(data)


def _copy_and_free_buf(buf: _FfiBuf) -> bytes:
    lib = _load_library()
    try:
        if buf.ptr is None or buf.len == 0:
            return b""
        return ctypes.string_at(buf.ptr, buf.len)
    finally:
        lib.boltffi_free_buf(buf)


def _decode_result(data: bytes, reader: Callable[[_WireReader], _T]) -> _T:
    wire = _WireReader(data)
    tag = wire.read_u8()
    if tag == 0:
        return reader(wire)
    raise XybridError._decode(wire)


def _decode_void_result(data: bytes) -> None:
    _decode_result(data, lambda reader: None)


def _take_last_error_message() -> str:
    lib = _load_library()
    out = _FfiString()
    status = lib.boltffi_last_error_message(ctypes.byref(out))
    if status.code != 0:
        raise LoadError(f"boltffi_last_error_message failed with status {status.code}")
    try:
        if out.ptr is None or out.len == 0:
            return ""
        return ctypes.string_at(out.ptr, out.len).decode("utf-8", errors="replace")
    finally:
        lib.boltffi_free_string(out)


_LAST_ERROR_PATTERN: Final = re.compile(r"^(?P<variant>[A-Za-z0-9_]+)(?: \{ (?P<field>[a-zA-Z0-9_]+): (?P<value>.*) \})?$")


def _parse_last_error(message: str) -> XybridError:
    match = _LAST_ERROR_PATTERN.match(message)
    if match is None:
        return LoadError(message)
    variant = match.group("variant")
    value_text = match.group("value")
    value = ""
    if value_text is not None:
        # The value is Rust `Debug` output, which is JSON-like but not JSON
        # (non-ASCII escapes as `\u{e9}`). Fall back to the raw text rather
        # than let a decode error escape the typed XybridError surface.
        try:
            value = json.loads(value_text)
        except ValueError:
            value = value_text.strip('"')
    match variant:
        case "ModelNotFound":
            return ModelNotFound(value)
        case "DirectoryNotFound":
            return DirectoryNotFound(value)
        case "MetadataNotFound":
            return MetadataNotFound(value)
        case "MetadataInvalid":
            return MetadataInvalid(value)
        case "LoadError":
            return LoadError(value)
        case "InferenceError":
            return InferenceError(value)
        case "AbortedForCloudFallback":
            return AbortedForCloudFallback(value)
        case "StreamingNotSupported":
            return StreamingNotSupported()
        case "NotLoaded":
            return NotLoaded()
        case "ConfigError":
            return ConfigError(value)
        case "NetworkError":
            return NetworkError(value)
        case "Offline":
            return Offline(value)
        case "IoError":
            return IoError(value)
        case "CacheError":
            return CacheError(value)
        case "PipelineError":
            return PipelineError(value)
        case "CircuitOpen":
            return CircuitOpen(value)
        case "RateLimited" | "Timeout" as variant_name:
            # The payload is u64 in the wire contract; if the Debug-string
            # fallback yielded non-numeric text, degrade to LoadError rather
            # than let ValueError escape the typed surface.
            try:
                seconds_or_ms = int(value)
            except (TypeError, ValueError):
                return LoadError(message)
            return RateLimited(seconds_or_ms) if variant_name == "RateLimited" else Timeout(seconds_or_ms)
        case "MissingArtifact":
            return MissingArtifact(value)
        case "UnsupportedModelCapability":
            return UnsupportedModelCapability(value)
        case "UnsupportedBackendCapability":
            return UnsupportedBackendCapability(value)
        case "InvalidImage":
            return InvalidImage(value)
        case _:
            return LoadError(message)


def _raise_last_error() -> None:
    raise _parse_last_error(_take_last_error_message())


def _encode_optional_record(value: XybridRunOptions | None) -> bytes:
    writer = _WireWriter()
    writer.write_optional(value, lambda nested_writer, item: item._encode(nested_writer))
    return writer.finalize()


def set_thermal_state(state: XybridThermalState) -> None:
    """Set the process thermal state hint."""

    _load_library().boltffi_set_thermal_state(state.value)


def clear_thermal_state() -> None:
    """Clear the process thermal state hint."""

    _load_library().boltffi_clear_thermal_state()


def set_battery_level(percent: int) -> None:
    """Set the process battery level hint, clamped to 0..=100."""

    # The C parameter is u8; clamp like the Kotlin wrapper (coerceIn) instead
    # of letting ctypes silently wrap out-of-range values modulo 256.
    _load_library().boltffi_set_battery_level(max(0, min(100, int(percent))))


def clear_battery_level() -> None:
    """Clear the process battery level hint."""

    _load_library().boltffi_clear_battery_level()


def init_sdk_cache_dir(cache_dir: str) -> None:
    """Set the SDK cache directory."""

    ptr, size = _string_arg(cache_dir)
    _load_library().boltffi_init_sdk_cache_dir(ptr, size)


def set_binding(binding: str) -> None:
    """Set the foreign binding name reported to xybrid."""

    ptr, size = _string_arg(binding)
    _load_library().boltffi_set_binding(ptr, size)


def set_api_key(api_key: str) -> None:
    """Set the xybrid API key."""

    ptr, size = _string_arg(api_key)
    _load_library().boltffi_set_api_key(ptr, size)


def set_platform_url(url: str) -> None:
    """Point the cloud gateway at a platform base URL (staging, self-hosted).

    Held in process memory rather than the environment. Pass a bare base URL;
    the ``/v1`` suffix is applied internally.
    """

    ptr, size = _string_arg(url)
    _load_library().boltffi_set_platform_url(ptr, size)


def set_speculative_cloud(enabled: bool) -> None:
    """Enable speculative cloud fallback globally.

    A registry model that is not downloaded yet is then served from the gateway
    while the weights download in the background. Only takes effect when an API
    key resolves. Speculation is LLM/chat only, so prefer
    :meth:`XybridModel.from_registry_speculative` when the process also loads
    ASR/TTS models, which cannot be served this way.
    """

    _load_library().boltffi_set_speculative_cloud(enabled)


def is_speculative_cloud_enabled() -> bool:
    """Return whether the global speculative-cloud default is on."""

    return bool(_load_library().boltffi_is_speculative_cloud_enabled())


def set_provider_api_key(provider: str, api_key: str) -> None:
    """Set a provider-specific API key."""

    provider_ptr, provider_size = _string_arg(provider)
    api_key_ptr, api_key_size = _string_arg(api_key)
    _load_library().boltffi_set_provider_api_key(provider_ptr, provider_size, api_key_ptr, api_key_size)


def configure_runtime(api_key: str | None = None, gateway_url: str | None = None, ingest_url: str | None = None) -> None:
    """Configure xybrid runtime auth and endpoint overrides."""

    writer = _WireWriter()
    writer.write_optional(api_key, lambda nested_writer, value: nested_writer.write_string(value))
    api_key_bytes = writer.finalize()
    writer = _WireWriter()
    writer.write_optional(gateway_url, lambda nested_writer, value: nested_writer.write_string(value))
    gateway_url_bytes = writer.finalize()
    writer = _WireWriter()
    writer.write_optional(ingest_url, lambda nested_writer, value: nested_writer.write_string(value))
    ingest_url_bytes = writer.finalize()
    _load_library().boltffi_configure_runtime(
        _bytes_arg(api_key_bytes),
        len(api_key_bytes),
        _bytes_arg(gateway_url_bytes),
        len(gateway_url_bytes),
        _bytes_arg(ingest_url_bytes),
        len(ingest_url_bytes),
    )


def json_schema_to_gbnf(schema_json: str) -> str:
    """Convert a JSON Schema string into GBNF grammar.

    Raises:
        XybridError: If the native converter rejects the schema.
    """

    ptr, size = _string_arg(schema_json)
    data = _copy_and_free_buf(_load_library().boltffi_json_schema_to_gbnf(ptr, size))
    return _decode_result(data, lambda reader: reader.read_string())


class XybridModel:
    """Opaque xybrid model handle."""

    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle
        self._handle_lock = threading.Lock()

    @classmethod
    def from_registry(cls, id: str) -> "XybridModel":
        """Load a model from the xybrid registry.

        Raises:
            XybridError: If the native constructor fails.
        """

        ptr, size = _string_arg(id)
        handle = _load_library().boltffi_xybrid_model_from_registry(ptr, size)
        if handle is None:
            _raise_last_error()
        return cls(handle)

    @classmethod
    def from_registry_speculative(cls, id: str) -> "XybridModel":
        """Load from the registry, serving from the cloud while it downloads.

        Returns almost immediately instead of blocking on the download, and
        switches to on-device by itself once the weights land. Requires a
        resolvable API key and an uncached model; otherwise this behaves exactly
        like :meth:`from_registry`. LLM/chat models only.

        Raises:
            XybridError: If the native constructor fails.
        """

        ptr, size = _string_arg(id)
        handle = _load_library().boltffi_xybrid_model_from_registry_speculative(ptr, size)
        if handle is None:
            _raise_last_error()
        return cls(handle)

    @classmethod
    def from_directory(cls, path: str) -> "XybridModel":
        """Load a model from a local model directory.

        Raises:
            XybridError: If the native constructor fails.
        """

        ptr, size = _string_arg(path)
        handle = _load_library().boltffi_xybrid_model_from_directory(ptr, size)
        if handle is None:
            _raise_last_error()
        return cls(handle)

    @classmethod
    def from_bundle(cls, path: str) -> "XybridModel":
        """Load a model from a local bundle.

        Raises:
            XybridError: If the native constructor fails.
        """

        ptr, size = _string_arg(path)
        handle = _load_library().boltffi_xybrid_model_from_bundle(ptr, size)
        if handle is None:
            _raise_last_error()
        return cls(handle)

    @classmethod
    def from_huggingface(cls, repo: str) -> "XybridModel":
        """Load a model from a Hugging Face repository.

        Raises:
            XybridError: If the native constructor fails.
        """

        ptr, size = _string_arg(repo)
        handle = _load_library().boltffi_xybrid_model_from_huggingface(ptr, size)
        if handle is None:
            _raise_last_error()
        return cls(handle)

    def __enter__(self) -> "XybridModel":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if _LIB is not None:
            self._free_handle()

    def close(self) -> None:
        """Free the native model handle.

        Idempotent and safe against a concurrent ``__del__``. Do not call
        while another thread has a call in flight on this model.
        """

        self._free_handle()

    def _free_handle(self) -> None:
        # Take-then-null under the lock so close()/__del__ racing from two
        # threads cannot both see a live handle and double-free it.
        with self._handle_lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            _load_library().boltffi_xybrid_model_free(handle)

    def _require_handle(self) -> int:
        if self._handle is None:
            raise NotLoaded()
        return self._handle

    @property
    def model_id(self) -> str:
        """Return the model identifier."""

        data = _copy_and_free_buf(_load_library().boltffi_xybrid_model_model_id(self._require_handle()))
        return _WireReader(data).read_string()

    @property
    def version(self) -> str:
        """Return the model version."""

        data = _copy_and_free_buf(_load_library().boltffi_xybrid_model_version(self._require_handle()))
        return _WireReader(data).read_string()

    @property
    def output_type(self) -> XybridOutputType:
        """Return the model output type."""

        return XybridOutputType(_load_library().boltffi_xybrid_model_output_type(self._require_handle()))

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is loaded. ``False`` after :meth:`close`."""

        if self._handle is None:
            return False
        return bool(_load_library().boltffi_xybrid_model_is_loaded(self._handle))

    @property
    def is_cloud_serving(self) -> bool:
        """Return whether runs are currently answered by the cloud.

        ``True`` while a speculative model's weights are still downloading.
        ``False`` for ordinary local models. This predicts the next run;
        :attr:`XybridResult.execution_target` reports what a run that already
        happened actually did.
        """

        if self._handle is None:
            return False
        return bool(_load_library().boltffi_xybrid_model_is_cloud_serving(self._handle))

    def download_status(self) -> XybridDownloadStatus:
        """Return download progress and state in one consistent read.

        Reports ``READY`` at 1.0 for an ordinary local model, so callers need no
        special case. Poll this to drive a progress bar.
        """

        # Infallible on the Rust side, so the buffer holds a bare
        # XybridDownloadStatus with no leading result discriminant — decoding it
        # through `_decode_result` would eat the state tag as the discriminant
        # and shift every field after it.
        data = _copy_and_free_buf(
            _load_library().boltffi_xybrid_model_download_status(self._require_handle())
        )
        return XybridDownloadStatus._decode(_WireReader(data))

    def await_download(self, timeout_ms: int) -> XybridDownloadStatus:
        """Block until the download finishes or ``timeout_ms`` elapses.

        The convenience helper for "tell me when it is on-device". A
        ``timeout_ms`` of 0 makes this a non-blocking read, identical to
        :meth:`download_status`. Returns immediately for a non-speculative
        model.
        """

        # Also infallible — see `download_status` for why this must not go
        # through `_decode_result`.
        data = _copy_and_free_buf(
            _load_library().boltffi_xybrid_model_await_download(
                self._require_handle(), max(0, timeout_ms)
            )
        )
        return XybridDownloadStatus._decode(_WireReader(data))

    @property
    def supports_streaming(self) -> bool:
        """Return whether streaming is supported."""

        return bool(_load_library().boltffi_xybrid_model_supports_streaming(self._require_handle()))

    @property
    def is_llm(self) -> bool:
        """Return whether the model is an LLM."""

        return bool(_load_library().boltffi_xybrid_model_is_llm(self._require_handle()))

    @property
    def has_voices(self) -> bool:
        """Return whether the model exposes voices."""

        return bool(_load_library().boltffi_xybrid_model_has_voices(self._require_handle()))

    def voices(self) -> list[XybridVoiceInfo]:
        """Return all voices for the model."""

        data = _copy_and_free_buf(_load_library().boltffi_xybrid_model_voices(self._require_handle()))
        return _WireReader(data).read_array(XybridVoiceInfo._decode)

    def default_voice(self) -> XybridVoiceInfo | None:
        """Return the default voice, if one exists."""

        data = _copy_and_free_buf(_load_library().boltffi_xybrid_model_default_voice(self._require_handle()))
        return _WireReader(data).read_optional(XybridVoiceInfo._decode)

    def voice(self, voice_id: str) -> XybridVoiceInfo | None:
        """Return a voice by identifier, if one exists."""

        ptr, size = _string_arg(voice_id)
        data = _copy_and_free_buf(_load_library().boltffi_xybrid_model_voice(self._require_handle(), ptr, size))
        return _WireReader(data).read_optional(XybridVoiceInfo._decode)

    def run(self, envelope: XybridEnvelope, options: XybridRunOptions | None = None) -> XybridResult:
        """Run inference.

        Raises:
            XybridError: If inference fails.
        """

        envelope_writer = _WireWriter()
        envelope._encode(envelope_writer)
        envelope_bytes = envelope_writer.finalize()
        options_bytes = _encode_optional_record(options)
        data = _copy_and_free_buf(
            _load_library().boltffi_xybrid_model_run(
                self._require_handle(),
                _bytes_arg(envelope_bytes),
                len(envelope_bytes),
                _bytes_arg(options_bytes),
                len(options_bytes),
            )
        )
        return _decode_result(data, XybridResult._decode)

    def warmup(self) -> None:
        """Warm up the model.

        Raises:
            XybridError: If warmup fails.
        """

        data = _copy_and_free_buf(_load_library().boltffi_xybrid_model_warmup(self._require_handle()))
        _decode_void_result(data)

    def unload(self) -> None:
        """Unload the model.

        Raises:
            XybridError: If unload fails.
        """

        data = _copy_and_free_buf(_load_library().boltffi_xybrid_model_unload(self._require_handle()))
        _decode_void_result(data)


__all__ = [
    "AbortedForCloudFallback",
    "CacheError",
    "CircuitOpen",
    "ConfigError",
    "DirectoryNotFound",
    "InferenceError",
    "InvalidImage",
    "IoError",
    "LoadError",
    "MetadataInvalid",
    "MetadataNotFound",
    "MissingArtifact",
    "ModelNotFound",
    "NetworkError",
    "NotLoaded",
    "Offline",
    "PipelineError",
    "RateLimited",
    "StreamingNotSupported",
    "Timeout",
    "UnsupportedBackendCapability",
    "UnsupportedModelCapability",
    "XybridAbortSignal",
    "XybridDownloadState",
    "XybridDownloadStatus",
    "XybridEnvelope",
    "XybridEnvelopeKind",
    "XybridError",
    "XybridExecutionTarget",
    "XybridGenerationConfig",
    "XybridInferenceMetrics",
    "XybridMessageRole",
    "XybridMetadataEntry",
    "XybridModel",
    "XybridOutputType",
    "XybridResult",
    "XybridRunOptions",
    "XybridStageLatency",
    "XybridThermalState",
    "XybridVoiceInfo",
    "clear_battery_level",
    "clear_thermal_state",
    "configure_runtime",
    "init_sdk_cache_dir",
    "is_speculative_cloud_enabled",
    "json_schema_to_gbnf",
    "set_api_key",
    "set_battery_level",
    "set_binding",
    "set_platform_url",
    "set_provider_api_key",
    "set_speculative_cloud",
    "set_thermal_state",
]
