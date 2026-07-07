"""Python SDK for local-first xybrid inference.

Xybrid runs models on-device by default with no account required. Passing an
API key to :func:`init` enables optional platform features such as telemetry and
cloud routing on top of the same local runtime. See https://docs.xybrid.dev.
"""

from __future__ import annotations

import threading
from typing import Final

from . import _bolt
from ._bolt import *  # noqa: F403 -- re-export the generated-style surface; _bolt defines __all__

_INIT_LOCK: Final = threading.Lock()
_INITIALIZED = False
_REASONING_METADATA_KEY: Final = "reasoning_content"


def init(api_key: str | None = None, gateway_url: str | None = None, ingest_url: str | None = None) -> None:
    """Initialize the Xybrid runtime.

    This function is idempotent and thread-safe. The first call registers the
    Python binding identifier and applies runtime configuration; later calls are
    no-ops even if they pass different arguments.

    Without ``api_key``, xybrid runs fully on-device and telemetry remains
    disabled. Passing ``api_key`` starts the platform telemetry exporter;
    ``gateway_url`` overrides the LLM gateway and ``ingest_url`` overrides the
    telemetry ingest endpoint. Desktop battery and thermal observers are handled
    inside Rust, so the Python SDK does not register host observers.
    """

    global _INITIALIZED
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        _bolt.set_binding("python")
        _bolt.configure_runtime(api_key=api_key, gateway_url=gateway_url, ingest_url=ingest_url)
        _INITIALIZED = True


def is_initialized() -> bool:
    """Return whether :func:`init` has run successfully."""

    with _INIT_LOCK:
        return _INITIALIZED


def _envelope_text(
    content: str,
    voice: str | None = None,
    speed: float | None = None,
    *,
    voice_id: str | None = None,
) -> XybridEnvelope:
    """Create a text envelope, optionally carrying TTS voice metadata."""

    selected_voice = voice if voice is not None else voice_id
    metadata: list[XybridMetadataEntry] = []
    if selected_voice is not None:
        metadata.append(XybridMetadataEntry(key="voice_id", value=selected_voice))
        metadata.append(XybridMetadataEntry(key="speed", value=str(1.0 if speed is None else speed)))
    elif speed is not None:
        metadata.append(XybridMetadataEntry(key="speed", value=str(speed)))
    return XybridEnvelope(kind=XybridEnvelopeKind.text(content), metadata=metadata)


def _envelope_audio(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> XybridEnvelope:
    """Create an audio envelope with sample-rate and channel metadata."""

    return XybridEnvelope(
        kind=XybridEnvelopeKind.audio(pcm_data),
        metadata=[
            XybridMetadataEntry(key="sample_rate", value=str(sample_rate)),
            XybridMetadataEntry(key="channels", value=str(channels)),
        ],
    )


def _envelope_embedding(data: list[float]) -> XybridEnvelope:
    """Create an embedding envelope from a float vector."""

    return XybridEnvelope(kind=XybridEnvelopeKind.embedding(data), metadata=[])


def _normalize_image_format(format: str) -> str:
    normalized = format.strip().lower()
    match normalized:
        case "jpg":
            return "jpeg"
        case "jpeg" | "png" | "webp":
            return normalized
        case _:
            raise ConfigError(f"Unsupported image format '{format}'. Supported formats: png, jpeg, jpg, webp")


def _envelope_image(data: bytes, format: str) -> XybridEnvelope:
    """Create an encoded image envelope for vision-language models."""

    return XybridEnvelope(kind=XybridEnvelopeKind.image(data, _normalize_image_format(format)), metadata=[])


def _is_image_envelope(envelope: XybridEnvelope) -> bool:
    return hasattr(envelope.kind, "format")


def _envelope_user_message(text: str, images: list[XybridEnvelope] | None = None) -> XybridEnvelope:
    """Create a multi-part user message from prompt text and image envelopes."""

    image_parts = [] if images is None else images
    if not all(_is_image_envelope(envelope) for envelope in image_parts):
        raise ConfigError("Envelope.user_message accepts only image envelopes")
    parts = [XybridEnvelope(kind=XybridEnvelopeKind.text(text), metadata=[])]
    parts.extend(image_parts)
    return XybridEnvelope(
        kind=XybridEnvelopeKind.multi_part(parts),
        metadata=[XybridMetadataEntry(key="xybrid.role", value="user")],
    )


def _result_text(result: XybridResult) -> str | None:
    return getattr(result.envelope.kind, "text", None)


def _result_audio_bytes(result: XybridResult) -> bytes | None:
    if result.output_type != XybridOutputType.AUDIO:
        return None
    value = getattr(result.envelope.kind, "bytes", None)
    return value if isinstance(value, bytes) else None


def _result_embedding(result: XybridResult) -> list[float] | None:
    value = getattr(result.envelope.kind, "values", None)
    return value if isinstance(value, list) else None


def _result_reasoning_content(result: XybridResult) -> str | None:
    for entry in result.envelope.metadata:
        if entry.key == _REASONING_METADATA_KEY:
            return entry.value
    return None


def _result_is_failure(result: XybridResult) -> bool:
    return result.output_type == XybridOutputType.UNKNOWN


def _result_latency_seconds(result: XybridResult) -> float:
    return result.latency_ms / 1000.0


def _voice_is_male(voice: XybridVoiceInfo) -> bool:
    return voice.gender == "male"


def _voice_is_female(voice: XybridVoiceInfo) -> bool:
    return voice.gender == "female"


class GenerationConfigs:
    """Preset factories for common LLM generation configurations."""

    @staticmethod
    def greedy() -> XybridGenerationConfig:
        """Return a deterministic greedy decoding preset."""

        return XybridGenerationConfig(
            max_tokens=None,
            temperature=0.0,
            top_p=1.0,
            min_p=None,
            top_k=0,
            repetition_penalty=None,
            stop_sequences=[],
            grammar=None,
        )

    @staticmethod
    def creative() -> XybridGenerationConfig:
        """Return a higher-temperature creative decoding preset."""

        return XybridGenerationConfig(
            max_tokens=None,
            temperature=0.9,
            top_p=0.95,
            min_p=None,
            top_k=50,
            repetition_penalty=None,
            stop_sequences=[],
            grammar=None,
        )


setattr(XybridEnvelope, "text", staticmethod(_envelope_text))
setattr(XybridEnvelope, "audio", staticmethod(_envelope_audio))
setattr(XybridEnvelope, "embedding", staticmethod(_envelope_embedding))
setattr(XybridEnvelope, "image", staticmethod(_envelope_image))
setattr(XybridEnvelope, "user_message", staticmethod(_envelope_user_message))
setattr(XybridResult, "text", property(_result_text))
setattr(XybridResult, "audio_bytes", property(_result_audio_bytes))
setattr(XybridResult, "embedding", property(_result_embedding))
setattr(XybridResult, "reasoning_content", property(_result_reasoning_content))
setattr(XybridResult, "is_failure", property(_result_is_failure))
setattr(XybridResult, "latency_seconds", property(_result_latency_seconds))
setattr(XybridVoiceInfo, "is_male", property(_voice_is_male))
setattr(XybridVoiceInfo, "is_female", property(_voice_is_female))

Model = XybridModel
Envelope = XybridEnvelope
GenerationConfig = XybridGenerationConfig
Result = XybridResult
VoiceInfo = XybridVoiceInfo

__all__ = [
    *_bolt.__all__,
    "Envelope",
    "GenerationConfig",
    "GenerationConfigs",
    "Model",
    "Result",
    "VoiceInfo",
    "init",
    "is_initialized",
]
