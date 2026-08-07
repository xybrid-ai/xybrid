"""Python SDK for local-first xybrid inference.

Xybrid runs models on-device by default with no account required. Passing an
API key to :func:`init` enables optional platform features such as telemetry and
cloud routing on top of the same local runtime. See https://docs.xybrid.dev.
"""

from __future__ import annotations

import threading
from typing import Final

from . import _bolt, _errors, _sugar
from ._bolt import *  # noqa: F403 -- re-export the generated surface; _bolt defines __all__
from ._bolt import XybridEnvelope, XybridGenerationConfig, XybridModel, XybridResult, XybridVoiceInfo

_errors._install()
_sugar.install()

# Imported after the star-import so the typed exceptions win the name
# `XybridError`. The generated class of that name is the error *payload* union
# (plain data, not an exception); it stays reachable as `_bolt.XybridError`,
# and each of its variants is still re-exported under its own name.
from ._errors import *  # noqa: E402,F403 -- ordering is deliberate, see above
from ._errors import XybridError  # noqa: E402

_INIT_LOCK: Final = threading.Lock()
_INITIALIZED = False


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


Model = XybridModel
Envelope = XybridEnvelope
GenerationConfig = XybridGenerationConfig
Result = XybridResult
VoiceInfo = XybridVoiceInfo

__all__ = sorted(
    {
        *_bolt.__all__,
        *_errors.__all__,
        "Envelope",
        "GenerationConfig",
        "GenerationConfigs",
        "Model",
        "Result",
        "VoiceInfo",
        "init",
        "is_initialized",
    }
)
