"""Python SDK for local-first xybrid inference.

Xybrid runs models on-device by default with no account required. Passing an
API key to :func:`init` configures cloud authentication. Observability export
requires explicit configuration. See https://docs.xybrid.dev.
"""

from __future__ import annotations

import threading
from typing import Final
from urllib.parse import urlsplit, urlunsplit

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
from ._errors import ConfigError, XybridError  # noqa: E402

# Attribute registry requests and telemetry to this SDK from the first call.
# `init()` is optional (local inference needs none of it), so registering
# there left every app that skips it reported as `rust`.
_bolt.set_binding("python")

_INIT_LOCK: Final = threading.Lock()
_INITIALIZED = False


def init(api_key: str | None = None, gateway_url: str | None = None, ingest_url: str | None = None) -> None:
    """Initialize the Xybrid runtime.

    This function is idempotent and thread-safe. The first call applies runtime
    configuration; later calls are no-ops even if they pass different arguments.

    ``api_key`` configures cloud authentication and ``gateway_url`` overrides
    the LLM gateway. Neither enables observability or changes an existing
    exporter. Configure export separately with ``XybridObservabilityConfig``.

    For legacy callers, a nonblank ``ingest_url`` explicitly enables
    bearer-authenticated, metadata-only OTLP traces using ``api_key`` (required
    in this form). A base URL maps to ``/v1/traces``; an explicit path is kept.
    The destination must accept OTLP/HTTP protobuf, not the retired JSON event
    format. Invalid configuration raises ``ConfigError``; failed initialization
    may be retried. Desktop resource observers are handled inside Rust.
    """

    global _INITIALIZED
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        endpoint = ingest_url.strip() if ingest_url is not None else ""
        if endpoint:
            key = api_key.strip() if api_key is not None else ""
            if not key:
                raise ConfigError("ingest_url requires a nonblank api_key")
            try:
                url = urlsplit(endpoint)
                if url.scheme not in ("http", "https") or not url.hostname:
                    raise ValueError("not an absolute HTTP(S) URL")
                # Accessing port also validates malformed/out-of-range ports.
                _ = url.port
            except ValueError:
                raise ConfigError("ingest_url must be an absolute HTTP(S) URL") from None
            endpoint = urlunsplit(
                url._replace(path=url.path if url.path not in ("", "/") else "/v1/traces")
            )
            config = _bolt.XybridObservabilityConfig.xybrid(key)
            config.set_traces_endpoint(endpoint)
            config.set_metadata_only(True)
            config.init()
        _bolt.configure_runtime(api_key=api_key, gateway_url=gateway_url)
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
            tools=[],
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
            tools=[],
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
