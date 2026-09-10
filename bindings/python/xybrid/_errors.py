"""Typed exceptions raised by native xybrid calls.

boltffi models errors as *data*: ``_bolt.XybridError`` is a union of frozen
dataclasses, and every failing call raises the single wrapper
``_bolt.XybridErrorException`` carrying one of them in ``.error``. That is
faithful to the Rust enum but not how Python callers handle failure — they
write ``except xybrid.ModelNotFound``.

This module restores the typed hierarchy the SDK documents, on top of the
generated payloads rather than beside them: every class here subclasses
``_bolt.XybridErrorException``, so ``except xybrid.XybridError`` still catches
everything, and ``exc.error`` still holds the generated payload.

Wiring: the generated ``_boltffi_error_exception`` resolves the exception class
for a payload by looking up ``f"{type(payload).__name__}Exception"`` in the
``_bolt`` module globals. Injecting those names (``_install``, called at import
from ``xybrid/__init__.py``) is enough to make every native call raise the
typed class — no generated code is modified.
"""

from __future__ import annotations

from typing import Any, ClassVar

from . import _bolt

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
    "XybridError",
]


class XybridError(_bolt.XybridErrorException):
    """Base class for every error raised by a native xybrid call.

    The generated payload stays available as ``.error``; its fields are also
    readable straight off the exception (``exc.id``, ``exc.message``).
    """

    __slots__ = ()

    #: Generated payload this exception wraps. ``None`` on the base class,
    #: which accepts any payload so an unrecognised variant still raises a
    #: xybrid error rather than a bare ``RuntimeError``.
    _payload: ClassVar[type[Any] | None] = None

    def __init__(self, error: Any = None) -> None:
        if not isinstance(error, _bolt.XybridError):
            payload_type = type(self)._payload
            if payload_type is None:
                raise TypeError(f"{type(self).__name__}() requires a XybridError payload")
            error = payload_type() if error is None else payload_type(error)
        super().__init__(error)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so `error` itself never
        # recurses here once __init__ has run.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            payload = object.__getattribute__(self, "error")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(payload, name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}") from None

    def __str__(self) -> str:
        payload = getattr(self, "error", None)
        for field in ("message", "reason"):
            value = getattr(payload, field, None)
            if isinstance(value, str):
                return value
        return repr(payload)


class ModelNotFound(XybridError):
    """The registry has no model with this id."""

    __slots__ = ()
    _payload = _bolt.XybridErrorModelNotFound

    def __str__(self) -> str:
        return f"model not found: {self.id}"


class DirectoryNotFound(XybridError):
    """The model directory does not exist."""

    __slots__ = ()
    _payload = _bolt.XybridErrorDirectoryNotFound

    def __str__(self) -> str:
        return f"directory not found: {self.path}"


class MetadataNotFound(XybridError):
    """The directory holds no ``model_metadata.json``."""

    __slots__ = ()
    _payload = _bolt.XybridErrorMetadataNotFound

    def __str__(self) -> str:
        return f"metadata not found: {self.path}"


class MetadataInvalid(XybridError):
    """``model_metadata.json`` could not be parsed."""

    __slots__ = ()
    _payload = _bolt.XybridErrorMetadataInvalid


class LoadError(XybridError):
    """The model could not be loaded."""

    __slots__ = ()
    _payload = _bolt.XybridErrorLoadError


class InferenceError(XybridError):
    """Inference failed."""

    __slots__ = ()
    _payload = _bolt.XybridErrorInferenceError


class AbortedForCloudFallback(XybridError):
    """Local execution was aborted so the run could fall back to cloud."""

    __slots__ = ()
    _payload = _bolt.XybridErrorAbortedForCloudFallback


class StreamingNotSupported(XybridError):
    """This model cannot stream."""

    __slots__ = ()
    _payload = _bolt.XybridErrorStreamingNotSupported

    def __str__(self) -> str:
        return "streaming not supported"


class NotLoaded(XybridError):
    """The model handle carries no loaded model."""

    __slots__ = ()
    _payload = _bolt.XybridErrorNotLoaded

    def __str__(self) -> str:
        return "model is not loaded"


class ConfigError(XybridError):
    """Invalid configuration or input."""

    __slots__ = ()
    _payload = _bolt.XybridErrorConfigError


class NetworkError(XybridError):
    """A network call failed."""

    __slots__ = ()
    _payload = _bolt.XybridErrorNetworkError


class Offline(XybridError):
    """The device is offline and the operation needs the network."""

    __slots__ = ()
    _payload = _bolt.XybridErrorOffline


class IoError(XybridError):
    """A filesystem operation failed."""

    __slots__ = ()
    _payload = _bolt.XybridErrorIoError


class CacheError(XybridError):
    """The model cache could not be read or written."""

    __slots__ = ()
    _payload = _bolt.XybridErrorCacheError


class PipelineError(XybridError):
    """A pipeline stage failed."""

    __slots__ = ()
    _payload = _bolt.XybridErrorPipelineError


class CircuitOpen(XybridError):
    """The cloud circuit breaker is open."""

    __slots__ = ()
    _payload = _bolt.XybridErrorCircuitOpen


class RateLimited(XybridError):
    """The gateway rate-limited this request."""

    __slots__ = ()
    _payload = _bolt.XybridErrorRateLimited

    def __str__(self) -> str:
        return f"rate limited; retry after {self.retry_after_secs}s"


class Timeout(XybridError):
    """The operation timed out."""

    __slots__ = ()
    _payload = _bolt.XybridErrorTimeout

    def __str__(self) -> str:
        return f"timed out after {self.timeout_ms}ms"


class MissingArtifact(XybridError):
    """A required model artifact is absent from the bundle."""

    __slots__ = ()
    _payload = _bolt.XybridErrorMissingArtifact


class UnsupportedModelCapability(XybridError):
    """The model does not support the requested capability."""

    __slots__ = ()
    _payload = _bolt.XybridErrorUnsupportedModelCapability


class UnsupportedBackendCapability(XybridError):
    """The backend does not support the requested capability."""

    __slots__ = ()
    _payload = _bolt.XybridErrorUnsupportedBackendCapability


class InvalidImage(XybridError):
    """The image payload could not be decoded."""

    __slots__ = ()
    _payload = _bolt.XybridErrorInvalidImage


def payload_variants() -> list[type[Any]]:
    """Return every generated ``XybridError`` payload variant."""

    return [
        value
        for value in vars(_bolt).values()
        if isinstance(value, type) and issubclass(value, _bolt.XybridError) and value is not _bolt.XybridError
    ]


def _install() -> None:
    """Register the typed exceptions with the generated error dispatcher."""

    typed = {
        exception_type._payload: exception_type
        for exception_type in globals().values()
        if isinstance(exception_type, type)
        and issubclass(exception_type, XybridError)
        and exception_type._payload is not None
    }
    for payload_type in payload_variants():
        # A variant added by a later boltffi release falls back to the base
        # class, so it still raises a xybrid error with the payload attached.
        exception_type = typed.get(payload_type, XybridError)
        setattr(_bolt, f"{payload_type.__name__}Exception", exception_type)
