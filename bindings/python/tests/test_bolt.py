"""Behavioural checks against the generated BoltFFI bindings.

Before boltffi 0.29 this file tested a hand-ported ctypes wire layer — its
loader precedence, `_WireReader`/`_WireWriter` round-trips, and the tagged
result decoding. boltffi generates all of that now
(`tools/scripts/gen_python_bolt.py`), so those internals are no longer ours to
test; asserting on them would just pin the generator's private shape. What is
still worth guarding is that the package loads its native bridge and that the
surface the SDK depends on is actually exported.
"""

from __future__ import annotations

import pytest

try:
    import xybrid._bolt as bolt
except ImportError as exc:  # pragma: no cover - native artifacts not built
    pytest.skip(str(exc), allow_module_level=True)


def test_native_bridge_loads_and_reports_a_version() -> None:
    """Proves the compiled bridge resolved against the staged cdylib.

    A mismatch between the two (for example a cdylib built without boltffi's
    export shims) fails at import with an unresolved-symbol ImportError.
    """

    assert bolt.version()


@pytest.mark.parametrize(
    "name",
    [
        # Free functions the `xybrid` wrapper calls directly.
        "set_api_key",
        "has_api_key",
        "set_binding",
        "configure_runtime",
        "set_platform_url",
        "set_speculative_cloud",
        "is_speculative_cloud_enabled",
        "will_speculate_for_model",
        # Handle + record types re-exported as the public API.
        "XybridModel",
        "XybridEnvelope",
        "XybridGenerationConfig",
        "XybridResult",
        "XybridVoiceInfo",
        "XybridDownloadStatus",
        "XybridExecutionTarget",
    ],
)
def test_surface_is_exported(name: str) -> None:
    assert name in bolt.__all__, f"{name} missing from generated __all__"
    assert hasattr(bolt, name)


def test_speculative_cloud_toggle_round_trips() -> None:
    """A real call through the bridge, not just an attribute lookup."""

    previous = bolt.is_speculative_cloud_enabled()
    try:
        bolt.set_speculative_cloud(True)
        assert bolt.is_speculative_cloud_enabled() is True
        bolt.set_speculative_cloud(False)
        assert bolt.is_speculative_cloud_enabled() is False
    finally:
        bolt.set_speculative_cloud(previous)


def test_will_speculate_is_false_without_an_api_key() -> None:
    """Speculation needs a resolvable key; absent one it must not engage."""

    if bolt.has_api_key():
        pytest.skip("an API key is configured in this environment")
    assert bolt.will_speculate_for_model("lfm2.5-350m") is False
