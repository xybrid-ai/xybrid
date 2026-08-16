"""Behavioural checks against the generated BoltFFI bindings.

Before boltffi 0.29 this file tested a hand-ported ctypes wire layer — its
loader precedence, `_WireReader`/`_WireWriter` round-trips, and the tagged
result decoding. boltffi generates those internals now, except for the
append-only result fallback owned by `tools/scripts/gen_python_bolt.py`.
These checks cover that compatibility transform, native loading, and exports.
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


def test_result_decoder_accepts_tool_calling_wire_without_reasoning_tail() -> None:
    tool_calling = bolt.XybridResult._boltffi_from_wire(_result_wire())
    assert [call.id for call in tool_calling.tool_calls] == ["call-1"]
    assert tool_calling.reasoning_content == "metadata reasoning"

    current = bolt.XybridResult._boltffi_from_wire(
        _result_wire(typed_reasoning="typed reasoning")
    )
    assert [call.id for call in current.tool_calls] == ["call-1"]
    assert current.reasoning_content == "typed reasoning"


def _result_wire(*, typed_reasoning: str | None = None) -> bytes:
    envelope = bolt.XybridEnvelope(
        kind=bolt.XybridEnvelopeKindText(text="answer"),
        metadata=[
            bolt.XybridMetadataEntry(
                key="reasoning_content", value="metadata reasoning"
            )
        ],
    )
    metrics = bolt.XybridInferenceMetrics(
        total_ms=7,
        ttft_ms=None,
        tokens_per_second=None,
        prefill_tps=None,
        decode_tps=None,
        tokens_out=None,
        stage_latencies_ms=[],
    )
    fields = [
        envelope._boltffi_wire(),
        bolt._boltffi_wire_i32(bolt.XybridOutputType.TEXT.value),
        bolt._boltffi_wire_string("model"),
        bolt._boltffi_wire_u32(9),
        bolt._boltffi_wire_i32(bolt.XybridExecutionTarget.LOCAL.value),
        metrics._boltffi_wire(),
    ]
    tool_calls = [
        bolt.XybridToolCall(id="call-1", name="lookup", arguments_json="{}")
    ]
    fields.append(
        bolt._boltffi_wire_sequence(
            tool_calls, len(tool_calls), lambda call: call._boltffi_wire()
        )
    )
    if typed_reasoning is not None:
        fields.append(
            bolt._boltffi_wire_optional(typed_reasoning, bolt._boltffi_wire_string)
        )
    return b"".join(fields)
