from __future__ import annotations

import pytest

try:
    import xybrid._bolt as bolt

    bolt._load_library()
except ImportError as exc:
    pytest.skip(str(exc), allow_module_level=True)


def test_loader_resolution_precedence() -> None:
    path = bolt._resolve_library_path()

    # A bundled copy in xybrid/_native wins when present (the state after
    # tools/scripts/build-python-bolt.sh); otherwise the loader falls back
    # to the workspace target/ directory.
    assert path.name == "libxybrid_bolt.dylib"
    bundled = path.parts[-3:-1] == ("xybrid", "_native")
    dev_fallback = path.parts[-3] == "target"
    assert bundled or dev_fallback


def test_primitive_calls_succeed() -> None:
    bolt.set_battery_level(50)
    bolt.clear_battery_level()
    bolt.set_thermal_state(bolt.XybridThermalState.WARM)
    bolt.clear_thermal_state()
    bolt.set_binding("python")


def test_json_schema_to_gbnf_returns_grammar() -> None:
    schema = '{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}'

    grammar = bolt.json_schema_to_gbnf(schema)

    assert grammar.strip()


def test_json_schema_to_gbnf_raises_typed_error_for_garbage() -> None:
    with pytest.raises(bolt.XybridError):
        bolt.json_schema_to_gbnf("{not-json")


def test_from_directory_raises_directory_not_found_for_missing_path() -> None:
    with pytest.raises(bolt.DirectoryNotFound) as raised:
        bolt.XybridModel.from_directory("/nonexistent/path/xyz")
    assert raised.value.path == "/nonexistent/path/xyz"


def test_envelope_and_run_options_wire_roundtrip() -> None:
    envelope = bolt.XybridEnvelope(
        kind=bolt.XybridEnvelopeKind.text("hello"),
        metadata=[bolt.XybridMetadataEntry(key="role", value="user")],
    )
    options = bolt.XybridRunOptions(
        generation_config=bolt.XybridGenerationConfig(
            max_tokens=16,
            temperature=0.5,
            top_p=0.75,
            min_p=None,
            top_k=40,
            repetition_penalty=1.25,
            stop_sequences=["</s>"],
            grammar="root ::= \"ok\"",
        ),
        abort_on=[bolt.XybridAbortSignal.THERMAL_HOT],
        fallback_to_cloud=True,
        max_grace_tokens=2,
        correlation_id="corr-1",
    )

    writer = bolt._WireWriter()
    envelope._encode(writer)
    options._encode(writer)
    reader = bolt._WireReader(writer.finalize())

    assert bolt.XybridEnvelope._decode(reader) == envelope
    assert bolt.XybridRunOptions._decode(reader) == options
