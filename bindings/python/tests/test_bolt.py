from __future__ import annotations

import threading

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


def test_parse_last_error_degrades_gracefully() -> None:
    # Non-numeric payload for an integer-carrying variant must stay inside
    # the typed error surface instead of leaking ValueError.
    error = bolt._parse_last_error("RateLimited { retry_after_secs: n/a }")
    assert isinstance(error, bolt.LoadError)
    assert isinstance(bolt._parse_last_error("Timeout { timeout_ms: 42 }"), bolt.Timeout)


def test_read_string_is_lossy_like_the_swift_reference() -> None:
    # 2-byte string with an invalid UTF-8 sequence decodes with replacement
    # characters (Swift String(decoding:) semantics), never raises.
    value = bolt._WireReader(bytes([2, 0, 0, 0, 0xFF, 0xFE])).read_string()
    assert value == "��"


def test_f32_array_roundtrip_bulk_codec() -> None:
    values = [float(i) / 7.0 for i in range(4096)]
    writer = bolt._WireWriter()
    writer.write_f32_array(values)
    decoded = bolt._WireReader(writer.finalize()).read_f32_array()
    assert len(decoded) == 4096
    assert decoded[1] == pytest.approx(values[1])


def test_battery_level_is_clamped() -> None:
    # u8 parameter: out-of-range input clamps (Kotlin coerceIn semantics)
    # instead of wrapping modulo 256.
    bolt.set_battery_level(300)
    bolt.set_battery_level(-5)
    bolt.clear_battery_level()


def test_is_loaded_reports_false_after_close() -> None:
    model = bolt.XybridModel.__new__(bolt.XybridModel)
    model._handle = None
    model._handle_lock = threading.Lock()
    assert model.is_loaded is False
    model.close()


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


def test_result_wire_roundtrip_carries_execution_target() -> None:
    """Provenance rides the wire: cloud fallback keeps the model id identical on
    both legs, so a decoded result must still say which one answered."""

    result = bolt.XybridResult(
        envelope=bolt.XybridEnvelope(
            kind=bolt.XybridEnvelopeKind.text("answer"),
            metadata=[],
        ),
        output_type=bolt.XybridOutputType.TEXT,
        model_id="lfm2.5-350m",
        latency_ms=42,
        execution_target=bolt.XybridExecutionTarget.CLOUD,
        metrics=bolt.XybridInferenceMetrics(total_ms=42),
    )

    writer = bolt._WireWriter()
    result._encode(writer)
    decoded = bolt.XybridResult._decode(bolt._WireReader(writer.finalize()))

    assert decoded == result
    assert decoded.execution_target is bolt.XybridExecutionTarget.CLOUD


def test_download_status_wire_roundtrip() -> None:
    status = bolt.XybridDownloadStatus(
        state=bolt.XybridDownloadState.DOWNLOADING,
        progress=0.5,
    )

    writer = bolt._WireWriter()
    status._encode(writer)
    decoded = bolt.XybridDownloadStatus._decode(bolt._WireReader(writer.finalize()))

    assert decoded == status


def _fake_model(handle: int) -> bolt.XybridModel:
    """A model bypassing the native constructor.

    Sets `_handle_lock` as well, so the finalizer's take-then-null free is a
    real no-op instead of raising inside `__del__`; the handle is a fake, so it
    must never reach `boltffi_xybrid_model_free`.
    """

    model = bolt.XybridModel.__new__(bolt.XybridModel)
    object.__setattr__(model, "_handle", handle)
    object.__setattr__(model, "_handle_lock", threading.Lock())
    return model


def _status_buffer(state: int, progress: float) -> bytes:
    """Exactly what the native side returns: a bare XybridDownloadStatus.

    Both `download_status` and `await_download` are infallible in Rust, so the
    buffer carries no leading result discriminant.
    """

    writer = bolt._WireWriter()
    writer.write_i32(state)
    writer.write_f32(progress)
    return writer.finalize()


@pytest.mark.parametrize(
    "state",
    [
        bolt.XybridDownloadState.DOWNLOADING,
        bolt.XybridDownloadState.READY,
        bolt.XybridDownloadState.FAILED,
    ],
)
def test_download_status_decodes_native_buffer(monkeypatch, state) -> None:
    """Guards the native call path, not just the codec.

    Routing these bare buffers through `_decode_result` consumed the state's
    first byte as a result discriminant and shifted every field after it, so
    each state failed differently (bogus enum value / unknown error tag).
    """

    captured: dict[str, object] = {}

    class _FakeLib:
        def boltffi_xybrid_model_download_status(self, handle):
            captured["handle"] = handle
            return _status_buffer(int(state), 0.25)

        def boltffi_xybrid_model_await_download(self, handle, timeout_ms):
            captured["timeout_ms"] = timeout_ms
            return _status_buffer(int(state), 0.75)

    monkeypatch.setattr(bolt, "_load_library", lambda: _FakeLib())
    monkeypatch.setattr(bolt, "_copy_and_free_buf", lambda buf: buf)

    model = _fake_model(1234)

    status = model.download_status()
    assert status.state is state
    assert status.progress == pytest.approx(0.25)
    assert captured["handle"] == 1234

    awaited = model.await_download(500)
    assert awaited.state is state
    assert awaited.progress == pytest.approx(0.75)
    assert captured["timeout_ms"] == 500

    object.__setattr__(model, "_handle", None)


def test_await_download_clamps_negative_timeout(monkeypatch) -> None:
    """A negative timeout would wrap through `c_uint64` to ~u64::MAX."""

    seen: dict[str, int] = {}

    class _FakeLib:
        def boltffi_xybrid_model_await_download(self, handle, timeout_ms):
            seen["timeout_ms"] = timeout_ms
            return _status_buffer(int(bolt.XybridDownloadState.READY), 1.0)

    monkeypatch.setattr(bolt, "_load_library", lambda: _FakeLib())
    monkeypatch.setattr(bolt, "_copy_and_free_buf", lambda buf: buf)

    model = _fake_model(7)

    assert model.await_download(-1).state is bolt.XybridDownloadState.READY
    assert seen["timeout_ms"] == 0

    object.__setattr__(model, "_handle", None)
