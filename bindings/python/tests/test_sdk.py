"""Behavioural checks for the hand-written SDK layer over the generated bindings.

The generated package (`xybrid/_bolt/`) is boltffi's output and is byte-compared
against a fresh generation in CI, so everything Pythonic about the SDK —
envelope factories, result accessors, model properties, typed exceptions — is
attached by `xybrid/_sugar.py` and `xybrid/_errors.py` at import. These tests
guard that surface: if a generator change renames or drops something they patch,
the failure lands here rather than in a user's code.
"""

from __future__ import annotations

import inspect

import pytest

try:
    import xybrid
    import xybrid._bolt as bolt
except ImportError as exc:  # pragma: no cover - native artifacts not built
    pytest.skip(str(exc), allow_module_level=True)


def _metrics(total_ms: int) -> xybrid.XybridInferenceMetrics:
    return xybrid.XybridInferenceMetrics(
        total_ms=total_ms,
        ttft_ms=None,
        tokens_per_second=None,
        prefill_tps=None,
        decode_tps=None,
        tokens_out=None,
        stage_latencies_ms=[],
    )


def _result(envelope: xybrid.XybridEnvelope, output_type: xybrid.XybridOutputType, latency_ms: int = 0):
    return xybrid.XybridResult(
        envelope=envelope,
        output_type=output_type,
        model_id="model",
        latency_ms=latency_ms,
        execution_target=xybrid.XybridExecutionTarget.LOCAL,
        metrics=_metrics(latency_ms),
        reasoning_content=next(
            (entry.value for entry in envelope.metadata if entry.key == "reasoning_content"),
            None,
        ),
        tool_calls=[],
    )


def test_init_is_idempotent() -> None:
    xybrid.init()
    xybrid.init(api_key="ignored-after-first-init")

    assert xybrid.is_initialized()


def test_envelope_factory_helpers_produce_expected_metadata() -> None:
    image = xybrid.XybridEnvelope.image(b"img", format="jpg")

    text = xybrid.XybridEnvelope.text("hello", voice="af_heart", speed=1.2)
    audio = xybrid.XybridEnvelope.audio(b"pcm", sample_rate=24000, channels=2)
    embedding = xybrid.XybridEnvelope.embedding([0.25, 0.5])
    message = xybrid.XybridEnvelope.user_message("describe", images=[image])

    assert text.kind.text == "hello"
    assert [(entry.key, entry.value) for entry in text.metadata] == [
        ("voice_id", "af_heart"),
        ("speed", "1.2"),
    ]
    assert audio.kind.bytes == b"pcm"
    assert [(entry.key, entry.value) for entry in audio.metadata] == [
        ("sample_rate", "24000"),
        ("channels", "2"),
    ]
    assert embedding.kind.values == [0.25, 0.5]
    assert image.kind.format == "jpeg"
    assert message.metadata == [xybrid.XybridMetadataEntry(key="xybrid.role", value="user")]


def test_image_factory_rejects_unsupported_format() -> None:
    with pytest.raises(xybrid.ConfigError):
        xybrid.XybridEnvelope.image(b"img", format="gif")


def test_user_message_rejects_non_image_parts() -> None:
    with pytest.raises(xybrid.ConfigError):
        xybrid.XybridEnvelope.user_message("describe", images=[xybrid.XybridEnvelope.text("not an image")])


def test_result_conveniences_on_synthetic_result() -> None:
    text_result = _result(
        xybrid.XybridEnvelope(
            kind=xybrid.XybridEnvelopeKind.text("answer"),
            metadata=[xybrid.XybridMetadataEntry(key="reasoning_content", value="thinking")],
        ),
        xybrid.XybridOutputType.TEXT,
        latency_ms=1234,
    )
    audio_result = _result(xybrid.XybridEnvelope.audio(b"audio"), xybrid.XybridOutputType.AUDIO)
    embedding_result = _result(xybrid.XybridEnvelope.embedding([1.0, 2.0]), xybrid.XybridOutputType.EMBEDDING)
    failed_result = _result(xybrid.XybridEnvelope.text(""), xybrid.XybridOutputType.UNKNOWN)

    assert text_result.text == "answer"
    assert text_result.reasoning_content == "thinking"
    assert text_result.latency_seconds == pytest.approx(1.234)
    assert not text_result.is_failure
    assert text_result.success
    assert audio_result.audio_bytes == b"audio"
    assert embedding_result.embedding == [1.0, 2.0]
    assert failed_result.is_failure

    # Non-matching kinds yield None, never the XybridEnvelopeKind factory
    # methods (the factories live on the union base, so `kind.text` on an audio
    # kind resolves to the inherited staticmethod).
    assert audio_result.text is None
    assert text_result.audio_bytes is None
    assert text_result.embedding is None

    # Payload presence follows the envelope kind, not output_type, matching
    # the Swift/Kotlin accessors.
    mislabeled_audio = _result(xybrid.XybridEnvelope.audio(b"pcm"), xybrid.XybridOutputType.UNKNOWN)
    assert mislabeled_audio.audio_bytes == b"pcm"

    # The conveniences are class members, visible to type checkers.
    assert isinstance(xybrid.XybridResult.text, property)
    assert isinstance(xybrid.XybridVoiceInfo.is_female, property)


def test_result_reasoning_field_defaults_to_none() -> None:
    result = xybrid.XybridResult(
        envelope=xybrid.XybridEnvelope.text("answer"),
        output_type=xybrid.XybridOutputType.TEXT,
        model_id="model",
        latency_ms=1,
        execution_target=xybrid.XybridExecutionTarget.LOCAL,
        metrics=_metrics(1),
        tool_calls=[],
    )

    assert result.reasoning_content is None


def test_voice_gender_helpers() -> None:
    female = xybrid.XybridVoiceInfo(id="af_heart", name="Heart", gender="female", language="en", style=None)
    male = xybrid.XybridVoiceInfo(id="am_adam", name="Adam", gender="male", language="en", style=None)
    unknown = xybrid.XybridVoiceInfo(id="x", name="X", gender=None, language=None, style=None)

    assert female.is_female and not female.is_male
    assert male.is_male and not male.is_female
    assert not unknown.is_male and not unknown.is_female


@pytest.mark.parametrize(
    "name",
    [
        "model_id",
        "version",
        "output_type",
        "is_loaded",
        "is_cloud_serving",
        "supports_streaming",
        "supports_token_streaming",
        "supports_tool_calling",
        "is_llm",
        "has_voices",
    ],
)
def test_model_accessors_are_properties(name: str) -> None:
    """The SDK documents these as attributes; boltffi generates them as methods."""

    assert isinstance(getattr(xybrid.XybridModel, name), property)


@pytest.mark.parametrize(
    "name",
    ["run", "run_stream", "run_with_context", "run_stream_with_context"],
)
def test_run_methods_default_their_options(name: str) -> None:
    parameter = inspect.signature(getattr(xybrid.XybridModel, name)).parameters["options"]

    assert parameter.default is None


def test_model_supports_explicit_release() -> None:
    assert callable(xybrid.XybridModel.close)
    assert hasattr(xybrid.XybridModel, "__enter__")
    assert hasattr(xybrid.XybridModel, "__exit__")


def test_generation_configs_presets_match_kotlin_values() -> None:
    greedy = xybrid.GenerationConfigs.greedy()
    creative = xybrid.GenerationConfigs.creative()

    # Preset values are pinned to bindings/kotlin/src/main/kotlin/ai/xybrid/Xybrid.kt.
    assert greedy == xybrid.XybridGenerationConfig(
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
    assert creative == xybrid.XybridGenerationConfig(
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


def test_public_import_surface() -> None:
    from xybrid import XybridEnvelope, XybridError, XybridModel, init

    assert XybridModel is xybrid.XybridModel
    assert XybridEnvelope is xybrid.XybridEnvelope
    assert XybridError is xybrid.XybridError
    assert init is xybrid.init


def test_xybrid_error_is_catchable_and_wraps_the_generated_payload() -> None:
    """`XybridError` is the exception; the generated union stays on `.error`."""

    assert issubclass(xybrid.XybridError, Exception)
    assert issubclass(xybrid.ModelNotFound, xybrid.XybridError)

    error = xybrid.ModelNotFound("missing-model")

    assert isinstance(error.error, bolt.XybridErrorModelNotFound)
    assert error.id == "missing-model"
    assert str(error) == "model not found: missing-model"


@pytest.mark.parametrize(
    ("exception_type", "payload", "message"),
    [
        (xybrid.ConfigError, bolt.XybridErrorConfigError(message="bad config"), "bad config"),
        (xybrid.NotLoaded, bolt.XybridErrorNotLoaded(), "model is not loaded"),
        (xybrid.StreamingNotSupported, bolt.XybridErrorStreamingNotSupported(), "streaming not supported"),
        (xybrid.RateLimited, bolt.XybridErrorRateLimited(retry_after_secs=30), "rate limited; retry after 30s"),
        (xybrid.Timeout, bolt.XybridErrorTimeout(timeout_ms=500), "timed out after 500ms"),
        (xybrid.DirectoryNotFound, bolt.XybridErrorDirectoryNotFound(path="/tmp/x"), "directory not found: /tmp/x"),
        (xybrid.AbortedForCloudFallback, bolt.XybridErrorAbortedForCloudFallback(reason="thermal"), "thermal"),
    ],
)
def test_native_errors_raise_the_typed_exception(exception_type: type, payload: object, message: str) -> None:
    """The generated dispatcher resolves our classes, so `except` clauses work."""

    raised = bolt._boltffi_error_exception(payload)

    assert type(raised) is exception_type
    assert isinstance(raised, xybrid.XybridError)
    assert str(raised) == message
    assert raised.error is payload


def test_every_generated_error_variant_maps_to_a_typed_exception() -> None:
    """Tripwire: a variant added by a boltffi bump must gain a class here."""

    unmapped = [
        variant.__name__
        for variant in xybrid._errors.payload_variants()
        if type(bolt._boltffi_error_exception(object.__new__(variant))) is xybrid.XybridError
    ]

    assert unmapped == [], f"add typed exceptions for {unmapped} in xybrid/_errors.py"
