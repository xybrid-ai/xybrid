from __future__ import annotations

import pytest

import xybrid


def test_init_is_idempotent_when_native_library_resolves() -> None:
    try:
        import xybrid._bolt as bolt

        bolt._load_library()
    except ImportError as exc:
        pytest.skip(str(exc))

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


def test_result_conveniences_on_synthetic_result() -> None:
    text_result = xybrid.XybridResult(
        envelope=xybrid.XybridEnvelope(
            kind=xybrid.XybridEnvelopeKind.text("answer"),
            metadata=[xybrid.XybridMetadataEntry(key="reasoning_content", value="thinking")],
        ),
        output_type=xybrid.XybridOutputType.TEXT,
        model_id="model",
        latency_ms=1234,
        execution_target=xybrid.XybridExecutionTarget.LOCAL,
        metrics=xybrid.XybridInferenceMetrics(total_ms=1234),
    )
    audio_result = xybrid.XybridResult(
        envelope=xybrid.XybridEnvelope.audio(b"audio"),
        output_type=xybrid.XybridOutputType.AUDIO,
        model_id="model",
        latency_ms=10,
        execution_target=xybrid.XybridExecutionTarget.LOCAL,
        metrics=xybrid.XybridInferenceMetrics(total_ms=10),
    )
    embedding_result = xybrid.XybridResult(
        envelope=xybrid.XybridEnvelope.embedding([1.0, 2.0]),
        output_type=xybrid.XybridOutputType.EMBEDDING,
        model_id="model",
        latency_ms=10,
        execution_target=xybrid.XybridExecutionTarget.LOCAL,
        metrics=xybrid.XybridInferenceMetrics(total_ms=10),
    )
    failed_result = xybrid.XybridResult(
        envelope=xybrid.XybridEnvelope.text(""),
        output_type=xybrid.XybridOutputType.UNKNOWN,
        model_id="model",
        latency_ms=0,
        execution_target=xybrid.XybridExecutionTarget.LOCAL,
        metrics=xybrid.XybridInferenceMetrics(total_ms=0),
    )

    assert text_result.text == "answer"
    assert text_result.reasoning_content == "thinking"
    assert text_result.latency_seconds == pytest.approx(1.234)
    assert not text_result.is_failure
    assert text_result.success
    assert audio_result.audio_bytes == b"audio"
    assert embedding_result.embedding == [1.0, 2.0]
    assert failed_result.is_failure

    # Non-matching kinds yield None, never the XybridEnvelopeKind factory
    # methods (regression: getattr on the kind used to resolve the inherited
    # `text` staticmethod for non-text kinds).
    assert audio_result.text is None
    assert text_result.audio_bytes is None
    assert text_result.embedding is None

    # Payload presence follows the envelope kind, not output_type, matching
    # the Swift/Kotlin accessors.
    mislabeled_audio = xybrid.XybridResult(
        envelope=xybrid.XybridEnvelope.audio(b"pcm"),
        output_type=xybrid.XybridOutputType.UNKNOWN,
        model_id="model",
        latency_ms=1,
        execution_target=xybrid.XybridExecutionTarget.LOCAL,
        metrics=xybrid.XybridInferenceMetrics(total_ms=1),
    )
    assert mislabeled_audio.audio_bytes == b"pcm"

    # The conveniences are class members now, visible to type checkers.
    assert isinstance(xybrid.XybridResult.text, property)
    assert isinstance(xybrid.XybridVoiceInfo.is_female, property)


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
    )


def test_public_import_surface() -> None:
    from xybrid import XybridEnvelope, XybridError, XybridModel, init

    assert XybridModel is xybrid.XybridModel
    assert XybridEnvelope is xybrid.XybridEnvelope
    assert XybridError is xybrid.XybridError
    assert init is xybrid.init
