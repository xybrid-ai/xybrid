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
import threading

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


def _stage(stage_id: str, envelope: xybrid.XybridEnvelope, output_type: xybrid.XybridOutputType):
    return xybrid.XybridStageResult(
        stage_id=stage_id,
        envelope=envelope,
        output_type=output_type,
        latency_ms=250,
        execution_target=xybrid.XybridExecutionTarget.LOCAL,
        metrics=_metrics(250),
    )


def test_pipeline_result_conveniences_reach_every_stage() -> None:
    asr = _stage("asr", xybrid.XybridEnvelope.text("what time is it"), xybrid.XybridOutputType.TEXT)
    tts = _stage("tts", xybrid.XybridEnvelope.audio(b"pcm"), xybrid.XybridOutputType.AUDIO)
    result = xybrid.XybridPipelineResult(
        envelope=tts.envelope,
        output_type=xybrid.XybridOutputType.AUDIO,
        latency_ms=500,
        stages=[asr, tts],
    )

    assert result.audio_bytes == b"pcm"
    assert result.text is None
    assert result.latency_seconds == pytest.approx(0.5)
    assert result.stage("asr").text == "what time is it"
    assert result.stage("tts").audio_bytes == b"pcm"
    assert result.stage("missing") is None


def test_pipeline_run_defaults_its_options() -> None:
    parameter = inspect.signature(xybrid.XybridPipeline.run).parameters["options"]

    assert parameter.default is None


def test_tool_call_conveniences_on_result_and_stream_token() -> None:
    call = xybrid.XybridToolCall(id="call_0", name="get_weather", arguments_json='{"city":"Paris"}')

    without = _result(xybrid.XybridEnvelope.text("plain answer"), xybrid.XybridOutputType.TEXT)
    assert not without.has_tool_calls

    with_calls = xybrid.XybridResult(
        envelope=xybrid.XybridEnvelope.text("checking"),
        output_type=xybrid.XybridOutputType.TEXT,
        model_id="model",
        latency_ms=0,
        execution_target=xybrid.XybridExecutionTarget.LOCAL,
        metrics=_metrics(0),
        reasoning_content=None,
        tool_calls=[call],
    )
    assert with_calls.has_tool_calls

    # The terminal stream token is where a streaming loop branches: call
    # blocks are suppressed from the emitted text, so the token text never
    # carries them.
    mid_stream = xybrid.XybridStreamToken(
        token="check",
        token_id=None,
        index=0,
        cumulative_text="check",
        finish_reason=None,
        tool_calls=[],
        raw_text=None,
    )
    terminal = xybrid.XybridStreamToken(
        token="",
        token_id=None,
        index=1,
        cumulative_text="checking",
        finish_reason="tool_calls",
        tool_calls=[call],
        raw_text="checking<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>",
    )

    assert not mid_stream.has_tool_calls
    assert terminal.has_tool_calls
    # raw_text, not cumulative_text, is what the continuation replays.
    assert terminal.raw_text is not None
    assert "tool_call_start" in terminal.raw_text
    assert "tool_call_start" not in terminal.cumulative_text

    assert isinstance(xybrid.XybridResult.has_tool_calls, property)
    assert isinstance(xybrid.XybridStreamToken.has_tool_calls, property)


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


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("run", ("envelope",)),
        ("run_stream", ("envelope",)),
        ("run_with_context", ("envelope", "context")),
        ("run_stream_with_context", ("envelope", "context")),
    ],
)
def test_run_methods_forward_every_required_generated_argument(
    name: str, args: tuple[str, ...]
) -> None:
    """The sugar must supply every argument the generated method requires.

    The generated run methods gained a required `cancel` handle -- BoltFFI
    cannot express an optional handle parameter -- and the sugar kept
    forwarding the old argument list, so every public call raised TypeError
    before reaching native code. Nothing caught it: the rest of this suite
    exercises factories and codecs, never a call through to the wire.

    Calling with an unbacked model reaches the native layer and fails on the
    missing handle. An arity mismatch would fail earlier, with TypeError.
    """

    model = xybrid.XybridModel.__new__(xybrid.XybridModel)
    supplied = {
        "envelope": xybrid.XybridEnvelope.text("hi"),
        "context": xybrid.XybridConversationContext(),
    }

    with pytest.raises(AttributeError, match="_handle"):
        getattr(model, name)(*(supplied[arg] for arg in args))


def test_run_methods_accept_an_explicit_cancellation_token() -> None:
    """`cancel` is keyword-only, so the positional shape is unchanged."""

    parameter = inspect.signature(xybrid.XybridModel.run).parameters["cancel"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_model_supports_explicit_release() -> None:
    assert callable(xybrid.XybridModel.close)
    assert hasattr(xybrid.XybridModel, "__enter__")
    assert hasattr(xybrid.XybridModel, "__exit__")


def test_active_native_borrow_delays_explicit_disposal() -> None:
    started = threading.Event()
    released = threading.Event()

    class Handle:
        def __init__(self):
            self.alive = True

        def __del__(self):
            if self.alive:
                self.alive = False
                released.set()

    xybrid._sugar._guard_handle_finalizer(Handle)
    handle = Handle()

    def dispose():
        started.set()
        handle.__del__()

    with xybrid._sugar._handle_lifetime_lock(handle):
        disposer = threading.Thread(target=dispose)
        disposer.start()
        assert started.wait(timeout=1)
        assert not released.wait(timeout=0.02)
        assert handle.alive
    disposer.join(timeout=1)
    assert released.is_set()
    assert id(handle) not in xybrid._sugar._HANDLE_LOCKS


def test_model_property_waiter_keeps_cancellation_live(monkeypatch) -> None:
    model = xybrid.XybridModel.__new__(xybrid.XybridModel)
    model._handle = None
    token = xybrid.XybridCancellationToken()
    getter_started = threading.Event()
    native_getter_entered = threading.Event()
    cancelled = threading.Event()
    values = []

    def native_getter(_handle):
        native_getter_entered.set()
        return True

    monkeypatch.setattr(bolt._native, "_boltffi_xybrid_model_is_loaded", native_getter)

    def read_property():
        getter_started.set()
        values.append(model.is_loaded)

    def cancel():
        token.cancel()
        cancelled.set()

    with xybrid._sugar._handle_lifetime_lock(model):
        reader = threading.Thread(target=read_property)
        reader.start()
        assert getter_started.wait(timeout=1)
        stopper = threading.Thread(target=cancel)
        stopper.start()
        assert cancelled.wait(timeout=1)
        assert token.is_cancelled()
        assert not native_getter_entered.is_set()
    reader.join(timeout=1)
    stopper.join(timeout=1)
    assert values == [True]


@pytest.mark.parametrize(
    "subscription",
    [bolt.XybridDownloadProgressSubscription, bolt.XybridModelDownloadProgressSubscription],
)
def test_download_progress_subscriptions_are_iterable(subscription: type) -> None:
    """The sugar turns boltffi's wait/pop_batch pair into a plain iterable.

    Without it every caller writes the same drain loop, and a generator rename
    would silently drop the documented `for status in download.progress()`.
    """

    assert hasattr(subscription, "__iter__")
    assert callable(subscription.__iter__)


def test_download_progress_iteration_drains_before_stopping() -> None:
    """The terminal status is pushed just before the stream closes.

    A loop that stopped on `unsubscribed` without popping first would drop the
    `Ready` frame -- exactly the event a progress bar is waiting for.
    """

    emitted = [
        xybrid.XybridDownloadStatus(
            state=xybrid.XybridDownloadState.READY,
            progress=1.0,
            downloaded_bytes=2048,
            total_bytes=2048,
        )
    ]

    class ClosedWithPendingItems:
        """Reports `unsubscribed` while a batch is still buffered."""

        __iter__ = bolt.XybridDownloadProgressSubscription.__iter__

        def wait(self, timeout_milliseconds: int) -> int:
            return -1

        def pop_batch(self, max_count: int = 16) -> list:
            return [emitted.pop(0)] if emitted else []

    assert [status.state for status in ClosedWithPendingItems()] == [
        xybrid.XybridDownloadState.READY
    ]


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
        (
            xybrid.Cancelled,
            bolt.XybridErrorCancelled(message="download cancelled by caller"),
            "download cancelled by caller",
        ),
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


def test_decision_json_factory_preserves_order_and_large_numbers():
    raw = '{"state":{"z":9223372036854775807,"a":9007199254740993,"f":1.0,"zero":-0.0},"questions":{"q":{"type":"choice","instructions":"choose","criteria":{"a":null}}}}'
    envelope = xybrid.XybridEnvelope.decision_request_json(raw)
    assert isinstance(envelope.kind, bolt.XybridEnvelopeKindDecisionRequest)
    assert envelope.kind.request.state.json_text == '{"z":9223372036854775807,"a":9007199254740993,"f":1.0,"zero":-0.0}'
    assert envelope.kind.request.questions[0].kind.options[0].key == 'a'
    typed = xybrid.XybridEnvelope.decision_request(envelope.kind.request)
    assert typed.kind.request == envelope.kind.request


def test_decision_result_accessor_preserves_double_precision_and_infinity():
    probability = 0.5000000018693014
    decisions = bolt.XybridDecisions(
        answers=[bolt.XybridDecisionAnswer(question_id='q', distribution=[
            bolt.XybridDecisionScore(key='0', raw_score=0.0, probability=1.0-probability, origin=bolt.XybridDecisionOrigin.CALLER),
            bolt.XybridDecisionScore(key='1', raw_score=1e-8, probability=probability, origin=bolt.XybridDecisionOrigin.CALLER),
            bolt.XybridDecisionScore(key='2', raw_score=float('-inf'), probability=0.0, origin=bolt.XybridDecisionOrigin.CALLER),
        ], value=bolt.XybridDecisionValueScore(value=1.0, statistic=bolt.XybridScoreStatistic.MEDIAN), confidence=probability)],
        passes=1, input_tokens=512,
    )
    envelope = bolt.XybridEnvelope(kind=bolt.XybridEnvelopeKindDecisions(decisions=decisions), metadata=[bolt.XybridMetadataEntry(key='tokens_in', value='512'), bolt.XybridMetadataEntry(key='tokens_out', value='0')])
    result = _result(envelope, bolt.XybridOutputType.DECISIONS)
    assert result.decisions is decisions
    assert result.decisions.answers[0].distribution[1].probability == probability
    assert result.decisions.answers[0].distribution[2].raw_score == float('-inf')
    assert result.tokens_in == 512
    assert result.tokens_out == 0
    assert result.text is None


def test_decision_json_factory_refuses_unsigned_overflow_without_content():
    raw = '{"state":{"private":18446744073709551615},"questions":{"q":{"type":"choice","instructions":"private instructions","criteria":{"a":null}}}}'
    with pytest.raises(xybrid.InferenceError) as raised:
        xybrid.XybridEnvelope.decision_request_json(raw)
    assert str(raised.value) == "Decision request refused: decision number_out_of_range"


@pytest.mark.parametrize("exceptional", [False, True])
def test_manual_observation_context_manager_releases_once(exceptional):
    observation = bolt.XybridWorkflowObservation("python-test", None)
    if exceptional:
        with pytest.raises(RuntimeError):
            with observation:
                raise RuntimeError("private application error")
    else:
        with observation:
            observation.complete()
            observation.cancel()  # first terminal call wins
    assert observation._handle is None
    observation.close()  # explicit close and later GC are idempotent


def _manual_observation(kind, options=None):
    if kind == "workflow":
        return bolt.XybridWorkflowObservation("python-test", options)
    if kind == "tool":
        call = bolt.XybridToolCallInfo("test-tool", None, None, None, None)
        return bolt.XybridToolObservation(call, options)
    if kind == "inference":
        request = bolt.XybridInferenceRequest(
            "chat", "test", "test-model", bolt.XybridExecutionLocation.LOCAL,
            None, None, False, None, None, None, None, None,
        )
        return bolt.XybridInferenceObservation(request, options)
    info = bolt.XybridModelLoadInfo("test-model", None, None, None, None)
    return bolt.XybridModelLoadObservation(info, options)


@pytest.mark.parametrize("kind", ["workflow", "tool", "inference", "load"])
def test_manual_observation_rejects_invalid_parent(kind):
    options = bolt.XybridObservationOptions(
        traceparent="invalid", tracestate=None, conversation_id=None, suppress_trace=False
    )
    with pytest.raises(xybrid.ConfigError, match="invalid traceparent") as raised:
        _manual_observation(kind, options)
    assert isinstance(raised.value, xybrid.XybridError)
    assert isinstance(raised.value.error, bolt.XybridErrorConfigError)


@pytest.mark.parametrize("kind", ["workflow", "tool", "inference", "load"])
def test_manual_observation_native_lifecycle(kind):
    observation = _manual_observation(kind)
    assert not observation.is_finished()
    observation.complete()
    observation.cancel()
    assert observation.is_finished()
    observation.close()
    observation.close()
    assert observation._handle is None


@pytest.mark.parametrize("kind,method", [
    ("tool", "record_result"),
    ("inference", "record_system_instructions"),
    ("inference", "record_input_messages"),
    ("inference", "record_output_messages"),
    ("inference", "record_tool_definitions"),
])
def test_manual_observation_rejects_invalid_content(kind, method):
    with _manual_observation(kind) as observation:
        with pytest.raises(xybrid.ConfigError):
            getattr(observation, method)("invalid json")


def test_manual_constructor_error_adaptation_preserves_python_errors():
    constructor = bolt.XybridWorkflowObservation.__init__
    xybrid._errors._install()
    assert bolt.XybridWorkflowObservation.__init__ is constructor
    with pytest.raises(TypeError):
        bolt.XybridWorkflowObservation()
