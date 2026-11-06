"""Decision smoke checks through the packaged Python extension and local model.

Set XYBRID_DECISION_MODEL_DIR to a staged Liquid text/v1 model directory.
XYBRID_REQUIRE_MODELS=1 makes missing model/native artifacts fail, not skip.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import pytest


def test_local_decisions_native_interface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    required = os.environ.get("XYBRID_REQUIRE_MODELS") == "1"
    path = os.environ.get("XYBRID_DECISION_MODEL_DIR")
    if not path or not (Path(path) / "model_metadata.json").is_file():
        if required:
            pytest.fail("XYBRID_DECISION_MODEL_DIR must contain a staged decision model")
        pytest.skip("set XYBRID_DECISION_MODEL_DIR to exercise the native decision model")
    try:
        import xybrid
    except ImportError as error:
        if required:
            pytest.fail(f"packaged Python native interface is unavailable: {error}")
        pytest.skip("build the packaged Python native interface first")

    xybrid.init_sdk_cache_dir(str(tmp_path / "cache"))
    xybrid.init()
    raw = (
        '{"state":{"z":9223372036854775807,"a":9007199254740993,'
        '"zero":-0.0,"nul":"\\u0000"},"questions":{'
        '"pick":{"type":"choice","instructions":"Choose a.",'
        '"criteria":{"a":"first","b":"second"}}}}'
    )
    request = xybrid.XybridEnvelope.decision_request_json(raw)
    assert request.kind.request.state.json_text == (
        '{"z":9223372036854775807,"a":9007199254740993,'
        '"zero":-0.0,"nul":"\\u0000"}'
    )
    with xybrid.XybridModel.from_directory(path) as model:
        token = xybrid.XybridCancellationToken()
        token.cancel()
        with pytest.raises(xybrid.Cancelled):
            model.run(request, cancel=token)

        active_token = xybrid.XybridCancellationToken()
        cancel_sent = threading.Event()

        def cancel_in_flight() -> None:
            active_token.cancel()
            cancel_sent.set()

        timer = threading.Timer(0.02, cancel_in_flight)
        timer.start()
        try:
            with pytest.raises(xybrid.Cancelled):
                model.run(request, cancel=active_token)
        finally:
            timer.join(timeout=5)
        assert cancel_sent.is_set(), "the native run must release the GIL for cancellation"

        result = model.run(request)
        assert result.output_type == xybrid.XybridOutputType.DECISIONS
        assert result.text is None
        assert result.tokens_out == 0
        assert result.tokens_in > 0
        assert result.decisions.passes == 1
        assert result.decisions.input_tokens == result.tokens_in
        answer = result.decisions.answers[0]
        assert answer.question_id == "pick"
        assert [entry.key for entry in answer.distribution] == ["a", "b"]
        assert all(math.isfinite(entry.probability) for entry in answer.distribution)
        assert sum(entry.probability for entry in answer.distribution) == pytest.approx(1.0)
        assert answer.value.key in ("a", "b")
        assert answer.confidence == max(entry.probability for entry in answer.distribution)

        with pytest.raises(xybrid.UnsupportedModelCapability):
            model.run_stream(request)
        # A refused surface must not poison the loaded model or its next context.
        recovered = model.run(request)
        assert recovered.decisions.passes == 1
        assert recovered.decisions.answers[0].question_id == "pick"
        assert recovered.tokens_out == 0

        # Explicit disposal on another thread must wait for the borrowed native
        # model handle while cancellation remains callable without that lock.
        closing_token = xybrid.XybridCancellationToken()
        closed = threading.Event()
        entered = threading.Event()
        original_native_run = xybrid._bolt._native._boltffi_xybrid_model_run

        def announce_native_run(*args):
            # The public run wrapper has acquired its lifetime locks before
            # entering the generated bridge. Gate the disposal race on that
            # event instead of relying on scheduler timing before run starts.
            entered.set()
            return original_native_run(*args)

        monkeypatch.setattr(
            xybrid._bolt._native, "_boltffi_xybrid_model_run", announce_native_run
        )

        def close_during_run() -> None:
            assert entered.wait(timeout=5)
            model.close()
            closed.set()

        def cancel_closing_run() -> None:
            assert entered.wait(timeout=5)
            closing_token.cancel()

        closer = threading.Thread(target=close_during_run)
        canceller = threading.Thread(target=cancel_closing_run)
        closer.start()
        canceller.start()
        try:
            with pytest.raises(xybrid.Cancelled):
                model.run(request, cancel=closing_token)
        finally:
            closer.join(timeout=5)
            canceller.join(timeout=5)
        assert closed.is_set()
        assert model._handle is None
