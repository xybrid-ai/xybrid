"""Decision smoke checks through the packaged Python extension and local model.

Set XYBRID_DECISION_MODEL_DIR to a staged Liquid text/v1 model directory.
XYBRID_REQUIRE_MODELS=1 makes missing model/native artifacts fail, not skip.
"""

from __future__ import annotations

import math
import os
import threading
import time
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
        '"criteria":{"z":"first","a":"second"}},'
        '"yes":{"type":"noul","instructions":"Is there a number?"},'
        '"score":{"type":"score","instructions":"How much data is present?",'
        '"criteria":["little","some","much"]}}}'
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
        entered_native = threading.Event()
        native_run = xybrid._bolt._native._boltffi_xybrid_model_run

        def enter_native(*args):
            entered_native.set()
            return native_run(*args)

        def cancel_in_flight() -> None:
            if not entered_native.wait(timeout=5):
                return
            time.sleep(0.02)
            active_token.cancel()
            cancel_sent.set()

        with monkeypatch.context() as patch:
            patch.setattr(xybrid._bolt._native, "_boltffi_xybrid_model_run", enter_native)
            timer = threading.Thread(target=cancel_in_flight)
            timer.start()
            try:
                with pytest.raises(xybrid.Cancelled):
                    model.run(request, cancel=active_token)
            finally:
                timer.join(timeout=5)
        assert entered_native.is_set()
        assert cancel_sent.is_set(), "the native run must release the GIL for cancellation"

        result = model.run(request)
        assert result.output_type == xybrid.XybridOutputType.DECISIONS
        assert result.text is None
        assert result.tokens_out == 0
        assert result.tokens_in > 0
        assert result.decisions.passes == 3
        assert result.decisions.input_tokens == result.tokens_in
        assert [answer.question_id for answer in result.decisions.answers] == ["pick", "yes", "score"]
        choice, yes_no, score = result.decisions.answers
        for answer, keys in zip(result.decisions.answers, [["z", "a"], ["true", "false"], ["0", "1", "2"]]):
            assert [entry.key for entry in answer.distribution] == keys
            assert all(math.isfinite(entry.probability) and 0 <= entry.probability <= 1 for entry in answer.distribution)
            assert all(not math.isnan(entry.raw_score) and entry.raw_score != math.inf for entry in answer.distribution)
            assert all(entry.origin == xybrid.XybridDecisionOrigin.CALLER for entry in answer.distribution)
            assert sum(entry.probability for entry in answer.distribution) == pytest.approx(1.0, rel=0, abs=1e-14)
        assert isinstance(choice.value, xybrid.XybridDecisionValueChoice)
        winner = max(choice.distribution, key=lambda entry: entry.probability)
        assert choice.value.key == winner.key
        assert choice.confidence == winner.probability
        assert isinstance(yes_no.value, xybrid.XybridDecisionValueYesNo)
        assert yes_no.value.p_yes == yes_no.distribution[0].probability
        assert yes_no.confidence is None
        assert isinstance(score.value, xybrid.XybridDecisionValueScore)
        assert score.value.statistic == xybrid.XybridScoreStatistic.MEDIAN
        cumulative = 0.0
        for index, entry in enumerate(score.distribution):
            cumulative += entry.probability
            if cumulative >= 0.5:
                assert score.value.value == index
                break
        assert score.confidence == max(entry.probability for entry in score.distribution)

        with pytest.raises(xybrid.UnsupportedModelCapability):
            model.run_stream(request)
        # A refused surface must not poison the loaded model or its next context.
        recovered = model.run(request)
        assert recovered.decisions == result.decisions
        assert recovered.tokens_in == result.tokens_in
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
