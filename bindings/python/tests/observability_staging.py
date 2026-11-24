"""The step 6 staging scenario, implemented with the public Python API.

This is the reference implementation of
``tools/scripts/observability_staging/README.md``: two sessions in one process,
``direct`` (the Xybrid preset, metadata only) and ``collector`` (OTLP to the
pinned Collector, content captured with one literal redaction), each loading a
real registry model under a remote parent, stopping one stream after three
chunks, streaming a second to the end, and recording feedback and an
evaluation on the run reference.

    python observability_staging.py --config CONFIG --expectations OUT

``run_staging.py`` writes ``CONFIG`` (mode 0600; it holds the staging key) and
fills both paths in. The scenario reads the config, deletes it, and never
prints or writes the key. It is not a pytest module: it needs the staging
driver, the network and a model download, so nothing in the test suite runs
it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import xybrid

SCHEMA = "xybrid-staging-scenario/1"
COUNTERS = ("accepted", "exported", "dropped", "failed", "pending")


def reference(value: Optional[xybrid.XybridObservationRef]) -> Optional[Dict[str, str]]:
    return None if value is None else {"trace_id": value.trace_id, "span_id": value.span_id}


def snapshot() -> Dict[str, Any]:
    stats = xybrid.observability_stats()
    return {
        "enabled": stats.enabled,
        **{signal: {key: getattr(getattr(stats, signal), key) for key in COUNTERS}
           for signal in ("traces", "logs", "metrics")},
    }


def outcome(call: Callable[[], Any]) -> str:
    try:
        call()
    except xybrid.XybridError as error:
        return f"error: {type(error).__name__}: {error}"
    return "ok"


def generation(max_tokens: int) -> xybrid.XybridGenerationConfig:
    return xybrid.XybridGenerationConfig(
        max_tokens=max_tokens, temperature=None, top_p=None, min_p=None, top_k=None,
        repetition_penalty=None, stop_sequences=[], grammar=None, tools=[])


def run_options(max_tokens: int, context: xybrid.XybridObservationContext,
                conversation_id: str) -> xybrid.XybridRunOptions:
    return xybrid.XybridRunOptions(
        generation_config=generation(max_tokens), abort_on=[], fallback_to_cloud=False, max_grace_tokens=0,
        correlation_id=None, traceparent=context.traceparent, tracestate=context.tracestate,
        conversation_id=conversation_id, suppress_trace=False, frame_session_id=None)


def configure(config: Dict[str, Any], session: str) -> None:
    """Step 1: the session's export, identity and content rules, then init."""
    constants = config["constants"]
    values = config["sessions"][session]
    if session == "direct":
        observability = xybrid.XybridObservabilityConfig.xybrid(values["api_key"])
        observability.set_traces_endpoint(values["traces_endpoint"])
        observability.set_metadata_only(True)
    else:
        observability = xybrid.XybridObservabilityConfig.otlp(values["endpoint"])
        observability.add_literal_redaction(constants["secret"], constants["replacement"])
    binding, lane = config["binding"], config["lane"]
    observability.set_service(constants["per_lane"]["service_name"].format(binding=binding),
                              constants["per_lane"]["service_version"])
    observability.set_device_id(constants["per_lane"]["device_id"].format(lane=lane))
    for key, value in (("run", config["run"]), ("lane", lane), ("commit", config["commit"]), ("session", session)):
        observability.set_resource_attribute(f"xybrid.acceptance.{key}", value)
    observability.init()


def stream(model: xybrid.XybridModel, prompt: str, options: xybrid.XybridRunOptions,
           stop_after: Optional[int]) -> Dict[str, Any]:
    """Steps 4 and 5: a pull stream, closed after ``stop_after`` chunks or
    read to the end."""
    stream_id = model.run_stream(xybrid.XybridEnvelope.text(prompt), options)
    chunks, finish = 0, None
    try:
        while True:
            event = model.stream_next(stream_id)
            if event.kind == xybrid.XybridStreamEventKind.COMPLETE:
                result = model.stream_result(stream_id)
                stream_id = None
                return {"outcome": "completed" if stop_after is not None else "ok", "chunks": chunks,
                        "result": result, "finish_reason": finish}
            if event.kind == xybrid.XybridStreamEventKind.TOKEN and event.token is not None:
                chunks += 1
                finish = event.token.finish_reason or finish
                if stop_after is not None and chunks >= stop_after:
                    # Closing before the terminal event cancels the run at
                    # its next token (a user cancellation, error.type cancelled).
                    model.stream_close(stream_id)
                    stream_id = None
                    return {"outcome": "cancelled", "chunks": chunks, "result": None, "finish_reason": None}
    finally:
        if stream_id is not None:
            model.stream_close(stream_id)


def run_session(config: Dict[str, Any], session: str, record: Dict[str, Any]) -> None:
    constants = config["constants"]
    values = config["sessions"][session]
    timeout_ms = constants["timeout_ms"]
    conversation = constants["per_lane"]["conversation_id"].format(run=config["run"], session=session)
    record.update({"trace_id": values["trace_id"], "parent_span_id": values["parent_span_id"],
                   "references": {"workflow": None, "load": None, "run": None}})
    record["init"] = outcome(lambda: configure(config, session))
    if record["init"] != "ok":
        raise RuntimeError(f"{session} init failed: {record['init']}")

    # Step 2: the workflow continues the driver's trace T under span P.
    workflow = xybrid.XybridWorkflowObservation(constants["workflow_name"], xybrid.XybridObservationOptions(
        traceparent=f"00-{values['trace_id']}-{values['parent_span_id']}-01", tracestate=None,
        conversation_id=None, suppress_trace=False))
    record["references"]["workflow"] = reference(workflow.observation_ref())
    context = workflow.context()

    # Step 3: a real registry model, loaded under the workflow.
    model = xybrid.XybridModel.load(
        xybrid.XybridModelSourceRegistry(config["model_id"], False),
        xybrid.XybridObservationOptions(traceparent=context.traceparent, tracestate=context.tracestate,
                                        conversation_id=conversation, suppress_trace=False))
    try:
        record["references"]["load"] = reference(model.load_observation_ref())

        # Step 4: stopped after three chunks by closing the stream.
        stopped = constants["stopped"]
        result = stream(model, stopped["prompt"], run_options(stopped["max_tokens"], context, conversation),
                        stopped["stop_after_chunks"])
        record["stopped"] = {"outcome": result["outcome"], "chunks_consumed": result["chunks"], "delivery": "pull"}

        # Step 5: streamed to the end; the result carries the run reference.
        completed = constants["completed"]
        result = stream(model, completed["prompt"], run_options(completed["max_tokens"], context, conversation),
                        None)
        final = result["result"]
        text = final.text
        record["completed"] = {"outcome": result["outcome"], "chunks": result["chunks"],
                               "tokens_out": final.metrics.tokens_out, "text_chars": len(text or ""),
                               "finish_reason": result["finish_reason"]}
        run_ref = final.observation
        record["references"]["run"] = reference(run_ref)

        # Step 6: quality results on the run reference.
        feedback, evaluation = constants["feedback"], constants["evaluation"]
        record["quality"] = {
            "feedback": xybrid.observability_record_feedback(run_ref, feedback["score"], feedback["label"],
                                                             feedback["comment"]),
            "evaluation": xybrid.observability_record_evaluation(
                run_ref, evaluation["name"], evaluation["score_value"], evaluation["score_label"],
                evaluation["explanation"]),
        }
    finally:
        model.close()

    # Step 7: lifecycle.
    workflow.complete()
    workflow.close()
    record["lifecycle"] = {"flush": outcome(lambda: xybrid.observability_flush(timeout_ms))}
    record["stats"] = {"after_flush": snapshot()}
    record["lifecycle"]["shutdown"] = outcome(lambda: xybrid.observability_shutdown(timeout_ms))
    record["stats"]["after_shutdown"] = snapshot()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--config", type=Path, default=os.environ.get("XYBRID_STAGING_CONFIG"))
    parser.add_argument("--expectations", type=Path, default=os.environ.get("XYBRID_STAGING_EXPECTATIONS"))
    args = parser.parse_args(argv)
    if args.config is None or args.expectations is None:
        parser.error("--config and --expectations are required (or XYBRID_STAGING_* variables)")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.config.unlink()  # it holds the staging key; nothing reads it again

    expectations: Dict[str, Any] = {
        "schema": SCHEMA, "binding": config["binding"], "lane": config["lane"], "run": config["run"],
        "sdk_version": xybrid.version(), "model_id": config["model_id"],
        "device": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                   "python": platform.python_version()},
        "sessions": {}, "scenario_error": None,
    }
    status = 0
    try:
        for session in ("direct", "collector"):
            expectations["sessions"][session] = {}
            run_session(config, session, expectations["sessions"][session])
    except Exception:  # noqa: BLE001 -- report every failure through expectations.json
        expectations["scenario_error"] = traceback.format_exc(limit=6)
        status = 1
        try:
            xybrid.observability_shutdown(config["constants"]["timeout_ms"])
        except xybrid.XybridError:
            pass
    config.clear()
    args.expectations.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.expectations.with_name(args.expectations.name + ".tmp")
    temporary.write_text(json.dumps(expectations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.expectations)
    print(json.dumps({session: {"references": values.get("references"), "stopped": values.get("stopped"),
                                "completed": values.get("completed"), "quality": values.get("quality")}
                      for session, values in expectations["sessions"].items()}, indent=1))
    if status:
        print(expectations["scenario_error"], file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
