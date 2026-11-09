"""The shared observability scenario, implemented with the public Python API.

This is the reference implementation of
``tools/scripts/observability_capture/scenario.md``: every binding performs the
same steps against ``capture_server.py`` and writes the same
``expectations.json``, which ``check_capture.py`` then checks against what was
exported. No model weights are involved.

    python observability_scenario.py --endpoint http://127.0.0.1:4318 \\
        --expectations out/expectations.json --workdir out/work

Each option falls back to ``XYBRID_SCENARIO_ENDPOINT``,
``XYBRID_SCENARIO_EXPECTATIONS`` and ``XYBRID_SCENARIO_WORKDIR``, which
``tools/scripts/observability_capture/run_scenario.py`` sets. Exits 1 when a
step raised; the error is recorded as ``scenario_error`` so the checker reports
it next to whatever was exported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import xybrid

BINDING = "python"
SCHEMA = "xybrid-observability-scenario/1"
SPEC_PATH = Path(__file__).resolve().parents[3] / "tools" / "scripts" / "observability_capture" / "scenario.json"
COUNTERS = ("accepted", "exported", "dropped", "failed", "pending")


def scenario_ids(binding: str) -> tuple:
    """T and P: hex of the UTF-8 name, right-padded with zeros (scenario.md, Identifiers)."""
    trace_id = ("xy:" + binding).encode("utf-8").hex().ljust(32, "0")[:32]
    parent_span_id = binding.encode("utf-8").hex().ljust(16, "0")[:16]
    return trace_id, parent_span_id


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


def options(traceparent: Optional[str], tracestate: Optional[str] = None,
            conversation_id: Optional[str] = None) -> xybrid.XybridObservationOptions:
    return xybrid.XybridObservationOptions(
        traceparent=traceparent, tracestate=tracestate,
        conversation_id=conversation_id, suppress_trace=False,
    )


def run_scenario(spec: Dict[str, Any], endpoint: str, workdir: Path, expectations: Dict[str, Any],
                 omit_rule: bool = True) -> None:
    """Performs steps (a)-(e), filling ``expectations`` as it goes.

    ``omit_rule=False`` is the control run: without the rule the recorded system
    instructions must reach the receiver, which shows that the scenario records
    them and so that their absence in a normal run is the rule's doing.
    """
    binding = expectations["binding"]
    trace_id, parent_span_id = expectations["trace_id"], expectations["parent_span_id"]
    conversation = f"scenario-conversation-{binding}"
    timeout_ms = spec["timeout_ms"]
    references = expectations["references"]
    stats = expectations["stats"]
    lifecycle = expectations["lifecycle"]
    automatic = expectations["automatic"]

    # (a) OTLP/JSON export to the receiver, with content rules and a device id.
    config = xybrid.XybridObservabilityConfig.otlp(endpoint)
    config.set_json_protocol(True)
    config.set_service(f"xybrid-scenario-{binding}", spec["per_binding"]["service_version"])
    config.set_device_id(f"scenario-device-{binding}")
    config.add_literal_redaction(spec["secret"], spec["replacement"])
    if omit_rule:
        config.omit_content_attribute(spec["omitted_attribute"])
    config.init()

    # (b) Parent-in: the workflow continues the caller's trace T under span P.
    workflow = xybrid.XybridWorkflowObservation(
        spec["workflow_name"], options(f"00-{trace_id}-{parent_span_id}-01"))
    references["workflow"] = reference(workflow.observation_ref())
    context = workflow.context()
    child = options(context.traceparent, context.tracestate)

    # (c) Manual children: a tool, a completed inference, a failed one and one
    # released unfinished.
    tool_spec = spec["tool"]
    call = xybrid.XybridToolCallInfo(
        tool_spec["name"], tool_spec["call_id"], tool_spec["tool_type"],
        tool_spec["description"], tool_spec["arguments_json"])
    with xybrid.XybridToolObservation(call, child) as tool:
        tool.record_result(tool_spec["result_json"])
        references["tool"] = reference(tool.observation_ref())

    request_spec = spec["inference_request"]
    request = xybrid.XybridInferenceRequest(
        request_spec["operation"], request_spec["provider"], request_spec["model"],
        xybrid.XybridExecutionLocation.LOCAL, request_spec["runtime"], request_spec["task"],
        request_spec["stream"], request_spec["max_tokens"], request_spec["temperature"],
        request_spec["top_p"], request_spec["server_address"], request_spec["server_port"])
    content = spec["inference"]
    with xybrid.XybridInferenceObservation(request, child) as inference:
        inference.record_system_instructions(content["system_instructions_json"])
        inference.record_input_messages(content["input_messages_json"])
        inference.record_output_messages(content["output_messages_json"])
        references["inference"] = reference(inference.observation_ref())

    failed = xybrid.XybridInferenceObservation(request, child)
    references["inference_failed"] = reference(failed.observation_ref())
    failed.fail(error_type=spec["failed_inference"]["error_type"], message=spec["failed_inference"]["message"])
    failed.close()

    released = xybrid.XybridInferenceObservation(request, child)
    released.record_input_messages(spec["released_inference"]["input_messages_json"])
    references["inference_released"] = reference(released.observation_ref())
    stats["before_release"] = snapshot()
    released.close()  # the last owner, never ended: cancels, does not flush
    stats["after_release"] = snapshot()

    # (d) The automatic path: a weightless model loads lazily and fails to run,
    # both under the workflow with a conversation id.
    model_dir = workdir / spec["model"]["directory_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_metadata.json").write_text(spec["model"]["metadata_json"], encoding="utf-8")
    automatic_parent = options(context.traceparent, context.tracestate, conversation)
    try:
        model = xybrid.XybridModel.load(xybrid.XybridModelSourceDirectory(str(model_dir)), automatic_parent)
    except xybrid.XybridError as error:
        automatic["load"] = {"outcome": "error", "error": f"{type(error).__name__}: {error}"}
        automatic["run"] = {"outcome": "skipped", "error": None}
    else:
        automatic["load"] = {"outcome": "ok", "error": None}
        references["load"] = reference(model.load_observation_ref())
        run_options = xybrid.XybridRunOptions(
            generation_config=None, abort_on=[], fallback_to_cloud=False, max_grace_tokens=0,
            correlation_id=None, traceparent=context.traceparent, tracestate=context.tracestate,
            conversation_id=conversation, suppress_trace=False)
        try:
            result = model.run(xybrid.XybridEnvelope.text(spec["model"]["run_input_text"]), run_options)
        except xybrid.XybridError as error:
            automatic["run"] = {"outcome": "error", "error": f"{type(error).__name__}: {error}"}
        else:
            automatic["run"] = {"outcome": "ok", "error": None}
            references["run"] = reference(result.observation)
        finally:
            model.close()

    workflow.complete()
    workflow.close()

    # (e) Lifecycle: flush, stats, shutdown twice, then a no-op call.
    lifecycle["flush"] = outcome(lambda: xybrid.observability_flush(timeout_ms))
    stats["after_flush"] = snapshot()
    lifecycle["shutdown"] = [outcome(lambda: xybrid.observability_shutdown(timeout_ms)) for _ in range(2)]
    stats["after_shutdown"] = snapshot()
    after = xybrid.XybridWorkflowObservation(spec["after_shutdown_workflow_name"], None)
    after_ref, after_context = after.observation_ref(), after.context()
    after.complete()
    after.close()
    lifecycle["after_shutdown"] = {
        "workflow_ref": reference(after_ref),
        "workflow_context": None if after_context is None else after_context.traceparent,
        "flush": outcome(lambda: xybrid.observability_flush(timeout_ms)),
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--endpoint", default=os.environ.get("XYBRID_SCENARIO_ENDPOINT"))
    parser.add_argument("--expectations", type=Path, default=os.environ.get("XYBRID_SCENARIO_EXPECTATIONS"))
    parser.add_argument("--workdir", type=Path, default=os.environ.get("XYBRID_SCENARIO_WORKDIR"))
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--without-omit-rule", action="store_true",
                        help="control run: configure no omit rule, so the omitted content is exported "
                             "and the checker must fail content.omit and inference.content")
    args = parser.parse_args(argv)
    if not args.endpoint or args.expectations is None:
        parser.error("--endpoint and --expectations are required (or XYBRID_SCENARIO_* variables)")
    workdir = args.workdir or args.expectations.parent / "work"

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    trace_id, parent_span_id = scenario_ids(BINDING)
    expectations: Dict[str, Any] = {
        "schema": SCHEMA,
        "binding": BINDING,
        "sdk_version": xybrid.version(),
        "trace_id": trace_id,
        "parent_span_id": parent_span_id,
        "references": {name: None for name in (
            "workflow", "tool", "inference", "inference_failed", "inference_released", "load", "run")},
        "automatic": {},
        "stats": {},
        "lifecycle": {},
        "scenario_error": None,
    }
    status = 0
    try:
        run_scenario(spec, args.endpoint, workdir, expectations, omit_rule=not args.without_omit_rule)
    except Exception:  # noqa: BLE001 -- report every failure through expectations.json
        expectations["scenario_error"] = traceback.format_exc(limit=5)
        status = 1
        try:
            xybrid.observability_shutdown(spec["timeout_ms"])
        except xybrid.XybridError:
            pass
    args.expectations.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.expectations.with_name(args.expectations.name + ".tmp")
    temporary.write_text(json.dumps(expectations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.expectations)
    if status:
        print(expectations["scenario_error"], file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
