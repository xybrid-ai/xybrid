"""Manual observations exported to an OpenTelemetry Collector, no model needed.

    python bindings/python/examples/observability.py [http://localhost:4318]

Starts OTLP export, records a workflow with a tool call and an inference run
by another runtime, then flushes. Without a Collector listening, the flush
raises ``xybrid.TelemetryDelivery`` and the per-signal counters show the loss.
"""

from __future__ import annotations

import sys

import xybrid


def child_options(operation) -> xybrid.XybridObservationOptions | None:
    """Explicit parent controls for a child of ``operation``."""
    context = operation.context()
    if context is None:  # export is off or the trace is suppressed
        return None
    return xybrid.XybridObservationOptions(
        traceparent=context.traceparent,
        tracestate=context.tracestate,
        conversation_id=None,
        suppress_trace=False,
    )


def main() -> None:
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4318"
    config = xybrid.XybridObservabilityConfig.otlp(endpoint)
    config.set_service("xybrid-python-example", xybrid.version())
    config.init()

    with xybrid.XybridWorkflowObservation("answer-question", None) as workflow:
        children = child_options(workflow)

        call = xybrid.XybridToolCallInfo("search", "call-1", None, None, '{"q": "otel"}')
        with xybrid.XybridToolObservation(call, children) as tool:
            tool.record_result('{"hits": 3}')

        request = xybrid.XybridInferenceRequest(
            "chat", "example-runtime", "example-model", xybrid.XybridExecutionLocation.LOCAL,
            None, None, True, None, None, None, None, None,
        )
        with xybrid.XybridInferenceObservation(request, children) as inference:
            inference.record_input_messages('[{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]')
            inference.record_chunk()  # a chunk is not a first token; record that only when measured
            inference.record_usage(xybrid.XybridUsage(3, 5, None, None, None))
            inference.record_finish_reasons(["stop"])

        reference = workflow.observation_ref()

    print("workflow reference:", reference)
    try:
        xybrid.observability_flush(5000)
    except xybrid.TelemetryDelivery as error:
        print("delivery incomplete:", error)
    print("stats:", xybrid.observability_stats())
    try:
        xybrid.observability_shutdown(5000)
    except xybrid.TelemetryDelivery:
        pass


if __name__ == "__main__":
    main()
