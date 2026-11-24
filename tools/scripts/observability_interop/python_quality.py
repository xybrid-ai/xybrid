"""Records an evaluation and user feedback on the Python scenario's inference.

    python python_quality.py --endpoint http://127.0.0.1:4318 \\
        --expectations out/python/expectations.json

The runtime scenario (``bindings/python/tests/observability_scenario.py``)
shuts observability down as its last step, so the quality results come from a
second process, the way an application records feedback that arrives later:
it reads the stored ``references.inference`` and records both results against
it through the public Python API, exporting OTLP/protobuf (the scenario itself
uses OTLP/JSON). Service, device id and redaction match the scenario, so the
feedback comment carrying the scenario secret must arrive redacted.

Prints one ``observation refs:`` line naming the target; exits 1 when a result
was not recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import xybrid

SPEC_PATH = Path(__file__).resolve().parents[1] / "observability_capture" / "scenario.json"
EVALUATION_NAME = "relevance"
FEEDBACK_LABEL = "thumbs_up"


def feedback_comment(secret: str) -> str:
    return f"helpful, and my token {secret} stayed private"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    expectations = json.loads(args.expectations.read_text(encoding="utf-8"))
    target = (expectations.get("references") or {}).get("inference")
    if not isinstance(target, dict):
        print("error: the scenario recorded no inference reference", file=sys.stderr)
        return 1

    config = xybrid.XybridObservabilityConfig.otlp(args.endpoint)
    config.set_service("xybrid-scenario-python", spec["per_binding"]["service_version"])
    config.set_device_id("scenario-device-python")
    config.add_literal_redaction(spec["secret"], spec["replacement"])
    config.init()

    reference = xybrid.XybridObservationRef(trace_id=target["trace_id"], span_id=target["span_id"])
    evaluated = xybrid.observability_record_evaluation(
        reference, EVALUATION_NAME, 1.0, "relevant", "answers the question")
    rated = xybrid.observability_record_feedback(reference, 1.0, FEEDBACK_LABEL, feedback_comment(spec["secret"]))
    xybrid.observability_flush(spec["timeout_ms"])
    xybrid.observability_shutdown(spec["timeout_ms"])
    print(f"evaluation recorded: {evaluated}, feedback recorded: {rated}")
    print("observation refs: " + json.dumps({"inference": target}))
    return 0 if evaluated and rated else 1


if __name__ == "__main__":
    sys.exit(main())
