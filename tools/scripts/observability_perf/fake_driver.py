#!/usr/bin/env python3
"""A stand-in for the ``observability_perf`` example driver, for dry runs of
``run_perf.py e2e`` and its unit tests: same arguments, same JSON schema, no
model. It synthesizes deterministic token timings (enabled modes a little
slower), holds a little extra memory in enabled modes so peak RSS differs,
and talks to the endpoint like the real pipeline would: one OTLP request to
an accepting sink, a request that waits for no answer to a blocked one.

    python3 run_perf.py e2e --driver-cmd "python3 fake_driver.py" \\
        --model-dir /tmp/fake-model --allow-unpinned-model --rounds 2

``FAKE_DRIVER_FAIL=<mode>`` makes that mode exit 1 (tests of failure paths);
``FAKE_DRIVER_SHORT=<mode>`` ends that mode's first run one token early (tests
of the sanity checks).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import sys
import time
import urllib.parse
from typing import Any, Dict

SLOWDOWN = {"off": 1.0, "metadata": 1.002, "content": 1.004, "blocked": 1.004}
EXTRA_MIB = {"off": 0, "metadata": 2, "content": 3, "blocked": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mode", default="off")
    parser.add_argument("--endpoint")
    parser.add_argument("--out")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-bytes", type=int, default=2048)
    parser.add_argument("--shutdown-timeout-ms", type=int, default=2000)
    return parser.parse_args()


# Request body bytes per mode: content spans carry prompts, metadata spans do not.
BODY_BYTES = {"metadata": 256, "content": 4096, "blocked": 4096}


def export(endpoint: str, mode: str, timeout_ms: int) -> Dict[str, Any]:
    """One OTLP/HTTP traces request; a blocked endpoint never answers."""
    blocked = mode == "blocked"
    url = urllib.parse.urlparse(endpoint)
    wait = min(timeout_ms, 300) / 1000 if blocked else 5
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=wait)
    body = b"\x0a" * BODY_BYTES.get(mode, 256)
    started = time.monotonic()
    try:
        connection.request("POST", "/v1/traces", body=body, headers={"Content-Type": "application/x-protobuf"})
        status = connection.getresponse().status
        return {"exported": 1 if status == 200 else 0, "failed": 0 if status == 200 else 1, "status": status,
                "shutdown_ms": (time.monotonic() - started) * 1e3}
    except OSError:
        return {"exported": 0, "failed": 1, "status": None, "shutdown_ms": (time.monotonic() - started) * 1e3}
    finally:
        connection.close()


def main() -> int:
    args = parse_args()
    if os.environ.get("FAKE_DRIVER_FAIL") == args.mode:
        print(f"fake_driver: failing {args.mode} on request", file=sys.stderr)
        return 1
    if args.mode != "off" and not args.endpoint:
        print("fake_driver: --endpoint required", file=sys.stderr)
        return 2
    rng = random.Random(f"{args.mode}-{os.getpid()}")
    ballast = bytearray(EXTRA_MIB.get(args.mode, 0) * 1024 * 1024)
    for i in range(0, len(ballast), 4096):
        ballast[i] = 1  # touch every page so it counts as resident
    factor = SLOWDOWN.get(args.mode, 1.0)
    time.sleep(0.01)
    runs = []
    for index in range(args.runs):
        tokens = args.max_tokens - (1 if index == 0 and os.environ.get("FAKE_DRIVER_SHORT") == args.mode else 0)
        gaps = [round(5.0 * factor * (1 + rng.uniform(-0.02, 0.02)), 4) for _ in range(tokens - 1)]
        first = 40.0 * factor * (1 + rng.uniform(-0.02, 0.02))
        wall = first + sum(gaps) + 1.0
        runs.append({
            "wall_ms": wall,
            "callbacks": tokens,
            "first_callback_ms": first,
            "callback_gaps_ms": gaps,
            "decode_callbacks_per_s": (tokens - 1) / (sum(gaps) / 1000),
            "finish_reason": "length",
            "runtime": {
                "ttft_ms": first - 0.5,
                "tokens_generated": float(tokens),
                "tokens_per_second": 1000.0 / (sum(gaps) / len(gaps)),
                "generation_time_ms": wall - 1.0,
                "mean_itl_ms": sum(gaps) / len(gaps),
                "p95_itl_ms": sorted(gaps)[int(0.95 * len(gaps))],
            },
            "has_observation_ref": args.mode != "off",
        })
    telemetry: Dict[str, Any] = {"enabled": args.mode != "off", "endpoint": args.endpoint,
                                 "stats_before_shutdown": None, "shutdown_result": None, "shutdown_ms": None,
                                 "shutdown_timeout_ms": args.shutdown_timeout_ms, "stats_after_shutdown": None}
    if args.mode != "off":
        outcome = export(args.endpoint, args.mode, args.shutdown_timeout_ms)
        traces = {"accepted": 1, "exported": outcome["exported"], "dropped": 0, "failed": outcome["failed"], "pending": 0}
        zero = {"accepted": 0, "exported": 0, "dropped": 0, "failed": 0, "pending": 0}
        telemetry.update(
            stats_before_shutdown={"traces": dict(traces, exported=0, failed=0, pending=1), "logs": zero, "metrics": zero},
            shutdown_result="Ok(())" if outcome["exported"] else "Err(Timeout)",
            shutdown_ms=outcome["shutdown_ms"],
            stats_after_shutdown={"traces": traces, "logs": zero, "metrics": zero},
        )
    report = {
        "driver": "fake_driver", "sdk_version": "0.0.0-fake", "mode": args.mode, "model_dir": args.model_dir,
        "max_tokens": args.max_tokens, "prompt_bytes": args.prompt_bytes, "warmup": args.warmup,
        "load_ms": 12.5, "runs": runs, "telemetry": telemetry, "process_ms": 50.0,
        "self_peak_rss_bytes": None, "ballast_bytes": len(ballast),
    }
    body = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(body)
    else:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
