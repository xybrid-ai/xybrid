#!/usr/bin/env python3
"""Runs one binding's scenario against a fresh receiver, then checks it.

    python3 run_scenario.py --binding python --out-dir out/python -- \\
        python3 bindings/python/tests/observability_scenario.py

Starts ``capture_server.py`` on ``--port`` (0 picks a free one), runs the
command after ``--``, waits ``--grace`` seconds so a late export would still
land, stops the receiver and runs ``check_capture.py``. ``--out-dir`` receives
``capture.jsonl``, ``ready.json``, ``server.log``, ``report.json`` and, unless
``--expectations`` points elsewhere, ``expectations.json``.

The command sees ``XYBRID_SCENARIO_BINDING``, ``_ENDPOINT``, ``_EXPECTATIONS``,
``_WORKDIR`` and ``_PORT`` in its environment, and no ``OTEL_*`` variables
(``--keep-otel-env`` keeps them): an inherited content-capture mode would change
what the scenario checks. For commands that cannot pass
environment through (``adb shell am instrument``, ``xcodebuild``, ``flutter
drive``), the literal placeholders ``{endpoint}``, ``{port}``,
``{expectations}``, ``{workdir}``, ``{binding}`` and ``{out_dir}`` in its
arguments are replaced. ``--endpoint-host`` sets the host in the endpoint:
``10.0.2.2`` for an Android emulator, the host's LAN address for a device. A
lane that copies ``expectations.json`` off a device does so inside its command
and names the copy with ``--expectations``.

Exit status 0 only when the command succeeded and every check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_capture  # noqa: E402 -- sibling module, imported after the path is set


def wait_for_ready(path: Path, server: subprocess.Popen, timeout: float) -> Dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if server.poll() is not None:
            raise RuntimeError(f"capture_server.py exited with {server.returncode} before it was ready")
        time.sleep(0.05)
    raise RuntimeError(f"capture_server.py was not ready after {timeout:.0f}s")


def substitute(command: List[str], values: Dict[str, str]) -> List[str]:
    out = []
    for argument in command:
        for key, value in values.items():
            argument = argument.replace("{" + key + "}", value)
        out.append(argument)
    return out


def stop(server: subprocess.Popen) -> None:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--binding", required=True, help="binding name, e.g. python, swift, kotlin")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0, help="receiver port (default: a free one)")
    parser.add_argument("--bind", default="0.0.0.0", help="receiver bind address (default 0.0.0.0)")
    parser.add_argument("--endpoint-host", default="127.0.0.1",
                        help="host the scenario uses to reach the receiver (10.0.2.2 on an Android emulator)")
    parser.add_argument("--expectations", type=Path, help="where the scenario's expectations.json ends up")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds the command may run")
    parser.add_argument("--grace", type=float, default=1.0, help="seconds to keep receiving after the command")
    parser.add_argument("--keep-otel-env", action="store_true",
                        help="pass OTEL_* variables through to the command (removed by default)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the scenario command")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the scenario command after --")
    try:
        check_capture.scenario_ids(args.binding)
    except ValueError as error:
        parser.error(str(error))

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "work"
    workdir.mkdir(exist_ok=True)
    capture = out_dir / "capture.jsonl"
    ready = out_dir / "ready.json"
    expectations = (args.expectations or out_dir / "expectations.json").resolve()
    for stale in (ready, expectations):
        if stale.exists():
            stale.unlink()

    with (out_dir / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, "-I", str(HERE / "capture_server.py"), "--host", args.bind,
             "--port", str(args.port), "--out", str(capture), "--ready-file", str(ready)],
            stdout=log, stderr=subprocess.STDOUT)
        try:
            try:
                port = str(wait_for_ready(ready, server, timeout=20)["port"])
            except RuntimeError as error:
                print(f"error: {error}; see {out_dir / 'server.log'}", file=sys.stderr)
                return 1
            values = {
                "binding": args.binding,
                "endpoint": f"http://{args.endpoint_host}:{port}",
                "port": port,
                "expectations": str(expectations),
                "workdir": str(workdir),
                "out_dir": str(out_dir),
            }
            # The scenario configures export explicitly; an inherited OTEL_*
            # variable (content capture mode, batch tuning) would change what
            # it checks, so the command starts without them.
            env = {key: value for key, value in os.environ.items()
                   if args.keep_otel_env or not key.startswith("OTEL_")}
            env.update({f"XYBRID_SCENARIO_{key.upper()}": value for key, value in values.items()})
            print(f"==> receiver on {values['endpoint']} (bound {args.bind}:{port}); running the {args.binding} scenario",
                  flush=True)
            try:
                status = subprocess.run(substitute(command, values), env=env, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                print(f"error: the scenario command ran past {args.timeout:.0f}s", file=sys.stderr)
                status = 124
            time.sleep(max(args.grace, 0.0))
        finally:
            stop(server)

    if status != 0:
        print(f"error: the scenario command exited with {status}", file=sys.stderr)
    if not expectations.exists():
        print(f"error: the scenario wrote no {expectations}", file=sys.stderr)
        return 1
    print("==> checking the capture", flush=True)
    try:
        results, captured, binding = check_capture.check_files(capture, expectations, args.binding)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    check_capture.print_results(results, binding, captured)
    (out_dir / "report.json").write_text(
        json.dumps(check_capture.report(results, binding, captured), indent=2) + "\n", encoding="utf-8")
    return 0 if status == 0 and all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
