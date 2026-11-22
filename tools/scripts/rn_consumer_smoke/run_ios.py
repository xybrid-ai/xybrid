#!/usr/bin/env python3
"""Install, stage model, launch the iOS consumer-smoke Release app, collect report.

Exit 0 on a valid successful report; 124 on timeout; 1 on failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rn_consumer_smoke.report import (  # noqa: E402
    IOS_BUNDLE_ID,
    REPORT_FILE_NAME,
    ReportError,
    validate_report_file,
    wait_for_report,
)
from rn_consumer_smoke.stage_model import stage_ios  # noqa: E402

DEFAULT_TIMEOUT_S = 600


def simctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    command = ["xcrun", "simctl", *args]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True, capture_output=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True, help="Path to the .app bundle")
    parser.add_argument("--udid", required=True, help="Simulator UDID")
    parser.add_argument("--model-dir", type=Path, required=True, help="FunctionGemma fixture dir")
    parser.add_argument("--out-dir", type=Path, required=True, help="Host directory for report + logs")
    parser.add_argument("--bundle-id", default=IOS_BUNDLE_ID)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_host = out_dir / REPORT_FILE_NAME

    if not args.app.is_dir():
        print(f"error: .app not found: {args.app}", file=sys.stderr)
        return 1

    simctl("uninstall", args.udid, args.bundle_id, check=False)
    simctl("install", args.udid, str(args.app))
    stage_ios(args.model_dir, udid=args.udid, bundle_id=args.bundle_id)

    launched_at = time.time()
    simctl("launch", args.udid, args.bundle_id)

    def pull() -> None:
        container = simctl("get_app_container", args.udid, args.bundle_id, "data", check=False)
        if container.returncode != 0:
            return
        data_root = Path(container.stdout.strip())
        remote = data_root / "Documents" / REPORT_FILE_NAME
        if remote.is_file():
            report_host.write_bytes(remote.read_bytes())

    try:
        wait_for_report(pull, report_host, timeout_s=args.timeout, started_after=launched_at)
        validate_report_file(report_host)
        print(f"ok: {report_host}", flush=True)
        return 0
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 124 if "timed out" in str(exc) else 1
    finally:
        if report_host.exists():
            print(f"report at {report_host}", flush=True)
        try:
            pull()
        except Exception:
            pass
        # Simulator system log snippet for failures.
        log_path = out_dir / "simctl-spawn-log.txt"
        spawned = simctl(
            "spawn",
            args.udid,
            "log",
            "show",
            "--last",
            "5m",
            "--predicate",
            'subsystem CONTAINS "react" OR eventMessage CONTAINS "xybrid"',
            check=False,
        )
        log_path.write_text(spawned.stdout or spawned.stderr or "", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
