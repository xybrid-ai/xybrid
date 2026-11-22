#!/usr/bin/env python3
"""Install, stage model, launch the Android consumer-smoke Release APK, collect report.

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
    ANDROID_PACKAGE,
    REPORT_FILE_NAME,
    ReportError,
    validate_report_file,
    wait_for_report,
)
from rn_consumer_smoke.stage_model import stage_android  # noqa: E402

DEFAULT_TIMEOUT_S = 600


def adb(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    command = ["adb", *args]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True, capture_output=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", type=Path, required=True, help="Release APK path")
    parser.add_argument("--model-dir", type=Path, required=True, help="FunctionGemma fixture dir")
    parser.add_argument("--out-dir", type=Path, required=True, help="Host directory for report + logs")
    parser.add_argument("--package", default=ANDROID_PACKAGE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_host = out_dir / REPORT_FILE_NAME
    logcat_path = out_dir / "logcat.txt"

    if not args.apk.is_file():
        print(f"error: APK not found: {args.apk}", file=sys.stderr)
        return 1

    adb("wait-for-device")
    # Clear any previous install so we never read a stale report.
    adb("uninstall", args.package, check=False)
    adb("install", "-r", str(args.apk))

    stage_android(args.model_dir, package=args.package)

    # Clear logcat and launch; ACTIVITY is the Expo main activity.
    adb("logcat", "-c", check=False)
    launched_at = time.time()
    launch = adb(
        "shell",
        "am",
        "start",
        "-n",
        f"{args.package}/.MainActivity",
        check=False,
    )
    if launch.returncode != 0:
        # Expo / RN 0.88 may use a different activity name; fall back to the launcher intent.
        adb(
            "shell",
            "monkey",
            "-p",
            args.package,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        )

    remote_report = f"/data/data/{args.package}/files/{REPORT_FILE_NAME}"

    def pull() -> None:
        adb("root", check=False)
        result = adb("shell", "test", "-f", remote_report, check=False)
        if result.returncode != 0:
            return
        adb("pull", remote_report, str(report_host))

    try:
        wait_for_report(pull, report_host, timeout_s=args.timeout, started_after=launched_at)
        validate_report_file(report_host)
        print(f"ok: {report_host}", flush=True)
        return 0
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 124 if "timed out" in str(exc) else 1
    finally:
        # Retain runtime logs and whatever report we have.
        logcat = adb("logcat", "-d", check=False)
        logcat_path.write_text(logcat.stdout or logcat.stderr or "", encoding="utf-8")
        if report_host.exists():
            print(f"report at {report_host}", flush=True)
        # Also try a last-chance pull for artifact upload.
        try:
            pull()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
