"""Validate the JSON report written by bindings/react-native/consumer-smoke."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

REPORT_SCHEMA_VERSION = 1
REPORT_FILE_NAME = "xybrid-consumer-smoke-report.json"

REQUIRED_STEP_IDS: Tuple[str, ...] = (
    "version_match",
    "model_load",
    "model_info",
    "warmup",
    "inference",
    "streaming",
    "cancel_settles",
    "post_cancel_inference",
    "conversation",
    "constrained_json",
    "release",
)

# Bundle / package identifiers must stay in sync with consumer-smoke/app.json.
ANDROID_PACKAGE = "dev.xybrid.consumersmoke"
IOS_BUNDLE_ID = "dev.xybrid.consumersmoke"
MODEL_DIR_RELATIVE = "models/functiongemma-270m-it"


class ReportError(Exception):
    """The report is missing, stale, incomplete, or failed."""


def load_report(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"report missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"report is not valid JSON: {path}: {exc}") from exc


def validate_report(
    report: Mapping[str, Any],
    *,
    expected_run_id: Optional[str] = None,
    require_success: bool = True,
) -> None:
    """Raise ReportError when the report is incomplete, stale, or failed."""
    if report.get("schemaVersion") != REPORT_SCHEMA_VERSION:
        raise ReportError(
            f"schemaVersion {report.get('schemaVersion')!r} != {REPORT_SCHEMA_VERSION}"
        )

    run_id = report.get("runId")
    if not isinstance(run_id, str) or not run_id:
        raise ReportError("runId missing")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ReportError(f"stale report: runId {run_id!r} != expected {expected_run_id!r}")

    for key in ("startedAt", "finishedAt", "versions", "results"):
        if key not in report:
            raise ReportError(f"missing field {key!r}")

    versions = report["versions"]
    if not isinstance(versions, Mapping):
        raise ReportError("versions must be an object")
    for key in ("npmPackage", "nativeSdk", "expo", "reactNative"):
        value = versions.get(key)
        if not isinstance(value, str) or not value or value == "unknown":
            raise ReportError(f"versions.{key} missing or unknown")

    if versions["npmPackage"] != versions["nativeSdk"]:
        raise ReportError(
            f"version mismatch: npm={versions['npmPackage']} native={versions['nativeSdk']}"
        )

    results = report["results"]
    if not isinstance(results, list) or not results:
        raise ReportError("results missing or empty")

    by_id: Dict[str, Mapping[str, Any]] = {}
    for entry in results:
        if not isinstance(entry, Mapping):
            raise ReportError("results entry is not an object")
        step_id = entry.get("id")
        if not isinstance(step_id, str):
            raise ReportError("results entry missing id")
        if step_id in by_id:
            raise ReportError(f"duplicate step id {step_id!r}")
        by_id[step_id] = entry

    missing = [step_id for step_id in REQUIRED_STEP_IDS if step_id not in by_id]
    if missing:
        raise ReportError(f"incomplete report; missing steps: {', '.join(missing)}")

    failed = [
        step_id
        for step_id in REQUIRED_STEP_IDS
        if by_id[step_id].get("ok") is not True
    ]
    if failed:
        raise ReportError(f"failed steps: {', '.join(failed)}")

    if require_success and report.get("success") is not True:
        raise ReportError("report.success is not true")

    # success must only be claimed after release succeeded (already checked via
    # REQUIRED_STEP_IDS, but keep the invariant explicit).
    if report.get("success") is True and by_id["release"].get("ok") is not True:
        raise ReportError("success true before release ok")


def validate_report_file(
    path: Path,
    *,
    expected_run_id: Optional[str] = None,
    require_success: bool = True,
) -> Dict[str, Any]:
    report = load_report(path)
    validate_report(report, expected_run_id=expected_run_id, require_success=require_success)
    return report


def wait_for_report(
    pull,
    dest: Path,
    *,
    timeout_s: float,
    poll_s: float = 2.0,
    started_after: Optional[float] = None,
) -> Path:
    """Poll ``pull()`` until a fresh, finished report lands at ``dest``.

    ``pull`` should copy the on-device report to ``dest`` (or raise/return
    without creating it when absent). ``started_after`` (epoch seconds) rejects
    reports whose ``startedAt`` is older than the launch, catching stale files.
    """
    deadline = time.monotonic() + timeout_s
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            if dest.exists():
                dest.unlink()
        except OSError:
            pass
        try:
            pull()
        except Exception as exc:  # noqa: BLE001 — surface later if we time out
            last_error = exc
            time.sleep(poll_s)
            continue
        if not dest.exists():
            time.sleep(poll_s)
            continue
        try:
            report = load_report(dest)
            if "finishedAt" not in report:
                time.sleep(poll_s)
                continue
            if started_after is not None:
                started_at = report.get("startedAt")
                if isinstance(started_at, str):
                    # Accept ISO-8601; if parsing fails, still validate below.
                    try:
                        from datetime import datetime

                        started_ts = datetime.fromisoformat(
                            started_at.replace("Z", "+00:00")
                        ).timestamp()
                        if started_ts + 1 < started_after:
                            raise ReportError(
                                f"stale report startedAt={started_at} before launch"
                            )
                    except ValueError:
                        pass
            validate_report(report)
            return dest
        except ReportError as exc:
            last_error = exc
            # A finished-but-failed report should fail immediately.
            try:
                report = load_report(dest)
                if report.get("finishedAt") and report.get("success") is False:
                    raise ReportError(f"qualification failed: {exc}") from exc
            except ReportError:
                raise
            except Exception:
                pass
            time.sleep(poll_s)
            continue
    message = f"timed out after {timeout_s:.0f}s waiting for a valid report"
    if last_error is not None:
        raise ReportError(f"{message}: {last_error}") from last_error
    raise ReportError(message)
