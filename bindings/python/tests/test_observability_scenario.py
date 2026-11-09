"""The shared observability scenario, end to end on the compiled bridge.

Runs ``tools/scripts/observability_capture/run_scenario.py``: a receiver in its
own process, ``observability_scenario.py`` against it, then
``check_capture.py`` over what arrived. This is the harness every native
binding lane uses, validated here on the Python bridge.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import xybrid  # noqa: F401 -- the scenario subprocess imports it too

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[2] / "tools" / "scripts" / "observability_capture"


def run_scenario(out_dir, *scenario_args):
    completed = subprocess.run(
        [sys.executable, str(HARNESS / "run_scenario.py"), "--binding", "python",
         "--out-dir", str(out_dir), "--grace", "0.5", "--",
         sys.executable, str(HERE / "observability_scenario.py"), *scenario_args],
        capture_output=True, text=True, timeout=600,
    )
    report_path = out_dir / "report.json"
    assert report_path.exists(), completed.stdout + completed.stderr
    return completed, json.loads(report_path.read_text(encoding="utf-8"))


def test_reference_scenario_passes_every_check(tmp_path):
    completed, report = run_scenario(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    failed = [check for check in report["checks"] if not check["ok"]]
    assert report["ok"] and not failed, failed
    assert report["capture"]["spans"] >= 7


def test_without_the_omit_rule_the_omitted_content_is_exported(tmp_path):
    # The positive control for content.omit. A rule-omitted value leaves no
    # trace on the wire (CONTRACT.md), so the passing run above cannot tell the
    # rule apart from content that was never recorded. Without the rule the
    # same scenario must export the system instructions and fail exactly the
    # two checks that look for them: the scenario does record them, so their
    # absence above is the rule's doing.
    completed, report = run_scenario(tmp_path, "--without-omit-rule")
    assert completed.returncode == 1, completed.stdout + completed.stderr
    failed = {check["name"]: check["failures"] for check in report["checks"] if not check["ok"]}
    assert sorted(failed) == ["content.omit", "inference.content"], failed
    marker = json.loads((HARNESS / "scenario.json").read_text(encoding="utf-8"))["omit_marker"]
    assert any(marker in failure for failure in failed["content.omit"]), failed
