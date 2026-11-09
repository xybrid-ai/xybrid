#!/usr/bin/env python3
"""Select CI jobs and verify the single required CI Success check."""

import argparse
import json
import re
import sys
from pathlib import Path


PATH_FILTERS = {
    # tools/scripts/tests also covers the zzz delivery contract, the Flutter
    # platform floors, JS release scripts/manifests and the workflows its hygiene tests read.
    # test_generation_gates.py reads build-flutter.yml and the FRB wrapper, and
    # globs bindings/*/scripts/gen* for generators that need a drift gate.
    # test_observability_capture.py reads each binding's observability scenario.
    "relevant": r"^(tools/scripts/|bindings/(python/tests/observability_scenario\.py|apple/Tests/XybridTests/ObservabilityRuntimeTests\.swift|kotlin/consumer-smoke/app/src/androidTest/kotlin/ai/xybrid/smoke/ObservabilityScenario\.kt|unity/Tests/Editor/ObservabilityRuntimeScenarioTests\.cs|flutter/test/observability_scenario_test\.dart)$|bindings/(web|observability-js)/(scripts/|package\.json$|pnpm-(lock|workspace)\.yaml$)|bindings/observability-js/src/version\.ts$|crates/zzz-sys/(zzz_pull\.py|natives-manifest\.json)$|bindings/flutter/((ios|macos)/xybrid_flutter\.podspec|android/build\.gradle|generate-frb\.sh)$|bindings/[^/]+/scripts/gen[^/]*$|\.bazelrc$|\.github/workflows/(ci|build-flutter|release-prep|release-publish|test-zzz-android|test-zzz-engine|trusted-zzz-sdk)\.yml$)",
    # The committed C# is generated from the whole bolt crate (its exports and
    # their doc comments) and the facade types it wraps, so a change to either
    # must run the drift check and recompile the hand-written Api layer on top.
    "unity_bolt": r"^(bindings/unity/Runtime/(Bolt/|BoltSupplement/|Api/)|tools/unity-bolt-compile-check/|tools/scripts/gen_unity_bolt_csharp\.py$|crates/xybrid-bolt/|crates/xybrid-ffi-facade/|\.github/workflows/ci\.yml$)",
    # The suite exports manual observations to a local receiver, so it also
    # covers the facade those handles wrap, and runs the shared observability
    # scenario (tools/scripts/observability_capture/) end to end, so it also
    # covers the SDK and the observability crate that export it.
    "python_sdk": r"^(bindings/python/|tools/scripts/(gen_python_bolt\.py|build_python_bolt\.py|build-python-bolt\.sh)$|tools/scripts/observability_capture/|crates/xybrid-bolt/|crates/xybrid-ffi-facade/|crates/xybrid-sdk/|crates/xybrid-observability/|\.github/workflows/ci\.yml$)",
    "kotlin_jni": r"^(bindings/kotlin/|tools/scripts/gen_kotlin_bolt\.py$|crates/xybrid-bolt/|crates/xybrid-ffi-facade/|\.github/workflows/ci\.yml$)",
    "apple_bolt": r"^(bindings/apple/|crates/xybrid-bolt/|crates/xybrid-ffi-facade/|\.github/workflows/ci\.yml$)",
}
OPTIONAL_JOBS = {
    "tooling-tests": "relevant",
    "unity-bolt-compile": "unity_bolt",
    "unity-bolt-drift": "unity_bolt",
    "python-sdk": "python_sdk",
    "kotlin-jni-drift": "kotlin_jni",
    "apple-bolt-drift": "apple_bolt",
}


def docs_path(path: str) -> bool:
    return path.endswith(".md") or path.startswith(
        ("docs/", "examples/", ".github/ISSUE_TEMPLATE/")
    ) or path == "LICENSE"


def classify(event: str, payload: object) -> dict[str, str]:
    if event == "pull_request":
        # gh api --paginate --slurp returns an array of pages. GitHub caps
        # pull-request files at 3,000; run every check if it may be truncated.
        if not isinstance(payload, list) or not all(isinstance(page, list) for page in payload):
            raise ValueError("Expected pull-request file pages")
        files = [entry for page in payload for entry in page]
        limit = 3000
    elif event == "push":
        # The compare endpoint returns at most 300 changed files.
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
            raise ValueError("Expected a commit comparison with files")
        files = payload["files"]
        limit = 300
    else:
        raise ValueError(f"Unsupported event: {event}")

    paths = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Expected a changed-file record")
        for key in ("filename", "previous_filename"):
            if key == "previous_filename" and key not in entry:
                continue
            path = entry.get(key)
            if not isinstance(path, str) or not path:
                raise ValueError(f"Missing or invalid {key}")
            paths.append(path)

    uncertain = not files or len(files) >= limit
    full_ci = uncertain or not all(docs_path(path) for path in paths)
    selected = {"full_ci": full_ci}
    selected.update({
        key: full_ci and (uncertain or any(re.search(pattern, path) for path in paths))
        for key, pattern in PATH_FILTERS.items()
    })
    return {key: str(value).lower() for key, value in selected.items()}


def check_jobs(needs: dict) -> None:
    detector = needs.get("tooling-changes", {})
    if detector.get("result") != "success":
        raise ValueError("CI change detection did not succeed")
    outputs = detector.get("outputs", {})
    for key in ("full_ci", *PATH_FILTERS):
        if outputs.get(key) not in ("true", "false"):
            raise ValueError(f"Missing or invalid change-detection output: {key}")

    for name, job in needs.items():
        result = job.get("result")
        if result == "success":
            continue
        if result == "skipped" and name != "tooling-changes":
            if outputs["full_ci"] == "false":
                continue
            flag = OPTIONAL_JOBS.get(name)
            if flag is not None and outputs[flag] == "false":
                continue
        raise ValueError(f"Required CI job {name} finished with {result!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    changes = commands.add_parser("changes")
    changes.add_argument("event", choices=("push", "pull_request"))
    changes.add_argument("files", type=Path)
    commands.add_parser("check")
    args = parser.parse_args()
    try:
        if args.command == "changes":
            for key, value in classify(args.event, json.loads(args.files.read_text())).items():
                print(f"{key}={value}")
        else:
            check_jobs(json.load(sys.stdin))
            print("All required CI jobs passed or were correctly skipped.")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(f"CI gate: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
