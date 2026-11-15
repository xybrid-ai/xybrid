#!/usr/bin/env python3
"""Select CI jobs and verify the single required CI Success check.

This file is the path registry: the one place that decides which ci.yml jobs a
change needs. `changes` prints one `key=true|false` line per flag for the
`tooling-changes` job's outputs; `check` reads `toJSON(needs)` in CI Success
and fails unless every required job passed or was skipped for a recorded
reason.

Two kinds of flag:

  * PATH_FILTERS: jobs that build from the workspace. A docs-only change
    (`full_ci` false) skips them whatever its paths.
  * LANE_PATHS: lanes (a backend with real models, ...) that run exactly when
    one of their paths changes, as the per-lane workflows' `paths:` filters
    did, in GitHub's own filter syntax.

A new backend or real-model suite is one ci.yml job gated on its flag
(`if: needs.tooling-changes.outputs.<flag> == 'true'`), one LANE_PATHS row,
its OPTIONAL_JOBS entry, its line in tooling-changes' outputs, and (once it is
reliable) an entry in CI Success's needs. test_ci_gate.py fails until they
agree. BAZEL_PATHS is bazel.yml's gate, read with `ci_gate.py bazel`.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PATH_FILTERS = {
    # The ONNX Runtime staging legs download and load the archives the script
    # pins; nothing else in the tree changes what they do.
    "ort_staging": r"^tools/scripts/stage_unity_desktop_ort\.py$",
    # bindings-drift regenerates the committed Kotlin/JNI, Swift/C header and
    # Unity C# bindings and compiles the Unity C# (with its hand-written Api
    # layer). Each is generated from the whole bolt crate (its exports and their
    # doc comments) over the facade types it wraps, so a change to either runs it.
    "bindings_drift": r"^(crates/xybrid-bolt/|crates/xybrid-ffi-facade/|bindings/(kotlin|apple)/|bindings/unity/Runtime/(Bolt/|BoltSupplement/|Api/)|tools/unity-bolt-compile-check/|tools/scripts/gen_(kotlin_bolt|unity_bolt_csharp)\.py$)",
    # The suite exports manual observations to a local receiver, so it also
    # covers the facade those handles wrap, and runs the shared observability
    # scenario (tools/scripts/observability_capture/) end to end, so it also
    # covers the SDK and the observability crate that export it.
    "python_sdk": r"^(bindings/python/|tools/scripts/(gen_python_bolt\.py|build-python-bolt\.sh)$|tools/scripts/observability_capture/|crates/xybrid-bolt/|crates/xybrid-ffi-facade/|crates/xybrid-sdk/|crates/xybrid-observability/)",
}

# Lane -> paths, in GitHub's `on.<event>.paths` syntax: `*` stays within a
# directory, `**` crosses directories, and the last pattern a path matches
# decides, so a later `!pattern` excludes and a later pattern re-includes. A
# submodule bump changes one path, the gitlink itself (`vendor/llama-cpp`), which
# `vendor/llama-cpp/**` does not match.
LANE_PATHS: Dict[str, List[str]] = {
    # Candle is the pure-Rust framework behind xybrid-core's `candle` feature.
    "candle": [
        "crates/xybrid-core/src/runtime_adapter/candle/**",
        "crates/xybrid-core/src/audio/**",
        "crates/xybrid-core/Cargo.toml",
        # The executor holds the candle-conditional dispatch arms (the
        # SafeTensors -> "candle" runtime mapping and its
        # #[cfg(not(feature = "candle"))] counterpart). Without this path the
        # Candle jobs did not run on either commit that retired Candle from the
        # platform presets.
        "crates/xybrid-core/src/execution/**",
        # The presets themselves live here; a change that drops candle from one
        # must be seen by the job that still builds with it.
        "crates/xybrid-sdk/Cargo.toml",
        "integration-tests/**",
        "Cargo.lock",
        "!**.md",
    ],
    "whispercpp": [
        "crates/whisper-cpp-sys/**",
        "crates/xybrid-whisper/**",
        "crates/llama-cpp-sys/**",
        "crates/xybrid-llama/**",
        "crates/xybrid-core/src/runtime_adapter/whisper_cpp/**",
        "crates/xybrid-core/src/audio/**",
        "crates/xybrid-core/src/execution/**",
        "crates/xybrid-core/Cargo.toml",
        "integration-tests/**",
        "vendor/llama-cpp",
        "vendor/whisper-cpp",
        "Cargo.lock",
        "!**.md",
    ],
    "policy_routing": [
        "crates/xybrid-core/**",
        "crates/xybrid-cli/**",
        "crates/xybrid-sdk/**",
        "crates/llama-cpp-sys/**",
        "crates/xybrid-llama/**",
        "integration-tests/**",
        "vendor/llama-cpp",
        "Cargo.toml",
        "Cargo.lock",
        ".cargo/**",
        "!**.md",
    ],
    "llamacpp": [
        "vendor/llama-cpp",
        "crates/llama-cpp-sys/**",
        "crates/xybrid-llama/**",
        "crates/xybrid-core/src/runtime_adapter/llama_cpp/**",
        "crates/xybrid-core/src/runtime_adapter/streaming_postprocess.rs",
        ".github/llamacpp-version.json",
        "tools/scripts/llamacpp_update.py",
        "integration-tests/tests/llamacpp_release.rs",
        "crates/xybrid-core/tests/llamacpp_stream_text.rs",
    ],
    # The SDK lanes (lane files ci.yml calls). Each lists its own file, which
    # holds the jobs. Markdown inside these paths never affects a build.
    "android": [
        "bindings/kotlin/**",
        "crates/xybrid-bolt/**",
        "crates/xybrid-ffi-facade/**",
        "crates/xybrid-core/**",
        "crates/xybrid-sdk/**",
        "crates/xybrid-observability/**",
        "tools/scripts/android-dlopen-gate.sh",
        "tools/scripts/android-dlopen-probe.c",
        "tools/scripts/verify_android_jni.py",
        "tools/scripts/verify_android_api_level.py",
        # The shared observability scenario's receiver and checker.
        "tools/scripts/observability_capture/**",
        "xtask/**",
        # llama.cpp's Android build config and sources (see //:llama_android_dl).
        "BUILD.bazel",
        "bazel/ggml/**",
        # The Android NDK toolchains: @androidndk (API 28) and the AAR arm64
        # slice's @androidndk_api29 (see //bazel/toolchains/android), the patch
        # that generates both, and the .bazelrc NDK download pins.
        "MODULE.bazel",
        "bazel/toolchains/android/**",
        "rules_android_ndk.patch",
        ".bazelrc",
        "crates/llama-cpp-sys/**",
        "vendor/llama-cpp",
        "vendor/ort-android/**",
        "Cargo.lock",
        "Cargo.toml",
        ".github/workflows/build-android.yml",
        "!**.md",
    ],
    "apple": [
        "bindings/apple/**",
        "crates/xybrid-bolt/**",
        "crates/xybrid-ffi-facade/**",
        "crates/xybrid-core/**",
        "crates/xybrid-sdk/**",
        "crates/xybrid-observability/**",
        "crates/xybrid-llama/**",
        "crates/llama-cpp-sys/**",
        # The Bazel graph that assembles the xcframework: sim triple + ort-sys
        # annotation, the llama iOS arms, --config=ios, the fetched sim-ORT
        # rule, and the committed device ORT slice.
        "MODULE.bazel",
        "BUILD.bazel",
        ".bazelrc",
        "bazel/**",
        "vendor/ort-ios/**",
        "Cargo.lock",
        "Cargo.toml",
        # The shared observability scenario the runtime job runs and checks.
        "tools/scripts/observability_capture/**",
        ".github/workflows/build-apple.yml",
        "!**.md",
    ],
    "flutter": [
        "bindings/flutter/**",
        "crates/xybrid-ffi-facade/**",
        # The Flutter crate links the SDK and core through the facade, and FRB
        # mirrors types it reaches there: a change in them can break the
        # native build or leave the committed glue stale.
        "crates/xybrid-sdk/**",
        "crates/xybrid-core/**",
        "crates/xybrid-observability/**",
        "xtask/**",
        # The shared observability scenario's receiver and checker.
        "tools/scripts/observability_capture/**",
        ".github/workflows/build-flutter.yml",
        "!**.md",
    ],
    "react_native": [
        "bindings/react-native/**",
        "bindings/apple/**",
        "bindings/kotlin/**",
        "crates/xybrid-bolt/**",
        "crates/xybrid-ffi-facade/**",
        ".github/workflows/build-react-native.yml",
        "!**.md",
    ],
    "unity_editor": [
        "bindings/unity/**",
        "crates/xybrid-bolt/**",
        "crates/xybrid-ffi-facade/**",
        # The SDK and the observability crate it links: the observability
        # runtime scenario exercises their export, content rules and
        # lifecycle through the C# API. xybrid-core is left to the Apple,
        # Android and Flutter lanes, which run the same scenario on its changes.
        "crates/xybrid-sdk/**",
        "crates/xybrid-observability/**",
        "tools/unity-runtime-smoke/**",
        "tools/scripts/stage_unity_native.py",
        "tools/scripts/stage_unity_desktop_ort.py",
        "tools/scripts/bazel-output.sh",
        "tools/scripts/write-buildbuddy-rc.sh",
        "tools/scripts/observability_capture/**",
        ".github/workflows/unity-editor.yml",
        "!**.md",
    ],
}

# bazel.yml's required job (`Bazel graph + RBE targets`) and its Vulkan lane
# run on a pull request when one of these paths changes (`ci_gate.py bazel`),
# and bazel.yml's push trigger lists the same paths (test_ci_gate.py keeps the
# two equal). Not selected by PIPELINE_PATHS: a ci.yml edit is no Bazel input.
BAZEL_PATHS = [
    "MODULE.bazel",
    ".bazelrc",
    ".bazelversion",
    ".bazelignore",
    "BUILD.bazel",
    "**/BUILD.bazel",
    # Starlark that BUILD files load: toolchain logic lives here, and
    # `**/BUILD.bazel` alone would let a .bzl-only change slip through.
    "**/*.bzl",
    "bazel/**",
    "tools/scripts/**",
    "rules_android_ndk.patch",
    "rules_foreign_cc.patch",
    "rules_rust.patch",
    "hermetic_llvm.patch",
    "emsdk.patch",
    "crates/**",
    "bindings/kotlin/**",
    "bindings/web/runtime/**",
    "macros/**",
    "vendor/llama-cpp",
    "vendor/ort-macos/**",
    "vendor/ort-android/**",
    "Cargo.toml",
    "Cargo.lock",
    "tools/unity-runtime-smoke/**",
    # Also built or read by the jobs: //:whisper's sources,
    # //integration-tests:all and the fixtures crate tests read,
    # //xtask:xtask_test, the Flutter cdylib, the Windows bolt smoke, the C#
    # the DLL audit reads its required exports from, and the Rust metadata
    # fixtures xybrid-core's inline tests read.
    "vendor/whisper-cpp",
    "integration-tests/**",
    "xtask/**",
    "bindings/flutter/rust/**",
    "tools/windows-smoke/**",
    "bindings/unity/Runtime/Bolt/XybridBolt.cs",
    "bindings/web/test/fixtures/rust-metadata-*.json",
    # The shared actions this workflow's jobs are built from.
    ".github/actions/**",
    ".github/workflows/bazel.yml",
    # Markdown inside any path above cannot change a Bazel build.
    "!**.md",
    "!**.mdx",
]

# Paths that select every flag and lane: the pipeline itself, the shared
# actions its jobs are built from, and this registry.
PIPELINE_PATHS = [".github/workflows/ci.yml", ".github/actions/**", "tools/scripts/ci_gate.py"]

# ci.yml job -> the flag that selects it. CI Success accepts a skip only when
# that flag is false. Every other job it needs runs on any change that is not
# docs-only (`full_ci`), except ALWAYS_JOBS, which run on every change.
OPTIONAL_JOBS = {
    "tooling-tests": "ort_staging",
    "bindings-drift": "bindings_drift",
    "python-sdk": "python_sdk",
    "test-candle": "candle",
    "test-whispercpp-integration": "whispercpp",
    "test-policy-routing": "policy_routing",
    "llamacpp-validate": "llamacpp",
    "android": "android",
    "apple": "apple",
    "flutter": "flutter",
    "react-native": "react_native",
    "unity-editor": "unity_editor",
}
ALWAYS_JOBS = {"tooling-changes", "repo-checks"}

# Jobs ci.yml runs that CI Success does not require: a failure shows on the
# pull request without blocking the merge. Every lane starts here, so moving it
# into ci.yml does not change merge policy; it joins CI Success's needs by
# leaving this set.
ADVISORY_JOBS = {
    "test-candle",
    "test-whispercpp-integration",
    "test-policy-routing",
    "llamacpp-validate",
    "android",
    "apple",
    "flutter",
    "react-native",
    "unity-editor",
}

_GLOB_SPECIAL = set("?+[]")


def glob_regex(pattern: str) -> "re.Pattern[str]":
    """GitHub's path-filter glob as a regex.

    `**/` is any directory prefix (none included), `**` anything, `*` anything
    but `/`. `?`, `+` and `[]` are refused: no registry row needs them, and
    their GitHub meanings are easy to get wrong.
    """
    if _GLOB_SPECIAL & set(pattern):
        raise ValueError(f"Unsupported path-filter character in {pattern!r}")
    out, index = [], 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            out.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif pattern[index] == "*":
            out.append("[^/]*")
            index += 1
        else:
            out.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(out))


def path_selected(patterns: Sequence[str], path: str) -> bool:
    """Whether `path` passes a GitHub `paths:` list: the last matching pattern decides."""
    selected = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        if glob_regex(pattern[1:] if negated else pattern).fullmatch(path):
            selected = not negated
    return selected


def any_selected(patterns: Sequence[str], paths: Iterable[str]) -> bool:
    return any(path_selected(patterns, path) for path in paths)


def docs_path(path: str) -> bool:
    return path.endswith(".md") or path.startswith(
        ("docs/", "examples/", ".github/ISSUE_TEMPLATE/")
    ) or path == "LICENSE"


def changed_paths(event: str, payload: object) -> Tuple[List[str], bool]:
    """Every changed path (a rename counts both names) and whether the list may be truncated."""
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
    return paths, not files or len(files) >= limit


def classify(event: str, payload: object) -> Dict[str, str]:
    paths, uncertain = changed_paths(event, payload)
    full_ci = uncertain or not all(docs_path(path) for path in paths)
    # A README beside the pipeline is still documentation.
    every = full_ci and (uncertain or any_selected(PIPELINE_PATHS, paths))
    selected = {"full_ci": full_ci}
    selected.update({
        key: every or (full_ci and any(re.search(pattern, path) for path in paths))
        for key, pattern in PATH_FILTERS.items()
    })
    selected.update({lane: every or any_selected(patterns, paths) for lane, patterns in LANE_PATHS.items()})
    return {key: str(value).lower() for key, value in selected.items()}


def bazel(event: str, payload: object) -> Dict[str, str]:
    """bazel.yml's gate: whether a Bazel input changed."""
    paths, uncertain = changed_paths(event, payload)
    return {"relevant": str(uncertain or any_selected(BAZEL_PATHS, paths)).lower()}


def dispatch(lanes: str, ref: str = "") -> Dict[str, str]:
    """A manual run: every job, and the named lanes (all of them when none are named).

    On master `lanes` is ignored: a skipped lane reports a check run under the
    same name as the push run's, and llamacpp_update.check_problems() reads the
    newest run per name on master's HEAD, so the skip would hide the success.
    Failing instead would skip every job and do the same to every check.
    """
    named = [lane for lane in lanes.replace(",", " ").split() if lane]
    if named and ref == "refs/heads/master":
        print(
            "CI gate: lanes are ignored on master, which runs every lane: skipped lanes would replace "
            "the checks the llama.cpp release reads on HEAD. Re-run a push run's failed jobs instead.",
            file=sys.stderr,
        )
        named = []
    unknown = sorted(set(named) - set(LANE_PATHS))
    if unknown:
        raise ValueError(f"Unknown lane(s) {', '.join(unknown)}; choose from {', '.join(sorted(LANE_PATHS))}")
    selected = {"full_ci": True}
    selected.update({key: True for key in PATH_FILTERS})
    selected.update({lane: not named or lane in named for lane in LANE_PATHS})
    return {key: str(value).lower() for key, value in selected.items()}


def check_jobs(needs: dict) -> None:
    detector = needs.get("tooling-changes", {})
    if detector.get("result") != "success":
        raise ValueError("CI change detection did not succeed")
    outputs = detector.get("outputs", {})
    for key in ("full_ci", *PATH_FILTERS, *LANE_PATHS):
        if outputs.get(key) not in ("true", "false"):
            raise ValueError(f"Missing or invalid change-detection output: {key}")

    for name, job in needs.items():
        result = job.get("result")
        if result == "success":
            continue
        if result == "skipped" and name not in ALWAYS_JOBS:
            if outputs[OPTIONAL_JOBS.get(name, "full_ci")] == "false":
                continue
        raise ValueError(f"Required CI job {name} finished with {result!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    changes = commands.add_parser("changes", help="classify a push or pull request's changed files")
    changes.add_argument("event", choices=("push", "pull_request"))
    changes.add_argument("files", type=Path)
    gate = commands.add_parser("bazel", help="whether a push or pull request changed a Bazel input")
    gate.add_argument("event", choices=("push", "pull_request"))
    gate.add_argument("files", type=Path)
    manual = commands.add_parser("dispatch", help="select jobs for a workflow_dispatch run")
    manual.add_argument("lanes", nargs="?", default="", help="comma-separated lanes; empty selects every lane")
    manual.add_argument("--ref", default="", help="the run's github.ref")
    commands.add_parser("check", help="verify CI Success's needs (JSON on stdin)")
    args = parser.parse_args()
    try:
        if args.command == "changes":
            outputs = classify(args.event, json.loads(args.files.read_text()))
        elif args.command == "bazel":
            outputs = bazel(args.event, json.loads(args.files.read_text()))
        elif args.command == "dispatch":
            outputs = dispatch(args.lanes, args.ref)
        else:
            check_jobs(json.load(sys.stdin))
            print("All required CI jobs passed or were correctly skipped.")
            return 0
        for key, value in outputs.items():
            print(f"{key}={value}")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(f"CI gate: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
