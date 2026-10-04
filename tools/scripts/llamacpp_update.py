#!/usr/bin/env python3
"""Discover official llama.cpp releases and prepare reviewed xybrid updates.

The read-only ``discover`` and ``release-plan`` commands work locally with gh.
Mutating commands run in clean CI checkouts. Neither command merges a PR,
creates a tag, or publishes an SDK; release branches use release-prep.yml.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "ggml-org/llama.cpp"
TRACKING = Path(".github/llamacpp-version.json")
SYS = Path("crates/llama-cpp-sys")
VENDOR = Path("vendor/llama-cpp")
UPDATE_BRANCH_PREFIX = "chore/llamacpp-"
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
SHA = re.compile(r"[0-9a-f]{40}")
PIN = re.compile(r'const LLAMA_CPP_COMMIT: &str = "([0-9a-f]{40})";')
REQUIRED_CHECKS = {
    "CI Success",
    "Bazel graph + RBE targets",
    "llama.cpp stable validation",
    "Test whisper.cpp (integration, real model)",
    "Test policy routing (real local model, fake DeepSeek)",
    "Provenance and fixture digests",
}
AUTOMATION_CHECKS = {"Discover llama.cpp stable", "Prepare llama.cpp release"}
NATIVE_TARGETS = {
    "aarch64-linux-android", "armv7-linux-androideabi", "x86_64-linux-android",
    "aarch64-apple-ios", "aarch64-apple-ios-sim", "aarch64-apple-darwin",
    "x86_64-apple-darwin", "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
}


class UpdateError(Exception):
    """An upstream response or local pin is inconsistent."""


def run(args: list[str], root: Path = ROOT, **kwargs: Any) -> str:
    return subprocess.check_output(args, cwd=root, text=True, **kwargs).strip()


def api(endpoint: str) -> Any:
    return json.loads(run(["gh", "api", endpoint]))


def version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise UpdateError(f"Expected a stable X.Y.Z version, got {value!r}")
    return tuple(int(part) for part in value.split("."))


def sdk_version(root: Path) -> str:
    text = (root / "Cargo.toml").read_text()
    match = re.search(r'\[workspace.package\]\s*\nversion = "([^"]+)"', text)
    if not match:
        raise UpdateError("Cannot read the workspace version")
    return match[1]


def pin(root: Path) -> dict[str, Any]:
    state = json.loads((root / TRACKING).read_text())
    matches = PIN.findall((root / SYS / "build.rs").read_text())
    gitlink = run(["git", "ls-files", "--stage", "--", str(VENDOR)], root).split()
    if (len(matches) != 1 or len(gitlink) < 3 or gitlink[0] != "160000"
            or matches[0] != gitlink[1] or state.get("commit") != matches[0]):
        raise UpdateError("llama.cpp tracking, build.rs and submodule pins disagree")
    if state.get("tag") is not None:
        tag_version(state["tag"])
        base = state.get("sdk_base_version")
        if not isinstance(base, str):
            raise UpdateError("Missing SDK baseline for the recorded upstream release")
        version(base.split("-", 1)[0].split("+", 1)[0])
        if state.get("update_kind") not in {"bootstrap", "patch", "minor", "major"}:
            raise UpdateError("Invalid recorded upstream update kind")
    released = state.get("sdk_release")
    if released is not None:
        if not isinstance(released, dict) or not isinstance(released.get("version"), str):
            raise UpdateError("Invalid recorded SDK release")
        version(released["version"].split("-", 1)[0].split("+", 1)[0])
        if not isinstance(released.get("commit"), str) or not SHA.fullmatch(released["commit"]):
            raise UpdateError("SDK release must record the upstream commit it includes")
    return state


def tag_version(tag: str) -> tuple[int, int, int]:
    if not isinstance(tag, str) or not tag.startswith("v"):
        raise UpdateError(f"Not an official llama.cpp release tag: {tag!r}")
    return version(tag[1:])


def update_kind(old_tag: str | None, new_tag: str) -> str:
    new = tag_version(new_tag)
    if old_tag is None:
        return "bootstrap"
    old = tag_version(old_tag)
    if new <= old:
        raise UpdateError(f"Refusing upstream version downgrade: {old_tag} -> {new_tag}")
    return next(name for index, name in enumerate(("major", "minor", "patch"))
                if new[index] != old[index])


def resolve_tag(tag: str, fetch: Callable[[str], Any] = api) -> str:
    obj = fetch(f"repos/{UPSTREAM}/git/ref/tags/{tag}")["object"]
    for _ in range(8):
        sha = obj.get("sha", "")
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise UpdateError("Upstream tag does not resolve to a full commit SHA")
        if obj.get("type") == "commit":
            return sha
        if obj.get("type") != "tag":
            break
        obj = fetch(f"repos/{UPSTREAM}/git/tags/{sha}")["object"]
    raise UpdateError("Cannot peel upstream release tag to a commit")


def discover(root: Path, repository: str, fetch: Callable[[str], Any] = api) -> dict:
    state = pin(root)
    release = fetch(f"repos/{UPSTREAM}/releases/latest")
    tag = release.get("tag_name")
    tag_version(tag)
    if release.get("draft") is not False or release.get("prerelease") is not False:
        raise UpdateError("Expected a published, non-prerelease llama.cpp release")
    commit = resolve_tag(tag, fetch)
    result = {
        "update": False, "tag": tag, "commit": commit,
        "branch": f"{UPDATE_BRANCH_PREFIX}{tag}",
        "previous_commit": state["commit"], "previous_tag": state["tag"],
        "url": f"https://github.com/{UPSTREAM}/releases/tag/{tag}",
        "sdk_base_version": sdk_version(root),
    }
    if commit == state["commit"]:
        return dict(result, reason="Already pinned to the latest official release")
    result["update_kind"] = update_kind(state["tag"], tag)
    # A workspace can intentionally be ahead of the latest stable tag. Never
    # move it backwards or onto a divergent maintenance branch automatically.
    comparison = fetch(f"repos/{UPSTREAM}/compare/{state['commit']}...{commit}")
    if comparison.get("status") != "ahead":
        return dict(result, reason="Latest stable does not advance the committed pin")
    pulls = run(["gh", "pr", "list", "--repo", repository, "--base", "master",
                 "--state", "open", "--limit", "1000", "--json", "url,headRefName"], root)
    pending = [pr for pr in json.loads(pulls)
               if pr["headRefName"].startswith(f"{UPDATE_BRANCH_PREFIX}v")]
    if pending:
        return dict(result, reason=f"Update already under review: {pending[0]['url']}")
    # Closed PRs remain an explicit rejection until their bot branch is deleted.
    if run(["git", "ls-remote", "--heads", "origin", f"refs/heads/{result['branch']}"], root):
        return dict(result, reason="Bot branch exists; review it or delete it to retry")
    return dict(result, update=True, reason="New official release available")


def add_changelog(root: Path, candidate: dict) -> None:
    path = root / "CHANGELOG.md"
    text = path.read_text()
    marker = "## [Unreleased]\n"
    if text.count(marker) != 1:
        raise UpdateError("Cannot find the Unreleased changelog section")
    line = (f"- **llama.cpp:** update to [{candidate['tag']}]({candidate['url']}) "
            f"(commit `{candidate['commit']}`).\n")
    start = text.index(marker) + len(marker)
    end = text.find("\n## ", start)
    end = len(text) if end == -1 else end
    section = text[start:end]
    if "### Changed\n" in section:
        section = section.replace("### Changed\n", f"### Changed\n\n{line}", 1)
    else:
        section = f"\n### Changed\n\n{line}" + section
    path.write_text(text[:start] + section + text[end:])


def apply(root: Path, candidate: dict) -> None:
    state = pin(root)
    tag_version(candidate["tag"])
    if (candidate.get("update") is not True
            or candidate["previous_commit"] != state["commit"]
            or candidate["sdk_base_version"] != sdk_version(root)
            or not SHA.fullmatch(candidate["commit"])):
        raise UpdateError("Candidate no longer matches this checkout")
    run(["git", "submodule", "update", "--init", "--depth", "1", "--", str(VENDOR)], root)
    run(["git", "fetch", "--depth", "1", "origin", candidate["commit"]], root / VENDOR)
    run(["git", "checkout", "--detach", candidate["commit"]], root / VENDOR)
    if run(["git", "rev-parse", "HEAD"], root / VENDOR) != candidate["commit"]:
        raise UpdateError("Submodule checkout did not resolve to the candidate commit")
    path = root / SYS / "build.rs"
    path.write_text(path.read_text().replace(state["commit"], candidate["commit"]))
    tracking = {key: candidate[key] for key in ("tag", "commit", "sdk_base_version", "update_kind")}
    tracking["sdk_release"] = None
    (root / TRACKING).write_text(json.dumps(tracking, indent=2) + "\n")
    add_changelog(root, candidate)
    run(["git", "add", "--", str(VENDOR)], root)
    pin(root)


def bindings(root: Path, check: bool) -> None:
    # Build through the real Cargo path: it owns the bindgen configuration and
    # also catches C++ wrapper breakage. No duplicate bindgen flags to maintain.
    env = dict(os.environ, XYBRID_NATIVES_FORCE_SOURCE="1")
    env.pop("XYBRID_NATIVES_PREBUILT_DIR", None)
    output = run(["cargo", "check", "--locked", "-p", "xybrid-llama-sys",
                  "--features", "bindings,vision", "--message-format=json"], root, env=env)
    generated = []
    for line in output.splitlines():
        message = json.loads(line)
        if (message.get("reason") == "build-script-executed"
                and "xybrid-llama-sys" in message.get("package_id", "")):
            generated.append(Path(message["out_dir"]) / "bindings.rs")
    if len(generated) != 1 or not generated[0].is_file():
        raise UpdateError("Cargo did not report the generated llama.cpp bindings")
    snapshot = root / SYS / "src/bindings.rs"
    # Match cargo fmt's formatting of the committed generator output.
    with tempfile.TemporaryDirectory() as directory:
        formatted = Path(directory) / "bindings.rs"
        shutil.copyfile(generated[0], formatted)
        run(["rustfmt", "--edition", "2021", str(formatted)], root)
        if check:
            if formatted.read_bytes() != snapshot.read_bytes():
                raise UpdateError("Committed llama.cpp bindings are stale; regenerate them")
        else:
            shutil.copyfile(formatted, snapshot)


def open_pr(root: Path, repository: str, candidate: dict, bindings_ok: bool) -> None:
    paths = [str(TRACKING), str(VENDOR), str(SYS / "build.rs"),
             str(SYS / "src/bindings.rs"), "CHANGELOG.md"]
    branch = f"{UPDATE_BRANCH_PREFIX}{candidate['tag']}"
    tag_version(candidate["tag"])
    if candidate.get("branch") != branch:
        raise UpdateError("Update branch does not match the discovered release tag")
    run(["git", "switch", "-c", branch], root)
    run(["git", "add", "--", *paths], root)
    run(["git", "commit", "-m", f"chore(llama): update to {candidate['tag']}"], root)
    run(["git", "push", "--set-upstream", "origin", branch], root)
    status = ("Bindings regenerated and the C++ wrapper compiled."
              if bindings_ok else "Binding generation or C++ compilation failed. Fix the workflow failure before marking this draft ready.")
    body = (
        f"Update llama.cpp from `{candidate['previous_commit']}` to "
        f"[{candidate['tag']}]({candidate['url']}) (`{candidate['commit']}`).\n\n"
        f"Upstream update: {candidate['update_kind']}. {status}\n\n"
        "Validation includes the Cargo and Bazel builds, real grammar and vision inference, "
        "SDK streaming and policy routing, and Whisper against the shared ggml. "
        "The choice-conformance provenance also pins the old converter; update its "
        "recipe and regenerate artifacts deliberately if that gate reports drift.\n\n"
        "After this PR and the generated native-manifest PR merge and validation passes, "
        "the daily automation can prepare the next 0.x minor release. "
        "Major upgrades and SDK versions >=1.0 require an explicit release version. "
        "The existing release PR remains the publishing approval point.\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "body.md"
        path.write_text(body)
        args = ["gh", "pr", "create", "--repo", repository, "--base", "master",
                "--head", branch, "--title", f"chore(llama): update to {candidate['tag']}",
                "--body-file", str(path)]
        if not bindings_ok:
            args.append("--draft")
        print(run(args, root))


def manifest_problems(root: Path, commit: str) -> list[str]:
    fields = {}
    slices = set()
    for line in (root / SYS / "natives-manifest.txt").read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "slice":
            if len(parts) < 4 or not re.fullmatch(r"sha256:[0-9a-f]{64}", parts[3]):
                raise UpdateError("Invalid native-manifest slice")
            slices.add((parts[1], parts[2]))
        else:
            fields[parts[0]] = parts[1]
    expected = {"version": "1", "llama_commit": commit}
    for field, filename in (("wrapper_cpp", "wrapper.cpp"), ("wrapper_h", "wrapper.h"),
                            ("build_rs", "build.rs")):
        expected[field] = hashlib.sha256((root / SYS / filename).read_bytes()).hexdigest()
    problems = [f"Native manifest has stale {key}" for key, value in expected.items()
                if fields.get(key) != value]
    missing = {(target, feature) for target in NATIVE_TARGETS for feature in ("base", "vision")} - slices
    if missing:
        problems.append("Native manifest missing: " + ", ".join(f"{t}/{f}" for t, f in sorted(missing)))
    return problems


def check_problems(checks: list[dict]) -> list[str]:
    latest = {}
    for check in checks:
        name = check["name"]
        if name not in latest or check["id"] > latest[name]["id"]:
            latest[name] = check
    problems = [f"Waiting for successful check: {name}" for name in sorted(REQUIRED_CHECKS)
                if name not in latest or latest[name].get("status") != "completed"
                or latest[name].get("conclusion") != "success"]
    for name, check in latest.items():
        if name in REQUIRED_CHECKS or name in AUTOMATION_CHECKS:
            continue
        if (check.get("status") != "completed"
                or check.get("conclusion") not in {"success", "skipped", "neutral"}):
            problems.append(f"Check has not passed: {name}")
    return problems


def release_plan(root: Path, repository: str, selected_version: str | None,
                 fetch: Callable[[str], Any] = api) -> dict:
    state = pin(root)
    current = sdk_version(root)
    head = run(["git", "rev-parse", "HEAD"], root)
    result = {"ready": False, "head": head, "tag": state["tag"], "sdk_base_version": current}
    released = state.get("sdk_release")
    if state["tag"] is None or (released is not None and released["commit"] == state["commit"]):
        return dict(result, reasons=["No unreleased automated llama.cpp update"])
    if not VERSION.fullmatch(current):
        return dict(result, reasons=["Finish the current SDK prerelease before preparing another release"])
    base = version(current)
    if selected_version:
        target = version(selected_version)
        if target <= base:
            raise UpdateError("Release version must advance the workspace version")
        if base[0] == 0 and target[0] == 0 and target[1] == base[1]:
            raise UpdateError("This POC reserves automated 0.x releases for minor updates")
        result["version"] = selected_version
    elif base[0] != 0 or state["update_kind"] == "major":
        return dict(result, reasons=["Choose sdk_version explicitly for major upstream updates or SDK >=1.0"])
    else:
        result["version"] = f"0.{base[1] + 1}.0"
    result["branch"] = f"release/v{result['version']}"
    problems = manifest_problems(root, state["commit"])
    checks = []
    page = 1
    while True:
        payload = fetch(f"repos/{repository}/commits/{head}/check-runs?filter=latest&per_page=100&page={page}")
        checks.extend(payload["check_runs"])
        if len(checks) >= payload["total_count"]:
            break
        if not payload["check_runs"]:
            raise UpdateError("Check-run pagination ended before all checks were read")
        page += 1
    problems.extend(check_problems(checks))
    pulls = json.loads(run(["gh", "pr", "list", "--repo", repository, "--base", "master",
                           "--state", "open", "--limit", "1000", "--json", "headRefName"], root))
    if any(pr["headRefName"].startswith("release/v") for pr in pulls):
        problems.append("A release PR is already open")
    refs = run(["git", "ls-remote", "origin", "refs/heads/release/v*",
                f"refs/tags/v{result['version']}"], root)
    for line in refs.splitlines():
        ref = line.split()[1]
        if ref == f"refs/tags/v{result['version']}":
            problems.append("The target release tag already exists")
        elif ref.startswith("refs/heads/release/v"):
            # An in-flight release may not have opened its PR yet. Ignore
            # historical branches, but do not start a concurrent release cut.
            ref_version = ref.removeprefix("refs/heads/release/v").split("-", 1)[0]
            if VERSION.fullmatch(ref_version) and version(ref_version) > base:
                problems.append(f"A release branch already exists: {ref}")
    return dict(result, ready=not problems, reasons=problems)


def bump_internal_pins(root: Path, target: str) -> None:
    # version-sync updates workspace/package versions but leaves internal
    # path-dependency constraints behind. Update only deps on workspace members.
    workspace = (root / "Cargo.toml").read_text()
    members_section = re.search(r"members\s*=\s*\[(.*?)\]", workspace, re.S)
    if not members_section:
        raise UpdateError("Cannot read workspace members")
    members = [root / member / "Cargo.toml" for member in re.findall(r'"([^"]+)"', members_section[1])]
    names = set()
    for manifest in members:
        match = re.search(r'^name\s*=\s*"([^"]+)"', manifest.read_text(), re.M)
        if not match:
            raise UpdateError(f"Cannot read package name: {manifest}")
        names.add(match[1])
    paths = run(["git", "ls-files", "--", "Cargo.toml", "**/Cargo.toml"], root).splitlines()
    for path in paths:
        manifest = root / path
        text = manifest.read_text()
        def replace(match: re.Match) -> str:
            name, entry = match[1], match[2]
            package = re.search(r'\bpackage\s*=\s*"([^"]+)"', entry)
            if (package[1] if package else name) in names and re.search(r'\bpath\s*=', entry):
                entry = re.sub(r'\bversion\s*=\s*"[^"]+"', f'version = "{target}"', entry)
            return f"{name}{entry}"
        manifest.write_text(re.sub(r'^([\w-]+)(\s*=\s*\{[^\n]*\})', replace, text, flags=re.M))


def promote_changelogs(root: Path, target: str) -> None:
    date = datetime.date.today().isoformat()
    entries = [
        ("CHANGELOG.md", "## [Unreleased]\n", f"## [Unreleased]\n\n## [{target}] - {date}\n"),
        ("bindings/flutter/CHANGELOG.md", "## Unreleased\n", f"## Unreleased\n\n## {target}\n"),
    ]
    for filename, marker, replacement in entries:
        path = root / filename
        text = path.read_text()
        if text.count(marker) != 1:
            raise UpdateError(f"Cannot find Unreleased in {filename}")
        path.write_text(text.replace(marker, replacement, 1))


def record_release(root: Path, target: str) -> None:
    """Stamp the exact upstream commit included by a prepared release branch."""
    if (sdk_version(root) != target
            or run(["git", "branch", "--show-current"], root) != f"release/v{target}"):
        raise UpdateError("Recording a release requires its matching version and release branch")
    state = pin(root)
    if state["tag"] is None:
        return
    released = {"version": target, "commit": state["commit"]}
    if state.get("sdk_release") != released:
        state["sdk_release"] = released
        (root / TRACKING).write_text(json.dumps(state, indent=2) + "\n")


def prepare_release(root: Path, plan: dict) -> None:
    if plan.get("ready") is not True or run(["git", "rev-parse", "HEAD"], root) != plan["head"]:
        raise UpdateError("Release plan is not ready or the checkout changed")
    if run(["git", "status", "--porcelain"], root):
        raise UpdateError("Release preparation requires a clean checkout")
    target = plan["version"]
    version(target)
    if plan["branch"] != f"release/v{target}":
        raise UpdateError("Release branch must match the chosen SDK version")
    run(["git", "switch", "-c", plan["branch"]], root)
    bump_internal_pins(root, target)
    run([str(root / "tools/scripts/version-sync.sh"), target], root)
    record_release(root, target)
    run([str(root / "bindings/apple/scripts/set-natives-mode.sh"), "--set-remote"], root)
    promote_changelogs(root, target)
    run([str(root / "tools/scripts/version-sync.sh"), "--check"], root)
    run(["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], root)
    run(["git", "diff", "--check"], root)
    # Long codegen/builds can outlive a merge onto master. Retry on the next
    # daily run rather than cut a release with an obsolete base.
    remote_head = run(["git", "ls-remote", "origin", "refs/heads/master"], root).split()
    if not remote_head or remote_head[0] != plan["head"]:
        raise UpdateError("master advanced during release preparation; rerun with the new head")
    run(["git", "add", "--all"], root)
    run(["git", "commit", "-m", f"bump: {target} (llama.cpp {plan['tag']})"], root)
    run(["git", "push", "--set-upstream", "origin", plan["branch"]], root)


def emit(payload: dict, destination: Path | None) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    if destination:
        destination.write_text(text)
    print(text, end="")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as handle:
            for key in ("update", "ready"):
                if key in payload:
                    handle.write(f"{key}={str(payload[key]).lower()}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as handle:
            handle.write(f"```json\n{text}```\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", "xybrid-ai/xybrid"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("discover", "release-plan"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path)
        if name == "release-plan":
            command.add_argument("--sdk-version")
    for name in ("apply", "open-pr", "prepare-release"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
        if name == "open-pr":
            command.add_argument("--bindings-ok", action="store_true")
    command = commands.add_parser("bindings")
    command.add_argument("--check", action="store_true")
    command = commands.add_parser("record-release")
    command.add_argument("version")
    commands.add_parser("check-pin")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "discover":
            emit(discover(root, args.repository), args.output)
        elif args.command == "release-plan":
            emit(release_plan(root, args.repository, args.sdk_version), args.output)
        elif args.command == "bindings":
            bindings(root, args.check)
        elif args.command == "check-pin":
            emit(pin(root), None)
        elif args.command == "record-release":
            record_release(root, args.version)
        else:
            payload = json.loads(args.input.read_text())
            if args.command == "apply":
                apply(root, payload)
            elif args.command == "open-pr":
                open_pr(root, args.repository, payload, args.bindings_ok)
            else:
                prepare_release(root, payload)
    except (UpdateError, subprocess.CalledProcessError, OSError, ValueError, KeyError) as exc:
        print(f"llamacpp-update: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
