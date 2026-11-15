#!/usr/bin/env python3
"""Generate the workflow inventory in .github/workflows/README.md.

The table between the BEGIN/END markers is derived from the workflow files, so
it cannot drift from them. The same pass enforces the conventions the README
describes, because a table can only be built from files that follow them:

  * `name:` is `Group: Subject`, Group is one of GROUPS, and names are unique.
  * The line directly above `name:` is `# Purpose: <one sentence>`.
  * A concurrency group never uses `github.workflow`, so renaming a workflow
    cannot silently change its concurrency group.
  * Every file is listed in WORKFLOW_FILES: CI grows by extending ci.yml (a
    job, or a lane file it calls), not by adding a workflow.
  * Only the files in PULL_REQUEST_FILES declare pull_request or
    pull_request_target. A lane file declares workflow_call (plus
    workflow_dispatch, plus schedule for a nightly), is called by a workflow,
    and never declares push or a pull-request event.
  * ci.yml and the lane files it calls hold exactly MACOS_JOB_BUDGET macOS jobs.

Standard library only: this runs on a bare CI runner and on a laptop without a
YAML package. It reads the small, regular subset of YAML these files use and
raises on anything else, rather than rendering a wrong row. When a workflow
needs a shape it does not read, extend the reader and its tests.

Usage:
    python3 tools/scripts/gen_workflow_index.py           # rewrite the table
    python3 tools/scripts/gen_workflow_index.py --check   # fail on drift or rule violations
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
README = WORKFLOWS_DIR / "README.md"

BEGIN = "<!-- BEGIN GENERATED: workflow inventory (tools/scripts/gen_workflow_index.py) -->"
END = "<!-- END GENERATED: workflow inventory -->"

# Display order of the inventory; the keys are the allowed `Group` prefixes.
GROUPS = {
    "CI": "The pull-request pipeline (ci.yml, bazel.yml) and suites that run only on master.",
    "SDK": "Per-platform SDK builds, wrapper tests and example apps.",
    "Artifacts": "Publish reusable native artifacts and their download manifests.",
    "Release": "Cut, validate, publish and announce a release.",
    "Security": "Static analysis and supply-chain checks.",
    "Maintenance": "Manual utilities, run on demand.",
    "Community": "Contributor-facing automation.",
}

# Every file allowed in .github/workflows/. CI coverage is added by extending
# ci.yml: a job built from the shared actions in .github/actions/, or a lane
# file it calls. A file missing here fails the check, and so does an entry
# whose file is gone, so the set only shrinks unless a PR adds a file on
# purpose and says why (README, "Adding CI").
WORKFLOW_FILES = frozenset(
    {
        "bazel.yml",
        "build-android.yml",
        "build-apple.yml",
        "build-flutter.yml",
        "build-natives.yml",
        "build-react-native.yml",
        "build-unity.yml",
        "ci.yml",
        "codeql.yml",
        "discord-notify.yml",
        "gradle-wrapper-validation.yml",
        "llamacpp-update.yml",
        "release-dryrun.yml",
        "release-notify.yml",
        "release-prep.yml",
        "release-publish.yml",
        "scorecard.yml",
        "star-history.yml",
        "test-choice-conformance.yml",
        "test-ci.yml",
        "test-decision-conformance.yml",
        "test-zzz-android.yml",
        "test-zzz-engine.yml",
        "trusted-zzz-sdk.yml",
        "unity-activation.yml",
        "unity-editor.yml",
        "web-webgpu.yml",
        "welcome.yml",
    }
)

# The files that may declare pull_request or pull_request_target. ci.yml and
# bazel.yml validate pull requests; the others scan them (codeql.yml,
# gradle-wrapper-validation.yml), dry-run a release (release-dryrun.yml),
# publish a merged release PR (release-publish.yml) or greet contributors
# (discord-notify.yml, welcome.yml).
PULL_REQUEST_FILES = frozenset(
    {
        "bazel.yml",
        "ci.yml",
        "codeql.yml",
        "discord-notify.yml",
        "gradle-wrapper-validation.yml",
        "release-dryrun.yml",
        "release-publish.yml",
        "welcome.yml",
    }
)

# macOS runners are shared: the organization's plan runs 5 macOS jobs at once,
# across every repository, so each one a pull request starts can queue behind
# another pull request's. The pull-request pipeline (ci.yml and the lane files
# it calls, matrix legs counted) holds exactly this many; macos_jobs() lists
# them. A new one goes into an existing macOS job as steps. If it truly needs
# its own runner, raise this number in the same PR and say why; when a macOS
# job goes, lower it, so the room is not reused silently.
#
# 7: check-platform-macos, the Candle, whisper.cpp and policy-routing lanes,
# the Apple build, React Native's iOS build, and the choice lane's Qwen GGUF
# job, which ran on every matching pull request from its own workflow before
# ci.yml called it and now runs only when the paths it reads change.
MACOS_JOB_BUDGET = 7

_PULL_REQUEST_EVENTS = ("pull_request", "pull_request_target")

# Events a lane file (one that declares workflow_call) may also declare.
_LANE_EVENTS = {"workflow_call", "workflow_dispatch", "schedule"}

_NAME = re.compile(r"^([A-Za-z]+): (\S.*)$")
_COMMENT = re.compile(r"\s+#.*$")
_KEY = re.compile(r"^\s*([A-Za-z_][\w-]*):\s*(.*)$")

# Keys that may appear under an event. Only branches/tags/types are rendered;
# the rest are accepted so their bodies can be skipped. Anything else (for
# example `branches-ignore`) would change what a row means, so it is refused.
_EVENT_KEYS = {"branches", "tags", "types", "paths", "paths-ignore", "inputs"}

# workflow_call declares its interface instead of filters; none of it is rendered.
_CALL_KEYS = {"inputs", "outputs", "secrets"}

# A job that runs a lane file: `uses: ./.github/workflows/<file>`.
_CALL = re.compile(r"""^\s+uses:\s*(["']?)\./\.github/workflows/([^"'\s#]+)\1\s*(?:#.*)?$""")

# Events are listed in this order so rows scan the same way; others follow alphabetically.
_EVENT_ORDER = (
    "push",
    "pull_request",
    "pull_request_target",
    "workflow_call",
    "issues",
    "release",
    "schedule",
    "workflow_dispatch",
)


class Unsupported(Exception):
    """A workflow uses YAML this generator does not read."""


@dataclass(frozen=True)
class Workflow:
    file: str
    name: str
    purpose: Optional[str]
    triggers: Tuple[str, ...]
    writes: Tuple[str, ...]
    concurrency: Tuple[str, ...]
    # Event names, in the order of `triggers`.
    events: Tuple[str, ...] = ()
    # Lane files this workflow runs (`uses: ./.github/workflows/<file>`).
    calls: Tuple[str, ...] = ()

    @property
    def group(self) -> Optional[str]:
        match = _NAME.match(self.name)
        return match.group(1) if match else None


# --- reading -----------------------------------------------------------------


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _skip(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _block(lines: Sequence[str], start: int) -> List[str]:
    """Lines nested under lines[start], up to the next line at its indent or less."""
    base = _indent(lines[start])
    block: List[str] = []
    for line in lines[start + 1 :]:
        if _skip(line):
            block.append(line)
        elif _indent(line) > base or (_indent(line) == base and line.lstrip().startswith("- ")):
            block.append(line)
        else:
            break
    return block


def _scalar(raw: str) -> str:
    raw = raw.strip()
    if raw[:1] in ('"', "'"):
        quote = raw[0]
        end = raw.find(quote, 1)
        tail = raw[end + 1 :].strip() if end != -1 else "?"
        if end == -1 or "\\" in raw[1:end] or (tail and not tail.startswith("#")):
            raise Unsupported(f"unsupported quoted scalar {raw!r}")
        return raw[1:end]
    return _COMMENT.sub("", raw).strip()


def _items(rest: str, block: Sequence[str]) -> List[str]:
    """A list written as `[a, b]`, as one scalar, or as `- a` lines under the key."""
    rest = _COMMENT.sub("", rest.strip()).strip()
    if rest.startswith("["):
        if not rest.endswith("]"):
            raise Unsupported(f"unsupported flow list {rest!r}")
        return [_scalar(item) for item in rest[1:-1].split(",") if item.strip()]
    if rest:
        return [_scalar(rest)]
    items: List[str] = []
    for line in block:
        if _skip(line):
            continue
        match = re.match(r"\s*-\s+(.*)$", line)
        if not match:
            raise Unsupported(f"unsupported list line {line.strip()!r}")
        items.append(_scalar(match.group(1)))
    return items


def _name(lines: Sequence[str]) -> Tuple[int, str]:
    for index, line in enumerate(lines):
        match = re.match(r"^name:\s*(.*)$", line)
        if match:
            return index, _scalar(match.group(1))
    raise Unsupported("no top-level name:")


def _purpose(lines: Sequence[str], name_index: int) -> Optional[str]:
    if name_index == 0:
        return None
    match = re.match(r"^# Purpose:\s*(\S.*?)\s*$", lines[name_index - 1])
    return match.group(1) if match else None


def _event_filters(event: str, rest: str, sub: Sequence[str]) -> Dict[str, List[str]]:
    rest = _COMMENT.sub("", rest).strip()
    if rest not in ("", "{}"):
        raise Unsupported(f"on.{event}: unsupported inline value {rest!r}")
    content = [line for line in sub if not _skip(line)]
    if not content:
        return {}
    if event == "schedule":
        crons: List[str] = []
        for line in content:
            match = re.match(r"^\s*(?:-\s+)?cron:\s*(.+)$", line)
            if not match:
                raise Unsupported(f"on.schedule: unsupported line {line.strip()!r}")
            crons.append(_scalar(match.group(1)))
        return {"cron": crons}
    filter_indent = min(_indent(line) for line in content)
    filters: Dict[str, List[str]] = {}
    allowed = _CALL_KEYS if event == "workflow_call" else _EVENT_KEYS
    for offset, line in enumerate(sub):
        if _skip(line) or _indent(line) != filter_indent:
            continue
        match = _KEY.match(line)
        if not match:
            raise Unsupported(f"on.{event}: unsupported line {line.strip()!r}")
        key, value = match.groups()
        if key not in allowed:
            raise Unsupported(f"on.{event}.{key}: the generator does not read this key")
        if key in ("branches", "tags", "types"):
            filters[key] = _items(value, _block(sub, offset))
    return filters


def _render_trigger(event: str, filters: Dict[str, List[str]]) -> str:
    if event == "schedule":
        return "schedule (" + ", ".join(f"`{cron}`" for cron in filters["cron"]) + ")"
    text = event
    if filters.get("types"):
        text += " (" + ", ".join(filters["types"]) + ")"
    if filters.get("branches"):
        text += " to " + ", ".join(f"`{branch}`" for branch in filters["branches"])
    if filters.get("tags"):
        text += " of tags " + ", ".join(f"`{tag}`" for tag in filters["tags"])
    return text


def _event_rank(event: str) -> Tuple[int, str]:
    return (_EVENT_ORDER.index(event) if event in _EVENT_ORDER else len(_EVENT_ORDER), event)


def _triggers(lines: Sequence[str]) -> List[Tuple[str, str]]:
    """(event, rendered trigger) pairs in display order."""
    start = next((i for i, line in enumerate(lines) if re.match(r"""^("on"|'on'|on):""", line)), None)
    if start is None:
        raise Unsupported("no top-level on:")
    head = re.match(r"""^(?:"on"|'on'|on):\s*(.*)$""", lines[start]).group(1)  # type: ignore[union-attr]
    head = _COMMENT.sub("", head).strip()
    if head.startswith("{"):
        # A flow mapping would read as one event and hide pull_request from the rules.
        raise Unsupported(f"on: unsupported flow mapping {head!r}")
    triggers: Dict[str, str] = {}
    if head:
        for event in _items(head, []):
            triggers[event] = _render_trigger(event, {})
    else:
        block = _block(lines, start)
        content = [line for line in block if not _skip(line)]
        if not content:
            raise Unsupported("empty on: block")
        event_indent = min(_indent(line) for line in content)
        for offset, line in enumerate(block):
            if _skip(line) or _indent(line) != event_indent:
                continue
            match = _KEY.match(line)
            if not match:
                raise Unsupported(f"on: unsupported line {line.strip()!r}")
            event, rest = match.groups()
            triggers[event] = _render_trigger(event, _event_filters(event, rest, _block(block, offset)))
    return [(event, triggers[event]) for event in sorted(triggers, key=_event_rank)]


def _calls(lines: Sequence[str]) -> List[str]:
    """Lane files run by this workflow's jobs, in file order, each once."""
    calls: List[str] = []
    for line in lines:
        match = _CALL.match(line)
        if match and match.group(2) not in calls:
            calls.append(match.group(2))
    return calls


def _writes(lines: Sequence[str]) -> List[str]:
    """Permission scopes granted `write` at any level (workflow or job)."""
    scopes = set()
    index = 0
    while index < len(lines):
        match = re.match(r"^\s*permissions:\s*(.*)$", lines[index])
        if not match:
            index += 1
            continue
        rest = _COMMENT.sub("", match.group(1)).strip()
        block = _block(lines, index)
        if rest:
            if rest == "write-all":
                scopes.add("all")
            elif rest not in ("read-all", "{}"):
                raise Unsupported(f"unsupported permissions value {rest!r}")
        else:
            for line in block:
                if _skip(line):
                    continue
                scope = re.match(r"^\s+([a-z-]+):\s*(\S+)", line)
                if not scope:
                    raise Unsupported(f"unsupported permissions line {line.strip()!r}")
                if scope.group(2) == "write":
                    scopes.add(scope.group(1))
        index += 1 + len(block)
    return sorted(scopes)


def _concurrency(lines: Sequence[str]) -> List[str]:
    groups: List[str] = []
    for index, line in enumerate(lines):
        match = re.match(r"^\s*concurrency:\s*(.*)$", line)
        if not match:
            continue
        rest = _COMMENT.sub("", match.group(1)).strip()
        if rest:
            groups.append(rest)  # `concurrency: <group>` or an inline map
            continue
        for sub in _block(lines, index):
            group = re.match(r"^\s+group:\s*(.*)$", sub)
            if group:
                groups.append(_COMMENT.sub("", group.group(1)).strip())
    return groups


def parse_workflow(path: Path) -> Workflow:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        name_index, name = _name(lines)
        triggers = _triggers(lines)
        return Workflow(
            file=path.name,
            name=name,
            purpose=_purpose(lines, name_index),
            triggers=tuple(text for _, text in triggers),
            writes=tuple(_writes(lines)),
            concurrency=tuple(_concurrency(lines)),
            events=tuple(event for event, _ in triggers),
            calls=tuple(_calls(lines)),
        )
    except Unsupported as exc:
        raise Unsupported(f"{path.name}: {exc}") from None


@dataclass(frozen=True)
class Job:
    """The keys of one job that decide when and where it runs."""

    id: str
    name: Optional[str]
    condition: Optional[str]
    needs: Tuple[str, ...]
    uses: Optional[str]
    runs_on: Optional[str]
    # `strategy.matrix.include` entries, each as its scalar keys.
    include: Tuple[Dict[str, str], ...] = ()
    # Other `strategy.matrix` keys (axes whose values multiply the legs).
    axes: Tuple[str, ...] = ()

    @property
    def check_name(self) -> str:
        """The check run GitHub reports for a job that calls no workflow."""
        return self.name or self.id


def _folded(rest: str, block: Sequence[str]) -> str:
    """A scalar written inline or as a `>-` / `|` block, joined into one line."""
    rest = _COMMENT.sub("", rest).strip()
    if rest[:1] in (">", "|"):
        return " ".join(line.strip() for line in block if not _skip(line))
    return _scalar(rest)


def _matrix_axes(job_lines: Sequence[str]) -> Tuple[str, ...]:
    """Keys under `strategy.matrix` other than include/exclude: axes the legs multiply over."""
    matrix = next((i for i, line in enumerate(job_lines) if re.match(r"^      matrix:\s*$", line)), None)
    if matrix is None:
        return ()
    block = [line for line in _block(job_lines, matrix) if not _skip(line)]
    if not block:
        return ()
    indent = min(_indent(line) for line in block)
    keys = [m.group(1) for line in block if _indent(line) == indent and (m := _KEY.match(line))]
    return tuple(key for key in keys if key not in ("include", "exclude"))


def _matrix_include(job_lines: Sequence[str]) -> Tuple[Dict[str, str], ...]:
    strategy = next((i for i, line in enumerate(job_lines) if re.match(r"^    strategy:\s*$", line)), None)
    if strategy is None:
        return ()
    block = _block(job_lines, strategy)
    start = next((i for i, line in enumerate(block) if re.match(r"^\s+include:\s*$", line)), None)
    if start is None:
        return ()
    entries: List[Dict[str, str]] = []
    for line in _block(block, start):
        if _skip(line):
            continue
        if re.match(r"^\s*-\s*\{", line):
            raise Unsupported(f"unsupported flow-mapping matrix entry {line.strip()!r}")
        match = re.match(r"^\s*(-\s+)?([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not match:
            continue
        if match.group(1) or not entries:
            entries.append({})
        value = _COMMENT.sub("", match.group(3)).strip()
        if value and value[:1] not in (">", "|"):
            entries[-1][match.group(2)] = _scalar(value)
    return tuple(entries)


def read_jobs(path: Path) -> Dict[str, Job]:
    """Every job in a workflow file, by id, in file order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(r"^jobs:\s*$", line)), None)
    if start is None:
        raise Unsupported(f"{path.name}: no top-level jobs:")
    jobs: Dict[str, Job] = {}
    body = _block(lines, start)
    content = [line for line in body if not _skip(line)]
    if content and min(_indent(line) for line in content) != 2:
        raise Unsupported(f"{path.name}: jobs must be indented two spaces")
    for offset, line in enumerate(body):
        if _skip(line) or _indent(line) != 2:
            continue
        match = re.match(r"^  ([A-Za-z_][\w-]*):\s*$", line)
        if not match:
            raise Unsupported(f"{path.name}: unsupported job line {line.strip()!r}")
        job_lines = _block(body, offset)
        keys: Dict[str, str] = {}
        needs: List[str] = []
        for index, job_line in enumerate(job_lines):
            key = re.match(r"^    ([A-Za-z_][\w-]*):\s*(.*)$", job_line)
            if not key:
                continue
            name, rest = key.groups()
            sub = _block(job_lines, index)
            if name == "needs":
                needs = _items(rest, sub)
            elif name in ("name", "if", "uses", "runs-on"):
                keys[name] = _folded(rest, sub)
        jobs[match.group(1)] = Job(
            id=match.group(1),
            name=keys.get("name"),
            condition=keys.get("if"),
            needs=tuple(needs),
            uses=keys.get("uses"),
            runs_on=keys.get("runs-on"),
            include=_matrix_include(job_lines),
            axes=_matrix_axes(job_lines),
        )
    return jobs


def _macos_legs(job: Job, file: str) -> List[str]:
    """This job's macOS legs. Raises on a runner shape it cannot count rather than miss one."""
    runner = job.runs_on
    if not runner or runner[:1] in ("[", "{"):
        # A missing value, a group/labels mapping or a label list.
        raise Unsupported(f"{file}: job {job.id} runs on {runner!r}, which macos_jobs() cannot count")
    matrix = re.fullmatch(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}", runner)
    if matrix:
        if not job.include or matrix.group(1) in job.axes:
            raise Unsupported(
                f"{file}: job {job.id} runs on a matrix axis; list its legs under `include:` only"
            )
        return [
            f"{file}:{job.id} ({leg.get(matrix.group(1), '?')})"
            for leg in job.include
            # GitHub matches runner labels case-insensitively.
            if leg.get(matrix.group(1), "").lower().startswith("macos")
        ]
    if "${{" in runner:
        raise Unsupported(f"{file}: job {job.id} runs on {runner!r}, which macos_jobs() cannot count")
    return [f"{file}:{job.id}"] if runner.lower().startswith("macos") else []


def macos_jobs(directory: Path, entry: str = "ci.yml") -> List[str]:
    """The macOS jobs a pull request can start through `entry` and the lane files it calls."""
    if not (directory / entry).is_file():
        return []
    found: List[str] = []
    for job in read_jobs(directory / entry).values():
        if job.uses and job.uses.startswith("./.github/workflows/"):
            found += macos_jobs(directory, job.uses.rsplit("/", 1)[1])
        elif job.uses:
            raise Unsupported(f"{entry}: job {job.id} calls {job.uses}, whose runners macos_jobs() cannot see")
        else:
            found += _macos_legs(job, entry)
    return found


def macos_problems(directory: Path, budget: Optional[int] = None) -> List[str]:
    budget = MACOS_JOB_BUDGET if budget is None else budget
    jobs = macos_jobs(directory)
    if len(jobs) > budget:
        return [
            f"the pull-request pipeline has {len(jobs)} macOS jobs ({', '.join(jobs)}), over "
            f"MACOS_JOB_BUDGET ({budget}). Add steps to an existing macOS job, or raise the "
            "budget in tools/scripts/gen_workflow_index.py in the same PR and say why"
        ]
    if len(jobs) < budget:
        return [
            f"the pull-request pipeline has {len(jobs)} macOS jobs ({', '.join(jobs)}); lower "
            f"MACOS_JOB_BUDGET from {budget} to {len(jobs)} so the room is not reused silently"
        ]
    return []


def load(directory: Path) -> List[Workflow]:
    paths = sorted(list(directory.glob("*.yml")) + list(directory.glob("*.yaml")))
    return [parse_workflow(path) for path in paths]


# --- rules -------------------------------------------------------------------


def problems(
    workflows: Sequence[Workflow],
    files: Optional[AbstractSet[str]] = None,
    pull_request_files: Optional[AbstractSet[str]] = None,
) -> List[str]:
    """Rule violations; `files` and `pull_request_files` default to the allowlists above."""
    files = WORKFLOW_FILES if files is None else files
    pull_request_files = PULL_REQUEST_FILES if pull_request_files is None else pull_request_files
    found: List[str] = []
    seen: Dict[str, str] = {}
    present = {workflow.file: workflow for workflow in workflows}
    names = set(present)
    for workflow in workflows:
        if workflow.file not in files:
            found.append(
                f"{workflow.file}: not an allowed workflow file. Add CI as a job in ci.yml or a lane "
                "file it calls (.github/workflows/README.md, Adding CI); a file the repository really "
                "needs is added to WORKFLOW_FILES in tools/scripts/gen_workflow_index.py by a PR that says why"
            )
        for event in _PULL_REQUEST_EVENTS:
            if event in workflow.events and workflow.file not in pull_request_files:
                found.append(
                    f"{workflow.file}: declares {event}, but only ci.yml and bazel.yml validate pull "
                    "requests. Run it from ci.yml: a job, or a lane file that declares workflow_call"
                )
        if "workflow_call" in workflow.events:
            extra = [event for event in workflow.events if event not in _LANE_EVENTS]
            if extra:
                found.append(
                    f"{workflow.file}: a lane file (workflow_call) runs when ci.yml calls it, so it "
                    f"declares only workflow_call, workflow_dispatch and schedule, not {', '.join(extra)}"
                )
            if not any(workflow.file in other.calls for other in workflows):
                found.append(f"{workflow.file}: declares workflow_call, but no workflow calls it")
        for callee in workflow.calls:
            if callee not in present:
                found.append(f"{workflow.file}: calls {callee}, which does not exist")
            elif "workflow_call" not in present[callee].events:
                found.append(f"{workflow.file}: calls {callee}, which does not declare workflow_call")
        match = _NAME.match(workflow.name)
        if not match or match.group(1) not in GROUPS:
            found.append(
                f"{workflow.file}: name {workflow.name!r} must be 'Group: Subject' "
                f"with Group one of: {', '.join(GROUPS)}"
            )
        if not workflow.purpose:
            found.append(f"{workflow.file}: add '# Purpose: <one sentence>' on the line directly above name:")
        for group in workflow.concurrency:
            if "github.workflow" in group:
                found.append(
                    f"{workflow.file}: concurrency group {group!r} uses github.workflow; "
                    "use a literal '<file-stem>-...' key so renaming the workflow cannot change it"
                )
        if workflow.name in seen:
            found.append(f"{workflow.file}: name {workflow.name!r} is already used by {seen[workflow.name]}")
        seen.setdefault(workflow.name, workflow.file)
    for file in sorted(set(files) - names):
        found.append(f"WORKFLOW_FILES lists {file}, which does not exist; remove it")
    for file in sorted(set(pull_request_files) & names):
        if not any(event in present[file].events for event in _PULL_REQUEST_EVENTS):
            found.append(f"PULL_REQUEST_FILES lists {file}, which declares no pull-request event; remove it")
    for file in sorted(set(pull_request_files) - set(files)):
        found.append(f"PULL_REQUEST_FILES lists {file}, which WORKFLOW_FILES does not allow; remove it")
    return found


# --- writing -----------------------------------------------------------------


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _called_by(workflow: Workflow, workflows: Sequence[Workflow]) -> str:
    callers = sorted(other.file for other in workflows if workflow.file in other.calls)
    if not callers:
        return "workflow_call"
    return "called by " + ", ".join(f"`{caller}`" for caller in callers)


def render(workflows: Sequence[Workflow]) -> str:
    sections: List[str] = []
    for group, blurb in GROUPS.items():
        rows = sorted((w for w in workflows if w.group == group), key=lambda w: w.name.lower())
        if not rows:
            continue
        lines = [
            f"### {group}",
            "",
            blurb,
            "",
            "| File | Name | Triggers | Writes | Purpose |",
            "| --- | --- | --- | --- | --- |",
        ]
        for w in rows:
            writes = ", ".join(w.writes) if w.writes else "none"
            # A lane file's row names its caller instead of the bare event.
            shown = [_called_by(w, workflows) if text == "workflow_call" else text for text in w.triggers]
            triggers = _cell(", ".join(shown))
            lines.append(
                f"| [`{w.file}`]({w.file}) | {_cell(w.name)} | {triggers} | {writes} | {_cell(w.purpose or '')} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def splice(readme: str, table: str) -> str:
    if readme.count(BEGIN) != 1 or readme.count(END) != 1 or readme.index(BEGIN) > readme.index(END):
        raise Unsupported(f"{README.name} must contain the BEGIN and END markers exactly once, in that order")
    head, rest = readme.split(BEGIN)
    _, tail = rest.split(END)
    return f"{head}{BEGIN}\n\n{table}\n{END}{tail}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the workflow inventory in .github/workflows/README.md.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the table is stale or a workflow breaks the naming rules; write nothing",
    )
    args = parser.parse_args(argv)

    try:
        workflows = load(WORKFLOWS_DIR)
        rule_violations = problems(workflows) + macos_problems(WORKFLOWS_DIR)
        if rule_violations:
            for violation in rule_violations:
                print(f"error: {violation}", file=sys.stderr)
            return 1
        current = README.read_text(encoding="utf-8")
        expected = splice(current, render(workflows))
    except (Unsupported, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if current == expected:
        print("workflow index is current")
        return 0
    if args.check:
        sys.stderr.writelines(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                "README.md (committed)",
                "README.md (generated)",
            )
        )
        print("\nerror: the workflow index is stale. Run: python3 tools/scripts/gen_workflow_index.py", file=sys.stderr)
        return 1
    with open(README, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(expected)
    print(f"updated {README}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
