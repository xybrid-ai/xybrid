#!/usr/bin/env python3
"""Generate the workflow inventory in .github/workflows/README.md.

The table between the BEGIN/END markers is derived from the workflow files, so
it cannot drift from them. The same pass enforces the conventions the README
describes, because a table can only be built from files that follow them:

  * `name:` is `Group: Subject`, Group is one of GROUPS, and names are unique.
  * The line directly above `name:` is `# Purpose: <one sentence>`.
  * A concurrency group never uses `github.workflow`, so renaming a workflow
    cannot silently change its concurrency group.

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
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
README = WORKFLOWS_DIR / "README.md"

BEGIN = "<!-- BEGIN GENERATED: workflow inventory (tools/scripts/gen_workflow_index.py) -->"
END = "<!-- END GENERATED: workflow inventory -->"

# Display order of the inventory; the keys are the allowed `Group` prefixes.
GROUPS = {
    "CI": "Workspace validation, pull-request gates and backend test suites.",
    "SDK": "Per-platform SDK builds, wrapper tests and example apps.",
    "Artifacts": "Publish reusable native artifacts and their download manifests.",
    "Release": "Cut, validate, publish and announce a release.",
    "Security": "Static analysis and supply-chain checks.",
    "Maintenance": "Manual utilities, run on demand.",
    "Community": "Contributor-facing automation.",
}

_NAME = re.compile(r"^([A-Za-z]+): (\S.*)$")
_COMMENT = re.compile(r"\s+#.*$")
_KEY = re.compile(r"^\s*([A-Za-z_][\w-]*):\s*(.*)$")

# Keys that may appear under an event. Only branches/tags/types are rendered;
# the rest are accepted so their bodies can be skipped. Anything else (for
# example `branches-ignore`) would change what a row means, so it is refused.
_EVENT_KEYS = {"branches", "tags", "types", "paths", "paths-ignore", "inputs"}

# Events are listed in this order so rows scan the same way; others follow alphabetically.
_EVENT_ORDER = ("push", "pull_request", "pull_request_target", "issues", "release", "schedule", "workflow_dispatch")


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
    for offset, line in enumerate(sub):
        if _skip(line) or _indent(line) != filter_indent:
            continue
        match = _KEY.match(line)
        if not match:
            raise Unsupported(f"on.{event}: unsupported line {line.strip()!r}")
        key, value = match.groups()
        if key not in _EVENT_KEYS:
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


def _triggers(lines: Sequence[str]) -> List[str]:
    start = next((i for i, line in enumerate(lines) if re.match(r"""^("on"|'on'|on):""", line)), None)
    if start is None:
        raise Unsupported("no top-level on:")
    head = re.match(r"""^(?:"on"|'on'|on):\s*(.*)$""", lines[start]).group(1)  # type: ignore[union-attr]
    head = _COMMENT.sub("", head).strip()
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
    return [triggers[event] for event in sorted(triggers, key=_event_rank)]


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
        return Workflow(
            file=path.name,
            name=name,
            purpose=_purpose(lines, name_index),
            triggers=tuple(_triggers(lines)),
            writes=tuple(_writes(lines)),
            concurrency=tuple(_concurrency(lines)),
        )
    except Unsupported as exc:
        raise Unsupported(f"{path.name}: {exc}") from None


def load(directory: Path) -> List[Workflow]:
    paths = sorted(list(directory.glob("*.yml")) + list(directory.glob("*.yaml")))
    return [parse_workflow(path) for path in paths]


# --- rules -------------------------------------------------------------------


def problems(workflows: Sequence[Workflow]) -> List[str]:
    found: List[str] = []
    seen: Dict[str, str] = {}
    for workflow in workflows:
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
    return found


# --- writing -----------------------------------------------------------------


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


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
            triggers = _cell(", ".join(w.triggers))
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
        rule_violations = problems(workflows)
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
