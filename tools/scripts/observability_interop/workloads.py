"""The producers run_interop.py drives through the Collector.

Each workload is a real program that exports to the Collector's OTLP/HTTP
endpoint, reports the ``{trace_id, span_id}`` references of what it recorded,
and records an evaluation and user feedback on its inference:

| Workload | Program | Transport | Content |
|---|---|---|---|
| ``rust`` | ``crates/xybrid-observability/examples/standalone.rs`` | http/protobuf | span + event, literal rule ``sk-demo-123`` -> ``[KEY]`` |
| ``js`` | ``bindings/observability-js/examples/node-basic.ts`` | http/protobuf | default (span only), rule ``Paris`` -> ``[CITY]`` |
| ``js-metadata`` | the same JS example with ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT=no_content`` | http/protobuf | none |
| ``python`` | ``bindings/python/tests/observability_scenario.py`` (the shared runtime scenario on the compiled bridge), then ``python_quality.py`` | http/json, then http/protobuf | default, literal rule and omit rule |

Every workload's resource carries ``xybrid.interop.workload=<name>`` (through
``OTEL_RESOURCE_ATTRIBUTES``), which is how the ground truth is split per
workload; the service name is the fallback.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAPTURE_DIR = REPO / "tools" / "scripts" / "observability_capture"
JS_DIR = REPO / "bindings" / "observability-js"
PYTHON_DIR = REPO / "bindings" / "python"

REFS_PREFIX = "observation refs: "
SCENARIO = json.loads((CAPTURE_DIR / "scenario.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Workload:
    name: str
    language: str
    program: str
    service_name: str
    # "captured": inference/tool content must reach a content-capable
    # destination; "metadata_only": no content may reach any destination.
    content: str
    # Strings the producer's redaction rules remove: they must reach neither
    # the Collector nor a destination.
    secrets: Tuple[str, ...]
    # What the rules put in their place (must reach a content destination).
    replacements: Tuple[str, ...]
    # gen_ai.evaluation.result records the workload emits.
    evaluations: int
    # Whether its inference measures first token apart from first chunk.
    streaming_timing: bool
    notes: str = ""


WORKLOADS: Dict[str, Workload] = {
    "rust": Workload(
        name="rust", language="rust", program="crates/xybrid-observability/examples/standalone.rs",
        service_name="xybrid-observability-standalone", content="captured",
        secrets=("sk-demo-123",), replacements=("[KEY]",), evaluations=2, streaming_timing=True,
        notes="workflow, model load, tool, chat with a failed attempt and a retry, chunks, measured first token"),
    "js": Workload(
        name="js", language="javascript", program="bindings/observability-js/examples/node-basic.ts",
        service_name="observability-example", content="captured",
        secrets=("Paris",), replacements=("[CITY]",), evaluations=2, streaming_timing=True,
        notes="workflow, streaming chat with measured first token, tool"),
    "js-metadata": Workload(
        name="js-metadata", language="javascript",
        program="bindings/observability-js/examples/node-basic.ts (no_content)",
        service_name="observability-example", content="metadata_only",
        secrets=("Paris", "[CITY]", "Looking up the weather", "helpful"), replacements=(),
        evaluations=2, streaming_timing=True,
        notes="the JS example with OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT=no_content"),
    "python": Workload(
        name="python", language="python (native bridge)",
        program="bindings/python/tests/observability_scenario.py + python_quality.py",
        service_name="xybrid-scenario-python", content="captured",
        secrets=(SCENARIO["secret"], SCENARIO["omit_marker"]), replacements=(SCENARIO["replacement"],),
        evaluations=2, streaming_timing=False,
        notes="the shared runtime scenario on the compiled Python bridge, then an evaluation and feedback "
              "recorded by a second process against the scenario's stored inference reference"),
}
DEFAULT_WORKLOADS = ("rust", "js", "js-metadata", "python")


@dataclass
class Step:
    argv: List[str]
    cwd: Path
    env: Dict[str, str] = field(default_factory=dict)
    label: str = ""


@dataclass
class Toolchain:
    python: str
    node: Optional[str]
    js_runtime: str  # "node" or "bun"
    cargo: str = "cargo"
    cargo_release: bool = False


def node_flags(node: str) -> List[str]:
    """Type stripping is on by default from Node 22.18 / 23.6; 22.6+ needs the flag."""
    try:
        output = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    match = re.match(r"v(\d+)\.(\d+)", output.strip())
    if match is None:
        return []
    major, minor = int(match.group(1)), int(match.group(2))
    if major > 23 or (major == 23 and minor >= 6) or (major == 22 and minor >= 18):
        return []
    return ["--experimental-strip-types"]


def steps_for(workload: Workload, endpoint: str, out_dir: Path, tools: Toolchain) -> List[Step]:
    """The commands of one workload. ``endpoint`` is the Collector's base URL."""
    base_env = {"OTEL_RESOURCE_ATTRIBUTES": f"xybrid.interop.workload={workload.name}"}
    if workload.name == "rust":
        argv = [tools.cargo, "run", "--quiet", "-p", "xybrid-observability", "--example", "standalone"]
        if tools.cargo_release:
            argv.insert(2, "--release")
        return [Step(argv, REPO, {**base_env, "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint}, "standalone example")]
    if workload.name in ("js", "js-metadata"):
        env = {**base_env, "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint}
        if workload.content == "metadata_only":
            env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT"] = "no_content"
        if tools.js_runtime == "bun":
            argv = ["bun", "examples/node-basic.ts"]
        else:
            node = tools.node or "node"
            argv = [node, *node_flags(node), "examples/node-basic.ts"]
        return [Step(argv, JS_DIR, env, "node-basic example")]
    if workload.name == "python":
        expectations = out_dir / "expectations.json"
        env = {**base_env, "PYTHONPATH": str(PYTHON_DIR),
               "XYBRID_SCENARIO_BINDING": "python", "XYBRID_SCENARIO_ENDPOINT": endpoint,
               "XYBRID_SCENARIO_EXPECTATIONS": str(expectations), "XYBRID_SCENARIO_WORKDIR": str(out_dir / "work")}
        return [
            Step([tools.python, str(PYTHON_DIR / "tests" / "observability_scenario.py")], REPO, env,
                 "runtime scenario"),
            Step([tools.python, str(HERE / "python_quality.py"), "--endpoint", endpoint,
                  "--expectations", str(expectations)], REPO, env, "evaluation and feedback"),
        ]
    raise ValueError(f"unknown workload {workload.name!r}")


def preflight(workload: Workload, tools: Toolchain) -> List[str]:
    """What is missing to run ``workload``, as instructions (empty when ready)."""
    problems = []
    if workload.name == "rust" and shutil.which(tools.cargo) is None:
        problems.append("cargo is not on PATH")
    if workload.name in ("js", "js-metadata"):
        runtime = "bun" if tools.js_runtime == "bun" else (tools.node or "node")
        if shutil.which(runtime) is None:
            problems.append(f"{runtime} is not on PATH")
        if not (JS_DIR / "dist" / "otlp.js").is_file() or not (JS_DIR / "node_modules").is_dir():
            problems.append("the JS package is not built: (cd bindings/observability-js && "
                            "pnpm install --frozen-lockfile && pnpm exec tsdown)")
    if workload.name == "python":
        executable = shutil.which(tools.python)
        if executable is None:
            problems.append(f"{tools.python} is not on PATH")
            return problems
        env = {key: value for key, value in os.environ.items() if not key.startswith("OTEL_")}
        env["PYTHONPATH"] = str(PYTHON_DIR)
        probe = subprocess.run([executable, "-c", "import xybrid; print(xybrid.version())"],
                               capture_output=True, text=True, env=env, timeout=120)
        if probe.returncode != 0:
            last = (probe.stderr.strip().splitlines() or ["?"])[-1]
            problems.append(f"{tools.python} cannot import the Python SDK ({last}); build the bridge with "
                            "./tools/scripts/build-python-bolt.sh and pass --python with the interpreter "
                            "it was built for (python >= 3.10)")
    return problems


def parse_refs_line(text: str) -> Dict[str, Dict[str, str]]:
    """``{role: {"trace_id", "span_id"}}`` from the last ``observation refs:`` line.

    Lists become ``role.0``, ``role.1``...; ``null`` entries are dropped.
    """
    payload = None
    for line in text.splitlines():
        if line.startswith(REFS_PREFIX):
            payload = line[len(REFS_PREFIX):]
    if payload is None:
        return {}
    try:
        data = json.loads(payload)
    except ValueError:
        return {}
    refs: Dict[str, Dict[str, str]] = {}

    def add(role: str, value: Any) -> None:
        if isinstance(value, dict) and value.get("trace_id") and value.get("span_id"):
            refs[role] = {"trace_id": str(value["trace_id"]).lower(), "span_id": str(value["span_id"]).lower()}
        elif isinstance(value, list):
            for index, item in enumerate(value):
                add(f"{role}.{index}", item)

    if isinstance(data, dict):
        for role, value in data.items():
            add(role, value)
    return refs


def reported_refs(workload: Workload, stdout: str, out_dir: Path) -> Dict[str, Dict[str, str]]:
    """References the workload reported: its stdout line, or for the Python
    scenario its ``expectations.json`` (scenario.md, step f)."""
    refs = parse_refs_line(stdout)
    if workload.name == "python":
        path = out_dir / "expectations.json"
        if path.is_file():
            expectations = json.loads(path.read_text(encoding="utf-8"))
            for role, value in (expectations.get("references") or {}).items():
                if isinstance(value, dict) and value.get("trace_id") and value.get("span_id"):
                    refs.setdefault(role, {"trace_id": value["trace_id"].lower(),
                                           "span_id": value["span_id"].lower()})
    return refs


def split_key(resource: Mapping[str, Any]) -> Optional[str]:
    """The workload named by a resource's ``xybrid.interop.workload``
    attribute (run_interop.py falls back to the service name)."""
    name = resource.get("xybrid.interop.workload")
    if isinstance(name, str) and name in WORKLOADS:
        return name
    return None
