"""Every committed code generator must have a CI drift gate.

Generated bindings are checked in next to hand-written wrappers. A generator
without a gate lets a crate change ship with stale output; a gate that does not
trigger on the generator's inputs is the same failure, just quieter. These tests
read the workflows and the CI path filters, so a new generator (or a gate that
loses its trigger) fails here rather than in a consumer's build.
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci_gate import OPTIONAL_JOBS, PATH_FILTERS

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Generator -> (workflow, job, path filter or None for an always-on job,
# committed outputs, inputs that must select the gate).
BOLT_INPUTS = ("crates/xybrid-bolt/src/lib.rs", "crates/xybrid-bolt/Cargo.toml")
GATES = {
    "tools/scripts/gen_python_bolt.py": (
        "ci.yml", "python-sdk", "python_sdk",
        ("bindings/python/xybrid/_bolt/__init__.py", "bindings/python/xybrid/_bolt/_native.c"),
        BOLT_INPUTS,
    ),
    "tools/scripts/gen_kotlin_bolt.py": (
        "ci.yml", "kotlin-jni-drift", "kotlin_jni",
        ("bindings/kotlin/src/main/kotlin/ai/xybrid/XybridBolt.kt", "bindings/kotlin/bazel/jni/jni_glue.c"),
        BOLT_INPUTS,
    ),
    "tools/scripts/gen_unity_bolt_csharp.py": (
        "ci.yml", "unity-bolt-drift", "unity_bolt",
        ("bindings/unity/Runtime/Bolt/XybridBolt.cs", "bindings/unity/Runtime/Bolt/XybridBolt.cs.meta"),
        BOLT_INPUTS + ("crates/xybrid-ffi-facade/src/lib.rs",),
    ),
    "bindings/apple/scripts/gen-bolt-bindings.sh": (
        "ci.yml", "apple-bolt-drift", "apple_bolt",
        ("bindings/apple/Sources/Xybrid/xybrid_bolt.swift", "bindings/apple/include/xybrid-bolt.h"),
        BOLT_INPUTS,
    ),
    "tools/scripts/gen_workflow_index.py": (
        "ci.yml", "workflow-index", None, (".github/workflows/README.md",), (),
    ),
}
# Generators that produce fixtures from downloaded models, not code from the
# tree: nothing committed drifts when the crates change.
NOT_SOURCE_GENERATORS = {
    "tools/scripts/gen_embedding_goldens.py": "writes model reference vectors; needs the downloaded model",
}

FRB_COMMAND = "./generate-frb.sh"
FRB_OUTPUTS = ("bindings/flutter/lib/src/rust/", "bindings/flutter/rust/src/frb_generated.rs")
# The Flutter crate reaches these through the facade; FRB mirrors their types.
FRB_INPUTS = (
    "bindings/flutter/**", "crates/xybrid-ffi-facade/**", "crates/xybrid-sdk/**",
    "crates/xybrid-core/**", "crates/xybrid-observability/**",
)


def job_block(workflow: str, job: str) -> str:
    match = re.search(rf"^  {re.escape(job)}:\n((?:    .*\n|\n)*)", workflow, re.M)
    if match is None:
        raise AssertionError(f"job {job} not found")
    return match[1]


def trigger_paths(workflow: str, event: str) -> list:
    block = re.search(rf"^  {event}:\n((?:    .*\n)*)", workflow, re.M)
    if block is None:
        raise AssertionError(f"no {event} trigger")
    paths = re.search(r"^    paths:\n((?:      .*\n)*)", block[1], re.M)
    if paths is None:
        raise AssertionError(f"{event} trigger has no path filter")
    return re.findall(r"^      - '([^']+)'", paths[1], re.M)


class GenerationGateTests(unittest.TestCase):
    def test_every_generator_is_listed(self):
        found = {
            str(path.relative_to(REPO_ROOT))
            for pattern in ("tools/scripts/gen_*.py", "tools/scripts/gen-*.sh", "bindings/*/scripts/gen*")
            for path in REPO_ROOT.glob(pattern)
        }
        unlisted = found - set(GATES) - set(NOT_SOURCE_GENERATORS)
        self.assertEqual(unlisted, set(), "add a CI drift gate for each new generator, then list it in GATES")
        self.assertEqual(set(GATES) - found, set(), "a listed generator no longer exists")

    def test_each_generator_runs_with_check_in_its_gate_job(self):
        for generator, (workflow_name, job, flag, _, _) in GATES.items():
            with self.subTest(generator=generator):
                block = job_block((WORKFLOWS / workflow_name).read_text(encoding="utf-8"), job)
                self.assertRegex(block, rf"run: (python3 )?{re.escape(generator)} --check\b")
                if flag is None:
                    self.assertNotRegex(block, r"(?m)^    if:", "an always-on gate has no path condition")
                else:
                    self.assertIn(f"if: needs.tooling-changes.outputs.{flag} == 'true'", block)
                    self.assertEqual(OPTIONAL_JOBS.get(job), flag, "the CI Success gate must know the job's flag")

    def test_each_gate_triggers_on_its_inputs_and_outputs(self):
        for generator, (_, job, flag, outputs, inputs) in GATES.items():
            if flag is None:
                continue
            for path in (generator, *outputs, *inputs):
                with self.subTest(job=job, path=path):
                    self.assertRegex(path, PATH_FILTERS[flag])

    def test_frb_drift_diffs_both_outputs_and_triggers_on_the_sdk(self):
        workflow = (WORKFLOWS / "build-flutter.yml").read_text(encoding="utf-8")
        self.assertIn(FRB_COMMAND, workflow)
        wrapper = (REPO_ROOT / "bindings/flutter/generate-frb.sh").read_text(encoding="utf-8")
        self.assertIn("flutter_rust_bridge_codegen generate", wrapper)
        self.assertIn("rustfmt", wrapper)
        self.assertIn("rust/src/frb_generated.rs", wrapper)
        check = re.search(r"- name: Check FRB bindings are up-to-date\n((?:        .*\n)*)", workflow)
        self.assertIsNotNone(check, "the FRB drift step is missing")
        for output in FRB_OUTPUTS:
            with self.subTest(output=output):
                self.assertIn(output, check[1])
        self.assertIn("git diff --exit-code", check[1])
        for event in ("pull_request", "push"):
            paths = trigger_paths(workflow, event)
            for path in FRB_INPUTS:
                with self.subTest(event=event, path=path):
                    self.assertIn(path, paths)


if __name__ == "__main__":
    unittest.main()
