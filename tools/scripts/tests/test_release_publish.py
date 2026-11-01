"""Check release-publish.yml: crate publication order and the checksum gates."""

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/release-publish.yml"

# The post-publish check every package publish waits for.
GATE = "verify-spm-checksum"
# Jobs after publish-release that may run without GATE, and why.
UNGATED = {
    "dispatch-unity-natives": "builds Unity natives from the tag; never reads the XybridFFI zip",
}
# Parser guard: these publish packages, so the split below must find them.
PUBLISHERS = {
    "publish-crates", "publish-flutter", "publish-kotlin", "publish-npm",
    "publish-web-npm", "publish-swift-branch",
}
# Status functions in a job's `if:` replace the implicit success() check,
# so any of them (even `success() || ...`) can run a job after a need failed.
BYPASS = re.compile(r"\b(?:always|cancelled|failure|success)\(\)")
# Turns a failed step or job into a success for everything after it.
IGNORE_FAILURE = re.compile(r"(?m)^\s*continue-on-error:")


def code(text):
    """Drop comment lines, so prose about a command never counts as one."""
    return "".join(line for line in text.splitlines(True) if not line.lstrip().startswith("#"))


def workflow_jobs():
    """Split the `jobs:` mapping into {job id: body} (CI's python has no YAML)."""
    body = WORKFLOW.read_text().split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([a-z0-9-]+):\n", body, flags=re.MULTILINE)
    return {job: code(text) for job, text in zip(parts[1::2], parts[2::2])}


def needs(job):
    match = re.search(r"^    needs:[ \t]*(.*)\n((?:      - .*\n)*)", job, re.MULTILINE)
    if not match:
        return set()
    if match.group(1).strip():
        return {name.strip() for name in match.group(1).strip("[] ").split(",")}
    return set(re.findall(r"^      - ([a-z0-9-]+)", match.group(2), re.MULTILINE))


def steps(job):
    return re.split(r"^      - ", job.split("\n    steps:\n", 1)[1], flags=re.MULTILINE)[1:]


def step_if(step):
    match = re.search(r"^        if: (.+)$", step, re.MULTILINE)
    return match and match.group(1).strip()


class PublishDependencyOrderTests(unittest.TestCase):
    def test_local_dependencies_publish_before_their_consumers(self):
        workflow = WORKFLOW.read_text()
        order = re.findall(r"^\s+publish_crate ([a-z0-9-]+)\s*$", workflow, re.MULTILINE)
        self.assertTrue(order, "release workflow has no crate publication steps")
        self.assertEqual(len(order), len(set(order)), "duplicate crate publication")

        workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())
        manifests = {}
        for member in workspace["workspace"]["members"]:
            for directory in ROOT.glob(member):
                path = directory / "Cargo.toml"
                manifest = tomllib.loads(path.read_text())
                manifests[manifest["package"]["name"]] = (path, manifest)

        for consumer in order:
            path, manifest = manifests[consumer]
            scopes = [manifest, *manifest.get("target", {}).values()]
            for scope in scopes:
                for kind in ("dependencies", "build-dependencies"):
                    for spec in scope.get(kind, {}).values():
                        if not isinstance(spec, dict) or "path" not in spec:
                            continue
                        dependency_path = path.parent / spec["path"] / "Cargo.toml"
                        dependency = tomllib.loads(dependency_path.read_text())
                        name = dependency["package"]["name"]
                        with self.subTest(consumer=consumer, dependency=name):
                            self.assertIn(name, order, f"publish {name} before {consumer}")
                            self.assertLess(order.index(name), order.index(consumer))


class ChecksumGateTests(unittest.TestCase):
    def setUp(self):
        self.jobs = workflow_jobs()

    def test_every_publisher_needs_the_checksum_gate(self):
        downstream = {job for job, body in self.jobs.items() if "publish-release" in needs(body)}
        self.assertLessEqual(PUBLISHERS | set(UNGATED), downstream)
        for job in sorted(downstream - {GATE} - set(UNGATED)):
            with self.subTest(job=job):
                self.assertIn(GATE, needs(self.jobs[job]), f"{job} must need {GATE}")
                preamble = self.jobs[job].split("\n    steps:\n", 1)[0]
                self.assertNotRegex(preamble, BYPASS, f"{job} would run after {GATE} failed")
                self.assertNotRegex(preamble, IGNORE_FAILURE, f"{job} would hide its own failure")

    def test_the_gate_runs_on_every_path_a_publisher_does(self):
        # A skipped need skips the job that needs it, and publish-flutter runs
        # on the flutter_only re-dispatch that every other publisher skips.
        self.assertNotRegex(self.jobs[GATE], r"(?m)^    if:")

    def test_no_failure_in_the_checks_is_ignored(self):
        for job in (GATE, "publish-release"):
            with self.subTest(job=job):
                self.assertNotRegex(self.jobs[job], IGNORE_FAILURE)

    def test_publish_release_checks_the_draft_zip_before_publishing(self):
        job = steps(self.jobs["publish-release"])
        publish = [i for i, step in enumerate(job) if re.search(r"gh release edit .*--draft=false", step)]
        check = [i for i, step in enumerate(job) if "sync-spm-checksum.sh --check" in step]
        self.assertEqual(len(publish), 1, "one step publishes the draft")
        self.assertEqual(len(check), 1, "one step checks the draft's zip")
        self.assertLess(check[0], publish[0], "check the zip before --draft=false")

        step = job[check[0]]
        self.assertIn(step_if(step), {None, step_if(job[publish[0]])},
                      "the check may only be skipped when the publish is")
        self.assertIn("XybridFFI-${TAG}.xcframework.zip", step)
        self.assertIn("pin-ios-checksum.sh --check", step)
        self.assertIn("gh release download", step)
        first_check = min(step.index("sync-spm-checksum.sh --check"),
                          step.index("pin-ios-checksum.sh --check"))
        # The draft's zip, judged by the pins the tag will carry (not master's).
        self.assertLess(step.index("gh release download"), first_check)
        merge = re.search(r"git checkout .*\"\$MERGE_SHA\"", step)
        self.assertIsNotNone(merge, "check out the merge commit before the --check scripts")
        self.assertLess(merge.start(), first_check)
        self.assertEqual(len(re.findall(r"\bgit (?:checkout|switch|reset|restore)\b", step)), 1,
                         "nothing may move the tree off the merge commit")
        # Both scripts run so both report; a failure in either must still fail.
        self.assertEqual(re.findall(r"\bok=true\b", step), ["ok=true"], "only the initial ok=true")
        self.assertLess(step.index("ok=true"), first_check)


if __name__ == "__main__":
    unittest.main()
