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
    "publish-web-npm", "publish-observability-npm", "publish-swift-branch",
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


class ObservabilityReleaseTests(unittest.TestCase):
    def setUp(self):
        self.jobs = workflow_jobs()
        self.prep = (ROOT / ".github/workflows/release-prep.yml").read_text()

    def test_routine_pr_ci_runs_package_regressions_before_wasm_build(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        web = workflow.split("  web:\n", 1)[1].split("\n  # All checks passed", 1)[0]
        self.assertIn("run: pnpm test:release", web)
        self.assertLess(web.index("run: pnpm test:release"), web.index("run: pnpm build"))
        observability = workflow.split("  observability:\n", 1)[1].split("\n  web:\n", 1)[0]
        self.assertIn("pnpm pack:release", observability)
        self.assertNotRegex(observability, r"(?m)^\s+pnpm build$")

    def test_observability_crate_readme_is_packaged(self):
        directory = ROOT / "crates/xybrid-observability"
        manifest = tomllib.loads((directory / "Cargo.toml").read_text())
        self.assertTrue((directory / manifest["package"]["readme"]).is_file())
        self.assertIn("README.md", manifest["package"]["include"])

    def test_prep_gates_draft_on_both_validated_checksummed_archives(self):
        build = self.prep.split("  build-web:\n", 1)[1].split("\n  # ====", 1)[0]
        self.assertIn("working-directory: bindings/observability-js", build)
        self.assertIn("pnpm pack:release", build)
        self.assertIn("XYBRID_OBSERVABILITY_ARCHIVE:", build)
        self.assertIn("name: observability-npm", build)
        self.assertIn('sha256sum "xybrid-observability-${VERSION}.tgz"', build)
        self.assertIn('sha256sum "xybrid-web-${VERSION}.tgz"', build)
        self.assertLess(build.index("Test and pack the Observability SDK"),
                        build.index("Pack and verify the Web SDK"))
        draft = self.prep.split("  create-draft-release:\n", 1)[1].split("\n  # ====", 1)[0]
        self.assertIn("build-web", needs(draft))
        self.assertIn("artifacts/*", draft)

    def test_publishers_validate_then_stage_original_release_archives(self):
        for name in ("observability", "web"):
            job = self.jobs[f"publish-{name}-npm"]
            with self.subTest(package=name):
                archive = f'xybrid-{name}-${{VERSION}}.tgz'
                self.assertIn("gh release download", job)
                self.assertIn(f'sha256sum --check "{archive}.sha256"', job)
                self.assertIn("node scripts/check-pack.mjs", job)
                stage = re.search(r"npm stage publish ([^\n]+)\n([^\n]+)", job)
                self.assertIsNotNone(stage)
                self.assertIn(archive, stage.group(1))
                self.assertIn("--ignore-scripts --provenance --access public", stage.group(2))
                self.assertLess(job.index("sha256sum --check"), job.index("node scripts/check-pack.mjs"))
                self.assertLess(job.index("node scripts/check-pack.mjs"), stage.start())
                self.assertIn(f'npm view "@xybrid/{name}@${{VERSION}}"', job)
                self.assertNotIn("pnpm build", job)
                self.assertIn('case "$VERSION" in *-*) DIST_TAG=next ;; esac', job)

    def test_web_staging_waits_for_public_observability(self):
        dependency = self.jobs["publish-observability-npm"]
        web = self.jobs["publish-web-npm"]
        self.assertIn("vars.NPM_WEB_PUBLISH_ENABLED == 'true'", dependency)
        self.assertIn("publish-observability-npm", needs(web))
        self.assertIn('npm view "@xybrid/observability@${VERSION}" version', web)
        self.assertIn('if [ "$dependency_ready" != true ]; then', web)
        self.assertLess(web.index('if [ "$dependency_ready" != true ]; then'),
                        web.index("npm stage publish"))
        self.assertIn('xybrid-observability-${VERSION}.tgz"', web)


if __name__ == "__main__":
    unittest.main()
