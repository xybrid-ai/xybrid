"""Exercise release discovery and the gates preventing an unsafe release cut."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "llamacpp_update.py"
SPEC = importlib.util.spec_from_file_location("llamacpp_update", SCRIPT)
update = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update)

OLD = "a" * 40
NEW = "b" * 40
TAG_OBJECT = "c" * 40
REAL_RUN = update.run


class ReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.write("Cargo.toml", '[workspace]\nmembers = ["crates/example"]\n\n'
                   '[workspace.package]\nversion = "0.10.1"\n')
        self.write("crates/example/Cargo.toml", '[package]\nname = "example"\n'
                   'version.workspace = true\n')
        self.write(str(update.SYS / "build.rs"),
                   f'//! Pinned upstream commit `{OLD}`.\nconst LLAMA_CPP_COMMIT: &str = "{OLD}";\n')
        self.write(str(update.SYS / "wrapper.cpp"), "// shim\n")
        self.write(str(update.SYS / "wrapper.h"), "// header\n")
        self.write("CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n\n### Changed\n\n- Existing change.\n\n## [0.10.1]\n")
        self.write("bindings/flutter/CHANGELOG.md", "# Changelog\n\n## Unreleased\n\n* Existing fix.\n\n## 0.10.1\n")
        self.track(None)
        self.manifest()
        self.git("init", "--quiet", "--initial-branch=master")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.git("add", ".")
        self.git("update-index", "--add", "--cacheinfo", f"160000,{OLD},{update.VENDOR}")
        (self.root / update.VENDOR).mkdir(parents=True)
        self.git("commit", "--quiet", "-m", "fixture")
        self.pending_updates = []
        self.release_prs = []
        self.remote_refs = ""
        self.commands = []
        self.checks = [{"id": index, "name": name, "status": "completed", "conclusion": "success"}
                       for index, name in enumerate(sorted(update.REQUIRED_CHECKS), 1)]
        self.release = {"tag_name": "v0.5.0", "draft": False, "prerelease": False,
                        # Deliberately wrong: only the peeled tag is authoritative.
                        "target_commitish": "master"}
        self.releases = [self.release]
        self.api_requests = []
        self.release_cursors = []
        self.comparison = "ahead"
        self.patcher = patch.object(update, "run", side_effect=self.fake_run)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def write(self, filename: str, text: str) -> None:
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=self.root, text=True,
                                       stderr=subprocess.DEVNULL).strip()

    def track(self, tag: str | None, sdk: str = "0.10.1", kind: str = "minor") -> None:
        self.write(str(update.TRACKING), json.dumps({
            "tag": tag, "commit": OLD, "sdk_base_version": sdk if tag else None,
            "update_kind": kind if tag else None,
            "sdk_release": None,
        }))

    def manifest(self, missing: tuple | None = None) -> None:
        fields = ["version 1", "registry ghcr.io/xybrid-ai/llama-natives", f"llama_commit {OLD}"]
        for field, filename in (("wrapper_cpp", "wrapper.cpp"), ("wrapper_h", "wrapper.h"), ("build_rs", "build.rs")):
            digest = hashlib.sha256((self.root / update.SYS / filename).read_bytes()).hexdigest()
            fields.append(f"{field} {digest}")
        for target in sorted(update.NATIVE_TARGETS):
            for feature in ("base", "vision"):
                if (target, feature) != missing:
                    fields.append(f"slice {target} {feature} sha256:{'d' * 64}")
        self.write(str(update.SYS / "natives-manifest.txt"), "\n".join(fields) + "\n")

    def fake_run(self, args: list[str], root: Path = None, **kwargs) -> str:
        self.commands.append(args)
        if args[:3] == ["gh", "pr", "list"]:
            return json.dumps(self.pending_updates if "url,headRefName" in args else self.release_prs)
        if args[:2] == ["git", "ls-remote"]:
            return self.remote_refs
        return REAL_RUN(args, root or self.root, **kwargs)

    def api(self, endpoint: str, **fields):
        self.api_requests.append(endpoint)
        if endpoint == "graphql":
            self.assertEqual(fields["owner"], "ggml-org")
            self.assertEqual(fields["name"], "llama.cpp")
            cursor = fields.get("cursor")
            self.release_cursors.append(cursor)
            start = 0 if cursor is None else int(cursor.removeprefix("cursor-"))
            batch = self.releases[start:start + 100]
            return {"data": {"repository": {"releases": {
                "nodes": [{"tagName": release.get("tag_name"),
                           "isDraft": release.get("draft"),
                           "isPrerelease": release.get("prerelease")} for release in batch],
                "pageInfo": {"hasNextPage": start + len(batch) < len(self.releases),
                             "endCursor": f"cursor-{start + len(batch)}" if batch else None},
            }}}}
        if "/git/ref/tags/" in endpoint:
            return {"object": {"type": "tag", "sha": TAG_OBJECT}}
        if endpoint.endswith(f"/git/tags/{TAG_OBJECT}"):
            return {"object": {"type": "commit", "sha": NEW}}
        if "/compare/" in endpoint:
            self.assertIn(f"{OLD}...{NEW}", endpoint)
            return {"status": self.comparison}
        if "/check-runs?" in endpoint:
            self.assertIn(self.git("rev-parse", "HEAD"), endpoint)
            return {"total_count": len(self.checks), "check_runs": self.checks}
        self.fail(f"Unexpected API endpoint: {endpoint}")

    def discover(self):
        return update.discover(self.root, "xybrid-ai/xybrid", self.api)

    def plan(self, selected: str | None = None):
        return update.release_plan(self.root, "xybrid-ai/xybrid", selected, self.api)

    def test_annotated_tag_is_peeled_and_untagged_pin_bootstraps(self):
        candidate = self.discover()
        self.assertTrue(candidate["update"])
        self.assertEqual(candidate["commit"], NEW)
        self.assertEqual(candidate["update_kind"], "bootstrap")

    def test_development_prerelease_draft_and_malformed_releases_are_ignored(self):
        for tag, prerelease, draft in [("b12345", False, False), ("v0.5.0-rc1", True, False),
                                       ("v0.5.0", True, False), ("v0.5.0", False, True),
                                       ("v00.5.0", False, False)]:
            with self.subTest(tag=tag, prerelease=prerelease, draft=draft):
                self.release.update(tag_name=tag, prerelease=prerelease, draft=draft)
                candidate = self.discover()
                self.assertFalse(candidate["update"])
                self.assertIn("No published stable", candidate["reason"])

    def test_development_builds_do_not_hide_a_stable_release_on_a_later_page(self):
        stable = dict(self.release)
        self.release["tag_name"] = "b20000"
        self.releases = [{"tag_name": f"b{20000 - index}", "draft": False, "prerelease": False}
                         for index in range(100)] + [stable]
        candidate = self.discover()
        self.assertTrue(candidate["update"])
        self.assertEqual(candidate["tag"], "v0.5.0")
        self.assertEqual(candidate["commit"], NEW)
        self.assertEqual(self.release_cursors, [None, "cursor-100"])
        self.assertNotIn(f"repos/{update.UPSTREAM}/releases/latest", self.api_requests)

    def test_highest_stable_version_is_selected_across_all_pages(self):
        self.releases = [dict(self.release, tag_name="v0.9.0")]
        self.releases += [dict(self.release, tag_name=f"b{index}") for index in range(99)]
        self.releases += [dict(self.release, tag_name="v0.10.0"),
                          dict(self.release, tag_name="v1.0.0", prerelease=True),
                          dict(self.release, tag_name="v2.0.0", draft=True)]
        candidate = self.discover()
        self.assertTrue(candidate["update"])
        self.assertEqual(candidate["tag"], "v0.10.0")

    def test_cursor_pagination_continues_past_a_rest_history_limit(self):
        self.releases = [dict(self.release, tag_name="v0.9.0")]
        self.releases += [dict(self.release, tag_name=f"b{index}") for index in range(10_000)]
        self.releases += [dict(self.release, tag_name="v0.10.0")]
        prefix = f"repos/{update.UPSTREAM}/releases?per_page=100&page="
        def fetch(endpoint, **fields):
            if endpoint.startswith(prefix):
                page = int(endpoint.removeprefix(prefix))
                if page > 100:
                    raise subprocess.CalledProcessError(1, ["gh", "api", endpoint],
                                                        output='{"message": "Only the first 10000 results are available."}')
                start = (page - 1) * 100
                return self.releases[start:start + 100]
            return self.api(endpoint, **fields)
        candidate = update.discover(self.root, "xybrid-ai/xybrid", fetch)
        self.assertTrue(candidate["update"])
        self.assertEqual(candidate["tag"], "v0.10.0")
        self.assertIn("cursor-10000", self.release_cursors)

    def test_full_final_cursor_page_does_not_request_another_page(self):
        self.releases += [dict(self.release, tag_name=f"b{index}") for index in range(99)]
        candidate = self.discover()
        self.assertTrue(candidate["update"])
        self.assertEqual(candidate["tag"], "v0.5.0")
        self.assertEqual(self.release_cursors, [None])

    def test_missing_or_repeated_release_cursor_blocks_discovery(self):
        self.releases *= 2
        for cursor in (None, "", "cursor-1"):
            with self.subTest(cursor=cursor):
                def fetch(endpoint, **fields):
                    result = self.api(endpoint, **fields)
                    if endpoint == "graphql":
                        result["data"]["repository"]["releases"]["pageInfo"] = {
                            "hasNextPage": True, "endCursor": cursor,
                        }
                    return result
                with self.assertRaisesRegex(update.UpdateError, "cursor"):
                    update.discover(self.root, "xybrid-ai/xybrid", fetch)

    def test_no_stable_release_is_a_successful_noop(self):
        for count in (0, 100):
            with self.subTest(release_count=count):
                self.releases = [dict(self.release, tag_name=f"b{index}") for index in range(count)]
                self.api_requests.clear()
                self.release_cursors.clear()
                candidate = self.discover()
                self.assertFalse(candidate["update"])
                self.assertIn("No published stable", candidate["reason"])
                self.assertNotIn("branch", candidate)
                self.assertEqual(self.api_requests, ["graphql"])
                self.assertEqual(self.release_cursors, [None])
                self.assertFalse(any(args[0] == "gh" for args in self.commands))

    def test_release_history_api_failure_is_not_a_successful_noop(self):
        def fail(endpoint, **fields):
            raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
        with self.assertRaises(subprocess.CalledProcessError):
            update.discover(self.root, "xybrid-ai/xybrid", fail)

    def test_tag_tree_is_not_treated_as_commit(self):
        with self.assertRaises(update.UpdateError):
            update.resolve_tag("v0.5.0", lambda _: {"object": {"type": "tree", "sha": NEW}})

    def test_upstream_update_kind_and_downgrade(self):
        for old, new, kind in [("v0.4.0", "v0.4.1", "patch"), ("v0.4.1", "v0.5.0", "minor"),
                               ("v0.5.0", "v1.0.0", "major")]:
            with self.subTest(old=old, new=new):
                self.assertEqual(update.update_kind(old, new), kind)
        for tag in ("v0.4.9", "v0.5.0"):
            with self.assertRaises(update.UpdateError):
                update.update_kind("v0.5.0", tag)

    def test_same_commit_does_not_open_an_update(self):
        def fetch(endpoint, **fields):
            if endpoint.endswith(f"/git/tags/{TAG_OBJECT}"):
                return {"object": {"type": "commit", "sha": OLD}}
            return self.api(endpoint, **fields)
        self.assertFalse(update.discover(self.root, "xybrid-ai/xybrid", fetch)["update"])

    def test_stable_release_behind_or_divergent_from_pin_is_skipped(self):
        for status in ("behind", "diverged", "identical"):
            with self.subTest(status=status):
                self.comparison = status
                self.assertFalse(self.discover()["update"])

    def test_pending_review_is_preserved(self):
        self.pending_updates = [{"url": "https://github.com/xybrid-ai/xybrid/pull/1", "headRefName": "chore/llamacpp-v0.4.0"}]
        candidate = self.discover()
        self.assertFalse(candidate["update"])
        self.assertIn("under review", candidate["reason"])
        self.assertFalse(any(args[:2] == ["git", "push"] for args in self.commands))

    def test_closed_pr_branch_prevents_reopening_rejected_update(self):
        self.remote_refs = f"{OLD}\trefs/heads/{update.UPDATE_BRANCH_PREFIX}v0.5.0"
        self.assertFalse(self.discover()["update"])

    def test_pin_drift_is_an_error(self):
        self.write(str(update.SYS / "build.rs"), f'const LLAMA_CPP_COMMIT: &str = "{NEW}";\n')
        with self.assertRaises(update.UpdateError):
            update.pin(self.root)

    def test_apply_keeps_gitlink_fallback_and_tracking_in_sync(self):
        candidate = self.discover()
        def commands(args, root=self.root, **kwargs):
            if args[:3] == ["git", "submodule", "update"] or args[:2] in (["git", "fetch"], ["git", "checkout"]):
                return ""
            if args == ["git", "rev-parse", "HEAD"] and root == self.root / update.VENDOR:
                return NEW
            if args == ["git", "add", "--", str(update.VENDOR)]:
                return self.git("update-index", "--cacheinfo", f"160000,{NEW},{update.VENDOR}")
            return self.fake_run(args, root, **kwargs)
        with patch.object(update, "run", side_effect=commands):
            update.apply(self.root, candidate)
        self.assertEqual(update.pin(self.root)["commit"], NEW)
        self.assertEqual(update.pin(self.root)["tag"], "v0.5.0")
        self.assertNotIn(OLD, (self.root / update.SYS / "build.rs").read_text())
        self.assertIn(f"llama_commit {OLD}", (self.root / update.SYS / "natives-manifest.txt").read_text())

    def test_stale_candidate_is_rejected_before_fetching(self):
        candidate = self.discover()
        candidate["previous_commit"] = NEW
        with self.assertRaisesRegex(update.UpdateError, "no longer matches"):
            update.apply(self.root, candidate)
        self.assertFalse(any(args[:2] == ["git", "fetch"] for args in self.commands))

    def test_failed_native_build_keeps_the_committed_binding_snapshot(self):
        snapshot = self.root / update.SYS / "src/bindings.rs"
        self.write(str(update.SYS / "src/bindings.rs"), "// committed snapshot\n")
        def commands(args, root=self.root, **kwargs):
            if args[:2] == ["cargo", "check"]:
                self.assertEqual(kwargs["env"]["XYBRID_NATIVES_FORCE_SOURCE"], "1")
                self.assertNotIn("XYBRID_NATIVES_PREBUILT_DIR", kwargs["env"])
                raise subprocess.CalledProcessError(1, args)
            return self.fake_run(args, root, **kwargs)
        with patch.object(update, "run", side_effect=commands):
            with self.assertRaises(subprocess.CalledProcessError):
                update.bindings(self.root, check=False)
        self.assertEqual(snapshot.read_text(), "// committed snapshot\n")

    def test_untracked_bootstrap_does_not_prepare_release(self):
        self.assertFalse(self.plan()["ready"])

    def test_green_checks_and_matching_slices_prepare_next_minor(self):
        self.track("v0.5.0")
        plan = self.plan()
        self.assertTrue(plan["ready"], plan["reasons"])
        self.assertEqual(plan["version"], "0.11.0")
        self.assertEqual(plan["branch"], "release/v0.11.0")

    def test_sdk_version_bump_alone_does_not_mark_upstream_update_released(self):
        self.track("v0.5.0", sdk="0.9.0")
        self.assertTrue(self.plan()["ready"])

    def test_recorded_release_of_current_pin_prevents_duplicate_cut(self):
        self.track("v0.5.0")
        state = update.pin(self.root)
        state["sdk_release"] = {"version": "0.10.1", "commit": OLD}
        self.write(str(update.TRACKING), json.dumps(state))
        self.assertFalse(self.plan()["ready"])

    def test_release_of_different_pin_does_not_mask_pending_update(self):
        self.track("v0.5.0", sdk="0.9.0")
        state = update.pin(self.root)
        state["sdk_release"] = {"version": "0.10.1", "commit": NEW}
        self.write(str(update.TRACKING), json.dumps(state))
        self.assertTrue(self.plan()["ready"])

    def test_major_upstream_upgrade_needs_explicit_release_version(self):
        self.track("v1.0.0", kind="major")
        self.assertFalse(self.plan()["ready"])
        self.assertTrue(self.plan("0.11.0")["ready"])

    def test_bootstrap_update_needs_explicit_release_version(self):
        for tag in ("v0.5.0", "v1.0.0"):
            with self.subTest(tag=tag):
                self.track(tag, kind=update.update_kind(None, tag))
                plan = self.plan()
                self.assertFalse(plan["ready"])
                self.assertNotIn("version", plan)
                self.assertNotIn("branch", plan)
                self.assertTrue(any("sdk_version" in reason for reason in plan["reasons"]))
                selected = self.plan("0.11.0")
                self.assertTrue(selected["ready"], selected["reasons"])
                self.assertEqual(selected["version"], "0.11.0")
                self.assertEqual(selected["branch"], "release/v0.11.0")

    def test_sdk_one_requires_explicit_version_and_poc_rejects_zero_patch(self):
        self.write("Cargo.toml", '[workspace.package]\nversion = "1.2.3"\n')
        self.track("v0.5.0", sdk="1.2.3")
        self.assertFalse(self.plan()["ready"])
        self.assertTrue(self.plan("2.0.0")["ready"])
        self.write("Cargo.toml", '[workspace.package]\nversion = "0.10.1"\n')
        self.track("v0.5.0")
        with self.assertRaises(update.UpdateError):
            self.plan("0.10.2")

    def test_sdk_prerelease_can_track_update_but_does_not_cut_another_release(self):
        self.write("Cargo.toml", '[workspace.package]\nversion = "0.11.0-rc1"\n')
        self.track("v0.5.0", sdk="0.11.0-rc1")
        self.assertEqual(update.pin(self.root)["sdk_base_version"], "0.11.0-rc1")
        self.assertFalse(self.plan()["ready"])

    def test_stale_wrapper_or_pin_blocks_release(self):
        self.track("v0.5.0")
        self.write(str(update.SYS / "wrapper.cpp"), "// edited shim\n")
        plan = self.plan()
        self.assertFalse(plan["ready"])
        self.assertIn("Native manifest has stale wrapper_cpp", plan["reasons"])

    def test_partial_native_matrix_blocks_release(self):
        self.track("v0.5.0")
        self.manifest(missing=("aarch64-apple-ios", "vision"))
        plan = self.plan()
        self.assertFalse(plan["ready"])
        self.assertTrue(any("aarch64-apple-ios/vision" in reason for reason in plan["reasons"]))

    def test_latest_check_failure_cannot_be_masked_by_old_success(self):
        self.track("v0.5.0")
        self.checks.append({"id": 1000, "name": "CI Success", "status": "completed", "conclusion": "failure"})
        self.assertFalse(self.plan()["ready"])

    def test_missing_pending_or_skipped_required_check_blocks_release(self):
        self.track("v0.5.0")
        self.checks = [check for check in self.checks if check["name"] != "CI Success"]
        for status, conclusion in [("in_progress", None), ("completed", "skipped")]:
            with self.subTest(status=status, conclusion=conclusion):
                self.checks = [check for check in self.checks if check["name"] != "CI Success"]
                self.checks.append({"id": 100, "name": "CI Success", "status": status, "conclusion": conclusion})
                self.assertFalse(self.plan()["ready"])

    def test_other_failed_checks_block_but_automation_does_not_wait_on_itself(self):
        self.track("v0.5.0")
        self.checks.append({"id": 100, "name": "Prepare llama.cpp release", "status": "in_progress"})
        self.assertTrue(self.plan()["ready"])
        self.checks.append({"id": 101, "name": "Android build", "status": "completed", "conclusion": "failure"})
        self.assertFalse(self.plan()["ready"])

    def test_open_release_and_existing_branch_prevent_duplicate_cut(self):
        self.track("v0.5.0")
        self.release_prs = [{"headRefName": "release/v0.11.0"}]
        self.assertFalse(self.plan()["ready"])
        self.release_prs = []
        self.remote_refs = f"{NEW}\trefs/heads/release/v0.11.0"
        self.assertFalse(self.plan()["ready"])

    def test_inflight_release_without_pr_blocks_but_historical_branch_does_not(self):
        self.track("v0.5.0")
        self.remote_refs = f"{NEW}\trefs/heads/release/v0.12.0-rc1"
        self.assertFalse(self.plan()["ready"])
        self.remote_refs = f"{OLD}\trefs/heads/release/v0.9.0"
        self.assertTrue(self.plan()["ready"])

    def test_changelog_update_preserves_existing_changes_and_release_history(self):
        candidate = self.discover()
        update.add_changelog(self.root, candidate)
        text = (self.root / "CHANGELOG.md").read_text()
        self.assertIn("Existing change.", text)
        self.assertIn(f"commit `{NEW}`", text)
        self.assertIn("## [0.10.1]", text)

    def test_internal_path_constraints_update_without_touching_external_deps(self):
        self.write("examples/demo/Cargo.toml", '[dependencies]\n'
                   'example = { path = "../../crates/example", version = "0.10.1" }\n'
                   'renamed = { package = "example", version = "0.10.1", path = "../../crates/example" }\n'
                   'other = { path = "../other", version = "1.0" }\n'
                   'serde = { version = "1.0" }\n')
        self.git("add", "examples/demo/Cargo.toml")
        update.bump_internal_pins(self.root, "0.11.0")
        text = (self.root / "examples/demo/Cargo.toml").read_text()
        self.assertEqual(text.count('version = "0.11.0"'), 2)
        self.assertEqual(text.count('version = "1.0"'), 2)

    def test_promoting_changelogs_includes_all_unreleased_changes(self):
        update.promote_changelogs(self.root, "0.11.0")
        root_text = (self.root / "CHANGELOG.md").read_text()
        flutter = (self.root / "bindings/flutter/CHANGELOG.md").read_text()
        self.assertLess(root_text.index("## [0.11.0]"), root_text.index("Existing change."))
        self.assertLess(flutter.index("## 0.11.0"), flutter.index("Existing fix."))

    def test_unready_or_stale_release_plan_never_pushes(self):
        for plan in ({"ready": False}, {"ready": True, "head": NEW}):
            with self.assertRaises(update.UpdateError):
                update.prepare_release(self.root, plan)
        self.assertFalse(any(args[:2] == ["git", "push"] for args in self.commands))

    def prepare_fixture(self):
        self.track("v0.5.0")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "tracked upstream")
        return self.plan()

    def release_commands(self, head: str):
        def execute(args, root=self.root, **kwargs):
            self.commands.append(args)
            if args[0].endswith("version-sync.sh"):
                if args[1] != "--check":
                    path = self.root / "Cargo.toml"
                    path.write_text(path.read_text().replace('version = "0.10.1"', f'version = "{args[1]}"'))
                return ""
            if args[0].endswith("set-natives-mode.sh") or args[:2] == ["cargo", "metadata"]:
                return ""
            if args[:2] == ["git", "ls-remote"]:
                return f"{head}\trefs/heads/master"
            if args[:2] == ["git", "push"]:
                return ""
            return REAL_RUN(args, root, **kwargs)
        return execute

    def test_release_preparation_bumps_and_pushes_only_release_branch(self):
        plan = self.prepare_fixture()
        self.assertTrue(plan["ready"])
        with patch.object(update, "run", side_effect=self.release_commands(plan["head"])):
            update.prepare_release(self.root, plan)
        self.assertEqual(self.git("branch", "--show-current"), "release/v0.11.0")
        self.assertEqual(update.sdk_version(self.root), "0.11.0")
        self.assertEqual(update.pin(self.root)["sdk_release"], {"version": "0.11.0", "commit": OLD})
        self.assertIn("## [0.11.0]", (self.root / "CHANGELOG.md").read_text())
        pushes = [args for args in self.commands if args[:2] == ["git", "push"]]
        self.assertEqual(pushes, [["git", "push", "--set-upstream", "origin", "release/v0.11.0"]])
        self.assertFalse(any(args[:2] == ["git", "tag"] for args in self.commands))

    def test_master_advancing_during_generation_prevents_push(self):
        plan = self.prepare_fixture()
        with patch.object(update, "run", side_effect=self.release_commands(NEW)):
            with self.assertRaisesRegex(update.UpdateError, "master advanced"):
                update.prepare_release(self.root, plan)
        self.assertFalse(any(args[:2] == ["git", "push"] for args in self.commands))

    def test_dirty_checkout_prevents_release_changes(self):
        plan = self.prepare_fixture()
        self.write("unexpected.txt", "user edit\n")
        with self.assertRaisesRegex(update.UpdateError, "clean checkout"):
            update.prepare_release(self.root, plan)
        self.assertEqual(self.git("branch", "--show-current"), "master")

    def test_recording_release_requires_the_release_branch(self):
        self.track("v0.5.0")
        with self.assertRaisesRegex(update.UpdateError, "release branch"):
            update.record_release(self.root, "0.10.1")
        self.assertIsNone(update.pin(self.root)["sdk_release"])


if __name__ == "__main__":
    unittest.main()
