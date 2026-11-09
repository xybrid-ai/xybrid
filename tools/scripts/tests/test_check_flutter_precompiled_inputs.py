"""Exercise the guard against releasing stale Flutter precompiled libraries."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "check_flutter_precompiled_inputs.py"
SPEC = importlib.util.spec_from_file_location("check_flutter_precompiled_inputs", SCRIPT)
guard = importlib.util.module_from_spec(SPEC)
# dataclasses resolves the module's string annotations through sys.modules.
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)

CARGO_TOML = "bindings/flutter/rust/Cargo.toml"
FLUTTER_SRC = "bindings/flutter/rust/src/lib.rs"
CORE_SRC = "crates/xybrid-core/src/lib.rs"
LLAMA = "vendor/llama-cpp"
# Commit dates for the tests that compare them with a release's.
BUMPED = 1_800_000_000
PUBLISHED = BUMPED + 600
LATER = BUMPED + 3600


def cargo_toml(version: str, note: str = "") -> str:
    return f'[package]\nname = "xybrid_flutter"\nversion = "{version}"\n{note}'


def stamp(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PathRulesTests(unittest.TestCase):
    def test_native_inputs_are_tracked(self):
        for path in (CORE_SRC, "crates/llama-cpp-sys/build.rs", "crates/zzz-sys/natives-manifest.json",
                     "macros/src/lib.rs", LLAMA, "vendor/ort-android/arm64-v8a/libonnxruntime.so",
                     ".gitmodules", "Cargo.toml", "Cargo.lock", "MODULE.bazel", "MODULE.bazel.lock",
                     "BUILD.bazel", ".bazelrc", ".bazelversion", "rules_rs.patch", "bazel/ggml/android.bzl",
                     ".cargo/config.toml", "bindings/flutter/rust/BUILD.bazel", FLUTTER_SRC,
                     "bindings/flutter/cargokit/build_tool/lib/src/android_environment.dart"):
            with self.subTest(path=path):
                self.assertTrue(guard.feeds_libraries(path))

    def test_every_workspace_manifest_is_tracked(self):
        # Bazel resolves third-party crates once for the whole workspace, with
        # every member's features merged into what the library links.
        for path in ("integration-tests/Cargo.toml", "xtask/Cargo.toml", "bindings/web/runtime/Cargo.toml",
                     "crates/xybrid-core/Cargo.toml", "macros/Cargo.toml"):
            with self.subTest(path=path):
                self.assertTrue(guard.feeds_libraries(path))

    def test_files_no_build_reads_are_ignored(self):
        for path in ("README.md", "crates/xybrid-sdk/README.md", "docs/content/docs/quickstart.mdx",
                     ".github/workflows/release-prep.yml", ".github/llamacpp-version.json",
                     guard.NATIVES_MANIFEST, "crates/xybrid-core/tests/system_validation.rs",
                     "crates/xybrid-core/benches/pipeline.rs", "crates/xybrid-sdk/examples/run.rs",
                     "macros/tests/expand.rs", "Package.swift", "bindings/flutter/lib/xybrid.dart",
                     "bindings/flutter/cargokit/build_pod.sh", "bindings/kotlin/BUILD.bazel",
                     "integration-tests/BUILD.bazel", "integration-tests/src/lib.rs", "xtask/src/main.rs",
                     "tools/scripts/BUILD.bazel"):
            with self.subTest(path=path):
                self.assertFalse(guard.feeds_libraries(path))


class GitFixture(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.root = Path(directory.name).resolve()
        # Real Git calls, including the script's, use a controlled environment
        # rather than host config, hooks or credentials.
        environment = {"PATH": os.defpath, "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
                       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                       "GIT_CONFIG_COUNT": "2",
                       "GIT_CONFIG_KEY_0": "maintenance.auto", "GIT_CONFIG_VALUE_0": "false",
                       "GIT_CONFIG_KEY_1": "gc.auto", "GIT_CONFIG_VALUE_1": "0"}
        environment_patch = patch.dict(os.environ, environment, clear=True)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        self.addCleanup(directory.cleanup)
        self.git("init", "--quiet", "--initial-branch=master")
        self.git("config", "core.hooksPath", os.devnull)
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        (self.root / LLAMA).mkdir(parents=True)
        self.commit("initial", {
            CARGO_TOML: cargo_toml("0.1.0"),
            FLUTTER_SRC: "pub fn api() {}\n",
            "bindings/flutter/rust/BUILD.bazel": "rust_shared_library()\n",
            CORE_SRC: "pub fn run() {}\n",
            "crates/xybrid-core/tests/smoke.rs": "#[test]\nfn smoke() {}\n",
            "integration-tests/Cargo.toml": '[dependencies]\ntokio = { version = "1", features = ["rt"] }\n',
            guard.NATIVES_MANIFEST: "version 1\n",
            "README.md": "# xybrid\n",
            ".github/workflows/release-prep.yml": "name: Release\n",
        }, gitlink="a" * 40)

    def git(self, *args: str, date: int | None = None) -> str:
        environment = dict(os.environ)
        if date is not None:
            environment["GIT_AUTHOR_DATE"] = environment["GIT_COMMITTER_DATE"] = f"@{date} +0000"
        return subprocess.check_output(["git", *args], cwd=self.root, text=True, env=environment,
                                       stderr=subprocess.DEVNULL).strip()

    def commit(self, message: str, files: dict[str, str | bytes] | None = None,
               gitlink: str | None = None, date: int | None = None) -> str:
        for name, content in (files or {}).items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content)
            self.git("add", "--", name)
        if gitlink:
            self.git("update-index", "--add", "--cacheinfo", f"160000,{gitlink},{LLAMA}")
        self.git("commit", "--quiet", "-m", message, date=date)
        return self.git("rev-parse", "HEAD")

    def cut_release(self, date: int | None = None) -> str:
        self.git("switch", "--quiet", "-c", "release/v0.2.0")
        return self.commit("bump: 0.2.0", {CARGO_TOML: cargo_toml("0.2.0")}, date=date)

    def merge_master(self, date: int | None = None) -> str:
        self.git("merge", "--quiet", "--no-ff", "--no-edit", "master", date=date)
        return self.git("rev-parse", "HEAD")

    def run_main(self, *args: str) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = guard.main(["--repo", str(self.root), *args])
        return status, output.getvalue()


class HistoryTests(GitFixture):
    def test_native_change_after_the_bump_fails(self):
        bump = self.cut_release()
        change = self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"})
        result = guard.check(self.root)
        self.assertEqual(result.origins, (bump,))
        self.assertEqual(result.stale, (CORE_SRC,))
        self.assertEqual(result.culprits, (change,))
        status, output = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn(f"precompiled_{result.crate_hash}", output)
        self.assertIn(f"  {CORE_SRC}\n", output)
        self.assertIn("bump: 0.2.0", output)
        self.assertIn("If it already holds libraries", output)
        self.assertIn(f"change the text of a line in {CARGO_TOML}", output)

    def test_vendored_llama_cpp_update_after_the_bump_fails(self):
        self.cut_release()
        self.commit("chore(llama): update", gitlink="b" * 40)
        self.assertEqual(guard.check(self.root).stale, (LLAMA,))

    def test_workspace_member_manifest_change_after_the_bump_fails(self):
        self.cut_release()
        self.commit("test: more tokio", {
            "integration-tests/Cargo.toml": '[dependencies]\ntokio = { version = "1", features = ["full"] }\n'})
        self.assertEqual(guard.check(self.root).stale, ("integration-tests/Cargo.toml",))

    def test_touching_cargo_toml_after_a_native_change_passes(self):
        self.cut_release()
        self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"})
        touch = self.commit("fix(flutter): rebuild", {CARGO_TOML: cargo_toml("0.2.0", "# rebuilt\n")})
        result = guard.check(self.root)
        self.assertEqual((result.origins, result.stale, result.culprits), ((touch,), (), ()))
        status, output = self.run_main()
        self.assertEqual(status, 0)
        self.assertIn(f"OK: precompiled_{result.crate_hash}", output)
        self.assertIn("Without --github-repo", output)

    def test_native_changes_before_the_bump_pass(self):
        self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"})
        bump = self.cut_release()
        result = guard.check(self.root)
        self.assertEqual((result.origins, result.stale), ((bump,), ()))

    def test_docs_workflow_manifest_and_test_changes_pass(self):
        self.cut_release()
        self.commit("docs: notes", {"README.md": "# xybrid\n\nNotes.\n"})
        self.commit("ci: tweak", {".github/workflows/release-prep.yml": "name: Release Prep\n"})
        self.commit("chore(natives): refresh manifest", {guard.NATIVES_MANIFEST: "version 1\nslice x\n"})
        self.commit("test(core): deflake", {"crates/xybrid-core/tests/smoke.rs": "#[test]\nfn smoke() { }\n"})
        self.assertEqual(guard.check(self.root).stale, ())
        self.assertEqual(self.run_main()[0], 0)

    def test_blank_lines_and_a_byte_order_mark_do_not_move_the_hash(self):
        bump = self.cut_release()
        self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"})
        for content in (cargo_toml("0.2.0", "\n\n"), guard.UTF8_BOM + cargo_toml("0.2.0").encode()):
            with self.subTest(content=content):
                self.commit("fix(flutter): rebuild?", {CARGO_TOML: content})
                result = guard.check(self.root)
                self.assertEqual(result.origins, (bump,))
                self.assertEqual(result.stale, (CARGO_TOML, CORE_SRC))
                self.assertIn(f"  {CARGO_TOML}  (edited without moving the hash)", self.run_main()[1])

    def test_reverting_back_to_the_bumped_hash_fails(self):
        # The release may still hold what the bump built: precompiled_<hash>
        # is named after content, not after the commit that last changed it.
        bump = self.cut_release()
        change = self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"})
        self.commit("feat(flutter): new API", {FLUTTER_SRC: "pub fn api() { newer() }\n"})
        self.assertEqual(guard.check(self.root).stale, ())
        self.git("revert", "--no-edit", "HEAD")
        result = guard.check(self.root)
        self.assertEqual(result.origins, (bump,))
        self.assertEqual(result.stale, (CORE_SRC,))
        self.assertEqual(result.culprits, (change,))

    def test_reverting_a_native_change_still_fails(self):
        # A run may have built the release while the change was in (test-ci.yml
        # can precompile any commit), so only moving the hash settles it.
        self.cut_release()
        change = self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"})
        self.git("revert", "--no-edit", "HEAD")
        revert = self.git("rev-parse", "HEAD")
        result = guard.check(self.root)
        self.assertEqual((result.stale, result.culprits), ((CORE_SRC,), (change, revert)))

    def test_master_merge_bringing_a_native_change_fails(self):
        bump = self.cut_release()
        self.git("switch", "--quiet", "master")
        self.commit("chore(llama): update", {CORE_SRC: "pub fn run() { newer() }\n"}, gitlink="b" * 40)
        self.git("switch", "--quiet", "release/v0.2.0")
        merge = self.merge_master()
        result = guard.check(self.root)
        self.assertEqual(result.origins, (bump,))
        self.assertEqual(result.stale, (CORE_SRC, LLAMA))
        self.assertEqual(result.culprits, (merge,))

    def test_master_merge_bringing_a_flutter_src_change_passes(self):
        self.cut_release()
        self.git("switch", "--quiet", "master")
        self.commit("feat: native and Flutter API", {CORE_SRC: "pub fn run() { newer() }\n",
                                                    FLUTTER_SRC: "pub fn api() { newer() }\n"})
        self.git("switch", "--quiet", "release/v0.2.0")
        merge = self.merge_master()
        result = guard.check(self.root)
        self.assertEqual((result.origins, result.stale), ((merge,), ()))

    def test_release_branch_merged_into_master_fails(self):
        # The merge's first parent is master, so the bump the release was
        # built from is only on its second-parent side.
        bump = self.cut_release()
        self.git("switch", "--quiet", "master")
        self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"})
        self.git("merge", "--quiet", "--no-ff", "--no-edit", "release/v0.2.0")
        merge = self.git("rev-parse", "HEAD")
        result = guard.check(self.root)
        self.assertEqual(result.origins, (bump,))
        self.assertEqual(result.stale, (CORE_SRC,))
        self.assertEqual(result.culprits, (merge,))

    def test_crate_hash_follows_cargokit(self):
        # cargokit hashes src/ in path order ("a.rs" before "a/"), then the
        # crate files it names, each decoded (dropping a leading byte order
        # mark, keeping any other) and split into lines without line endings.
        self.commit("crate files", {
            "bindings/flutter/rust/src/a.rs": b"one\r\ntwo\n",
            "bindings/flutter/rust/src/a/b.rs": b"three\rfour",
            "bindings/flutter/rust/src/a/c.rs": guard.UTF8_BOM + b"bom\n",
            "bindings/flutter/rust/src/a/d.rs": b"in" + guard.UTF8_BOM + b"side\n",
            "bindings/flutter/rust/build.rs": b"fn main() {}\n",
            "bindings/flutter/rust/cargokit.yaml": b"cargo: {}\n",
        })
        lines = [b"onetwo", b"threefour", b"bom", b"in" + guard.UTF8_BOM + b"side", b"pub fn api() {}",
                 cargo_toml("0.1.0").replace("\n", "").encode(), b"fn main() {}", b"cargo: {}"]
        expected = hashlib.sha256(b"".join(lines)).hexdigest()[:32]
        inputs = guard.hash_inputs(self.root, "HEAD")
        blobs = guard.read_blobs(self.root, {oid for _, oid in inputs})
        self.assertEqual(guard.crate_hash(inputs, blobs), expected)
        self.assertEqual(guard.check(self.root).crate_hash, expected)

    def test_shallow_history_is_an_error(self):
        self.cut_release()
        self.commit("docs: notes", {"README.md": "# xybrid\n\nNotes.\n"})
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "clone"
            self.git("clone", "--quiet", "--depth", "1", self.root.as_uri(), str(clone))
            with self.assertRaisesRegex(guard.CheckError, "fetch-depth: 0"):
                guard.check(clone)

    def test_unknown_ref_is_an_error(self):
        status, output = self.run_main("no-such-ref")
        self.assertEqual(status, 2)
        self.assertIn("no-such-ref is not a commit", output)


class ReleaseLookupTests(GitFixture):
    """--github-repo, against a fake gh that answers one release lookup."""

    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.gh = Path(directory.name)
        script = self.gh / "gh"
        script.write_text(f'#!/bin/sh\necho "$*" > "{self.gh}/args"\ncat "{self.gh}/stdout"\n'
                          f'cat "{self.gh}/stderr" >&2\nexit "$(cat "{self.gh}/status")"\n')
        script.chmod(0o755)
        # GitFixture's environment patch restores PATH.
        os.environ["PATH"] = f"{self.gh}{os.pathsep}{os.environ['PATH']}"
        self.no_release()

    def answer(self, body: dict, status: int = 0, stderr: str = "") -> None:
        (self.gh / "stdout").write_text(json.dumps(body))
        (self.gh / "stderr").write_text(stderr)
        (self.gh / "status").write_text(str(status))

    def no_release(self) -> None:
        self.answer({"message": "Not Found", "status": "404"}, 1, "gh: Not Found (HTTP 404)\n")

    def release(self, published: int | None, *assets: int) -> None:
        # created_at is the date of the commit the release's tag points at.
        self.answer({"created_at": stamp(BUMPED - 86400), "published_at": published and stamp(published),
                     "assets": [{"name": f"asset{i}", "created_at": stamp(at)} for i, at in enumerate(assets)]})

    def lookup(self) -> tuple[int, str]:
        return self.run_main("--github-repo", "xybrid-ai/xybrid")

    def test_lookup_asks_for_the_release_named_after_the_crate_hash(self):
        self.cut_release(date=BUMPED)
        self.lookup()
        crate_hash = guard.check(self.root).crate_hash
        self.assertEqual((self.gh / "args").read_text().split(),
                         ["api", f"repos/xybrid-ai/xybrid/releases/tags/precompiled_{crate_hash}"])

    def test_release_published_takes_the_earliest_upload(self):
        self.release(PUBLISHED, PUBLISHED + 5, PUBLISHED + 900)
        self.assertEqual(guard.release_published("o/r", "t"), PUBLISHED)
        self.release(None, PUBLISHED + 5)
        self.assertEqual(guard.release_published("o/r", "t"), PUBLISHED + 5)
        # Nothing uploaded, or no release at all: every target builds afresh.
        self.release(PUBLISHED)
        self.assertIsNone(guard.release_published("o/r", "t"))
        self.no_release()
        self.assertIsNone(guard.release_published("o/r", "t"))

    def test_a_failed_lookup_is_an_error(self):
        self.cut_release(date=BUMPED)
        self.answer({"message": "Server Error"}, 1, "gh: Server Error (HTTP 502)\n")
        status, output = self.lookup()
        self.assertEqual(status, 2)
        self.assertIn("HTTP 502", output)

    def test_no_release_yet_passes(self):
        self.cut_release(date=BUMPED)
        status, output = self.lookup()
        self.assertEqual(status, 0)
        self.assertIn("It holds no libraries yet, so every precompile job builds them from HEAD", output)

    def test_a_native_change_fails_before_anything_is_uploaded(self):
        # This run would build from HEAD, but a later one could not tell its
        # uploads from libraries built at the bump.
        self.cut_release(date=BUMPED)
        self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"}, date=BUMPED + 60)
        status, output = self.lookup()
        self.assertEqual(status, 1)
        self.assertIn("It holds no libraries yet, so this run would build them from HEAD", output)

    def test_release_built_on_this_branch_passes(self):
        self.cut_release(date=BUMPED)
        self.release(PUBLISHED, PUBLISHED)
        self.commit("docs: notes", {"README.md": "# xybrid\n\nNotes.\n"}, date=LATER)
        self.assertEqual(self.lookup()[0], 0)

    def test_release_built_before_a_native_change_fails(self):
        self.cut_release(date=BUMPED)
        self.release(PUBLISHED, PUBLISHED)
        self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"}, date=LATER)
        status, output = self.lookup()
        self.assertEqual(status, 1)
        self.assertIn(f"It holds libraries since {guard.utc(PUBLISHED)}", output)
        self.assertIn(f"  {CORE_SRC}\n", output)

    def test_rebase_onto_a_native_change_fails(self):
        self.cut_release(date=BUMPED)
        self.release(PUBLISHED, PUBLISHED)
        self.git("switch", "--quiet", "master")
        self.commit("fix(core): native fix", {CORE_SRC: "pub fn run() { fixed() }\n"}, date=LATER)
        self.git("switch", "--quiet", "release/v0.2.0")
        self.git("rebase", "--quiet", "master", date=LATER + 60)
        self.assertEqual(guard.check(self.root).stale, ())
        self.assertEqual(self.run_main()[0], 0)
        status, output = self.lookup()
        self.assertEqual(status, 1)
        self.assertIn(f"That release was published {guard.utc(PUBLISHED)}", output)
        self.assertIn("bump: 0.2.0", output)

    def test_recut_with_the_same_bump_fails(self):
        self.cut_release(date=BUMPED)
        self.release(PUBLISHED, PUBLISHED)
        self.git("switch", "--quiet", "master")
        self.commit("chore(llama): update", gitlink="b" * 40, date=LATER)
        self.git("branch", "--quiet", "-D", "release/v0.2.0")
        self.cut_release(date=LATER + 60)
        self.assertEqual(guard.check(self.root).stale, ())
        self.assertEqual(self.lookup()[0], 1)


if __name__ == "__main__":
    unittest.main()
