"""Tests for clean.sh (repo root) and tools/scripts/clean-lib.sh.

Both delete directories, so what is worth pinning is what they refuse to
delete: anything git tracks or does not ignore, worktrees holding work that
exists nowhere else, and Bazel output bases whose workspace is still there.
Each test builds a throwaway repo holding copies of the two scripts; `gh`,
Bazel and the per-folder clean commands are stubs on PATH.

RepoCleanScriptsTests checks the real per-folder clean.sh files instead: every
path they list must be one git ignores, or clean-lib.sh would refuse it.
"""

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ROOT_SCRIPT = REPO / "clean.sh"
LIB = REPO / "tools" / "scripts" / "clean-lib.sh"


class CleanTestCase(unittest.TestCase):
    def setUp(self):
        # resolve(): git reports the real path (/private/var on macOS).
        self.temp = Path(tempfile.mkdtemp()).resolve()
        # chmod first: a test leaves a read-only tree behind when it fails.
        self.addCleanup(
            subprocess.run, f"chmod -R u+w '{self.temp}'; rm -rf '{self.temp}'", shell=True
        )
        self.bin = self.temp / "bin"
        self.bin.mkdir()
        # No `gh` answer by default: nothing counts as a merged PR.
        self.stub("gh", "exit 0")
        self.env = dict(
            os.environ,
            PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            GIT_CONFIG_GLOBAL="/dev/null",
            GIT_CONFIG_NOSYSTEM="1",
            GIT_AUTHOR_NAME="test",
            GIT_AUTHOR_EMAIL="test@example.com",
            GIT_COMMITTER_NAME="test",
            GIT_COMMITTER_EMAIL="test@example.com",
        )
        self.env.pop("CLEAN_KB_FILE", None)

        self.origin = self.temp / "origin.git"
        self.git(self.temp, "init", "-q", "--bare", "-b", "main", str(self.origin))
        self.repo = self.temp / "repo"
        self.git(self.temp, "clone", "-q", str(self.origin), str(self.repo))
        # No `.context/` here: Conductor ignores it through .git/info/exclude
        # only, so a fresh clone does not, and the scripts must not rely on it.
        write(self.repo / ".gitignore", "/target/\nnode_modules/\nbuild/\n*.so\n")
        write(self.repo / "src.txt")
        shutil.copy(ROOT_SCRIPT, self.repo / "clean.sh")
        (self.repo / "tools" / "scripts").mkdir(parents=True)
        shutil.copy(LIB, self.repo / "tools" / "scripts" / "clean-lib.sh")
        self.git(self.repo, "add", ".")
        self.git(self.repo, "commit", "-qm", "init")
        self.git(self.repo, "push", "-q", "origin", "HEAD:main")

    def git(self, cwd, *args):
        return subprocess.run(
            ["git", *args], cwd=cwd, env=self.env, check=True, capture_output=True, text=True
        ).stdout.strip()

    def stub(self, name, body):
        path = self.bin / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(0o755)

    def folder_script(self, folder, body):
        """Write <folder>/clean.sh sourcing the lib, with `body` as its declarations."""
        depth = len(Path(folder).parts)
        path = self.repo / folder / "clean.sh"
        write(
            path,
            "#!/usr/bin/env bash\n# Test folder.\n"
            f'. "$(dirname "$0")/{"../" * depth}tools/scripts/clean-lib.sh"\n'
            f'{body}\nclean_run "$@"\n',
        )
        path.chmod(0o755)
        # The root only runs clean scripts git tracks.
        self.git(self.repo, "add", str(path))
        return path

    def run_script(self, script, *args, **env):
        return subprocess.run(
            ["bash", str(script), *args],
            capture_output=True,
            text=True,
            env=dict(self.env, **env),
        )

    def run_root(self, *args, **env):
        return self.run_script(self.repo / "clean.sh", *args, **env)


def write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class FolderScriptTests(CleanTestCase):
    def test_dry_run_lists_outputs_and_deletes_nothing(self):
        script = self.folder_script("app", "clean_paths build")
        write(self.repo / "app" / "build" / "out.bin")

        result = self.run_script(script)

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("app/build", result.stdout)
        self.assertIn("Would free", result.stdout)
        self.assertTrue((self.repo / "app" / "build").is_dir())

    def test_apply_deletes_outputs(self):
        script = self.folder_script("app", "clean_paths build")
        write(self.repo / "app" / "build" / "out.bin")

        result = self.run_script(script, "--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse((self.repo / "app" / "build").exists())

    def test_apply_keeps_listed_paths_git_does_not_ignore(self):
        # A wrong entry: `src` is source, `notes` is untracked but not ignored.
        script = self.folder_script("app", "clean_paths src notes")
        write(self.repo / "app" / "src" / "main.c")
        self.git(self.repo, "add", "app/src")
        self.git(self.repo, "commit", "-qm", "src")
        write(self.repo / "app" / "notes" / "todo.md")

        result = self.run_script(script, "--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue((self.repo / "app" / "src" / "main.c").is_file())
        self.assertTrue((self.repo / "app" / "notes" / "todo.md").is_file())
        self.assertIn("app/src (git does not ignore it", result.stdout)

    def test_apply_keeps_ignored_dir_holding_a_tracked_file(self):
        script = self.folder_script("app", "clean_paths build")
        write(self.repo / "app" / "build" / "vendored.js")
        self.git(self.repo, "add", "-f", "app/build/vendored.js")
        self.git(self.repo, "commit", "-qm", "vendor")

        result = self.run_script(script, "--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue((self.repo / "app" / "build" / "vendored.js").is_file())

    def test_glob_reports_one_line_and_deletes_every_match(self):
        script = self.folder_script("app", "clean_paths 'libs/*/*.so'")
        for abi in ("arm64-v8a", "x86_64", "armeabi-v7a"):
            write(self.repo / "app" / "libs" / abi / "libfoo.so")
        write(self.repo / "app" / "libs" / ".gitkeep")
        self.git(self.repo, "add", "app/libs/.gitkeep")
        self.git(self.repo, "commit", "-qm", "keep")

        dry = self.run_script(script)
        applied = self.run_script(script, "--apply")

        self.assertIn("app/libs/*/*.so (3 matches)", dry.stdout)
        self.assertEqual(0, applied.returncode, applied.stderr)
        self.assertEqual([], list((self.repo / "app" / "libs").glob("*/*.so")))
        self.assertTrue((self.repo / "app" / "libs" / ".gitkeep").is_file())

    def test_command_runs_in_the_folder_on_apply_only(self):
        calls = self.temp / "calls"
        self.stub("fake-clean", f'echo "$PWD $*" >> {calls}')
        script = self.folder_script("app", "clean_command fake-clean --all\nclean_paths build")
        write(self.repo / "app" / "build" / "out.bin")

        dry = self.run_script(script)
        self.assertFalse(calls.exists())
        self.assertIn("then: fake-clean --all", dry.stdout)

        applied = self.run_script(script, "--apply")
        self.assertEqual(0, applied.returncode, applied.stderr)
        self.assertEqual(f"{self.repo / 'app'} --all", calls.read_text().strip())
        # The paths go too, whatever the command left.
        self.assertFalse((self.repo / "app" / "build").exists())

    def test_command_is_skipped_when_nothing_was_built(self):
        calls = self.temp / "calls"
        self.stub("fake-clean", f"touch {calls}")
        script = self.folder_script("app", "clean_command fake-clean\nclean_paths build")

        result = self.run_script(script, "--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse(calls.exists())

    def test_missing_command_still_deletes_the_paths(self):
        script = self.folder_script("app", "clean_command no-such-tool clean\nclean_paths build")
        write(self.repo / "app" / "build" / "out.bin")

        result = self.run_script(script, "--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("no-such-tool is not installed", result.stdout)
        self.assertFalse((self.repo / "app" / "build").exists())

    def test_unknown_argument_fails(self):
        script = self.folder_script("app", "clean_paths build")

        result = self.run_script(script, "--everything")

        self.assertEqual(2, result.returncode)
        self.assertIn("unknown argument", result.stderr)


class RootScriptTests(CleanTestCase):
    def add_worktree(self, name, branch):
        path = self.repo / ".context" / name
        self.git(self.repo, "worktree", "add", "-q", "-b", branch, str(path))
        write(path / f"{name}.txt")
        self.git(path, "add", ".")
        self.git(path, "commit", "-qm", name)
        return path

    def test_runs_every_folder_script_and_reports_in_path_order(self):
        self.folder_script("b-app", "clean_paths build")
        self.folder_script("a-app", "clean_paths build")
        write(self.repo / "a-app" / "build" / "out.bin")
        write(self.repo / "b-app" / "build" / "out.bin")
        write(self.repo / "target" / "debug" / "bin")

        dry = self.run_root()
        applied = self.run_root("--apply")

        self.assertEqual(0, dry.returncode, dry.stderr)
        self.assertLess(dry.stdout.index("a-app/build"), dry.stdout.index("b-app/build"))
        self.assertIn("target", dry.stdout)
        self.assertEqual(0, applied.returncode, applied.stderr)
        for path in ("a-app/build", "b-app/build", "target"):
            self.assertFalse((self.repo / path).exists(), path)

    def test_runs_only_clean_scripts_git_tracks(self):
        # A package's own clean.sh inside node_modules/, or a stray untracked
        # one, is not ours to run.
        marker = self.temp / "ran"
        for folder in ("web/node_modules/pkg", "stray"):
            write(self.repo / folder / "clean.sh", f"touch {marker}\n")

        result = self.run_root("--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse(marker.exists())

    def test_never_runs_clean_scripts_under_context_even_when_tracked(self):
        marker = self.temp / "ran"
        write(self.repo / ".context" / "scratch" / "clean.sh", f"touch {marker}\n")
        self.git(self.repo, "add", ".context/scratch/clean.sh")

        result = self.run_root("--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse(marker.exists())

    def test_failing_folder_script_fails_the_run_but_not_the_others(self):
        write(self.repo / "broken" / "clean.sh", "echo boom; exit 3\n")
        self.git(self.repo, "add", "broken/clean.sh")
        self.folder_script("app", "clean_paths build")
        write(self.repo / "app" / "build" / "out.bin")

        result = self.run_root("--apply")

        self.assertEqual(1, result.returncode)
        self.assertIn("broken/clean.sh", result.stdout)
        self.assertIn("boom", result.stdout)
        self.assertFalse((self.repo / "app" / "build").exists())

    def test_sweeps_outputs_in_plain_context_scratch_but_keeps_notes(self):
        # .context/ is not ignored here, as on a machine without Conductor.
        write(self.repo / ".context" / "smoke" / "node_modules" / "pkg" / "i.js")
        write(self.repo / ".context" / "smoke" / "ios" / "Pods" / "Pod.h")
        write(self.repo / ".context" / "notes.md")

        result = self.run_root("--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse((self.repo / ".context" / "smoke" / "node_modules").exists())
        self.assertFalse((self.repo / ".context" / "smoke" / "ios" / "Pods").exists())
        self.assertTrue((self.repo / ".context" / "notes.md").is_file())

    def test_sweeps_only_what_a_nested_checkout_under_context_ignores(self):
        app = self.repo / ".context" / "app"
        app.mkdir(parents=True)
        self.git(app, "init", "-q")
        write(app / ".gitignore", "node_modules/\n")
        write(app / "node_modules" / "pkg" / "i.js")
        write(app / "build" / "handwritten.txt")

        result = self.run_root("--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse((app / "node_modules").exists())
        self.assertTrue((app / "build" / "handwritten.txt").is_file())

    def test_keeps_context_dirs_this_checkout_tracks(self):
        write(self.repo / ".context" / "kept" / "build" / "keep.txt")
        self.git(self.repo, "add", "-f", ".context/kept/build/keep.txt")

        result = self.run_root("--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue((self.repo / ".context" / "kept" / "build" / "keep.txt").is_file())

    def test_worktrees_removes_only_published_clean_ones(self):
        pushed = self.add_worktree("pushed", "feat/pushed")
        self.git(pushed, "push", "-q", "origin", "feat/pushed")
        write(pushed / "target" / "big")
        local_only = self.add_worktree("local-only", "feat/local-only")
        dirty = self.add_worktree("dirty", "feat/dirty")
        self.git(dirty, "push", "-q", "origin", "feat/dirty")
        write(dirty / "wip.txt", "unsaved")

        result = self.run_root("--worktrees", "--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse(pushed.exists())
        self.assertNotIn("pushed", self.git(self.repo, "worktree", "list"))
        self.assertTrue(local_only.is_dir())
        self.assertIn("local-only (HEAD is on no remote branch", result.stdout)
        self.assertTrue((dirty / "wip.txt").is_file())
        self.assertIn("dirty (uncommitted or untracked changes)", result.stdout)

    def test_worktrees_treats_merged_pr_commit_as_published(self):
        # Squash merge: the remote branch is gone, only the PR remembers HEAD.
        merged = self.add_worktree("merged", "fix/merged")
        head = self.git(merged, "rev-parse", "HEAD")
        self.stub("gh", f"echo {'0' * 40}; echo {head}")

        result = self.run_root("--worktrees", "--apply")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse(merged.exists())

    def test_bazel_expunges_this_workspace_output_base(self):
        base = self.temp / "bazel-root" / ("a" * 32)
        (base / "execroot" / "_main" / "bazel-out").mkdir(parents=True)
        (self.repo / "bazel-out").symlink_to(base / "execroot" / "_main" / "bazel-out")
        calls = self.temp / "bazel-calls"
        self.stub("fake-bazel", f'echo "$PWD $*" >> {calls}')

        result = self.run_root("--bazel", "--apply", BAZEL=str(self.bin / "fake-bazel"))

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(f"{self.repo} clean --expunge", calls.read_text().strip())

    def test_bazel_orphans_removes_only_bases_of_missing_workspaces(self):
        user_root = self.temp / "bazel-root"
        orphan = user_root / ("b" * 32)
        write(orphan / "DO_NOT_BUILD_HERE", str(self.temp / "archived-workspace"))
        write(orphan / "external" / "readonly.txt")
        (orphan / "external" / "readonly.txt").chmod(0o444)
        (orphan / "external").chmod(0o555)
        live = user_root / ("c" * 32)
        write(live / "DO_NOT_BUILD_HERE", str(self.repo))

        result = self.run_root("--bazel-orphans", "--apply", BAZEL_USER_ROOT=str(user_root))

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse(orphan.exists())
        self.assertTrue(live.is_dir())

    def test_unknown_argument_fails(self):
        result = self.run_root("--everything")

        self.assertEqual(2, result.returncode)
        self.assertIn("unknown argument", result.stderr)


def folder_scripts():
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--", ":(glob)**/clean.sh", ":!.context"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return [REPO / path for path in listed if path != "clean.sh"]


def listed_paths(script):
    """The patterns a clean.sh passes to clean_paths."""
    for line in script.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words[:1] == ["clean_paths"]:
            yield from words[1:]


def sample(pattern):
    """A concrete path a glob pattern would match, for git check-ignore."""
    return re.sub(r"\[(.)[^]]*\]", r"\1", pattern).replace("*", "x")


class RepoCleanScriptsTests(unittest.TestCase):
    def test_repo_has_folder_scripts(self):
        self.assertTrue(folder_scripts())

    def test_every_folder_script_sources_the_lib(self):
        for script in folder_scripts():
            with self.subTest(script=str(script.relative_to(REPO))):
                match = re.search(r'\. "\$\(dirname "\$0"\)/(.*)"', script.read_text())
                self.assertIsNotNone(match)
                self.assertEqual(LIB, (script.parent / match.group(1)).resolve())
                self.assertIn('clean_run "$@"', script.read_text())

    def test_every_listed_path_is_git_ignored(self):
        for script in folder_scripts():
            for pattern in listed_paths(script):
                path = (script.parent / sample(pattern)).relative_to(REPO)
                with self.subTest(script=str(script.relative_to(REPO)), pattern=pattern):
                    # A trailing slash tells git the path is a directory,
                    # which is what `build/`-style patterns need.
                    ignored = any(
                        subprocess.run(
                            ["git", "check-ignore", "-q", "--no-index", candidate],
                            cwd=REPO,
                        ).returncode
                        == 0
                        for candidate in (str(path), f"{path}/")
                    )
                    self.assertTrue(ignored, f"{path} is not git-ignored")


if __name__ == "__main__":
    unittest.main()
