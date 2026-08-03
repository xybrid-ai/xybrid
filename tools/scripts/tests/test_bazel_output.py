"""Tests for tools/scripts/bazel-output.sh.

The script shells out to Bazel, which a plain CI runner has no workspace for,
so these drive it against a stub binary via its `$BAZEL` hook. That covers
everything worth pinning — the argument parsing, the literal `--ext` suffix
match, and above all the strict single-output assertion, which is the reason
the script exists (the `| head -1` it replaced silently returned whichever
output sorted first when a target grew a second one).
"""

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "bazel-output.sh"

MSVC_CDYLIB = [
    "bazel-out/k8-opt/bin/crates/xybrid-bolt/xybrid_bolt.dll",
    "bazel-out/k8-opt/bin/crates/xybrid-bolt/xybrid_bolt.dll.lib",
]
SINGLE = ["bazel-out/k8-opt/bin/crates/xybrid-cli/xybrid"]


class BazelOutputTests(unittest.TestCase):
    def run_script(self, args, outputs):
        """Run bazel-output.sh with a stub Bazel that prints `outputs`."""
        with tempfile.TemporaryDirectory() as temp:
            stub = Path(temp) / "fake-bazel"
            stub.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    printf '%s\\n' {}
                    """
                ).format(" ".join(f"'{o}'" for o in outputs))
                if outputs
                else "#!/usr/bin/env bash\nexit 0\n"
            )
            stub.chmod(0o755)
            env = dict(os.environ, BAZEL=str(stub))
            return subprocess.run(
                ["bash", str(SCRIPT), *args],
                capture_output=True,
                text=True,
                env=env,
            )

    def test_single_output_is_printed(self):
        result = self.run_script(["//crates/xybrid-cli:xybrid"], SINGLE)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(SINGLE[0], result.stdout.strip())

    def test_two_outputs_fail_instead_of_guessing(self):
        # The whole point: `head -1` would have silently returned the .dll.
        result = self.run_script(["//crates/xybrid-bolt:cdylib"], MSVC_CDYLIB)
        self.assertEqual(1, result.returncode)
        self.assertIn("expected exactly 1 output", result.stderr)
        # The diagnostic must name what it actually found, both paths.
        for path in MSVC_CDYLIB:
            self.assertIn(path, result.stderr)

    def test_ext_selects_the_matching_output(self):
        result = self.run_script(
            ["--ext", "dll", "//crates/xybrid-bolt:cdylib"], MSVC_CDYLIB
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(MSVC_CDYLIB[0], result.stdout.strip())

    def test_ext_dll_does_not_swallow_dll_lib(self):
        # A regex `\\.dll$` would be fine here, but `--ext dll.lib` needs the
        # dot treated literally; both directions are checked.
        result = self.run_script(
            ["--ext", "dll.lib", "//crates/xybrid-bolt:cdylib"], MSVC_CDYLIB
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(MSVC_CDYLIB[1], result.stdout.strip())

    def test_ext_matching_nothing_fails(self):
        result = self.run_script(
            ["--ext", "exe", "//crates/xybrid-bolt:cdylib"], MSVC_CDYLIB
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("expected exactly 1 output", result.stderr)

    def test_all_prints_every_output(self):
        result = self.run_script(["--all", "//crates/xybrid-bolt:cdylib"], MSVC_CDYLIB)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(MSVC_CDYLIB, result.stdout.split())

    def test_bazel_args_are_forwarded_after_the_target(self):
        with tempfile.TemporaryDirectory() as temp:
            stub = Path(temp) / "fake-bazel"
            argv_log = Path(temp) / "argv"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                f'printf "%s\\n" "$@" > {argv_log}\n'
                f"printf '%s\\n' '{SINGLE[0]}'\n"
            )
            stub.chmod(0o755)
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "//crates/xybrid-cli:xybrid",
                    "--config=remote",
                    "-c",
                    "opt",
                ],
                capture_output=True,
                text=True,
                env=dict(os.environ, BAZEL=str(stub)),
            )
            self.assertEqual(0, result.returncode, result.stderr)
            argv = argv_log.read_text().split()
            self.assertEqual(
                ["cquery", "--config=remote", "-c", "opt", "--output=files",
                 "//crates/xybrid-cli:xybrid"],
                argv,
            )

    def test_unknown_option_is_rejected(self):
        result = self.run_script(["--nope", "//t"], SINGLE)
        self.assertEqual(2, result.returncode)
        self.assertIn("unknown option", result.stderr)


if __name__ == "__main__":
    unittest.main()
