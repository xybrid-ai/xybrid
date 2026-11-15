"""The shared CI setup actions stay the one definition, and Dependabot sees them.

.github/actions/ holds the composite actions ci.yml's jobs are built from
(setup-rust, install-boltffi, free-disk-space, ...). These tests keep three
things true that nothing else would catch:

  * Dependabot's github-actions entry lists every action directory: "/" scans
    only .github/workflows/, so an unlisted action keeps stale pins forever.
  * The workflows that use the shared actions do not grow inline copies again.
  * install-boltffi installs the boltffi CLI the generators pin and Cargo.lock
    resolves; a skew mis-generates the bindings.
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ACTIONS = REPO / ".github" / "actions"
WORKFLOWS = REPO / ".github" / "workflows"

# Workflows whose Rust toolchain, cargo cache, boltffi install and disk cleanup
# come from the shared actions. Release and zzz workflows keep their own copies,
# whose text their tests pin.
SHARED_SETUP_WORKFLOWS = ("ci.yml", "bazel.yml", "build-android.yml", "unity-editor.yml")

INLINE_COPIES = {
    "dtolnay/rust-toolchain": "use ./.github/actions/setup-rust",
    "Swatinem/rust-cache": "use ./.github/actions/setup-rust",
    "cargo install boltffi_cli": "use ./.github/actions/install-boltffi",
    "sudo rm -rf /usr/share/dotnet": "use ./.github/actions/free-disk-space",
}

BOLTFFI_PINS = {
    "tools/scripts/gen_python_bolt.py": r'^PINNED_BOLTFFI = "([^"]+)"$',
    "tools/scripts/gen_kotlin_bolt.py": r'^PINNED_BOLTFFI = "([^"]+)"$',
    "tools/scripts/gen_unity_bolt_csharp.py": r'^PINNED_BOLTFFI = "([^"]+)"$',
    "bindings/apple/scripts/gen-bolt-bindings.sh": r'^PINNED_BOLTFFI="([^"]+)"$',
}


def action_dirs() -> list:
    return sorted(path.parent.name for path in ACTIONS.glob("*/action.yml"))


def dependabot_action_dirs() -> list:
    text = (REPO / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    entry = re.search(r"- package-ecosystem: github-actions\n((?:    .*\n|\s*#.*\n|\n)*)", text)
    if entry is None:
        raise AssertionError("dependabot.yml has no github-actions entry")
    listed = re.search(r"^    directories:\n((?:      - .*\n)+)", entry[1], re.M)
    if listed is None:
        raise AssertionError("the github-actions entry must use `directories:`, one per action")
    return re.findall(r'^      - "([^"]+)"$', listed[1], re.M)


class SharedActionTests(unittest.TestCase):
    def test_dependabot_updates_every_action_directory(self):
        listed = dependabot_action_dirs()
        self.assertIn("/", listed, "keep '/' so the workflows are still scanned")
        expected = [f"/.github/actions/{name}" for name in action_dirs()]
        self.assertEqual(sorted(entry for entry in listed if entry != "/"), expected)

    def test_shared_setup_workflows_have_no_inline_copies(self):
        for workflow in SHARED_SETUP_WORKFLOWS:
            text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
            for needle, fix in INLINE_COPIES.items():
                with self.subTest(workflow=workflow, copy=needle):
                    self.assertNotIn(needle, text, f"{workflow}: {fix}")

    def test_every_action_step_that_runs_a_script_names_its_shell(self):
        # A composite action's `run:` step fails at startup without `shell:`.
        for name in action_dirs():
            text = (ACTIONS / name / "action.yml").read_text(encoding="utf-8")
            steps = re.split(r"(?m)^    - ", text)[1:]
            for step in steps:
                if re.search(r"(?m)^      run:", step):
                    with self.subTest(action=name, step=step.splitlines()[0]):
                        self.assertRegex(step, r"(?m)^      shell: ")

    def test_install_boltffi_matches_the_generators_and_cargo_lock(self):
        action = (ACTIONS / "install-boltffi" / "action.yml").read_text(encoding="utf-8")
        default = re.search(r"^  version:\n(?:    .*\n)*?    default: '([^']+)'$", action, re.M)
        self.assertIsNotNone(default, "install-boltffi must default its version")
        version = default[1]
        for path, pattern in BOLTFFI_PINS.items():
            with self.subTest(pin=path):
                pin = re.search(pattern, (REPO / path).read_text(encoding="utf-8"), re.M)
                self.assertIsNotNone(pin, f"{path} no longer pins boltffi")
                self.assertEqual(pin[1], version)
        lock = (REPO / "Cargo.lock").read_text(encoding="utf-8")
        crate = re.search(r'^name = "boltffi"\nversion = "([^"]+)"$', lock, re.M)
        self.assertIsNotNone(crate, "Cargo.lock no longer resolves the boltffi crate")
        self.assertEqual(crate[1], version, "the boltffi CLI must match the runtime crate")

    def test_setup_rust_saves_the_cache_only_on_master(self):
        action = (ACTIONS / "setup-rust" / "action.yml").read_text(encoding="utf-8")
        self.assertIn("save-if: ${{ github.ref == 'refs/heads/master' }}", action)


if __name__ == "__main__":
    unittest.main()
