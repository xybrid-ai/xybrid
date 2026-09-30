"""Unit tests for the choice-scoring provenance checker (standard library only)."""

import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_choice_provenance import REPO_ROOT, Checker, ProvenanceError, main

UPSTREAM_REV = "a" * 40
TOOL_REV = "b" * 40
FIXTURE_TEMPLATE = "integration-tests/fixtures/choice/templates/t.jinja"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pinned(payload: bytes) -> dict:
    return {"sha256": sha(payload), "size": len(payload)}


def write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_json(path: Path, value: object) -> None:
    write(path, (json.dumps(value, indent=2) + "\n").encode())


class Tree:
    """A minimal repository whose provenance is consistent."""

    WEIGHTS = b"upstream weights"
    LICENSE = b"MIT License"
    TEMPLATE = b"{{ messages }}"
    REQUIREMENTS = b"numpy==2.4.6\n"
    MODEL_A = b"derived onnx bytes"
    MODEL_B = b"derived gguf bytes"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.fixtures = root / "integration-tests/fixtures/choice"
        self.models_json = root / "integration-tests/fixtures/models/models.json"
        self.models_dir = root / "integration-tests/fixtures/models"
        write(root / "tools/requirements.txt", self.REQUIREMENTS)
        write(root / "tools/build.py", b"# builds the fixtures\n")
        write(self.fixtures / "templates/t.jinja", self.TEMPLATE)
        write_json(
            self.fixtures / "cases/c.json",
            {"provenance": {"sources": {"upstream": UPSTREAM_REV}}, "cases": []},
        )
        self.write_golden()
        self.manifest = {
            "schema": "xybrid/choice-provenance/v1",
            "sources": {
                "upstream": {
                    "kind": "huggingface",
                    "repository": "org/model",
                    "revision": UPSTREAM_REV,
                    "license": "MIT",
                    "license_evidence": "LICENSE",
                    "files": {
                        "LICENSE": pinned(self.LICENSE),
                        "weights.bin": pinned(self.WEIGHTS),
                        "template.jinja": pinned(self.TEMPLATE),
                    },
                },
                "tool": {
                    "kind": "git",
                    "repository": "https://github.com/org/tool",
                    "revision": TOOL_REV,
                    "license": "MIT",
                    "license_evidence": "LICENSE",
                    "files": {"LICENSE": pinned(self.LICENSE)},
                },
            },
            "environments": {
                "env": {
                    "python": "3.12",
                    "requirements": "tools/requirements.txt",
                    "sha256": sha(self.REQUIREMENTS),
                }
            },
            "artifacts": {
                "model-a": {
                    "files": {"model.onnx": pinned(self.MODEL_A)},
                    "inputs": [{"source": "upstream", "file": "weights.bin"}],
                    "tool": "tools/build.py",
                    "environment": "env",
                },
                "model-b": {
                    "files": {"model-b.gguf": pinned(self.MODEL_B)},
                    "inputs": [{"artifact": "model-a", "file": "model.onnx"}],
                    "tool": "tools/build.py",
                    "environment": "env",
                },
            },
            "templates": {"templates/t.jinja": {"source": "upstream", "file": "template.jinja"}},
            "cases": ["cases/c.json"],
            "generated": {
                "goldens/g.json": {
                    "generator": "tools/build.py",
                    "environment": "env",
                    "cases": "cases/c.json",
                    "template": "templates/t.jinja",
                }
            },
            "gates": {
                "note": "text",
                "track": {"status": "provisional", "g1": {"max_abs_logit": 0.15, "note": "text"}},
            },
            "fixtures": {},
        }
        self.models = {
            "models": {
                "unrelated": {"source": "url", "files": []},
                "model-a": {
                    "source": "derived",
                    "inputs": [
                        {
                            "url": f"https://huggingface.co/org/model/resolve/{UPSTREAM_REV}/weights.bin",
                            "output": "weights.bin",
                            "sha256": sha(self.WEIGHTS),
                        }
                    ],
                    "files": [{"output": "model.onnx", "sha256": sha(self.MODEL_A)}],
                },
                "model-b": {
                    "source": "derived",
                    "inputs": [
                        {"model": "model-a", "output": "model.onnx", "sha256": sha(self.MODEL_A)}
                    ],
                    "files": [{"output": "model-b.gguf", "sha256": sha(self.MODEL_B)}],
                },
            }
        }
        self.save()

    def write_golden(self, **overrides: object) -> None:
        cited = {
            "sources": {"upstream": UPSTREAM_REV},
            "generator": "tools/build.py",
            "environment": "env",
            "requirements_sha256": sha(self.REQUIREMENTS),
            "cases_sha256": sha((self.fixtures / "cases/c.json").read_bytes()),
            "template_sha256": sha(self.TEMPLATE),
            "inputs": [{"source": "upstream", "file": "weights.bin", "sha256": sha(self.WEIGHTS)}],
        }
        cited.update(overrides)
        write_json(self.fixtures / "goldens/g.json", {"provenance": cited, "cases": []})

    def save(self, *, refresh_digests: bool = True) -> None:
        if refresh_digests:
            self.manifest["fixtures"] = {
                path.relative_to(self.fixtures).as_posix(): sha(path.read_bytes())
                for path in sorted(self.fixtures.rglob("*"))
                if path.is_file() and path.name != "provenance.json"
            }
        write_json(self.fixtures / "provenance.json", self.manifest)
        write_json(self.models_json, self.models)

    def checker(self) -> Checker:
        return Checker(self.root, use_git=False)


class CheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.tree = Tree(Path(self._temp.name))

    def tearDown(self) -> None:
        self._temp.cleanup()

    def assertRejects(self, pattern: str) -> None:
        with self.assertRaisesRegex(ProvenanceError, pattern):
            self.tree.checker().check()

    def test_consistent_tree_passes(self) -> None:
        self.tree.checker().check()

    def test_changed_fixture_is_rejected(self) -> None:
        write(self.tree.fixtures / "templates/t.jinja", b"{{ other }}")
        self.assertRejects(r"templates/t\.jinja changed")

    def test_unpinned_fixture_is_rejected(self) -> None:
        write(self.tree.fixtures / "extra.json", b"{}")
        self.assertRejects(r"extra\.json is not pinned")

    def test_every_stale_fixture_is_reported(self) -> None:
        write(self.tree.fixtures / "extra.json", b"{}")
        write(self.tree.fixtures / "templates/t.jinja", b"{{ other }}")
        self.assertRejects(r"extra\.json is not pinned; templates/t\.jinja changed")

    def test_generated_file_citing_a_missing_input_is_rejected(self) -> None:
        self.tree.manifest["generated"]["goldens/g.json"]["spec"] = "specs/missing.json"
        self.tree.save()
        self.assertRejects(r"generated\.goldens/g\.json\.spec: specs/missing\.json is missing")

    def test_missing_pinned_fixture_is_rejected(self) -> None:
        self.tree.manifest["fixtures"]["gone.json"] = sha(b"")
        self.tree.save(refresh_digests=False)
        self.assertRejects(r"gone\.json is pinned but missing")

    def test_abbreviated_revision_is_rejected(self) -> None:
        self.tree.manifest["sources"]["upstream"]["revision"] = "aaaaaaa"
        self.tree.save()
        self.assertRejects(r"full 40-character commit SHA")

    def test_license_evidence_must_be_pinned(self) -> None:
        self.tree.manifest["sources"]["tool"]["license_evidence"] = "COPYING"
        self.tree.save()
        self.assertRejects(r"license_evidence: 'COPYING' is not a pinned file")

    def test_changed_requirements_are_rejected(self) -> None:
        write(self.tree.root / "tools/requirements.txt", b"numpy==2.5.0\n")
        self.assertRejects(r"requirements\.txt changed")

    def test_models_json_must_use_the_pinned_revision(self) -> None:
        entry = self.tree.models["models"]["model-a"]["inputs"][0]
        entry["url"] = entry["url"].replace(UPSTREAM_REV, "main")
        self.tree.save()
        self.assertRejects(r"models\.model-a\.inputs: inputs disagree")

    def test_models_json_output_digest_must_match(self) -> None:
        self.tree.models["models"]["model-b"]["files"][0]["sha256"] = sha(b"other")
        self.tree.save()
        self.assertRejects(r"models\.model-b\.files: outputs .* disagree")

    def test_models_json_derived_outputs_have_no_url(self) -> None:
        self.tree.models["models"]["model-a"]["files"][0]["url"] = "https://example.invalid/x"
        self.tree.save()
        self.assertRejects(r"derived outputs have no published URL")

    def test_models_json_entry_must_be_derived(self) -> None:
        self.tree.models["models"]["model-a"]["source"] = "url"
        self.tree.save()
        self.assertRejects(r"models\.model-a\.source: expected 'derived'")

    def test_artifact_input_must_resolve(self) -> None:
        self.tree.manifest["artifacts"]["model-b"]["inputs"] = [
            {"artifact": "model-a", "file": "missing.onnx"}
        ]
        self.tree.save()
        self.assertRejects(r"'missing\.onnx' is not pinned there")

    def test_golden_citing_another_revision_is_rejected(self) -> None:
        self.tree.write_golden(sources={"upstream": "c" * 40})
        self.tree.save()
        self.assertRejects(r"goldens/g\.json: cites upstream@c+, pinned a+")

    def test_golden_from_other_cases_is_rejected(self) -> None:
        self.tree.write_golden(cases_sha256=sha(b"older cases"))
        self.tree.save()
        self.assertRejects(r"generated from a different cases/c\.json")

    def test_case_file_citing_unknown_source_is_rejected(self) -> None:
        write_json(
            self.tree.fixtures / "cases/c.json",
            {"provenance": {"sources": {"nobody": UPSTREAM_REV}}, "cases": []},
        )
        self.tree.write_golden()
        self.tree.save()
        self.assertRejects(r"cases/c\.json: cites unknown source 'nobody'")

    def test_golden_from_another_environment_is_rejected(self) -> None:
        self.tree.write_golden(requirements_sha256=sha(b"numpy==1.0\n"))
        self.tree.save()
        self.assertRejects(r"goldens/g\.json: generated in another env environment")

    def test_golden_naming_another_generator_is_rejected(self) -> None:
        self.tree.write_golden(generator="tools/other.py")
        self.tree.save()
        self.assertRejects(r"goldens/g\.json: says generator 'tools/other\.py'")

    def test_golden_input_must_match_the_pinned_digest(self) -> None:
        self.tree.write_golden(
            inputs=[{"source": "upstream", "file": "weights.bin", "sha256": sha(b"other weights")}]
        )
        self.tree.save()
        self.assertRejects(r"provenance\.inputs\[0\]: read 'weights\.bin' at another digest")

    def test_golden_must_list_its_inputs(self) -> None:
        self.tree.write_golden(inputs=[])
        self.tree.save()
        self.assertRejects(r"goldens/g\.json: provenance\.inputs must list the files it read")

    def test_template_must_match_its_pinned_source(self) -> None:
        self.tree.manifest["sources"]["upstream"]["files"]["template.jinja"] = pinned(b"{{ x }}")
        self.tree.save()
        self.assertRejects(r"templates\.templates/t\.jinja: differs from the pinned")

    def test_gate_bounds_must_be_positive(self) -> None:
        self.tree.manifest["gates"]["track"]["g1"]["max_abs_logit"] = 0
        self.tree.save()
        self.assertRejects(r"gates\.track\.g1\.max_abs_logit: expected a finite positive bound")

    def test_gate_bounds_must_be_finite_numbers(self) -> None:
        for bad in ("0.15", float("inf"), True):
            self.tree.manifest["gates"]["track"]["g1"]["max_abs_logit"] = bad
            self.tree.save()
            self.assertRejects(r"gates\.track\.g1\.max_abs_logit: expected a finite positive bound")

    def test_each_gate_track_has_its_own_status(self) -> None:
        for bad in (None, "measured"):
            self.tree.manifest["gates"]["track"]["status"] = bad
            self.tree.save()
            self.assertRejects(r"gates\.track\.status: expected one of \['frozen', 'provisional'\]")
        self.tree.manifest["gates"]["track"]["status"] = "frozen"
        self.tree.save()
        self.assertEqual(self.run_main("--check")[0], 0)

    def test_gate_status_is_not_global(self) -> None:
        self.tree.manifest["gates"]["status"] = "provisional"
        self.tree.save()
        self.assertRejects(r"gates\.status: status is per track")

    def test_gate_tracks_need_bounds(self) -> None:
        for track in ({"status": "frozen"}, {"status": "frozen", "note": "measured"}):
            self.tree.manifest["gates"]["track"] = track
            self.tree.save()
            self.assertRejects(r"gates\.track: expected at least one gate")
        self.tree.manifest["gates"] = {"note": "text"}
        self.tree.save()
        self.assertRejects(r"gates: expected at least one track")

    def run_main(self, *args: str) -> tuple[int, str]:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            status = main([*args, "--root", str(self.tree.root), "--no-git"])
        return status, stderr.getvalue()

    def test_update_rewrites_only_the_fixture_table(self) -> None:
        # New cases and a golden regenerated from them: only the table is stale.
        write_json(
            self.tree.fixtures / "cases/c.json",
            {"provenance": {"sources": {"upstream": UPSTREAM_REV}}, "cases": [1]},
        )
        self.tree.write_golden()
        self.tree.save(refresh_digests=False)
        self.assertEqual(self.run_main("--update")[0], 0)
        rewritten = json.loads((self.tree.fixtures / "provenance.json").read_text())
        cases = (self.tree.fixtures / "cases/c.json").read_bytes()
        self.assertEqual(rewritten["fixtures"]["cases/c.json"], sha(cases))
        self.assertEqual(rewritten["sources"], self.tree.manifest["sources"])
        self.assertEqual(self.run_main("--check")[0], 0)

    def test_update_writes_nothing_when_the_result_would_not_check(self) -> None:
        # New cases without a regenerated golden: updating the digests alone
        # cannot make the tree consistent, so the manifest must stay as it was.
        write_json(
            self.tree.fixtures / "cases/c.json",
            {"provenance": {"sources": {"upstream": UPSTREAM_REV}}, "cases": [2]},
        )
        before = (self.tree.fixtures / "provenance.json").read_bytes()
        status, stderr = self.run_main("--update")
        self.assertEqual(status, 1)
        self.assertIn("generated from a different cases/c.json", stderr)
        self.assertEqual((self.tree.fixtures / "provenance.json").read_bytes(), before)

    def test_staged_artifacts_are_verified(self) -> None:
        checker = self.tree.checker()
        with self.assertRaisesRegex(ProvenanceError, r"not staged"):
            checker.check_staged(["model-a"], self.tree.models_dir)
        write(self.tree.models_dir / "model-a/model.onnx", b"tampered")
        with self.assertRaisesRegex(ProvenanceError, r"sha256 differs"):
            checker.check_staged(["model-a"], self.tree.models_dir)
        write(self.tree.models_dir / "model-a/model.onnx", Tree.MODEL_A)
        checker.check_staged(["model-a"], self.tree.models_dir)
        with self.assertRaisesRegex(ProvenanceError, r"'unknown' is not a pinned artifact"):
            checker.check_staged(["unknown"], self.tree.models_dir)

    def test_main_reports_failure_with_exit_status(self) -> None:
        write(self.tree.fixtures / "extra.json", b"{}")
        status, stderr = self.run_main("--check")
        self.assertEqual(status, 1)
        self.assertIn("extra.json is not pinned", stderr)


@unittest.skipUnless(shutil.which("git"), "git is required for the submodule check")
class SubmoduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.tree = Tree(Path(self._temp.name))
        self.tree.manifest["sources"]["tool"]["submodule"] = "vendor/tool"
        self.tree.save()
        subprocess.run(["git", "init", "-q", str(self.tree.root)], check=True)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def record_gitlink(self, revision: str) -> None:
        subprocess.run(
            [
                "git",
                "-C",
                str(self.tree.root),
                "update-index",
                "--add",
                "--cacheinfo",
                f"160000,{revision},vendor/tool",
            ],
            check=True,
        )

    def test_matching_gitlink_passes(self) -> None:
        self.record_gitlink(TOOL_REV)
        Checker(self.tree.root).check()

    def test_moved_submodule_is_rejected(self) -> None:
        self.record_gitlink("d" * 40)
        with self.assertRaisesRegex(ProvenanceError, r"vendor/tool is at d+, provenance pins b+"):
            Checker(self.tree.root).check()

    def test_missing_submodule_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProvenanceError, r"vendor/tool is not a submodule"):
            Checker(self.tree.root).check()

    def test_git_ignored_fixture_is_rejected(self) -> None:
        # A fixture that only exists locally passes every digest check but is
        # missing from the commit, so CI would fail on the pinned-but-missing file.
        self.record_gitlink(TOOL_REV)
        write(self.tree.root / ".gitignore", b"*.jinja\n")
        with self.assertRaisesRegex(
            ProvenanceError, r"templates/t\.jinja would not be committed \(git-ignored\)"
        ):
            Checker(self.tree.root).check()

    def test_tracked_fixture_matching_an_ignore_rule_is_accepted(self) -> None:
        self.record_gitlink(TOOL_REV)
        subprocess.run(
            ["git", "-C", str(self.tree.root), "add", "-f", str(FIXTURE_TEMPLATE)], check=True
        )
        write(self.tree.root / ".gitignore", b"*.jinja\n")
        Checker(self.tree.root).check()


@unittest.skipUnless(
    (REPO_ROOT / ".git").exists() and shutil.which("git"), "needs the repository checkout"
)
class RepositoryTests(unittest.TestCase):
    def test_committed_fixtures_match_their_provenance(self) -> None:
        Checker(REPO_ROOT).check()


if __name__ == "__main__":
    unittest.main()
