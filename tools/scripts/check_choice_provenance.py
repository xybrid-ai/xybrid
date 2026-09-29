#!/usr/bin/env python3
"""Verify the choice-scoring conformance provenance manifest.

``integration-tests/fixtures/choice/provenance.json`` pins every input of the
choice-scoring conformance suite: upstream code and weights by full commit
SHA and per-file sha256, the environments that produced each derived file,
and the digests of the derived model artifacts, which are regenerated from
those inputs rather than committed. This script checks that the manifest,
the committed fixtures and ``models.json`` all agree, using only the Python
standard library, so it never needs the torch environment in
``tools/conformance``.

    python3 tools/scripts/check_choice_provenance.py --check
    python3 tools/scripts/check_choice_provenance.py --staged cua-s1-forms
    python3 tools/scripts/check_choice_provenance.py --update

``--check`` verifies every digest and citation. ``--staged`` also verifies
prepared model artifacts under ``integration-tests/fixtures/models``.
``--update`` rewrites only the committed-fixture digest table, after a
generator deliberately changed a fixture, and only if everything else checks
out with the new table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "xybrid/choice-provenance/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path("integration-tests/fixtures/choice")
MODELS_JSON = Path("integration-tests/fixtures/models/models.json")
MODELS_DIR = Path("integration-tests/fixtures/models")
MANIFEST_NAME = "provenance.json"

_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GATE_STATUS = {"provisional", "frozen"}
_CHUNK = 1 << 20


class ProvenanceError(Exception):
    """The manifest, a fixture or models.json disagrees with the pinned state."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProvenanceError(f"{path}: missing") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"{path}: invalid JSON ({exc})") from None


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceError(f"{where}: expected an object")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProvenanceError(f"{where}: expected a non-empty string")
    return value


def _revision(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ProvenanceError(f"{where}: expected a full 40-character commit SHA")
    return value


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProvenanceError(f"{where}: expected a sha256 digest")
    return value


def _relative(value: Any, where: str) -> str:
    text = _text(value, where)
    parts = Path(text).parts
    if Path(text).is_absolute() or ".." in parts or "\\" in text:
        raise ProvenanceError(f"{where}: unsafe relative path {text!r}")
    return text


def _file_entry(value: Any, where: str) -> tuple[str, int]:
    entry = _mapping(value, where)
    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ProvenanceError(f"{where}.size: expected a non-negative integer")
    return _digest(entry.get("sha256"), f"{where}.sha256"), size


def source_url(source: Mapping[str, Any], file_name: str) -> str:
    """The immutable download URL of one pinned source file."""
    repository = source["repository"]
    revision = source["revision"]
    if source["kind"] == "huggingface":
        return f"https://huggingface.co/{repository}/resolve/{revision}/{file_name}"
    owner_repo = repository.removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{revision}/{file_name}"


class Checker:
    """Cross-checks provenance.json against the tree rooted at ``root``."""

    def __init__(
        self, root: Path, *, use_git: bool = True, manifest: Mapping[str, Any] | None = None
    ) -> None:
        self.root = root
        self.fixture_dir = root / FIXTURE_DIR
        self.manifest_path = self.fixture_dir / MANIFEST_NAME
        self.use_git = use_git
        if manifest is None:
            manifest = load_json(self.manifest_path)
        self.manifest = _mapping(manifest, MANIFEST_NAME)

    # -- structure -----------------------------------------------------------

    def check(self) -> None:
        if self.manifest.get("schema") != SCHEMA:
            raise ProvenanceError(f"{MANIFEST_NAME}: schema must be {SCHEMA!r}")
        self.check_sources()
        self.check_environments()
        self.check_artifacts()
        self.check_models_json()
        self.check_fixture_digests()
        self.check_templates()
        self.check_citing_files()
        self.check_gates()

    @property
    def sources(self) -> Mapping[str, Any]:
        return _mapping(self.manifest.get("sources"), "sources")

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return _mapping(self.manifest.get("artifacts"), "artifacts")

    def check_sources(self) -> None:
        if not self.sources:
            raise ProvenanceError("sources: no pinned sources")
        for name, raw in self.sources.items():
            where = f"sources.{name}"
            source = _mapping(raw, where)
            kind = source.get("kind")
            if kind not in {"git", "huggingface"}:
                raise ProvenanceError(f"{where}.kind: expected 'git' or 'huggingface'")
            repository = _text(source.get("repository"), f"{where}.repository")
            if kind == "git" and not repository.startswith("https://github.com/"):
                raise ProvenanceError(f"{where}.repository: expected a GitHub URL")
            revision = _revision(source.get("revision"), f"{where}.revision")
            _text(source.get("license"), f"{where}.license")
            files = _mapping(source.get("files"), f"{where}.files")
            for file_name, entry in files.items():
                _relative(file_name, f"{where}.files")
                _file_entry(entry, f"{where}.files.{file_name}")
            evidence = _text(source.get("license_evidence"), f"{where}.license_evidence")
            if evidence not in files:
                raise ProvenanceError(
                    f"{where}.license_evidence: {evidence!r} is not a pinned file"
                )
            submodule = source.get("submodule")
            if submodule is not None:
                self.check_submodule(_relative(submodule, f"{where}.submodule"), revision, where)

    def check_submodule(self, path: str, revision: str, where: str) -> None:
        if not self.use_git:
            return
        try:
            listing = subprocess.run(
                ["git", "-C", str(self.root), "ls-files", "--stage", "--", path],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ProvenanceError(f"{where}.submodule: cannot read the git index ({exc})") from None
        if len(listing) < 2 or listing[0] != "160000":
            raise ProvenanceError(f"{where}.submodule: {path} is not a submodule")
        if listing[1] != revision:
            raise ProvenanceError(
                f"{where}.submodule: {path} is at {listing[1]}, provenance pins {revision}"
            )

    def check_environments(self) -> None:
        environments = _mapping(self.manifest.get("environments"), "environments")
        for name, raw in environments.items():
            where = f"environments.{name}"
            environment = _mapping(raw, where)
            _text(environment.get("python"), f"{where}.python")
            requirements = _relative(environment.get("requirements"), f"{where}.requirements")
            expected = _digest(environment.get("sha256"), f"{where}.sha256")
            path = self.root / requirements
            if not path.is_file():
                raise ProvenanceError(f"{where}.requirements: {requirements} is missing")
            if sha256_file(path) != expected:
                raise ProvenanceError(f"{where}: {requirements} changed; regenerate what it built")

    def environment(self, name: Any, where: str) -> None:
        environments = _mapping(self.manifest.get("environments"), "environments")
        if name not in environments:
            raise ProvenanceError(f"{where}: unknown environment {name!r}")

    def check_artifacts(self) -> None:
        if not self.artifacts:
            raise ProvenanceError("artifacts: no derived artifacts")
        for model_id, raw in self.artifacts.items():
            where = f"artifacts.{model_id}"
            artifact = _mapping(raw, where)
            files = _mapping(artifact.get("files"), f"{where}.files")
            if not files:
                raise ProvenanceError(f"{where}.files: no outputs")
            for file_name, entry in files.items():
                _relative(file_name, f"{where}.files")
                _file_entry(entry, f"{where}.files.{file_name}")
            tool = _relative(artifact.get("tool"), f"{where}.tool")
            if not (self.root / tool).is_file():
                raise ProvenanceError(f"{where}.tool: {tool} is missing")
            self.environment(artifact.get("environment"), f"{where}.environment")
            inputs = artifact.get("inputs")
            if not isinstance(inputs, list) or not inputs:
                raise ProvenanceError(f"{where}.inputs: expected a non-empty list")
            for index, item in enumerate(inputs):
                self.resolve_input(item, f"{where}.inputs[{index}]")

    def resolve_input(self, raw: Any, where: str) -> tuple[str, int]:
        """Return the (sha256, size) an artifact input cites."""
        item = _mapping(raw, where)
        file_name = _text(item.get("file"), f"{where}.file")
        if "source" in item:
            source = self.sources.get(item["source"])
            if source is None:
                raise ProvenanceError(f"{where}: unknown source {item['source']!r}")
            entry = source["files"].get(file_name)
        elif "artifact" in item:
            artifact = self.artifacts.get(item["artifact"])
            if artifact is None:
                raise ProvenanceError(f"{where}: unknown artifact {item['artifact']!r}")
            entry = artifact["files"].get(file_name)
        else:
            raise ProvenanceError(f"{where}: cite a 'source' or an 'artifact'")
        if entry is None:
            raise ProvenanceError(f"{where}: {file_name!r} is not pinned there")
        return _file_entry(entry, where)

    # -- models.json ---------------------------------------------------------

    def check_models_json(self) -> None:
        manifest = _mapping(load_json(self.root / MODELS_JSON), str(MODELS_JSON))
        models = _mapping(manifest.get("models"), f"{MODELS_JSON}: models")
        for model_id, raw in self.artifacts.items():
            where = f"{MODELS_JSON}: models.{model_id}"
            entry = models.get(model_id)
            if entry is None:
                raise ProvenanceError(f"{where}: missing")
            entry = _mapping(entry, where)
            if entry.get("source") != "derived":
                raise ProvenanceError(f"{where}.source: expected 'derived' (no published URL)")
            artifact = _mapping(raw, f"artifacts.{model_id}")
            self.compare_outputs(entry.get("files"), artifact["files"], f"{where}.files")
            self.compare_inputs(entry.get("inputs"), artifact["inputs"], f"{where}.inputs")

    @staticmethod
    def compare_outputs(raw: Any, pinned: Mapping[str, Any], where: str) -> None:
        if not isinstance(raw, list):
            raise ProvenanceError(f"{where}: expected a list")
        declared: dict[str, str] = {}
        for index, item in enumerate(raw):
            item = _mapping(item, f"{where}[{index}]")
            if "url" in item:
                raise ProvenanceError(f"{where}[{index}]: derived outputs have no published URL")
            output = _relative(item.get("output"), f"{where}[{index}].output")
            declared[output] = _digest(item.get("sha256"), f"{where}[{index}].sha256")
        expected = {name: entry["sha256"] for name, entry in pinned.items()}
        if declared != expected:
            raise ProvenanceError(
                f"{where}: outputs {declared} disagree with provenance {expected}"
            )

    def compare_inputs(self, raw: Any, pinned: list[Any], where: str) -> None:
        if not isinstance(raw, list):
            raise ProvenanceError(f"{where}: expected a list")
        expected = []
        for index, item in enumerate(pinned):
            digest, _size = self.resolve_input(item, f"{where}[{index}]")
            if "source" in item:
                source = self.sources[item["source"]]
                expected.append(
                    {
                        "url": source_url(source, item["file"]),
                        "output": item.get("output", item["file"]),
                        "sha256": digest,
                    }
                )
            else:
                expected.append(
                    {"model": item["artifact"], "output": item["file"], "sha256": digest}
                )
        if raw != expected:
            raise ProvenanceError(
                f"{where}: inputs disagree with provenance; expected {json.dumps(expected)}"
            )

    # -- committed fixtures --------------------------------------------------

    def fixture_files(self) -> dict[str, str]:
        """Digest of every committed fixture file except the manifest itself."""
        digests = {}
        for path in sorted(self.fixture_dir.rglob("*")):
            if not path.is_file() or path == self.manifest_path:
                continue
            digests[path.relative_to(self.fixture_dir).as_posix()] = sha256_file(path)
        return digests

    def check_fixture_digests(self) -> None:
        pinned = _mapping(self.manifest.get("fixtures"), "fixtures")
        actual = self.fixture_files()
        problems = [f"{name} is pinned but missing" for name in sorted(set(pinned) - set(actual))]
        problems += [f"{name} is not pinned" for name in sorted(set(actual) - set(pinned))]
        problems += [
            f"{name} changed; its sha256 is now {digest}"
            for name, digest in sorted(actual.items())
            if name in pinned and _digest(pinned[name], f"fixtures.{name}") != digest
        ]
        if problems:
            raise ProvenanceError(
                "fixtures: " + "; ".join(problems) + " (run --update once the change is reviewed)"
            )
        ignored = self.ignored_fixtures(sorted(actual))
        if ignored:
            raise ProvenanceError(
                f"fixtures: {', '.join(ignored)} would not be committed (git-ignored); "
                "add a .gitignore exception"
            )

    def ignored_fixtures(self, names: list[str]) -> list[str]:
        """The fixtures git would leave out of a commit (untracked and ignored)."""
        if not self.use_git or not names:
            return []
        paths = {(FIXTURE_DIR / name).as_posix(): name for name in names}
        try:
            result = subprocess.run(
                ["git", "-C", str(self.root), "check-ignore", "--stdin"],
                input="\n".join(paths),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise ProvenanceError(f"fixtures: cannot run git check-ignore ({exc})") from None
        # Exit status 1 means no path is ignored; anything above it is an error.
        if result.returncode not in (0, 1):
            raise ProvenanceError(f"fixtures: git check-ignore failed: {result.stderr.strip()}")
        return [paths[line] for line in result.stdout.splitlines() if line in paths]

    def check_templates(self) -> None:
        templates = _mapping(self.manifest.get("templates"), "templates")
        for name, raw in templates.items():
            where = f"templates.{name}"
            template = _mapping(raw, where)
            path = self.fixture_dir / _relative(name, where)
            if not path.is_file():
                raise ProvenanceError(f"{where}: missing")
            digest, _size = self.resolve_input(template, where)
            if sha256_file(path) != digest:
                raise ProvenanceError(f"{where}: differs from the pinned {template['file']}")

    def check_citing_files(self) -> None:
        """Cases, goldens and generated files cite pinned sources and inputs."""
        cases = self.manifest.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ProvenanceError("cases: expected a non-empty list of case files")
        for index, name in enumerate(cases):
            name = _relative(name, f"cases[{index}]")
            if not (self.fixture_dir / name).is_file():
                raise ProvenanceError(f"cases[{index}]: {name} is missing")
            self.check_citations(name, {})
        generated = _mapping(self.manifest.get("generated"), "generated")
        for name, raw in generated.items():
            where = f"generated.{name}"
            item = _mapping(raw, where)
            if not (self.fixture_dir / _relative(name, where)).is_file():
                raise ProvenanceError(f"{where}: missing")
            tool = _relative(item.get("generator"), f"{where}.generator")
            if not (self.root / tool).is_file():
                raise ProvenanceError(f"{where}.generator: {tool} is missing")
            self.environment(item.get("environment"), f"{where}.environment")
            if name.endswith(".json"):
                self.check_citations(name, item)

    def check_citations(self, name: str, item: Mapping[str, Any]) -> None:
        document = _mapping(load_json(self.fixture_dir / name), name)
        cited = _mapping(document.get("provenance"), f"{name}: provenance")
        sources = cited.get("sources")
        if not isinstance(sources, dict) or not sources:
            raise ProvenanceError(f"{name}: provenance.sources must name the pinned sources")
        for source, revision in sources.items():
            pinned = self.sources.get(source)
            if pinned is None:
                raise ProvenanceError(f"{name}: cites unknown source {source!r}")
            if revision != pinned["revision"]:
                raise ProvenanceError(
                    f"{name}: cites {source}@{revision}, pinned {pinned['revision']}"
                )
        for key in ("cases", "spec", "template"):
            if key not in item:
                continue
            path = _relative(item[key], f"generated.{name}.{key}")
            if not (self.fixture_dir / path).is_file():
                raise ProvenanceError(f"generated.{name}.{key}: {path} is missing")
            if cited.get(f"{key}_sha256") != sha256_file(self.fixture_dir / path):
                raise ProvenanceError(f"{name}: generated from a different {path}; regenerate it")
        if item:
            self.check_generated_by(name, cited, item)

    def check_generated_by(
        self, name: str, cited: Mapping[str, Any], item: Mapping[str, Any]
    ) -> None:
        """A generated file names the generator, environment and inputs that made it."""
        environment = item["environment"]
        if cited.get("generator") != item["generator"]:
            raise ProvenanceError(f"{name}: says generator {cited.get('generator')!r}")
        if cited.get("environment") != environment:
            raise ProvenanceError(f"{name}: says environment {cited.get('environment')!r}")
        pinned = self.manifest["environments"][environment]["sha256"]
        if cited.get("requirements_sha256") != pinned:
            raise ProvenanceError(f"{name}: generated in another {environment} environment")
        inputs = cited.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ProvenanceError(f"{name}: provenance.inputs must list the files it read")
        for index, raw in enumerate(inputs):
            where = f"{name}: provenance.inputs[{index}]"
            entry = _mapping(raw, where)
            digest, _size = self.resolve_input(entry, where)
            if entry.get("sha256") != digest:
                raise ProvenanceError(f"{where}: read {entry.get('file')!r} at another digest")

    def check_gates(self) -> None:
        gates = _mapping(self.manifest.get("gates"), "gates")
        if gates.get("status") not in _GATE_STATUS:
            raise ProvenanceError(f"gates.status: expected one of {sorted(_GATE_STATUS)}")

        def walk(value: Any, where: str, key: str) -> None:
            if isinstance(value, dict):
                for child_key, child in value.items():
                    walk(child, f"{where}.{child_key}", child_key)
            elif key == "note":
                _text(value, where)
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ProvenanceError(f"{where}: expected a finite positive bound")

        for key, value in gates.items():
            if key != "status":
                walk(value, f"gates.{key}", key)

    # -- staged artifacts ----------------------------------------------------

    def check_staged(self, model_ids: Iterable[str], models_dir: Path) -> None:
        for model_id in model_ids:
            artifact = self.artifacts.get(model_id)
            if artifact is None:
                raise ProvenanceError(f"--staged: {model_id!r} is not a pinned artifact")
            for file_name, entry in artifact["files"].items():
                path = models_dir / model_id / file_name
                if not path.is_file():
                    raise ProvenanceError(f"{path}: not staged; run tools/conformance/prepare.sh")
                if sha256_file(path) != entry["sha256"]:
                    raise ProvenanceError(f"{path}: sha256 differs from provenance")

    def with_current_fixture_digests(self) -> dict[str, Any]:
        """A copy of the manifest whose fixture table matches the files on disk."""
        return {**self.manifest, "fixtures": self.fixture_files()}

    def write_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify every digest and citation")
    mode.add_argument("--update", action="store_true", help="rewrite the fixture digest table")
    mode.add_argument(
        "--staged", nargs="+", metavar="MODEL_ID", help="--check, then verify prepared artifacts"
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--no-git", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="staged models (default: fixtures/models)"
    )
    args = parser.parse_args(argv)

    try:
        checker = Checker(args.root, use_git=not args.no_git)
        if args.update:
            candidate = checker.with_current_fixture_digests()
            changed = candidate != checker.manifest
            checker = Checker(args.root, use_git=not args.no_git, manifest=candidate)
        checker.check()
        if args.update:
            # Written only now: a stale golden or citation leaves the file untouched.
            if changed:
                checker.write_manifest()
            print("fixture digests updated" if changed else "fixture digests already current")
        if args.staged:
            checker.check_staged(args.staged, args.models_dir or args.root / MODELS_DIR)
    except ProvenanceError as exc:
        print(f"choice provenance: {exc}", file=sys.stderr)
        return 1
    print("choice provenance: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
