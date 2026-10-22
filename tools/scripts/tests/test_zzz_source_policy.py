"""Offline security contracts for the draft source gate; no candidate code runs."""

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/zzz_source_policy.py"
SPEC = importlib.util.spec_from_file_location("zzz_source_policy", SCRIPT)
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)
GIT = shutil.which("git", path=os.defpath)
OLD = "0.13.0"
NEW = "0.14.0"
DATE = "2026-10-22"
CHECKSUM = "b" * 64


def context(master):
    return {
        "repository": "xybrid-ai/xybrid",
        "workflow_path": ".github/workflows/trusted-zzz-sdk.yml",
        "workflow_commit": master,
        "ref": "refs/heads/master", "event": "workflow_dispatch",
        "run_id": 123, "run_attempt": 1,
    }


def synthetic_files():
    return {
        "Cargo.toml": '''[workspace]
members = ["crates/demo", "crates/other"]
[workspace.package]
version = "0.13.0"
[workspace.dependencies]
demo = { path = "crates/demo", version = "0.13.0" }
external = "0.13.0"
''',
        "crates/demo/Cargo.toml": '''[package]
name = "demo"
version.workspace = true
[dependencies]
renamed = { package = "other", path = "../other", version = "0.13.0" }
external = "0.13.0"
[target.'cfg(unix)'.build-dependencies]
other = { path = "../other", version = "0.13.0" }
''',
        "crates/other/Cargo.toml": '[package]\nname = "other"\nversion = "0.13.0"\n',
        "Cargo.lock": '''version = 4
[[package]]
name = "demo"
version = "0.13.0"
dependencies = ["external", "other"]
[[package]]
name = "other"
version = "0.13.0"
[[package]]
name = "external"
version = "0.13.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "trusted-external-checksum"
''',
        POLICY.MANIFEST: '{"schema_version":4,"release":"v0.1.0-rc.8"}\n',
        POLICY.VERIFIER: '# Frozen verifier; importing this is forbidden.\nraise RuntimeError("DO NOT RUN")\n',
        "Package.swift": 'import PackageDescription\nlet useLocalNatives = false\nlet sdkVersion = "0.13.0"\nlet xybridFFIChecksum = "' + "a" * 64 + '"\n',
        "bindings/react-native/package.json": '{\n  "version": "0.13.0",\n  "scripts": {"prepack": "trusted-command"},\n  "xybrid": {"iosXcframeworkSha256": "' + "a" * 64 + '"}\n}\n',
        "bindings/unity/package.json": '{"version": "0.13.0", "name": "trusted"}\n',
        "bindings/web/package.json": '{"version": "0.13.0", "scripts": {"build": "trusted"}}\n',
        "bindings/python/pyproject.toml": '[project]\nname = "xybrid"\nversion = "0.13.0"\n[build-system]\nrequires = ["trusted"]\n',
        "bindings/python/xybrid/_bolt/__init__.py": 'PACKAGE_VERSION = "0.13.0"\nraise RuntimeError("DO NOT IMPORT CANDIDATE")\n',
        "CHANGELOG.md": '# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- Already reviewed.\n',
        "bindings/flutter/CHANGELOG.md": '# Changelog\n\n## Unreleased\n\n* Already reviewed.\n',
        "bindings/flutter/pubspec.yaml": 'name: xybrid_flutter\nversion: 0.13.0\n',
        "bindings/flutter/example/pubspec.lock": 'packages:\n  xybrid_flutter:\n    dependency: direct main\n    source: path\n    version: "0.13.0"\n  external:\n    version: "0.13.0"\n',
        "bindings/kotlin/build.gradle.kts": 'version = "0.13.0"\n// reviewed Kotlin code\n',
        "bindings/react-native/android/build.gradle": 'dependencies { api "ai.xybrid:xybrid-kotlin:0.13.0" }\n',
        "bindings/web/src/version.ts": 'export const SDK_VERSION = "0.13.0";\n',
        "bindings/apple/XybridFFI-Info.plist": '<plist><string>0.13.0</string><string>0.13.0</string></plist>\n',
        "README.md": '# Trusted SDK\nInstall version 0.13.0.\n',
        "docs/content/docs/quickstart.mdx": 'import Reviewed from "./reviewed";\n<Reviewed version="0.13.0" />\n',
        "src/frozen.rs": 'fn main() { panic!("DO NOT EXECUTE"); }\n',
        ".gitattributes": '*.md filter=never-run\n',
    }


def canonical_bump(files, old, new, date=DATE, checksum=CHECKSUM):
    """An independent fixture editor, deliberately not the policy renderer."""
    result = dict(files)
    root = tomllib.loads(files["Cargo.toml"])
    members = root["workspace"]["members"]
    names = {tomllib.loads(files[m + "/Cargo.toml"])["package"]["name"] for m in members}
    cargo_paths = {"Cargo.toml", *(m + "/Cargo.toml" for m in members)}
    for path, text in files.items():
        if path in cargo_paths or path == "bindings/python/pyproject.toml":
            section = ""
            lines = []
            for line in text.splitlines(keepends=True):
                if line.startswith("["):
                    section = line.strip()
                if section in ("[workspace.package]", "[package]", "[project]") and re.match(r"^version\s*=", line):
                    line = line.replace('"' + old + '"', '"' + new + '"')
                elif "path" in line and "version" in line:
                    name = line.split("=", 1)[0].strip()
                    alias = re.search(r'package\s*=\s*"([^"]+)"', line)
                    if (alias[1] if alias else name) in names:
                        line = line.replace('"' + old + '"', '"' + new + '"')
                lines.append(line)
            result[path] = "".join(lines)
        elif path == "Cargo.lock":
            parts = text.split("[[package]]")
            for index in range(1, len(parts)):
                package = tomllib.loads(parts[index])
                if package["name"] in names and "source" not in package:
                    parts[index] = parts[index].replace('version = "' + old + '"', 'version = "' + new + '"', 1)
            result[path] = "[[package]]".join(parts)
        elif path in POLICY.JSON_PACKAGES:
            text = text.replace('"version": "' + old + '"', '"version": "' + new + '"', 1)
            if path == "bindings/react-native/package.json" and checksum is not None:
                previous = json.loads(files[path])["xybrid"]["iosXcframeworkSha256"]
                text = text.replace('"' + previous + '"', '"' + checksum + '"', 1)
            result[path] = text
        elif path == "Package.swift":
            text = text.replace('let sdkVersion = "' + old + '"', 'let sdkVersion = "' + new + '"', 1)
            if checksum is not None:
                text = re.sub(r'(?m)^let xybridFFIChecksum = "[a-f0-9]{64}"$',
                              'let xybridFFIChecksum = "' + checksum + '"', text)
            result[path] = text
        elif path == "CHANGELOG.md" and new != old:
            result[path] = text.replace("## [Unreleased]\n", f"## [Unreleased]\n\n---\n\n## [{new}] - {date}\n", 1)
        elif path == "bindings/flutter/CHANGELOG.md" and new != old:
            result[path] = text.replace("## Unreleased\n", f"## Unreleased\n\n## {new}\n", 1)
        elif path == "bindings/flutter/example/pubspec.lock":
            result[path] = re.sub(r'(?m)(^  xybrid_flutter:\n(?:    .*\n)*?    version: ")' + re.escape(old) + r'("$)',
                                  lambda match: match[1] + new + match[2], text)
        elif path in POLICY.VERSION_TEXT:
            result[path] = text.replace(old, new)
        elif path == "bindings/apple/XybridFFI-Info.plist":
            result[path] = text.replace(f"<string>{old.split('-')[0]}</string>", f"<string>{new.split('-')[0]}</string>")
        elif path == "bindings/python/xybrid/_bolt/__init__.py":
            result[path] = text.replace(f'PACKAGE_VERSION = "{old}"', f'PACKAGE_VERSION = "{new}"', 1)
        elif path == "bindings/web/src/version.ts":
            result[path] = text.replace(f'export const SDK_VERSION = "{old}";', f'export const SDK_VERSION = "{new}";', 1)
        elif path == "bindings/flutter/pubspec.yaml":
            result[path] = text.replace(f"version: {old}", f"version: {new}", 1)
        elif path == "bindings/kotlin/build.gradle.kts":
            result[path] = text.replace(f'version = "{old}"', f'version = "{new}"', 1)
        elif path == "bindings/react-native/android/build.gradle":
            result[path] = text.replace(f"ai.xybrid:xybrid-kotlin:{old}", f"ai.xybrid:xybrid-kotlin:{new}", 1)
    return result


class Repository:
    def __init__(self, directory, files):
        self.path = Path(directory)
        self.environment = {"PATH": os.defpath, "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
                            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_AUTHOR_NAME": "Policy Test",
                            "GIT_AUTHOR_EMAIL": "policy@example.invalid", "GIT_COMMITTER_NAME": "Policy Test",
                            "GIT_COMMITTER_EMAIL": "policy@example.invalid"}
        self.git("init", "--quiet")
        self.git("config", "core.hooksPath", os.devnull)
        self.write(files)
        # An unchanged gitlink need not exist in the superproject object store.
        self.gitlink = "1" * 40
        self.base = self.commit()

    def git(self, *arguments):
        return subprocess.run([GIT, "--no-replace-objects", "-C", str(self.path), *arguments],
                              env=self.environment, check=True, capture_output=True, timeout=20).stdout.decode().strip()

    def write(self, files):
        for path, text in files.items():
            target = self.path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    def commit(self):
        self.git("add", ".")
        # Keep an intentionally uninitialized submodule in each fixture tree.
        self.git("update-index", "--add", "--cacheinfo", "160000," + self.gitlink + ",vendor/pinned")
        self.git("commit", "--quiet", "--allow-empty", "-m", "policy fixture")
        return self.git("rev-parse", "HEAD")

    def bind(self, candidate=None, **kwargs):
        options = {"repo": self.path, "trusted_master": self.base,
                   "candidate": candidate or self.git("rev-parse", "HEAD"),
                   "release_version": NEW, "release_date": DATE,
                   "context": context(self.base), "checksum": CHECKSUM}
        options.update(kwargs)
        return POLICY.source_binding(**options)


class JsonParsingTests(unittest.TestCase):
    def test_container_depth_boundary_is_explicit_and_parser_independent(self):
        depth = POLICY.MAX_JSON_DEPTH
        for opening, closing in ((b"[", b"]"), (b'{"a":', b"}")):
            with self.subTest(opening=opening):
                valid = opening * depth + b"0" + closing * depth
                self.assertIsNotNone(POLICY.json_bytes(valid, 8192))
                invalid = opening * (depth + 1) + b"0" + closing * (depth + 1)
                with self.assertRaisesRegex(POLICY.PolicyError, "depth bound"):
                    POLICY.json_bytes(invalid, 8192)
        with self.assertRaisesRegex(POLICY.PolicyError, "depth bound"):
            POLICY.json_bytes(b"[" * 2000 + b"0" + b"]" * 2000, 8192)
        mixed = b'[{"a":' * (depth // 2) + b"0" + b"}]" * (depth // 2)
        self.assertIsNotNone(POLICY.json_bytes(mixed, 8192))
        with self.assertRaisesRegex(POLICY.PolicyError, "depth bound"):
            POLICY.json_bytes(b"[" + mixed + b"]", 8192)

    def test_quoted_escaped_and_unicode_brackets_do_not_count(self):
        value = '[{' * 2000 + '\\"' + '}]' * 2000 + '\\' + '\u005b'
        encoded = json.dumps(value).encode()
        self.assertEqual(POLICY.json_bytes(encoded, len(encoded)), value)
        # Escaped quotes must not end the string, and escaped backslashes must
        # not hide the actual closing quote/container that follows.
        nested = b"[" * POLICY.MAX_JSON_DEPTH + json.dumps(value).encode() + b"]" * POLICY.MAX_JSON_DEPTH
        self.assertIsNotNone(POLICY.json_bytes(nested, len(nested)))
        self.assertEqual(POLICY.json_bytes(b'"\\u005b"', 64), "[")
        self.assertEqual(POLICY.json_bytes(b'["ends\\\\"]', 64), ["ends\\"])

    def test_malformed_duplicate_nonfinite_and_non_utf8_json_still_reject(self):
        for data in (b'{"a":1,"a":2}', b"NaN", b"Infinity", b"-Infinity", b"[{]}",
                     b"]", b'"unterminated', b'"\\q"', b'"\xff"', b'\xff\xfe{\x00}\x00'):
            with self.subTest(data=data), self.assertRaises(POLICY.PolicyError):
                POLICY.json_bytes(data, 8192)


class SourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.files = synthetic_files()
        self.repo = Repository(self.temporary.name, self.files)
        self.bump = canonical_bump(self.files, OLD, NEW)
        self.repo.write(self.bump)
        self.good = self.repo.commit()

    def rejected(self, files, message=None):
        self.repo.write(files)
        bad = self.repo.commit()
        with self.assertRaises(POLICY.PolicyError) as caught:
            self.repo.bind(bad)
        if message:
            self.assertIn(message, str(caught.exception))

    def test_canonical_bump_binds_content_and_producer_without_claiming_authentication(self):
        result = self.repo.bind(self.good)
        self.assertEqual(result["candidate_commit"], self.good)
        self.assertEqual(result["producer_context"], context(self.repo.base))
        self.assertEqual(result["producer_authentication"], "NOT_AUTHENTICATED_BY_THIS_HELPER")
        self.assertEqual(result["engine_manifest_sha256"], hashlib.sha256(self.files[POLICY.MANIFEST].encode()).hexdigest())
        self.assertEqual(result["packaged_verifier_sha256"], hashlib.sha256(self.files[POLICY.VERIFIER].encode()).hexdigest())
        self.assertRegex(result["candidate_tree_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(result, self.repo.bind(self.good))
        self.assertIn('external = "0.13.0"', self.bump["crates/demo/Cargo.toml"])
        self.assertIn('name = "external"\nversion = "0.13.0"', self.bump["Cargo.lock"])

    def test_unchanged_tree_and_checksum_only_are_supported(self):
        same = self.repo.bind(self.repo.base, release_version=OLD, checksum=None)
        self.repo.write(canonical_bump(self.files, OLD, OLD))
        checksum_only = self.repo.bind(self.repo.commit(), release_version=OLD)
        self.assertNotEqual(same["candidate_tree_sha256"], checksum_only["candidate_tree_sha256"])
        self.assertEqual(same["engine_manifest_sha256"], checksum_only["engine_manifest_sha256"])

    def test_rejects_code_appended_to_each_executable_metadata_format(self):
        injections = {
            "Package.swift": '\nimport Foundation\nlet stolen = ProcessInfo.processInfo.environment\n',
            "bindings/kotlin/build.gradle.kts": '\nexec { commandLine("curl", "evil") }\n',
            "bindings/react-native/android/build.gradle": '\nexec { commandLine "evil" }\n',
            "bindings/python/xybrid/_bolt/__init__.py": '\nimport subprocess; subprocess.run(["evil"])\n',
            "bindings/web/src/version.ts": '\nfetch("https://evil.invalid");\n',
            "docs/content/docs/quickstart.mdx": '\nexport const stolen = process.env;\n',
            "bindings/flutter/pubspec.yaml": '\nhooks: {build: evil}\n',
            "bindings/python/pyproject.toml": '\n[tool.evil]\nscript = "steal"\n',
        }
        for path, injection in injections.items():
            with self.subTest(path=path):
                self.rejected({path: self.bump[path] + injection})
                self.repo.write({path: self.bump[path]})

    def test_rejects_changes_to_native_mode_script_urls_and_package_hooks(self):
        variants = [
            ("Package.swift", self.bump["Package.swift"].replace("useLocalNatives = false", "useLocalNatives = true")),
            ("Package.swift", self.bump["Package.swift"] + '\nlet url = "https://evil.invalid"\n'),
            ("bindings/react-native/package.json", self.bump["bindings/react-native/package.json"].replace("trusted-command", "evil-command")),
            ("bindings/web/package.json", self.bump["bindings/web/package.json"].replace('"trusted"', '"evil"')),
        ]
        for path, text in variants:
            with self.subTest(path=path):
                self.rejected({path: text})
                self.repo.write({path: self.bump[path]})

    def test_rejects_external_dependency_and_lock_checksum_changes(self):
        for path, before, after in [
            ("crates/demo/Cargo.toml", 'external = "0.13.0"', 'external = "0.14.0"'),
            ("Cargo.toml", 'external = "0.13.0"', 'external = "0.14.0"'),
            ("Cargo.lock", 'name = "external"\nversion = "0.13.0"', 'name = "external"\nversion = "0.14.0"'),
            ("Cargo.lock", "trusted-external-checksum", "evil-checksum"),
        ]:
            with self.subTest(path=path, token=before):
                self.rejected({path: self.bump[path].replace(before, after)})
                self.repo.write({path: self.bump[path]})

    def test_rejects_internal_dependency_destination_or_features_changes(self):
        original = self.bump["crates/demo/Cargo.toml"]
        for text in (original.replace('path = "../other"', 'path = "../evil"'),
                     original.replace('version = "0.14.0" }', 'version = "0.14.0", features = ["evil"] }')):
            self.rejected({"crates/demo/Cargo.toml": text})
            self.repo.write({"crates/demo/Cargo.toml": original})

    def test_rejects_new_freeform_changelog_even_for_plausible_current_release(self):
        self.rejected({"CHANGELOG.md": self.bump["CHANGELOG.md"] + "\n### Release notes\n\n- Newly supplied release prose.\n"})

    def test_rejects_malformed_duplicate_and_nonfinite_package_json(self):
        path = "bindings/react-native/package.json"
        for text in (self.bump[path].replace('"version":', '"version":"evil", "version":', 1),
                     self.bump[path].replace('"scripts":', '"unexpected":NaN, "scripts":', 1)):
            self.rejected({path: text})
            self.repo.write({path: self.bump[path]})

    def test_rejects_duplicate_swift_literals_and_wrong_checksum(self):
        path = "Package.swift"
        for text in (self.bump[path] + f'let sdkVersion = "{NEW}"\n',
                     self.bump[path].replace(CHECKSUM, "c" * 64)):
            self.rejected({path: text})
            self.repo.write({path: self.bump[path]})

    def test_rejects_manifest_verifier_or_other_source_changes(self):
        for path in (POLICY.MANIFEST, POLICY.VERIFIER, "src/frozen.rs", ".gitattributes"):
            with self.subTest(path=path):
                self.rejected({path: self.bump[path] + "\n# modified\n"}, "nonmetadata")
                self.repo.write({path: self.bump[path]})

    def test_rejects_added_removed_or_renamed_paths(self):
        self.rejected({"src/new.rs": "// added\n"}, "paths")
        (self.repo.path / "src/new.rs").unlink()
        self.repo.git("mv", "src/frozen.rs", "src/moved.rs")
        with self.assertRaisesRegex(POLICY.PolicyError, "paths"):
            self.repo.bind(self.repo.commit())

    def test_rejects_mode_or_symlink_substitution(self):
        path = self.repo.path / "Package.swift"
        path.chmod(0o755)
        with self.assertRaisesRegex(POLICY.PolicyError, "mode/type"):
            self.repo.bind(self.repo.commit())
        path.unlink()
        path.symlink_to("/private/should-never-be-read")
        with self.assertRaisesRegex(POLICY.PolicyError, "mode/type"):
            self.repo.bind(self.repo.commit())

    def test_rejects_gitlink_changes_without_reading_submodule(self):
        self.repo.gitlink = "2" * 40
        with self.assertRaisesRegex(POLICY.PolicyError, "nonmetadata"):
            self.repo.bind(self.repo.commit())

    def test_git_replace_cannot_hide_a_bad_candidate(self):
        self.repo.write({"src/frozen.rs": "// malicious\n"})
        bad = self.repo.commit()
        self.repo.git("replace", bad, self.good)
        with self.assertRaisesRegex(POLICY.PolicyError, "nonmetadata"):
            self.repo.bind(bad)

    def test_immutable_commit_objects_ignore_worktree_and_smudge_filter(self):
        sentinel = self.repo.path / "FILTER-RAN"
        self.repo.git("config", "filter.never-run.smudge", "touch " + str(sentinel))
        self.repo.write({"Package.swift": "uncommitted malicious bytes\n"})
        result = self.repo.bind(self.good)
        self.assertEqual(result["candidate_commit"], self.good)
        self.assertFalse(sentinel.exists())

    def test_mirror_environment_is_refused_before_any_git_or_input_read(self):
        for name in POLICY.SECRET_NAMES:
            for value in ("", "DO-NOT-LOG-THIS-TOKEN"):
                with self.subTest(name=name, empty=not value), patch.dict(os.environ, {name: value}), \
                        patch.object(POLICY, "Git") as git:
                    with self.assertRaisesRegex(POLICY.PolicyError, "outside the mirror-credential") as caught:
                        self.repo.bind(self.good)
                    self.assertNotIn(value or name, str(caught.exception))
                    git.assert_not_called()

    def test_fixed_git_environment_does_not_inherit_credentials_or_git_overrides(self):
        with patch.dict(os.environ, {"GH_TOKEN": "secret", "GIT_CONFIG_COUNT": "1", "GIT_DIR": "evil"}):
            git = POLICY.Git(self.repo.path)
        self.assertNotIn("GH_TOKEN", git.environment)
        self.assertNotIn("GIT_DIR", git.environment)
        self.assertNotIn("GIT_CONFIG_COUNT", git.environment)
        self.assertIn("--no-replace-objects", git.command)
        self.assertEqual(git.environment["GIT_NO_LAZY_FETCH"], "1")

    def test_cli_refuses_mirror_environment_before_opening_context(self):
        arguments = [str(SCRIPT), "--repo", "not-opened", "--trusted-master", self.repo.base,
                     "--candidate", self.good, "--version", NEW, "--release-date", DATE,
                     "--producer-context", "not-opened.json"]
        with patch.dict(os.environ, {"ZZZ_GITHUB_TOKEN": "TEST-VALUE-DO-NOT-PRINT"}), \
                patch.object(sys, "argv", arguments), patch.object(POLICY, "load_json") as load, \
                patch.object(sys, "stderr", io.StringIO()) as errors:
            self.assertEqual(POLICY.main(), 1)
            load.assert_not_called()
            self.assertIn("outside the mirror-credential", errors.getvalue())
            self.assertNotIn("TEST-VALUE-DO-NOT-PRINT", errors.getvalue())

    def test_only_full_lowercase_immutable_commit_shas_are_accepted(self):
        for value in ("HEAD", "master", self.good[:12], self.good.upper(), self.good + "^{tree}", 123):
            with self.subTest(value=value), self.assertRaises(POLICY.PolicyError):
                self.repo.bind(value)
        tree = self.repo.git("rev-parse", self.good + "^{tree}")
        with self.assertRaisesRegex(POLICY.PolicyError, "identify a commit"):
            self.repo.bind(tree)

    def test_non_descendant_candidate_is_rejected(self):
        tree = self.repo.git("rev-parse", self.good + "^{tree}")
        orphan = self.repo.git("commit-tree", tree, "-m", "unrelated history")
        with self.assertRaises(POLICY.PolicyError):
            self.repo.bind(orphan)

    def test_versions_dates_and_checksums_reject_injection_or_bad_types(self):
        for value in ('0.14.0"; evil()', "0.14.0\n", "v0.14.0", "01.14.0", 123):
            with self.subTest(value=value), self.assertRaises(POLICY.PolicyError):
                self.repo.bind(release_version=value)
        for value in ("2026-02-30", "2026-1-1", "2026-10-22\n", None):
            with self.subTest(date=value), self.assertRaises(POLICY.PolicyError):
                self.repo.bind(release_date=value)
        for value in ("C" * 64, "b" * 63, "b" * 64 + "\n", 123):
            with self.subTest(checksum=value), self.assertRaises(POLICY.PolicyError):
                self.repo.bind(checksum=value)

    def test_producer_context_is_strict_and_cannot_override_source_identity(self):
        changes = {"repository": "attacker/fork", "workflow_path": ".github/workflows/release-prep.yml",
                   "workflow_commit": self.good, "ref": "refs/heads/release/v0.14.0",
                   "event": "pull_request_target", "run_id": True, "run_attempt": 0,
                   "extra": "not allowed"}
        for field, value in changes.items():
            supplied = context(self.repo.base)
            supplied[field] = value
            with self.subTest(field=field), self.assertRaises(POLICY.PolicyError):
                self.repo.bind(context=supplied)

    def test_output_and_tree_limits_fail_closed(self):
        with self.assertRaisesRegex(POLICY.PolicyError, "output exceeds"):
            POLICY.Git(self.repo.path).read("ls-tree", "-r", self.good, limit=8)
        for constant, limit in (("MAX_METADATA", 16), ("MAX_BLOB", 16), ("MAX_TREE_BYTES", 16)):
            with self.subTest(constant=constant), patch.object(POLICY, constant, limit), \
                    self.assertRaisesRegex(POLICY.PolicyError, "byte bound"):
                self.repo.bind(self.good)
        with self.assertRaises(POLICY.PolicyError):
            POLICY.json_bytes(b'{"a":1,"a":2}', 64)
        with self.assertRaises(POLICY.PolicyError):
            POLICY.json_bytes(b"{}" * 10, 4)
        with self.assertRaises(POLICY.PolicyError):
            POLICY.json_bytes(b"[" * 2000 + b"0" + b"]" * 2000, 8192)

    def cli(self, *extra):
        context_file = self.repo.path / "caller-context.json"
        context_file.write_text(json.dumps(context(self.repo.base)))
        return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(self.repo.path),
                               "--trusted-master", self.repo.base, "--candidate", self.good,
                               "--version", NEW, "--release-date", DATE,
                               "--producer-context", str(context_file), "--xcframework-sha256", CHECKSUM, *extra],
                              env={"PATH": os.defpath}, capture_output=True, timeout=30)

    def test_cli_recomputes_binding_and_rejects_forged_fields(self):
        initial = self.cli()
        self.assertEqual(initial.returncode, 0, initial.stderr)
        binding = json.loads(initial.stdout)
        path = self.repo.path / "binding.json"
        path.write_bytes(initial.stdout)
        self.assertEqual(self.cli("--compare-binding", str(path)).returncode, 0)
        reordered = dict(reversed(list(binding.items())))
        reordered["producer_context"] = dict(reversed(list(binding["producer_context"].items())))
        path.write_text(json.dumps(reordered, indent=3))
        self.assertEqual(self.cli("--compare-binding", str(path)).returncode, 0)
        for field, value in (("candidate_commit", self.repo.base), ("engine_manifest_sha256", "0" * 64),
                             ("packaged_verifier_sha256", "0" * 64), ("candidate_tree_sha256", "0" * 64),
                             ("producer_authentication", "authenticated"), ("schema_version", True),
                             ("schema_version", 1.0), ("unknown", "claim")):
            forged = copy.deepcopy(binding)
            forged[field] = value
            path.write_text(json.dumps(forged))
            result = self.cli("--compare-binding", str(path))
            self.assertEqual(result.returncode, 1, field)
            self.assertEqual(result.stdout, b"")
            self.assertIn(b"source binding/context mismatch", result.stderr)
        forged = copy.deepcopy(binding)
        forged["producer_context"]["run_attempt"] += 1
        path.write_text(json.dumps(forged))
        self.assertEqual(self.cli("--compare-binding", str(path)).returncode, 1)
        for field, value in (("run_attempt", True), ("run_id", 123.0)):
            forged = copy.deepcopy(binding)
            forged["producer_context"][field] = value
            path.write_text(json.dumps(forged))
            self.assertEqual(self.cli("--compare-binding", str(path)).returncode, 1)


class ActualRepositoryMetadataTests(unittest.TestCase):
    def test_complete_immutable_repository_tree_including_large_ort_archives(self):
        # Exercise real >64MiB blobs and >256MiB unique content by streaming
        # existing objects, without copying, checking out or building anything.
        with patch.dict(os.environ, {}, clear=True):
            git = POLICY.Git(ROOT)
            head = git.read("rev-parse", "--verify", "HEAD", limit=64).decode().strip()
            entries = git.entries(head)
            root = tomllib.loads(git.blob(entries["Cargo.toml"]).decode())
            current = root["workspace"]["package"]["version"]
            result = POLICY.source_binding(ROOT, head, head, current, DATE, context(head))
        self.assertEqual(result["candidate_commit"], head)
        self.assertRegex(result["candidate_tree_sha256"], r"^[0-9a-f]{64}$")

    def test_real_current_metadata_structure_accepts_only_canonical_bump(self):
        """Includes the real generated PACKAGE_VERSION and all workspace manifests."""
        root = tomllib.loads((ROOT / "Cargo.toml").read_text())
        old = root["workspace"]["package"]["version"]
        major, minor, _ = old.split(".")
        new = f"{major}.{int(minor) + 1}.0"
        paths = POLICY.VERSION_TEXT | POLICY.JSON_PACKAGES | {
            "Cargo.toml", "Cargo.lock", POLICY.MANIFEST, POLICY.VERIFIER, "Package.swift", "CHANGELOG.md",
            "bindings/flutter/CHANGELOG.md", "bindings/flutter/pubspec.yaml",
            "bindings/flutter/example/pubspec.lock", "bindings/kotlin/build.gradle.kts",
            "bindings/react-native/android/build.gradle", "bindings/python/pyproject.toml",
            "bindings/python/xybrid/_bolt/__init__.py", "bindings/web/src/version.ts",
            "bindings/apple/XybridFFI-Info.plist",
        } | {member + "/Cargo.toml" for member in root["workspace"]["members"]}
        files = {path: (ROOT / path).read_text() for path in paths}
        self.assertIn(f'PACKAGE_VERSION = "{old}"', files["bindings/python/xybrid/_bolt/__init__.py"])
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            repo = Repository(directory, files)
            bumped = canonical_bump(files, old, new)
            self.assertNotEqual(files["Cargo.toml"], bumped["Cargo.toml"])
            self.assertNotEqual(files["bindings/python/xybrid/_bolt/__init__.py"],
                                bumped["bindings/python/xybrid/_bolt/__init__.py"])
            self.assertIn(f'PACKAGE_VERSION = "{new}"', bumped["bindings/python/xybrid/_bolt/__init__.py"])
            repo.write(bumped)
            candidate = repo.commit()
            binding = repo.bind(candidate, release_version=new)
            self.assertEqual(binding["candidate_commit"], candidate)
            self.assertEqual(binding["release_metadata"]["version"], new)
            self.assertEqual(binding["engine_manifest_sha256"], hashlib.sha256(files[POLICY.MANIFEST].encode()).hexdigest())
            # A plausible, manually edited release note still exceeds this draft policy.
            changelog = (repo.path / "CHANGELOG.md").read_text()
            repo.write({"CHANGELOG.md": changelog + "\n### Added\n\n- Custom 0.13.1 release prose.\n"})
            with self.assertRaises(POLICY.PolicyError):
                repo.bind(repo.commit(), release_version=new)


if __name__ == "__main__":
    unittest.main()
