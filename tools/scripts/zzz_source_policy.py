#!/usr/bin/env python3
"""Draft source-policy checks, not producer authentication or runtime activation."""

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
import tomllib


MAX_METADATA = 2 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_TREE_OUTPUT = 8 * 1024 * 1024
# The reviewed tree already tracks an 83,134,752-byte ORT archive and
# 272,602,448 unique blob bytes. Stream these; retain explicit upper bounds.
MAX_BLOB = 128 * 1024 * 1024
MAX_TREE_BYTES = 512 * 1024 * 1024
MANIFEST = "crates/zzz-sys/natives-manifest.json"
VERIFIER = "crates/zzz-sys/zzz_pull.py"
WORKFLOW = ".github/workflows/trusted-zzz-sdk.yml"  # Proposed; not wired by this patch.
SECRET_NAMES = ("ZZZ_GITHUB_TOKEN", "ZZZ_RELEASE_REPOSITORY", "XYBRID_ZZZ_USE_GH_AUTH")
JSON_PACKAGES = {
    "bindings/react-native/package.json", "bindings/unity/package.json",
    "bindings/web/package.json",
}
VERSION_TEXT = {
    "README.md", "docs/content/docs/quickstart.mdx", "docs/content/docs/quickstart.zh.mdx",
    "docs/content/docs/(integration)/sdks/android.mdx",
    "docs/content/docs/(integration)/sdks/android.zh.mdx",
    "docs/content/docs/(integration)/sdks/flutter.mdx",
    "docs/content/docs/(integration)/sdks/flutter.zh.mdx",
    "docs/content/docs/(integration)/sdks/ios.mdx",
    "docs/content/docs/(integration)/sdks/ios.zh.mdx",
    "docs/content/docs/(integration)/sdks/react-native.mdx",
}


class PolicyError(Exception):
    """An input is outside the deliberately narrow source policy."""


def require(condition, message):
    if not condition:
        raise PolicyError(message)


def refuse_mirror_environment():
    # Check names only: never read, copy, log or pass their values to Git.
    require(not any(name in os.environ for name in SECRET_NAMES),
            "source policy must run outside the mirror-credential step")


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
            "a full lowercase 40-hex commit SHA is required")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def json_bytes(data, limit):
    require(len(data) <= limit, "JSON input exceeds its byte bound")
    try:
        text = data.decode("utf-8")
        depth, quoted, escaped = 0, False, False
        # Bound containers before parsing, independently of the interpreter's
        # recursion limit. Brackets inside JSON strings do not count.
        for character in text:
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character in "[{":
                depth += 1
                require(depth <= MAX_JSON_DEPTH, "JSON nesting exceeds its depth bound")
            elif character in "]}":
                depth -= 1
                require(depth >= 0, "invalid JSON input")
        def invalid_constant(_value):
            raise PolicyError("nonfinite JSON value")
        return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise PolicyError("invalid JSON input") from error


def load_json(path, limit):
    with Path(path).open("rb") as stream:
        return json_bytes(stream.read(limit + 1), limit)


def producer_context(value, trusted_master):
    fields = {"repository", "workflow_path", "workflow_commit", "ref", "event", "run_id", "run_attempt"}
    require(isinstance(value, dict) and set(value) == fields, "unexpected producer context fields")
    require(value["repository"] == "xybrid-ai/xybrid", "unexpected producer repository")
    require(value["workflow_path"] == WORKFLOW, "unexpected producer workflow path")
    require(sha(value["workflow_commit"]) == trusted_master, "producer/master SHA mismatch")
    require(value["ref"] == "refs/heads/master" and value["event"] == "workflow_dispatch",
            "producer context must name a master dispatch")
    for field in ("run_id", "run_attempt"):
        require(type(value[field]) is int and 0 < value[field] < 2**63, "invalid producer numeric context")
    return dict(value)


class Git:
    """Raw local Git objects only: no checkout, diff drivers, filters or fetch."""

    def __init__(self, repo):
        refuse_mirror_environment()
        executable = shutil.which("git", path=os.defpath)
        require(executable is not None, "system Git is required")
        self.command = [executable, "--no-pager", "--no-replace-objects", "-c", "protocol.allow=never",
                        "-c", "core.fsmonitor=false", "-c", "core.hooksPath=" + os.devnull, "-C", str(repo)]
        # A fixed allowlist, not os.environ.copy(): no credentials inherited.
        self.environment = {
            "PATH": os.defpath, "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1", "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
        }

    def read(self, *arguments, limit=MAX_TREE_OUTPUT, allowed_codes=(0,)):
        with subprocess.Popen(self.command + list(arguments), env=self.environment,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
            data = process.stdout.read(limit + 1)
            if len(data) > limit:
                process.kill()
                process.wait()
                raise PolicyError("Git output exceeds its byte bound")
            require(process.wait() in allowed_codes, "local Git object request failed")
            return data

    def entries(self, commit):
        sha(commit)
        require(self.read("cat-file", "-t", commit, limit=32) == b"commit\n", "SHA must identify a commit")
        result = {}
        for row in self.read("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
            if not row:
                continue
            metadata, raw_path = row.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split()
            require(len(raw_path) <= 4096, "Git path exceeds its byte bound")
            try:
                path = raw_path.decode("utf-8")
            except UnicodeError as error:
                raise PolicyError("non-UTF8 Git path is outside this policy") from error
            require(path not in result and not path.startswith("/") and
                    all(part not in ("", ".", "..") for part in path.split("/")), "invalid Git tree path")
            require(mode in ("100644", "100755", "120000", "160000"), "unsupported Git entry mode")
            require(kind == ("commit" if mode == "160000" else "blob"), "unexpected Git entry type")
            sha(oid)
            result[path] = (mode, kind, oid)
            require(len(result) <= 20000, "Git tree exceeds its entry bound")
        return result

    def blob(self, entry):
        require(entry[1] == "blob", "metadata must be a blob")
        size = self.read("cat-file", "-s", entry[2], limit=32)
        require(size.strip().isdigit() and int(size) <= MAX_METADATA, "metadata blob exceeds its byte bound")
        return self.read("cat-file", "blob", entry[2], limit=MAX_METADATA)

    def tree_digest(self, entries):
        digest = hashlib.sha256(b"xybrid-zzz-source-tree-v1\0")
        total = 0
        cache = {}
        with subprocess.Popen(self.command + ["cat-file", "--batch"], env=self.environment,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL) as process:
            try:
                for path, (mode, kind, oid) in sorted(entries.items()):
                    if kind == "commit":
                        content_hash = hashlib.sha256(b"gitlink\0" + oid.encode()).digest()
                    elif oid in cache:
                        content_hash = cache[oid]
                    else:
                        process.stdin.write(oid.encode() + b"\n")
                        process.stdin.flush()
                        header = process.stdout.readline(128).split()
                        require(len(header) == 3 and header[0] == oid.encode() and header[1] == b"blob" and
                                header[2].isdigit(), "invalid raw Git blob response")
                        size = int(header[2])
                        total += size
                        require(size <= MAX_BLOB and total <= MAX_TREE_BYTES, "source tree exceeds its byte bound")
                        content = hashlib.sha256()
                        remaining = size
                        while remaining:
                            block = process.stdout.read(min(remaining, 65536))
                            require(block, "truncated Git blob")
                            content.update(block)
                            remaining -= len(block)
                        require(process.stdout.read(1) == b"\n", "invalid Git blob terminator")
                        content_hash = content.digest()
                        cache[oid] = content_hash
                    encoded_path = path.encode()
                    digest.update(len(encoded_path).to_bytes(4, "big") + encoded_path)
                    digest.update(mode.encode() + b"\0" + kind.encode() + b"\0" + content_hash)
                process.stdin.close()
                require(process.wait() == 0, "Git blob reader failed")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        return digest.hexdigest()


def version(value):
    require(isinstance(value, str) and len(value) <= 80 and re.fullmatch(
        r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?", value),
        "unsupported release version")
    return value


def replace_one(data, old, new):
    require(data.count(old) == 1, "trusted metadata literal is not unique")
    return data.replace(old, new, 1)


def literals_only(before, after, pairs):
    require(b"\0" not in before and b"\0" not in after, "NUL in metadata")
    for index, (old, new) in enumerate(pairs):
        marker = b"\0policy-" + str(index).encode() + b"\0"
        before = before.replace(old, marker).replace(new, marker)
        after = after.replace(old, marker).replace(new, marker)
    require(before == after, "metadata changed outside approved literals")


def toml(data):
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise PolicyError("invalid TOML metadata") from error


def expected_toml(data, path, old, new, workspace):
    value = copy.deepcopy(toml(data))
    if path == "Cargo.toml":
        value["workspace"]["package"]["version"] = new
    elif isinstance(value.get("package", {}).get("version"), str):
        require(value["package"]["version"] == old, "unsupported package version baseline")
        value["package"]["version"] = new
    elif path == "bindings/python/pyproject.toml":
        require(value["project"]["version"] == old, "unsupported Python version baseline")
        value["project"]["version"] = new

    def dependencies(table):
        for key, child in table.items():
            if key in ("dependencies", "dev-dependencies", "build-dependencies") and isinstance(child, dict):
                for name, dependency in child.items():
                    if not isinstance(dependency, dict) or "path" not in dependency or "version" not in dependency:
                        continue
                    package = dependency.get("package", name)
                    destination = posixpath.normpath(posixpath.join(posixpath.dirname(path), dependency["path"]))
                    if package in workspace and destination == workspace[package]:
                        require(dependency["version"] == old, "unsupported internal dependency version baseline")
                        dependency["version"] = new
            elif isinstance(child, dict):
                dependencies(child)
    dependencies(value)
    return value


def check_metadata(path, before, after, old, new, date, checksum, workspace):
    quoted = [(json.dumps(old).encode(), json.dumps(new).encode())]
    if path == "Cargo.lock":
        expected = copy.deepcopy(toml(before))
        for package in expected.get("package", []):
            if package["name"] in workspace and "source" not in package:
                require(package["version"] == old, "unsupported lockfile workspace version baseline")
                package["version"] = new
        require(toml(after) == expected, "lockfile changed outside workspace package versions")
        literals_only(before, after, quoted)
    elif path == "Cargo.toml" or path in {p + "/Cargo.toml" for p in workspace.values()} or path == "bindings/python/pyproject.toml":
        require(toml(after) == expected_toml(before, path, old, new, workspace),
                "TOML changed outside package/internal dependency versions")
        literals_only(before, after, quoted)
    elif path in JSON_PACKAGES:
        expected = copy.deepcopy(json_bytes(before, MAX_METADATA))
        require(expected["version"] == old, "unsupported JSON package version baseline")
        expected["version"] = new
        if path == "bindings/react-native/package.json" and checksum is not None:
            original = expected["xybrid"]["iosXcframeworkSha256"]
            expected["xybrid"]["iosXcframeworkSha256"] = checksum
            quoted.append((json.dumps(original).encode(), json.dumps(checksum).encode()))
        require(json_bytes(after, MAX_METADATA) == expected, "JSON changed outside approved version/checksum values")
        literals_only(before, after, quoted)
    else:
        expected = before
        if path == "Package.swift":
            expected = replace_one(expected, f'let sdkVersion = "{old}"'.encode(), f'let sdkVersion = "{new}"'.encode())
            if checksum is not None:
                matches = re.findall(rb'^let xybridFFIChecksum = "([0-9a-f]{64})"$', before, re.M)
                require(len(matches) == 1, "unsupported Swift checksum baseline")
                expected = replace_one(expected, b'let xybridFFIChecksum = "' + matches[0] + b'"',
                                       f'let xybridFFIChecksum = "{checksum}"'.encode())
        elif path == "CHANGELOG.md" and old != new:
            expected = replace_one(before, b"## [Unreleased]\n", f"## [Unreleased]\n\n---\n\n## [{new}] - {date}\n".encode())
        elif path == "bindings/flutter/CHANGELOG.md" and old != new:
            expected = replace_one(before, b"## Unreleased\n", f"## Unreleased\n\n## {new}\n".encode())
        elif path in VERSION_TEXT:
            expected = before.replace(old.encode(), new.encode())
        elif path == "bindings/apple/XybridFFI-Info.plist":
            expected = before.replace(f"<string>{old.split('-')[0]}</string>".encode(),
                                      f"<string>{new.split('-')[0]}</string>".encode())
        elif path == "bindings/flutter/example/pubspec.lock":
            pattern = rb'(?m)(^  xybrid_flutter:\n(?:    .*\n)*?    version: ")' + re.escape(old.encode()) + rb'("$)'
            expected, count = re.subn(pattern, lambda match: match[1] + new.encode() + match[2], before)
            require(count == 1, "unsupported Flutter lockfile baseline")
        else:
            patterns = {
                "bindings/flutter/pubspec.yaml": (f"version: {old}", f"version: {new}"),
                "bindings/kotlin/build.gradle.kts": (f'version = "{old}"', f'version = "{new}"'),
                "bindings/react-native/android/build.gradle": (f"ai.xybrid:xybrid-kotlin:{old}", f"ai.xybrid:xybrid-kotlin:{new}"),
                "bindings/python/xybrid/_bolt/__init__.py": (f'PACKAGE_VERSION = "{old}"', f'PACKAGE_VERSION = "{new}"'),
                "bindings/web/src/version.ts": (f'export const SDK_VERSION = "{old}";', f'export const SDK_VERSION = "{new}";'),
            }
            if path in patterns:
                first, second = patterns[path]
                expected = replace_one(before, first.encode(), second.encode())
        require(after == expected, "file changed outside exact trusted metadata rendering")


def source_binding(repo, trusted_master, candidate, release_version, release_date, context, checksum=None):
    refuse_mirror_environment()
    trusted_master, candidate = sha(trusted_master), sha(candidate)
    release_version = version(release_version)
    require(isinstance(release_date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", release_date),
            "invalid release date")
    try:
        datetime.date.fromisoformat(release_date)
    except ValueError as error:
        raise PolicyError("invalid release date") from error
    require(checksum is None or isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum),
            "invalid approved SDK checksum")
    producer = producer_context(context, trusted_master)
    git = Git(repo)
    baseline, source = git.entries(trusted_master), git.entries(candidate)
    require(set(baseline) == set(source), "tracked source paths were added or removed")
    for path in baseline:
        require(baseline[path][:2] == source[path][:2], "tracked source mode/type changed")
    require(git.read("merge-base", "--is-ancestor", trusted_master, candidate, limit=32) == b"",
            "candidate must descend from trusted master")
    for path in ("Cargo.toml", "Cargo.lock", MANIFEST, VERIFIER, "Package.swift", "bindings/react-native/package.json"):
        require(path in baseline and baseline[path][0] == "100644", "required regular source input missing")
    root = toml(git.blob(baseline["Cargo.toml"]))
    old = version(root["workspace"]["package"]["version"])
    workspace = {}
    for member in root["workspace"]["members"]:
        require(isinstance(member, str) and member == posixpath.normpath(member) and not member.startswith(("/", "../")) and
                not any(char in member for char in "*?["), "unsupported workspace member path")
        path = member + "/Cargo.toml"
        require(path in baseline and baseline[path][0] == "100644", "workspace manifest missing")
        name = toml(git.blob(baseline[path]))["package"]["name"]
        require(name not in workspace, "duplicate workspace package name")
        workspace[name] = member
    metadata = VERSION_TEXT | JSON_PACKAGES | {
        "Cargo.toml", "Cargo.lock", "Package.swift", "CHANGELOG.md", "bindings/flutter/CHANGELOG.md",
        "bindings/apple/XybridFFI-Info.plist", "bindings/flutter/example/pubspec.lock",
        "bindings/flutter/pubspec.yaml", "bindings/kotlin/build.gradle.kts",
        "bindings/react-native/android/build.gradle", "bindings/python/pyproject.toml",
        "bindings/python/xybrid/_bolt/__init__.py", "bindings/web/src/version.ts",
    } | {path + "/Cargo.toml" for path in workspace.values()}
    for path in sorted(baseline):
        if path in metadata:
            require(baseline[path][0] == "100644", "metadata must remain a regular nonexecutable file")
            check_metadata(path, git.blob(baseline[path]), git.blob(source[path]), old,
                           release_version, release_date, checksum, workspace)
        else:
            require(baseline[path] == source[path], "code or other nonmetadata source changed")
    result = {
        "schema_version": 1, "kind": "xybrid-zzz-sdk-source-binding",
        "producer_authentication": "NOT_AUTHENTICATED_BY_THIS_HELPER",
        "producer_context": producer, "trusted_master_commit": trusted_master,
        "candidate_commit": candidate,
        "candidate_tree_oid": git.read("rev-parse", "--verify", candidate + "^{tree}", limit=64).decode().strip(),
        "candidate_tree_sha256": git.tree_digest(source),
        "engine_manifest_sha256": hashlib.sha256(git.blob(source[MANIFEST])).hexdigest(),
        "packaged_verifier_sha256": hashlib.sha256(git.blob(source[VERIFIER])).hexdigest(),
        "release_metadata": {"version": release_version, "date": release_date, "approved_xcframework_sha256": checksum},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--trusted-master", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--release-date", required=True)
    parser.add_argument("--producer-context", required=True,
                        help="caller-supplied context; this helper does NOT authenticate it")
    parser.add_argument("--xcframework-sha256")
    parser.add_argument("--compare-binding", help="compare against freshly recomputed source facts and supplied context")
    args = parser.parse_args()
    try:
        refuse_mirror_environment()
        binding = source_binding(args.repo, args.trusted_master, args.candidate, args.version,
                                 args.release_date, load_json(args.producer_context, 8192), args.xcframework_sha256)
        if args.compare_binding:
            # Python treats True == 1 and 1.0 == 1. Canonical JSON preserves
            # their distinct JSON types while ignoring object key ordering.
            require(canonical(load_json(args.compare_binding, 16384)) == canonical(binding),
                    "source binding/context mismatch")
        sys.stdout.buffer.write(canonical(binding))
    except (PolicyError, KeyError, TypeError, OSError) as error:
        # Never print blob/context contents, credential values or Git stderr.
        message = str(error) if isinstance(error, PolicyError) else "invalid or unavailable local input"
        print("source-policy: " + message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
