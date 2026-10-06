#!/usr/bin/env python3
"""Fetch and privately stage the pinned Kitten TTS 2 engine before native builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2] / "crates/zzz-sys/natives-manifest.json"
)
DEFAULT_ROOT = Path.home() / ".cache/xybrid/zzz"
PROFILE = "kitten-tts2"
STAGED_ARCHIVE = ".pinned-archive.tar.gz"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
CHUNK_BYTES = 1024 * 1024
ZIG_TARGETS = {
    "aarch64-apple-darwin": "aarch64-macos-none",
    "aarch64-linux-android": "aarch64-linux-android",
    "x86_64-unknown-linux-gnu": "x86_64-linux-gnu",
}


class StageError(ValueError):
    """An error safe to display without private paths, API bodies or receipts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StageError(message)


def expect(record: dict, key: str, value: object, label: str) -> None:
    # JSON comparison also distinguishes true from 1, including nested API versions.
    require(
        key in record
        and json.dumps(record[key], sort_keys=True)
        == json.dumps(value, sort_keys=True),
        f"{label} mismatch: {key}",
    )


def json_object(data: bytes, label: str) -> dict:
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError):
        raise StageError(f"invalid {label}") from None
    require(isinstance(result, dict), f"invalid {label}")
    return result


@dataclass(frozen=True)
class Pin:
    release_tag: str
    entry: dict

    @property
    def files(self) -> set[str]:
        return {
            self.entry["library"],
            *self.entry["headers"],
            "RECEIPT.json",
            "share/zzz_embed.json",
        }

    @property
    def directories(self) -> set[str]:
        return {"lib", "include", "share"}

    @property
    def key(self) -> tuple[str, ...]:
        return (self.release_tag, self.entry["target"], PROFILE, self.entry["sha256"])


def select_pin(target: str, manifest: Path = DEFAULT_MANIFEST) -> Pin:
    require(target in ZIG_TARGETS, "unsupported zzz target")
    data = json_object(manifest.read_bytes(), "engine manifest")
    expect(data, "schema_version", 1, "manifest")
    expect(data, "engine", "zzz", "manifest")
    tag = data.get("release_tag")
    require(
        isinstance(tag, str)
        and re.fullmatch(r"v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", tag),
        "invalid release tag",
    )
    entries = data.get("archives")
    require(
        isinstance(entries, list) and all(isinstance(e, dict) for e in entries),
        "invalid archive pins",
    )
    matches = [
        e for e in entries if e.get("target") == target and e.get("profile") == PROFILE
    ]
    require(len(matches) == 1, "missing or duplicate target/profile pin")
    entry = matches[0]
    for field, value in {
        "profile": PROFILE,
        "models": [PROFILE],
        "backend": "cpu",
        "receipt_schema": 3,
        "apis": {"zzz_embed": 1},
        "library": "lib/libzzz_embed.a",
        "headers": ["include/zzz_embed.h"],
    }.items():
        expect(entry, field, value, "pin")
    slices = {
        "aarch64-apple-darwin": "macos-arm64",
        "aarch64-linux-android": "android-arm64",
        "x86_64-unknown-linux-gnu": "linux-x86_64",
    }
    expect(entry, "slice", slices[target], "pin")
    expect(entry, "archive", f"zzz-{PROFILE}-{slices[target]}-{tag}.tar.gz", "pin")
    require(
        isinstance(entry.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]),
        "invalid archive checksum pin",
    )
    for field in ("cpu", "deployment_minimum"):
        require(
            isinstance(entry.get(field), str) and bool(entry[field]),
            f"invalid pin: {field}",
        )
    libs = entry.get("system_libraries")
    require(
        isinstance(libs, list) and libs and all(isinstance(v, str) and v for v in libs),
        "invalid system library pins",
    )
    return Pin(tag, entry)


def validate_receipt(receipt: dict, metadata: dict, pin: Pin) -> None:
    entry = pin.entry
    for field, value in {
        "schema_version": entry["receipt_schema"],
        "release_tag": pin.release_tag,
        "engine_version": pin.release_tag[1:],
        "abi_version": entry["apis"]["zzz_embed"],
        "target": ZIG_TARGETS[entry["target"]],
        "position_independent": True,
        "header": entry["headers"][0],
    }.items():
        expect(receipt, field, value, "receipt")
    for field in (
        "slice",
        "profile",
        "models",
        "apis",
        "library",
        "headers",
        "backend",
        "cpu",
        "deployment_minimum",
    ):
        expect(receipt, field, entry[field], "receipt")
    dependencies = receipt.get("dependencies")
    require(isinstance(dependencies, dict), "invalid receipt dependencies")
    expect(
        dependencies,
        "system_libraries",
        entry["system_libraries"],
        "receipt dependencies",
    )
    expect(dependencies, "compiler_runtime", "bundled", "receipt dependencies")
    resolved = receipt.get("resolved_target")
    minimum = re.escape(entry["deployment_minimum"])
    target_patterns = {
        "aarch64-apple-darwin": rf"aarch64-macos\.{minimum}(?:\.\.\.[0-9.]+)?-none",
        "aarch64-linux-android": rf"aarch64-linux(?:\.[0-9.]+)?-android\.{minimum}",
        "x86_64-unknown-linux-gnu": rf"x86_64-linux(?:\.[0-9.]+)?-gnu\.{re.escape(entry['deployment_minimum'].removeprefix('glibc-'))}",
    }
    require(
        isinstance(resolved, str)
        and re.fullmatch(target_patterns[entry["target"]], resolved),
        "receipt mismatch: resolved_target",
    )
    if entry["target"] == "aarch64-linux-android":
        expect(receipt, "android_api", int(entry["deployment_minimum"]), "receipt")
        expect(receipt, "elf_max_page_size", 16384, "receipt")
    for field in ("apis", "models", "headers", "backend"):
        expect(metadata, field, entry[field], "embedded metadata")
    expect(metadata, "abi_version", entry["apis"]["zzz_embed"], "embedded metadata")
    expect(metadata, "target", resolved, "embedded metadata")


def open_regular(path: Path):
    require(not path.is_symlink(), "input links are not allowed")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise StageError("expected a regular input file")
    return os.fdopen(descriptor, "rb")


def inspect_archive(
    path: Path, pin: Pin, destination: Path | None = None
) -> dict[str, str]:
    """Verify compressed bytes first; never use tarfile.extract or trust a stamp."""
    with open_regular(path) as compressed:
        require(
            os.fstat(compressed.fileno()).st_size <= MAX_ARCHIVE_BYTES,
            "archive exceeds size limit",
        )
        digest = hashlib.file_digest(compressed, "sha256").hexdigest()
        require(digest == pin.entry["sha256"], "archive checksum mismatch")
        compressed.seek(0)
        with tarfile.open(fileobj=compressed, mode="r:gz") as archive:
            root = pin.entry["archive"].removesuffix(".tar.gz")
            files: dict[str, tarfile.TarInfo] = {}
            seen = set()
            total = 0
            for member in archive:
                require(
                    member.name not in seen and len(seen) < 32,
                    "duplicate or excessive archive entries",
                )
                seen.add(member.name)
                relative = member.name.removeprefix(root + "/")
                if member.isdir():
                    require(
                        member.name == root
                        or (
                            relative in pin.directories
                            and member.name == f"{root}/{relative}"
                        ),
                        "unexpected archive directory",
                    )
                else:
                    require(
                        member.isreg()
                        and relative in pin.files
                        and member.name == f"{root}/{relative}",
                        "unsafe or unexpected archive entry",
                    )
                    require(
                        0 < member.size <= MAX_PAYLOAD_BYTES, "invalid payload size"
                    )
                    total += member.size
                    files[relative] = member
                require(total <= MAX_PAYLOAD_BYTES, "payload exceeds size limit")
            require(set(files) == pin.files, "missing archive payload")
            documents = []
            for name in ("RECEIPT.json", "share/zzz_embed.json"):
                require(
                    files[name].size <= MAX_JSON_BYTES, "metadata exceeds size limit"
                )
                with archive.extractfile(files[name]) as source:
                    documents.append(json_object(source.read(), name))
            validate_receipt(*documents, pin)
            hashes = {}
            for name, member in files.items():
                digest = hashlib.sha256()
                with archive.extractfile(member) as source:
                    if destination is None:
                        hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
                        continue
                    output = destination / name
                    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    with output.open("xb") as staged:
                        output.chmod(0o600)
                        while chunk := source.read(CHUNK_BYTES):
                            digest.update(chunk)
                            staged.write(chunk)
                hashes[name] = digest.hexdigest()
            return hashes


def private_directory(path: Path) -> None:
    require(not path.is_symlink(), "private directory must not be a link")
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    require(
        path.is_dir() and not path.stat().st_mode & 0o077,
        "staging and cache directories require private permissions (0700)",
    )


def verify_staged(path: Path, pin: Pin) -> Path:
    require(
        not path.is_symlink() and path.is_dir(), "missing or unsafe staged directory"
    )
    expected = pin.files | {STAGED_ARCHIVE}
    found = set()
    for current, directories, files in os.walk(path, followlinks=False):
        directory = Path(current)
        require(
            not directory.stat().st_mode & 0o077,
            "staged directory requires private permissions",
        )
        for name in directories + files:
            item = directory / name
            relative = item.relative_to(path).as_posix()
            require(not item.is_symlink(), "staged links are not allowed")
            require(
                not item.stat().st_mode & 0o077,
                "staged inputs require private permissions",
            )
            if name in directories:
                require(relative in pin.directories, "unexpected staged directory")
            else:
                require(
                    item.is_file() and relative in expected, "unexpected staged file"
                )
                found.add(relative)
    require(found == expected, "missing staged payload or pinned archive")
    hashes = inspect_archive(path / STAGED_ARCHIVE, pin)
    for name, expected_hash in hashes.items():
        with open_regular(path / name) as source:
            require(
                hashlib.file_digest(source, "sha256").hexdigest() == expected_hash,
                "staged payload differs from pinned archive",
            )
    return path.absolute()


def github_environment(use_gh_auth: bool) -> tuple[str, dict[str, str]]:
    repository = os.environ.get("ZZZ_RELEASE_REPOSITORY", "")
    require(
        re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository) is not None,
        "set ZZZ_RELEASE_REPOSITORY to OWNER/REPO",
    )
    token = os.environ.get("ZZZ_GITHUB_TOKEN")
    require(
        bool(token) or (use_gh_auth and not os.environ.get("CI")),
        "set ZZZ_GITHUB_TOKEN; local gh credentials require --use-gh-auth",
    )
    environment = dict(os.environ)
    for name in (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GH_DEBUG",
        "GH_HOST",
        "ZZZ_GITHUB_TOKEN",
        "ZZZ_RELEASE_REPOSITORY",
    ):
        environment.pop(name, None)
    environment["GH_PROMPT_DISABLED"] = "1"
    if token:
        environment["GH_TOKEN"] = token
    return repository, environment


def github_api(endpoint: str, environment: dict[str, str], output=None):
    accept = (
        "application/octet-stream"
        if output is not None
        else "application/vnd.github+json"
    )
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                "--hostname",
                "github.com",
                "--method",
                "GET",
                "-H",
                f"Accept: {accept}",
                "-H",
                "X-GitHub-Api-Version: 2022-11-28",
                endpoint,
            ],
            env=environment,
            stdout=output if output is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise StageError(
            "authenticated release request failed; check gh installation and access"
        ) from None
    require(
        result.returncode == 0,
        "authenticated release request failed; check credentials and release access",
    )
    if output is None:
        require(
            len(result.stdout) <= MAX_JSON_BYTES, "release response exceeds size limit"
        )
        return json_object(result.stdout, "release response")


def fetch_archive(pin: Pin, cache: Path, use_gh_auth: bool = False) -> Path:
    directory = cache.joinpath(*pin.key)
    archive = directory / "archive.tar.gz"
    private_directory(cache)
    for component in pin.key:
        cache /= component
        private_directory(cache)
    if archive.exists() or archive.is_symlink():
        require(
            not archive.stat().st_mode & 0o077,
            "cached archive requires private permissions",
        )
        inspect_archive(archive, pin)
        return archive
    repository, environment = github_environment(use_gh_auth)
    release = github_api(
        f"repos/{repository}/releases/tags/{pin.release_tag}", environment
    )
    expect(release, "tag_name", pin.release_tag, "release")
    expect(release, "draft", False, "release")
    assets = release.get("assets")
    require(
        isinstance(assets, list) and all(isinstance(a, dict) for a in assets),
        "invalid release assets",
    )
    matches = [a for a in assets if a.get("name") == pin.entry["archive"]]
    require(len(matches) == 1, "missing or duplicate pinned release asset")
    asset = matches[0]
    require(
        type(asset.get("id")) is int and asset["id"] > 0, "invalid release asset id"
    )
    require(
        type(asset.get("size")) is int and 0 < asset["size"] <= MAX_ARCHIVE_BYTES,
        "invalid release asset size",
    )
    expect(asset, "state", "uploaded", "release asset")
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=directory) as temp:
        downloaded = Path(temp) / "archive.tar.gz"
        with downloaded.open("xb") as output:
            downloaded.chmod(0o600)
            github_api(
                f"repos/{repository}/releases/assets/{asset['id']}", environment, output
            )
        require(
            downloaded.stat().st_size == asset["size"], "release asset size mismatch"
        )
        inspect_archive(downloaded, pin)
        os.replace(downloaded, archive)
    return archive


def stage_archive(archive: Path, pin: Pin, destination: Path) -> Path:
    inspect_archive(archive, pin)
    private_directory(destination)
    parent = destination
    for component in pin.key[:-1]:
        parent /= component
        private_directory(parent)
    staged = parent / pin.entry["sha256"]
    if staged.exists() or staged.is_symlink():
        return verify_staged(staged, pin)
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=parent) as temp:
        temporary = Path(temp)
        copied = temporary / STAGED_ARCHIVE
        with open_regular(archive) as source, copied.open("xb") as output:
            copied.chmod(0o600)
            shutil.copyfileobj(source, output, CHUNK_BYTES)
        inspect_archive(copied, pin, temporary)
        verify_staged(temporary, pin)
        try:
            temporary.rename(staged)
        except OSError:
            if not staged.exists():
                raise
            verify_staged(staged, pin)
    return verify_staged(staged, pin)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Requires Python 3.11+ and gh for downloads. Set ZZZ_RELEASE_REPOSITORY and ZZZ_GITHUB_TOKEN in private CI configuration. Outputs the slice directory for XYBRID_ZZZ_PREBUILT_DIR. Keep staging, receipts and archives out of public artifacts and caches.",
    )
    parser.add_argument("--target", required=True, choices=sorted(ZIG_TARGETS))
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_ROOT / "staged",
        help="Private staging base; slices are keyed by release/target/profile/checksum.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_ROOT / "archives",
        help="Private archive cache; every hit is reverified.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use a local compressed archive instead of downloading.",
    )
    parser.add_argument(
        "--use-gh-auth",
        action="store_true",
        help="Explicitly allow configured gh credentials for local downloads (never CI).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify XYBRID_ZZZ_PREBUILT_DIR without fetching or staging.",
    )
    args = parser.parse_args(argv)
    try:
        pin = select_pin(args.target)
        prebuilt = os.environ.get("XYBRID_ZZZ_PREBUILT_DIR")
        require(
            not (prebuilt and args.archive), "choose a staged override or local archive"
        )
        if prebuilt:
            result = verify_staged(Path(prebuilt), pin)
        else:
            require(
                not args.verify_only, "--verify-only requires XYBRID_ZZZ_PREBUILT_DIR"
            )
            archive = args.archive or fetch_archive(
                pin, args.cache_dir, args.use_gh_auth
            )
            result = stage_archive(archive, pin, args.dest)
        print(result)
        return 0
    except StageError as error:
        print(f"zzz-pull: {error}", file=sys.stderr)
    except (OSError, ValueError, tarfile.TarError, EOFError):
        # Exceptions can embed private paths, receipt values, or API response bodies.
        print(
            "zzz-pull: unable to verify or stage the pinned engine inputs",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
