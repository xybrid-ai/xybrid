#!/usr/bin/env python3
"""Fail when release-prep would reuse Flutter libraries built before a native change.

cargokit publishes the Flutter plugin's native libraries as a GitHub release
named ``precompiled_<crate hash>``, and release-prep's precompile jobs skip
every target that release already holds. The hash covers only the Flutter
crate (every file under ``bindings/flutter/rust/src`` plus its ``Cargo.toml``,
``Cargo.lock``, ``build.rs`` and ``cargokit.yaml``), not the workspace crates,
vendored llama.cpp or Bazel configuration the libraries are built from. A
native change that reaches a release branch after the version bump therefore
leaves the hash alone, and the release ships libraries built without it:
v0.16.0 nearly shipped llama.cpp v0.7.0 that way, after a master merge.

The release may have been built from any commit in REF's history that has
REF's crate hash: the bump, a later push, or a commit before a revert or on
the other side of a merge. Any file that feeds the libraries and differs
between one of those commits and REF fails the check. That holds even while
the release is still empty and this run would build from REF: once it
uploads, later runs cannot tell which of those commits it built. The fix is
to change a line of ``bindings/flutter/rust/Cargo.toml``, which moves the hash
and makes every precompile job rebuild.

History does not show commits a rebase or a re-cut of the branch dropped. With
``--github-repo`` the check also looks the release up, and fails when it was
published before the first commit with REF's hash was committed: its
libraries then came from a commit REF does not contain. Commit dates stand in
for push times, so a branch rewritten while an earlier run was still building
can slip through.

    python3 tools/scripts/check_flutter_precompiled_inputs.py [--github-repo OWNER/NAME] [REF]

It needs REF's full history (``fetch-depth: 0``), and ``gh`` with a token for
the lookup. Exit status is 0 when the release is current, 1 when it may not
be, and 2 when the check cannot run.

Only a release branch after its bump gives a meaningful answer. On master the
history check reports every native change since the Flutter crate last
changed, and right after a release the lookup fails: the release's squash
merge replaced the commits ``precompiled_<hash>`` was built from. The next
bump moves the hash either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CRATE = "bindings/flutter/rust"
# Hashed besides everything under src/ (CrateHash.getFiles in
# bindings/flutter/cargokit/build_tool/lib/src/crate_hash.dart).
CRATE_FILES = ("Cargo.toml", "Cargo.lock", "build.rs", "cargokit.yaml")
# Dart's UTF-8 decoder drops it from the start of a file before cargokit
# splits the text into lines.
UTF8_BOM = b"\xef\xbb\xbf"

# What the libraries are built from. Bazel builds every target but
# i686-linux-android from //bindings/flutter/rust, which links the workspace
# crates (and macros, a proc-macro of xybrid-sdk) and vendored native code,
# using toolchains set up by bazel/, the root Bazel files and the patches
# MODULE.bazel applies. Its third-party crates come from one resolution of the
# whole workspace that merges every member's features (the library's tokio
# gets "full" from integration-tests), so every Cargo.toml counts, wherever it
# lives. cargokit's build tool cargo-builds the i686 slice, which also reads
# .cargo/. Workflows are not tracked: release-prep's build flags change
# rarely, and CI-only edits after the bump are routine.
NATIVE_DIRS = (
    "crates/", "macros/", "vendor/", "bazel/", ".cargo/",
    f"{CRATE}/", "bindings/flutter/cargokit/build_tool/",
)
NATIVE_ROOT_FILES = frozenset({
    ".bazelignore", ".bazelrc", ".bazelversion", ".gitmodules", "BUILD.bazel",
    "Cargo.lock", "MODULE.bazel", "MODULE.bazel.lock", "rust-toolchain", "rust-toolchain.toml",
})
# cargo's download list of prebuilt llama.cpp archives. No Bazel target reads
# it, and cargo uses a row only while wrapper.cpp, wrapper.h, build.rs and the
# llama.cpp pin match the ones it records; those are checked themselves, so a
# manifest change can only swap a source build for an archive of the same
# sources.
NATIVES_MANIFEST = "crates/llama-cpp-sys/natives-manifest.txt"
# A crate's tests, benches and examples compile only into their own targets:
# Bazel's library targets take src/**/*.rs and each test target names its
# file, and cargo builds them only for cargo test, bench or run --example. No
# library include!s a file from them. Release branches do pick up test fixes
# after the bump, and failing on those would teach maintainers to apply the
# fix without reading the list.
_CRATE_ONLY = re.compile(r"^(?:crates/[^/]+|macros|bindings/flutter/rust)/(?:tests|benches|examples)/")


class CheckError(Exception):
    """The check cannot run: an unknown ref, shallow history or a failed lookup."""


@dataclass(frozen=True)
class Commit:
    sha: str
    committed: int  # committer date, seconds since the epoch
    tree: str
    parents: tuple[str, ...]


@dataclass(frozen=True)
class Result:
    commit: str
    crate_hash: str
    # Commits in REF's history with REF's crate hash that no other such commit
    # precedes, oldest first: usually just the bump.
    origins: tuple[str, ...]
    # The commit with REF's crate hash committed first, and when (seconds since
    # the epoch): the release cannot hold anything built from REF's history
    # before then.
    oldest: str
    since: int
    # Feeding the libraries and differing between REF and a commit with its
    # hash; empty when the history check passes.
    stale: tuple[str, ...]
    # Commits after an origin that changed one of `stale`, oldest first.
    culprits: tuple[str, ...]


def git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(["git", *args], cwd=repo, input=stdin, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise CheckError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def feeds_libraries(path: str) -> bool:
    """Whether a change to PATH can change the shipped Flutter libraries."""
    if path.endswith(".md") or path == NATIVES_MANIFEST:
        return False
    if path.rsplit("/", 1)[-1] == "Cargo.toml":
        return True
    if _CRATE_ONLY.match(path):
        return False
    if "/" not in path:
        return path in NATIVE_ROOT_FILES or path.endswith(".patch")
    return path.startswith(NATIVE_DIRS)


def is_hash_input(path: str) -> bool:
    return path.startswith(f"{CRATE}/src/") or path in {f"{CRATE}/{name}" for name in CRATE_FILES}


def history(repo: Path, commit: str) -> list[Commit]:
    """Every commit COMMIT contains, children before their parents."""
    out = git(repo, "rev-list", "--topo-order", "--no-commit-header", "--format=%H %ct %T %P", commit)
    commits = []
    for line in out.decode().splitlines():
        sha, committed, tree, *parents = line.split()
        commits.append(Commit(sha, int(committed), tree, tuple(parents)))
    return commits


def crate_keys(repo: Path, shas: list[str]) -> list[tuple[str, ...]]:
    """Per commit, the object ids of everything cargokit hashes: equal ids, equal hashes."""
    names = ("src", *CRATE_FILES)
    request = "".join(f"{sha}:{CRATE}/{name}\n" for sha in shas for name in names)
    out = git(repo, "cat-file", "--batch-check=%(objectname)", stdin=request.encode()).decode().splitlines()
    oids = ["" if line.endswith(" missing") else line for line in out]
    return [tuple(oids[i:i + len(names)]) for i in range(0, len(oids), len(names))]


def hash_inputs(repo: Path, commit: str) -> tuple[tuple[str, str], ...] | None:
    """The (path, blob) pairs cargokit hashes at COMMIT, in its order.

    None when the crate has no src/ there, where cargokit cannot hash at all.
    """
    listing = git(repo, "ls-tree", "-r", "-z", "--full-tree", commit, "--", CRATE)
    blobs = {}
    has_src = False
    for entry in listing.decode(errors="surrogateescape").split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, kind, oid = meta.split()
        has_src = has_src or path.startswith(f"{CRATE}/src/")
        # cargokit lists src/ without following links and keeps only files.
        if kind == "blob" and mode != "120000":
            blobs[path] = oid
    if not has_src:
        return None
    paths = sorted(path for path in blobs if path.startswith(f"{CRATE}/src/"))
    paths += [f"{CRATE}/{name}" for name in CRATE_FILES if f"{CRATE}/{name}" in blobs]
    return tuple((path, blobs[path]) for path in paths)


def read_blobs(repo: Path, oids: set[str]) -> dict[str, bytes]:
    ordered = sorted(oids)
    batch = git(repo, "cat-file", "--batch", stdin="".join(f"{oid}\n" for oid in ordered).encode())
    blobs = {}
    position = 0
    for oid in ordered:
        header_end = batch.index(b"\n", position)
        size = int(batch[position:header_end].split()[2])
        blobs[oid] = batch[header_end + 1:header_end + 1 + size]
        position = header_end + 1 + size + 1
    return blobs


def crate_hash(inputs: tuple[tuple[str, str], ...], blobs: dict[str, bytes]) -> str:
    """cargokit's crate hash: the inputs' lines, hashed back to back, cut to 128 bits.

    cargokit decodes each file, which drops a leading byte order mark, and
    hashes its lines without their line endings, so blank lines and line
    breaks do not count.
    """
    digest = hashlib.sha256()
    for _, oid in inputs:
        content = blobs[oid]
        if content.startswith(UTF8_BOM):
            content = content[len(UTF8_BOM):]
        digest.update(content.replace(b"\r", b"").replace(b"\n", b""))
    return digest.hexdigest()[:32]


def changed_files(repo: Path, old: str, new: str, *paths: str) -> set[str]:
    # Plumbing: no rename pairing (both sides of a move are listed) and no
    # user config hiding submodule pointer changes.
    pathspecs = [f":(literal){path}" for path in paths]
    out = git(repo, "diff-tree", "-r", "-z", "--name-only", "--no-renames", old, new, "--", *pathspecs)
    return {path for path in out.decode(errors="surrogateescape").split("\0") if path}


def check(repo: Path, ref: str = "HEAD") -> Result:
    """Compare REF with every commit in its history that has its crate hash."""
    try:
        commit = git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()
    except CheckError:
        raise CheckError(f"{ref} is not a commit") from None
    if git(repo, "rev-parse", "--is-shallow-repository").decode().strip() == "true":
        raise CheckError("the history is shallow, so older commits with the same Flutter crate hash "
                         "may be missing; fetch the full history (actions/checkout fetch-depth: 0)")
    commits = history(repo, commit)
    keys = crate_keys(repo, [c.sha for c in commits])
    first_with_key = {}
    for c, key in zip(commits, keys):
        first_with_key.setdefault(key, c.sha)
    listings = {key: hash_inputs(repo, sha) for key, sha in first_with_key.items()}
    if listings[keys[0]] is None:
        raise CheckError(f"{ref} has no {CRATE}/src for cargokit to hash")
    blobs = read_blobs(repo, {oid for listing in listings.values() if listing for _, oid in listing})
    hashes = {key: None if listing is None else crate_hash(listing, blobs) for key, listing in listings.items()}
    current = hashes[keys[0]]
    same = {c.sha for c, key in zip(commits, keys) if hashes[key] == current}

    # Parents come first in this order, so `after` (descends from a commit
    # with REF's hash) is known for every parent before its child.
    after: set[str] = set()
    origins = []
    for c in reversed(commits):
        if any(parent in same or parent in after for parent in c.parents):
            after.add(c.sha)
        elif c.sha in same:
            origins.append(c.sha)
    oldest = min((c for c in commits if c.sha in same), key=lambda c: c.committed)

    stale = set()
    for tree in {c.tree for c in commits if c.sha in same} - {commits[0].tree}:
        stale |= {path for path in changed_files(repo, tree, commits[0].tree) if feeds_libraries(path)}
    culprits = []
    if stale:
        lineage = same | after
        for c in reversed(commits):
            if c.sha in after and any(parent in lineage and changed_files(repo, parent, c.sha, *stale)
                                      for parent in c.parents):
                culprits.append(c.sha)
    return Result(commit, current, tuple(origins), oldest.sha, oldest.committed, tuple(sorted(stale)),
                  tuple(culprits))


def release_published(github_repo: str, tag: str) -> int | None:
    """When release TAG first held an asset, in seconds since the epoch; None if it holds none."""
    failure = f"looking up release {tag} on {github_repo} failed"
    try:
        result = subprocess.run(["gh", "api", f"repos/{github_repo}/releases/tags/{tag}"],
                                capture_output=True, text=True)
    except OSError as error:
        raise CheckError(f"{failure}: {error}") from None
    if result.returncode != 0:
        if "(HTTP 404)" in result.stderr:
            return None
        raise CheckError(f"{failure}: {result.stderr.strip()}")
    try:
        release = json.loads(result.stdout)
    except ValueError as error:
        raise CheckError(f"{failure}: {error}") from None
    if not release.get("assets"):
        return None
    # A release's created_at is the date of the commit its tag points at
    # (master, for these); published_at is when cargokit created it.
    stamps = [release.get("published_at")] + [asset.get("created_at") for asset in release["assets"]]
    return min(int(datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
               for stamp in stamps if stamp)


def describe(repo: Path, sha: str) -> str:
    return git(repo, "log", "-1", "--no-show-signature", "--format=%h %s", sha).decode().strip()


def utc(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


FIX = f"""\
Fix: change the text of a line in {CRATE}/Cargo.toml (rewording
its comment about this check will do) and commit. The new crate hash makes
every precompile job rebuild; blank lines and line breaks do not move it."""


def report(repo: Path, ref: str, result: Result, *, looked_up: bool = False,
           published: int | None = None) -> tuple[bool, str]:
    """Whether REF passes, and what to print.

    PUBLISHED is what release_published returned, when LOOKED_UP.
    """
    target = f"{ref} at {result.commit[:12]}"
    release = f"precompiled_{result.crate_hash}"
    origins = "\n".join(f"  {describe(repo, sha)}" for sha in result.origins)
    if published is not None and published < result.since:
        return False, f"""\
ERROR: {release} may hold stale Flutter libraries for {target}.

That release was published {utc(published)}, before any commit in {ref}'s
history had its crate hash; the first that has it was committed {utc(result.since)}:
  {describe(repo, result.oldest)}
Its libraries came from a commit {ref} does not contain, one that a rebase,
a re-cut or a squash merge replaced, so nothing shows what they were built
from. The precompile jobs would reuse them.

{FIX}"""
    empty = looked_up and published is None
    if not result.stale:
        if empty:
            state = f"It holds no libraries yet, so every precompile job builds them from {ref}."
        else:
            state = (f"The Flutter crate hash first appeared in\n{origins}\n"
                     "and nothing the libraries are built from has changed since.")
            if not looked_up:
                state += "\nWithout --github-repo, a rebased or re-cut branch is not detected."
        return True, f"OK: {release} matches {target}.\n{state}"
    if empty:
        state = (f"It holds no libraries yet, so this run would build them from {ref}, but\n"
                 "once they are uploaded, later runs cannot tell them from libraries built\n"
                 "before these files changed:")
    elif looked_up:
        state = (f"It holds libraries since {utc(published)}, and the precompile jobs reuse\n"
                 "them, but they may have been built before these files changed:")
    else:
        state = ("If it already holds libraries, the precompile jobs reuse them, and they\n"
                 "may have been built before these files changed:")
    files = "\n".join(
        f"  {path}" + ("  (edited without moving the hash)" if is_hash_input(path) else "")
        for path in result.stale)
    culprits = "\n".join(f"  {describe(repo, sha)}" for sha in result.culprits)
    return False, f"""\
ERROR: {release} may hold stale Flutter libraries for {target}.

cargokit names that release after a hash of {CRATE}, which
first appeared in
{origins}
{state}
{files}
They came in with:
{culprits}

{FIX}"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("ref", nargs="?", default="HEAD", help="commit to check (default: HEAD)")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)")
    parser.add_argument("--github-repo", metavar="OWNER/NAME",
                        help="also look the release up on this GitHub repository (needs gh and GH_TOKEN)")
    args = parser.parse_args(argv)
    try:
        result = check(args.repo, args.ref)
        published = None
        if args.github_repo:
            published = release_published(args.github_repo, f"precompiled_{result.crate_hash}")
        ok, message = report(args.repo, args.ref, result, looked_up=bool(args.github_repo), published=published)
    except CheckError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(message, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
