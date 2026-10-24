#!/usr/bin/env python3
"""Trusted Apple SDK candidate producer; genuine schema-5 pins are mandatory."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tempfile


def sibling(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
SOURCE = sibling("xybrid_stage_source_policy", ROOT / "tools/scripts/zzz_source_policy.py")
PRODUCER = sibling("xybrid_stage_producer_verify", ROOT / "tools/scripts/zzz_producer_verify.py")
PULL = sibling("xybrid_stage_pinned_pull", ROOT / "crates/zzz-sys/zzz_pull.py")
Error = PRODUCER.VerificationError
require = PRODUCER.require
FORBIDDEN_ENV = (*PRODUCER.FORBIDDEN_ENV, "BUILDBUDDY_API_KEY", "SCCACHE_GHA_ENABLED",
                 "SCCACHE_S3_KEY_PREFIX", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                 "AWS_SESSION_TOKEN", "AZURE_STORAGE_KEY", "GOOGLE_APPLICATION_CREDENTIALS")
STAGING_REPOSITORIES = {
    "zzz_linux": "x86_64-unknown-linux-gnu", "zzz_macos": "aarch64-apple-darwin",
    "zzz_android": "aarch64-linux-android", "zzz_ios": "aarch64-apple-ios",
    "zzz_ios_sim": "aarch64-apple-ios-sim", "zzz_linux_arm64": "aarch64-unknown-linux-gnu",
}
STAGING_TARGETS = tuple(STAGING_REPOSITORIES.values())
SUBMODULES = {"vendor/llama-cpp": "https://github.com/ggml-org/llama.cpp",
              "vendor/whisper-cpp": "https://github.com/ggml-org/whisper.cpp"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # Reserve >12 MiB for the 512 MiB Actions ZIP wrapper.
MAX_BINDING = 16 * 1024
PATH = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"
APPLE_TARGET = "//bindings/apple:XybridFFI_zzz"
APPLE_ARCHIVE = "XybridFFI_zzz.xcframework.zip"


def credential_free():
    require(not any(name in os.environ for name in FORBIDDEN_ENV),
            "trusted producer preparation requires a credential-free process")


def github_context(environment):
    sha = environment.get("GITHUB_SHA")
    require(type(sha) is str and re.fullmatch(r"[0-9a-f]{40}", sha), "invalid runtime master SHA")
    expected = {"GITHUB_REPOSITORY": "xybrid-ai/xybrid", "GITHUB_REF": "refs/heads/master",
                "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_WORKFLOW_SHA": sha,
                "GITHUB_WORKFLOW_REF": "xybrid-ai/xybrid/.github/workflows/trusted-zzz-sdk.yml@refs/heads/master",
                "GITHUB_RUN_ATTEMPT": "1"}
    require(all(environment.get(key) == value for key, value in expected.items()),
            "producer requires its exact master workflow dispatch on attempt 1")
    run = environment.get("GITHUB_RUN_ID")
    require(type(run) is str and re.fullmatch(r"[1-9][0-9]{0,18}", run) and int(run) < 2**63,
            "invalid runtime run ID")
    return SOURCE.producer_context({"repository": expected["GITHUB_REPOSITORY"],
                                   "workflow_path": SOURCE.WORKFLOW, "workflow_commit": sha,
                                   "ref": expected["GITHUB_REF"], "event": expected["GITHUB_EVENT_NAME"],
                                   "run_id": int(run), "run_attempt": 1}, sha)


def schema5_apple_manifest(manifest):
    require(type(manifest) is dict and set(manifest) == {"schema_version", "engine", "release_tag", "archives"} and
            type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1 and
            manifest.get("engine") == "zzz" and type(manifest.get("release_tag")) is str and
            re.fullmatch(r"v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", manifest["release_tag"]),
            "published schema-5 Apple pins are required")
    require(manifest["release_tag"] not in PULL.FROZEN_RELEASE_TAGS,
            "frozen releases cannot supply schema-5 Apple pins")
    entries = manifest.get("archives")
    require(type(entries) is list and 0 < len(entries) <= 16 and all(type(entry) is dict for entry in entries),
            "published schema-5 Apple pins are required")
    require(all(type(entry.get("receipt_schema")) is int and entry["receipt_schema"] == 5 and
                entry.get("notice_files") == PULL.SCHEMA5_NOTICE_FILES for entry in entries),
            "all producer engine inputs require the reviewed schema-5 notices")
    for target in STAGING_TARGETS:
        require(sum(entry.get("target") == target for entry in entries) == 1,
                "genuine schema-5 pins are required for the exact six staging repositories")


def trusted_pins(manifest_path=None):
    manifest_path = manifest_path or PULL.DEFAULT_MANIFEST
    manifest = SOURCE.json_bytes(manifest_path.read_bytes(), SOURCE.MAX_METADATA)
    schema5_apple_manifest(manifest)
    return {target: PULL.select_pin(target, manifest_path) for target in STAGING_TARGETS}


def file_identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def source_tree_digest(root, entries):
    """Compare actual source bytes, including ignored files; never run Git filters."""
    tree = {}
    for path, metadata in entries.items():
        parts = path.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = metadata
    actual, total = {}, 0

    def walk(fd, expected, prefix="", depth=0):
        nonlocal total
        require(depth <= 64, "materialized source depth exceeds its bound")
        before = os.fstat(fd)
        seen = set()
        with os.scandir(fd) as rows:
            for row in rows:
                if not prefix and row.name == ".git":
                    continue  # Fresh checkout metadata is trusted infrastructure, never build input.
                require(row.name in expected and row.name not in seen and len(seen) < len(expected),
                        "materialized source contains untracked or unexpected bytes")
                seen.add(row.name)
                path = prefix + row.name
                wanted = expected[row.name]
                metadata = row.stat(follow_symlinks=False)
                if type(wanted) is dict or wanted[0] == "160000":
                    require(stat.S_ISDIR(metadata.st_mode), "source directory or submodule is not a real directory")
                    child = os.open(row.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                    dir_fd=fd)
                    try:
                        require(file_identity(metadata) == file_identity(os.fstat(child)), "source directory changed")
                        if type(wanted) is dict:
                            walk(child, wanted, path + "/", depth + 1)
                        else:
                            actual[path] = hashlib.sha256(b"gitlink\0" + wanted[2].encode()).digest()
                    finally:
                        os.close(child)
                elif wanted[0] == "120000":
                    require(stat.S_ISLNK(metadata.st_mode), "tracked source symlink type changed")
                    data = os.fsencode(os.readlink(row.name, dir_fd=fd))
                    require(len(data) <= 4096, "source symlink exceeds its bound")
                    actual[path] = hashlib.sha256(data).digest()
                else:
                    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1 and
                            bool(metadata.st_mode & 0o111) == (wanted[0] == "100755") and
                            0 <= metadata.st_size <= SOURCE.MAX_BLOB, "source file type, mode or size changed")
                    total += metadata.st_size
                    require(total <= SOURCE.MAX_TREE_BYTES, "materialized source exceeds its byte bound")
                    child = os.open(row.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                    try:
                        require(file_identity(metadata) == file_identity(os.fstat(child)), "source file identity changed")
                        digest, count = hashlib.sha256(), 0
                        while True:
                            data = os.read(child, min(65536, metadata.st_size + 1 - count))
                            if not data:
                                break
                            count += len(data)
                            require(count <= metadata.st_size, "source file grew beyond its bound")
                            digest.update(data)
                        require(count == metadata.st_size and file_identity(metadata) == file_identity(os.fstat(child)),
                                "source file changed while hashing")
                        actual[path] = digest.digest()
                    finally:
                        os.close(child)
                require(file_identity(metadata) == file_identity(os.stat(row.name, dir_fd=fd, follow_symlinks=False)),
                        "source member changed while checking")
        require(seen == set(expected) and file_identity(before) == file_identity(os.fstat(fd)),
                "materialized source paths or directory changed")

    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        walk(fd, tree)
    finally:
        os.close(fd)
    digest = hashlib.sha256(b"xybrid-zzz-source-tree-v1\0")
    for path, (mode, kind, _oid) in sorted(entries.items()):
        encoded = path.encode()
        digest.update(len(encoded).to_bytes(4, "big") + encoded)
        digest.update(mode.encode() + b"\0" + kind.encode() + b"\0" + actual[path])
    return digest.hexdigest()


def verify_materialized_source(root, object_repo, candidate, submodule_repos=None):
    credential_free()
    PRODUCER.require_posix_host()
    git = SOURCE.Git(object_repo)
    entries = git.entries(candidate)
    links = {path: value for path, value in entries.items() if value[0] == "160000"}
    require(set(links) == set(SUBMODULES), "unexpected or missing reviewed submodule paths")
    require(source_tree_digest(root, entries) == git.tree_digest(entries), "materialized candidate bytes differ")
    result = {}
    for path, entry in sorted(links.items()):
        submodule = Path(root) / path
        sub_git = SOURCE.Git(submodule_repos[path] if submodule_repos else submodule)
        require(sub_git.read("rev-parse", "--verify", "HEAD", limit=64).decode().strip() == entry[2],
                "submodule HEAD differs from the exact gitlink")
        sub_entries = sub_git.entries(entry[2])
        require(all(value[0] != "160000" for value in sub_entries.values()), "nested submodules require review")
        digest = sub_git.tree_digest(sub_entries)
        require(source_tree_digest(submodule, sub_entries) == digest, "materialized submodule bytes differ")
        result[path] = {"commit": entry[2], "tree_sha256": digest}
    return result


def materialize_tree(repo, commit, destination):
    """Write raw Git blobs, never checkout filters, hooks or candidate helpers."""
    git = SOURCE.Git(repo)
    entries = git.entries(commit)
    destination.mkdir(mode=0o700)  # Exclusive new source root.
    total = 0
    with subprocess.Popen(git.command + ["cat-file", "--batch"], env=git.environment,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        try:
            for path, (mode, _kind, oid) in sorted(entries.items()):
                target = destination / path
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if mode == "160000":
                    target.mkdir(mode=0o700)
                    continue
                process.stdin.write(oid.encode() + b"\n")
                process.stdin.flush()
                header = process.stdout.readline(128).split()
                require(len(header) == 3 and header[0] == oid.encode() and header[1] == b"blob" and
                        header[2].isdigit(), "invalid materialization Git blob response")
                size = int(header[2])
                total += size
                require(size <= SOURCE.MAX_BLOB and total <= SOURCE.MAX_TREE_BYTES,
                        "materialized source exceeds its byte bound")
                if mode == "120000":
                    require(size <= 4096, "source symlink exceeds its byte bound")
                    contents = process.stdout.read(size)
                    link = os.fsdecode(contents)
                    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), link))
                    require(len(contents) == size and not posixpath.isabs(link) and "\0" not in link and
                            resolved not in ("..", ".") and not resolved.startswith("../"),
                            "tracked source symlink escapes its fresh source tree")
                    target.symlink_to(link)
                else:
                    remaining = size
                    with target.open("xb") as output:
                        while remaining:
                            block = process.stdout.read(min(remaining, 65536))
                            require(block, "truncated materialization Git blob")
                            output.write(block)
                            remaining -= len(block)
                        os.fchmod(output.fileno(), 0o700 if mode == "100755" else 0o600)
                require(process.stdout.read(1) == b"\n", "invalid materialization Git blob terminator")
            process.stdin.close()
            require(process.wait() == 0, "materialization Git blob reader failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    metadata = git.read("rev-parse", "--absolute-git-dir", limit=4096).decode().strip()
    require("\n" not in metadata and Path(metadata).is_absolute(), "invalid private Git metadata path")
    (destination / ".git").write_text("gitdir: " + metadata + "\n")
    (destination / ".git").chmod(0o600)


def materialize_candidate(repo, candidate, destination, submodule_repos):
    credential_free()
    materialize_tree(repo, candidate, destination)
    entries = SOURCE.Git(repo).entries(candidate)
    links = {path: entry for path, entry in entries.items() if entry[0] == "160000"}
    require(set(links) == set(SUBMODULES) == set(submodule_repos), "exact reviewed submodule stores required")
    for path, entry in links.items():
        (destination / path).rmdir()  # Only the empty directory made above.
        materialize_tree(submodule_repos[path], entry[2], destination / path)
    return verify_materialized_source(destination, repo, candidate, submodule_repos)


def apple_build_plan(private_root, staging_root):
    """Unexecuted command specification; callers must create fresh private roots."""
    require(private_root.is_absolute() and staging_root.is_absolute() and private_root != staging_root,
            "private absolute isolation roots are required")
    command = ["bazelisk", "--nosystem_rc", "--nohome_rc", "--noworkspace_rc", "--bazelrc=" + os.devnull,
               "--output_user_root=" + str(private_root / "bazel-user"),
               "--output_base=" + str(private_root / "bazel-output"), "build",
               "--@rules_cc//cc/toolchains/args/archiver_flags:use_libtool_on_macos=False",
               "--@rules_rust//rust/settings:experimental_use_allocator_libraries_with_mangled_symbols",
               "--@rules_python//python/config_settings:bootstrap_impl=script",
               "--repo_env=BAZEL_MSVC_RUNTIME_VISUAL_STUDIO_EULA=1",
               "--repo_env=XYBRID_NDK_URL=https://dl.google.com/android/repository/android-ndk-r27-linux.zip",
               "--repo_env=XYBRID_NDK_SHA256=2f17eb8bcbfdc40201c0b36e9a70826fcd2524ab7a2a235e2c71186c302da1dc",
               "--repo_env=XYBRID_NDK_HOST_TAG=linux-x86_64",
               "--platforms=@rules_rs//rs/platforms:aarch64-apple-ios", "--compilation_mode=opt",
               "--ios_minimum_os=16.0", "--action_env=IPHONEOS_DEPLOYMENT_TARGET=16.0",
               "--@rules_rust//rust/settings:lto=fat", "--//:tts_zzz=true",
               "--remote_executor=", "--remote_cache=", "--bes_backend=", "--disk_cache=",
               "--noremote_accept_cached", "--noremote_upload_local_results",
               "--experimental_convenience_symlinks=ignore", "--lockfile_mode=off",
               "--repository_cache=" + str(private_root / "repository-cache"),
               "--repo_env=XYBRID_ZZZ_PREBUILT_DIR=" + str(staging_root),
               "--repo_env=XYBRID_ZZZ_PYTHON=" + sys.executable,
               "--extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain",
               "--extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_make_toolchain",
               "--extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_pkgconfig_toolchain",
               APPLE_TARGET]
    environment = {"PATH": PATH, "LC_ALL": "C", "HOME": str(private_root / "home"),
                   "TMPDIR": str(private_root / "tmp"), "XDG_CACHE_HOME": str(private_root / "xdg-cache"),
                   "BAZELISK_HOME": str(private_root / "bazelisk"),
                   "BAZELISK_SKIP_WRAPPER": "1",
                   "CARGO_HOME": str(private_root / "cargo-home"),
                   "CARGO_TARGET_DIR": str(private_root / "cargo-target"), "CARGO_INCREMENTAL": "0",
                   "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                   "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1",
                   "DEVELOPER_DIR": "/Applications/Xcode_26.6.app/Contents/Developer"}
    return command, environment


def download_environment(private_root):
    environment = {"PATH": PATH, "LC_ALL": "C", "CI": "true", "HOME": str(private_root / "mirror-home"),
                   "TMPDIR": str(private_root / "tmp"), "GH_CONFIG_DIR": str(private_root / "mirror-home/gh")}
    for name in ("ZZZ_GITHUB_TOKEN", "ZZZ_RELEASE_REPOSITORY"):
        require(name in os.environ and os.environ[name], "dedicated mirror-read credentials are required")
        environment[name] = os.environ[name]
    return environment


def download_engine(private_root, environment):
    github_context(environment)
    pins = trusted_pins()  # Before even constructing a token-bearing child environment.
    allowed = {"ZZZ_GITHUB_TOKEN", "ZZZ_RELEASE_REPOSITORY"}
    require(not any(name in os.environ for name in FORBIDDEN_ENV if name not in allowed),
            "mirror step refuses unrelated credentials or cache configuration")
    child_environment = download_environment(private_root)
    for directory in ("mirror-home", "tmp"):
        (private_root / directory).mkdir(mode=0o700, exist_ok=True)
    for target in pins:
        command = [sys.executable, "-I", str(ROOT / "crates/zzz-sys/zzz_pull.py"),
                   "--target", target, "--cache-dir", str(private_root / "download-cache"),
                   "--dest", str(private_root / "download-stage")]
        run_checked(command, environment=child_environment, timeout=600)


def run_checked(command, *, environment, cwd=None, timeout=300, output_limit=None):
    # Output is private and bounded. Never print candidate/compiler/Git stderr.
    if output_limit is None:
        try:
            subprocess.run(command, env=environment, cwd=cwd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout, check=True)
            return b""
        except (OSError, subprocess.SubprocessError) as error:
            raise Error("trusted producer subprocess failed") from error
    with tempfile.TemporaryFile(dir=environment.get("TMPDIR")) as output:
        try:
            subprocess.run(command, env=environment, cwd=cwd, stdout=output, stderr=subprocess.DEVNULL,
                           timeout=timeout, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            raise Error("trusted producer subprocess failed") from error
        require(output.tell() <= output_limit, "producer subprocess output exceeds its bound")
        output.seek(0)
        return output.read(output_limit + 1)


def upload_members(directory, version):
    """Exact outer allowlist; nested SDK qualification is a separate check."""
    PRODUCER.require_posix_host()
    SOURCE.version(version)
    package = "packages/xybrid-apple-v" + version + ".xcframework.zip"
    expected = {package, "source-binding.json"}
    actual, total = {}, 0
    root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        exact_names(root_fd, {"packages", "source-binding.json"})
        packages_fd = os.open("packages", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                              dir_fd=root_fd)
        try:
            exact_names(packages_fd, {Path(package).name})
            for name in sorted(expected):
                parent_fd = root_fd if name == "source-binding.json" else packages_fd
                actual[name] = upload_file(parent_fd, Path(name).name, name == package)
                total += actual[name]["size"]
                require(total <= MAX_UPLOAD_BYTES, "final SDK upload exceeds its wrapper budget")
            exact_names(packages_fd, {Path(package).name})
        finally:
            os.close(packages_fd)
        exact_names(root_fd, {"packages", "source-binding.json"})
    finally:
        os.close(root_fd)
    return actual


def exact_names(fd, expected):
    seen = set()
    with os.scandir(fd) as rows:
        for row in rows:
            require(row.name in expected and len(seen) < len(expected),
                    "upload tree differs from the exact final SDK allowlist")
            seen.add(row.name)
    require(seen == expected, "upload tree is missing a final SDK member")


def upload_file(parent_fd, name, package):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    with os.fdopen(fd, "rb") as source:
        before = os.fstat(source.fileno())
        bound = MAX_UPLOAD_BYTES if package else MAX_BINDING
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= bound,
                "upload input must be a bounded single-link regular file")
        digest, count = hashlib.sha256(), 0
        prefix = b""
        while True:
            data = source.read(min(65536, before.st_size + 1 - count))
            if not data:
                break
            count += len(data)
            require(count <= before.st_size, "upload member grew beyond its bound")
            prefix = (prefix + data)[:4]
            digest.update(data)
        require(count == before.st_size and file_identity(before) == file_identity(os.fstat(source.fileno())) and
                file_identity(before) == file_identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)),
                "upload member changed while hashing")
        if package:
            require(prefix == b"PK\x03\x04", "outer SDK member is not a ZIP package container")
        return {"size": count, "sha256": digest.hexdigest(), "kind": "sdk-package" if package else "source-binding"}


def fetch_candidate(repo, candidate):
    SOURCE.sha(candidate)
    git = SOURCE.Git(repo)
    command = git.command + ["-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
                             "-c", "gc.auto=0", "-c", "maintenance.auto=false",
                             "-c", "credential.helper=", "-c", "http.extraheader=",
                             "fetch", "--no-tags", "--no-write-fetch-head", "--no-recurse-submodules",
                             "https://github.com/xybrid-ai/xybrid.git", candidate]
    try:
        subprocess.run(command, env=git.environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=180, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise Error("fixed-origin candidate object fetch failed") from error


def preflight(tooling, repo, candidate, version, release_date, environment, *, fetch=True):
    credential_free()
    PRODUCER.require_posix_host()
    context = github_context(environment)
    trusted_master = context["workflow_commit"]
    trusted = SOURCE.Git(tooling)
    require(trusted.read("rev-parse", "--verify", "HEAD", limit=64).decode().strip() == trusted_master,
            "trusted tooling checkout does not match runtime master")
    entries = trusted.entries(trusted_master)
    module = trusted.blob(entries["MODULE.bazel"]).decode()
    declarations = re.findall(r'^zzz_prebuilt_repository\(name = "([^"]+)", target = "([^"]+)"\)$', module, re.MULTILINE)
    require(len(re.findall(r'^zzz_prebuilt_repository\s*\(', module, re.MULTILINE)) == len(STAGING_REPOSITORIES) and
            len(declarations) == len(STAGING_REPOSITORIES) and dict(declarations) == STAGING_REPOSITORIES,
            "reviewed MODULE staging repositories differ from the fixed six-target set")
    manifest = tooling / SOURCE.MANIFEST
    require(manifest.read_bytes() == trusted.blob(entries[SOURCE.MANIFEST]),
            "trusted manifest checkout bytes changed")
    trusted_pins(manifest)  # Frozen/schema-4 pins fail before fetch or any mirror-token step.
    if fetch:
        fetch_candidate(repo, candidate)
    return SOURCE.source_binding(repo, trusted_master, candidate, version, release_date, context, checksum=None)


def source_stores(private_root):
    return {path: private_root / "git" / ("submodule-" + str(index))
            for index, path in enumerate(SUBMODULES)}


def initialize_repository(path):
    path.mkdir(mode=0o700)
    git = SOURCE.Git(path)
    git.read("init", "--quiet", limit=4096)
    for name, value in (("core.hooksPath", os.devnull), ("gc.auto", "0"), ("maintenance.auto", "false"),
                        ("credential.helper", ""), ("core.fsmonitor", "false")):
        git.read("config", "--local", name, value, limit=4096)
    return git


def prepare(tooling, repo, candidate, version, release_date, environment, private_root):
    binding = preflight(tooling, repo, candidate, version, release_date, environment)
    require(private_root.is_absolute(), "absolute private producer root required")
    private_root.mkdir(mode=0o700)
    (private_root / "git").mkdir(mode=0o700)
    stores = source_stores(private_root)
    links = {path: entry for path, entry in SOURCE.Git(repo).entries(candidate).items() if entry[0] == "160000"}
    require(set(links) == set(SUBMODULES), "only the exact two reviewed submodule origins are allowed")
    for path, store in stores.items():
        git = initialize_repository(store)
        fetch_objects(store, links[path][2], SUBMODULES[path])
        git.read("update-ref", "HEAD", links[path][2], limit=4096)
    SOURCE.Git(repo).read("update-ref", "HEAD", candidate, limit=4096)
    materialize_candidate(repo, candidate, private_root / "candidate", stores)
    (private_root / "source-binding.json").write_bytes(SOURCE.canonical(binding))
    (private_root / "source-binding.json").chmod(0o400)
    return binding


def fetch_objects(repo, commit, origin):
    require(origin in SUBMODULES.values(), "only fixed reviewed submodule origins are allowed")
    SOURCE.sha(commit)
    git = SOURCE.Git(repo)
    command = git.command + ["-c", "protocol.https.allow=always", "-c", "gc.auto=0", "-c", "maintenance.auto=false",
                             "-c", "credential.helper=", "-c", "http.extraheader=", "fetch", "--no-tags",
                             "--no-write-fetch-head", "--no-recurse-submodules", origin, commit]
    run_checked(command, environment=git.environment, timeout=180)


def private_directory(path):
    metadata = path.lstat()
    require(stat.S_ISDIR(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o700 and
            metadata.st_uid == os.geteuid(), "producer state must remain a private owned directory")


def check_source(tooling, repo, candidate, version, release_date, environment, private_root):
    binding = preflight(tooling, repo, candidate, version, release_date, environment, fetch=False)
    private_directory(private_root)
    require(SOURCE.canonical(PRODUCER.strict_json(read_binding(private_root / "source-binding.json"), MAX_BINDING)) ==
            SOURCE.canonical(binding), "prepared binding differs from freshly recomputed source facts")
    verify_materialized_source(private_root / "candidate", repo, candidate, source_stores(private_root))
    return binding


def read_binding(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        before = os.fstat(source.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= MAX_BINDING and
                stat.S_IMODE(before.st_mode) == 0o400, "binding must remain a bounded read-only single-link file")
        data = source.read(MAX_BINDING + 1)
        require(len(data) == before.st_size and file_identity(before) == file_identity(os.fstat(source.fileno())),
                "binding changed while reading")
        return data


def verify_staging(private_root):
    credential_free()
    inputs = private_root / "engine-inputs"
    private_directory(inputs)
    pins = trusted_pins()
    fd = os.open(inputs, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        exact_names(fd, set(STAGING_TARGETS))
    finally:
        os.close(fd)
    for target, pin in pins.items():
        PULL.verify_staged(inputs / target, pin)
    return inputs


def fresh_staging(private_root):
    credential_free()
    inputs = private_root / "engine-inputs"
    inputs.mkdir(mode=0o700)
    for target, pin in trusted_pins().items():
        downloaded = (private_root / "download-stage").joinpath(*pin.key)
        PULL.verify_staged(downloaded, pin)
        staged = PULL.stage_archive(downloaded / PULL.STAGED_ARCHIVE, pin, private_root / "restage")
        staged.rename(inputs / target)
    return verify_staging(private_root)


def final_package(private_root, version):
    SOURCE.version(version)
    return private_root / "upload/packages" / ("xybrid-apple-v" + version + ".xcframework.zip")


def verify_final_package(private_root, version):
    credential_free()
    _command, environment = apple_build_plan(private_root / "build", private_root / "engine-inputs")
    run_checked([sys.executable, "-I", str(ROOT / "bindings/apple/pack_zzz_notices.py"), "verify", "--archive",
                 str(final_package(private_root, version))], environment=environment, timeout=180)


def configure_sdk_mirror(command, build_root, mirror):
    if not mirror:
        return command
    require(type(mirror) is str and re.fullmatch(r"[A-Za-z0-9.-]+(?::[0-9]{1,5})?/[A-Za-z0-9._~%/-]+", mirror),
            "public SDK mirror must be a host/path without credentials or whitespace")
    config = build_root / "downloader.cfg"
    with config.open("x") as output:
        output.write(r"rewrite swcdn\.apple\.com/.*/CLTools_macOSNMOS_SDK\.pkg " + mirror + "\n")
    return command[:-1] + ["--downloader_config=" + str(config), command[-1]]


def ensure_host_tools(build_root, environment):
    require(shutil.which("bazelisk", path=PATH) is not None, "preinstalled Bazelisk is required")
    for program in ("cmake", "pkg-config", "xz"):
        if shutil.which(program, path=PATH) is None:
            brew = shutil.which("brew", path=PATH)
            require(brew is not None, "required Apple host tools and Homebrew are missing")
            installer_environment = dict(environment, HOMEBREW_NO_AUTO_UPDATE="1", HOMEBREW_NO_ANALYTICS="1",
                                         HOMEBREW_CACHE=str(build_root / "brew-cache"),
                                         HOMEBREW_LOGS=str(build_root / "brew-logs"))
            run_checked([brew, "install", program], environment=installer_environment, cwd=build_root, timeout=900)
            require(shutil.which(program, path=PATH) is not None, "required Apple host tool installation failed")


def build_candidate(tooling, repo, candidate, version, release_date, environment, private_root):
    require(sys.platform == "darwin", "Apple candidate build requires a real macOS Xcode host")
    binding = check_source(tooling, repo, candidate, version, release_date, environment, private_root)
    staging = fresh_staging(private_root)
    build_root = private_root / "build"
    build_root.mkdir(mode=0o700)
    for name in ("home", "tmp", "xdg-cache", "bazelisk", "cargo-home", "cargo-target", "repository-cache"):
        (build_root / name).mkdir(mode=0o700)
    command, child_environment = apple_build_plan(build_root, staging)
    command = configure_sdk_mirror(command, build_root, environment.get("MACOS_SDK_MIRROR", ""))
    ensure_host_tools(build_root, child_environment)  # Credential-free, after the mirror broker has exited.
    require(Path(child_environment["DEVELOPER_DIR"]).is_dir(), "reviewed Xcode installation is missing")
    verify_materialized_source(private_root / "candidate", repo, candidate, source_stores(private_root))
    verify_staging(private_root)
    run_checked(command, environment=child_environment, cwd=private_root / "candidate", timeout=3600)
    query = list(command)
    query[query.index("build")] = "cquery"
    query.insert(-1, "--output=files")
    output = run_checked(query, environment=child_environment, cwd=private_root / "candidate", timeout=300,
                         output_limit=4096).decode().strip()
    require("\n" not in output and output.startswith("bazel-out/") and
            all(part not in ("", ".", "..") for part in output.split("/")) and
            Path(output).name == APPLE_ARCHIVE, "Bazel did not identify the exact final notice-bearing SDK ZIP")
    info = command[:command.index("build")] + ["info", "--lockfile_mode=off", "execution_root"]
    execution_root = run_checked(info, environment=child_environment, cwd=private_root / "candidate", timeout=120,
                                 output_limit=4096).decode().strip()
    require("\n" not in execution_root and Path(execution_root).is_absolute(), "invalid private Bazel execution root")
    archive = (Path(execution_root) / output).resolve()
    require(archive.is_relative_to((build_root / "bazel-output").resolve()), "Bazel archive escaped private outputs")
    (private_root / "upload").mkdir(mode=0o700)
    (private_root / "upload/packages").mkdir(mode=0o700)
    with PULL.open_regular(archive) as source, final_package(private_root, version).open("xb") as target:
        before = os.fstat(source.fileno())
        require(0 < before.st_size <= MAX_UPLOAD_BYTES, "final package exceeds its upload bound")
        count = 0
        while True:
            block = source.read(min(65536, before.st_size + 1 - count))
            if not block:
                break
            count += len(block)
            require(count <= before.st_size, "Bazel output changed during final package copy")
            target.write(block)
        require(count == before.st_size and file_identity(before) == file_identity(os.fstat(source.fileno())),
                "Bazel output changed during final package copy")
        os.fchmod(target.fileno(), 0o400)
    (private_root / "upload/source-binding.json").write_bytes(SOURCE.canonical(binding))
    (private_root / "upload/source-binding.json").chmod(0o400)
    pre_upload(tooling, repo, candidate, version, release_date, environment, private_root)


def pre_upload(tooling, repo, candidate, version, release_date, environment, private_root):
    binding = check_source(tooling, repo, candidate, version, release_date, environment, private_root)
    verify_staging(private_root)
    verify_final_package(private_root, version)
    require(read_binding(private_root / "upload/source-binding.json") == SOURCE.canonical(binding),
            "final source binding differs from freshly recomputed source facts")
    members = upload_members(private_root / "upload", version)
    context = github_context(environment)  # Repeat immediately before the sole success-only workflow upload.
    return PRODUCER.artifact_name({"artifact_role": "apple", "run_id": context["run_id"]}), members


def main():
    parser = PRODUCER.PrivateArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "download", "check-source", "build", "pre-upload"))
    parser.add_argument("--tooling", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--release-date", required=True)
    parser.add_argument("--root", required=True, type=Path)
    try:
        arguments = parser.parse_args()
        common = (arguments.tooling, arguments.repo, arguments.candidate, arguments.version,
                  arguments.release_date, os.environ, arguments.root)
        if arguments.phase == "download":
            private_directory(arguments.root)
            download_engine(arguments.root, os.environ)
        elif arguments.phase == "prepare":
            prepare(*common)
        elif arguments.phase == "check-source":
            check_source(*common)
        elif arguments.phase == "build":
            build_candidate(*common)
        else:
            pre_upload(*common)
        return 0
    except Exception as error:
        message = str(error) if isinstance(error, (Error, SOURCE.PolicyError, PULL.StageError)) else "invalid or unavailable local input"
        sys.stderr.write("trusted producer failed: " + message + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
