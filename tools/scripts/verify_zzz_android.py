#!/usr/bin/env python3
"""Gate the Android SDK's arm64-v8a Phonon/Kitten TTS 2 speech payload.

The Kotlin AAR (Maven Central `ai.xybrid:xybrid-kotlin`, which React Native
also consumes) links the zzz speech engine statically into
arm64-v8a/libxybrid_bolt.so and nowhere else. armeabi-v7a and x86_64 ship
without speech. The engine's 14 notices travel inside the package at
`assets/xybrid/zzz/<path>`, byte for byte the schema-5 inventory pinned in
crates/zzz-sys/zzz_pull.py (SCHEMA5_NOTICE_FILES). This tool proves all of
that from the artifact itself, wherever it is built or repackaged.

Subcommands:

  aar <file.aar> (--speech | --no-speech) [--min-sdk 29] [--readelf R]
      A Bazel- or Gradle-built AAR. Checks: zip safety (no absolute or `..`
      paths, duplicates, links or unexpected top-level entries); the text
      AndroidManifest.xml declares minSdkVersion == --min-sdk; jni/<abi>/
      holds exactly the expected libraries for arm64-v8a, armeabi-v7a and
      x86_64 (the arm64 llama.cpp set comes from bazel/ggml/android.bzl);
      machine, NEEDED closure, 16 KB LOAD alignment, the generated JNI
      exports (verify_android_jni.py) and the `.note.android.ident` API level
      (verify_android_api_level.py: 29 for every arm64-v8a library we build,
      28 for armeabi-v7a and x86_64; the bundled libc++_shared.so and
      libonnxruntime.so keep their vendor's level). --speech: arm64-v8a
      carries the engine-linked signal below, the other ABIs are speech-free,
      and assets/xybrid/zzz/ holds exactly the 14 notices with their pinned
      SHA-256 and nothing else. --no-speech: no engine anywhere and no
      assets/xybrid/zzz/. Either way no private staging file (libzzz_embed.a,
      zzz_embed.h, RECEIPT.json, .pinned-archive.tar.gz, share/zzz_embed.json)
      may appear anywhere.

  pack-dist <file.aar> <out.zip>
      Repackage an AAR as the release zip (xybrid-android-<tag>.zip): its
      jni/<abi>/ directories at the zip root, which is exactly what
      bindings/kotlin/libs/ (the Gradle jniLibs srcDir) expects, plus
      assets/xybrid/zzz/. Run `aar` before and `dist` after.

  dist <file.zip> [--readelf R]
      The release zip: the `aar --speech` checks minus the manifest. Only the
      three ABI directories and assets/xybrid/zzz/ may appear.

  stage-gradle <file.zip> --kotlin-dir bindings/kotlin
      Put a release zip where Gradle packages it: <abi>/*.so into
      <kotlin-dir>/libs/<abi>/ and the notices into
      <kotlin-dir>/src/main/assets/xybrid/zzz/. Revalidates the layout and
      notice bytes first, never writes anywhere else, and refuses non-empty
      libs/<abi>/ directories, notice directories holding anything but the
      exact notices, and any other file under src/main/assets/.

  symbol-overlap --archive <libzzz_embed.a> [--readelf R] <shared libs...>
      Count the archive's GLOBAL/WEAK defined symbols that the given public
      shared libraries (libggml*.so, libllama.so, libmtmd.so,
      libc++_shared.so, libonnxruntime.so) also export. Prints only the count
      and the overlapping names, which are public by construction; fails if
      any overlap, since a static engine copy would interpose on them.

  compare-imports <tts-off.so> <tts-on.so> [--readelf R]
      Fail if an undefined ggml_/gguf_/llama_/mtmd_ import of the default
      arm64-v8a libxybrid_bolt.so is missing from the speech build: the
      engine would then be satisfying it with its own copy.

Engine-linked signal for arm64-v8a/libxybrid_bolt.so. A default (tts-off)
build has none of the first three, which the CI proof lane
(test-zzz-android.yml) recalibrates on every relevant change:
  - it contains the string "zzz engine unavailable", compiled only with the
    tts-zzz feature (crates/zzz-sys/src/session.rs and
    crates/xybrid-core/src/runtime_adapter/zzz/runtime.rs);
  - no zzz_embed_* symbol in .dynsym: an undefined one means the archive was
    not linked in, a defined one would export engine API;
  - a PT_TLS program header (the engine uses thread-local storage);
  - `.note.android.ident` API 29 (Android 10 has native ELF TLS).

Messages are fixed text: no readelf stderr, receipt or private path is
printed, and symbol names only where public libraries export them. Exit
status: 0 passed, 1 failed, 2 usage. Standard library only, Python 3.9+.
GNU readelf and llvm-readelf output both parse.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
import zipfile
import zlib
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

SCRIPTS = Path(os.path.abspath(__file__)).parent
ROOT = SCRIPTS.parents[1]


def _sibling(name: str, path: Path):
    # Loaded by file path, never through sys.path or the working directory.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The immutable schema-5 notice inventory: relative path -> SHA-256.
PULL = _sibling("xybrid_zzz_android_contract", ROOT / "crates/zzz-sys/zzz_pull.py")
API_LEVEL = _sibling("xybrid_zzz_android_api_level", SCRIPTS / "verify_android_api_level.py")
JNI = _sibling("xybrid_zzz_android_jni", SCRIPTS / "verify_android_jni.py")
# The readelf parsers (machine, NEEDED, LOAD alignment, symbols) the Unity
# Android gate already proves against both readelf flavours.
ELF = _sibling("xybrid_zzz_android_elf", SCRIPTS / "verify_unity_android_natives.py")

ABIS = ("arm64-v8a", "armeabi-v7a", "x86_64")
SPEECH_ABI = "arm64-v8a"
# The `.note.android.ident` level of every library we build, per ABI: only
# the AAR's arm64 slice links with @androidndk_api29 (//bazel/toolchains/android).
API_LEVELS = {"arm64-v8a": 29, "armeabi-v7a": 28, "x86_64": 28}
ALIGNED_ABIS = frozenset({"arm64-v8a", "x86_64"})  # 16 KB pages on Android 15+
BOLT = "libxybrid_bolt.so"
CXX = "libc++_shared.so"
ORT = "libonnxruntime.so"
# Prebuilt libraries we only bundle: they keep their vendor's API level and
# alignment, so only their machine and NEEDED closure are checked.
BUNDLED = frozenset({CXX, ORT})
ORT_ABIS = frozenset({"arm64-v8a", "x86_64"})
GGML_BZL = ROOT / "bazel/ggml/android.bzl"
GLUE = ROOT / "bindings/kotlin/bazel/jni/jni_glue.c"

NOTICE_ROOT = "assets/xybrid/zzz/"
MARKER = b"zzz engine unavailable"
ENGINE_PREFIX = "zzz_embed_"
# Staged engine inputs (zzz_pull.py's slice layout) and the retired Android
# companion library: none may reach a shipped package.
PRIVATE_NAMES = frozenset({
    "libzzz_embed.a", "zzz_embed.h", "RECEIPT.json", PULL.STAGED_ARCHIVE,
    "zzz_embed.json", "libxybrid_zzz_tts.so",
})
SHARED_PREFIXES = ("ggml_", "gguf_", "llama_", "mtmd_")
EXPORT_BINDS = frozenset({"GLOBAL", "WEAK", "UNIQUE", "GNU_UNIQUE"})

# The documented AAR anatomy (developer.android.com, "Anatomy of an AAR
# file") plus what AGP adds (META-INF/, annotations.zip, baseline-prof.txt).
AAR_FILES = frozenset({
    "AndroidManifest.xml", "classes.jar", "R.txt", "public.txt", "proguard.txt",
    "lint.jar", "api.jar", "annotations.zip", "baseline-prof.txt",
})
AAR_DIRS = frozenset({"jni", "assets", "res", "libs", "META-INF", "prefab"})
AAR_REQUIRED = ("AndroidManifest.xml", "classes.jar")
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"

MAX_PACKAGE = 1024 * 1024 * 1024
MAX_MEMBER = 512 * 1024 * 1024
MAX_TOTAL = 2 * 1024 * 1024 * 1024
MAX_MEMBERS = 4096
MAX_RATIO = 200
MAX_NOTICE = 1024 * 1024
MAX_MANIFEST = 1024 * 1024
MAX_NAME = 512
CHUNK = 1024 * 1024
_NAME = re.compile(r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*/?")


class GateError(ValueError):
    """A fixed message, safe for public logs."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise GateError(message)


def shown(name: str) -> str:
    return repr(name if len(name) <= 96 else name[:96] + "...")


def _bzl_strings(text: str, variable: str) -> List[str]:
    match = re.search(rf"^{variable} = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    require(match, f"{GGML_BZL.name}: {variable} not found")
    body = "\n".join(line.split("#", 1)[0] for line in match.group(1).splitlines())
    names = re.findall(r'"([^"]+)"', body)
    require(names, f"{GGML_BZL.name}: {variable} is empty")
    return names


def expected_natives(bzl: Path = GGML_BZL) -> Dict[str, FrozenSet[str]]:
    """The exact jni/<abi>/ inventory of the AAR (bindings/kotlin/BUILD.bazel)."""
    text = bzl.read_text()
    ggml = set(_bzl_strings(text, "ANDROID_ARM64_GGML_LIBS"))
    ggml |= set(_bzl_strings(text, "ANDROID_ARM64_CPU_VARIANTS"))
    natives = {}
    for abi in ABIS:
        names = {BOLT, CXX}
        if abi in ORT_ABIS:
            names.add(ORT)
        if abi == SPEECH_ABI:
            names |= ggml
        natives[abi] = frozenset(names)
    return natives


# --- readelf ---------------------------------------------------------------

Runner = Callable[[Sequence[str]], str]


def run_readelf(argv: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=600, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise GateError("could not run readelf") from None
    require(result.returncode == 0, "readelf failed")
    return result.stdout.decode("utf-8", "replace")


class Readelf:
    def __init__(self, program: str, runner: Optional[Runner] = None):
        self.program = program
        self.runner = runner or run_readelf

    def __call__(self, path: Path, *flags: str) -> str:
        return self.runner([self.program, "-W", *flags, str(path)])


def has_tls(program_headers: str) -> bool:
    """A PT_TLS row in `readelf -l` text (only program-header rows start with TLS)."""
    return any(line.split()[:1] == ["TLS"] for line in program_headers.splitlines())


def defined_globals(symbol_text: str) -> Set[str]:
    return {
        symbol.name for symbol in ELF.parse_symbols(symbol_text)
        if symbol.defined and symbol.bind in EXPORT_BINDS
        and symbol.kind not in ("SECTION", "FILE")
    }


def dynamic_exports(symbol_text: str) -> Set[str]:
    return {
        symbol.name for symbol in ELF.parse_symbols(symbol_text)
        if symbol.defined and symbol.bind in EXPORT_BINDS
        and symbol.visibility in ("DEFAULT", "PROTECTED")
    }


def shared_imports(symbol_text: str) -> Set[str]:
    return {
        symbol.name for symbol in ELF.parse_symbols(symbol_text)
        if not symbol.defined and symbol.name.startswith(SHARED_PREFIXES)
    }


def contains(path: Path, needle: bytes) -> bool:
    tail = b""
    with open(path, "rb") as stream:
        while True:
            block = stream.read(CHUNK)
            if not block:
                return False
            window = tail + block
            if needle in window:
                return True
            tail = window[-(len(needle) - 1):]


# --- zip packages ----------------------------------------------------------


class Package:
    """A bounded, safely indexed zip (AAR or release zip), opened read-only."""

    def __init__(self, path: Path, label: Optional[str] = None):
        self.label = label or Path(path).name
        try:
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        except OSError:
            raise GateError(f"{self.label}: cannot open (missing, a link, or unreadable)") from None
        self._stream = os.fdopen(descriptor, "rb")
        try:
            metadata = os.fstat(descriptor)
            require(stat.S_ISREG(metadata.st_mode) and 0 < metadata.st_size <= MAX_PACKAGE,
                    f"{self.label}: not a bounded regular file")
            try:
                self._zip = zipfile.ZipFile(self._stream)
            except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError):
                raise GateError(f"{self.label}: not a readable zip") from None
            self.files, self.directories = self._index()
        except BaseException:
            self._stream.close()
            raise

    def __enter__(self) -> "Package":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        zip_file = getattr(self, "_zip", None)
        if zip_file is not None:
            zip_file.close()
        self._stream.close()

    def _index(self) -> Tuple[Dict[str, zipfile.ZipInfo], Set[str]]:
        label = self.label
        infos = self._zip.infolist()
        require(0 < len(infos) <= MAX_MEMBERS, f"{label}: empty, or more than {MAX_MEMBERS} entries")
        files: Dict[str, zipfile.ZipInfo] = {}
        directories: Set[str] = set()
        seen: Set[str] = set()
        total = 0
        for info in infos:
            name = info.filename
            key = name.rstrip("/")
            require(name == info.orig_filename and len(name) <= MAX_NAME
                    and _NAME.fullmatch(name) is not None
                    and all(part not in (".", "..") for part in key.split("/")),
                    f"{label}: unsafe entry name {shown(name)} (absolute, '..', or unexpected characters)")
            # Case-folded: a case-insensitive filesystem would merge the two.
            require(key.lower() not in seen, f"{label}: duplicate entry {shown(name)}")
            seen.add(key.lower())
            require(not info.flag_bits & 0x1, f"{label}: encrypted entry {shown(name)}")
            require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    f"{label}: unsupported compression for {shown(name)}")
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            require(not mode & 0o7000, f"{label}: setuid, setgid or sticky entry {shown(name)}")
            if info.is_dir():
                require(kind in (0, stat.S_IFDIR) and info.file_size == 0,
                        f"{label}: {shown(name)} is not a plain directory entry")
                directories.add(key)
                continue
            require(kind in (0, stat.S_IFREG), f"{label}: {shown(name)} is a link or special file")
            require(info.file_size <= MAX_MEMBER
                    and info.file_size <= MAX_RATIO * max(1, info.compress_size),
                    f"{label}: {shown(name)} exceeds its size or compression bound")
            total += info.file_size
            require(total <= MAX_TOTAL, f"{label}: total uncompressed size exceeds its bound")
            files[name] = info
        parents = directories | {name[:index] for name in files for index, char in enumerate(name) if char == "/"}
        for name in files:
            require(name not in parents, f"{label}: {shown(name)} is both a file and a directory")
        return files, directories

    def copy(self, info: zipfile.ZipInfo, target=None) -> str:
        """Stream one entry (into `target` if given); returns its SHA-256."""
        digest, count = hashlib.sha256(), 0
        try:
            with self._zip.open(info) as source:
                while True:
                    block = source.read(min(CHUNK, info.file_size + 1 - count))
                    if not block:
                        break
                    count += len(block)
                    require(count <= info.file_size, f"{self.label}: {shown(info.filename)} grew beyond its size")
                    digest.update(block)
                    if target is not None:
                        target.write(block)
        except GateError:
            raise
        except (zipfile.BadZipFile, zlib.error, EOFError, RuntimeError, ValueError):
            # CRC, inflate and header errors; a write failure stays an OSError.
            raise GateError(f"{self.label}: {shown(info.filename)} is corrupt") from None
        require(count == info.file_size, f"{self.label}: {shown(info.filename)} is truncated")
        return digest.hexdigest()

    def read(self, info: zipfile.ZipInfo, limit: int) -> bytes:
        require(info.file_size <= limit, f"{self.label}: {shown(info.filename)} is too large")
        buffer = io.BytesIO()
        self.copy(info, buffer)
        return buffer.getvalue()

    def extract(self, info: zipfile.ZipInfo, destination: Path) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(destination, flags, 0o600), "wb") as target:
            self.copy(info, target)


# --- checks ----------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.summaries: List[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def ok(self, message: str) -> None:
        self.summaries.append(message)

    def emit(self, subject: str) -> int:
        for summary in self.summaries:
            print(summary)
        if self.failures:
            print(f"\n{len(self.failures)} problem(s) in {subject}:", file=sys.stderr)
            for failure in self.failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1
        print(f"{subject}: OK")
        return 0


def check_private_names(package: Package, report: Report) -> None:
    for name in sorted(set(package.files) | package.directories):
        if name.rstrip("/").rsplit("/", 1)[-1] in PRIVATE_NAMES:
            report.fail(f"{shown(name)}: private engine staging file in a shipped package")


def classify(package: Package, layout: str, report: Report) -> Dict[str, Dict[str, zipfile.ZipInfo]]:
    """Sort entries into natives per ABI; report anything the layout does not allow.

    assets/ belongs to check_notices. In an AAR the natives sit under jni/ and
    the other documented AAR entries are allowed; the release zip holds only
    the ABI directories (and assets/).
    """
    natives: Dict[str, Dict[str, zipfile.ZipInfo]] = {abi: {} for abi in ABIS}
    for name in sorted(package.files):
        if name.startswith("assets/"):
            continue
        relative = name
        if layout == "aar":
            top, slash, relative = name.partition("/")
            if not slash:
                if name not in AAR_FILES:
                    report.fail(f"{shown(name)}: unexpected top-level AAR entry")
                continue
            if top != "jni":
                if top not in AAR_DIRS:
                    report.fail(f"{shown(name)}: unexpected top-level AAR directory")
                continue
        parts = relative.split("/")
        if len(parts) == 2 and parts[0] in ABIS and parts[1].endswith(".so"):
            natives[parts[0]][parts[1]] = package.files[name]
        elif layout == "dist" and parts[0] not in ABIS:
            report.fail(f"{shown(name)}: unexpected entry (the release zip holds only "
                        f"{', '.join(ABIS)} and {NOTICE_ROOT})")
        else:
            report.fail(f"{shown(name)}: unexpected native payload (only <abi>/<lib>.so "
                        f"for {', '.join(ABIS)})")
    for directory in sorted(package.directories):
        parts = directory.split("/")
        if parts[0] == "assets":
            continue
        if layout == "dist":
            native = parts
        elif parts[0] not in AAR_DIRS:
            report.fail(f"{shown(directory + '/')}: unexpected top-level AAR directory")
            continue
        elif parts[0] == "jni":
            native = parts[1:]
        else:
            continue
        if len(native) > 1 or (native and native[0] not in ABIS):
            report.fail(f"{shown(directory + '/')}: unexpected native directory")
    expected = expected_natives()
    for abi in ABIS:
        have = set(natives[abi])
        for name in sorted(expected[abi] - have):
            report.fail(f"{abi}/{name}: missing")
        for name in sorted(have - expected[abi]):
            report.fail(f"{abi}/{name}: unexpected native library")
    return natives


def check_notices(package: Package, speech: bool, report: Report) -> None:
    notices = PULL.SCHEMA5_NOTICE_FILES
    found = {}
    for name in sorted(package.files):
        if name.startswith(NOTICE_ROOT):
            found[name[len(NOTICE_ROOT):]] = package.files[name]
        elif name.startswith("assets/"):
            report.fail(f"{shown(name)}: unexpected asset (only the engine notices "
                        f"ship under {NOTICE_ROOT})")
    notice_dirs = {"assets", "assets/xybrid", NOTICE_ROOT.rstrip("/")} | {
        (NOTICE_ROOT + name).rsplit("/", 1)[0] for name in notices}
    for directory in sorted(package.directories):
        if directory == "assets" or directory.startswith("assets/"):
            if not speech or directory not in notice_dirs:
                report.fail(f"{shown(directory + '/')}: unexpected asset directory")
    if not speech:
        if found:
            report.fail(f"{NOTICE_ROOT}: {len(found)} engine notice file(s) in a package "
                        "built without speech")
        return
    good = 0
    for relative, digest in sorted(notices.items()):
        info = found.get(relative)
        if info is None:
            report.fail(f"{NOTICE_ROOT}{relative}: missing engine notice")
        elif info.file_size > MAX_NOTICE or package.copy(info) != digest:
            report.fail(f"{NOTICE_ROOT}{relative}: bytes differ from the schema-5 contract")
        else:
            good += 1
    for relative in sorted(set(found) - set(notices)):
        report.fail(f"{shown(NOTICE_ROOT + relative)}: unexpected file among the engine notices")
    if good == len(notices) and set(found) == set(notices):
        report.ok(f"{NOTICE_ROOT}: all {good} engine notices match the schema-5 contract")


def check_manifest(package: Package, min_sdk: int, report: Report) -> None:
    for name in AAR_REQUIRED:
        if name not in package.files:
            report.fail(f"{name}: missing from the AAR")
    info = package.files.get("AndroidManifest.xml")
    if info is None:
        return
    data = package.read(info, MAX_MANIFEST)
    if data[:4] == b"\x03\x00\x08\x00":
        report.fail("AndroidManifest.xml: binary XML; an AAR carries a text manifest")
        return
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        report.fail("AndroidManifest.xml: declares a DTD or entities")
        return
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError:
        report.fail("AndroidManifest.xml: not well-formed XML")
        return
    uses_sdk = root.findall("uses-sdk") if root.tag == "manifest" else []
    if root.tag != "manifest" or len(uses_sdk) != 1:
        report.fail("AndroidManifest.xml: expected one <manifest> with exactly one <uses-sdk>")
        return
    value = uses_sdk[0].get(ANDROID_NS + "minSdkVersion")
    if value != str(min_sdk):
        report.fail(f"AndroidManifest.xml: minSdkVersion is {shown(value or '')}, expected {min_sdk}")
    else:
        report.ok(f"AndroidManifest.xml: minSdkVersion {min_sdk}")


def check_library(abi: str, name: str, path: Path, shipped: FrozenSet[str], speech: bool,
                  readelf: Readelf, glue: str, report: Report) -> None:
    """ELF checks for one shipped library; `speech` means it must carry the engine."""
    where = f"{abi}/{name}"
    before = len(report.failures)
    try:
        headers = readelf(path, "-h", "-l", "-d")
        notes = "" if name in BUNDLED else readelf(path, "-n")
        dynsyms = readelf(path, "--dyn-syms") if name == BOLT else ""
    except GateError as error:
        report.fail(f"{where}: {error}")
        return
    machine = ELF.parse_machine(headers)
    if machine not in ELF.ABI_MACHINES[abi]:
        report.fail(f"{where}: ELF machine is {shown(machine or 'unknown')}, "
                    f"expected {ELF.ABI_MACHINES[abi][0]}")
    needed = ELF.parse_needed(headers)
    for dependency in needed:
        if dependency not in ELF.SYSTEM_LIBS and dependency not in shipped:
            report.fail(f"{where}: NEEDs {shown(dependency)}, which is neither an Android "
                        f"system library nor shipped in {abi}/")
    if name in BUNDLED:
        return
    if abi in ALIGNED_ABIS:
        try:
            aligns = ELF.parse_load_aligns(headers)
        except ValueError:
            aligns = []
        if not aligns or min(aligns) < ELF.MIN_LOAD_ALIGN:
            report.fail(f"{where}: a LOAD segment is aligned below 16 KB (or none was found)")
    levels = set(API_LEVEL.parse_api_levels(notes))
    if levels != {API_LEVELS[abi]}:
        found = ", ".join(str(level) for level in sorted(levels)) or "none"
        report.fail(f"{where}: linked for Android API {found}, expected {API_LEVELS[abi]}")
    if name != BOLT:
        return
    if CXX not in needed:
        report.fail(f"{where}: does not NEED {CXX}")
    try:
        JNI.verify(glue, dynsyms)
    except ValueError as error:
        # The names are the generated, public Java_* entry points.
        report.fail(f"{where}: " + str(error).replace("\n", "; "))
    engine = [s for s in ELF.parse_symbols(dynsyms) if s.name.startswith(ENGINE_PREFIX)]
    undefined = sum(1 for symbol in engine if not symbol.defined)
    defined = len(engine) - undefined
    marker = contains(path, MARKER)
    tls = has_tls(headers)
    problems = []
    if speech:
        if not marker:
            problems.append("lacks the speech runtime's marker string")
        if undefined:
            problems.append(f"imports {undefined} engine symbol(s), so the archive was not linked in")
        if defined:
            problems.append(f"exports {defined} engine symbol(s); the engine must stay internal")
        if not tls:
            problems.append("has no PT_TLS segment")
        if problems:
            report.fail(f"{where}: the speech engine is not linked as required: " + "; ".join(problems))
    else:
        if marker:
            problems.append("carries the speech runtime's marker string")
        if engine:
            problems.append(f"references {len(engine)} engine symbol(s)")
        if tls:
            problems.append("has a PT_TLS segment")
        if problems:
            report.fail(f"{where}: must ship without speech, but it " + "; ".join(problems))
    if len(report.failures) == before:
        state = ("speech engine linked statically (marker, PT_TLS, no engine symbols in .dynsym)"
                 if speech else "speech-free")
        report.ok(f"{where}: {state}; JNI exports, NEEDED, Android API {API_LEVELS[abi]} OK")


def check_natives(package: Package, natives: Dict[str, Dict[str, zipfile.ZipInfo]], speech: bool,
                  readelf: Readelf, glue: str, report: Report) -> None:
    scratch = Path(tempfile.mkdtemp(prefix="xybrid-android-gate-"))
    try:
        for abi in ABIS:
            directory = scratch / abi
            directory.mkdir(mode=0o700)
            shipped = frozenset(natives[abi])
            for name in sorted(natives[abi]):
                path = directory / name
                package.extract(natives[abi][name], path)
                check_library(abi, name, path, shipped, speech and abi == SPEECH_ABI,
                              readelf, glue, report)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def verify_package(path: Path, layout: str, speech: bool, readelf: Readelf,
                   min_sdk: Optional[int] = None, glue_path: Path = GLUE) -> Report:
    report = Report()
    if min_sdk is not None and min_sdk < API_LEVELS[SPEECH_ABI]:
        report.fail(f"--min-sdk {min_sdk} is below the arm64-v8a libraries' Android API "
                    f"{API_LEVELS[SPEECH_ABI]}")
    glue = glue_path.read_text()
    with Package(path) as package:
        check_private_names(package, report)
        if layout == "aar":
            check_manifest(package, min_sdk if min_sdk is not None else API_LEVELS[SPEECH_ABI], report)
        natives = classify(package, layout, report)
        check_notices(package, speech, report)
        check_natives(package, natives, speech, readelf, glue, report)
    return report


# --- repackaging -----------------------------------------------------------


def _output_info(name: str, original: zipfile.ZipInfo) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, original.date_time)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def pack_dist(aar: Path, output: Path) -> int:
    """Copy jni/<abi>/* to <abi>/* and assets/xybrid/zzz/** as-is; nothing else."""
    with Package(aar) as package:
        entries = []
        for name in sorted(package.files):
            if name.startswith("jni/"):
                entries.append((name[len("jni/"):], package.files[name]))
            elif name.startswith(NOTICE_ROOT):
                entries.append((name, package.files[name]))
        require(entries, f"{aar.name}: no jni/ or {NOTICE_ROOT} entries to package")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(output, flags, 0o644)
        except OSError:
            raise GateError(f"{output.name}: cannot create (it must not exist yet)") from None
        try:
            with os.fdopen(descriptor, "wb") as stream, \
                    zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as result:
                for target, info in entries:
                    with result.open(_output_info(target, info), "w") as entry:
                        package.copy(info, entry)
        except BaseException:
            output.unlink()
            raise
    print(f"{output.name}: packed {len(entries)} file(s) from {aar.name}")
    return 0


def _plain_directory(path: Path, label: str) -> bool:
    """True if `path` is an existing real directory; False if absent."""
    if not os.path.lexists(path):
        return False
    require(not path.is_symlink() and path.is_dir(), f"{label}: not a plain directory")
    return True


def _existing_notices(directory: Path) -> Set[str]:
    """Verify an existing notice directory holds only exact notices; return them."""
    present = set()
    allowed_dirs = {(name.rsplit("/", 1)[0] if "/" in name else "") for name in PULL.SCHEMA5_NOTICE_FILES}
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        relative_dir = Path(current).relative_to(directory).as_posix()
        relative_dir = "" if relative_dir == "." else relative_dir
        for dirname in dirnames:
            child = (relative_dir + "/" + dirname).lstrip("/")
            require(not (Path(current) / dirname).is_symlink() and child in allowed_dirs,
                    f"{NOTICE_ROOT}: existing directory holds unexpected content")
        for filename in filenames:
            relative = (relative_dir + "/" + filename).lstrip("/")
            path = Path(current) / filename
            require(relative in PULL.SCHEMA5_NOTICE_FILES and not path.is_symlink() and path.is_file(),
                    f"{NOTICE_ROOT}: existing directory holds unexpected content")
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                digest.update(stream.read(MAX_NOTICE + 1))
            require(digest.hexdigest() == PULL.SCHEMA5_NOTICE_FILES[relative],
                    f"{NOTICE_ROOT}: existing directory holds unexpected content")
            present.add(relative)
    return present


def _create(path: Path, root: Path) -> int:
    """Create a new file under `root` only: no overwrite, no link followed."""
    require(path.parent.resolve().is_relative_to(root.resolve()),
            "refusing to write outside the staging roots")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, 0o644)


def stage_gradle(release_zip: Path, kotlin_dir: Path) -> int:
    require(_plain_directory(kotlin_dir, "--kotlin-dir")
            and (kotlin_dir / "build.gradle.kts").is_file(),
            "--kotlin-dir must be the Kotlin binding (bindings/kotlin, with build.gradle.kts)")
    libs = kotlin_dir / "libs"
    assets = kotlin_dir / "src" / "main" / "assets"
    notices = assets / "xybrid" / "zzz"
    for part in (kotlin_dir / "src", kotlin_dir / "src" / "main"):
        require(_plain_directory(part, part.name), f"{part.name}: missing Kotlin source directory")
    libs_exist = _plain_directory(libs, "libs")

    report = Report()
    with Package(release_zip) as package:
        check_private_names(package, report)
        natives = classify(package, "dist", report)
        check_notices(package, True, report)
        if report.failures:
            report.summaries.clear()
            return report.emit(release_zip.name)

        for abi in ABIS:
            directory = libs / abi
            if libs_exist and _plain_directory(directory, f"libs/{abi}"):
                require(not any(directory.iterdir()),
                        f"libs/{abi}: already holds files; stage into a clean checkout")
        existing: Set[str] = set()
        if _plain_directory(assets, "src/main/assets"):
            for current, dirnames, filenames in os.walk(assets, followlinks=False):
                relative = Path(current).relative_to(assets).as_posix()
                for entry in dirnames + filenames:
                    child = entry if relative == "." else f"{relative}/{entry}"
                    require(child in ("xybrid", "xybrid/zzz") or child.startswith("xybrid/zzz/"),
                            "src/main/assets: holds other assets, which would ship in the AAR")
                    require(not (Path(current) / entry).is_symlink(), "src/main/assets: holds a link")
            if _plain_directory(assets / "xybrid", "src/main/assets/xybrid") \
                    and _plain_directory(notices, NOTICE_ROOT):
                existing = _existing_notices(notices)

        written = 0
        if not libs_exist:
            libs.mkdir()
        for abi in ABIS:
            directory = libs / abi
            if not _plain_directory(directory, f"libs/{abi}"):
                directory.mkdir()
            for name, info in sorted(natives[abi].items()):
                with os.fdopen(_create(directory / name, libs), "wb") as target:
                    package.copy(info, target)
                written += 1
        for relative in sorted(PULL.SCHEMA5_NOTICE_FILES):
            if relative in existing:
                continue
            chain = [assets, assets / "xybrid", notices]
            for part in relative.split("/")[:-1]:
                chain.append(chain[-1] / part)
            for directory in chain:
                if not _plain_directory(directory, f"src/main/{NOTICE_ROOT}"):
                    directory.mkdir()
            with os.fdopen(_create(notices / relative, notices), "wb") as target:
                package.copy(package.files[NOTICE_ROOT + relative], target)
            written += 1
    print(f"{release_zip.name}: staged {written} file(s) into libs/<abi>/ and "
          f"src/main/{NOTICE_ROOT}")
    return 0


# --- symbol comparisons ----------------------------------------------------


def symbol_overlap(archive: Path, libraries: Sequence[Path], readelf: Readelf) -> int:
    try:
        engine = defined_globals(readelf(archive, "-s"))
    except GateError:
        # Never echo the archive's path or readelf's stderr.
        print("symbol-overlap: readelf could not read the engine archive", file=sys.stderr)
        return 1
    if not engine:
        print("symbol-overlap: no defined global symbols in the engine archive; wrong file?",
              file=sys.stderr)
        return 1
    overlap: Dict[str, List[str]] = {}
    for library in libraries:
        try:
            exports = dynamic_exports(readelf(library, "--dyn-syms"))
        except GateError:
            print(f"symbol-overlap: readelf could not read {library.name}", file=sys.stderr)
            return 1
        for name in engine & exports:
            overlap.setdefault(name, []).append(library.name)
    print(f"symbol-overlap: {len(overlap)} engine symbol(s) also exported by "
          f"{len(libraries)} public librar{'y' if len(libraries) == 1 else 'ies'}")
    for name in sorted(overlap):
        # Public by construction: these libraries export the name.
        print(f"  {name} (exported by {', '.join(sorted(overlap[name]))})")
    return 1 if overlap else 0


def compare_imports(default: Path, speech: Path, readelf: Readelf) -> int:
    try:
        before = shared_imports(readelf(default, "--dyn-syms"))
        after = shared_imports(readelf(speech, "--dyn-syms"))
    except GateError as error:
        print(f"compare-imports: {error}", file=sys.stderr)
        return 1
    if not before:
        print(f"compare-imports: {default.name} imports no {'/'.join(SHARED_PREFIXES)} symbols; "
              "expected the default arm64-v8a libxybrid_bolt.so", file=sys.stderr)
        return 1
    missing = sorted(before - after)
    if missing:
        print(f"compare-imports: {len(missing)} of {len(before)} shared llama.cpp import(s) of "
              f"{default.name} are no longer imported by the speech build, so a static copy "
              "now satisfies them:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1
    print(f"compare-imports: all {len(before)} shared llama.cpp imports are still imported "
          f"({len(after - before)} new)")
    return 0


# --- command line ----------------------------------------------------------


class Parser(argparse.ArgumentParser):
    def error(self, _message):  # argparse echoes values, which may be private paths
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: invalid arguments\n")


def main(argv: Optional[Sequence[str]] = None, runner: Optional[Runner] = None,
         glue_path: Path = GLUE) -> int:
    parser = Parser(prog="verify_zzz_android", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)

    def with_readelf(sub):
        sub.add_argument("--readelf", default="readelf",
                         help="GNU readelf or llvm-readelf (default: %(default)s)")
        return sub

    aar = with_readelf(commands.add_parser("aar", help="gate a Bazel- or Gradle-built AAR"))
    aar.add_argument("aar", type=Path)
    mode = aar.add_mutually_exclusive_group(required=True)
    mode.add_argument("--speech", dest="speech", action="store_true")
    mode.add_argument("--no-speech", dest="speech", action="store_false")
    aar.add_argument("--min-sdk", type=int, default=API_LEVELS[SPEECH_ABI])

    pack = commands.add_parser("pack-dist", help="repackage an AAR as the release zip")
    pack.add_argument("aar", type=Path)
    pack.add_argument("output", type=Path)

    dist = with_readelf(commands.add_parser("dist", help="gate the release zip"))
    dist.add_argument("zip", type=Path)

    stage = commands.add_parser("stage-gradle", help="stage the release zip for Gradle")
    stage.add_argument("zip", type=Path)
    stage.add_argument("--kotlin-dir", type=Path, required=True)

    overlap = with_readelf(commands.add_parser("symbol-overlap", help="engine vs public exports"))
    overlap.add_argument("--archive", type=Path, required=True)
    overlap.add_argument("libraries", type=Path, nargs="+")

    compare = with_readelf(commands.add_parser("compare-imports", help="tts-off vs tts-on imports"))
    compare.add_argument("default", type=Path)
    compare.add_argument("speech", type=Path)

    args = parser.parse_args(argv)
    readelf = Readelf(getattr(args, "readelf", "readelf"), runner)
    try:
        if args.command == "aar":
            report = verify_package(args.aar, "aar", args.speech, readelf, args.min_sdk, glue_path)
            return report.emit(args.aar.name)
        if args.command == "dist":
            report = verify_package(args.zip, "dist", True, readelf, None, glue_path)
            return report.emit(args.zip.name)
        if args.command == "pack-dist":
            return pack_dist(args.aar, args.output)
        if args.command == "stage-gradle":
            return stage_gradle(args.zip, args.kotlin_dir)
        if args.command == "symbol-overlap":
            return symbol_overlap(args.archive, args.libraries, readelf)
        return compare_imports(args.default, args.speech, readelf)
    except GateError as error:
        print(f"verify_zzz_android: {error}", file=sys.stderr)
    except OSError:
        print("verify_zzz_android: an input or output could not be read or written",
              file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
