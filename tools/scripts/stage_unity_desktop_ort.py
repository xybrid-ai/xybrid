#!/usr/bin/env python3
"""Stage the pinned ONNX Runtime shared library into a Unity plugin tree."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


ORT_VERSION = "1.23.2"


@dataclass(frozen=True)
class RuntimeSpec:
    platform: str
    url: str
    sha256: str
    archive_member: str
    runtime_name: str


# These are Microsoft's CPU release archives. The digests are published by the
# GitHub release API for v1.23.2 and deliberately match ort-sys 2.0.0-rc.11's
# ONNX Runtime version.
RUNTIMES = {
    "linux": RuntimeSpec(
        platform="linux",
        url=(
            "https://github.com/microsoft/onnxruntime/releases/download/"
            f"v{ORT_VERSION}/onnxruntime-linux-x64-{ORT_VERSION}.tgz"
        ),
        sha256="1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
        # Extract the real ELF payload, not the archive's libonnxruntime.so
        # symlink, then install it under the name ort/load-dynamic opens.
        archive_member=(
            f"onnxruntime-linux-x64-{ORT_VERSION}/lib/"
            f"libonnxruntime.so.{ORT_VERSION}"
        ),
        runtime_name="libonnxruntime.so",
    ),
    "windows": RuntimeSpec(
        platform="windows",
        url=(
            "https://github.com/microsoft/onnxruntime/releases/download/"
            f"v{ORT_VERSION}/onnxruntime-win-x64-{ORT_VERSION}.zip"
        ),
        sha256="0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd",
        archive_member=(
            f"onnxruntime-win-x64-{ORT_VERSION}/lib/onnxruntime.dll"
        ),
        runtime_name="onnxruntime.dll",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec: RuntimeSpec, destination: Path) -> None:
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "xybrid-unity-ort-stager/1"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def read_runtime(spec: RuntimeSpec, archive: Path) -> bytes:
    try:
        if spec.platform == "linux":
            with tarfile.open(archive, "r:gz") as tar:
                member = tar.extractfile(spec.archive_member)
                if member is None:
                    raise KeyError(spec.archive_member)
                return member.read()

        with zipfile.ZipFile(archive) as zip_file:
            return zip_file.read(spec.archive_member)
    except (KeyError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"ONNX Runtime archive does not contain {spec.archive_member}"
        ) from error


def unity_guid(spec: RuntimeSpec) -> str:
    logical_name = f"xybrid-unity:{spec.platform}:{spec.runtime_name}"
    return hashlib.sha256(logical_name.encode()).hexdigest()[:32]


def unity_meta(spec: RuntimeSpec) -> str:
    guid = unity_guid(spec)
    if spec.platform == "windows":
        target_data = """\
        Exclude Linux64: 1
        Exclude OSXUniversal: 1
        Exclude Win: 0
        Exclude Win64: 0
    Editor:
      enabled: 1
      settings:
        CPU: AnyCPU
        DefaultValueInitialized: true
        OS: Windows
    Win:
      enabled: 1
      settings:
        CPU: x86
    Win64:
      enabled: 1
      settings:
        CPU: x86_64"""
    else:
        target_data = """\
        Exclude Linux64: 0
        Exclude OSXUniversal: 1
        Exclude Win: 1
        Exclude Win64: 1
    Editor:
      enabled: 1
      settings:
        CPU: AnyCPU
        DefaultValueInitialized: true
        OS: AnyOS
    Linux64:
      enabled: 1
      settings:
        CPU: x86_64"""

    return f"""\
fileFormatVersion: 2
guid: {guid}
PluginImporter:
  externalObjects: {{}}
  serializedVersion: 3
  iconMap: {{}}
  executionOrder: {{}}
  defineConstraints: []
  isPreloaded: 0
  isOverridable: 1
  isExplicitlyReferenced: 0
  validateReferences: 1
  platformData:
    Any:
      enabled: 0
      settings:
        Exclude Editor: 0
{target_data}
  userData:
  assetBundleName:
  assetBundleVariant:
"""


def stage_from_archive(
    spec: RuntimeSpec, archive: Path, output_dir: Path
) -> Path:
    actual_sha = sha256(archive)
    if actual_sha != spec.sha256:
        raise ValueError(
            "ONNX Runtime archive SHA-256 mismatch: "
            f"expected {spec.sha256}, got {actual_sha}"
        )

    runtime = read_runtime(spec, archive)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / spec.runtime_name
    temporary = output_dir / f".{spec.runtime_name}.tmp"
    try:
        temporary.write_bytes(runtime)
        temporary.replace(destination)
    finally:
        with suppress(OSError):
            temporary.unlink()
    destination.with_name(f"{destination.name}.meta").write_text(
        unity_meta(spec), encoding="utf-8"
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, verify, and stage desktop ONNX Runtime for Unity."
    )
    parser.add_argument("platform", choices=sorted(RUNTIMES))
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Unity platform plugin directory (for example Runtime/Plugins/Linux)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = RUNTIMES[args.platform]
    suffix = ".tgz" if spec.platform == "linux" else ".zip"
    try:
        with tempfile.TemporaryDirectory(prefix="xybrid-unity-ort-") as temp:
            archive = Path(temp) / f"onnxruntime-{spec.platform}{suffix}"
            print(f"Downloading ONNX Runtime {ORT_VERSION} for {spec.platform}...")
            download(spec, archive)
            destination = stage_from_archive(spec, archive, args.output_dir)
            print(f"Staged {destination}")
        return 0
    except Exception as error:
        print(f"stage-unity-desktop-ort: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
