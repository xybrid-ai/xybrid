#!/usr/bin/env python3
"""Stage the pinned ONNX Runtime shared library into a Unity plugin tree."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
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


# Both runtimes come from Microsoft's NuGet package Microsoft.ML.OnnxRuntime,
# the official native CPU distribution. The GitHub release archives this used
# to download are gone: github.com/microsoft/onnxruntime answers 404. The
# SHA-256 is of the .nupkg (a zip); its SHA-512 equals the packageHash that
# nuget.org publishes in the package's catalog entry. The version deliberately
# matches ort-sys 2.0.0-rc.11's ONNX Runtime.
NUPKG_URL = (
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/"
    f"{ORT_VERSION}/microsoft.ml.onnxruntime.{ORT_VERSION}.nupkg"
)
NUPKG_SHA256 = "25fe172f7fcdf34f2b5b02c8b997f6b88abc282956849d4af4335af23d0c4e4a"

RUNTIMES = {
    "linux": RuntimeSpec(
        platform="linux",
        url=NUPKG_URL,
        sha256=NUPKG_SHA256,
        # The real ELF (SONAME libonnxruntime.so.1), installed under the name
        # ort/load-dynamic opens.
        archive_member="runtimes/linux-x64/native/libonnxruntime.so",
        runtime_name="libonnxruntime.so",
    ),
    "windows": RuntimeSpec(
        platform="windows",
        url=NUPKG_URL,
        sha256=NUPKG_SHA256,
        archive_member="runtimes/win-x64/native/onnxruntime.dll",
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
        with zipfile.ZipFile(archive) as zip_file:
            return zip_file.read(spec.archive_member)
    except (KeyError, zipfile.BadZipFile) as error:
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
        Exclude WebGL: 1
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
        Exclude WebGL: 1
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
    try:
        with tempfile.TemporaryDirectory(prefix="xybrid-unity-ort-") as temp:
            archive = Path(temp) / f"onnxruntime-{ORT_VERSION}.nupkg"
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
