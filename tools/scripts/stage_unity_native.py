#!/usr/bin/env python3
"""Stage a pre-built native library and its Unity plugin metadata."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PLUGINS_ROOT = Path("bindings/unity/Runtime/Plugins")
DEFAULT_ANDROID_DEPENDENCIES = Path("vendor/ort-android")
ANDROID_DEPENDENCY_NAMES = ("libonnxruntime.so", "libc++_shared.so")


@dataclass(frozen=True)
class TargetSpec:
    platform: str
    android_abi: str | None = None
    android_cpu: str | None = None


TARGETS = {
    "aarch64-apple-darwin": TargetSpec("macOS"),
    "x86_64-apple-darwin": TargetSpec("macOS"),
    "aarch64-apple-ios": TargetSpec("iOS"),
    "aarch64-apple-ios-sim": TargetSpec("iOS"),
    "x86_64-apple-ios": TargetSpec("iOS"),
    "aarch64-linux-android": TargetSpec("Android", "arm64-v8a", "ARM64"),
    "armv7-linux-androideabi": TargetSpec(
        "Android", "armeabi-v7a", "ARMv7"
    ),
    "x86_64-linux-android": TargetSpec("Android", "x86_64", "x86_64"),
    "x86_64-pc-windows-msvc": TargetSpec("Windows"),
    "x86_64-pc-windows-gnu": TargetSpec("Windows"),
    "x86_64-unknown-linux-gnu": TargetSpec("Linux"),
}


def unity_guid(logical_name: str) -> str:
    return hashlib.sha256(f"xybrid-unity:{logical_name}".encode()).hexdigest()[
        :32
    ]


def folder_meta(guid: str) -> str:
    return f"""\
fileFormatVersion: 2
guid: {guid}
folderAsset: yes
DefaultImporter:
  externalObjects: {{}}
  userData:
  assetBundleName:
  assetBundleVariant:
"""


def plugin_meta(guid: str, spec: TargetSpec) -> str:
    if spec.platform == "macOS":
        target_data = """\
        Exclude Editor: 0
        Exclude Linux64: 1
        Exclude OSXUniversal: 0
        Exclude WebGL: 1
        Exclude Win: 1
        Exclude Win64: 1
    Editor:
      enabled: 1
      settings:
        CPU: AnyCPU
        DefaultValueInitialized: true
        OS: AnyOS
    OSXUniversal:
      enabled: 1
      settings:
        CPU: AnyCPU"""
    elif spec.platform == "iOS":
        target_data = """\
        Exclude Editor: 1
        Exclude Linux64: 1
        Exclude OSXUniversal: 1
        Exclude WebGL: 1
        Exclude Win: 1
        Exclude Win64: 1
    Editor:
      enabled: 0
      settings:
        CPU: AnyCPU
        DefaultValueInitialized: true
        OS: AnyOS
    iOS:
      enabled: 1
      settings:
        AddToEmbeddedBinaries: false
        CPU: ARM64
        CompileFlags:
        FrameworkDependencies:"""
    elif spec.platform == "Android":
        target_data = f"""\
        Exclude Editor: 1
        Exclude Linux64: 1
        Exclude OSXUniversal: 1
        Exclude WebGL: 1
        Exclude Win: 1
        Exclude Win64: 1
    Android:
      enabled: 1
      settings:
        CPU: {spec.android_cpu}
    Editor:
      enabled: 0
      settings:
        CPU: AnyCPU
        DefaultValueInitialized: true
        OS: AnyOS"""
    elif spec.platform == "Windows":
        target_data = """\
        Exclude Editor: 0
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
        Exclude Editor: 0
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
{target_data}
  userData:
  assetBundleName:
  assetBundleVariant:
"""


def write_meta_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")
    print(f"  Created metadata: {path}")


def copy_file(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def stage_file(
    source: Path,
    destination: Path,
    spec: TargetSpec,
    logical_name: str,
) -> None:
    copy_file(source, destination)
    metadata = destination.with_name(f"{destination.name}.meta")
    write_meta_if_missing(
        metadata,
        plugin_meta(unity_guid(logical_name), spec),
    )
    print(f"  Staged: {destination}")


def stage_native(
    native: Path,
    target: str,
    *,
    plugins_root: Path = DEFAULT_PLUGINS_ROOT,
    android_dependencies: Path = DEFAULT_ANDROID_DEPENDENCIES,
) -> Path:
    native = Path(native)
    if not native.is_file():
        raise FileNotFoundError(f"native library not found: {native}")

    try:
        spec = TARGETS[target]
    except KeyError as error:
        raise ValueError(f"unsupported target for Unity staging: {target}") from error

    platform_dir = plugins_root / spec.platform
    destination_dir = (
        platform_dir / spec.android_abi
        if spec.android_abi is not None
        else platform_dir
    )
    destination_dir.mkdir(parents=True, exist_ok=True)

    platform_metadata = plugins_root / f"{spec.platform}.meta"
    write_meta_if_missing(
        platform_metadata,
        folder_meta(unity_guid(f"folder:{spec.platform}")),
    )

    if spec.android_abi is not None:
        abi_metadata = platform_dir / f"{spec.android_abi}.meta"
        write_meta_if_missing(
            abi_metadata,
            folder_meta(
                unity_guid(f"folder:{spec.platform}/{spec.android_abi}")
            ),
        )

    destination = destination_dir / native.name
    logical_directory = spec.platform
    if spec.android_abi is not None:
        logical_directory = f"{logical_directory}/{spec.android_abi}"
    stage_file(
        native,
        destination,
        spec,
        f"{logical_directory}/{native.name}",
    )

    if spec.android_abi is not None:
        dependencies_dir = android_dependencies / spec.android_abi
        if not dependencies_dir.is_dir():
            print(
                "  Warning: Android dependencies not found for "
                f"{spec.android_abi} at {dependencies_dir}",
                file=sys.stderr,
            )
            return destination

        for dependency_name in ANDROID_DEPENDENCY_NAMES:
            dependency = dependencies_dir / dependency_name
            if dependency.is_file():
                stage_file(
                    dependency,
                    destination_dir / dependency_name,
                    spec,
                    f"{logical_directory}/{dependency_name}",
                )

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a pre-built native library into Unity plugins."
    )
    parser.add_argument("--lib", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument(
        "--plugins-root",
        type=Path,
        default=DEFAULT_PLUGINS_ROOT,
        help="Unity Runtime/Plugins directory.",
    )
    parser.add_argument(
        "--android-dependencies",
        type=Path,
        default=DEFAULT_ANDROID_DEPENDENCIES,
        help="Directory containing Android dependency libraries by ABI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        destination = stage_native(
            args.lib,
            args.target,
            plugins_root=args.plugins_root,
            android_dependencies=args.android_dependencies,
        )
        print(f"Unity native staged at {destination}")
        return 0
    except (OSError, ValueError) as error:
        print(f"stage-unity-native: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
