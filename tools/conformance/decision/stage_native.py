#!/usr/bin/env python3
"""Stage the oracle's exact CPU engine archives for matched-build Rust G2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import oracle

TARGET = "x86_64-unknown-linux-gnu"
ARCHIVES = (
    "src/libllama.a", "ggml/src/libggml.a", "ggml/src/libggml-base.a",
    "ggml/src/libggml-cpu.a", "tools/mtmd/libmtmd.a", "vendor/hash/libvendor-hash.a",
)


def inputs(build: Path, target: str) -> tuple[dict, dict[str, Path]]:
    identity, _ = oracle.build_identity(build / "bin", False)
    profile = identity["build"]
    if target != TARGET or (profile["system"], profile["processor"]) != ("Linux", "x86_64"):
        raise ValueError("matched-engine staging currently requires Linux x86_64 GNU")
    if profile["build_type"] != "Release" or profile["backends"] != ["cpu"]:
        raise ValueError("matched-engine staging requires the official Release CPU build")
    files = {f"{target}/lib/{Path(name).name}": build / "llama.cpp" / name for name in ARCHIVES}
    source = Path(identity["llama_cpp_dir"])
    files.update({f"{target}/include/{name}": source / "include" / name for name in ("llama.h", "llama-cpp.h")})
    for name, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty oracle build input: {name}")
        if name.endswith(".a"):
            with path.open("rb") as stream:
                if stream.read(8) != b"!<arch>\n":
                    raise ValueError(f"not a regular static archive: {name}")
    receipt = {
        "schema_version": 1, "target": target,
        "purpose": "exact oracle engine for matched CPU G2; not a published native package",
        "stager_sha256": oracle.sha256_file(Path(__file__)),
        "oracle_identity": identity,
        "executables": {name: oracle.sha256_file(build / "bin" / name) for name in ("decision-oracle", "llama-server")},
        "files": {name: {"bytes": path.stat().st_size, "sha256": oracle.sha256_file(path)} for name, path in files.items()},
        "local_wrapper": "Cargo compiles the current wrapper and pinned Jinja sources locally",
    }
    return receipt, files


def verify(build: Path, output: Path, target: str) -> dict:
    expected, _ = inputs(build, target)
    receipt = json.loads((output / "native-stage.json").read_text())
    if receipt != expected:
        raise ValueError("staged engine identity differs from the current official oracle build")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if observed != set(receipt["files"]) | {"native-stage.json"}:
        raise ValueError("staged engine file set is incomplete or contains unrecorded files")
    for name, entry in receipt["files"].items():
        path = output / name
        if path.stat().st_size != entry["bytes"] or oracle.sha256_file(path) != entry["sha256"]:
            raise ValueError(f"staged engine digest mismatch: {name}")
    return receipt


def stage(build: Path, output: Path, target: str) -> dict:
    receipt, files = inputs(build, target)
    if output.exists():
        raise ValueError("staging destination must not exist; use --check to verify an existing prefix")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="decision-native-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        for name, path in files.items():
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        (temporary / "native-stage.json").write_text(json.dumps(receipt, indent=2) + "\n")
        verify(build, temporary, target)
        temporary.rename(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path(__file__).parent / "build")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        target = args.target
        if target is None:
            rustc = subprocess.check_output(["rustc", "-vV"], text=True)
            target = next(line.removeprefix("host: ") for line in rustc.splitlines() if line.startswith("host: "))
        receipt = (verify if args.check else stage)(args.build.resolve(), args.out.resolve(), target)
    except (ValueError, OSError, KeyError, StopIteration, subprocess.CalledProcessError) as error:
        print(f"stage_native: {error}", file=sys.stderr)
        return 1
    print(f"Verified {len(receipt['files'])} files for {target}; oracle {receipt['oracle_identity']['fingerprint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
