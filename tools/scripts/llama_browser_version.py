#!/usr/bin/env python3
"""Generate browser version macros from the same sources as native CMake."""

import argparse
import json
import re
from pathlib import Path


def cmake_version(path: Path, project: str) -> str:
    source = path.read_text()
    parts = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"set\({project}_VERSION_{component}\s+(\d+)\)", source)
        if match is None:
            raise ValueError(f"{path}: missing {project}_VERSION_{component}")
        parts.append(match[1])
    return ".".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llama_cmake", type=Path)
    parser.add_argument("ggml_cmake", type=Path)
    parser.add_argument("tracking", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    commit = json.loads(args.tracking.read_text())["commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Invalid tracked llama.cpp commit")
    args.output.mkdir(parents=True, exist_ok=True)
    for project, cmake in (("LLAMA", args.llama_cmake), ("GGML", args.ggml_cmake)):
        header = args.output / f"{project.lower()}-version.h"
        header.write_text(
            "#pragma once\n"
            f"#define {project}_VERSION {json.dumps(cmake_version(cmake, project))}\n"
            f"#define {project}_COMMIT {json.dumps(commit)}\n"
        )


if __name__ == "__main__":
    main()
