#!/usr/bin/env python3
"""Add complete pinned native licenses to a Bazel-produced XCFramework ZIP."""

import argparse
from pathlib import Path
import shutil
import stat
import zipfile


NOTICE_PATH = "XybridFFI.xcframework/Notices/llama/NOTICE"


def package(archive: Path, notice: Path, output: Path) -> None:
    payload = notice.read_bytes()
    if not payload or len(payload) > 1024 * 1024:
        raise ValueError("native notice is missing or oversized")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or NOTICE_PATH in names:
            raise ValueError("native archive contains duplicate notices or members")
        if "XybridFFI.xcframework/Info.plist" not in names:
            raise ValueError("native archive is not the expected XCFramework")
        with zipfile.ZipFile(output, "w") as target:
            for member in source.infolist():
                # Stream the large static archives; retain all entry metadata.
                with source.open(member) as reader, target.open(member, "w") as writer:
                    shutil.copyfileobj(reader, writer)
            info = zipfile.ZipInfo(NOTICE_PATH, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--notice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    package(arguments.archive, arguments.notice, arguments.output)
