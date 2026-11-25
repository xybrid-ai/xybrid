"""Stage the FunctionGemma fixture into an app sandbox (Android / iOS)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from .report import ANDROID_PACKAGE, IOS_BUNDLE_ID, MODEL_DIR_RELATIVE


def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(list(command), check=check)


def require_model_dir(model_dir: Path) -> Path:
    metadata = model_dir / "model_metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"missing model_metadata.json under {model_dir}")
    weights = list(model_dir.glob("*.gguf"))
    if not weights:
        raise FileNotFoundError(f"no .gguf weights under {model_dir}")
    return model_dir


def stage_android(
    model_dir: Path,
    *,
    package: str = ANDROID_PACKAGE,
    adb: str = "adb",
) -> None:
    """Copy model files into the app's files/{MODEL_DIR_RELATIVE} via root adb.

    Release APKs are not debuggable, so ``run-as`` is unavailable. The CI
    emulator is rooted; ``adb root`` + direct writes under /data/data work.
    """
    model_dir = require_model_dir(model_dir)
    _run([adb, "root"])
    _run([adb, "wait-for-device"])
    remote_tmp = "/data/local/tmp/xybrid-consumer-smoke-model"
    _run([adb, "shell", "rm", "-rf", remote_tmp])
    _run([adb, "shell", "mkdir", "-p", remote_tmp])
    for path in sorted(model_dir.iterdir()):
        if path.name.startswith("."):
            continue
        _run([adb, "push", str(path), f"{remote_tmp}/{path.name}"])

    app_files = f"/data/data/{package}/files"
    dest = f"{app_files}/{MODEL_DIR_RELATIVE}"
    # Ensure the package has been installed at least once so its data dir exists.
    _run([adb, "shell", "mkdir", "-p", dest])
    _run([adb, "shell", f"cp -R {remote_tmp}/. {dest}/"])
    # App uid must own the files.
    _run(
        [
            adb,
            "shell",
            f"uid=$(stat -c %u /data/data/{package}) && "
            f"chown $uid:$uid {app_files} {app_files}/models && chmod 700 {app_files} && "
            f"chown -R $uid:$uid {dest} && restorecon -DR {app_files}",
        ]
    )


def stage_ios(
    model_dir: Path,
    *,
    udid: str,
    bundle_id: str = IOS_BUNDLE_ID,
) -> Path:
    """Copy model files into the simulator app's Documents/{MODEL_DIR_RELATIVE}."""
    model_dir = require_model_dir(model_dir)
    container = subprocess.check_output(
        ["xcrun", "simctl", "get_app_container", udid, bundle_id, "data"],
        text=True,
    ).strip()
    if not container:
        raise RuntimeError(f"no data container for {bundle_id} on {udid}")
    dest = Path(container) / "Documents" / MODEL_DIR_RELATIVE
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, dest)
    print(f"staged model at {dest}", flush=True)
    return dest
