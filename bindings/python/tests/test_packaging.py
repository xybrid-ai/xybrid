"""Packaging checks that do not need compiled native artifacts."""

import runpy
import sys
import sysconfig
from pathlib import Path
from unittest.mock import patch

import pytest
from setuptools import Distribution


@pytest.fixture
def wheel_command(tmp_path, monkeypatch):
    with patch("setuptools.setup"):
        namespace = runpy.run_path(str(Path(__file__).parents[1] / "setup.py"))
    command_type = namespace["_NativeBdistWheel"]
    monkeypatch.setitem(command_type.run.__globals__, "_BOLT_DIR", tmp_path)
    command = command_type(Distribution({"packages": []}))
    command.ensure_finalized()
    return command


def test_macos_wheel_tag_preserves_intel_architecture(wheel_command, monkeypatch):
    monkeypatch.delenv("MACOSX_DEPLOYMENT_TARGET", raising=False)
    with patch.object(
        type(wheel_command).__bases__[0], "get_tag",
        return_value=("cp312", "cp312", "macosx_26_0_x86_64"),
    ):
        assert wheel_command.get_tag() == ("cp312", "cp312", "macosx_11_0_x86_64")


def test_wheel_rejects_bridge_from_another_python(wheel_command, tmp_path):
    library = {"darwin": "libxybrid_bolt.dylib", "linux": "libxybrid_bolt.so", "win32": "xybrid_bolt.dll"}[sys.platform]
    (tmp_path / library).touch()
    (tmp_path / "_native.cpython-999-other.so").touch()
    with pytest.raises(RuntimeError, match="interpreter|mismatched"):
        wheel_command.run()


def test_wheel_rejects_extra_stale_native_files(wheel_command, tmp_path):
    library = {"darwin": "libxybrid_bolt.dylib", "linux": "libxybrid_bolt.so", "win32": "xybrid_bolt.dll"}[sys.platform]
    (tmp_path / library).touch()
    (tmp_path / f"_native{sysconfig.get_config_var('EXT_SUFFIX')}").touch()
    (tmp_path / "_native.cpython-999-other.so").touch()
    with pytest.raises(RuntimeError, match="stale|mismatched"):
        wheel_command.run()
