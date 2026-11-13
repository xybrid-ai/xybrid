"""The pinned OpenTelemetry Collector: fetch, verify, validate, run, read.

``collector.lock.json`` pins the ``otelcol-contrib`` release and the sha256 of
each platform's archive (the release's own ``.sha256`` assets). The first use
downloads the archive into a cache directory, refuses it unless the checksum
matches, extracts only the ``otelcol-contrib`` binary and deletes the archive;
later runs reuse the binary.

Runs on python3 >= 3.9 with no third-party packages.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOCK_PATH = HERE / "collector.lock.json"
PROFILES_DIR = REPO / "crates" / "xybrid-observability" / "collector"
HARNESS_OVERLAY = HERE / "collector-harness.yaml"
SINKS_JSON_OVERLAY = HERE / "collector-sinks-json.yaml"
BINARY_NAME = "otelcol-contrib"

# The destination exporter of each profile, as named in the profile.
DESTINATION_EXPORTER = {
    "langfuse": "otlphttp/langfuse",
    "phoenix": "otlphttp/phoenix",
    "xybrid": "otlphttp/xybrid",
}

# Every profile/overlay combination collector/README.md lists, with the
# environment variables each file reads. ``validate_profiles`` sets only the
# variables of the files in a combination, so a profile that needs a variable
# it does not document fails here.
PROFILE_COMBINATIONS: Tuple[Tuple[str, ...], ...] = (
    ("xybrid.yaml",),
    ("xybrid.yaml", "sinks.yaml"),
    ("langfuse.yaml",),
    ("langfuse.yaml", "sinks.yaml"),
    ("phoenix.yaml",),
    ("phoenix.yaml", "sinks.yaml"),
    ("phoenix.yaml", "phoenix-auth.yaml"),
    ("phoenix.yaml", "phoenix-auth.yaml", "sinks.yaml"),
)
PROFILE_ENV: Dict[str, Dict[str, str]] = {
    # Placeholder values: validation never connects anywhere.
    "xybrid.yaml": {"XYBRID_API_KEY": "validate-only"},
    "langfuse.yaml": {"LANGFUSE_OTLP_ENDPOINT": "http://127.0.0.1:9/api/public/otel",
                      "LANGFUSE_AUTH": "dmFsaWRhdGU6b25seQ=="},
    "phoenix.yaml": {"PHOENIX_OTLP_ENDPOINT": "http://127.0.0.1:9"},
    "phoenix-auth.yaml": {"PHOENIX_API_KEY": "validate-only"},
    "sinks.yaml": {"METRICS_OTLP_ENDPOINT": "http://127.0.0.1:9", "LOGS_OTLP_ENDPOINT": "http://127.0.0.1:9"},
}
ALL_PROFILE_VARIABLES = sorted({key for env in PROFILE_ENV.values() for key in env}
                               | {"XYBRID_TRACES_ENDPOINT", "XYBRID_LOGS_ENDPOINT"})


class CollectorError(RuntimeError):
    """The pinned Collector cannot be fetched, verified or started."""


@dataclass
class CollectorBinary:
    path: Path
    version: str
    platform: str
    archive_sha256: Optional[str]
    source: str  # "cache", "download" or "explicit"

    def describe(self) -> Dict[str, object]:
        return {"distribution": BINARY_NAME, "version": self.version, "platform": self.platform,
                "archive_sha256": self.archive_sha256, "binary": str(self.path), "source": self.source}


def load_lock(path: Path = LOCK_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def host_platform(system: Optional[str] = None, machine: Optional[str] = None) -> str:
    """The release's platform name for this host: ``darwin_arm64``, ``linux_amd64``..."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    os_name = {"darwin": "darwin", "linux": "linux"}.get(system)
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "amd64": "amd64"}.get(machine)
    if os_name is None or arch is None:
        raise CollectorError(f"no pinned {BINARY_NAME} build for {system}/{machine}")
    return f"{os_name}_{arch}"


def default_cache_dir() -> Path:
    override = os.environ.get("XYBRID_INTEROP_CACHE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "xybrid" / "otelcol"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_version(binary: Path) -> str:
    """``0.165.0`` from ``otelcol-contrib version 0.165.0``."""
    try:
        output = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CollectorError(f"cannot run {binary} --version: {error}") from None
    match = re.search(r"version\s+v?(\d+\.\d+\.\d+\S*)", output)
    if match is None:
        raise CollectorError(f"{binary} --version printed {output.strip()!r}")
    return match.group(1)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "xybrid-observability-interop"})
    with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out, 1 << 20)


def extract_binary(archive: Path, destination: Path) -> None:
    """Extracts only the regular file named ``otelcol-contrib`` (no paths, no links)."""
    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.name.lstrip("./") == BINARY_NAME]
        if len(members) != 1 or not members[0].isfile():
            raise CollectorError(f"{archive.name} has no single regular {BINARY_NAME} file")
        source = tar.extractfile(members[0])
        if source is None:
            raise CollectorError(f"cannot read {BINARY_NAME} from {archive.name}")
        temporary = destination.with_name(destination.name + ".part")
        with source, temporary.open("wb") as out:
            shutil.copyfileobj(source, out, 1 << 20)
        temporary.chmod(0o755)
        os.replace(temporary, destination)


def ensure_collector(lock: Optional[Dict[str, object]] = None, cache_dir: Optional[Path] = None,
                     explicit: Optional[Path] = None, platform_name: Optional[str] = None,
                     download: Callable[[str, Path], None] = _download,
                     log: Callable[[str], None] = print) -> CollectorBinary:
    """Returns the pinned Collector, downloading and verifying it on first use.

    ``explicit`` (``--collector-binary``) skips the download but must report
    the pinned version; its archive checksum is then unknown and recorded as
    such in the report.
    """
    lock = lock or load_lock()
    version = str(lock["version"])
    if explicit is not None:
        found = binary_version(explicit)
        if found != version:
            raise CollectorError(f"{explicit} is {BINARY_NAME} {found}; the harness pins {version}")
        return CollectorBinary(explicit.resolve(), found, platform_name or host_platform(), None, "explicit")

    platform_name = platform_name or host_platform()
    checksums = lock["sha256"]
    if platform_name not in checksums:  # type: ignore[operator]
        raise CollectorError(f"collector.lock.json pins no {platform_name} archive")
    expected = str(checksums[platform_name])  # type: ignore[index]
    directory = (cache_dir or default_cache_dir()) / version / platform_name
    binary = directory / BINARY_NAME
    marker = directory / "archive.sha256"
    if binary.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == expected:
        return CollectorBinary(binary, version, platform_name, expected, "cache")

    directory.mkdir(parents=True, exist_ok=True)
    url = str(lock["url"]).format(version=version, platform=platform_name)
    archive = directory / f"{BINARY_NAME}_{version}_{platform_name}.tar.gz"
    log(f"==> downloading {url}")
    try:
        download(url, archive)
        actual = sha256_file(archive)
        if actual != expected:
            raise CollectorError(f"{archive.name}: sha256 {actual}, collector.lock.json pins {expected}; refusing it")
        extract_binary(archive, binary)
    finally:
        if archive.exists():
            archive.unlink()
    marker.write_text(expected + "\n", encoding="utf-8")
    found = binary_version(binary)
    if found != version:
        raise CollectorError(f"the verified archive holds {BINARY_NAME} {found}, not {version}")
    log(f"==> {BINARY_NAME} {version} ({platform_name}) verified and cached in {directory}")
    return CollectorBinary(binary, version, platform_name, expected, "download")


def scrubbed_environment(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The process environment without OTEL_* and without any profile variable."""
    source = os.environ if base is None else base
    return {key: value for key, value in source.items()
            if not key.startswith("OTEL_") and key not in ALL_PROFILE_VARIABLES}


def validate(binary: Path, configs: Sequence[Path], env: Mapping[str, str]) -> Tuple[bool, str]:
    arguments = [str(binary), "validate"]
    for config in configs:
        arguments += ["--config", str(config)]
    completed = subprocess.run(arguments, capture_output=True, text=True, env=dict(env), timeout=120)
    return completed.returncode == 0, (completed.stdout + completed.stderr).strip()


def validate_profiles(binary: Path, base_env: Optional[Mapping[str, str]] = None) -> List[Dict[str, object]]:
    """``otelcol-contrib validate`` for every combination in PROFILE_COMBINATIONS.

    Each combination sees only the variables its own files read, so the
    profiles' optional variables (Xybrid endpoints, the Phoenix key) are really
    optional.
    """
    results = []
    for combination in PROFILE_COMBINATIONS:
        env = scrubbed_environment(base_env)
        for name in combination:
            env.update(PROFILE_ENV[name])
        ok, output = validate(binary, [PROFILES_DIR / name for name in combination], env)
        results.append({"configs": list(combination), "ok": ok, "variables": sorted(
            key for name in combination for key in PROFILE_ENV[name]), "output": output[-2000:]})
    return results


def wire_overlay(destination: str) -> str:
    """YAML that makes the destination exporter send uncompressed OTLP/JSON,
    so the harness's capture server can stand in for the backend (``--wire``)."""
    exporter = DESTINATION_EXPORTER[destination]
    return ("# Generated by run_interop.py --wire: the destination exporter sends\n"
            "# uncompressed OTLP/JSON to the harness's capture server.\n"
            f"exporters:\n  {exporter}:\n    encoding: json\n    compression: none\n")


def run_configs(destination: str, sinks: bool, extra: Iterable[Path] = ()) -> List[Path]:
    configs = [PROFILES_DIR / f"{destination}.yaml"]
    if sinks:
        configs += [PROFILES_DIR / "sinks.yaml", SINKS_JSON_OVERLAY]
    configs += list(extra)
    configs.append(HARNESS_OVERLAY)
    return configs


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class RunningCollector:
    """One Collector process; ``stop`` sends SIGTERM so exporters flush."""

    def __init__(self, binary: Path, configs: Sequence[Path], env: Mapping[str, str], log_path: Path) -> None:
        self.binary = binary
        self.configs = list(configs)
        self.env = dict(env)
        self.log_path = log_path
        self.process: Optional[subprocess.Popen] = None
        self._log = None

    def start(self, port: int, timeout: float = 60.0) -> None:
        if port_open("127.0.0.1", port):
            raise CollectorError(f"127.0.0.1:{port} is already in use; stop whatever listens there "
                                 f"(another Collector?) or pass --collector-port")
        arguments = [str(self.binary)]
        for config in self.configs:
            arguments += ["--config", str(config)]
        self._log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(arguments, env=self.env, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise CollectorError(f"{BINARY_NAME} exited with {self.process.returncode}; see {self.log_path}")
            if port_open("127.0.0.1", port):
                return
            time.sleep(0.2)
        self.stop()
        raise CollectorError(f"{BINARY_NAME} did not listen on 127.0.0.1:{port} within {timeout:.0f}s")

    def stop(self, timeout: float = 30.0) -> Optional[int]:
        if self.process is None:
            return None
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self._log is not None:
            self._log.close()
            self._log = None
        return self.process.returncode


# ---------------------------------------------------------------------------
# The Collector's own metrics (Prometheus text format)
# ---------------------------------------------------------------------------

_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> List[Tuple[str, Dict[str, str], float]]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        name, _, labels, value = match.groups()
        try:
            number = float(value)
        except ValueError:
            continue
        samples.append((name, dict(_LABEL.findall(labels or "")), number))
    return samples


EXPORTER_COUNTERS = {
    "otelcol_exporter_sent_spans": "sent_spans",
    "otelcol_exporter_send_failed_spans": "failed_spans",
    "otelcol_exporter_sent_log_records": "sent_log_records",
    "otelcol_exporter_send_failed_log_records": "failed_log_records",
    "otelcol_exporter_sent_metric_points": "sent_metric_points",
    "otelcol_exporter_send_failed_metric_points": "failed_metric_points",
}
RECEIVER_COUNTERS = {
    "otelcol_receiver_accepted_spans": "accepted_spans",
    "otelcol_receiver_refused_spans": "refused_spans",
    "otelcol_receiver_accepted_log_records": "accepted_log_records",
    "otelcol_receiver_refused_log_records": "refused_log_records",
    "otelcol_receiver_accepted_metric_points": "accepted_metric_points",
    "otelcol_receiver_refused_metric_points": "refused_metric_points",
}


def summarize_metrics(samples: Iterable[Tuple[str, Dict[str, str], float]]) -> Dict[str, Dict[str, float]]:
    """Per exporter and per receiver totals: ``{"exporter:otlphttp/xybrid": {"sent_spans": 6, ...}}``."""
    out: Dict[str, Dict[str, float]] = {}
    for name, labels, value in samples:
        base = name[:-len("_total")] if name.endswith("_total") else name
        if base in EXPORTER_COUNTERS and "exporter" in labels:
            key, field = f"exporter:{labels['exporter']}", EXPORTER_COUNTERS[base]
        elif base in RECEIVER_COUNTERS and "receiver" in labels:
            key, field = f"receiver:{labels['receiver']}", RECEIVER_COUNTERS[base]
        else:
            continue
        bucket = out.setdefault(key, {})
        bucket[field] = bucket.get(field, 0.0) + value
    return out


def scrape_metrics(port: int, timeout: float = 10.0) -> Tuple[Dict[str, Dict[str, float]], str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as response:  # noqa: S310
        text = response.read().decode("utf-8", "replace")
    return summarize_metrics(parse_prometheus(text)), text


def export_errors(log_text: str) -> List[str]:
    """Collector log lines that report a failed or dropped export."""
    needles = ("Exporting failed", "Dropping data", "dropped_items", "Permanent error", "rejected")
    return [line.strip()[:500] for line in log_text.splitlines() if any(needle in line for needle in needles)]
