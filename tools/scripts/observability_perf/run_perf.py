#!/usr/bin/env python3
"""Observability overhead and memory benchmarks: end-to-end real-model runs
and the micro benchmarks, with machine-readable reports.

    python3 tools/scripts/observability_perf/run_perf.py fetch-model
    python3 tools/scripts/observability_perf/run_perf.py e2e --rounds 10
    python3 tools/scripts/observability_perf/run_perf.py micro
    python3 tools/scripts/observability_perf/run_perf.py report --e2e DIR [--micro DIR]

``e2e`` builds ``crates/xybrid-sdk/examples/observability_perf.rs`` once
(release), then runs it ``--rounds`` times per telemetry mode, one process per
(round, mode), rotating the mode order every round so drift (thermal, page
cache, background load) spreads over every mode instead of biasing one. Each
process loads the model, discards ``--warmup`` runs and measures
``--runs-per-process`` streaming runs. Peak RSS and the peak memory footprint
come from ``/usr/bin/time -l`` on macOS (``wait4`` elsewhere).

Modes: ``off`` (no session), ``metadata`` (OTLP pipeline, metadata only),
``content`` (OTLP pipeline, default content capture), ``blocked`` (content to
an endpoint that accepts connections and never answers). The accepting
endpoints are ``observability_capture/capture_server.py``, one per mode.

Nothing here contacts a hosted service: the endpoints are local, ``OTEL_*``,
the Xybrid API keys and ``XYBRID_RESOURCE_TELEMETRY`` are removed from the
driver's environment, and the model comes from a local directory
(``fetch-model`` downloads the pinned GGUF from Hugging Face once).

Reports (``report.json`` and ``report.md``) carry the environment, workload,
schedule, per-mode distributions, bootstrap comparisons against ``off`` and
the proposed thresholds from ``thresholds.json``, evaluated but labelled as
proposals until agreed. See README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import micro  # noqa: E402 -- sibling module
import perf_common as pc  # noqa: E402 -- sibling module

REPO = pc.REPO
CAPTURE_SERVER = REPO / "tools" / "scripts" / "observability_capture" / "capture_server.py"
MODELS_JSON = REPO / "integration-tests" / "fixtures" / "models" / "models.json"

MODES = ("off", "metadata", "content", "blocked")
SINK_MODES = ("metadata", "content")

# The smallest text-generation GGUF the repo already tracks with a pinned
# Qwen3.5-0.8B recipe (models.json `qwen3.5-0.8b`; the CI choice-conformance
# lane converts the same model). Pinned to a Hugging Face revision and the
# file's LFS SHA-256 so every run measures the same bytes.
MODEL = {
    "id": "qwen3.5-0.8b",
    "repo": "unsloth/Qwen3.5-0.8B-GGUF",
    "revision": "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
    "file": "Qwen3.5-0.8B-Q4_K_M.gguf",
    "sha256": "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517",
    "size": 532517120,
}
DEFAULT_MODEL_DIR = REPO / "integration-tests" / "fixtures" / "models" / MODEL["id"]

# Variables that would change what a mode measures or reach a hosted service.
STRIPPED_ENV_PREFIXES = ("OTEL_",)
STRIPPED_ENV = ("XYBRID_API_KEY", "XYBRID_STAGING_API_KEY", "XYBRID_RESOURCE_TELEMETRY", "XYBRID_PLATFORM_API_KEY")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def model_url(spec: Dict[str, Any] = MODEL) -> str:
    return f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def model_metadata(models_json: Path = MODELS_JSON, model_id: str = MODEL["id"]) -> Dict[str, Any]:
    """The repo's ``model_metadata`` for the pinned model (models.json)."""
    manifest = json.loads(models_json.read_text(encoding="utf-8"))
    return manifest["models"][model_id]["model_metadata"]


def fetch_model(dest: Path, spec: Dict[str, Any] = MODEL, url: Optional[str] = None) -> Path:
    """Downloads the pinned GGUF into ``dest`` (verified, atomic) and writes
    ``model_metadata.json`` from models.json. A verified file is kept."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / spec["file"]
    if target.exists() and sha256_file(target) == spec["sha256"]:
        pc.stage(f"{target} already verified")
    else:
        partial = target.with_name(target.name + ".part")
        source = url or model_url(spec)
        pc.stage(f"downloading {source}")
        request = urllib.request.Request(source, headers={"User-Agent": "xybrid-observability-perf"})
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                size += len(block)
        if digest.hexdigest() != spec["sha256"] or size != spec["size"]:
            partial.unlink(missing_ok=True)
            raise SystemExit(
                f"error: {source} has sha256 {digest.hexdigest()} / {size} bytes, "
                f"expected {spec['sha256']} / {spec['size']}"
            )
        os.replace(partial, target)
    metadata_path = dest / "model_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(model_metadata(), indent=2) + "\n", encoding="utf-8")
    return target


def verify_model(model_dir: Path, allow_unpinned: bool) -> Dict[str, Any]:
    """Identity of the model the run measures: the file ``model_metadata.json``
    names (``execution_template.model_file``), which is the one the driver
    loads, whatever else the directory holds. Fails when the metadata names no
    file or a missing one, and, unless ``allow_unpinned``, when that file is
    not the pinned one with the pinned bytes."""
    info: Dict[str, Any] = {"dir": str(model_dir), "pinned": False}
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"error: {metadata_path} not found; run `run_perf.py fetch-model`")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise SystemExit(f"error: {metadata_path} is not JSON ({error})") from None
    template = metadata.get("execution_template") if isinstance(metadata, dict) else None
    model_file = template.get("model_file") if isinstance(template, dict) else None
    if not isinstance(model_file, str) or not model_file:
        raise SystemExit(f"error: {metadata_path} names no execution_template.model_file, so the measured "
                         "model is unknown")
    measured = model_dir / model_file
    if not measured.is_file():
        raise SystemExit(f"error: {metadata_path} names {model_file}, which is not in {model_dir}")
    pc.stage(f"hashing {model_file}")
    info.update(file=model_file, size=measured.stat().st_size, sha256=sha256_file(measured))
    if model_file == MODEL["file"] and info["sha256"] == MODEL["sha256"]:
        info.update(pinned=True, repo=MODEL["repo"], revision=MODEL["revision"])
    elif not allow_unpinned:
        raise SystemExit(
            f"error: {metadata_path} names {model_file} (sha256 {info['sha256']}), not the pinned "
            f"{MODEL['file']} (sha256 {MODEL['sha256']}); run `run_perf.py fetch-model` or pass "
            "--allow-unpinned-model to measure another model"
        )
    return info


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class BlockedEndpoint:
    """Accepts connections and reads requests but never answers: every export
    to it waits until its deadline or the pipeline's shutdown."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(128)
        self._sock.settimeout(0.1)
        self.port = self._sock.getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self.connections = 0
        self.bytes = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                self.connections += 1
            threading.Thread(target=self._drain, args=(conn,), daemon=True).start()

    def _drain(self, conn: socket.socket) -> None:
        conn.settimeout(0.2)
        with conn:
            while not self._stop.is_set():
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not data:
                    return
                with self._lock:
                    self.bytes += len(data)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {"endpoint": self.endpoint, "connections": self.connections, "bytes_read": self.bytes}

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2)


class Sink:
    """One ``capture_server.py`` (answers 200 and records every request)."""

    def __init__(self, out_dir: Path, name: str, timeout: float = 15.0) -> None:
        self.capture = out_dir / f"sink-{name}.jsonl"
        ready = out_dir / f"sink-{name}.ready.json"
        ready.unlink(missing_ok=True)
        self.log = (out_dir / f"sink-{name}.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, str(CAPTURE_SERVER), "--host", "127.0.0.1", "--port", "0",
             "--out", str(self.capture), "--ready-file", str(ready)],
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + timeout
        while not ready.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                # Nobody holds this half-built sink, so nothing else would stop
                # a server that is still starting.
                self.close()
                raise SystemExit(f"error: capture_server.py ({name}) did not start; see {self.log.name}")
            time.sleep(0.05)
        self.endpoint = json.loads(ready.read_text(encoding="utf-8"))["endpoint"]

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()

    def summary(self) -> Dict[str, Any]:
        return summarize_capture(self.capture)


def summarize_capture(path: Path) -> Dict[str, Any]:
    """Requests and body bytes per signal in a capture file."""
    out: Dict[str, Any] = {"requests": 0, "bytes": 0, "signals": {}}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        signal_name = entry.get("signal") or "other"
        per = out["signals"].setdefault(signal_name, {"requests": 0, "bytes": 0})
        per["requests"] += 1
        per["bytes"] += int(entry.get("size") or 0)
        out["requests"] += 1
        out["bytes"] += int(entry.get("size") or 0)
    return out


# ---------------------------------------------------------------------------
# Process measurement
# ---------------------------------------------------------------------------

_TIME_HEADER = re.compile(r"^\s*([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys\s*$")
_TIME_FIELD = re.compile(r"^\s*(\d+)\s+([a-z][a-z ]+[a-z])\s*$")


def parse_bsd_time(stderr: str) -> Optional[Dict[str, Any]]:
    """Parses the ``/usr/bin/time -l`` block (macOS) at the end of ``stderr``:
    real/user/sys seconds and every counter, keyed by its label. Sizes
    (``maximum resident set size``, ``peak memory footprint``) are bytes."""
    lines = stderr.splitlines()
    start = None
    for index in range(len(lines) - 1, -1, -1):
        if _TIME_HEADER.match(lines[index]):
            start = index
            break
    if start is None:
        return None
    header = _TIME_HEADER.match(lines[start])
    assert header is not None
    parsed: Dict[str, Any] = {
        "real_s": float(header.group(1)),
        "user_s": float(header.group(2)),
        "sys_s": float(header.group(3)),
    }
    for line in lines[start + 1:]:
        field = _TIME_FIELD.match(line)
        if field:
            parsed[field.group(2)] = int(field.group(1))
    return parsed


def use_bsd_time() -> bool:
    return platform.system() == "Darwin" and Path("/usr/bin/time").exists()


def driver_env() -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(STRIPPED_ENV_PREFIXES) and k not in STRIPPED_ENV}
    return env


def kill_process_group(process: "subprocess.Popen[bytes]") -> None:
    """SIGKILL to ``process``'s whole process group (its own session): on
    macOS ``process`` is ``/usr/bin/time``, and killing only the wrapper
    would leave the driver running (GPU, memory) through every later round."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        process.kill()


def run_measured(command: List[str], stdout_path: Path, stderr_path: Path, timeout: float,
                 env: Dict[str, str]) -> Dict[str, Any]:
    """Runs ``command`` once; returns exit status, wall time and peak memory.

    The process runs in its own session, so a timeout (or an interrupt of this
    harness) kills the driver along with the ``/usr/bin/time`` wrapper."""
    wrapped = (["/usr/bin/time", "-l"] + command) if use_bsd_time() else command
    started = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        process = subprocess.Popen(wrapped, stdout=out, stderr=err, env=env, start_new_session=True)
        rusage = None
        timed_out = False
        deadline = started + timeout
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    rusage = usage
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    kill_process_group(process)
                    pid, status, usage = os.wait4(process.pid, 0)
                    rusage = usage
                    break
                time.sleep(0.01)
        except BaseException:
            kill_process_group(process)
            process.wait()
            raise
        process.returncode = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else (status >> 8)
    wall = time.monotonic() - started
    result: Dict[str, Any] = {
        "command": command,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "wall_s": wall,
    }
    if use_bsd_time():
        parsed = parse_bsd_time(stderr_path.read_text(encoding="utf-8", errors="replace"))
        result["rss_source"] = "/usr/bin/time -l"
        result["time"] = parsed
        result["peak_rss_bytes"] = parsed.get("maximum resident set size") if parsed else None
        result["peak_footprint_bytes"] = parsed.get("peak memory footprint") if parsed else None
    else:
        # ru_maxrss: kilobytes on Linux, bytes on macOS.
        scale = 1 if platform.system() == "Darwin" else 1024
        result["rss_source"] = "wait4 ru_maxrss"
        result["peak_rss_bytes"] = int(rusage.ru_maxrss) * scale if rusage is not None else None
        result["peak_footprint_bytes"] = None
        if rusage is not None:
            result["time"] = {"user_s": rusage.ru_utime, "sys_s": rusage.ru_stime}
    return result


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def williams_rows(count: int) -> List[List[int]]:
    """A Williams design over ``count`` treatments: a Latin square (each
    treatment once per row and once per column) in which every treatment
    directly follows every other one equally often across the rows. Even
    ``count`` needs ``count`` rows; odd ``count`` needs ``2 * count`` (the
    square and its mirror)."""
    if count <= 0:
        return []
    first = []
    for j in range(count):
        if j == 0:
            first.append(0)
        elif j % 2:
            first.append((j + 1) // 2)
        else:
            first.append(count - j // 2)
    rows = [[(value + shift) % count for value in first] for shift in range(count)]
    if count % 2:
        rows += [list(reversed(row)) for row in rows]
    return rows


def schedule(modes: Sequence[str], rounds: int, seed: Optional[int] = None) -> List[Tuple[int, str]]:
    """Interleaved (round, mode) order.

    Without a seed, rounds follow a Williams design: over every
    ``len(williams_rows(n))`` rounds each mode takes each position equally
    often *and* follows each other mode equally often within a round, so no
    mode (``off``, the baseline, included) keeps the same predecessor and
    inherits its carryover (thermal state, GPU clocks, page cache, a slow
    exit). A plain rotation gave every mode a fixed predecessor. With a seed
    each round is shuffled instead."""
    rng = random.Random(seed) if seed is not None else None
    rows = williams_rows(len(modes))
    order: List[Tuple[int, str]] = []
    for r in range(rounds):
        if rng is not None:
            round_modes = list(modes)
            rng.shuffle(round_modes)
        else:
            round_modes = [modes[index] for index in rows[r % len(rows)]]
        order.extend((r, m) for m in round_modes)
    return order


def predecessor_counts(order: Sequence[Tuple[int, str]]) -> Dict[str, Dict[str, int]]:
    """How often each mode ran right after each other mode (process order),
    recorded so a report shows how balanced the carryover was."""
    modes = [mode for _, mode in order]
    counts: Dict[str, Dict[str, int]] = {mode: {} for mode in dict.fromkeys(modes)}
    for previous, mode in zip(modes, modes[1:]):
        counts[mode][previous] = counts[mode].get(previous, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_driver(features: str, profile: str = "release") -> Tuple[Path, Dict[str, Any]]:
    command = ["cargo", "build", "-p", "xybrid-sdk", "--example", "observability_perf",
               "--message-format=json-render-diagnostics"]
    if profile == "release":
        command.append("--release")
    else:
        command += ["--profile", profile]
    if features:
        command += ["--features", features]
    pc.stage("building: " + " ".join(command))
    started = time.monotonic()
    process = subprocess.run(command, cwd=REPO, stdout=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise SystemExit(f"error: driver build failed ({process.returncode})")
    executable = None
    for line in process.stdout.splitlines():
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get("reason") == "compiler-artifact" and message.get("executable") \
                and message.get("target", {}).get("name") == "observability_perf":
            executable = Path(message["executable"])
    if executable is None:
        raise SystemExit("error: cargo reported no observability_perf executable")
    return executable, {"command": command, "seconds": time.monotonic() - started, "executable": str(executable)}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _runs(processes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [run for p in processes for run in (p.get("driver") or {}).get("runs", [])]


def _gap_percentile(run: Dict[str, Any], q: float) -> Optional[float]:
    gaps = sorted(run.get("callback_gaps_ms") or [])
    return pc.percentile(gaps, q) if gaps else None


RUN_METRICS = {
    # name: (getter over one measured run, unit, higher_is_better)
    "runtime_tokens_per_s": (lambda r: r["runtime"]["tokens_per_second"], "tok/s", True),
    "callback_decode_per_s": (lambda r: r["decode_callbacks_per_s"], "callbacks/s", True),
    "first_callback_ms": (lambda r: r["first_callback_ms"], "ms", False),
    "runtime_ttft_ms": (lambda r: r["runtime"]["ttft_ms"], "ms", False),
    "wall_ms": (lambda r: r["wall_ms"], "ms", False),
    "callback_gap_p50_ms": (lambda r: _gap_percentile(r, 0.50), "ms", False),
    "callback_gap_p95_ms": (lambda r: _gap_percentile(r, 0.95), "ms", False),
    "callback_gap_p99_ms": (lambda r: _gap_percentile(r, 0.99), "ms", False),
    "callback_gap_max_ms": (lambda r: max(r["callback_gaps_ms"]) if r.get("callback_gaps_ms") else None, "ms", False),
    "callbacks": (lambda r: r["callbacks"], "count", True),
}

PROCESS_METRICS = {
    "peak_rss_mib": (lambda p: pc.mib(p["measure"]["peak_rss_bytes"]), "MiB", False),
    "peak_footprint_mib": (lambda p: pc.mib(p["measure"]["peak_footprint_bytes"]), "MiB", False),
    "load_ms": (lambda p: p["driver"]["load_ms"], "ms", False),
    "shutdown_ms": (lambda p: p["driver"]["telemetry"]["shutdown_ms"], "ms", False),
    "process_wall_s": (lambda p: p["measure"]["wall_s"], "s", False),
}


def analyze(raw: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    processes = raw["processes"]
    modes = raw["workload"]["modes"]
    by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    failures: Dict[str, int] = {m: 0 for m in modes}
    for p in processes:
        if p.get("ok"):
            by_mode[p["mode"]].append(p)
        else:
            failures[p["mode"]] += 1

    summary: Dict[str, Any] = {}
    for mode in modes:
        procs = by_mode[mode]
        runs = _runs(procs)
        gaps = [g for run in runs for g in run.get("callback_gaps_ms") or []]
        stats_after = [((p["driver"].get("telemetry") or {}).get("stats_after_shutdown") or {}) for p in procs]
        traces_after = [s.get("traces") or {} for s in stats_after if s]
        summary[mode] = {
            "processes": len(procs),
            "failed_processes": failures[mode],
            "runs": len(runs),
            "run_metrics": {name: dict(pc.distribution(pc.collect(runs, get)), unit=unit)
                            for name, (get, unit, _) in RUN_METRICS.items()},
            "process_metrics": {name: dict(pc.distribution(pc.collect(procs, get)), unit=unit)
                                for name, (get, unit, _) in PROCESS_METRICS.items()},
            "callback_gaps_pooled_ms": pc.distribution(gaps),
            "finish_reasons": sorted({str(r.get("finish_reason")) for r in runs}),
            "telemetry": {
                "shutdown_results": sorted({str((p["driver"].get("telemetry") or {}).get("shutdown_result")) for p in procs}),
                "traces_exported": sum(int(t.get("exported") or 0) for t in traces_after),
                "traces_failed": sum(int(t.get("failed") or 0) for t in traces_after),
                "traces_dropped": sum(int(t.get("dropped") or 0) for t in traces_after),
                "traces_accepted": sum(int(t.get("accepted") or 0) for t in traces_after),
            },
        }

    comparisons: Dict[str, Any] = {}
    baseline = "off" if "off" in modes else None
    if baseline:
        base_runs = _runs(by_mode[baseline])
        for mode in modes:
            if mode == baseline:
                continue
            runs = _runs(by_mode[mode])
            per: Dict[str, Any] = {}
            for name, (get, unit, higher) in RUN_METRICS.items():
                delta = pc.bootstrap_median_delta(pc.collect(base_runs, get), pc.collect(runs, get))
                if delta is not None:
                    delta.update(unit=unit, higher_is_better=higher)
                per[name] = delta
            for name, (get, unit, higher) in PROCESS_METRICS.items():
                delta = pc.bootstrap_median_delta(pc.collect(by_mode[baseline], get), pc.collect(by_mode[mode], get))
                if delta is not None:
                    delta.update(unit=unit, higher_is_better=higher)
                per[name] = delta
            comparisons[mode] = per

    return {
        "modes": summary,
        "comparisons": comparisons,
        "sanity": sanity_checks(raw, summary),
        "thresholds": evaluate_thresholds(thresholds, comparisons, summary, raw),
    }


def sanity_checks(raw: Dict[str, Any], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Checks that each mode measured what it claims (not performance gates)."""
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: Optional[bool], detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    for mode, s in summary.items():
        check(f"{mode}: every process succeeded", s["failed_processes"] == 0,
              f"{s['failed_processes']} failed of {s['processes'] + s['failed_processes']}")
    callbacks = {m: s["run_metrics"]["callbacks"] for m, s in summary.items() if s["runs"]}
    counts = {value for c in callbacks.values() for value in (c.get("min"), c.get("max"))}
    check("greedy decoding produced the same callback count in every run and mode",
          None if not callbacks else len(counts) == 1 and None not in counts,
          json.dumps({m: [c.get("min"), c.get("max")] for m, c in callbacks.items()}))
    sinks = raw.get("sinks", {})
    if "metadata" in sinks and "content" in sinks:
        meta, content = sinks["metadata"], sinks["content"]
        check("content mode exported more bytes per trace request than metadata mode",
              _per_request(content, "traces") > _per_request(meta, "traces"),
              f"content {_per_request(content, 'traces'):.0f} B/request, metadata {_per_request(meta, 'traces'):.0f} B/request")
    for mode in SINK_MODES:
        if mode in summary and mode in sinks:
            check(f"{mode}: the sink received traces", (sinks[mode]["signals"].get("traces") or {}).get("requests", 0) > 0,
                  json.dumps(sinks[mode]["signals"]))
    if "blocked" in summary:
        t = summary["blocked"]["telemetry"]
        check("blocked: nothing was acknowledged", t["traces_exported"] == 0,
              f"exported {t['traces_exported']}, failed {t['traces_failed']}, dropped {t['traces_dropped']}")
        blocked = raw.get("blocked_endpoint") or {}
        check("blocked: the endpoint received connections", (blocked.get("connections") or 0) > 0, json.dumps(blocked))
    if "off" in summary:
        check("off: no pipeline counters", summary["off"]["telemetry"]["traces_accepted"] == 0,
              f"accepted {summary['off']['telemetry']['traces_accepted']}")
    return checks


def _per_request(sink: Dict[str, Any], signal_name: str) -> float:
    s = (sink.get("signals") or {}).get(signal_name) or {}
    return (s.get("bytes", 0) / s["requests"]) if s.get("requests") else 0.0


def evaluate_thresholds(thresholds: Dict[str, Any], comparisons: Dict[str, Any], summary: Dict[str, Any],
                        raw: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the e2e thresholds. Results are proposals unless
    ``thresholds.json`` says ``status: agreed``; then ``within`` (the median
    delta) is a gate (``perf_common.failed_gates``)."""
    out = {"status": thresholds.get("status", "proposed"), "source": "thresholds.json", "results": []}
    for item in thresholds.get("e2e", []):
        for mode in item["modes"]:
            comparison = (comparisons.get(mode) or {}).get(item["metric"])
            observed = worst = reference = None
            if comparison:
                # Regressions are positive: slower, later or larger. The worst
                # case is the CI end furthest in the regressing direction.
                if comparison.get("higher_is_better"):
                    observed, worst = -comparison["delta"], -comparison["ci95_low"]
                else:
                    observed, worst = comparison["delta"], comparison["ci95_high"]
                reference = comparison["baseline_median"]
            out["results"].append({
                "id": item["id"], "mode": mode, "metric": item["metric"], "bound": item["bound"],
                "observed_regression": observed, "worst_case_regression": worst, "baseline_median": reference,
                "observed_regression_pct": None if observed is None or not reference else observed / abs(reference) * 100.0,
                "ci95": None if not comparison else [comparison["ci95_low"], comparison["ci95_high"]],
                "within": pc.evaluate_bound(observed, item["bound"], reference),
                "within_at_ci_worst_case": pc.evaluate_bound(worst, item["bound"], reference),
                "rationale": item.get("rationale"),
            })
    for item in thresholds.get("e2e_absolute", []):
        for mode in item["modes"]:
            dist = ((summary.get(mode) or {}).get("process_metrics") or {}).get(item["metric"]) or {}
            observed = dist.get(item.get("statistic", "max"))
            limit = item["bound"].get("max_abs")
            if item.get("plus_shutdown_timeout") and limit is not None:
                limit = limit + raw["workload"]["shutdown_timeout_ms"]
            within = None if observed is None or limit is None else observed <= limit
            out["results"].append({
                "id": item["id"], "mode": mode, "metric": item["metric"], "statistic": item.get("statistic", "max"),
                "limit": limit, "observed": observed, "within": within, "rationale": item.get("rationale"),
            })
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def render_markdown(report: Dict[str, Any]) -> str:
    env = report["environment"]
    work = report["workload"]
    status = report["thresholds"]["status"]
    agreed = status == "agreed"
    lines = ["# Observability end-to-end overhead", ""]
    if agreed:
        lines.append(f"Generated {env.get('captured_at')} — thresholds are **agreed**: each row's `within` "
                     "is a gate (the median delta against `off`, or for an absolute row its statistic against "
                     "the limit), and a failed one makes `e2e` exit 1.")
    else:
        lines.append(f"Generated {env.get('captured_at')} — thresholds are **{status}**; "
                     "results below are measurements, not pass/fail verdicts.")
    lines += ["", "## Environment", ""]
    lines.append(pc.table(["field", "value"], [
        ("chip", env.get("chip")), ("model", env.get("model")),
        ("cores (P/E/logical)", f"{env.get('performance_cores')}/{env.get('efficiency_cores')}/{env.get('logical_cpus')}"),
        ("memory GiB", None if not env.get("memory_bytes") else env["memory_bytes"] / 2**30),
        ("OS", env.get("macos") or env.get("os_release") or env.get("platform")),
        ("power", env.get("power")), ("low power mode", env.get("low_power_mode")),
        ("rustc", (env.get("toolchain") or {}).get("rustc")),
        ("commit", (env.get("source") or {}).get("commit")), ("dirty", (env.get("source") or {}).get("dirty")),
    ]))
    model = work.get("model") or {}
    lines += ["", "## Workload", ""]
    lines.append(pc.table(["field", "value"], [
        ("model", f"{model.get('file')} ({'pinned ' + str(model.get('revision'))[:12] if model.get('pinned') else 'UNPINNED'})"),
        ("sha256", model.get("sha256")),
        ("modes", ", ".join(work["modes"])), ("rounds (processes per mode)", work["rounds"]),
        ("measured runs per process", work["runs_per_process"]), ("warmup runs per process", work["warmup"]),
        ("max tokens", work["max_tokens"]), ("prompt bytes", work["prompt_bytes"]),
        ("order", work["order"]), ("cooldown s", work["cooldown_s"]),
        ("driver", work.get("driver")), ("memory source", work.get("rss_source")),
    ]))
    lines += ["", "## Per mode (medians; [p5, p95] over runs or processes)", ""]
    rows = []
    for mode, s in report["analysis"]["modes"].items():
        rm, pm = s["run_metrics"], s["process_metrics"]

        def span(d: Dict[str, Any]) -> str:
            if not d.get("n"):
                return "–"
            return f"{pc.fmt(d['p50'])} [{pc.fmt(d['p5'])}, {pc.fmt(d['p95'])}]"
        rows.append((mode, f"{s['processes']}/{s['runs']}", span(rm["runtime_tokens_per_s"]),
                     span(rm["first_callback_ms"]), span(rm["callback_gap_p50_ms"]), span(rm["callback_gap_p99_ms"]),
                     span(pm["peak_rss_mib"]), span(pm["peak_footprint_mib"]), span(pm["shutdown_ms"])))
    lines.append(pc.table(["mode", "processes/runs", "tok/s (runtime)", "first callback ms", "gap p50 ms",
                           "gap p99 ms", "peak RSS MiB", "peak footprint MiB", "shutdown ms"], rows))
    lines += ["", "## Against `off` (median delta, bootstrap 95% CI)", ""]
    rows = []
    for mode, per in report["analysis"]["comparisons"].items():
        for metric in ("runtime_tokens_per_s", "callback_decode_per_s", "first_callback_ms", "callback_gap_p50_ms",
                       "callback_gap_p99_ms", "peak_rss_mib", "peak_footprint_mib"):
            c = per.get(metric)
            if not c:
                continue
            rows.append((mode, metric, c["baseline_median"], c["median"], c["delta"], c["delta_pct"],
                         f"[{pc.fmt(c['ci95_low'])}, {pc.fmt(c['ci95_high'])}]"))
    lines.append(pc.table(["mode", "metric", "off", "candidate", "delta", "delta %", "CI95 of delta"], rows))
    lines += ["", "## Sanity (does each mode measure what it claims)", ""]
    lines.append(pc.table(["check", "ok", "detail"],
                          [(c["check"], c["ok"], c["detail"]) for c in report["analysis"]["sanity"]]))
    lines += ["", f"## Thresholds ({report['thresholds']['status']})", ""]
    rows = []
    for r in report["thresholds"]["results"]:
        observed = r.get("observed_regression", r.get("observed"))
        bound = r["bound"] if "bound" in r else {"statistic": r.get("statistic"), "max_abs": r.get("limit")}
        rows.append((r["id"], r["mode"], r["metric"], json.dumps(bound),
                     observed, r.get("observed_regression_pct"), r.get("worst_case_regression"), r["within"],
                     r.get("within_at_ci_worst_case")))
    lines.append(pc.table(["id", "mode", "metric", "bound", "observed regression", "observed %",
                           "CI worst case", "within", "within at CI worst case"], rows))
    lines.append("")
    lines.append("Regressions are positive (slower, later, larger). `within` uses the median delta (an "
                 "absolute row, which has a `statistic` in its bound, compares that statistic over processes "
                 "with `max_abs`), `within at CI worst case` the end of the delta's bootstrap 95% interval in "
                 "the regressing direction"
                 + (" (for information; it does not gate)." if agreed else ". Agree the bounds (thresholds.json "
                    "`status`) before treating any row as a gate."))
    if agreed:
        failed = pc.failed_gates(report["thresholds"])
        unevaluated = sum(1 for r in report["thresholds"]["results"] if r.get("within") is None)
        verdict = (f"**Failed gates:** {', '.join(f'`{row}`' for row in failed)}." if failed
                   else "Every gate held.")
        if unevaluated:
            verdict += f" {unevaluated} row(s) had nothing to evaluate (`within` –) and do not gate."
        lines += ["", verdict]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, raw: Dict[str, Any], thresholds_path: Optional[Path] = None) -> Dict[str, Any]:
    thresholds = pc.load_thresholds(thresholds_path)
    analysis = analyze(raw, thresholds)
    report = {
        "schema": "xybrid-observability-perf/e2e/v1",
        "environment": raw["environment"],
        "workload": raw["workload"],
        "build": raw.get("build"),
        "sinks": raw.get("sinks"),
        "blocked_endpoint": raw.get("blocked_endpoint"),
        "analysis": {k: v for k, v in analysis.items() if k != "thresholds"},
        "thresholds": analysis["thresholds"],
    }
    pc.write_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    pc.stage(f"wrote {out_dir / 'report.json'} and report.md")
    return report


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fetch_model(args: argparse.Namespace) -> int:
    path = fetch_model(Path(args.dest))
    print(path)
    return 0


def cmd_e2e(args: argparse.Namespace) -> int:
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"error: unknown modes {unknown}; choose from {MODES}")
    out_dir = Path(args.out_dir) if args.out_dir else pc.timestamp_dir(pc.default_target_dir() / "observability-perf", "e2e")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    pc.stage(f"output: {out_dir}")

    environment = pc.environment()
    model_dir = Path(args.model_dir)
    model = verify_model(model_dir, args.allow_unpinned_model)

    build = None
    if args.driver_cmd:
        driver = shlex.split(args.driver_cmd)
        driver_desc = args.driver_cmd
    elif args.driver:
        driver = [str(Path(args.driver).resolve())]
        driver_desc = driver[0]
    else:
        executable, build = build_driver(args.cargo_features, args.profile)
        driver = [str(executable)]
        driver_desc = str(executable)

    sinks: Dict[str, Sink] = {}
    blocked: Optional[BlockedEndpoint] = None
    order = schedule(modes, args.rounds, args.shuffle_seed)
    workload = {
        "modes": modes, "rounds": args.rounds, "runs_per_process": args.runs_per_process, "warmup": args.warmup,
        "max_tokens": args.max_tokens, "prompt_bytes": args.prompt_bytes,
        "shutdown_timeout_ms": args.shutdown_timeout_ms, "cooldown_s": args.cooldown_s,
        "order": ("shuffled per round (seed %d)" % args.shuffle_seed if args.shuffle_seed is not None
                  else "Williams design (positions and within-round predecessors balanced)"),
        "schedule": [f"{r}:{m}" for r, m in order], "predecessors": predecessor_counts(order),
        "model": model, "driver": driver_desc,
        "cargo_features": None if (args.driver or args.driver_cmd) else args.cargo_features,
        "rss_source": "/usr/bin/time -l" if use_bsd_time() else "wait4 ru_maxrss",
    }
    processes: List[Dict[str, Any]] = []
    env = driver_env()
    try:
        for mode in modes:
            if mode in SINK_MODES:
                sinks[mode] = Sink(out_dir, mode)
        if "blocked" in modes:
            blocked = BlockedEndpoint()
        for index, (round_index, mode) in enumerate(order):
            endpoint = sinks[mode].endpoint if mode in sinks else (blocked.endpoint if mode == "blocked" and blocked else None)
            stem = f"r{round_index:02d}-{mode}"
            result_path = runs_dir / f"{stem}.json"
            result_path.unlink(missing_ok=True)
            command = driver + [
                "--model-dir", str(model_dir), "--mode", mode, "--out", str(result_path),
                "--runs", str(args.runs_per_process), "--warmup", str(args.warmup),
                "--max-tokens", str(args.max_tokens), "--prompt-bytes", str(args.prompt_bytes),
                "--shutdown-timeout-ms", str(args.shutdown_timeout_ms),
            ]
            if endpoint:
                command += ["--endpoint", endpoint]
            pc.stage(f"[{index + 1}/{len(order)}] round {round_index} mode {mode}")
            measure = run_measured(command, runs_dir / f"{stem}.stdout", runs_dir / f"{stem}.stderr",
                                   args.process_timeout_s, env)
            driver_result = None
            if result_path.exists():
                try:
                    driver_result = json.loads(result_path.read_text(encoding="utf-8"))
                except ValueError:
                    driver_result = None
            ok = measure["exit_code"] == 0 and not measure["timed_out"] and driver_result is not None
            if not ok:
                pc.stage(f"  failed: exit {measure['exit_code']}; see {runs_dir / (stem + '.stderr')}")
            processes.append({"round": round_index, "mode": mode, "position": index, "ok": ok,
                              "measure": measure, "driver": driver_result})
            if args.cooldown_s > 0 and index + 1 < len(order):
                time.sleep(args.cooldown_s)
    finally:
        for sink in sinks.values():
            sink.close()
        blocked_summary = blocked.summary() if blocked else None
        if blocked:
            blocked.close()

    raw = {
        "schema": "xybrid-observability-perf/e2e-raw/v1",
        "environment": environment,
        "workload": workload,
        "build": build,
        "sinks": {m: s.summary() for m, s in sinks.items()},
        "blocked_endpoint": blocked_summary,
        "processes": processes,
    }
    pc.write_json(out_dir / "raw.json", raw)
    report = write_report(out_dir, raw, Path(args.thresholds) if args.thresholds else None)
    failed = sum(1 for p in processes if not p["ok"])
    # A failed sanity check means a mode did not measure what it claims, so
    # its comparison with `off` is not evidence either way.
    sanity_failed = [c["check"] for c in report["analysis"]["sanity"] if c["ok"] is False]
    thresholds_failed = pc.failed_gates(report["thresholds"])
    print(json.dumps({"out_dir": str(out_dir), "processes": len(processes), "failed": failed,
                      "sanity_failed": sanity_failed, "thresholds_status": report["thresholds"]["status"],
                      "thresholds_failed": thresholds_failed}))
    return 1 if failed or sanity_failed or thresholds_failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.e2e)
    raw = json.loads((out_dir / "raw.json").read_text(encoding="utf-8"))
    write_report(out_dir, raw, Path(args.thresholds) if args.thresholds else None)
    if args.micro:
        micro.write_report(Path(args.micro), e2e_report=out_dir / "report.json",
                           thresholds_path=Path(args.thresholds) if args.thresholds else None)
    return 0


def cmd_micro(args: argparse.Namespace) -> int:
    return micro.main_from_args(args)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-model", help="download the pinned GGUF and write model_metadata.json")
    fetch.add_argument("--dest", default=str(DEFAULT_MODEL_DIR))
    fetch.set_defaults(func=cmd_fetch_model)

    e2e = sub.add_parser("e2e", help="real-model runs per telemetry mode, interleaved")
    e2e.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    e2e.add_argument("--modes", default=",".join(MODES))
    e2e.add_argument("--rounds", type=int, default=10, help="processes per mode (default 10)")
    e2e.add_argument("--runs-per-process", type=int, default=3)
    e2e.add_argument("--warmup", type=int, default=1)
    e2e.add_argument("--max-tokens", type=int, default=128)
    e2e.add_argument("--prompt-bytes", type=int, default=2048)
    e2e.add_argument("--shutdown-timeout-ms", type=int, default=2000)
    e2e.add_argument("--cooldown-s", type=float, default=2.0)
    e2e.add_argument("--process-timeout-s", type=float, default=600.0)
    e2e.add_argument("--shuffle-seed", type=int, default=None, help="shuffle each round instead of rotating")
    e2e.add_argument("--cargo-features", default="llm-llamacpp", help="features for the driver build")
    e2e.add_argument("--profile", default="release", help="cargo profile for the driver build")
    e2e.add_argument("--driver", help="a prebuilt observability_perf executable (skips the build)")
    e2e.add_argument("--driver-cmd", help="a command line standing in for the driver (dry runs, tests)")
    e2e.add_argument("--allow-unpinned-model", action="store_true")
    e2e.add_argument("--thresholds", help="thresholds file (default thresholds.json)")
    e2e.add_argument("--out-dir")
    e2e.set_defaults(func=cmd_e2e)

    report = sub.add_parser("report", help="re-render reports from raw results")
    report.add_argument("--e2e", required=True, help="an e2e output directory (holds raw.json)")
    report.add_argument("--micro", help="a micro output directory, re-rendered against this e2e report")
    report.add_argument("--thresholds")
    report.set_defaults(func=cmd_report)

    m = sub.add_parser("micro", help="run the Rust and JS micro benchmarks and collect their results")
    micro.add_arguments(m)
    m.set_defaults(func=cmd_micro)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
