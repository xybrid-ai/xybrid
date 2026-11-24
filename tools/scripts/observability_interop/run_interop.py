#!/usr/bin/env python3
"""Runs real producers through the pinned Collector into a destination, then
reads the destination back and checks it against what the Collector received.

    python3 run_interop.py --destination phoenix          # needs ./up.sh phoenix
    python3 run_interop.py --destination langfuse         # needs ./up.sh langfuse
    XYBRID_STAGING_API_KEY=... python3 run_interop.py --destination xybrid
    python3 run_interop.py --destination langfuse --wire  # no backend: checks the profile's output
    python3 run_interop.py --validate-profiles            # otelcol validate, every combination

Per destination it
  1. fetches the pinned otelcol-contrib (checksum-verified, cached),
  2. starts it with the destination profile, sinks.yaml pointed at a local
     capture server, and collector-harness.yaml, which copies everything
     received (before any mapping) to another capture server: the ground truth,
  3. runs the workloads (workloads.py) against the Collector's OTLP endpoint,
     recording the references each reports,
  4. reads the destination's API (destinations.py) until every span the
     Collector received is there, and the quality results where it keeps them,
  5. runs the table-driven checks (expectations.py) and writes ``report.json``
     and ``report.md`` naming the producer commit, Collector and backend versions.

Exit status: 0 when every check passed, 1 when one failed, 2 when the run
could not start (missing prerequisites, unreachable backend, refused target).
Never contacts a Xybrid host other than a staging or loopback one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "observability_capture"))

import check_capture  # noqa: E402
import destinations  # noqa: E402
import expectations  # noqa: E402
import otelcol  # noqa: E402
import workloads  # noqa: E402

REPO = HERE.parents[2]
CAPTURE_SERVER = HERE.parent / "observability_capture" / "capture_server.py"
SETTINGS_PATH = HERE / "interop.env"
SCHEMA = "xybrid-observability-interop/1"
ALL_DESTINATIONS = ("langfuse", "phoenix", "xybrid")
PRODUCTION_XYBRID_HOSTS = ("api.xybrid.dev", "ingest.xybrid.dev", "xybrid.dev")


class SetupError(RuntimeError):
    """The run cannot start; the message says what to do."""


# ---------------------------------------------------------------------------
# Settings and guards
# ---------------------------------------------------------------------------


def parse_env_file(text: str) -> Dict[str, str]:
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def load_settings(path: Path = SETTINGS_PATH, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """interop.env, with any of its keys overridden from the environment."""
    environ = os.environ if environ is None else environ
    settings = parse_env_file(path.read_text(encoding="utf-8"))
    for key in list(settings):
        if environ.get(key):
            settings[key] = environ[key]
    return settings


def check_xybrid_target(url: str) -> str:
    """Returns ``url`` when it is a staging or loopback host; raises otherwise.

    The harness must never write to or read from production Xybrid.
    """
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise SetupError(f"{url!r} is not an http(s) URL")
    if host in ("127.0.0.1", "localhost", "::1"):
        return url.rstrip("/")
    if host.startswith("staging.") and host.endswith(".xybrid.dev"):
        return url.rstrip("/")
    if host in PRODUCTION_XYBRID_HOSTS or host.endswith(".xybrid.dev"):
        raise SetupError(f"refusing {url}: the interop harness only targets Xybrid staging "
                         f"(staging.*.xybrid.dev) or a loopback host")
    raise SetupError(f"refusing {url}: not a known Xybrid staging host")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def command_output(argv: Sequence[str], cwd: Path = REPO, timeout: float = 60.0) -> Optional[str]:
    try:
        completed = subprocess.run(list(argv), cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def producer_provenance(tools: workloads.Toolchain) -> Dict[str, Any]:
    # Tracked changes make the producers differ from the commit; untracked
    # files (build leftovers, scratch directories) are listed, not counted.
    status = command_output(["git", "status", "--porcelain", "--untracked-files=no"]) or ""
    untracked = command_output(["git", "ls-files", "--others", "--exclude-standard", "--directory"]) or ""
    cargo_toml = (REPO / "Cargo.toml").read_text(encoding="utf-8")
    workspace_version = re.search(r'\[workspace\.package\][^\[]*?\nversion\s*=\s*"([^"]+)"', cargo_toml, re.S)
    js_package = json.loads((workloads.JS_DIR / "package.json").read_text(encoding="utf-8"))
    return {
        "git_sha": command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(status),
        "git_modified": [line[3:] for line in status.splitlines()][:50],
        "git_untracked": untracked.splitlines()[:50],
        "xybrid_observability_crate": workspace_version.group(1) if workspace_version else None,
        "observability_js_package": js_package.get("version"),
        "runtimes": {
            "rustc": command_output(["rustc", "--version"]),
            "node": command_output([tools.node or "node", "--version"]),
            "bun": command_output(["bun", "--version"]) if tools.js_runtime == "bun" else None,
            "python_sdk_interpreter": command_output([tools.python, "--version"]),
            "harness_python": platform.python_version(),
        },
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    }


def container_images(names: Sequence[str]) -> Dict[str, str]:
    """Image reference and image id of each running harness container (best effort)."""
    if shutil.which("docker") is None:
        return {}
    images = {}
    for name in names:
        value = command_output(["docker", "inspect", "-f", "{{.Config.Image}} {{.Image}}", name], timeout=20)
        if value:
            images[name] = value
    return images


CONTAINERS = {
    "langfuse": ["xybrid-interop-langfuse-web", "xybrid-interop-langfuse-worker", "xybrid-interop-clickhouse",
                 "xybrid-interop-postgres", "xybrid-interop-redis", "xybrid-interop-minio"],
    "phoenix": ["xybrid-interop-phoenix"],
    "xybrid": [],
}


# ---------------------------------------------------------------------------
# Capture servers
# ---------------------------------------------------------------------------


class CaptureServer:
    """``capture_server.py`` on a free loopback port."""

    def __init__(self, directory: Path, name: str) -> None:
        self.out = directory / f"{name}.jsonl"
        self.ready = directory / f"{name}.ready.json"
        self.log = directory / f"{name}.log"
        self.process: Optional[subprocess.Popen] = None
        self.endpoint = ""

    def start(self, timeout: float = 20.0) -> "CaptureServer":
        if self.ready.exists():
            self.ready.unlink()
        with self.log.open("w", encoding="utf-8") as handle:  # the child keeps its own copy
            self.process = subprocess.Popen(
                [sys.executable, "-I", str(CAPTURE_SERVER), "--host", "127.0.0.1", "--port", "0",
                 "--out", str(self.out), "--ready-file", str(self.ready)], stdout=handle, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready.exists():
                self.endpoint = json.loads(self.ready.read_text(encoding="utf-8"))["endpoint"]
                return self
            if self.process.poll() is not None:
                break
            time.sleep(0.05)
        # A server that is still starting past the deadline would outlive the run.
        self.stop()
        raise SetupError(f"capture server {self.out.name} did not start; see {self.log}")

    def lines(self) -> int:
        if not self.out.exists():
            return 0
        with self.out.open("rb") as handle:
            return sum(1 for _ in handle)

    def entries(self) -> List[Dict[str, Any]]:
        if not self.out.exists():
            return []
        return [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines() if line.strip()]

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def settle(servers: Sequence[CaptureServer], quiet: float = 1.5, limit: float = 20.0) -> None:
    """Waits until no capture server has received anything for ``quiet`` seconds."""
    deadline = time.monotonic() + limit
    last = [server.lines() for server in servers]
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(0.25)
        now = [server.lines() for server in servers]
        if now != last:
            last, quiet_since = now, time.monotonic()
        elif time.monotonic() - quiet_since >= quiet:
            return


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


# Credentials a producer never needs: the workloads export to the harness's
# Collector only, so a hosted key in the caller's shell must not reach them.
PRODUCER_STRIPPED_ENV = ("XYBRID_API_KEY", "XYBRID_STAGING_API_KEY", "XYBRID_PLATFORM_API_KEY")


def producer_env(step_env: Mapping[str, str], base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    base = os.environ if base is None else base
    env = {key: value for key, value in base.items()
           if not key.startswith("OTEL_") and key not in PRODUCER_STRIPPED_ENV}
    env.update(step_env)
    return env


def kill_group(process: "subprocess.Popen[Any]") -> None:
    """Kills ``process`` and everything it started (its own session)."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    process.kill()


def run_step(step: workloads.Step, log_path: Path, timeout: float) -> Tuple[int, str]:
    """Runs one workload step. The step gets its own session, so a timeout
    (or Ctrl-C) kills what it started too: ``cargo run`` leaves the example
    running when only cargo is killed."""
    env = producer_env(step.env)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(step.argv)}  (cwd {step.cwd})\n")
        log.flush()
        try:
            process = subprocess.Popen(step.argv, cwd=str(step.cwd), env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=os.name == "posix")
        except OSError as error:
            log.write(f"cannot start: {error}\n")
            return 127, ""
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_group(process)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:  # something outside the group holds the pipes
                stdout, stderr = "", ""
            log.write(stdout or "")
            log.write(stderr or "")
            log.write(f"timed out after {timeout:.0f}s\n")
            return 124, stdout or ""
        except BaseException:
            kill_group(process)
            process.wait()
            raise
        log.write(stdout)
        log.write(stderr)
    return process.returncode, stdout


def assign_workloads(entries: Sequence[Dict[str, Any]], selected: Sequence[str],
                     windows: Mapping[str, Tuple[int, int]]) -> Dict[str, List[Dict[str, Any]]]:
    """Splits the ground-truth capture per workload.

    A resource is assigned by its ``xybrid.interop.workload`` attribute, else
    by a service name only one selected workload uses, else by the time window
    in which the workload ran (they run one after another, with the capture
    settled in between).
    """
    services: Dict[str, List[str]] = {}
    for name in selected:
        services.setdefault(workloads.WORKLOADS[name].service_name, []).append(name)

    def owner(resource: Mapping[str, Any], received: int) -> Optional[str]:
        named = workloads.split_key(resource)
        if named in selected:
            return named
        candidates = services.get(str(resource.get("service.name")), [])
        if len(candidates) == 1:
            return candidates[0]
        for name in candidates or selected:
            start, end = windows.get(name, (0, -1))
            if start <= received <= end:
                return name
        return None

    out: Dict[str, List[Dict[str, Any]]] = {name: [] for name in selected}
    for entry in entries:
        received = int(entry.get("received_unix_nano") or 0)
        for name in selected:
            kept = expectations.filter_entries([entry], lambda resource, n=name: owner(resource, received) == n)
            out[name].extend(kept)
    return out


# ---------------------------------------------------------------------------
# Destinations
# ---------------------------------------------------------------------------


class RawLog:
    """Every API response, kept for the report and written to raw/."""

    def __init__(self) -> None:
        self.items: List[Tuple[str, str, Any]] = []

    def __call__(self, label: str, url: str, value: Any) -> None:
        self.items.append((label, url, value))

    def answers(self, labels: Collection[str]) -> List[Any]:
        """Every recorded answer (each poll's) whose label is one of ``labels``."""
        return [value for label, _, value in self.items if label in labels]

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        latest: Dict[str, Tuple[str, Any]] = {}
        for label, url, value in self.items:
            latest[label] = (url, value)
        for label, (url, value) in latest.items():
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:120]
            (directory / f"{safe}.json").write_text(json.dumps({"url": url, "response": value}, indent=2) + "\n",
                                                    encoding="utf-8")


def own_answers_text(truth: expectations.GroundTruth, spans: Sequence[destinations.DestSpan],
                     evaluations: Optional[Sequence[destinations.DestEvaluation]],
                     raw: Optional[RawLog] = None) -> str:
    """What the destination answered for this run's spans, as the text the
    ``content`` and ``content.redaction`` checks search.

    Only spans with a ground-truth id count, their quality results, and (from
    ``raw``) the Xybrid content route's answer for each of them. A trace
    holds every run's spans when its id is fixed (the Python scenario's), so
    the whole trace's answers would fail a clean run on a secret an earlier
    run stored, and pass a broken one on a replacement an earlier run stored.
    """
    ids = truth.span_ids()
    traces = set(truth.trace_ids())
    own = [span for span in spans if span.span_id in ids and span.trace_id in traces]
    parts = [json.dumps(span.raw, sort_keys=True) for span in own]
    if raw is not None:
        parts += [json.dumps(value, sort_keys=True)
                  for value in raw.answers({f"content-{span.span_id}" for span in own})]
    parts += [json.dumps(evaluation.raw, sort_keys=True) for evaluation in evaluations or []
              if evaluation.span_id in ids]
    return "\n".join(parts)


def make_client(destination: str, settings: Mapping[str, str], raw: RawLog,
                environ: Mapping[str, str]) -> Any:
    if destination == "langfuse":
        return destinations.Langfuse(f"http://127.0.0.1:{settings['LANGFUSE_HOST_PORT']}",
                                     settings["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"],
                                     settings["LANGFUSE_INIT_PROJECT_SECRET_KEY"], raw)
    if destination == "phoenix":
        return destinations.Phoenix(f"http://127.0.0.1:{settings['PHOENIX_HOST_PORT']}",
                                    settings.get("PHOENIX_PROJECT", "default"), raw)
    key = environ.get("XYBRID_STAGING_API_KEY")
    if not key:
        raise SetupError("XYBRID_STAGING_API_KEY is not set: the xybrid destination needs a staging key")
    return destinations.Xybrid(check_xybrid_target(settings["XYBRID_STAGING_API_URL"]), key, raw,
                               ingest_url=check_xybrid_target(settings["XYBRID_STAGING_INGEST_URL"]))


def destination_env(destination: str, settings: Mapping[str, str], environ: Mapping[str, str],
                    wire_endpoint: Optional[str]) -> Dict[str, str]:
    """The profile's variables, pointing at the backend or, with --wire, at
    the harness's capture server."""
    import base64

    if destination == "langfuse":
        auth = base64.b64encode(f"{settings['LANGFUSE_INIT_PROJECT_PUBLIC_KEY']}:"
                                f"{settings['LANGFUSE_INIT_PROJECT_SECRET_KEY']}".encode()).decode()
        endpoint = wire_endpoint or f"http://127.0.0.1:{settings['LANGFUSE_HOST_PORT']}/api/public/otel"
        return {"LANGFUSE_OTLP_ENDPOINT": endpoint, "LANGFUSE_AUTH": auth}
    if destination == "phoenix":
        return {"PHOENIX_OTLP_ENDPOINT": wire_endpoint or f"http://127.0.0.1:{settings['PHOENIX_HOST_PORT']}"}
    if wire_endpoint:
        return {"XYBRID_API_KEY": "wire-mode-not-a-key", "XYBRID_TRACES_ENDPOINT": f"{wire_endpoint}/v1/traces",
                "XYBRID_LOGS_ENDPOINT": f"{wire_endpoint}/v1/logs"}
    key = environ.get("XYBRID_STAGING_API_KEY")
    if not key:
        raise SetupError("XYBRID_STAGING_API_KEY is not set: the xybrid destination needs a staging key")
    ingest = check_xybrid_target(settings["XYBRID_STAGING_INGEST_URL"])
    return {"XYBRID_API_KEY": key, "XYBRID_TRACES_ENDPOINT": f"{ingest}/v1/traces",
            "XYBRID_LOGS_ENDPOINT": f"{ingest}/v1/logs"}


def poll(fetch: Callable[[], Any], done: Callable[[Any], bool], timeout: float, interval: float = 3.0) -> Any:
    deadline = time.monotonic() + timeout
    value = fetch()
    while not done(value) and time.monotonic() < deadline:
        time.sleep(interval)
        value = fetch()
    return value


@dataclass
class Readback:
    """What a destination answered for one workload.

    ``errors`` are reads whose last attempt failed (an earlier failure that a
    later poll recovered from is only a note). ``evaluations`` is ``None``
    when any quality-result read failed, so the ``evaluations`` check reports
    "not read" instead of passing a "dropped" expectation on an empty answer.
    """

    spans: List[destinations.DestSpan] = field(default_factory=list)
    evaluations: Optional[List[destinations.DestEvaluation]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def read_back(client: Any, truth: expectations.GroundTruth, table: Mapping[str, Any], started: datetime,
              timeout: float) -> Readback:
    """Spans of every trace the workload touched, polled until all are there,
    then its quality results (polled where the table says they are kept)."""
    result = Readback()
    evaluations: List[destinations.DestEvaluation] = []
    evaluations_failed = False

    def attempt(read: Callable[[], List[Any]], label: str, last_error: Dict[str, str]) -> List[Any]:
        try:
            value = read()
        except destinations.HttpError as error:
            last_error["error"] = f"{label}: {error}"
            result.notes.append(last_error["error"])
            return []
        last_error.pop("error", None)
        return value

    for trace_id in truth.trace_ids():
        want = {span.span_id for span in truth.spans if span.trace_id == trace_id}
        span_error: Dict[str, str] = {}

        def fetch(trace_id: str = trace_id, span_error: Dict[str, str] = span_error) -> List[Any]:
            return attempt(lambda: client.spans(trace_id, (iso(started - timedelta(minutes=30)),
                                                           iso(utc_now() + timedelta(minutes=30)))),
                           f"{trace_id} spans", span_error)

        result.spans.extend(poll(fetch, lambda got, want=want: want <= {span.span_id for span in got}, timeout))
        if span_error:
            result.errors.append(span_error["error"])
    for trace_id in truth.trace_ids():
        ids = [span.span_id for span in truth.spans if span.trace_id == trace_id]
        expected_here = len([log for log in truth.evaluations() if log.trace_id == trace_id])
        evaluation_error: Dict[str, str] = {}

        def fetch_evaluations(trace_id: str = trace_id, ids: Sequence[str] = ids,
                              evaluation_error: Dict[str, str] = evaluation_error) -> List[Any]:
            return attempt(lambda: client.evaluations(trace_id, ids), f"{trace_id} evaluations", evaluation_error)

        if table["evaluations"] == "linked" and expected_here:
            got = poll(fetch_evaluations,
                       lambda value, ids=ids, n=expected_here: len([e for e in value if e.span_id in ids]) >= n,
                       timeout)
        else:
            got = fetch_evaluations()
        if evaluation_error:
            result.errors.append(evaluation_error["error"])
            evaluations_failed = True
        evaluations.extend(got)
    result.evaluations = None if evaluations_failed else evaluations
    return result


def destination_readback(destination: str, truth: expectations.GroundTruth, table: Mapping[str, Any],
                         client: Any, raw: RawLog, started: datetime, timeout: float,
                         wire_entries: Optional[Sequence[Dict[str, Any]]] = None,
                         ) -> Tuple[List[expectations.Check], Optional[Dict[str, Any]]]:
    """Reads one workload back and checks it against its ground truth.

    With ``wire_entries`` (``--wire``) the spans come from the wire capture,
    which holds only this run's requests; otherwise ``client`` reads the
    destination's API, and the text the content checks search is built from
    this run's own spans only (``own_answers_text``). Returns the checks and
    the report's ``destination`` entry, ``None`` when there was nothing to read.
    """
    name = truth.workload.name
    check = expectations.Check("destination", name, "readback")
    if not truth.spans:
        check.failures.append("nothing to read back: the workload exported no spans")
        return [check], None
    if wire_entries is not None:
        spans = [span for span in destinations.wire_spans(destination, wire_entries)
                 if span.trace_id in truth.trace_ids()]
        evaluations = destinations.wire_evaluations(wire_entries) if destination == "xybrid" else []
        raw_text = "\n".join([json.dumps(span.raw, sort_keys=True) for span in spans] + [
            json.dumps(evaluation.raw, sort_keys=True) for evaluation in evaluations
            if evaluation.trace_id in truth.trace_ids()])
        notes: List[str] = []
        errors: List[str] = []
    else:
        print(f"==> reading {name} back from {destination}", flush=True)
        readback = read_back(client, truth, table, started, timeout)
        spans, evaluations = readback.spans, readback.evaluations
        notes, errors = readback.notes, readback.errors
        raw_text = own_answers_text(truth, spans, evaluations, raw)
    check.failures.extend(f"read failed: {error}" for error in errors)
    check.notes.extend(notes)
    checks = [check] + expectations.destination_checks(destination, truth, spans, evaluations, raw_text, table)
    return checks, {"spans_read": len(spans), "evaluations_read": None if evaluations is None else len(evaluations),
                    "read_errors": errors}


# ---------------------------------------------------------------------------
# One destination
# ---------------------------------------------------------------------------


def run_destination(destination: str, args: argparse.Namespace, settings: Mapping[str, str],
                    tools: workloads.Toolchain) -> Dict[str, Any]:
    out = args.out_dir / (f"{destination}-wire" if args.wire else destination)
    out.mkdir(parents=True, exist_ok=True)
    table = expectations.DESTINATIONS[destination]
    raw = RawLog()
    selected = list(args.workloads)
    started = utc_now()
    report: Dict[str, Any] = {
        "schema": SCHEMA, "destination": destination, "mode": "wire" if args.wire else "api",
        "started_at": iso(started), "out_dir": str(out), "workloads": {}, "checks": [],
        "expectations": {key: value for key, value in table.items()},
    }
    checks: List[expectations.Check] = []
    print(f"==> {destination}{' (wire)' if args.wire else ''}: output in {out}", flush=True)

    binary = otelcol.ensure_collector(cache_dir=args.cache_dir, explicit=args.collector_binary)
    report["collector"] = binary.describe()
    report["producers"] = producer_provenance(tools)

    client = None
    if not args.wire:
        client = make_client(destination, settings, raw, os.environ)
        try:
            report["destination_info"] = client.version()
        except (destinations.HttpError, OSError, ValueError) as error:
            raise SetupError(f"{destination} is not reachable ({error}); start it with "
                             f"tools/scripts/observability_interop/up.sh {destination}") from None
        report["destination_info"]["containers"] = container_images(CONTAINERS[destination])
    else:
        report["destination_info"] = {"product": f"{table['label']} wire format (no backend)",
                                      "version": None}

    problems = {name: workloads.preflight(workloads.WORKLOADS[name], tools) for name in selected}
    problems = {name: found for name, found in problems.items() if found}
    if problems:
        raise SetupError("workloads not ready:\n" + "\n".join(
            f"  {name}: {problem}" for name, found in problems.items() for problem in found))

    tee = CaptureServer(out, "ground-truth")
    sink = CaptureServer(out, "sinks") if args.sinks else None
    wire = CaptureServer(out, "wire") if args.wire else None
    servers = [server for server in (tee, sink, wire) if server is not None]
    collector = None
    exit_codes: Dict[str, List[int]] = {}
    refs: Dict[str, Dict[str, Dict[str, str]]] = {}
    windows: Dict[str, Tuple[int, int]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    try:
        # Started inside the try, so a server that fails to start still has
        # the ones before it stopped by the finally below.
        for server in servers:
            server.start()
        port = int(args.collector_port or settings["INTEROP_COLLECTOR_PORT"])
        metrics_port = int(settings["INTEROP_COLLECTOR_METRICS_PORT"])
        env = otelcol.scrubbed_environment()
        env.update({"INTEROP_COLLECTOR_PORT": str(port), "INTEROP_COLLECTOR_METRICS_PORT": str(metrics_port),
                    "INTEROP_CAPTURE_ENDPOINT": tee.endpoint})
        env.update(destination_env(destination, settings, os.environ, wire.endpoint if wire else None))
        if sink is not None:
            env.update({"METRICS_OTLP_ENDPOINT": sink.endpoint, "LOGS_OTLP_ENDPOINT": sink.endpoint})
        extra = []
        if wire is not None:
            overlay = out / "collector-wire-json.yaml"
            overlay.write_text(otelcol.wire_overlay(destination), encoding="utf-8")
            extra.append(overlay)
        configs = otelcol.run_configs(destination, args.sinks, extra)
        report["collector"]["configs"] = [str(path.relative_to(REPO)) if REPO in path.parents else str(path)
                                          for path in configs]
        ok, output = otelcol.validate(binary.path, configs, env)
        check = expectations.Check("collector", None, "collector.config_valid")
        if not ok:
            check.failures.append(output[-1500:])
            checks.append(check)
            raise SetupError(f"otelcol-contrib validate failed:\n{output[-1500:]}")
        checks.append(check)
        collector = otelcol.RunningCollector(binary.path, configs, env, out / "collector.log")
        collector.start(port)
        endpoint = f"http://127.0.0.1:{port}"
        print(f"==> collector {binary.version} on {endpoint}", flush=True)

        for name in selected:
            workload = workloads.WORKLOADS[name]
            directory = out / name
            directory.mkdir(exist_ok=True)
            begin = time.time_ns()
            stdout_all = ""
            exit_codes[name] = []
            for index, step in enumerate(workloads.steps_for(workload, endpoint, directory, tools)):
                print(f"==> {name}: {step.label}", flush=True)
                code, stdout = run_step(step, directory / f"step{index + 1}.log", args.workload_timeout)
                exit_codes[name].append(code)
                stdout_all += stdout
                if code != 0:
                    print(f"    exit {code}; see {directory / f'step{index + 1}.log'}", flush=True)
                    break
            settle(servers)
            windows[name] = (begin, time.time_ns())
            refs[name] = workloads.reported_refs(workload, stdout_all, directory)
            report["workloads"][name] = {
                "program": workload.program, "language": workload.language, "content": workload.content,
                "notes": workload.notes, "exit_codes": exit_codes[name], "reported_refs": refs[name],
                "duration_s": round((windows[name][1] - begin) / 1e9, 2),
            }
        try:
            metrics, text = otelcol.scrape_metrics(metrics_port)
            (out / "collector-metrics.txt").write_text(text, encoding="utf-8")
        except OSError as error:
            report["collector"]["metrics_error"] = str(error)
    finally:
        if collector is not None:
            report["collector"]["exit_status"] = collector.stop()
        settle(servers, quiet=1.0, limit=5.0)
        for server in servers:
            server.stop()

    report["collector"]["exporters"] = metrics
    log_text = (out / "collector.log").read_text(encoding="utf-8", errors="replace") \
        if (out / "collector.log").exists() else ""

    # Ground truth, per workload.
    entries = tee.entries()
    split = assign_workloads(entries, selected, windows)
    truths = {name: expectations.GroundTruth.build(workloads.WORKLOADS[name], split[name], refs.get(name))
              for name in selected}
    sink_split = assign_workloads(sink.entries(), selected, windows) if sink is not None else {}

    for name in selected:
        truth = truths[name]
        report["workloads"][name]["ground_truth"] = {
            "trace_ids": truth.trace_ids(), "spans": len(truth.spans), "log_records": len(truth.capture.logs),
            "evaluation_results": len(truth.evaluations()), "metric_points": truth.capture.metric_points}
        checks += expectations.collector_checks(truth, exit_codes.get(name, []))
        if sink is not None:
            checks += expectations.sink_checks(
                truth, expectations.GroundTruth.build(workloads.WORKLOADS[name], sink_split[name]), destination)
        if name == "python":
            checks += scenario_checks(truth, out / "python" / "expectations.json")
        (out / name / "ground-truth.jsonl").write_text(
            "".join(json.dumps(entry) + "\n" for entry in split[name]), encoding="utf-8")

    checks += collector_run_checks(destination, binary, truths, metrics, log_text, wire)

    # Destination readback.
    for name in selected:
        found, read = destination_readback(destination, truths[name], table, client, raw, started,
                                           args.readback_timeout, wire.entries() if wire is not None else None)
        checks += found
        if read is not None:
            report["workloads"][name]["destination"] = read

    if not args.wire:
        raw.write(out / "raw")
    report["checks"] = [check.as_dict() for check in checks]
    report["summary"] = expectations.summarize(checks)
    report["ok"] = report["summary"]["fail"] == 0
    report["finished_at"] = iso(utc_now())
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print_summary(report, checks)
    return report


def scenario_checks(truth: expectations.GroundTruth, expectations_path: Path) -> List[expectations.Check]:
    """The shared runtime scenario's own checks, on what crossed the Collector."""
    if not expectations_path.is_file():
        check = expectations.Check("collector", "python", "scenario")
        check.failures.append(f"the scenario wrote no {expectations_path}")
        return [check]
    expected = json.loads(expectations_path.read_text(encoding="utf-8"))
    checker = check_capture.Checker(truth.capture, expected, check_capture.load_spec(), "python")
    out = []
    for result in checker.run():
        check = expectations.Check("collector", "python", f"scenario.{result.name}")
        check.failures, check.notes = list(result.failures), list(result.notes)
        out.append(check)
    return out


def collector_run_checks(destination: str, binary: otelcol.CollectorBinary,
                         truths: Mapping[str, expectations.GroundTruth], metrics: Mapping[str, Mapping[str, float]],
                         log_text: str, wire: Optional[CaptureServer]) -> List[expectations.Check]:
    checks = []
    check = expectations.Check("collector", None, "collector.version")
    pinned = str(otelcol.load_lock()["version"])
    if binary.version != pinned:
        check.failures.append(f"ran {binary.version}, the harness pins {pinned}")
    check.notes.append(f"otelcol-contrib {binary.version} {binary.platform} ({binary.source}), "
                       f"archive sha256 {binary.archive_sha256 or 'unknown (--collector-binary)'}")
    checks.append(check)

    check = expectations.Check("collector", None, "collector.export")
    exporter = metrics.get(f"exporter:{otelcol.DESTINATION_EXPORTER[destination]}")
    spans = sum(len(truth.spans) for truth in truths.values())
    evaluations = sum(len(truth.evaluations()) for truth in truths.values())
    if exporter is None:
        if destination == "xybrid" and wire is None:
            # The only proof (in API mode) that nothing but evaluation results
            # went to Xybrid's /v1/logs is the exporter's sent_log_records.
            check.failures.append(f"the Collector's own metrics were not readable, so nothing shows that only the "
                                  f"{evaluations} evaluation results went to Xybrid /v1/logs")
        else:
            check.skipped = "the Collector's own metrics were not readable"
    else:
        if exporter.get("sent_spans", 0) < spans:
            check.failures.append(f"{otelcol.DESTINATION_EXPORTER[destination]} sent {exporter.get('sent_spans', 0):.0f} "
                                  f"spans of {spans} received")
        for key in ("failed_spans", "failed_log_records", "failed_metric_points"):
            if exporter.get(key, 0) > 0:
                check.failures.append(f"{key} = {exporter[key]:.0f}")
        if destination == "xybrid" and exporter.get("sent_log_records", 0) != evaluations:
            check.failures.append(f"sent {exporter.get('sent_log_records', 0):.0f} log records to Xybrid, "
                                  f"expected the {evaluations} evaluation results only")
        check.notes.append(json.dumps(exporter, sort_keys=True))
    errors = otelcol.export_errors(log_text)
    if errors:
        check.failures.append(f"{len(errors)} export errors in collector.log, first: {errors[0]}")
    checks.append(check)

    if wire is not None and destination == "xybrid":
        check = expectations.Check("collector", None, "routing.xybrid_logs")
        names = destinations.wire_log_event_names(wire.entries())
        other = [name for name in names if name != expectations.EVALUATION_EVENT]
        if other:
            check.failures.append(f"non-evaluation log records sent to Xybrid /v1/logs: {sorted(set(map(str, other)))}")
        if len(names) != evaluations:
            check.failures.append(f"{len(names)} log records sent to Xybrid, {evaluations} evaluation results received")
        checks.append(check)
    return checks


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def render_markdown(report: Mapping[str, Any]) -> str:
    producers = report.get("producers") or {}
    collector = report.get("collector") or {}
    info = report.get("destination_info") or {}
    summary = report.get("summary") or {}
    lines = [
        f"# Observability interop: {report['destination']} ({report['mode']})",
        "",
        f"Result: **{'PASS' if report.get('ok') else 'FAIL'}** ({summary.get('pass', 0)} passed, "
        f"{summary.get('fail', 0)} failed, {summary.get('skip', 0)} skipped), "
        f"{report.get('started_at')} to {report.get('finished_at')}.",
        "",
        "## Versions",
        "",
        "| Component | Version |",
        "|---|---|",
        f"| Producers (git) | `{producers.get('git_sha')}` on `{producers.get('git_branch')}`"
        f"{' (uncommitted changes)' if producers.get('git_dirty') else ''} |",
        f"| xybrid-observability (Rust) | {producers.get('xybrid_observability_crate')} |",
        f"| @xybrid/observability (JS) | {producers.get('observability_js_package')} |",
    ]
    for name, data in (report.get("workloads") or {}).items():
        lines.append(f"| workload `{name}` | {data.get('program')} (exit {data.get('exit_codes')}) |")
    runtimes = producers.get("runtimes") or {}
    lines.append("| runtimes | " + ", ".join(f"{key} {value}" for key, value in runtimes.items() if value) + " |")
    lines.append(f"| Collector | {collector.get('distribution')} {collector.get('version')} "
                 f"{collector.get('platform')}, archive sha256 `{collector.get('archive_sha256')}` |")
    lines.append(f"| Collector configs | {', '.join(f'`{c}`' for c in collector.get('configs') or [])} |")
    lines.append(f"| Destination | {info.get('product')} {info.get('version') or ''} |")
    for name, image in (info.get("containers") or {}).items():
        lines.append(f"| container `{name}` | `{image}` |")
    lines += ["", "## What the destination keeps", "", expectations.table_markdown(report["destination"]), ""]
    lines += ["## Checks", "", "| Stage | Workload | Check | Status | Detail |", "|---|---|---|---|---|"]
    for check in report.get("checks") or []:
        detail = "; ".join(check["failures"]) or check.get("skipped") or "; ".join(check["notes"])
        detail = detail.replace("|", "\\|").replace("\n", " ")
        if len(detail) > 400:
            detail = detail[:400] + "..."
        lines.append(f"| {check['stage']} | {check['workload'] or '-'} | `{check['name']}` | "
                     f"{check['status'].upper()} | {detail} |")
    lines += ["", "## Workloads", ""]
    for name, data in (report.get("workloads") or {}).items():
        truth = data.get("ground_truth") or {}
        lines.append(f"- `{name}`: {truth.get('spans')} spans in {truth.get('trace_ids')}, "
                     f"{truth.get('evaluation_results')} evaluation results, {truth.get('metric_points')} metric "
                     f"points; reported refs {json.dumps(data.get('reported_refs'))}")
    return "\n".join(lines) + "\n"


def print_summary(report: Mapping[str, Any], checks: Sequence[expectations.Check]) -> None:
    for check in checks:
        if check.status == "fail":
            print(f"FAIL {check.stage}/{check.workload or '-'}/{check.name}")
            for failure in check.failures:
                print(f"     - {failure}")
    summary = report["summary"]
    print(f"{report['destination']}{' (wire)' if report['mode'] == 'wire' else ''}: "
          f"{summary['pass']} passed, {summary['fail']} failed, {summary['skip']} skipped -> "
          f"{report['out_dir']}/report.md", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--destination", choices=(*ALL_DESTINATIONS, "all"),
                        help="backend to read back (all = the three in turn; xybrid only with a staging key)")
    parser.add_argument("--wire", action="store_true",
                        help="no backend: point the profile's exporter at a local capture server and check "
                             "the dialect it sends")
    parser.add_argument("--validate-profiles", action="store_true",
                        help="only run otelcol-contrib validate on every profile combination")
    parser.add_argument("--workloads", default=",".join(workloads.DEFAULT_WORKLOADS),
                        help=f"comma-separated subset of {', '.join(workloads.WORKLOADS)}")
    parser.add_argument("--out-dir", type=Path,
                        help="where reports go (default: $TMPDIR/xybrid-interop/<UTC time>)")
    parser.add_argument("--collector-binary", type=Path, help="use this otelcol-contrib (must be the pinned version)")
    parser.add_argument("--cache-dir", type=Path, help="Collector cache (default ~/.cache/xybrid/otelcol)")
    parser.add_argument("--collector-port", type=int, help="OTLP/HTTP port (default interop.env, 4318)")
    parser.add_argument("--no-sinks", dest="sinks", action="store_false",
                        help="leave sinks.yaml out (metrics and other logs go to nop)")
    parser.add_argument("--python", default=os.environ.get("XYBRID_INTEROP_PYTHON", "python3"),
                        help="interpreter with the built Python SDK (default $XYBRID_INTEROP_PYTHON or python3)")
    parser.add_argument("--js-runtime", choices=("node", "bun"), default="node")
    parser.add_argument("--cargo-release", action="store_true", help="build the Rust example in release mode")
    parser.add_argument("--workload-timeout", type=float, default=900.0)
    parser.add_argument("--readback-timeout", type=float, default=180.0,
                        help="seconds to wait for a destination to show every span")
    args = parser.parse_args(argv)
    if not args.validate_profiles and not args.destination:
        parser.error("give --destination (or --validate-profiles)")
    names = [name.strip() for name in args.workloads.split(",") if name.strip()]
    unknown = [name for name in names if name not in workloads.WORKLOADS]
    if unknown:
        parser.error(f"unknown workloads {unknown}; choose from {list(workloads.WORKLOADS)}")
    args.workloads = names
    if args.out_dir is None:
        stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "xybrid-interop" / stamp
    args.out_dir = args.out_dir.resolve()
    return args


def run_validate(args: argparse.Namespace) -> int:
    binary = otelcol.ensure_collector(cache_dir=args.cache_dir, explicit=args.collector_binary)
    results = otelcol.validate_profiles(binary.path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "xybrid-observability-collector-validate/1", "collector": binary.describe(),
               "results": results, "ok": all(result["ok"] for result in results)}
    (args.out_dir / "validate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"{'PASS' if result['ok'] else 'FAIL'} {' + '.join(result['configs'])}"
              + ("" if result["ok"] else f"\n     {result['output'][-400:]}"))
    print(f"otelcol-contrib {binary.version} ({binary.platform}): {sum(r['ok'] for r in results)}/{len(results)} "
          f"combinations valid -> {args.out_dir / 'validate.json'}")
    return 0 if payload["ok"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_profiles:
            return run_validate(args)
        settings = load_settings()
        python = shutil.which(args.python) or args.python
        tools = workloads.Toolchain(python=python, node=shutil.which("node"), js_runtime=args.js_runtime,
                                    cargo_release=args.cargo_release)
        targets = list(ALL_DESTINATIONS) if args.destination == "all" else [args.destination]
        if args.destination == "all" and not args.wire and not os.environ.get("XYBRID_STAGING_API_KEY"):
            print("note: XYBRID_STAGING_API_KEY is not set; skipping the xybrid destination", flush=True)
            targets.remove("xybrid")
        reports = [run_destination(target, args, settings, tools) for target in targets]
    except (SetupError, otelcol.CollectorError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if len(reports) > 1:
        index = {"schema": SCHEMA + "/index", "reports": [
            {"destination": r["destination"], "mode": r["mode"], "ok": r["ok"], "summary": r["summary"],
             "report": f"{r['out_dir']}/report.json"} for r in reports]}
        (args.out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
