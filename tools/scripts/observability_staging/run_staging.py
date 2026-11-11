#!/usr/bin/env python3
"""Step 6 staging acceptance driver: one lane launch, read back from staging.

    run_staging.py run --lane python-mac -- \\
        python3 bindings/python/tests/observability_staging.py --config {config} --expectations {expectations}

``run`` starts the pinned Collector (Xybrid profile + sinks + the harness tee)
and two capture servers, writes the lane's ``config.json`` (mode 0600, outside
the evidence), runs the lane command, stops the Collector, reads every row of
the launch's traces back from the staging API, reads the dashboard aggregate,
runs ``staging_checks.check_launch`` and writes ``report.json`` and
``report.md``. Exit status 0 only when the command succeeded and every check
passed; 2 when the launch could not start.

    run_staging.py isolation --traces traces.json --env-file staging.env [--probe]
    run_staging.py recheck --run-dir DIR [--probe-file isolation-probe.json]
    run_staging.py readback --run-dir DIR     # re-read and re-check a finished launch
    run_staging.py export-check --export run.json --out DIR   # the CLI lane's --trace-export files
    run_staging.py summary --root DIR         # every launch's result, and traces.json

``isolation`` runs where the other tenants' keys live (never on the host that
holds the launch key): every launch trace read with tenant A's key (another
organization) and tenant B's key (another project of the same organization)
must answer 404; ``--probe`` also writes one span under each trace with B's
key and checks B sees only that span. ``recheck`` then shows, with the launch
key, that each launch's span set did not change.

See README.md. Python 3.9+ standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "observability_interop"))
sys.path.insert(0, str(HERE.parent / "observability_capture"))

import destinations  # noqa: E402 -- sibling tool directories, imported after the path is set
import otelcol  # noqa: E402
import run_interop  # noqa: E402
import staging_checks  # noqa: E402

REPORT_SCHEMA = "xybrid-staging-acceptance/1"
DEFAULT_KEY_FILE = Path.home() / ".xybrid-step6" / "key"
PLATFORM_REPO = "xybrid-ai/xybrid-platform"
# Platform workflows whose staging jobs deploy what the lanes talk to.
DEPLOY_JOBS = {
    "GCP - CD (auth, API)": ("deploy (api)", "deploy (auth)"),
    "Cloudflare - Workers CI/CD": ("deploy-ingest-worker-staging", "deploy-gateway-worker-staging",
                                   "deploy-staging", "deploy-registry-staging"),
    "Tinybird - CD": ("deploy-staging",),
}
STRIPPED_ENV = ("XYBRID_API_KEY", "XYBRID_STAGING_API_KEY", "XYBRID_STAGING_API_KEY_B", "XYBRID_PLATFORM_API_KEY")


class SetupError(RuntimeError):
    """The launch cannot start; the message says what to do."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(moment: Optional[datetime] = None) -> str:
    return (moment or utc_now()).strftime("%Y%m%dt%H%M%Sz")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Key, hosts, private files
# ---------------------------------------------------------------------------


def read_key(path: Path) -> str:
    """The launch key, from a file only the user can read. Never printed."""
    try:
        info = path.stat()
    except OSError:
        raise SetupError(f"no key file at {path}; create it with mode 0600 (README, Never production)") from None
    if os.name == "posix" and info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SetupError(f"{path} is readable by others (mode {stat.S_IMODE(info.st_mode):o}); chmod 600 it")
    key = path.read_text(encoding="utf-8").strip()
    if not key or any(ch.isspace() for ch in key):
        raise SetupError(f"{path} does not hold exactly one key")
    return key


def staging_urls(settings: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
    settings = settings or run_interop.load_settings()
    try:
        return (run_interop.check_xybrid_target(settings["XYBRID_STAGING_INGEST_URL"]),
                run_interop.check_xybrid_target(settings["XYBRID_STAGING_API_URL"]))
    except run_interop.SetupError as error:
        raise SetupError(str(error)) from None


def private_dir(key_file: Path, run: str) -> Path:
    directory = key_file.parent / "run" / run
    directory.mkdir(parents=True, exist_ok=True)
    for path in (key_file.parent / "run", directory):
        path.chmod(0o700)
    return directory


def write_private(path: Path, value: Any) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def remove_private(directory: Path) -> None:
    for path in directory.glob("*"):
        try:
            size = path.stat().st_size
            with path.open("r+b") as handle:
                handle.write(b"\0" * size)
            path.unlink()
        except OSError:
            pass
    try:
        directory.rmdir()
    except OSError:
        pass


def lane_env(values: Mapping[str, str], base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    base = os.environ if base is None else base
    env = {key: value for key, value in base.items() if not key.startswith("OTEL_") and key not in STRIPPED_ENV}
    env.update({f"XYBRID_STAGING_{key.upper()}": value for key, value in values.items()})
    return env


def substitute(command: Sequence[str], values: Mapping[str, str]) -> List[str]:
    out = []
    for argument in command:
        for key, value in values.items():
            argument = argument.replace("{" + key + "}", value)
        out.append(argument)
    return out


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def scan_secrets(root: Path, needles: Mapping[str, str]) -> List[str]:
    """Files under ``root`` holding any needle value; reported by label only
    (``{"the key": key, "Bearer": "Bearer "}``), never by value."""
    hits = []
    encoded = {label: value.encode("utf-8") for label, value in needles.items() if value}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        hits += [f"{path.relative_to(root)}: {label}" for label, raw in encoded.items() if raw in data]
    return hits


class HostCollector(otelcol.RunningCollector):
    """The pinned Collector whose receiver binds ``host`` (``collector-lan.yaml``
    for a LAN address): readiness is probed there, not on loopback."""

    def __init__(self, host: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.host = host

    def start(self, port: int, timeout: float = 60.0) -> None:
        if self.host in ("127.0.0.1", "localhost"):
            super().start(port, timeout)
            return
        if otelcol.port_open(self.host, port):
            raise otelcol.CollectorError(f"{self.host}:{port} is already in use")
        arguments = [str(self.binary)]
        for config in self.configs:
            arguments += ["--config", str(config)]
        self._log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(arguments, env=self.env, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise otelcol.CollectorError(f"the Collector exited with {self.process.returncode}; see {self.log_path}")
            if otelcol.port_open(self.host, port):
                return
            time.sleep(0.2)
        self.stop()
        raise otelcol.CollectorError(f"the Collector did not listen on {self.host}:{port} within {timeout:.0f}s")


# ---------------------------------------------------------------------------
# Platform deployments (read with gh, never written)
# ---------------------------------------------------------------------------


def gh_json(argv: Sequence[str], runner: Callable[..., Any] = subprocess.run, attempts: int = 4,
            pause: float = 5.0) -> Any:
    """``gh <argv>`` as JSON, retried: GitHub's API answers 502 now and then,
    and a missed deployment record voids the launch's deployment check."""
    error = ""
    for attempt in range(attempts):
        completed = runner(["gh", *argv], capture_output=True, text=True, timeout=60)
        if completed.returncode == 0:
            return json.loads(completed.stdout or "null")
        error = completed.stderr.strip()[:300]
        if attempt + 1 < attempts:
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"gh {' '.join(argv[:3])}: {error}")


def deployments(runner: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    """The head commit of the newest successful run of every staging deploy
    job on the platform's main branch: ``{"<workflow> / <job>": sha}``."""
    found: Dict[str, Any] = {}
    for workflow, jobs in DEPLOY_JOBS.items():
        runs = gh_json(["run", "list", "-R", PLATFORM_REPO, "--branch", "main", "--workflow", workflow,
                        "--status", "success", "--limit", "20", "--json", "databaseId,headSha"], runner) or []
        wanted = set(jobs)
        for item in runs:
            if not wanted:
                break
            view = gh_json(["run", "view", str(item["databaseId"]), "-R", PLATFORM_REPO, "--json", "jobs"], runner)
            for job in (view or {}).get("jobs") or []:
                if job.get("name") in wanted and job.get("conclusion") == "success":
                    found[f"{workflow} / {job['name']}"] = item["headSha"]
                    wanted.discard(job["name"])
        for job in sorted(wanted):
            found[f"{workflow} / {job}"] = None
    return found


# ---------------------------------------------------------------------------
# Staging reads
# ---------------------------------------------------------------------------


def read_trace(client: destinations.Client, trace_id: str, session: str,
               want_spans: int, want_refs: Sequence[str], want_evaluations: int, timeout: float,
               interval: float = 4.0, log: Callable[[str], None] = print) -> staging_checks.SessionReadback:
    """Polls one trace until its spans and evaluations are all there (or the
    timeout passes), then reads the object-stored content of each span."""
    readback = staging_checks.SessionReadback()
    deadline = time.monotonic() + timeout

    def get(path: str, label: str) -> Tuple[Optional[Any], Optional[str]]:
        try:
            return client.get(path, label=label), None
        except destinations.HttpError as error:
            if error.status == 404:
                return None, None
            return None, str(error)

    last_error: Optional[str] = None
    while True:
        body, last_error = get(f"/v1/telemetry/traces/{trace_id}/spans", f"{session}-spans")
        spans = (body or {}).get("spans") or []
        ids = {str(span.get("span_id", "")).lower() for span in spans}
        if body is not None:
            readback.spans = body
        if len(spans) >= want_spans and set(want_refs) <= ids:
            break
        if time.monotonic() >= deadline:
            log(f"    {session}: {len(spans)} of {want_spans} spans after {timeout:.0f}s")
            break
        time.sleep(interval)
    while True:
        body, error = get(f"/v1/telemetry/traces/{trace_id}/evaluations", f"{session}-evaluations")
        readback.evaluations = body
        last_eval_error = error
        if len((body or {}).get("items") or []) >= want_evaluations or time.monotonic() >= deadline:
            break
        time.sleep(interval)
    if want_evaluations == 0:
        # Nothing to wait for: read once more after a pause, so a late record is seen.
        time.sleep(min(interval * 2, max(0.0, deadline - time.monotonic())))
        body, last_eval_error = get(f"/v1/telemetry/traces/{trace_id}/evaluations", f"{session}-evaluations")
        readback.evaluations = body
    for error in (last_error, last_eval_error):
        if error:
            readback.errors.append(error)
    for span in ((readback.spans or {}).get("spans") or []):
        stored = span.get("content") or {}
        if stored.get("storage") == "object" and stored.get("values") is None:
            span_id = str(span.get("span_id", "")).lower()
            body, error = get(f"/v1/telemetry/traces/{trace_id}/spans/{span_id}/content",
                              f"{session}-content-{span_id}")
            if body is not None:
                readback.contents[span_id] = body
            if error:
                readback.errors.append(error)
    return readback


def read_stats(client: destinations.Client, label: str) -> Optional[Dict[str, Any]]:
    try:
        return client.get("/v1/telemetry/stats", label=label)
    except destinations.HttpError as error:
        print(f"warning: stats read failed: {error}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def stream_command(command: Sequence[str], env: Mapping[str, str], log_path: Path, timeout: float,
                   cwd: Path) -> int:
    """Runs the lane command in its own process group, copying its output to
    ``log_path`` and to stdout; a timeout stops everything it started."""
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        try:
            process = subprocess.Popen(list(command), cwd=str(cwd), env=dict(env), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace",
                                       start_new_session=os.name == "posix")
        except OSError as error:
            log.write(f"cannot start: {error}\n")
            return 127

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write("    | " + line)
                sys.stdout.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write(f"timed out after {timeout:.0f}s\n")
            run_interop.kill_group(process)
            process.wait()
            status = 124
        except BaseException:
            run_interop.kill_group(process)
            process.wait()
            raise
        reader.join(timeout=10)
        return status


def session_configs(names: Sequence[str], key: str, ingest: str, api: str, collector: str) -> Dict[str, Any]:
    """Each session's export settings and its own random T and P (README, config.json)."""
    sessions: Dict[str, Any] = {}
    for name in names:
        kind = staging_checks.session_kind(name)
        values: Dict[str, Any] = {"trace_id": secrets.token_hex(16), "parent_span_id": secrets.token_hex(8)}
        if kind == "direct":
            values.update({"traces_endpoint": f"{ingest}/v1/traces", "api_key": key})
        elif kind == "direct-logs":
            values.update({"traces_endpoint": f"{ingest}/v1/traces", "logs_endpoint": f"{ingest}/v1/logs",
                           "api_key": key})
        else:
            values["endpoint"] = collector
        if name == "case-fallback":
            values.update({"platform_url": api, "api_key": key})
        sessions[name] = values
    return sessions


def expected_span_counts(config: Mapping[str, Any], expectations: Mapping[str, Any],
                         truth_counts: Mapping[str, int]) -> Dict[str, int]:
    """Spans to wait for per session: what the Collector received, or (no
    Collector) what the binding reported exported."""
    counts = {}
    for session in config.get("sessions") or {}:
        values = ((expectations.get("sessions") or {}).get(session)) or {}
        if staging_checks.session_kind(session) in ("collector", "case"):
            counts[session] = truth_counts.get(session, 0)
        else:
            exported = (((values.get("stats") or {}).get("after_shutdown") or {}).get("traces") or {}).get("exported")
            counts[session] = int(exported) if isinstance(exported, int) else 3
    return counts


def truth_span_counts(entries: Sequence[Mapping[str, Any]], run: str) -> Dict[str, int]:
    import check_capture  # noqa: PLC0415 -- on the path set above

    counts: Dict[str, int] = {}
    for span in check_capture.Capture.from_entries(list(entries)).spans:
        if span.resource.get("xybrid.acceptance.run") == run:
            session = str(span.resource.get("xybrid.acceptance.session"))
            counts[session] = counts.get(session, 0) + 1
    return counts


def readback_launch(out: Path, client: destinations.Client, config: Mapping[str, Any],
                    expectations: Mapping[str, Any], ground_truth: Sequence[Mapping[str, Any]],
                    timeout: float) -> Dict[str, staging_checks.SessionReadback]:
    counts = expected_span_counts(config, expectations, truth_span_counts(ground_truth, str(config["run"])))
    readback = {}
    for session in config["sessions"]:
        trace_id = config["sessions"][session]["trace_id"]
        values = ((expectations.get("sessions") or {}).get(session)) or {}
        refs = [str(ref.get("span_id", "")).lower() for ref in (values.get("references") or {}).values()
                if isinstance(ref, dict) and ref.get("span_id")]
        want_evaluations = 2 if session in staging_checks.EVALUATING_SESSIONS else 0
        print(f"==> reading {session} trace {trace_id} ({counts[session]} spans, {want_evaluations} evaluations)",
              flush=True)
        readback[session] = read_trace(client, trace_id, session, counts[session], refs, want_evaluations, timeout)
        write_json(out / "readback" / f"{session}.json", readback[session].as_dict())
    return readback


def finish_launch(out: Path, config: Mapping[str, Any], expectations: Mapping[str, Any],
                  readback: Mapping[str, staging_checks.SessionReadback], lane_exit: int,
                  header: Mapping[str, Any], key: Optional[str]) -> int:
    def load(name: str) -> Any:
        path = out / name
        return read_json(path) if path.exists() else None

    def entries(name: str) -> List[Dict[str, Any]]:
        path = out / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    launch = staging_checks.Launch(
        config=dict(config), expectations=dict(expectations), readback=dict(readback),
        ground_truth=entries("ground-truth.jsonl"), sinks=entries("sinks.jsonl"),
        stats_before=load("stats-before.json"), stats_after=load("stats-after.json"),
        deployments_before=load("deployments-before.json"), deployments_after=load("deployments-after.json"),
        lane_exit=lane_exit)
    launch.secret_hits = scan_secrets(out, {"the key": key or "", "Bearer": "Bearer "})
    checks = staging_checks.check_launch(launch)
    if header.get("deployments") == "not recorded":
        for check in checks:
            if check.name == "deployment.unchanged":
                check.failures, check.skipped = [], "--no-deployments"
    summary = staging_checks.summarize(checks)
    passed = lane_exit == 0 and summary.get("fail", 0) == 0
    report = {
        "schema": REPORT_SCHEMA, "lane": config.get("lane"), "run": config.get("run"), "passed": passed,
        "lane_exit": lane_exit, "summary": summary, "header": dict(header),
        "traces": {session: {"trace_id": values.get("trace_id"), "parent_span_id": values.get("parent_span_id")}
                   for session, values in (config.get("sessions") or {}).items()},
        "counted_requests": staging_checks.counted_requests(launch),
        "checks": [check.as_dict() for check in checks],
    }
    write_json(out / "report.json", report)
    (out / "report.md").write_text(staging_checks.render_markdown(report), encoding="utf-8")
    for check in checks:
        marker = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[check.status]
        detail = "; ".join(check.failures) or check.skipped or ""
        print(f"{marker} {check.workload} {check.name}" + (f": {detail[:300]}" if detail else ""))
    print(f"==> {'PASS' if passed else 'FAIL'}: {summary} (lane exit {lane_exit}); report {out / 'report.md'}")
    return 0 if passed else 1


def cmd_run(args: argparse.Namespace) -> int:
    spec = staging_checks.load_spec()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SetupError("give the lane command after --")
    if args.lane not in spec["lanes"]:
        raise SetupError(f"unknown lane {args.lane!r}; staging.json lists {sorted(spec['lanes'])}")
    key = read_key(args.key_file)
    ingest, api = staging_urls()
    started = utc_now()
    run = f"s6-{stamp(started)}-{args.lane}"
    out = (args.out_root / args.lane / run).resolve()
    out.mkdir(parents=True, exist_ok=True)
    harness = run_interop.command_output(["git", "rev-parse", "HEAD"])
    commit = args.commit or harness
    lane = spec["lanes"][args.lane]
    print(f"==> lane {args.lane} ({lane['binding']} on {lane['host']}), run {run}, commit {commit}", flush=True)

    raw = run_interop.RawLog()
    client = destinations.Client(api, {"Authorization": f"Bearer {key}"}, raw)
    header: Dict[str, Any] = {"sdk_commit": commit, "harness_commit": harness,
                              "harness_dirty": bool(run_interop.command_output(
                                  ["git", "status", "--porcelain", "--untracked-files=no"])),
                              "binding": lane["binding"], "host": lane["host"],
                              "started": started.isoformat(), "staging": {"api": api, "ingest": ingest}}
    try:
        header["platform"] = {"api_health": health(api), "ingest_health": health(ingest)}
    except Exception as error:  # noqa: BLE001 -- recorded, the checks decide
        header["platform"] = f"unavailable: {error}"
    if args.no_deployments:
        header["deployments"] = "not recorded"
    else:
        try:
            write_json(out / "deployments-before.json", deployments())
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
            print(f"warning: deployments not recorded: {error}", file=sys.stderr)
    stats_before = read_stats(client, "stats-before")
    if stats_before is not None:
        write_json(out / "stats-before.json", stats_before)

    binary = otelcol.ensure_collector()
    header["collector"] = binary.describe()
    port = args.collector_port or free_port(args.collector_host)
    metrics_port = free_port()
    tee = run_interop.CaptureServer(out, "ground-truth").start()
    sinks = run_interop.CaptureServer(out, "sinks").start()
    configs = otelcol.run_configs("xybrid", sinks=True)
    if args.collector_host not in ("127.0.0.1", "localhost"):
        configs.append(HERE / "collector-lan.yaml")
    env = otelcol.scrubbed_environment()
    env.update({
        "XYBRID_API_KEY": key, "XYBRID_TRACES_ENDPOINT": f"{ingest}/v1/traces",
        "XYBRID_LOGS_ENDPOINT": f"{ingest}/v1/logs", "METRICS_OTLP_ENDPOINT": sinks.endpoint,
        "LOGS_OTLP_ENDPOINT": sinks.endpoint, "INTEROP_COLLECTOR_PORT": str(port),
        "INTEROP_COLLECTOR_METRICS_PORT": str(metrics_port), "INTEROP_CAPTURE_ENDPOINT": tee.endpoint,
        "STAGING_COLLECTOR_HOST": args.collector_host,
    })
    collector = HostCollector(args.collector_host, binary.path, configs, env, out / "collector.log")
    header["collector"]["configs"] = [str(path.relative_to(REPO)) for path in configs]
    secret_dir = private_dir(args.key_file, run)
    status = 1
    try:
        collector.start(port)
        sessions = session_configs(staging_checks.lane_sessions(spec, args.lane), key, ingest, api,
                                   f"http://{args.collector_host}:{port}")
        constants = {key_: value for key_, value in spec.items() if key_ not in ("lanes",)}
        config = {"schema": staging_checks.SCHEMA, "binding": lane["binding"], "lane": args.lane, "run": run,
                  "commit": commit, "model_id": args.model or lane.get("model") or spec["model"]["id"],
                  "sessions": sessions,
                  "constants": constants}
        config_path = secret_dir / "config.json"
        write_private(config_path, config)
        public = staging_checks.public_config(config)
        write_json(out / "config.public.json", public)
        expectations_path = out / "expectations.json"
        values = {"config": str(config_path), "expectations": str(expectations_path), "out_dir": str(out),
                  "lane": args.lane, "run": run, "collector_port": str(port), "repo": str(REPO),
                  "collector_host": args.collector_host}
        print(f"==> Collector on {args.collector_host}:{port}; running the lane command", flush=True)
        status = stream_command(substitute(command, values), lane_env(values), out / "lane.log", args.timeout, REPO)
        header["lane_exit"] = status
        print(f"==> lane command exited {status}", flush=True)
    finally:
        remove_private(secret_dir)
        run_interop.settle([tee, sinks], quiet=3.0, limit=60.0)
        try:
            metrics, text = otelcol.scrape_metrics(metrics_port)
            write_json(out / "collector-metrics.json", metrics)
            (out / "collector-metrics.txt").write_text(text, encoding="utf-8")
        except OSError as error:
            print(f"warning: Collector metrics unreadable: {error}", file=sys.stderr)
        collector.stop()
        run_interop.settle([tee, sinks], quiet=2.0, limit=30.0)
        tee.stop()
        sinks.stop()

    expectations_path = out / "expectations.json"
    if not expectations_path.exists():
        print(f"error: the lane wrote no {expectations_path}", file=sys.stderr)
        expectations: Dict[str, Any] = {"schema": staging_checks.SCHEMA, "binding": lane["binding"],
                                        "lane": args.lane, "run": run, "sessions": {},
                                        "scenario_error": "the lane wrote no expectations.json"}
    else:
        expectations = read_json(expectations_path)
    header.update({"sdk_version": expectations.get("sdk_version"), "model_id": expectations.get("model_id"),
                   "device": expectations.get("device")})
    ground_truth = [json.loads(line) for line in (out / "ground-truth.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()] if (out / "ground-truth.jsonl").exists() else []
    # Nothing to wait for when the lane never ran its scenario.
    timeout = args.readback_timeout if expectations_path.exists() else 10.0
    readback = readback_launch(out, client, public, expectations, ground_truth, timeout)
    counted = sum(1 for session in readback.values()
                  for span in session.stored() if span.counted is True)
    if stats_before is not None:
        expected_total = stats_before.get("total_executions_24h", 0) + counted
        stats_after = run_interop.poll(lambda: read_stats(client, "stats-after"),
                                       lambda value: isinstance(value, dict)
                                       and value.get("total_executions_24h", -1) >= expected_total,
                                       timeout=args.readback_timeout, interval=5.0)
        # One more read after a pause: a delta above the expected one must show.
        time.sleep(5.0)
        stats_after = read_stats(client, "stats-after") or stats_after
        if stats_after is not None:
            write_json(out / "stats-after.json", stats_after)
    if not args.no_deployments:
        try:
            write_json(out / "deployments-after.json", deployments())
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
            print(f"warning: deployments not recorded: {error}", file=sys.stderr)
    raw.write(out / "raw")
    write_json(out / "header.json", header)
    return finish_launch(out, public, expectations, readback, status, header, key)


def health(base: str) -> Any:
    """``GET /health`` of a staging service (unauthenticated)."""
    return destinations.Client(base, {}).get("/health", label="health")


def cmd_readback(args: argparse.Namespace) -> int:
    key = read_key(args.key_file)
    _, api = staging_urls()
    out = args.run_dir.resolve()
    config = read_json(out / "config.public.json")
    expectations = read_json(out / "expectations.json") if (out / "expectations.json").exists() else {}
    header = read_json(out / "header.json") if (out / "header.json").exists() else {}
    raw = run_interop.RawLog()
    client = destinations.Client(api, {"Authorization": f"Bearer {key}"}, raw)
    ground_truth = [json.loads(line) for line in (out / "ground-truth.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()] if (out / "ground-truth.jsonl").exists() else []
    readback = readback_launch(out, client, config, expectations, ground_truth, args.readback_timeout)
    stats_after = read_stats(client, "stats-after")
    if stats_after is not None:
        write_json(out / "stats-after.json", stats_after)
    raw.write(out / "raw")
    lane_exit = args.lane_exit if args.lane_exit is not None else int(header.get("lane_exit", 1))
    return finish_launch(out, config, expectations, readback, lane_exit, header, key)


# ---------------------------------------------------------------------------
# export-check (the CLI lane: --trace-export is the ground truth)
# ---------------------------------------------------------------------------


def cmd_export_check(args: argparse.Namespace) -> int:
    """Reads every trace of each ``xybrid run --trace-export`` file back from
    staging and runs the interop destination checks with the export as the
    ground truth, plus the CLI's identity and the dashboard aggregate."""
    import expectations as interop  # noqa: PLC0415 -- on the path set above
    import workloads  # noqa: PLC0415

    key = read_key(args.key_file)
    _, api = staging_urls()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = run_interop.RawLog()
    client = destinations.Client(api, {"Authorization": f"Bearer {key}"}, raw)
    checks: List[Any] = []
    counted = 0
    for export in args.export:
        name = export.stem
        body = read_json(export)
        entries = [{"seq": 1, "signal": "traces", "encoding": "json", "path": "/v1/traces", "body": body}]
        workload = workloads.Workload(name=name, language="cli", program=str(export), service_name="xybrid-cli",
                                      content="captured", secrets=(), replacements=(), evaluations=0,
                                      streaming_timing=True)
        truth = interop.GroundTruth.build(workload, entries)
        spans: List[destinations.DestSpan] = []
        text = []
        for trace_id in truth.trace_ids():
            want = [span.span_id for span in truth.spans if span.trace_id == trace_id]
            readback = read_trace(client, trace_id, name, len(want), want, 0, args.readback_timeout)
            write_json(out / "readback" / f"{name}-{trace_id}.json", readback.as_dict())
            stored = [span for span in readback.stored() if span.trace_id == trace_id]
            spans += stored
            text.append(readback.text())
            counted += sum(1 for span in stored if span.counted is True)
        found = interop.destination_checks("xybrid", truth, spans, [], "\n".join(text))
        identity = interop.Check("staging", name, "identity")
        for span in spans:
            values = staging_checks.resource(span)
            if values.get("xybrid.sdk.binding") != "cli" or values.get("service.name") != "xybrid-cli":
                identity.failures.append(f"{staging_checks.label(span)}: binding {values.get('xybrid.sdk.binding')!r}, "
                                         f"service {values.get('service.name')!r}")
        if not spans:
            identity.failures.append("nothing stored")
        identity.notes.append(f"{len(truth.spans)} exported spans in {len(truth.trace_ids())} trace(s), "
                              f"{len(spans)} stored")
        checks += found + [identity]
    stats_after = read_stats(client, "stats-after")
    aggregate = interop.Check("staging", "cli", "aggregate.stats")
    if args.stats_before is not None and stats_after is not None:
        delta = staging_checks.stats_delta(read_json(args.stats_before), stats_after)
        if delta != counted:
            aggregate.failures.append(f"total_executions_24h changed by {delta}, the exports hold {counted} "
                                      f"counted requests")
        aggregate.notes.append(f"+{delta}, counted {counted}")
    else:
        aggregate.skipped = "no --stats-before"
    checks.append(aggregate)
    raw.write(out / "raw")
    summary = interop.summarize(checks)
    hits = scan_secrets(out, {"the key": key, "Bearer": "Bearer "})
    report = {"schema": "xybrid-staging-cli/1", "at": utc_now().isoformat(), "passed": not summary.get("fail")
              and not hits, "summary": summary, "secret_hits": hits, "counted_requests": counted,
              "exports": [str(path) for path in args.export], "checks": [check.as_dict() for check in checks]}
    write_json(out / "report.json", report)
    for check in checks:
        detail = "; ".join(check.failures) or check.skipped or ""
        print(f"{check.status.upper()} {check.workload} {check.name}" + (f": {detail[:300]}" if detail else ""))
    print(f"==> {'PASS' if report['passed'] else 'FAIL'}: {summary}; {out / 'report.json'}")
    return 0 if report["passed"] else 1


# ---------------------------------------------------------------------------
# isolation and recheck
# ---------------------------------------------------------------------------


def probe_span(trace_id: str, span_id: str, now_ns: int) -> Dict[str, Any]:
    """One OTLP/JSON span under ``trace_id``, written with tenant B's key."""
    return {"resourceSpans": [{
        "resource": {"attributes": [
            {"key": "service.name", "value": {"stringValue": "xybrid-staging-isolation-probe"}},
            {"key": "xybrid.acceptance.session", "value": {"stringValue": "isolation-probe"}}]},
        "scopeSpans": [{"scope": {"name": "xybrid-staging-isolation"}, "spans": [{
            "traceId": trace_id, "spanId": span_id, "name": "chat isolation-probe", "kind": 1,
            "startTimeUnixNano": str(now_ns - 1_000_000), "endTimeUnixNano": str(now_ns),
            "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                           {"key": "gen_ai.request.model", "value": {"stringValue": "isolation-probe"}},
                           {"key": "gen_ai.provider.name", "value": {"stringValue": "isolation-probe"}}],
            "status": {"code": 1}}]}]}]}


def post_json(url: str, key: str, body: Mapping[str, Any]) -> int:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with destinations.OPENER.open(request, timeout=30) as response:  # noqa: S310
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def cmd_isolation(args: argparse.Namespace) -> int:
    ingest, api = staging_urls()
    env = run_interop.parse_env_file(args.env_file.read_text(encoding="utf-8"))
    keys = {"A (other organization)": env.get("XYBRID_STAGING_API_KEY"),
            "B (same organization, other project)": env.get("XYBRID_STAGING_API_KEY_B")}
    if not all(keys.values()):
        raise SetupError(f"{args.env_file} must define XYBRID_STAGING_API_KEY and XYBRID_STAGING_API_KEY_B")
    traces = read_json(args.traces)
    results: List[Dict[str, Any]] = []
    probes: Dict[str, str] = {}
    failures = 0
    for tenant, key in keys.items():
        client = destinations.Client(api, {"Authorization": f"Bearer {key}"})
        for item in traces:
            trace_id = item["trace_id"]
            paths = [f"/v1/telemetry/traces/{trace_id}/spans", f"/v1/telemetry/traces/{trace_id}/evaluations"]
            if item.get("run_span_id"):
                paths.append(f"/v1/telemetry/traces/{trace_id}/spans/{item['run_span_id']}/content")
            for path in paths:
                try:
                    client.get(path, label="isolation")
                    status = 200
                except destinations.HttpError as error:
                    status = error.status
                ok = status == 404
                failures += 0 if ok else 1
                results.append({"tenant": tenant, "lane": item.get("lane"), "session": item.get("session"),
                                "path": path.replace(trace_id, "{T}"), "trace_id": trace_id, "status": status,
                                "ok": ok})
    if args.probe:
        key_b = str(keys["B (same organization, other project)"])
        client_b = destinations.Client(api, {"Authorization": f"Bearer {key_b}"})
        for item in traces:
            span_id = secrets.token_hex(8)
            status = post_json(f"{ingest}/v1/traces", key_b, probe_span(item["trace_id"], span_id, time.time_ns()))
            probes[item["trace_id"]] = span_id
            results.append({"tenant": "B probe write", "lane": item.get("lane"), "session": item.get("session"),
                            "path": "/v1/traces", "trace_id": item["trace_id"], "status": status,
                            "ok": status == 200, "probe_span_id": span_id})
            failures += 0 if status == 200 else 1
        time.sleep(args.probe_wait)
        for item in traces:
            body = run_interop.poll(
                lambda: _get_or_none(client_b, f"/v1/telemetry/traces/{item['trace_id']}/spans"),
                lambda value: value is not None, timeout=60, interval=5)
            ids = sorted(str(span.get("span_id", "")).lower() for span in ((body or {}).get("spans") or []))
            ok = ids == [probes[item["trace_id"]]]
            failures += 0 if ok else 1
            results.append({"tenant": "B probe read", "lane": item.get("lane"), "session": item.get("session"),
                            "path": "/v1/telemetry/traces/{T}/spans", "trace_id": item["trace_id"],
                            "status": 200 if body is not None else 404, "ok": ok, "span_ids": ids})
    report = {"schema": "xybrid-staging-isolation/1", "at": utc_now().isoformat(), "passed": failures == 0,
              "failures": failures, "results": results, "probes": probes}
    write_json(args.out, report)
    for result in results:
        print(f"{'PASS' if result['ok'] else 'FAIL'} {result['tenant']}: {result['lane']}/{result['session']} "
              f"{result['path']} -> {result['status']}")
    print(f"==> {'PASS' if failures == 0 else 'FAIL'}: {len(results) - failures}/{len(results)}; {args.out}")
    return 0 if failures == 0 else 1


def _get_or_none(client: destinations.Client, path: str) -> Optional[Any]:
    try:
        return client.get(path, label="probe")
    except destinations.HttpError:
        return None


def cmd_recheck(args: argparse.Namespace) -> int:
    key = read_key(args.key_file)
    _, api = staging_urls()
    client = destinations.Client(api, {"Authorization": f"Bearer {key}"})
    probes = read_json(args.probe_file).get("probes", {}) if args.probe_file else {}
    failures = 0
    results = []
    for run_dir in args.run_dir:
        config = read_json(run_dir / "config.public.json")
        for session in config["sessions"]:
            trace_id = config["sessions"][session]["trace_id"]
            saved = read_json(run_dir / "readback" / f"{session}.json")
            before = sorted(str(span.get("span_id", "")).lower() for span in ((saved.get("spans") or {}).get("spans")
                                                                              or []))
            body = _get_or_none(client, f"/v1/telemetry/traces/{trace_id}/spans")
            now = sorted(str(span.get("span_id", "")).lower() for span in ((body or {}).get("spans") or []))
            probe = probes.get(trace_id)
            ok = now == before and (probe is None or probe not in now)
            failures += 0 if ok else 1
            results.append({"lane": config.get("lane"), "session": session, "trace_id": trace_id, "ok": ok,
                            "spans_before": len(before), "spans_now": len(now), "probe": probe})
            print(f"{'PASS' if ok else 'FAIL'} {config.get('lane')}/{session}: {len(before)} -> {len(now)} spans"
                  + (f", probe {probe} {'absent' if probe not in now else 'PRESENT'}" if probe else ""))
    if args.out:
        write_json(args.out, {"schema": "xybrid-staging-recheck/1", "at": utc_now().isoformat(),
                              "passed": failures == 0, "results": results})
    return 0 if failures == 0 else 1


def cmd_stats(args: argparse.Namespace) -> int:
    """The launch project's dashboard aggregate, and optionally its newest traces."""
    key = read_key(args.key_file)
    _, api = staging_urls()
    client = destinations.Client(api, {"Authorization": f"Bearer {key}"})
    value: Dict[str, Any] = {"at": utc_now().isoformat(), "stats": read_stats(client, "stats")}
    if args.traces:
        try:
            value["traces"] = client.get("/v1/telemetry/traces", [("limit", args.traces)], label="traces")
        except destinations.HttpError as error:
            value["traces"] = f"unavailable: {error}"
    stats = value["stats"] if args.traces == 0 else value
    write_json(args.out, stats)
    print(json.dumps(value["stats"]))
    return 0 if value["stats"] is not None else 1


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def cmd_summary(args: argparse.Namespace) -> int:
    rows, traces = [], []
    for report_path in sorted(args.root.rglob("report.json")):
        report = read_json(report_path)
        if report.get("schema") != REPORT_SCHEMA:
            continue
        rows.append(report)
        run_dir = report_path.parent
        expectations = read_json(run_dir / "expectations.json") if (run_dir / "expectations.json").exists() else {}
        for session, ids in (report.get("traces") or {}).items():
            run_ref = (((expectations.get("sessions") or {}).get(session) or {}).get("references") or {}).get("run")
            traces.append({"lane": report.get("lane"), "run": report.get("run"), "session": session,
                           "trace_id": ids.get("trace_id"),
                           "run_span_id": (run_ref or {}).get("span_id") if isinstance(run_ref, dict) else None})
    lines = ["| Lane | Run | Result | Passed | Failed | Skipped | Counted requests |", "|---|---|---|---|---|---|---|"]
    for report in rows:
        summary = report.get("summary") or {}
        lines.append(f"| {report.get('lane')} | {report.get('run')} | {'PASS' if report.get('passed') else 'FAIL'} | "
                     f"{summary.get('pass', 0)} | {summary.get('fail', 0)} | {summary.get('skip', 0)} | "
                     f"{report.get('counted_requests')} |")
    (args.root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(args.root / "traces.json", traces)
    print("\n".join(lines))
    print(f"==> {len(rows)} launches; {args.root / 'summary.md'}, {args.root / 'traces.json'}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_out_root() -> Path:
    return Path.home() / "xybrid-acceptance-evidence" / utc_now().strftime("%Y-%m-%d") / "step6" / "lanes"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    commands = parser.add_subparsers(dest="action", required=True)

    run = commands.add_parser("run", help="one lane launch")
    run.add_argument("--lane", required=True)
    run.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    run.add_argument("--out-root", type=Path, default=None)
    run.add_argument("--collector-host", default="127.0.0.1",
                     help="address the Collector binds and the collector session exports to (the host's LAN "
                          "address for an iPhone)")
    run.add_argument("--collector-port", type=int, default=0)
    run.add_argument("--commit", default=None, help="candidate commit (default: git HEAD)")
    run.add_argument("--model", default=None, help="registry model id (default: staging.json model.id)")
    run.add_argument("--timeout", type=float, default=1800.0, help="seconds the lane command may run")
    run.add_argument("--readback-timeout", type=float, default=180.0)
    run.add_argument("--no-deployments", action="store_true", help="do not read the platform deployments (gh)")
    run.add_argument("command", nargs=argparse.REMAINDER)

    readback = commands.add_parser("readback", help="re-read and re-check a finished launch")
    readback.add_argument("--run-dir", type=Path, required=True)
    readback.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    readback.add_argument("--readback-timeout", type=float, default=60.0)
    readback.add_argument("--lane-exit", type=int, default=None)

    isolation = commands.add_parser("isolation", help="cross-tenant reads (where tenant A's and B's keys live)")
    isolation.add_argument("--traces", type=Path, required=True)
    isolation.add_argument("--env-file", type=Path, required=True)
    isolation.add_argument("--probe", action="store_true")
    isolation.add_argument("--probe-wait", type=float, default=20.0)
    isolation.add_argument("--out", type=Path, default=Path("isolation.json"))

    recheck = commands.add_parser("recheck", help="each launch's span set is unchanged (after isolation --probe)")
    recheck.add_argument("--run-dir", type=Path, action="append", required=True)
    recheck.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    recheck.add_argument("--probe-file", type=Path, default=None)
    recheck.add_argument("--out", type=Path, default=None)

    export_check = commands.add_parser("export-check", help="CLI --trace-export files read back from staging")
    export_check.add_argument("--export", type=Path, action="append", required=True)
    export_check.add_argument("--out", type=Path, required=True)
    export_check.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    export_check.add_argument("--stats-before", type=Path, default=None)
    export_check.add_argument("--readback-timeout", type=float, default=120.0)

    stats = commands.add_parser("stats", help="the launch project's dashboard aggregate (and newest traces)")
    stats.add_argument("--out", type=Path, required=True)
    stats.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    stats.add_argument("--traces", type=int, default=0, help="also list this many newest traces")

    summary = commands.add_parser("summary", help="every launch's result under a root, and traces.json")
    summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "run" and args.out_root is None:
        args.out_root = default_out_root()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    handlers = {"run": cmd_run, "readback": cmd_readback, "isolation": cmd_isolation, "recheck": cmd_recheck,
                "export-check": cmd_export_check, "stats": cmd_stats, "summary": cmd_summary}
    try:
        return handlers[args.action](args)
    except (SetupError, otelcol.CollectorError, run_interop.SetupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
