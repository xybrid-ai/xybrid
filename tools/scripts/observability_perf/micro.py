"""Micro benchmarks of the observability packages: runs the Rust Criterion
benches of ``crates/xybrid-observability/benches`` and the Node benchmark of
``bindings/observability-js/scripts/bench.mjs``, then collects Criterion's
estimates and samples, the benches' JSON side reports and the JS results into
``micro-report.json`` / ``micro-report.md``.

Run through ``run_perf.py micro`` (see README.md). ``--collect-only`` re-reads
existing Criterion output instead of running anything.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import perf_common as pc

REPO = pc.REPO
JS_DIR = REPO / "bindings" / "observability-js"
RUST_BENCHES = ("stream_overhead", "content_pipeline", "saturation")
CRITERION_GROUPS = ("stream_op", "chunk", "content", "burst")
ENABLED_SETUPS = ("memory_metadata", "memory_content", "otlp_blackhole", "otlp_blocked")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir")
    parser.add_argument("--benches", default=",".join(RUST_BENCHES))
    parser.add_argument("--quick", action="store_true", help="Criterion --quick and shorter side reports")
    parser.add_argument("--criterion-args", default="", help="extra Criterion arguments, e.g. '--measurement-time 3'")
    parser.add_argument("--profile", default="bench", help="cargo profile (default bench; dev for smoke runs)")
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--skip-js", action="store_true")
    parser.add_argument("--collect-only", action="store_true", help="collect existing Criterion output only")
    parser.add_argument("--e2e-report", help="an e2e report.json: enables thresholds relative to token time")
    parser.add_argument("--thresholds")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_logged(command: List[str], cwd: Path, log: Path, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    pc.stage("running: " + " ".join(command))
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode
    return {"command": command, "exit_code": code, "seconds": time.monotonic() - started, "log": str(log)}


def run_rust(benches: List[str], out_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    env = dict(os.environ, XYBRID_OBS_BENCH_DIR=str(out_dir / "rust"))
    # The benches read no OTEL_* variable, but keep runs comparable.
    env = {k: v for k, v in env.items() if not k.startswith("OTEL_")}
    results = []
    for bench in benches:
        command = ["cargo", "bench", "-p", "xybrid-observability", "--features", "test-util", "--bench", bench]
        if args.profile != "bench":
            command += ["--profile", args.profile]
        extra = (["--quick"] if args.quick else []) + shlex.split(args.criterion_args)
        command += ["--"] + extra
        results.append(run_logged(command, REPO, out_dir / "logs" / f"{bench}.log", env))
    return results


# What the JS benchmark needs; `pnpm exec tsdown` builds dist/ without bun
# (the package's `build` script runs through bun).
JS_SETUP = "(cd bindings/observability-js && pnpm install --frozen-lockfile && pnpm exec tsdown)"


def js_prerequisite_problem() -> Optional[str]:
    if shutil.which("node") is None:
        return "node is not on PATH"
    if not (JS_DIR / "node_modules").exists() or not (JS_DIR / "dist" / "index.js").exists():
        return f"bindings/observability-js is not built; run {JS_SETUP}"
    return None


def run_js(out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Runs the Node benchmark. When it cannot run, the result is a failed run
    (exit 127) rather than a silent skip: J1/J2 would otherwise just vanish
    from the report. ``--skip-js`` is the way to leave it out."""
    problem = js_prerequisite_problem()
    if problem is not None:
        pc.stage(f"error: the JS benchmark cannot run: {problem} (or pass --skip-js)")
        return {"command": ["node", "scripts/bench.mjs"], "exit_code": 127, "seconds": 0.0, "log": None,
                "error": problem}
    command = ["node", "--expose-gc", "scripts/bench.mjs", "--out", str(out_dir / "js" / "observability-js-bench.json")]
    if args.quick:
        command.append("--quick")
    return run_logged(command, JS_DIR, out_dir / "logs" / "js-bench.log")


# ---------------------------------------------------------------------------
# Collecting
# ---------------------------------------------------------------------------

def collect_criterion(criterion_dir: Path, since: Optional[float] = None) -> List[Dict[str, Any]]:
    """Every benchmark of the observability groups under ``criterion_dir``:
    Criterion's estimates (ns per iteration) and the per-sample
    per-iteration times. Entries written before ``since`` are marked stale."""
    out = []
    if not criterion_dir.exists():
        return out
    for meta_path in sorted(criterion_dir.rglob("new/benchmark.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("group_id") not in CRITERION_GROUPS:
            continue
        new = meta_path.parent
        estimates = json.loads((new / "estimates.json").read_text(encoding="utf-8"))
        sample = json.loads((new / "sample.json").read_text(encoding="utf-8"))
        per_iter = [t / i for t, i in zip(sample.get("times", []), sample.get("iters", [])) if i]

        def est(name: str) -> Optional[Dict[str, float]]:
            e = estimates.get(name)
            if not e:
                return None
            ci = e.get("confidence_interval") or {}
            return {"point": e.get("point_estimate"), "low": ci.get("lower_bound"), "high": ci.get("upper_bound")}

        out.append({
            "id": meta.get("full_id"),
            "group": meta.get("group_id"),
            "function": meta.get("function_id"),
            "value": meta.get("value_str"),
            "throughput": meta.get("throughput"),
            "mean_ns": est("mean"),
            "median_ns": est("median"),
            "slope_ns": est("slope"),
            "std_dev_ns": est("std_dev"),
            "samples": len(per_iter),
            "sample_per_iteration_ns": pc.distribution(per_iter),
            "stale": bool(since is not None and (new / "estimates.json").stat().st_mtime < since - 1),
            "path": str(new.relative_to(criterion_dir)),
        })
    return out


def load_json_dir(directory: Path) -> Dict[str, Any]:
    if not directory.exists():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))}


def load_fresh_json(path: Path, since: float) -> Optional[Any]:
    """``path`` when it was written by this run (at or after ``since``), else
    None: a reused ``--out-dir`` must not let an earlier run's results stand
    in for a benchmark this run left out or failed to produce."""
    if not path.exists() or path.stat().st_mtime < since - 1:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Deriving
# ---------------------------------------------------------------------------

def _mean(index: Dict[str, Dict[str, Any]], key: str) -> Optional[float]:
    entry = index.get(key)
    if not entry or entry.get("stale") or not entry.get("mean_ns"):
        return None
    return entry["mean_ns"]["point"]


def derive_rust(criterion: List[Dict[str, Any]], side: Dict[str, Any]) -> Dict[str, Any]:
    """Overheads against the disabled observer, from Criterion means."""
    index = {c["id"]: c for c in criterion}
    setups = ["disabled", *ENABLED_SETUPS]
    out: Dict[str, Any] = {"per_setup": {}}
    base_op1 = _mean(index, "stream_op/disabled/1")
    base_op256 = _mean(index, "stream_op/disabled/256")
    base_chunk = _mean(index, "chunk/disabled")
    base_slope = (base_op256 - base_op1) / 255 if base_op1 is not None and base_op256 is not None else None
    tails = {s["setup"]: s for s in ((side.get("stream_overhead") or {}).get("setups") or [])}
    for setup in setups:
        op1 = _mean(index, f"stream_op/{setup}/1")
        op32 = _mean(index, f"stream_op/{setup}/32")
        op256 = _mean(index, f"stream_op/{setup}/256")
        chunk = _mean(index, f"chunk/{setup}")
        slope = (op256 - op1) / 255 if op1 is not None and op256 is not None else None
        tail = tails.get(setup) or {}
        out["per_setup"][setup] = {
            "op_1_chunk_ns": op1,
            "op_32_chunks_ns": op32,
            "op_256_chunks_ns": op256,
            "per_chunk_slope_ns": slope,
            "chunk_call_ns": chunk,
            "fixed_op_overhead_ns": None if op1 is None or base_op1 is None else op1 - base_op1,
            "per_chunk_overhead_ns": None if slope is None or base_slope is None else slope - base_slope,
            "chunk_call_overhead_ns": None if chunk is None or base_chunk is None else chunk - base_chunk,
            "op_256_overhead_ns": None if op256 is None or base_op256 is None else op256 - base_op256,
            "record_chunk_tail_ns": tail.get("record_chunk_ns"),
            "whole_op_tail_ns": tail.get("whole_op_ns"),
        }
    # Content rows: Criterion means joined with the side report's individually
    # timed calls and checked outcomes (either may be missing).
    rows: Dict[Any, Dict[str, Any]] = {}
    for c in criterion:
        if c["group"] == "content" and not c["stale"]:
            rows[(c["function"], c["value"])] = {
                "policy": c["function"], "workload": c["value"],
                "mean_ns": (c["mean_ns"] or {}).get("point"), "throughput": c["throughput"],
            }
    side_content = side.get("content_pipeline") or {}
    outcomes = {(o["policy"], o["workload"]): o["outcome"] for o in side_content.get("outcomes") or []}
    for m in side_content.get("measurements") or []:
        key = (m["policy"], m["workload"])
        row = rows.setdefault(key, {"policy": key[0], "workload": key[1], "mean_ns": None, "throughput": None})
        row["tail_ns"] = m.get("record_input_messages_ns")
    for key, row in rows.items():
        row.setdefault("tail_ns", None)
        row["outcome"] = outcomes.get(key)
    content = list(rows.values())
    out["content"] = content
    out["saturation"] = [
        {"scenario": s["scenario"], **s.get("summary", {})}
        for s in ((side.get("saturation") or {}).get("scenarios") or [])
    ]
    return out


def derive_js(js: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not js:
        return None
    results = js.get("results") or {}
    ops = {(r["setup"], r["chunks"]): r for r in results.get("stream_op") or []}
    chunks = {r["setup"]: r for r in results.get("chunk") or []}

    def p50(r: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        return None if not r else (r.get(key) or {}).get("p50")

    base1, base256 = p50(ops.get(("disabled", 1)), "per_op_ns"), p50(ops.get(("disabled", 256)), "per_op_ns")
    base_chunk = p50(chunks.get("disabled"), "per_call_ns")
    base_slope = (base256 - base1) / 255 if base1 is not None and base256 is not None else None
    per = {}
    for setup in sorted({s for s, _ in ops}):
        o1, o256 = p50(ops.get((setup, 1)), "per_op_ns"), p50(ops.get((setup, 256)), "per_op_ns")
        slope = (o256 - o1) / 255 if o1 is not None and o256 is not None else None
        c = p50(chunks.get(setup), "per_call_ns")
        per[setup] = {
            "op_1_chunk_ns": o1, "op_256_chunks_ns": o256, "per_chunk_slope_ns": slope,
            "fixed_op_overhead_ns": None if o1 is None or base1 is None else o1 - base1,
            "per_chunk_overhead_ns": None if slope is None or base_slope is None else slope - base_slope,
            "chunk_call_overhead_ns": None if c is None or base_chunk is None else c - base_chunk,
            "record_chunk_tail_ns": (chunks.get(setup) or {}).get("per_call_ns"),
        }
    return {"per_setup": per, "content": results.get("content") or []}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def e2e_reference(e2e_report: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Per-token time and time to first callback of the ``off`` mode (ns)."""
    if not e2e_report:
        return {"token_ns": None, "ttft_ns": None}
    off = ((e2e_report.get("analysis") or {}).get("modes") or {}).get("off") or {}
    rm = off.get("run_metrics") or {}
    gap = (rm.get("callback_gap_p50_ms") or {}).get("p50")
    first = (rm.get("first_callback_ms") or {}).get("p50")
    return {"token_ns": None if gap is None else gap * 1e6, "ttft_ns": None if first is None else first * 1e6}


def evaluate(thresholds: Dict[str, Any], rust: Dict[str, Any], js: Optional[Dict[str, Any]],
             reference: Dict[str, Optional[float]], js_requested: bool = True) -> Dict[str, Any]:
    results = []

    def add(item: Dict[str, Any], subject: str, observed: Optional[float], within: Optional[bool], **extra: Any) -> None:
        results.append({"id": item["id"], "kind": item["kind"], "subject": subject, "bound": item["bound"],
                        "observed": observed, "within": within, "rationale": item.get("rationale"), **extra})

    for item in thresholds.get("micro", []):
        kind, bound = item["kind"], item["bound"]
        source = js if item.get("package") == "js" else rust
        if source is None:
            if item.get("package") == "js" and js_requested:
                # Listed as not run, so a missing JS result cannot pass for a clean report.
                add(item, "JS benchmark", None, None, not_run="the JS benchmark produced no results")
            continue
        per = source.get("per_setup") or {}
        if kind in ("chunk_overhead_vs_token", "op_overhead_vs_ttft"):
            ref = reference["token_ns"] if kind == "chunk_overhead_vs_token" else reference["ttft_ns"]
            field = "per_chunk_overhead_ns" if kind == "chunk_overhead_vs_token" else "fixed_op_overhead_ns"
            for setup in item["setups"]:
                value = (per.get(setup) or {}).get(field)
                pct = None if value is None or not ref else value / ref * 100.0
                within = None if pct is None else pct <= bound["max_pct"]
                add(item, setup, pct, within, unit="% of reference", value_ns=value, reference_ns=ref)
        elif kind == "chunk_tail_ns":
            for setup in item["setups"]:
                tail = (per.get(setup) or {}).get("record_chunk_tail_ns") or {}
                value = tail.get(item.get("statistic", "p99"))
                add(item, setup, value, pc.evaluate_bound(value, bound), unit="ns", statistic=item.get("statistic", "p99"))
        elif kind == "content_tail_ns":
            rows = source.get("content") or []
            for workload in item["workloads"]:
                row = next((r for r in rows if r.get("policy") == item["policy"] and r.get("workload") == workload), None)
                tail = (row or {}).get("tail_ns") or (row or {}).get("per_call_ns") or {}
                value = tail.get(item.get("statistic", "p99"))
                add(item, f"{item['policy']}/{workload}", value, pc.evaluate_bound(value, bound), unit="ns",
                    statistic=item.get("statistic", "p99"))
        elif kind == "saturation_bounds":
            for s in source.get("saturation") or []:
                held = s.get("bounds_held")
                add(item, s["scenario"], held, None if held is None else bool(held))
        elif kind == "saturation_p99_ratio":
            scenarios = {s["scenario"]: s for s in source.get("saturation") or []}
            ref = ((scenarios.get(item["reference"]) or {}).get("op_ns_p99") or {}).get("p50")
            for name in item["scenarios"]:
                value = ((scenarios.get(name) or {}).get("op_ns_p99") or {}).get("p50")
                ratio = None if value is None or not ref else value / ref
                add(item, f"{name} vs {item['reference']}", ratio,
                    None if ratio is None else ratio <= bound["max_ratio"], unit="ratio of p99 op latency")
    return {"status": thresholds.get("status", "proposed"), "results": results}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def render_markdown(report: Dict[str, Any]) -> str:
    env = report["environment"]
    lines = ["# Observability micro benchmarks", ""]
    lines.append(f"Generated {env.get('captured_at')} on {env.get('chip')} ({env.get('platform')}), "
                 f"commit {(env.get('source') or {}).get('commit')}"
                 f"{' (dirty)' if (env.get('source') or {}).get('dirty') else ''}. "
                 f"Thresholds are **{report['thresholds']['status']}**.")
    lines.append("")
    lines.append(f"Rust benches: cargo profile `{report.get('cargo_profile')}`, crate build "
                 f"{', '.join(report.get('rust_build') or []) or 'unknown'}"
                 f"{', Criterion --quick' if report.get('quick') else ''}. Only `bench`-profile "
                 "(optimized) full runs are comparable with the proposed absolute bounds.")
    rust = report.get("rust") or {}
    lines += ["", "## Rust: streaming operation (Criterion means, ns)", ""]
    rows = []
    for setup, d in (rust.get("per_setup") or {}).items():
        tail = d.get("record_chunk_tail_ns") or {}
        rows.append((setup, d["op_1_chunk_ns"], d["op_256_chunks_ns"], d["per_chunk_slope_ns"], d["chunk_call_ns"],
                     d["fixed_op_overhead_ns"], d["per_chunk_overhead_ns"], tail.get("p50"), tail.get("p99"),
                     tail.get("p999"), tail.get("max")))
    lines.append(pc.table(["setup", "op N=1", "op N=256", "per-chunk slope", "record_chunk", "fixed op overhead",
                           "per-chunk overhead", "chunk p50", "chunk p99", "chunk p99.9", "chunk max"], rows))
    lines += ["", "## Rust: content pipeline (record_input_messages, ns)", ""]
    rows = []
    for row in rust.get("content") or []:
        tail = row.get("tail_ns") or {}
        rows.append((row["policy"], row["workload"], row.get("outcome"), row["mean_ns"], tail.get("p50"),
                     tail.get("p99"), tail.get("max")))
    lines.append(pc.table(["policy", "workload", "outcome", "mean", "p50", "p99", "max"], rows))
    lines += ["", "## Rust: bursts and saturation (medians over repetitions)", ""]
    rows = []
    for s in rust.get("saturation") or []:
        rows.append((s["scenario"], (s.get("wall_ms") or {}).get("p50"), (s.get("op_ns_p99") or {}).get("p50"),
                     (s.get("op_ns_max") or {}).get("p50"), (s.get("dropped") or {}).get("p50"),
                     pc.mib((s.get("peak_buffered_bytes_sampled") or {}).get("max")), s.get("bounds_held")))
    lines.append(pc.table(["scenario", "wall ms", "op p99 ns", "op max ns", "dropped", "peak buffered MiB (max)",
                           "bounds held"], rows))
    js = report.get("js")
    if js:
        lines += ["", "## JS (Node, p50 of samples, ns)", ""]
        rows = []
        for setup, d in js["per_setup"].items():
            tail = d.get("record_chunk_tail_ns") or {}
            rows.append((setup, d["op_1_chunk_ns"], d["op_256_chunks_ns"], d["per_chunk_slope_ns"],
                         d["fixed_op_overhead_ns"], d["per_chunk_overhead_ns"], tail.get("p50"), tail.get("p99")))
        lines.append(pc.table(["setup", "op N=1", "op N=256", "per-chunk slope", "fixed op overhead",
                               "per-chunk overhead", "recordChunk p50", "recordChunk p99"], rows))
        rows = [(c.get("policy"), c.get("workload"), (c.get("per_call_ns") or {}).get("p50"),
                 (c.get("per_call_ns") or {}).get("p99"), c.get("outcome")) for c in js.get("content") or []]
        lines += ["", pc.table(["policy", "workload", "p50 ns", "p99 ns", "outcome"], rows)]
    lines += ["", f"## Thresholds ({report['thresholds']['status']})", ""]
    ref = report.get("reference") or {}
    lines.append(f"Reference from the e2e `off` mode: token time {pc.fmt(ref.get('token_ns'))} ns, "
                 f"time to first callback {pc.fmt(ref.get('ttft_ns'))} ns"
                 f"{'' if ref.get('token_ns') else ' (no --e2e-report: relative rows are not evaluated)'}.")
    lines.append("")
    rows = [(r["id"], r["subject"], json.dumps(r["bound"]), r["observed"], r.get("unit"),
             "not run" if r.get("not_run") else r["within"])
            for r in report["thresholds"]["results"]]
    lines.append(pc.table(["id", "subject", "bound", "observed", "unit", "within"], rows))
    stale = [c["id"] for c in report.get("criterion") or [] if c.get("stale")]
    if stale:
        lines += ["", f"Stale Criterion entries (older than this run, excluded): {', '.join(stale)}"]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, e2e_report: Optional[Path] = None,
                 thresholds_path: Optional[Path] = None) -> Dict[str, Any]:
    raw = json.loads((out_dir / "raw.json").read_text(encoding="utf-8"))
    thresholds = pc.load_thresholds(thresholds_path)
    e2e = json.loads(e2e_report.read_text(encoding="utf-8")) if e2e_report else None
    rust = derive_rust(raw.get("criterion") or [], raw.get("side_reports") or {})
    js = derive_js(raw.get("js"))
    reference = e2e_reference(e2e)
    report = {
        "schema": "xybrid-observability-perf/micro/v1",
        "environment": raw["environment"],
        "cargo_profile": raw.get("cargo_profile"),
        "quick": raw.get("quick"),
        "rust_build": sorted({str((r.get("environment") or {}).get("profile"))
                              for r in (raw.get("side_reports") or {}).values()}),
        "runs": raw.get("runs"),
        "reference": reference,
        "e2e_report": str(e2e_report) if e2e_report else None,
        "rust": rust,
        "js": js,
        "criterion": raw.get("criterion"),
        "side_reports": raw.get("side_reports"),
        "js_raw": raw.get("js"),
        "thresholds": evaluate(thresholds, rust, js, reference, bool(raw.get("js_requested", True))),
    }
    pc.write_json(out_dir / "micro-report.json", report)
    (out_dir / "micro-report.md").write_text(render_markdown(report), encoding="utf-8")
    pc.stage(f"wrote {out_dir / 'micro-report.json'} and micro-report.md")
    return report


def main_from_args(args: argparse.Namespace) -> int:
    # Absolute: the children run elsewhere (cargo bench binaries in the crate,
    # Node in bindings/observability-js) and write under this directory.
    out_dir = (Path(args.out_dir) if args.out_dir
               else pc.timestamp_dir(pc.default_target_dir() / "observability-perf", "micro")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pc.stage(f"output: {out_dir}")
    started = time.time()
    runs: List[Dict[str, Any]] = []
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    if not args.collect_only:
        if not args.skip_rust:
            runs += run_rust(benches, out_dir, args)
        if not args.skip_js:
            runs.append(run_js(out_dir, args))
    js_path = out_dir / "js" / "observability-js-bench.json"
    if args.collect_only:
        side_reports = load_json_dir(out_dir / "rust")
        js = json.loads(js_path.read_text(encoding="utf-8")) if js_path.exists() else None
    else:
        # Only what the benchmarks selected for this run wrote during it.
        fresh = {} if args.skip_rust else {
            name: load_fresh_json(out_dir / "rust" / f"{name}.json", started) for name in benches}
        side_reports = {name: report for name, report in fresh.items() if report is not None}
        js = None if args.skip_js else load_fresh_json(js_path, started)
    raw = {
        "schema": "xybrid-observability-perf/micro-raw/v1",
        "environment": pc.environment(),
        "cargo_profile": None if args.collect_only or args.skip_rust else args.profile,
        "quick": bool(args.quick),
        "runs": runs,
        "criterion": collect_criterion(pc.default_target_dir() / "criterion",
                                       None if args.collect_only else started),
        "side_reports": side_reports,
        "js": js,
        "js_requested": not args.skip_js,
    }
    pc.write_json(out_dir / "raw.json", raw)
    report = write_report(out_dir, Path(args.e2e_report) if args.e2e_report else None,
                          Path(args.thresholds) if args.thresholds else None)
    failed = [r for r in runs if r["exit_code"] != 0]
    thresholds_failed = pc.failed_gates(report["thresholds"])
    print(json.dumps({"out_dir": str(out_dir), "runs": len(runs), "failed": len(failed),
                      "thresholds_status": report["thresholds"]["status"], "thresholds_failed": thresholds_failed}))
    return 1 if failed or thresholds_failed else 0
