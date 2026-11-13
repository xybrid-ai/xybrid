"""Tests for the observability performance harness.

tools/scripts/observability_perf/ runs the end-to-end overhead measurements
(a real model per telemetry mode, interleaved) and collects the micro
benchmarks. These tests cover what can go wrong without a model or a Mac:
the /usr/bin/time parser, the interleaved schedule, the statistics and
threshold evaluation, the model pin, a whole e2e run against the fake
driver (sinks, blocked endpoint, report), the Criterion collector, and that
the driver, the harness and the Rust/JS workloads stay in step.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HARNESS = Path(__file__).resolve().parents[1] / "observability_perf"
sys.path.insert(0, str(HARNESS))

import fake_driver  # noqa: E402
import micro  # noqa: E402
import perf_common as pc  # noqa: E402
import run_perf  # noqa: E402

REPO = HARNESS.parents[2]
FAKE_DRIVER = HARNESS / "fake_driver.py"

# What `/usr/bin/time -l` appends on macOS, after the program's own stderr.
BSD_TIME = """llama_model_loader: loaded meta data with 32 key-value pairs
ggml_metal_init: found device: Apple M5 Max
        4.21 real         3.10 user         0.52 sys
           812646400  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               52011  page reclaims
                  12  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                  18  voluntary context switches
                1412  involuntary context switches
         41234567890  instructions retired
         12345678901  cycles elapsed
           698765312  peak memory footprint
"""


class BsdTimeTests(unittest.TestCase):
    def test_parses_the_block_after_program_output(self):
        parsed = run_perf.parse_bsd_time(BSD_TIME)
        self.assertEqual(parsed["real_s"], 4.21)
        self.assertEqual(parsed["user_s"], 3.10)
        self.assertEqual(parsed["maximum resident set size"], 812646400)
        self.assertEqual(parsed["peak memory footprint"], 698765312)
        self.assertEqual(parsed["involuntary context switches"], 1412)

    def test_uses_the_last_block(self):
        twice = BSD_TIME.replace("4.21 real", "9.99 real") + BSD_TIME
        self.assertEqual(run_perf.parse_bsd_time(twice)["real_s"], 4.21)

    def test_missing_block_is_none(self):
        self.assertIsNone(run_perf.parse_bsd_time("no timing here\n"))


class ScheduleTests(unittest.TestCase):
    MODES = ("off", "metadata", "content", "blocked")

    def test_every_round_runs_every_mode_once(self):
        order = run_perf.schedule(self.MODES, 10)
        self.assertEqual(len(order), 40)
        for r in range(10):
            self.assertEqual(sorted(m for rr, m in order if rr == r), sorted(self.MODES))

    def test_default_order_puts_each_mode_in_each_position_equally(self):
        order = run_perf.schedule(self.MODES, 8)
        positions = {m: [] for m in self.MODES}
        for index, (_, mode) in enumerate(order):
            positions[mode].append(index % len(self.MODES))
        for mode, seen in positions.items():
            self.assertEqual(sorted(seen), sorted([0, 1, 2, 3] * 2), mode)

    def test_default_order_balances_predecessors_within_rounds(self):
        # Williams design: within the rounds of one cycle every mode directly
        # follows every other mode exactly once (twice for an odd count).
        for modes in (self.MODES, ("off", "content", "blocked"), ("off", "content")):
            cycle = len(run_perf.williams_rows(len(modes)))
            order = run_perf.schedule(modes, cycle)
            pairs = {}
            for r in range(cycle):
                row = [m for rr, m in order if rr == r]
                for previous, mode in zip(row, row[1:]):
                    pairs[(previous, mode)] = pairs.get((previous, mode), 0) + 1
            expected = 2 if len(modes) % 2 else 1
            self.assertEqual(set(pairs), {(a, b) for a in modes for b in modes if a != b}, modes)
            self.assertEqual(set(pairs.values()), {expected}, modes)

    def test_the_baseline_has_no_fixed_predecessor(self):
        # A rotation put `off` right after `blocked` 7 of 9 times in 10 rounds.
        counts = run_perf.predecessor_counts(run_perf.schedule(self.MODES, 10))
        self.assertEqual(sum(counts["off"].values()), 9)
        self.assertLessEqual(max(counts["off"].values()), 4, counts["off"])
        for mode, seen in counts.items():
            self.assertEqual(set(seen), set(self.MODES) - {mode}, mode)

    def test_seeded_shuffle_is_reproducible(self):
        a = run_perf.schedule(self.MODES, 5, seed=7)
        self.assertEqual(a, run_perf.schedule(self.MODES, 5, seed=7))
        self.assertNotEqual(a, run_perf.schedule(self.MODES, 5))


class StatisticsTests(unittest.TestCase):
    def test_distribution_ignores_missing_and_non_finite(self):
        d = pc.distribution([3.0, None, 1.0, float("nan"), 2.0])
        self.assertEqual((d["n"], d["min"], d["p50"], d["max"]), (3, 1.0, 2.0, 3.0))
        self.assertEqual(pc.distribution([]), {"n": 0})

    def test_nearest_rank_percentile(self):
        values = list(range(1, 101))
        self.assertEqual(pc.percentile(values, 0.99), 99)
        self.assertEqual(pc.percentile(values, 0.0), 1)
        self.assertEqual(pc.percentile(values, 1.0), 100)

    def test_bootstrap_delta_is_deterministic_and_brackets_the_point(self):
        base = [100.0 + (i % 7) for i in range(30)]
        cand = [103.0 + (i % 7) for i in range(30)]
        a = pc.bootstrap_median_delta(base, cand)
        self.assertEqual(a, pc.bootstrap_median_delta(base, cand))
        self.assertAlmostEqual(a["delta"], 3.0)
        self.assertLessEqual(a["ci95_low"], a["delta"])
        self.assertGreaterEqual(a["ci95_high"], a["delta"])
        self.assertIsNone(pc.bootstrap_median_delta([], cand))

    def test_bounds_take_the_larger_allowance(self):
        bound = {"max_abs": 2.0, "max_pct": 2.0}
        self.assertTrue(pc.evaluate_bound(1.9, bound, reference=10.0))
        self.assertFalse(pc.evaluate_bound(2.1, bound, reference=10.0))
        self.assertTrue(pc.evaluate_bound(3.9, bound, reference=200.0))
        self.assertIsNone(pc.evaluate_bound(None, bound, 1.0))
        self.assertIsNone(pc.evaluate_bound(1.0, {"max_pct": 1.0}, None))

    def test_failed_gates_only_count_agreed_failures(self):
        rows = [{"id": "E1", "mode": "content", "within": False}, {"id": "E2", "mode": "content", "within": True},
                {"id": "M1", "subject": "memory_content", "within": None},
                {"id": "J1", "subject": "JS benchmark", "within": None, "not_run": "no results"}]
        self.assertEqual(pc.failed_gates({"status": "proposed", "results": rows}), [])
        self.assertEqual(pc.failed_gates({"status": "agreed", "results": rows}), ["E1/content", "J1/JS benchmark"])

    def test_thresholds_file_is_agreed_with_known_kinds(self):
        thresholds = pc.load_thresholds()
        self.assertEqual(thresholds["status"], "agreed")
        self.assertTrue(thresholds["agreed"]["on"])
        metrics = set(run_perf.RUN_METRICS) | set(run_perf.PROCESS_METRICS)
        for item in thresholds["e2e"] + thresholds["e2e_absolute"]:
            self.assertIn(item["metric"], metrics, item["id"])
            self.assertTrue(item.get("rationale"), item["id"])
        kinds = {"chunk_overhead_vs_token", "op_overhead_vs_ttft", "chunk_tail_ns", "content_tail_ns",
                 "saturation_bounds", "saturation_p99_ratio"}
        for item in thresholds["micro"]:
            self.assertIn(item["kind"], kinds, item["id"])
            self.assertTrue(item.get("rationale"), item["id"])


def write_metadata(model_dir, model_file):
    """A model_metadata.json whose GGUF template loads ``model_file``."""
    template = {"type": "Gguf", "model_file": model_file, "context_length": 4096}
    (model_dir / "model_metadata.json").write_text(json.dumps({"model_id": "m", "execution_template": template}))


class ModelPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_metadata(self.dir, run_perf.MODEL["file"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_wrong_bytes_under_the_pinned_name_fail(self):
        (self.dir / run_perf.MODEL["file"]).write_bytes(b"not the model")
        with self.assertRaises(SystemExit):
            run_perf.verify_model(self.dir, allow_unpinned=False)
        info = run_perf.verify_model(self.dir, allow_unpinned=True)
        self.assertFalse(info["pinned"])
        self.assertEqual(info["sha256"], hashlib.sha256(b"not the model").hexdigest())

    def test_the_pinned_bytes_pass(self):
        (self.dir / run_perf.MODEL["file"]).write_bytes(b"the model")
        with mock.patch.dict(run_perf.MODEL, sha256=hashlib.sha256(b"the model").hexdigest()):
            info = run_perf.verify_model(self.dir, allow_unpinned=False)
        self.assertEqual((info["pinned"], info["file"], info["revision"]),
                         (True, run_perf.MODEL["file"], run_perf.MODEL["revision"]))

    def test_the_file_the_metadata_names_is_the_one_recorded(self):
        # The driver loads execution_template.model_file; a.gguf sorts first.
        (self.dir / "a.gguf").write_bytes(b"a projector, say")
        (self.dir / "b.gguf").write_bytes(b"the model that runs")
        write_metadata(self.dir, "b.gguf")
        info = run_perf.verify_model(self.dir, allow_unpinned=True)
        self.assertEqual((info["file"], info["sha256"], info["pinned"]),
                         ("b.gguf", hashlib.sha256(b"the model that runs").hexdigest(), False))

    def test_the_pinned_file_counts_only_when_the_metadata_names_it(self):
        (self.dir / run_perf.MODEL["file"]).write_bytes(b"the model")
        (self.dir / "other.gguf").write_bytes(b"another model")
        write_metadata(self.dir, "other.gguf")
        with mock.patch.dict(run_perf.MODEL, sha256=hashlib.sha256(b"the model").hexdigest()):
            with self.assertRaises(SystemExit):
                run_perf.verify_model(self.dir, allow_unpinned=False)
            info = run_perf.verify_model(self.dir, allow_unpinned=True)
        self.assertEqual((info["file"], info["pinned"]), ("other.gguf", False))
        self.assertNotIn("revision", info)

    def test_metadata_without_a_loadable_file_fails(self):
        (self.dir / "a.gguf").write_bytes(b"gguf")
        for metadata in ("{}", json.dumps({"execution_template": {"type": "Gguf"}}), "not json"):
            (self.dir / "model_metadata.json").write_text(metadata)
            with self.subTest(metadata=metadata), self.assertRaises(SystemExit):
                run_perf.verify_model(self.dir, allow_unpinned=True)
        write_metadata(self.dir, "missing.gguf")
        with self.assertRaisesRegex(SystemExit, "missing.gguf"):
            run_perf.verify_model(self.dir, allow_unpinned=True)

    def test_missing_metadata_fails(self):
        (self.dir / "model_metadata.json").unlink()
        with self.assertRaises(SystemExit):
            run_perf.verify_model(self.dir, allow_unpinned=True)

    def test_fetch_verifies_and_writes_metadata(self):
        source = self.dir / "source.gguf"
        source.write_bytes(b"gguf bytes")
        spec = dict(run_perf.MODEL, file="m.gguf", sha256=hashlib.sha256(b"gguf bytes").hexdigest(), size=10)
        dest = self.dir / "dest"
        path = run_perf.fetch_model(dest, spec, url=source.as_uri())
        self.assertEqual(path.read_bytes(), b"gguf bytes")
        metadata = json.loads((dest / "model_metadata.json").read_text())
        self.assertEqual(metadata["model_id"], run_perf.MODEL["id"])
        self.assertEqual(metadata["execution_template"]["model_file"], run_perf.MODEL["file"])

    def test_fetch_rejects_other_bytes(self):
        source = self.dir / "source.gguf"
        source.write_bytes(b"tampered")
        spec = dict(run_perf.MODEL, file="m.gguf", sha256="0" * 64, size=8)
        with self.assertRaises(SystemExit):
            run_perf.fetch_model(self.dir / "dest", spec, url=source.as_uri())
        self.assertFalse((self.dir / "dest" / "m.gguf").exists())
        self.assertFalse((self.dir / "dest" / "m.gguf.part").exists())


class EnvironmentTests(unittest.TestCase):
    def test_driver_env_strips_otel_and_hosted_credentials(self):
        with mock.patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "x", "XYBRID_API_KEY": "k",
                                          "XYBRID_STAGING_API_KEY": "s", "XYBRID_RESOURCE_TELEMETRY": "on",
                                          "KEEP_ME": "1"}):
            env = run_perf.driver_env()
        self.assertNotIn("OTEL_EXPORTER_OTLP_ENDPOINT", env)
        self.assertNotIn("XYBRID_API_KEY", env)
        self.assertNotIn("XYBRID_STAGING_API_KEY", env)
        self.assertNotIn("XYBRID_RESOURCE_TELEMETRY", env)
        self.assertEqual(env["KEEP_ME"], "1")


class EndToEndFakeDriverTests(unittest.TestCase):
    """A whole e2e run against fake_driver.py: sinks, blocked endpoint, report."""

    def run_e2e(self, out_dir, model_dir, extra_env=None, modes="off,metadata,content,blocked", thresholds=None):
        # The fake driver is deliberately slower when enabled, so these runs
        # use a proposed copy of the thresholds unless a test passes its own.
        argv = ["e2e", "--driver-cmd", f"{sys.executable} {FAKE_DRIVER}", "--model-dir", str(model_dir),
                "--allow-unpinned-model", "--rounds", "2", "--runs-per-process", "2", "--max-tokens", "16",
                "--cooldown-s", "0", "--modes", modes, "--out-dir", str(out_dir),
                "--thresholds", str(thresholds or self.proposed)]
        with mock.patch.dict(os.environ, extra_env or {}):
            return run_perf.main(argv)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.model = root / "model"
        self.model.mkdir()
        write_metadata(self.model, "tiny.gguf")
        (self.model / "tiny.gguf").write_bytes(b"\0" * 64)
        self.out = root / "out"
        self.proposed = root / "thresholds-proposed.json"
        self.proposed.write_text(json.dumps({**pc.load_thresholds(), "status": "proposed"}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_agreed_thresholds_gate_the_exit_status(self):
        # The fake driver is slower when enabled: a 0% tokens/s bound fails
        # it, a 1000% bound does not. Proposed bounds never gate.
        def thresholds(status, max_pct):
            path = Path(self.tmp.name) / f"thresholds-{status}-{max_pct}.json"
            path.write_text(json.dumps({"status": status, "e2e": [
                {"id": "T1", "metric": "runtime_tokens_per_s", "modes": ["content"], "bound": {"max_pct": max_pct},
                 "rationale": "test"}], "e2e_absolute": [
                # `off` never shuts a pipeline down: a row with nothing to evaluate.
                {"id": "A1", "metric": "shutdown_ms", "modes": ["off"], "statistic": "max",
                 "bound": {"max_abs": 1.0}, "rationale": "test"}], "micro": []}))
            return path

        # report.md says what the exit status did: a gate under agreed bounds,
        # a measurement under proposed ones. An absolute row (A1) gates on its
        # statistic, not on a median delta, and its bound cell names it.
        agreed_header = ("each row's `within` is a gate (the median delta against `off`, or for an absolute row "
                         "its statistic against the limit)")
        proposed_header = "results below are measurements, not pass/fail verdicts"
        unevaluated = "1 row(s) had nothing to evaluate (`within` –) and do not gate."
        absolute = ("`within` uses the median delta (an absolute row, which has a `statistic` in its bound, "
                    "compares that statistic over processes with `max_abs`)")
        absolute_bound = '| A1 | off | shutdown_ms | {"statistic": "max", "max_abs": 1.0} |'
        for status, max_pct, code, failed, says, never in (
                ("agreed", 0.0, 1, ["T1/content"], [agreed_header, "**Failed gates:** `T1/content`.", unevaluated,
                                                    absolute, absolute_bound],
                 [proposed_header, "Every gate held.", "before treating any row as a gate", "(the median delta)"]),
                ("agreed", 1000.0, 0, [], [agreed_header, "Every gate held.", unevaluated, absolute, absolute_bound],
                 [proposed_header, "Failed gates", "before treating any row as a gate", "(the median delta)"]),
                ("proposed", 0.0, 0, [], [proposed_header, "before treating any row as a gate", absolute,
                                          absolute_bound],
                 [agreed_header, "Every gate held.", "Failed gates", unevaluated])):
            with self.subTest(status=status, max_pct=max_pct):
                out = Path(self.tmp.name) / f"out-{status}-{max_pct}"
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(self.run_e2e(out, self.model, modes="off,content",
                                                  thresholds=thresholds(status, max_pct)), code)
                summary = json.loads(stdout.getvalue().strip().splitlines()[-1])
                self.assertEqual((summary["thresholds_status"], summary["thresholds_failed"]), (status, failed))
                markdown = (out / "report.md").read_text()
                for text in says:
                    self.assertIn(text, markdown)
                for text in never:
                    self.assertNotIn(text, markdown)

    def test_the_shipped_agreed_bounds_hold_for_the_fake_driver(self):
        # fake_driver.py's docstring dry run gates on the shipped thresholds.json;
        # a fake slower than those bounds made it exit 1 on every run. The rows
        # on synthesized timings (and on the fake's resident memory) are kept;
        # the shutdown rows (e2e_absolute) time real requests and are left out.
        shipped = pc.load_thresholds()
        self.assertEqual(shipped["status"], "agreed")
        path = Path(self.tmp.name) / "thresholds-shipped-relative.json"
        path.write_text(json.dumps({**shipped, "e2e_absolute": []}))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = self.run_e2e(self.out, self.model, thresholds=path)
        summary = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertEqual((code, summary["thresholds_status"], summary["thresholds_failed"]), (0, "agreed", []))
        # Held, not skipped: every timing row was evaluated for every mode.
        report = json.loads((self.out / "report.json").read_text())
        held = {(r["id"], r["mode"]) for r in report["thresholds"]["results"] if r["within"] is True}
        self.assertEqual({row for row in held if row[0] in ("E1", "E2", "E3", "E4", "E5")},
                         {(row, mode) for row in ("E1", "E2", "E3", "E4", "E5")
                          for mode in ("metadata", "content", "blocked")})

    def test_report_covers_every_mode_and_passes_sanity(self):
        self.assertEqual(self.run_e2e(self.out, self.model), 0)
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["schema"], "xybrid-observability-perf/e2e/v1")
        modes = report["analysis"]["modes"]
        self.assertEqual(set(modes), {"off", "metadata", "content", "blocked"})
        for mode, summary in modes.items():
            self.assertEqual((summary["processes"], summary["runs"]), (2, 4), mode)
            self.assertEqual(summary["run_metrics"]["callbacks"]["p50"], 16, mode)
            self.assertIsNotNone(summary["process_metrics"]["peak_rss_mib"].get("p50"), mode)
        failed = [c for c in report["analysis"]["sanity"] if c["ok"] is not True]
        self.assertEqual(failed, [])
        self.assertEqual(set(report["analysis"]["comparisons"]), {"metadata", "content", "blocked"})
        tps = report["analysis"]["comparisons"]["content"]["runtime_tokens_per_s"]
        self.assertLess(tps["delta"], 0, "the fake driver is slower when enabled")
        self.assertEqual(report["thresholds"]["status"], "proposed")
        ids = {r["id"] for r in report["thresholds"]["results"]}
        self.assertTrue({"E1", "E5", "E6", "E8"} <= ids)
        self.assertEqual(report["workload"]["schedule"][:8], ["0:off", "0:metadata", "0:blocked", "0:content",
                                                              "1:metadata", "1:content", "1:off", "1:blocked"])
        self.assertEqual(report["workload"]["predecessors"]["off"], {"content": 1})
        self.assertTrue(report["blocked_endpoint"]["connections"] >= 2)
        self.assertTrue((self.out / "report.md").read_text().startswith("# Observability end-to-end overhead"))
        # `report` re-renders from raw.json.
        (self.out / "report.json").unlink()
        self.assertEqual(run_perf.main(["report", "--e2e", str(self.out)]), 0)
        self.assertTrue((self.out / "report.json").exists())

    def test_unequal_callback_counts_fail_the_sanity_check_and_the_run(self):
        code = self.run_e2e(self.out, self.model, {"FAKE_DRIVER_SHORT": "content"}, modes="off,content")
        self.assertEqual(code, 1)
        report = json.loads((self.out / "report.json").read_text())
        check = next(c for c in report["analysis"]["sanity"] if c["check"].startswith("greedy decoding"))
        self.assertFalse(check["ok"])
        self.assertTrue(all(p["ok"] for p in json.loads((self.out / "raw.json").read_text())["processes"]))

    def test_a_failing_mode_is_counted_and_fails_the_run(self):
        code = self.run_e2e(self.out, self.model, {"FAKE_DRIVER_FAIL": "content"}, modes="off,content")
        self.assertEqual(code, 1)
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["analysis"]["modes"]["content"]["failed_processes"], 2)
        check = next(c for c in report["analysis"]["sanity"] if c["check"].startswith("content: every process"))
        self.assertFalse(check["ok"])


class SinkTests(unittest.TestCase):
    def test_a_sink_that_does_not_start_is_stopped(self):
        # Nothing holds a Sink whose constructor raised, so it must stop its
        # own server: cmd_e2e's finally only closes the sinks it was given.
        with tempfile.TemporaryDirectory() as tmp:
            stalled = Path(tmp) / "stalled.py"
            stalled.write_text("import time\ntime.sleep(60)\n")  # alive, never ready
            created = []
            real_popen = subprocess.Popen

            def popen(*args, **kwargs):
                created.append(real_popen(*args, **kwargs))
                return created[-1]

            try:
                with mock.patch.object(run_perf, "CAPTURE_SERVER", stalled), \
                        mock.patch.object(run_perf.subprocess, "Popen", side_effect=popen):
                    with self.assertRaisesRegex(SystemExit, "did not start"):
                        run_perf.Sink(Path(tmp), "content", timeout=0.5)
                self.assertEqual(len(created), 1)
                self.assertIsNotNone(created[0].poll(), "the stalled sink is still running")
            finally:
                for process in created:
                    if process.poll() is None:
                        process.kill()
                        process.wait()


class FakeDriverTests(unittest.TestCase):
    """The e2e tests above expect `content` to be slower than `off`; the fake
    driver's timings must make that certain, not likely."""

    def test_every_enabled_run_is_slower_than_every_off_run(self):
        # A 0.4% slowdown under +-2% jitter left the order to chance: the
        # 0%-bound gate passed and the delta came out positive about 1 run in 10.
        slowest_off = fake_driver.SLOWDOWN["off"] * (1 + fake_driver.JITTER)
        for mode in ("metadata", "content", "blocked"):
            self.assertGreater(fake_driver.SLOWDOWN[mode] * (1 - fake_driver.JITTER), slowest_off, mode)
        self.assertGreater(fake_driver.SLOWDOWN["content"] * (1 - fake_driver.JITTER),
                           fake_driver.SLOWDOWN["metadata"] * (1 + fake_driver.JITTER))

    def test_the_same_invocation_gives_the_same_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for attempt in ("a", "b"):
                out = Path(tmp) / attempt / "r00-off.json"
                out.parent.mkdir()
                subprocess.run([sys.executable, str(FAKE_DRIVER), "--model-dir", tmp, "--mode", "off",
                                "--out", str(out), "--runs", "2", "--max-tokens", "16"], check=True, timeout=60)
                outputs.append(json.loads(out.read_text())["runs"])
            self.assertEqual(outputs[0], outputs[1])
            self.assertNotEqual(outputs[0][0]["callback_gaps_ms"], outputs[0][1]["callback_gaps_ms"])


def criterion_entry(root, group, function, value, mean_ns, iters=(10.0, 20.0)):
    parts = [group] + ([function] if function else []) + [value]
    new = root.joinpath(*parts, "new")
    new.mkdir(parents=True)
    full = "/".join(parts)
    (new / "benchmark.json").write_text(json.dumps({
        "group_id": group, "function_id": function, "value_str": value, "full_id": full, "throughput": None}))
    est = {"point_estimate": mean_ns, "confidence_interval": {"lower_bound": mean_ns * 0.99,
                                                               "upper_bound": mean_ns * 1.01}}
    (new / "estimates.json").write_text(json.dumps({"mean": est, "median": est, "slope": None, "std_dev": est}))
    (new / "sample.json").write_text(json.dumps({"iters": list(iters), "times": [mean_ns * i for i in iters]}))


class SanityTests(unittest.TestCase):
    @staticmethod
    def summary(callbacks):
        return {mode: {"runs": 4, "failed_processes": 0, "processes": 2,
                       "run_metrics": {"callbacks": {"min": low, "max": high}},
                       "telemetry": {"traces_accepted": 0, "traces_exported": 0, "traces_failed": 0,
                                     "traces_dropped": 0}}
                for mode, (low, high) in callbacks.items()}

    def callback_check(self, callbacks):
        checks = run_perf.sanity_checks({"sinks": {}}, self.summary(callbacks))
        return next(c for c in checks if c["check"].startswith("greedy decoding"))["ok"]

    def test_callback_counts_must_agree_within_and_across_modes(self):
        self.assertTrue(self.callback_check({"off": (128, 128), "content": (128, 128)}))
        self.assertFalse(self.callback_check({"off": (64, 128), "content": (64, 128)}))  # varies within modes
        self.assertFalse(self.callback_check({"off": (128, 128), "content": (127, 127)}))


class ProcessTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "process groups are POSIX")
    def test_a_timeout_kills_the_wrapped_driver_too(self):
        # On macOS the measured process is /usr/bin/time, which waits on the
        # driver; killing only the wrapper left the driver running.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_file = root / "driver.pid"
            wrapper = ("import subprocess, sys, pathlib; "
                       "driver = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                       f"pathlib.Path({str(pid_file)!r}).write_text(str(driver.pid)); driver.wait()")
            with mock.patch.object(run_perf, "use_bsd_time", return_value=False):
                result = run_perf.run_measured([sys.executable, "-c", wrapper], root / "out", root / "err",
                                               timeout=2, env=dict(os.environ))
            self.assertTrue(result["timed_out"])
            driver = int(pid_file.read_text())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and process_runs(driver):
                time.sleep(0.05)
            self.assertFalse(process_runs(driver), f"driver {driver} outlived the timeout")


def process_runs(pid):
    """Whether ``pid`` still runs (a zombie awaiting its reaper does not)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split(") ")[-1].startswith("Z"))


class MicroCollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        means = {"disabled": (1000.0, 11200.0, 40.0), "memory_content": (8000.0, 94500.0, 330.0)}
        for setup, (op1, op256, chunk) in means.items():
            criterion_entry(self.root, "stream_op", setup, "1", op1)
            criterion_entry(self.root, "stream_op", setup, "256", op256)
            criterion_entry(self.root, "chunk", None, setup, chunk)
        criterion_entry(self.root, "content", "rules", "msg_15k", 20000.0)
        criterion_entry(self.root, "unrelated", "x", "y", 1.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_collects_only_observability_groups(self):
        entries = micro.collect_criterion(self.root)
        self.assertEqual({e["group"] for e in entries}, {"stream_op", "chunk", "content"})
        chunk = next(e for e in entries if e["id"] == "chunk/memory_content")
        self.assertEqual(chunk["mean_ns"]["point"], 330.0)
        self.assertEqual(chunk["sample_per_iteration_ns"]["p50"], 330.0)
        self.assertFalse(chunk["stale"])

    def test_entries_older_than_the_run_are_stale(self):
        entries = micro.collect_criterion(self.root, since=4102444800.0)  # 2100-01-01
        self.assertTrue(all(e["stale"] for e in entries))

    def test_overheads_are_against_the_disabled_observer(self):
        rust = micro.derive_rust(micro.collect_criterion(self.root), {})
        d = rust["per_setup"]["memory_content"]
        self.assertAlmostEqual(d["fixed_op_overhead_ns"], 7000.0)
        self.assertAlmostEqual(d["per_chunk_slope_ns"], (94500.0 - 8000.0) / 255)
        self.assertAlmostEqual(d["per_chunk_overhead_ns"], (94500.0 - 8000.0) / 255 - (11200.0 - 1000.0) / 255)
        self.assertAlmostEqual(d["chunk_call_overhead_ns"], 290.0)

    def test_relative_thresholds_need_the_e2e_reference(self):
        rust = micro.derive_rust(micro.collect_criterion(self.root), {})
        thresholds = {"status": "proposed", "micro": [
            {"id": "M1", "kind": "chunk_overhead_vs_token", "setups": ["memory_content"], "bound": {"max_pct": 0.1}}]}
        none = micro.evaluate(thresholds, rust, None, micro.e2e_reference(None))
        self.assertIsNone(none["results"][0]["within"])
        e2e = {"analysis": {"modes": {"off": {"run_metrics": {
            "callback_gap_p50_ms": {"p50": 5.0}, "first_callback_ms": {"p50": 40.0}}}}}}
        evaluated = micro.evaluate(thresholds, rust, None, micro.e2e_reference(e2e))
        result = evaluated["results"][0]
        self.assertTrue(result["within"])
        self.assertAlmostEqual(result["observed"], rust["per_setup"]["memory_content"]["per_chunk_overhead_ns"] / 5e6 * 100)

    def test_a_js_benchmark_that_cannot_run_fails_and_is_listed_as_not_run(self):
        args = mock.Mock(quick=True)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(micro, "js_prerequisite_problem", return_value="not built"):
            result = micro.run_js(Path(tmp), args)
        self.assertEqual(result["exit_code"], 127)
        self.assertIn("tsdown", micro.JS_SETUP)
        self.assertIn("pnpm exec tsdown)", (HARNESS / "README.md").read_text())  # no bun needed
        thresholds = {"status": "proposed", "micro": [
            {"id": "J1", "kind": "chunk_overhead_vs_token", "package": "js", "setups": ["memory_content"],
             "bound": {"max_pct": 0.1}}]}
        rust = micro.derive_rust([], {})
        rows = micro.evaluate(thresholds, rust, None, micro.e2e_reference(None))["results"]
        self.assertEqual([(r["id"], r["within"], bool(r.get("not_run"))) for r in rows], [("J1", None, True)])
        skipped = micro.evaluate(thresholds, rust, None, micro.e2e_reference(None), js_requested=False)
        self.assertEqual(skipped["results"], [])

    def micro_args(self, *argv):
        parser = argparse.ArgumentParser()
        micro.add_arguments(parser)
        return parser.parse_args(list(argv))

    def test_a_relative_out_dir_reaches_the_children_as_an_absolute_path(self):
        seen = {}

        def fake_js(out_dir, args):
            seen["js"] = out_dir
            target = out_dir / "js" / "observability-js-bench.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"results": {}}))
            return {"command": ["node"], "exit_code": 0, "seconds": 0.0, "log": None}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(micro, "run_js", side_effect=fake_js), \
                mock.patch.object(micro, "collect_criterion", return_value=[]):
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    micro.main_from_args(self.micro_args("--out-dir", "results", "--skip-rust"))
            finally:
                os.chdir(cwd)
            self.assertTrue(seen["js"].is_absolute())
            self.assertEqual(seen["js"], (Path(tmp) / "results").resolve())
            raw = json.loads((Path(tmp) / "results" / "raw.json").read_text())
            self.assertEqual(raw["js"], {"results": {}})

    def test_a_reused_out_dir_never_lends_earlier_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            (out / "rust").mkdir(parents=True)
            (out / "js").mkdir()
            old = time.time() - 3600
            for name in ("stream_overhead", "content_pipeline", "saturation"):
                path = out / "rust" / f"{name}.json"
                path.write_text(json.dumps({"bench": name, "old": True}))
                os.utime(path, (old, old))
            js = out / "js" / "observability-js-bench.json"
            js.write_text(json.dumps({"results": {}, "old": True}))
            os.utime(js, (old, old))

            def fake_rust(benches, out_dir, args):
                # stream_overhead writes a fresh report; content_pipeline fails to.
                (out_dir / "rust" / "stream_overhead.json").write_text(json.dumps({"bench": "stream_overhead"}))
                return [{"command": ["cargo"], "exit_code": 0, "seconds": 0.0, "log": None}]

            with mock.patch.object(micro, "run_rust", side_effect=fake_rust), \
                    mock.patch.object(micro, "collect_criterion", return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()):
                micro.main_from_args(self.micro_args("--out-dir", str(out), "--skip-js",
                                                     "--benches", "stream_overhead,content_pipeline"))
            raw = json.loads((out / "raw.json").read_text())
            self.assertEqual(raw["side_reports"], {"stream_overhead": {"bench": "stream_overhead"}})
            self.assertIsNone(raw["js"])
            # --collect-only still reads whatever the directory holds.
            with mock.patch.object(micro, "collect_criterion", return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()):
                micro.main_from_args(self.micro_args("--out-dir", str(out), "--collect-only"))
            raw = json.loads((out / "raw.json").read_text())
            self.assertEqual(set(raw["side_reports"]), {"stream_overhead", "content_pipeline", "saturation"})
            self.assertTrue(raw["js"]["old"])

    def test_js_overheads(self):
        dist = lambda v: {"p50": v}  # noqa: E731
        js = {"results": {
            "stream_op": [{"setup": "disabled", "chunks": 1, "per_op_ns": dist(20000.0)},
                          {"setup": "disabled", "chunks": 256, "per_op_ns": dist(40400.0)},
                          {"setup": "memory_content", "chunks": 1, "per_op_ns": dist(45000.0)},
                          {"setup": "memory_content", "chunks": 256, "per_op_ns": dist(300000.0)}],
            "chunk": [{"setup": "disabled", "per_call_ns": dist(80.0)},
                      {"setup": "memory_content", "per_call_ns": dist(950.0)}],
            "content": []}}
        derived = micro.derive_js(js)["per_setup"]["memory_content"]
        self.assertAlmostEqual(derived["fixed_op_overhead_ns"], 25000.0)
        self.assertAlmostEqual(derived["per_chunk_overhead_ns"], (300000.0 - 45000.0) / 255 - (40400.0 - 20000.0) / 255)
        self.assertAlmostEqual(derived["chunk_call_overhead_ns"], 870.0)


class ConsistencyTests(unittest.TestCase):
    """The driver writes what the harness reads, and the Rust and JS benches
    measure the same workloads."""

    def test_driver_writes_every_field_the_harness_reads(self):
        driver = (REPO / "crates" / "xybrid-sdk" / "examples" / "observability_perf.rs").read_text()
        fake = FAKE_DRIVER.read_text()
        for field in ("wall_ms", "callbacks", "first_callback_ms", "callback_gaps_ms", "decode_callbacks_per_s",
                      "finish_reason", "ttft_ms", "tokens_per_second", "tokens_generated", "load_ms",
                      "shutdown_ms", "shutdown_result", "stats_after_shutdown", "stats_before_shutdown"):
            self.assertIn(f'"{field}"', driver, field)
            self.assertIn(f'"{field}"', fake, field)
        for flag in ("model-dir", "mode", "endpoint", "out", "runs", "warmup", "max-tokens", "prompt-bytes",
                     "shutdown-timeout-ms"):
            self.assertIn(f'"{flag}"', driver, flag)
            self.assertIn(f'"--{flag}"', fake, flag)
        for mode in run_perf.MODES:
            self.assertIn(f'"{mode}"', driver, mode)

    def test_rust_and_js_content_workloads_match(self):
        rust = (REPO / "crates" / "xybrid-observability" / "benches" / "content_pipeline.rs").read_text()
        js = (REPO / "bindings" / "observability-js" / "scripts" / "bench.mjs").read_text()
        ids = lambda text: set(re.findall(r'"((?:msg|hist)_[a-z0-9_]+)"', text))  # noqa: E731
        self.assertEqual(ids(rust), ids(js))
        self.assertGreaterEqual(len(ids(rust)), 11)
        policies = lambda text: set(re.findall(r'"(metadata_only|default|rules|hook_and_rules)"', text))  # noqa: E731
        self.assertEqual(policies(rust), policies(js))

    def test_bench_targets_are_declared(self):
        manifest = (REPO / "crates" / "xybrid-observability" / "Cargo.toml").read_text()
        for bench in micro.RUST_BENCHES:
            self.assertIn(f'name = "{bench}"', manifest)
            self.assertTrue((REPO / "crates" / "xybrid-observability" / "benches" / f"{bench}.rs").exists())


if __name__ == "__main__":
    unittest.main()
