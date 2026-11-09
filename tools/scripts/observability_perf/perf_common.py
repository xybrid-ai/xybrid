"""Shared helpers of the observability performance harness: environment
capture, distribution statistics, bootstrap comparisons, threshold
evaluation and Markdown tables. Standard library only, Python 3.9+.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _run(command: Sequence[str], cwd: Optional[Path] = None, timeout: float = 20) -> Optional[str]:
    """Output of ``command`` (stripped), or ``None`` when it cannot run."""
    if shutil.which(command[0]) is None and not Path(command[0]).exists():
        return None
    try:
        out = subprocess.run(list(command), cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _sysctl(name: str) -> Optional[str]:
    return _run(["sysctl", "-n", name])


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def environment(repo: Path = REPO) -> Dict[str, Any]:
    """Machine, OS, toolchain and source identity of this run."""
    env: Dict[str, Any] = {
        "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "logical_cpus": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        env.update({
            "chip": _sysctl("machdep.cpu.brand_string"),
            "model": _sysctl("hw.model"),
            "physical_cpus": _int(_sysctl("hw.physicalcpu")),
            "performance_cores": _int(_sysctl("hw.perflevel0.physicalcpu")),
            "efficiency_cores": _int(_sysctl("hw.perflevel1.physicalcpu")),
            "memory_bytes": _int(_sysctl("hw.memsize")),
            "macos": _run(["sw_vers", "-productVersion"]),
            "macos_build": _run(["sw_vers", "-buildVersion"]),
            # Power source and thermal state change sustained performance.
            "power": _first_line(_run(["pmset", "-g", "batt"])),
            "low_power_mode": _low_power_mode(_run(["pmset", "-g"])),
            "thermal": _run(["pmset", "-g", "therm"]),
        })
    else:
        env.update({
            "chip": _cpuinfo_model(),
            "memory_bytes": _meminfo_total(),
            "os_release": _os_release(),
            "kernel": platform.release(),
        })
    env["toolchain"] = {
        "rustc": _run(["rustc", "-V"]),
        "rustc_verbose": _run(["rustc", "-vV"]),
        "cargo": _run(["cargo", "-V"]),
        "node": _run(["node", "--version"]),
    }
    env["source"] = source_identity(repo)
    return env


def source_identity(repo: Path = REPO) -> Dict[str, Any]:
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=repo),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo),
        "dirty": None if status is None else bool(status),
        "dirty_files": None if not status else len(status.splitlines()),
    }


def _first_line(text: Optional[str]) -> Optional[str]:
    return text.splitlines()[0] if text else None


def _low_power_mode(text: Optional[str]) -> Optional[bool]:
    if not text:
        return None
    match = re.search(r"lowpowermode\s+(\d)", text)
    return None if match is None else match.group(1) == "1"


def _cpuinfo_model() -> Optional[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _meminfo_total() -> Optional[int]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _os_release() -> Optional[str]:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of already sorted values (``q`` in 0..1)."""
    if not sorted_values:
        return math.nan
    rank = math.ceil(q * len(sorted_values))
    return sorted_values[min(max(rank, 1), len(sorted_values)) - 1]


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def distribution(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    """n, min, p5, p50, p90, p95, p99, max, mean and stdev of the finite values."""
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return {"n": 0}
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1) if len(clean) > 1 else 0.0
    return {
        "n": len(clean),
        "min": clean[0],
        "p5": percentile(clean, 0.05),
        "p50": median(clean),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": clean[-1],
        "mean": mean,
        "stdev": math.sqrt(var),
    }


def bootstrap_median_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    resamples: int = 2000,
    seed: int = 20261109,
) -> Optional[Dict[str, float]]:
    """Median of ``candidate`` minus median of ``baseline`` with a percentile
    bootstrap 95% interval (units resampled independently, fixed seed)."""
    base = [float(v) for v in baseline if v is not None and math.isfinite(float(v))]
    cand = [float(v) for v in candidate if v is not None and math.isfinite(float(v))]
    if not base or not cand:
        return None
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        b = [base[rng.randrange(len(base))] for _ in base]
        c = [cand[rng.randrange(len(cand))] for _ in cand]
        deltas.append(median(c) - median(b))
    deltas.sort()
    point = median(cand) - median(base)
    base_median = median(base)
    return {
        "baseline_median": base_median,
        "median": median(cand),
        "delta": point,
        "delta_pct": (point / base_median * 100.0) if base_median else math.nan,
        "ci95_low": percentile(deltas, 0.025),
        "ci95_high": percentile(deltas, 0.975),
        "n_baseline": len(base),
        "n": len(cand),
    }


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def load_thresholds(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or HERE / "thresholds.json"
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_bound(observed: Optional[float], bound: Dict[str, Any], reference: Optional[float] = None) -> Optional[bool]:
    """Whether ``observed`` is within ``bound``: ``max_abs`` (same unit),
    ``max_pct`` (of ``reference``), and/or ``min_abs``. With both ``max_abs``
    and ``max_pct`` the larger allowance applies (``max(abs, pct)``). ``None``
    when nothing can be evaluated."""
    if observed is None or (isinstance(observed, float) and not math.isfinite(observed)):
        return None
    allowances = []
    if "max_abs" in bound:
        allowances.append(float(bound["max_abs"]))
    if "max_pct" in bound and reference is not None and math.isfinite(reference):
        allowances.append(abs(reference) * float(bound["max_pct"]) / 100.0)
    if "min_abs" in bound and observed < float(bound["min_abs"]):
        return False
    if not allowances:
        return None if "min_abs" not in bound else True
    return observed <= max(allowances)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "–"
        if abs(value) >= 1e5:
            return f"{value:,.0f}"
        if value != 0 and abs(value) < 10 ** -digits:
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    if isinstance(value, int) and abs(value) >= 100_000:
        return f"{value:,}"
    return str(value)


def mib(value: Optional[float]) -> Optional[float]:
    return None if value is None else value / (1024 * 1024)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(c) for c in row) + " |")
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def timestamp_dir(root: Path, prefix: str) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{prefix}-{stamp}"


def default_target_dir() -> Path:
    """The workspace target directory (``CARGO_TARGET_DIR`` honored)."""
    configured = os.environ.get("CARGO_TARGET_DIR")
    return Path(configured) if configured else REPO / "target"


def stage(message: str) -> None:
    print(f"[observability-perf] {message}", file=sys.stderr, flush=True)


Getter = Callable[[Dict[str, Any]], Optional[float]]


def collect(items: Iterable[Dict[str, Any]], getter: Getter) -> List[float]:
    out = []
    for item in items:
        try:
            value = getter(item)
        except (KeyError, TypeError, IndexError):
            value = None
        if value is not None and math.isfinite(float(value)):
            out.append(float(value))
    return out


def pairs(mapping: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return sorted(mapping.items())
