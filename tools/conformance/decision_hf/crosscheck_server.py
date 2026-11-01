#!/usr/bin/env python3
"""Send the cases to a running llama-server and compare its answers with the HF reference's (stdlib only).

    crosscheck_server.py --url http://127.0.0.1:8089 --hf OBSERVATIONS --temperature 1.3373980522155762 \
        --artifact d1-2b-f16 --settings cpu-t4-fa-off --label 'what ran' --out FILE CASE...

For each case the exact `request_text` goes to `POST /v1/systemone`; the status and body are kept. Where both
references answered, per question: whether the answer agrees (choice, median level, yes/no at one half), the largest
probability difference, and the largest difference of logit differences. The server returns probabilities only, but
T * log(p_i / p_top) recovers its max-pooled logit differences s_i - s_top exactly enough (T: the temperature the
server used, `--temperature`), which compare with the reference's `hf.evaluate.scores` (raw logits, max-pooled) and
are independent of either temperature. Options whose server probability underflowed to 0 are skipped. This is a
black-box cross-check of the difference matrix and a G1 feasibility measurement, not the upstream oracle.

The output records what it was computed from: the artifact and settings ids the server ran (as given), this script's
sha256, and the sha256 of every HF observation it compared, which provenance.py checks against the committed files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def f32(bits: str) -> float:
    return struct.unpack(">f", bytes.fromhex(bits[2:]))[0]


def post(url: str, body: str, timeout: float) -> tuple[int, Any, float]:
    request = urllib.request.Request(
        url, data=body.encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    seconds = time.perf_counter() - started
    try:
        return status, json.loads(raw), seconds
    except ValueError:
        return status, raw.decode("utf-8", "replace"), seconds


def logit_gaps(probs: list[float], temperature: float) -> list[float | None]:
    top = max(probs)
    return [temperature * math.log(p / top) if p > 0 else None for p in probs]


def compare_question(hf_answer: dict, hf_scores: list[str], server: dict, temperature: float) -> dict[str, Any]:
    row: dict[str, Any] = {"type": hf_answer["type"]}
    s = [f32(v) for v in hf_scores]
    if hf_answer["type"] == "noul":
        p_server = float(server["noul"])
        p_hf = float(hf_answer["noul"])
        row["same_answer"] = (p_server >= 0.5) == (p_hf >= 0.5)
        row["max_abs_p"] = abs(p_server - p_hf)
        if 0 < p_server < 1:
            row["max_abs_gap"] = abs(temperature * math.log(p_server / (1 - p_server)) - (s[0] - s[1]))
        return row
    keys = list(hf_answer["probabilities"])
    server_probs = [float(server["probabilities"][k]) for k in keys]
    hf_probs = [float(hf_answer["probabilities"][k]) for k in keys]
    if hf_answer["type"] == "choice":
        row["same_answer"] = server["choice"] == hf_answer["choice"]
    else:
        row["same_answer"] = float(server["score"]) == float(hf_answer["score"])
    row["max_abs_p"] = max(abs(a - b) for a, b in zip(server_probs, hf_probs))
    top = max(range(len(server_probs)), key=server_probs.__getitem__)
    gaps = logit_gaps(server_probs, temperature)
    diffs = [abs(g - (s[i] - s[top])) for i, g in enumerate(gaps) if g is not None]
    row["max_abs_gap"] = max(diffs) if diffs else None
    row["skipped_underflow"] = sum(g is None for g in gaps)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", type=Path, nargs="+")
    parser.add_argument("--url", default="http://127.0.0.1:8089")
    parser.add_argument("--hf", type=Path, required=True, help="the HF reference's observation directory")
    parser.add_argument("--temperature", type=float, required=True, help="the server's effective temperature")
    parser.add_argument("--artifact", required=True, help="the artifact id of the GGUF the server ran")
    parser.add_argument("--settings", required=True, help="the settings id the server ran under")
    parser.add_argument("--label", default="", help="how the server was built and started, recorded as given")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    files = []
    for path in args.cases:
        files.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    results = []
    compared: dict[str, str] = {}
    for path in files:
        case = json.loads(path.read_text(encoding="utf-8"))
        status, body, seconds = post(args.url.rstrip("/") + "/v1/systemone", case["request_text"], args.timeout)
        entry: dict[str, Any] = {"case": case["id"], "status": status, "seconds": round(seconds, 2), "body": body}
        hf_path = args.hf / f"{case['id']}.json"
        hf = json.loads(hf_path.read_text(encoding="utf-8")) if hf_path.is_file() else None
        if hf is not None:
            compared[case["id"]] = hashlib.sha256(hf_path.read_bytes()).hexdigest()
        hf_final = (hf or {}).get("hf", {}).get("finalize")
        if status == 200 and isinstance(body, dict) and hf_final and "answers" in hf_final:
            entry["same_usage"] = body["usage"]["input_tokens"] == hf_final["usage"]["input_tokens"]
            names = [p["question"] for p in hf["hf"]["prepare"]["passes"]]
            scores = hf["hf"]["evaluate"]["scores"]
            entry["questions"] = {
                name: compare_question(hf_final["answers"][name], scores[i], body["answers"][name], args.temperature)
                for i, name in enumerate(names)
                if name in body.get("answers", {})
            }
        elif hf is not None:
            refused = [s for s in ("parse", "prepare", "evaluate") if "refusal" in hf["hf"].get(s, {})]
            entry["hf"] = (
                f"refused at {refused[0]}" if refused else ("typed answer" if "typed" in hf["detail"] else "answered")
            )
        results.append(entry)
        questions = entry.get("questions", {})
        gaps = [q["max_abs_gap"] for q in questions.values() if q.get("max_abs_gap") is not None]
        summary = (
            f"usage={'same' if entry.get('same_usage') else 'DIFF'} answers="
            f"{'same' if all(q['same_answer'] for q in questions.values()) else 'CHANGED'} "
            f"max|dp|={max((q['max_abs_p'] for q in questions.values()), default=0):.3g} "
            f"max|dgap|={max(gaps, default=0):.3g}"
            if questions
            else f"hf {entry.get('hf', '-')}"
        )
        print(f"{case['id']:34s} {status:3d} {summary}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "server": args.label,
        "artifact": args.artifact,
        "settings": args.settings,
        "temperature": args.temperature,
        "recipe": {"crosscheck_server.py": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()},
        "observations": compared,
        "cases": results,
    }
    args.out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
