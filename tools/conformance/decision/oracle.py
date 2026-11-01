#!/usr/bin/env python3
"""Write the decision conformance cases from llama.cpp's own decision code.

Runs every case of cases.py through decision-oracle (the pinned upstream routines, see oracle.cpp)
and sends the same request to the stock llama-server built from the same sources, then writes
integration-tests/fixtures/decision/cases/<id>.json and oracle.json. With --check it writes
nothing and fails unless a fresh run gives the committed files, from the same recipe and build.
With --collect it runs any directory of schema-v1 case files on any artifact of artifacts.json
and writes the observations to --out.

Usage:
  tools/conformance/decision/oracle.py [--check] [--model PATH] [--download] [--settings ID] [--only ID]...
  tools/conformance/decision/oracle.py --collect DIR --artifact ID --settings ID --out DIR [--only ID]...

Official runs need executables built by build.sh from the pinned llama.cpp commit, in a clean
checkout, from the tool sources on disk; --dev runs anyway and marks the output. Stdlib only.
README.md next to this file explains the setup.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURES = REPO / "integration-tests" / "fixtures" / "decision"
MODELS = REPO / "integration-tests" / "fixtures" / "models"
ARTIFACT = "d1-2b-q4km"

# The llama.cpp commit official observations come from; move it only together with the
# vendor/llama-cpp gitlink, and regenerate (README.md, "When the llama.cpp pin moves").
PINNED_LLAMA_CPP = "8642541b255bfb5dc39cd351a178be09013a55d6"

# What an observation depends on: the recipe. The first four are compiled into the executables,
# whose stamped identity must hash them the same.
RECIPE = ("CMakeLists.txt", "identity.cmake", "oracle.cpp", "test_access.h", "build.sh", "cases.py", "oracle.py")
IDENTITY_MARKER = b"XYBRID_ORACLE_IDENTITY {"

TAGS = set("""
state-string state-empty state-null state-scalar state-object state-array state-messages
json-order json-float json-int json-unicode json-escapes json-literals json-depth
instructions-structured description-structured description-falsy description-null
choice-letter-keys choice-lowercase-keys choice-one-option choice-26 choice-27 choice-pool-fallback choice-max-options choice-order
noul-no-criteria noul-partial-criteria noul-null-criteria noul-order
score-2-levels score-10-levels score-null-level
escape-control escape-keys escape-overlap escape-key-collision escape-user-defined
multi-question budget transport media
math-reference math-median-b3 math-median-not-mode math-tie math-underflow math-neg-inf math-non-finite math-alias-max
math-temperature-bucket math-temperature-default math-score-bias math-media-temperature
refusal-load refusal-parse refusal-prepare refusal-evaluate refusal-finalize
divergence
""".split())

KINDS = set("""
invalid_json invalid_request duplicate_key unknown_field missing_state invalid_questions invalid_question
missing_instructions invalid_question_type invalid_criteria level_count too_many_options empty_question_id
empty_option_key media_unsupported spoken_instructions json_too_deep number_out_of_range escaped_key_collision
no_single_token_label budget_exceeded non_finite_scores unsupported_metadata invalid_metadata
""".split())

STAGES = ("load", "parse", "prepare", "evaluate", "finalize")
F32_SPECIAL = {"nan": "0x7fc00000", "inf": "0x7f800000", "-inf": "0xff800000"}


def die(msg):
    sys.exit(f"oracle.py: {msg}")


#
# JSON: numbers keep the text they were printed with
#

class Num:
    """A JSON float kept as printed: f64 values with 17 significant digits, legend values as upstream wrote them."""

    def __init__(self, text):
        self.text = text

    def __eq__(self, other):
        return isinstance(other, Num) and other.text == self.text


def load_json(text):
    return json.loads(text, parse_float=Num)


def dump(value, indent=0):
    """Two-space indentation; arrays of scalars on one line."""
    pad = "  " * indent
    if isinstance(value, Num):
        return value.text
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        raise TypeError("a float reached dump(): numbers must stay Num")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(v, (list, dict)) for v in value):
            return "[" + ", ".join(dump(v) for v in value) + "]"
        inner = ",\n".join(pad + "  " + dump(v, indent + 1) for v in value)
        return "[\n" + inner + "\n" + pad + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner = ",\n".join(pad + "  " + json.dumps(k, ensure_ascii=True) + ": " + dump(v, indent + 1) for k, v in value.items())
        return "{\n" + inner + "\n" + pad + "}"
    raise TypeError(f"cannot dump {type(value)}")


def plain(value):
    """Num -> float, for comparisons with what Python's json module reads."""
    if isinstance(value, Num):
        return float(value.text)
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def strict_equal(a, b):
    """JSON equality that tells 1 from 1.0 and keeps key order."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a.keys()) == list(b.keys()) and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def f32_hex(x):
    if isinstance(x, str):
        return F32_SPECIAL[x]
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#
# what runs: the executables' identity, the model, the settings
#

def binary_identity(path):
    """The identity identity.cmake stamped into an executable."""
    data = Path(path).read_bytes()
    start = data.find(IDENTITY_MARKER)
    if start < 0:
        die(f"{path} carries no build identity: rebuild with tools/conformance/decision/build.sh")
    start += len(IDENTITY_MARKER) - 1
    return json.loads(data[start:data.index(b"\0", start)].decode("utf-8"))


def gitlink_commit():
    """The llama.cpp commit this repository pins (its vendor/llama-cpp gitlink), or None if unreadable."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-tree", "HEAD", "vendor/llama-cpp"],
                             capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out[2] if len(out) >= 3 and out[0] == "160000" else None


def build_identity(bin_dir, dev):
    """Both executables' identity, and why they cannot produce official observations (empty if they can)."""
    oracle_id = binary_identity(bin_dir / "decision-oracle")
    server_id = binary_identity(bin_dir / "llama-server")
    if oracle_id != server_id:
        die("decision-oracle and llama-server were built from different identities: rebuild both with build.sh")
    problems = []
    source = oracle_id["llama_cpp"]
    if source["checkout"] != "git":
        problems.append(f"llama.cpp at {oracle_id['llama_cpp_dir']} is not a git checkout of its own ({source['checkout']})")
    elif source["commit"] != PINNED_LLAMA_CPP:
        problems.append(f"built from llama.cpp {source['commit']}, the pin is {PINNED_LLAMA_CPP}")
    if source["dirty"]:
        problems.append("built from a llama.cpp checkout with local changes")
    for name, digest in oracle_id["tool_files"].items():
        if digest != sha256_file(HERE / name):
            problems.append(f"{name} changed since the executables were built: run build.sh")
    pin = gitlink_commit()
    if pin is not None and pin != PINNED_LLAMA_CPP:
        problems.append(f"this repository pins llama.cpp {pin}, oracle.py {PINNED_LLAMA_CPP}: move both together")
    if problems and not dev:
        die("these executables cannot produce official observations:\n  " + "\n  ".join(problems)
            + "\n(--dev runs anyway and marks the output as a developer run)")
    return oracle_id, problems


def recorded_identity(identity):
    """The identity as oracle.json records it: the llama.cpp directory relative to the repository."""
    out = dict(identity)
    directory = Path(identity["llama_cpp_dir"])
    try:
        out["llama_cpp_dir"] = directory.relative_to(REPO).as_posix()
    except ValueError:
        out["llama_cpp_dir"] = str(directory)
    return out


def settings_args(settings):
    """llama-server arguments for a settings id of artifacts.json; decision-oracle parses the same list."""
    if settings["backend"] != "cpu":
        die(f"backend {settings['backend']}: only cpu settings are defined")
    return ["-c", str(settings["n_ctx"]), "-b", str(settings["n_batch"]), "-ub", str(settings["n_ubatch"]),
            "-np", str(settings["n_seq_max"]), "-t", str(settings["threads"]), "-tb", str(settings["threads"]),
            "-fa", "on" if settings["flash_attn"] else "off",
            # the CPU only: no layer offloaded, no offload device, no automatic fitting
            "-ngl", "0", "--device", "none", "-fit", "off"]


def check_context(facts, settings):
    """The context the oracle ran is the one the settings define, on the CPU only."""
    ctx = facts.get("context") or {}
    expected = {
        "n_ctx": settings["n_ctx"], "n_batch": settings["n_batch"], "n_ubatch": settings["n_ubatch"],
        "n_seq_max": settings["n_seq_max"], "n_outputs_max": settings["n_outputs_max"],
        "n_threads": settings["threads"], "n_threads_batch": settings["threads"],
        "flash_attn": "on" if settings["flash_attn"] else "off", "causal_attn": False, "memory": False,
        "n_gpu_layers": 0, "offload_devices": "none",
    }
    wrong = {k: (ctx.get(k), v) for k, v in expected.items() if ctx.get(k) != v}
    if wrong:
        die(f"the oracle's context is not the settings': {wrong}")
    if any(d["type"] != "cpu" for d in facts["devices"]) or facts["model_devices"]:
        die(f"the oracle did not run on the CPU only: {facts['devices']}, model devices {facts['model_devices']}")


def tool_env():
    """The environment both executables run in: no LLAMA_*, GGML_* or MTMD_* variable may change a setting."""
    return {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_", "MTMD_"))}


def resolve_model(args, artifacts, artifact_id):
    """The artifact's GGUF: --model, else integration-tests/fixtures/models/<id>/<file> (download.sh's place)."""
    art = artifacts["artifacts"].get(artifact_id)
    if art is None:
        die(f"artifact {artifact_id} is not in artifacts.json")
    manifest = json.loads((MODELS / "models.json").read_text())["models"].get(artifact_id)
    if manifest is not None:
        files = manifest.get("files", [])
        if len(files) != 1 or (files[0].get("url"), files[0].get("output"), files[0].get("sha256")) != (art["url"], art["file"], art["sha256"]):
            die(f"models.json and artifacts.json disagree on {artifact_id}")
    path = Path(args.model) if args.model else MODELS / artifact_id / art["file"]
    if not path.exists():
        if not args.download:
            die(f"no {path}: pass --model, or --download to fetch {art['url']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        print(f"downloading {art['url']}", file=sys.stderr)
        urllib.request.urlretrieve(art["url"], part)
        part.rename(path)
    if path.stat().st_size != art["bytes"]:
        die(f"{path} has {path.stat().st_size} bytes, artifacts.json pins {art['bytes']}")
    digest = sha256_file(path)
    if digest != art["sha256"]:
        die(f"{path} has sha256 {digest}, artifacts.json pins {art['sha256']}")
    return path


#
# running the oracle and the server
#

def run_oracle(binary, jobs, work, name, model_args):
    """jobs: [{id, request_text, select_logits?, has_media?}] -> the oracle's JSON output, or None if it failed."""
    inp = work / f"{name}-in.json"
    out = work / f"{name}-out.json"
    inp.write_text(json.dumps({"cases": jobs}, ensure_ascii=False), encoding="utf-8")
    with open(work / f"{name}.log", "w") as log:
        result = subprocess.run([str(binary), "run", "--input", str(inp), "--output", str(out)] + model_args,
                                stdout=log, stderr=subprocess.STDOUT, env=tool_env())
    if result.returncode != 0:
        return None
    return load_json(out.read_text(encoding="utf-8"))


def log_tail(work, name):
    lines = (work / f"{name}.log").read_text(errors="replace").strip().splitlines()
    return lines[-1] if lines else "no output"


def make_variant(binary, base, metadata, work):
    key = hashlib.sha256((str(base) + json.dumps(metadata, sort_keys=True)).encode()).hexdigest()[:16]
    path = work / f"variant-{key}.gguf"
    if path.exists():
        return path
    cmd = [str(binary), "variant", "--base", str(base), "--output", str(path)]
    for k, v in sorted(metadata.get("set", {}).items()):
        cmd += ["--set", f"{k}={v}"]
    for k in metadata.get("remove", []):
        cmd += ["--remove", k]
    for token, ttype in sorted(metadata.get("token_types", {}).items()):
        cmd += ["--token-type", f"{token}={ttype}"]
    subprocess.run(cmd, check=True, env=tool_env())
    return path


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ask_server(binary, model, args, jobs, work):
    """POST each request_text to a fresh llama-server; returns {id: (status, body)}."""
    devices = subprocess.run([str(binary), "--list-devices"], capture_output=True, text=True, env=tool_env()).stdout
    if devices.split() != ["Available", "devices:", "(none)"]:
        die(f"llama-server can reach a device other than the CPU:\n{devices}")
    port = free_port()
    log = open(work / "llama-server.log", "w")
    proc = subprocess.Popen([str(binary), "-m", str(model)] + args + ["--host", "127.0.0.1", "--port", str(port)],
                            stdout=log, stderr=subprocess.STDOUT, env=tool_env())
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 300
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if proc.poll() is not None or time.time() > deadline:
                die(f"llama-server did not start, see {work / 'llama-server.log'}")
            time.sleep(0.5)
        answers = {}
        for job in jobs:
            req = urllib.request.Request(base + "/v1/systemone", data=job["request_text"].encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    answers[job["id"]] = (r.status, r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                answers[job["id"]] = (e.code, e.read().decode("utf-8"))
        return answers
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


class Job:
    """One request to run: from cases.py or from a case file."""

    def __init__(self, id, request_text, metadata=None, has_media=False, select_logits=None, synthetic=None):
        self.id = id
        self.request_text = request_text
        self.metadata = metadata
        self.has_media = has_media
        self.select_logits = select_logits  # per pass, f32 bit patterns
        self.synthetic = synthetic          # per pass, one score per option (cases.py only)

    def sendable(self):
        return not self.metadata and self.select_logits is None and self.synthetic is None and not self.has_media


def expand_synthetic(job, prepared):
    """Synthetic select logits: one score per option, a group's other labels one lower."""
    if job.select_logits is not None or job.synthetic is None:
        return job.select_logits
    if prepared is None:
        return []
    tasks = prepared["prepare"]["tasks"]
    if len(tasks) != len(job.synthetic):
        die(f"{job.id}: synthetic scores for {len(job.synthetic)} passes, the request has {len(tasks)}")
    out = []
    for task, scores in zip(tasks, job.synthetic):
        if len(scores) != len(task["label_groups"]):
            die(f"{job.id}: {len(scores)} scores for {len(task['label_groups'])} options")
        logits = []
        for score, n in zip(scores, task["label_groups"]):
            logits.append(f32_hex(score))
            for _ in range(n - 1):
                logits.append(f32_hex(score if isinstance(score, str) else float(score) - 1.0))
        out.append(logits)
    return out


def run_jobs(jobs, model, settings_id, settings, bin_dir, work):
    """Every job through the oracle, and the sendable ones through the server.

    Returns ({id: result or {"error": message}}, the model's facts, {id: (status, body)}).
    """
    oracle_bin, server_bin = bin_dir / "decision-oracle", bin_dir / "llama-server"
    args = settings_args(settings)

    variants = {job.id: make_variant(oracle_bin, model, job.metadata, work) for job in jobs if job.metadata}

    # synthetic scores per option need each pass's group sizes: a prepare-only run first
    prepared, groups = {}, {}
    for job in jobs:
        if job.synthetic is not None and job.select_logits is None:
            groups.setdefault(str(variants.get(job.id, model)), []).append(job)
    for target, group in groups.items():
        out = run_oracle(oracle_bin, [{"id": j.id, "request_text": j.request_text} for j in group], work,
                         f"prepare-{hashlib.sha256(target.encode()).hexdigest()[:8]}", ["--vocab-only", target])
        if out is None:
            die(f"decision-oracle failed to prepare synthetic cases, see {work}")
        for j, r in zip(group, out.get("cases", [])):
            prepared[j.id] = r if "tasks" in r.get("prepare", {}) else None

    runs = {}
    for job in jobs:
        key = ("variant", str(variants[job.id])) if job.id in variants else ("model", str(model))
        runs.setdefault(key, []).append(job)
    stamped = binary_identity(oracle_bin)
    results, facts = {}, None
    for (kind, target), group in runs.items():
        model_args = ["--", "-m", target] + args if kind == "model" else ["--vocab-only", target]
        name = "model" if kind == "model" else f"variant-{Path(target).stem}"
        items = []
        for j in group:
            item = {"id": j.id, "request_text": j.request_text, "has_media": j.has_media}
            logits = expand_synthetic(j, prepared.get(j.id))
            if logits is not None:
                item["select_logits"] = logits
            items.append(item)
        out = run_oracle(oracle_bin, items, work, name, model_args)
        if out is None:
            # one case at a time, to name the cases the oracle cannot run
            outs = []
            for i, item in enumerate(items):
                one = run_oracle(oracle_bin, [item], work, f"{name}-{i}", model_args)
                outs.append(one if one is not None else {"error": log_tail(work, f"{name}-{i}")})
        else:
            outs = [out] * len(items)
        for i, (j, o) in enumerate(zip(group, outs)):
            if "error" in o:
                results[j.id] = ({"error": o["error"]}, None)
                continue
            if o.get("identity") != stamped:
                die("decision-oracle reported another identity than its executable carries")
            if kind == "model":
                facts = facts or o.get("facts")
            if "load" in o:
                results[j.id] = ({"id": j.id, "load": o["load"]}, None)
                continue
            r = o["cases"][0] if out is None else o["cases"][i]
            if r["id"] != j.id:
                die("decision-oracle returned cases out of order")
            results[j.id] = (r, o["facts"])
    if facts is not None:
        check_context(facts, settings)

    sendable = [j for j in jobs if j.sendable() and "error" not in results[j.id][0]]
    server = ask_server(server_bin, model, args, [{"id": j.id, "request_text": j.request_text} for j in sendable], work) if sendable else None
    return results, facts, server


#
# the observations
#

def bucket(n):
    return "2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11"


def temperature_label(facts, qtype, n_options, f32):
    """The metadata string a question's temperature was read from, checked against upstream's value."""
    temps = facts["temperatures"]
    meta = facts["decision_metadata"]
    for name in (f"{qtype}.{bucket(n_options)}", qtype):
        if name in temps:
            if temps[name] != f32:
                die(f"temperature {name} is {temps[name]} in the table, upstream used {f32}")
            return meta[f"{facts['architecture']}.decision.temperature.{name}"]
    if f32 != "0x3f800000":
        die(f"no temperature key for {qtype}/{n_options}, upstream used {f32}")
    return None


def build_upstream(result, facts):
    up = {}
    if "load" in result:
        up["load"] = result["load"]
        return up
    up["parse"] = result["parse"]
    if "refusal" in result["parse"]:
        return up
    prep = result["prepare"]
    if "refusal" in prep:
        up["prepare"] = prep
        return up
    passes, answers = [], []
    for i, t in enumerate(prep["tasks"]):
        passes.append({"question": t["question"], "prompt": t["prompt"], "token_ids": t["token_ids"],
                       "output_index": t["output_index"], "select": t["select"]})
        groups, start = [], 0
        for key, code, n in zip(t["keys"], t["codes"], t["label_groups"]):
            groups.append({"key": key, "code": code, "range": [start, start + n]})
            start += n
        if start != len(t["select"]) or t["variant"] != 0 or not t["no_batching"]:
            die(f"{result['id']}: unexpected task layout")
        answers.append({"question": t["question"], "type": t["type"], "pass": i, "groups": groups,
                        "temperature": temperature_label(facts, t["type"], len(t["keys"]), t["temperature"]),
                        "temperature_f32": t["temperature"]})
    up["prepare"] = {"passes": passes, "answers": answers}
    if "evaluate" not in result:
        return up
    ev = result["evaluate"]
    if "refusal" in ev:
        up["evaluate"] = {"refusal": ev["refusal"]}
        return up
    up["evaluate"] = {"source": ev["source"], "raw_scores": [t["select_logits"] for t in ev["tasks"]],
                      "scores": [t["scores"] for t in ev["tasks"]]}
    up["finalize"] = result["finalize"]
    return up


def first_refusal(upstream):
    for stage in STAGES:
        if stage in upstream and "refusal" in upstream[stage]:
            return stage, upstream[stage]["refusal"]
    return None, None


def server_check(upstream, status, body):
    """Does llama-server's answer agree with the oracle's stages?"""
    stage, refusal = first_refusal(upstream)
    try:
        data = load_json(body)
    except ValueError:
        return False
    if refusal is not None:
        err = data.get("error", {}) if isinstance(data, dict) else {}
        return (status == refusal["status"] and plain(err.get("code")) == refusal["status"]
                and err.get("type") == refusal["type"] and err.get("message") == refusal["message"])
    fin = upstream.get("finalize")
    if status != 200 or fin is None or not isinstance(data, dict):
        return False
    return strict_equal(plain(data.get("answers")), plain(fin["answers"])) and strict_equal(plain(data.get("usage")), plain(fin["usage"]))


def server_block(upstream, response):
    status, body = response
    agrees = server_check(upstream, status, body)
    block = {"status": status, "agrees": agrees}
    if not agrees:
        block["body"] = load_json(body) if body.strip().startswith("{") else body
    return block


def settings_used(upstream, settings_id):
    ev = upstream.get("evaluate")
    return settings_id if ev is not None and (ev.get("source") == "model" or "refusal" in ev) else None


def check_expectations(case, upstream):
    """xybrid may refuse earlier than upstream, never later; match needs an observation."""
    stage, _ = first_refusal(upstream)
    refused = False
    for s in STAGES:
        exp = case.xybrid.get(s)
        if exp is None:
            if s == "load" and case.metadata:
                die(f"{case.id}: a case with metadata edits needs a load expectation")
            continue
        if refused:
            if exp != "not_reached":
                die(f"{case.id}: {s} must be not_reached after a refusal")
            continue
        if isinstance(exp, dict):
            refused = True
            continue
        if stage is not None and STAGES.index(s) >= STAGES.index(stage):
            die(f"{case.id}: upstream refuses at {stage}, xybrid expects {exp} at {s}")
        if exp in ("match", "reference", "given") and s != "load" and s not in upstream:
            die(f"{case.id}: {s} expects {exp} but upstream has no observation")
        if exp == "reference" and upstream.get("evaluate", {}).get("source") != "model":
            die(f"{case.id}: reference needs model scores")
        if exp == "given" and upstream.get("evaluate", {}).get("source") != "synthetic":
            die(f"{case.id}: given needs synthetic scores")


def case_document(case, upstream, settings, server):
    doc = {"schema_version": 1, "id": case.id, "profile": "lfm3-d1/text/v1", "title": case.title, "covers": case.covers}
    if case.notes:
        doc["notes"] = case.notes
    doc["artifact"] = ARTIFACT
    if case.metadata:
        doc["metadata"] = case.metadata
    doc["settings"] = settings
    doc["request_text"] = case.request_text
    if case.has_media:
        doc["has_media"] = True
    up = dict(upstream)
    if server is not None:
        up["server"] = server
    doc["upstream"] = up
    doc["xybrid"] = {s: case.xybrid[s] for s in STAGES if s in case.xybrid}
    return doc


def platform_record():
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor() or None
    return {"system": platform.system(), "machine": platform.machine(), "cpu": cpu, "libc": " ".join(platform.libc_ver())}


def run_record(identity, problems, settings_id, settings, facts, server):
    """Who made the observations: the collector, the build, the settings as they ran, the platform."""
    devices = [{"name": d["name"], "type": d["type"]} for d in facts["devices"]] if facts else None
    return {
        "collector": {"official": not problems, "problems": problems},
        "build": recorded_identity(identity),
        "settings": {settings_id: {
            "llama_server_args": settings_args(settings),
            "context": facts.get("context") if facts else None,
            "oracle_devices": devices,
            "model_devices": facts["model_devices"] if facts else None,
            # llama-server --list-devices printed no device but the CPU before it ran
            "server_devices": [] if server is not None else None,
        }},
        "platform": platform_record(),
    }


def oracle_record(facts, record):
    commit = record["build"]["llama_cpp"]["commit"]
    out = {
        "schema_version": 1,
        "llama_cpp": {"commit": commit, "path": record["build"]["llama_cpp_dir"],
                      "routines": "tools/server/server-decision.cpp, linked unmodified from the server-context library"},
        "oracle": {"tool": "tools/conformance/decision", "sources": {p: sha256_file(HERE / p) for p in RECIPE}},
        "collector": record["collector"],
        "build": record["build"],
        "settings": record["settings"],
        "platform": record["platform"],
        "artifacts": {},
    }
    f = dict(facts)
    for k in ("devices", "model_devices", "context"):
        f.pop(k, None)
    tmpl = f.pop("template")
    spellings = f.pop("quoted_spellings")
    f["template_sha256"] = sha256_text(tmpl)
    f["template"] = tmpl
    f["control_spellings"] = {"count": len(spellings), "sha256": sha256_text("\n".join(spellings)), "longest_first": spellings}
    out["artifacts"][ARTIFACT] = f
    return out


# what --check compares in oracle.json; the platform is reported, not compared
CHECKED = ("llama_cpp", "oracle", "collector", "build", "settings", "artifacts")


def comparable(section, value):
    """A section as --check compares it: where the llama.cpp checkout sits is recorded, not compared."""
    if not isinstance(value, dict):
        return value
    value = dict(value)
    if section == "build":
        value.pop("llama_cpp_dir", None)
    if section == "llama_cpp":
        value.pop("path", None)
    return value


#
# generate and check: the case set of cases.py
#

def load_case_defs():
    spec = importlib.util.spec_from_file_location("decision_cases", HERE / "cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = set()
    for case in module.CASES:
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", case.id) or case.id in seen:
            die(f"bad or repeated case id: {case.id}")
        seen.add(case.id)
        unknown = set(case.covers) - TAGS
        if unknown:
            die(f"{case.id}: unknown coverage tags {sorted(unknown)}")
        for stage, exp in case.xybrid.items():
            if stage not in STAGES:
                die(f"{case.id}: unknown stage {stage}")
            if isinstance(exp, dict) and exp.get("refuse") not in KINDS:
                die(f"{case.id}: unknown refusal kind {exp}")
    return module.CASES


def generate(args, artifacts, identity, problems, bin_dir, work):
    settings = artifacts["settings"][args.settings]
    model = resolve_model(args, artifacts, ARTIFACT)
    defs = load_case_defs()
    if args.only:
        defs = [c for c in defs if c.id in set(args.only)]
        if len(defs) != len(set(args.only)):
            die("--only names an unknown case")
    jobs = [Job(c.id, c.request_text, c.metadata, c.has_media,
                [[f32_hex(x) for x in t] for t in c.select_logits] if c.select_logits is not None else None, c.synthetic)
            for c in defs]
    results, facts, server = run_jobs(jobs, model, args.settings, settings, bin_dir, work)

    docs, disagreements = {}, []
    for case in defs:
        result, case_facts = results[case.id]
        if "error" in result:
            die(f"{case.id}: decision-oracle cannot run it: {result['error']}")
        upstream = build_upstream(result, case_facts or facts)
        check_expectations(case, upstream)
        srv = server_block(upstream, server[case.id]) if server and case.id in server else None
        if srv is not None and not srv["agrees"]:
            disagreements.append(case.id)
        docs[case.id] = dump(case_document(case, upstream, settings_used(upstream, args.settings), srv)) + "\n"

    record = None
    if not args.only:
        if facts is None:
            die("no case ran on the model")
        record = oracle_record(facts, run_record(identity, problems, args.settings, settings, facts, server))

    out_dir = Path(args.out) if args.out else FIXTURES
    cases_dir = out_dir / "cases"
    status = 0
    if disagreements:
        print(f"llama-server disagrees with the oracle on: {', '.join(disagreements)}", file=sys.stderr)
        status = 1
    if args.check:
        stale = [i for i, text in docs.items() if not (cases_dir / f"{i}.json").exists() or (cases_dir / f"{i}.json").read_text(encoding="utf-8") != text]
        if not args.only:
            stale += sorted({p.stem for p in cases_dir.glob("*.json")} - set(docs))
            old = load_json((out_dir / "oracle.json").read_text(encoding="utf-8")) if (out_dir / "oracle.json").exists() else {}
            if not old.get("collector", {}).get("official"):
                print("oracle.json is not an official record: regenerate it without --dev", file=sys.stderr)
                status = 1
            for section in CHECKED:
                if dump(comparable(section, old.get(section))) != dump(comparable(section, record.get(section))):
                    if args.dev and section in ("oracle", "collector", "build", "settings"):
                        print(f"warning: oracle.json:{section} differs from this developer run", file=sys.stderr)
                    else:
                        stale.append(f"oracle.json:{section}")
            if dump(old.get("platform")) != dump(record["platform"]):
                print(f"note: recorded platform {dump(old.get('platform'))}, this one {dump(record['platform'])}", file=sys.stderr)
        if stale:
            print("stale: " + ", ".join(stale), file=sys.stderr)
            status = 1
        elif status == 0:
            print(f"{len(docs)} cases match the committed files" + (" (developer run)" if problems else ""))
    else:
        if problems and out_dir.resolve() == FIXTURES.resolve():
            die("a developer run does not write the committed fixtures: pass --out")
        cases_dir.mkdir(parents=True, exist_ok=True)
        if not args.only:
            for p in cases_dir.glob("*.json"):
                if p.stem not in docs:
                    p.unlink()
            (out_dir / "oracle.json").write_text(dump(record) + "\n", encoding="utf-8")
        for i, text in docs.items():
            (cases_dir / f"{i}.json").write_text(text, encoding="utf-8")
        print(f"wrote {len(docs)} cases to {cases_dir}")
    return status


#
# collect: any directory of case files, on any artifact
#

def collect(args, artifacts, identity, problems, bin_dir, work):
    if not args.out:
        die("--collect needs --out")
    settings = artifacts["settings"][args.settings]
    model = resolve_model(args, artifacts, args.artifact)
    out_dir = Path(args.out)
    if out_dir.resolve() in (FIXTURES.resolve(), (FIXTURES / "cases").resolve()):
        die("--collect writes observations, not the committed fixtures: pick another --out")

    entries, jobs = [], []
    for path in sorted(Path(args.collect).glob("*.json")):
        entry = {"file": path.name}
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if case.get("schema_version") != 1 or not isinstance(case.get("id"), str) or not isinstance(case.get("request_text"), str):
                raise ValueError("not a schema-v1 case: schema_version 1, id and request_text are required")
            if args.only and case["id"] not in args.only:
                continue
            entry["id"] = case["id"]
            entry["request_sha256"] = hashlib.sha256(case["request_text"].encode("utf-8")).hexdigest()
            ev = case.get("upstream", {}).get("evaluate", {})
            logits = ev.get("raw_scores") if ev.get("source") == "synthetic" else None
            if case.get("metadata") and logits is None and "load" not in case.get("upstream", {}):
                # a metadata-only GGUF has no tensors: without synthetic scores the case can only be prepared
                entry["note"] = "metadata edits and no synthetic scores: prepared, not evaluated"
            jobs.append(Job(case["id"], case["request_text"], case.get("metadata"), bool(case.get("has_media")), logits))
        except (ValueError, UnicodeEncodeError) as e:
            entry.update({"status": "error", "error": str(e)})
        entries.append(entry)
    ids = [j.id for j in jobs]
    if len(ids) != len(set(ids)):
        die("two case files share an id")
    if args.only and set(args.only) - set(ids):
        die(f"--only names cases that are not in {args.collect}: {sorted(set(args.only) - set(ids))}")

    results, facts, server = run_jobs(jobs, model, args.settings, settings, bin_dir, work) if jobs else ({}, None, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for entry in entries:
        if "id" not in entry:
            failed += 1
            continue
        result, case_facts = results[entry["id"]]
        if "error" in result:
            entry.update({"status": "error", "error": result["error"]})
            failed += 1
            continue
        upstream = build_upstream(result, case_facts or facts)
        if server and entry["id"] in server:
            upstream["server"] = server_block(upstream, server[entry["id"]])
            if not upstream["server"]["agrees"]:
                failed += 1
        name = f"{entry['id']}.{entry['request_sha256']}.json"
        doc = {"schema_version": 1, "id": entry["id"], "request_sha256": entry["request_sha256"], "source_file": entry["file"],
               "artifact": args.artifact, "settings": settings_used(upstream, args.settings), "upstream": upstream}
        (out_dir / name).write_text(dump(doc) + "\n", encoding="utf-8")
        entry.update({"status": "observed", "observation": name})
        if "server" in upstream:
            entry["server_agrees"] = upstream["server"]["agrees"]
    art = artifacts["artifacts"][args.artifact]
    summary = {"schema_version": 1, "cases_dir": str(args.collect),
               "artifact": {"id": args.artifact, "file": art["file"], "bytes": art["bytes"], "sha256": art["sha256"]}}
    summary.update(run_record(identity, problems, args.settings, settings, facts, server))
    summary["cases"] = entries
    (out_dir / "collection.json").write_text(dump(summary) + "\n", encoding="utf-8")
    observed = sum(1 for e in entries if e.get("status") == "observed")
    print(f"{observed} observed, {failed} failed or disagreeing, of {len(entries)} case files; see {out_dir / 'collection.json'}")
    return 1 if failed else 0


#
# main
#

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="write nothing; fail unless a fresh run gives the committed files")
    ap.add_argument("--collect", metavar="DIR", help="run the schema-v1 case files of DIR instead of cases.py; needs --out")
    ap.add_argument("--artifact", default=ARTIFACT, help=f"artifact id of artifacts.json for --collect (default {ARTIFACT})")
    ap.add_argument("--model", help="the artifact's GGUF (default: integration-tests/fixtures/models/<artifact>/<file>)")
    ap.add_argument("--download", action="store_true", help="download the GGUF if it is missing")
    ap.add_argument("--build", default=str(HERE / "build"), help="the CMake build directory (build.sh)")
    ap.add_argument("--settings", default="cpu-t4-fa-off", help="settings id of artifacts.json")
    ap.add_argument("--only", action="append", help="only this case id (repeatable); oracle.json is left alone")
    ap.add_argument("--out", help="where cases/ and oracle.json go (default: the committed fixtures), or --collect's output")
    ap.add_argument("--dev", action="store_true", help="run executables that cannot produce official observations, and mark the output")
    ap.add_argument("--keep-work", action="store_true", help="keep the oracle's inputs, outputs and logs")
    args = ap.parse_args()

    bin_dir = Path(args.build) / "bin"
    for b in (bin_dir / "decision-oracle", bin_dir / "llama-server"):
        if not b.exists():
            die(f"no {b}: run tools/conformance/decision/build.sh")
    artifacts = json.loads((FIXTURES / "artifacts.json").read_text())
    if args.settings not in artifacts["settings"]:
        die(f"settings {args.settings} is not in artifacts.json")
    if args.collect is None and args.artifact != ARTIFACT:
        die(f"the case set is observed on {ARTIFACT}; --artifact goes with --collect")
    identity, problems = build_identity(bin_dir, args.dev)

    work = HERE / "work" / f"run-{os.getpid()}"
    work.mkdir(parents=True)
    try:
        if args.collect is not None:
            return collect(args, artifacts, identity, problems, bin_dir, work)
        return generate(args, artifacts, identity, problems, bin_dir, work)
    finally:
        if args.keep_work:
            print(f"work files kept in {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
