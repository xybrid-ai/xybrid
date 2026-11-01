#!/usr/bin/env python3
"""Write the decision conformance cases from llama.cpp's own decision code.

Runs every case of cases.py through decision-oracle (the pinned upstream routines, see oracle.cpp)
and sends the same request to the stock llama-server built from the same sources, then writes
integration-tests/fixtures/decision/cases/<id>.json and oracle.json. With --check it writes
nothing and fails unless a fresh run gives the committed files.

Usage:
  tools/conformance/decision/oracle.py [--check] [--model PATH] [--download] [--settings ID] [--only ID]...

Stdlib only. README.md next to this file explains the setup.
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
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURES = REPO / "integration-tests" / "fixtures" / "decision"
ARTIFACT = "d1-2b-q4km"

# llama-server arguments for each settings id of artifacts.json. decision-oracle parses the same
# list with the server's own parser.
SERVER_ARGS = {
    "cpu-t4-fa-off": ["-c", "4096", "-b", "4096", "-ub", "4096", "-np", "1", "-t", "4", "-tb", "4", "-fa", "off"],
    "cpu-t4-fa-on": ["-c", "4096", "-b", "4096", "-ub", "4096", "-np", "1", "-t", "4", "-tb", "4", "-fa", "on"],
}

TAGS = set("""
state-string state-empty state-null state-object state-array state-messages
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
# inputs
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


def resolve_model(args, artifacts):
    art = artifacts["artifacts"][ARTIFACT]
    path = Path(args.model) if args.model else Path(os.environ.get("XYBRID_DECISION_MODELS", HERE / "work" / "models")) / art["file"]
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

def run_oracle(binary, cases, out_dir, name, model_args):
    """cases: [{id, request_text, select_logits?, has_media?}] -> the oracle's JSON output."""
    inp = out_dir / f"{name}-in.json"
    out = out_dir / f"{name}-out.json"
    inp.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    cmd = [str(binary), "run", "--input", str(inp), "--output", str(out)] + model_args
    with open(out_dir / f"{name}.log", "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        die(f"decision-oracle failed ({name}), see {out_dir / (name + '.log')}")
    return load_json(out.read_text(encoding="utf-8"))


def make_variant(binary, base, metadata, out_dir):
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    path = out_dir / f"variant-{key}.gguf"
    cmd = [str(binary), "variant", "--base", str(base), "--output", str(path)]
    for k, v in sorted(metadata.get("set", {}).items()):
        cmd += ["--set", f"{k}={v}"]
    for k in metadata.get("remove", []):
        cmd += ["--remove", k]
    for token, ttype in sorted(metadata.get("token_types", {}).items()):
        cmd += ["--token-type", f"{token}={ttype}"]
    subprocess.run(cmd, check=True)
    return key, path


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ask_server(binary, model, settings, cases, out_dir):
    """POST each request_text to a fresh llama-server; returns {id: (status, body)}."""
    port = free_port()
    log = open(out_dir / "llama-server.log", "w")
    proc = subprocess.Popen([str(binary), "-m", str(model)] + SERVER_ARGS[settings] + ["--host", "127.0.0.1", "--port", str(port)],
                            stdout=log, stderr=subprocess.STDOUT)
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
                die(f"llama-server did not start, see {out_dir / 'llama-server.log'}")
            time.sleep(0.5)
        answers = {}
        for case in cases:
            req = urllib.request.Request(base + "/v1/systemone", data=case["request_text"].encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    answers[case["id"]] = (r.status, r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                answers[case["id"]] = (e.code, e.read().decode("utf-8"))
        return answers
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


#
# the case files
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


def select_logits_for(case, prepared):
    """Synthetic select logits: given in full, or one score per option with lower aliases."""
    if case.select_logits is not None:
        return [[f32_hex(x) for x in task] for task in case.select_logits]
    if case.synthetic is None:
        return None
    if prepared is None:
        return []
    tasks = prepared["prepare"]["tasks"]
    if len(tasks) != len(case.synthetic):
        die(f"{case.id}: synthetic scores for {len(case.synthetic)} passes, the request has {len(tasks)}")
    out = []
    for task, scores in zip(tasks, case.synthetic):
        if len(scores) != len(task["label_groups"]):
            die(f"{case.id}: {len(scores)} scores for {len(task['label_groups'])} options")
        logits = []
        for score, n in zip(scores, task["label_groups"]):
            logits.append(f32_hex(score))
            for _ in range(n - 1):
                logits.append(f32_hex(score if isinstance(score, str) else float(score) - 1.0))
        out.append(logits)
    return out


def oracle_record(facts_by_artifact, settings, binary_dir):
    commit = subprocess.run(["git", "-C", str(REPO / "vendor" / "llama-cpp"), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    cache = {}
    cache_file = binary_dir.parent / "CMakeCache.txt"
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            m = re.match(r"([A-Z0-9_]+):[A-Z]+=(.*)", line)
            if m:
                cache[m.group(1)] = m.group(2)
    options = {k: cache[k] for k in sorted(cache) if k.startswith(("GGML_", "LLAMA_")) and k in (
        "GGML_NATIVE", "GGML_OPENMP", "GGML_SSE42", "GGML_AVX", "GGML_AVX2", "GGML_BMI2", "GGML_FMA", "GGML_F16C",
        "GGML_AVX512", "GGML_LLAMAFILE", "GGML_CUDA", "GGML_METAL", "GGML_VULKAN", "LLAMA_BUILD_SERVER", "LLAMA_OPENSSL")}
    record = {
        "schema_version": 1,
        "llama_cpp": {"commit": commit, "path": "vendor/llama-cpp",
                      "routines": "tools/server/server-decision.cpp, linked unmodified from the server-context library"},
        "oracle": {"tool": "tools/conformance/decision",
                   "sources": {p: sha256_file(HERE / p) for p in ("CMakeLists.txt", "oracle.cpp", "test_access.h", "cases.py", "oracle.py")}},
        "build": {"build_type": cache.get("CMAKE_BUILD_TYPE"), "compiler": compiler_version(cache.get("CMAKE_CXX_COMPILER")),
                  "options": options},
        "platform": {"system": platform.system(), "machine": platform.machine(), "cpu": cpu_model(),
                     "libc": " ".join(platform.libc_ver())},
        "settings": {settings: {"llama_server_args": SERVER_ARGS[settings]}},
        "artifacts": {},
    }
    for art, facts in facts_by_artifact.items():
        f = dict(facts)
        tmpl = f.pop("template")
        spellings = f.pop("quoted_spellings")
        context = f.pop("context", None)
        f["template_sha256"] = sha256_text(tmpl)
        f["template"] = tmpl
        f["control_spellings"] = {"count": len(spellings), "sha256": sha256_text("\n".join(spellings)), "longest_first": spellings}
        record["artifacts"][art] = f
        if context is not None:
            record["settings"][settings]["context"] = context
    return record


def compiler_version(path):
    if not path:
        return None
    try:
        return subprocess.run([path, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except OSError:
        return None


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


#
# main
#

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="write nothing; fail unless a fresh run gives the committed files")
    ap.add_argument("--model", help=f"the {ARTIFACT} GGUF (default: $XYBRID_DECISION_MODELS or work/models)")
    ap.add_argument("--download", action="store_true", help="download the GGUF if it is missing")
    ap.add_argument("--build", default=str(HERE / "build"), help="the CMake build directory (build.sh)")
    ap.add_argument("--settings", default="cpu-t4-fa-off", choices=sorted(SERVER_ARGS))
    ap.add_argument("--only", action="append", help="only this case id (repeatable); oracle.json is left alone")
    ap.add_argument("--out", default=str(FIXTURES), help="where cases/ and oracle.json go (default: the committed fixtures)")
    ap.add_argument("--keep-work", action="store_true", help="keep the oracle's inputs, outputs and logs in work/run")
    args = ap.parse_args()

    bin_dir = Path(args.build) / "bin"
    oracle_bin, server_bin = bin_dir / "decision-oracle", bin_dir / "llama-server"
    for b in (oracle_bin, server_bin):
        if not b.exists():
            die(f"no {b}: run tools/conformance/decision/build.sh")
    artifacts = json.loads((FIXTURES / "artifacts.json").read_text())
    if args.settings not in artifacts["settings"]:
        die(f"settings {args.settings} is not in artifacts.json")
    model = resolve_model(args, artifacts)
    defs = load_case_defs()
    if args.only:
        defs = [c for c in defs if c.id in set(args.only)]
        if len(defs) != len(set(args.only)):
            die("--only names an unknown case")

    work = HERE / "work" / "run"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # metadata variants: one metadata-only GGUF per distinct edit set
    variants = {}
    for case in defs:
        if case.metadata:
            key, path = make_variant(oracle_bin, model, case.metadata, work)
            variants[case.id] = (key, path)

    # synthetic scores need each pass's group sizes: a prepare-only run first
    prepared = {}
    groups = {}
    for case in defs:
        if case.synthetic is not None and case.select_logits is None:
            target = variants[case.id][1] if case.id in variants else model
            groups.setdefault(str(target), []).append(case)
    for target, group in groups.items():
        out = run_oracle(oracle_bin, [{"id": c.id, "request_text": c.request_text} for c in group], work,
                         f"prepare-{hashlib.sha256(target.encode()).hexdigest()[:8]}", ["--vocab-only", target])
        for c, r in zip(group, out.get("cases", [])):
            prepared[c.id] = r if "tasks" in r.get("prepare", {}) else None

    # the real runs: the model under the settings, or a variant's vocabulary only
    runs = {}
    for case in defs:
        target = ("variant", variants[case.id][1]) if case.id in variants else ("model", model)
        runs.setdefault(target, []).append(case)
    results, facts_by_target = {}, {}
    for (kind, target), group in runs.items():
        items = []
        for c in group:
            item = {"id": c.id, "request_text": c.request_text, "has_media": c.has_media}
            logits = select_logits_for(c, prepared.get(c.id))
            if logits is not None:
                item["select_logits"] = logits
            items.append(item)
        model_args = ["--", "-m", str(target)] + SERVER_ARGS[args.settings] if kind == "model" else ["--vocab-only", str(target)]
        name = "model" if kind == "model" else f"variant-{Path(target).stem}"
        out = run_oracle(oracle_bin, items, work, name, model_args)
        facts_by_target[(kind, str(target))] = out.get("facts")
        if "load" in out:
            for c in group:
                results[c.id] = ({"id": c.id, "load": out["load"]}, None)
            continue
        for c, r in zip(group, out["cases"]):
            if r["id"] != c.id:
                die("decision-oracle returned cases out of order")
            results[c.id] = (r, out["facts"])

    # the stock server, for every request it can be given as is
    sendable = [c for c in defs if not c.metadata and c.synthetic is None and c.select_logits is None and not c.has_media]
    server = {}
    if sendable:
        server = ask_server(server_bin, model, args.settings, [{"id": c.id, "request_text": c.request_text} for c in sendable], work)

    out_dir = Path(args.out)
    cases_dir = out_dir / "cases"
    docs = {}
    disagreements = []
    base_facts = facts_by_target.get(("model", str(model)))
    for case in defs:
        result, facts = results[case.id]
        upstream = build_upstream(result, facts or base_facts)
        check_expectations(case, upstream)
        ev = upstream.get("evaluate")
        settings = args.settings if ev is not None and (ev.get("source") == "model" or "refusal" in ev) else None
        srv = None
        if case.id in server:
            status, body = server[case.id]
            agrees = server_check(upstream, status, body)
            srv = {"status": status, "agrees": agrees}
            if not agrees:
                srv["body"] = load_json(body) if body.strip().startswith("{") else body
                disagreements.append(case.id)
        docs[case.id] = dump(case_document(case, upstream, settings, srv)) + "\n"

    record = None
    if not args.only:
        record = dump(oracle_record({ARTIFACT: base_facts}, args.settings, bin_dir)) + "\n"

    status = 0
    if disagreements:
        print(f"llama-server disagrees with the oracle on: {', '.join(disagreements)}", file=sys.stderr)
        status = 1
    if args.check:
        stale = [i for i, text in docs.items() if not (cases_dir / f"{i}.json").exists() or (cases_dir / f"{i}.json").read_text(encoding="utf-8") != text]
        if not args.only:
            committed = {p.stem for p in cases_dir.glob("*.json")}
            stale += sorted(committed - set(docs))
            old = load_json((out_dir / "oracle.json").read_text(encoding="utf-8")) if (out_dir / "oracle.json").exists() else {}
            new = load_json(record)
            for section in ("llama_cpp", "artifacts"):
                if dump(old.get(section)) != dump(new.get(section)):
                    stale.append(f"oracle.json:{section}")
        if stale:
            print("stale: " + ", ".join(stale), file=sys.stderr)
            status = 1
        else:
            print(f"{len(docs)} cases match the committed files")
    else:
        cases_dir.mkdir(parents=True, exist_ok=True)
        if not args.only:
            for p in cases_dir.glob("*.json"):
                if p.stem not in docs:
                    p.unlink()
            (out_dir / "oracle.json").write_text(record, encoding="utf-8")
        for i, text in docs.items():
            (cases_dir / f"{i}.json").write_text(text, encoding="utf-8")
        print(f"wrote {len(docs)} cases to {cases_dir}")
    if not args.keep_work:
        shutil.rmtree(work)
    return status


if __name__ == "__main__":
    sys.exit(main())
