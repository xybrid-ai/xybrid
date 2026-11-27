#!/usr/bin/env python3
"""Collect pinned HF d1 observations; never infer numerical acceptance bounds.

Asset staging has an explicit network phase. Collection verifies every asset,
then loads local code/checkpoint with HF offline and set_size=1, one question
and one batch row per call. Failed cases remain failed receipts, not skips.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURES = ROOT / "integration-tests/fixtures/decision"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def verify_assets(directory: Path, manifest: dict) -> None:
    for name, entry in manifest["assets"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"missing or corrupt pinned asset: {name}")


def stage(directory: Path, manifest: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, entry in manifest["assets"].items():
        path = directory / name
        if path.is_file() and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]:
            continue
        url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
        temporary = path.with_suffix(path.suffix + ".download")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > entry["bytes"]:
                    raise ValueError(f"oversized pinned asset: {name}")
                output.write(block)
        if total != entry["bytes"] or digest(temporary) != entry["sha256"]:
            raise ValueError(f"pinned asset digest mismatch: {name}")
        temporary.replace(path)
    verify_assets(directory, manifest)


def corpus(only: list[str] | None) -> list[tuple[dict, dict]]:
    manifest = read_json(FIXTURES / "hf/corpus.json")
    entries = manifest["cases"]
    if len(entries) != 63 or len({entry["id"] for entry in entries}) != 63:
        raise ValueError("HF corpus must contain all 63 distinct pinned cases")
    if only and not set(only) <= {entry["id"] for entry in entries}:
        raise ValueError("unknown --only case")
    result = []
    for entry in entries:
        path = FIXTURES / "hf/cases" / (entry["id"] + ".json")
        value = read_json(path)
        if digest(path) != entry["file_sha256"] or hashlib.sha256(value["request_text"].encode()).hexdigest() != entry["request_sha256"]:
            raise ValueError(f"HF corpus digest mismatch: {entry['id']}")
        if not only or entry["id"] in only:
            result.append((entry, value))
    return result


def verify_environment() -> dict[str, str]:
    actual = {}
    for line in (HERE / "requirements-hf.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        package, expected = line.split("==")
        observed = importlib.metadata.version(package)
        if observed != expected:
            raise ValueError(f"reference environment mismatch: {package}: {observed}, expected {expected}")
        actual[package] = observed
    return actual


def collect(directory: Path, output: Path, cases: list[tuple[dict, dict]], manifest: dict, dtype: str) -> int:
    if output.exists() and any(output.iterdir()):
        raise ValueError("--out must be new or empty")
    if output.resolve().is_relative_to(FIXTURES.resolve()):
        raise ValueError("collect outside committed fixtures; review observations before promotion")
    verify_assets(directory, manifest)
    environment = verify_environment()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(output.resolve() / "cache")
    os.environ["HF_MODULES_CACHE"] = str(output.resolve() / "cache/modules")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModel

    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(
        str(directory.resolve()), trust_remote_code=True,
        local_files_only=True, dtype=getattr(torch, dtype),
    ).to("cpu").eval()
    engine = model.engine
    # Import the exact readout implementation loaded with the model, not a
    # separately reimplemented label map or probability calculation.
    prompt_module = importlib.import_module(type(engine).__module__.rsplit(".", 1)[0] + ".prompt")
    original_readouts = engine._readouts
    selected = []
    raw_rows = []

    def observe_head(_module, _arguments, output):
        # The runner subtracts a vocabulary-wide logsumexp before _readouts.
        # Capture the actual output projection separately; normalized values
        # cannot be compared to GGUF raw logits without this lost constant.
        raw_rows.append(output.detach().float().cpu())

    head_hook = model.get_output_embeddings().register_forward_hook(observe_head)

    def observe(qs, logits, tempered):
        if len(raw_rows) != 1 or raw_rows[0].shape[0] != len(qs):
            raise ValueError("HF reference did not project exactly one output row per question")
        for index, (question, row) in enumerate(zip(qs, logits)):
            groups = prompt_module.readout_ids(engine.tokenizer, question)
            tokens = [token for group in groups for token in group]
            values = row[tokens].detach().float().cpu().tolist()
            raw = raw_rows[0][index, tokens].tolist()
            selected.append({"groups": groups,
                             "raw_scores": ["0x" + struct.pack(">f", value).hex() for value in raw],
                             "log_probabilities": ["0x" + struct.pack(">f", value).hex() for value in values]})
        return original_readouts(qs, logits, tempered)

    engine._readouts = observe
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    # Process short inputs first without changing the frozen corpus/splits.
    # Long/depth-boundary cases still execute and are counted explicitly.
    for entry, case in sorted(cases, key=lambda item: len(item[1]["request_text"])):
        started = time.monotonic()
        request = json.loads(case["request_text"])
        result = {"id": entry["id"], "split": entry["split"], "request_sha256": entry["request_sha256"],
                  "case_file_sha256": entry["file_sha256"], "status": "complete", "passes": []}
        try:
            for name, spec in request["questions"].items():
                question = engine.ask(spec)
                prompt = engine.render(request["state"], question)
                tokens = engine.tokenizer.encode(prompt, add_special_tokens=False)
                if len(tokens) > 4096:
                    raise ValueError("independent HF prompt exceeds text/v1 token budget")
                selected.clear()
                raw_rows.clear()
                with torch.inference_mode():
                    answer = model.system_one(request["state"], {name: spec}, set_size=1, score_readout="median")
                if len(selected) != 1:
                    raise ValueError("HF reference did not produce exactly one readout")
                result["passes"].append({"question": name, "prompt": prompt, "token_ids": tokens,
                                         "selected": selected[0], "result": answer})
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
        result["duration_seconds"] = time.monotonic() - started
        path = output / (entry["id"] + ".json")
        path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        receipts.append({"id": entry["id"], "status": result["status"], "sha256": digest(path)})
        print(f"{entry['id']}: {result['status']}", flush=True)
    head_hook.remove()
    failed = sum(entry["status"] != "complete" for entry in receipts)
    report = {
        "schema_version": 1, "status": "incomplete" if failed else "complete",
        "purpose": "independent observations; not numerical promotion thresholds",
        "profile": "lfm3-d1/text/v1", "repository": manifest["repository"], "revision": manifest["revision"],
        "assets_manifest_sha256": digest(FIXTURES / "hf/assets.json"),
        "corpus_manifest_sha256": digest(FIXTURES / "hf/corpus.json"),
        "collector_sha256": digest(Path(__file__)), "requirements_sha256": digest(HERE / "requirements-hf.txt"),
        "environment": environment, "python": platform.python_version(), "platform": platform.platform(),
        "torch_config": torch.__config__.show(),
        "settings": {"dtype": dtype, "device": "cpu", "threads": 4, "interop_threads": 4,
                     "attention": engine.attention, "set_size": 1, "batch_rows": 1, "compiled": False},
        "counts": {"selected": len(cases), "executed": len(receipts), "failed": failed}, "cases": receipts,
    }
    (output / "collection.json").write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--stage", action="store_true", help="download and verify pinned assets, then exit")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", action="append")
    parser.add_argument("--expect-cases", type=int, default=63)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    manifest = read_json(FIXTURES / "hf/assets.json")
    if args.stage:
        stage(args.model_dir, manifest)
        return 0
    cases = corpus(args.only)
    if len(cases) != args.expect_cases:
        raise ValueError("selected case count differs from --expect-cases")
    if args.out is None:
        parser.error("--out is required for collection")
    return collect(args.model_dir, args.out, cases, manifest, args.dtype)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f"hf_reference: {error}", file=sys.stderr)
        sys.exit(1)
