"""Helpers shared by the conformance generators: pins, numerics, JSON layout."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

# How every reference runs; recorded in each golden.
NUMERICS = {
    "dtype": "float32",
    "device": "cpu",
    "torch_threads": 1,
    "deterministic_algorithms": True,
    "mode": "eval, no_grad",
}


def deterministic_torch(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verified_inputs(
    sources: Mapping[str, Any], inputs: Iterable[tuple[str, str, Path]]
) -> list[dict[str, str]]:
    """Refuse unless every ``(source, pinned file, local path)`` matches its pin.

    A golden cites the revisions it was generated from, so a generator reads
    nothing unpinned; the verified list goes into the golden, where
    tools/scripts/check_choice_provenance.py checks it again.
    """
    verified = []
    for source, file_name, path in inputs:
        pinned = sources[source]["files"][file_name]["sha256"]
        actual = sha256_file(path)
        if actual != pinned:
            raise SystemExit(
                f"{path} is not the pinned {source}:{file_name} (sha256 {actual}, pinned {pinned})"
            )
        verified.append({"source": source, "file": file_name, "sha256": actual})
    return verified


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits.astype(np.float64) - np.max(logits))
    return shifted / shifted.sum()


def best_and_margin(scores: np.ndarray) -> tuple[int, float]:
    """Best candidate (first on ties) and the top-1 minus top-2 margin."""
    ordered = np.sort(scores)[::-1]
    return int(np.argmax(scores)), float(ordered[0] - ordered[1])


def dumps(value: Any, level: int = 0) -> str:
    """JSON with indented objects and one-line scalar arrays, so ids read as rows."""
    pad = "  " * (level + 1)
    if isinstance(value, dict) and value:
        items = [
            f"{pad}{json.dumps(k, ensure_ascii=False)}: {dumps(v, level + 1)}"
            for k, v in value.items()
        ]
        text = "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    elif isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        items = [f"{pad}{dumps(item, level + 1)}" for item in value]
        text = "[\n" + ",\n".join(items) + "\n" + "  " * level + "]"
    elif isinstance(value, list):
        text = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    else:
        text = json.dumps(value, ensure_ascii=False)
    return text + ("\n" if level == 0 else "")
