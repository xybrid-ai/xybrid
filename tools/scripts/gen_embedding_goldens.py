#!/usr/bin/env python3
"""Write the reference embeddings for the EmbeddingGemma 2 parity test.

Runs the fixture's own ONNX graph with ONNX Runtime and keeps its in-graph
``sentence_embedding`` output: mean pooling plus L2 normalization, the same
modules sentence-transformers' ``modules.json`` declares for the model. Text is
tokenized with the tokenizer's special tokens (``<bos>`` ... ``<eos>``).
xybrid's Tokenize -> ONNX -> MeanPool -> L2Normalize path must reproduce these
vectors (``integration-tests/tests/embedding_parity.rs``).

Requirements: ``onnxruntime==1.23.2`` (the version xybrid vendors), ``tokenizers``
and ``numpy``.

Usage::

    ./integration-tests/download.sh embeddinggemma-2
    python3 tools/scripts/gen_embedding_goldens.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "integration-tests/fixtures/models/embeddinggemma-2"
DEFAULT_OUT = REPO_ROOT / "integration-tests/fixtures/embedding/embeddinggemma-2.json"
MODEL_FILE = "model_q4.onnx"

# Queries and documents in the model card's prompt formats. Two languages, so a
# tokenization or pooling slip on non-ASCII text shows up too.
TEXTS = [
    "task: search result | query: How do I reset my password?",
    "task: search result | query: Comment réinitialiser mon mot de passe ?",
    "title: none | text: To reset your password, open Settings, choose Account, "
    "tap Reset password and follow the link we email you.",
    "title: none | text: Bananas are rich in potassium and are a popular breakfast fruit.",
]

# A text-only run gives the model's image, video and audio inputs zero rows.
FEATURE_INPUTS = ("image_features", "video_features", "audio_features")
FEATURE_WIDTH = 512


def embed(session: ort.InferenceSession, tokenizer: Tokenizer, text: str) -> list[float]:
    ids = np.array([tokenizer.encode(text, add_special_tokens=True).ids], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    feeds.update({name: np.zeros((0, FEATURE_WIDTH), dtype=np.float32) for name in FEATURE_INPUTS})
    (embedding,) = session.run(["sentence_embedding"], feeds)
    return [round(float(value), 7) for value in embedding[0]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    session = ort.InferenceSession(
        str(args.model_dir / MODEL_FILE), providers=["CPUExecutionProvider"]
    )

    # One case per line keeps the 768 floats from turning into 768 lines.
    cases = ",\n".join(
        "    "
        + json.dumps({"text": text, "embedding": embed(session, tokenizer, text)}, ensure_ascii=False)
        for text in TEXTS
    )
    header = {
        "generator": "tools/scripts/gen_embedding_goldens.py",
        "model": "onnx-community/embeddinggemma-2-ONNX@daa72c51243991dfcaf9f9137d2c573d8f7790c0",
        "model_file": MODEL_FILE,
        "onnxruntime": ort.__version__,
        "output": "sentence_embedding",
    }
    fields = "".join(f"  {json.dumps(key)}: {json.dumps(value)},\n" for key, value in header.items())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("{\n" + fields + '  "cases": [\n' + cases + "\n  ]\n}\n", encoding="utf-8")
    print(f"wrote {len(TEXTS)} reference embeddings to {args.out}")


if __name__ == "__main__":
    main()
