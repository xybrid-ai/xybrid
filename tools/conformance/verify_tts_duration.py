#!/usr/bin/env python3
"""Check that a grafted TTS model still renders byte-identical audio.

Runs the original model (single ``audio`` output) and the model produced by
``expose_tts_duration.py`` on the same token ids, style vector and speeds, and
requires: identical ``audio`` bytes, a ``duration`` output shaped
``[1, len(tokens)]`` with every value >= 1, and reports how the audio length
relates to ``frame_samples * sum(duration)``.

Token ids are built like the Xybrid Phonemize step: pad 0, one id per phoneme
character from ``tokens.txt``, pad 0 (unknown characters are skipped).

Usage::

    uv run --no-project --python 3.12 --with-requirements tools/conformance/requirements-forms.txt \\
        python tools/conformance/verify_tts_duration.py ORIGINAL/model.onnx GRAFTED/model.onnx ORIGINAL/tokens.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Misaki phonemes for a few lines (us_gold), covering stress marks, diphthong capitals and punctuation
SAMPLES = [
    "həlˈO wˈɜɹld!",
    "wˈɛl nˈW, pˈʊl ˈʌp ɐ stˈul.",
    "ðə ɹˈʌsti flˈæɡən ɪz fˈɔɹti jˈɪɹz ˈOld?",
]


def load_vocab(path: Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        symbol, _, index = line.rpartition(" ")
        vocab[symbol if symbol else " "] = int(index)
    return vocab


def token_ids(phonemes: str, vocab: dict[str, int]) -> np.ndarray:
    ids = [0] + [vocab[c] for c in phonemes if c in vocab] + [0]
    return np.array([ids], dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("original", type=Path)
    parser.add_argument("grafted", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--frame-samples", type=int, default=600)
    args = parser.parse_args()

    vocab = load_vocab(args.tokens)
    style = np.random.default_rng(7).standard_normal((1, 256)).astype(np.float32) * 0.1
    failures = 0

    for threads in (1, 0):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        original = ort.InferenceSession(str(args.original), options)
        grafted = ort.InferenceSession(str(args.grafted), options)
        out_names = [o.name for o in grafted.get_outputs()]
        if out_names[:1] != ["audio"] or "duration" not in out_names:
            print(f"FAIL grafted outputs are {out_names}")
            return 1

        for phonemes in SAMPLES:
            tokens = token_ids(phonemes, vocab)
            for speed in (0.8, 1.0, 1.25):
                feeds = {"tokens": tokens, "style": style, "speed": np.array([speed], dtype=np.float32)}
                (audio_a,) = original.run(["audio"], feeds)
                audio_b, duration = grafted.run(["audio", "duration"], feeds)
                same = audio_a.shape == audio_b.shape and audio_a.tobytes() == audio_b.tobytes()
                shape_ok = duration.shape == (1, tokens.shape[1]) and int(duration.min()) >= 1
                total = int(duration.sum())
                ratio = audio_b.size / (args.frame_samples * total) if total else float("nan")
                status = "ok" if same and shape_ok else "FAIL"
                failures += status == "FAIL"
                print(
                    f"{status} threads={threads or 'default'} speed={speed} tokens={tokens.shape[1]} "
                    f"identical={same} duration{tuple(duration.shape)} min={int(duration.min())} "
                    f"sum={total} samples={audio_b.size} "
                    f"samples/({args.frame_samples}*sum)={ratio:.4f}"
                )

    print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
