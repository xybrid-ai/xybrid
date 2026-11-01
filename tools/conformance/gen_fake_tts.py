#!/usr/bin/env python3
"""Write the tiny fake ONNX TTS models used by the phoneme-timing tests.

Two model directories under ``integration-tests/fixtures/tts/fake/``, which
``integration-tests/tests/tts_phoneme_timings.rs`` runs through the SDK:

- ``with_duration/``: outputs ``duration`` (INT64 ``[1, N]``, frames per input
  token) and ``audio`` (float32 ``[samples]``), declared in that order, so the
  waveform has to be picked by name, not by position.
- ``without_duration/``: the same graph with ``audio`` as its only output.

Both take Kokoro's inputs (``tokens`` INT64 ``[1, N]``, ``style`` float32
``[1, 256]``, ``speed`` float32 ``[1]``) and give token id ``t``:

    frames(t)      = max(1, round(BASE_FRAMES[t] / speed))   (ONNX Round: half to even)
    BASE_FRAMES[t] = 4 for the pad (0), 3 for the silence token (30), else 1 + t % 5

The waveform is the tokens' frames back to back, ``FRAME_SAMPLES`` (600, 25 ms
at 24 kHz) samples per frame, so its length is exactly ``600 * sum(duration)``.
A pad or silence frame is zero. Any other frame alternates ``+a, -a`` sample by
sample, with ``a = 0.3 + 0.1 * (t % 4)``: a tone that TTSAudioEncode's 80 Hz
high-pass filter keeps and its -40 dB silence trim never mistakes for silence,
so the trim removes exactly the silent pads, less its 50 ms lead-in and tail,
and every phoneme's span in the returned audio is exactly where its tone is.
``style`` is accepted and ignored. Both graphs compute the waveform with the
same nodes, so the two models return byte-identical audio.

Each directory also holds ``tokens.txt`` (Kokoro's ids for the symbols the
dictionary uses, plus punctuation and space; ids 0 and 30 have no symbol, as in
Kokoro), ``misaki/us_gold.json`` (a few words with their Misaki US gold
pronunciations, so the default MisakiDictionary backend runs offline) and
``model_metadata.json`` (Phonemize with padding and two silence tokens,
TTSAudioEncode at 24 kHz, and ``trim_trailing_samples`` inside the trailing
pad). There is no voice file, so the SDK feeds a zero style vector.

Built with the ONNX helper API, so the bytes depend only on the ``onnx`` pinned
in ``requirements-forms.txt``; run it in that environment, or the models can
differ from the committed ones. Before writing, the script runs both graphs
through ONNX Runtime and checks the durations, the waveform length, the silent
and loud frames, and that the two models' audio is byte-identical. It prints
each file's sha256.

Usage (pinned ``forms`` env, see README.md)::

    uv run --no-project --python 3.12 --with-requirements tools/conformance/requirements-forms.txt \\
        python tools/conformance/gen_fake_tts.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "integration-tests/fixtures/tts/fake"

OPSET = 18
# IR version 8 is the one ONNX pairs with opset 18; ONNX Runtime 1.23 loads it.
IR_VERSION = 8

SAMPLE_RATE = 24_000
# Samples per duration frame, as in Kokoro (25 ms at 24 kHz).
FRAME_SAMPLES = 600
STYLE_DIM = 256
# Kokoro's vocabulary size: token ids run from 0 to 177.
VOCAB_SIZE = 178
PAD_ID = 0
# The id the Phonemize step's `silence_tokens` prepends (Kokoro's silence token).
SILENCE_ID = 30
PAD_FRAMES = 4
SILENCE_FRAMES = 3
SILENCE_TOKENS = 2
# Cut from the end of each raw chunk before postprocessing; it lands in the
# trailing pad, which stays longer than the 480-sample chunk crossfade.
TRIM_TRAILING_SAMPLES = 240

# Kokoro's token ids for the symbols the fixture can emit: punctuation, space,
# and every character of the dictionary's pronunciations.
VOCAB = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10,
    '"': 11, "(": 12, ")": 13, "“": 14, "”": 15, " ": 16,
    "I": 25, "O": 31, "d": 46, "f": 48, "h": 50, "i": 51, "k": 53, "l": 54,
    "m": 55, "n": 56, "s": 61, "t": 62, "u": 63, "w": 65, "z": 68, "æ": 72,
    "ɔ": 76, "ð": 81, "ə": 83, "ɛ": 86, "ɡ": 92, "ɪ": 102, "ɹ": 123,
    "ʃ": 131, "ʌ": 138, "ˈ": 156, "ˌ": 157,
}

# Misaki US gold pronunciations (`misaki/us_gold.json` of Kokoro-82M).
DICTIONARY = {
    "fire": "fˈIəɹ",
    "flagon": "flˈæɡən",
    "hello": "həlˈO",
    "hi": "hˈI",
    "is": "ɪz",
    "rusty": "ɹˈʌsti",
    "said": "sˈɛd",
    "she": "ʃi",
    "the": "ði",
    "to": "tu",
    "warm": "wˈɔɹm",
    "welcome": "wˈɛlkəm",
}

VARIANTS = {
    "with_duration": "Fake TTS for the phoneme-timing tests: per-token durations and a tone per phoneme",
    "without_duration": "Fake TTS for the phoneme-timing tests: the with_duration waveform, no duration output",
}


def base_frames() -> list[float]:
    """Frames per token id at speed 1."""
    frames = [float(1 + token % 5) for token in range(VOCAB_SIZE)]
    frames[PAD_ID] = float(PAD_FRAMES)
    frames[SILENCE_ID] = float(SILENCE_FRAMES)
    return frames


def amplitudes() -> list[float]:
    """Tone amplitude per token id; the pad and silence token are silent."""
    amps = [0.3 + 0.1 * (token % 4) for token in range(VOCAB_SIZE)]
    amps[PAD_ID] = 0.0
    amps[SILENCE_ID] = 0.0
    return amps


def _ints(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


def _int_scalar(name: str, value: int) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, [], [value])


def build_model(with_duration: bool) -> onnx.ModelProto:
    """The fake TTS graph; ``with_duration`` adds the ``duration`` output."""
    carrier = [1.0 if sample % 2 == 0 else -1.0 for sample in range(FRAME_SAMPLES)]
    initializers = [
        helper.make_tensor("base_frames", TensorProto.FLOAT, [VOCAB_SIZE], base_frames()),
        helper.make_tensor("amplitudes", TensorProto.FLOAT, [VOCAB_SIZE], amplitudes()),
        helper.make_tensor("carrier", TensorProto.FLOAT, [1, FRAME_SAMPLES], carrier),
        helper.make_tensor("one_frame", TensorProto.FLOAT, [], [1.0]),
        _ints("axis_0", [0]),
        _ints("axis_1", [1]),
        _ints("flat", [-1]),
        _int_scalar("scalar_axis_0", 0),
        _int_scalar("zero", 0),
        _int_scalar("one", 1),
    ]
    nodes = [
        helper.make_node("Squeeze", ["tokens", "axis_0"], ["ids"]),
        # Durations, the way Kokoro's predictor ends: divide by speed, round,
        # at least one frame, INT64.
        helper.make_node("Gather", ["base_frames", "ids"], ["frames_at_speed_1"], axis=0),
        helper.make_node("Div", ["frames_at_speed_1", "speed"], ["frames_scaled"]),
        helper.make_node("Round", ["frames_scaled"], ["frames_rounded"]),
        helper.make_node("Max", ["frames_rounded", "one_frame"], ["frames_clipped"]),
        helper.make_node("Cast", ["frames_clipped"], ["frames"], to=TensorProto.INT64),
        # Which token each frame belongs to: the number of tokens that end at
        # or before it.
        helper.make_node("CumSum", ["frames", "scalar_axis_0"], ["token_ends"]),
        helper.make_node("ReduceSum", ["frames", "axis_0"], ["total_frames"], keepdims=0),
        helper.make_node("Range", ["zero", "total_frames", "one"], ["frame_index"]),
        helper.make_node("Unsqueeze", ["token_ends", "axis_0"], ["token_ends_row"]),
        helper.make_node("Unsqueeze", ["frame_index", "axis_1"], ["frame_index_column"]),
        helper.make_node("LessOrEqual", ["token_ends_row", "frame_index_column"], ["ended"]),
        helper.make_node("Cast", ["ended"], ["ended_count"], to=TensorProto.INT64),
        helper.make_node("ReduceSum", ["ended_count", "axis_1"], ["frame_token"], keepdims=0),
        helper.make_node("Gather", ["ids", "frame_token"], ["frame_ids"], axis=0),
        # One tone frame per frame, scaled by its token's amplitude.
        helper.make_node("Gather", ["amplitudes", "frame_ids"], ["frame_amplitude"], axis=0),
        helper.make_node("Unsqueeze", ["frame_amplitude", "axis_1"], ["frame_amplitude_column"]),
        helper.make_node("Mul", ["frame_amplitude_column", "carrier"], ["frame_samples"]),
        helper.make_node("Reshape", ["frame_samples", "flat"], ["audio"]),
    ]
    outputs = [helper.make_tensor_value_info("audio", TensorProto.FLOAT, ["audio_length"])]
    if with_duration:
        nodes.append(helper.make_node("Unsqueeze", ["frames", "axis_0"], ["duration"]))
        outputs.insert(
            0,
            helper.make_tensor_value_info("duration", TensorProto.INT64, [1, "sequence_length"]),
        )
    graph = helper.make_graph(
        nodes,
        "fake_tts_with_duration" if with_duration else "fake_tts_without_duration",
        [
            helper.make_tensor_value_info("tokens", TensorProto.INT64, [1, "sequence_length"]),
            helper.make_tensor_value_info("style", TensorProto.FLOAT, [1, STYLE_DIM]),
            helper.make_tensor_value_info("speed", TensorProto.FLOAT, [1]),
        ],
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=IR_VERSION,
        producer_name="xybrid-conformance",
        producer_version="1",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def metadata(variant: str) -> dict:
    """The variant's ``model_metadata.json``."""
    return {
        "model_id": f"fake-tts-{variant.replace('_', '-')}",
        "version": "1.0",
        "description": VARIANTS[variant],
        "execution_template": {"type": "Onnx", "model_file": "model.onnx"},
        "preprocessing": [
            {
                "type": "Phonemize",
                "tokens_file": "tokens.txt",
                "backend": "MisakiDictionary",
                "add_padding": True,
                "normalize_text": True,
                "silence_tokens": SILENCE_TOKENS,
            }
        ],
        "postprocessing": [
            {"type": "TTSAudioEncode", "sample_rate": SAMPLE_RATE, "apply_postprocessing": True}
        ],
        "files": ["model.onnx", "tokens.txt", "misaki/us_gold.json"],
        "metadata": {
            "task": "text-to-speech",
            "sample_rate": SAMPLE_RATE,
            "notes": "Written by tools/conformance/gen_fake_tts.py; not a real voice",
        },
        "trim_trailing_samples": TRIM_TRAILING_SAMPLES,
    }


def tokens_txt() -> str:
    """``tokens.txt``: one ``SYMBOL ID`` line per symbol, by id."""
    return "".join(f"{symbol} {token}\n" for symbol, token in sorted(VOCAB.items(), key=lambda kv: kv[1]))


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _token_ids(phonemes: str) -> np.ndarray:
    """Ids the way the Phonemize step builds them: pad, silence tokens, one id
    per phoneme character, pad."""
    ids = [PAD_ID] + [SILENCE_ID] * SILENCE_TOKENS + [VOCAB[c] for c in phonemes] + [PAD_ID]
    return np.array([ids], dtype=np.int64)


def check(payloads: dict[str, bytes]) -> None:
    """Run both graphs through ONNX Runtime and check what the tests rely on."""
    for word, phonemes in DICTIONARY.items():
        missing = sorted(set(phonemes) - set(VOCAB))
        assert not missing, f"{word}: {missing} not in VOCAB"
    assert PAD_ID not in VOCAB.values() and SILENCE_ID not in VOCAB.values()
    assert max(VOCAB.values()) < VOCAB_SIZE

    sessions = {
        name: ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
        for name, payload in payloads.items()
    }
    names = {name: [o.name for o in s.get_outputs()] for name, s in sessions.items()}
    assert names == {"with_duration": ["duration", "audio"], "without_duration": ["audio"]}, names

    frames_table = np.array(base_frames(), dtype=np.float32)
    amps = np.array(amplitudes(), dtype=np.float32)
    carrier = np.where(np.arange(FRAME_SAMPLES) % 2 == 0, 1.0, -1.0).astype(np.float32)
    lines = [
        '"həlˈO," ʃi sˈɛd. wˈɛlkəm tu ði ɹˈʌsti flˈæɡən; ði fˈIəɹ ɪz wˈɔɹm!',
        "hˈI",
        "—…()“”?:",
    ]
    for line in lines:
        tokens = _token_ids(line)
        for speed in (0.8, 1.0, 1.25):
            feeds = {
                "tokens": tokens,
                "style": np.zeros((1, STYLE_DIM), dtype=np.float32),
                "speed": np.array([speed], dtype=np.float32),
            }
            duration, audio = sessions["with_duration"].run(["duration", "audio"], feeds)
            (plain,) = sessions["without_duration"].run(["audio"], feeds)
            assert audio.tobytes() == plain.tobytes(), (line, speed)
            assert audio.dtype == np.float32 and audio.ndim == 1, (audio.dtype, audio.shape)
            assert duration.dtype == np.int64 and duration.shape == tokens.shape, duration.shape

            scaled = frames_table[tokens[0]] / np.float32(speed)
            expected = np.maximum(np.round(scaled), 1).astype(np.int64)
            assert np.array_equal(duration[0], expected), (line, speed, duration, expected)
            assert audio.size == FRAME_SAMPLES * int(duration.sum()), (audio.size, duration.sum())

            frames = audio.reshape(-1, FRAME_SAMPLES)
            frame_ids = np.repeat(tokens[0], duration[0])
            assert np.array_equal(frames, amps[frame_ids][:, None] * carrier[None, :])
            silent = np.isin(frame_ids, [PAD_ID, SILENCE_ID])
            assert not frames[silent].any() and np.abs(frames[~silent]).min() >= 0.3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args(argv)

    payloads = {
        variant: build_model(variant == "with_duration").SerializeToString(deterministic=True)
        for variant in VARIANTS
    }
    check(payloads)

    shared = {
        "tokens.txt": tokens_txt(),
        "misaki/us_gold.json": _json(DICTIONARY),
    }
    for variant, payload in payloads.items():
        directory = args.out / variant
        (directory / "misaki").mkdir(parents=True, exist_ok=True)
        files = {
            "model.onnx": payload,
            "model_metadata.json": _json(metadata(variant)).encode("utf-8"),
            **{name: text.encode("utf-8") for name, text in shared.items()},
        }
        for name, data in files.items():
            (directory / name).write_bytes(data)
            print(f"{hashlib.sha256(data).hexdigest()}  {variant}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
