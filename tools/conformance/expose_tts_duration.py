#!/usr/bin/env python3
"""Expose a TTS model's per-token durations as a second graph output.

Kokoro-82M (and other StyleTTS2-style exports) compute how many acoustic frames
each input token lasts before generating any audio, then use those counts to
upsample the features. The exported graph keeps that tensor internal and
declares only ``audio``. This tool appends an ``Identity`` named ``duration``
over it, so the Xybrid TTS executor can report per-phoneme timings, while the
audio stays byte-identical (no node, weight or initializer changes).

The duration tensor is found by structure, not by name, so a re-export with
different internal names still works: it is the single INT64 ``Cast`` whose
output feeds a ``Gather`` that feeds the ``CumSum`` building the alignment.
In the shipped Kokoro-82M v1.0 export that tensor is ``/encoder/Cast_output_0``
(``Cast(Clip(Round(ReduceSum(Sigmoid(duration_proj)) / speed), min=1))``,
shape ``[1, sequence_length]``, one frame = 600 samples = 25 ms at 24 kHz).

Usage (pinned env, see README.md)::

    uv run --no-project --python 3.12 --with-requirements tools/conformance/requirements-forms.txt \\
        python tools/conformance/expose_tts_duration.py SRC/model.onnx DST/model.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

DURATION_OUTPUT = "duration"


def find_duration_tensor(graph: onnx.GraphProto) -> str:
    """Return the name of the INT64 Cast output that drives the alignment CumSum."""
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def casts_to_int64(node: onnx.NodeProto) -> bool:
        return node.op_type == "Cast" and any(
            a.name == "to" and a.i == TensorProto.INT64 for a in node.attribute
        )

    def feeds_alignment(name: str) -> bool:
        return any(
            gather.op_type == "Gather"
            and any(c.op_type == "CumSum" for c in consumers.get(gather.output[0], []))
            for gather in consumers.get(name, [])
        )

    hits = [n.output[0] for n in graph.node if casts_to_int64(n) and feeds_alignment(n.output[0])]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one duration tensor, found {len(hits)}: {hits}")
    return hits[0]


def initializer_digest(graph: onnx.GraphProto) -> str:
    return hashlib.sha256(b"".join(i.SerializeToString() for i in graph.initializer)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="source model.onnx (single 'audio' output)")
    parser.add_argument("dst", type=Path, help="where to write the grafted model.onnx")
    parser.add_argument("--frame-samples", type=int, default=600, help="samples per duration frame (metadata only)")
    args = parser.parse_args()

    model = onnx.load(str(args.src))
    graph = model.graph
    outputs = [o.name for o in graph.output]
    if DURATION_OUTPUT in outputs:
        raise SystemExit(f"{args.src} already declares a '{DURATION_OUTPUT}' output")
    if len(outputs) != 1:
        raise SystemExit(f"expected a single audio output, found {outputs}")

    source = find_duration_tensor(graph)
    taken = {t for n in graph.node for t in (*n.input, *n.output)}
    taken |= {i.name for i in graph.initializer} | {i.name for i in graph.input}
    if DURATION_OUTPUT in taken:
        raise SystemExit(f"tensor name '{DURATION_OUTPUT}' is already used inside the graph")

    digest = initializer_digest(graph)
    graph.node.append(helper.make_node("Identity", [source], [DURATION_OUTPUT], name="xybrid_duration"))
    graph.output.append(helper.make_tensor_value_info(DURATION_OUTPUT, TensorProto.INT64, [1, "sequence_length"]))
    helper.set_model_props(
        model,
        {
            "xybrid.duration_output": DURATION_OUTPUT,
            "xybrid.duration_unit": "frames",
            "xybrid.duration_frame_samples": str(args.frame_samples),
            "xybrid.duration_source": source,
        },
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.dst))

    check = onnx.load(str(args.dst))
    if [o.name for o in check.graph.output] != [*outputs, DURATION_OUTPUT]:
        raise SystemExit(f"unexpected outputs after graft: {[o.name for o in check.graph.output]}")
    if initializer_digest(check.graph) != digest:
        raise SystemExit("initializers changed during the graft")
    print(f"ok: {source} -> '{DURATION_OUTPUT}' in {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
