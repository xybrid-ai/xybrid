#!/usr/bin/env python3
"""Write the tiny fake ONNX choice scorers used by the contract tests.

Each fake takes the five named tensors of the FORMS scorer spec
(``specs/cua-s1-forms.json``) and computes a logit a test can predict:

    logit[n] = 0.01 * sum(option_ids[n] * option_token_mask[n])
             + 0.001 * sum(context_ids * context_mask)

- ``dynamic.onnx``: dynamic choice axis, output ``[1, N]``.
- ``static_masked.onnx``: static axis of 8, output ``[1, 8]``; masked choices are
  NaN, so a caller must drop the padding tail before checking active logits.
- ``short_output.onnx``: dynamic axis, output ``[1, N - 1]``.
- ``wrong_dtype.onnx``: dynamic axis, float64 output.

Built with the ONNX helper API, so the bytes depend only on the pinned ``onnx``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "integration-tests/fixtures/choice/fake"

CONTEXT_LEN = 224
CHOICE_LEN = 96
STATIC_CHOICES = 8
OPSET = 18
# IR version 8 is the one ONNX pairs with opset 18; ONNX Runtime 1.23 loads it.
IR_VERSION = 8
OPTION_WEIGHT = 0.01
CONTEXT_WEIGHT = 0.001


def _inputs(choices: int | str) -> list[onnx.ValueInfoProto]:
    return [
        helper.make_tensor_value_info("context_ids", TensorProto.INT64, [1, CONTEXT_LEN]),
        helper.make_tensor_value_info("context_mask", TensorProto.BOOL, [1, CONTEXT_LEN]),
        helper.make_tensor_value_info("option_ids", TensorProto.INT64, [1, choices, CHOICE_LEN]),
        helper.make_tensor_value_info(
            "option_token_mask", TensorProto.BOOL, [1, choices, CHOICE_LEN]
        ),
        helper.make_tensor_value_info("option_mask", TensorProto.BOOL, [1, choices]),
    ]


def _scalar(name: str, value: float) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.FLOAT, [], [value])


def _score_nodes() -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """Nodes computing ``raw`` ([1, N] float32) from the five inputs."""
    initializers = [
        _scalar("option_weight", OPTION_WEIGHT),
        _scalar("context_weight", CONTEXT_WEIGHT),
        helper.make_tensor("last_axis", TensorProto.INT64, [1], [-1]),
    ]
    nodes = [
        helper.make_node("Cast", ["option_ids"], ["option_ids_f"], to=TensorProto.FLOAT),
        helper.make_node(
            "Cast", ["option_token_mask"], ["option_token_mask_f"], to=TensorProto.FLOAT
        ),
        helper.make_node("Mul", ["option_ids_f", "option_token_mask_f"], ["option_bytes"]),
        helper.make_node("ReduceSum", ["option_bytes", "last_axis"], ["option_sum"], keepdims=0),
        helper.make_node("Mul", ["option_sum", "option_weight"], ["option_term"]),
        helper.make_node("Cast", ["context_ids"], ["context_ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["context_mask"], ["context_mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["context_ids_f", "context_mask_f"], ["context_bytes"]),
        # keepdims=1 leaves [1, 1], which broadcasts over the [1, N] option term.
        helper.make_node("ReduceSum", ["context_bytes", "last_axis"], ["context_sum"], keepdims=1),
        helper.make_node("Mul", ["context_sum", "context_weight"], ["context_term"]),
        helper.make_node("Add", ["option_term", "context_term"], ["raw"]),
    ]
    return nodes, initializers


def _model(
    name: str,
    choices: int | str,
    output_shape: list[int | str],
    tail: list[onnx.NodeProto],
    extra_initializers: list[onnx.TensorProto],
    output_type: int = TensorProto.FLOAT,
) -> onnx.ModelProto:
    nodes, initializers = _score_nodes()
    graph = helper.make_graph(
        nodes + tail,
        name,
        _inputs(choices),
        [helper.make_tensor_value_info("logits", output_type, output_shape)],
        initializer=initializers + extra_initializers,
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


def _masked(output: str, fill: str) -> onnx.NodeProto:
    """Replace the logits of masked choices with the scalar initializer ``fill``."""
    return helper.make_node("Where", ["option_mask", "raw", fill], [output])


def build_models() -> dict[str, onnx.ModelProto]:
    """Return every fake scorer keyed by its output file name."""
    # On a dynamic axis every offered choice is active, so masking with -inf
    # never changes a logit; it keeps all five inputs live in the graph.
    neg_inf = _scalar("neg_inf", float("-inf"))
    slice_bounds = [
        helper.make_tensor("slice_start", TensorProto.INT64, [1], [0]),
        helper.make_tensor("slice_end", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("slice_axis", TensorProto.INT64, [1], [1]),
    ]
    return {
        "dynamic.onnx": _model(
            "fake_dynamic",
            "choices",
            [1, "choices"],
            [_masked("logits", "neg_inf")],
            [neg_inf],
        ),
        "static_masked.onnx": _model(
            "fake_static_masked",
            STATIC_CHOICES,
            [1, STATIC_CHOICES],
            [_masked("logits", "nan")],
            [_scalar("nan", float("nan"))],
        ),
        "short_output.onnx": _model(
            "fake_short_output",
            "choices",
            [1, "choices_minus_one"],
            [
                _masked("full", "neg_inf"),
                helper.make_node(
                    "Slice", ["full", "slice_start", "slice_end", "slice_axis"], ["logits"]
                ),
            ],
            [neg_inf, *slice_bounds],
        ),
        "wrong_dtype.onnx": _model(
            "fake_wrong_dtype",
            "choices",
            [1, "choices"],
            [
                _masked("full", "neg_inf"),
                helper.make_node("Cast", ["full"], ["logits"], to=TensorProto.DOUBLE),
            ],
            [neg_inf],
            output_type=TensorProto.DOUBLE,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for file_name, model in build_models().items():
        payload = model.SerializeToString(deterministic=True)
        (args.out / file_name).write_bytes(payload)
        print(f"{hashlib.sha256(payload).hexdigest()}  {file_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
