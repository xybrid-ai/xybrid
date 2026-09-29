#!/usr/bin/env python3
"""Export the pinned CUA-S1-FORMS checkpoint to ONNX, or refuse to write it.

The graph takes five tensors at fixed byte lengths with a dynamic choice axis N
and returns ``logits`` of shape ``[1, N]``:

    context_ids        int64 [1, 224]     UTF-8 bytes + 1, zero padded
    context_mask       bool  [1, 224]     context_ids != 0
    option_ids         int64 [1, N, 96]   UTF-8 bytes + 1 per choice, zero padded
    option_token_mask  bool  [1, N, 96]   option_ids != 0
    option_mask        bool  [1, N]       true for every offered choice

The checkpoint loads only through the pinned upstream loader
(``cua_s1.model.load_checkpoint``), which refuses pickles and verifies the
tensor signature. Nothing is written unless the export gates hold on the
committed cases plus seeded random ones (README.md, "Export gates"): ONNX
Runtime vs PyTorch on the fixed tensors, the fixed padding vs the upstream
collator, and both at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from common import deterministic_torch, softmax

CONTEXT_LEN = 224
CHOICE_LEN = 96
BYTE_OFFSET = 1
FIXED_CHOICES = ("check", "click", "skip")
MAX_CHOICES = 64
INPUT_NAMES = ("context_ids", "context_mask", "option_ids", "option_token_mask", "option_mask")
OUTPUT_NAME = "logits"
OPSET = 18
# IR version 8 is the one ONNX pairs with opset 18; ONNX Runtime 1.23 loads it.
IR_VERSION = 8
EXPECTED_PARAMETERS = 706_048
EXPECTED_CONFIG = {"encoder": "tinyx", "context_tokens": CONTEXT_LEN, "option_tokens": CHOICE_LEN}
RANDOM_CASES = 200
SEED = 20260928
SCORE_TOLERANCE = 1e-5
# Just above the float32 floor: logits reach |37| (README.md, "Export gates").
LOGIT_TOLERANCE = 5e-5


def encode(text: str, max_len: int) -> list[int]:
    """The xybrid byte encoder: raw UTF-8 bytes, truncated, offset, zero padded."""
    raw = text.encode("utf-8")[:max_len]
    return [byte + BYTE_OFFSET for byte in raw] + [0] * (max_len - len(raw))


def fixed_tensors(context: str, options: list[str]) -> dict[str, np.ndarray]:
    context_ids = np.array([encode(context, CONTEXT_LEN)], dtype=np.int64)
    option_ids = np.array([[encode(option, CHOICE_LEN) for option in options]], dtype=np.int64)
    return {
        "context_ids": context_ids,
        "context_mask": context_ids != 0,
        "option_ids": option_ids,
        "option_token_mask": option_ids != 0,
        "option_mask": np.ones((1, len(options)), dtype=bool),
    }


class FormsGraph(torch.nn.Module):
    """The upstream scorer called with the five tensors as positional inputs."""

    def __init__(self, scorer: torch.nn.Module) -> None:
        super().__init__()
        self.scorer = scorer

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        option_ids: torch.Tensor,
        option_token_mask: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.scorer(
            {
                "context_ids": context_ids,
                "context_mask": context_mask,
                "option_ids": option_ids,
                "option_token_mask": option_token_mask,
                "option_mask": option_mask,
            }
        )


def torch_fixed_logits(graph: FormsGraph, tensors: dict[str, np.ndarray]) -> np.ndarray:
    with torch.no_grad():
        return graph(*[torch.from_numpy(tensors[name]) for name in INPUT_NAMES]).numpy()[0]


def torch_collator_logits(
    model: torch.nn.Module, collator: Any, example_type: Any, context: str, options: list[str]
) -> np.ndarray:
    """Logits through the upstream collator, as the reference implementation pads."""
    batch = collator([example_type(context=context, options=tuple(options), label=0)])
    with torch.no_grad():
        return model(batch).numpy()[0]


def export_graph(graph: FormsGraph, workdir: Path) -> onnx.ModelProto:
    sample = fixed_tensors('TASK sample\nFORM x\nELEMENT Edit "x" value=""', ["a", "bb", "c"])
    args = tuple(torch.from_numpy(sample[name]) for name in INPUT_NAMES)
    choices = torch.export.Dim("choices", min=1, max=MAX_CHOICES)
    dynamic_shapes = {
        "context_ids": None,
        "context_mask": None,
        "option_ids": {1: choices},
        "option_token_mask": {1: choices},
        "option_mask": {1: choices},
    }
    path = workdir / "raw.onnx"
    # The attention fast path is an eval-mode kernel with no ONNX lowering, so
    # the export traces the plain operators; the gates check them.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        torch.onnx.export(
            graph,
            args,
            str(path),
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_shapes=dynamic_shapes,
            opset_version=OPSET,
            dynamo=True,
            external_data=False,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    return onnx.load(str(path))


def normalize(model: onnx.ModelProto) -> onnx.ModelProto:
    """Strip exporter metadata (source locations, producer) that makes bytes machine-dependent."""

    def clear(*items: Any) -> None:
        for item in items:
            item.doc_string = ""
            del item.metadata_props[:]

    def clean_graph(graph: onnx.GraphProto) -> None:
        clear(graph, *graph.input, *graph.output, *graph.value_info)
        clear(*graph.initializer, *graph.node)
        for attribute in (attribute for node in graph.node for attribute in node.attribute):
            attribute.doc_string = ""
            if attribute.HasField("g"):
                clean_graph(attribute.g)
            for subgraph in attribute.graphs:
                clean_graph(subgraph)

    clear(model)
    model.producer_name = "xybrid-conformance"
    model.producer_version = "1"
    model.domain = ""
    model.model_version = 0
    model.ir_version = IR_VERSION
    clean_graph(model.graph)
    for function in model.functions:
        clear(function, *function.node)
    onnx.checker.check_model(model, full_check=True)
    return model


def check_contract(model: onnx.ModelProto) -> None:
    """The exported graph must expose exactly the scorer's tensor contract."""
    opsets = {entry.domain: entry.version for entry in model.opset_import}
    if opsets.get("", opsets.get("ai.onnx")) != OPSET:
        raise SystemExit(f"export used opset {opsets}, expected {OPSET}")

    def signature(values: Any) -> list[tuple[str, int, list[int | str]]]:
        return [
            (
                value.name,
                value.type.tensor_type.elem_type,
                [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim],
            )
            for value in values
        ]

    expected = [
        ("context_ids", onnx.TensorProto.INT64, [1, CONTEXT_LEN]),
        ("context_mask", onnx.TensorProto.BOOL, [1, CONTEXT_LEN]),
        ("option_ids", onnx.TensorProto.INT64, [1, "choices", CHOICE_LEN]),
        ("option_token_mask", onnx.TensorProto.BOOL, [1, "choices", CHOICE_LEN]),
        ("option_mask", onnx.TensorProto.BOOL, [1, "choices"]),
    ]
    inputs = signature(model.graph.input)
    if inputs != expected:
        raise SystemExit(f"exported inputs {inputs} differ from the contract {expected}")
    outputs = signature(model.graph.output)
    if outputs != [(OUTPUT_NAME, onnx.TensorProto.FLOAT, [1, "choices"])]:
        raise SystemExit(f"exported outputs {outputs} differ from the contract")


def verification_cases(cases_path: Path) -> list[tuple[str, str, list[str]]]:
    """The committed cases plus seeded random ones, each with the fixed choices."""
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = [
        (case["id"], case["context"], [c["text"] for c in case["choices"]] + list(FIXED_CHOICES))
        for case in document["cases"]
    ]
    rng = random.Random(SEED)
    alphabet = (
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 :;,.-()/@\"'"
        "éèàçÉöüßñ€—–’“”…中文日本語한국어📞✓"
    )

    def text(max_chars: int) -> str:
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, max_chars)))

    for index in range(RANDOM_CASES):
        context = (
            "TASK fill the form from the document, then submit\n"
            f'FORM {text(64)}\nELEMENT Edit "{text(72)}" value="{text(48)}"'
        )
        callers = [f"fill {text(20)}: {text(90)}" for _ in range(rng.randint(0, MAX_CHOICES - 3))]
        cases.append((f"random-{index:03d}", context, callers + list(FIXED_CHOICES)))
    return cases


def verify(
    onnx_bytes: bytes,
    graph: FormsGraph,
    model: torch.nn.Module,
    collator: Any,
    example_type: Any,
    cases: list[tuple[str, str, list[str]]],
) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # xybrid opens ONNX sessions at Level3, which is ORT_ENABLE_ALL.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_bytes, options, providers=["CPUExecutionProvider"])
    gates = ("export_parity", "padding", "export_vs_reference")
    worst = {gate: {"logit": 0.0, "score": 0.0} for gate in gates}
    failures = []
    for case_id, context, choice_texts in cases:
        tensors = fixed_tensors(context, choice_texts)
        onnx_logits = session.run([OUTPUT_NAME], {k: tensors[k] for k in INPUT_NAMES})[0][0]
        fixed = torch_fixed_logits(graph, tensors)
        reference = torch_collator_logits(model, collator, example_type, context, choice_texts)
        if not (onnx_logits.shape == fixed.shape == reference.shape == (len(choice_texts),)):
            failures.append(
                f"{case_id}: shapes {onnx_logits.shape} {fixed.shape} {reference.shape}"
            )
            continue
        pairs = zip(gates, ((onnx_logits, fixed), (fixed, reference), (onnx_logits, reference)))
        for gate, (left, right) in pairs:
            logit = float(np.max(np.abs(left - right)))
            score = float(np.max(np.abs(softmax(left) - softmax(right))))
            worst[gate]["logit"] = max(worst[gate]["logit"], logit)
            worst[gate]["score"] = max(worst[gate]["score"], score)
            if not (logit <= LOGIT_TOLERANCE and score <= SCORE_TOLERANCE):
                failures.append(f"{case_id}: {gate} max |Δlogit| {logit:.3g}, |Δscore| {score:.3g}")
    if failures:
        raise SystemExit(
            f"refusing to write the export (bounds: |Δlogit| <= {LOGIT_TOLERANCE:g}, "
            f"|Δscore| <= {SCORE_TOLERANCE:g}):\n  " + "\n  ".join(failures)
        )
    return {
        "cases": len(cases),
        "bounds": {"logit": LOGIT_TOLERANCE, "score": SCORE_TOLERANCE},
        "max_abs": worst,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cua-src", type=Path, required=True, help="directory holding the pinned cua_s1 package"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="cua-s1-forms.safetensors")
    parser.add_argument("--cases", type=Path, required=True, help="cases/forms.json")
    parser.add_argument("--out", type=Path, required=True, help="ONNX file to write")
    parser.add_argument("--report", type=Path, help="write the verification report as JSON")
    args = parser.parse_args(argv)

    deterministic_torch(SEED)
    sys.path.insert(0, str(args.cua_src.resolve()))
    from cua_s1.model import ChoiceExample, load_checkpoint, parameter_count

    model, collator, config = load_checkpoint(args.checkpoint, "cpu")
    mismatched = {k: config.get(k) for k, v in EXPECTED_CONFIG.items() if config.get(k) != v}
    if mismatched or parameter_count(model) != EXPECTED_PARAMETERS:
        raise SystemExit(
            f"unexpected checkpoint: config {mismatched}, {parameter_count(model)} parameters"
        )
    graph = FormsGraph(model).eval()

    with tempfile.TemporaryDirectory() as workdir:
        exported = normalize(export_graph(graph, Path(workdir)))
    check_contract(exported)
    payload = exported.SerializeToString(deterministic=True)
    report = verify(payload, graph, model, collator, ChoiceExample, verification_cases(args.cases))
    report.update(
        {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "opset": OPSET,
            "ir_version": IR_VERSION,
            "versions": {
                "torch": torch.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": ort.__version__,
            },
        }
    )

    # Written under a temporary name and renamed, so a failure never leaves a
    # partial file at --out.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(f".{args.out.name}.partial")
    partial.write_bytes(payload)
    partial.replace(args.out)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
