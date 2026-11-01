#!/usr/bin/env python3
"""Run LiquidAI's own d1 code (the pinned LiquidAI/d1-2B snapshot) on decision cases, one question per sequence.

For each case file (case schema v1, integration-tests/fixtures/decision/schema.json) the runner reads `id` and
`request_text`, nothing else. It parses the request with Python's json module, validates its questions with the
reference's own `check_question`, and hands `state` and each question to the snapshot's own
`model.system_one(state, {id: question}, set_size=1)`: one question alone in one sequence, a batch of one, no padding,
the reading llama.cpp's server gives every question. Hooks record what that call evaluated, without changing it.

Each case gets one observation file, `<out>/<id>.json`, whose `hf` block has the stages and shapes of the case
schema's `upstream` block, so the two references read alike:

- parse: each question's id, type and option keys in model order;
- prepare: per pass the prompt the reference renders, its token ids, the output index (the answer slot) and the
  selected label tokens; per question the groups (key, code, range of `select`) and the temperature;
- evaluate: per pass the answer slot's raw logits for the selected tokens (the LM head's output, before any
  normalization), and each group's maximum;
- finalize: the answers `system_one` returns, and the usage.

The `detail` block holds what only the reference has: the vocabulary log-probabilities its runner actually reads
(`runner.py` `_slot_logits`: logits minus their logsumexp, in float32), their group maxima, the logsumexp, the
distribution it softmaxes from them (`prompt.readouts`, float64), its float64 temperature and the JSON parser's notes.
A refusal is `{"refusal": {question?, exception, message}}`: the Python exception and the reference's own message (the
reference has no HTTP status). f32 values are bit patterns, f64 values 17 significant digits (an integral one keeps a
".0"), non-finite values strings, as in the case schema. Timings and memory go to `--run-log`, not the observations.

Numerics: eval and inference mode, deterministic algorithms, a fixed thread count, and a float32 forward. The
checkpoint stores bfloat16 weights; `--precision f32` (the default) keeps them in bfloat16 between uses and gives each
decoder layer, the embeddings, the final norm and the LM head exact float32 copies of their own parameters for the
duration of their forward, so every operation runs in float32 on the values a float32 load holds, within 8 GB.
`--check-f32-parity` proves it during the run: every module call is replayed on a float32 copy of the module and
must return the same bytes from float32 inputs, and every hooked module must be verified at least once, or the run
fails. `--precision f32-load` is the model card's CPU recipe (`from_pretrained(dtype=torch.float32)`, about 9 GB, too
large for this machine); `--precision bf16` runs the bfloat16 forward, for comparison.

`--set-size default` runs each request in one call at the checkpoint's own set size (16 questions packed after one
state), the reference's default serving mode, for the packing row of DIFFERENCES.md.

A run writes its observations only once every case has run and the proof (if asked for) holds; any broken invariant
exits 1 with nothing written. `--run-log` records what the observations were made from, as it stood when the run
started (the pins, the verified snapshot files, the lock, the recipe sources `RECIPE`, the request digests) and the
digest of every observation written; provenance.py carries those identities and refuses any that no longer hold.

Usage: README.md. Run it through run.sh, which supplies the pinned environment.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import platform
import resource
import struct
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PINS = HERE / "pins.json"
REQUIREMENTS = HERE / "requirements.txt"
FIXTURES = REPO_ROOT / "integration-tests/fixtures/decision/hf"
# Snapshot, module cache, runs: git-ignored here and declared by clean.sh.
DEFAULT_CACHE = HERE / ".cache"
# The sources whose bytes decide an observation's bytes besides the pins and the lock: every run log records their
# sha256 at run time, and provenance.py refuses observations whose recorded sources differ from the current ones.
RECIPE = ("run_reference.py", "run.sh")
# The one pinned file a weight-free entry point (synthetic_median.py) neither needs nor verifies.
WEIGHTS = "model.safetensors"
# The coordinator's settings ids (cpu-t4-fa-off, ...) name llama.cpp contexts; these name the reference's runs and
# are defined in the fixtures' provenance.json.
SETTINGS_ID = "hf-cpu-{precision}-t{threads}"
SCHEMA_VERSION = 1
# The question types whose answer is a softmax over option tokens, as in the case schema; the reference's typed
# answers (reads, estimates, boxes, multiple choices) fall outside it and are recorded as it answers them.
OPTION_TYPES = ("noul", "choice", "score")
MEDIA_FIELDS = ("files", "images", "videos", "media")


# --------------------------------------------------------------------------- #
# value encoding (case schema v1)
# --------------------------------------------------------------------------- #


class F64(float):
    """An f64 value, written with 17 significant digits (`f64_text`)."""


def f64(value: float) -> F64 | str:
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return F64(value)


def f32_hex(value: float) -> str:
    """An f32 value as its bit pattern (exact for a value that came from float32)."""
    return "0x" + struct.pack(">f", value).hex()


def f64_tree(value: Any) -> Any:
    """Every float of a reference answer as an f64 value; the rest as JSON."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return f64(value)
    if isinstance(value, dict):
        return {str(k): f64_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [f64_tree(v) for v in value]
    return repr(value)


def f64_text(value: float) -> str:
    """%.17g, with ".0" on an integral value so that it reads back as a float (case schema v1)."""
    text = "%.17g" % value
    return text if any(c in text for c in ".e") else text + ".0"


def scalar(value: Any) -> str:
    if isinstance(value, F64):
        return f64_text(value)
    return json.dumps(value, ensure_ascii=False)


def dumps(value: Any, level: int = 0) -> str:
    """JSON with indented objects and one-line scalar arrays, so ids and bit patterns read as rows."""
    pad = "  " * (level + 1)
    if isinstance(value, dict) and value:
        items = [f"{pad}{json.dumps(k, ensure_ascii=False)}: {dumps(v, level + 1)}" for k, v in value.items()]
        text = "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    elif isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        items = [f"{pad}{dumps(item, level + 1)}" for item in value]
        text = "[\n" + ",\n".join(items) + "\n" + "  " * level + "]"
    elif isinstance(value, list):
        text = "[" + ", ".join(scalar(item) for item in value) + "]"
    elif isinstance(value, dict):
        text = "{}"
    else:
        text = scalar(value)
    return text + ("\n" if level == 0 else "")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# pins
# --------------------------------------------------------------------------- #


def recipe(names: tuple[str, ...]) -> dict[str, str]:
    """The sha256 of each of this folder's `names`: a run's recipe, recorded when the run starts."""
    return {name: sha256_file(HERE / name) for name in names}


def verify_snapshot(snapshot: Path, pins: dict, without: tuple[str, ...] = ()) -> list[dict[str, str]]:
    """Refuse unless every pinned file but `without` is in `snapshot` with its size and sha256, and no unpinned code
    is. Model inference verifies all of them; a weight-free entry point passes `without=(WEIGHTS,)`."""
    verified = []
    for name, pin in sorted(pins["files"].items()):
        if name in without:
            continue
        path = snapshot / name
        if not path.is_file():
            hint = " (download.sh --no-weights fetches all but the weights)" if without else ""
            raise SystemExit(f"{path} is missing: run download.sh{hint}")
        if path.stat().st_size != pin["bytes"]:
            raise SystemExit(f"{path} has {path.stat().st_size} bytes, pinned {pin['bytes']}")
        actual = sha256_file(path)
        if actual != pin["sha256"]:
            raise SystemExit(f"{path} is not the pinned file (sha256 {actual}, pinned {pin['sha256']})")
        verified.append({"file": name, "sha256": actual})
    stray = sorted(p.name for p in snapshot.glob("*.py") if p.name not in pins["files"])
    if stray:
        raise SystemExit(f"{snapshot} holds unpinned code: {stray}")
    return verified


def loaded_code(root: str, pins: dict) -> list[dict[str, str]]:
    """The snapshot modules this process imported, each byte-identical to its pinned file.

    transformers copies a remote-code snapshot into `transformers_modules.<name>.<hash>` once for the config and
    once for the model class, so a pinned file can run from two such copies; every one is checked."""
    modules = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(root + ".") or module is None or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__)
        if path.name == "__init__.py":
            continue
        pin = pins["files"].get(path.name)
        if pin is None:
            raise SystemExit(f"the reference imported {name} ({path}), which is not pinned")
        actual = sha256_file(path)
        if actual != pin["sha256"]:
            raise SystemExit(f"{name} runs from {path}, which differs from the pinned {path.name}")
        modules.append({"module": name[len(root) + 1 :], "sha256": actual})
    return modules


# --------------------------------------------------------------------------- #
# request parsing
# --------------------------------------------------------------------------- #


def parse_request(text: str) -> tuple[Any, dict[str, Any]]:
    """`request_text` as the Python value the reference API takes, and what Python's json module did with it.

    The reference has no request parser of its own: its API takes Python objects. This is json.loads with its
    defaults (a repeated key keeps its first position and its last value; NaN and Infinity are read; integers keep
    every digit), observed: repeated keys, non-finite constants, integers outside int64, lone surrogates."""
    notes: dict[str, Any] = {"duplicate_keys": [], "non_finite": [], "integers_beyond_int64": [], "lone_surrogates": 0}

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                notes["duplicate_keys"].append(key)
            out[key] = value
        return out

    def constant(name: str) -> float:
        notes["non_finite"].append(name)
        return float(name.replace("Infinity", "inf"))

    def integer(digits: str) -> int:
        value = int(digits)
        if not -(2**63) <= value < 2**63:
            notes["integers_beyond_int64"].append(digits)
        return value

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_int=integer)

    def surrogates(item: Any) -> int:
        if isinstance(item, str):
            return sum(0xD800 <= ord(c) <= 0xDFFF for c in item)
        if isinstance(item, dict):
            return sum(surrogates(k) + surrogates(v) for k, v in item.items())
        if isinstance(item, list):
            return sum(surrogates(v) for v in item)
        return 0

    notes["lone_surrogates"] = surrogates(value)
    return value, {k: v for k, v in notes.items() if v}


class RunnerError(Exception):
    """A broken invariant of this runner, never a refusal by the reference."""


def refusal(error: BaseException | str, message: str | None = None, question: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if question is not None:
        out["question"] = question
    out["exception"] = error if isinstance(error, str) else type(error).__name__
    out["message"] = message if message is not None else ("" if isinstance(error, str) else str(error))
    return {"refusal": out}


# --------------------------------------------------------------------------- #
# the reference
# --------------------------------------------------------------------------- #


class Float32Compute:
    """Each listed module runs in float32 over parameters stored in bfloat16.

    A pre-hook replaces the data of every bfloat16 parameter under the module with its float32 widening (exact) and
    the forward hook puts the bfloat16 data back, so the module computes exactly what a float32 load computes, while
    only one module's float32 copy exists at a time. The modules must cover every parameter the forward reads."""

    def __init__(self, torch_module, modules: list) -> None:
        self.torch = torch_module
        self.saved: dict[int, list] = {}
        for module in modules:
            module.register_forward_pre_hook(self.widen)
            module.register_forward_hook(self.restore)

    def widen(self, module, args) -> None:
        saved = []
        for param in module.parameters():
            if param.dtype == self.torch.bfloat16:
                saved.append((param, param.data))
                param.data = param.data.to(self.torch.float32)
        self.saved[id(module)] = saved

    def restore(self, module, args, output) -> None:
        for param, data in self.saved.pop(id(module), ()):
            param.data = data
        hidden = output[0] if isinstance(output, tuple) else output
        if self.torch.is_tensor(hidden) and hidden.dtype != self.torch.float32:
            raise RunnerError(f"{type(module).__name__} returned {hidden.dtype} under float32 compute")

    def restore_all(self) -> None:
        """After a failed forward: no module keeps its float32 copy."""
        for saved in self.saved.values():
            for param, data in saved:
                param.data = data
        self.saved.clear()


class ParityCheck:
    """Proves `Float32Compute` equals a float32 load, module by module, on real inputs, or stops the run.

    Every module `Float32Compute` covers records the inputs and the output of each call. `verify` replays each
    recorded call on a float32 deep copy of the module (`copy.deepcopy(module).to(torch.float32)`: exactly the
    parameters a float32 load holds) and raises `RunnerError` unless every floating input tensor is float32 and the
    replay returns the recorded output's structure (types, lengths, keys), dtypes and shapes with the same bytes.
    `require_coverage` raises unless at least one call was verified and every hooked module at least once. The
    modules chain (embedding, each decoder layer, final norm, LM head), so equal outputs from equal inputs make the
    whole forward a float32 load's. A float32 load itself (about 9 GB) does not fit this machine's 8 GB."""

    HOOKS = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
    )

    def __init__(self, torch_module, modules: list) -> None:
        self.torch = torch_module
        self.modules = list(modules)
        self.pending: dict[int, tuple] = {}
        self.records: list[tuple] = []
        self.verified: set[int] = set()
        self.calls = 0
        for module in self.modules:
            module.register_forward_pre_hook(self.before, with_kwargs=True)
            module.register_forward_hook(self.after, with_kwargs=True)

    def before(self, module, args, kwargs) -> None:
        self.pending[id(module)] = (args, kwargs)

    def after(self, module, args, kwargs, output) -> None:
        recorded_args, recorded_kwargs = self.pending.pop(id(module))
        self.records.append((module, recorded_args, recorded_kwargs, output))

    def discard(self) -> None:
        """Forget the calls of a forward that raised: they reach no observation."""
        self.pending.clear()
        self.records.clear()

    def float32_copy(self, module):
        saved = {name: getattr(module, name) for name in self.HOOKS if hasattr(module, name)}
        for name, hooks in saved.items():
            setattr(module, name, type(hooks)())
        try:
            clone = copy.deepcopy(module)
        finally:
            for name, hooks in saved.items():
                setattr(module, name, hooks)
        return clone.to(self.torch.float32)

    def require_float32(self, value: Any, where: str) -> None:
        """Every floating tensor in `value` (tuples, lists, dicts, dataclasses, walked) is float32."""
        torch = self.torch
        if torch.is_tensor(value):
            if value.is_floating_point() and value.dtype != torch.float32:
                raise RunnerError(f"float32 parity: {where} is {value.dtype}, not float32")
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                self.require_float32(item, f"{where}[{i}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                self.require_float32(item, f"{where}[{key!r}]")
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                self.require_float32(getattr(value, field.name), f"{where}.{field.name}")

    def bits(self, tensor):
        return tensor.detach().contiguous().reshape(-1).view(self.torch.uint8)

    def compare(self, recorded: Any, again: Any, where: str) -> None:
        """Raise unless `again` has `recorded`'s structure and every tensor in it the same dtype, shape and bytes."""
        torch = self.torch
        if torch.is_tensor(recorded) or torch.is_tensor(again):
            if not (torch.is_tensor(recorded) and torch.is_tensor(again)):
                raise RunnerError(f"float32 parity: {where}: {type(recorded).__name__} recorded, "
                                  f"{type(again).__name__} replayed")
            if recorded.dtype != again.dtype or recorded.shape != again.shape:
                raise RunnerError(f"float32 parity: {where}: {recorded.dtype} {tuple(recorded.shape)} recorded, "
                                  f"{again.dtype} {tuple(again.shape)} replayed")
            a, b = self.bits(recorded), self.bits(again)
            if not torch.equal(a, b):
                raise RunnerError(f"float32 parity: {where}: {int((a != b).sum())} of {a.numel()} bytes differ")
            return
        if isinstance(recorded, (tuple, list, dict)) or isinstance(again, (tuple, list, dict)):
            if type(recorded) is not type(again) or len(recorded) != len(again):
                raise RunnerError(f"float32 parity: {where}: {type(recorded).__name__} of {len(recorded)} recorded, "
                                  f"{type(again).__name__} of {len(again)} replayed")
            if isinstance(recorded, dict):
                if list(recorded) != list(again):
                    raise RunnerError(
                        f"float32 parity: {where}: keys {list(recorded)} recorded, {list(again)} replayed"
                    )
                for key in recorded:
                    self.compare(recorded[key], again[key], f"{where}[{key!r}]")
            else:
                for i in range(len(recorded)):
                    self.compare(recorded[i], again[i], f"{where}[{i}]")
            return
        if recorded != again:
            raise RunnerError(f"float32 parity: {where}: {recorded!r} recorded, {again!r} replayed")

    def verify(self) -> None:
        """Replay every call recorded since the last `verify`; raise `RunnerError` at the first difference."""
        torch = self.torch
        if self.pending:
            raise RunnerError("float32 parity: a hooked module started a call that never returned")
        records, self.records = self.records, []
        for module, args, kwargs, output in records:
            name = type(module).__name__
            self.require_float32(args, f"{name} args")
            self.require_float32(kwargs, f"{name} kwargs")
            clone = self.float32_copy(module)
            with torch.inference_mode():
                again = clone(*args, **kwargs)
            self.compare(output, again, f"{name} output")
            self.calls += 1
            self.verified.add(id(module))
            del clone, again

    def require_coverage(self) -> dict[str, int]:
        """The proof's coverage, if it is a proof: at least one call verified, every hooked module at least once."""
        if self.pending or self.records:
            raise RunnerError("float32 parity: recorded module calls were never verified")
        if self.calls == 0:
            raise RunnerError("float32 parity: no module call was verified")
        unseen = [type(m).__name__ for m in self.modules if id(m) not in self.verified]
        if unseen:
            raise RunnerError(f"float32 parity: {len(unseen)} of {len(self.modules)} hooked modules never ran")
        return self.results()

    def results(self) -> dict[str, int]:
        # every verified call was bit-identical with float32 inputs: anything else raised
        return {
            "calls": self.calls,
            "bit_identical": self.calls,
            "non_float32_inputs": 0,
            "modules_hooked": len(self.modules),
            "modules_verified": len(self.verified),
        }


class Capture:
    """What one `system_one` call evaluated: each backbone forward's input ids, the LM head's slot logits, and the
    runner's readout input (vocabulary log-probabilities) and output (option probabilities)."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.inputs: list[list[list[int]]] = []
        self.logits: list = []
        self.readouts: list[tuple] = []


def rss_anon_gib() -> float | None:
    """This process's anonymous resident memory now (Linux), which excludes the mapped checkpoint."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


class Reference:
    def __init__(self, args: argparse.Namespace, pins: dict) -> None:
        import torch
        import transformers
        from transformers import AutoModel

        self.torch = torch
        torch.manual_seed(0)
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        self.args = args
        self.pins = pins
        self.capture = Capture()
        self.peak_anon_gib = 0.0

        started = time.perf_counter()
        dtype = torch.float32 if args.precision == "f32-load" else torch.bfloat16
        model, loading = AutoModel.from_pretrained(
            str(args.snapshot), trust_remote_code=True, dtype=dtype, output_loading_info=True
        )
        problems = {key: sorted(map(str, value)) for key, value in loading.items() if value}
        if problems:
            raise SystemExit(f"the model did not load cleanly: {problems}")
        package = type(model).__module__.rsplit(".", 1)[0]
        if not package.startswith("transformers_modules."):
            raise SystemExit(f"{type(model).__name__} is not the snapshot's class ({type(model).__module__})")
        self.root = ".".join(package.split(".")[:2])  # transformers_modules.<snapshot name>
        self.modules = {name: sys.modules[f"{package}.{name}"] for name in ("prompt", "runner", "attention", "api")}
        attention = self.modules["attention"]
        language_model = attention.language_model(model)
        if not type(language_model).__module__.startswith(self.root + "."):
            raise SystemExit(
                f"the text model is {type(language_model).__module__}.{type(language_model).__name__}, not the "
                f"snapshot's vendored LFM3 (transformers {transformers.__version__} ships its own)"
            )
        head = model.get_output_embeddings()
        if head.weight is not language_model.embed_tokens.weight:
            raise SystemExit("expected the LM head tied to the input embeddings")
        self.model = model
        self.engine = model.engine  # SystemOne: eval mode, fused projections, the pinned calibration
        self.load_seconds = time.perf_counter() - started
        if self.engine.model.training:
            raise SystemExit("the reference engine is not in eval mode")
        self.code = loaded_code(self.root, pins)
        if not any(m["module"].endswith(".modeling_lfm3") for m in self.code):
            raise SystemExit("the snapshot's vendored LFM3 text model was not imported")

        # Text only: the media encoders must never run.
        def no_media(module, args):
            raise RunnerError(f"{type(module).__name__} ran in a text-only reference run")

        body = attention.text_backbone(model)
        for name in ("vision_tower", "multi_modal_projector", "audio_tower"):
            if getattr(body, name, None) is not None:
                getattr(body, name).register_forward_pre_hook(no_media)

        self.compute = None
        if args.precision == "f32":
            hooked = [language_model.embed_tokens, *language_model.layers, language_model.embedding_norm, head]
            covered = {id(p) for m in hooked for p in m.parameters()}
            needed = [*language_model.parameters(), *head.parameters()]
            if any(id(p) not in covered for p in needed):
                raise SystemExit("float32 compute does not cover every text-model parameter")
            self.compute = Float32Compute(torch, hooked)
        self.parity = None
        if args.check_f32_parity:
            if args.precision != "f32":
                raise SystemExit("--check-f32-parity checks --precision f32")
            self.parity = ParityCheck(torch, hooked)

        capture = self.capture

        def backbone_inputs(module, args, kwargs):
            capture.inputs.append(kwargs["input_ids"].tolist())

        body.register_forward_pre_hook(backbone_inputs, with_kwargs=True)

        def slot_logits(module, args, output):
            capture.logits.append(output.detach().to(torch.float32).clone())

        head.register_forward_hook(slot_logits)

        readouts = self.engine._readouts

        def captured_readouts(qs, logz, tempered):
            result = readouts(qs, logz, tempered)
            capture.readouts.append((list(qs), logz.detach().clone(), [list(map(float, p)) for p in result]))
            return result

        self.engine._readouts = captured_readouts
        self.tokenizer = self.engine.tokenizer
        self.versions = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "numpy": __import__("numpy").__version__,
            "safetensors": __import__("safetensors").__version__,
            "torchvision": __import__("torchvision").__version__,
        }

    def note_memory(self) -> None:
        anon = rss_anon_gib()
        if anon is not None:
            self.peak_anon_gib = max(self.peak_anon_gib, anon)

    def call(self, state: Any, questions: dict, set_size: int | None) -> tuple[dict, float]:
        """`system_one`, then, with `--check-f32-parity`, the proof of every module call it made. A proof failure is
        a `RunnerError`, never a refusal by the reference, so it ends the run before any observation is written."""
        self.capture.clear()
        started = time.perf_counter()
        try:
            result = self.model.system_one(state, questions, set_size=set_size)
        except BaseException:
            if self.parity is not None:
                self.parity.discard()  # a forward that raised reaches no observation
            raise
        finally:
            if self.compute is not None:
                self.compute.restore_all()
        seconds = time.perf_counter() - started
        self.note_memory()
        if self.parity is not None:
            try:
                self.parity.verify()
            except RunnerError:
                raise
            except Exception as error:  # the replay itself failed: no proof
                raise RunnerError(f"float32 parity: replaying a module call failed: {error!r}") from error
        return result, seconds

    # ------------------------------------------------------------------ stages

    def parse(self, text: str) -> tuple[dict, dict, Any, dict, dict]:
        """`request_text` to `system_one`'s arguments: Python's json module, the request's shape, then the
        reference's own question checks (`SystemOneApi.named`: `check_question` and `as_question`, in order)."""
        detail: dict[str, Any] = {}
        try:
            request, notes = parse_request(text)
        except ValueError as error:
            return refusal(error), detail, None, {}, {}
        if notes:
            detail["json"] = notes
        if not isinstance(request, dict):
            return refusal("NotAnObject", "the request is not a JSON object"), detail, None, {}, {}
        if "state" not in request or not isinstance(request.get("questions"), dict):
            return refusal("MissingField", 'system_one(state, questions) needs "state" and a "questions" object'), (
                detail
            ), None, {}, {}
        media = [k for k in MEDIA_FIELDS if request.get(k) not in (None, [])]
        if media:
            return refusal("Media", f"media fields {media}: this runner reads text only"), detail, None, {}, {}
        ignored = sorted(k for k in request if k not in ("state", "questions", *MEDIA_FIELDS))
        if ignored:
            detail["ignored_fields"] = ignored
        raws = request["questions"]
        questions = {}
        for name, raw in raws.items():
            if isinstance(raw, dict) and self.modules["prompt"].is_media(raw.get("instructions")):
                return refusal("Media", "spoken instructions: this runner reads text only", name), detail, None, {}, {}
            try:
                questions[name] = self.engine.ask(raw)
            except Exception as error:  # the reference's own refusal, whatever its type
                return refusal(error, question=name), detail, None, {}, {}
        typed = {n: q.type for n, q in questions.items() if q.type not in OPTION_TYPES or getattr(q, "multiple", False)}
        if typed:
            detail["typed"] = typed
            return {}, detail, request["state"], raws, questions
        parsed = {"questions": [{"id": n, "type": q.type, "keys": self.keys(q)} for n, q in questions.items()]}
        return parsed, detail, request["state"], raws, questions

    @staticmethod
    def keys(question: Any) -> list[str]:
        """Option keys in model order: choice in criteria order, score "0".."n-1", noul true then false."""
        if question.type == "choice":
            return [str(k) for k in question.criteria]
        if question.type == "score":
            return [str(i) for i in range(len(question.criteria))]
        return ["true", "false"]  # prompt.py reads yes before no

    def prepare_one(self, state: Any, name: str, question: Any, index: int) -> tuple[dict, dict, dict]:
        """The reference's own preparation of one lone question: rendered text, tokens, codes, groups, temperature."""
        prompt, engine, tok = self.modules["prompt"], self.engine, self.tokenizer
        prefix, _ = engine._prefix(state, [])
        suffixes, _, _ = engine._suffixes([question])
        text = prefix + suffixes[0]
        ((_, _, ids, state_len, _, ends, _),) = engine._sequences([(state, [question], [])], set_size=1)
        if ids != tok.encode(text, add_special_tokens=False):
            raise RunnerError("the reference's sequence is not its rendered text, tokenized")
        if ends != [[len(ids) - 1]]:
            raise RunnerError("a lone option question is read at its last token")
        groups = prompt.readout_ids(tok, question)
        if question.type == "choice":
            codes = [code for code, _ in prompt.aliases(tok, list(question.criteria))]
        else:
            codes = self.keys(question)
        select, ranges = [], []
        for group in groups:
            ranges.append([len(select), len(select) + len(group)])
            select.extend(group)
        temperature = engine.calibration.temperature(question) if engine.calibration is not None else 1.0
        passed = {"question": name, "prompt": text, "token_ids": ids, "output_index": len(ids) - 1, "select": select}
        answer = {
            "question": name,
            "type": question.type,
            "pass": index,
            "groups": [{"key": k, "code": c, "range": r} for k, c, r in zip(self.keys(question), codes, ranges)],
            # the reference reads calibration.json's number with Python's float(); this is its repr
            "temperature": repr(temperature) if engine.calibration is not None else None,
            "temperature_f32": f32_hex(temperature),
        }
        detail = {
            "question": name,
            "prompt_sha256": sha256_bytes(text.encode("utf-8")),
            "state_tokens": state_len,
            "select_tokens": tok.convert_ids_to_tokens(select),
            "temperature_f64": f64(temperature),
        }
        return passed, answer, detail

    def evaluate_one(
        self, state: Any, name: str, raw: Any, passed: dict, answer: dict
    ) -> tuple[list, list, dict, dict]:
        """One lone question through the reference's public call, `system_one(state, {name: q}, set_size=1)`."""
        result, _ = self.call(state, {name: raw}, 1)
        cap = self.capture
        if cap.inputs != [[passed["token_ids"]]]:
            raise RunnerError("the evaluated sequence differs from the prepared one")
        if len(cap.logits) != 1 or len(cap.readouts) != 1 or cap.logits[0].shape[0] != 1:
            raise RunnerError("expected one answer slot, read once")
        select = passed["select"]
        logits = cap.logits[0][0][select].tolist()
        logprobs = cap.readouts[0][1][0][select].tolist()
        ranges = [g["range"] for g in answer["groups"]]
        detail = {
            "question": name,
            "raw_logprobs": [f32_hex(v) for v in logprobs],
            "logprob_scores": [f32_hex(max(logprobs[a:b])) for a, b in ranges],
            "logsumexp": f32_hex(float(self.torch.logsumexp(cap.logits[0][0], dim=-1))),
            "distribution": [f64(p) for p in cap.readouts[0][2][0]],
        }
        raw_scores = [f32_hex(v) for v in logits]
        scores = [f32_hex(max(logits[a:b])) for a, b in ranges]
        return raw_scores, scores, detail, result

    # ----------------------------------------------------------------- per case

    def run_case(self, case: dict) -> dict[str, Any]:
        text = case.get("request_text")
        observation = self.run_case_header(case)
        hf, detail = observation["hf"], observation["detail"]

        parsed, detail["parse"], state, raws, questions = self.parse(text)
        if "refusal" in parsed:
            hf["parse"] = parsed
            return observation
        if not parsed:  # typed answers: outside the schema's stages, recorded as the reference gives them
            try:
                result, _ = self.call(state, raws, 1)
            except RunnerError:
                raise
            except Exception as error:
                detail["typed"] = refusal(error)
                return observation
            detail["typed"] = {"answers": f64_tree(result["answers"]), "usage": result["usage"]}
            return observation
        hf["parse"] = parsed

        # The reference renders and labels every question before any forward pass.
        passes, answers, prepared = [], [], []
        for index, (name, question) in enumerate(questions.items()):
            try:
                passed, answer, extra = self.prepare_one(state, name, question, index)
            except RunnerError:
                raise
            except Exception as error:
                hf["prepare"] = refusal(error, question=name)
                return observation
            passes.append(passed)
            answers.append(answer)
            prepared.append(extra)
        hf["prepare"] = {"passes": passes, "answers": answers}
        detail["prepare"] = prepared

        raw_scores, scores, evaluated, answered, input_tokens = [], [], [], {}, 0
        for passed, answer in zip(passes, answers):
            name = passed["question"]
            try:
                raw, best, extra, result = self.evaluate_one(state, name, raws[name], passed, answer)
            except RunnerError:
                raise
            except Exception as error:
                hf["evaluate"] = refusal(error, question=name)
                return observation
            raw_scores.append(raw)
            scores.append(best)
            evaluated.append(extra)
            answered[name] = f64_tree(result["answers"][name])
            input_tokens += result["usage"]["input_tokens"]
        hf["evaluate"] = {"source": "model", "raw_scores": raw_scores, "scores": scores}
        detail["evaluate"] = evaluated
        hf["finalize"] = {"answers": answered, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
        return observation

    def run_packed(self, case: dict) -> dict[str, Any]:
        """The whole request in one call at the checkpoint's own set size: its questions packed after one state."""
        observation = self.run_case_header(case, packed=True)
        parsed, detail_parse, state, raws, questions = self.parse(case["request_text"])
        observation["detail"]["parse"] = detail_parse
        engine, prompt, tok = self.engine, self.modules["prompt"], self.tokenizer
        if "refusal" in parsed or not parsed or not questions or len(questions) > engine.set_size:
            raise RunnerError(f"{case['id']}: packed runs take 1 to {engine.set_size} noul, choice or score questions")
        names, qs = list(questions), list(questions.values())
        ((_, _, ids, state_len, question_of_token, ends, _),) = engine._sequences([(state, qs, [])])
        result, _ = self.call(state, raws, None)
        cap = self.capture
        if cap.inputs != [[ids]] or len(cap.logits) != 1 or cap.logits[0].shape[0] != len(qs):
            raise RunnerError("the packed forward differs from the prepared sequence")
        evaluated = []
        for j, (name, question) in enumerate(zip(names, qs)):
            groups = prompt.readout_ids(tok, question)
            select = [t for g in groups for t in g]
            logits = cap.logits[0][j][select].tolist()
            logprobs = cap.readouts[0][1][j][select].tolist()
            bounds, start = [], 0
            for g in groups:
                bounds.append((start, start + len(g)))
                start += len(g)
            evaluated.append(
                {
                    "question": name,
                    "output_index": ends[j][0],
                    "select": select,
                    "raw_scores": [f32_hex(v) for v in logits],
                    "scores": [f32_hex(max(logits[a:b])) for a, b in bounds],
                    "raw_logprobs": [f32_hex(v) for v in logprobs],
                    "distribution": [f64(p) for p in cap.readouts[0][2][j]],
                }
            )
        observation["hf"]["finalize"] = {
            "answers": {n: f64_tree(result["answers"][n]) for n in names},
            "usage": {"input_tokens": result["usage"]["input_tokens"], "output_tokens": 0},
        }
        observation["detail"]["packed"] = {
            "set_size": engine.set_size,
            "token_ids": ids,
            "state_tokens": state_len,
            "question_of_token": question_of_token,
            "evaluate": evaluated,
        }
        return observation

    def run_case_header(self, case: dict, packed: bool = False) -> dict[str, Any]:
        text = case.get("request_text")
        if not isinstance(text, str):
            raise SystemExit(f"{case.get('id')}: no request_text")
        settings = SETTINGS_ID.format(precision=self.args.precision, threads=self.args.threads)
        return {
            "schema_version": SCHEMA_VERSION,
            "case": case["id"],
            "source": "hf-reference-packed" if packed else "hf-reference",
            "artifact": self.pins["artifact"],
            "settings": settings + ("-packed" if packed else ""),
            "request_text_sha256": sha256_bytes(text.encode("utf-8")),
            "hf": {},
            "detail": {},
        }


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def case_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    return files


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"machine": platform.machine(), "system": platform.system(), "logical_cpus": os.cpu_count()}
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()
        info["cpu"] = next(line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name"))
        flags = next(line.split(":", 1)[1].split() for line in cpuinfo if line.startswith("flags"))
        info["simd"] = sorted(f for f in flags if f.startswith(("avx", "amx", "fma", "f16c")))
        mem = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal"))
        info["memory_gib"] = round(int(mem.split()[1]) / 2**20, 1)
    except (OSError, StopIteration):
        pass
    return info


def status_of(observation: dict) -> str:
    for stage in ("parse", "prepare", "evaluate", "finalize"):
        if "refusal" in observation["hf"].get(stage, {}):
            return f"refused at {stage}"
    typed = observation["detail"].get("typed")
    if typed is not None:
        return "typed: refused" if "refusal" in typed else "typed answer"
    return "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", type=Path, nargs="+", help="case files, or directories of them")
    parser.add_argument("--cache", type=Path, default=Path(os.environ.get("XYBRID_DECISION_HF_CACHE", DEFAULT_CACHE)))
    parser.add_argument("--snapshot", type=Path, help="the pinned snapshot (default: <cache>/snapshots/<rev>/d1-2B)")
    parser.add_argument("--out", type=Path, help="observation directory (default: the fixtures' observations/)")
    parser.add_argument("--run-log", type=Path, help="write the run's environment, timings and memory here (JSON)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--precision", choices=("f32", "f32-load", "bf16"), default="f32")
    parser.add_argument("--set-size", choices=("1", "default"), default="1")
    parser.add_argument(
        "--check-f32-parity",
        action="store_true",
        help="replay every module call of every forward as its float32 copy; any difference, a non-float32 input or "
        "a module never verified fails the run (exit 1) before anything is written",
    )
    args = parser.parse_args(argv)

    packed = args.set_size == "default"
    if args.out is None:
        args.out = FIXTURES / ("packed" if packed else "observations")
    # What can change an observation's bytes, as it stands when the run starts (the run log records it).
    inputs = {
        "pins_sha256": sha256_file(PINS),
        "requirements_sha256": sha256_file(REQUIREMENTS),
        "recipe": recipe(RECIPE),
    }
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    if args.snapshot is None:
        args.snapshot = args.cache / "snapshots" / pins["revision"] / "d1-2B"
    os.environ.setdefault("HF_MODULES_CACHE", str(args.cache / "modules"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    inputs["snapshot_files"] = verify_snapshot(args.snapshot, pins)
    files = case_files(args.cases)
    if not files:
        raise SystemExit("no case files")

    # Observations stay in memory until every case ran and the parity proof (if asked for) holds: a failed run
    # writes nothing, and exits nonzero.
    written: list[tuple[str, str]] = []
    requests: dict[str, str] = {}
    timings = []
    try:
        reference = Reference(args, pins)
        reference.note_memory()
        for path in files:
            case = json.loads(path.read_text(encoding="utf-8"))
            started = time.perf_counter()
            observation = reference.run_packed(case) if packed else reference.run_case(case)
            seconds = time.perf_counter() - started
            written.append((case["id"], dumps(observation)))
            requests[case["id"]] = observation["request_text_sha256"]
            status = status_of(observation)
            tokens = observation["hf"].get("finalize", {}).get("usage", {}).get("input_tokens")
            timings.append({"case": case["id"], "seconds": round(seconds, 2), "input_tokens": tokens, "status": status})
            print(f"{case['id']:34s} {status:20s} tokens={tokens!s:>6s} {seconds:7.1f}s", flush=True)
        parity = reference.parity.require_coverage() if reference.parity is not None else None
        if recipe(RECIPE) != inputs["recipe"] or sha256_file(PINS) != inputs["pins_sha256"]:
            raise RunnerError("the recipe or the pins changed during the run")
    except RunnerError as error:
        print(f"run_reference: {error}; no observation written", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for case_id, text in written:
        (args.out / f"{case_id}.json").write_text(text, encoding="utf-8")
        outputs[case_id] = sha256_bytes(text.encode("utf-8"))
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(
        f"model load {reference.load_seconds:.1f}s, peak RSS {peak_gib:.2f} GiB "
        f"(anonymous {reference.peak_anon_gib:.2f} GiB at most between calls)",
        flush=True,
    )
    if parity is not None:
        print(f"float32 parity: {parity}", flush=True)
    if args.run_log:
        settings = SETTINGS_ID.format(precision=args.precision, threads=args.threads) + ("-packed" if packed else "")
        log = {
            "reference": {"artifact": pins["artifact"], "repo": pins["repo"], "revision": pins["revision"]},
            "inputs": {**inputs, "requests": requests},
            "outputs": outputs,
            "code_imported": reference.code,
            "environment": {
                "requirements_sha256": inputs["requirements_sha256"],
                "versions": reference.versions,
                "torch_parallel": reference.torch.__config__.parallel_info().splitlines()[:6],
                "cpu_capability": reference.torch.backends.cpu.get_cpu_capability(),
            },
            "numerics": {
                "settings": settings,
                "precision": args.precision,
                "threads": args.threads,
                "set_size": "default (the checkpoint's d1_set_size)" if packed else 1,
                "deterministic_algorithms": True,
                "mode": "eval, inference_mode",
                "float32_parity_gate": parity is not None,
            },
            "hardware": hardware(),
            "model_load_seconds": round(reference.load_seconds, 2),
            "peak_rss_gib": round(peak_gib, 2),
            "peak_anon_gib_between_calls": round(reference.peak_anon_gib, 2),
            "cases": timings,
        }
        if parity is not None:
            log["float32_parity"] = parity
        args.run_log.parent.mkdir(parents=True, exist_ok=True)
        args.run_log.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
