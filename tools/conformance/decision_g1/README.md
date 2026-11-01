# G1: the d1-2b-f16 GGUF through llama.cpp against LiquidAI's d1 forward

Gate G1 of the decision-model work (LFM3 d1, profile `lfm3-d1/text/v1`) checks
that llama.cpp evaluates d1 the way LiquidAI's own code does: the same prompts,
read out by two independent implementations, must give the same offered
distribution up to a frozen numerical bound. It compares

- the **upstream oracle** ([`../decision`](../decision)): llama.cpp's own
  decision code on the `d1-2b-f16` GGUF, CPU only, under settings
  `cpu-t4-fa-off` and `cpu-t4-fa-on` (the product default), and
- the **HF reference** ([`../decision_hf`](../decision_hf)): the pinned
  `LiquidAI/d1-2B` code and weights, a float32 forward, one question per
  sequence (`hf-cpu-f32-t1`).

G1 says nothing about prompts or answer arithmetic (G0, exact against
upstream), about xybrid's own wiring (G2), or about the quantized artifacts
(G3); this folder also writes G3's Q4 drift report, which is a report, not a
gate.

| File | Role |
|---|---|
| `compare.py` | The gate: joins, identity checks, the metric, the freeze, the drift report (stdlib) |
| `make_heldout.py` | The held-out recipe: 40 fixed slots, a literal seed, deterministic padding (stdlib) |
| `tests/` | Unit and mutation tests of both |
| `clean.sh` | Lists or removes `out/` (reports) and bytecode |

The data lives in
[`integration-tests/fixtures/decision/g1/`](../../../integration-tests/fixtures/decision/g1):
the shared-domain allowlist, the frozen bounds, the oracle observations, the
held-out cases with their HF and oracle observations, and the committed
reports.

## What is compared, and how

**Nothing numeric before identity.** Cases, HF observations and oracle
observations are joined by case id and the sha256 recomputed from the exact
request bytes. Each side's run identity is checked and fingerprinted: the
oracle's `collection.json` (official collection, pinned clean llama.cpp
commit, Release CPU-only build, the settings as they ran, CPU devices, the
artifact's bytes) and the HF side's provenance (`decision_hf/provenance.py
--check`, or a held-out run log with its float32 parity proof, pins, lock,
recipe, request and output digests). Inside each observation, questions,
answers, passes, readouts and final answers must form complete, ordered
bijections, and usage must equal the evaluated tokens. For every question of
the shared domain the parsed question, prompt bytes, token ids, output index,
selected tokens and label groups must be identical on both sides, and each
side's temperature must be its own reference's (`0x3fab2fdc` for llama.cpp,
calibration.json's float64 for HF). Envelopes with duplicate keys are refused.

**The domain** is `g1/allowlist.json`: explicit inclusions, and exclusions that
name what they expect (the refusal's stage, exception and status, or the exact
fields that differ); a stale entry fails. It replaces the blanket `hf-diff-`
prefix rule.

**The metric** is the anchored gap error, in logits:

    AGE = max_g |(s_g^gguf - s_w^gguf) - (s_g^hf - s_w^hf)|

`s_g` is a group's maximum selected raw logit (upstream's reduction) and `w` the
HF winner. The gate is `AGE <= B_s` for every question, every group counting.
Raw differences, probability differences and margins are reported. Discrete
answers use upstream's arithmetic on both sides; a changed answer must pass a
necessary-condition certificate (a choice flip's HF margin within the bound,
every separating score prefix inside its one-sided band) whenever its AGE is
within the bound. HF's own answers against upstream's arithmetic are reported
apart.

**The bounds** (`g1/bounds.json`) were frozen once from the representative set,
before any held-out request existed, by the clipped-headroom rule
`B_s = min(0.020, max(0.005, ceil(3 * M_s, 0.001)))`, `M_s` the set's maximum
AGE, refused unless `M_s <= 0.010` and every changed answer is consistent. They
bind the computation identities they were frozen on: observations from another
build, artifact, platform or reference environment fail the gate until they are
revalidated, never refitted. Changing the cap or the rule is a new experiment.

## Running it

```bash
# Representative oracle observations (P3a's collection path), per artifact and settings:
python3 tools/conformance/decision/oracle.py --collect integration-tests/fixtures/decision/hf/cases \
    --artifact d1-2b-f16 --settings cpu-t4-fa-off \
    --out integration-tests/fixtures/decision/g1/oracle/d1-2b-f16/cpu-t4-fa-off
python3 tools/conformance/decision_g1/compare.py freeze   # once, before any held-out request exists
python3 tools/conformance/decision_g1/compare.py gate     # exit 1 on any failure; out/report.json
python3 tools/conformance/decision_g1/compare.py drift    # G3 report: Q4 vs F16, FA on vs off
python3 tools/conformance/decision_g1/compare.py check    # the committed reports equal a fresh run
python3 -m unittest discover -s tools/conformance/decision_g1/tests
```

The held-out set is generated once, after the freeze, with the oracle's
vocabulary-only preparation measuring prompt lengths:

```bash
python3 tools/conformance/decision_g1/make_heldout.py \
    --oracle tools/conformance/decision/build/bin/decision-oracle \
    --vocab integration-tests/fixtures/models/d1-2b-q4km/d1-2B-Q4_K_M.gguf
tools/conformance/decision_hf/run.sh --check-f32-parity \
    --run-log integration-tests/fixtures/decision/g1/heldout/hf/run.json \
    --out integration-tests/fixtures/decision/g1/heldout/hf integration-tests/fixtures/decision/g1/heldout/cases
# then oracle.py --collect on g1/heldout/cases for each artifact and settings, into g1/heldout/oracle/...
```

Every heavy step runs alone on an 8 GB machine: the HF forward peaks near
7 GB, the F16 oracle near 5 GB.

## When llama.cpp moves

The oracle's identity records the llama.cpp commit; `compare.py` refuses
observations whose commit is not the tree's `vendor/llama-cpp` gitlink, with a
"regenerate" message, and the gate refuses any computation identity the bounds
were not frozen or revalidated on. A llama.cpp update therefore fails G1 loudly
until:

1. the oracle observations are collected again (`oracle.py --collect`, both
   settings, representative and held-out; the HF observations do not depend on
   llama.cpp and stay);
2. `compare.py revalidate` passes: every corpus is evaluated under the
   **same** frozen bounds and, only if all of it passes, the new identities are
   appended to `bounds.json` as a revalidation. The bounds are never refitted;
   a failure is investigated like any G1 failure.

For the update automation (`llamacpp-update.yml`,
`docs/development/llamacpp-updates.md`): the decision conformance check belongs
with the checks release preparation waits on, so an update PR stays blocked
until a maintainer regenerates the observations (about an hour on a 4-vCPU,
16 GB runner, through the manual regeneration workflow) and commits them with
the revalidation. As with the choice-conformance provenance, never edit digests
or identities by hand to make the check green.
