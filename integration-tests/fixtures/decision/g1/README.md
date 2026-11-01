# G1 fixtures: d1-2b-f16 through llama.cpp against LiquidAI's d1 forward

The evidence of gate G1 (profile `lfm3-d1/text/v1`), and G3's Q4 drift report.
The comparator and the held-out recipe are in
[`tools/conformance/decision_g1`](../../../../tools/conformance/decision_g1),
whose README explains the metric, the rules and the commands.

| Path | Contents | Made by |
|---|---|---|
| `allowlist.json` | The shared domain of the representative corpus: 26 included cases (32 question observations, 26 distinct token sequences) and 20 exclusions, each with the refusal or the exact differing fields it expects | hand, checked by `compare.py` |
| `oracle/<artifact>/<settings>/` | The upstream oracle's observations of the representative corpus (P3b's `hf/cases/` request bytes): `<id>.<sha256 of the request>.json` per case and `collection.json` (collector, build identity, settings as they ran, devices, artifact, per-case status) | `tools/conformance/decision/oracle.py --collect` |
| `bounds.json` | The frozen bounds per settings id, the rule that made them, the observed maxima and headroom, the computation identities they hold for, the comparator and held-out recipe they were frozen with (`frozen_with`), and any later revalidation | `compare.py freeze` (once), `compare.py revalidate` |
| `freeze-record.md` | What the freeze is bound to: the recipe's lineage and the generating recipe, the frozen comparator, the evidence that the recipe amendments are content-neutral, the disclosure, the inputs and frozen values, the post-freeze amendment policy, and what generation must satisfy | by hand, in the freeze commit |
| `heldout/cases/` | The held-out requests, generated after the freeze | `make_heldout.py` |
| `heldout/generation.json`, `heldout/allowlist.json` | How they were generated (the frozen recipe and seed; the measuring tool's repository path, sha256 and stamped identity; the vocabulary and its sha256; the measurement count; every request digest and measured length); every case included | `make_heldout.py` |
| `heldout/hf/` | The HF reference's observations of the held-out cases and its run log (`run.json`: pins, lock, recipe, requests, outputs, float32 parity proof) | `tools/conformance/decision_hf/run.sh --check-f32-parity --run-log` |
| `heldout/oracle/<artifact>/<settings>/` | The oracle's observations of the held-out cases | `oracle.py --collect` |
| `report.json` | G1's verdict on every corpus and setting, per question | `compare.py gate` |
| `q4-drift.json` | G3's report: `d1-2b-q4km` against `d1-2b-f16` under each setting, and Flash Attention on against off | `compare.py drift` |

`compare.py integrity` runs every check that compares no number; `gate`,
`drift` and `revalidate` run it first and refuse unless it passes.
`compare.py check` recomputes both reports from the files here and fails on
any difference. Every file in this folder is checked out byte for byte
(`.gitattributes`), because the reports and bounds pin their digests.

## Order

1. The held-out recipe (`make_heldout.py`, its seed and slot table) was
   committed before any representative F16 observation existed (`a530d1b`).
   Three reviewed amendments followed before the freeze, none changing a slot,
   target, topic or the seed: the padding search (`c7dbf77`), the module
   docstring (`f59a21e`) and the generation record's tool fields (`596a3e2`).
2. The representative corpus was observed under both settings, and
   `bounds.json` frozen from it (`412117c`), stamping the comparator and the
   recipe it binds.
3. Only then were the held-out requests generated, once (`4b22769`), and
   observed by both references (`3ac28a9`). `compare.py integrity` passed
   before any held-out number was read; then the set was evaluated against the
   frozen bounds. A held-out failure is reported and stays a blocking open gate; the
   bounds are not widened (`freeze-record.md`, condition 5).

## Results

`report.json` has the results per question, and `q4-drift.json` per
comparison. Both match a fresh run (`compare.py check`).

**G1 passes on both corpora under the frozen bounds** (`bounds.json`,
`freeze-record.md`).

| Corpus | Setting | Questions (cases) | Max AGE (logits) | At | Frozen bound | Bound / max | Max \|raw Δ\| | Max \|Δp\| | Changed answers |
|---|---|---|---|---|---|---|---|---|---|
| representative | `cpu-t4-fa-off` | 32 (26) | 0.00340 | `hf-choice-letter-keys/planet` (61 tokens) | 0.011 | 3.23 | 0.0038 | 0.00037 | 0 |
| representative | `cpu-t4-fa-on` | 32 (26) | 0.00843 | `hf-state-special-token-key/injection` (58 tokens) | 0.020 (cap) | 2.37 | 0.0069 | 0.0014 | 0 |
| held-out | `cpu-t4-fa-off` | 44 (40) | 0.00344 | `g1h-09-choice-lower-letter-keys` (87 tokens) | 0.011 | 3.20 | 0.0029 | 0.00041 | 0 |
| held-out | `cpu-t4-fa-on` | 44 (40) | 0.00910 | `g1h-30-score-3` (60 tokens) | 0.020 (cap) | 2.20 | 0.0066 | 0.0013 | 0 |

The two corpora:
- **Representative:** P3b's 46 HF cases. The shared domain holds 26 cases,
  32 question observations and 26 distinct token sequences. The oracle's
  official collections were made with P3a's collector `7072b46b`, and the HF
  observations are P3b's `6ed89e09`.
- **Held-out:** `make_heldout.py`'s 40 cases and 44 questions, generated once
  after the freeze, every one in the domain. They were observed by the same
  oracle build (`01ef0d0d…`) and the same HF environment (identity
  `a43c3ae8…`) as the representative corpus.

On both corpora:
- There is no integrity failure, and every computation identity is the
  frozen one.
- The comparator reproduces the oracle's own answers.
- HF's own answers and upstream's arithmetic on HF's scores agree on every
  question.
- No question changes its answer.

**Flash attention.** On both corpora, the largest errors with flash
attention on are on short prompts: 58 and 52 tokens on the representative
corpus, 60 and 58 on the held-out set. That fits the CPU kernel accumulating
V in f16 below 64 query rows. The held-out slots at the boundary show no step
at 64:

| Prompt tokens | AGE, flash attention on | AGE, flash attention off |
|---|---|---|
| 63 | 0.00136 | 0.00005 |
| 64 | 0.00100 | 0.00071 |
| 65 | 0.00089 | 0.00056 |

The long held-out prompts stay small with flash attention on: 0.00099 at 600
tokens, 0.0023 at 1,500 and 0.0025 at 4,064.

**G3 drift** (a report, not a gate):

| Comparison | Corpus | Max AGE | At | Max \|Δp\| | Changed answers |
|---|---|---|---|---|---|
| `d1-2b-q4km` vs `d1-2b-f16`, `cpu-t4-fa-off` | representative | 0.851 | `hf-noul-criteria/damaged` | 0.094 | 0 |
| `d1-2b-q4km` vs `d1-2b-f16`, `cpu-t4-fa-on` | representative | 1.000 | `hf-noul-criteria/damaged` | 0.121 | 0 |
| `d1-2b-q4km` vs `d1-2b-f16`, `cpu-t4-fa-off` | held-out | 0.871 | `g1h-20-noul-criteria-b` | 0.099 | 4 |
| `d1-2b-q4km` vs `d1-2b-f16`, `cpu-t4-fa-on` | held-out | 0.991 | `g1h-20-noul-criteria-b` | 0.106 | 4 |
| `d1-2b-f16`, fa-on vs fa-off | representative | 0.0073 | `hf-state-special-token-key/injection` | 0.0013 | 0 |
| `d1-2b-f16`, fa-on vs fa-off | held-out | 0.0064 | `g1h-30-score-3` | 0.00088 | 0 |
| `d1-2b-q4km`, fa-on vs fa-off | representative | 0.282 | `hf-state-unicode/language` | 0.038 | 0 |
| `d1-2b-q4km`, fa-on vs fa-off | held-out | 0.217 | `g1h-18-choice-4064-tokens` | 0.030 | 0 |

Quantization's gap errors (0.4 to 1.0 logits) are about a hundred times
llama.cpp's F16 error against the reference (at most 0.0091). On the held-out
set, Q4 changes the same four answers under both settings:
- two choices where F16's own margin is small:
  - `g1h-10-choice-27`: scheldt becomes seine; F16 margin 0.24 logits, Q4
    AGE 0.42 (fa-off) and 0.41 (fa-on);
  - `g1h-12-choice-255`: item_168 becomes item_059; F16 margin 0.15 logits,
    Q4 AGE 0.41 and 0.40;
- two score medians:
  - `g1h-30-score-3`: 2 becomes 1, with max |Δp| 0.099 (fa-off) and 0.106
    (fa-on);
  - `g1h-33-score-10`: 4 becomes 5, with max |Δp| 0.023 and 0.021.
