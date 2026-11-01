# HF reference observations (LFM3 d1, text)

What LiquidAI's own d1 code (`LiquidAI/d1-2B` at
`10e7671d44e4b37700020b1eed3f594c5f2af0b4`, artifact `d1-2b-hf`) does with
decision requests, one question per sequence: the independent reference for
G1. [`tools/conformance/decision_hf`](../../../../tools/conformance/decision_hf)
wrote everything here; its README says how, and how to rebuild it.

| Path | Contents |
|---|---|
| `cases/<id>.json` | Requests in the decision case schema, version 1 (`../schema.json`), each naming the G1 artifact `d1-2b-f16` (contract C10). `upstream` is empty: the upstream oracle fills it. The `xybrid` expectations are proposals. `hf-*` were written inside the domain both references share, `hf-diff-*` to demonstrate the differences in `DIFFERENCES.md`; which cases bound G1 is P11's allowlist (contract C16), not these names (it takes `hf-diff-packing` and `hf-diff-duplicate-keys` too) |
| `observations/<id>.json` | The reference's observation of each case at `hf-cpu-f32-t1`. `hf` has the stages and shapes of the schema's `upstream` block (`raw_scores` are raw logits); `detail` has the reference's log-probabilities, distributions and float64 temperature |
| `packed/<id>.json` | The two multi-question cases at the checkpoint's own set size (16), the reference's default serving mode, for `DIFFERENCES.md` 1.1 |
| `synthetic/median.json` | Synthetic scores where the two references' score medians disagree (`DIFFERENCES.md` 1.3, 1.5) |
| `crosscheck/<artifact>.<settings>.json` | The stock `llama-server` (v0.8.0) answering the same requests, compared with the observations: a black-box check, not the upstream oracle. `d1-2b-f16` is the G1 counterpart; the BF16 and Q4_K_M files are reports. Each records the script and the observations it compared |
| `provenance.json` | What every file here was made from, as its run recorded it (pins, snapshot files, lock, recipe sources, requests, each run's complete numerics, and the package versions, CPU capability and hardware every run shared), the parity proofs, timings and memory, verification runs (threads, bf16, packing), and the sha256 of every other file here; `provenance.py --check` verifies it all against the current inputs, the settings definitions and the lock |

The reference matches upstream only where both read the same prompt, so
compare numbers there alone: on P11's allowlist (C16). The 24 pilot cases the
cross-check summaries count are a historical selection by name, not that list.
The differences, with their cases, are in
[`DIFFERENCES.md`](../../../../tools/conformance/decision_hf/DIFFERENCES.md).
