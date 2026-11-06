# Independent HF reference

`assets.json` pins the d1-2B checkpoint and every Python, configuration,
calibration, tokenizer and template companion at revision
`10e7671d44e4b37700020b1eed3f594c5f2af0b4`. Assets are downloaded and verified
outside this directory. `corpus.json` fixes 42 calibration and 21 held-out
requests before numerical observations; `cases/` contains their exact bytes.

These files are reference **inputs**, not a claim that HF and GGUF agree.
The collector and comparison commands are documented in
[`decision-conformance.md`](../../../../docs/development/decision-conformance.md).
They use one question per call, one batch row, and explicit `set_size=1`.
The collector records both the raw output-head logits and the log-probabilities
that the HF runner actually passes to its finalizer.

The [recorded CPU observations](observations/README.md) cover all 63 attempted
HF cases, the full Q4 upstream reference, a deliberately partial F16 smoke,
and matched-engine Rust G2. HF completes 58 cases and refuses five; 12 accepted
cases differ in preparation, and four choices/medians differ against Q4 even
with matching preparation. All reports remain explicitly unqualified for
numerical promotion.

## Compatibility observations

The initial pinned HF investigation already demonstrates semantic differences
that cannot be accounted for by numerical error tolerances:

| Input | Pinned HF custom implementation | Pinned upstream text/v1 |
|---|---|---|
| `instructions-array` | Python list formatting, including single quotes and `None` | Ordered JSON formatting, double quotes and `null` |
| `state-unicode-escapes` (its structured instructions) | Python dictionary formatting | Ordered JSON formatting |
| `score-null-levels` | `check_question` refuses non-string score descriptions | Null descriptions are accepted and rendered by the named template |

The relevant publisher source is `prompt.py` (`check_question`, `as_question`
and question rendering) at the pinned revision; its byte hash is in
`assets.json`. The Rust implementation follows the approved native text/v1
contract and its exact G0 fixtures. Substituting Python formatting would change
that protocol and must not be used to hide these differences.

The investigation must distinguish shared preparation from mismatched
preparation, then compare F16/BF16 separately from Q4_K_M quantization.
`compare_reference.py` reports every refused/different case and every changed
selected answer. It records observed maxima only for matching preparation and
always reports `promotion_qualified: false`; choosing release bounds requires
a reviewed compatibility assessment and held-out evidence. No placeholder
`provenance.json` or guessed threshold is provided.
