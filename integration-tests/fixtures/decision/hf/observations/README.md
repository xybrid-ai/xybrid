# Observed CPU references

These are public-fixture observations from the implementation work, not
numerical promotion thresholds or device/package qualification. Data files are
bound by [`../../integrity.json`](../../integrity.json), and `check.py` rejects
an edited, missing or extra observation. No model weights are included.

| Collection | Actual coverage | Scope |
|---|---|---|
| `hf-f32-linux-x86_64/` | All 63 cases attempted: 58 complete, five explicit refusals; 72 successful question passes including partial multi-question cases. | Pinned independent HF custom code/checkpoint, float32 CPU, four threads, `set_size=1`, one question and batch row per call. The collector exits nonzero and marks the receipt `incomplete` for the five refusals. |
| `q4km-linux-x86_64/` | 63 selected/observed cases, 78 whole-prompt passes; all stock `llama-server` cross-checks agree. | Fresh official upstream Q4_K_M reference on the same CPU/compiler as the Rust G2 run. |
| `f16-smoke-linux-x86_64/` | **One** selected/observed case (`card-text-example`), three questions; stock server agrees. | Partial independent F16 smoke only. The receipt explicitly marks the other 62 input cases `not_selected`. |
| `xybrid-linux-x86_64/g2.json` | 63 model cases, 78 whole-prompt passes; every selected f32 logit bit-equal to the fresh Q4 reference. | Matched-engine CPU G2 only, including a successful actual 4096-token pass. |
| `xybrid-linux-x86_64/native-stage.json` | Six exact upstream engine archives and two headers, each byte-counted and hashed. | These eight files match the prefix linked by the tested Rust binary. The current wrapper/Jinja bridge was compiled locally. |
| `xybrid-linux-x86_64/integration-tests.json` and `.log` | Seven tests passed, zero failed/skipped: wire, metadata, G0, lifecycle, G2, SDK and independent finalization. | Records the executed binary hash, command, environment and bound log. Native cancellation recovery and Rust sync/async/concurrent calls executed with the real model. |
| `xybrid-linux-x86_64/cli-integration-tests.json` and `.log` | Three required CLI tests passed, zero failed/skipped. | Actual Q4 file/stdin/output-file results match the Rust SDK in order, f64 values and usage; transport/flag refusals occur before model loading. |

The native receipts identify the model SHA-256, pinned llama.cpp source,
compiler/flags, CPU/libc, context and offload settings, recipe digests, request
identities and observations. Their absolute build paths describe the collector
machine; they are not prerequisites for consuming these fixtures. Both native
collections used four CPU threads, full non-causal attention, one sequence,
one selected output, 4096-token logical/physical batches and no offload devices.

`hf-q4km.json` compares every attempted HF case with the corresponding fresh
Q4 observation. Five HF refusals are explicit: `score-null-levels`,
`score-structured-levels`, `depth-score-level-at-limit` and `escape-everywhere`
use score levels rejected by the pinned HF question validator;
`instructions-scalars` contains non-text instructions rejected there.

Twelve accepted cases differ in prompt bytes and token IDs. The report names
each one. Differences include Python representation versus ordered JSON for
structured descriptions/instructions, `None` for null yes/no descriptions,
CONTROL escaping and numeric-domain representation. Those cases are excluded
from numerical-error maxima; a larger tolerance cannot reconcile different
inputs.

Four selected answers differ even with matching preparation:

| Case | Frozen split | HF | Q4_K_M |
|---|---|---|---|
| `score-10-levels` | held-out | median 3 | median 2 |
| `choice-26-options` | calibration | `option_005` | `option_006` |
| `choice-36-options` | calibration | `option_005` | `option_001` |
| `choice-120-options` | held-out | `option_005` | `option_001` |

For matching preparations, the **observed** maximum absolute raw-logit
difference is `1.08827543258667` and probability difference is
`0.08273522700751684`. These are neither acceptance limits nor a quantization
attribution: the full HF/F16 compatibility study has not been completed.

`hf-f16-smoke.json` explicitly selects only `card-text-example` from the full
HF receipt and the one-case F16 receipt. All three prompts/token arrays/label
maps and selected answers agree. Observed maxima are raw-logit
`0.0015583038330078125` and probability `0.00002926999361130278` differences.
This small smoke does not qualify the remaining corpus. Both comparison
reports set `promotion_qualified` to `false`.

Durations are collection wall time on a shared CPU worker, including resource
contention and an intentional pause of the HF collector. They must not be used
as latency, memory-floor or thermal qualification evidence.

Full F16/BF16, repeated native runs, Q8_0, calibrated bounds and compatibility
sign-off remain release prerequisites. See the
[qualification status](../../../../../docs/development/decision-conformance.md).
