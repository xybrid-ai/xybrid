# Liquid text decisions

The `lfm3-d1/text/v1` profile runs the pinned LiquidAI d1-2B GGUF locally and
returns typed answers without generating text. It accepts text or ordered JSON
state and 1–16 named questions: one choice among 1–255 options, yes/no (`noul`),
or an ordinal score with 2–10 levels.

The [protocol contract](../development/liquid-decision-protocol.md) specifies
exact wire, preparation, answer and compatibility semantics.

This is an implementation profile, not a claim that every packaged SDK, device,
quantization or accelerator has been qualified. See
[decision conformance](../development/decision-conformance.md) for the evidence
gates and outstanding reference/device qualification. Image, video, audio,
browser execution, additional question types and legacy decision models remain
separate profiles. The registry capability for this profile is
`llamacpp-lfm3-d1-text-v1`. No SDK build reports it yet, so the registry does
not serve decision models: load a local artifact, as below.

## Loading a local artifact

Build with `llm-llamacpp` and use a directory containing the pinned GGUF and its
`model_metadata.json`. The committed candidate metadata is
[`integration-tests/fixtures/models/d1-2b-q4km/model_metadata.json`](../../integration-tests/fixtures/models/d1-2b-q4km/model_metadata.json).
The runtime validates the artifact's architecture, named `systemone` template,
vocabulary and calibration against the profile when loading. A model name or a
generic LFM3 architecture declaration alone does not establish compatibility.

A GGUF that loads but is not this model (another architecture or decision
type, no `systemone` template, or decision metadata the profile does not know)
is refused with `SdkError::UnsupportedModelCapability`. A missing file (named
in the message), a file that cannot be loaded and metadata that cannot be read
are `SdkError::LoadError`.

The reference configuration is CPU-only: 4,096 context, logical batch and
physical batch tokens, with four CPU threads. Each question is one complete
model pass. The 4,096-token bound is inclusive; requests are never truncated or
split into smaller prefill batches. These values are fixed by the profile, not
tunable: `DecisionTemplate` refuses any other value. A device that needs a
different thread count needs its own qualified profile.

```rust,no_run
use xybrid_sdk::{decode_decision_request, ModelLoader, ir::Envelope};

fn main() -> Result<(), Box<dyn std::error::Error>> {
let request = decode_decision_request(r#"{
  "state": {"ticket": "The order arrived damaged", "refunds": 0},
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Choose the most appropriate support action.",
      "criteria": {"replace": "Send a replacement", "review": "Ask for more information"}
    },
    "urgent": {"type": "noul", "instructions": "Does the customer need an urgent response?"},
    "severity": {
      "type": "score",
      "instructions": "How severe is this issue?",
      "criteria": ["minor", "moderate", "severe"]
    }
  }
}"#)?;
let model = ModelLoader::from_directory("./d1-model")?.load()?;
let input = Envelope::decision_request(request);
let result = model.run(&input, None)?;
let decisions = result.decisions().ok_or("expected decisions")?;
assert_eq!(decisions.answers().len(), 3);
assert_eq!(result.tokens_out(), Some(0));
Ok(())
}
```

`run_async(&input, None).await` has the same result semantics and executes native
work on a blocking worker. Typed Rust builders are re-exported from `xybrid_sdk`
(`DecisionRequest`, `DecisionState`, `DecisionQuestion`, `DecisionQuestionKind`,
`DecisionOption`, `DecisionLevel`, `DecisionContent`, and `OrderedJson`).

For cancellation, use the existing `run_with_options` and a shared
`CancellationToken`. Cancellation is checked before preparation, between passes
and by native CPU evaluation. The native context serializes concurrent calls.
An aborted or failed request returns an error, with no partial answers; the
context is recovered before reuse. A cancelled token ends a decision run with
`SdkError::Cancelled` whatever the abort policy observes; only decision models
poll the token during a batch run, and every other model keeps its existing
batch contract. The Rust async batch signature still accepts
`GenerationConfig`, not `RunOptions`; an async application needing cancellation
can move `run_with_options` onto `tokio::task::spawn_blocking`.
Resource abort policies are checked before batch execution; during-call
cancellation uses the explicit cancellation token.

## CLI

Put the Liquid JSON in `request.json` and run:

```sh
xybrid run --directory ./d1-model --decision-request request.json
cat request.json | xybrid run --directory ./d1-model --decision-request -
```

`--model`, `--bundle` and `--huggingface` are also accepted load sources. Their
download behavior remains the existing explicit loading behavior; cached local
inference does not need a network connection. `--model-file` is not accepted by
this CLI mode; use a local directory with reviewed execution metadata.

Standard output contains one JSON record, or `--output result.json` writes it
to a file:

```json
{
  "format": "xybrid.decision.v1",
  "model_id": "d1-2b-q4km",
  "decisions": {"answers": [], "usage": {"passes": 0, "evaluated_tokens": 0}},
  "tokens_in": 0,
  "tokens_out": 0
}
```

This illustration shows the container shape; a successful request has one
answer per question. Answers and distributions keep caller order. Raw negative
infinity is serialized as `"-Infinity"`, never `null`. Probabilities retain
`f64` precision. The CLI refuses conflicting chat, choice, generation, tracing,
policy and speculative-cloud flags before loading a model.

## Semantics and limits

The Liquid decoder requires `state` (explicit `null` is valid) and `questions`.
Instructions are required and cannot be null. Duplicate keys, unknown fields,
unsupported media and invalid question shapes fail with content-free,
non-retryable errors. The optional wire `model` field is ignored; the loaded
model stays authoritative. The decoder does not guess between Liquid JSON and
serialized xybrid IR.

Keep structured content as raw JSON text through foreign-language boundaries.
Converting it through a generic dictionary or JavaScript object may reorder keys
or lose integers above 2^53. Ordered JSON preserves object order, integer/float
identity, signed zero, embedded NUL and null values.
The neutral IR preserves `u64`, but Liquid text/v1 refuses unsigned values above
`i64::MAX` because the pinned Jinja implementation cannot render them faithfully.
Signed integers above 2^53 and within `i64` remain exact.

The transport limit is 8 MiB, aggregate request content is bounded to 1 MiB and
65,536 JSON nodes, and each rendered prompt is limited to 8 MiB and 4,096 tokens.
The IR depth limit is 128; this protocol limits state/instructions to 127 and
descriptions to 125. Structural validity does not guarantee that content fits
the rendered or token budgets.

Choice ties select the first maximum. Yes/no returns the canonical yes
probability without confidence. Ordinal scores return a zero-based median
level, with distribution keys `"0"` through `"n-1"`; keep the level descriptions
with the original request. Choice/score confidence is the largest distribution
probability. These are protocol probabilities, not an application's action,
abstention policy or a claim of calibrated quantized-device confidence.

Only local batch execution is supported. Explicit cloud routing, cloud
fallback, speculative execution, streaming, conversation history, tools,
generation options and pipeline input/output refuse before dispatch, including
nested decision envelopes. Telemetry includes protocol/backend, counts, token
usage and durations; state, instructions, IDs, option keys and answers are not
exported.
