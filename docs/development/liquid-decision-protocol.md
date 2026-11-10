# Liquid `lfm3-d1/text/v1` contract

This document specifies the implemented text/JSON protocol. It does not promote
an artifact, SDK package or device configuration to release-qualified status.
The separate [conformance record](decision-conformance.md) tracks G0–G4 evidence;
[SDK usage](../sdk/decisions.md) documents application calls and CLI output.

The protocol implementation lives in
[`execution/decision`](../../crates/xybrid-core/src/execution/decision), above
the protocol-neutral [decision IR](../../crates/xybrid-core/src/ir/decision).
It follows the decision implementation at llama.cpp commit
`8642541b255bfb5dc39cd351a178be09013a55d6`. The registry compatibility token is
`llamacpp-lfm3-d1-text-v1`, separate from generic `llamacpp-lfm3`.

## Admission and execution metadata

Model metadata uses the appended `ExecutionTemplate::Decision(DecisionTemplate)`
variant, with these fields:

| Field | Contract |
|---|---|
| `protocol` | Exactly `lfm3-d1/text/v1`; unknown versions refuse |
| `model_file` | Nonempty GGUF path relative to the model directory |
| `context_length` | Exactly 4096, also the default |
| `batch_size` | Exactly 4096, also the default |
| `ubatch_size` | Exactly 4096, also the default |
| `n_threads` | Exactly 4, also the default, used for both native thread settings |

Unknown execution fields refuse. Admission inspects the actual artifact:
`general.architecture` must be `lfm3`, `lfm3.decision.type` must be `lfm3-d1`, and
the explicitly named `systemone` template must have SHA-256
`ee5fc62461a9bbbcb8389c57f4cd2cd694ccb0dec19521576bd1419049490d8b`.
There is no fallback to a default chat template. A new template needs new
fixture evidence and compatibility review; this whitelist does not authenticate
arbitrary weights.

Temperature entries are read using native metadata string formatting, then
parsed as finite, positive `f32` values. For each question, the lookup order is
`<type>.<bucket>`, then `<type>`, then the implicit temperature 1. The buckets are
`2` for 1–2 options, `3_5` for 3–5, `6_10` for 6–10, and `11` for larger choices.
The type names are `choice`, `noul` and `score`. Absent temperature metadata is
valid. A value rounded to zero by native formatting refuses. Nonempty score-bias
metadata and unknown decision metadata refuse. Unused `temperature.image`
metadata does not invalidate a text artifact and never overrides a text pass.

Two limits of this check follow from how llama.cpp exposes metadata. Its string
map holds scalar values only, so an array-typed `lfm3.decision.*` entry is not
seen and cannot refuse; upstream writes scalars at this pin. And it formats
floats under the process's `LC_NUMERIC`, so a temperature can arrive as
`0,700000`: admission accepts that `%f` form with any decimal separator, as
upstream's `strtof` does under the same locale.

Vocabulary-only admission uses a distinct handle that cannot create an
inference context. Metadata reads are bounded to 1 MiB per string and 8 MiB
aggregate; the text profile admits at most 1,048,576 vocabulary tokens and 256
CONTROL spellings with at most 64 KiB of aggregate spelling bytes. The pinned
artifact has 124 CONTROL spellings. Spellings that cannot be broken after their
first Unicode character, or that already contain the inserted U+200B separator,
refuse. These admission limits also apply to metadata-only fixture variants.

## Typed requests and Liquid transport

The regular batch `run`/`run_async` surface accepts
`EnvelopeKind::DecisionRequest(DecisionRequest)`. There is no decision-specific
run method. The optional Liquid transport decoder is an explicit boundary:
`execution::decision::wire::decode(&str)` (re-exported as the SDK helper
`decode_decision_request`). It never guesses whether input is serialized IR.

The Liquid body has the shape `{ "state": ..., "questions": {...}, "model"?: ... }`.
Its rules are:

| Element | Accepted form and semantics |
|---|---|
| Body | One JSON object, with no trailing data |
| `state` | Required; text, ordered JSON or explicit `null` |
| `questions` | Required, nonempty ordered object; each key is a nonempty question ID |
| Question | Object with exactly `type`, `instructions`, and optional `criteria` |
| `instructions` | Required and non-null; empty text and other JSON values remain distinct |
| `type` | Exactly `choice`, `noul` or `score` |
| Choice `criteria` | Required nonempty ordered object with 1–255 nonempty keys; descriptions may be any admitted JSON value or null |
| Noul `criteria` | Absent, null, or an object containing only optional `true` and `false` descriptions |
| Score `criteria` | Required array of 2–10 descriptions, including null descriptions |
| `model` | Validated JSON and ignored; it cannot change the explicitly loaded artifact or fetch another one |

Duplicate keys refuse at every layer, including structured content. Unknown
fields refuse at the top level, in questions and in noul criteria. Choice
criteria keys are caller option keys, not transport field names. `files`,
`images` and `videos` refuse on presence, including null or empty values. Media
parts in state messages and spoken `input_audio` instructions also refuse;
ordinary ordered JSON message data remains content, not conversation history.

Absent state is an error; null state omits the state block in the prompt; empty
text is present state. Missing and null instructions both refuse. A null option
description remains different from empty text, false, zero, an empty object and
an empty array. The pinned template determines which falsy choice descriptions
fall back to the option key. Noul descriptions always render in true/false order,
regardless of criteria insertion order.

`OrderedJson` retains member order and number identity: `1` differs from `1.0`,
`-0` is integer zero, and `-0.0` is floating negative zero. Signed integer values
above 2^53 remain exact. Although the neutral IR retains `u64`, this profile
refuses unsigned values above `i64::MAX`: the pinned Jinja implementation cannot
render those without wrapping. Non-finite/overflowing numbers refuse. U+0000 and
U+200B remain length-delimited UTF-8 through rendering and tokenization.

## Bounds and preparation

| Limit | Maximum |
|---|---:|
| Raw Liquid transport | 8 MiB per request, checked before parsing |
| Questions/passes | 16 per request |
| Aggregate IR content | 1 MiB, using the IR's content-byte accounting |
| Aggregate structured-content JSON nodes | 65,536 |
| Neutral IR value depth | 128 |
| State/instructions depth | 127 |
| Choice/noul/score description depth | 125 |
| Rendered prompt | 8 MiB per question |
| Rendered input tokens | 4,096 per question, inclusive |

A scalar or empty container has depth 1. Transport wrappers are sliced and
validated separately from raw content so their overhead cannot reject a valid
field. The wire stage accepts neutral IR depth; preparation applies the smaller
field limits before invoking the renderer. Content limits and conservative
escaping/indentation/number-expansion bounds are checked before native rendering.
A structurally valid request can still exceed a render or token budget. Nothing
is silently truncated.

The renderer compiles the admitted template once and creates a fresh native
Jinja context per invocation, using the pinned
`jinja::global_from_json(..., false)` semantics. Compact `OrderedJson` text is
only the bridge input, never a substitute for the upstream `tojson` filter.
Every CONTROL spelling is broken by inserting U+200B after its first Unicode
character, longest spellings first, repeating for overlapping matches.
USER_DEFINED alone does not trigger escaping. Escaping applies recursively to
keys and values; collisions in escaped object keys refuse. Original caller IDs
and option keys are retained for results.

Choice codes use original keys when all keys are single ASCII letters;
otherwise they start A–Z, then AA, AB, and so on. An unusable or already-used
primary token falls back through the pinned pool: A–Z, 00–99, a–z, #0–#199,
AA–ZZ. A code and its space-prefixed spelling contribute only when each is a
single token. Yes aliases are `yes`, `Yes`, `YES`; no aliases are `no`, `No`,
`NO`; score labels are `0`–`9`. Empty groups and token aliases shared between
answer groups refuse. Label tokenization disables added/special tokens; prompt
tokenization disables added tokens and enables special-token recognition.

Every question's schema, prompt and tokens are prepared before the first model
pass. Runtime preparation discards prompt strings and retains bounded token and
answer plans. Each question then uses one complete extended batch: sequence 0,
positions 0 through n−1, and the final token's output flag only. The runtime
calls native processing once per question and copies selected logits after
synchronization. Exactly 4,096 input tokens is valid; no next-token slot is
reserved. Chunked prefill and packing multiple questions together are outside
this contract.

The context is explicitly CPU-only with no offload devices, flash attention
disabled, one sequence and one output. Effective context, physical/logical batch
sizes, thread settings, full attention and no-memory behavior are checked.
Acceleration requires separate qualification, not an implicit default change.

## Answers and numeric behavior

Every selected raw logit is validated before reduction. NaN and positive
infinity refuse even in a losing alias. Negative infinity is allowed provided
the question has at least one finite option. Each group's raw score is its
maximum alias logit; equal aliases preserve the first value. Computation then
subtracts the maximum option score, divides by the chosen `f32` temperature
promoted to `f64`, and performs ordered, stable `f64` exponentiation,
summation and normalization.

The result has one `DecisionAnswer` per question in request order, with ordered
`DecisionScore` distributions. Every text/v1 entry has `DecisionOrigin::Caller`.

| Kind | Canonical value | Confidence |
|---|---|---|
| Choice | First option with maximum probability, using the original key | Maximum distribution probability |
| Noul | Probability of `true`; distribution order is `true`, then `false` | Absent |
| Score | Zero-based median: first cumulative probability reaching 0.5, including equality | Maximum distribution probability |

Score distribution keys are `"0"` through `"n-1"`; descriptions remain with the
original request. The profile never substitutes an expectation for the median.
Raw scores remain `f32`; probabilities/confidence remain `f64` across bindings.
Human-readable JSON uses the string `"-Infinity"` for a negative-infinite raw
score, never null. The neutral IR's probability-sum tolerance is a structural
invariant, not a numerical promotion threshold or a quantized-device calibration
claim.

Usage records the number of passes and the sum of evaluated input tokens. The
existing envelope metadata keys `tokens_in` and `tokens_out` carry the same
input sum and zero generated tokens, exposed through result accessors. A
multi-question request is not represented as one forward pass.

## Compatibility, policy and failures

All existing IR records retain field order and types. Existing enum variants
retain their order: `EnvelopeKind::DecisionRequest` and `Decisions` append at
indices 7 and 8; SDK `OutputType::Decisions` appends at index 5. The decision
execution template also appends. Legacy `ChoiceRequest`/`ChoiceScores` and
FORMS/System One ONNX scoring keep their separate contracts. Foreign bindings
preserve JSON content as text and convert unsupported variants fallibly; they
must not round-trip structured content through generic maps or JavaScript
objects. Generated code and native bridges ship together.

The profile accepts local batch execution only. Explicit cloud routing, cloud
fallback, speculative execution, generation options, streaming, tools,
conversation history, and pipeline inputs/outputs refuse before dispatch,
including nested envelopes. Downloads at model load remain separate explicit
network actions. Cached inference does not require a network connection.

Cancellation is checked before preparation, between prepared/evaluated passes
and through the CPU native abort callback. Mutable contexts serialize callers.
Any failure or cancellation returns an error with no successful partial answers;
the failed context is dropped and recreated before reuse. Warmup uses a valid
decision request. No token streaming or retained request/KV history is exposed.

Protocol refusals carry only stable categories (`DecisionProtocolError::category`),
not parser snippets, question IDs, option keys, instructions or state. They map
through the canonical core/SDK errors and are non-retryable. Telemetry may report
protocol/backend, counts, token usage, durations and refusal categories; it must
not carry selected answers, logits, distributions or request-derived identifiers.

The [143-case corpus](../../integration-tests/fixtures/decision/README.md) assigns
each case to a stage. Its executable runner asserts 98 accepted and 42 refused
parse cases, 89 accepted and 9 refused preparation cases, and 3 load refusals.
Independent finalization uses upstream selected logits, asserting 82 accepted
and 6 refused cases; media-only informational finalization cases stay outside
this text profile. These expected counts are test requirements, not a statement
that unrun qualification gates have passed.
