# Decision conformance fixtures (LFM3 d1, text)

Reference cases for xybrid's decision support, profile `lfm3-d1/text/v1`. Each
case is a SystemOne request, what the pinned upstream llama.cpp routines did
with it stage by stage, and what xybrid must do.

| Path | Contents |
|---|---|
| `artifacts.json` | The pinned artifacts (repo, revision, file, bytes, sha256) and the settings ids. Shared by every decision gate |
| `schema.json` | Case schema, version 1 (JSON Schema 2020-12) |
| `cases/<id>.json` | One case per file |
| `oracle.json` | How the upstream observations were made: llama.cpp commit, build, platform, server flags per settings id, the facts read from each artifact |

The cases are written by the upstream oracle,
[`tools/conformance/decision`](../../../tools/conformance/decision), which runs
llama.cpp's own decision code (`tools/server/server-decision.cpp`, unmodified)
and cross-checks every case against the stock `llama-server`. Its README says
how to rebuild them.

## A case

```jsonc
{
  "schema_version": 1,
  "id": "choice-36-options",            // = the file name
  "profile": "lfm3-d1/text/v1",
  "title": "36 options: code AJ is not one token, the first free pool entry (00) replaces it",
  "covers": ["choice-pool-fallback"],
  "notes": "...",                       // optional: why xybrid differs, what else to know
  "artifact": "d1-2b-q4km",             // template, vocabulary and metadata come from it
  "metadata": { "set": { "lfm3.decision.temperature.choice.2": "f32:0.5" } },  // optional edits
  "settings": "cpu-t4-fa-off",          // null unless upstream.evaluate ran the model
  "request_text": "{\"state\": ...}",   // the body, exact
  "has_media": false,                   // optional: format_answer(has_media = true)
  "upstream": { "load": ..., "parse": ..., "prepare": ..., "evaluate": ..., "finalize": ..., "server": ... },
  "xybrid":   { "load": ..., "parse": ..., "prepare": ..., "evaluate": ..., "finalize": ... }
}
```

`upstream` holds observations; `xybrid` holds acceptance expectations. They are
separate on purpose: where upstream behaves in a way xybrid must not copy (it
silently overwrites colliding escaped keys, keeps the last of duplicate keys,
ignores unknown fields, answers NaN), the observation is still recorded and the
expectation says `refuse`.

### Upstream stages

The oracle runs the request the way `llama-server`'s `POST /v1/systemone`
handler does. Stages run in order; a refusal ends the run and later stages are
absent.

| Stage | Upstream routines | Observation |
|---|---|---|
| `load` | `server_decision_context::init` on the artifact (and its `metadata` edits) | present only as a refusal: `llama-server` would not start |
| `parse` | `common_json::parse`, `parse_questions`, `parse_state`, `prepare_state`, the media check | `questions`: id, type and option keys of each question, in model order |
| `prepare` | `fill_task` per question (`render`, `d1_break`, `d1_labels`, `common_tokenize`), `get_temperature` | `passes` and `answers`, below |
| `evaluate` | one decode per pass as a server slot runs it: whole prompt in one batch, sequence 0, positions 0..n-1, output on the last token; `send_decision` reads the select logits and takes each group's max | `source`, `raw_scores`, `scores` |
| `finalize` | `format_answer` per question | `answers`, `usage` |
| `server` | the stock `llama-server` given `request_text` | `status`, `agrees` (and `body` when it does not agree) |

`prepare`:

- `passes[i]`: `question`, the exact `prompt`, its `token_ids`, the
  `output_index` whose logits are read (the last prompt token), and `select`,
  the label tokens of every group, concatenated.
- `answers[j]`: `question`, `type`, `pass` (the pass it reads), `groups` (one
  per option in model order: the requested `key`, unescaped; the `code`, the
  label text, which for noul and score is the key; `range`, a half-open interval
  of `select`), `temperature` (the metadata string it was read from, as
  `llama_model_meta_val_str` returns it, e.g. `"1.337398"`; null when no key
  applies and T = 1) and `temperature_f32` (the value used).

`evaluate.source` is `model` (the artifact under `settings`) or `synthetic`
(scores given to test the answer arithmetic). `raw_scores[i]` are pass i's
select logits in `select` order; `scores[i]` the max of each group, which is
what `format_answer` receives.

`finalize.answers` is the SystemOne answers object in request order:
choice `{type, choice, probabilities, confidence}`, noul `{type, noul}`, score
`{type, score, legend, probabilities, confidence}`. `usage` is
`{input_tokens, output_tokens: 0}`; `input_tokens` sums every pass's tokens.

A refusal is `{"refusal": {question?, exception, status, type, message}}`:
`status` and `type` are what `llama-server` answers (400
`invalid_request_error` for `std::invalid_argument` and JSON errors, 500
`server_error` for other exceptions, 501 `not_supported_error`, and the slot
checks' own), `exception` names what was thrown (`slot` for a slot check,
`not_supported` for the 501), `question` the question being processed.

### xybrid expectations

Each stage holds one of:

| Value | Meaning |
|---|---|
| `"match"` | Reproduce `upstream.<stage>` exactly, starting from upstream's inputs to that stage |
| `"reference"` | `evaluate` only: upstream's logits are the numeric reference. Bit-identical only under G2 conditions (same GGUF bytes, natives, backend, threads, flash attention, context and batch shape); otherwise the G3 bounds in `provenance.json` |
| `"given"` | `evaluate` only: the scores are synthetic; use `upstream.evaluate.raw_scores` as the readout |
| `"not_reached"` | An earlier stage refuses |
| `"out_of_scope"` | Upstream behavior xybrid has no counterpart for in this profile (the media temperature override); informational |
| `{"refuse": "<kind>"}` | Refuse at this stage, with an error of this kind (below). Refusing at an earlier stage than upstream is the norm: xybrid checks budgets in `prepare`, and metadata at load |

`load` is present only for cases with `metadata` edits. Such a case runs on a
metadata-only copy of the artifact's GGUF (no tensors, so synthetic scores);
`decision-oracle variant` rebuilds it from the artifact and the edits, and a
consumer that tokenizes needs that copy's vocabulary when the edits change token
types.

Stage isolation: `prepare` is checked from the parsed request and the artifact;
`evaluate` from upstream's `token_ids` and `select`; `finalize` from upstream's
`raw_scores`, not xybrid's own logits. A wrong stage cannot hide behind another,
and the answer arithmetic is checked exactly even where logits differ.

### Comparing values

- Strings (`prompt`, keys, codes, `request_text`) compare as exact UTF-8 bytes.
  The prompt holds the U+200B that `d1_break` inserts, and whatever the request
  carried, U+0000 included.
- f32 values are their bit pattern, `"0x3fab2fdc"`. Rust:
  `f32::from_bits(u32::from_str_radix(&s[2..], 16)?)`.
- f64 values are JSON numbers printed with 17 significant digits (`%.17g`), so
  any correctly rounded parser gets the exact double back: Rust's
  `str::parse::<f64>` on the number's text, or serde_json with its
  `float_roundtrip` feature. **serde_json's default float parser is not
  guaranteed to round-trip 17 digits**; do not compare through it. An integral
  value keeps a `.0` (`0.0`, `1.0`), so every f64 reads back as a float.
- Non-finite values are the strings `"NaN"`, `"Infinity"`, `"-Infinity"`
  (llama-server itself writes NaN as `null`).
- `legend` values are the request's descriptions as upstream serializes JSON
  (`1.0` stays a float, `1` an integer); compare them as JSON values.
- f64 answers are exact on the reference platform (Linux x86_64, glibc `exp`),
  where Rust's `f64::exp` and C++'s `std::exp` call the same function.
  Elsewhere a different `exp` can move the last bits.

### Refusal kinds

Content-free categories for `{"refuse": ...}`; the implementation maps them to
its own error variants.

| Kind | When |
|---|---|
| `invalid_json` | The body is not one JSON value (syntax, trailing data) |
| `invalid_request` | The body is JSON but not an object |
| `duplicate_key` | An object has the same key twice, anywhere in the request (upstream keeps the last value) |
| `unknown_field` | A field the v1 transport does not define, at the top level (other than `model`) or in a question or its criteria (upstream ignores it) |
| `missing_state` | No `state` |
| `invalid_questions` | `questions` missing, not an object, or empty |
| `invalid_question` | A question that is not an object |
| `missing_instructions` | `instructions` missing or null |
| `invalid_question_type` | `type` missing, not a string, or not choice/noul/score |
| `invalid_criteria` | Choice criteria not a non-empty object, score criteria not an array, noul criteria neither null nor an object |
| `level_count` | A score with fewer than 2 or more than 10 levels |
| `too_many_options` | More options than the model supports (255) |
| `empty_question_id` | A question id that is the empty string (upstream accepts it) |
| `empty_option_key` | A choice key that is the empty string (upstream accepts it) |
| `media_unsupported` | `files`, `images`, `videos` or a media part in the state's messages: the text profile has no media |
| `spoken_instructions` | Instructions that are an audio part |
| `json_too_deep` | Nesting beyond the limit |
| `number_out_of_range` | A number upstream cannot render faithfully (see the cases) |
| `escaped_key_collision` | Two keys of one object that `d1_break` makes equal (upstream keeps one) |
| `no_single_token_label` | No single-token label left for an option |
| `budget_exceeded` | A pass longer than the token budget of `settings` |
| `non_finite_scores` | A NaN or +inf score, or no finite score among a question's options |
| `unsupported_metadata` | Optional decision metadata xybrid does not implement (e.g. `score_bias`) |
| `invalid_metadata` | Decision metadata that upstream also refuses (a temperature <= 0 after its string round trip) |

### Coverage tags

`covers` lists what a case exercises, so a gate can check that nothing is
missing:

| Area | Tags |
|---|---|
| State | `state-string`, `state-empty`, `state-null`, `state-object`, `state-array`, `state-messages` |
| JSON rendering | `json-order`, `json-float`, `json-int`, `json-unicode`, `json-escapes`, `json-literals`, `json-depth` |
| Instructions and descriptions | `instructions-structured`, `description-structured`, `description-falsy`, `description-null` |
| Choice labels | `choice-letter-keys`, `choice-lowercase-keys`, `choice-one-option`, `choice-26`, `choice-27`, `choice-pool-fallback`, `choice-max-options`, `choice-order` |
| Noul | `noul-no-criteria`, `noul-partial-criteria`, `noul-null-criteria`, `noul-order` |
| Score | `score-2-levels`, `score-10-levels`, `score-null-level` |
| Escaping | `escape-control`, `escape-keys`, `escape-overlap`, `escape-key-collision`, `escape-user-defined` |
| Requests | `multi-question`, `budget`, `transport`, `media` |
| Answer arithmetic | `math-reference`, `math-median-b3`, `math-median-not-mode`, `math-tie`, `math-underflow`, `math-neg-inf`, `math-non-finite`, `math-alias-max`, `math-temperature-bucket`, `math-temperature-default`, `math-score-bias`, `math-media-temperature` |
| Refusals | `refusal-load`, `refusal-parse`, `refusal-prepare`, `refusal-evaluate`, `refusal-finalize` |
| Divergence | `divergence` (xybrid's expectation differs from upstream's behavior) |

## The G0 checklist, case by case

| Review item | Cases |
|---|---|
| Structured, null and empty values | `state-null`, `state-empty-string`, `state-whitespace`, `instructions-object`, `instructions-array`, `instructions-scalars`, `choice-null-descriptions`, `choice-falsy-descriptions`, `choice-structured-descriptions`, `score-null-levels`, `score-structured-levels`, `noul-structured-descriptions` |
| Nested, unsorted objects | `state-object-nested`, `state-wrapped-messages`, `instructions-object`, `choice-criteria-order` |
| Float and integer formatting | `state-numbers`, `state-u64-above-i64`, `state-number-overflow`, `choice-structured-descriptions` |
| All-letter and lowercase keys | `choice-letter-keys`, `choice-lowercase-keys`, `choice-mixed-case-keys`, `choice-letter-and-word-keys`, `choice-non-letter-short-keys` |
| One option; 26 and 27 options | `choice-one-option`, `math-one-option`; `choice-26-options`, `choice-27-options` |
| A real pool fallback | `choice-36-options` (AJ becomes 00), `choice-120-options`, `choice-255-options` |
| Overlapping control spellings | `escape-adjacent-and-partial`, `escape-overlap-variant` (the d1-2B spellings cannot overlap; the variant makes `\|>` CONTROL) |
| Escaped-key collisions | `escape-key-collision-state`, `escape-key-collision-nested` |
| Noul criteria: partial, all null | `noul-partial-true`, `noul-partial-false`, `noul-empty-string-description`, `noul-all-null`, `noul-criteria-null`, `noul-criteria-empty-object`, `noul-false-first` |
| Score with 2 and 10 levels | `score-2-levels`, `score-10-levels` |
| Null state against empty text | `state-null`, `state-empty-string` |
| Malformed requests | the `parse-*` cases, `instructions-null`, `instructions-missing`, `choice-criteria-*`, `choice-256-options`, `score-1-level`, `score-11-levels`, `score-criteria-object`, `noul-criteria-array`, `depth-*-over-limit`, `budget-over-limit` |
| Answer arithmetic | `math-median-b3` (the review's float32 counterexample), `math-median-exact-half`, `math-median-not-mode`, `math-choice-tie`, `math-noul-tie`, `math-underflow`, `math-neg-inf-some`, `math-neg-inf-all`, `math-nan*`, `math-pos-inf*`, `math-alias-max`, `math-temperature-*`, `math-score-bias`, `math-media-temperature*` |
