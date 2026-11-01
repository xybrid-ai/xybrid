//! The SystemOne transport for decision requests: subset v1.
//!
//! Internal: the CLI's and its tests' interface (`#[doc(hidden)]`), with no
//! stability promise.
//!
//! SystemOne is the JSON shape llama.cpp's server answers decision requests
//! in (`POST /v1/systemone`). `xybrid run --decision` reads a request in a
//! subset of it, and writes the answers back in its response shape.
//! [`parse_request`] turns a request body into the core's
//! [`DecisionRequest`]; [`render_response`] writes [`Decisions`] as the
//! response.
//!
//! The subset is the text profile of LiquidAI's LFM3 d1 (`lfm3-d1/text/v1`):
//! a state and typed questions, no media, no spoken instructions. Parsing
//! follows llama.cpp's own request reading (`parse_questions` and
//! `parse_state` in `tools/server/server-decision.cpp`) wherever it accepts
//! a request, and refuses what it would silently drop or misread. The
//! decision fixtures (`integration-tests/fixtures/decision/`) hold the
//! reference cases and name the refusal kinds ([`RefusalKind`]).
//!
//! # Request
//!
//! ```text
//! {
//!   "model": <any; ignored>,
//!   "state": <string | object | array | number | boolean | null>,
//!   "questions": {
//!     "<id>": {
//!       "type": "choice" | "noul" | "score",
//!       "instructions": <any JSON but null>,
//!       "criteria": <by type, below>
//!     },
//!     ...
//!   }
//! }
//! ```
//!
//! - **state** is required; `null` is a null state, which LFM3 d1 renders
//!   without the part that introduces the state, and `""` is an empty one.
//!   A string is text, anything else JSON.
//! - **questions** is a non-empty object; its member order is the question
//!   order, and each id names its answer.
//! - **instructions** are text (a string) or JSON; missing or `null` is
//!   refused.
//! - **criteria**: a *choice* takes a non-empty object of option key to
//!   description, in order, at most 255 options; a *score* takes an array of
//!   2 to 10 level descriptions, lowest first; a *noul* (yes/no) takes
//!   `null`, nothing, or an object describing `"true"` and `"false"`, either
//!   or both. A description is text or JSON; `null` is no description.
//!
//! Content (the state, instructions, descriptions) is read as written: key
//! order and number spelling kept (`1` and `1.0` differ), through
//! [`OrderedJson::parse`], never `serde_json::Value`.
//!
//! # Refusals and their order
//!
//! The first refusal ends the parse. Each [`RequestError`] names its
//! [`RefusalKind`] (the fixtures' name) and positions counting from 0, never
//! content: no id, key, text, or field name the caller made up. The checks
//! run in this order:
//!
//! 1. **The body**, as JSON: at most 8 MiB and 131,072 values (the
//!    transport's ceilings, set well above what the defaults of
//!    `validate_structure` accept: `request_too_large`, the one kind no
//!    fixture names); UTF-8; one JSON value under
//!    [`OrderedJson::parse`]'s grammar (`invalid_json`); no number beyond
//!    the `f64` range (`number_out_of_range`, as `1e400`); then no object
//!    anywhere that repeats a key (`duplicate_key`; llama.cpp keeps the
//!    last value). The body itself has no depth limit: content nests up to
//!    128 levels counting from the content, as [`OrderedJson`] does,
//!    however deep it sits in the request.
//! 2. **What llama.cpp refuses, in its order**, so such a request is refused
//!    for the same reason: a body that is not an object
//!    (`invalid_request`), no `state` (`missing_state`), `questions`
//!    (`invalid_questions`); then each question in order: not an object
//!    (`invalid_question`), instructions (`missing_instructions`,
//!    `spoken_instructions` for an object whose `type` is `"input_audio"`),
//!    the type (`invalid_question_type`; a type that is not a string reads
//!    as none), the criteria (`invalid_criteria`, `level_count`), the option
//!    count (`too_many_options`); then media (`media_unsupported`): a
//!    `videos`, `files` or `images` field, whatever its value (`[]` and
//!    `null` included), or an `image_url`, `input_audio` or `video_url`
//!    part in the `content` array of a chat message of the state (the state
//!    when it is an array, its `messages` when it is an object holding
//!    them).
//! 3. **What the subset refuses although llama.cpp accepts it**, in reading
//!    order: a request field other than `state`, `questions` and `model`;
//!    then each question: an empty id (`empty_question_id`), a field other
//!    than `type`, `instructions` and `criteria`, an empty option key
//!    (`empty_option_key`), and noul criteria members other than `"true"`
//!    and `"false"` (`unknown_field`; llama.cpp ignores them, `"yes"` and
//!    `"no"` included). Unknown fields are looked for in these objects only,
//!    never inside content.
//! 4. **Content**, in request order (the state, then each question's
//!    instructions and descriptions): nested deeper than 128 levels
//!    (`json_too_deep`), an integer above `i64::MAX` (`number_out_of_range`:
//!    llama.cpp's template engine reads it wrapped to a negative number;
//!    integers beyond `u64::MAX` are floats there, and stay floats here), or
//!    more than 65,536 values in one piece (`request_too_large`, a ceiling
//!    like those of step 1).
//!
//! The protocol's own limits come later, when the model's protocol prepares
//! the request: LFM3 d1 caps the state and instructions at 127 levels and
//! descriptions at 125, and every pass at its token budget.
//!
//! # Response
//!
//! ```text
//! {
//!   "model": "<the model that answered>",
//!   "answers": {
//!     "<id>": {"type": "choice", "choice": "<key>", "probabilities": {"<key>": p, ...}, "confidence": c},
//!     "<id>": {"type": "noul", "noul": p_true},
//!     "<id>": {"type": "score", "score": s, "legend": {"0": <description>, ...}, "probabilities": {"0": p, ...}, "confidence": c}
//!   },
//!   "usage": {"input_tokens": n, "output_tokens": 0}
//! }
//! ```
//!
//! Members come in llama.cpp's order (`format_answer`), answers in request
//! order, probabilities in option or level order. `legend` gives each
//! level's description back as the JSON value it was in the request (`null`
//! when absent), key order and number spelling kept. `input_tokens` counts
//! the tokens of every pass; `confidence` is `null` for a protocol that
//! defines none (LFM3 d1 always does).
//!
//! Every computed number (probabilities, confidence, `noul`, `score`) is an
//! `f64`, written with the fewest digits that read back as the same `f64`
//! (serde_json's formatter: fixed notation for decimal exponents -5 to 15,
//! with `.0` after an integral value, else `d.ddde±x`), so a correctly
//! rounded parser gets the exact value back; serde_json's own default float
//! parser is not one. A valid answer holds only finite numbers, so none is
//! written as `NaN` or `Infinity`: a non-finite number refuses the response
//! ([`ResponseError::NonFinite`]).
//!
//! [`DecisionRequest`]: xybrid_core::ir::DecisionRequest
//! [`Decisions`]: xybrid_core::ir::Decisions
//! [`OrderedJson`]: xybrid_core::ir::OrderedJson
//! [`OrderedJson::parse`]: xybrid_core::ir::OrderedJson::parse

mod body;
mod request;
mod response;

pub use request::{
    parse_request, ContentAt, CriteriaShape, FieldAt, MediaAt, RefusalKind, RequestError,
    RequestLimits, DEFAULT_MAX_BODY_BYTES, DEFAULT_MAX_BODY_VALUES,
};
pub use response::{render_response, ResponseError};
