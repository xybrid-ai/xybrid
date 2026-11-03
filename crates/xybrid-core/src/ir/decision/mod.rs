//! Decision data: the request a caller sends to a decision model and the
//! answers it returns.
//!
//! A decision model reads one state (text, JSON, or nothing) and answers
//! typed questions about it: pick an option, yes or no, or a level on a
//! scale. It reads each answer from the model without generating text.
//! [`DecisionRequest`] carries the request and [`Decisions`] the answers.
//! LiquidAI's LFM3 d1 models are the first protocol; the types are
//! protocol-neutral.
//!
//! The legacy choice types ([`ChoiceRequest`](super::ChoiceRequest),
//! [`ChoiceScores`](super::ChoiceScores)) are unchanged beside these, and
//! the two differ on purpose. A choice id only names a candidate, while a
//! decision option key may change what the model reads ([`DecisionOption`]).
//! Decision content can be structured JSON or absent, not only text. And
//! decision probabilities are `f64`, so a protocol can finalize answers
//! exactly as its reference does.
//!
//! | Item | Role |
//! |------|------|
//! | [`OrderedJson`] | JSON content that keeps key order and number identity |
//! | [`DecisionContent`] | Text, or a JSON value that is not a string |
//! | [`DecisionRequest`], [`DecisionState`], [`DecisionQuestion`], [`DecisionQuestionKind`], [`DecisionOption`], [`DecisionLevel`] | The request |
//! | [`Decisions`], [`DecisionAnswer`], [`DecisionScore`], [`DecisionValue`] | One answer per question, in question order |
//! | [`validate_structure`], [`StructuralLimits`] | The checks every protocol needs |
//! | [`DecisionError`] | Every refusal, with content-free messages |
//!
//! These types are plain data. A protocol decides which questions a model
//! accepts (kinds, numbers of options and levels, required or forbidden
//! descriptions and instructions, a null state, media), renders the prompt,
//! reads the model, and builds the answers; none of that lives here.
//!
//! # Absent, null and empty
//!
//! Every piece of content a request may leave out (a state body,
//! instructions, an option, yes/no or level description) sits in an
//! `Option`. JSON `null` is never content: [`DecisionContent::from_json`]
//! (and [`DecisionState::from_json`]) turn `null` into `None` and a JSON
//! string into text, so each of those has one form. `None` and
//! an empty text differ: LFM3 d1, for one, renders a null state by leaving
//! the state out, and an empty one as an empty state. Nulls nested inside a
//! JSON value are kept.
//!
//! # Ordered JSON
//!
//! [`OrderedJson`] is built only by [`OrderedJson::parse`], a parser written
//! for it, never through `serde_json::Value`. It keeps object members in
//! document order and classifies numbers as llama.cpp's JSON library
//! (nlohmann::json) does ([`JsonNumber`]). It refuses repeated keys,
//! numbers that overflow `f64`, nesting deeper than [`MAX_JSON_DEPTH`], and
//! text over its [`JsonLimits`]. [`OrderedJson::to_json_text`] writes compact
//! JSON in serde_json's style; that is not llama.cpp's `tojson` filter,
//! which a protocol renders itself.
//!
//! # Answers
//!
//! [`DecisionAnswer::new`] (and deserialization, which calls it) refuses an
//! answer that disagrees with its distribution: the chosen key must have the
//! highest probability, `p_yes` must be the probability of `"true"`, a score
//! must be the statistic it names, and probabilities must sum to 1 within
//! [`PROBABILITY_TOLERANCE`]. Probabilities follow the temperature the
//! model's publisher set; they are the protocol's canonical distribution,
//! not calibrated probabilities. A raw score may be `-inf`.
//!
//! # Content never reaches errors or `Debug`
//!
//! State, question ids, option keys, instructions, descriptions and every
//! part of a JSON value are caller content. `Debug` on every type here
//! prints kinds, lengths, counts and model numbers only, and every
//! [`DecisionError`] (its message, its `Debug`, and the `AdapterError` it
//! converts into) names byte offsets, positions, counts and limits only. A
//! format's own deserialization errors (serde_json quoting an unexpected
//! value, say) are outside that promise; the boundary that decodes envelopes
//! must withhold them.
//!
//! # Decoding is bounded
//!
//! Deserializing these types, in every format, applies fixed ceilings
//! before it allocates what they bound, nested values included:
//!
//! | Field | Ceiling |
//! |-------|---------|
//! | a request's questions, a result's answers | 16 |
//! | a choice's options, a score's levels, an answer's distribution | 255 |
//! | each string: question id, option key, text content, answer question id, distribution key, chosen key | 1 MiB |
//! | an [`OrderedJson`]'s text | 2 MiB, and 65,536 values |
//! | a state's media | 0, while [`DecisionMedia`] has no variant |
//!
//! A sequence is refused on its declared length when the format declares
//! one (bincode does), before any element is read, and otherwise at the
//! element past its ceiling, before that element is decoded; a string is
//! measured before it is copied, and JSON text before it is parsed. A
//! refusal names the kind of field and the ceiling, never content. The
//! ceilings are the IR's, the same for every model, and at or above the
//! default [`StructuralLimits`], so data within the defaults always decodes
//! (JSON text has room for its compact form, which can be longer than the
//! text it was parsed from); configured limits should stay within them.
//!
//! With bincode, how the input arrives decides what bounds allocation:
//!
//! - **From a byte slice**, these ceilings do: a declared length is refused
//!   before anything is reserved, and a string arrives borrowed and is
//!   measured before it is copied. bincode's own `with_limit` does not apply
//!   to slices in bincode 1.3.x, which replaces it with no limit, so
//!   `options.with_limit(n).deserialize(&bytes)` adds no bound of its own.
//! - **From an `io::Read`**, use `with_limit` with `deserialize_from`:
//!   bincode reads a string's declared length into its own buffer before
//!   any visitor runs, so only its limit bounds that allocation.
//!
//! # Compatibility
//!
//! The representation is frozen; these are the rules for changing it.
//!
//! - **Enums are append-only.** A new variant goes last, and existing
//!   variants never move or go away: bincode and the generated bindings
//!   encode a variant by its index (pinned by tests). Every enum that can
//!   grow is `#[non_exhaustive]` ([`DecisionMedia`], [`DecisionInstructions`],
//!   [`DecisionQuestionKind`], [`DecisionValue`], [`DecisionOrigin`],
//!   [`ScoreStatistic`], [`DecisionError`] and its kinds), and so is every
//!   variant of [`DecisionQuestionKind`] and [`DecisionValue`]: code outside
//!   this crate builds them with their constructors
//!   ([`DecisionQuestionKind::choice`], [`DecisionValue::score`], ...) and
//!   reads them with their accessors or `{ .. }` patterns, with a wildcard
//!   arm for variants added later. Sets that cannot grow stay exhaustive:
//!   [`JsonKind`], [`JsonView`], [`JsonNumber`], [`DecisionContentView`].
//! - **Record field sets freeze at the first release.** Structs have private
//!   fields, constructors and accessors, and [`OrderedJson`] and
//!   [`DecisionContent`] are opaque, so Rust callers never name a field. But
//!   bincode and the generated FFI records are positional, so after the
//!   first release a need for more data is met by a new variant or a new
//!   type, never by a new field.
//! - **JSON.** Every struct refuses unknown fields, so a decoder built
//!   before a field was added rejects data that has it: deliberate, so that
//!   nothing is silently dropped. Enums use serde's default external
//!   tagging, and an old decoder refuses a new variant. Floats are JSON
//!   numbers, written with the shortest digits that read back as the same
//!   `f64`. xybrid-core turns on serde_json's `float_roundtrip` feature, so
//!   serde_json reads each one back as exactly that `f64` (its default
//!   parsing is best-effort and can be an ulp off, enough to move a median
//!   across its one-half boundary). Decoding, in every format, checks
//!   exactly what [`DecisionAnswer::new`] checks; there is no tolerance and
//!   no exception.
//! - **`OrderedJson` in serde.** In every format it is one string holding
//!   its compact JSON text, which keeps key order and number identity
//!   exactly and works in formats that cannot carry an ordered map
//!   (bincode). Decoding parses the string under the decode ceilings
//!   (2 MiB of text, 65,536 values). Floats are spelled as serde_json
//!   spells them, so the bytes (and [`StructuralLimits::max_bytes`] totals)
//!   can change with serde_json; the values cannot.
//! - **Raw scores.** In human-readable formats a finite
//!   [`DecisionScore::raw_score`] is a number and `-inf` is the string
//!   `"-Infinity"`, never `null`; binary formats keep the `f32` itself.
//! - **bincode and envelopes.** bincode writes no field names or tags: a
//!   struct is its fields in order and a variant its index, which is why
//!   the two rules above keep bytes readable across releases. Envelope kinds
//!   that carry these types are appended to
//!   [`EnvelopeKind`](super::EnvelopeKind) without moving existing ordinals,
//!   by the envelope's own rule; this module adds none.
//! - **FFI.** Generated bindings carry their own records (content as
//!   `{text}` or `{json_text}`, ordered JSON as text for
//!   [`OrderedJson::parse`]) and are versioned with the bindings. Converting
//!   this IR to them is fallible: a variant added after a binding was
//!   generated is refused as unrepresentable, never dropped or
//!   approximated.
//!
//! Nothing promises that an older client understands a newer variant or
//! field; it refuses it.
//!
//! # Example
//!
//! ```
//! use xybrid_core::ir::decision::{validate_structure, StructuralLimits};
//! use xybrid_core::ir::{
//!     DecisionAnswer, DecisionContent, DecisionOption, DecisionOrigin, DecisionQuestion,
//!     DecisionQuestionKind, DecisionRequest, DecisionScore, DecisionState, DecisionUsage,
//!     DecisionValue, Decisions, OrderedJson,
//! };
//!
//! let state = DecisionState::from_json(OrderedJson::parse(
//!     r#"{"ticket": "Order 1042 arrived damaged", "tier": "gold", "refunds": 0}"#,
//! )?);
//! let action = DecisionQuestion::new(
//!     "action",
//!     Some(DecisionContent::text("What should support do next?").into()),
//!     DecisionQuestionKind::choice(vec![
//!         DecisionOption::new("refund", Some(DecisionContent::text("Refund the order"))),
//!         DecisionOption::new("send_replacement", None),
//!     ]),
//! );
//! let request = DecisionRequest::new(state, vec![action]);
//! validate_structure(&request, &StructuralLimits::default())?;
//! assert!(!format!("{request:?}").contains("1042"));
//!
//! let answer = DecisionAnswer::new(
//!     "action",
//!     vec![
//!         DecisionScore::new("refund", 2.5, 0.75, DecisionOrigin::Caller),
//!         DecisionScore::new("send_replacement", 1.0, 0.25, DecisionOrigin::Caller),
//!     ],
//!     DecisionValue::choice("refund"),
//!     Some(0.75),
//! )?;
//! let decisions = Decisions::new(vec![answer], DecisionUsage::new(1, 412));
//! assert_eq!(decisions.answer("action").unwrap().confidence(), Some(0.75));
//! assert!(!format!("{decisions:?}").contains("refund"));
//! # Ok::<(), xybrid_core::ir::DecisionError>(())
//! ```

mod answer;
mod decode;
mod error;
mod json;
mod request;
mod validate;

#[cfg(test)]
mod tests;

pub use answer::{
    DecisionAnswer, DecisionOrigin, DecisionScore, DecisionUsage, DecisionValue, Decisions,
    ScoreStatistic, PROBABILITY_TOLERANCE,
};
pub use error::{AnswerInvariant, DecisionError, JsonErrorKind};
pub use json::{
    JsonKind, JsonLimits, JsonNumber, JsonView, OrderedJson, DEFAULT_MAX_JSON_BYTES,
    DEFAULT_MAX_JSON_NODES, MAX_JSON_DEPTH,
};
pub use request::{
    DecisionContent, DecisionContentView, DecisionInstructions, DecisionLevel, DecisionMedia,
    DecisionOption, DecisionQuestion, DecisionQuestionKind, DecisionRequest, DecisionState,
};
pub use validate::{
    validate_structure, StructuralLimits, DEFAULT_MAX_OPTIONS, DEFAULT_MAX_QUESTIONS,
    DEFAULT_MAX_REQUEST_BYTES, DEFAULT_MAX_REQUEST_JSON_NODES,
};
