//! Choice-scorer specifications: how a decision model turns one context and a
//! list of candidates into tensors, and which output holds their logits.
//!
//! A spec is metadata, not behaviour. It is checked by
//! [`validate_scorer_spec`](crate::execution::choice::validate_scorer_spec),
//! bound to a loaded model by the runtime for its variant, and used to encode
//! every request.
//!
//! # Example
//!
//! ```
//! use xybrid_core::execution::template::ChoiceScorerSpec;
//!
//! let spec: ChoiceScorerSpec = serde_json::from_str(r#"{
//!     "type": "OnnxByteOptionScorer",
//!     "model_file": "scorer.onnx",
//!     "context": { "offset": 1, "max_len": 224, "overflow": "Truncate",
//!                  "ids_input": "context_ids", "mask_input": "context_mask" },
//!     "choice": { "offset": 1, "max_len": 96, "overflow": "Truncate",
//!                 "ids_input": "option_ids", "mask_input": "option_token_mask" },
//!     "choice_mask_input": "option_mask",
//!     "logits_output": "logits",
//!     "max_choices": 64,
//!     "fixed_choices": [{ "id": "skip", "text": "skip" }]
//! }"#)?;
//! let ChoiceScorerSpec::OnnxByteOptionScorer(onnx) = &spec else { unreachable!() };
//! assert_eq!(onnx.context.max_len, 224);
//! # Ok::<(), serde_json::Error>(())
//! ```

use crate::ir::Choice;
use serde::{Deserialize, Serialize};

#[cfg(feature = "schema")]
use schemars::JsonSchema;

/// How a choice scorer reads its inputs and produces one logit per candidate.
///
/// Serialized with a `"type"` tag naming the variant. New scorer kinds are
/// added as variants, so match with a wildcard arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum ChoiceScorerSpec {
    /// An ONNX model that reads the context and every candidate as UTF-8
    /// byte ids and returns a `[1, N]` float32 logit per candidate.
    OnnxByteOptionScorer(OnnxByteOptionScorerSpec),
}

/// An ONNX choice scorer over UTF-8 byte ids.
///
/// The model takes five tensors:
///
/// | Input | Type | Shape |
/// |---|---|---|
/// | `context.ids_input` | int64 | `[1, context.max_len]` |
/// | `context.mask_input` | bool | `[1, context.max_len]` |
/// | `choice.ids_input` | int64 | `[1, N, choice.max_len]` |
/// | `choice.mask_input` | bool | `[1, N, choice.max_len]` |
/// | `choice_mask_input` | bool | `[1, N]` |
///
/// and returns `logits_output` as float32 `[1, N]`. `N` is either a dynamic
/// axis (one row per candidate) or a fixed size of at least `max_choices`,
/// whose unused rows are masked out and never scored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct OnnxByteOptionScorerSpec {
    /// ONNX file, relative to the model directory.
    pub model_file: String,
    /// Encoding of the context.
    pub context: ByteFieldSpec,
    /// Encoding of each candidate's text.
    pub choice: ByteFieldSpec,
    /// Bool input marking which of the `N` candidate rows are offered.
    pub choice_mask_input: String,
    /// Float32 output holding one logit per candidate row.
    pub logits_output: String,
    /// Most candidates one request may offer, counting the fixed choices.
    pub max_choices: usize,
    /// Candidates the model always offers, appended after the caller's.
    /// Their ids are reserved: a caller choice may not reuse one.
    #[serde(default)]
    pub fixed_choices: Vec<Choice>,
    /// Divides every logit before the softmax. Finite and greater than zero;
    /// `None` means 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoring_temperature: Option<f32>,
}

/// How one text field becomes a row of byte ids and its mask.
///
/// Each UTF-8 byte `b` becomes the id `b + offset`. The row is cut at
/// `max_len` bytes (possibly inside a multi-byte character) and zero padded;
/// the mask is `true` exactly where an id was written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ByteFieldSpec {
    /// Added to every byte. At least 1, so a byte id is never the padding id 0.
    pub offset: u32,
    /// Row length in bytes.
    pub max_len: usize,
    /// What to do with text longer than `max_len` bytes.
    pub overflow: ByteOverflow,
    /// Int64 input receiving the ids.
    pub ids_input: String,
    /// Bool input receiving the mask.
    pub mask_input: String,
}

/// What a byte field does with text longer than its `max_len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
pub enum ByteOverflow {
    /// Keep the first `max_len` bytes.
    Truncate,
    /// Refuse the request.
    Reject,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_json() -> String {
        // The runtime CARGO_MANIFEST_DIR, not `env!`: under Bazel the test runs
        // in a sandbox where the compile-time path points nowhere.
        let manifest_dir =
            std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set for tests");
        let path = std::path::Path::new(&manifest_dir)
            .join("../../integration-tests/fixtures/choice/specs/cua-s1-forms.json");
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    #[test]
    fn the_committed_forms_spec_deserializes() {
        let spec: ChoiceScorerSpec = serde_json::from_str(&fixture_json()).unwrap();
        let ChoiceScorerSpec::OnnxByteOptionScorer(onnx) = &spec;
        assert_eq!(onnx.model_file, "cua-s1-forms.onnx");
        assert_eq!((onnx.context.offset, onnx.context.max_len), (1, 224));
        assert_eq!((onnx.choice.offset, onnx.choice.max_len), (1, 96));
        assert_eq!(onnx.context.overflow, ByteOverflow::Truncate);
        assert_eq!(onnx.max_choices, 64);
        let fixed: Vec<&str> = onnx.fixed_choices.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(fixed, ["check", "click", "skip"]);
        assert_eq!(onnx.scoring_temperature, None);

        let round_trip: ChoiceScorerSpec =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(round_trip, spec);
    }

    #[test]
    fn unknown_fields_and_types_are_rejected() {
        let mut value: serde_json::Value = serde_json::from_str(&fixture_json()).unwrap();
        value["max_choice"] = 8.into();
        assert!(serde_json::from_value::<ChoiceScorerSpec>(value).is_err());

        let mut value: serde_json::Value = serde_json::from_str(&fixture_json()).unwrap();
        value["context"]["overflow"] = "Wrap".into();
        assert!(serde_json::from_value::<ChoiceScorerSpec>(value).is_err());

        let mut value: serde_json::Value = serde_json::from_str(&fixture_json()).unwrap();
        value["type"] = "GgufLabelToken".into();
        assert!(serde_json::from_value::<ChoiceScorerSpec>(value).is_err());
    }
}
