//! Choice-scorer specifications: which runtime scores a decision model's
//! requests, and how.
//!
//! A spec is metadata, not behaviour. It is checked by
//! [`validate_scorer_spec`](crate::execution::choice::validate_scorer_spec),
//! bound to a loaded model by the runtime for its variant, and used to encode
//! every request. Two kinds exist:
//!
//! - [`OnnxByteOptionScorerSpec`]: an ONNX model that reads one context and a
//!   list of candidates as UTF-8 byte ids, and the output holding their
//!   logits.
//! - [`GgufDecisionSpec`]: a llama.cpp decision model in one GGUF, such as
//!   LiquidAI's LFM3 d1. Its behaviour (prompt template, temperatures,
//!   attention) is read from the GGUF; the spec names the file, the
//!   [`DecisionProtocol`] the file must carry, and resource ceilings.
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

use crate::gguf::DecisionProfileId;
use crate::ir::decision::{DEFAULT_MAX_OPTIONS, DEFAULT_MAX_QUESTIONS};
use crate::ir::Choice;
use serde::{Deserialize, Serialize};
use std::fmt;

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
    /// A llama.cpp decision model in one GGUF, such as LiquidAI's LFM3 d1:
    /// it answers decision questions with one readout per question instead
    /// of generating text.
    ///
    /// The GGUF must classify as a decision model of `protocol`
    /// ([`crate::gguf::classify`]); every load checks it with
    /// [`crate::gguf::admit_gguf`].
    GgufDecision(GgufDecisionSpec),
}

impl ChoiceScorerSpec {
    /// The candidates the scorer appends after the caller's, in order. Their
    /// ids are reserved: a caller choice may not reuse one.
    ///
    /// Always empty for a [`GgufDecisionSpec`]: its protocol's own labels
    /// (yes/no, score levels) are not caller-visible choice ids.
    pub fn fixed_choices(&self) -> &[Choice] {
        match self {
            Self::OnnxByteOptionScorer(spec) => &spec.fixed_choices,
            Self::GgufDecision(_) => &[],
        }
    }
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

/// A llama.cpp decision model in one GGUF file.
///
/// The GGUF defines the model's behaviour: its `systemone` prompt template,
/// vocabulary, attention and decision temperatures are read from the file at
/// load and never restated here, so the spec cannot override them. It holds
/// what the file cannot know: where the file is, which protocol the bundle
/// expects it to carry, and how much one request may cost.
///
/// `protocol` is an assertion: a load refuses a GGUF whose decision type
/// differs (see [`crate::gguf::admit_gguf`]).
///
/// The resource defaults and ceilings are provisional: they come from CPU
/// measurements of d1-2B (Q4_K_M, four threads) and will be revised once
/// mobile and GPU devices are measured. A bundle may lower them for a
/// platform.
///
/// ```
/// use xybrid_core::execution::template::{ChoiceScorerSpec, DecisionProtocol, GgufDecisionSpec};
///
/// let spec: ChoiceScorerSpec = serde_json::from_str(r#"{
///     "type": "GgufDecision",
///     "model_file": "d1-2B-Q4_K_M.gguf",
///     "protocol": "lfm3-d1"
/// }"#)?;
/// assert_eq!(
///     spec,
///     ChoiceScorerSpec::GgufDecision(GgufDecisionSpec::new("d1-2B-Q4_K_M.gguf", DecisionProtocol::Lfm3D1))
/// );
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct GgufDecisionSpec {
    /// GGUF file, relative to the model directory.
    pub model_file: String,
    /// The decision protocol the GGUF must carry. Checked against the file's
    /// decision type at every load, never applied to it.
    pub protocol: DecisionProtocol,
    /// Most tokens one readout may evaluate: one question's whole prompt. It
    /// sizes the native context (context, batch and micro-batch alike), so
    /// memory and latency grow with it. At least 1 and at most
    /// [`Self::MAX_TOKEN_BUDGET`], or
    /// [`Self::MAX_TOKEN_BUDGET_WITHOUT_FLASH_ATTN`] when `flash_attn` is
    /// off. Provisional.
    #[serde(default = "default_token_budget")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 8192)))]
    pub token_budget: u32,
    /// Whether the native context uses flash attention: on by default. Off
    /// is a diagnostic setting (it changes the scores slightly, and the
    /// compute buffer then grows quadratically with `token_budget`), allowed
    /// only up to [`Self::MAX_TOKEN_BUDGET_WITHOUT_FLASH_ATTN`] tokens.
    #[serde(default = "default_flash_attn")]
    pub flash_attn: bool,
    /// Most questions one request may ask, each one readout. Between 1 and
    /// [`Self::MAX_QUESTIONS`].
    #[serde(default = "default_max_questions")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 16)))]
    pub max_questions: usize,
    /// Most options of one choice question, and most levels of one score
    /// question. Between 1 and [`Self::MAX_OPTIONS`].
    #[serde(default = "default_max_options")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 255)))]
    pub max_options: usize,
    /// Most tokens one request may evaluate across all its readouts: a
    /// per-readout budget alone does not bound a request of many questions.
    /// At least `token_budget` (else a readout the budget admits could never
    /// run) and at most [`Self::MAX_TOTAL_TOKENS`]. Provisional.
    #[serde(default = "default_max_total_tokens")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 131072)))]
    pub max_total_tokens: u32,
}

impl GgufDecisionSpec {
    /// Default `token_budget`: 4,096 tokens, about 0.6 GiB of compute buffer
    /// (2.3 GiB peak with d1-2B's Q4_K_M weights) on CPU with flash
    /// attention. Provisional, pending device measurements.
    pub const DEFAULT_TOKEN_BUDGET: u32 = 4096;

    /// The largest `token_budget`, which needs flash attention: 8,192 tokens,
    /// about 3.0 GiB peak with d1-2B's Q4_K_M weights on CPU. Provisional.
    pub const MAX_TOKEN_BUDGET: u32 = 8192;

    /// The largest `token_budget` without flash attention: 4,096 tokens.
    /// Without it the compute buffer grows quadratically (1.34 GiB at 4,096,
    /// 4.99 GiB at 8,192 for d1-2B). Provisional.
    pub const MAX_TOKEN_BUDGET_WITHOUT_FLASH_ATTN: u32 = 4096;

    /// The most questions per request, and the default: 16, a product
    /// ceiling on the readouts one run may take
    /// ([`DEFAULT_MAX_QUESTIONS`]).
    pub const MAX_QUESTIONS: usize = DEFAULT_MAX_QUESTIONS;

    /// The most options per choice question (and levels per score question),
    /// and the default: 255, llama.cpp's ceiling for LFM3 d1
    /// ([`DEFAULT_MAX_OPTIONS`]).
    pub const MAX_OPTIONS: usize = DEFAULT_MAX_OPTIONS;

    /// Default `max_total_tokens`: 16,384 tokens, four full readouts at the
    /// default budget. Provisional, pending device measurements.
    pub const DEFAULT_MAX_TOTAL_TOKENS: u32 = 16_384;

    /// The largest `max_total_tokens`: every question at the largest budget,
    /// 16 × 8,192 = 131,072 tokens.
    pub const MAX_TOTAL_TOKENS: u32 = Self::MAX_QUESTIONS as u32 * Self::MAX_TOKEN_BUDGET;

    /// A spec for `model_file` carrying `protocol`, with the default budget
    /// and limits and flash attention on.
    pub fn new(model_file: impl Into<String>, protocol: DecisionProtocol) -> Self {
        Self {
            model_file: model_file.into(),
            protocol,
            token_budget: Self::DEFAULT_TOKEN_BUDGET,
            flash_attn: true,
            max_questions: Self::MAX_QUESTIONS,
            max_options: Self::MAX_OPTIONS,
            max_total_tokens: Self::DEFAULT_MAX_TOTAL_TOKENS,
        }
    }
}

fn default_token_budget() -> u32 {
    GgufDecisionSpec::DEFAULT_TOKEN_BUDGET
}

fn default_flash_attn() -> bool {
    true
}

fn default_max_questions() -> usize {
    GgufDecisionSpec::MAX_QUESTIONS
}

fn default_max_options() -> usize {
    GgufDecisionSpec::MAX_OPTIONS
}

fn default_max_total_tokens() -> u32 {
    GgufDecisionSpec::DEFAULT_MAX_TOTAL_TOKENS
}

/// A decision protocol: the `<arch>.decision.type` a decision GGUF carries,
/// which fixes how a request is rendered, read out and turned into answers.
///
/// Serialized as that decision type. Only protocols this build implements
/// have a variant, so metadata naming another one fails to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[non_exhaustive]
pub enum DecisionProtocol {
    /// LiquidAI's LFM3 d1 (`lfm3-d1`), run by the `lfm3-d1/text/v1`
    /// profile: a text, JSON or absent state, and no media.
    #[serde(rename = "lfm3-d1")]
    Lfm3D1,
}

impl DecisionProtocol {
    /// The decision type, as both the GGUF and the metadata spell it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lfm3D1 => "lfm3-d1",
        }
    }
}

impl fmt::Display for DecisionProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<DecisionProfileId> for DecisionProtocol {
    /// The protocol a classification profile implements.
    fn from(profile: DecisionProfileId) -> Self {
        match profile {
            DecisionProfileId::Lfm3D1Text => Self::Lfm3D1,
        }
    }
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
        let ChoiceScorerSpec::OnnxByteOptionScorer(onnx) = &spec else {
            panic!("expected an ONNX byte scorer, got {spec:?}");
        };
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

    /// The second scorer kind leaves the ONNX scorer's JSON as it was: the
    /// same fields, in the same order, spelled the same.
    #[test]
    fn the_onnx_scorer_serializes_as_before() {
        let spec: ChoiceScorerSpec = serde_json::from_str(&fixture_json()).unwrap();
        assert_eq!(
            serde_json::to_string(&spec).unwrap(),
            concat!(
                r#"{"type":"OnnxByteOptionScorer","model_file":"cua-s1-forms.onnx","#,
                r#""context":{"offset":1,"max_len":224,"overflow":"Truncate","#,
                r#""ids_input":"context_ids","mask_input":"context_mask"},"#,
                r#""choice":{"offset":1,"max_len":96,"overflow":"Truncate","#,
                r#""ids_input":"option_ids","mask_input":"option_token_mask"},"#,
                r#""choice_mask_input":"option_mask","logits_output":"logits","max_choices":64,"#,
                r#""fixed_choices":[{"id":"check","text":"check"},{"id":"click","text":"click"},"#,
                r#"{"id":"skip","text":"skip"}]}"#
            )
        );
    }

    const D1_SPEC: &str =
        r#"{"type":"GgufDecision","model_file":"d1-2B-Q4_K_M.gguf","protocol":"lfm3-d1"}"#;

    fn d1_spec_with(field: &str, value: serde_json::Value) -> serde_json::Value {
        let mut spec: serde_json::Value = serde_json::from_str(D1_SPEC).unwrap();
        spec[field] = value;
        spec
    }

    #[test]
    fn a_gguf_decision_spec_defaults_its_limits_and_writes_them_out() {
        let spec: ChoiceScorerSpec = serde_json::from_str(D1_SPEC).unwrap();
        assert_eq!(
            spec,
            ChoiceScorerSpec::GgufDecision(GgufDecisionSpec {
                model_file: "d1-2B-Q4_K_M.gguf".to_string(),
                protocol: DecisionProtocol::Lfm3D1,
                token_budget: 4096,
                flash_attn: true,
                max_questions: 16,
                max_options: 255,
                max_total_tokens: 16_384,
            })
        );
        assert_eq!(
            spec,
            ChoiceScorerSpec::GgufDecision(GgufDecisionSpec::new(
                "d1-2B-Q4_K_M.gguf",
                DecisionProtocol::Lfm3D1
            ))
        );
        assert!(spec.fixed_choices().is_empty());

        // Every field is written, so a bundle states the limits it runs at.
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(
            json,
            concat!(
                r#"{"type":"GgufDecision","model_file":"d1-2B-Q4_K_M.gguf","protocol":"lfm3-d1","#,
                r#""token_budget":4096,"flash_attn":true,"max_questions":16,"max_options":255,"#,
                r#""max_total_tokens":16384}"#
            )
        );
        assert_eq!(
            serde_json::from_str::<ChoiceScorerSpec>(&json).unwrap(),
            spec
        );
    }

    /// The GGUF defines the model's behaviour; a spec that tries to restate
    /// it, or names a protocol this build does not implement, does not parse.
    #[test]
    fn a_gguf_decision_spec_cannot_restate_the_gguf() {
        for field in [
            "chat_template",
            "temperature",
            "scoring_temperature",
            "attention_causal",
            "fixed_choices",
            "context_length",
            "generation_params",
        ] {
            assert!(
                serde_json::from_value::<ChoiceScorerSpec>(d1_spec_with(field, 1.into())).is_err(),
                "{field} is not a GgufDecision field"
            );
        }
        for protocol in ["lfm3-d1-omni", "LFM3-D1", "lfm3-d1/text/v1", "lfm2-d1", ""] {
            assert!(
                serde_json::from_value::<ChoiceScorerSpec>(d1_spec_with(
                    "protocol",
                    protocol.into()
                ))
                .is_err(),
                "protocol {protocol:?}"
            );
        }
        for required in ["model_file", "protocol"] {
            let mut spec: serde_json::Value = serde_json::from_str(D1_SPEC).unwrap();
            spec.as_object_mut().unwrap().remove(required);
            assert!(
                serde_json::from_value::<ChoiceScorerSpec>(spec).is_err(),
                "{required} is required"
            );
        }
        assert!(serde_json::from_value::<ChoiceScorerSpec>(d1_spec_with(
            "token_budget",
            (-1).into()
        ))
        .is_err());
    }

    #[test]
    fn gguf_decision_specs_are_validated() {
        use crate::execution::choice::{validate_scorer_spec, ChoiceError};

        let edited = |edit: &dyn Fn(&mut GgufDecisionSpec)| {
            let mut spec = GgufDecisionSpec::new("d1-2B-Q4_K_M.gguf", DecisionProtocol::Lfm3D1);
            edit(&mut spec);
            validate_scorer_spec(&ChoiceScorerSpec::GgufDecision(spec))
        };

        type Edit = fn(&mut GgufDecisionSpec);
        let valid: &[(&str, Edit)] = &[
            ("the defaults", |_| {}),
            ("a one-token budget", |s| {
                s.token_budget = 1;
                s.max_total_tokens = 1;
            }),
            ("8,192 tokens with flash attention", |s| {
                s.token_budget = 8192
            }),
            ("4,096 tokens without flash attention", |s| {
                s.flash_attn = false
            }),
            ("one question, one option", |s| {
                s.max_questions = 1;
                s.max_options = 1;
            }),
            ("the ceilings", |s| {
                s.token_budget = 8192;
                s.max_options = 255;
                s.max_total_tokens = 131_072;
            }),
            ("a total equal to the budget", |s| s.max_total_tokens = 4096),
        ];
        for (what, edit) in valid {
            edited(edit).unwrap_or_else(|e| panic!("{what}: {e}"));
        }

        let invalid: &[(Edit, &str)] = &[
            (|s| s.model_file.clear(), "model_file is empty"),
            (|s| s.token_budget = 0, "token_budget is 0; at least 1"),
            (
                |s| s.token_budget = 8193,
                "token_budget is 8193 with flash attention on; at most 8192",
            ),
            (
                |s| {
                    s.flash_attn = false;
                    s.token_budget = 4097;
                },
                "token_budget is 4097 with flash attention off; at most 4096",
            ),
            (
                |s| s.max_questions = 0,
                "max_questions is 0; between 1 and 16",
            ),
            (
                |s| s.max_questions = 17,
                "max_questions is 17; between 1 and 16",
            ),
            (|s| s.max_options = 0, "max_options is 0; between 1 and 255"),
            (
                |s| s.max_options = 256,
                "max_options is 256; between 1 and 255",
            ),
            (
                |s| s.max_total_tokens = 0,
                "max_total_tokens is 0; between token_budget (4096) and 131072",
            ),
            (
                |s| s.max_total_tokens = 4095,
                "max_total_tokens is 4095; between token_budget (4096) and 131072",
            ),
            (
                |s| s.max_total_tokens = 131_073,
                "max_total_tokens is 131073; between token_budget (4096) and 131072",
            ),
        ];
        for (edit, reason) in invalid {
            match edited(edit) {
                Err(ChoiceError::InvalidSpec(message)) => assert_eq!(message, *reason),
                other => panic!("{reason}: {other:?}"),
            }
        }
    }

    /// Until the decision runtime is wired, a GgufDecision scorer refuses
    /// every request with a typed capability refusal, and nothing reaches
    /// the ONNX scorer path (a mock ONNX runtime would answer).
    #[test]
    fn a_gguf_decision_scorer_refuses_every_request_for_now() {
        use crate::execution::choice::{validate_choice_request, EntryPoint};
        use crate::execution::template::{ExecutionTemplate, ModelMetadata};
        use crate::execution::TemplateExecutor;
        use crate::ir::{Envelope, EnvelopeKind};
        use crate::runtime_adapter::{AdapterError, ModelRuntime};

        let mut metadata = ModelMetadata::onnx("d1", "1.0", "d1-2B-Q4_K_M.gguf");
        metadata.execution_template = ExecutionTemplate::ChoiceScorer {
            scorer: serde_json::from_str(D1_SPEC).unwrap(),
        };
        let request = Envelope::choice_request(
            Envelope::new(EnvelopeKind::Text("context".to_string())),
            vec![Choice::new("a", "first"), Choice::new("b", "second")],
        );
        let text = Envelope::new(EnvelopeKind::Text("hello".to_string()));

        let refused = |result: Result<Envelope, AdapterError>| match result {
            Err(AdapterError::UnsupportedModelCapability {
                model_id,
                capability,
                hint,
            }) => {
                assert_eq!(model_id, "d1");
                (capability, hint)
            }
            other => panic!("expected a capability refusal, got {other:?}"),
        };
        let (capability, hint) = refused(
            validate_choice_request(&metadata, &request, EntryPoint::Batch).map(|()| text.clone()),
        );
        assert_eq!(capability, "choice requests");
        assert!(
            hint.contains("GgufDecision") && hint.contains("llama.cpp"),
            "{hint}"
        );

        let mut runtimes: std::collections::HashMap<String, Box<dyn ModelRuntime>> =
            std::collections::HashMap::new();
        runtimes.insert(
            "onnx".to_string(),
            Box::new(crate::testing::mocks::MockRuntime::with_text("ran")),
        );
        let mut executor = TemplateExecutor::with_runtimes("", runtimes);
        for input in [&request, &text] {
            refused(executor.execute(&metadata, input, None));
            for entry in [
                EntryPoint::Streaming,
                EntryPoint::Conversation,
                EntryPoint::Pipeline,
            ] {
                refused(validate_choice_request(&metadata, input, entry).map(|()| text.clone()));
            }
        }
    }

    /// The telemetry backend id follows the scorer kind, whatever the hint.
    #[test]
    fn each_scorer_kind_reports_its_runtime() {
        use crate::execution::template::{backend_label_from_template, ExecutionTemplate};

        let gguf = ExecutionTemplate::ChoiceScorer {
            scorer: serde_json::from_str(D1_SPEC).unwrap(),
        };
        for hint in [None, Some("mistralrs"), Some("llamacpp")] {
            assert_eq!(backend_label_from_template(&gguf, hint), Some("llamacpp"));
        }
        let onnx = ExecutionTemplate::ChoiceScorer {
            scorer: serde_json::from_str(&fixture_json()).unwrap(),
        };
        assert_eq!(
            backend_label_from_template(&onnx, Some("llamacpp")),
            Some("ort")
        );
    }

    #[test]
    fn the_protocol_is_the_profile_s_decision_type() {
        assert_eq!(
            DecisionProtocol::from(DecisionProfileId::Lfm3D1Text),
            DecisionProtocol::Lfm3D1
        );
        assert_eq!(DecisionProtocol::Lfm3D1.to_string(), "lfm3-d1");
        assert_eq!(
            serde_json::to_string(&DecisionProtocol::Lfm3D1).unwrap(),
            r#""lfm3-d1""#
        );
    }

    #[cfg(feature = "schema")]
    #[test]
    fn the_metadata_schema_describes_gguf_decision_scorers() {
        let schema = serde_json::to_string(&schemars::schema_for!(
            crate::execution::template::ModelMetadata
        ))
        .unwrap();
        for name in [
            "GgufDecision",
            "lfm3-d1",
            "token_budget",
            "max_total_tokens",
        ] {
            assert!(schema.contains(name), "the schema is missing {name}");
        }
    }
}
