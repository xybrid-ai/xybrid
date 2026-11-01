//! Decision records for Dart (P4 binding spike — not production surface).
//!
//! flutter_rust_bridge has to scan type definitions in this crate, so the
//! facade's decision records are re-declared here and converted with `From`;
//! the facade still owns the one translation to the decision IR. Production
//! decisions would ride `FfiModel::run` (a request envelope in, a typed field
//! on `FfiResult` out) in P10b, not these `decision_spike_*` functions, which
//! exist so a Dart test can round-trip the records through the real library.

use flutter_rust_bridge::frb;
use xybrid_ffi_facade::decision as facade;
use xybrid_ffi_facade::decision::stand_in::{
    JsonKind, JsonNumber, JsonView, OrderedJson, StructuralLimits,
};

/// Text, or a non-string JSON value carried as JSON text that Rust parses
/// with an order-preserving parser.
#[derive(Clone, PartialEq, Eq)]
pub enum FfiDecisionContent {
    Text { value: String },
    Json { json_text: String },
}

/// How a question is put to the model.
#[derive(Clone, PartialEq, Eq)]
pub enum FfiDecisionInstructions {
    Content { value: FfiDecisionContent },
}

/// One choice option. The key may affect inference.
#[derive(Clone, PartialEq, Eq)]
pub struct FfiDecisionOption {
    pub key: String,
    pub description: Option<FfiDecisionContent>,
}

/// One score level; level `i` has key `"i"`.
#[derive(Clone, PartialEq, Eq)]
pub struct FfiDecisionLevel {
    pub description: Option<FfiDecisionContent>,
}

/// The question types.
#[derive(Clone, PartialEq, Eq)]
pub enum FfiDecisionQuestionKind {
    Choice {
        options: Vec<FfiDecisionOption>,
    },
    YesNo {
        yes: Option<FfiDecisionContent>,
        no: Option<FfiDecisionContent>,
    },
    Score {
        levels: Vec<FfiDecisionLevel>,
    },
}

/// One question.
#[derive(Clone, PartialEq, Eq)]
pub struct FfiDecisionQuestion {
    pub id: String,
    pub instructions: Option<FfiDecisionInstructions>,
    pub kind: FfiDecisionQuestionKind,
}

/// A state and the questions asked about it, in order.
#[derive(Clone, PartialEq, Eq)]
pub struct FfiDecisionRequest {
    pub state: Option<FfiDecisionContent>,
    pub questions: Vec<FfiDecisionQuestion>,
}

/// Where a scored key came from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FfiDecisionOrigin {
    Caller,
    Fixed,
}

/// Which statistic a score answer reports.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FfiScoreStatistic {
    Median,
    Expected,
}

/// One key of an answer's distribution.
#[derive(Clone, PartialEq)]
pub struct FfiDecisionScore {
    pub key: String,
    /// `-inf` is valid when another key is finite.
    pub raw_score: f32,
    pub probability: f64,
    pub origin: FfiDecisionOrigin,
}

/// The answer's value.
#[derive(Clone, PartialEq)]
pub enum FfiDecisionValue {
    Choice {
        key: String,
    },
    YesNo {
        p_yes: f64,
    },
    Score {
        value: f64,
        statistic: FfiScoreStatistic,
    },
}

/// One question's answer.
#[derive(Clone, PartialEq)]
pub struct FfiDecisionAnswer {
    pub question_id: String,
    pub distribution: Vec<FfiDecisionScore>,
    pub value: FfiDecisionValue,
    pub confidence: Option<f64>,
}

/// Content-free work counters.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FfiDecisionUsage {
    pub passes: u32,
    pub evaluated_tokens: u64,
}

/// Every answer, in request order.
#[derive(Clone, PartialEq)]
pub struct FfiDecisions {
    pub answers: Vec<FfiDecisionAnswer>,
    pub usage: FfiDecisionUsage,
}

impl From<FfiDecisionContent> for facade::DecisionContent {
    fn from(content: FfiDecisionContent) -> Self {
        match content {
            FfiDecisionContent::Text { value } => Self::Text { text: value },
            FfiDecisionContent::Json { json_text } => Self::Json { json_text },
        }
    }
}

impl From<facade::DecisionContent> for FfiDecisionContent {
    fn from(content: facade::DecisionContent) -> Self {
        match content {
            facade::DecisionContent::Text { text } => Self::Text { value: text },
            facade::DecisionContent::Json { json_text } => Self::Json { json_text },
        }
    }
}

impl From<FfiDecisionRequest> for facade::DecisionRequest {
    fn from(request: FfiDecisionRequest) -> Self {
        Self {
            state: request.state.map(Into::into),
            questions: request
                .questions
                .into_iter()
                .map(|question| facade::DecisionQuestion {
                    id: question.id,
                    instructions: question
                        .instructions
                        .map(|instructions| match instructions {
                            FfiDecisionInstructions::Content { value } => {
                                facade::DecisionInstructions::Content {
                                    content: value.into(),
                                }
                            }
                        }),
                    kind: match question.kind {
                        FfiDecisionQuestionKind::Choice { options } => {
                            facade::DecisionQuestionKind::Choice {
                                options: options
                                    .into_iter()
                                    .map(|option| facade::DecisionOption {
                                        key: option.key,
                                        description: option.description.map(Into::into),
                                    })
                                    .collect(),
                            }
                        }
                        FfiDecisionQuestionKind::YesNo { yes, no } => {
                            facade::DecisionQuestionKind::YesNo {
                                yes: yes.map(Into::into),
                                no: no.map(Into::into),
                            }
                        }
                        FfiDecisionQuestionKind::Score { levels } => {
                            facade::DecisionQuestionKind::Score {
                                levels: levels
                                    .into_iter()
                                    .map(|level| facade::DecisionLevel {
                                        description: level.description.map(Into::into),
                                    })
                                    .collect(),
                            }
                        }
                    },
                })
                .collect(),
        }
    }
}

impl From<facade::DecisionRequest> for FfiDecisionRequest {
    fn from(request: facade::DecisionRequest) -> Self {
        Self {
            state: request.state.map(Into::into),
            questions: request
                .questions
                .into_iter()
                .map(|question| FfiDecisionQuestion {
                    id: question.id,
                    instructions: question
                        .instructions
                        .map(|instructions| match instructions {
                            facade::DecisionInstructions::Content { content } => {
                                FfiDecisionInstructions::Content {
                                    value: content.into(),
                                }
                            }
                        }),
                    kind: match question.kind {
                        facade::DecisionQuestionKind::Choice { options } => {
                            FfiDecisionQuestionKind::Choice {
                                options: options
                                    .into_iter()
                                    .map(|option| FfiDecisionOption {
                                        key: option.key,
                                        description: option.description.map(Into::into),
                                    })
                                    .collect(),
                            }
                        }
                        facade::DecisionQuestionKind::YesNo { yes, no } => {
                            FfiDecisionQuestionKind::YesNo {
                                yes: yes.map(Into::into),
                                no: no.map(Into::into),
                            }
                        }
                        facade::DecisionQuestionKind::Score { levels } => {
                            FfiDecisionQuestionKind::Score {
                                levels: levels
                                    .into_iter()
                                    .map(|level| FfiDecisionLevel {
                                        description: level.description.map(Into::into),
                                    })
                                    .collect(),
                            }
                        }
                    },
                })
                .collect(),
        }
    }
}

impl From<facade::Decisions> for FfiDecisions {
    fn from(decisions: facade::Decisions) -> Self {
        Self {
            answers: decisions
                .answers
                .into_iter()
                .map(|answer| FfiDecisionAnswer {
                    question_id: answer.question_id,
                    distribution: answer
                        .distribution
                        .into_iter()
                        .map(|score| FfiDecisionScore {
                            key: score.key,
                            raw_score: score.raw_score,
                            probability: score.probability,
                            origin: match score.origin {
                                facade::DecisionOrigin::Caller => FfiDecisionOrigin::Caller,
                                facade::DecisionOrigin::Fixed => FfiDecisionOrigin::Fixed,
                            },
                        })
                        .collect(),
                    value: match answer.value {
                        facade::DecisionValue::Choice { key } => FfiDecisionValue::Choice { key },
                        facade::DecisionValue::YesNo { p_yes } => FfiDecisionValue::YesNo { p_yes },
                        facade::DecisionValue::Score { value, statistic } => {
                            FfiDecisionValue::Score {
                                value,
                                statistic: match statistic {
                                    facade::ScoreStatistic::Median => FfiScoreStatistic::Median,
                                    facade::ScoreStatistic::Expected => FfiScoreStatistic::Expected,
                                },
                            }
                        }
                    },
                    confidence: answer.confidence,
                })
                .collect(),
            usage: FfiDecisionUsage {
                passes: decisions.usage.passes,
                evaluated_tokens: decisions.usage.evaluated_tokens,
            },
        }
    }
}

impl From<FfiDecisions> for facade::Decisions {
    fn from(decisions: FfiDecisions) -> Self {
        Self {
            answers: decisions
                .answers
                .into_iter()
                .map(|answer| facade::DecisionAnswer {
                    question_id: answer.question_id,
                    distribution: answer
                        .distribution
                        .into_iter()
                        .map(|score| facade::DecisionScore {
                            key: score.key,
                            raw_score: score.raw_score,
                            probability: score.probability,
                            origin: match score.origin {
                                FfiDecisionOrigin::Caller => facade::DecisionOrigin::Caller,
                                FfiDecisionOrigin::Fixed => facade::DecisionOrigin::Fixed,
                            },
                        })
                        .collect(),
                    value: match answer.value {
                        FfiDecisionValue::Choice { key } => facade::DecisionValue::Choice { key },
                        FfiDecisionValue::YesNo { p_yes } => facade::DecisionValue::YesNo { p_yes },
                        FfiDecisionValue::Score { value, statistic } => {
                            facade::DecisionValue::Score {
                                value,
                                statistic: match statistic {
                                    FfiScoreStatistic::Median => facade::ScoreStatistic::Median,
                                    FfiScoreStatistic::Expected => facade::ScoreStatistic::Expected,
                                },
                            }
                        }
                    },
                    confidence: answer.confidence,
                })
                .collect(),
            usage: facade::DecisionUsage {
                passes: decisions.usage.passes,
                evaluated_tokens: decisions.usage.evaluated_tokens,
            },
        }
    }
}

fn request_into_ir(
    request: FfiDecisionRequest,
) -> Result<facade::stand_in::DecisionRequest, String> {
    facade::DecisionRequest::from(request)
        .into_ir(&StructuralLimits::default())
        .map_err(|error| error.to_string())
}

/// Spike: Dart request -> facade -> IR (JSON parsed, nulls normalized,
/// structure checked) -> facade -> Dart.
#[frb(sync)]
pub fn decision_spike_echo_request(
    request: FfiDecisionRequest,
) -> Result<FfiDecisionRequest, String> {
    Ok(facade::DecisionRequest::from_ir(&request_into_ir(request)?).into())
}

/// Spike: what the Rust IR saw, one line per item in request order.
#[frb(sync)]
pub fn decision_spike_request_outline(request: FfiDecisionRequest) -> Result<Vec<String>, String> {
    Ok(facade::spike::outline(&request_into_ir(request)?))
}

/// Spike: the stub engine's result, through the Dart records.
#[frb(sync)]
pub fn decision_spike_decide(request: FfiDecisionRequest) -> Result<FfiDecisions, String> {
    Ok(facade::spike::decide(&request_into_ir(request)?).into())
}

/// Spike: the bit patterns of every float [`decision_spike_decide`] returns.
#[frb(sync)]
pub fn decision_spike_decide_bits(request: FfiDecisionRequest) -> Result<Vec<String>, String> {
    Ok(facade::spike::bits(&facade::spike::decide(
        &request_into_ir(request)?,
    )))
}

/// Spike: compare a result Dart re-encoded with the stub engine's own, bit
/// for bit. `None` means identical.
#[frb(sync)]
pub fn decision_spike_verify_decisions(
    request: FfiDecisionRequest,
    decisions: FfiDecisions,
) -> Result<Option<String>, String> {
    facade::spike::verify(&request_into_ir(request)?, decisions.into())
        .map_err(|error| error.to_string())
}

/// Probe: `Vec<Option<data enum>>` and `Vec<Option<String>>`.
#[derive(Clone, PartialEq, Eq)]
pub struct FfiProbeOptionalList {
    pub levels: Vec<Option<FfiDecisionContent>>,
    pub labels: Vec<Option<String>>,
}

/// Probe: nested lists and an optional record.
#[derive(Clone, PartialEq)]
pub struct FfiProbeNesting {
    pub raw_scores: Vec<Vec<f32>>,
    pub probabilities: Vec<Vec<f64>>,
    pub usage: Option<FfiDecisionUsage>,
}

/// Probe: a recursive JSON tree, the alternative to carrying JSON as text.
#[derive(Clone, PartialEq)]
pub enum FfiProbeJson {
    Null,
    Bool { value: bool },
    Int { value: i64 },
    Uint { value: u64 },
    Float { value: f64 },
    Text { value: String },
    Seq { items: Vec<FfiProbeJson> },
    Obj { entries: Vec<FfiProbeJsonEntry> },
}

/// One entry of an [`FfiProbeJson::Obj`], in document order.
#[derive(Clone, PartialEq)]
pub struct FfiProbeJsonEntry {
    pub key: String,
    pub value: FfiProbeJson,
}

/// Probe echo for [`FfiProbeOptionalList`].
#[frb(sync)]
pub fn decision_spike_probe_optional_list(probe: FfiProbeOptionalList) -> FfiProbeOptionalList {
    probe
}

/// Probe echo for [`FfiProbeNesting`]; floats come back bit for bit.
#[frb(sync)]
pub fn decision_spike_probe_nesting(probe: FfiProbeNesting) -> FfiProbeNesting {
    probe
}

/// Probe: parse JSON text with the order-preserving parser into a tree.
#[frb(sync)]
pub fn decision_spike_probe_json(json_text: String) -> Result<FfiProbeJson, String> {
    fn tree(view: JsonView<'_>) -> FfiProbeJson {
        match view.kind() {
            JsonKind::Null => FfiProbeJson::Null,
            JsonKind::Bool => FfiProbeJson::Bool {
                value: view.as_bool().expect("bool node"),
            },
            JsonKind::Number => match view.as_number().expect("number node") {
                JsonNumber::I64(value) => FfiProbeJson::Int { value },
                JsonNumber::U64(value) => FfiProbeJson::Uint { value },
                JsonNumber::F64(value) => FfiProbeJson::Float { value },
            },
            JsonKind::String => FfiProbeJson::Text {
                value: view.as_str().expect("string node").to_string(),
            },
            JsonKind::Array => FfiProbeJson::Seq {
                items: view
                    .items()
                    .expect("array node")
                    .into_iter()
                    .map(tree)
                    .collect(),
            },
            JsonKind::Object => FfiProbeJson::Obj {
                entries: view
                    .entries()
                    .expect("object node")
                    .into_iter()
                    .map(|(key, value)| FfiProbeJsonEntry {
                        key: key.to_string(),
                        value: tree(value),
                    })
                    .collect(),
            },
        }
    }
    let json = OrderedJson::parse(&json_text, &StructuralLimits::default())
        .map_err(|e| format!("invalid JSON at byte {}: {}", e.offset, e.reason))?;
    Ok(tree(json.view()))
}

/// Probe echo for [`FfiProbeJson`].
#[frb(sync)]
pub fn decision_spike_probe_json_echo(value: FfiProbeJson) -> FfiProbeJson {
    value
}
