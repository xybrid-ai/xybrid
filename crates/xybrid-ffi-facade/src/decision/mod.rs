//! Decision requests and results across the FFI boundary (P4 feasibility
//! spike — not production code).
//!
//! The facade owns the one translation between the decision IR and the shapes
//! the binding generators can describe. Bolt and Flutter re-declare these
//! records and forward through [`DecisionRequest::into_ir`] and
//! [`Decisions::from_ir`]; neither touches the IR directly.
//!
//! Representation (contract 4, framework v2 §5):
//!
//! - questions, options and score levels are ordered arrays of records, never
//!   a foreign dictionary: no binding preserves dictionary key order (a
//!   JavaScript object reorders integer-like keys, `NSDictionary` and Java's
//!   `HashMap` have no order at all);
//! - [`DecisionContent`] is `Text { text }` or `Json { json_text }`, and the
//!   JSON text is parsed here, in Rust, by the order-preserving
//!   [`stand_in::OrderedJson::parse`]. Hosts never parse it, so their JSON
//!   libraries cannot reorder keys or turn `1.0` into `1`;
//! - a JSON `null` or JSON string arriving as `json_text` normalizes to
//!   `None` or text (v2.1 C3), so the IR never holds `Some(Json(null))`;
//! - results are records: `f64` probabilities, `f32` raw scores that may be
//!   `-inf` (v2.1 C4), answers in request order.
//!
//! `xybrid_core::ir::decision` does not exist yet, so [`stand_in`] mirrors
//! contract 1 locally. Swapping it for the real IR changes the `use` below,
//! not the conversions.
//!
//! Every type's `Debug` prints counts, lengths and kinds only, and every
//! error names fields and indexes, never ids, keys, text or JSON.

pub mod stand_in;

use std::fmt;

use self::stand_in as ir;
use crate::{Error, Result};

// ============================================================================
// Request
// ============================================================================

/// Text, or a non-string JSON value carried as JSON text.
#[derive(Clone, PartialEq, Eq)]
pub enum DecisionContent {
    Text {
        text: String,
    },
    /// Parsed by [`stand_in::OrderedJson::parse`]: object order and integer
    /// vs float spelling survive. `"null"` is absence and a JSON string is
    /// text once converted.
    Json {
        json_text: String,
    },
}

/// How a question is put to the model. One variant in v1; spoken
/// instructions (O2) append a variant, which keeps old payloads decodable.
#[derive(Clone, PartialEq, Eq)]
pub enum DecisionInstructions {
    Content { content: DecisionContent },
}

/// One choice option. The key may affect inference (d1 label codes).
#[derive(Clone, PartialEq, Eq)]
pub struct DecisionOption {
    pub key: String,
    pub description: Option<DecisionContent>,
}

/// One score level. Level `i` has key `"i"`; a record rather than a bare
/// `Option` so a level can gain fields without a new list type.
#[derive(Clone, PartialEq, Eq)]
pub struct DecisionLevel {
    pub description: Option<DecisionContent>,
}

/// The question types.
#[derive(Clone, PartialEq, Eq)]
pub enum DecisionQuestionKind {
    Choice {
        options: Vec<DecisionOption>,
    },
    YesNo {
        yes: Option<DecisionContent>,
        no: Option<DecisionContent>,
    },
    Score {
        levels: Vec<DecisionLevel>,
    },
}

/// One question, identified by a caller-chosen id unique in the request.
#[derive(Clone, PartialEq, Eq)]
pub struct DecisionQuestion {
    pub id: String,
    pub instructions: Option<DecisionInstructions>,
    pub kind: DecisionQuestionKind,
}

/// A state and the questions asked about it, in order.
#[derive(Clone, PartialEq, Eq)]
pub struct DecisionRequest {
    /// `None` is a null state; `Text { "" }` is an empty one.
    pub state: Option<DecisionContent>,
    pub questions: Vec<DecisionQuestion>,
}

// ============================================================================
// Result
// ============================================================================

/// Where a scored key came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionOrigin {
    Caller,
    Fixed,
}

/// Which statistic a score answer reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreStatistic {
    Median,
    Expected,
}

/// One key of an answer's distribution.
#[derive(Clone, PartialEq)]
pub struct DecisionScore {
    /// Option key, `"0"`..`"n-1"` for a score, `"true"`/`"false"` for yes/no.
    pub key: String,
    /// `-inf` is valid when another key is finite.
    pub raw_score: f32,
    pub probability: f64,
    pub origin: DecisionOrigin,
}

/// The answer's value.
#[derive(Clone, PartialEq)]
pub enum DecisionValue {
    Choice {
        key: String,
    },
    YesNo {
        p_yes: f64,
    },
    Score {
        value: f64,
        statistic: ScoreStatistic,
    },
}

/// One question's answer.
#[derive(Clone, PartialEq)]
pub struct DecisionAnswer {
    pub question_id: String,
    /// Choice: option order; score: level order; yes/no: `true`, `false`.
    pub distribution: Vec<DecisionScore>,
    pub value: DecisionValue,
    /// Protocol-defined; d1 reports none for yes/no.
    pub confidence: Option<f64>,
}

/// Content-free work counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionUsage {
    pub passes: u32,
    pub evaluated_tokens: u64,
}

/// Every answer, in request order.
#[derive(Clone, PartialEq)]
pub struct Decisions {
    pub answers: Vec<DecisionAnswer>,
    pub usage: DecisionUsage,
}

// ============================================================================
// FFI -> IR
// ============================================================================

/// An invalid request is a request error, as `SdkError::InferenceError` is
/// for choice requests. The message is content-free (paths and indexes).
fn invalid_request(error: ir::DecisionError) -> Error {
    Error::InferenceError {
        message: error.to_string(),
    }
}

fn content_into_ir(
    content: Option<DecisionContent>,
    field: impl FnOnce() -> String,
    limits: &ir::StructuralLimits,
) -> Result<Option<ir::DecisionContent>> {
    match content {
        None => Ok(None),
        Some(DecisionContent::Text { text }) => Ok(Some(ir::DecisionContent::text(text))),
        Some(DecisionContent::Json { json_text }) => {
            let json = ir::OrderedJson::parse(&json_text, limits).map_err(|e| {
                invalid_request(ir::DecisionError::InvalidJson {
                    field: field(),
                    offset: e.offset,
                    reason: e.reason,
                })
            })?;
            Ok(ir::DecisionContent::from_json(json))
        }
    }
}

impl DecisionRequest {
    /// Convert to the IR: parse every JSON text in order, normalize JSON
    /// `null`/string content, then run the structural checks.
    ///
    /// # Errors
    /// [`Error::InferenceError`] naming the field path and byte offset of
    /// invalid JSON, or the first structural violation.
    pub fn into_ir(self, limits: &ir::StructuralLimits) -> Result<ir::DecisionRequest> {
        let state = content_into_ir(self.state, || "state".into(), limits)?;
        let questions = self
            .questions
            .into_iter()
            .enumerate()
            .map(|(q, question)| {
                let instructions = match question.instructions {
                    None => None,
                    Some(DecisionInstructions::Content { content }) => content_into_ir(
                        Some(content),
                        || format!("questions[{q}].instructions"),
                        limits,
                    )?
                    .map(ir::DecisionInstructions::Content),
                };
                let kind = match question.kind {
                    DecisionQuestionKind::Choice { options } => ir::DecisionQuestionKind::Choice {
                        options: options
                            .into_iter()
                            .enumerate()
                            .map(|(o, option)| {
                                let description = content_into_ir(
                                    option.description,
                                    || format!("questions[{q}].options[{o}].description"),
                                    limits,
                                )?;
                                Ok(ir::DecisionOption::new(option.key, description))
                            })
                            .collect::<Result<_>>()?,
                    },
                    DecisionQuestionKind::YesNo { yes, no } => ir::DecisionQuestionKind::YesNo {
                        yes: content_into_ir(yes, || format!("questions[{q}].yes"), limits)?,
                        no: content_into_ir(no, || format!("questions[{q}].no"), limits)?,
                    },
                    DecisionQuestionKind::Score { levels } => ir::DecisionQuestionKind::Score {
                        levels: levels
                            .into_iter()
                            .enumerate()
                            .map(|(l, level)| {
                                content_into_ir(
                                    level.description,
                                    || format!("questions[{q}].levels[{l}].description"),
                                    limits,
                                )
                            })
                            .collect::<Result<_>>()?,
                    },
                };
                Ok(ir::DecisionQuestion::new(question.id, instructions, kind))
            })
            .collect::<Result<Vec<_>>>()?;
        let request = ir::DecisionRequest::new(ir::DecisionState::new(state), questions);
        ir::validate_structure(&request, limits).map_err(invalid_request)?;
        Ok(request)
    }
}

// ============================================================================
// IR -> FFI
// ============================================================================

fn content_from_ir(content: Option<&ir::DecisionContent>) -> Option<DecisionContent> {
    content.map(|content| match content.view() {
        ir::DecisionContentView::Text(text) => DecisionContent::Text {
            text: text.to_string(),
        },
        ir::DecisionContentView::Json(json) => DecisionContent::Json {
            json_text: json.to_json_text(),
        },
    })
}

impl DecisionRequest {
    /// Render an IR request back into FFI form; JSON content becomes compact
    /// JSON text in the same order. Used by echo tests, not by a run.
    pub fn from_ir(request: &ir::DecisionRequest) -> Self {
        let questions = request
            .questions()
            .iter()
            .map(|question| DecisionQuestion {
                id: question.id().to_string(),
                instructions: question
                    .instructions()
                    .map(|instructions| match instructions {
                        ir::DecisionInstructions::Content(content) => {
                            DecisionInstructions::Content {
                                content: content_from_ir(Some(content))
                                    .expect("Some content converts to Some"),
                            }
                        }
                    }),
                kind: match question.kind() {
                    ir::DecisionQuestionKind::Choice { options } => DecisionQuestionKind::Choice {
                        options: options
                            .iter()
                            .map(|option| DecisionOption {
                                key: option.key().to_string(),
                                description: content_from_ir(option.description()),
                            })
                            .collect(),
                    },
                    ir::DecisionQuestionKind::YesNo { yes, no } => DecisionQuestionKind::YesNo {
                        yes: content_from_ir(yes.as_ref()),
                        no: content_from_ir(no.as_ref()),
                    },
                    ir::DecisionQuestionKind::Score { levels } => DecisionQuestionKind::Score {
                        levels: levels
                            .iter()
                            .map(|level| DecisionLevel {
                                description: content_from_ir(level.as_ref()),
                            })
                            .collect(),
                    },
                },
            })
            .collect();
        Self {
            state: content_from_ir(request.state().body()),
            questions,
        }
    }
}

impl DecisionOrigin {
    fn from_ir(origin: ir::DecisionOrigin) -> Self {
        match origin {
            ir::DecisionOrigin::Caller => Self::Caller,
            ir::DecisionOrigin::Fixed => Self::Fixed,
        }
    }

    fn into_ir(self) -> ir::DecisionOrigin {
        match self {
            Self::Caller => ir::DecisionOrigin::Caller,
            Self::Fixed => ir::DecisionOrigin::Fixed,
        }
    }
}

impl ScoreStatistic {
    fn from_ir(statistic: ir::ScoreStatistic) -> Self {
        match statistic {
            ir::ScoreStatistic::Median => Self::Median,
            ir::ScoreStatistic::Expected => Self::Expected,
        }
    }

    fn into_ir(self) -> ir::ScoreStatistic {
        match self {
            Self::Median => ir::ScoreStatistic::Median,
            Self::Expected => ir::ScoreStatistic::Expected,
        }
    }
}

impl Decisions {
    /// Copy an IR result into FFI form. Lossless: every `f64`/`f32` is moved
    /// bit for bit and the answer and distribution order is kept.
    pub fn from_ir(decisions: &ir::Decisions) -> Self {
        Self {
            answers: decisions
                .answers()
                .iter()
                .map(|answer| DecisionAnswer {
                    question_id: answer.question_id().to_string(),
                    distribution: answer
                        .distribution()
                        .iter()
                        .map(|score| DecisionScore {
                            key: score.key().to_string(),
                            raw_score: score.raw_score(),
                            probability: score.probability(),
                            origin: DecisionOrigin::from_ir(score.origin()),
                        })
                        .collect(),
                    value: match answer.value() {
                        ir::DecisionValue::Choice { key } => {
                            DecisionValue::Choice { key: key.clone() }
                        }
                        ir::DecisionValue::YesNo { p_yes } => {
                            DecisionValue::YesNo { p_yes: *p_yes }
                        }
                        ir::DecisionValue::Score { value, statistic } => DecisionValue::Score {
                            value: *value,
                            statistic: ScoreStatistic::from_ir(*statistic),
                        },
                    },
                    confidence: answer.confidence(),
                })
                .collect(),
            usage: DecisionUsage {
                passes: decisions.usage().passes(),
                evaluated_tokens: decisions.usage().evaluated_tokens(),
            },
        }
    }

    /// Rebuild an IR result through its invariant-checking constructor.
    /// Results only flow out of Rust in production; this exists so a foreign
    /// round trip can be checked on the Rust side.
    ///
    /// # Errors
    /// [`Error::InferenceError`] naming the first broken answer invariant.
    pub fn into_ir(self) -> Result<ir::Decisions> {
        let answers = self
            .answers
            .into_iter()
            .enumerate()
            .map(|(index, answer)| {
                ir::DecisionAnswer::new(
                    index,
                    answer.question_id,
                    answer
                        .distribution
                        .into_iter()
                        .map(|s| {
                            ir::DecisionScore::new(
                                s.key,
                                s.raw_score,
                                s.probability,
                                s.origin.into_ir(),
                            )
                        })
                        .collect(),
                    match answer.value {
                        DecisionValue::Choice { key } => ir::DecisionValue::Choice { key },
                        DecisionValue::YesNo { p_yes } => ir::DecisionValue::YesNo { p_yes },
                        DecisionValue::Score { value, statistic } => ir::DecisionValue::Score {
                            value,
                            statistic: statistic.into_ir(),
                        },
                    },
                    answer.confidence,
                )
                .map_err(invalid_request)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ir::Decisions::new(
            answers,
            ir::DecisionUsage::new(self.usage.passes, self.usage.evaluated_tokens),
        ))
    }
}

/// Bit-exact equality of two results: `f64`/`f32` compared by bits, so
/// `-0.0` vs `0.0` and NaN payloads count as differences. Returns the path of
/// the first difference (content-free).
pub fn first_bit_difference(left: &Decisions, right: &Decisions) -> Option<String> {
    if left.usage != right.usage {
        return Some("usage".into());
    }
    if left.answers.len() != right.answers.len() {
        return Some("answers.len".into());
    }
    for (a, (l, r)) in left.answers.iter().zip(&right.answers).enumerate() {
        if l.question_id != r.question_id {
            return Some(format!("answers[{a}].question_id"));
        }
        if l.confidence.map(f64::to_bits) != r.confidence.map(f64::to_bits) {
            return Some(format!("answers[{a}].confidence"));
        }
        let same_value = match (&l.value, &r.value) {
            (DecisionValue::Choice { key: lk }, DecisionValue::Choice { key: rk }) => lk == rk,
            (DecisionValue::YesNo { p_yes: lp }, DecisionValue::YesNo { p_yes: rp }) => {
                lp.to_bits() == rp.to_bits()
            }
            (
                DecisionValue::Score {
                    value: lv,
                    statistic: ls,
                },
                DecisionValue::Score {
                    value: rv,
                    statistic: rs,
                },
            ) => lv.to_bits() == rv.to_bits() && ls == rs,
            _ => false,
        };
        if !same_value {
            return Some(format!("answers[{a}].value"));
        }
        if l.distribution.len() != r.distribution.len() {
            return Some(format!("answers[{a}].distribution.len"));
        }
        for (s, (ls, rs)) in l.distribution.iter().zip(&r.distribution).enumerate() {
            if ls.key != rs.key
                || ls.raw_score.to_bits() != rs.raw_score.to_bits()
                || ls.probability.to_bits() != rs.probability.to_bits()
                || ls.origin != rs.origin
            {
                return Some(format!("answers[{a}].distribution[{s}]"));
            }
        }
    }
    None
}

// ============================================================================
// Redacted Debug
// ============================================================================

impl fmt::Debug for DecisionContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text { text } => f.debug_struct("Text").field("bytes", &text.len()).finish(),
            Self::Json { json_text } => f
                .debug_struct("Json")
                .field("bytes", &json_text.len())
                .finish(),
        }
    }
}

impl fmt::Debug for DecisionInstructions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Content { content } => f.debug_tuple("Content").field(content).finish(),
        }
    }
}

impl fmt::Debug for DecisionOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionOption")
            .field("key_bytes", &self.key.len())
            .field("description", &self.description)
            .finish()
    }
}

impl fmt::Debug for DecisionLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionLevel")
            .field("description", &self.description)
            .finish()
    }
}

impl fmt::Debug for DecisionQuestionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Choice { options } => f
                .debug_struct("Choice")
                .field("options", &options.len())
                .finish(),
            Self::YesNo { yes, no } => f
                .debug_struct("YesNo")
                .field("yes", yes)
                .field("no", no)
                .finish(),
            Self::Score { levels } => f
                .debug_struct("Score")
                .field("levels", &levels.len())
                .finish(),
        }
    }
}

impl fmt::Debug for DecisionQuestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionQuestion")
            .field("id_bytes", &self.id.len())
            .field("instructions", &self.instructions)
            .field("kind", &self.kind)
            .finish()
    }
}

impl fmt::Debug for DecisionRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionRequest")
            .field("state", &self.state)
            .field("questions", &self.questions)
            .finish()
    }
}

impl fmt::Debug for DecisionScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionScore")
            .field("key_bytes", &self.key.len())
            .field("raw_score", &self.raw_score)
            .field("probability", &self.probability)
            .field("origin", &self.origin)
            .finish()
    }
}

impl fmt::Debug for DecisionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Choice { key } => f
                .debug_struct("Choice")
                .field("key_bytes", &key.len())
                .finish(),
            Self::YesNo { p_yes } => f.debug_struct("YesNo").field("p_yes", p_yes).finish(),
            Self::Score { value, statistic } => f
                .debug_struct("Score")
                .field("value", value)
                .field("statistic", statistic)
                .finish(),
        }
    }
}

impl fmt::Debug for DecisionAnswer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionAnswer")
            .field("question_id_bytes", &self.question_id.len())
            .field("distribution", &self.distribution.len())
            .field("value", &self.value)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl fmt::Debug for Decisions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Decisions")
            .field("answers", &self.answers)
            .field("usage", &self.usage)
            .finish()
    }
}

#[cfg(test)]
mod tests;
