//! Ordered, lossless foreign decision data and fallible conversion to the shared IR.
//!
//! JSON content crosses language boundaries as text. Never parse it through a
//! foreign dictionary: doing so can reorder keys or round large integers.

use crate::{unrepresentable, Error, Result};
use xybrid_core::execution::decision::liquid::DecisionProtocolError as Refusal;
use xybrid_core::ir;

/// Text or raw ordered JSON. JSON null belongs in the containing optional field.
#[derive(Clone, PartialEq)]
pub enum DecisionContent {
    Text { text: String },
    Json { json_text: String },
}

/// One option, in caller order.
#[derive(Clone, PartialEq)]
pub struct DecisionOption {
    pub key: String,
    pub description: Option<DecisionContent>,
}

/// One ordinal level, in ascending order.
#[derive(Clone, PartialEq)]
pub struct DecisionLevel {
    pub description: Option<DecisionContent>,
}

/// The supported question kinds. New variants are appended.
#[derive(Clone, PartialEq)]
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

/// A named question with optional structured instructions.
#[derive(Clone, PartialEq)]
pub struct DecisionQuestion {
    pub id: String,
    pub instructions: Option<DecisionContent>,
    pub kind: DecisionQuestionKind,
}

/// A text/JSON decision request. An absent state is explicit null.
#[derive(Clone, PartialEq)]
pub struct DecisionRequest {
    pub state: Option<DecisionContent>,
    pub questions: Vec<DecisionQuestion>,
}

/// Provenance of one distribution entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionOrigin {
    Caller,
    Fixed,
}

/// The statistic represented by an ordinal score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreStatistic {
    Median,
    Expected,
}

/// One ordered distribution entry. Probabilities retain all 64 bits.
#[derive(Clone, PartialEq)]
pub struct DecisionScore {
    pub key: String,
    pub raw_score: f32,
    pub probability: f64,
    pub origin: DecisionOrigin,
}

/// A typed answer. New variants are appended.
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

/// The answer to one question, with the distribution used to derive it.
#[derive(Clone, PartialEq)]
pub struct DecisionAnswer {
    pub question_id: String,
    pub distribution: Vec<DecisionScore>,
    pub value: DecisionValue,
    pub confidence: Option<f64>,
}

/// Atomic answers in request order, with total evaluation usage.
#[derive(Clone, PartialEq)]
pub struct Decisions {
    pub answers: Vec<DecisionAnswer>,
    pub passes: u32,
    pub input_tokens: u64,
}

// Foreign data is often inspected while debugging. Keep every request and
// decision (including logits and probabilities) out of diagnostic formatting.
macro_rules! redacted_debug {
    ($($ty:ty),+ $(,)?) => {$(
        impl std::fmt::Debug for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($ty)).finish_non_exhaustive()
            }
        }
    )+};
}
redacted_debug!(
    DecisionContent,
    DecisionOption,
    DecisionLevel,
    DecisionQuestionKind,
    DecisionQuestion,
    DecisionRequest,
    DecisionScore,
    DecisionValue,
    DecisionAnswer,
    Decisions
);

/// A refusal naming its stable category (`duplicate_key`, `too_many_options`,
/// ...), so a binding can tell failures apart. Never request content.
fn invalid(refusal: impl Into<Refusal>) -> Error {
    Error::ConfigError {
        message: format!("invalid decision data: {}", refusal.into().category()),
    }
}

impl DecisionContent {
    fn into_ir(self) -> Result<ir::DecisionContent> {
        match self {
            Self::Text { text } => Ok(ir::DecisionContent::text(text)),
            Self::Json { json_text } => {
                let json = ir::OrderedJson::parse(&json_text).map_err(invalid)?;
                // JSON null belongs in the containing optional field.
                ir::DecisionContent::from_json(json).ok_or_else(|| invalid(Refusal::InvalidRequest))
            }
        }
    }

    fn from_ir(content: &ir::DecisionContent) -> Self {
        match content.view() {
            ir::DecisionContentView::Text(text) => Self::Text {
                text: text.to_owned(),
            },
            ir::DecisionContentView::Json(json) => Self::Json {
                json_text: json.to_json_text(),
            },
        }
    }
}

fn content_into_ir(content: Option<DecisionContent>) -> Result<Option<ir::DecisionContent>> {
    content.map(DecisionContent::into_ir).transpose()
}

impl DecisionRequest {
    /// Convert and validate before entering inference. Errors never quote content.
    pub fn into_sdk(self) -> Result<ir::DecisionRequest> {
        if self.questions.len() > 16 {
            return Err(invalid(Refusal::BudgetExceeded));
        }
        let questions = self
            .questions
            .into_iter()
            .map(|question| {
                let kind = match question.kind {
                    DecisionQuestionKind::Choice { options } => {
                        if options.len() > 255 {
                            return Err(invalid(Refusal::TooManyOptions));
                        }
                        ir::DecisionQuestionKind::choice(
                            options
                                .into_iter()
                                .map(|option| {
                                    Ok(ir::DecisionOption::new(
                                        option.key,
                                        content_into_ir(option.description)?,
                                    ))
                                })
                                .collect::<Result<Vec<_>>>()?,
                        )
                    }
                    DecisionQuestionKind::YesNo { yes, no } => ir::DecisionQuestionKind::yes_no(
                        content_into_ir(yes)?,
                        content_into_ir(no)?,
                    ),
                    DecisionQuestionKind::Score { levels } => {
                        if levels.len() > 255 {
                            return Err(invalid(Refusal::LevelCount));
                        }
                        ir::DecisionQuestionKind::score(
                            levels
                                .into_iter()
                                .map(|level| {
                                    Ok(ir::DecisionLevel::new(content_into_ir(level.description)?))
                                })
                                .collect::<Result<Vec<_>>>()?,
                        )
                    }
                };
                Ok(ir::DecisionQuestion::new(
                    question.id,
                    content_into_ir(question.instructions)?.map(Into::into),
                    kind,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let request = ir::DecisionRequest::new(
            ir::DecisionState::new(content_into_ir(self.state)?),
            questions,
        );
        ir::decision::validate_structure(&request, &ir::decision::StructuralLimits::default())
            .map_err(invalid)?;
        Ok(request)
    }

    /// Preserve ordered JSON and refuse future variants this binding cannot carry.
    pub fn try_from_sdk(request: &ir::DecisionRequest) -> Result<Self> {
        if !request.state().media().is_empty() {
            return Err(unrepresentable("decision media"));
        }
        let questions = request
            .questions()
            .iter()
            .map(|question| {
                let instructions = question
                    .instructions()
                    .map(|instructions| {
                        instructions
                            .as_content()
                            .map(DecisionContent::from_ir)
                            .ok_or_else(|| unrepresentable("decision instructions"))
                    })
                    .transpose()?;
                let kind = match question.kind() {
                    ir::DecisionQuestionKind::Choice { options, .. } => {
                        DecisionQuestionKind::Choice {
                            options: options
                                .iter()
                                .map(|option| DecisionOption {
                                    key: option.key().to_owned(),
                                    description: option.description().map(DecisionContent::from_ir),
                                })
                                .collect(),
                        }
                    }
                    ir::DecisionQuestionKind::YesNo { yes, no, .. } => {
                        DecisionQuestionKind::YesNo {
                            yes: yes.as_ref().map(DecisionContent::from_ir),
                            no: no.as_ref().map(DecisionContent::from_ir),
                        }
                    }
                    ir::DecisionQuestionKind::Score { levels, .. } => DecisionQuestionKind::Score {
                        levels: levels
                            .iter()
                            .map(|level| DecisionLevel {
                                description: level.description().map(DecisionContent::from_ir),
                            })
                            .collect(),
                    },
                    _ => return Err(unrepresentable("decision question kind")),
                };
                Ok(DecisionQuestion {
                    id: question.id().to_owned(),
                    instructions,
                    kind,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            state: request.state().body().map(DecisionContent::from_ir),
            questions,
        })
    }
}

impl Decisions {
    /// Convert a validated core result without rounding probabilities.
    pub fn try_from_sdk(decisions: &ir::Decisions) -> Result<Self> {
        let answers = decisions
            .answers()
            .iter()
            .map(|answer| {
                let value = match answer.value() {
                    ir::DecisionValue::Choice { key, .. } => {
                        DecisionValue::Choice { key: key.clone() }
                    }
                    ir::DecisionValue::YesNo { p_yes, .. } => {
                        DecisionValue::YesNo { p_yes: *p_yes }
                    }
                    ir::DecisionValue::Score {
                        value, statistic, ..
                    } => DecisionValue::Score {
                        value: *value,
                        statistic: match statistic {
                            ir::ScoreStatistic::Median => ScoreStatistic::Median,
                            ir::ScoreStatistic::Expected => ScoreStatistic::Expected,
                            _ => return Err(unrepresentable("decision score statistic")),
                        },
                    },
                    _ => return Err(unrepresentable("decision value")),
                };
                let distribution = answer
                    .distribution()
                    .iter()
                    .map(|score| {
                        Ok(DecisionScore {
                            key: score.key().to_owned(),
                            raw_score: score.raw_score(),
                            probability: score.probability(),
                            origin: match score.origin() {
                                ir::DecisionOrigin::Caller => DecisionOrigin::Caller,
                                ir::DecisionOrigin::Fixed => DecisionOrigin::Fixed,
                                _ => return Err(unrepresentable("decision origin")),
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(DecisionAnswer {
                    question_id: answer.question_id().to_owned(),
                    distribution,
                    value,
                    confidence: answer.confidence(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            answers,
            passes: decisions.usage().passes(),
            input_tokens: decisions.usage().evaluated_tokens(),
        })
    }

    /// Validate foreign output data when reconstructing an envelope.
    pub fn into_sdk(self) -> Result<ir::Decisions> {
        if self.answers.len() > 16 {
            return Err(invalid(Refusal::InvalidOutput));
        }
        let answers = self
            .answers
            .into_iter()
            .map(|answer| {
                if answer.distribution.len() > 255 {
                    return Err(invalid(Refusal::InvalidOutput));
                }
                let value = match answer.value {
                    DecisionValue::Choice { key } => ir::DecisionValue::choice(key),
                    DecisionValue::YesNo { p_yes } => ir::DecisionValue::yes_no(p_yes),
                    DecisionValue::Score { value, statistic } => ir::DecisionValue::score(
                        value,
                        match statistic {
                            ScoreStatistic::Median => ir::ScoreStatistic::Median,
                            ScoreStatistic::Expected => ir::ScoreStatistic::Expected,
                        },
                    ),
                };
                let distribution = answer
                    .distribution
                    .into_iter()
                    .map(|score| {
                        ir::DecisionScore::new(
                            score.key,
                            score.raw_score,
                            score.probability,
                            match score.origin {
                                DecisionOrigin::Caller => ir::DecisionOrigin::Caller,
                                DecisionOrigin::Fixed => ir::DecisionOrigin::Fixed,
                            },
                        )
                    })
                    .collect();
                ir::DecisionAnswer::new(answer.question_id, distribution, value, answer.confidence)
                    .map_err(invalid)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ir::Decisions::new(
            answers,
            ir::DecisionUsage::new(self.passes, self.input_tokens),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_content_round_trip_preserves_numeric_identity() {
        let raw =
            r#"{"z":18446744073709551615,"a":9007199254740993,"f":1.0,"zero":-0.0,"nul":"\u0000"}"#;
        let request = DecisionRequest {
            state: Some(DecisionContent::Json {
                json_text: raw.into(),
            }),
            questions: vec![DecisionQuestion {
                id: "q".into(),
                instructions: Some(DecisionContent::Text {
                    text: "choose".into(),
                }),
                kind: DecisionQuestionKind::Choice {
                    options: vec![DecisionOption {
                        key: "a".into(),
                        description: None,
                    }],
                },
            }],
        };
        let core = request.into_sdk().unwrap();
        let rebuilt = DecisionRequest::try_from_sdk(&core)
            .unwrap()
            .into_sdk()
            .unwrap();
        assert_eq!(core, rebuilt);
        let text = rebuilt
            .state()
            .body()
            .unwrap()
            .as_json()
            .unwrap()
            .to_json_text();
        assert_eq!(text, raw);
        assert!(
            !format!("{:?}", DecisionRequest::try_from_sdk(&core).unwrap())
                .contains("9007199254740993")
        );
    }

    #[test]
    fn result_round_trip_keeps_half_boundary_and_negative_infinity() {
        let probabilities = [0.4999999981306987, 0.5000000018693014, 0.0];
        let distribution = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                ir::DecisionScore::new(
                    i.to_string(),
                    if i == 2 { f32::NEG_INFINITY } else { 0.0 },
                    p,
                    ir::DecisionOrigin::Caller,
                )
            })
            .collect();
        let core = ir::Decisions::new(
            vec![ir::DecisionAnswer::new(
                "private question",
                distribution,
                ir::DecisionValue::score(1.0, ir::ScoreStatistic::Median),
                Some(probabilities[1]),
            )
            .unwrap()],
            ir::DecisionUsage::new(1, 400),
        );
        let foreign = Decisions::try_from_sdk(&core).unwrap();
        assert!(!format!("{foreign:?}").contains("private question"));
        assert_eq!(
            foreign.answers[0].distribution[1].probability.to_bits(),
            probabilities[1].to_bits()
        );
        assert_eq!(foreign.into_sdk().unwrap(), core);
    }

    #[test]
    fn malformed_json_errors_do_not_disclose_content() {
        let error = DecisionContent::Json {
            json_text: "PRIVATE_CANARY invalid".into(),
        }
        .into_ir()
        .unwrap_err();
        assert!(!format!("{error:?}").contains("PRIVATE_CANARY"));
        assert!(!error.is_retryable());
        let raw_error = crate::Envelope::decision_request_json(
            r#"{"state":"PRIVATE_CANARY","questions":{"q":{"type":"invalid"}}}"#.into(),
        )
        .unwrap_err();
        assert!(!raw_error.to_string().contains("PRIVATE_CANARY"));
        assert!(!raw_error.is_retryable());
        assert!(DecisionContent::Json {
            json_text: "null".into()
        }
        .into_ir()
        .is_err());
        assert!(DecisionContent::Json {
            json_text: r#"{"a":1,"a":2}"#.into()
        }
        .into_ir()
        .is_err());
    }

    #[test]
    fn refusals_name_their_category_and_nothing_else() {
        let message = |content: DecisionContent| match content.into_ir().unwrap_err() {
            Error::ConfigError { message } => message,
            other => panic!("{other:?}"),
        };
        let json = |text: &str| DecisionContent::Json {
            json_text: text.into(),
        };
        assert_eq!(
            message(json(r#"{"PRIVATE":1,"PRIVATE":2}"#)),
            "invalid decision data: duplicate_key"
        );
        assert_eq!(
            message(json("PRIVATE")),
            "invalid decision data: invalid_json"
        );
        assert_eq!(
            message(json("null")),
            "invalid decision data: invalid_request"
        );
        let question = |id: usize| DecisionQuestion {
            id: id.to_string(),
            instructions: Some(DecisionContent::Text {
                text: "PRIVATE".into(),
            }),
            kind: DecisionQuestionKind::YesNo {
                yes: None,
                no: None,
            },
        };
        let error = DecisionRequest {
            state: None,
            questions: (0..17).map(question).collect(),
        }
        .into_sdk()
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            Error::ConfigError {
                message: "invalid decision data: budget_exceeded".into()
            }
            .to_string()
        );
    }

    #[test]
    fn decision_history_and_nested_history_are_refused_before_storage() {
        let request = ir::DecisionRequest::new(
            ir::DecisionState::text("private"),
            vec![ir::DecisionQuestion::new(
                "q",
                Some(ir::DecisionContent::text("choose").into()),
                ir::DecisionQuestionKind::choice(vec![ir::DecisionOption::new("a", None)]),
            )],
        );
        let input = crate::Envelope::try_from_sdk(ir::Envelope::decision_request(request)).unwrap();
        let context = crate::ConversationContextHandle::new();
        assert!(context.push(input.clone()).is_err());
        assert!(context
            .set_system(crate::Envelope::multipart(vec![input]))
            .is_err());
        assert_eq!(context.history_len(), 0);
        assert!(!context.has_system());
    }
}
