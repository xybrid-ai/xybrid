//! [`DecisionError`]: every refusal of the decision IR, with messages that
//! carry no caller content.

use super::ScoreStatistic;
use crate::runtime_adapter::AdapterError;
use std::fmt;

/// Why a JSON text could not be parsed, named without quoting the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JsonErrorKind {
    /// The text ended inside a value.
    UnexpectedEnd,
    /// No value starts here.
    ExpectedValue,
    /// An object member does not start with a string key.
    ExpectedKey,
    /// An object key is not followed by `:`.
    ExpectedColon,
    /// An array item or object member is not followed by `,` or the closing
    /// bracket.
    ExpectedCommaOrEnd,
    /// A word other than `null`, `true` or `false`.
    InvalidLiteral,
    /// A number outside the JSON grammar (a leading zero, `+`, a bare `.` or
    /// `e`).
    InvalidNumber,
    /// A number too large for a finite `f64`. nlohmann::json refuses these
    /// too ("number overflow").
    NumberOutOfRange,
    /// A backslash escape outside the JSON grammar.
    InvalidEscape,
    /// A `\u` escape of a UTF-16 surrogate that is not half of a pair.
    UnpairedSurrogate,
    /// A control character (U+0000 to U+001F) written raw inside a string.
    ControlCharacter,
    /// Something other than whitespace after the value.
    TrailingCharacters,
    /// An object repeats a key (after unescaping).
    DuplicateKey,
}

impl JsonErrorKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::UnexpectedEnd => "unexpected end of input",
            Self::ExpectedValue => "expected a value",
            Self::ExpectedKey => "expected a string key",
            Self::ExpectedColon => "expected ':' after an object key",
            Self::ExpectedCommaOrEnd => "expected ',' or a closing bracket",
            Self::InvalidLiteral => "invalid literal",
            Self::InvalidNumber => "invalid number",
            Self::NumberOutOfRange => "number out of range",
            Self::InvalidEscape => "invalid escape sequence",
            Self::UnpairedSurrogate => "unpaired UTF-16 surrogate escape",
            Self::ControlCharacter => "unescaped control character in a string",
            Self::TrailingCharacters => "characters after the value",
            Self::DuplicateKey => "an object repeats a key",
        }
    }
}

impl fmt::Display for JsonErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An invariant a [`DecisionAnswer`](super::DecisionAnswer) breaks.
///
/// Entries are named by their position in the distribution; keys and
/// question ids are never quoted.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AnswerInvariant {
    /// The question id is empty.
    EmptyQuestionId,
    /// The distribution has no entries.
    EmptyDistribution,
    /// An entry has an empty key.
    EmptyKey {
        /// Position in the distribution.
        index: usize,
    },
    /// Two entries share a key.
    DuplicateKey {
        /// Position of the repeat.
        index: usize,
        /// Position of the first use.
        first: usize,
    },
    /// An entry's raw score is NaN or `+inf`. `-inf` is allowed.
    InvalidRawScore {
        /// Position in the distribution.
        index: usize,
    },
    /// Every raw score is `-inf`: the model ruled every entry out, so no
    /// distribution follows from them.
    NoFiniteRawScore,
    /// An entry's probability is not a finite number in `[0, 1]`.
    InvalidProbability {
        /// Position in the distribution.
        index: usize,
    },
    /// The probabilities do not sum to 1 within
    /// [`PROBABILITY_TOLERANCE`](super::PROBABILITY_TOLERANCE).
    ProbabilitySum {
        /// Their sum, added in distribution order.
        sum: f64,
    },
    /// The chosen key is not in the distribution.
    ChoiceNotInDistribution,
    /// The chosen key does not have the highest probability.
    ChoiceNotHighest,
    /// A yes/no distribution does not have exactly the keys `"true"` and
    /// `"false"`.
    YesNoKeys,
    /// `p_yes` is not the probability of `"true"`.
    PYesMismatch,
    /// Entry `index` of a score distribution is not keyed `"index"`.
    ScoreKey {
        /// Position in the distribution.
        index: usize,
    },
    /// The score value is not the statistic it names.
    ScoreMismatch {
        /// The statistic the value claims to be.
        statistic: ScoreStatistic,
    },
    /// The confidence is not a finite number in `[0, 1]`.
    InvalidConfidence,
}

impl fmt::Display for AnswerInvariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuestionId => f.write_str("the question id is empty"),
            Self::EmptyDistribution => f.write_str("the distribution is empty"),
            Self::EmptyKey { index } => write!(f, "distribution entry {index} has an empty key"),
            Self::DuplicateKey { index, first } => {
                write!(
                    f,
                    "distribution entry {index} repeats the key of entry {first}"
                )
            }
            Self::InvalidRawScore { index } => {
                write!(f, "distribution entry {index} has a NaN or +inf raw score")
            }
            Self::NoFiniteRawScore => f.write_str("no raw score is finite"),
            Self::InvalidProbability { index } => {
                write!(
                    f,
                    "distribution entry {index} has a probability outside [0, 1]"
                )
            }
            Self::ProbabilitySum { sum } => write!(
                f,
                "the probabilities sum to {sum}, not 1 within {}",
                super::PROBABILITY_TOLERANCE
            ),
            Self::ChoiceNotInDistribution => {
                f.write_str("the chosen key is not in the distribution")
            }
            Self::ChoiceNotHighest => {
                f.write_str("the chosen key does not have the highest probability")
            }
            Self::YesNoKeys => {
                f.write_str("a yes/no distribution needs exactly the keys \"true\" and \"false\"")
            }
            Self::PYesMismatch => f.write_str("p_yes is not the probability of \"true\""),
            Self::ScoreKey { index } => {
                write!(
                    f,
                    "score distribution entry {index} is not keyed \"{index}\""
                )
            }
            Self::ScoreMismatch { statistic } => {
                write!(
                    f,
                    "the score value is not the {statistic} of the distribution"
                )
            }
            Self::InvalidConfidence => f.write_str("the confidence is outside [0, 1]"),
        }
    }
}

/// Why JSON text, a decision request or a decision answer was refused.
///
/// Messages hold byte offsets, positions, counts and limits only: never a
/// question id, an option key, text, or any part of a JSON value. The one
/// exception is [`DecisionError::Runtime`], which passes a runtime's own
/// error through, message included.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DecisionError {
    /// The JSON text breaks the JSON grammar.
    #[error("invalid JSON at byte {offset}: {kind}")]
    InvalidJson {
        /// What is wrong.
        kind: JsonErrorKind,
        /// Byte offset of the problem in the text.
        offset: usize,
    },
    /// The JSON text is longer than its [`JsonLimits`](super::JsonLimits)
    /// allow.
    #[error("JSON text is {len} bytes; at most {max} are accepted")]
    JsonTooLong {
        /// Length of the text in bytes.
        len: usize,
        /// [`JsonLimits::max_bytes`](super::JsonLimits::max_bytes).
        max: usize,
    },
    /// The JSON text holds more values than its
    /// [`JsonLimits`](super::JsonLimits) allow.
    #[error("JSON text has more than {max} values")]
    JsonTooManyNodes {
        /// [`JsonLimits::max_nodes`](super::JsonLimits::max_nodes).
        max: usize,
    },
    /// The JSON text nests deeper than
    /// [`MAX_JSON_DEPTH`](super::MAX_JSON_DEPTH).
    #[error("JSON is nested more than {max} levels deep")]
    JsonTooDeep {
        /// [`MAX_JSON_DEPTH`](super::MAX_JSON_DEPTH).
        max: usize,
    },
    /// The request asks no question.
    #[error("a decision request needs at least one question")]
    NoQuestions,
    /// The request asks more questions than the limits allow.
    #[error("{count} questions; at most {max} are accepted")]
    TooManyQuestions {
        /// Number of questions.
        count: usize,
        /// [`StructuralLimits::max_questions`](super::StructuralLimits::max_questions).
        max: usize,
    },
    /// A question has an empty id.
    #[error("question {question} has an empty id")]
    EmptyQuestionId {
        /// Position of the question.
        question: usize,
    },
    /// Two questions share an id.
    #[error("question {question} repeats the id of question {first}")]
    DuplicateQuestionId {
        /// Position of the repeat.
        question: usize,
        /// Position of the first use.
        first: usize,
    },
    /// A choice question has no options.
    #[error("choice question {question} has no options")]
    NoOptions {
        /// Position of the question.
        question: usize,
    },
    /// A choice question has more options than the limits allow.
    #[error("choice question {question} has {count} options; at most {max} are accepted")]
    TooManyOptions {
        /// Position of the question.
        question: usize,
        /// Number of options.
        count: usize,
        /// [`StructuralLimits::max_options`](super::StructuralLimits::max_options).
        max: usize,
    },
    /// An option has an empty key.
    #[error("option {option} of question {question} has an empty key")]
    EmptyOptionKey {
        /// Position of the question.
        question: usize,
        /// Position of the option in its question.
        option: usize,
    },
    /// Two options of one question share a key.
    #[error("option {option} of question {question} repeats the key of option {first}")]
    DuplicateOptionKey {
        /// Position of the question.
        question: usize,
        /// Position of the repeat.
        option: usize,
        /// Position of the first use.
        first: usize,
    },
    /// A score question has no levels.
    #[error("score question {question} has no levels")]
    NoLevels {
        /// Position of the question.
        question: usize,
    },
    /// A score question has more levels than the limits allow.
    #[error("score question {question} has {count} levels; at most {max} are accepted")]
    TooManyLevels {
        /// Position of the question.
        question: usize,
        /// Number of levels.
        count: usize,
        /// [`StructuralLimits::max_options`](super::StructuralLimits::max_options).
        max: usize,
    },
    /// The request holds more content than the limits allow.
    #[error("the request holds {bytes} bytes of content; at most {max} are accepted")]
    RequestTooLarge {
        /// Content bytes, counted as
        /// [`StructuralLimits::max_bytes`](super::StructuralLimits::max_bytes)
        /// describes.
        bytes: usize,
        /// [`StructuralLimits::max_bytes`](super::StructuralLimits::max_bytes).
        max: usize,
    },
    /// The request's JSON content holds more values than the limits allow.
    #[error("the request holds {count} JSON values; at most {max} are accepted")]
    TooManyJsonNodes {
        /// JSON values across the request.
        count: usize,
        /// [`StructuralLimits::max_json_nodes`](super::StructuralLimits::max_json_nodes).
        max: usize,
    },
    /// An answer breaks one of its invariants.
    #[error("invalid decision answer: {0}")]
    InvalidAnswer(AnswerInvariant),
    /// The runtime failed.
    #[error(transparent)]
    Runtime(#[from] AdapterError),
}

impl From<DecisionError> for AdapterError {
    /// JSON and request problems become [`AdapterError::InvalidInput`]. An
    /// invalid answer becomes [`AdapterError::InferenceFailed`]: a protocol
    /// builds answers from model output, so a broken invariant is a bad
    /// output, not a bad request. Runtime errors pass through.
    fn from(error: DecisionError) -> Self {
        match error {
            DecisionError::Runtime(inner) => inner,
            DecisionError::InvalidAnswer(_) => Self::InferenceFailed(error.to_string()),
            _ => Self::InvalidInput(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_and_request_refusals_become_invalid_input() {
        for error in [
            DecisionError::InvalidJson {
                kind: JsonErrorKind::DuplicateKey,
                offset: 7,
            },
            DecisionError::JsonTooDeep { max: 128 },
            DecisionError::NoQuestions,
            DecisionError::DuplicateOptionKey {
                question: 0,
                option: 2,
                first: 1,
            },
            DecisionError::RequestTooLarge { bytes: 10, max: 9 },
        ] {
            let message = error.to_string();
            match AdapterError::from(error) {
                AdapterError::InvalidInput(text) => assert_eq!(text, message),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn invalid_answers_become_inference_failures() {
        let error = DecisionError::InvalidAnswer(AnswerInvariant::ChoiceNotHighest);
        assert!(matches!(
            AdapterError::from(error),
            AdapterError::InferenceFailed(text)
                if text == "invalid decision answer: the chosen key does not have the highest probability"
        ));
    }

    #[test]
    fn runtime_errors_pass_through_unchanged() {
        let error = DecisionError::from(AdapterError::ModelNotLoaded("m".into()));
        assert_eq!(error.to_string(), "Model not loaded: m");
        assert!(matches!(
            AdapterError::from(error),
            AdapterError::ModelNotLoaded(id) if id == "m"
        ));
    }

    #[test]
    fn messages_name_offsets_positions_and_counts() {
        let cases: [(DecisionError, &str); 8] = [
            (
                DecisionError::InvalidJson {
                    kind: JsonErrorKind::UnpairedSurrogate,
                    offset: 12,
                },
                "invalid JSON at byte 12: unpaired UTF-16 surrogate escape",
            ),
            (
                DecisionError::JsonTooLong { len: 9, max: 8 },
                "JSON text is 9 bytes; at most 8 are accepted",
            ),
            (
                DecisionError::JsonTooManyNodes { max: 4 },
                "JSON text has more than 4 values",
            ),
            (
                DecisionError::TooManyQuestions { count: 17, max: 16 },
                "17 questions; at most 16 are accepted",
            ),
            (
                DecisionError::DuplicateQuestionId {
                    question: 3,
                    first: 1,
                },
                "question 3 repeats the id of question 1",
            ),
            (
                DecisionError::TooManyLevels {
                    question: 0,
                    count: 300,
                    max: 255,
                },
                "score question 0 has 300 levels; at most 255 are accepted",
            ),
            (
                DecisionError::InvalidAnswer(AnswerInvariant::ScoreMismatch {
                    statistic: ScoreStatistic::Median,
                }),
                "invalid decision answer: the score value is not the median of the distribution",
            ),
            (
                DecisionError::InvalidAnswer(AnswerInvariant::ProbabilitySum { sum: 0.5 }),
                "invalid decision answer: the probabilities sum to 0.5, not 1 within 0.000001",
            ),
        ];
        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
        }
    }
}
