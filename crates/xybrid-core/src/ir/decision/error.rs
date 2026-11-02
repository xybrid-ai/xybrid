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
/// Every variant carries byte offsets, positions, counts, limits and kinds
/// only, so its message, its `Debug` and the [`AdapterError`] it converts
/// into never hold a question id, an option key, text, or any part of a
/// JSON value. A variant added later keeps to that rule; there is no
/// passthrough for another error's message.
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
}

impl From<DecisionError> for AdapterError {
    /// JSON and request problems become [`AdapterError::InvalidInput`]. An
    /// invalid answer becomes [`AdapterError::InferenceFailed`]: a protocol
    /// builds answers from model output, so a broken invariant is a bad
    /// output, not a bad request. Either way the message is the
    /// content-free [`DecisionError`] message.
    fn from(error: DecisionError) -> Self {
        match error {
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

    const CANARY: &str = "PRIVATE-REQUEST-CANARY";
    const NUMBER_CANARY: &str = "918273645";

    /// Every variant and kind, refusing input full of caller content.
    fn every_refusal() -> Vec<DecisionError> {
        use crate::ir::decision::{
            validate_structure, DecisionAnswer, DecisionContent, DecisionLevel, DecisionOption,
            DecisionOrigin, DecisionQuestion, DecisionQuestionKind, DecisionRequest, DecisionScore,
            DecisionState, DecisionValue, JsonLimits, OrderedJson, StructuralLimits,
        };
        let c = CANARY;
        let n = NUMBER_CANARY;
        let mut refusals = Vec::new();

        // JSON text, one refusal per syntax kind, then each ceiling.
        for text in [
            format!(r#""{c}"#),
            format!(r#"["{c}", ]"#),
            format!(r#"{{"{c}": {n},}}"#),
            format!(r#"{{"{c}"}}"#),
            format!(r#"["{c}" {n}]"#),
            format!(r#"["{c}", nul]"#),
            format!(r#"["{c}", 0{n}]"#),
            format!(r#"["{c}", {n}e999]"#),
            format!(r#""{c}\q""#),
            format!(r#""{c}\uD800""#),
            format!("\"{c}\t\""),
            format!(r#""{c}" {n}"#),
            format!(r#"{{"{c}": 1, "{c}": {n}}}"#),
        ] {
            refusals.push(OrderedJson::parse(&text).unwrap_err());
        }
        let text = format!(r#"["{c}", {n}]"#);
        let tight = JsonLimits::default();
        refusals.push(OrderedJson::parse_with_limits(&text, &tight.with_max_bytes(4)).unwrap_err());
        refusals.push(OrderedJson::parse_with_limits(&text, &tight.with_max_nodes(2)).unwrap_err());
        let deep = format!("{}\"{c}\"{}", "[".repeat(200), "]".repeat(200));
        refusals.push(OrderedJson::parse(&deep).unwrap_err());

        // Request structure.
        let text = |text: String| Some(DecisionContent::text(text));
        let yes_no = |id: String| {
            DecisionQuestion::new(id, None, DecisionQuestionKind::yes_no(text(c.into()), None))
        };
        let choice = |keys: Vec<String>| {
            let options = keys
                .into_iter()
                .map(|key| DecisionOption::new(key, text(c.into())))
                .collect();
            DecisionQuestion::new(c, None, DecisionQuestionKind::choice(options))
        };
        let score = |levels: usize| {
            DecisionQuestion::new(
                c,
                None,
                DecisionQuestionKind::score(vec![DecisionLevel::new(text(c.into())); levels]),
            )
        };
        let state = || {
            DecisionState::from_json(
                OrderedJson::parse(&format!(r#"{{"{c}": [{n}, "{c}"]}}"#)).unwrap(),
            )
        };
        let check = |questions: Vec<DecisionQuestion>, limits: StructuralLimits| {
            validate_structure(&DecisionRequest::new(state(), questions), &limits).unwrap_err()
        };
        let limits = StructuralLimits::default;
        refusals.push(check(vec![], limits()));
        refusals.push(check(
            (0..17).map(|i| yes_no(format!("{c}-{i}"))).collect(),
            limits(),
        ));
        refusals.push(check(
            vec![yes_no(c.into()), yes_no(String::new())],
            limits(),
        ));
        refusals.push(check(vec![yes_no(c.into()), yes_no(c.into())], limits()));
        refusals.push(check(vec![choice(vec![])], limits()));
        refusals.push(check(
            vec![choice((0..256).map(|i| format!("{c}-{i}")).collect())],
            limits(),
        ));
        refusals.push(check(vec![choice(vec![c.into(), String::new()])], limits()));
        refusals.push(check(vec![choice(vec![c.into(), c.into()])], limits()));
        refusals.push(check(vec![score(0)], limits()));
        refusals.push(check(vec![score(256)], limits()));
        refusals.push(check(vec![yes_no(c.into())], limits().with_max_bytes(8)));
        refusals.push(check(
            vec![yes_no(c.into())],
            limits().with_max_json_nodes(1),
        ));

        // Answers, one refusal per invariant.
        let entry =
            |key: &str, raw: f32, p: f64| DecisionScore::new(key, raw, p, DecisionOrigin::Caller);
        let answer =
            |id: &str, distribution: Vec<DecisionScore>, value: DecisionValue, confidence| {
                DecisionAnswer::new(id, distribution, value, confidence).unwrap_err()
            };
        let pick = || DecisionValue::choice(c);
        let one = || vec![entry(c, 0.0, 1.0)];
        refusals.push(answer("", one(), pick(), None));
        refusals.push(answer(c, vec![], pick(), None));
        refusals.push(answer(c, vec![entry("", 0.0, 1.0)], pick(), None));
        refusals.push(answer(
            c,
            vec![entry(c, 0.0, 0.5), entry(c, 0.0, 0.5)],
            pick(),
            None,
        ));
        refusals.push(answer(c, vec![entry(c, f32::NAN, 1.0)], pick(), None));
        refusals.push(answer(
            c,
            vec![entry(c, f32::NEG_INFINITY, 1.0)],
            pick(),
            None,
        ));
        refusals.push(answer(c, vec![entry(c, 0.0, 2.0)], pick(), None));
        refusals.push(answer(c, vec![entry(c, 0.0, 0.5)], pick(), None));
        refusals.push(answer(
            c,
            one(),
            DecisionValue::choice(format!("{c}-other")),
            None,
        ));
        refusals.push(answer(
            c,
            vec![entry(c, 0.0, 0.25), entry("b", 0.0, 0.75)],
            pick(),
            None,
        ));
        refusals.push(answer(c, one(), DecisionValue::yes_no(1.0), None));
        let yes_no_entries = vec![entry("true", 0.0, 0.75), entry("false", 0.0, 0.25)];
        refusals.push(answer(c, yes_no_entries, DecisionValue::yes_no(0.5), None));
        refusals.push(answer(
            c,
            one(),
            DecisionValue::score(0.0, ScoreStatistic::Median),
            None,
        ));
        let half = vec![entry("0", 0.0, 0.5), entry("1", 0.0, 0.5)];
        refusals.push(answer(
            c,
            half,
            DecisionValue::score(1.0, ScoreStatistic::Median),
            None,
        ));
        refusals.push(answer(c, one(), pick(), Some(2.0)));
        refusals
    }

    /// One name per variant and kind. The matches are exhaustive, so a new
    /// variant stops this from compiling until it is named here and
    /// `every_refusal` produces it.
    fn variant(error: &DecisionError) -> String {
        let name = match error {
            DecisionError::InvalidJson { kind, .. } => {
                return format!(
                    "InvalidJson::{}",
                    match kind {
                        JsonErrorKind::UnexpectedEnd => "UnexpectedEnd",
                        JsonErrorKind::ExpectedValue => "ExpectedValue",
                        JsonErrorKind::ExpectedKey => "ExpectedKey",
                        JsonErrorKind::ExpectedColon => "ExpectedColon",
                        JsonErrorKind::ExpectedCommaOrEnd => "ExpectedCommaOrEnd",
                        JsonErrorKind::InvalidLiteral => "InvalidLiteral",
                        JsonErrorKind::InvalidNumber => "InvalidNumber",
                        JsonErrorKind::NumberOutOfRange => "NumberOutOfRange",
                        JsonErrorKind::InvalidEscape => "InvalidEscape",
                        JsonErrorKind::UnpairedSurrogate => "UnpairedSurrogate",
                        JsonErrorKind::ControlCharacter => "ControlCharacter",
                        JsonErrorKind::TrailingCharacters => "TrailingCharacters",
                        JsonErrorKind::DuplicateKey => "DuplicateKey",
                    }
                )
            }
            DecisionError::JsonTooLong { .. } => "JsonTooLong",
            DecisionError::JsonTooManyNodes { .. } => "JsonTooManyNodes",
            DecisionError::JsonTooDeep { .. } => "JsonTooDeep",
            DecisionError::NoQuestions => "NoQuestions",
            DecisionError::TooManyQuestions { .. } => "TooManyQuestions",
            DecisionError::EmptyQuestionId { .. } => "EmptyQuestionId",
            DecisionError::DuplicateQuestionId { .. } => "DuplicateQuestionId",
            DecisionError::NoOptions { .. } => "NoOptions",
            DecisionError::TooManyOptions { .. } => "TooManyOptions",
            DecisionError::EmptyOptionKey { .. } => "EmptyOptionKey",
            DecisionError::DuplicateOptionKey { .. } => "DuplicateOptionKey",
            DecisionError::NoLevels { .. } => "NoLevels",
            DecisionError::TooManyLevels { .. } => "TooManyLevels",
            DecisionError::RequestTooLarge { .. } => "RequestTooLarge",
            DecisionError::TooManyJsonNodes { .. } => "TooManyJsonNodes",
            DecisionError::InvalidAnswer(invariant) => {
                return format!(
                    "InvalidAnswer::{}",
                    match invariant {
                        AnswerInvariant::EmptyQuestionId => "EmptyQuestionId",
                        AnswerInvariant::EmptyDistribution => "EmptyDistribution",
                        AnswerInvariant::EmptyKey { .. } => "EmptyKey",
                        AnswerInvariant::DuplicateKey { .. } => "DuplicateKey",
                        AnswerInvariant::InvalidRawScore { .. } => "InvalidRawScore",
                        AnswerInvariant::NoFiniteRawScore => "NoFiniteRawScore",
                        AnswerInvariant::InvalidProbability { .. } => "InvalidProbability",
                        AnswerInvariant::ProbabilitySum { .. } => "ProbabilitySum",
                        AnswerInvariant::ChoiceNotInDistribution => "ChoiceNotInDistribution",
                        AnswerInvariant::ChoiceNotHighest => "ChoiceNotHighest",
                        AnswerInvariant::YesNoKeys => "YesNoKeys",
                        AnswerInvariant::PYesMismatch => "PYesMismatch",
                        AnswerInvariant::ScoreKey { .. } => "ScoreKey",
                        AnswerInvariant::ScoreMismatch { .. } => "ScoreMismatch",
                        AnswerInvariant::InvalidConfidence => "InvalidConfidence",
                    }
                )
            }
        };
        name.to_string()
    }

    /// The text of `error` and of everything its `source()` chain reaches.
    fn texts(error: &dyn std::error::Error) -> Vec<String> {
        let mut texts = vec![
            error.to_string(),
            format!("{error:?}"),
            format!("{error:#?}"),
        ];
        let mut source = error.source();
        while let Some(inner) = source {
            texts.extend([inner.to_string(), format!("{inner:?}")]);
            source = inner.source();
        }
        texts
    }

    #[test]
    fn no_refusal_carries_caller_content() {
        let refusals = every_refusal();
        let names: std::collections::BTreeSet<String> = refusals.iter().map(variant).collect();
        // 13 JSON syntax kinds, 3 JSON ceilings, 12 request refusals and 15
        // answer invariants: every one, each once.
        assert_eq!(names.len(), 43, "{names:#?}");
        assert_eq!(refusals.len(), names.len());

        for error in refusals {
            let name = variant(&error);
            let mut seen = texts(&error);
            let adapter = AdapterError::from(error);
            assert!(
                matches!(
                    adapter,
                    AdapterError::InvalidInput(_) | AdapterError::InferenceFailed(_)
                ),
                "{name}"
            );
            seen.extend(texts(&adapter));
            for text in seen {
                assert!(!text.contains(CANARY), "{name}: {text}");
                assert!(!text.contains(NUMBER_CANARY), "{name}: {text}");
            }
        }
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
