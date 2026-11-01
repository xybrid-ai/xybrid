//! [`Decisions`] to a SystemOne response.
//!
//! See the [module docs](super) for the response format.

use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use std::collections::HashSet;
use std::fmt;
use xybrid_core::ir::{
    DecisionAnswer, DecisionContent, DecisionContentView, DecisionLevel, DecisionQuestion,
    DecisionRequest, DecisionScore, Decisions, JsonNumber, JsonView, OrderedJson,
};

/// Why answers could not be written as the response to a request: they do
/// not answer it. Answers a runtime returned for the same request always
/// fit, so this is a bug in whatever produced them.
///
/// Messages name positions (counting from 0) only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseError {
    /// The answers and the questions differ in number.
    AnswerCount {
        /// Number of answers.
        answers: usize,
        /// Number of questions.
        questions: usize,
    },
    /// Two questions share an id, so one response cannot hold both
    /// answers. Requests that passed
    /// [`validate_structure`](xybrid_core::ir::decision::validate_structure)
    /// never do.
    RepeatedQuestionId {
        /// Position of the repeat.
        question: usize,
    },
    /// An answer is to another question than the one at its position.
    QuestionMismatch {
        /// Position of the answer.
        answer: usize,
    },
    /// An answer is not of its question's type, or of a type the subset
    /// cannot write.
    KindMismatch {
        /// Position of the answer.
        answer: usize,
    },
    /// An answer's distribution does not list its question's options, or
    /// levels, in order.
    DistributionMismatch {
        /// Position of the answer.
        answer: usize,
    },
    /// An answer holds a number that is not finite, which a valid answer
    /// never does.
    NonFinite {
        /// Position of the answer.
        answer: usize,
    },
}

impl fmt::Display for ResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AnswerCount { answers, questions } => {
                write!(f, "{answers} answers to {questions} questions")
            }
            Self::RepeatedQuestionId { question } => {
                write!(
                    f,
                    "question {question} repeats the id of an earlier question"
                )
            }
            Self::QuestionMismatch { answer } => {
                write!(
                    f,
                    "answer {answer} answers another question than question {answer}"
                )
            }
            Self::KindMismatch { answer } => {
                write!(f, "answer {answer} is not of the type of question {answer}")
            }
            Self::DistributionMismatch { answer } => write!(
                f,
                "the distribution of answer {answer} does not list the options or levels of \
                 question {answer} in order"
            ),
            Self::NonFinite { answer } => {
                write!(f, "answer {answer} holds a number that is not finite")
            }
        }
    }
}

impl std::error::Error for ResponseError {}

/// Writes `decisions`, the answers to `request`, as a SystemOne response,
/// pretty-printed: `model`, then `answers` keyed by question id in request
/// order, then `usage`.
///
/// `model` names the model that answered. `input_tokens` is
/// [`DecisionUsage::evaluated_tokens`](xybrid_core::ir::DecisionUsage::evaluated_tokens),
/// the tokens of every pass, as llama.cpp counts them; `output_tokens` is
/// always 0. See the [module docs](super) for each answer's members and the
/// spelling of numbers.
///
/// # Errors
///
/// A [`ResponseError`] when `decisions` do not answer `request`: another
/// number of answers, an answer to another question or of another type,
/// a distribution that does not list the question's options or levels, or
/// a number that is not finite.
pub fn render_response(
    model: &str,
    request: &DecisionRequest,
    decisions: &Decisions,
) -> Result<String, ResponseError> {
    let questions = request.questions();
    let answers = decisions.answers();
    if answers.len() != questions.len() {
        return Err(ResponseError::AnswerCount {
            answers: answers.len(),
            questions: questions.len(),
        });
    }
    let mut ids = HashSet::with_capacity(questions.len());
    let answers = questions
        .iter()
        .zip(answers)
        .enumerate()
        .map(|(index, (question, answer))| {
            if !ids.insert(question.id()) {
                return Err(ResponseError::RepeatedQuestionId { question: index });
            }
            Answer::new(index, question, answer)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let response = Response {
        model,
        answers,
        input_tokens: decisions.usage().evaluated_tokens(),
    };
    Ok(serde_json::to_string_pretty(&response)
        .expect("a response holds strings, finite numbers and maps with string keys only"))
}

/// The response, ready to serialize: every number finite.
struct Response<'a> {
    model: &'a str,
    answers: Vec<Answer<'a>>,
    input_tokens: u64,
}

/// One answer, checked against its question.
struct Answer<'a> {
    id: &'a str,
    body: Body<'a>,
}

enum Body<'a> {
    Choice {
        choice: &'a str,
        distribution: &'a [DecisionScore],
        confidence: Option<f64>,
    },
    Noul {
        p_yes: f64,
    },
    Score {
        score: f64,
        levels: &'a [DecisionLevel],
        distribution: &'a [DecisionScore],
        confidence: Option<f64>,
    },
}

impl<'a> Answer<'a> {
    fn new(
        index: usize,
        question: &'a DecisionQuestion,
        answer: &'a DecisionAnswer,
    ) -> Result<Self, ResponseError> {
        if answer.question_id() != question.id() {
            return Err(ResponseError::QuestionMismatch { answer: index });
        }
        let kind = question.kind();
        let value = answer.value();
        let distribution = answer.distribution();
        let keys = || distribution.iter().map(DecisionScore::key);
        let body = if let (Some(options), Some(choice)) = (kind.as_choice(), value.as_choice()) {
            if !keys().eq(options.iter().map(|option| option.key())) {
                return Err(ResponseError::DistributionMismatch { answer: index });
            }
            Body::Choice {
                choice,
                distribution,
                confidence: answer.confidence(),
            }
        } else if let (Some(_), Some(p_yes)) = (kind.as_yes_no(), value.as_yes_no()) {
            Body::Noul { p_yes }
        } else if let (Some(levels), Some((score, _))) = (kind.as_score(), value.as_score()) {
            // A protocol's statistic (LFM3 d1's median) is the score.
            if !keys().eq((0..levels.len()).map(|level| level.to_string())) {
                return Err(ResponseError::DistributionMismatch { answer: index });
            }
            Body::Score {
                score,
                levels,
                distribution,
                confidence: answer.confidence(),
            }
        } else {
            return Err(ResponseError::KindMismatch { answer: index });
        };
        let confidence = match &body {
            Body::Choice { confidence, .. } | Body::Score { confidence, .. } => *confidence,
            Body::Noul { .. } => None,
        };
        let finite = distribution
            .iter()
            .map(DecisionScore::probability)
            .chain(confidence)
            .chain(value.as_yes_no())
            .chain(value.as_score().map(|(score, _)| score))
            .all(f64::is_finite);
        if !finite {
            return Err(ResponseError::NonFinite { answer: index });
        }
        Ok(Self {
            id: question.id(),
            body,
        })
    }
}

impl Serialize for Response<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("model", self.model)?;
        map.serialize_entry("answers", &Answers(&self.answers))?;
        map.serialize_entry("usage", &Usage(self.input_tokens))?;
        map.end()
    }
}

/// The answers object: question id to answer, in request order.
struct Answers<'a>(&'a [Answer<'a>]);

impl Serialize for Answers<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for answer in self.0 {
            map.serialize_entry(answer.id, &answer.body)?;
        }
        map.end()
    }
}

/// llama.cpp's members, in its order (`format_answer`).
impl Serialize for Body<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Choice {
                choice,
                distribution,
                confidence,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "choice")?;
                map.serialize_entry("choice", choice)?;
                map.serialize_entry("probabilities", &Probabilities(distribution))?;
                map.serialize_entry("confidence", confidence)?;
                map.end()
            }
            Self::Noul { p_yes } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "noul")?;
                map.serialize_entry("noul", p_yes)?;
                map.end()
            }
            Self::Score {
                score,
                levels,
                distribution,
                confidence,
            } => {
                let mut map = serializer.serialize_map(Some(5))?;
                map.serialize_entry("type", "score")?;
                map.serialize_entry("score", score)?;
                map.serialize_entry("legend", &Legend(levels))?;
                map.serialize_entry("probabilities", &Probabilities(distribution))?;
                map.serialize_entry("confidence", confidence)?;
                map.end()
            }
        }
    }
}

/// Key to probability, in distribution order.
struct Probabilities<'a>(&'a [DecisionScore]);

impl Serialize for Probabilities<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for entry in self.0 {
            map.serialize_entry(entry.key(), &entry.probability())?;
        }
        map.end()
    }
}

/// Level to its description, as the request gave it: `"0"` to `"n-1"`.
struct Legend<'a>(&'a [DecisionLevel]);

impl Serialize for Legend<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (level, entry) in self.0.iter().enumerate() {
            map.serialize_entry(&level.to_string(), &Content(entry.description()))?;
        }
        map.end()
    }
}

/// A description back as the JSON value it was: `null` when absent, a
/// string for text, the value itself for JSON.
struct Content<'a>(Option<&'a DecisionContent>);

impl Serialize for Content<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0.map(DecisionContent::view) {
            None => serializer.serialize_unit(),
            Some(DecisionContentView::Text(text)) => serializer.serialize_str(text),
            Some(DecisionContentView::Json(value)) => Json(value).serialize(serializer),
        }
    }
}

/// An [`OrderedJson`] value inline, keys in order and numbers as they were
/// read: integers as integers, floats with a `.` or an exponent.
struct Json<'a>(&'a OrderedJson);

impl Serialize for Json<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0.view() {
            JsonView::Null => serializer.serialize_unit(),
            JsonView::Bool(value) => serializer.serialize_bool(value),
            JsonView::Number(JsonNumber::I64(value)) => serializer.serialize_i64(value),
            JsonView::Number(JsonNumber::U64(value)) => serializer.serialize_u64(value),
            // Finite: an `OrderedJson` holds no other.
            JsonView::Number(JsonNumber::F64(value)) => serializer.serialize_f64(value),
            JsonView::String(text) => serializer.serialize_str(text),
            JsonView::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(&Json(item))?;
                }
                seq.end()
            }
            JsonView::Object(members) => {
                let mut map = serializer.serialize_map(Some(members.len()))?;
                for (key, value) in members {
                    map.serialize_entry(key, &Json(value))?;
                }
                map.end()
            }
        }
    }
}

/// `{"input_tokens": n, "output_tokens": 0}`.
struct Usage(u64);

impl Serialize for Usage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("input_tokens", &self.0)?;
        map.serialize_entry("output_tokens", &0u64)?;
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_core::ir::{
        DecisionOption, DecisionOrigin, DecisionQuestionKind, DecisionState, DecisionUsage,
        DecisionValue, ScoreStatistic,
    };

    const CANARY: &str = "canary-9b44d2";

    fn json(text: &str) -> OrderedJson {
        OrderedJson::parse(text).unwrap()
    }

    fn entry(key: &str, raw: f32, probability: f64) -> DecisionScore {
        DecisionScore::new(key, raw, probability, DecisionOrigin::Caller)
    }

    fn request() -> DecisionRequest {
        DecisionRequest::new(
            DecisionState::text("state"),
            vec![
                DecisionQuestion::new(
                    "route",
                    None,
                    DecisionQuestionKind::choice(vec![
                        DecisionOption::new("billing", None),
                        DecisionOption::new("tech", None),
                    ]),
                ),
                DecisionQuestion::new("angry", None, DecisionQuestionKind::yes_no(None, None)),
                DecisionQuestion::new(
                    "urgency",
                    None,
                    DecisionQuestionKind::score(vec![
                        DecisionLevel::new(Some(DecisionContent::text("can wait"))),
                        DecisionLevel::new(None),
                        DecisionLevel::new(DecisionContent::from_json(json(
                            r#"{"z": 1, "a": [1.0, -0.0, 18446744073709551615, "x\n"]}"#,
                        ))),
                    ]),
                ),
            ],
        )
    }

    fn answers() -> Vec<DecisionAnswer> {
        vec![
            DecisionAnswer::new(
                "route",
                vec![entry("billing", 3.5, 0.75), entry("tech", 1.0, 0.25)],
                DecisionValue::choice("billing"),
                Some(0.75),
            )
            .unwrap(),
            DecisionAnswer::new(
                "angry",
                vec![
                    entry("true", 2.0, 0.1),
                    entry("false", f32::NEG_INFINITY, 0.9),
                ],
                DecisionValue::yes_no(0.1),
                None,
            )
            .unwrap(),
            DecisionAnswer::new(
                "urgency",
                vec![
                    entry("0", 0.0, 0.2),
                    entry("1", 1.0, 0.5),
                    entry("2", 0.5, 0.3),
                ],
                DecisionValue::score(1.0, ScoreStatistic::Median),
                Some(0.5),
            )
            .unwrap(),
        ]
    }

    fn decisions(answers: Vec<DecisionAnswer>) -> Decisions {
        Decisions::new(answers, DecisionUsage::new(3, 412))
    }

    #[test]
    fn a_response_follows_llama_cpp_member_by_member() {
        let text = render_response("d1-2B", &request(), &decisions(answers())).unwrap();
        assert_eq!(
            text,
            r#"{
  "model": "d1-2B",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.75,
        "tech": 0.25
      },
      "confidence": 0.75
    },
    "angry": {
      "type": "noul",
      "noul": 0.1
    },
    "urgency": {
      "type": "score",
      "score": 1.0,
      "legend": {
        "0": "can wait",
        "1": null,
        "2": {
          "z": 1,
          "a": [
            1.0,
            -0.0,
            18446744073709551615,
            "x\n"
          ]
        }
      },
      "probabilities": {
        "0": 0.2,
        "1": 0.5,
        "2": 0.3
      },
      "confidence": 0.5
    }
  },
  "usage": {
    "input_tokens": 412,
    "output_tokens": 0
  }
}"#
        );
    }

    #[test]
    fn numbers_take_the_shortest_spelling_that_reads_back_exactly() {
        // Probabilities near the edges of the fixed notation, and values
        // whose shortest spelling needs all 17 digits.
        let values = [
            0.1 + 0.2,
            1.0 / 3.0,
            1e-5,
            1e-7,
            2.5e-300,
            0.9999999999999999,
            5e-324,
        ];
        for value in values {
            let answer = DecisionAnswer::new(
                "q",
                vec![entry("true", 0.0, value), entry("false", 0.0, 1.0 - value)],
                DecisionValue::yes_no(value),
                None,
            )
            .unwrap();
            let request = DecisionRequest::new(
                DecisionState::default(),
                vec![DecisionQuestion::new(
                    "q",
                    None,
                    DecisionQuestionKind::yes_no(None, None),
                )],
            );
            let text = render_response("m", &request, &decisions(vec![answer])).unwrap();
            let parsed = json(&text);
            let noul = parsed
                .get("answers")
                .and_then(|answers| answers.get("q"))
                .and_then(|answer| answer.get("noul"))
                .and_then(OrderedJson::as_number);
            assert_eq!(noul, Some(JsonNumber::F64(value)), "{text}");
            // The spelling is the shortest that reads back: no 17-digit
            // %.17g form where fewer digits do.
            let written = text
                .lines()
                .find(|line| line.trim_start().starts_with("\"noul\":"))
                .and_then(|line| line.split(": ").nth(1))
                .unwrap();
            assert_eq!(written.parse::<f64>().unwrap().to_bits(), value.to_bits());
            assert_eq!(written, serde_json::to_string(&value).unwrap());
        }
    }

    #[test]
    fn answers_must_answer_the_request() {
        let request = request();
        let render = |answers: Vec<DecisionAnswer>| {
            render_response("m", &request, &decisions(answers)).unwrap_err()
        };

        let mut fewer = answers();
        fewer.pop();
        assert_eq!(
            render(fewer),
            ResponseError::AnswerCount {
                answers: 2,
                questions: 3
            }
        );

        let mut swapped = answers();
        swapped.swap(0, 1);
        assert_eq!(
            render(swapped),
            ResponseError::QuestionMismatch { answer: 0 }
        );

        let mut wrong_kind = answers();
        wrong_kind[1] = DecisionAnswer::new(
            "angry",
            vec![entry("true", 0.0, 0.5), entry("false", 0.0, 0.5)],
            DecisionValue::choice("true"),
            None,
        )
        .unwrap();
        assert_eq!(
            render(wrong_kind),
            ResponseError::KindMismatch { answer: 1 }
        );

        let mut other_keys = answers();
        other_keys[0] = DecisionAnswer::new(
            "route",
            vec![entry("tech", 1.0, 0.25), entry("billing", 3.5, 0.75)],
            DecisionValue::choice("billing"),
            Some(0.75),
        )
        .unwrap();
        assert_eq!(
            render(other_keys),
            ResponseError::DistributionMismatch { answer: 0 }
        );

        let mut fixed = answers();
        fixed[0] = DecisionAnswer::new(
            "route",
            vec![
                entry("billing", 3.5, 0.5),
                entry("tech", 1.0, 0.25),
                DecisionScore::new("skip", 1.0, 0.25, DecisionOrigin::Fixed),
            ],
            DecisionValue::choice("billing"),
            Some(0.5),
        )
        .unwrap();
        assert_eq!(
            render(fixed),
            ResponseError::DistributionMismatch { answer: 0 }
        );

        let mut fewer_levels = answers();
        fewer_levels[2] = DecisionAnswer::new(
            "urgency",
            vec![entry("0", 0.0, 0.5), entry("1", 1.0, 0.5)],
            DecisionValue::score(0.0, ScoreStatistic::Median),
            Some(0.5),
        )
        .unwrap();
        assert_eq!(
            render(fewer_levels),
            ResponseError::DistributionMismatch { answer: 2 }
        );

        let repeated = DecisionRequest::new(
            DecisionState::default(),
            vec![
                DecisionQuestion::new("q", None, DecisionQuestionKind::yes_no(None, None)),
                DecisionQuestion::new("q", None, DecisionQuestionKind::yes_no(None, None)),
            ],
        );
        let yes_no = DecisionAnswer::new(
            "q",
            vec![entry("true", 0.0, 0.5), entry("false", 0.0, 0.5)],
            DecisionValue::yes_no(0.5),
            None,
        )
        .unwrap();
        assert_eq!(
            render_response("m", &repeated, &decisions(vec![yes_no.clone(), yes_no])),
            Err(ResponseError::RepeatedQuestionId { question: 1 })
        );
    }

    #[test]
    fn an_absent_confidence_is_null() {
        let mut answers = answers();
        answers[0] = DecisionAnswer::new(
            "route",
            vec![entry("billing", 3.5, 0.75), entry("tech", 1.0, 0.25)],
            DecisionValue::choice("billing"),
            None,
        )
        .unwrap();
        let text = render_response("m", &request(), &decisions(answers)).unwrap();
        let confidence = json(&text)
            .get("answers")
            .and_then(|answers| answers.get("route"))
            .and_then(|answer| answer.get("confidence"))
            .map(OrderedJson::is_null);
        assert_eq!(confidence, Some(true));
    }

    #[test]
    fn errors_never_quote_ids_or_keys() {
        let request = DecisionRequest::new(
            DecisionState::default(),
            vec![DecisionQuestion::new(
                CANARY,
                None,
                DecisionQuestionKind::choice(vec![DecisionOption::new(CANARY, None)]),
            )],
        );
        let answer = DecisionAnswer::new(
            format!("{CANARY}-other"),
            vec![entry(CANARY, 0.0, 1.0)],
            DecisionValue::choice(CANARY),
            Some(1.0),
        )
        .unwrap();
        let error = render_response(CANARY, &request, &decisions(vec![answer])).unwrap_err();
        for rendered in [error.to_string(), format!("{error:?}")] {
            assert!(!rendered.contains(CANARY), "{rendered}");
        }
    }
}
