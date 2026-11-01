//! [`validate_structure`]: the checks every decision protocol needs.

use super::{
    DecisionContent, DecisionContentView, DecisionError, DecisionInstructions,
    DecisionQuestionKind, DecisionRequest,
};
use std::collections::HashMap;

/// Default [`StructuralLimits::max_questions`]: 16.
pub const DEFAULT_MAX_QUESTIONS: usize = 16;

/// Default [`StructuralLimits::max_options`]: 255.
pub const DEFAULT_MAX_OPTIONS: usize = 255;

/// Default [`StructuralLimits::max_bytes`]: 1 MiB.
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 1 << 20;

/// Default [`StructuralLimits::max_json_nodes`]: 65,536 values.
pub const DEFAULT_MAX_REQUEST_JSON_NODES: usize = 1 << 16;

/// Protocol-neutral ceilings on one decision request.
///
/// They bound what a request can make a runtime read, render and tokenize
/// before its protocol's own checks (token budgets, cardinalities) run, and
/// a protocol may accept less. The defaults:
///
/// - [`DEFAULT_MAX_QUESTIONS`] (16): a product ceiling on the passes one run
///   may take, not a model capacity;
/// - [`DEFAULT_MAX_OPTIONS`] (255): llama.cpp's option ceiling for most
///   decision model families, LFM3 d1 included;
/// - [`DEFAULT_MAX_REQUEST_BYTES`] (1 MiB) and
///   [`DEFAULT_MAX_REQUEST_JSON_NODES`] (65,536): ceilings against oversized
///   input, far above what a decision prompt reads.
///
/// The IR's decode ceilings (16 questions or answers; 255 options, levels or
/// distribution entries; 1 MiB per string field; 2 MiB of text and 65,536
/// values per JSON value) bound whatever limits are configured here: a
/// request beyond them can be built and validated in memory, but its serde
/// form is refused when decoded, so keep these limits at or below them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StructuralLimits {
    /// Most questions in one request.
    pub max_questions: usize,
    /// Most options of one choice question, and most levels of one score
    /// question.
    pub max_options: usize,
    /// Most content bytes in one request: the UTF-8 bytes of every question
    /// id, option key and text, plus each JSON value's compact JSON text
    /// ([`OrderedJson::json_text_len`](super::OrderedJson::json_text_len)).
    pub max_bytes: usize,
    /// Most JSON values across the request's JSON content
    /// ([`OrderedJson::node_count`](super::OrderedJson::node_count)).
    pub max_json_nodes: usize,
}

impl Default for StructuralLimits {
    fn default() -> Self {
        Self {
            max_questions: DEFAULT_MAX_QUESTIONS,
            max_options: DEFAULT_MAX_OPTIONS,
            max_bytes: DEFAULT_MAX_REQUEST_BYTES,
            max_json_nodes: DEFAULT_MAX_REQUEST_JSON_NODES,
        }
    }
}

impl StructuralLimits {
    /// Override the question ceiling.
    pub fn with_max_questions(mut self, max_questions: usize) -> Self {
        self.max_questions = max_questions;
        self
    }

    /// Override the option and level ceiling.
    pub fn with_max_options(mut self, max_options: usize) -> Self {
        self.max_options = max_options;
        self
    }

    /// Override the content-byte ceiling.
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Override the JSON value ceiling.
    pub fn with_max_json_nodes(mut self, max_json_nodes: usize) -> Self {
        self.max_json_nodes = max_json_nodes;
        self
    }
}

/// Checks the structure every decision protocol relies on, before any
/// protocol-specific check.
///
/// - at least one question, at most [`StructuralLimits::max_questions`];
/// - question ids non-empty and unique;
/// - a choice question has at least one option and at most
///   [`StructuralLimits::max_options`], with non-empty, unique keys;
/// - a score question has at least one level and at most
///   [`StructuralLimits::max_options`];
/// - the request's content bytes and JSON values within
///   [`StructuralLimits::max_bytes`] and
///   [`StructuralLimits::max_json_nodes`].
///
/// JSON content needs no depth, key or number check here:
/// [`OrderedJson`](super::OrderedJson) cannot hold a value deeper than
/// [`MAX_JSON_DEPTH`](super::MAX_JSON_DEPTH), a repeated key or a
/// non-finite number. Whether a model accepts a question's kind, its number
/// of options or levels, missing or present descriptions and instructions,
/// a null state, or media is its protocol's call, checked after this.
///
/// # Errors
///
/// The first problem found, as a [`DecisionError`] naming positions,
/// counts and limits only.
///
/// # Example
///
/// ```
/// use xybrid_core::ir::decision::{validate_structure, StructuralLimits};
/// use xybrid_core::ir::{DecisionError, DecisionOption, DecisionQuestion, DecisionQuestionKind, DecisionRequest, DecisionState};
///
/// let options = vec![DecisionOption::new("a", None), DecisionOption::new("a", None)];
/// let request = DecisionRequest::new(
///     DecisionState::text("a state"),
///     vec![DecisionQuestion::new("q", None, DecisionQuestionKind::choice(options))],
/// );
/// let error = validate_structure(&request, &StructuralLimits::default()).unwrap_err();
/// assert!(matches!(error, DecisionError::DuplicateOptionKey { question: 0, option: 1, first: 0 }));
/// assert_eq!(error.to_string(), "option 1 of question 0 repeats the key of option 0");
/// ```
pub fn validate_structure(
    request: &DecisionRequest,
    limits: &StructuralLimits,
) -> Result<(), DecisionError> {
    let questions = request.questions();
    if questions.is_empty() {
        return Err(DecisionError::NoQuestions);
    }
    if questions.len() > limits.max_questions {
        return Err(DecisionError::TooManyQuestions {
            count: questions.len(),
            max: limits.max_questions,
        });
    }

    let mut totals = Totals::default();
    totals.content(request.state().body());
    let mut first_id = HashMap::with_capacity(questions.len());
    for (question, entry) in questions.iter().enumerate() {
        if entry.id().is_empty() {
            return Err(DecisionError::EmptyQuestionId { question });
        }
        if let Some(&first) = first_id.get(entry.id()) {
            return Err(DecisionError::DuplicateQuestionId { question, first });
        }
        first_id.insert(entry.id(), question);
        totals.bytes(entry.id().len());
        match entry.instructions() {
            Some(DecisionInstructions::Content(content)) => totals.content(Some(content)),
            None => {}
        }

        match entry.kind() {
            DecisionQuestionKind::Choice { options } => {
                if options.is_empty() {
                    return Err(DecisionError::NoOptions { question });
                }
                if options.len() > limits.max_options {
                    return Err(DecisionError::TooManyOptions {
                        question,
                        count: options.len(),
                        max: limits.max_options,
                    });
                }
                let mut first_key = HashMap::with_capacity(options.len());
                for (option, entry) in options.iter().enumerate() {
                    if entry.key().is_empty() {
                        return Err(DecisionError::EmptyOptionKey { question, option });
                    }
                    if let Some(&first) = first_key.get(entry.key()) {
                        return Err(DecisionError::DuplicateOptionKey {
                            question,
                            option,
                            first,
                        });
                    }
                    first_key.insert(entry.key(), option);
                    totals.bytes(entry.key().len());
                    totals.content(entry.description());
                }
            }
            DecisionQuestionKind::YesNo { yes, no } => {
                totals.content(yes.as_ref());
                totals.content(no.as_ref());
            }
            DecisionQuestionKind::Score { levels } => {
                if levels.is_empty() {
                    return Err(DecisionError::NoLevels { question });
                }
                if levels.len() > limits.max_options {
                    return Err(DecisionError::TooManyLevels {
                        question,
                        count: levels.len(),
                        max: limits.max_options,
                    });
                }
                for level in levels {
                    totals.content(level.description());
                }
            }
        }
    }

    if totals.bytes > limits.max_bytes {
        return Err(DecisionError::RequestTooLarge {
            bytes: totals.bytes,
            max: limits.max_bytes,
        });
    }
    if totals.json_nodes > limits.max_json_nodes {
        return Err(DecisionError::TooManyJsonNodes {
            count: totals.json_nodes,
            max: limits.max_json_nodes,
        });
    }
    Ok(())
}

/// Running content totals of one request.
#[derive(Default)]
struct Totals {
    bytes: usize,
    json_nodes: usize,
}

impl Totals {
    fn bytes(&mut self, bytes: usize) {
        self.bytes = self.bytes.saturating_add(bytes);
    }

    fn content(&mut self, content: Option<&DecisionContent>) {
        match content.map(DecisionContent::view) {
            None => {}
            Some(DecisionContentView::Text(text)) => self.bytes(text.len()),
            Some(DecisionContentView::Json(value)) => {
                self.bytes(value.json_text_len());
                self.json_nodes = self.json_nodes.saturating_add(value.node_count());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::decision::{
        DecisionLevel, DecisionOption, DecisionQuestion, DecisionState, OrderedJson,
    };

    const CANARY: &str = "canary-7f3a9c";

    fn json(text: &str) -> OrderedJson {
        OrderedJson::parse(text).unwrap()
    }

    fn choice(id: &str, keys: &[&str]) -> DecisionQuestion {
        let options = keys
            .iter()
            .map(|key| DecisionOption::new(*key, None))
            .collect();
        DecisionQuestion::new(id, None, DecisionQuestionKind::Choice { options })
    }

    fn yes_no(id: &str) -> DecisionQuestion {
        DecisionQuestion::new(
            id,
            None,
            DecisionQuestionKind::YesNo {
                yes: None,
                no: None,
            },
        )
    }

    fn score(id: &str, levels: usize) -> DecisionQuestion {
        DecisionQuestion::new(
            id,
            None,
            DecisionQuestionKind::score(vec![DecisionLevel::new(None); levels]),
        )
    }

    fn check(questions: Vec<DecisionQuestion>) -> Result<(), DecisionError> {
        let request = DecisionRequest::new(DecisionState::text("state"), questions);
        validate_structure(&request, &StructuralLimits::default())
    }

    #[test]
    fn a_well_formed_request_passes() {
        let content = |text: &str| DecisionContent::from_json(json(text));
        let request = DecisionRequest::new(
            DecisionState::from_json(json(r#"{"messages":[{"role":"user","content":"hi"}]}"#)),
            vec![
                DecisionQuestion::new(
                    "pick",
                    Some(DecisionContent::text("Which?").into()),
                    DecisionQuestionKind::Choice {
                        options: vec![
                            DecisionOption::new("a", content(r#"{"why":[1,2]}"#)),
                            DecisionOption::new("b", None),
                        ],
                    },
                ),
                DecisionQuestion::new(
                    "ok",
                    Some(DecisionInstructions::Content(content("[1]").unwrap())),
                    DecisionQuestionKind::YesNo {
                        yes: content(r#""fine""#),
                        no: None,
                    },
                ),
                DecisionQuestion::new(
                    "level",
                    None,
                    DecisionQuestionKind::score(
                        [None, content("0"), content(r#"{"k":null}"#)]
                            .map(DecisionLevel::new)
                            .to_vec(),
                    ),
                ),
            ],
        );
        validate_structure(&request, &StructuralLimits::default()).unwrap();
        // Protocol questions (a null state, one option, no instructions) are
        // not structural.
        let null_state =
            DecisionRequest::new(DecisionState::default(), vec![choice("q", &["only"])]);
        validate_structure(&null_state, &StructuralLimits::default()).unwrap();
    }

    #[test]
    fn a_request_asks_one_to_max_questions() {
        assert!(matches!(check(vec![]), Err(DecisionError::NoQuestions)));
        let questions = |n: usize| (0..n).map(|i| yes_no(&format!("q{i}"))).collect::<Vec<_>>();
        check(questions(DEFAULT_MAX_QUESTIONS)).unwrap();
        assert!(matches!(
            check(questions(DEFAULT_MAX_QUESTIONS + 1)),
            Err(DecisionError::TooManyQuestions { count: 17, max: 16 })
        ));
        let request = DecisionRequest::new(DecisionState::text("s"), questions(3));
        let limits = StructuralLimits::default().with_max_questions(2);
        assert!(matches!(
            validate_structure(&request, &limits),
            Err(DecisionError::TooManyQuestions { count: 3, max: 2 })
        ));
    }

    #[test]
    fn question_ids_are_non_empty_and_unique() {
        assert!(matches!(
            check(vec![yes_no("a"), yes_no("")]),
            Err(DecisionError::EmptyQuestionId { question: 1 })
        ));
        assert!(matches!(
            check(vec![yes_no("a"), yes_no("b"), score("a", 2)]),
            Err(DecisionError::DuplicateQuestionId {
                question: 2,
                first: 0
            })
        ));
    }

    #[test]
    fn option_keys_are_non_empty_and_unique_within_a_question() {
        assert!(matches!(
            check(vec![choice("q", &["a", ""])]),
            Err(DecisionError::EmptyOptionKey {
                question: 0,
                option: 1
            })
        ));
        assert!(matches!(
            check(vec![yes_no("p"), choice("q", &["a", "b", "c", "b"])]),
            Err(DecisionError::DuplicateOptionKey {
                question: 1,
                option: 3,
                first: 1
            })
        ));
        check(vec![choice("q", &["a", "b"]), choice("r", &["a", "b"])]).unwrap();
        check(vec![choice("q", &["a", "A", "a "])]).unwrap();
    }

    #[test]
    fn choices_and_scores_offer_one_to_max_alternatives() {
        assert!(matches!(
            check(vec![choice("q", &[])]),
            Err(DecisionError::NoOptions { question: 0 })
        ));
        assert!(matches!(
            check(vec![yes_no("p"), score("q", 0)]),
            Err(DecisionError::NoLevels { question: 1 })
        ));
        let keys: Vec<String> = (0..=DEFAULT_MAX_OPTIONS).map(|i| format!("k{i}")).collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        check(vec![choice("q", &keys[..DEFAULT_MAX_OPTIONS])]).unwrap();
        assert!(matches!(
            check(vec![choice("q", &keys)]),
            Err(DecisionError::TooManyOptions {
                question: 0,
                count: 256,
                max: 255
            })
        ));
        check(vec![score("q", DEFAULT_MAX_OPTIONS)]).unwrap();
        assert!(matches!(
            check(vec![score("q", DEFAULT_MAX_OPTIONS + 1)]),
            Err(DecisionError::TooManyLevels {
                question: 0,
                count: 256,
                max: 255
            })
        ));
        check(vec![choice("q", &["one"]), score("r", 1)]).unwrap();
    }

    #[test]
    fn content_bytes_count_ids_keys_text_and_compact_json() {
        let request = DecisionRequest::new(
            DecisionState::from_json(json(r#" { "a" : [ 1 , 2.5 ] } "#)),
            vec![
                DecisionQuestion::new(
                    "id",
                    Some(DecisionContent::text("four").into()),
                    DecisionQuestionKind::Choice {
                        options: vec![
                            DecisionOption::new("k", Some(DecisionContent::text("é"))),
                            DecisionOption::new("kk", None),
                        ],
                    },
                ),
                DecisionQuestion::new(
                    "yn",
                    None,
                    DecisionQuestionKind::YesNo {
                        yes: DecisionContent::from_json(json("true")),
                        no: None,
                    },
                ),
                DecisionQuestion::new(
                    "s",
                    None,
                    DecisionQuestionKind::score(vec![
                        DecisionLevel::new(None),
                        DecisionLevel::new(Some(DecisionContent::text("hi"))),
                    ]),
                ),
            ],
        );
        // {"a":[1,2.5]} 13 + "id" 2 + "four" 4 + "k" 1 + "é" 2 + "kk" 2
        // + "yn" 2 + true 4 + "s" 1 + "hi" 2 = 33
        let bytes = 33;
        let exact = StructuralLimits::default().with_max_bytes(bytes);
        validate_structure(&request, &exact).unwrap();
        let short = StructuralLimits::default().with_max_bytes(bytes - 1);
        assert!(matches!(
            validate_structure(&request, &short),
            Err(DecisionError::RequestTooLarge { bytes: 33, max: 32 })
        ));

        // {"a":[1,2.5]} has 4 values and true 1.
        let exact = StructuralLimits::default().with_max_json_nodes(5);
        validate_structure(&request, &exact).unwrap();
        let short = StructuralLimits::default().with_max_json_nodes(4);
        assert!(matches!(
            validate_structure(&request, &short),
            Err(DecisionError::TooManyJsonNodes { count: 5, max: 4 })
        ));
    }

    #[test]
    fn default_limits() {
        let limits = StructuralLimits::default();
        assert_eq!(
            (
                limits.max_questions,
                limits.max_options,
                limits.max_bytes,
                limits.max_json_nodes
            ),
            (16, 255, 1 << 20, 1 << 16)
        );
        let custom = limits
            .with_max_questions(1)
            .with_max_options(2)
            .with_max_bytes(3)
            .with_max_json_nodes(4);
        assert_eq!(
            (
                custom.max_questions,
                custom.max_options,
                custom.max_bytes,
                custom.max_json_nodes
            ),
            (1, 2, 3, 4)
        );
    }

    #[test]
    fn refusals_name_positions_never_content() {
        let duplicate = format!("{CANARY}-key");
        let cases = [
            check(vec![yes_no(CANARY), yes_no(CANARY)]),
            check(vec![choice(CANARY, &[&duplicate, &duplicate])]),
            check(vec![choice(CANARY, &[CANARY, ""])]),
            validate_structure(
                &DecisionRequest::new(DecisionState::text(CANARY), vec![yes_no(CANARY)]),
                &StructuralLimits::default().with_max_bytes(4),
            ),
        ];
        for result in cases {
            let error = result.unwrap_err();
            for text in [error.to_string(), format!("{error:?}")] {
                assert!(!text.contains(CANARY), "{text}");
            }
        }
    }
}
