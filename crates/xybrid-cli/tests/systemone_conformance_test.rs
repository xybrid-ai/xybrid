//! The SystemOne transport against the decision fixtures
//! (`integration-tests/fixtures/decision/`, schema version 1).
//!
//! - **Parse**: every case's `xybrid.parse` expectation holds. `match`: the
//!   request parses, its questions are `upstream.parse.questions` (ids,
//!   types, keys, in order), and its content equals what the whole request
//!   reads as through `OrderedJson::parse` (where the request is shallow
//!   enough to be read whole). `{"refuse": kind}`: the request is refused
//!   with that kind, and the message quotes none of its strings.
//!   `not_reached`: an earlier stage (load) refuses, and the request itself
//!   parses.
//! - **Response**: for every case whose `xybrid.finalize` is `match`, the
//!   answers upstream computed, rebuilt as `Decisions`, are written as
//!   exactly `upstream.finalize`'s answers and usage, value for value.
//!
//! Case files are read with `OrderedJson::parse`, which reads floats
//! correctly rounded, so each 17-digit f64 comes back exact (serde_json's
//! default float parser does not promise that).

use std::path::{Path, PathBuf};

use xybrid_cli::systemone::{parse_request, render_response, RefusalKind, RequestLimits};
use xybrid_core::ir::decision::{validate_structure, StructuralLimits};
use xybrid_core::ir::{
    DecisionAnswer, DecisionContent, DecisionInstructions, DecisionLevel, DecisionOption,
    DecisionOrigin, DecisionQuestion, DecisionQuestionKind, DecisionRequest, DecisionScore,
    DecisionState, DecisionUsage, DecisionValue, Decisions, JsonNumber, JsonView, OrderedJson,
    ScoreStatistic,
};

/// The parse census of the current case set: cases whose request must
/// parse, be refused, or is not reached (an earlier stage refuses). Update
/// it when cases are added or their expectations change.
const CENSUS: Census = Census {
    matched: 90,
    refused: 40,
    not_reached: 3,
};

/// Cases whose answers the response is checked against.
const RESPONSE_CASES: usize = 76;

#[derive(Debug, Default, PartialEq, Eq)]
struct Census {
    matched: usize,
    refused: usize,
    not_reached: usize,
}

fn decision_fixtures() -> PathBuf {
    // The runtime CARGO_MANIFEST_DIR, not `env!`: under Bazel the test runs
    // in a sandbox where the compile-time path points nowhere.
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set for tests");
    Path::new(&manifest_dir).join("../../integration-tests/fixtures/decision")
}

/// One case file, read whole.
struct Case {
    id: String,
    json: OrderedJson,
}

impl Case {
    /// `path` through the case's members, which must exist.
    fn at(&self, path: &[&str]) -> &OrderedJson {
        lookup(&self.json, path).unwrap_or_else(|| panic!("{}: no {}", self.id, path.join(".")))
    }

    fn request_text(&self) -> &str {
        self.at(&["request_text"])
            .as_str()
            .expect("request_text is a string")
    }

    /// The expectation of `stage`: `"match"`, `"not_reached"`, ... or
    /// `"refuse:<kind>"`.
    fn expectation(&self, stage: &str) -> Option<String> {
        let value = lookup(&self.json, &["xybrid", stage])?;
        Some(match value.as_str() {
            Some(word) => word.to_string(),
            None => format!(
                "refuse:{}",
                value
                    .get("refuse")
                    .and_then(OrderedJson::as_str)
                    .unwrap_or_else(|| panic!("{}: a {stage} expectation without a kind", self.id))
            ),
        })
    }
}

fn lookup<'a>(value: &'a OrderedJson, path: &[&str]) -> Option<&'a OrderedJson> {
    path.iter().try_fold(value, |value, key| value.get(key))
}

fn cases() -> Vec<Case> {
    let dir = decision_fixtures().join("cases");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no cases in {}", dir.display());
    paths
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let json =
                OrderedJson::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let case = Case {
                id: path.file_stem().unwrap().to_string_lossy().into_owned(),
                json,
            };
            assert_eq!(
                case.at(&["schema_version"]).as_number(),
                Some(JsonNumber::I64(1)),
                "{}",
                case.id
            );
            assert_eq!(case.at(&["id"]).as_str(), Some(case.id.as_str()));
            case
        })
        .collect()
}

fn parse(text: &str) -> Result<DecisionRequest, xybrid_cli::systemone::RequestError> {
    parse_request(text.as_bytes(), &RequestLimits::default())
}

/// A question as `upstream.parse.questions` lists it: id, type, and keys in
/// model order (LFM3 d1 puts `true` before `false`).
fn upstream_shape(question: &DecisionQuestion) -> (String, &'static str, Vec<String>) {
    let kind = question.kind();
    let (name, keys) = if let Some(options) = kind.as_choice() {
        (
            "choice",
            options
                .iter()
                .map(|option| option.key().to_string())
                .collect(),
        )
    } else if kind.as_yes_no().is_some() {
        ("noul", vec!["true".to_string(), "false".to_string()])
    } else if let Some(levels) = kind.as_score() {
        (
            "score",
            (0..levels.len()).map(|level| level.to_string()).collect(),
        )
    } else {
        panic!("a question kind the transport does not build")
    };
    (question.id().to_string(), name, keys)
}

fn expected_shapes(case: &Case) -> Vec<(String, &'static str, Vec<String>)> {
    let questions = case
        .at(&["upstream", "parse", "questions"])
        .as_array()
        .expect("upstream.parse.questions is an array");
    questions
        .iter()
        .map(|question| {
            let id = question.get("id").and_then(OrderedJson::as_str).unwrap();
            let kind = match question.get("type").and_then(OrderedJson::as_str) {
                Some("choice") => "choice",
                Some("noul") => "noul",
                Some("score") => "score",
                other => panic!("{}: question type {other:?}", case.id),
            };
            let keys = question
                .get("keys")
                .and_then(OrderedJson::as_array)
                .unwrap()
                .iter()
                .map(|key| key.as_str().unwrap().to_string())
                .collect();
            (id.to_string(), kind, keys)
        })
        .collect()
}

/// The request as the whole text reads through `OrderedJson::parse`, built
/// without the transport: `None` when the text is too deep to read whole.
fn reference_request(text: &str) -> Option<DecisionRequest> {
    let root = OrderedJson::parse(text).ok()?;
    let content = |value: Option<&OrderedJson>| value.cloned().and_then(DecisionContent::from_json);
    let state = DecisionState::from_json(root.get("state").cloned().unwrap());
    let questions = root
        .get("questions")
        .and_then(OrderedJson::as_object)
        .unwrap()
        .iter()
        .map(|(id, question)| {
            let criteria = question.get("criteria");
            let kind = match question.get("type").and_then(OrderedJson::as_str) {
                Some("choice") => DecisionQuestionKind::choice(
                    criteria
                        .and_then(OrderedJson::as_object)
                        .unwrap()
                        .iter()
                        .map(|(key, description)| {
                            DecisionOption::new(key.as_str(), content(Some(description)))
                        })
                        .collect(),
                ),
                Some("noul") => DecisionQuestionKind::yes_no(
                    content(criteria.and_then(|c| c.get("true"))),
                    content(criteria.and_then(|c| c.get("false"))),
                ),
                Some("score") => DecisionQuestionKind::score(
                    criteria
                        .and_then(OrderedJson::as_array)
                        .unwrap()
                        .iter()
                        .map(|description| DecisionLevel::new(content(Some(description))))
                        .collect(),
                ),
                other => panic!("question type {other:?}"),
            };
            DecisionQuestion::new(
                id.as_str(),
                content(question.get("instructions")).map(DecisionInstructions::from),
                kind,
            )
        })
        .collect();
    Some(DecisionRequest::new(state, questions))
}

/// Every string of a JSON value, keys included.
fn strings(value: &OrderedJson, out: &mut Vec<String>) {
    match value.view() {
        JsonView::String(text) => out.push(text.to_string()),
        JsonView::Array(items) => items.iter().for_each(|item| strings(item, out)),
        JsonView::Object(members) => {
            for (key, value) in members {
                out.push(key.clone());
                strings(value, out);
            }
        }
        _ => {}
    }
}

/// Words the refusals use themselves: the subset's field and type names,
/// and the media part types.
const PROTOCOL_WORDS: &[&str] = &[
    "state",
    "questions",
    "model",
    "type",
    "instructions",
    "criteria",
    "choice",
    "noul",
    "score",
    "true",
    "false",
    "files",
    "images",
    "videos",
    "messages",
    "content",
    "input_audio",
    "image_url",
    "video_url",
];

#[test]
fn every_case_parses_or_refuses_as_expected() {
    let mut census = Census::default();
    let mut reference_checked = 0;
    let mut too_deep_to_read_whole = Vec::new();
    let mut failures = Vec::new();
    for case in cases() {
        let text = case.request_text();
        let expectation = case
            .expectation("parse")
            .unwrap_or_else(|| panic!("{}: no xybrid.parse", case.id));
        let result = parse(text);
        match (expectation.as_str(), &result) {
            ("match", Ok(request)) => {
                census.matched += 1;
                let shapes: Vec<_> = request.questions().iter().map(upstream_shape).collect();
                if shapes != expected_shapes(&case) {
                    failures.push(format!("{}: questions {shapes:?}", case.id));
                }
                if let Err(error) = validate_structure(request, &StructuralLimits::default()) {
                    failures.push(format!("{}: validate_structure: {error}", case.id));
                }
                match reference_request(text) {
                    Some(reference) => {
                        reference_checked += 1;
                        if &reference != request {
                            failures
                                .push(format!("{}: content differs from the reference", case.id));
                        }
                    }
                    None => too_deep_to_read_whole.push(case.id.clone()),
                }
            }
            ("not_reached", Ok(_)) => {
                census.not_reached += 1;
                // Only the load stage comes before parse.
                let load = case.expectation("load").unwrap_or_default();
                if !load.starts_with("refuse:") {
                    failures.push(format!("{}: not reached, yet load is {load:?}", case.id));
                }
            }
            (expected, Err(error)) if expected.starts_with("refuse:") => {
                census.refused += 1;
                let kind = error.kind().as_str();
                if Some(kind) != expected.strip_prefix("refuse:") {
                    failures.push(format!("{}: refused as {kind:?}: {error}", case.id));
                }
                // Content-free: no string of the request in the message.
                let mut words = Vec::new();
                if let Ok(json) = OrderedJson::parse(text) {
                    strings(&json, &mut words);
                }
                let message = format!("{error} {error:?}");
                for word in words {
                    if word.len() >= 3
                        && !PROTOCOL_WORDS.contains(&word.as_str())
                        && message.contains(&word)
                    {
                        failures.push(format!("{}: the refusal quotes {word:?}", case.id));
                    }
                }
            }
            (expected, result) => failures.push(format!(
                "{}: expected {expected}, got {:?}",
                case.id,
                result
                    .as_ref()
                    .map(|_| "a request")
                    .map_err(ToString::to_string)
            )),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(census, CENSUS, "the case set changed; update CENSUS");
    // Only requests nested past 128 levels as a whole cannot be read whole;
    // their content is what the depth cases check.
    assert!(
        too_deep_to_read_whole
            .iter()
            .all(|id| id.starts_with("depth-")),
        "{too_deep_to_read_whole:?}"
    );
    eprintln!(
        "parse: {} matched ({reference_checked} against the whole-text reference), {} refused, \
         {} not reached",
        census.matched, census.refused, census.not_reached
    );
}

#[test]
fn the_parse_kinds_are_the_fixture_kinds() {
    let schema_path = decision_fixtures().join("schema.json");
    let schema = OrderedJson::parse(&std::fs::read_to_string(&schema_path).unwrap()).unwrap();
    let kinds: Vec<&str> = lookup(&schema, &["$defs", "expectation", "oneOf"])
        .and_then(OrderedJson::as_array)
        .and_then(|options| {
            options
                .iter()
                .find_map(|option| lookup(option, &["properties", "refuse", "enum"]))
        })
        .and_then(OrderedJson::as_array)
        .expect("the schema lists the refusal kinds")
        .iter()
        .map(|kind| kind.as_str().unwrap())
        .collect();
    for kind in [
        RefusalKind::InvalidJson,
        RefusalKind::InvalidRequest,
        RefusalKind::DuplicateKey,
        RefusalKind::UnknownField,
        RefusalKind::MissingState,
        RefusalKind::InvalidQuestions,
        RefusalKind::InvalidQuestion,
        RefusalKind::MissingInstructions,
        RefusalKind::InvalidQuestionType,
        RefusalKind::InvalidCriteria,
        RefusalKind::LevelCount,
        RefusalKind::TooManyOptions,
        RefusalKind::EmptyQuestionId,
        RefusalKind::EmptyOptionKey,
        RefusalKind::MediaUnsupported,
        RefusalKind::SpokenInstructions,
        RefusalKind::JsonTooDeep,
        RefusalKind::NumberOutOfRange,
    ] {
        assert!(kinds.contains(&kind.as_str()), "{kind}");
    }
    // `RefusalKind::RequestTooLarge` is the transport's own kind, for its
    // ceilings: no case reaches one, so the schema need not name it.
    // Every kind a case expects at parse is one the transport has.
    for case in cases() {
        if let Some(kind) = case
            .expectation("parse")
            .and_then(|expectation| expectation.strip_prefix("refuse:").map(String::from))
        {
            assert!(
                PARSE_KINDS.contains(&kind.as_str()),
                "{}: parse refuses with {kind}",
                case.id
            );
        }
    }
}

/// The kinds of [`RefusalKind`], by name.
const PARSE_KINDS: &[&str] = &[
    "invalid_json",
    "invalid_request",
    "duplicate_key",
    "unknown_field",
    "missing_state",
    "invalid_questions",
    "invalid_question",
    "missing_instructions",
    "invalid_question_type",
    "invalid_criteria",
    "level_count",
    "too_many_options",
    "empty_question_id",
    "empty_option_key",
    "media_unsupported",
    "spoken_instructions",
    "json_too_deep",
    "number_out_of_range",
];

/// An f64 of the fixtures: a number, or a non-finite value as a string.
fn f64_of(value: &OrderedJson) -> f64 {
    match value.view() {
        JsonView::Number(JsonNumber::F64(value)) => value,
        JsonView::Number(JsonNumber::I64(value)) => value as f64,
        JsonView::String("NaN") => f64::NAN,
        JsonView::String("Infinity") => f64::INFINITY,
        JsonView::String("-Infinity") => f64::NEG_INFINITY,
        other => panic!("not an f64: {other:?}"),
    }
}

/// An f32 of the fixtures: its bit pattern, `"0x3f800000"`.
fn f32_of(value: &OrderedJson) -> f32 {
    let bits = value
        .as_str()
        .and_then(|text| text.strip_prefix("0x"))
        .unwrap();
    f32::from_bits(u32::from_str_radix(bits, 16).unwrap())
}

/// The answers upstream computed for `case`, rebuilt as `Decisions`: the
/// distribution in group order with upstream's group scores and
/// probabilities, upstream's value and confidence. A noul answer gives only
/// p(true); p(false) is its complement, which the response never shows.
fn upstream_decisions(case: &Case, request: &DecisionRequest) -> Decisions {
    let answers = case.at(&["upstream", "finalize", "answers"]);
    let plans = case
        .at(&["upstream", "prepare", "answers"])
        .as_array()
        .unwrap();
    let scores = case
        .at(&["upstream", "evaluate", "scores"])
        .as_array()
        .unwrap();
    let rebuilt = request
        .questions()
        .iter()
        .zip(plans)
        .map(|(question, plan)| {
            let id = question.id();
            assert_eq!(plan.get("question").and_then(OrderedJson::as_str), Some(id));
            let answer = answers
                .get(id)
                .unwrap_or_else(|| panic!("{}: no answer to a question", case.id));
            let pass = match plan.get("pass").and_then(OrderedJson::as_number) {
                Some(JsonNumber::I64(pass)) => usize::try_from(pass).unwrap(),
                other => panic!("{}: pass {other:?}", case.id),
            };
            let group_scores = scores[pass].as_array().unwrap();
            let keys: Vec<&str> = plan
                .get("groups")
                .and_then(OrderedJson::as_array)
                .unwrap()
                .iter()
                .map(|group| group.get("key").and_then(OrderedJson::as_str).unwrap())
                .collect();
            let number = |name: &str| f64_of(answer.get(name).unwrap());
            let kind = answer.get("type").and_then(OrderedJson::as_str).unwrap();
            let value = match kind {
                "choice" => DecisionValue::choice(
                    answer.get("choice").and_then(OrderedJson::as_str).unwrap(),
                ),
                "score" => DecisionValue::score(number("score"), ScoreStatistic::Median),
                "noul" => DecisionValue::yes_no(number("noul")),
                other => panic!("{}: answer type {other}", case.id),
            };
            let probability = |key: &str| match kind {
                "noul" if key == "true" => number("noul"),
                "noul" => 1.0 - number("noul"),
                _ => f64_of(answer.get("probabilities").unwrap().get(key).unwrap()),
            };
            let distribution = keys
                .iter()
                .zip(group_scores)
                .map(|(key, score)| {
                    DecisionScore::new(
                        *key,
                        f32_of(score),
                        probability(key),
                        DecisionOrigin::Caller,
                    )
                })
                .collect();
            let confidence = answer.get("confidence").map(f64_of);
            DecisionAnswer::new(id, distribution, value, confidence)
                .unwrap_or_else(|e| panic!("{}: {e}", case.id))
        })
        .collect();
    let input_tokens = match case
        .at(&["upstream", "finalize", "usage", "input_tokens"])
        .as_number()
    {
        Some(JsonNumber::I64(tokens)) => u64::try_from(tokens).unwrap(),
        other => panic!("{}: input_tokens {other:?}", case.id),
    };
    let passes = case
        .at(&["upstream", "prepare", "passes"])
        .as_array()
        .unwrap()
        .len();
    Decisions::new(
        rebuilt,
        DecisionUsage::new(u32::try_from(passes).unwrap(), input_tokens),
    )
}

#[test]
fn responses_match_upstream_value_for_value() {
    let mut checked = 0;
    let mut failures = Vec::new();
    for case in cases() {
        if case.expectation("finalize").as_deref() != Some("match") {
            continue;
        }
        checked += 1;
        let request = parse(case.request_text())
            .unwrap_or_else(|e| panic!("{}: a finalize case refused at parse: {e}", case.id));
        let decisions = upstream_decisions(&case, &request);
        let text = render_response("d1-2B", &request, &decisions)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        let response = OrderedJson::parse(&text).expect("the response is JSON");
        let members: Vec<&str> = response
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        assert_eq!(members, ["model", "answers", "usage"], "{}", case.id);
        assert_eq!(
            response.get("model").and_then(OrderedJson::as_str),
            Some("d1-2B")
        );
        // Exact equality: members in order, f64 bits, integer and float
        // identity in legends.
        for part in ["answers", "usage"] {
            let expected = case.at(&["upstream", "finalize", part]);
            let written = response.get(part).unwrap();
            if written != expected {
                failures.push(format!(
                    "{}: {part}\n  written:  {}\n  expected: {}",
                    case.id,
                    written.to_json_text(),
                    expected.to_json_text()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(
        checked, RESPONSE_CASES,
        "the case set changed; update RESPONSE_CASES"
    );
    eprintln!("response: {checked} cases written as upstream answered them");
}
