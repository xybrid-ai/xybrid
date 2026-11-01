//! The decision conformance cases, stage by stage (G0: preparation and
//! answer arithmetic).
//!
//! The fixture README defines the stages. Each is checked in isolation, from
//! upstream's inputs to it: `finalize` from upstream's plan and raw scores,
//! never from xybrid's own logits, so the arithmetic is held exactly even
//! where logits would differ.
//!
//! | Test | Needs | Checks |
//! |------|-------|--------|
//! | `the_case_census_is_the_one_expected` | — | every case has an expectation per stage; the counts |
//! | `every_case_finalizes_as_upstream` | — | `finish` on upstream's plan and raw scores: answers bit for bit, refusals |
//! | `every_load_stage_is_upstreams` | — | metadata edits on d1-2B's decision metadata: refusals, temperatures |
//! | `prompts_render_as_upstream` | `llm-llamacpp` | `prepare` with d1-2B's spellings and labels, without the tokenizer: prompt bytes, codes, refusals |
//! | `model::every_case_prepares_as_upstream_on_d1_2b` | `llm-llamacpp`, d1-2B | `prepare` with the real tokenizer: everything upstream observed |
//! | `model::metadata_variants_built_by_the_oracle` | `llm-llamacpp`, d1-2B, `decision-oracle` | the metadata cases on the oracle's own variant GGUFs |
//!
//! An answer's `f64`s are held as [`parity`](super::parity) says: bit for bit
//! in the reference environment the fixtures were observed in (Linux x86-64,
//! glibc 2.34, FMA and AVX2), measured and reported elsewhere; zeros, ties
//! and order exactly everywhere, as keys, choices and score levels are.

use std::collections::BTreeMap;

use super::cases::{self, f32_of, f64_of, int, items, Case, Fixture};
use super::fake::{d1_header, GgufWriter};
use super::parity::F64Parity;
use crate::execution::decision::facts::Temperature;
use crate::execution::decision::plan::{AnswerPlan, FinalizeRule, LabelGroup, RequestDigest};
use crate::execution::decision::{
    finish, DecisionLimits, DecisionModelFacts, DecisionPass, DecisionPlan,
};
use crate::gguf::DecisionProfileId;
use crate::ir::decision::{
    DecisionAnswer, DecisionError, DecisionOrigin, DecisionRequest, Decisions,
};

/// How many cases expect `outcome` (`"match"`, `"refuse"` for any refusal,
/// `"not_reached"`, …) at `stage`.
pub(super) fn expected(cases: &[Case], stage: &str, outcome: &str) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.expect(stage).is_some_and(|expectation| {
                expectation == outcome
                    || (outcome == "refuse" && expectation.starts_with("refuse:"))
            })
        })
        .count()
}

#[test]
fn the_case_census_is_the_one_expected() {
    let cases = cases::all();
    assert_eq!(cases.len(), 143);
    for case in &cases {
        assert_eq!(
            case.str_at(&["profile"]),
            Some("lfm3-d1/text/v1"),
            "{}",
            case.id
        );
        assert_eq!(
            case.at(&["schema_version"]).map(int),
            Some(1),
            "{}",
            case.id
        );
        for stage in ["parse", "prepare", "evaluate", "finalize"] {
            assert!(
                case.expect(stage).is_some(),
                "{}: no {stage} expectation",
                case.id
            );
        }
    }
    // The contract's census (C14, with P3a's 143 cases): every case of these
    // stages either passes, refuses as intended, or is legitimately out of
    // scope. The stage tests check each of them; these numbers notice a
    // fixture set that changed under them.
    let count = |stage, outcome| expected(&cases, stage, outcome);
    assert_eq!(count("prepare", "match"), 89);
    assert_eq!(count("prepare", "refuse"), 9);
    assert_eq!(count("prepare", "not_reached"), 45);
    assert_eq!(count("finalize", "match"), 82);
    assert_eq!(count("finalize", "refuse"), 6);
    assert_eq!(count("finalize", "out_of_scope"), 1);
    assert_eq!(count("finalize", "not_reached"), 54);
}

#[test]
fn the_refusal_kinds_are_the_readmes() {
    use crate::ir::decision::DecisionRefusalKind;
    let readme = std::fs::read_to_string(cases::fixtures_dir().join("README.md"))
        .expect("the fixture README");
    let table = readme
        .split("### Refusal kinds")
        .nth(1)
        .and_then(|rest| rest.split("\n### ").next())
        .expect("a refusal-kind table");
    let kinds: Vec<&str> = table
        .lines()
        .filter_map(|line| line.strip_prefix("| `"))
        .filter_map(|line| line.split('`').next())
        .collect();
    let ours: Vec<&str> = DecisionRefusalKind::ALL
        .iter()
        .map(|kind| kind.as_str())
        .collect();
    // The README's kinds are ours: all of them but `request_too_large`,
    // which the table may list too.
    for kind in &kinds {
        assert!(ours.contains(kind), "{kind} is not a DecisionRefusalKind");
    }
    for kind in &ours {
        assert!(
            kinds.contains(kind) || *kind == "request_too_large",
            "{kind} is not in the README's table"
        );
    }
    let mut unique = kinds.clone();
    unique.dedup();
    assert_eq!(unique.len(), kinds.len(), "{kinds:?}");
    assert!(kinds.len() >= 24, "{kinds:?}");
    // Every refusal a case expects is a kind.
    for case in cases::all() {
        for stage in ["load", "parse", "prepare", "evaluate", "finalize"] {
            if let Some(kind) = case
                .expect(stage)
                .and_then(|e| e.strip_prefix("refuse:").map(String::from))
            {
                assert!(ours.contains(&kind.as_str()), "{}: {kind}", case.id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// finalize

/// The plan upstream prepared for `case` (`upstream.prepare`).
pub(super) fn upstream_plan(case: &Case, request: &DecisionRequest) -> DecisionPlan {
    let prepared = case
        .at(&["upstream", "prepare"])
        .expect("upstream prepared the case");
    let passes = items(prepared.get("passes").expect("passes"))
        .iter()
        .map(|pass| DecisionPass {
            tokens: items(pass.get("token_ids").unwrap())
                .iter()
                .map(|t| int(t) as i32)
                .collect(),
            select: items(pass.get("select").unwrap())
                .iter()
                .map(|t| int(t) as i32)
                .collect(),
        })
        .collect();
    let answers = items(prepared.get("answers").expect("answers"))
        .iter()
        .enumerate()
        .map(|(index, answer)| {
            assert_eq!(
                answer.get("question").and_then(Fixture::as_str),
                Some(request.questions()[index].id()),
                "{}: answers in request order",
                case.id
            );
            let rule = match answer.get("type").and_then(Fixture::as_str) {
                Some("choice") => FinalizeRule::Choice,
                Some("noul") => FinalizeRule::YesNo,
                Some("score") => FinalizeRule::Score,
                other => panic!("{}: type {other:?}", case.id),
            };
            let groups = items(answer.get("groups").unwrap())
                .iter()
                .map(|group| {
                    let range = items(group.get("range").unwrap());
                    LabelGroup {
                        key: group
                            .get("key")
                            .and_then(Fixture::as_str)
                            .unwrap()
                            .to_string(),
                        code: group
                            .get("code")
                            .and_then(Fixture::as_str)
                            .unwrap()
                            .to_string(),
                        select: int(&range[0]) as usize..int(&range[1]) as usize,
                        origin: DecisionOrigin::Caller,
                    }
                })
                .collect();
            AnswerPlan {
                question: index,
                pass: int(answer.get("pass").unwrap()) as usize,
                groups,
                rule,
                temperature: Temperature {
                    source: answer
                        .get("temperature")
                        .and_then(Fixture::as_str)
                        .map(str::to_string),
                    value: f32_of(answer.get("temperature_f32").unwrap()),
                },
            }
        })
        .collect();
    DecisionPlan {
        passes,
        answers,
        request: RequestDigest::of(request),
    }
}

/// Upstream's readouts for `case` (`upstream.evaluate.raw_scores`).
pub(super) fn upstream_readouts(case: &Case) -> Vec<Vec<f32>> {
    items(
        case.at(&["upstream", "evaluate", "raw_scores"])
            .expect("raw scores"),
    )
    .iter()
    .map(|pass| items(pass).iter().map(f32_of).collect())
    .collect()
}

/// `decisions` are upstream's `finalize` of `case`: the answer of every
/// question (its value, every probability in order, the confidence, the
/// `f64`s as `parity` holds them), each option's raw score (the group maxima
/// upstream's `format_answer` was given), and the usage.
pub(super) fn assert_upstream_answers(
    case: &Case,
    plan: &DecisionPlan,
    decisions: &Decisions,
    parity: &mut F64Parity,
) {
    let id = &case.id;
    let expected = case
        .at(&["upstream", "finalize", "answers"])
        .and_then(Fixture::as_object)
        .expect("upstream answers");
    assert_eq!(decisions.answers().len(), expected.len(), "{id}");
    let scores = items(
        case.at(&["upstream", "evaluate", "scores"])
            .expect("scores"),
    );
    for (index, ((question, expected), answer)) in
        expected.iter().zip(decisions.answers()).enumerate()
    {
        let what = format!("{id}: question {question}");
        assert_eq!(answer.question_id(), question, "{what}");
        let pass = plan.answers[index].pass;
        let maxima = items(&scores[pass]);
        assert_eq!(answer.distribution().len(), maxima.len(), "{what}");
        for (entry, maximum) in answer.distribution().iter().zip(maxima) {
            assert_eq!(
                entry.raw_score().to_bits(),
                f32_of(maximum).to_bits(),
                "{what}: raw score of option {:?}",
                entry.key()
            );
        }
        assert_answer(&what, expected, answer, parity);
    }
    let usage = decisions.usage();
    assert_eq!(
        usage.evaluated_tokens() as i64,
        int(case
            .at(&["upstream", "finalize", "usage", "input_tokens"])
            .unwrap()),
        "{id}: input tokens"
    );
    assert_eq!(
        case.at(&["upstream", "finalize", "usage", "output_tokens"])
            .map(int),
        Some(0)
    );
    assert_eq!(usage.passes() as usize, plan.passes().len(), "{id}");
}

/// One SystemOne answer against a [`DecisionAnswer`]: compared through
/// independently parsed `f64`s and discrete values, never through
/// `DecisionAnswer` deserialization. Keys, their order, the choice and the
/// score level are exact everywhere; the `f64`s go to `parity`.
fn assert_answer(what: &str, expected: &Fixture, answer: &DecisionAnswer, parity: &mut F64Parity) {
    let number = |key: &str| {
        f64_of(
            expected
                .get(key)
                .unwrap_or_else(|| panic!("{what}: no {key}")),
        )
    };
    // The answer's f64s: (name, upstream's, xybrid's).
    let mut values = Vec::new();
    match expected.get("type").and_then(Fixture::as_str) {
        Some(kind @ ("choice" | "score")) => {
            let listed = expected
                .get("probabilities")
                .and_then(Fixture::as_object)
                .expect("probabilities");
            assert_eq!(listed.len(), answer.distribution().len(), "{what}");
            for ((key, p), entry) in listed.iter().zip(answer.distribution()) {
                assert_eq!(entry.key(), key, "{what}: option order");
                values.push((format!("p({key})"), f64_of(p), entry.probability()));
            }
            values.push((
                "confidence".to_string(),
                number("confidence"),
                answer.confidence().expect("a confidence"),
            ));
            if kind == "choice" {
                assert_eq!(
                    answer.value().as_choice(),
                    expected.get("choice").and_then(Fixture::as_str),
                    "{what}: choice"
                );
            } else {
                let (level, statistic) = answer.value().as_score().expect("a score answer");
                assert_eq!(
                    statistic,
                    crate::ir::decision::ScoreStatistic::Median,
                    "{what}"
                );
                // A level, selected rather than computed: exact everywhere.
                assert_eq!(
                    level.to_bits(),
                    number("score").to_bits(),
                    "{what}: score level {level}"
                );
            }
        }
        Some("noul") => {
            let keys: Vec<&str> = answer.distribution().iter().map(|e| e.key()).collect();
            assert_eq!(keys, ["true", "false"], "{what}");
            assert_eq!(answer.confidence(), None, "{what}: no confidence");
            let p_yes = answer.value().as_yes_no().expect("a yes/no answer");
            assert_eq!(
                p_yes.to_bits(),
                answer.distribution()[0].probability().to_bits(),
                "{what}: the answer is p(true)"
            );
            values.push(("noul".to_string(), number("noul"), p_yes));
        }
        other => panic!("{what}: type {other:?}"),
    }
    parity.answer(what, &values);
}

#[test]
fn every_case_finalizes_as_upstream() {
    let mut checked = BTreeMap::<String, usize>::new();
    let mut parity = F64Parity::new();
    for case in cases::all() {
        let expectation = case.expect("finalize").unwrap();
        if expectation == "not_reached" {
            continue;
        }
        let request = cases::request(case.request_text())
            .unwrap_or_else(|reason| panic!("{}: {reason}", case.id));
        if expectation == "out_of_scope" {
            // `math-media-temperature`: format_answer with media, which the
            // text profile never sees. Its prepare stage is checked.
            assert_eq!(case.id, "math-media-temperature");
            *checked.entry("out_of_scope".into()).or_default() += 1;
            continue;
        }
        let plan = upstream_plan(&case, &request);
        let result = finish(&request, &plan, upstream_readouts(&case));
        match expectation.strip_prefix("refuse:") {
            None => {
                assert_eq!(expectation, "match", "{}", case.id);
                let decisions = result.unwrap_or_else(|error| panic!("{}: {error}", case.id));
                assert_upstream_answers(&case, &plan, &decisions, &mut parity);
                *checked.entry("match".into()).or_default() += 1;
            }
            Some(kind) => {
                let error = result
                    .err()
                    .unwrap_or_else(|| panic!("{}: not refused", case.id));
                assert_eq!(
                    error.kind().map(|k| k.as_str()),
                    Some(kind),
                    "{}: {error}",
                    case.id
                );
                *checked.entry("refuse".into()).or_default() += 1;
            }
        }
    }
    let cases = cases::all();
    for outcome in ["match", "refuse", "out_of_scope"] {
        assert_eq!(
            checked.get(outcome).copied().unwrap_or(0),
            expected(&cases, "finalize", outcome),
            "{outcome}: {checked:?}"
        );
    }
    parity.report("every_case_finalizes_as_upstream");
}

// ---------------------------------------------------------------------------
// load

/// d1-2B's decision metadata with `case`'s metadata edits (`set`, `remove`;
/// token types are the vocabulary's, not the facts').
pub(super) fn edited_header(case: &Case) -> GgufWriter {
    let mut header = d1_header();
    let Some(metadata) = case.at(&["metadata"]) else {
        return header;
    };
    if let Some(set) = metadata.get("set").and_then(Fixture::as_object) {
        for (key, value) in set {
            let value = value.as_str().expect("an edit is a string");
            if let Some(number) = value.strip_prefix("f32:") {
                // The oracle writes `std::stof` of the text: correctly
                // rounded, as Rust's parse is.
                header = header.set_f32(key, number.parse::<f32>().expect("an f32 edit"));
            } else if let Some(text) = value.strip_prefix("str:") {
                header = header.without(key).str(key, text);
            } else {
                panic!("{}: edit {value:?}", case.id);
            }
        }
    }
    if let Some(remove) = metadata.get("remove").and_then(Fixture::as_array) {
        for key in remove {
            header = header.without(key.as_str().expect("a key"));
        }
    }
    header
}

/// The facts `case` prepares with: d1-2B's (edited), with `limits`.
pub(super) fn case_facts(
    case: &Case,
    limits: DecisionLimits,
) -> Result<DecisionModelFacts, DecisionError> {
    DecisionModelFacts::new(
        DecisionProfileId::Lfm3D1Text,
        &edited_header(case).header(),
        limits,
    )
}

/// The limits of `case`'s settings (`artifacts.json`): its token budget.
pub(super) fn case_limits(case: &Case) -> DecisionLimits {
    let artifacts = cases::fixture_file("artifacts.json");
    let settings = case.str_at(&["settings"]).unwrap_or("cpu-t4-fa-off");
    let budget = cases::at(&artifacts, &["settings", settings, "token_budget"])
        .map(int)
        .expect("a token budget");
    DecisionLimits::default().with_token_budget(budget as u32)
}

#[test]
fn every_load_stage_is_upstreams() {
    let mut checked = 0;
    for case in cases::all() {
        let Some(expectation) = case.expect("load") else {
            assert!(
                case.at(&["metadata"]).is_none(),
                "{}: edits need a load stage",
                case.id
            );
            continue;
        };
        let facts = case_facts(&case, DecisionLimits::default());
        match expectation.strip_prefix("refuse:") {
            Some(kind) => {
                let error = facts.err().unwrap_or_else(|| panic!("{}: loads", case.id));
                assert_eq!(
                    error.kind().map(|k| k.as_str()),
                    Some(kind),
                    "{}: {error}",
                    case.id
                );
                assert!(
                    case.at(&["upstream", "load"]).is_some() || kind == "unsupported_metadata",
                    "{}",
                    case.id
                );
            }
            None => {
                assert_eq!(expectation, "match", "{}", case.id);
                let facts = facts.unwrap_or_else(|error| panic!("{}: {error}", case.id));
                // Every question's temperature is upstream's: its string and
                // its value.
                let request = cases::request(case.request_text()).unwrap();
                let answers = items(case.at(&["upstream", "prepare", "answers"]).unwrap());
                for (index, (question, answer)) in
                    request.questions().iter().zip(answers).enumerate()
                {
                    let (type_name, count) = match question.kind() {
                        crate::ir::decision::DecisionQuestionKind::Choice { options } => {
                            ("choice", options.len())
                        }
                        crate::ir::decision::DecisionQuestionKind::YesNo { .. } => ("noul", 2),
                        crate::ir::decision::DecisionQuestionKind::Score { levels } => {
                            ("score", levels.len())
                        }
                    };
                    let temperature = facts.temperatures.lookup(type_name, count);
                    assert_eq!(
                        temperature.source.as_deref(),
                        answer.get("temperature").and_then(Fixture::as_str),
                        "{}: question {index}",
                        case.id
                    );
                    assert_eq!(
                        temperature.value.to_bits(),
                        f32_of(answer.get("temperature_f32").unwrap()).to_bits(),
                        "{}",
                        case.id
                    );
                }
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 11, "the metadata cases");
}

// ---------------------------------------------------------------------------
// prepare, without the tokenizer

/// Upstream's prompts and codes for `case`.
pub(super) fn upstream_prompts(case: &Case) -> Vec<String> {
    items(case.at(&["upstream", "prepare", "passes"]).unwrap())
        .iter()
        .map(|pass| {
            pass.get("prompt")
                .and_then(Fixture::as_str)
                .unwrap()
                .to_string()
        })
        .collect()
}

pub(super) fn upstream_codes(case: &Case) -> Vec<Vec<String>> {
    items(case.at(&["upstream", "prepare", "answers"]).unwrap())
        .iter()
        .map(|answer| {
            items(answer.get("groups").unwrap())
                .iter()
                .map(|group| {
                    group
                        .get("code")
                        .and_then(Fixture::as_str)
                        .unwrap()
                        .to_string()
                })
                .collect()
        })
        .collect()
}

/// The CONTROL spellings of d1-2B, longest first, as the oracle read them.
pub(super) fn d1_control_spellings() -> Vec<String> {
    let oracle = cases::fixture_file("oracle.json");
    let spellings = cases::at(
        &oracle,
        &[
            "artifacts",
            "d1-2b-q4km",
            "control_spellings",
            "longest_first",
        ],
    )
    .expect("the oracle's spellings");
    let spellings: Vec<String> = items(spellings)
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect();
    assert_eq!(spellings.len(), 124);
    spellings
}

/// The parse-stage refusals a request built as data can still carry, which
/// `prepare` must refuse with the same kind.
pub(super) const PROTOCOL_KINDS_AT_PARSE: [&str; 5] = [
    "too_many_options",
    "level_count",
    "missing_instructions",
    "number_out_of_range",
    "invalid_criteria",
];

#[cfg(feature = "llm-llamacpp")]
#[test]
fn prompts_render_as_upstream() {
    use super::fake::FakeVocab;
    use crate::execution::decision::prepare::prepare;

    let spellings = d1_control_spellings();
    let vocab = FakeVocab::new()
        .d1_labels()
        .control(spellings.iter().map(String::as_str));
    // The fake tokenizer counts bytes, not tokens: budgets are the real
    // tokenizer's to check (`model::every_case_prepares_as_upstream_on_d1_2b`).
    let limits = DecisionLimits::default()
        .with_token_budget(u32::MAX)
        .with_max_total_tokens(u64::MAX);
    let mut checked = BTreeMap::<&str, usize>::new();
    let mut skipped = Vec::new();
    for case in cases::all() {
        let parse = case.expect("parse").unwrap();
        let prepare_expectation = case.expect("prepare").unwrap();
        if case.at(&["metadata", "token_types"]).is_some() {
            // The variant's vocabulary: checked by the model tests.
            skipped.push(case.id.clone());
            continue;
        }
        let Ok(facts) = case_facts(&case, limits) else {
            continue; // refused at load (`every_load_stage_is_upstreams`)
        };
        if let Some(kind) = parse.strip_prefix("refuse:") {
            if !PROTOCOL_KINDS_AT_PARSE.contains(&kind) {
                continue; // a transport's refusal
            }
            let Ok(request) = cases::request(case.request_text()) else {
                continue; // not a request the data types can hold
            };
            let error = prepare(&request, &facts, &vocab, None).unwrap_err();
            assert_eq!(
                error.kind().map(|k| k.as_str()),
                Some(kind),
                "{}: {error}",
                case.id
            );
            *checked.entry("parse refusals").or_default() += 1;
            continue;
        }
        let request = cases::request(case.request_text())
            .unwrap_or_else(|reason| panic!("{}: {reason}", case.id));
        let mut prompts = Vec::new();
        let result = prepare(&request, &facts, &vocab, Some(&mut prompts));
        match prepare_expectation.strip_prefix("refuse:") {
            Some("budget_exceeded") => {
                skipped.push(case.id.clone());
            }
            Some(kind) => {
                let error = result
                    .err()
                    .unwrap_or_else(|| panic!("{}: not refused", case.id));
                assert_eq!(
                    error.kind().map(|k| k.as_str()),
                    Some(kind),
                    "{}: {error}",
                    case.id
                );
                *checked.entry("prepare refusals").or_default() += 1;
            }
            None => {
                let plan = result.unwrap_or_else(|error| panic!("{}: {error}", case.id));
                let expected = upstream_prompts(&case);
                assert_eq!(prompts.len(), expected.len(), "{}", case.id);
                for (index, (prompt, expected)) in prompts.iter().zip(&expected).enumerate() {
                    assert!(
                        prompt == expected,
                        "{}: pass {index} differs from byte {}:\n  xybrid:   {:?}\n  upstream: {:?}",
                        case.id,
                        first_difference(prompt, expected),
                        tail_from(prompt, first_difference(prompt, expected)),
                        tail_from(expected, first_difference(prompt, expected)),
                    );
                }
                let codes: Vec<Vec<String>> = plan
                    .answers
                    .iter()
                    .map(|answer| {
                        answer
                            .groups
                            .iter()
                            .map(|group| group.code.clone())
                            .collect()
                    })
                    .collect();
                assert_eq!(codes, upstream_codes(&case), "{}", case.id);
                *checked.entry("prompts").or_default() += 1;
            }
        }
    }
    eprintln!("prompts_render_as_upstream: {checked:?}; left to the model tests: {skipped:?}");
    // Every prepared case but the token-type variant; every prepare-stage
    // refusal but the budget ones (both need the real tokenizer); and the 7
    // parse-stage refusals a request built as data can carry (256 options,
    // 1 and 11 levels, missing and null instructions, an integer above
    // i64::MAX, no options).
    let cases = cases::all();
    let budget = cases
        .iter()
        .filter(|case| case.expect("prepare").as_deref() == Some("refuse:budget_exceeded"))
        .count();
    assert_eq!(
        checked.get("prompts").copied(),
        Some(expected(&cases, "prepare", "match") - 1),
        "{checked:?}"
    );
    assert_eq!(
        checked.get("prepare refusals").copied(),
        Some(expected(&cases, "prepare", "refuse") - budget),
        "{checked:?}"
    );
    assert_eq!(checked.get("parse refusals"), Some(&7), "{checked:?}");
}

pub(super) fn first_difference(a: &str, b: &str) -> usize {
    a.bytes()
        .zip(b.bytes())
        .position(|(x, y)| x != y)
        .unwrap_or(a.len().min(b.len()))
}

pub(super) fn tail_from(text: &str, at: usize) -> String {
    let start = at.saturating_sub(40);
    String::from_utf8_lossy(&text.as_bytes()[start..text.len().min(at + 60)]).into_owned()
}
