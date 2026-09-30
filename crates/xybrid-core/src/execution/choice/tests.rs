//! Choice-scoring conformance and contract tests.
//!
//! Inputs come from `integration-tests/fixtures/choice/` (pinned by its
//! `provenance.json`). Groups:
//!
//! - pure rules: candidates, spec validation, encoding, math, error content;
//! - G0: the encoder reproduces every `encoded` entry of `goldens/forms.json`;
//! - fakes: the tiny ONNX scorers in `fake/`, run through ONNX Runtime;
//! - G1: the staged CUA-S1-FORMS export against the PyTorch reference goldens,
//!   within the bounds in `provenance.json` `gates.forms.g1`.
//!
//! Fake and G1 tests skip without ONNX Runtime and G1 without the staged
//! model, unless `XYBRID_REQUIRE_MODELS` is set, where both fail instead.

use super::encode::encode_request;
use super::math::{choice_scores, softmax_scores};
use super::onnx::{ChoiceAxis, OnnxChoiceScorer};
use super::*;
use crate::execution::template::{ByteOverflow, ChoiceScorerSpec, OnnxByteOptionScorerSpec};
use crate::ir::Choice;
use crate::runtime_adapter::onnx::{ExecutionProviderKind, ONNXSession, SessionOptions};
use crate::testing::model_fixtures::{models_required, staged_artifact, ENV_REQUIRE_MODELS};
use serde::Deserialize;
use std::path::PathBuf;

const CANARY: &str = "canary-5d1e0b";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn choice_dir() -> PathBuf {
    // The runtime CARGO_MANIFEST_DIR, not `env!`: under Bazel the test runs in
    // a sandbox where the compile-time path points nowhere.
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set for tests");
    PathBuf::from(manifest_dir).join("../../integration-tests/fixtures/choice")
}

fn read_json<T: for<'de> Deserialize<'de>>(relative: &str) -> T {
    let path = choice_dir().join(relative);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn forms_spec() -> OnnxByteOptionScorerSpec {
    match read_json("specs/cua-s1-forms.json") {
        ChoiceScorerSpec::OnnxByteOptionScorer(spec) => spec,
    }
}

#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    context: String,
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Golden>,
}

#[derive(Deserialize)]
struct Golden {
    id: String,
    candidates: Vec<GoldenCandidate>,
    encoded: GoldenEncoding,
    logits: Vec<f64>,
    scores: Vec<f64>,
    best: usize,
    margin: f64,
}

#[derive(Deserialize)]
struct GoldenCandidate {
    id: String,
    fixed: bool,
}

#[derive(Deserialize)]
struct GoldenEncoding {
    context_ids: Vec<i64>,
    context_truncated: bool,
    choice_ids: Vec<Vec<i64>>,
    choice_truncated: Vec<bool>,
}

#[derive(Deserialize)]
struct Provenance {
    gates: Gates,
}

#[derive(Deserialize)]
struct Gates {
    forms: FormsGates,
}

#[derive(Deserialize)]
struct FormsGates {
    status: String,
    g1: G1Bounds,
}

#[derive(Deserialize)]
struct G1Bounds {
    max_abs_logit: f64,
    max_abs_score: f64,
    near_tie_margin: f64,
}

/// Every FORMS case paired with its golden, in file order.
fn forms_cases() -> Vec<(Case, Golden)> {
    let cases: Cases = read_json("cases/forms.json");
    let goldens: Goldens = read_json("goldens/forms.json");
    assert_eq!(cases.cases.len(), goldens.cases.len());
    let paired: Vec<(Case, Golden)> = cases.cases.into_iter().zip(goldens.cases).collect();
    for (case, golden) in &paired {
        assert_eq!(case.id, golden.id, "cases and goldens out of order");
    }
    paired
}

fn session(path: &std::path::Path) -> ONNXSession {
    ONNXSession::build(
        path.to_str().expect("fixture path is UTF-8"),
        ExecutionProviderKind::Cpu,
        SessionOptions::default(),
    )
    .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

/// Whether ONNX Runtime can run here. Without it the test skips, unless
/// models are required: a missing runtime would skip exactly the tests that
/// `XYBRID_REQUIRE_MODELS` forces, so it fails instead.
fn onnx_runtime_for_test() -> bool {
    if crate::runtime_adapter::onnx::ort_runtime_available() {
        return true;
    }
    assert!(
        !models_required(),
        "{ENV_REQUIRE_MODELS} is set but ONNX Runtime is not loadable; set ORT_DYLIB_PATH"
    );
    eprintln!("Skipping test: ONNX Runtime is not loadable (set ORT_DYLIB_PATH)");
    false
}

/// A fake scorer's session, or `None` to skip without ONNX Runtime.
fn fake(name: &str) -> Option<ONNXSession> {
    onnx_runtime_for_test().then(|| session(&choice_dir().join("fake").join(name)))
}

fn choices(pairs: &[(&str, &str)]) -> Vec<Choice> {
    pairs
        .iter()
        .map(|(id, text)| Choice::new(*id, *text))
        .collect()
}

fn ids(candidates: &[Candidate<'_>]) -> Vec<(String, bool)> {
    candidates
        .iter()
        .map(|c| (c.choice.id.clone(), c.fixed))
        .collect()
}

// ---------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------

#[test]
fn candidates_put_fixed_choices_after_the_callers() {
    let fixed = choices(&[("skip", "skip"), ("click", "click")]);
    let caller = choices(&[("b", "second"), ("a", "first")]);
    let e = effective_candidates(&caller, &fixed, 4).unwrap();
    assert_eq!(
        ids(&e),
        [
            ("b".into(), false),
            ("a".into(), false),
            ("skip".into(), true),
            ("click".into(), true)
        ]
    );
}

#[test]
fn candidates_reject_empty_duplicate_and_reserved_ids() {
    let fixed = choices(&[("skip", "skip"), ("click", "click")]);

    let empty = choices(&[("a", "x"), ("", "y")]);
    assert!(matches!(
        effective_candidates(&empty, &fixed, 64),
        Err(ChoiceError::EmptyChoiceId { index: 1 })
    ));

    let duplicate = choices(&[("a", "x"), ("b", "y"), ("a", "z")]);
    assert!(matches!(
        effective_candidates(&duplicate, &fixed, 64),
        Err(ChoiceError::DuplicateChoiceId { index: 2, first: 0 })
    ));

    let reserved = choices(&[("a", "x"), ("click", "y")]);
    assert!(matches!(
        effective_candidates(&reserved, &fixed, 64),
        Err(ChoiceError::ReservedChoiceId {
            index: 1,
            fixed_index: 1
        })
    ));

    // Two fixed choices sharing an id is a spec bug, reported as a duplicate.
    let clashing_fixed = choices(&[("skip", "a"), ("skip", "b")]);
    assert!(matches!(
        effective_candidates(&[], &clashing_fixed, 64),
        Err(ChoiceError::DuplicateChoiceId { index: 1, first: 0 })
    ));
}

#[test]
fn candidates_count_after_the_fixed_choices_are_appended() {
    let two_fixed = choices(&[("skip", "skip"), ("click", "click")]);
    let one_fixed = choices(&[("skip", "skip")]);

    // Zero caller choices is fine with at least two fixed ones.
    assert_eq!(effective_candidates(&[], &two_fixed, 2).unwrap().len(), 2);
    assert!(matches!(
        effective_candidates(&[], &one_fixed, 64),
        Err(ChoiceError::TooFewCandidates { count: 1 })
    ));
    assert!(matches!(
        effective_candidates(&[], &[], 64),
        Err(ChoiceError::TooFewCandidates { count: 0 })
    ));

    // Two caller choices fit a limit of 3 alone, not with the fixed ones.
    let caller = choices(&[("a", "x"), ("b", "y")]);
    assert_eq!(effective_candidates(&caller, &[], 3).unwrap().len(), 2);
    assert!(matches!(
        effective_candidates(&caller, &two_fixed, 3),
        Err(ChoiceError::TooManyCandidates { count: 4, max: 3 })
    ));
    assert_eq!(
        effective_candidates(&caller, &two_fixed, 4).unwrap().len(),
        4
    );
}

// ---------------------------------------------------------------------------
// Spec validation
// ---------------------------------------------------------------------------

#[test]
fn the_committed_forms_spec_is_valid() {
    let spec = ChoiceScorerSpec::OnnxByteOptionScorer(forms_spec());
    validate_scorer_spec(&spec).unwrap();
}

#[test]
fn spec_validation_rejects_unusable_specs() {
    type Edit = fn(&mut OnnxByteOptionScorerSpec);
    let cases: [(&str, Edit); 13] = [
        ("model_file", |s| s.model_file.clear()),
        ("context.ids_input is empty", |s| {
            s.context.ids_input.clear()
        }),
        ("names the same input", |s| {
            s.choice_mask_input = s.choice.mask_input.clone()
        }),
        ("logits_output", |s| s.logits_output.clear()),
        ("context.offset", |s| s.context.offset = 0),
        ("choice.max_len", |s| s.choice.max_len = 0),
        ("max_choices", |s| s.max_choices = 1),
        ("encode more than", |s| s.choice.max_len = MAX_ENCODED_IDS),
        ("exceed max_choices", |s| s.max_choices = 2),
        ("empty id", |s| s.fixed_choices[1].id.clear()),
        ("repeats the id", |s| {
            s.fixed_choices[2].id = s.fixed_choices[0].id.clone()
        }),
        ("with Reject", |s| {
            s.choice.overflow = ByteOverflow::Reject;
            s.fixed_choices[0].text = "x".repeat(97);
        }),
        ("scoring_temperature", |s| s.scoring_temperature = Some(0.0)),
    ];
    for (expected, edit) in cases {
        let mut spec = forms_spec();
        edit(&mut spec);
        let error = validate_scorer_spec(&ChoiceScorerSpec::OnnxByteOptionScorer(spec))
            .expect_err(expected);
        assert!(
            matches!(&error, ChoiceError::InvalidSpec(message) if message.contains(expected)),
            "{expected}: {error}"
        );
    }
    for bad in [f32::NAN, f32::INFINITY, -1.0] {
        let mut spec = forms_spec();
        spec.scoring_temperature = Some(bad);
        assert!(validate_scorer_spec(&ChoiceScorerSpec::OnnxByteOptionScorer(spec)).is_err());
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

fn small_spec(overflow: ByteOverflow) -> OnnxByteOptionScorerSpec {
    let mut spec = forms_spec();
    spec.context.max_len = 4;
    spec.choice.max_len = 3;
    spec.choice.offset = 2;
    spec.context.overflow = overflow;
    spec.choice.overflow = overflow;
    spec.fixed_choices.clear();
    spec
}

#[test]
fn encoder_cuts_raw_bytes_offsets_them_and_zero_pads() {
    let spec = small_spec(ByteOverflow::Truncate);
    // "é" is C3 A9; "日" is E6 97 A5, cut after two bytes at a limit of 3.
    let offered = choices(&[("a", "é"), ("b", "x日"), ("c", "")]);
    let e = effective_candidates(&offered, &[], 8).unwrap();
    let encoded = encode_request(&spec, "ab", &e, None).unwrap();

    assert_eq!(encoded.context_ids.as_slice().unwrap(), &[98, 99, 0, 0]);
    assert_eq!(
        encoded.context_mask.as_slice().unwrap(),
        &[true, true, false, false]
    );
    assert!(!encoded.context_truncated);
    assert_eq!(
        encoded.choice_ids.as_slice().unwrap(),
        &[
            0xC3 + 2,
            0xA9 + 2,
            0,
            b'x' as i64 + 2,
            0xE6 + 2,
            0x97 + 2,
            0,
            0,
            0
        ]
    );
    assert_eq!(
        encoded.choice_token_mask.as_slice().unwrap(),
        &[true, true, false, true, true, true, false, false, false]
    );
    assert_eq!(encoded.choice_mask.as_slice().unwrap(), &[true; 3]);
    assert_eq!(encoded.choice_truncated, [false, true, false]);
}

#[test]
fn encoder_rejects_an_empty_context() {
    // With every context position masked, the export averages over all 224
    // padding positions while the reference collator pads to one: the logits
    // differ by ~1.5 (measured), so the request is refused instead.
    let spec = small_spec(ByteOverflow::Truncate);
    let offered = choices(&[("a", "x"), ("b", "")]);
    let e = effective_candidates(&offered, &[], 8).unwrap();
    assert!(matches!(
        encode_request(&spec, "", &e, None),
        Err(ChoiceError::EmptyContext)
    ));
    // An empty candidate text is fine: its pooled representation is zero at
    // any padding length.
    assert!(encode_request(&spec, " ", &e, None).is_ok());
}

#[test]
fn encoder_rejects_overlong_fields_under_reject() {
    let spec = small_spec(ByteOverflow::Reject);
    let fits = choices(&[("a", "abc"), ("b", "é")]);
    let e = effective_candidates(&fits, &[], 8).unwrap();
    assert!(encode_request(&spec, "abcd", &e, None).is_ok());
    assert!(matches!(
        encode_request(&spec, "abcde", &e, None),
        Err(ChoiceError::InputTooLong {
            field: ChoiceField::Context,
            len: 5,
            max_len: 4
        })
    ));
    // Three characters, four bytes.
    let long = choices(&[("a", "abc"), ("b", "aéb")]);
    let e = effective_candidates(&long, &[], 8).unwrap();
    assert!(matches!(
        encode_request(&spec, "c", &e, None),
        Err(ChoiceError::InputTooLong {
            field: ChoiceField::Choice(1),
            len: 4,
            max_len: 3
        })
    ));
}

#[test]
fn encoder_masks_the_rows_after_the_candidates() {
    let spec = small_spec(ByteOverflow::Truncate);
    let offered = choices(&[("a", "x"), ("b", "y")]);
    let e = effective_candidates(&offered, &[], 8).unwrap();

    let encoded = encode_request(&spec, "c", &e, Some(4)).unwrap();
    assert_eq!(encoded.choice_ids.shape(), &[1, 4, 3]);
    assert_eq!(
        encoded.choice_mask.as_slice().unwrap(),
        &[true, true, false, false]
    );
    assert!(encoded.choice_ids.as_slice().unwrap()[6..]
        .iter()
        .all(|&id| id == 0));

    assert!(matches!(
        encode_request(&spec, "c", &e, Some(1)),
        Err(ChoiceError::TooManyCandidates { count: 2, max: 1 })
    ));
}

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

#[test]
fn softmax_divides_by_the_temperature() {
    let logits = [2.0_f32, 1.0, -0.5];
    for temperature in [None, Some(0.5), Some(2.0)] {
        let t = f64::from(temperature.unwrap_or(1.0));
        let exps: Vec<f64> = logits.iter().map(|&l| (f64::from(l) / t).exp()).collect();
        let total: f64 = exps.iter().sum();
        let scores = softmax_scores(&logits, temperature).unwrap();
        for (score, exp) in scores.iter().zip(&exps) {
            assert!(
                (f64::from(*score) - exp / total).abs() < 1e-7,
                "{temperature:?}"
            );
        }
    }
    assert!(softmax_scores(&logits, Some(0.0)).is_err());
    assert!(softmax_scores(&logits, Some(f32::NAN)).is_err());
}

#[test]
fn softmax_rejects_empty_and_non_finite_logits() {
    assert!(matches!(
        softmax_scores(&[], None),
        Err(ChoiceError::NoCandidates)
    ));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            softmax_scores(&[0.0, 1.0, bad], None),
            Err(ChoiceError::NonFiniteLogit { index: 2 })
        ));
    }
}

#[test]
fn softmax_is_stable_and_ties_go_to_the_first_candidate() {
    let scores = softmax_scores(&[f32::MAX, f32::MAX, -f32::MAX], None).unwrap();
    assert_eq!(scores, [0.5, 0.5, 0.0]);

    let offered = choices(&[("a", "x"), ("b", "y"), ("c", "z")]);
    let e = effective_candidates(&offered, &[], 8).unwrap();
    let scored = choice_scores(ids(&e), &[1.0, 3.0, 3.0], None).unwrap();
    assert_eq!(scored.best(), Some(1));
    assert_eq!(scored.margin(), Some(0.0));
    assert_eq!(scored.entries[2].id, "c");
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[test]
fn errors_carry_no_caller_content() {
    let fixed = vec![Choice::new(CANARY, "skip")];
    let text = format!("{CANARY} {CANARY} {CANARY}");
    let errors = [
        effective_candidates(&[Choice::new(CANARY, &text)], &fixed, 8).unwrap_err(),
        effective_candidates(
            &[Choice::new(CANARY, &text), Choice::new(CANARY, &text)],
            &[],
            8,
        )
        .unwrap_err(),
        encode_request(
            &small_spec(ByteOverflow::Reject),
            &text,
            &effective_candidates(&choices(&[("a", "x"), ("b", "y")]), &[], 8).unwrap(),
            None,
        )
        .unwrap_err(),
    ];
    let offered = [Choice::new(CANARY, &text), Choice::new("b", &text)];
    let e = effective_candidates(&offered, &[], 8).unwrap();
    let encoded = encode_request(&forms_spec(), &text, &e, None).unwrap();
    let rendered = format!("{encoded:?}");
    assert!(
        !rendered.contains(CANARY) && !rendered.contains("99"),
        "{rendered}"
    );
    if let Some(session) = fake("dynamic.onnx") {
        let scorer = OnnxChoiceScorer::bind(&session, &forms_spec()).unwrap();
        let prepared = scorer.prepare(&text, &offered).unwrap();
        let rendered = format!("{prepared:?}");
        assert!(!rendered.contains(CANARY), "{rendered}");
    }

    for error in errors {
        let rendered = format!("{error} {error:?}");
        let rendered = format!("{rendered} {}", AdapterError::from(error));
        assert!(!rendered.contains(CANARY), "{rendered}");
    }
}

#[test]
fn choice_errors_map_onto_adapter_errors() {
    let request = ChoiceError::TooFewCandidates { count: 1 };
    assert!(matches!(
        AdapterError::from(request),
        AdapterError::InvalidInput(_)
    ));
    let spec = ChoiceError::InvalidSpec("x".into());
    assert!(matches!(
        AdapterError::from(spec),
        AdapterError::InvalidInput(_)
    ));
    let contract = ChoiceError::ModelContract("x".into());
    assert!(matches!(
        AdapterError::from(contract),
        AdapterError::InvalidInput(_)
    ));
    let output = ChoiceError::NonFiniteLogit { index: 0 };
    assert!(matches!(
        AdapterError::from(output),
        AdapterError::InferenceFailed(_)
    ));
    let runtime = ChoiceError::Runtime(AdapterError::ModelNotLoaded("m".into()));
    assert!(matches!(
        AdapterError::from(runtime),
        AdapterError::ModelNotLoaded(_)
    ));
}

// ---------------------------------------------------------------------------
// G0: encoding parity with the reference collator
// ---------------------------------------------------------------------------

#[test]
fn g0_forms_candidates_and_encoding_match_the_goldens() {
    let spec = forms_spec();
    let cases = forms_cases();
    assert_eq!(cases.len(), 13);
    for (case, golden) in &cases {
        let e = effective_candidates(&case.choices, &spec.fixed_choices, spec.max_choices)
            .unwrap_or_else(|error| panic!("{}: {error}", case.id));
        let expected: Vec<(String, bool)> = golden
            .candidates
            .iter()
            .map(|c| (c.id.clone(), c.fixed))
            .collect();
        assert_eq!(ids(&e), expected, "{}: candidate order", case.id);

        let encoded = encode_request(&spec, &case.context, &e, None).unwrap();
        let context = encoded.context_ids.as_slice().unwrap();
        let used = golden.encoded.context_ids.len();
        assert_eq!(&context[..used], golden.encoded.context_ids, "{}", case.id);
        assert!(context[used..].iter().all(|&id| id == 0), "{}", case.id);
        let mask: Vec<bool> = context.iter().map(|&id| id != 0).collect();
        assert_eq!(
            encoded.context_mask.as_slice().unwrap(),
            mask,
            "{}",
            case.id
        );
        assert_eq!(
            encoded.context_truncated, golden.encoded.context_truncated,
            "{}",
            case.id
        );

        assert_eq!(
            encoded.choice_ids.shape(),
            [1, e.len(), spec.choice.max_len]
        );
        for (row, want) in golden.encoded.choice_ids.iter().enumerate() {
            let ids = encoded.choice_ids.slice(ndarray::s![0, row, ..]);
            let ids = ids.as_slice().unwrap();
            assert_eq!(&ids[..want.len()], want.as_slice(), "{} row {row}", case.id);
            assert!(ids[want.len()..].iter().all(|&id| id == 0), "{}", case.id);
            let mask = encoded.choice_token_mask.slice(ndarray::s![0, row, ..]);
            assert!(
                mask.iter().zip(ids).all(|(&set, &id)| set == (id != 0)),
                "{} row {row}: mask",
                case.id
            );
        }
        assert_eq!(
            encoded.choice_truncated, golden.encoded.choice_truncated,
            "{}",
            case.id
        );
        assert!(encoded.choice_mask.iter().all(|&offered| offered));
    }
}

// ---------------------------------------------------------------------------
// Fake scorers: the ONNX contract
// ---------------------------------------------------------------------------

/// The fakes' logit for one encoded candidate row (see
/// `tools/conformance/gen_fake_scorers.py`).
fn fake_logit(context: &str, text: &str, spec: &OnnxByteOptionScorerSpec) -> f64 {
    let sum = |text: &str, max_len: usize, offset: u32| -> f64 {
        text.bytes()
            .take(max_len)
            .map(|b| f64::from(b) + f64::from(offset))
            .sum()
    };
    0.01 * sum(text, spec.choice.max_len, spec.choice.offset)
        + 0.001 * sum(context, spec.context.max_len, spec.context.offset)
}

const FAKE_CONTEXT: &str = "FORM Intake\nELEMENT Edit \"Phone\" value=\"\"";

fn fake_choices() -> Vec<Choice> {
    choices(&[
        ("tel", "fill Tel: (503) 555-0142"),
        ("name", "fill Name: Jane Doe"),
        ("dob", "fill DOB: 03/14/1987"),
    ])
}

fn assert_follows_formula(scores: &crate::ir::ChoiceScores, offered: &[Choice]) {
    let spec = forms_spec();
    let e = effective_candidates(offered, &spec.fixed_choices, spec.max_choices).unwrap();
    assert_eq!(scores.entries.len(), e.len());
    let logits: Vec<f32> = e
        .iter()
        .map(|c| fake_logit(FAKE_CONTEXT, &c.choice.text, &spec) as f32)
        .collect();
    let expected = choice_scores(ids(&e), &logits, None).unwrap();
    for (got, want) in scores.entries.iter().zip(&expected.entries) {
        assert_eq!((&got.id, got.fixed), (&want.id, want.fixed));
        assert!((got.logit - want.logit).abs() < 1e-3, "{got:?} vs {want:?}");
        assert!((got.score - want.score).abs() < 1e-4, "{got:?} vs {want:?}");
    }
}

#[test]
fn fake_dynamic_scores_follow_the_formula_in_candidate_order() {
    let Some(session) = fake("dynamic.onnx") else {
        return;
    };
    let scorer = OnnxChoiceScorer::bind(&session, &forms_spec()).unwrap();
    assert_eq!(scorer.axis(), ChoiceAxis::Dynamic);

    let offered = fake_choices();
    let scores = scorer.score(&session, FAKE_CONTEXT, &offered).unwrap();
    assert_follows_formula(&scores, &offered);
    let ids: Vec<&str> = scores.entries.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["tel", "name", "dob", "check", "click", "skip"]);

    // Reordering the caller's choices reorders the scores with them.
    let mut reordered = offered.clone();
    reordered.reverse();
    let again = scorer.score(&session, FAKE_CONTEXT, &reordered).unwrap();
    assert_follows_formula(&again, &reordered);
    for entry in &scores.entries {
        let moved = again.entries.iter().find(|e| e.id == entry.id).unwrap();
        assert_eq!(moved.logit, entry.logit, "{}", entry.id);
    }
}

#[test]
fn fake_static_axis_drops_its_nan_padding_rows() {
    let Some(session) = fake("static_masked.onnx") else {
        return;
    };
    let mut spec = forms_spec();
    spec.max_choices = 8;
    let scorer = OnnxChoiceScorer::bind(&session, &spec).unwrap();
    assert_eq!(scorer.axis(), ChoiceAxis::Static(8));

    // 3 + 3 fixed candidates in 8 rows: the two NaN rows must not count.
    let offered = fake_choices();
    let prepared = scorer.prepare(FAKE_CONTEXT, &offered).unwrap();
    assert_eq!(prepared.encoded().choice_ids.shape(), &[1, 8, 96]);
    let scores = prepared.run(&session).unwrap();
    assert_follows_formula(&scores, &offered);
    let total: f32 = scores.entries.iter().map(|s| s.score).sum();
    assert!((total - 1.0).abs() < 1e-5);

    spec.max_choices = 9;
    assert!(matches!(
        OnnxChoiceScorer::bind(&session, &spec),
        Err(ChoiceError::ModelContract(message)) if message.contains("8 rows")
    ));
}

#[test]
fn fake_short_output_is_rejected() {
    let Some(session) = fake("short_output.onnx") else {
        return;
    };
    let scorer = OnnxChoiceScorer::bind(&session, &forms_spec()).unwrap();
    let error = scorer
        .score(&session, FAKE_CONTEXT, &fake_choices())
        .unwrap_err();
    assert!(
        matches!(
            &error,
            ChoiceError::OutputShape { expected, actual, .. }
                if expected == &[1, 6] && actual == &[1, 5]
        ),
        "{error}"
    );
}

#[test]
fn fake_float64_output_is_rejected() {
    let (Some(wrong), Some(dynamic)) = (fake("wrong_dtype.onnx"), fake("dynamic.onnx")) else {
        return;
    };
    let spec = forms_spec();
    assert!(matches!(
        OnnxChoiceScorer::bind(&wrong, &spec),
        Err(ChoiceError::ModelContract(message)) if message.contains("float32")
    ));

    // The per-call read refuses it too, never coercing to f32.
    let prepared = OnnxChoiceScorer::bind(&dynamic, &spec)
        .unwrap()
        .prepare(FAKE_CONTEXT, &fake_choices())
        .unwrap();
    assert!(matches!(
        prepared.run(&wrong),
        Err(ChoiceError::Runtime(AdapterError::RuntimeError(message)))
            if message.contains("not float32")
    ));
}

#[test]
fn runtime_failures_withhold_the_backend_message() {
    let (Some(dynamic), Some(fixed)) = (fake("dynamic.onnx"), fake("static_masked.onnx")) else {
        return;
    };
    // Six rows prepared for the dynamic model; the static one wants eight, so
    // ONNX Runtime itself rejects the run with a message quoting shapes.
    let prepared = OnnxChoiceScorer::bind(&dynamic, &forms_spec())
        .unwrap()
        .prepare(FAKE_CONTEXT, &fake_choices())
        .unwrap();
    let error = prepared.run(&fixed).unwrap_err();
    assert!(matches!(error, ChoiceError::InferenceFailed), "{error:?}");
    let rendered = format!("{error} {error:?}");
    let rendered = format!("{rendered} {}", AdapterError::from(error));
    assert!(!rendered.contains("option_ids"), "{rendered}");
    assert!(rendered.contains("withheld"), "{rendered}");
}

#[test]
fn fake_models_reject_specs_naming_missing_or_mistyped_tensors() {
    let Some(session) = fake("dynamic.onnx") else {
        return;
    };
    type Edit = fn(&mut OnnxByteOptionScorerSpec);
    let cases: [(&str, Edit); 4] = [
        ("no output 'scores'", |s| s.logits_output = "scores".into()),
        ("no input 'context_tokens'", |s| {
            s.context.ids_input = "context_tokens".into()
        }),
        ("needs Int64", |s| {
            std::mem::swap(&mut s.context.ids_input, &mut s.context.mask_input)
        }),
        ("fixes axis 1 at 224", |s| s.context.max_len = 256),
    ];
    for (expected, edit) in cases {
        let mut spec = forms_spec();
        edit(&mut spec);
        let error = OnnxChoiceScorer::bind(&session, &spec).expect_err(expected);
        assert!(
            matches!(&error, ChoiceError::ModelContract(message) if message.contains(expected)),
            "{expected}: {error}"
        );
    }

    // The named-output read looks the name up; it never falls back to index 0.
    let spec = forms_spec();
    let prepared = OnnxChoiceScorer::bind(&session, &spec)
        .unwrap()
        .prepare(FAKE_CONTEXT, &fake_choices())
        .unwrap();
    let encoded = prepared.encoded();
    let tensor = |array: ndarray::ArrayD<i64>| -> ort::value::Value {
        ort::value::Tensor::from_array(array).unwrap().into_dyn()
    };
    let flag = |array: ndarray::ArrayD<bool>| -> ort::value::Value {
        ort::value::Tensor::from_array(array).unwrap().into_dyn()
    };
    let inputs = || {
        HashMap::from([
            (
                spec.context.ids_input.clone(),
                tensor(encoded.context_ids.clone().into_dyn()),
            ),
            (
                spec.context.mask_input.clone(),
                flag(encoded.context_mask.clone().into_dyn()),
            ),
            (
                spec.choice.ids_input.clone(),
                tensor(encoded.choice_ids.clone().into_dyn()),
            ),
            (
                spec.choice.mask_input.clone(),
                flag(encoded.choice_token_mask.clone().into_dyn()),
            ),
            (
                spec.choice_mask_input.clone(),
                flag(encoded.choice_mask.clone().into_dyn()),
            ),
        ])
    };
    assert_eq!(
        session
            .run_single_output_f32(inputs(), "logits")
            .unwrap()
            .shape(),
        &[1, 6]
    );
    assert!(matches!(
        session.run_single_output_f32(inputs(), "scores"),
        Err(AdapterError::RuntimeError(message)) if message.contains("no output named 'scores'")
    ));
    assert!(matches!(
        session.run_single_output_f32(inputs(), ""),
        Err(AdapterError::InvalidInput(_))
    ));
}

// ---------------------------------------------------------------------------
// G1: the staged CUA-S1-FORMS export against the reference
// ---------------------------------------------------------------------------

#[test]
fn g1_forms_logits_and_scores_match_the_reference() {
    if !onnx_runtime_for_test() {
        return;
    }
    let Some(model) = staged_artifact("cua-s1-forms", "cua-s1-forms.onnx") else {
        return;
    };
    let gates = read_json::<Provenance>("provenance.json").gates.forms;
    assert_eq!(
        gates.status, "frozen",
        "the FORMS bounds were measured and frozen; loosening them needs a justification, not a status change"
    );
    let bounds = gates.g1;
    let session = session(&model);
    let spec = forms_spec();
    let scorer = OnnxChoiceScorer::bind(&session, &spec).unwrap();
    assert_eq!(scorer.axis(), ChoiceAxis::Dynamic);

    let (mut worst_logit, mut worst_score) = (0.0_f64, 0.0_f64);
    let mut exempt = Vec::new();
    for (case, golden) in forms_cases() {
        let scores = scorer
            .score(&session, &case.context, &case.choices)
            .unwrap_or_else(|error| panic!("{}: {error}", case.id));
        assert_eq!(scores.entries.len(), golden.candidates.len(), "{}", case.id);
        for (index, entry) in scores.entries.iter().enumerate() {
            assert_eq!(entry.id, golden.candidates[index].id, "{}", case.id);
            let logit = (f64::from(entry.logit) - golden.logits[index]).abs();
            let score = (f64::from(entry.score) - golden.scores[index]).abs();
            assert!(
                logit <= bounds.max_abs_logit,
                "{} candidate {index}: |Δlogit| {logit:.3e} > {:.1e}",
                case.id,
                bounds.max_abs_logit
            );
            assert!(
                score <= bounds.max_abs_score,
                "{} candidate {index}: |Δscore| {score:.3e} > {:.1e}",
                case.id,
                bounds.max_abs_score
            );
            worst_logit = worst_logit.max(logit);
            worst_score = worst_score.max(score);
        }
        if golden.margin < bounds.near_tie_margin {
            exempt.push(case.id.clone());
        } else {
            assert_eq!(scores.best(), Some(golden.best), "{}: argmax", case.id);
        }
    }
    eprintln!(
        "forms G1: max |Δlogit| {worst_logit:.3e}, max |Δscore| {worst_score:.3e}; \
         argmax exempt (margin < {}): {exempt:?}",
        bounds.near_tie_margin
    );
    assert_eq!(exempt, ["tie-duplicate-text", "near-tie-telephone"]);
}
