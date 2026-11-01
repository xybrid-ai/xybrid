//! `finish` on synthetic plans and readouts.

use crate::execution::decision::facts::Temperature;
use crate::execution::decision::plan::{AnswerPlan, FinalizeRule, LabelGroup, RequestDigest};
use crate::execution::decision::{finish, DecisionPass, DecisionPlan, Token};
use crate::ir::decision::{
    DecisionContent, DecisionError, DecisionLevel, DecisionOption, DecisionOrigin,
    DecisionQuestion, DecisionQuestionKind, DecisionRequest, DecisionState, OrderedJson,
    ScoreStatistic,
};

/// d1-2B's effective temperature, `strtof("1.337398")`.
const D1_T: u32 = 0x3fab2fdc;

/// A question and its plan ([`plan_for`]).
fn single(
    kind: DecisionQuestionKind,
    rule: FinalizeRule,
    keys: &[&str],
    sizes: &[usize],
    t: f32,
) -> (DecisionRequest, DecisionPlan) {
    let question = DecisionQuestion::new("q", Some(DecisionContent::text("x").into()), kind);
    let request = DecisionRequest::new(DecisionState::text("s"), vec![question]);
    let plan = plan_for(&request, rule, keys, sizes, t);
    (request, plan)
}

/// The plan of `request`'s one question: one pass, one token per option
/// unless `sizes` says more, on a 10-token prompt.
fn plan_for(
    request: &DecisionRequest,
    rule: FinalizeRule,
    keys: &[&str],
    sizes: &[usize],
    t: f32,
) -> DecisionPlan {
    let mut select: Vec<Token> = Vec::new();
    let groups = keys
        .iter()
        .zip(sizes)
        .map(|(key, &size)| {
            let start = select.len();
            select.extend((0..size).map(|i| (start + i) as Token));
            LabelGroup {
                key: key.to_string(),
                code: key.to_string(),
                select: start..select.len(),
                origin: DecisionOrigin::Caller,
            }
        })
        .collect();
    DecisionPlan {
        passes: vec![DecisionPass {
            tokens: (0..10).collect(),
            select,
        }],
        answers: vec![AnswerPlan {
            question: 0,
            pass: 0,
            groups,
            rule,
            temperature: Temperature {
                source: None,
                value: t,
            },
        }],
        request: RequestDigest::of(request),
    }
}

fn choice(keys: &[&str], t: f32) -> (DecisionRequest, DecisionPlan) {
    let options = keys
        .iter()
        .map(|key| DecisionOption::new(*key, None))
        .collect();
    single(
        DecisionQuestionKind::choice(options),
        FinalizeRule::Choice,
        keys,
        &vec![1; keys.len()],
        t,
    )
}

fn score(levels: usize, t: f32) -> (DecisionRequest, DecisionPlan) {
    let keys: Vec<String> = (0..levels).map(|i| i.to_string()).collect();
    let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
    single(
        DecisionQuestionKind::score(vec![DecisionLevel::new(None); levels]),
        FinalizeRule::Score,
        &keys,
        &vec![1; levels],
        t,
    )
}

#[test]
fn a_choice_is_the_first_most_probable_option() {
    let (request, plan) = choice(&["a", "b", "c"], 1.0);
    let decisions = finish(&request, &plan, vec![vec![2.0, 2.0, 1.0]]).unwrap();
    let answer = &decisions.answers()[0];
    assert_eq!(answer.value().as_choice(), Some("a"), "first of the tied");
    let p: Vec<f64> = answer
        .distribution()
        .iter()
        .map(|e| e.probability())
        .collect();
    assert_eq!(p[0], p[1]);
    assert_eq!(answer.confidence(), Some(p[0]));
    // exp((s - max) / T): the maximum is subtracted before dividing.
    let w = ((1.0f64 - 2.0) / 1.0).exp();
    assert_eq!(p[2], w / (1.0 + 1.0 + w));
    assert_eq!(answer.question_id(), "q");
}

#[test]
fn yes_no_is_the_probability_of_true_without_confidence() {
    let (request, plan) = single(
        DecisionQuestionKind::yes_no(None, None),
        FinalizeRule::YesNo,
        &["true", "false"],
        &[2, 3],
        f32::from_bits(D1_T),
    );
    // Groups take their highest logit: true 5.0, false 1.0.
    let decisions = finish(&request, &plan, vec![vec![5.0, -1.0, 1.0, 0.5, -3.0]]).unwrap();
    let answer = &decisions.answers()[0];
    let raw: Vec<f32> = answer
        .distribution()
        .iter()
        .map(|e| e.raw_score())
        .collect();
    assert_eq!(raw, [5.0, 1.0]);
    let p_yes = answer.value().as_yes_no().unwrap();
    assert_eq!(p_yes, answer.distribution()[0].probability());
    assert!(p_yes > 0.9);
    assert_eq!(answer.confidence(), None);
}

#[test]
fn a_score_is_the_median_level_and_confidence_the_highest_probability() {
    // Probabilities about [0.45, 0.1, 0.45]: the mode is level 0 (first of
    // the highest), the median level 1.
    let (request, plan) = score(3, 1.0);
    let logits = vec![0.45f64.ln() as f32, 0.1f64.ln() as f32, 0.45f64.ln() as f32];
    let decisions = finish(&request, &plan, vec![logits]).unwrap();
    let answer = &decisions.answers()[0];
    assert_eq!(
        answer.value().as_score(),
        Some((1.0, ScoreStatistic::Median))
    );
    let p0 = answer.distribution()[0].probability();
    assert_eq!(answer.confidence(), Some(p0), "max p, not the median's");
    assert!(answer.distribution()[1].probability() < p0);

    // No level reaches one half until the last: the last.
    let (request, plan) = score(2, 1.0);
    let decisions = finish(&request, &plan, vec![vec![0.0, 0.0]]).unwrap();
    assert_eq!(
        decisions.answers()[0].value().as_score(),
        Some((0.0, ScoreStatistic::Median)),
        "0.5 reaches one half"
    );
}

#[test]
fn the_median_is_read_from_f64_probabilities() {
    // Review B3: float32 logits [0, 1e-8] at d1's temperature give
    // [0.4999999981306987, 0.5000000018693014] in f64, whose median is level
    // 1; rounded to float32 first they would be [0.5, 0.5], level 0.
    let (request, plan) = score(2, f32::from_bits(D1_T));
    let decisions = finish(&request, &plan, vec![vec![0.0, 1e-8]]).unwrap();
    let answer = &decisions.answers()[0];
    let p0 = answer.distribution()[0].probability();
    assert!(p0 < 0.5 && (p0 as f32) == 0.5, "{p0}");
    assert_eq!(
        answer.value().as_score(),
        Some((1.0, ScoreStatistic::Median))
    );
}

#[test]
fn minus_infinity_is_allowed_while_one_score_is_finite() {
    let (request, plan) = choice(&["a", "b"], 1.0);
    let decisions = finish(&request, &plan, vec![vec![f32::NEG_INFINITY, 0.0]]).unwrap();
    let answer = &decisions.answers()[0];
    assert_eq!(answer.distribution()[0].probability(), 0.0);
    assert_eq!(answer.distribution()[0].raw_score(), f32::NEG_INFINITY);
    assert_eq!(answer.value().as_choice(), Some("b"));
    let error = finish(&request, &plan, vec![vec![f32::NEG_INFINITY; 2]]).unwrap_err();
    assert!(
        matches!(error, DecisionError::NonFiniteScores { question: 0 }),
        "{error:?}"
    );
}

#[test]
fn nan_or_plus_infinity_anywhere_in_a_pass_refuses() {
    let (request, plan) = single(
        DecisionQuestionKind::yes_no(None, None),
        FinalizeRule::YesNo,
        &["true", "false"],
        &[2, 1],
        1.0,
    );
    // A NaN that a group maximum would skip (upstream then answers by label
    // order), a NaN maximum, +inf.
    for readout in [
        vec![1.0, f32::NAN, 0.0],
        vec![f32::NAN, 1.0, 0.0],
        vec![1.0, 0.0, f32::INFINITY],
    ] {
        let error = finish(&request, &plan, vec![readout]).unwrap_err();
        assert_eq!(error.kind().map(|k| k.as_str()), Some("non_finite_scores"));
    }
}

#[test]
fn a_token_in_two_groups_scores_both() {
    // Aliases are not reserved: token 7 is option a's alias and option b's
    // code. The readout carries it twice, once per place in `select`.
    let (request, mut plan) = choice(&["a", "b"], 1.0);
    plan.passes[0].select = vec![3, 7, 7];
    plan.answers[0].groups[0].select = 0..2;
    plan.answers[0].groups[1].select = 2..3;
    let decisions = finish(&request, &plan, vec![vec![1.0, 4.0, 4.0]]).unwrap();
    let raw: Vec<f32> = decisions.answers()[0]
        .distribution()
        .iter()
        .map(|e| e.raw_score())
        .collect();
    assert_eq!(raw, [4.0, 4.0]);
    assert_eq!(decisions.answers()[0].value().as_choice(), Some("a"));
}

#[test]
fn readouts_must_match_the_plan() {
    let (request, plan) = choice(&["a", "b"], 1.0);
    assert!(matches!(
        finish(&request, &plan, vec![]),
        Err(DecisionError::ReadoutCount {
            expected: 1,
            found: 0
        })
    ));
    assert!(matches!(
        finish(&request, &plan, vec![vec![0.0, 1.0], vec![0.0, 1.0]]),
        Err(DecisionError::ReadoutCount {
            expected: 1,
            found: 2
        })
    ));
    assert!(matches!(
        finish(&request, &plan, vec![vec![0.0]]),
        Err(DecisionError::ReadoutLength {
            pass: 0,
            expected: 2,
            found: 1
        })
    ));
    let usage = finish(&request, &plan, vec![vec![0.0, 1.0]])
        .unwrap()
        .usage();
    assert_eq!((usage.passes(), usage.evaluated_tokens()), (1, 10));
    assert_eq!(plan.passes()[0].output_index(), 9);
}

#[test]
#[should_panic(expected = "a plan prepared for another request")]
fn a_plan_for_another_request_is_a_bug() {
    let (_, plan) = choice(&["a", "b"], 1.0);
    let (other, _) = choice(&["a", "b", "c"], 1.0);
    let _ = finish(&other, &plan, vec![vec![0.0, 1.0]]);
}

/// Another request of the same shape: the plan's digest tells it apart.
#[test]
#[should_panic(expected = "a plan prepared for another request")]
fn a_plan_for_other_keys_of_the_same_shape_is_a_bug() {
    let (_, plan) = choice(&["a", "b"], 1.0);
    let (other, _) = choice(&["a", "c"], 1.0);
    let _ = finish(&other, &plan, vec![vec![0.0, 1.0]]);
}

#[test]
#[should_panic(expected = "a plan prepared for another request")]
fn a_plan_for_another_state_of_the_same_shape_is_a_bug() {
    let (request, plan) = choice(&["a", "b"], 1.0);
    let (_, questions) = request.into_parts();
    let other = DecisionRequest::new(DecisionState::text("t"), questions);
    let _ = finish(&other, &plan, vec![vec![0.0, 1.0]]);
}

/// The digest is the request's value. A JSON request read back through
/// serde, as JSON or as bincode, is the same request: floats by their bits
/// (`-0.0` included), integers by value, members in order.
#[test]
fn a_json_request_decoded_through_serde_is_the_same_request() {
    let json = |text: &str| OrderedJson::parse(text).expect("JSON");
    let state = r#"{"z": {"b": -0.0, "a": [1.5, 0.1, 42, -7]}, "nested": {"k": {"k": 1e-300}}}"#;
    let options = vec![
        DecisionOption::new(
            "a",
            DecisionContent::from_json(json(r#"{"w": [-0.0, 2, {"deep": 0.5}]}"#)),
        ),
        DecisionOption::new("b", None),
    ];
    let question = DecisionQuestion::new(
        "q",
        DecisionContent::from_json(json(r#"{"ask": "which?", "n": 3}"#)).map(Into::into),
        DecisionQuestionKind::choice(options),
    );
    let request = DecisionRequest::new(DecisionState::from_json(json(state)), vec![question]);
    let plan = plan_for(&request, FinalizeRule::Choice, &["a", "b"], &[1, 1], 1.0);

    let through_json: DecisionRequest =
        serde_json::from_str(&serde_json::to_string(&request).unwrap()).unwrap();
    let through_bincode: DecisionRequest =
        bincode::deserialize(&bincode::serialize(&request).unwrap()).unwrap();
    for decoded in [through_json, through_bincode] {
        assert!(decoded == request);
        assert!(RequestDigest::of(&decoded) == RequestDigest::of(&request));
        let decisions = finish(&decoded, &plan, vec![vec![0.0, 1.0]]).unwrap();
        assert_eq!(decisions.answers()[0].value().as_choice(), Some("b"));
    }

    // One value changed is another request: `0.0` for `-0.0`, a float for
    // an integer, members in another order.
    for other in [
        r#"{"z": {"b": 0.0, "a": [1.5, 0.1, 42, -7]}, "nested": {"k": {"k": 1e-300}}}"#,
        r#"{"z": {"b": -0.0, "a": [1.5, 0.1, 42.0, -7]}, "nested": {"k": {"k": 1e-300}}}"#,
        r#"{"z": {"a": [1.5, 0.1, 42, -7], "b": -0.0}, "nested": {"k": {"k": 1e-300}}}"#,
    ] {
        let (_, questions) = request.clone().into_parts();
        let other = DecisionRequest::new(DecisionState::from_json(json(other)), questions);
        assert!(RequestDigest::of(&other) != RequestDigest::of(&request));
    }
}

#[test]
fn an_equal_request_is_the_same_request() {
    let (request, plan) = choice(&["a", "b"], 1.0);
    let decoded: DecisionRequest =
        bincode::deserialize(&bincode::serialize(&request).unwrap()).unwrap();
    for same in [request.clone(), decoded] {
        assert!(same == request);
        let decisions = finish(&same, &plan, vec![vec![0.0, 1.0]]).unwrap();
        assert_eq!(decisions.answers()[0].value().as_choice(), Some("b"));
    }
}
