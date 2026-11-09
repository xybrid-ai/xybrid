//! The decision IR from outside `xybrid-core`, through the SDK's re-exports.
//!
//! The question kinds, answer values, origins and statistics are
//! `#[non_exhaustive]`, and so are the variants of the kinds and values.
//! This file compiles only if code outside the defining crate can still build
//! every variant (constructors) and read every variant (accessors and `{ .. }`
//! patterns, with wildcard arms for what later releases add).

use xybrid_sdk::ir::decision::{validate_structure, StructuralLimits};
use xybrid_sdk::{
    DecisionAnswer, DecisionContent, DecisionLevel, DecisionOption, DecisionOrigin,
    DecisionQuestion, DecisionQuestionKind, DecisionRequest, DecisionScore, DecisionState,
    DecisionUsage, DecisionValue, Decisions, OrderedJson, ScoreStatistic,
};

fn kinds() -> [DecisionQuestionKind; 3] {
    [
        DecisionQuestionKind::choice(vec![
            DecisionOption::new("refund", Some(DecisionContent::text("Refund the order"))),
            DecisionOption::new("send_replacement", None),
        ]),
        DecisionQuestionKind::yes_no(Some(DecisionContent::text("it is damaged")), None),
        DecisionQuestionKind::score(vec![
            DecisionLevel::new(None),
            DecisionLevel::new(Some(DecisionContent::text("severe"))),
        ]),
    ]
}

/// What a caller outside the crate can read from a kind, by pattern.
fn alternatives(kind: &DecisionQuestionKind) -> Option<usize> {
    match kind {
        DecisionQuestionKind::Choice { options, .. } => Some(options.len()),
        DecisionQuestionKind::YesNo { yes, no, .. } => {
            Some(usize::from(yes.is_some()) + usize::from(no.is_some()))
        }
        DecisionQuestionKind::Score { levels, .. } => Some(levels.len()),
        _ => None,
    }
}

#[test]
fn every_question_kind_builds_and_reads_outside_the_crate() {
    let [choice, yes_no, score] = kinds();
    assert_eq!(alternatives(&choice), Some(2));
    assert_eq!(alternatives(&yes_no), Some(1));
    assert_eq!(alternatives(&score), Some(2));

    let options = choice.as_choice().unwrap();
    assert_eq!(options[0].key(), "refund");
    assert!(options[1].description().is_none());
    let (yes, no) = yes_no.as_yes_no().unwrap();
    assert_eq!(
        yes.and_then(DecisionContent::as_text),
        Some("it is damaged")
    );
    assert!(no.is_none());
    let levels = score.as_score().unwrap();
    assert!(levels[0].description().is_none());
    assert_eq!(
        levels[1].description().and_then(DecisionContent::as_text),
        Some("severe")
    );
    assert!(choice.as_score().is_none() && score.as_yes_no().is_none());
    assert_eq!(
        [&choice, &yes_no, &score].map(DecisionQuestionKind::name),
        ["choice", "yes/no", "score"]
    );
}

#[test]
fn every_answer_value_builds_and_reads_outside_the_crate() {
    let values = [
        DecisionValue::choice("refund"),
        DecisionValue::yes_no(0.75),
        DecisionValue::score(1.0, ScoreStatistic::Median),
        DecisionValue::score(0.5, ScoreStatistic::Expected),
    ];
    assert_eq!(values[0].as_choice(), Some("refund"));
    assert_eq!(values[1].as_yes_no(), Some(0.75));
    assert_eq!(values[2].as_score(), Some((1.0, ScoreStatistic::Median)));
    let read: Vec<String> = values
        .iter()
        .map(|value| match value {
            DecisionValue::Choice { key, .. } => format!("choice {key}"),
            DecisionValue::YesNo { p_yes, .. } => format!("yes {p_yes}"),
            DecisionValue::Score {
                value, statistic, ..
            } => match statistic {
                ScoreStatistic::Median => format!("median {value}"),
                ScoreStatistic::Expected => format!("expected {value}"),
                _ => format!("other statistic {value}"),
            },
            _ => "other value".to_string(),
        })
        .collect();
    assert_eq!(
        read,
        ["choice refund", "yes 0.75", "median 1", "expected 0.5"]
    );

    let origin = |origin: DecisionOrigin| match origin {
        DecisionOrigin::Caller => "caller",
        DecisionOrigin::Fixed => "fixed",
        _ => "other",
    };
    assert_eq!(
        [DecisionOrigin::Caller, DecisionOrigin::Fixed].map(origin),
        ["caller", "fixed"]
    );
}

#[test]
fn a_request_and_its_answers_round_trip_outside_the_crate() {
    let state = OrderedJson::parse(r#"{"order": 1042, "notes": ["arrived damaged"]}"#).unwrap();
    let questions = kinds()
        .into_iter()
        .zip(["action", "damaged", "severity"])
        .map(|(kind, id)| {
            DecisionQuestion::new(id, Some(DecisionContent::text("Decide").into()), kind)
        })
        .collect();
    let request = DecisionRequest::new(DecisionState::from_json(state), questions);
    validate_structure(&request, &StructuralLimits::default()).unwrap();
    let text = serde_json::to_string(&request).unwrap();
    assert_eq!(
        serde_json::from_str::<DecisionRequest>(&text).unwrap(),
        request
    );

    let caller = DecisionOrigin::Caller;
    let answers = vec![
        DecisionAnswer::new(
            "action",
            vec![
                DecisionScore::new("refund", 1.0, 0.75, caller),
                DecisionScore::new("send_replacement", 0.0, 0.25, caller),
            ],
            DecisionValue::choice("refund"),
            Some(0.75),
        )
        .unwrap(),
        DecisionAnswer::new(
            "damaged",
            vec![
                DecisionScore::new("true", 2.0, 0.875, caller),
                DecisionScore::new("false", f32::NEG_INFINITY, 0.125, caller),
            ],
            DecisionValue::yes_no(0.875),
            None,
        )
        .unwrap(),
        DecisionAnswer::new(
            "severity",
            vec![
                DecisionScore::new("0", 0.0, 0.375, caller),
                DecisionScore::new("1", 0.5, 0.625, caller),
            ],
            DecisionValue::score(1.0, ScoreStatistic::Median),
            Some(0.625),
        )
        .unwrap(),
    ];
    let decisions = Decisions::new(answers, DecisionUsage::new(3, 96));
    let text = serde_json::to_string(&decisions).unwrap();
    assert!(text.contains(r#""raw_score":"-Infinity""#), "{text}");
    let back: Decisions = serde_json::from_str(&text).unwrap();
    assert_eq!(back, decisions);
    assert_eq!(
        back.answer("severity")
            .map(|answer| answer.value().as_score()),
        Some(Some((1.0, ScoreStatistic::Median)))
    );
}
