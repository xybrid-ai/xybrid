use super::stand_in::{
    self as ir, DecisionContentView, JsonKind, JsonNumber, OrderedJson, StructuralLimits,
};
use super::*;

const CANARY: &str = "CANARY-7f3a";

/// Insertion order that differs from sorted order, integer-like keys a JS
/// object would reorder, number spellings a host JSON library would rewrite.
const STATE_JSON: &str = r#"{"zeta":1,"alpha":{"2":true,"10":false,"1":null},"mid":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808,9223372036854775808,18446744073709551616,-0],"é":"é\n","nested":[[{"b":1,"a":2}]],"CANARY-7f3a":"CANARY-7f3a"}"#;

/// The same value as [`STATE_JSON`] in compact canonical form.
const STATE_JSON_CANONICAL: &str = r#"{"zeta":1,"alpha":{"2":true,"10":false,"1":null},"mid":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808,9223372036854775808,1.8446744073709552e19,0],"é":"é\n","nested":[[{"b":1,"a":2}]],"CANARY-7f3a":"CANARY-7f3a"}"#;

fn text(value: &str) -> Option<DecisionContent> {
    Some(DecisionContent::Text { text: value.into() })
}

fn json(value: &str) -> Option<DecisionContent> {
    Some(DecisionContent::Json {
        json_text: value.into(),
    })
}

/// Option keys in a scrambled, integer-like order: `0, 7, 14, 21, 28, 5, …`.
fn option_keys() -> Vec<String> {
    (0..30).map(|i| ((i * 7) % 30).to_string()).collect()
}

fn option_description(i: usize) -> Option<DecisionContent> {
    match i % 4 {
        0 => text(&format!("option {i} {CANARY}")),
        1 => json(&format!(r#"{{"z":{i},"a":[{i}.0,"{CANARY}"]}}"#)),
        2 => None,
        _ => json(r#""a JSON string becomes text""#),
    }
}

fn sample_request() -> DecisionRequest {
    DecisionRequest {
        state: json(STATE_JSON),
        questions: vec![
            DecisionQuestion {
                id: "10".into(),
                instructions: Some(DecisionInstructions::Content {
                    content: json(r#"{"task":"pick","why":[3,1,2]}"#).unwrap(),
                }),
                kind: DecisionQuestionKind::Choice {
                    options: option_keys()
                        .into_iter()
                        .enumerate()
                        .map(|(i, key)| DecisionOption {
                            key,
                            description: option_description(i),
                        })
                        .collect(),
                },
            },
            DecisionQuestion {
                id: "2".into(),
                instructions: Some(DecisionInstructions::Content {
                    content: text("Is it urgent?").unwrap(),
                }),
                kind: DecisionQuestionKind::YesNo {
                    yes: json(r#"{"means":"urgent","rank":1}"#),
                    no: None,
                },
            },
            DecisionQuestion {
                id: "1".into(),
                instructions: None,
                kind: DecisionQuestionKind::Score {
                    levels: vec![
                        DecisionLevel { description: None },
                        DecisionLevel {
                            description: text("low"),
                        },
                        DecisionLevel {
                            description: json("[1,2.5]"),
                        },
                        DecisionLevel {
                            description: json(r#""mid""#),
                        },
                        DecisionLevel {
                            description: json("null"),
                        },
                    ],
                },
            },
        ],
    }
}

fn limits() -> StructuralLimits {
    StructuralLimits::default()
}

#[test]
fn a_request_crosses_into_the_ir_with_order_and_number_identity_intact() {
    let request = sample_request().into_ir(&limits()).expect("valid request");

    let Some(state) = request.state().body() else {
        panic!("state must be present")
    };
    let DecisionContentView::Json(state) = state.view() else {
        panic!("state must stay JSON")
    };
    assert_eq!(state.to_json_text(), STATE_JSON_CANONICAL);
    let entries = state.view().entries().expect("object state");
    let keys: Vec<&str> = entries.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, ["zeta", "alpha", "mid", "é", "nested", CANARY]);
    let alpha: Vec<&str> = entries[1]
        .1
        .entries()
        .unwrap()
        .iter()
        .map(|(k, _)| *k)
        .collect();
    assert_eq!(
        alpha,
        ["2", "10", "1"],
        "integer-like keys keep document order"
    );
    let numbers: Vec<JsonNumber> = entries[2]
        .1
        .items()
        .unwrap()
        .iter()
        .map(|v| v.as_number().unwrap())
        .collect();
    assert_eq!(
        numbers,
        [
            JsonNumber::F64(1.0),
            JsonNumber::F64(-0.0),
            JsonNumber::F64(1e300),
            JsonNumber::F64(5e-324),
            JsonNumber::U64(u64::MAX),
            JsonNumber::I64(i64::MIN),
            JsonNumber::U64(9_223_372_036_854_775_808),
            JsonNumber::F64(18_446_744_073_709_551_616.0),
            JsonNumber::I64(0),
        ]
    );
    assert!(
        matches!(numbers[1], JsonNumber::F64(z) if z.is_sign_negative()),
        "-0.0 keeps its sign"
    );

    let ids: Vec<&str> = request.questions().iter().map(|q| q.id()).collect();
    assert_eq!(ids, ["10", "2", "1"], "questions stay in request order");

    let ir::DecisionQuestionKind::Choice { options } = request.questions()[0].kind() else {
        panic!("first question is a choice")
    };
    let keys: Vec<&str> = options.iter().map(|o| o.key()).collect();
    assert_eq!(keys, option_keys(), "30 options keep their order");
    assert!(matches!(
        options[3].description().unwrap().view(),
        DecisionContentView::Text("a JSON string becomes text")
    ));
    assert!(options[2].description().is_none());

    let ir::DecisionQuestionKind::Score { levels } = request.questions()[2].kind() else {
        panic!("third question is a score")
    };
    assert_eq!(levels.len(), 5);
    assert!(
        levels[0].is_none() && levels[4].is_none(),
        "JSON null is absence"
    );
    assert!(matches!(
        levels[3].as_ref().unwrap().view(),
        DecisionContentView::Text("mid")
    ));
}

#[test]
fn a_request_round_trips_to_its_normalized_ffi_form() {
    let request = sample_request().into_ir(&limits()).expect("valid request");
    let back = DecisionRequest::from_ir(&request);

    let mut expected = sample_request();
    expected.state = json(STATE_JSON_CANONICAL);
    if let DecisionQuestionKind::Choice { options } = &mut expected.questions[0].kind {
        for (i, option) in options.iter_mut().enumerate() {
            option.description = match i % 4 {
                1 => json(&format!(r#"{{"z":{i},"a":[{i}.0,"{CANARY}"]}}"#)),
                3 => text("a JSON string becomes text"),
                _ => option.description.clone(),
            };
        }
    }
    if let DecisionQuestionKind::Score { levels } = &mut expected.questions[2].kind {
        levels[2].description = json("[1,2.5]");
        levels[3].description = text("mid");
        levels[4].description = None;
    }
    expected.questions[0].instructions = Some(DecisionInstructions::Content {
        content: json(r#"{"task":"pick","why":[3,1,2]}"#).unwrap(),
    });
    assert_eq!(back, expected);

    // Idempotent: the normalized form converts to the same IR.
    assert_eq!(back.into_ir(&limits()).expect("still valid"), request);
}

#[test]
fn json_numbers_are_classified_as_contract_v2_1_says() {
    let classify = |text: &str| {
        OrderedJson::parse(text, &limits())
            .expect("valid number")
            .view()
            .as_number()
            .unwrap()
    };
    assert_eq!(classify("1"), JsonNumber::I64(1));
    assert_eq!(classify("-0"), JsonNumber::I64(0));
    assert_eq!(classify("9223372036854775807"), JsonNumber::I64(i64::MAX));
    assert_eq!(
        classify("9223372036854775808"),
        JsonNumber::U64(9_223_372_036_854_775_808)
    );
    assert_eq!(classify("18446744073709551615"), JsonNumber::U64(u64::MAX));
    assert_eq!(
        classify("18446744073709551616"),
        JsonNumber::F64(18_446_744_073_709_551_616.0)
    );
    assert_eq!(
        classify("-9223372036854775809"),
        JsonNumber::F64(-9_223_372_036_854_775_809.0)
    );
    assert_eq!(classify("1.0"), JsonNumber::F64(1.0));
    assert_eq!(classify("1E2"), JsonNumber::F64(100.0));
    let JsonNumber::F64(negative_zero) = classify("-0.0") else {
        panic!("-0.0 is a float")
    };
    assert!(negative_zero == 0.0 && negative_zero.is_sign_negative());
}

#[test]
fn invalid_json_is_refused_with_a_content_free_path() {
    let deep = format!("{}{}", "[".repeat(129), "]".repeat(129));
    let cases = [
        (
            r#"{"CANARY-7f3a":1,"CANARY-7f3a":2}"#,
            "duplicate object key",
        ),
        ("1e400", "does not fit a finite float"),
        (deep.as_str(), "depth limit"),
        (r#""\ud800""#, "unpaired UTF-16 surrogate"),
        ("{\"a\":\"CANARY-7f3a\u{1}\"}", "control character"),
        ("[1,2] CANARY-7f3a", "characters after the value"),
        ("[01]", "invalid number"),
        ("", "unexpected end"),
    ];
    for (bad, reason) in cases {
        let mut request = sample_request();
        if let DecisionQuestionKind::Choice { options } = &mut request.questions[0].kind {
            options[29].description = json(bad);
        }
        let error = request.into_ir(&limits()).expect_err(reason);
        let message = error.to_string();
        assert!(matches!(error, Error::InferenceError { .. }), "{message}");
        assert!(
            message.contains("questions[0].options[29].description"),
            "{message}"
        );
        assert!(message.contains(reason), "{message}");
        assert!(!message.contains(CANARY), "{message}");
    }

    let at_depth_limit = format!("{}{}", "[".repeat(128), "]".repeat(128));
    assert!(OrderedJson::parse(&at_depth_limit, &limits()).is_ok());
}

#[test]
fn nested_nulls_stay_json_but_a_null_field_is_absent() {
    let request = DecisionRequest {
        state: json(r#"{"a":null}"#),
        questions: vec![DecisionQuestion {
            id: "q".into(),
            instructions: Some(DecisionInstructions::Content {
                content: json("null").unwrap(),
            }),
            kind: DecisionQuestionKind::YesNo {
                yes: json("null"),
                no: json(r#""no""#),
            },
        }],
    }
    .into_ir(&limits())
    .expect("valid request");

    let DecisionContentView::Json(state) = request.state().body().unwrap().view() else {
        panic!("an object state stays JSON")
    };
    assert_eq!(state.view().entries().unwrap()[0].1.kind(), JsonKind::Null);
    let question = &request.questions()[0];
    assert!(
        question.instructions().is_none(),
        "null instructions are absent"
    );
    let ir::DecisionQuestionKind::YesNo { yes, no } = question.kind() else {
        panic!("yes/no")
    };
    assert!(yes.is_none());
    assert!(matches!(
        no.as_ref().unwrap().view(),
        DecisionContentView::Text("no")
    ));

    let null_state = DecisionRequest {
        state: json("null"),
        ..sample_request()
    }
    .into_ir(&limits())
    .expect("valid request");
    assert!(
        null_state.state().body().is_none(),
        "a null state is no state"
    );
    let empty_state = DecisionRequest {
        state: text(""),
        ..sample_request()
    }
    .into_ir(&limits())
    .expect("valid request");
    assert!(matches!(
        empty_state.state().body().unwrap().view(),
        DecisionContentView::Text("")
    ));
}

#[test]
fn structural_errors_name_indexes_never_ids_or_keys() {
    let with_questions = |questions: Vec<DecisionQuestion>| DecisionRequest {
        state: text("s"),
        questions,
    };
    let choice = |id: &str, keys: &[&str]| DecisionQuestion {
        id: id.into(),
        instructions: None,
        kind: DecisionQuestionKind::Choice {
            options: keys
                .iter()
                .map(|k| DecisionOption {
                    key: (*k).into(),
                    description: None,
                })
                .collect(),
        },
    };
    let cases = [
        (with_questions(vec![]), "no question"),
        (
            with_questions(vec![choice(CANARY, &["a"]), choice(CANARY, &["a"])]),
            "questions[1].id repeats questions[0].id",
        ),
        (
            with_questions(vec![choice("q", &[CANARY, "b", CANARY])]),
            "questions[0].options[2].key repeats options[0].key",
        ),
        (
            with_questions(vec![choice("q", &["a", ""])]),
            "questions[0].options[1].key is empty",
        ),
        (
            with_questions(vec![choice("q", &[])]),
            "questions[0] has no option",
        ),
        (
            with_questions(vec![DecisionQuestion {
                id: "q".into(),
                instructions: None,
                kind: DecisionQuestionKind::Score { levels: vec![] },
            }]),
            "questions[0] has no level",
        ),
    ];
    for (request, expected) in cases {
        let message = request.into_ir(&limits()).unwrap_err().to_string();
        assert!(message.contains(expected), "{message}");
        assert!(!message.contains(CANARY), "{message}");
    }
}

#[test]
fn results_cross_bit_for_bit_including_negative_infinity() {
    let ir_request = sample_request().into_ir(&limits()).expect("valid request");
    let decided = ir::stub_decide(&ir_request);
    let wire = Decisions::from_ir(&decided);

    let ids: Vec<&str> = wire
        .answers
        .iter()
        .map(|a| a.question_id.as_str())
        .collect();
    assert_eq!(ids, ["10", "2", "1"], "answers in request order");
    let keys: Vec<&str> = wire.answers[0]
        .distribution
        .iter()
        .map(|s| s.key.as_str())
        .collect();
    assert_eq!(keys, option_keys(), "distribution in option order");
    let yes_no: Vec<&str> = wire.answers[1]
        .distribution
        .iter()
        .map(|s| s.key.as_str())
        .collect();
    assert_eq!(yes_no, ["true", "false"]);
    assert!(wire.answers[1].confidence.is_none());
    let levels = &wire.answers[2].distribution;
    assert_eq!(levels[0].raw_score, f32::NEG_INFINITY);
    assert_eq!(levels[0].probability.to_bits(), 0.0f64.to_bits());
    for answer in &wire.answers {
        let total: f64 = answer.distribution.iter().map(|s| s.probability).sum();
        assert!((total - 1.0).abs() < 1e-12, "{total}");
    }

    let rebuilt = wire
        .clone()
        .into_ir()
        .expect("a valid result converts back");
    assert_eq!(
        first_bit_difference(&Decisions::from_ir(&rebuilt), &wire),
        None
    );
}

/// Breaks one invariant of a valid result.
type Breaker = fn(&mut Decisions);

#[test]
fn a_result_breaking_an_invariant_is_refused() {
    let ir_request = sample_request().into_ir(&limits()).expect("valid request");
    let good = Decisions::from_ir(&ir::stub_decide(&ir_request));
    let broken: [(Breaker, &str); 6] = [
        (
            |d| d.answers[0].distribution[0].raw_score = f32::NAN,
            "NaN or +inf",
        ),
        (
            |d| d.answers[0].distribution[0].raw_score = f32::INFINITY,
            "NaN or +inf",
        ),
        (
            |d| d.answers[0].distribution[0].probability = 1.5,
            "outside [0, 1]",
        ),
        (
            |d| {
                d.answers[1]
                    .distribution
                    .iter_mut()
                    .for_each(|s| s.raw_score = f32::NEG_INFINITY)
            },
            "every raw score is -inf",
        ),
        (
            |d| d.answers[0].value = DecisionValue::Choice { key: "nope".into() },
            "not a distribution key",
        ),
        (|d| d.answers[0].confidence = Some(f64::NAN), "confidence"),
    ];
    for (breaker, reason) in broken {
        let mut decisions = good.clone();
        breaker(&mut decisions);
        let message = decisions.into_ir().unwrap_err().to_string();
        assert!(message.contains(reason), "{message}");
    }
}

#[test]
fn debug_output_never_shows_content() {
    let request = sample_request();
    let ir_request = request.clone().into_ir(&limits()).expect("valid request");
    let decisions = Decisions::from_ir(&ir::stub_decide(&ir_request));
    for rendered in [
        format!("{request:?}"),
        format!("{ir_request:?}"),
        format!("{decisions:?}"),
        format!("{:?}", ir::stub_decide(&ir_request)),
    ] {
        assert!(!rendered.contains(CANARY), "{rendered}");
        assert!(!rendered.contains("zeta"), "{rendered}");
        assert!(!rendered.contains("Is it urgent"), "{rendered}");
    }
}
