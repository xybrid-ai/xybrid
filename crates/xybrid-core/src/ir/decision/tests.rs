//! Cross-module tests: serde round trips, pinned wire forms and content
//! canaries over a whole request and its answers.

use super::*;

const CANARY: &str = "canary-7f3a9c";
const NUMBER_CANARY: &str = "918273645";

fn json(text: &str) -> OrderedJson {
    OrderedJson::parse(text).unwrap()
}

fn content(text: &str) -> Option<DecisionContent> {
    DecisionContent::from_json(json(text))
}

/// A request with caller content in every field that can hold it.
fn canary_request() -> DecisionRequest {
    let structured = format!(
        r#"{{"{CANARY}": ["{CANARY}", {NUMBER_CANARY}, -0, -0.0, 18446744073709551615, 1e300, true, null], "z": {{"{CANARY}": "\u00e9\u20ac"}}}}"#
    );
    DecisionRequest::new(
        DecisionState::from_json(json(&structured)),
        vec![
            DecisionQuestion::new(
                CANARY,
                Some(DecisionContent::text(format!("Which {CANARY}?")).into()),
                DecisionQuestionKind::Choice {
                    options: vec![
                        DecisionOption::new(CANARY, Some(DecisionContent::text(CANARY))),
                        DecisionOption::new(format!("{CANARY}_b"), content(&structured)),
                        DecisionOption::new(format!("{CANARY}_c"), None),
                    ],
                },
            ),
            DecisionQuestion::new(
                format!("{CANARY}-yes-no"),
                Some(DecisionInstructions::Content(
                    content(&format!(r#"{{"ask": "{CANARY}"}}"#)).unwrap(),
                )),
                DecisionQuestionKind::YesNo {
                    yes: Some(DecisionContent::text(CANARY)),
                    no: None,
                },
            ),
            DecisionQuestion::new(
                format!("{CANARY}-score"),
                None,
                DecisionQuestionKind::score(
                    [
                        None,
                        Some(DecisionContent::text(CANARY)),
                        content(&format!("[{NUMBER_CANARY}]")),
                    ]
                    .map(DecisionLevel::new)
                    .to_vec(),
                ),
            ),
        ],
    )
}

/// Answers with caller ids and keys, and every kind of value.
fn canary_decisions() -> Decisions {
    let caller = DecisionOrigin::Caller;
    let choice = DecisionAnswer::new(
        CANARY,
        vec![
            DecisionScore::new(CANARY, 3.25, 0.75, caller),
            DecisionScore::new(format!("{CANARY}_b"), 2.0, 0.25, caller),
            DecisionScore::new(format!("{CANARY}_c"), f32::NEG_INFINITY, 0.0, caller),
        ],
        DecisionValue::Choice { key: CANARY.into() },
        Some(0.75),
    )
    .unwrap();
    let yes_no = DecisionAnswer::new(
        format!("{CANARY}-yes-no"),
        vec![
            DecisionScore::new("true", 0.5, 0.375, caller),
            DecisionScore::new("false", 1.0, 0.625, caller),
        ],
        DecisionValue::YesNo { p_yes: 0.375 },
        None,
    )
    .unwrap();
    let score = DecisionAnswer::new(
        format!("{CANARY}-score"),
        vec![
            DecisionScore::new("0", 0.0, 0.4999999981306987, caller),
            DecisionScore::new("1", 1e-8, 0.5000000018693014, caller),
            DecisionScore::new("2", f32::NEG_INFINITY, 0.0, caller),
        ],
        DecisionValue::Score {
            value: 1.0,
            statistic: ScoreStatistic::Median,
        },
        Some(0.5000000018693014),
    )
    .unwrap();
    Decisions::new(vec![choice, yes_no, score], DecisionUsage::new(3, 1_234))
}

#[test]
fn a_request_round_trips_through_json_and_bincode() {
    let request = canary_request();
    validate_structure(&request, &StructuralLimits::default()).unwrap();

    let text = serde_json::to_string(&request).unwrap();
    let back: DecisionRequest = serde_json::from_str(&text).unwrap();
    assert_eq!(back, request);

    let bytes = bincode::serialize(&request).unwrap();
    let back: DecisionRequest = bincode::deserialize(&bytes).unwrap();
    assert_eq!(back, request);

    // Order and number identity survive both: the state's compact text is
    // unchanged, keys first-to-last as written.
    let state = back
        .state()
        .body()
        .and_then(DecisionContent::as_json)
        .unwrap();
    assert_eq!(
        state.to_json_text(),
        format!(
            r#"{{"{CANARY}":["{CANARY}",{NUMBER_CANARY},0,-0.0,18446744073709551615,1e+300,true,null],"z":{{"{CANARY}":"é€"}}}}"#
        )
    );
}

#[test]
fn answers_round_trip_through_json_and_bincode() {
    let decisions = canary_decisions();

    let text = serde_json::to_string(&decisions).unwrap();
    assert!(text.contains(r#""raw_score":"-Infinity""#), "{text}");
    assert!(!text.contains("null,\"probability\""), "{text}");
    let back: Decisions = serde_json::from_str(&text).unwrap();
    assert_eq!(back, decisions);

    let bytes = bincode::serialize(&decisions).unwrap();
    let back: Decisions = bincode::deserialize(&bytes).unwrap();
    assert_eq!(back, decisions);
    for (answer, original) in back.answers().iter().zip(decisions.answers()) {
        for (entry, original) in answer.distribution().iter().zip(original.distribution()) {
            assert_eq!(entry.raw_score().to_bits(), original.raw_score().to_bits());
            assert_eq!(
                entry.probability().to_bits(),
                original.probability().to_bits()
            );
        }
    }
}

/// The serde forms other packages build against (bindings, envelopes, the
/// CLI's debug output). Changing them is a wire change.
#[test]
fn serde_forms_are_pinned() {
    let request = DecisionRequest::new(
        DecisionState::from_json(json(r#"{"b": 1, "a": -0.0}"#)),
        vec![
            DecisionQuestion::new(
                "q",
                Some(DecisionContent::text("Pick").into()),
                DecisionQuestionKind::Choice {
                    options: vec![
                        DecisionOption::new("x", None),
                        DecisionOption::new("y", content("[1]")),
                    ],
                },
            ),
            DecisionQuestion::new(
                "n",
                None,
                DecisionQuestionKind::YesNo {
                    yes: None,
                    no: content(r#""no""#),
                },
            ),
            DecisionQuestion::new(
                "s",
                None,
                DecisionQuestionKind::score(vec![
                    DecisionLevel::new(None),
                    DecisionLevel::new(Some(DecisionContent::text("high"))),
                ]),
            ),
        ],
    );
    assert_eq!(
        serde_json::to_string(&request).unwrap(),
        concat!(
            r#"{"state":{"body":{"Json":"{\"b\":1,\"a\":-0.0}"},"media":[]},"questions":["#,
            r#"{"id":"q","instructions":{"Content":{"Text":"Pick"}},"kind":{"Choice":{"options":["#,
            r#"{"key":"x","description":null},{"key":"y","description":{"Json":"[1]"}}]}}},"#,
            r#"{"id":"n","instructions":null,"kind":{"YesNo":{"yes":null,"no":{"Text":"no"}}}},"#,
            r#"{"id":"s","instructions":null,"kind":{"Score":{"levels":["#,
            r#"{"description":null},{"description":{"Text":"high"}}]}}}]}"#,
        )
    );

    let answer = DecisionAnswer::new(
        "q",
        vec![
            DecisionScore::new("x", f32::NEG_INFINITY, 0.0, DecisionOrigin::Caller),
            DecisionScore::new("y", 1.5, 1.0, DecisionOrigin::Fixed),
        ],
        DecisionValue::Choice { key: "y".into() },
        Some(1.0),
    )
    .unwrap();
    let decisions = Decisions::new(vec![answer], DecisionUsage::new(1, 12));
    assert_eq!(
        serde_json::to_string(&decisions).unwrap(),
        concat!(
            r#"{"answers":[{"question_id":"q","distribution":["#,
            r#"{"key":"x","raw_score":"-Infinity","probability":0.0,"origin":"Caller"},"#,
            r#"{"key":"y","raw_score":1.5,"probability":1.0,"origin":"Fixed"}],"#,
            r#""value":{"Choice":{"key":"y"}},"confidence":1.0}],"#,
            r#""usage":{"passes":1,"evaluated_tokens":12}}"#,
        )
    );
}

/// bincode writes an enum variant as its index, so reordering variants is
/// a wire change: new variants go last.
#[test]
fn bincode_variant_indexes_are_pinned() {
    fn index(value: &impl serde::Serialize) -> u32 {
        let bytes = bincode::serialize(value).unwrap();
        u32::from_le_bytes(bytes[..4].try_into().unwrap())
    }
    assert_eq!(index(&DecisionContent::text("")), 0);
    assert_eq!(index(&content("[]").unwrap()), 1);
    assert_eq!(
        index(&DecisionInstructions::Content(DecisionContent::text(""))),
        0
    );
    let kinds = [
        DecisionQuestionKind::choice(vec![]),
        DecisionQuestionKind::yes_no(None, None),
        DecisionQuestionKind::score(vec![]),
    ];
    assert_eq!(kinds.map(|kind| index(&kind)), [0, 1, 2]);
    let values = [
        DecisionValue::choice(""),
        DecisionValue::yes_no(0.0),
        DecisionValue::score(0.0, ScoreStatistic::Median),
    ];
    assert_eq!(values.map(|value| index(&value)), [0, 1, 2]);
    assert_eq!(
        [DecisionOrigin::Caller, DecisionOrigin::Fixed].map(|origin| index(&origin)),
        [0, 1]
    );
    assert_eq!(
        [ScoreStatistic::Median, ScoreStatistic::Expected].map(|statistic| index(&statistic)),
        [0, 1]
    );
}

/// `DecisionLevel` is a one-field record, which bincode writes as the field
/// alone: a score question's bytes are what they were when levels were bare
/// `Option<DecisionContent>`s, variant index included.
#[test]
fn score_levels_as_records_keep_their_bincode_bytes() {
    let descriptions = vec![None, Some(DecisionContent::text("high")), content("[1]")];
    let kind = DecisionQuestionKind::score(
        descriptions
            .iter()
            .cloned()
            .map(DecisionLevel::new)
            .collect(),
    );
    let mut before = 2u32.to_le_bytes().to_vec();
    before.extend(bincode::serialize(&descriptions).unwrap());
    assert_eq!(bincode::serialize(&kind).unwrap(), before);
    let back: DecisionQuestionKind = bincode::deserialize(&before).unwrap();
    assert_eq!(back, kind);
}

#[test]
fn constructors_and_accessors_cover_every_variant() {
    let options = vec![DecisionOption::new("a", None)];
    let yes = DecisionContent::text("it is");
    let levels = vec![
        DecisionLevel::new(None),
        DecisionLevel::new(Some(yes.clone())),
    ];
    let choice = DecisionQuestionKind::choice(options.clone());
    let yes_no = DecisionQuestionKind::yes_no(Some(yes.clone()), None);
    let score = DecisionQuestionKind::score(levels.clone());
    assert_eq!(
        choice,
        DecisionQuestionKind::Choice {
            options: options.clone()
        }
    );
    assert_eq!(choice.as_choice(), Some(options.as_slice()));
    assert_eq!(yes_no.as_yes_no(), Some((Some(&yes), None)));
    assert_eq!(score.as_score(), Some(levels.as_slice()));
    for (kind, accessors) in [
        (&choice, [true, false, false]),
        (&yes_no, [false, true, false]),
        (&score, [false, false, true]),
    ] {
        let found = [
            kind.as_choice().is_some(),
            kind.as_yes_no().is_some(),
            kind.as_score().is_some(),
        ];
        assert_eq!(found, accessors, "{}", kind.name());
    }

    let values = [
        DecisionValue::choice("a"),
        DecisionValue::yes_no(0.25),
        DecisionValue::score(1.5, ScoreStatistic::Expected),
    ];
    assert_eq!(values[0], DecisionValue::Choice { key: "a".into() });
    assert_eq!(
        values.each_ref().map(DecisionValue::as_choice),
        [Some("a"), None, None]
    );
    assert_eq!(
        values.each_ref().map(DecisionValue::as_yes_no),
        [None, Some(0.25), None]
    );
    assert_eq!(
        values.each_ref().map(DecisionValue::as_score),
        [None, None, Some((1.5, ScoreStatistic::Expected))]
    );
}

#[test]
fn debug_never_prints_caller_content() {
    let request = canary_request();
    let decisions = canary_decisions();
    let mut rendered = vec![
        format!("{request:?}"),
        format!("{request:#?}"),
        format!("{decisions:?}"),
        format!("{decisions:#?}"),
        format!("{:?}", request.state()),
    ];
    for question in request.questions() {
        rendered.push(format!(
            "{question:?} {:?} {:?}",
            question.kind(),
            question.instructions()
        ));
        if let DecisionQuestionKind::Choice { options } = question.kind() {
            rendered.extend(options.iter().map(|option| format!("{option:?}")));
        }
    }
    for answer in decisions.answers() {
        rendered.push(format!(
            "{answer:?} {:?} {:?}",
            answer.value(),
            answer.distribution()
        ));
    }
    for text in &rendered {
        assert!(!text.contains(CANARY), "{text}");
        assert!(!text.contains(NUMBER_CANARY), "{text}");
        assert!(!text.contains('é') && !text.contains('€'), "{text}");
    }
}

#[test]
fn refusals_of_a_whole_request_never_print_caller_content() {
    let (state, mut questions) = canary_request().into_parts();
    questions.push(questions[0].clone());
    let request = DecisionRequest::new(state, questions);
    let tight = [
        StructuralLimits::default(),
        StructuralLimits::default().with_max_questions(1),
        StructuralLimits::default().with_max_options(1),
        StructuralLimits::default().with_max_bytes(10),
    ];
    for limits in tight {
        let error = validate_structure(&request, &limits).unwrap_err();
        let adapter = crate::runtime_adapter::AdapterError::from(error);
        let text = format!("{adapter} {adapter:?}");
        assert!(!text.contains(CANARY), "{text}");
    }
}

#[cfg(feature = "schema")]
#[test]
fn json_schemas_generate() {
    let request = serde_json::to_value(schemars::schema_for!(DecisionRequest)).unwrap();
    let decisions = serde_json::to_value(schemars::schema_for!(Decisions)).unwrap();
    let text = format!("{request} {decisions}");
    assert!(text.contains("OrderedJson"), "{text}");
    assert!(text.contains("-Infinity"), "{text}");
    let ordered = &request["$defs"]["OrderedJson"];
    assert_eq!(ordered["type"], "string", "{request}");
}
