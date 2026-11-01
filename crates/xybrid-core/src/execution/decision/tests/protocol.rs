//! What LFM3 d1 reads (`protocol::check`), and the facts it reads from a
//! GGUF header.

use super::fake::{d1_header, D1_TEMPERATURE_BITS};
use crate::execution::decision::facts::{sha256_hex, SYSTEMONE_TEMPLATE};
use crate::execution::decision::protocol::check;
use crate::execution::decision::{
    DecisionLimits, DecisionModelFacts, ProfileRules, SYSTEMONE_TEMPLATE_SHA256,
};
use crate::gguf::DecisionProfileId;
use crate::ir::decision::{
    ContentPart, DecisionContent, DecisionError, DecisionLevel, DecisionOption, DecisionQuestion,
    DecisionQuestionKind, DecisionRequest, DecisionState, OrderedJson,
};

fn json(text: &str) -> Option<DecisionContent> {
    DecisionContent::from_json(OrderedJson::parse(text).unwrap())
}

fn ask(kind: DecisionQuestionKind) -> DecisionQuestion {
    DecisionQuestion::new("q", Some(DecisionContent::text("Which?").into()), kind)
}

fn request(state: DecisionState, questions: Vec<DecisionQuestion>) -> DecisionRequest {
    DecisionRequest::new(state, questions)
}

fn checked(request: &DecisionRequest) -> Result<(), DecisionError> {
    check(request, &DecisionLimits::default())
}

/// JSON nested `depth` levels deep (counting the value itself).
fn nested(depth: usize) -> String {
    format!("{}0{}", "[".repeat(depth - 1), "]".repeat(depth - 1))
}

#[test]
fn a_score_has_2_to_10_levels() {
    for count in [2, 10] {
        let r = request(
            DecisionState::text("s"),
            vec![ask(DecisionQuestionKind::score(vec![
                DecisionLevel::new(
                    None
                );
                count
            ]))],
        );
        assert!(checked(&r).is_ok(), "{count}");
    }
    for count in [1, 11] {
        let r = request(
            DecisionState::text("s"),
            vec![ask(DecisionQuestionKind::score(vec![
                DecisionLevel::new(
                    None
                );
                count
            ]))],
        );
        assert!(
            matches!(checked(&r), Err(DecisionError::LevelCount { question: 0, count: c, min: 2, max: 10 }) if c == count),
            "{count}"
        );
    }
}

#[test]
fn a_choice_has_at_most_255_options_whatever_the_limits() {
    let options = |count: usize| {
        (0..count)
            .map(|i| DecisionOption::new(format!("k{i}"), None))
            .collect::<Vec<_>>()
    };
    let r = request(
        DecisionState::text("s"),
        vec![ask(DecisionQuestionKind::choice(options(255)))],
    );
    assert!(checked(&r).is_ok());
    let r = request(
        DecisionState::text("s"),
        vec![ask(DecisionQuestionKind::choice(options(256)))],
    );
    assert!(matches!(
        checked(&r),
        Err(DecisionError::TooManyOptions {
            count: 256,
            max: 255,
            ..
        })
    ));
    let wide = DecisionLimits::default().with_max_options(1000);
    assert!(matches!(
        check(&r, &wide),
        Err(DecisionError::TooManyOptions {
            count: 256,
            max: 255,
            ..
        })
    ));
    let narrow = DecisionLimits::default().with_max_options(10);
    let r = request(
        DecisionState::text("s"),
        vec![ask(DecisionQuestionKind::choice(options(11)))],
    );
    assert!(matches!(
        check(&r, &narrow),
        Err(DecisionError::TooManyOptions {
            count: 11,
            max: 10,
            ..
        })
    ));
}

#[test]
fn instructions_are_required_and_questions_bounded() {
    let r = request(
        DecisionState::text("s"),
        vec![DecisionQuestion::new(
            "q",
            None,
            DecisionQuestionKind::yes_no(None, None),
        )],
    );
    assert!(matches!(
        checked(&r),
        Err(DecisionError::MissingInstructions { question: 0 })
    ));
    let questions: Vec<DecisionQuestion> = (0..3)
        .map(|i| {
            DecisionQuestion::new(
                format!("q{i}"),
                Some(DecisionContent::text("x").into()),
                DecisionQuestionKind::yes_no(None, None),
            )
        })
        .collect();
    let r = request(DecisionState::default(), questions);
    assert!(checked(&r).is_ok(), "a null state is accepted");
    assert!(matches!(
        check(&r, &DecisionLimits::default().with_max_questions(2)),
        Err(DecisionError::TooManyQuestions { count: 3, max: 2 })
    ));
}

#[test]
fn json_depth_limits_are_127_for_state_and_instructions_and_125_for_descriptions() {
    // At the limit: accepted.
    let ok = request(
        DecisionState::from_json(OrderedJson::parse(&nested(127)).unwrap()),
        vec![DecisionQuestion::new(
            "q",
            json(&nested(127)).map(Into::into),
            DecisionQuestionKind::choice(vec![DecisionOption::new("a", json(&nested(125)))]),
        )],
    );
    assert!(checked(&ok).is_ok());

    // One level more, part by part.
    let deep_state = request(
        DecisionState::from_json(OrderedJson::parse(&nested(128)).unwrap()),
        vec![ask(DecisionQuestionKind::yes_no(None, None))],
    );
    let deep_instructions = request(
        DecisionState::text("s"),
        vec![DecisionQuestion::new(
            "q",
            json(&nested(128)).map(Into::into),
            DecisionQuestionKind::yes_no(None, None),
        )],
    );
    let deep_option = request(
        DecisionState::text("s"),
        vec![ask(DecisionQuestionKind::choice(vec![
            DecisionOption::new("a", None),
            DecisionOption::new("b", json(&nested(126))),
        ]))],
    );
    let deep_no = request(
        DecisionState::text("s"),
        vec![ask(DecisionQuestionKind::yes_no(
            json(&nested(125)),
            json(&nested(126)),
        ))],
    );
    let deep_level = request(
        DecisionState::text("s"),
        vec![ask(DecisionQuestionKind::score(vec![
            DecisionLevel::new(json(&nested(126))),
            DecisionLevel::new(None),
        ]))],
    );
    for (r, part, depth, max) in [
        (deep_state, ContentPart::State, 128, 127),
        (
            deep_instructions,
            ContentPart::Instructions { question: 0 },
            128,
            127,
        ),
        (
            deep_option,
            ContentPart::OptionDescription {
                question: 0,
                option: 1,
            },
            126,
            125,
        ),
        (
            deep_no,
            ContentPart::YesNoDescription {
                question: 0,
                yes: false,
            },
            126,
            125,
        ),
        (
            deep_level,
            ContentPart::LevelDescription {
                question: 0,
                level: 0,
            },
            126,
            125,
        ),
    ] {
        let error = checked(&r).unwrap_err();
        assert!(
            matches!(error, DecisionError::ContentTooDeep { part: p, depth: d, max: m } if p == part && d == depth && m == max),
            "{error:?}"
        );
        assert_eq!(error.kind().map(|k| k.as_str()), Some("json_too_deep"));
    }
}

#[test]
fn integers_above_i64_max_are_refused_anywhere() {
    let big = r#"{"a": [1, {"b": 18446744073709551615}]}"#;
    let r = request(
        DecisionState::text("s"),
        vec![ask(DecisionQuestionKind::score(vec![
            DecisionLevel::new(None),
            DecisionLevel::new(json(big)),
        ]))],
    );
    assert!(matches!(
        checked(&r),
        Err(DecisionError::NumberOutOfRange {
            part: ContentPart::LevelDescription {
                question: 0,
                level: 1
            }
        })
    ));
    // i64::MAX itself, and integers past u64::MAX (floats), are fine.
    for text in [
        "9223372036854775807",
        "-9223372036854775808",
        "18446744073709551616",
        "1e300",
    ] {
        let r = request(
            DecisionState::from_json(OrderedJson::parse(text).unwrap()),
            vec![ask(DecisionQuestionKind::yes_no(None, None))],
        );
        assert!(checked(&r).is_ok(), "{text}");
    }
}

#[test]
fn the_pinned_template_is_the_one_embedded() {
    assert_eq!(SYSTEMONE_TEMPLATE.len(), 1380);
    assert!(SYSTEMONE_TEMPLATE.ends_with("<think></think>"));
    assert_eq!(
        sha256_hex(SYSTEMONE_TEMPLATE.as_bytes()),
        SYSTEMONE_TEMPLATE_SHA256
    );
    let oracle = super::cases::fixture_file("oracle.json");
    assert_eq!(
        super::cases::at(&oracle, &["artifacts", "d1-2b-q4km", "template"])
            .and_then(super::cases::Fixture::as_str),
        Some(SYSTEMONE_TEMPLATE)
    );
}

fn facts(header: &super::fake::GgufWriter) -> Result<DecisionModelFacts, DecisionError> {
    DecisionModelFacts::new(
        DecisionProfileId::Lfm3D1Text,
        &header.header(),
        DecisionLimits::default(),
    )
}

#[test]
fn facts_need_the_profile_the_template_and_valid_temperatures() {
    let facts_of = |header| {
        facts(&header)
            .map(|_| ())
            .map_err(|error| error.kind().map(|k| k.as_str()))
    };
    assert_eq!(facts_of(d1_header()), Ok(()));
    // Not a decision model; another protocol; causal attention; no
    // template.
    let not_decision = super::fake::GgufWriter::new().str("general.architecture", "lfm3");
    assert!(matches!(
        facts(&not_decision),
        Err(DecisionError::ProfileMismatch)
    ));
    let omni = d1_header()
        .without("lfm3.decision.type")
        .str("lfm3.decision.type", "lfm3-d1-omni");
    assert!(matches!(facts(&omni), Err(DecisionError::ProfileMismatch)));
    let causal = d1_header().without("lfm3.attention.causal");
    assert!(matches!(
        facts(&causal),
        Err(DecisionError::ProfileMismatch)
    ));
    let no_template = d1_header().without("tokenizer.chat_template.systemone");
    assert!(matches!(
        facts(&no_template),
        Err(DecisionError::ProfileMismatch)
    ));
    // Another template: a capability this build lacks.
    let other = d1_header()
        .without("tokenizer.chat_template.systemone")
        .str(
            "tokenizer.chat_template.systemone",
            &format!("{SYSTEMONE_TEMPLATE}\n"),
        );
    assert_eq!(facts_of(other), Err(Some("unsupported_metadata")));
    assert!(matches!(
        facts(
            &d1_header()
                .without("tokenizer.chat_template.systemone")
                .str("tokenizer.chat_template.systemone", "x")
        ),
        Err(DecisionError::UnsupportedTemplate)
    ));
    // A score bias, which this build does not apply.
    let bias = d1_header().set_f32("lfm3.decision.score_bias.3.0", 0.5);
    assert!(matches!(
        facts(&bias),
        Err(DecisionError::UnsupportedMetadata)
    ));
    // Temperatures upstream refuses: zero or less after the string round trip.
    for value in [-1.0, 0.0, 4e-7] {
        let bad = d1_header().set_f32("lfm3.decision.temperature.choice", value);
        assert!(
            matches!(facts(&bad), Err(DecisionError::InvalidMetadata)),
            "{value}"
        );
    }
}

#[test]
fn temperatures_are_looked_up_by_bucket_then_type_then_one() {
    let header = d1_header()
        .set_f32("lfm3.decision.temperature.choice.2", 0.5)
        .set_f32("lfm3.decision.temperature.choice.6_10", 3.25)
        .set_f32("lfm3.decision.temperature.score.11", 9.0)
        .set_f32("lfm3.decision.temperature.image", 2.5)
        .without("lfm3.decision.temperature.noul");
    let facts = facts(&header).unwrap();
    let names: Vec<&str> = facts.temperatures.names().collect();
    assert!(names.contains(&"image"));
    let t = |kind: &str, n: usize| {
        let t = facts.temperatures.lookup(kind, n);
        (t.source, t.value.to_bits())
    };
    let d1 = (Some("1.337398".to_string()), 0x3fab2fdc);
    assert_eq!(t("choice", 1), (Some("0.500000".into()), 0.5f32.to_bits()));
    assert_eq!(t("choice", 2), (Some("0.500000".into()), 0.5f32.to_bits()));
    assert_eq!(t("choice", 3), d1, "no 3_5 key: the type's");
    assert_eq!(t("choice", 5), d1);
    assert_eq!(t("choice", 6), (Some("3.250000".into()), 3.25f32.to_bits()));
    assert_eq!(
        t("choice", 10),
        (Some("3.250000".into()), 3.25f32.to_bits())
    );
    assert_eq!(t("choice", 11), d1);
    assert_eq!(t("score", 10), d1);
    assert_eq!(t("score", 11), (Some("9.000000".into()), 9.0f32.to_bits()));
    assert_eq!(t("noul", 2), (None, 1.0f32.to_bits()), "no key: 1");
    // The stored value is not the one used: the six-decimal string is.
    assert_ne!(D1_TEMPERATURE_BITS, 0x3fab2fdc);
}

#[test]
fn the_profile_rules_are_what_prepare_checks() {
    let rules = super::fake::d1_facts(DecisionLimits::default()).rules();
    assert_eq!(rules, ProfileRules::of(DecisionProfileId::Lfm3D1Text));
    assert_eq!(
        (rules.max_options, rules.min_levels, rules.max_levels),
        (255, 2, 10)
    );
    assert_eq!(
        (
            rules.max_state_depth,
            rules.max_instructions_depth,
            rules.max_description_depth
        ),
        (127, 127, 125)
    );
    // The check refuses just past each rule.
    let at_limit = request(
        DecisionState::from_json(OrderedJson::parse(&nested(rules.max_state_depth)).unwrap()),
        vec![ask(DecisionQuestionKind::score(vec![
            DecisionLevel::new(
                None
            );
            rules.max_levels
        ]))],
    );
    assert!(checked(&at_limit).is_ok());
    let past = request(
        DecisionState::from_json(OrderedJson::parse(&nested(rules.max_state_depth + 1)).unwrap()),
        vec![ask(DecisionQuestionKind::yes_no(None, None))],
    );
    assert!(matches!(
        checked(&past),
        Err(DecisionError::ContentTooDeep { .. })
    ));
}
