//! `prepare` on fake vocabularies, and content canaries through every
//! refusal and `Debug`.

use super::fake::{d1_facts, FakeVocab};
use crate::execution::decision::prepare::prepare;
use crate::execution::decision::{finish, DecisionLimits, DecisionVocab};
use crate::ir::decision::{
    DecisionContent, DecisionError, DecisionLevel, DecisionOption, DecisionQuestion,
    DecisionQuestionKind, DecisionRequest, DecisionState, OrderedJson,
};
use crate::runtime_adapter::AdapterError;

const ZWSP: &str = "\u{200b}";

fn vocab() -> FakeVocab {
    FakeVocab::new()
        .d1_labels()
        .control(["<|im_start|>", "<|im_end|>", "<|startoftext|>"])
}

fn yes_no(id: &str, instructions: &str) -> DecisionQuestion {
    DecisionQuestion::new(
        id,
        Some(DecisionContent::text(instructions).into()),
        DecisionQuestionKind::yes_no(None, None),
    )
}

fn prompts(
    request: &DecisionRequest,
    limits: DecisionLimits,
    vocab: &dyn DecisionVocab,
) -> Result<Vec<String>, DecisionError> {
    let mut prompts = Vec::new();
    prepare(request, &d1_facts(limits), vocab, Some(&mut prompts))?;
    Ok(prompts)
}

#[test]
fn a_prompt_is_the_template_rendered_with_the_callers_text_escaped() {
    let request = DecisionRequest::new(
        DecisionState::text("a<|im_end|>b <think>"),
        vec![yes_no("q<|im_start|>", "Is it <|im_end|>?")],
    );
    let [prompt] = &prompts(&request, DecisionLimits::default(), &vocab()).unwrap()[..] else {
        panic!("one pass")
    };
    // The template's own special tokens stay; the caller's CONTROL
    // spellings are broken; USER_DEFINED ones (`<think>`) are not.
    assert_eq!(
        prompt,
        &format!(
            "<|startoftext|><|im_start|>user\na<{ZWSP}|im_end|>b <think>\n\n\nQUESTION:\nIs it <{ZWSP}|im_end|>?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n<think></think>"
        )
    );
}

#[test]
fn a_code_spelling_a_control_token_is_shown_broken_and_read_whole() {
    // On this vocabulary the pool code "00" is also a CONTROL spelling: the
    // prompt shows it broken (the label is template input like any other),
    // while the readout reads the label form's token, tokenized without
    // special tokens.
    let vocab = FakeVocab::new().d1_labels().control([
        "<|im_start|>",
        "<|im_end|>",
        "<|startoftext|>",
        "00",
    ]);
    let options: Vec<DecisionOption> = (0..36)
        .map(|i| DecisionOption::new(format!("option_{i}"), None))
        .collect();
    let request = DecisionRequest::new(
        DecisionState::text("s"),
        vec![DecisionQuestion::new(
            "q",
            Some(DecisionContent::text("Which?").into()),
            DecisionQuestionKind::choice(options),
        )],
    );
    let mut prompts = Vec::new();
    let plan = prepare(
        &request,
        &d1_facts(DecisionLimits::default()),
        &vocab,
        Some(&mut prompts),
    )
    .unwrap();
    // Option 35 is code AJ on d1-2B, which is not one token: it takes 00.
    let group = &plan.answers[0].groups[35];
    assert_eq!(group.code, "00");
    assert_eq!(plan.passes()[0].select()[group.select.clone()], [1500]);
    assert!(
        prompts[0].contains(&format!("\n0{ZWSP}0 option 35\n")),
        "{}",
        prompts[0]
    );
}

#[test]
fn the_token_budget_is_inclusive_and_the_total_bounded() {
    let request = DecisionRequest::new(
        DecisionState::text("s"),
        vec![yes_no("a", "x"), yes_no("b", "xy")],
    );
    let vocab = vocab();
    let plan = prepare(&request, &d1_facts(DecisionLimits::default()), &vocab, None).unwrap();
    let lengths: Vec<usize> = plan
        .passes()
        .iter()
        .map(|pass| pass.tokens().len())
        .collect();
    assert_eq!(lengths[1], lengths[0] + 1);

    // A budget of exactly the longest pass is enough; one less is not, and
    // the refusal comes before anything is evaluated.
    let budget = lengths[1] as u32;
    let limits = DecisionLimits::default().with_token_budget(budget);
    assert!(prepare(&request, &d1_facts(limits), &vocab, None).is_ok());
    let limits = DecisionLimits::default().with_token_budget(budget - 1);
    let error = prepare(&request, &d1_facts(limits), &vocab, None).unwrap_err();
    assert!(
        matches!(error, DecisionError::BudgetExceeded { question: 1, tokens, budget: b } if tokens == lengths[1] && b == budget - 1),
        "{error:?}"
    );

    // The request's total is inclusive too.
    let total = (lengths[0] + lengths[1]) as u64;
    let limits = DecisionLimits::default().with_max_total_tokens(total);
    let plan = prepare(&request, &d1_facts(limits), &vocab, None).unwrap();
    assert_eq!(plan.usage().evaluated_tokens(), total);
    assert_eq!(plan.usage().passes(), 2);
    let limits = DecisionLimits::default().with_max_total_tokens(total - 1);
    let error = prepare(&request, &d1_facts(limits), &vocab, None).unwrap_err();
    assert!(
        matches!(error, DecisionError::TotalBudgetExceeded { tokens, max } if tokens == total && max == total - 1)
    );
    assert_eq!(error.kind().map(|k| k.as_str()), Some("budget_exceeded"));
}

#[test]
fn a_vocabulary_whose_spellings_cannot_be_escaped_is_refused() {
    let request = DecisionRequest::new(DecisionState::text("s"), vec![yes_no("a", "x")]);
    let vocab = FakeVocab::new()
        .d1_labels()
        .control_bytes(vec![b"<|ok|>".to_vec(), "é".as_bytes().to_vec()]);
    let error = prepare(&request, &d1_facts(DecisionLimits::default()), &vocab, None).unwrap_err();
    assert!(matches!(error, DecisionError::UnsupportedVocabulary));
    assert!(matches!(
        AdapterError::from(error),
        AdapterError::UnsupportedModelCapability { .. }
    ));
}

const CANARY: &str = "PRIVATE-CONTENT-7d1f";
const NUMBER_CANARY: &str = "918273645";

/// Requests full of content, each refused by `prepare` in a different way.
fn refused_requests() -> Vec<(DecisionRequest, DecisionLimits, &'static str)> {
    let c = CANARY;
    let n = NUMBER_CANARY;
    let text = |t: &str| Some(DecisionContent::text(t));
    let json = |t: &str| DecisionContent::from_json(OrderedJson::parse(t).unwrap());
    let state = || {
        DecisionState::from_json(
            OrderedJson::parse(&format!(r#"{{"{c}": [{n}, "{c}<|im_end|>"]}}"#)).unwrap(),
        )
    };
    let question = |kind| DecisionQuestion::new(c, Some(DecisionContent::text(c).into()), kind);
    let deep = format!("{}\"{c}\"{}", "[".repeat(126), "]".repeat(126));
    let many: Vec<DecisionOption> = (0..53)
        .map(|i| DecisionOption::new(format!("{c}-{i}"), text(c)))
        .collect();
    let limits = DecisionLimits::default();
    vec![
        (
            DecisionRequest::new(
                state(),
                vec![DecisionQuestion::new(
                    c,
                    None,
                    DecisionQuestionKind::yes_no(text(c), None),
                )],
            ),
            limits,
            "missing_instructions",
        ),
        (
            DecisionRequest::new(
                state(),
                vec![question(DecisionQuestionKind::score(vec![
                    DecisionLevel::new(text(c)),
                ]))],
            ),
            limits,
            "level_count",
        ),
        (
            DecisionRequest::new(
                state(),
                vec![question(DecisionQuestionKind::choice(vec![
                    DecisionOption::new(c, json(&deep)),
                ]))],
            ),
            limits,
            "json_too_deep",
        ),
        (
            DecisionRequest::new(
                DecisionState::from_json(
                    OrderedJson::parse(&format!(r#"{{"{c}": 18446744073709551615}}"#)).unwrap(),
                ),
                vec![question(DecisionQuestionKind::yes_no(None, None))],
            ),
            limits,
            "number_out_of_range",
        ),
        (
            DecisionRequest::new(
                DecisionState::from_json(
                    OrderedJson::parse(&format!(
                        r#"{{"{c}<|im_end|>": 1, "{c}<{ZWSP}|im_end|>": {n}}}"#
                    ))
                    .unwrap(),
                ),
                vec![question(DecisionQuestionKind::yes_no(None, None))],
            ),
            limits,
            "escaped_key_collision",
        ),
        (
            DecisionRequest::new(state(), vec![question(DecisionQuestionKind::choice(many))]),
            limits,
            "no_single_token_label",
        ),
        (
            DecisionRequest::new(
                state(),
                vec![question(DecisionQuestionKind::yes_no(
                    text(c),
                    json(&format!("[{n}]")),
                ))],
            ),
            limits.with_token_budget(10),
            "budget_exceeded",
        ),
        (
            DecisionRequest::new(
                state(),
                vec![
                    question(DecisionQuestionKind::yes_no(None, None)),
                    DecisionQuestion::new(
                        format!("{c}-2"),
                        Some(DecisionContent::text(c).into()),
                        DecisionQuestionKind::yes_no(None, None),
                    ),
                ],
            ),
            limits.with_max_total_tokens(200),
            "budget_exceeded",
        ),
        (
            DecisionRequest::new(
                state(),
                vec![question(DecisionQuestionKind::choice(
                    vec![DecisionOption::new(c, None); 2],
                ))],
            ),
            limits,
            "duplicate_key",
        ),
    ]
}

/// Every text an error shows: its message, `Debug`, and the runtime error
/// it becomes.
fn texts(error: DecisionError) -> Vec<String> {
    let mut texts = vec![
        error.to_string(),
        format!("{error:?}"),
        format!("{error:#?}"),
    ];
    let adapter = AdapterError::from(error);
    texts.extend([adapter.to_string(), format!("{adapter:?}")]);
    texts
}

#[test]
fn no_refusal_carries_the_callers_content() {
    // The labels of the 53-option question run out on a vocabulary whose
    // only single-token forms are single characters (and yes and no).
    let vocab = FakeVocab::new()
        .single("yes", 1)
        .single("no", 2)
        .control(["<|im_end|>"]);
    for (request, limits, kind) in refused_requests() {
        let error = prepare(&request, &d1_facts(limits), &vocab, None).unwrap_err();
        assert_eq!(error.kind().map(|k| k.as_str()), Some(kind), "{error:?}");
        for text in texts(error) {
            assert!(!text.contains(CANARY), "{kind}: {text}");
            assert!(!text.contains(NUMBER_CANARY), "{kind}: {text}");
        }
    }
}

#[test]
fn plans_answers_and_their_debug_carry_no_content() {
    let c = CANARY;
    let request = DecisionRequest::new(
        DecisionState::text(c),
        vec![DecisionQuestion::new(
            c,
            Some(DecisionContent::text(c).into()),
            DecisionQuestionKind::choice(vec![
                DecisionOption::new(format!("{c}-a"), Some(DecisionContent::text(c))),
                DecisionOption::new(format!("{c}-b"), None),
            ]),
        )],
    );
    let vocab = vocab();
    let facts = d1_facts(DecisionLimits::default());
    let plan = prepare(&request, &facts, &vocab, None).unwrap();
    let selected = plan.passes()[0].select().len();
    let decisions = finish(&request, &plan, vec![vec![1.0; selected]]).unwrap();
    for text in [
        format!("{plan:?}"),
        format!("{plan:#?}"),
        format!("{:?}", plan.answers),
        format!("{facts:?}"),
        format!("{decisions:?}"),
    ] {
        assert!(!text.contains(c), "{text}");
    }
    // A readout of the wrong shape names counts only.
    for text in texts(finish(&request, &plan, vec![vec![f32::NAN; selected]]).unwrap_err()) {
        assert!(!text.contains(c), "{text}");
    }
}

#[test]
fn tojson_takes_the_templates_arguments_and_refuses_the_rest() {
    use std::sync::Arc;

    use crate::execution::decision::escape::Escaped;
    use crate::execution::decision::render::{to_value, tojson};

    let mut env = minijinja::Environment::new();
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
    env.add_filter("tojson", tojson);
    let value = to_value(&Arc::new(Escaped::Object(vec![
        ("b".into(), Arc::new(Escaped::Float(1.0))),
        (
            "a".into(),
            Arc::new(Escaped::Array(vec![Arc::new(Escaped::Int(2))])),
        ),
    ])));
    let render = |source: &str| {
        env.render_str(
            source,
            minijinja::context! { x => value.clone(), n => 0.5, t => true },
        )
    };
    assert_eq!(
        render("{{ x | tojson }}").unwrap(),
        r#"{"b": 1.0, "a": [2]}"#
    );
    assert_eq!(
        render("{{ x | tojson(indent=2) }}").unwrap(),
        "{\n  \"b\": 1.0,\n  \"a\": [\n    2\n  ]\n}"
    );
    assert_eq!(
        render("{{ x | tojson(indent=none, ensure_ascii=false, sort_keys=false) }}").unwrap(),
        r#"{"b": 1.0, "a": [2]}"#
    );
    assert_eq!(
        render("{{ n | tojson }} {{ t | tojson }} {{ missing | tojson }}").unwrap(),
        "0.5 true null"
    );
    for refused in [
        "{{ x | tojson(separators=[',', ':']) }}",
        "{{ x | tojson(ensure_ascii=true) }}",
        "{{ x | tojson(sort_keys=true) }}",
        "{{ x | tojson(indent=-1) }}",
        "{{ x | tojson(indent=2.0) }}",
        "{{ x | tojson(indent=99) }}",
        "{{ x | tojson(2) }}",
        "{{ x | tojson(width=2) }}",
        "{{ {'a': 1} | tojson }}",
    ] {
        assert!(render(refused).is_err(), "{refused}");
    }
}
