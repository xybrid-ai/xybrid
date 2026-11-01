//! The conformance cases on the real d1-2B vocabulary (`d1-2b-q4km`).
//!
//! The model is the pinned artifact of `artifacts.json`, staged by
//! `integration-tests/download.sh d1-2b-q4km`; without it these tests skip,
//! unless `XYBRID_REQUIRE_MODELS` is set, where they fail. Only the
//! vocabulary is used: nothing is evaluated.
//!
//! The metadata cases run twice: here, on d1-2B's decision metadata with the
//! case's edits (and, for a token-type edit, the edited token's spelling
//! among the CONTROL ones), and in
//! [`metadata_variants_built_by_the_oracle`], on the variant GGUF P3a's
//! oracle builds (`tools/conformance/decision`, `decision-oracle variant`),
//! read as the model's header. That test needs the oracle binary:
//! `XYBRID_DECISION_ORACLE=<path to decision-oracle>`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use xybrid_llama::LlamaModel;

use super::cases::{self, f32_of, int, items, Case};
use super::conformance::{
    assert_upstream_answers, case_facts, case_limits, d1_control_spellings, first_difference,
    tail_from, upstream_codes, upstream_readouts, PROTOCOL_KINDS_AT_PARSE,
};
use super::fake::d1_facts;
use crate::execution::decision::facts::sha256_hex;
use crate::execution::decision::llama::LlamaVocab;
use crate::execution::decision::prepare::prepare;
use crate::execution::decision::{
    finish, DecisionLimits, DecisionModelFacts, DecisionPlan, DecisionVocab, Token,
    SYSTEMONE_TEMPLATE_SHA256,
};
use crate::gguf::{DecisionProfileId, GgufArray, GgufHeader, GgufValue};
use crate::ir::decision::{DecisionError, OrderedJson};
use crate::testing::model_fixtures;

const MODEL_ID: &str = "d1-2b-q4km";
const MODEL_FILE: &str = "d1-2B-Q4_K_M.gguf";

/// The staged model, loaded once (`None`, after a skip notice, when it is
/// not staged).
fn d1_2b() -> Option<&'static (PathBuf, LlamaModel)> {
    static MODEL: OnceLock<Option<(PathBuf, LlamaModel)>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let path = model_fixtures::staged_artifact(MODEL_ID, MODEL_FILE)?;
            let model =
                LlamaModel::load(path.to_str().expect("a UTF-8 path"), 0).expect("load d1-2B");
            Some((path, model))
        })
        .as_ref()
}

/// A vocabulary with another vocabulary's tokenizer and its own CONTROL
/// spellings: a metadata variant that changes token types only.
struct SpellingsOf<'a> {
    tokenizer: &'a LlamaVocab<'a>,
    control: Vec<Vec<u8>>,
}

impl DecisionVocab for SpellingsOf<'_> {
    fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Token>, DecisionError> {
        self.tokenizer.tokenize(text, add_special, parse_special)
    }

    fn control_spellings(&self) -> &[Vec<u8>] {
        &self.control
    }
}

/// The CONTROL spellings a GGUF's own vocabulary arrays give (token type 3,
/// 2 bytes or more), in token-id order: what llama.cpp derives its CONTROL
/// attribute from.
fn header_control_spellings(header: &GgufHeader) -> Vec<Vec<u8>> {
    let Some(GgufValue::Array(GgufArray::String(tokens))) = header.get("tokenizer.ggml.tokens")
    else {
        panic!("a vocabulary");
    };
    let Some(GgufValue::Array(GgufArray::I32(types))) = header.get("tokenizer.ggml.token_type")
    else {
        panic!("token types");
    };
    assert_eq!(tokens.len(), types.len());
    tokens
        .iter()
        .zip(types)
        .filter(|(text, &kind)| kind == 3 && text.len() >= 2)
        .map(|(text, _)| text.clone())
        .collect()
}

/// `case`'s token-type edits applied to d1-2B's CONTROL spellings.
fn edited_spellings(case: &Case, model: &LlamaModel, vocab: &LlamaVocab<'_>) -> Vec<Vec<u8>> {
    let Some(edits) = case
        .at(&["metadata", "token_types"])
        .and_then(OrderedJson::as_object)
    else {
        return vocab.control_spellings().to_vec();
    };
    // Rebuild in token-id order with the edited types.
    let mut control: Vec<(Token, Vec<u8>)> = Vec::new();
    let edited: Vec<(Token, i64)> = edits
        .iter()
        .map(|(id, kind)| (id.parse().expect("a token id"), int(kind)))
        .collect();
    for id in 0..model.n_vocab() as Token {
        let text = model
            .token_text(id)
            .expect("a token's text")
            .expect("a token of the vocabulary");
        let control_now = match edited.iter().find(|(edited, _)| *edited == id) {
            Some(&(_, kind)) => kind == 3,
            None => model
                .token_attr(id)
                .expect("a token's attributes")
                .is_some_and(|attr| attr.intersects(xybrid_llama::TokenAttr::CONTROL)),
        };
        if control_now && text.len() >= 2 {
            control.push((id, text));
        }
    }
    control.into_iter().map(|(_, text)| text).collect()
}

#[test]
fn d1_2b_facts_and_vocabulary_are_the_oracles() {
    let Some((path, model)) = d1_2b() else { return };
    let header = GgufHeader::read_file(path).expect("the d1-2B header");
    let facts = DecisionModelFacts::new(
        DecisionProfileId::Lfm3D1Text,
        &header,
        DecisionLimits::default(),
    )
    .expect("d1-2B's facts");
    // The synthetic d1-2B metadata the model-free tests use is the model's.
    assert_eq!(
        facts.temperatures,
        d1_facts(DecisionLimits::default()).temperatures
    );
    let template = header
        .get("tokenizer.chat_template.systemone")
        .and_then(GgufValue::as_bytes)
        .unwrap();
    assert_eq!(sha256_hex(template), SYSTEMONE_TEMPLATE_SHA256);
    assert_eq!(
        model.named_chat_template("systemone").unwrap().as_deref(),
        std::str::from_utf8(template).ok()
    );

    // The vocabulary's CONTROL spellings are the oracle's, and the ones the
    // header's arrays give (which the variant test reads).
    let vocab = LlamaVocab::new(model).expect("d1-2B's vocabulary");
    let spellings = vocab.control_spellings();
    assert_eq!(spellings.len(), 124);
    assert_eq!(header_control_spellings(&header), spellings);
    let mut longest_first: Vec<String> = spellings
        .iter()
        .map(|s| String::from_utf8(s.clone()).unwrap())
        .collect();
    longest_first.sort_by_key(|s| std::cmp::Reverse(s.len()));
    assert_eq!(longest_first, d1_control_spellings());

    // The fake d1 labels of the model-free tests are d1-2B's: every
    // two-letter code that is one token, and only those.
    let not_single: Vec<&str> = super::fake::D1_NOT_SINGLE_TWO_LETTER
        .split_whitespace()
        .collect();
    for a in b'A'..=b'Z' {
        for b in b'A'..=b'Z' {
            let code: String = [char::from(a), char::from(b)].iter().collect();
            let single = vocab.tokenize(&code, false, false).unwrap().len() == 1;
            assert_eq!(single, !not_single.contains(&code.as_str()), "{code}");
        }
    }
    for i in 0..100 {
        assert_eq!(
            vocab
                .tokenize(&format!("{i:02}"), false, false)
                .unwrap()
                .len(),
            1
        );
    }
    // A NUL byte is text (the `state-string-escapes` prompt holds one).
    assert_eq!(vocab.tokenize("a\0b", false, true).unwrap().len(), 3);
}

/// `plan` and `prompts` are what upstream prepared for `case`.
fn assert_upstream_plan(case: &Case, plan: &DecisionPlan, prompts: &[String]) {
    let id = &case.id;
    let passes = items(case.at(&["upstream", "prepare", "passes"]).unwrap());
    assert_eq!(plan.passes().len(), passes.len(), "{id}");
    assert_eq!(prompts.len(), passes.len(), "{id}");
    for (index, ((pass, prompt), expected)) in
        plan.passes().iter().zip(prompts).zip(passes).enumerate()
    {
        let expected_prompt = expected
            .get("prompt")
            .and_then(OrderedJson::as_str)
            .unwrap();
        let at = first_difference(prompt, expected_prompt);
        assert!(
            prompt == expected_prompt,
            "{id}: pass {index} differs from byte {at}:\n  xybrid:   {:?}\n  upstream: {:?}",
            tail_from(prompt, at),
            tail_from(expected_prompt, at),
        );
        let tokens: Vec<Token> = items(expected.get("token_ids").unwrap())
            .iter()
            .map(|t| int(t) as Token)
            .collect();
        assert_eq!(pass.tokens(), tokens, "{id}: pass {index} tokens");
        assert_eq!(
            pass.output_index() as i64,
            int(expected.get("output_index").unwrap()),
            "{id}: pass {index}"
        );
        let select: Vec<Token> = items(expected.get("select").unwrap())
            .iter()
            .map(|t| int(t) as Token)
            .collect();
        assert_eq!(pass.select(), select, "{id}: pass {index} select");
    }
    let answers = items(case.at(&["upstream", "prepare", "answers"]).unwrap());
    assert_eq!(plan.answers.len(), answers.len(), "{id}");
    for (answer, expected) in plan.answers.iter().zip(answers) {
        assert_eq!(
            answer.pass as i64,
            int(expected.get("pass").unwrap()),
            "{id}"
        );
        let groups = items(expected.get("groups").unwrap());
        assert_eq!(answer.groups.len(), groups.len(), "{id}");
        for (group, expected) in answer.groups.iter().zip(groups) {
            assert_eq!(
                Some(group.key.as_str()),
                expected.get("key").and_then(OrderedJson::as_str),
                "{id}"
            );
            assert_eq!(
                Some(group.code.as_str()),
                expected.get("code").and_then(OrderedJson::as_str),
                "{id}"
            );
            let range = items(expected.get("range").unwrap());
            assert_eq!(
                (group.select.start as i64, group.select.end as i64),
                (int(&range[0]), int(&range[1])),
                "{id}"
            );
        }
        assert_eq!(
            answer.temperature.source.as_deref(),
            expected.get("temperature").and_then(OrderedJson::as_str),
            "{id}"
        );
        assert_eq!(
            answer.temperature.value.to_bits(),
            f32_of(expected.get("temperature_f32").unwrap()).to_bits(),
            "{id}"
        );
    }
}

/// Prepares `case` with `facts` and `vocab` and checks it against upstream:
/// the plan or the refusal, then (when upstream finalized it) the answers
/// `finish` reads from upstream's raw scores with xybrid's own plan.
/// Returns what was checked.
fn check_case(case: &Case, facts: &DecisionModelFacts, vocab: &dyn DecisionVocab) -> &'static str {
    let parse = case.expect("parse").unwrap();
    if let Some(kind) = parse.strip_prefix("refuse:") {
        if !PROTOCOL_KINDS_AT_PARSE.contains(&kind) {
            return "transport";
        }
        let Ok(request) = cases::request(case.request_text()) else {
            return "transport";
        };
        let error = prepare(&request, facts, vocab, None).unwrap_err();
        assert_eq!(
            error.kind().map(|k| k.as_str()),
            Some(kind),
            "{}: {error}",
            case.id
        );
        return "parse refusal";
    }
    let request = cases::request(case.request_text())
        .unwrap_or_else(|reason| panic!("{}: {reason}", case.id));
    let mut prompts = Vec::new();
    let result = prepare(&request, facts, vocab, Some(&mut prompts));
    let expectation = case.expect("prepare").unwrap();
    if let Some(kind) = expectation.strip_prefix("refuse:") {
        let error = result
            .err()
            .unwrap_or_else(|| panic!("{}: not refused", case.id));
        assert_eq!(
            error.kind().map(|k| k.as_str()),
            Some(kind),
            "{}: {error}",
            case.id
        );
        return "prepare refusal";
    }
    assert_eq!(expectation, "match", "{}", case.id);
    let plan = result.unwrap_or_else(|error| panic!("{}: {error}", case.id));
    assert_upstream_plan(case, &plan, &prompts);
    let codes: Vec<Vec<String>> = plan
        .answers
        .iter()
        .map(|answer| answer.groups.iter().map(|g| g.code.clone()).collect())
        .collect();
    assert_eq!(codes, upstream_codes(case), "{}", case.id);

    match case.expect("finalize").as_deref() {
        Some("match") => {
            let decisions = finish(&request, &plan, upstream_readouts(case))
                .unwrap_or_else(|error| panic!("{}: {error}", case.id));
            assert_upstream_answers(case, &plan, &decisions);
        }
        Some(refusal) if refusal.starts_with("refuse:") => {
            let error = finish(&request, &plan, upstream_readouts(case)).unwrap_err();
            assert_eq!(
                error
                    .kind()
                    .map(|k| format!("refuse:{}", k.as_str()))
                    .as_deref(),
                Some(refusal),
                "{}",
                case.id
            );
        }
        _ => {}
    }
    "prepared"
}

#[test]
fn every_case_prepares_as_upstream_on_d1_2b() {
    let Some((_, model)) = d1_2b() else { return };
    let vocab = LlamaVocab::new(model).expect("d1-2B's vocabulary");
    let mut checked = std::collections::BTreeMap::<&str, usize>::new();
    for case in cases::all() {
        if case.expect("prepare").as_deref() == Some("not_reached")
            && !case
                .expect("parse")
                .is_some_and(|parse| parse.starts_with("refuse:"))
        {
            continue; // refused at load
        }
        let Ok(facts) = case_facts(&case, case_limits(&case)) else {
            continue; // refused at load (`every_load_stage_is_upstreams`)
        };
        let outcome = if case.at(&["metadata", "token_types"]).is_some() {
            let variant = SpellingsOf {
                tokenizer: &vocab,
                control: edited_spellings(&case, model, &vocab),
            };
            check_case(&case, &facts, &variant)
        } else {
            check_case(&case, &facts, &vocab)
        };
        *checked.entry(outcome).or_default() += 1;
    }
    eprintln!("every_case_prepares_as_upstream_on_d1_2b: {checked:?}");
    assert_eq!(checked.get("prepared"), Some(&83), "{checked:?}");
    assert_eq!(checked.get("prepare refusal"), Some(&7), "{checked:?}");
    assert_eq!(checked.get("parse refusal"), Some(&7), "{checked:?}");
}

/// The `decision-oracle` binary, from `XYBRID_DECISION_ORACLE`.
fn oracle() -> Option<PathBuf> {
    let path = std::env::var_os("XYBRID_DECISION_ORACLE").map(PathBuf::from);
    match path {
        Some(path) if path.is_file() => Some(path),
        _ => {
            let reason = "XYBRID_DECISION_ORACLE does not name a decision-oracle binary \
                          (build it with tools/conformance/decision/build.sh)";
            if model_fixtures::models_required() {
                panic!("XYBRID_REQUIRE_MODELS is set but {reason}");
            }
            eprintln!("Skipping test: {reason}");
            None
        }
    }
}

/// Builds `case`'s variant GGUF from `base` with the oracle, as `oracle.py`
/// does (`make_variant`).
fn build_variant(oracle: &Path, base: &Path, case: &Case, dir: &Path) -> PathBuf {
    let output = dir.join(format!("{}.gguf", case.id));
    let mut command = std::process::Command::new(oracle);
    command
        .arg("variant")
        .arg("--base")
        .arg(base)
        .arg("--output")
        .arg(&output);
    let metadata = case.at(&["metadata"]).expect("edits");
    let mut sets: Vec<(&str, &str)> = metadata
        .get("set")
        .and_then(OrderedJson::as_object)
        .map(|set| {
            set.iter()
                .map(|(k, v)| (k.as_str(), v.as_str().unwrap()))
                .collect()
        })
        .unwrap_or_default();
    sets.sort();
    for (key, value) in sets {
        command.arg("--set").arg(format!("{key}={value}"));
    }
    for key in metadata
        .get("remove")
        .and_then(OrderedJson::as_array)
        .unwrap_or(&[])
    {
        command.arg("--remove").arg(key.as_str().unwrap());
    }
    let mut types: Vec<(&str, i64)> = metadata
        .get("token_types")
        .and_then(OrderedJson::as_object)
        .map(|types| types.iter().map(|(k, v)| (k.as_str(), int(v))).collect())
        .unwrap_or_default();
    types.sort();
    for (token, kind) in types {
        command.arg("--token-type").arg(format!("{token}={kind}"));
    }
    let status = command.status().expect("run decision-oracle");
    assert!(
        status.success(),
        "{}: decision-oracle variant failed",
        case.id
    );
    output
}

#[test]
fn metadata_variants_built_by_the_oracle() {
    let Some((base, model)) = d1_2b() else { return };
    let Some(oracle) = oracle() else { return };
    let dir = tempfile::tempdir().expect("a temporary directory");
    let vocab = LlamaVocab::new(model).expect("d1-2B's vocabulary");
    let mut checked = std::collections::BTreeMap::<&str, usize>::new();
    for case in cases::all() {
        if case.at(&["metadata"]).is_none() {
            continue;
        }
        let variant = build_variant(&oracle, base, &case, dir.path());
        let header = GgufHeader::read_file(&variant).expect("the variant's header");
        let facts =
            DecisionModelFacts::new(DecisionProfileId::Lfm3D1Text, &header, case_limits(&case));
        let load = case.expect("load").expect("a load stage");
        if let Some(kind) = load.strip_prefix("refuse:") {
            let error = facts.err().unwrap_or_else(|| panic!("{}: loads", case.id));
            assert_eq!(
                error.kind().map(|k| k.as_str()),
                Some(kind),
                "{}: {error}",
                case.id
            );
            *checked.entry("load refusal").or_default() += 1;
            continue;
        }
        let facts = facts.unwrap_or_else(|error| panic!("{}: {error}", case.id));
        // The variant keeps d1-2B's vocabulary but for its token types: its
        // tokenizer is d1-2B's, its CONTROL spellings its own.
        let variant_vocab = SpellingsOf {
            tokenizer: &vocab,
            control: header_control_spellings(&header),
        };
        let outcome = check_case(&case, &facts, &variant_vocab);
        *checked.entry(outcome).or_default() += 1;
    }
    eprintln!("metadata_variants_built_by_the_oracle: {checked:?}");
    assert_eq!(checked.get("load refusal"), Some(&3), "{checked:?}");
    assert_eq!(checked.get("prepared"), Some(&8), "{checked:?}");
}
