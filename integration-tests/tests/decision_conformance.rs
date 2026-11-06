//! Stage-isolated Liquid text/v1 conformance against the pinned upstream corpus.
//!
//! The corpus's `upstream` fields are observations, whereas `xybrid` declares
//! acceptance. These tests never treat the latter as executed model evidence.
//! Model-required CI sets `XYBRID_REQUIRE_MODELS=1`; missing assets then fail.

use integration_tests::fixtures;
use serde::Deserialize;
use serde_json::Value;
use xybrid_core::execution::decision::{
    liquid::DecisionProtocolError,
    math,
    prepare::{AnswerKind, AnswerPlan, LabelGroup, PreparedQuestion},
    wire,
};
use xybrid_core::ir::decision::{DecisionAnswer, DecisionOrigin, ScoreStatistic};

// These are trusted fixture envelopes, not application requests: wrapper
// overhead puts their 125-level structured legends beyond serde's depth128.
// The production wire decoder continues enforcing its independent field limits.
fn fixture_json(text: &str) -> Value {
    let mut decoder = serde_json::Deserializer::from_str(text);
    decoder.disable_recursion_limit();
    let value = Value::deserialize(&mut decoder).expect("fixture JSON is valid");
    decoder.end().expect("fixture has no trailing data");
    value
}

fn cases() -> Vec<Value> {
    let mut paths: Vec<_> = std::fs::read_dir(fixtures::fixtures_dir().join("decision/cases"))
        .expect("decision corpus exists")
        .map(|entry| entry.expect("case entry is readable").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    paths.sort();
    assert_eq!(
        paths.len(),
        143,
        "an empty or partial corpus must never pass"
    );
    paths
        .into_iter()
        .map(|path| fixture_json(&std::fs::read_to_string(path).expect("case is readable")))
        .collect()
}

fn f32_bits(value: &Value) -> f32 {
    f32::from_bits(
        u32::from_str_radix(
            value
                .as_str()
                .expect("f32 is recorded as bits")
                .trim_start_matches("0x"),
            16,
        )
        .expect("f32 bits are hexadecimal"),
    )
}

fn plan(pass: &Value, answer: &Value) -> PreparedQuestion {
    PreparedQuestion {
        tokens: serde_json::from_value(pass["token_ids"].clone()).expect("token ids"),
        select: serde_json::from_value(pass["select"].clone()).expect("selected token ids"),
        plan: AnswerPlan {
            question: answer["question"].as_str().unwrap().to_owned(),
            kind: match answer["type"].as_str().unwrap() {
                "choice" => AnswerKind::Choice,
                "noul" => AnswerKind::Noul,
                "score" => AnswerKind::Score,
                _ => panic!("unknown fixture question type"),
            },
            groups: answer["groups"]
                .as_array()
                .unwrap()
                .iter()
                .map(|group| LabelGroup {
                    key: group["key"].as_str().unwrap().to_owned(),
                    code: group["code"].as_str().unwrap().to_owned(),
                    range: group["range"][0].as_u64().unwrap() as usize
                        ..group["range"][1].as_u64().unwrap() as usize,
                })
                .collect(),
            temperature: f32_bits(&answer["temperature_f32"]),
            temperature_text: answer["temperature"].as_str().map(str::to_owned),
        },
    }
}

fn assert_answer(case: &str, actual: &DecisionAnswer, expected: &Value) {
    match expected["type"].as_str().unwrap() {
        "choice" => assert_eq!(
            actual.value().as_choice(),
            expected["choice"].as_str(),
            "{case}"
        ),
        "noul" => {
            assert_eq!(
                actual.value().as_yes_no(),
                expected["noul"].as_f64(),
                "{case}"
            );
            assert!(actual.confidence().is_none(), "{case}");
        }
        "score" => assert_eq!(
            actual.value().as_score(),
            Some((expected["score"].as_f64().unwrap(), ScoreStatistic::Median)),
            "{case}"
        ),
        _ => panic!("unknown fixture answer type"),
    }
    if expected["type"] != "noul" {
        assert_eq!(
            actual.confidence(),
            expected["confidence"].as_f64(),
            "{case}"
        );
        assert_eq!(
            actual.distribution().len(),
            expected["probabilities"].as_object().unwrap().len()
        );
        for entry in actual.distribution() {
            assert_eq!(
                entry.probability().to_bits(),
                expected["probabilities"][entry.key()]
                    .as_f64()
                    .unwrap()
                    .to_bits(),
                "{case}: f64 probability must round-trip exactly"
            );
        }
    }
    assert!(actual
        .distribution()
        .iter()
        .all(|entry| entry.origin() == DecisionOrigin::Caller));
    let round_trip: DecisionAnswer =
        serde_json::from_str(&serde_json::to_string(actual).unwrap()).unwrap();
    assert_eq!(&round_trip, actual, "{case}: JSON round trip");
}

#[test]
fn all_143_wire_cases_match_their_parse_stage() {
    let mut accepted = 0;
    let mut refused = 0;
    for case in cases() {
        let expected = &case["xybrid"]["parse"];
        if expected == "not_reached" {
            continue;
        }
        let actual = wire::decode(case["request_text"].as_str().unwrap());
        let id = case["id"].as_str().unwrap();
        if expected == "match" {
            let request = actual.unwrap_or_else(|error| panic!("{id}: {error}"));
            let ids: Vec<_> = request
                .questions()
                .iter()
                .map(|question| question.id())
                .collect();
            let expected_ids: Vec<_> = case["upstream"]["parse"]["questions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|question| question["id"].as_str().unwrap())
                .collect();
            assert_eq!(ids, expected_ids, "{id}: question order");
            accepted += 1;
        } else {
            let error = actual.expect_err("the fixture requires refusal");
            assert_eq!(
                error.category(),
                expected["refuse"].as_str().unwrap(),
                "{id}"
            );
            refused += 1;
        }
    }
    assert_eq!((accepted, refused), (98, 42));
}

#[test]
fn upstream_selected_logits_finalize_independently_of_inference() {
    if !cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )) {
        assert!(
            std::env::var_os(fixtures::REQUIRE_MODELS_ENV).is_none(),
            "exact numerical reference is Linux x86_64 glibc; qualify this platform independently"
        );
        eprintln!(
            "Skipping exact upstream f64 bit comparison: no frozen libm bounds for this platform"
        );
        return;
    }
    let mut matched = 0;
    let mut refused = 0;
    for case in cases() {
        let expected = &case["xybrid"]["finalize"];
        if expected == "not_reached" || expected == "out_of_scope" {
            continue;
        }
        let id = case["id"].as_str().unwrap();
        let upstream = &case["upstream"];
        for answer in upstream["prepare"]["answers"].as_array().unwrap() {
            let index = answer["pass"].as_u64().unwrap() as usize;
            let prepared = plan(&upstream["prepare"]["passes"][index], answer);
            let raw: Vec<_> = upstream["evaluate"]["raw_scores"][index]
                .as_array()
                .unwrap()
                .iter()
                .map(f32_bits)
                .collect();
            let actual = math::finalize(&prepared, &raw);
            if expected == "match" {
                let result = actual.unwrap_or_else(|error| panic!("{id}: {error}"));
                assert_eq!(result.question_id(), prepared.plan.question);
                assert_answer(
                    id,
                    &result,
                    &upstream["finalize"]["answers"][result.question_id()],
                );
            } else {
                assert_eq!(
                    actual.unwrap_err(),
                    DecisionProtocolError::NonFiniteScores,
                    "{id}"
                );
            }
        }
        if expected == "match" {
            matched += 1;
        } else {
            refused += 1;
        }
    }
    assert_eq!((matched, refused), (82, 6));
}

#[test]
fn committed_decision_metadata_names_the_pinned_protocol() {
    let metadata: Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures::model_path("d1-2b-q4km").join("model_metadata.json"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(metadata["execution_template"]["type"], "Decision");
    assert_eq!(
        metadata["execution_template"]["protocol"],
        "lfm3-d1/text/v1"
    );
    assert_eq!(metadata["execution_template"]["context_length"], 4096);
    assert_eq!(metadata["metadata"]["task"], "decision");
}

#[cfg(feature = "llm-llamacpp")]
mod native {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::Read;
    use std::path::{Path, PathBuf};
    use std::sync::{atomic::AtomicBool, Mutex, OnceLock};
    use xybrid_core::execution::decision::liquid::LiquidProtocol;
    use xybrid_llama::{DecisionContext, LlamaModel, VocabularyModel};

    // A 4096-token whole-attention pass is intentionally large. Do not run
    // the corpus's separate ownership checks concurrently on a CI worker.
    static NATIVE: Mutex<()> = Mutex::new(());
    static MODEL: OnceLock<Option<PathBuf>> = OnceLock::new();

    fn model() -> Option<PathBuf> {
        MODEL
            .get_or_init(|| {
                let path = fixtures::model_path("d1-2b-q4km").join("d1-2B-Q4_K_M.gguf");
                if !path.is_file() {
                    assert!(
                        std::env::var_os(fixtures::REQUIRE_MODELS_ENV).is_none(),
                        "required d1 GGUF missing: run integration-tests/download.sh d1-2b-q4km"
                    );
                    eprintln!("Skipping d1 native conformance: Q4_K_M is not staged");
                    return None;
                }
                assert_eq!(
                    path.metadata().unwrap().len(),
                    1_099_629_888,
                    "artifact length"
                );
                let mut file = std::fs::File::open(&path).unwrap();
                let mut hash = Sha256::new();
                let mut buffer = [0u8; 1024 * 1024];
                loop {
                    let count = file.read(&mut buffer).unwrap();
                    if count == 0 {
                        break;
                    }
                    hash.update(&buffer[..count]);
                }
                assert_eq!(
                    format!("{:x}", hash.finalize()),
                    "adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b"
                );
                Some(path)
            })
            .clone()
    }

    fn oracle_binary() -> PathBuf {
        std::env::var_os("XYBRID_DECISION_ORACLE_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                fixtures::fixtures_dir()
                    .join("../../tools/conformance/decision/build/bin/decision-oracle")
            })
    }

    fn variant(case: &Value, base: &Path, directory: &Path) -> PathBuf {
        if case["metadata"].is_null() {
            return base.to_owned();
        }
        let output = directory.join(format!("{}.gguf", case["id"].as_str().unwrap()));
        let mut command = std::process::Command::new(oracle_binary());
        command
            .args(["variant", "--base"])
            .arg(base)
            .arg("--output")
            .arg(&output);
        if let Some(values) = case["metadata"]["set"].as_object() {
            for (key, value) in values {
                command
                    .arg("--set")
                    .arg(format!("{key}={}", value.as_str().unwrap()));
            }
        }
        if let Some(values) = case["metadata"]["remove"].as_array() {
            for key in values {
                command.arg("--remove").arg(key.as_str().unwrap());
            }
        }
        if let Some(values) = case["metadata"]["token_types"].as_object() {
            for (key, value) in values {
                command.arg("--token-type").arg(format!("{key}={value}"));
            }
        }
        let status = command
            .output()
            .expect("build the pinned oracle: tools/conformance/decision/build.sh");
        assert!(
            status.status.success(),
            "metadata-only fixture generation failed"
        );
        output
    }

    #[test]
    fn all_preparation_and_metadata_variants_match_the_upstream() {
        let _guard = NATIVE.lock().unwrap();
        let Some(path) = model() else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        let vocabulary = VocabularyModel::load(path.to_str().unwrap()).unwrap();
        let mut protocol = LiquidProtocol::load(&vocabulary).unwrap();
        let mut matched = 0;
        let mut refused = 0;
        let mut load_refused = 0;
        for case in cases() {
            let id = case["id"].as_str().unwrap();
            let own = if case["metadata"].is_null() {
                None
            } else {
                Some(
                    VocabularyModel::load(
                        variant(&case, &path, directory.path()).to_str().unwrap(),
                    )
                    .unwrap(),
                )
            };
            let loaded = own.as_ref().map(LiquidProtocol::load);
            if let Some(expected) = case["xybrid"]["load"]["refuse"].as_str() {
                assert_eq!(
                    loaded.unwrap().unwrap_err().category(),
                    expected,
                    "{id}: load"
                );
                load_refused += 1;
                continue;
            }
            let mut own_protocol = loaded.map(Result::unwrap);
            let vocabulary = own.as_ref().unwrap_or(&vocabulary);
            let protocol = own_protocol.as_mut().unwrap_or(&mut protocol);
            if case["xybrid"]["parse"] != "match" {
                continue;
            }
            let request = wire::decode(case["request_text"].as_str().unwrap()).unwrap();
            let expected = &case["xybrid"]["prepare"];
            if let Some(category) = expected["refuse"].as_str() {
                assert_eq!(
                    protocol
                        .prepare(vocabulary, &request)
                        .unwrap_err()
                        .category(),
                    category,
                    "{id}: prepare"
                );
                refused += 1;
                continue;
            }
            assert_eq!(expected, "match", "{id}");
            for (index, question) in request.questions().iter().enumerate() {
                let (prompt, actual) = protocol
                    .prepare_one(vocabulary, request.state(), question)
                    .unwrap_or_else(|error| panic!("{id}: {error}"));
                let upstream_pass = &case["upstream"]["prepare"]["passes"][index];
                let upstream_answer = &case["upstream"]["prepare"]["answers"][index];
                let expected = plan(upstream_pass, upstream_answer);
                assert_eq!(
                    prompt.as_bytes(),
                    upstream_pass["prompt"].as_str().unwrap().as_bytes(),
                    "{id}: prompt bytes"
                );
                assert_eq!(actual.tokens, expected.tokens, "{id}: tokens");
                assert_eq!(
                    actual.tokens.len() - 1,
                    upstream_pass["output_index"].as_u64().unwrap() as usize,
                    "{id}: output index"
                );
                assert_eq!(actual.select, expected.select, "{id}: selected labels");
                assert_eq!(actual.plan.kind, expected.plan.kind, "{id}: kind");
                assert_eq!(
                    actual.plan.question, expected.plan.question,
                    "{id}: question"
                );
                assert_eq!(
                    actual.plan.temperature.to_bits(),
                    expected.plan.temperature.to_bits(),
                    "{id}: temperature bits"
                );
                assert_eq!(
                    actual.plan.temperature_text, expected.plan.temperature_text,
                    "{id}: temperature representation"
                );
                assert_eq!(
                    actual.plan.groups.len(),
                    expected.plan.groups.len(),
                    "{id}: group count"
                );
                for (actual, expected) in actual.plan.groups.iter().zip(&expected.plan.groups) {
                    assert_eq!(
                        (&actual.key, &actual.code, &actual.range),
                        (&expected.key, &expected.code, &expected.range),
                        "{id}: group"
                    );
                }
            }
            matched += 1;
        }
        assert_eq!((matched, refused, load_refused), (89, 9, 3));
    }

    fn references() -> Option<PathBuf> {
        let directory = std::env::var_os("XYBRID_DECISION_REFERENCE_DIR").map(PathBuf::from);
        if directory.is_none() {
            assert!(std::env::var_os(fixtures::REQUIRE_MODELS_ENV).is_none(),
                "set XYBRID_DECISION_REFERENCE_DIR to a fresh official oracle --collect output (see just integration-tests decision-native)");
            eprintln!("Skipping matched-build G2: XYBRID_DECISION_REFERENCE_DIR is absent");
            return None;
        }
        let directory = directory.unwrap();
        let receipt: Value = serde_json::from_str(
            &std::fs::read_to_string(directory.join("collection.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            receipt["status"], "complete",
            "upstream collection must be complete"
        );
        assert_eq!(
            receipt["collector"]["official"], true,
            "upstream collection must be official"
        );
        assert_eq!(
            receipt["artifact"]["sha256"],
            "adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b"
        );
        assert_eq!(
            receipt["build"]["llama_cpp"]["commit"],
            "8642541b255bfb5dc39cd351a178be09013a55d6"
        );
        assert_eq!(receipt["corpus"]["selected"], 63);
        let settings = &receipt["settings"]["cpu-t4-fa-off"];
        assert_eq!(settings["model_devices"], serde_json::json!([]));
        assert_eq!(settings["context"]["n_ctx"], 4096);
        assert_eq!(settings["context"]["n_batch"], 4096);
        assert_eq!(settings["context"]["n_ubatch"], 4096);
        Some(directory)
    }

    fn reference(directory: &Path, case: &Value) -> Value {
        let request_hash = Sha256::digest(case["request_text"].as_str().unwrap().as_bytes());
        let filename = format!("{}.{request_hash:x}.json", case["id"].as_str().unwrap());
        let observation = fixture_json(&std::fs::read_to_string(directory.join(filename)).unwrap());
        assert_eq!(
            observation["inputs"]["model"]["sha256"],
            "adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b"
        );
        observation
    }

    #[test]
    fn native_selected_logits_match_a_fresh_matched_build_reference() {
        let _guard = NATIVE.lock().unwrap();
        let Some(path) = model() else {
            return;
        };
        let Some(directory) = references() else {
            return;
        };
        let model = LlamaModel::load_decision_cpu(path.to_str().unwrap()).unwrap();
        let mut context = DecisionContext::new(&model).unwrap();
        let settings = context.settings().unwrap();
        assert_eq!(
            (
                settings.n_ctx,
                settings.n_ctx_seq,
                settings.n_batch,
                settings.n_ubatch
            ),
            (4096, 4096, 4096, 4096)
        );
        assert_eq!(
            (
                settings.n_seq_max,
                settings.n_threads,
                settings.n_threads_batch,
                settings.n_devices
            ),
            (1, 4, 4, 0)
        );
        assert!(!settings.causal_attn && !settings.has_memory);
        eprintln!("decision effective context: {settings:?}");
        let mut executed = 0;
        let mut passes = 0;
        for case in cases() {
            if case["xybrid"]["evaluate"] != "reference" {
                continue;
            }
            eprintln!("G2 case {}/63: {}", executed + 1, case["id"]);
            let observed = reference(&directory, &case);
            let expected = &observed["upstream"];
            for (index, answer) in expected["prepare"]["answers"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let prepared = plan(&expected["prepare"]["passes"][index], answer);
                let raw = context
                    .evaluate(&prepared.tokens, &prepared.select, None)
                    .unwrap();
                let expected: Vec<_> = expected["evaluate"]["raw_scores"][index]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(f32_bits)
                    .collect();
                assert_eq!(
                    raw.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{}: matched-build selected logits",
                    case["id"]
                );
                passes += 1;
            }
            executed += 1;
        }
        assert_eq!(executed, 63, "a skipped or narrowed model corpus must fail");
        assert_eq!(passes, 78, "every question must execute exactly one pass");
        eprintln!("G2 executed {executed} model cases, {passes} independent whole-prompt passes");
        if let Some(output) = std::env::var_os("XYBRID_DECISION_REPORT_DIR").map(PathBuf::from) {
            std::fs::create_dir_all(&output).unwrap();
            let reference_hash =
                Sha256::digest(std::fs::read(directory.join("collection.json")).unwrap());
            let report = serde_json::json!({
                "schema_version": 1, "gate": "G2", "status": "passed",
                "profile": "lfm3-d1/text/v1", "model_cases_executed": executed,
                "whole_prompt_passes_executed": passes, "selected_logits_bit_equal": true,
                "reference_collection_sha256": format!("{reference_hash:x}"),
                "model_sha256": "adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b",
                "effective_context": {
                    "n_ctx": settings.n_ctx, "n_ctx_seq": settings.n_ctx_seq,
                    "n_batch": settings.n_batch, "n_ubatch": settings.n_ubatch,
                    "n_seq_max": settings.n_seq_max, "n_threads": settings.n_threads,
                    "n_threads_batch": settings.n_threads_batch, "offload_devices": settings.n_devices,
                    "causal_attention": settings.causal_attn, "has_memory": settings.has_memory
                },
                "scope": "matched native CPU configuration; no portable numerical tolerance"
            });
            std::fs::write(
                output.join("g2.json"),
                serde_json::to_vec_pretty(&report).unwrap(),
            )
            .unwrap();
        }
    }

    #[test]
    fn native_context_recovers_after_reordering_refusal_and_cancellation() {
        let _guard = NATIVE.lock().unwrap();
        let Some(path) = model() else {
            return;
        };
        let model = LlamaModel::load_decision_cpu(path.to_str().unwrap()).unwrap();
        let mut context = DecisionContext::new(&model).unwrap();
        let corpus = cases();
        let mut plans = corpus
            .iter()
            .filter(|case| case["xybrid"]["evaluate"] == "reference")
            .filter(|case| {
                case["upstream"]["prepare"]["passes"][0]["token_ids"]
                    .as_array()
                    .unwrap()
                    .len()
                    < 100
            })
            .map(|case| {
                plan(
                    &case["upstream"]["prepare"]["passes"][0],
                    &case["upstream"]["prepare"]["answers"][0],
                )
            });
        let a = plans.next().unwrap();
        let b = plans.next().unwrap();
        let first = context.evaluate(&a.tokens, &a.select, None).unwrap();
        context.evaluate(&b.tokens, &b.select, None).unwrap();
        assert_eq!(
            context.evaluate(&a.tokens, &a.select, None).unwrap(),
            first,
            "A → B → A"
        );
        assert!(context
            .evaluate(&vec![a.tokens[0]; 4097], &a.select, None)
            .is_err());
        assert!(context.evaluate(&a.tokens, &[-1], None).is_err());
        assert!(matches!(
            context.evaluate(&a.tokens, &a.select, Some(&AtomicBool::new(true))),
            Err(xybrid_llama::LlamaError::DecisionCancelled)
        ));
        assert_eq!(
            context.evaluate(&a.tokens, &a.select, None).unwrap(),
            first,
            "refusal/abort → valid"
        );
        let longest = corpus
            .iter()
            .find(|case| case["id"] == "budget-at-limit")
            .unwrap();
        let long = plan(
            &longest["upstream"]["prepare"]["passes"][0],
            &longest["upstream"]["prepare"]["answers"][0],
        );
        assert_eq!(long.tokens.len(), 4096);
        let abort = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                abort.store(true, std::sync::atomic::Ordering::Release);
            });
            assert!(
                matches!(
                    context.evaluate(&long.tokens, &long.select, Some(&abort)),
                    Err(xybrid_llama::LlamaError::DecisionCancelled)
                ),
                "abort during whole-prompt native evaluation"
            );
        });
        assert_eq!(
            context.evaluate(&a.tokens, &a.select, None).unwrap(),
            first,
            "mid-pass abort → valid"
        );
    }

    #[test]
    fn sdk_sync_async_returns_ordered_decisions_and_usage() {
        let _guard = NATIVE.lock().unwrap();
        let Some(path) = model() else {
            return;
        };
        let Some(directory) = references() else {
            return;
        };
        let model = xybrid_sdk::ModelLoader::from_directory(path.parent().unwrap())
            .unwrap()
            .load()
            .unwrap();
        model.warmup().unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let mut executed = 0;
        for case in cases() {
            if !matches!(
                case["id"].as_str().unwrap(),
                "card-text-example" | "multi-question-order" | "state-numbers" | "noul-false-first"
            ) {
                continue;
            }
            let request =
                xybrid_sdk::decode_decision_request(case["request_text"].as_str().unwrap())
                    .unwrap();
            let envelope = xybrid_core::ir::Envelope::decision_request(request);
            let sync = model.run(&envelope, None).unwrap();
            let asynchronous = runtime.block_on(model.run_async(&envelope, None)).unwrap();
            assert_eq!(
                sync.decisions(),
                asynchronous.decisions(),
                "sync/async parity"
            );
            if case["id"] == "multi-question-order" {
                let (first, second) = runtime.block_on(async {
                    tokio::join!(
                        model.run_async(&envelope, None),
                        model.run_async(&envelope, None)
                    )
                });
                assert_eq!(
                    first.unwrap().decisions(),
                    sync.decisions(),
                    "first queued caller"
                );
                assert_eq!(
                    second.unwrap().decisions(),
                    sync.decisions(),
                    "second queued caller"
                );
            }
            let reference = reference(&directory, &case);
            let expected = &reference["upstream"]["finalize"];
            let answers = sync.decisions().expect("typed decisions").answers();
            let prepared = reference["upstream"]["prepare"]["answers"]
                .as_array()
                .unwrap();
            assert_eq!(answers.len(), prepared.len());
            for (answer, plan) in answers.iter().zip(prepared) {
                assert_eq!(
                    answer.question_id(),
                    plan["question"].as_str().unwrap(),
                    "request order"
                );
                assert_answer(
                    case["id"].as_str().unwrap(),
                    answer,
                    &expected["answers"][answer.question_id()],
                );
            }
            assert_eq!(sync.tokens_in(), expected["usage"]["input_tokens"].as_u64());
            assert_eq!(sync.tokens_out(), Some(0));
            executed += 1;
        }
        assert_eq!(executed, 4);
    }
}
