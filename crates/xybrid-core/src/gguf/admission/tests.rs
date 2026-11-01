//! Admission's test vectors: the admission matrix (every classification
//! outcome under a decision declaration and a chat declaration), the load
//! routes' stale-metadata cases, what an admitted model carries, and a check
//! against LiquidAI's d1-2B GGUF when it is staged.

use std::path::PathBuf;

use super::super::testing::{d1_builder, gguf_string, GgufBuilder};
use super::super::{GgufLimits, UnsupportedReason};
use super::*;

const MODEL_FILE: &str = "model.gguf";

/// Metadata as a bundle's `model_metadata.json` holds it.
fn bundle(template: serde_json::Value) -> ModelMetadata {
    serde_json::from_value(serde_json::json!({
        "model_id": "bundle-model",
        "version": "1.0",
        "execution_template": template,
        "files": [MODEL_FILE],
    }))
    .expect("valid metadata")
}

fn decision_with(fields: serde_json::Value) -> ModelMetadata {
    let mut scorer = serde_json::json!({
        "type": "GgufDecision", "model_file": MODEL_FILE, "protocol": "lfm3-d1",
    });
    scorer
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    bundle(serde_json::json!({ "type": "ChoiceScorer", "scorer": scorer }))
}

fn decision() -> ModelMetadata {
    decision_with(serde_json::json!({}))
}

fn chat() -> ModelMetadata {
    bundle(serde_json::json!({ "type": "Gguf", "model_file": MODEL_FILE }))
}

fn vision() -> ModelMetadata {
    let mut vision = bundle(serde_json::json!({
        "type": "VisionLanguage", "model_file": MODEL_FILE,
    }));
    vision.files.push("mmproj.gguf".to_string());
    vision.vision_encoder = serde_json::from_value(serde_json::json!({
        "file": "mmproj.gguf", "preprocessing_preset": "siglip", "image_size": 512,
    }))
    .unwrap();
    vision
}

/// A model directory holding `gguf` as [`MODEL_FILE`].
fn model_dir(gguf: &[u8]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(MODEL_FILE), gguf).unwrap();
    dir
}

fn admit(metadata: &ModelMetadata, gguf: &[u8]) -> Result<GgufAdmission, AdapterError> {
    admit_gguf(metadata, model_dir(gguf).path())
}

fn admitted(admission: GgufAdmission) -> AdmittedDecisionModel {
    match admission {
        GgufAdmission::Decision(model) => model,
        other => panic!("expected a decision model, got {other:?}"),
    }
}

/// The capability and hint of an `UnsupportedModelCapability` refusal.
fn capability_refusal(result: Result<GgufAdmission, AdapterError>) -> (String, String) {
    match result {
        Err(AdapterError::UnsupportedModelCapability {
            model_id,
            capability,
            hint,
        }) => {
            assert_eq!(model_id, "bundle-model");
            (capability, hint)
        }
        other => panic!("expected an unsupported capability, got {other:?}"),
    }
}

fn invalid_input(result: Result<GgufAdmission, AdapterError>) -> String {
    match result {
        Err(AdapterError::InvalidInput(message)) => message,
        other => panic!("expected invalid input, got {other:?}"),
    }
}

/// A decision GGUF of `decision_type`, otherwise shaped like d1-2B's.
fn typed(decision_type: &str) -> GgufBuilder {
    d1_builder()
        .without("lfm3.decision.type")
        .str("lfm3.decision.type", decision_type)
}

fn llama() -> GgufBuilder {
    GgufBuilder::new()
        .str("general.architecture", "llama")
        .u32("llama.context_length", 8192)
        .str("tokenizer.chat_template", "{{ messages }}")
}

// ---------------------------------------------------------------------------
// The admission matrix
// ---------------------------------------------------------------------------

/// What classification makes of a row's GGUF.
#[derive(Debug, Clone)]
enum Class {
    /// `lfm3-d1/text/v1`.
    Supported,
    /// A decision model no profile admits, for this reason.
    Unsupported(UnsupportedReason),
    /// Not a decision model.
    NotDecision,
    /// An error: the header does not read or classify completely.
    Malformed,
}

struct Row {
    name: &'static str,
    gguf: Vec<u8>,
    class: Class,
}

fn row(name: &'static str, gguf: GgufBuilder, class: Class) -> Row {
    Row {
        name,
        gguf: gguf.build(),
        class,
    }
}

/// Checks one row: its classification, then its admission as a
/// `GgufDecision` scorer for `lfm3-d1` and as a chat model.
fn check_row(row: &Row) {
    let Row { name, gguf, class } = row;
    let classification =
        GgufHeader::read(&gguf[..], &GgufLimits::default()).and_then(|header| classify(&header));
    let decision_type = match (class, &classification) {
        (Class::Supported, Ok(DecisionClassification::Supported(profile))) => {
            assert_eq!(*profile, DecisionProfileId::Lfm3D1Text, "{name}");
            None
        }
        (
            Class::Unsupported(expected),
            Ok(DecisionClassification::Unsupported {
                decision_type,
                reason,
            }),
        ) => {
            assert_eq!(reason, expected, "{name}");
            Some(decision_type.clone())
        }
        (Class::NotDecision, Ok(DecisionClassification::NotDecision))
        | (Class::Malformed, Err(_)) => None,
        (class, classified) => panic!("{name}: expected {class:?}, classified {classified:?}"),
    };

    let as_decision = admit(&decision(), gguf);
    let as_chat = admit(&chat(), gguf);
    match class {
        Class::Supported => {
            let model = admitted(as_decision.unwrap_or_else(|e| panic!("{name}: {e}")));
            assert_eq!(model.profile(), DecisionProfileId::Lfm3D1Text, "{name}");
            let (capability, hint) = capability_refusal(as_chat);
            assert_eq!(capability, "Gguf execution", "{name}");
            assert!(
                hint.contains("lfm3-d1/text/v1 decision model"),
                "{name}: {hint}"
            );
        }
        Class::Unsupported(_) => {
            let decision_type = decision_type.expect("an unsupported decision type");
            if decision_type == "lfm3-d1" {
                let (capability, hint) = capability_refusal(as_decision);
                assert_eq!(capability, "lfm3-d1 decisions", "{name}");
                assert!(
                    hint.contains("no profile in this build admits it"),
                    "{name}: {hint}"
                );
            } else {
                let message = invalid_input(as_decision);
                assert!(
                    message.contains("for the lfm3-d1 protocol")
                        && message.contains(&format!("decision type is {decision_type:?}")),
                    "{name}: {message}"
                );
            }
            let (capability, hint) = capability_refusal(as_chat);
            assert_eq!(capability, "Gguf execution", "{name}");
            assert!(
                hint.contains("that this build does not run")
                    && hint.contains("does not load as any other model"),
                "{name}: {hint}"
            );
        }
        Class::NotDecision => {
            let (capability, hint) = capability_refusal(as_decision);
            assert_eq!(capability, "lfm3-d1 decisions", "{name}");
            assert!(hint.contains("not a decision model"), "{name}: {hint}");
            assert!(
                matches!(as_chat, Ok(GgufAdmission::NotDecision)),
                "{name}: {as_chat:?}"
            );
        }
        Class::Malformed => {
            for result in [as_decision, as_chat] {
                let message = invalid_input(result);
                assert!(
                    message.starts_with("model 'bundle-model': its GGUF cannot be classified"),
                    "{name}: {message}"
                );
            }
        }
    }
}

fn decision_key(key: &str) -> UnsupportedReason {
    UnsupportedReason::DecisionKey {
        key: key.to_string(),
    }
}

/// Every row of the admission matrix the classifier on this base decides.
fn matrix() -> Vec<Row> {
    use Class::*;
    let without = |key: &str| d1_builder().without(key);
    let causal = || without("lfm3.attention.causal");
    let temperature = "lfm3.decision.temperature.choice";
    let untyped = || without("lfm3.decision.type");

    let mut large = GgufBuilder::new()
        .str("general.architecture", "lfm3")
        .u32("lfm3.context_length", 128_000)
        .str("tokenizer.chat_template", "{{ messages }}");
    for i in 0..1500 {
        large = large.u32(&format!("filler.{i}"), i);
    }
    let large = large
        .bool("lfm3.attention.causal", false)
        .str("tokenizer.chat_template.systemone", "{{ state }}")
        .str("lfm3.decision.type", "lfm3-d1");

    let mut rows = vec![
        // Decision types.
        row("lfm3-d1, shaped like d1-2B", d1_builder(), Supported),
        row(
            "lfm3-d1-omni, known but not admitted yet",
            typed("lfm3-d1-omni"),
            Unsupported(UnsupportedReason::ProtocolNotYetAdmitted),
        ),
        row(
            "an unknown decision type",
            typed("laya"),
            Unsupported(UnsupportedReason::Protocol),
        ),
        row(
            "a decision type in another case",
            typed("LFM3-D1"),
            Unsupported(UnsupportedReason::Protocol),
        ),
        row(
            "lfm3-d1 on another architecture",
            GgufBuilder::new()
                .str("general.architecture", "lfm2")
                .bool("lfm2.attention.causal", false)
                .str("lfm2.decision.type", "lfm3-d1")
                .str("tokenizer.chat_template.systemone", "{{ state }}"),
            Unsupported(UnsupportedReason::Architecture),
        ),
        // `<arch>.attention.causal`.
        row(
            "attention.causal absent (llama.cpp reads causal)",
            causal(),
            Unsupported(UnsupportedReason::CausalAttention),
        ),
        row(
            "attention.causal true",
            causal().bool("lfm3.attention.causal", true),
            Unsupported(UnsupportedReason::CausalAttention),
        ),
        row(
            "attention.causal as a u8",
            causal().u8("lfm3.attention.causal", 0),
            Malformed,
        ),
        row(
            "attention.causal as a string",
            causal().str("lfm3.attention.causal", "false"),
            Malformed,
        ),
        // The `systemone` template.
        row(
            "systemone template missing",
            without("tokenizer.chat_template.systemone"),
            Unsupported(UnsupportedReason::MissingTemplate),
        ),
        row(
            "only the default chat template",
            without("tokenizer.chat_template.systemone")
                .str("tokenizer.chat_template", "{{ messages }}"),
            Unsupported(UnsupportedReason::MissingTemplate),
        ),
        row(
            "systemone template as an array",
            without("tokenizer.chat_template.systemone")
                .str_array("tokenizer.chat_template.systemone", &["{{ state }}"]),
            Malformed,
        ),
        // Other decision keys.
        row(
            "score_bias, which llama.cpp would apply",
            d1_builder().f32("lfm3.decision.score_bias.5.0", 0.5),
            Unsupported(decision_key("lfm3.decision.score_bias.5.0")),
        ),
        row(
            "max_head_tokens",
            d1_builder().u32("lfm3.decision.max_head_tokens", 64),
            Unsupported(decision_key("lfm3.decision.max_head_tokens")),
        ),
        // Temperatures.
        row(
            "temperature.image, unused by the text profile",
            d1_builder().f32("lfm3.decision.temperature.image", 3.0),
            Supported,
        ),
        row(
            "a temperature bucket",
            d1_builder().f32("lfm3.decision.temperature.choice.many", 2.0),
            Supported,
        ),
        row(
            "no temperatures (T = 1)",
            {
                d1_builder()
                    .without("lfm3.decision.temperature.noul")
                    .without(temperature)
                    .without("lfm3.decision.temperature.score")
            },
            Supported,
        ),
        // The profile accepts only f32 temperatures. llama.cpp would read
        // another numeric type through its string accessor; the profile
        // refuses it instead of guessing at its precision.
        row(
            "an f64 temperature (profile restriction)",
            without(temperature).f64(temperature, 1.3),
            Malformed,
        ),
        row(
            "a u32 temperature (profile restriction)",
            without(temperature).u32(temperature, 1),
            Malformed,
        ),
        row(
            "an f32-array temperature",
            without(temperature).f32_array(temperature, &[1.3]),
            Malformed,
        ),
        row(
            "a zero temperature, which llama.cpp refuses",
            without(temperature).f32(temperature, 0.0),
            Malformed,
        ),
        row(
            "a temperature that reads as zero (1e-7)",
            without(temperature).f32(temperature, 1e-7),
            Malformed,
        ),
        row(
            "a temperature without a name",
            d1_builder().f32("lfm3.decision.temperature.", 1.0),
            Malformed,
        ),
        // Malformed decision tagging.
        row(
            "an array-typed decision type",
            untyped().str_array("lfm3.decision.type", &["lfm3-d1"]),
            Malformed,
        ),
        row(
            "a numeric decision type",
            untyped().u32("lfm3.decision.type", 1),
            Malformed,
        ),
        row(
            "a decision type that is not UTF-8",
            untyped().bytes_str("lfm3.decision.type", &[0xff]),
            Malformed,
        ),
        row(
            "decision keys without a decision type",
            untyped(),
            Malformed,
        ),
        // Foreign-prefix decision types (other foreign decision keys are in
        // `the_admission_matrix_foreign_decision_keys`).
        row(
            "a decision type under another architecture's prefix",
            d1_builder().str("lfm2.decision.type", "lfm2-d1"),
            Malformed,
        ),
        row(
            "a decision type under general",
            d1_builder().str("general.decision.type", "lfm3-d1"),
            Malformed,
        ),
        row(
            "an ordinary model tagged for another architecture",
            llama().str("lfm3.decision.type", "lfm3-d1"),
            Malformed,
        ),
        // Ordinary models.
        row("an ordinary llama model", llama(), NotDecision),
        row(
            "an LFM3 chat model",
            GgufBuilder::new()
                .str("general.architecture", "lfm3")
                .bool("lfm3.attention.causal", true)
                .str("tokenizer.chat_template", "{{ messages }}"),
            NotDecision,
        ),
        row(
            "a model without an architecture",
            GgufBuilder::new().u32("x", 1),
            Malformed,
        ),
        // Where the decision type sits.
        row(
            "the decision type after 1,500 keys and the context length",
            large,
            Supported,
        ),
    ];

    let d1 = d1_builder().build();
    rows.push(Row {
        name: "a header cut inside its decision type",
        gguf: d1[..d1.len() / 2].to_vec(),
        class: Class::Malformed,
    });
    rows.push(Row {
        name: "not a GGUF",
        gguf: b"not a GGUF at all".to_vec(),
        class: Class::Malformed,
    });
    let mut huge_key = GgufBuilder::new().declare(0, 1).build();
    huge_key.extend(gguf_string(b"general.architecture"));
    huge_key.extend(8u32.to_le_bytes());
    huge_key.extend((1u64 << 29).to_le_bytes());
    rows.push(Row {
        name: "a length past the end of the file",
        gguf: huge_key,
        class: Class::Malformed,
    });
    rows
}

#[test]
fn the_admission_matrix() {
    for row in matrix() {
        check_row(&row);
    }
}

/// Decision keys other than `*.decision.type` under a prefix that is not
/// the architecture's (C12: any `<other_arch>.decision.*` key is an error).
/// The classifier on this base checks only a foreign `*.decision.type`; P2
/// fixes the rule in `classify.rs` on its own branch.
#[test]
#[ignore = "needs P2's classify fix"]
fn the_admission_matrix_foreign_decision_keys() {
    for row in [
        row(
            "a temperature under another architecture's prefix",
            d1_builder().f32("lfm2.decision.temperature.choice", 1.0),
            Class::Malformed,
        ),
        row(
            "an ordinary model with another architecture's temperature",
            llama().f32("lfm3.decision.temperature.choice", 1.0),
            Class::Malformed,
        ),
        row(
            "an ordinary model with another architecture's score_bias",
            llama().f32("lfm3.decision.score_bias.5.0", 0.5),
            Class::Malformed,
        ),
    ] {
        check_row(&row);
    }
}

// ---------------------------------------------------------------------------
// Load routes and declarations
// ---------------------------------------------------------------------------

#[test]
fn every_declaration_but_a_decision_scorer_refuses_a_decision_gguf() {
    let d1 = d1_builder().build();
    for (label, metadata) in [("Gguf", chat()), ("VisionLanguage", vision())] {
        let (capability, hint) = capability_refusal(admit(&metadata, &d1));
        assert_eq!(capability, format!("{label} execution"));
        assert!(
            hint.contains("(decision type \"lfm3-d1\"), which loads only as a ChoiceScorer with a GgufDecision scorer"),
            "{label}: {hint}"
        );
        assert!(matches!(
            admit(&metadata, &llama().build()),
            Ok(GgufAdmission::NotDecision)
        ));
    }
}

/// Templates llama.cpp builds no model for are admitted unread, so even a
/// missing file passes.
#[test]
fn templates_that_build_no_llama_cpp_model_are_not_read() {
    let empty = tempfile::tempdir().unwrap();
    for template in [
        serde_json::json!({ "type": "Onnx", "model_file": MODEL_FILE }),
        serde_json::json!({ "type": "ZzzPhonon", "model_file": MODEL_FILE }),
        serde_json::json!({ "type": "GgmlWhisper", "model_file": MODEL_FILE }),
        serde_json::json!({ "type": "SafeTensors", "model_file": MODEL_FILE }),
        serde_json::json!({
            "type": "ChoiceScorer",
            "scorer": {
                "type": "OnnxByteOptionScorer", "model_file": MODEL_FILE,
                "context": { "offset": 1, "max_len": 8, "overflow": "Truncate",
                             "ids_input": "c", "mask_input": "cm" },
                "choice": { "offset": 1, "max_len": 8, "overflow": "Truncate",
                            "ids_input": "o", "mask_input": "om" },
                "choice_mask_input": "m", "logits_output": "logits", "max_choices": 4,
            },
        }),
    ] {
        let metadata = bundle(template.clone());
        assert!(
            matches!(
                admit_gguf(&metadata, empty.path()),
                Ok(GgufAdmission::NotDecision)
            ),
            "{template}"
        );
    }

    // The projector is the clip loader's: only the language model is read.
    let dir = model_dir(&llama().build());
    assert!(!dir.path().join("mmproj.gguf").exists());
    assert!(matches!(
        admit_gguf(&vision(), dir.path()),
        Ok(GgufAdmission::NotDecision)
    ));
}

/// A stale sidecar or cached bundle: metadata read back from its directory,
/// as a load reads it, checked against the GGUF the directory holds now.
#[test]
fn stale_sidecars_and_cached_bundles_are_checked_against_their_gguf() {
    fn cached(dir: &Path, metadata: &ModelMetadata, gguf: &[u8]) -> ModelMetadata {
        std::fs::write(
            dir.join("model_metadata.json"),
            serde_json::to_string_pretty(metadata).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join(MODEL_FILE), gguf).unwrap();
        serde_json::from_str(&std::fs::read_to_string(dir.join("model_metadata.json")).unwrap())
            .unwrap()
    }
    let dir = tempfile::tempdir().unwrap();
    let d1 = d1_builder().build();

    // A chat sidecar written before the file was classified.
    let metadata = cached(dir.path(), &chat(), &d1);
    capability_refusal(admit_gguf(&metadata, dir.path()));

    // A cached vision bundle around a decision model.
    let metadata = cached(dir.path(), &vision(), &d1);
    capability_refusal(admit_gguf(&metadata, dir.path()));

    // A cached decision bundle whose GGUF is now a chat model.
    let metadata = cached(dir.path(), &decision(), &llama().build());
    capability_refusal(admit_gguf(&metadata, dir.path()));

    // The intact decision bundle.
    let metadata = cached(dir.path(), &decision(), &d1);
    let model = admitted(admit_gguf(&metadata, dir.path()).unwrap());
    assert_eq!(model.profile(), DecisionProfileId::Lfm3D1Text);
}

#[test]
fn an_admitted_model_carries_what_its_load_needs() {
    let d1 = d1_builder().build();
    let dir = model_dir(&d1);
    let metadata = decision_with(serde_json::json!({ "token_budget": 2048, "max_questions": 4 }));
    let model = admitted(admit_gguf(&metadata, dir.path()).unwrap());

    let ExecutionTemplate::ChoiceScorer {
        scorer: ChoiceScorerSpec::GgufDecision(spec),
    } = &metadata.execution_template
    else {
        unreachable!("a decision bundle");
    };
    assert_eq!(model.spec(), spec);
    assert_eq!(model.profile(), DecisionProfileId::Lfm3D1Text);
    assert_eq!(model.path(), dir.path().join(MODEL_FILE));
    assert_eq!(
        model.header(),
        &GgufHeader::read(&d1[..], &GgufLimits::default()).unwrap()
    );
    assert_eq!(model.file_len(), d1.len() as u64);
    let on_disk = std::fs::metadata(dir.path().join(MODEL_FILE)).unwrap();
    assert_eq!(model.modified(), on_disk.modified().unwrap());

    model.verify_unchanged().expect("the file is as admitted");
    std::fs::write(dir.path().join(MODEL_FILE), llama().build()).unwrap();
    match model.verify_unchanged() {
        Err(AdapterError::IOError(error)) => {
            assert!(error.to_string().contains("changed since it was admitted"))
        }
        other => panic!("a replaced file passed: {other:?}"),
    }
    std::fs::remove_file(dir.path().join(MODEL_FILE)).unwrap();
    assert!(matches!(
        model.verify_unchanged(),
        Err(AdapterError::IOError(error)) if error.kind() == std::io::ErrorKind::NotFound
    ));
}

/// The spec is validated before anything is read: an invalid one fails
/// even without a file.
#[test]
fn an_invalid_spec_is_refused_before_the_gguf_is_read() {
    let empty = tempfile::tempdir().unwrap();
    for fields in [
        serde_json::json!({ "token_budget": 0 }),
        serde_json::json!({ "token_budget": 8193 }),
        serde_json::json!({ "flash_attn": false, "token_budget": 8192 }),
        serde_json::json!({ "max_questions": 0 }),
        serde_json::json!({ "max_options": 256 }),
        serde_json::json!({ "max_total_tokens": 0 }),
        serde_json::json!({ "model_file": "" }),
    ] {
        let message = invalid_input(admit_gguf(&decision_with(fields.clone()), empty.path()));
        assert!(
            message.starts_with("invalid choice scorer spec:"),
            "{fields}: {message}"
        );
    }
}

/// The decision profiles read no media, so a media companion is refused
/// with the spec, before anything is read, and the message names no file.
#[test]
fn a_media_companion_on_a_decision_bundle_is_refused_before_the_gguf_is_read() {
    let mut metadata = decision();
    metadata.files.push("mmproj.gguf".to_string());
    metadata.vision_encoder = serde_json::from_value(serde_json::json!({
        "file": "mmproj.gguf", "preprocessing_preset": "siglip", "image_size": 512,
    }))
    .unwrap();
    let empty = tempfile::tempdir().unwrap();
    assert_eq!(
        invalid_input(admit_gguf(&metadata, empty.path())),
        "model 'bundle-model': a GgufDecision scorer reads no media, so it takes no \
         vision_encoder block"
    );

    metadata.vision_encoder = None;
    let dir = model_dir(&d1_builder().build());
    admitted(admit_gguf(&metadata, dir.path()).unwrap());
}

#[test]
fn a_missing_gguf_is_an_io_error_naming_the_model() {
    let empty = tempfile::tempdir().unwrap();
    for metadata in [decision(), chat()] {
        match admit_gguf(&metadata, empty.path()) {
            Err(AdapterError::IOError(error)) => {
                assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
                assert!(
                    error
                        .to_string()
                        .starts_with("model 'bundle-model': cannot read its GGUF"),
                    "{error}"
                );
            }
            other => panic!("expected an I/O error, got {other:?}"),
        }
    }
}

#[test]
fn every_truncation_of_a_decision_header_is_refused() {
    let d1 = d1_builder().build();
    let dir = tempfile::tempdir().unwrap();
    for len in 0..d1.len() {
        std::fs::write(dir.path().join(MODEL_FILE), &d1[..len]).unwrap();
        for metadata in [decision(), chat()] {
            let message = invalid_input(admit_gguf(&metadata, dir.path()));
            assert!(
                message.contains("cannot be classified"),
                "cut at {len}: {message}"
            );
        }
    }
}

#[test]
fn messages_quote_the_file_s_facts_cut_short() {
    let long_type = "x".repeat(10_000);
    let message = invalid_input(admit(&decision(), &typed(&long_type).build()));
    assert!(message.len() < 400, "{} bytes", message.len());
    assert!(message.contains(&format!("\"{}\"…", "x".repeat(64))));

    let long_key = format!("lfm3.decision.{}", "k".repeat(10_000));
    let (_, hint) = capability_refusal(admit(
        &decision(),
        &d1_builder().f32(&long_key, 0.5).build(),
    ));
    assert!(hint.len() < 500, "{} bytes", hint.len());

    let wrong_type = format!("lfm3.decision.temperature.{}", "t".repeat(10_000));
    let message = invalid_input(admit(&chat(), &d1_builder().u32(&wrong_type, 1).build()));
    assert!(message.len() < 500, "{} bytes", message.len());

    assert_eq!(quoted(&"é".repeat(100)), format!("{:?}…", "é".repeat(64)));
    assert_eq!(quoted("lfm3-d1"), "\"lfm3-d1\"");
}

#[test]
fn an_admitted_model_prints_a_summary_not_its_header() {
    let model = admitted(admit(&decision(), &d1_builder().build()).unwrap());
    let debug = format!("{model:?}");
    assert!(
        debug.starts_with("AdmittedDecisionModel { spec: GgufDecisionSpec {"),
        "{debug}"
    );
    assert!(debug.contains("profile: Lfm3D1Text"), "{debug}");
    assert!(debug.contains("metadata_keys: 10, tensors: 1"), "{debug}");
    assert!(!debug.contains("{{"), "the template stays out: {debug}");
}

#[test]
fn every_profile_s_protocol_classifies_as_that_profile() {
    // A new profile joins this list with its protocol.
    const PROFILES: &[DecisionProfileId] = &[DecisionProfileId::Lfm3D1Text];
    for &profile in PROFILES {
        let protocol = DecisionProtocol::from(profile);
        let header = GgufHeader::read(
            &typed(protocol.as_str()).build()[..],
            &GgufLimits::default(),
        )
        .unwrap();
        assert_eq!(
            classify(&header).unwrap(),
            DecisionClassification::Supported(profile)
        );
    }
}

// ---------------------------------------------------------------------------
// The published d1-2B GGUF (artifact `d1-2b-q4km`), when staged
// ---------------------------------------------------------------------------

mod published {
    use super::*;
    use crate::testing::model_fixtures;

    /// The artifact id and file of the coordinator's pin
    /// (`integration-tests/fixtures/decision/artifacts.json`):
    /// LiquidAI/d1-2B-GGUF at e968f6575f0d4fad928312d7af15cad90fd365c0,
    /// sha256 adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b.
    const MODEL_ID: &str = "d1-2b-q4km";
    const D1_FILE: &str = "d1-2B-Q4_K_M.gguf";

    /// The staged model directory, or `None` (with a note) when the model is
    /// absent.
    ///
    /// # Panics
    ///
    /// When `XYBRID_REQUIRE_MODELS` is set and the model is not staged.
    fn staged_d1() -> Option<PathBuf> {
        let dir = model_fixtures::model_path(MODEL_ID).filter(|dir| dir.join(D1_FILE).is_file());
        if dir.is_none() {
            let reason = format!("'{MODEL_ID}/{D1_FILE}' is not staged");
            assert!(
                !model_fixtures::models_required(),
                "{} is set but {reason}",
                model_fixtures::ENV_REQUIRE_MODELS
            );
            eprintln!("Skipping test: {reason}");
        }
        dir
    }

    fn named(mut metadata: ModelMetadata) -> ModelMetadata {
        let text = serde_json::to_string(&metadata).unwrap();
        metadata = serde_json::from_str(&text.replace(MODEL_FILE, D1_FILE)).unwrap();
        metadata
    }

    #[test]
    fn d1_2b_is_admitted_only_as_a_gguf_decision_scorer() {
        let Some(dir) = staged_d1() else { return };
        let model = admitted(admit_gguf(&named(decision()), &dir).expect("d1-2B is admitted"));
        assert_eq!(model.profile(), DecisionProfileId::Lfm3D1Text);
        assert_eq!(model.path(), dir.join(D1_FILE));
        assert_eq!(model.file_len(), 1_099_629_888);
        // Nothing but the decision type tells it apart from a chat model: it
        // carries an ordinary chat template too.
        assert!(model.header().get("tokenizer.chat_template").is_some());
        model.verify_unchanged().unwrap();

        for (label, metadata) in [("Gguf", chat()), ("VisionLanguage", vision())] {
            let (capability, _) = capability_refusal(admit_gguf(&named(metadata), &dir));
            assert_eq!(capability, format!("{label} execution"));
        }
    }

    /// The sequence a load follows: admit, load natively, then check the
    /// file is still the one admitted and llama.cpp read the metadata the
    /// admitted header holds.
    #[cfg(feature = "llm-llamacpp")]
    #[test]
    fn an_admitted_d1_2b_loads_and_matches_its_admission() {
        let Some(dir) = staged_d1() else { return };
        let model = admitted(admit_gguf(&named(decision()), &dir).expect("d1-2B is admitted"));
        xybrid_llama::backend_init();
        let native =
            xybrid_llama::LlamaModel::load(model.path().to_str().expect("a UTF-8 path"), 0)
                .expect("load d1-2B");
        model.verify_unchanged().expect("the admitted file");
        crate::gguf::verify_native_metadata(model.header(), &native)
            .expect("llama.cpp read the admitted header");
    }
}
