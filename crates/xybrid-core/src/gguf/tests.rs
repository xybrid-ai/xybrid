//! The scanner's and the classification rule's test vectors, then checks
//! against LiquidAI's d1-2B GGUF when it is staged.

use super::native::compare_native;
use super::testing::{d1_builder, gguf_string, GgufBuilder, D1_TEMPERATURE};
use super::*;

fn read(bytes: &[u8]) -> Result<GgufHeader, GgufError> {
    GgufHeader::read(bytes, &GgufLimits::default())
}

fn classify_bytes(bytes: &[u8]) -> Result<DecisionClassification, GgufError> {
    classify(&read(bytes)?)
}

fn unsupported(decision_type: &str, reason: UnsupportedReason) -> DecisionClassification {
    DecisionClassification::Unsupported {
        decision_type: decision_type.to_string(),
        reason,
    }
}

const D1: DecisionClassification = DecisionClassification::Supported(DecisionProfileId::Lfm3D1Text);

// ---------------------------------------------------------------------------
// The header scanner
// ---------------------------------------------------------------------------

#[test]
fn every_type_reads_back_typed_in_file_order() {
    let bytes = GgufBuilder::new()
        .u8("u8", 200)
        .i8("i8", -5)
        .u16("u16", 65535)
        .i16("i16", -32768)
        .u32("u32", 4096)
        .i32("i32", -1)
        .f32("f32", 1.5)
        .bool("bool", true)
        .raw(b"bool2", 7, &[2]) // llama.cpp reads any non-zero byte as true
        .str("str", "lfm3")
        .bytes_str("bytes", &[0xff, 0xfe])
        .u64("u64", u64::MAX)
        .i64("i64", i64::MIN)
        .f64("f64", -0.25)
        .str_array("strs", &["a", "", "ccc"])
        .i32_array("i32s", &[3, -4])
        .f32_array("f32s", &[])
        .array("bools", 7, 3, &[0, 1, 9])
        .tensor("w", &[4, 2], 0, 0)
        .tensor("b", &[], 1, 32)
        .build();
    let header = read(&bytes).expect("a valid header");

    assert_eq!(header.version(), 3);
    assert_eq!(header.header_bytes(), bytes.len() as u64);
    let keys: Vec<&str> = header.metadata().map(|(key, _)| key).collect();
    assert_eq!(
        keys,
        [
            "u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "bool2", "str", "bytes", "u64",
            "i64", "f64", "strs", "i32s", "f32s", "bools"
        ]
    );
    assert_eq!(header.get("u8"), Some(&GgufValue::U8(200)));
    assert_eq!(header.get("i8"), Some(&GgufValue::I8(-5)));
    assert_eq!(header.get("u16"), Some(&GgufValue::U16(65535)));
    assert_eq!(header.get("i16"), Some(&GgufValue::I16(-32768)));
    assert_eq!(header.get("i32"), Some(&GgufValue::I32(-1)));
    assert_eq!(header.get("f32").and_then(GgufValue::as_f32), Some(1.5));
    assert_eq!(header.get("bool").and_then(GgufValue::as_bool), Some(true));
    assert_eq!(header.get("bool2").and_then(GgufValue::as_bool), Some(true));
    assert_eq!(header.get("str").and_then(GgufValue::as_str), Some("lfm3"));
    assert_eq!(
        header.get("bytes").and_then(GgufValue::as_str),
        None,
        "not UTF-8"
    );
    assert_eq!(
        header.get("bytes").and_then(GgufValue::as_bytes),
        Some(&[0xff, 0xfe][..]),
        "kept as bytes"
    );
    assert_eq!(
        header.get("u64").and_then(GgufValue::as_u64),
        Some(u64::MAX)
    );
    assert_eq!(
        header.get("i64").and_then(GgufValue::as_u64),
        None,
        "negative"
    );
    assert_eq!(header.get("u32").and_then(GgufValue::as_u64), Some(4096));
    assert_eq!(header.get("f64"), Some(&GgufValue::F64(-0.25)));

    let strs = header.get("strs").and_then(GgufValue::as_array).unwrap();
    assert_eq!(strs.elem_type(), GgufType::String);
    assert_eq!(
        strs.strings().unwrap().collect::<Vec<_>>(),
        [Some("a"), Some(""), Some("ccc")]
    );
    assert_eq!(
        header.get("i32s"),
        Some(&GgufValue::Array(GgufArray::I32(vec![3, -4])))
    );
    assert!(header
        .get("f32s")
        .and_then(GgufValue::as_array)
        .unwrap()
        .is_empty());
    assert_eq!(
        header.get("bools"),
        Some(&GgufValue::Array(GgufArray::Bool(vec![false, true, true])))
    );
    assert_eq!(header.get("strs").unwrap().type_name(), "arr[str]");
    assert_eq!(header.get("missing"), None);

    let tensors = header.tensors();
    assert_eq!(tensors.len(), 2);
    assert_eq!(
        (
            tensors[0].name(),
            tensors[0].dims(),
            tensors[0].ggml_type(),
            tensors[0].offset()
        ),
        ("w", &[4, 2][..], 0, 0)
    );
    assert_eq!((tensors[1].name(), tensors[1].dims()), ("b", &[][..]));
}

#[test]
fn type_codes_and_names_are_llama_cpps() {
    for code in 0..13 {
        let ty = GgufType::from_code(code).expect("codes 0..13 are types");
        assert_eq!(ty.code(), code);
    }
    assert_eq!(GgufType::from_code(13), None);
    let names: Vec<&str> = (0..13)
        .map(|code| GgufType::from_code(code).unwrap().name())
        .collect();
    assert_eq!(
        names,
        [
            "u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "str", "arr", "u64", "i64",
            "f64"
        ]
    );
}

#[test]
fn every_truncation_of_a_header_is_an_error() {
    let bytes = d1_builder().build();
    assert!(read(&bytes).is_ok());
    for len in 0..bytes.len() {
        match read(&bytes[..len]) {
            Err(GgufError::Truncated { .. }) => {}
            other => panic!(
                "a header cut at {len} of {} bytes read as {other:?}",
                bytes.len()
            ),
        }
    }
}

#[test]
fn a_length_past_the_end_of_a_file_is_refused_before_it_is_allocated() {
    let dir = tempfile::tempdir().unwrap();
    // One string key claiming 512 MiB (under llama.cpp's 1 GiB limit), in a
    // tiny file.
    let mut bytes = GgufBuilder::new().declare(0, 1).build();
    bytes.extend(gguf_string(b"general.name"));
    bytes.extend(8u32.to_le_bytes());
    bytes.extend((1u64 << 29).to_le_bytes());
    let cut = dir.path().join("cut.gguf");
    std::fs::write(&cut, &bytes).unwrap();
    assert!(matches!(
        GgufHeader::read_file(&cut),
        Err(GgufError::Truncated { .. })
    ));

    let good = dir.path().join("d1.gguf");
    std::fs::write(&good, d1_builder().build()).unwrap();
    assert_eq!(
        classify(&GgufHeader::read_file(&good).unwrap()).unwrap(),
        D1
    );
    assert!(matches!(
        GgufHeader::read_file(dir.path().join("absent.gguf")),
        Err(GgufError::Io(_))
    ));
}

#[test]
fn declared_counts_do_not_allocate_without_bytes() {
    // 2^62 keys, then arrays of 2^29 and 2^30 + 1 strings, in a few bytes.
    let huge_kv = GgufBuilder::new().declare(0, 1 << 62).build();
    assert!(matches!(read(&huge_kv), Err(GgufError::Truncated { .. })));
    let huge_array = GgufBuilder::new()
        .array("tokenizer.ggml.tokens", 8, 1 << 29, &[])
        .build();
    assert!(matches!(
        read(&huge_array),
        Err(GgufError::LimitExceeded { .. })
    ));
    let too_many = GgufBuilder::new()
        .array("tokenizer.ggml.tokens", 8, (1 << 30) + 1, &[])
        .build();
    assert!(matches!(read(&too_many), Err(GgufError::Malformed(_))));
}

#[test]
fn the_scan_stops_at_its_limit() {
    let bytes = d1_builder().build();
    let exact = GgufLimits::new(bytes.len() as u64);
    assert!(GgufHeader::read(&bytes[..], &exact).is_ok());
    let short = GgufLimits::new(bytes.len() as u64 - 1);
    assert!(matches!(
        GgufHeader::read(&bytes[..], &short),
        Err(GgufError::LimitExceeded { .. })
    ));
    assert_eq!(
        GgufLimits::default().max_header_bytes(),
        GgufLimits::DEFAULT_MAX_HEADER_BYTES
    );
}

#[test]
fn headers_llama_cpp_refuses_are_refused() {
    let base = || GgufBuilder::new().str("general.architecture", "llama");
    let malformed =
        |builder: GgufBuilder| matches!(read(&builder.build()), Err(GgufError::Malformed(_)));

    let mut bad_magic = base().build();
    bad_magic[0] = b'g';
    assert!(matches!(read(&bad_magic), Err(GgufError::NotGguf)));
    for version in [0, 1, 4, 0x0003_0000] {
        assert!(
            matches!(
                read(&base().version(version).build()),
                Err(GgufError::UnsupportedVersion(v)) if v == version
            ),
            "version {version:#x}"
        );
    }
    assert!(read(&base().version(2).build()).is_ok(), "v2 is read");

    assert!(
        malformed(base().str("general.architecture", "again")),
        "duplicate key"
    );
    assert!(malformed(base().str("", "empty")), "empty key");
    assert!(
        malformed(base().raw(&[0xff], 4, &1u32.to_le_bytes())),
        "key not UTF-8"
    );
    assert!(malformed(base().raw(b"k", 13, &[])), "unknown type");
    assert!(malformed(base().array("k", 9, 0, &[])), "array of arrays");
    assert!(
        malformed(base().array("k", 99, 0, &[])),
        "unknown element type"
    );
    assert!(
        malformed(base().declare(0, 1u64 << 63)),
        "negative key count"
    );
    assert!(
        malformed(base().declare(1u64 << 63, 1)),
        "negative tensor count"
    );

    assert!(read(&base().u32("general.alignment", 64).build()).is_ok());
    assert!(
        malformed(base().u32("general.alignment", 48)),
        "not a power of 2"
    );
    assert!(malformed(base().u32("general.alignment", 0)), "zero");
    assert!(matches!(
        read(&base().u64("general.alignment", 32).build()),
        Err(GgufError::WrongType { .. })
    ));

    assert!(read(&base().tensor(&"t".repeat(63), &[1], 0, 0).build()).is_ok());
    assert!(
        malformed(base().tensor(&"t".repeat(64), &[1], 0, 0)),
        "name too long"
    );
    assert!(
        malformed(base().tensor("t", &[1, 1, 1, 1, 1], 0, 0)),
        "five dimensions"
    );
    assert!(
        malformed(base().tensor("t", &[1u64 << 63], 0, 0)),
        "negative dimension"
    );
    assert!(
        malformed(base().tensor("t", &[1 << 32, 1 << 32], 0, 0)),
        "too many elements"
    );
    assert!(
        read(&base().tensor("t", &[0, 1 << 62, 1 << 62], 0, 0).build()).is_ok(),
        "an empty tensor never overflows"
    );
    assert!(
        malformed(base().tensor("t", &[1], 0, 0).tensor("t", &[1], 0, 32)),
        "duplicate tensor"
    );
    assert!(
        malformed(base().raw(b"long", 8, &(1u64 << 31).to_le_bytes())),
        "a string past llama.cpp's 1 GiB limit"
    );
}

#[test]
fn arrays_print_as_their_type_and_length() {
    let header = read(&d1_builder().build()).unwrap();
    let tokens = header.get("tokenizer.ggml.tokens").unwrap();
    assert_eq!(
        format!("{tokens:?}"),
        "Array(GgufArray { elem_type: String, len: 3 })"
    );
}

// ---------------------------------------------------------------------------
// The classification rule's test vectors
// ---------------------------------------------------------------------------

#[test]
fn an_ordinary_model_is_not_a_decision_model() {
    let llama = GgufBuilder::new()
        .str("general.architecture", "llama")
        .str("tokenizer.chat_template", "{{ messages }}")
        .bytes_str("general.description", &[0xff]) // unrelated bytes are fine
        .build();
    assert_eq!(
        classify_bytes(&llama).unwrap(),
        DecisionClassification::NotDecision
    );

    // LFM3 chat models share the architecture, not the decision keys.
    let lfm3_chat = GgufBuilder::new()
        .str("general.architecture", "lfm3")
        .bool("lfm3.attention.causal", true)
        .build();
    assert_eq!(
        classify_bytes(&lfm3_chat).unwrap(),
        DecisionClassification::NotDecision
    );
}

#[test]
fn lfm3_d1_is_the_supported_text_profile() {
    assert_eq!(classify_bytes(&d1_builder().build()).unwrap(), D1);
    assert_eq!(DecisionProfileId::Lfm3D1Text.to_string(), "lfm3-d1/text/v1");

    // Temperatures are optional, a temperature name may have dots (buckets),
    // and an empty template is still a template.
    let minimal = GgufBuilder::new()
        .str("general.architecture", "lfm3")
        .bool("lfm3.attention.causal", false)
        .str("lfm3.decision.type", "lfm3-d1")
        .str("tokenizer.chat_template.systemone", "")
        .f32("lfm3.decision.temperature.choice.many", 2.0)
        .build();
    assert_eq!(classify_bytes(&minimal).unwrap(), D1);
}

#[test]
fn a_decision_key_late_in_a_large_header_is_found() {
    let mut builder = GgufBuilder::new().str("general.architecture", "lfm3");
    for i in 0..3000 {
        builder = builder.u32(&format!("filler.{i}"), i);
    }
    let tokens: Vec<String> = (0..20_000).map(|i| format!("tok{i}")).collect();
    let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
    let bytes = builder
        .str_array("tokenizer.ggml.tokens", &tokens)
        .bool("lfm3.attention.causal", false)
        .str("tokenizer.chat_template.systemone", "{{ state }}")
        .str("lfm3.decision.type", "lfm3-d1")
        .build();
    assert_eq!(classify_bytes(&bytes).unwrap(), D1);
}

#[test]
fn other_decision_types_are_unsupported_not_ordinary() {
    let typed = |arch: &str, decision_type: &str| {
        GgufBuilder::new()
            .str("general.architecture", arch)
            .bool(&format!("{arch}.attention.causal"), false)
            .str(&format!("{arch}.decision.type"), decision_type)
            .str("tokenizer.chat_template.systemone", "{{ state }}")
            .build()
    };
    for (arch, decision_type) in [
        ("lfm3", "lfm3-d1-omni"), // admitted by a later profile
        ("lfm2", "lfm2-d1"),
        ("lfm2", "lfm2-d1-omni"),
        ("modern-bert", "laya"),
        ("lfm3", ""),
        ("lfm3", "LFM3-D1"),
    ] {
        assert_eq!(
            classify_bytes(&typed(arch, decision_type)).unwrap(),
            unsupported(decision_type, UnsupportedReason::Protocol),
            "{arch} / {decision_type:?}"
        );
    }
    assert_eq!(
        classify_bytes(&typed("lfm2", "lfm3-d1")).unwrap(),
        unsupported("lfm3-d1", UnsupportedReason::Architecture)
    );
}

#[test]
fn the_profile_needs_non_causal_attention_and_the_template() {
    let classified = |builder: GgufBuilder| classify_bytes(&builder.build()).unwrap();
    let causal = unsupported("lfm3-d1", UnsupportedReason::CausalAttention);
    assert_eq!(
        classified(d1_builder().without("lfm3.attention.causal")),
        causal,
        "absent reads as causal"
    );
    assert_eq!(
        classified(
            d1_builder()
                .without("lfm3.attention.causal")
                .bool("lfm3.attention.causal", true)
        ),
        causal
    );
    assert_eq!(
        classified(d1_builder().without("tokenizer.chat_template.systemone")),
        unsupported("lfm3-d1", UnsupportedReason::MissingTemplate)
    );
    // The default template is not the decision template.
    assert_eq!(
        classified(
            d1_builder()
                .without("tokenizer.chat_template.systemone")
                .str("tokenizer.chat_template", "{{ messages }}")
        ),
        unsupported("lfm3-d1", UnsupportedReason::MissingTemplate)
    );
}

#[test]
fn decision_keys_the_profile_does_not_implement_are_unsupported() {
    for key in [
        "lfm3.decision.score_bias.5.0",
        "lfm3.decision.max_head_tokens",
        "lfm3.decision.block_count",
        "lfm3.decision.labels",
        "lfm3.decision.temperatures",
        "lfm3.decision.something_new",
    ] {
        assert_eq!(
            classify_bytes(&d1_builder().f32(key, 0.5).build()).unwrap(),
            unsupported(
                "lfm3-d1",
                UnsupportedReason::DecisionKey {
                    key: key.to_string()
                }
            ),
            "{key}"
        );
    }
    let labels = d1_builder()
        .str_array("lfm3.decision.labels", &["yes", "no"])
        .f32("lfm3.decision.score_bias.2.0", 0.5)
        .build();
    assert_eq!(
        classify_bytes(&labels).unwrap(),
        unsupported(
            "lfm3-d1",
            UnsupportedReason::DecisionKey {
                key: "lfm3.decision.labels".to_string()
            }
        ),
        "the first in file order"
    );
}

#[test]
fn malformed_decision_metadata_is_an_error_not_absence() {
    let wrong_type = |builder: GgufBuilder, key: &str| match classify_bytes(&builder.build()) {
        Err(GgufError::WrongType { key: k, .. }) => k == key,
        other => panic!("{key}: {other:?}"),
    };
    let without_type = || d1_builder().without("lfm3.decision.type");
    assert!(wrong_type(
        without_type().str_array("lfm3.decision.type", &["lfm3-d1"]),
        "lfm3.decision.type"
    ));
    assert!(wrong_type(
        without_type().u32("lfm3.decision.type", 3),
        "lfm3.decision.type"
    ));
    assert!(matches!(
        classify_bytes(
            &without_type()
                .bytes_str("lfm3.decision.type", &[0xff])
                .build()
        ),
        Err(GgufError::Malformed(_))
    ));

    let with = |key: &str| d1_builder().without(key);
    assert!(wrong_type(
        with("lfm3.attention.causal").str("lfm3.attention.causal", "false"),
        "lfm3.attention.causal"
    ));
    assert!(wrong_type(
        with("lfm3.attention.causal").u8("lfm3.attention.causal", 0),
        "lfm3.attention.causal"
    ));
    assert!(wrong_type(
        with("tokenizer.chat_template.systemone")
            .str_array("tokenizer.chat_template.systemone", &["{{ state }}"]),
        "tokenizer.chat_template.systemone"
    ));
    let temperature = "lfm3.decision.temperature.choice";
    assert!(wrong_type(
        with(temperature).f64(temperature, 1.3),
        temperature
    ));
    assert!(wrong_type(
        with(temperature).u32(temperature, 1),
        temperature
    ));
    assert!(wrong_type(
        with(temperature).f32_array(temperature, &[1.3]),
        temperature
    ));

    let invalid = |builder: GgufBuilder| {
        matches!(
            classify_bytes(&builder.build()),
            Err(GgufError::InvalidDecision(_))
        )
    };
    for value in [0.0, -1.0, 1e-7, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            invalid(with(temperature).f32(temperature, value)),
            "temperature {value}"
        );
    }
    assert!(
        !invalid(with(temperature).f32(temperature, 1e-6)),
        "1e-6 reads as 0.000001"
    );
    assert!(
        invalid(d1_builder().f32("lfm3.decision.temperature.", 1.0)),
        "a temperature without a name"
    );

    // Malformed beats unsupported: a causal model with a bad temperature is
    // an error, not merely unsupported.
    assert!(wrong_type(
        with(temperature)
            .u32(temperature, 1)
            .without("lfm3.attention.causal")
            .bool("lfm3.attention.causal", true),
        temperature
    ));
}

#[test]
fn inconsistent_decision_tagging_is_an_error() {
    let invalid = |builder: GgufBuilder| {
        matches!(
            classify_bytes(&builder.build()),
            Err(GgufError::InvalidDecision(_))
        )
    };
    assert!(
        invalid(d1_builder().str("lfm2.decision.type", "lfm2-d1")),
        "a decision type under another architecture"
    );
    assert!(
        invalid(
            GgufBuilder::new()
                .str("general.architecture", "llama")
                .str("lfm3.decision.type", "lfm3-d1")
        ),
        "a decision model labelled for another architecture"
    );
    assert!(
        invalid(d1_builder().without("lfm3.decision.type")),
        "temperatures without a decision type"
    );
}

#[test]
fn classification_needs_the_architecture() {
    let missing = GgufBuilder::new().u32("x", 1).build();
    assert!(matches!(
        classify_bytes(&missing),
        Err(GgufError::MissingKey { key }) if key == "general.architecture"
    ));
    let array = GgufBuilder::new()
        .str_array("general.architecture", &["lfm3"])
        .build();
    assert!(matches!(
        classify_bytes(&array),
        Err(GgufError::WrongType { .. })
    ));
}

#[test]
fn errors_become_invalid_input_unless_they_are_io() {
    use crate::runtime_adapter::AdapterError;
    let malformed = AdapterError::from(GgufError::Malformed("x".to_string()));
    assert!(matches!(malformed, AdapterError::InvalidInput(_)));
    let io = AdapterError::from(GgufError::Io(std::io::Error::other("disk")));
    assert!(matches!(io, AdapterError::IOError(_)));
}

// ---------------------------------------------------------------------------
// llama.cpp's view against the header
// ---------------------------------------------------------------------------

/// What `LlamaModel::meta` reports for `d1_builder()`'s header: every
/// scalar, as llama.cpp writes it, sorted by key.
fn d1_native() -> Vec<(String, String)> {
    let mut native: Vec<(String, String)> = [
        ("general.architecture", "lfm3"),
        ("lfm3.attention.causal", "false"),
        ("lfm3.block_count", "24"),
        ("lfm3.decision.temperature.choice", "1.337398"),
        ("lfm3.decision.temperature.noul", "1.337398"),
        ("lfm3.decision.temperature.score", "1.337398"),
        ("lfm3.decision.type", "lfm3-d1"),
        ("tokenizer.chat_template.systemone", "{{ state }}"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    native.sort();
    native
}

#[test]
fn the_native_view_matches_the_header_it_was_loaded_from() {
    let header = read(&d1_builder().build()).unwrap();
    assert!(compare_native(&header, &d1_native()).is_ok());
    assert_eq!(
        native_f32(f32::from_bits(D1_TEMPERATURE)),
        "1.337398",
        "the temperature string llama.cpp reads"
    );

    let mismatch = |native: Vec<(String, String)>| match compare_native(&header, &native) {
        Err(GgufError::NativeMismatch { key }) => key,
        other => panic!("{other:?}"),
    };
    let mut changed = d1_native();
    changed[3].1 = "1.337399".to_string();
    assert_eq!(mismatch(changed), "lfm3.decision.temperature.choice");
    let mut missing = d1_native();
    missing.remove(6);
    assert_eq!(mismatch(missing), "lfm3.decision.type");
    let mut extra = d1_native();
    extra.push(("zzz.extra".to_string(), "1".to_string()));
    assert_eq!(mismatch(extra), "zzz.extra");
}

// ---------------------------------------------------------------------------
// The published d1-2B GGUF (artifact `d1-2b-q4km`), when staged
// ---------------------------------------------------------------------------

mod published {
    use std::path::PathBuf;

    use sha2::{Digest, Sha256};

    use super::*;

    /// The artifact id and file of the coordinator's pin: LiquidAI/d1-2B-GGUF
    /// at e968f6575f0d4fad928312d7af15cad90fd365c0, sha256
    /// adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b.
    const MODEL_ID: &str = "d1-2b-q4km";
    const MODEL_FILE: &str = "d1-2B-Q4_K_M.gguf";
    /// The `systemone` template d1-2B and d1-omni-2B share: 1,380 bytes,
    /// ending in `<think></think>`. (Contract v2 quotes `ac78e268…`, which
    /// matches neither GGUF.)
    pub(super) const SYSTEMONE_SHA256: &str =
        "ee5fc62461a9bbbcb8389c57f4cd2cd694ccb0dec19521576bd1419049490d8b";

    /// The staged GGUF, or `None` (with a note) when it is absent.
    ///
    /// # Panics
    ///
    /// When `XYBRID_REQUIRE_MODELS` is set and the model is not staged.
    pub(super) fn staged_d1() -> Option<PathBuf> {
        let models = std::env::var_os("XYBRID_TEST_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../integration-tests/fixtures/models")
            });
        let path = models.join(MODEL_ID).join(MODEL_FILE);
        if path.is_file() {
            return Some(path);
        }
        let reason = format!(
            "'{MODEL_ID}/{MODEL_FILE}' is not staged under {}",
            models.display()
        );
        if std::env::var_os("XYBRID_REQUIRE_MODELS").is_some() {
            panic!("XYBRID_REQUIRE_MODELS is set but {reason}");
        }
        eprintln!("Skipping test: {reason}");
        None
    }

    pub(super) fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    #[test]
    fn d1_2b_classifies_as_the_text_profile() {
        let Some(path) = staged_d1() else { return };
        let header = GgufHeader::read_file(&path).expect("read the d1-2B header");
        assert_eq!(classify(&header).unwrap(), D1);

        let str_of = |key: &str| header.get(key).and_then(GgufValue::as_str);
        assert_eq!(str_of("general.architecture"), Some("lfm3"));
        assert_eq!(str_of("lfm3.decision.type"), Some("lfm3-d1"));
        assert_eq!(
            header
                .get("lfm3.attention.causal")
                .and_then(GgufValue::as_bool),
            Some(false)
        );
        for name in ["noul", "choice", "score"] {
            let key = format!("lfm3.decision.temperature.{name}");
            let stored = header.get(&key).and_then(GgufValue::as_f32).unwrap();
            assert_eq!(stored.to_bits(), D1_TEMPERATURE, "{key}");
            assert_eq!(native_f32(stored), "1.337398");
            assert_eq!(effective_f32(stored).to_bits(), 0x3fab2fdc);
        }
        let template = header
            .get("tokenizer.chat_template.systemone")
            .and_then(GgufValue::as_bytes)
            .unwrap();
        assert_eq!(sha256_hex(template), SYSTEMONE_SHA256);
        let names: Vec<Option<&str>> = header
            .get("tokenizer.chat_templates")
            .and_then(GgufValue::as_array)
            .and_then(GgufArray::strings)
            .unwrap()
            .collect();
        assert_eq!(names, [Some("systemone")]);
        eprintln!(
            "d1-2B header: {} bytes, {} keys, {} tensors",
            header.header_bytes(),
            header.metadata().len(),
            header.tensors().len()
        );
    }

    /// Another, ordinary GGUF is not a decision model. Set
    /// `XYBRID_OTHER_GGUF` to one (the xybrid-llama smoke test uses the same
    /// variable to check its readout context refuses it).
    #[test]
    fn an_ordinary_gguf_is_not_a_decision_model() {
        let Some(path) = std::env::var_os("XYBRID_OTHER_GGUF") else {
            eprintln!(
                "Skipping test: set XYBRID_OTHER_GGUF to a GGUF that is not a decision model"
            );
            return;
        };
        let header = GgufHeader::read_file(&path).expect("read the header");
        assert_eq!(
            classify(&header).unwrap(),
            DecisionClassification::NotDecision
        );
        #[cfg(feature = "llm-llamacpp")]
        {
            xybrid_llama::backend_init();
            let path = std::path::Path::new(&path);
            let model = xybrid_llama::LlamaModel::load(path.to_str().expect("a UTF-8 path"), 0)
                .expect("load the GGUF");
            verify_native_metadata(&header, &model).expect("the same metadata");
        }
    }

    /// llama.cpp loads exactly the scalar metadata the header holds, and its
    /// named template is the header's.
    #[cfg(feature = "llm-llamacpp")]
    #[test]
    fn llama_cpp_loads_what_the_header_says() {
        let Some(path) = staged_d1() else { return };
        let header = GgufHeader::read_file(&path).expect("read the d1-2B header");
        xybrid_llama::backend_init();
        let model = xybrid_llama::LlamaModel::load(path.to_str().expect("a UTF-8 path"), 0)
            .expect("load d1-2B");

        verify_native_metadata(&header, &model).expect("the same metadata");
        let template = model
            .named_chat_template("systemone")
            .unwrap()
            .expect("the systemone template");
        assert_eq!(sha256_hex(template.as_bytes()), SYSTEMONE_SHA256);
        assert_eq!(
            model
                .meta_str("lfm3.decision.temperature.choice")
                .unwrap()
                .as_deref(),
            Some("1.337398")
        );

        // The native profile assertion agrees with the classification.
        let params = xybrid_llama::MemorylessParams::new(256, 4, false);
        let context = xybrid_llama::MemorylessContext::new(&model, params)
            .expect("a Supported model gets a readout context");
        assert!(!context.facts().has_memory);
        assert!(!context.facts().causal_attn);
    }
}
