//! Tests for ordered JSON: parsing, compact text, serde.

use super::write::{self, Counter};
use super::*;
use crate::ir::decision::JsonErrorKind;

const CANARY: &str = "canary-7f3a9c";

fn parse(text: &str) -> OrderedJson {
    OrderedJson::parse(text).unwrap_or_else(|error| panic!("{text:?}: {error}"))
}

fn syntax_error(text: &str) -> (JsonErrorKind, usize) {
    match OrderedJson::parse(text) {
        Err(DecisionError::InvalidJson { kind, offset }) => (kind, offset),
        other => panic!("{text:?}: expected a syntax error, got {other:?}"),
    }
}

fn number(text: &str) -> JsonNumber {
    parse(text)
        .as_number()
        .unwrap_or_else(|| panic!("{text:?} is not a number"))
}

fn compact(text: &str) -> String {
    parse(text).to_json_text()
}

fn keys(value: &OrderedJson) -> Vec<&str> {
    value
        .as_object()
        .expect("an object")
        .iter()
        .map(|(key, _)| key.as_str())
        .collect()
}

fn nested_arrays(depth: usize, innermost: &str) -> String {
    format!("{}{innermost}{}", "[".repeat(depth), "]".repeat(depth))
}

fn nested_objects(depth: usize, innermost: &str) -> String {
    format!(
        "{}{innermost}{}",
        r#"{"k":"#.repeat(depth),
        "}".repeat(depth)
    )
}

/// Deterministic xorshift64* so the float sample is the same on every run.
struct Bits(u64);

impl Bits {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn float_text(value: f64) -> String {
    let mut text = String::new();
    write::number(JsonNumber::F64(value), &mut text);
    text
}

// --- order and structure ---------------------------------------------------

#[test]
fn members_keep_document_order_at_every_level() {
    let text = r#"{"z":1,"a":{"y":2,"b":3},"m":[{"k":1,"c":2}],"":0}"#;
    let value = parse(text);
    assert_eq!(keys(&value), ["z", "a", "m", ""]);
    assert_eq!(keys(value.get("a").unwrap()), ["y", "b"]);
    assert_eq!(
        keys(&value.get("m").unwrap().as_array().unwrap()[0]),
        ["k", "c"]
    );
    assert_eq!(value.to_json_text(), text);
}

#[test]
fn compact_text_drops_whitespace_and_keeps_everything_else() {
    assert_eq!(
        compact(" { \"a\" : [ 1 , -2.5 , true , false , null , \"x\" ] ,\n\t\"b\" : { } , \"c\" : [ ] }\r\n"),
        r#"{"a":[1,-2.5,true,false,null,"x"],"b":{},"c":[]}"#
    );
    for scalar in ["null", "true", "false", "0", "-7", "\"\"", "\"text\""] {
        assert_eq!(compact(scalar), scalar);
    }
}

#[test]
fn member_order_is_part_of_equality() {
    assert_ne!(parse(r#"{"a":1,"b":2}"#), parse(r#"{"b":2,"a":1}"#));
    assert_eq!(
        parse(r#"{"a":1,"b":2}"#),
        parse(r#" { "a" : 1 , "b" : 2 } "#)
    );
}

#[test]
fn objects_refuse_repeated_keys() {
    assert_eq!(
        syntax_error(r#"{"a":1,"a":2}"#),
        (JsonErrorKind::DuplicateKey, 7)
    );
    // The first repeat in document order is reported.
    assert_eq!(
        syntax_error(r#"{"a":1,"b":2,"b":3,"a":4}"#),
        (JsonErrorKind::DuplicateKey, 13)
    );
    // Nested objects, and keys equal once unescaped.
    assert_eq!(
        syntax_error(r#"{"a":1,"b":{"c":1,"c":2}}"#),
        (JsonErrorKind::DuplicateKey, 18)
    );
    assert_eq!(
        syntax_error(r#"{"a":1,"\u0061":2}"#),
        (JsonErrorKind::DuplicateKey, 7)
    );
    // The same key in different objects, or in another case, is fine.
    parse(r#"[{"a":1},{"a":2}]"#);
    parse(r#"{"a":{"a":{"a":1}}}"#);
    parse(r#"{"a":1,"A":2}"#);
}

#[test]
fn objects_with_many_members_refuse_a_late_repeat() {
    let members: Vec<String> = (0..2_000)
        .map(|index| format!("\"k{index}\":{index}"))
        .collect();
    let unique = format!("{{{}}}", members.join(","));
    assert_eq!(parse(&unique).as_object().unwrap().len(), 2_000);
    let repeated = format!("{{{},\"k1999\":0}}", members.join(","));
    assert!(matches!(
        syntax_error(&repeated),
        (JsonErrorKind::DuplicateKey, _)
    ));
}

// --- depth -----------------------------------------------------------------

#[test]
fn depth_counts_the_value_itself() {
    assert_eq!(parse("1").depth(), 1);
    assert_eq!(parse("[]").depth(), 1);
    assert_eq!(parse("{}").depth(), 1);
    assert_eq!(parse("[[]]").depth(), 2);
    assert_eq!(parse(r#"{"a":[1]}"#).depth(), 3);
    assert_eq!(parse(r#"[1,[2,[3]],{"a":{}}]"#).depth(), 4);
}

#[test]
fn nesting_up_to_max_depth_is_accepted() {
    for text in [
        nested_arrays(MAX_JSON_DEPTH, ""),
        nested_arrays(MAX_JSON_DEPTH - 1, "1"),
        nested_objects(MAX_JSON_DEPTH - 1, "{}"),
        nested_objects(MAX_JSON_DEPTH - 1, "null"),
    ] {
        let value = parse(&text);
        assert_eq!(value.depth(), MAX_JSON_DEPTH);
        assert_eq!(value.to_json_text(), text);
        assert_eq!(OrderedJson::parse(&value.to_json_text()).unwrap(), value);
    }
}

#[test]
fn nesting_past_max_depth_is_refused() {
    for text in [
        nested_arrays(MAX_JSON_DEPTH + 1, ""),
        nested_arrays(MAX_JSON_DEPTH, "1"),
        nested_objects(MAX_JSON_DEPTH, "{}"),
        nested_objects(MAX_JSON_DEPTH, "\"x\""),
    ] {
        assert!(matches!(
            OrderedJson::parse(&text),
            Err(DecisionError::JsonTooDeep {
                max: MAX_JSON_DEPTH
            })
        ));
    }
    let error = OrderedJson::parse(&nested_arrays(200, "")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "JSON is nested more than 128 levels deep"
    );
}

#[test]
fn very_deep_text_is_refused_without_descending() {
    // Unclosed, and far deeper than any stack would allow if each level
    // recursed: the parser stops at level 129.
    let text = "[".repeat(1_000_000);
    let limits = JsonLimits::default().with_max_bytes(text.len());
    assert!(matches!(
        OrderedJson::parse_with_limits(&text, &limits),
        Err(DecisionError::JsonTooDeep { .. })
    ));
    let text = r#"{"a":"#.repeat(200_000);
    let limits = JsonLimits::default().with_max_bytes(text.len());
    assert!(matches!(
        OrderedJson::parse_with_limits(&text, &limits),
        Err(DecisionError::JsonTooDeep { .. })
    ));
}

// --- numbers ---------------------------------------------------------------

#[test]
fn integers_are_classified_as_nlohmann_classifies_them() {
    // Expectations cross-checked against the vendored nlohmann::json 3.12.0
    // (vendor/llama-cpp/vendor/nlohmann/json.hpp): integer literal ->
    // i64 if it fits, else u64 if it fits, else a float.
    let cases: [(&str, JsonNumber, &str); 12] = [
        ("0", JsonNumber::I64(0), "0"),
        ("-0", JsonNumber::I64(0), "0"),
        ("7", JsonNumber::I64(7), "7"),
        ("-7", JsonNumber::I64(-7), "-7"),
        (
            "9223372036854775807",
            JsonNumber::I64(i64::MAX),
            "9223372036854775807",
        ),
        (
            "-9223372036854775808",
            JsonNumber::I64(i64::MIN),
            "-9223372036854775808",
        ),
        (
            "9223372036854775808",
            JsonNumber::U64(1 << 63),
            "9223372036854775808",
        ),
        (
            "18446744073709551615",
            JsonNumber::U64(u64::MAX),
            "18446744073709551615",
        ),
        (
            "18446744073709551616",
            JsonNumber::F64(18446744073709551616.0),
            "1.8446744073709552e+19",
        ),
        (
            "-9223372036854775809",
            JsonNumber::F64(-9223372036854775808.0),
            "-9.223372036854776e+18",
        ),
        (
            "123456789012345678901234567890",
            JsonNumber::F64(1.2345678901234568e29),
            "1.2345678901234568e+29",
        ),
        (
            "-123456789012345678901234567890",
            JsonNumber::F64(-1.2345678901234568e29),
            "-1.2345678901234568e+29",
        ),
    ];
    for (literal, expected, text) in cases {
        let actual = number(literal);
        assert_eq!(actual, expected, "{literal}");
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"), "{literal}");
        assert_eq!(compact(literal), text, "{literal}");
    }
}

#[test]
fn minus_zero_is_the_integer_zero() {
    // nlohmann::json reads `-0` with strtoll: the integer 0, no sign.
    // `matches!` checks the variant itself; `==` would also accept U64(0).
    assert!(matches!(number("-0"), JsonNumber::I64(0)));
    let nested = parse(r#"[-0, {"k": -0}]"#);
    let items = nested.as_array().unwrap();
    assert!(matches!(items[0].as_number(), Some(JsonNumber::I64(0))));
    assert!(matches!(
        items[1].get("k").and_then(OrderedJson::as_number),
        Some(JsonNumber::I64(0))
    ));
    assert_eq!(compact("-0"), "0");
    let JsonNumber::F64(float) = number("-0.0") else {
        panic!("-0.0 is a float");
    };
    assert!(float == 0.0 && float.is_sign_negative());
}

#[test]
fn a_fraction_or_an_exponent_makes_a_float() {
    let cases: [(&str, f64, &str); 14] = [
        ("1.0", 1.0, "1.0"),
        ("1e0", 1.0, "1.0"),
        ("1E2", 100.0, "100.0"),
        ("1e+2", 100.0, "100.0"),
        ("1e-2", 0.01, "0.01"),
        ("2.5E-3", 0.0025, "0.0025"),
        ("-0.0", -0.0, "-0.0"),
        ("0.0", 0.0, "0.0"),
        ("-0e0", -0.0, "-0.0"),
        ("0.5", 0.5, "0.5"),
        ("-1.5e300", -1.5e300, "-1.5e+300"),
        ("1e-400", 0.0, "0.0"),
        ("-1e-400", -0.0, "-0.0"),
        ("1e-99999999999999999999", 0.0, "0.0"),
    ];
    for (literal, expected, text) in cases {
        let JsonNumber::F64(actual) = number(literal) else {
            panic!("{literal} is not a float");
        };
        assert_eq!(actual.to_bits(), expected.to_bits(), "{literal}");
        assert_eq!(compact(literal), text, "{literal}");
    }
}

#[test]
fn integer_and_float_identity_survive_a_round_trip() {
    let value = parse("[1, 1.0, -0, -0.0, 0, 0.0, 100, 1e2]");
    assert_eq!(value.to_json_text(), "[1,1.0,0,-0.0,0,0.0,100,100.0]");
    let items = value.as_array().unwrap();
    assert_ne!(items[0], items[1]);
    assert_eq!(items[2], items[4]);
    assert_ne!(items[3], items[5]);
    assert_eq!(OrderedJson::parse(&value.to_json_text()).unwrap(), value);
}

#[test]
fn floats_that_overflow_are_refused() {
    for literal in [
        "1e400",
        "-1e400",
        "1e99999999999999999999",
        "2e308",
        "1.8e308",
    ] {
        assert_eq!(syntax_error(literal), (JsonErrorKind::NumberOutOfRange, 0));
    }
    assert_eq!(
        syntax_error("[0, 1e999]"),
        (JsonErrorKind::NumberOutOfRange, 4)
    );
    // The largest finite double is fine.
    assert_eq!(number("1.7976931348623157e308"), JsonNumber::F64(f64::MAX));
}

#[test]
fn numbers_outside_the_grammar_are_refused() {
    let cases: [(&str, JsonErrorKind, usize); 16] = [
        ("01", JsonErrorKind::InvalidNumber, 0),
        ("-01", JsonErrorKind::InvalidNumber, 0),
        ("[1, 007]", JsonErrorKind::InvalidNumber, 4),
        ("1.", JsonErrorKind::InvalidNumber, 0),
        ("1.e5", JsonErrorKind::InvalidNumber, 0),
        ("1e", JsonErrorKind::InvalidNumber, 0),
        ("1e+", JsonErrorKind::InvalidNumber, 0),
        ("--1", JsonErrorKind::InvalidNumber, 0),
        ("-a", JsonErrorKind::InvalidNumber, 0),
        ("-Infinity", JsonErrorKind::InvalidNumber, 0),
        ("-", JsonErrorKind::UnexpectedEnd, 1),
        (".5", JsonErrorKind::ExpectedValue, 0),
        ("+1", JsonErrorKind::ExpectedValue, 0),
        ("NaN", JsonErrorKind::ExpectedValue, 0),
        ("Infinity", JsonErrorKind::ExpectedValue, 0),
        ("1.2.3", JsonErrorKind::TrailingCharacters, 3),
    ];
    for (text, kind, offset) in cases {
        assert_eq!(syntax_error(text), (kind, offset), "{text:?}");
    }
    assert_eq!(syntax_error("0x10"), (JsonErrorKind::TrailingCharacters, 1));
}

#[test]
fn floats_read_back_bit_for_bit() {
    for literal in [
        "0.1",
        "0.30000000000000004",
        "5e-324",
        "2.2250738585072014e-308",
        "2.225073858507201e-308",
        "1.7976931348623157e308",
        "9007199254740993.0",
        "4.9406564584124654e-324",
        "123456789.123456789",
        "0.000001",
        "1e15",
        "1e16",
        "1.5e-5",
        "1.3373980522155762",
        "0.4999999981306987",
    ] {
        let first = parse(literal);
        let again = parse(&first.to_json_text());
        assert_eq!(first, again, "{literal}");
        let (Some(JsonNumber::F64(a)), Some(JsonNumber::F64(b))) =
            (first.as_number(), again.as_number())
        else {
            panic!("{literal} is not a float");
        };
        assert_eq!(a.to_bits(), b.to_bits(), "{literal}");
        assert_eq!(a.to_bits(), literal.parse::<f64>().unwrap().to_bits());
    }
}

#[test]
fn float_text_matches_serde_json_and_reads_back() {
    let mut special = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        0.5,
        1.5,
        100.0,
        123.456,
        1e-5,
        1e-6,
        1.5e-5,
        9.999999999999999e-6,
        1.0000000000000002e-5,
        0.0001,
        1e15,
        1e16,
        1e17,
        9.999999999999998e15,
        1.2345678901234567e15,
        1.2345678901234567e16,
        123456789012345680.0,
        1e21,
        1e22,
        1e100,
        1e-100,
        1e300,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        5e-324,
        // The largest subnormal.
        f64::from_bits(0x000f_ffff_ffff_ffff),
        9007199254740992.0,
        9007199254740993.0,
        0.30000000000000004,
        1.3373980522155762,
        std::f64::consts::PI,
        std::f64::consts::E,
    ];
    for exponent in -330..=310 {
        special.push(10f64.powi(exponent));
        special.push(-1.25 * 10f64.powi(exponent));
    }
    let mut bits = Bits(0x9e37_79b9_7f4a_7c15);
    let random = (0..100_000).map(|_| f64::from_bits(bits.next()));
    for value in special.into_iter().chain(random) {
        if !value.is_finite() {
            continue;
        }
        let text = float_text(value);
        assert_eq!(text, serde_json::to_string(&value).unwrap(), "{value:e}");
        let back: f64 = text.parse().unwrap();
        assert_eq!(back.to_bits(), value.to_bits(), "{text}");
        let mut counter = Counter::default();
        write::number(JsonNumber::F64(value), &mut counter);
        assert_eq!(counter.bytes(), text.len(), "{text}");
    }
}

#[test]
fn a_tie_between_shortest_digits_goes_to_the_even_digit() {
    // 1394865425023536.25 is exactly halfway between the 17-digit
    // candidates ...536.2 and ...536.3; both read back as it. serde_json
    // writes the even one (std's `{:e}` would write ...536.3).
    assert_eq!(compact("1394865425023536.25"), "1394865425023536.2");
    assert_eq!(compact("1394865425023536.2"), "1394865425023536.2");
    assert_eq!(compact("1394865425023536.3"), "1394865425023536.2");
}

#[test]
fn non_finite_floats_are_written_as_null() {
    // Unreachable through `parse`, which refuses them; the writer still
    // never emits text that is not JSON.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(float_text(value), "null");
    }
}

#[test]
fn number_equality_is_identity() {
    assert_eq!(JsonNumber::I64(5), JsonNumber::U64(5));
    assert_eq!(JsonNumber::U64(5), JsonNumber::I64(5));
    assert_ne!(JsonNumber::I64(-1), JsonNumber::U64(u64::MAX));
    assert_ne!(JsonNumber::I64(1), JsonNumber::F64(1.0));
    assert_ne!(JsonNumber::F64(0.0), JsonNumber::F64(-0.0));
    assert_eq!(JsonNumber::F64(0.1), JsonNumber::F64(0.1));
    let nan = JsonNumber::F64(f64::NAN);
    assert_eq!(nan, nan);
}

// --- strings ---------------------------------------------------------------

#[test]
fn escapes_unescape() {
    let value = parse(r#""\"\\\/\b\f\n\r\t\u0000\u001f\u007f\u00e9\u00E9\u20AC\uD83D\uDE00""#);
    assert_eq!(
        value.as_str(),
        Some("\"\\/\u{8}\u{c}\n\r\t\u{0}\u{1f}\u{7f}\u{e9}\u{e9}\u{20ac}\u{1f600}")
    );
    // Only quotes, backslashes and control characters are escaped on the
    // way out; `/`, DEL and non-ASCII stay raw.
    assert_eq!(
        value.to_json_text(),
        "\"\\\"\\\\/\\b\\f\\n\\r\\t\\u0000\\u001f\u{7f}\u{e9}\u{e9}\u{20ac}\u{1f600}\""
    );
    let raw = "\"h\u{e9}llo \u{1f600} \u{2028}\"";
    assert_eq!(parse(raw).as_str(), Some("h\u{e9}llo \u{1f600} \u{2028}"));
    assert_eq!(compact(raw), raw);
}

#[test]
fn string_text_matches_serde_json() {
    let mut samples: Vec<String> = (0u8..0x80)
        .map(|byte| char::from(byte).to_string())
        .collect();
    samples.extend(
        [
            "",
            "plain",
            "quote \" and backslash \\ and slash /",
            "tab\tnewline\nreturn\rbell\u{7}nul\u{0}",
            "é ü 漢字 😀 \u{2028}\u{2029} \u{feff} \u{fffd}",
            CANARY,
        ]
        .map(String::from),
    );
    for sample in samples {
        let text = serde_json::to_string(&sample).unwrap();
        let value = parse(&text);
        assert_eq!(value.as_str(), Some(sample.as_str()));
        assert_eq!(value.to_json_text(), text, "{sample:?}");
        assert_eq!(value.json_text_len(), text.len());
    }
}

#[test]
fn bad_escapes_and_lone_surrogates_are_refused() {
    let cases: [(&str, JsonErrorKind, usize); 12] = [
        (r#""\x""#, JsonErrorKind::InvalidEscape, 1),
        (r#""a\'""#, JsonErrorKind::InvalidEscape, 2),
        (r#""\u12""#, JsonErrorKind::InvalidEscape, 1),
        (r#""\uZZZZ""#, JsonErrorKind::InvalidEscape, 1),
        (r#""\u00g0""#, JsonErrorKind::InvalidEscape, 1),
        ("\"\\u12", JsonErrorKind::UnexpectedEnd, 5),
        ("\"\\", JsonErrorKind::UnexpectedEnd, 2),
        (r#""\uD83D""#, JsonErrorKind::UnpairedSurrogate, 1),
        (r#""\uD83Dx""#, JsonErrorKind::UnpairedSurrogate, 1),
        (r#""\uD83D\u0041""#, JsonErrorKind::UnpairedSurrogate, 1),
        (r#""\uDE00""#, JsonErrorKind::UnpairedSurrogate, 1),
        (r#""ok \uD83D\uD83D""#, JsonErrorKind::UnpairedSurrogate, 4),
    ];
    for (text, kind, offset) in cases {
        assert_eq!(syntax_error(text), (kind, offset), "{text:?}");
    }
    assert_eq!(
        syntax_error(r#""\uD83D\uZZZZ""#),
        (JsonErrorKind::InvalidEscape, 7)
    );
}

#[test]
fn raw_control_characters_are_refused() {
    assert_eq!(
        syntax_error("\"a\tb\""),
        (JsonErrorKind::ControlCharacter, 2)
    );
    assert_eq!(
        syntax_error("\"a\nb\""),
        (JsonErrorKind::ControlCharacter, 2)
    );
    assert_eq!(
        syntax_error("\"\u{0}\""),
        (JsonErrorKind::ControlCharacter, 1)
    );
    assert_eq!(
        syntax_error("{\"\u{1f}\":1}"),
        (JsonErrorKind::ControlCharacter, 2)
    );
    // DEL and everything above U+001F may be raw.
    assert_eq!(
        parse("\"\u{7f}\u{80}\u{9f}\"").as_str(),
        Some("\u{7f}\u{80}\u{9f}")
    );
}

// --- whitespace, BOM, malformed text ---------------------------------------

#[test]
fn only_json_whitespace_separates_tokens() {
    assert_eq!(compact(" \t\n\r[ 1 ,\r\n2 ]\t"), "[1,2]");
    for (text, kind, offset) in [
        ("\u{a0}1", JsonErrorKind::ExpectedValue, 0),
        ("\u{b}1", JsonErrorKind::ExpectedValue, 0),
        ("\u{c}1", JsonErrorKind::ExpectedValue, 0),
        ("[1,\u{2028}2]", JsonErrorKind::ExpectedValue, 3),
        ("1\u{a0}", JsonErrorKind::TrailingCharacters, 1),
    ] {
        assert_eq!(syntax_error(text), (kind, offset), "{text:?}");
    }
}

#[test]
fn a_leading_byte_order_mark_is_skipped_like_nlohmann() {
    assert_eq!(compact("\u{feff}{\"a\":1}"), r#"{"a":1}"#);
    assert_eq!(compact("\u{feff} 1"), "1");
    assert_eq!(syntax_error("\u{feff}"), (JsonErrorKind::UnexpectedEnd, 3));
    assert_eq!(
        syntax_error(" \u{feff}1"),
        (JsonErrorKind::ExpectedValue, 1)
    );
    assert_eq!(
        syntax_error("\u{feff}\u{feff}1"),
        (JsonErrorKind::ExpectedValue, 3)
    );
    assert_eq!(
        syntax_error("1\u{feff}"),
        (JsonErrorKind::TrailingCharacters, 1)
    );
}

#[test]
fn malformed_text_is_refused_with_its_offset() {
    let cases: [(&str, JsonErrorKind, usize); 24] = [
        ("", JsonErrorKind::UnexpectedEnd, 0),
        ("   ", JsonErrorKind::UnexpectedEnd, 3),
        ("[", JsonErrorKind::UnexpectedEnd, 1),
        ("[1", JsonErrorKind::UnexpectedEnd, 2),
        ("[1,", JsonErrorKind::UnexpectedEnd, 3),
        ("[1,]", JsonErrorKind::ExpectedValue, 3),
        ("[,1]", JsonErrorKind::ExpectedValue, 1),
        ("[1 2]", JsonErrorKind::ExpectedCommaOrEnd, 3),
        ("[1}", JsonErrorKind::ExpectedCommaOrEnd, 2),
        ("{", JsonErrorKind::UnexpectedEnd, 1),
        (r#"{"a"}"#, JsonErrorKind::ExpectedColon, 4),
        (r#"{"a":}"#, JsonErrorKind::ExpectedValue, 5),
        (r#"{"a":1,}"#, JsonErrorKind::ExpectedKey, 7),
        (r#"{"a":1 "b":2}"#, JsonErrorKind::ExpectedCommaOrEnd, 7),
        ("{1:2}", JsonErrorKind::ExpectedKey, 1),
        ("{'a':1}", JsonErrorKind::ExpectedKey, 1),
        ("[1]]", JsonErrorKind::TrailingCharacters, 3),
        ("1 2", JsonErrorKind::TrailingCharacters, 2),
        ("nul", JsonErrorKind::InvalidLiteral, 0),
        ("[tru]", JsonErrorKind::InvalidLiteral, 1),
        ("nulll", JsonErrorKind::TrailingCharacters, 4),
        ("True", JsonErrorKind::ExpectedValue, 0),
        ("\"abc", JsonErrorKind::UnexpectedEnd, 4),
        ("// comment\n1", JsonErrorKind::ExpectedValue, 0),
    ];
    for (text, kind, offset) in cases {
        assert_eq!(syntax_error(text), (kind, offset), "{text:?}");
    }
    assert_eq!(
        syntax_error("[1,/*c*/2]"),
        (JsonErrorKind::ExpectedValue, 3)
    );
}

// --- limits ----------------------------------------------------------------

#[test]
fn default_limits() {
    let limits = JsonLimits::default();
    assert_eq!(limits.max_bytes, 1 << 20);
    assert_eq!(limits.max_nodes, 1 << 16);
    let custom = limits.with_max_bytes(10).with_max_nodes(3);
    assert_eq!((custom.max_bytes, custom.max_nodes), (10, 3));
}

#[test]
fn the_byte_ceiling_counts_the_whole_text() {
    let text = r#"  {"a": [1, 2]}  "#;
    let exact = JsonLimits::default().with_max_bytes(text.len());
    assert!(OrderedJson::parse_with_limits(text, &exact).is_ok());
    let short = JsonLimits::default().with_max_bytes(text.len() - 1);
    assert!(matches!(
        OrderedJson::parse_with_limits(text, &short),
        Err(DecisionError::JsonTooLong { len, max }) if len == text.len() && max == text.len() - 1
    ));
    let bom = "\u{feff}1";
    let limits = JsonLimits::default().with_max_bytes(3);
    assert!(matches!(
        OrderedJson::parse_with_limits(bom, &limits),
        Err(DecisionError::JsonTooLong { len: 4, max: 3 })
    ));
    let big = format!("\"{}\"", "x".repeat(DEFAULT_MAX_JSON_BYTES));
    assert!(matches!(
        OrderedJson::parse(&big),
        Err(DecisionError::JsonTooLong { .. })
    ));
}

#[test]
fn the_value_ceiling_counts_scalars_and_containers() {
    let text = r#"[1,[2,{"a":null,"b":"x"}],{}]"#;
    let value = parse(text);
    assert_eq!(value.node_count(), 8);
    let exact = JsonLimits::default().with_max_nodes(8);
    assert!(OrderedJson::parse_with_limits(text, &exact).is_ok());
    let short = JsonLimits::default().with_max_nodes(7);
    assert!(matches!(
        OrderedJson::parse_with_limits(text, &short),
        Err(DecisionError::JsonTooManyNodes { max: 7 })
    ));
    let many = format!("[{}]", vec!["0"; DEFAULT_MAX_JSON_NODES].join(","));
    assert!(matches!(
        OrderedJson::parse(&many),
        Err(DecisionError::JsonTooManyNodes {
            max: DEFAULT_MAX_JSON_NODES
        })
    ));
}

// --- accessors -------------------------------------------------------------

#[test]
fn views_and_accessors_follow_the_kind() {
    let value = parse(r#"{"n":null,"b":true,"i":-3,"s":"x","a":[1],"o":{"k":2}}"#);
    let kinds: Vec<&str> = value
        .as_object()
        .unwrap()
        .iter()
        .map(|(_, item)| item.kind().as_str())
        .collect();
    assert_eq!(
        kinds,
        ["null", "boolean", "number", "string", "array", "object"]
    );
    assert_eq!(JsonKind::Object.to_string(), "object");

    let get = |key| value.get(key).unwrap();
    assert!(get("n").is_null() && matches!(get("n").view(), JsonView::Null));
    assert_eq!(get("b").as_bool(), Some(true));
    assert_eq!(get("i").as_number(), Some(JsonNumber::I64(-3)));
    assert_eq!(get("s").as_str(), Some("x"));
    assert_eq!(get("a").as_array().map(<[_]>::len), Some(1));
    assert_eq!(
        get("o").get("k").and_then(OrderedJson::as_number),
        Some(JsonNumber::I64(2))
    );
    assert!(matches!(get("o").view(), JsonView::Object(members) if members.len() == 1));
    assert!(value.get("missing").is_none());
    assert!(get("a").get("k").is_none());
    for (_, item) in value.as_object().unwrap() {
        assert_eq!(item.view().kind(), item.kind());
    }
    assert!(get("s").as_bool().is_none() && get("b").as_str().is_none());
    assert!(get("i").as_array().is_none() && get("a").as_object().is_none());
}

#[test]
fn into_methods_move_parts_out_or_give_the_value_back() {
    let value = parse(r#"{"s":"text","a":[1,2],"o":{}}"#);
    let members = value.clone().into_object().unwrap();
    let [(_, s), (_, a), (_, o)]: [(String, OrderedJson); 3] = members.try_into().unwrap();
    assert_eq!(s.clone().into_string().unwrap(), "text");
    assert_eq!(a.clone().into_array().unwrap().len(), 2);
    assert!(o.clone().into_object().unwrap().is_empty());
    assert_eq!(s.clone().into_array().unwrap_err(), s);
    assert_eq!(a.clone().into_string().unwrap_err(), a);
    assert_eq!(a.clone().into_object().unwrap_err(), a);
    assert_eq!(value.to_json_text().parse::<OrderedJson>().unwrap(), value);
}

#[test]
fn text_length_matches_the_text() {
    for text in [
        "null",
        r#"{"a":[1,2.5,-0.0,1e300,18446744073709551615],"b":"\u0001\"\\é😀","c":{}}"#,
        "[true,false,null,[],{},\"\"]",
        "-1.25e-7",
    ] {
        let value = parse(text);
        assert_eq!(value.json_text_len(), value.to_json_text().len(), "{text}");
    }
}

// --- serde -----------------------------------------------------------------

#[test]
fn serde_form_is_the_compact_text_in_json_and_bincode() {
    let value = parse(
        r#"{"b":-0,"a":[1.0,1,-0.0,18446744073709551615,1e300,"\u00e9\u0000"],"c":{"z":null,"y":{}}}"#,
    );
    let text = value.to_json_text();

    let json = serde_json::to_string(&value).unwrap();
    assert_eq!(json, serde_json::to_string(&text).unwrap());
    let back: OrderedJson = serde_json::from_str(&json).unwrap();
    assert_eq!(back, value);
    assert_eq!(keys(&back), ["b", "a", "c"]);

    let bytes = bincode::serialize(&value).unwrap();
    assert_eq!(bytes, bincode::serialize(&text).unwrap());
    let back: OrderedJson = bincode::deserialize(&bytes).unwrap();
    assert_eq!(back, value);
    assert_eq!(back.to_json_text(), text);
}

#[test]
fn deserializing_refuses_invalid_text_without_quoting_it() {
    let inner = format!(r#"{{"{CANARY}": 1, "{CANARY}": 2}}"#);
    let json = serde_json::to_string(&inner).unwrap();
    let error = serde_json::from_str::<OrderedJson>(&json)
        .unwrap_err()
        .to_string();
    assert!(error.contains("an object repeats a key"), "{error}");
    assert!(!error.contains(CANARY), "{error}");

    let bytes = bincode::serialize(&format!("[\"{CANARY}\" 1]")).unwrap();
    let error = bincode::deserialize::<OrderedJson>(&bytes)
        .unwrap_err()
        .to_string();
    assert!(error.contains("invalid JSON at byte"), "{error}");
    assert!(!error.contains(CANARY), "{error}");

    // The serde form is a string, never the JSON value itself.
    assert!(serde_json::from_str::<OrderedJson>(r#"{"a":1}"#).is_err());
    assert!(serde_json::from_str::<OrderedJson>("1").is_err());
}

#[test]
fn decoding_applies_the_default_value_ceiling() {
    let text = format!("[{}]", vec!["0"; DEFAULT_MAX_JSON_NODES].join(","));
    let roomy = JsonLimits::default().with_max_nodes(DEFAULT_MAX_JSON_NODES + 1);
    let value = OrderedJson::parse_with_limits(&text, &roomy).unwrap();
    let bytes = bincode::serialize(&value).unwrap();
    let error = bincode::deserialize::<OrderedJson>(&bytes)
        .unwrap_err()
        .to_string();
    assert!(error.contains("more than 65536 values"), "{error}");
}

#[test]
fn compact_text_longer_than_the_byte_ceiling_still_decodes() {
    // `1E15` is 4 bytes in, 18 out: the defaults accept the text, and its
    // compact form is longer than the default byte ceiling.
    let text = format!("[{}]", vec!["1E15"; DEFAULT_MAX_JSON_NODES - 1].join(","));
    assert!(text.len() < DEFAULT_MAX_JSON_BYTES);
    let value = parse(&text);
    let compact = value.to_json_text();
    assert!(compact.len() > DEFAULT_MAX_JSON_BYTES);
    assert!(matches!(
        OrderedJson::parse(&compact),
        Err(DecisionError::JsonTooLong { .. })
    ));

    let json = serde_json::to_string(&value).unwrap();
    assert_eq!(serde_json::from_str::<OrderedJson>(&json).unwrap(), value);
    let bytes = bincode::serialize(&value).unwrap();
    assert_eq!(bincode::deserialize::<OrderedJson>(&bytes).unwrap(), value);
}

#[test]
fn the_deepest_values_fit_a_small_thread_stack() {
    // Debug builds included: an app may parse on a 512 KiB thread (an iOS
    // secondary thread, a GCD queue). Parsing, refusing, copying, comparing,
    // writing, encoding and dropping the deepest values must fit half that.
    let worker = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            for text in [
                nested_arrays(MAX_JSON_DEPTH, ""),
                nested_arrays(MAX_JSON_DEPTH - 1, "1.5"),
                nested_objects(MAX_JSON_DEPTH - 1, "{}"),
                nested_objects(MAX_JSON_DEPTH - 1, r#""text""#),
            ] {
                let value = OrderedJson::parse(&text).unwrap();
                let copy = value.clone();
                assert!(copy == value);
                assert_eq!(value.depth(), MAX_JSON_DEPTH);
                assert_eq!(value.node_count(), MAX_JSON_DEPTH);
                assert_eq!(value.to_json_text(), text);
                assert_eq!(value.json_text_len(), text.len());
                let bytes = bincode::serialize(&value).unwrap();
                let decoded: OrderedJson = bincode::deserialize(&bytes).unwrap();
                assert!(decoded == value);
                let json = serde_json::to_string(&value).unwrap();
                assert!(serde_json::from_str::<OrderedJson>(&json).unwrap() == value);
                drop((value, copy, decoded));
            }
            for text in [
                nested_arrays(MAX_JSON_DEPTH + 1, ""),
                "[".repeat(100_000),
                r#"{"a":"#.repeat(100_000),
            ] {
                assert!(matches!(
                    OrderedJson::parse(&text),
                    Err(DecisionError::JsonTooDeep { .. })
                ));
            }
        })
        .unwrap();
    worker.join().unwrap();
}

// --- redaction -------------------------------------------------------------

#[test]
fn debug_prints_kind_and_length_only() {
    let value = parse(&format!(
        r#"{{"{CANARY}": ["{CANARY}", 918273645, 0.918273645, true], "k": {{"{CANARY}": null}}}}"#
    ));
    let mut rendered = vec![format!("{value:?}"), format!("{:?}", value.view())];
    for (_, item) in value.as_object().unwrap() {
        rendered.push(format!("{item:?}"));
        for child in item.as_array().into_iter().flatten() {
            rendered.push(format!("{child:?} {:?}", child.view()));
            if let Some(number) = child.as_number() {
                rendered.push(format!("{number:?}"));
            }
        }
    }
    for text in &rendered {
        assert!(!text.contains(CANARY), "{text}");
        assert!(!text.contains("918273645"), "{text}");
        assert!(!text.contains("true"), "{text}");
    }
    assert_eq!(rendered[0], "OrderedJson { kind: \"object\", members: 2 }");
    assert_eq!(rendered[1], "JsonView { kind: \"object\", members: 2 }");
    assert!(rendered.contains(&"OrderedJson { kind: \"array\", items: 4 }".to_string()));
    assert!(rendered
        .iter()
        .any(|text| text.starts_with("OrderedJson { kind: \"string\", bytes: 13 }")));
    assert!(rendered.contains(&"F64".to_string()));
}

#[test]
fn parse_errors_never_quote_the_text() {
    for text in [
        format!(r#"{{"{CANARY}":1,"{CANARY}":2}}"#),
        format!(r#"["{CANARY}" "{CANARY}"]"#),
        format!(r#"{{"{CANARY}"}}"#),
        format!(r#""{CANARY}\q""#),
        format!("\"{CANARY}\t\""),
        CANARY.to_string(),
        format!(r#"["{CANARY}", 1e999]"#),
    ] {
        let error = OrderedJson::parse(&text).unwrap_err();
        for rendered in [error.to_string(), format!("{error:?}")] {
            assert!(!rendered.contains(CANARY), "{rendered}");
        }
    }
}
