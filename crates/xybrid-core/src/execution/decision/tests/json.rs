//! llama.cpp's `tojson`: layout, string escapes and Python's float `repr`.

use std::sync::Arc;

use crate::execution::decision::escape::Escaped;
use crate::execution::decision::json::{float_json, float_repr, to_json, JsonStyle};

/// `repr()` of these doubles in CPython (which llama.cpp's `float_repr`
/// reproduces). The port also agrees with CPython on 601,954 more: random
/// bit patterns, every power of two, decimal spellings and the ties near
/// 2^49 (checked when it was written; not part of the suite).
const REPR: &[(u64, &str)] = &[
    (0x0000000000000000, "0.0"),
    (0x8000000000000000, "-0.0"),
    (0x3ff0000000000000, "1.0"),
    (0xbff0000000000000, "-1.0"),
    (0x3fb999999999999a, "0.1"),
    (0x3fd3333333333333, "0.3"),
    (0x3fd3333333333334, "0.30000000000000004"),
    (0x3fd5555555555555, "0.3333333333333333"),
    (0x4341c37937e08000, "1e+16"),
    (0x430c6bf526340000, "1000000000000000.0"),
    (0x4341c37937e07fff, "9999999999999998.0"),
    (0x3f1a36e2eb1c432d, "0.0001"),
    (0x3ee4f8b588e368f1, "1e-05"),
    (0x3f1cd5f99c38b04b, "0.00011"),
    (0x3e7ad7f29abcaf48, "1e-07"),
    (0x0000000000000001, "5e-324"),
    (0x8000000000000001, "-5e-324"),
    (0x0010000000000000, "2.2250738585072014e-308"),
    (0x000fffffffffffff, "2.225073858507201e-308"),
    (0x7fefffffffffffff, "1.7976931348623157e+308"),
    (0x7fe0000000000000, "8.98846567431158e+307"),
    (0x7e37e43c8800759c, "1e+300"),
    (0x4480f0cf064dd592, "1e+22"),
    (0x44b52d02c7e14af6, "1e+23"),
    (0x4330000000000000, "4503599627370496.0"),
    (0x4340000000000000, "9007199254740992.0"),
    (0x43e0000000000000, "9.223372036854776e+18"),
    (0x43f0000000000000, "1.8446744073709552e+19"),
    (0x43e56a95319d63e1, "1.2345678901234567e+19"),
    (0x419d6f34547df3b6, "123456789.123"),
    (0x400921fb54442d18, "3.141592653589793"),
    (0x4011666666666666, "4.35"),
    (0x4059000000000000, "100.0"),
    (0x3de49da7e361ce4c, "1.5e-10"),
    // 562949953421312.25: the two 16-digit candidates are equally close;
    // ties go to even (`...312.2`), as in CPython. Rust's own shortest
    // formatting writes `...312.3`.
    (0x4300000000000002, "562949953421312.2"),
];

#[test]
fn floats_print_as_pythons_repr() {
    for &(bits, expected) in REPR {
        assert_eq!(float_repr(f64::from_bits(bits)), expected, "{bits:#018x}");
    }
    assert_eq!(float_repr(f64::NAN), "nan");
    assert_eq!(float_repr(f64::INFINITY), "inf");
    assert_eq!(float_repr(f64::NEG_INFINITY), "-inf");
    assert_eq!(float_json(f64::NAN), "NaN");
    assert_eq!(float_json(f64::INFINITY), "Infinity");
    assert_eq!(float_json(f64::NEG_INFINITY), "-Infinity");
    assert_eq!(float_json(0.5), "0.5");
}

fn arc(value: Escaped) -> Arc<Escaped> {
    Arc::new(value)
}

fn sample() -> Escaped {
    Escaped::Object(vec![
        ("z".into(), arc(Escaped::Int(1))),
        (
            "a".into(),
            arc(Escaped::Array(vec![
                arc(Escaped::Float(1.0)),
                arc(Escaped::Null),
                arc(Escaped::Bool(false)),
                arc(Escaped::Array(vec![])),
                arc(Escaped::Object(vec![])),
            ])),
        ),
        (
            "nested".into(),
            arc(Escaped::Object(vec![(
                "k".into(),
                arc(Escaped::Str("v".into())),
            )])),
        ),
    ])
}

#[test]
fn compact_json_has_pythons_separators_and_keeps_order() {
    assert_eq!(
        to_json(&sample(), JsonStyle { indent: None }),
        r#"{"z": 1, "a": [1.0, null, false, [], {}], "nested": {"k": "v"}}"#
    );
    assert_eq!(
        to_json(&Escaped::Int(-42), JsonStyle { indent: None }),
        "-42"
    );
    assert_eq!(
        to_json(&Escaped::Bool(true), JsonStyle { indent: None }),
        "true"
    );
    assert_eq!(
        to_json(&Escaped::Float(-0.0), JsonStyle { indent: None }),
        "-0.0"
    );
}

#[test]
fn indented_json_puts_each_item_on_its_line() {
    let expected = "{\n  \"z\": 1,\n  \"a\": [\n    1.0,\n    null,\n    false,\n    [],\n    {}\n  ],\n  \"nested\": {\n    \"k\": \"v\"\n  }\n}";
    assert_eq!(to_json(&sample(), JsonStyle { indent: Some(2) }), expected);
    // Indent 0: newlines without spaces, as Python's json.dumps(indent=0).
    let flat = to_json(
        &Escaped::Array(vec![arc(Escaped::Int(1)), arc(Escaped::Int(2))]),
        JsonStyle { indent: Some(0) },
    );
    assert_eq!(flat, "[\n1,\n2\n]");
    // A scalar has no layout.
    assert_eq!(
        to_json(&Escaped::Str("x".into()), JsonStyle { indent: Some(2) }),
        "\"x\""
    );
}

#[test]
fn strings_escape_quotes_backslashes_and_control_characters_only() {
    let text = "q\" b\\ \u{8}\u{c}\n\r\t \u{1}\u{1f} \u{7f} é 中 😀 \u{2028} \u{200b} </script>";
    assert_eq!(
        to_json(&Escaped::Str(text.into()), JsonStyle { indent: None }),
        "\"q\\\" b\\\\ \\b\\f\\n\\r\\t \\u0001\\u001f \u{7f} é 中 😀 \u{2028} \u{200b} </script>\""
    );
}
