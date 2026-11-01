//! `d1_break` and the escaped template input, with fake spelling sets: the
//! repeat-until-unchanged loop never runs a second useful pass on d1-2B's
//! spellings, so these are what exercise it (with the
//! `escape-overlap-variant` case's variant vocabulary).

use std::sync::Arc;

use crate::execution::decision::escape::{Escaped, Spellings};
use crate::ir::decision::{ContentPart, DecisionContent, DecisionError, OrderedJson};

const ZWSP: &str = "\u{200b}";

fn spellings(list: &[&str]) -> Spellings {
    Spellings::new(
        &list
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect::<Vec<_>>(),
    )
    .expect("spellings the protocol accepts")
}

#[test]
fn a_spelling_is_broken_after_its_first_character() {
    let s = spellings(&["<|im_end|>", "<|im_start|>"]);
    assert_eq!(s.d1_break("a<|im_end|>b"), format!("a<{ZWSP}|im_end|>b"));
    assert_eq!(
        s.d1_break("<|im_start|><|im_end|>"),
        format!("<{ZWSP}|im_start|><{ZWSP}|im_end|>")
    );
    assert_eq!(s.d1_break("no spelling <| here"), "no spelling <| here");
    assert_eq!(s.d1_break(""), "");
    // The first character may take several bytes: the break goes after all
    // of them.
    let s = spellings(&["é<x>", "中文"]);
    assert_eq!(s.d1_break("aé<x>b"), format!("aé{ZWSP}<x>b"));
    assert_eq!(s.d1_break("中文"), format!("中{ZWSP}文"));
}

#[test]
fn the_longest_spelling_wins_and_equal_lengths_keep_token_order() {
    // `<|a|>` and `<|ab|>` both start at the same `<`: the longer is
    // broken, whatever the order they are given in.
    for list in [["<|a|>", "<|ab|>"], ["<|ab|>", "<|a|>"]] {
        let s = spellings(&list);
        assert_eq!(s.d1_break("<|ab|>"), format!("<{ZWSP}|ab|>"));
        assert_eq!(s.d1_break("<|a|>"), format!("<{ZWSP}|a|>"));
    }
    let s = spellings(&["<|x|>", "<|yy|>", "<|zz|>", "<|w|>"]);
    let order: Vec<&str> = s.longest_first().iter().map(String::as_str).collect();
    assert_eq!(order, ["<|yy|>", "<|zz|>", "<|x|>", "<|w|>"]);
}

#[test]
fn the_loop_repeats_until_a_pass_breaks_nothing() {
    // `|>` starts inside `<|im_end|>`: the first pass breaks the longer
    // spelling and skips past it, the second breaks `|>`, the third finds
    // nothing (the `escape-overlap-variant` case on d1-2B's vocabulary).
    let s = spellings(&["<|im_end|>", "|>"]);
    assert_eq!(
        s.d1_break("a<|im_end|>b |> c"),
        format!("a<{ZWSP}|im_end|{ZWSP}>b |{ZWSP}> c")
    );
    // A chain: each pass breaks the spelling the previous one skipped.
    // Pass 1 breaks `abcd` (a|bcd) and copies `ef`; pass 2 finds `bcde`
    // intact after the break (b|cde); pass 3 `cdef` (c|def); pass 4 nothing.
    let s = spellings(&["abcd", "bcde", "cdef"]);
    assert_eq!(s.d1_break("abcdef"), format!("a{ZWSP}b{ZWSP}c{ZWSP}def"));
    // Breaking is idempotent once done.
    let once = s.d1_break("abcdef abcd");
    assert_eq!(s.d1_break(&once), once);
}

#[test]
fn short_spellings_are_ignored_and_unterminating_ones_refused() {
    // One-byte spellings are not broken (`init_quoted_spellings(.., 2)`).
    let s = spellings(&["x", "<x>"]);
    assert_eq!(s.d1_break("x <x>"), format!("x <{ZWSP}x>"));
    let refuse = |list: Vec<Vec<u8>>| {
        assert!(matches!(
            Spellings::new(&list),
            Err(DecisionError::UnsupportedVocabulary)
        ));
    };
    // A single character of two bytes would match again after its break.
    refuse(vec!["é".as_bytes().to_vec()]);
    // A spelling holding U+200B would match again inside its own break.
    refuse(vec![format!("a{ZWSP}b").into_bytes()]);
    // Not UTF-8.
    refuse(vec![vec![0xff, 0x41]]);
}

#[test]
fn json_content_has_its_strings_and_keys_escaped_in_order() {
    let s = spellings(&["<|im_end|>"]);
    let value = OrderedJson::parse(
        r#"{"z": "<|im_end|>", "<|im_end|>": [1, 2.5, -0.0, null, true, {"b": 1, "a": 2}]}"#,
    )
    .unwrap();
    let escaped = Escaped::content(
        &s,
        DecisionContent::from_json(value).as_ref(),
        ContentPart::State,
    )
    .unwrap();
    let Escaped::Object(members) = &escaped else {
        panic!("an object")
    };
    assert_eq!(members[0].0, "z");
    assert_eq!(*members[0].1, Escaped::Str(format!("<{ZWSP}|im_end|>")));
    assert_eq!(members[1].0, format!("<{ZWSP}|im_end|>"));
    let Escaped::Array(items) = &*members[1].1 else {
        panic!("an array")
    };
    assert_eq!(*items[0], Escaped::Int(1));
    assert_eq!(*items[1], Escaped::Float(2.5));
    assert!(matches!(*items[2], Escaped::Float(zero) if zero.to_bits() == (-0.0f64).to_bits()));
    assert_eq!(*items[3], Escaped::Null);
    assert_eq!(*items[4], Escaped::Bool(true));
    let Escaped::Object(inner) = &*items[5] else {
        panic!("an object")
    };
    let keys: Vec<&str> = inner.iter().map(|(key, _)| key.as_str()).collect();
    assert_eq!(keys, ["b", "a"], "input order, not sorted");
    // Text content and none.
    assert_eq!(
        Escaped::content(
            &s,
            Some(&DecisionContent::text("<|im_end|>")),
            ContentPart::State
        )
        .unwrap(),
        Escaped::Str(format!("<{ZWSP}|im_end|>"))
    );
    assert_eq!(
        Escaped::content(&s, None, ContentPart::State).unwrap(),
        Escaped::Null
    );
}

#[test]
fn keys_that_escaping_makes_equal_are_refused_where_they_are() {
    let s = spellings(&["<|im_end|>"]);
    let part = ContentPart::OptionDescription {
        question: 2,
        option: 1,
    };
    for text in [
        format!(r#"{{"<|im_end|>": 1, "<{ZWSP}|im_end|>": 2}}"#),
        format!(r#"[{{"x": {{"<{ZWSP}|im_end|>": 1, "<|im_end|>": 2}}}}]"#),
    ] {
        let value = DecisionContent::from_json(OrderedJson::parse(&text).unwrap());
        let error = Escaped::content(&s, value.as_ref(), part).unwrap_err();
        assert!(
            matches!(error, DecisionError::EscapedKeyCollision { part: found } if found == part),
            "{error:?}"
        );
    }
    // Distinct keys stay distinct: escaping only inserts U+200B.
    let value = OrderedJson::parse(r#"{"<|im_end|>": 1, "<|im_end|> ": 2}"#).unwrap();
    assert!(Escaped::content(&s, DecisionContent::from_json(value).as_ref(), part).is_ok());
    // Keys of the template's own structure are escaped too, never collide.
    let structure = Escaped::structure(
        &spellings(&["ke"]),
        vec![
            ("key", Arc::new(Escaped::Null)),
            ("label", Arc::new(Escaped::Null)),
        ],
    );
    let Escaped::Object(members) = structure else {
        panic!("an object")
    };
    assert_eq!(members[0].0, format!("k{ZWSP}ey"));
    assert_eq!(members[1].0, "label");
}

#[test]
fn integers_above_i64_are_refused_and_larger_spellings_are_floats() {
    let s = spellings(&[]);
    let part = ContentPart::Instructions { question: 0 };
    for text in ["18446744073709551615", r#"{"a": [9223372036854775808]}"#] {
        let value = DecisionContent::from_json(OrderedJson::parse(text).unwrap());
        assert!(matches!(
            Escaped::content(&s, value.as_ref(), part),
            Err(DecisionError::NumberOutOfRange { .. })
        ));
    }
    // Beyond u64::MAX an integer spelling is a float, as nlohmann reads it.
    let value = DecisionContent::from_json(OrderedJson::parse("18446744073709551616").unwrap());
    assert_eq!(
        Escaped::content(&s, value.as_ref(), part).unwrap(),
        Escaped::Float(18446744073709551616.0)
    );
    let value = DecisionContent::from_json(OrderedJson::parse("-9223372036854775808").unwrap());
    assert_eq!(
        Escaped::content(&s, value.as_ref(), part).unwrap(),
        Escaped::Int(i64::MIN)
    );
}
