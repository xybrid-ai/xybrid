//! `d1_break` and the escaped template input, with fake spelling sets: the
//! repeat-until-unchanged loop never runs a second useful pass on d1-2B's
//! spellings, so these are what exercise it (with the
//! `escape-overlap-variant` case's variant vocabulary).

use std::sync::Arc;

use crate::execution::decision::escape::{
    Escaped, Spellings, ESCAPE_STEPS_PER_BYTE, ESCAPE_STEPS_PER_STRING, MAX_SPELLINGS,
    MAX_SPELLINGS_BYTES, MAX_SPELLING_BYTES,
};
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
    assert_eq!(
        s.d1_break("a<|im_end|>b").unwrap(),
        format!("a<{ZWSP}|im_end|>b")
    );
    assert_eq!(
        s.d1_break("<|im_start|><|im_end|>").unwrap(),
        format!("<{ZWSP}|im_start|><{ZWSP}|im_end|>")
    );
    assert_eq!(
        s.d1_break("no spelling <| here").unwrap(),
        "no spelling <| here"
    );
    assert_eq!(s.d1_break("").unwrap(), "");
    // The first character may take several bytes: the break goes after all
    // of them.
    let s = spellings(&["é<x>", "中文"]);
    assert_eq!(s.d1_break("aé<x>b").unwrap(), format!("aé{ZWSP}<x>b"));
    assert_eq!(s.d1_break("中文").unwrap(), format!("中{ZWSP}文"));
}

#[test]
fn the_longest_spelling_wins() {
    // `<|a|>` and `<|ab|>` both start at the same `<`: the longer is
    // broken, whatever the order they are given in.
    for list in [["<|a|>", "<|ab|>"], ["<|ab|>", "<|a|>"]] {
        let s = spellings(&list);
        assert_eq!(s.d1_break("<|ab|>").unwrap(), format!("<{ZWSP}|ab|>"));
        assert_eq!(s.d1_break("<|a|>").unwrap(), format!("<{ZWSP}|a|>"));
    }
    // Spellings of one length cannot both match at a position unless they
    // are the same text, so their order is immaterial.
    let s = spellings(&["<|x|>", "<|yy|>", "<|zz|>", "<|w|>", "<|yy|>"]);
    assert_eq!(
        s.d1_break("<|w|><|yy|><|x|>").unwrap(),
        format!("<{ZWSP}|w|><{ZWSP}|yy|><{ZWSP}|x|>")
    );
}

#[test]
fn the_loop_repeats_until_a_pass_breaks_nothing() {
    // `|>` starts inside `<|im_end|>`: the first pass breaks the longer
    // spelling and skips past it, the second breaks `|>`, the third finds
    // nothing (the `escape-overlap-variant` case on d1-2B's vocabulary).
    let s = spellings(&["<|im_end|>", "|>"]);
    assert_eq!(
        s.d1_break("a<|im_end|>b |> c").unwrap(),
        format!("a<{ZWSP}|im_end|{ZWSP}>b |{ZWSP}> c")
    );
    // A chain: each pass breaks the spelling the previous one skipped.
    // Pass 1 breaks `abcd` (a|bcd) and copies `ef`; pass 2 finds `bcde`
    // intact after the break (b|cde); pass 3 `cdef` (c|def); pass 4 nothing.
    let s = spellings(&["abcd", "bcde", "cdef"]);
    assert_eq!(
        s.d1_break("abcdef").unwrap(),
        format!("a{ZWSP}b{ZWSP}c{ZWSP}def")
    );
    // Breaking is idempotent once done.
    let once = s.d1_break("abcdef abcd").unwrap();
    assert_eq!(s.d1_break(&once).unwrap(), once);
}

#[test]
fn short_spellings_are_ignored_and_unterminating_ones_refused() {
    // One-byte spellings are not broken (`init_quoted_spellings(.., 2)`).
    let s = spellings(&["x", "<x>"]);
    assert_eq!(s.d1_break("x <x>").unwrap(), format!("x <{ZWSP}x>"));
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
    )
    .unwrap();
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

/// Upstream's `d1_break`, ported as written (`server-decision.cpp:630-661`):
/// at each position whose byte starts a spelling, every spelling, longest
/// first, until one matches. The trie must give the same text.
fn upstream_d1_break(spellings: &[&str], text: &str) -> String {
    let mut sorted: Vec<&str> = spellings.iter().copied().filter(|s| s.len() >= 2).collect();
    sorted.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let first: Vec<u8> = sorted.iter().map(|s| s.as_bytes()[0]).collect();
    let mut cur = text.as_bytes().to_vec();
    loop {
        let mut changed = false;
        let mut out = Vec::new();
        let mut i = 0;
        while i < cur.len() {
            let hit = first
                .contains(&cur[i])
                .then(|| sorted.iter().find(|s| cur[i..].starts_with(s.as_bytes())))
                .flatten();
            match hit {
                None => {
                    out.push(cur[i]);
                    i += 1;
                }
                Some(spelling) => {
                    let n = spelling.chars().next().unwrap().len_utf8();
                    out.extend_from_slice(&spelling.as_bytes()[..n]);
                    out.extend_from_slice(ZWSP.as_bytes());
                    out.extend_from_slice(&spelling.as_bytes()[n..]);
                    i += spelling.len();
                    changed = true;
                }
            }
        }
        cur = out;
        if !changed {
            return String::from_utf8(cur).unwrap();
        }
    }
}

#[test]
fn the_trie_breaks_what_upstreams_scan_breaks() {
    // Random spelling sets over a small alphabet, so that spellings share
    // prefixes, nest and overlap, against random text: the same output.
    let alphabet: Vec<char> = "<|>ab".chars().collect();
    let mut seed: u64 = 0x005e_edd1;
    let mut next = |bound: usize| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) % bound as u64) as usize
    };
    for _ in 0..400 {
        let spelling_count = 1 + next(8);
        let mut list: Vec<String> = Vec::new();
        for _ in 0..spelling_count {
            let len = 2 + next(5);
            list.push((0..len).map(|_| alphabet[next(alphabet.len())]).collect());
        }
        let refs: Vec<&str> = list.iter().map(String::as_str).collect();
        let s = spellings(&refs);
        for _ in 0..10 {
            let len = next(40);
            let text: String = (0..len).map(|_| alphabet[next(alphabet.len())]).collect();
            assert_eq!(
                s.d1_break(&text).unwrap(),
                upstream_d1_break(&refs, &text),
                "{refs:?} on {text:?}"
            );
        }
    }
}

#[test]
fn spellings_past_the_caps_are_refused() {
    let refused = |list: Vec<Vec<u8>>| {
        matches!(
            Spellings::new(&list),
            Err(DecisionError::UnsupportedVocabulary)
        )
    };
    // Codex's probe: 16,384 spellings of 72 bytes sharing a 63-byte prefix.
    let probe: Vec<Vec<u8>> = (0..16_384)
        .map(|i| format!("{}{i:08x}X", "a".repeat(63)).into_bytes())
        .collect();
    assert!(refused(probe));
    // One past each cap.
    let many: Vec<Vec<u8>> = (0..=MAX_SPELLINGS)
        .map(|i| format!("<{i}>").into_bytes())
        .collect();
    assert!(refused(many));
    assert!(refused(vec![format!(
        "<{}>",
        "a".repeat(MAX_SPELLING_BYTES - 1)
    )
    .into_bytes()]));
    let heavy: Vec<Vec<u8>> = (0..MAX_SPELLINGS_BYTES / 200 + 1)
        .map(|i| format!("<{i:04}{}>", "b".repeat(194)).into_bytes())
        .collect();
    assert!(heavy.iter().all(|s| s.len() == 200));
    assert!(refused(heavy));
    // At the caps: accepted.
    let at_cap: Vec<Vec<u8>> = (0..MAX_SPELLINGS)
        .map(|i| format!("<{i:05}>").into_bytes())
        .collect();
    assert!(Spellings::new(&at_cap).is_ok());
    assert!(
        Spellings::new(&[format!("<{}>", "a".repeat(MAX_SPELLING_BYTES - 2)).into_bytes()]).is_ok()
    );
}

#[test]
fn a_common_prefix_that_never_completes_costs_bounded_work() {
    // 900 spellings sharing a long prefix of `a`, which the text matches at
    // every position without ever completing a spelling: each position walks
    // the shared prefix once, never every spelling.
    let text = "a".repeat(64 << 10);
    let short: Vec<String> = (0..900)
        .map(|i| format!("{}{i:03}X", "a".repeat(30)))
        .collect();
    let refs: Vec<&str> = short.iter().map(String::as_str).collect();
    let started = std::time::Instant::now();
    assert_eq!(spellings(&refs).d1_break(&text).unwrap(), text);
    assert!(started.elapsed() < std::time::Duration::from_secs(5));

    // With a prefix longer than the work ceiling allows per byte, the
    // escaping stops at the ceiling and refuses the vocabulary.
    let long: Vec<String> = (0..200)
        .map(|i| format!("{}{i:03}X", "a".repeat(200)))
        .collect();
    let refs: Vec<&str> = long.iter().map(String::as_str).collect();
    let started = std::time::Instant::now();
    assert!(matches!(
        spellings(&refs).d1_break(&text),
        Err(DecisionError::EscapeWorkExceeded)
    ));
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
    // The ceiling is per byte of text, plus a constant per string that lets
    // a short string walk the longest spelling.
    const { assert!(ESCAPE_STEPS_PER_BYTE >= 32 && ESCAPE_STEPS_PER_STRING >= MAX_SPELLING_BYTES) };
}
