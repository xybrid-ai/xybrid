//! Label allocation on fake vocabularies: what d1-2B's vocabulary cannot
//! show (pool exhaustion, colliding aliases, a token in two groups).

use super::fake::FakeVocab;
use crate::execution::decision::labels::{self, Label, LabelTokens};
use crate::ir::decision::DecisionError;

fn choice(vocab: &FakeVocab, keys: &[&str]) -> Result<Vec<Label>, DecisionError> {
    labels::choice(0, keys, &mut LabelTokens::new(vocab))
}

fn codes(labels: &[Label]) -> Vec<&str> {
    labels.iter().map(|label| label.code.as_str()).collect()
}

fn keys(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("option_{i:03}")).collect()
}

fn refs(keys: &[String]) -> Vec<&str> {
    keys.iter().map(String::as_str).collect()
}

#[test]
fn letter_keys_are_their_own_codes_up_to_fifty_two() {
    let vocab = FakeVocab::new().d1_labels();
    let labels = choice(&vocab, &["b", "a", "Z"]).unwrap();
    assert_eq!(codes(&labels), ["b", "a", "Z"]);
    // Each group: the code's token, then " " + code's when it is a single,
    // different token.
    assert_eq!(
        labels[0].group,
        [100 + i32::from(b'b'), 300 + i32::from(b'b')]
    );

    let letters: Vec<String> = (b'A'..=b'Z')
        .chain(b'a'..=b'z')
        .map(|c| char::from(c).to_string())
        .collect();
    let labels = choice(&vocab, &refs(&letters)).unwrap();
    assert_eq!(
        codes(&labels),
        refs(&letters),
        "52 letter keys keep their letters"
    );

    // One key that is not a single ASCII letter, and every option gets a
    // code: A, B, C, … (a multi-byte letter is not ASCII).
    let labels = choice(&vocab, &["a", "é"]).unwrap();
    assert_eq!(codes(&labels), ["A", "B"]);
    let labels = choice(&vocab, &["a", "bb"]).unwrap();
    assert_eq!(codes(&labels), ["A", "B"]);
}

#[test]
fn past_26_options_codes_go_on_to_two_letters_and_fall_back_to_the_pool() {
    let vocab = FakeVocab::new().d1_labels();
    let keys = keys(40);
    let labels = choice(&vocab, &refs(&keys)).unwrap();
    let codes = codes(&labels);
    assert_eq!(codes[..26].concat(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    // AA..AI are single tokens on d1-2B; AJ is not, and takes the first
    // free pool entry: A..Z are taken, so 00.
    assert_eq!(
        codes[26..36],
        ["AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "00"]
    );
    assert_eq!(codes[36..40], ["AK", "AL", "AM", "AN"]);
    // A pool code without a single-token " " + code form has a one-token
    // group.
    assert_eq!(labels[35].group.len(), 1);
}

#[test]
fn the_pool_runs_out() {
    // A vocabulary where only single characters are single tokens: A..Z,
    // then a..z from the pool, then nothing (00..99, #k and AA..ZZ are two
    // tokens each).
    let vocab = FakeVocab::new();
    let keys = keys(53);
    let error = choice(&vocab, &refs(&keys)).unwrap_err();
    assert!(
        matches!(
            error,
            DecisionError::NoSingleTokenLabel {
                question: 0,
                option: 52
            }
        ),
        "{error:?}"
    );
    let labels = choice(&vocab, &refs(&keys[..52])).unwrap();
    assert_eq!(codes(&labels)[26..28], ["a", "b"]);
    assert_eq!(codes(&labels)[51], "z");
}

#[test]
fn aliases_are_not_reserved_so_one_token_can_score_two_options() {
    // " A" is the token of "B": option A's group reads it as its alias,
    // and option B, whose code is "B", reads it too. Upstream does not
    // reserve aliases; both groups select it.
    let vocab = FakeVocab::new()
        .single("A", 41)
        .single("B", 42)
        .single(" A", 42)
        .single(" B", 378);
    let labels = choice(&vocab, &["one", "two"]).unwrap();
    assert_eq!(codes(&labels), ["A", "B"]);
    assert_eq!(labels[0].group, [41, 42]);
    assert_eq!(labels[1].group, [42, 378]);

    // A primary already taken is not reused: if "B" were option A's token,
    // option B falls back to the pool.
    let vocab = FakeVocab::new().single("A", 41).single("B", 41);
    let labels = choice(&vocab, &["one", "two"]).unwrap();
    assert_eq!(codes(&labels), ["A", "C"]);
    // An alias equal to its own primary is dropped.
    let vocab = FakeVocab::new().single("A", 41).single(" A", 41);
    assert_eq!(choice(&vocab, &["one"]).unwrap()[0].group, [41]);
}

#[test]
fn yes_no_and_score_labels_have_fixed_forms() {
    let vocab = FakeVocab::new().d1_labels();
    let [yes, no] = labels::yes_no(0, &mut LabelTokens::new(&vocab)).unwrap();
    assert_eq!((yes.code.as_str(), no.code.as_str()), ("true", "false"));
    // `YES` is two tokens on d1-2B: dropped.
    assert_eq!(yes.group, [11683, 12447]);
    assert_eq!(no.group, [2243, 4547, 19598]);
    let levels = labels::score(0, 3, &mut LabelTokens::new(&vocab)).unwrap();
    assert_eq!(codes(&levels), ["0", "1", "2"]);
    assert_eq!(levels[2].group, [26], "no space alias for a level");

    // Distinct forms only: `Yes` spelling the same token as `yes` counts once.
    let vocab = FakeVocab::new()
        .single("yes", 7)
        .single("Yes", 7)
        .single("no", 8);
    let [yes, no] = labels::yes_no(0, &mut LabelTokens::new(&vocab)).unwrap();
    assert_eq!((yes.group, no.group), (vec![7], vec![8]));

    // No form of no is a single token; a level number that is not.
    let vocab = FakeVocab::new()
        .single("yes", 7)
        .single("no", 8)
        .single("NO", 8);
    assert!(
        labels::yes_no(3, &mut LabelTokens::new(&FakeVocab::new().single("yes", 7))).is_err_and(
            |error| matches!(
                error,
                DecisionError::NoSingleTokenLabel {
                    question: 3,
                    option: 1
                }
            )
        )
    );
    assert!(labels::yes_no(3, &mut LabelTokens::new(&vocab)).is_ok());
    let digits_but_7 = FakeVocab::new()
        .singles(["0", "1", "2", "3", "4", "5", "6"], 24)
        .pair("7");
    assert!(
        labels::score(1, 8, &mut LabelTokens::new(&digits_but_7)).is_err_and(|error| matches!(
            error,
            DecisionError::NoSingleTokenLabel {
                question: 1,
                option: 7
            }
        ))
    );
}
