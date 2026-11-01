//! LFM3 d1's labels: the code each option is shown with, and the tokens
//! whose logits score it (`d1_labels`, `tools/server/server-decision.cpp:968-1065`;
//! the reference is `prompt.py`'s `option_codes`, `aliases` and
//! `readout_ids`).
//!
//! Every label form is tokenized without special tokens and without BOS
//! (`common_tokenize(vocab, form, false, false)`, `:974`); a form is usable
//! only if it is exactly one token.

use std::collections::{HashMap, HashSet};

use super::{DecisionVocab, Token};
use crate::ir::decision::DecisionError;

/// One option's label: the code shown in the prompt, and its token group
/// (the code's own token first, then any alias).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Label {
    pub(super) code: String,
    pub(super) group: Vec<Token>,
}

/// Tokenizes label forms, remembering each: the pool fallback tries the
/// same forms again for every code that misses.
pub(super) struct LabelTokens<'v> {
    vocab: &'v dyn DecisionVocab,
    single: HashMap<String, Option<Token>>,
}

impl<'v> LabelTokens<'v> {
    pub(super) fn new(vocab: &'v dyn DecisionVocab) -> Self {
        Self {
            vocab,
            single: HashMap::new(),
        }
    }

    /// The token `form` is, if it is exactly one.
    fn single(&mut self, form: &str) -> Result<Option<Token>, DecisionError> {
        if let Some(&token) = self.single.get(form) {
            return Ok(token);
        }
        let tokens = self.vocab.tokenize(form, false, false)?;
        let token = match tokens[..] {
            [token] => Some(token),
            _ => None,
        };
        self.single.insert(form.to_string(), token);
        Ok(token)
    }

    /// The distinct single tokens of `forms`, in order
    /// (`get_single_tokens`, `server-decision.cpp:971-980`).
    fn singles(&mut self, forms: &[&str]) -> Result<Vec<Token>, DecisionError> {
        let mut out = Vec::with_capacity(forms.len());
        for form in forms {
            if let Some(token) = self.single(form)? {
                if !out.contains(&token) {
                    out.push(token);
                }
            }
        }
        Ok(out)
    }
}

/// The labels of a yes/no question, in protocol order (true, then false):
/// the distinct single tokens of `yes`, `Yes`, `YES` and of `no`, `No`,
/// `NO`, with no space-prefixed alias (`server-decision.cpp:982-998`).
///
/// # Errors
///
/// [`DecisionError::NoSingleTokenLabel`] when no form of yes (option 0) or
/// no (option 1) is a single token.
pub(super) fn yes_no(
    question: usize,
    tokens: &mut LabelTokens<'_>,
) -> Result<[Label; 2], DecisionError> {
    let mut label = |option: usize, key: &str, forms: &[&str]| {
        let group = tokens.singles(forms)?;
        if group.is_empty() {
            return Err(DecisionError::NoSingleTokenLabel { question, option });
        }
        Ok(Label {
            code: key.to_string(),
            group,
        })
    };
    Ok([
        label(0, "true", &["yes", "Yes", "YES"])?,
        label(1, "false", &["no", "No", "NO"])?,
    ])
}

/// The labels of a score question of `levels` levels: level `i` is read
/// at the single token of its decimal number, with no alias and no pool
/// (`server-decision.cpp:982-998`).
///
/// # Errors
///
/// [`DecisionError::NoSingleTokenLabel`] when a level's number is not a
/// single token.
pub(super) fn score(
    question: usize,
    levels: usize,
    tokens: &mut LabelTokens<'_>,
) -> Result<Vec<Label>, DecisionError> {
    (0..levels)
        .map(|level| {
            let code = level.to_string();
            let group = tokens.singles(&[&code])?;
            if group.is_empty() {
                return Err(DecisionError::NoSingleTokenLabel {
                    question,
                    option: level,
                });
            }
            Ok(Label { code, group })
        })
        .collect()
}

/// The labels of a choice question with these option `keys`, in option
/// order (`server-decision.cpp:1000-1064`).
///
/// The codes are the keys themselves when every key is one ASCII letter
/// (up to 52 of them); otherwise `A`..`Z` for up to 26 options, and past 26
/// `A`..`Z` then `AA`, `AB`, …. Each code must be a single token not taken
/// by an earlier option; a code that is not falls back to the first free
/// entry of the pool `A`..`Z`, `00`..`99`, `a`..`z`, `#0`..`#199`,
/// `AA`..`ZZ`. An option's group is its code's token, then the token of
/// `" " + code` when that is a single, different token. Aliases are not
/// reserved: an alias may equal another option's code token.
///
/// # Errors
///
/// [`DecisionError::NoSingleTokenLabel`] naming the first option the pool
/// cannot label.
pub(super) fn choice(
    question: usize,
    keys: &[&str],
    tokens: &mut LabelTokens<'_>,
) -> Result<Vec<Label>, DecisionError> {
    let letters = keys
        .iter()
        .all(|key| key.len() == 1 && key.as_bytes()[0].is_ascii_alphabetic());
    let count = keys.len();
    let codes = keys.iter().enumerate().map(|(index, key)| {
        if letters {
            key.to_string()
        } else if count <= 26 || index < 26 {
            letter(index).to_string()
        } else {
            let index = index - 26;
            [letter(index / 26), letter(index % 26)].iter().collect()
        }
    });

    let pool = pool();
    let mut used: HashSet<Token> = HashSet::with_capacity(count);
    let mut labels = Vec::with_capacity(count);
    for (option, code) in codes.enumerate() {
        let mut label = take(&code, &mut used, tokens)?;
        for candidate in &pool {
            if label.is_some() {
                break;
            }
            label = take(candidate, &mut used, tokens)?;
        }
        labels.push(label.ok_or(DecisionError::NoSingleTokenLabel { question, option })?);
    }
    Ok(labels)
}

/// `take(code)`: the label of `code` if it is a single token nobody took
/// yet, which it then takes.
fn take(
    code: &str,
    used: &mut HashSet<Token>,
    tokens: &mut LabelTokens<'_>,
) -> Result<Option<Label>, DecisionError> {
    let Some(token) = tokens.single(code)? else {
        return Ok(None);
    };
    if !used.insert(token) {
        return Ok(None);
    }
    let mut group = vec![token];
    for alias in tokens.singles(&[&format!(" {code}")])? {
        if alias != token {
            group.push(alias);
        }
    }
    Ok(Some(Label {
        code: code.to_string(),
        group,
    }))
}

/// `A`..`Z`, by index.
fn letter(index: usize) -> char {
    char::from(b'A' + u8::try_from(index).expect("a letter index is below 26"))
}

/// The fallback codes, in the order they are tried.
fn pool() -> Vec<String> {
    let mut pool = Vec::with_capacity(26 + 100 + 26 + 200 + 26 * 26);
    pool.extend((b'A'..=b'Z').map(|c| char::from(c).to_string()));
    pool.extend((0..100).map(|i| format!("{i:02}")));
    pool.extend((b'a'..=b'z').map(|c| char::from(c).to_string()));
    pool.extend((0..200).map(|i| format!("#{i}")));
    for a in b'A'..=b'Z' {
        for b in b'A'..=b'Z' {
            pool.push([char::from(a), char::from(b)].iter().collect());
        }
    }
    pool
}
