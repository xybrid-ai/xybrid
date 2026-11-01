//! `d1_break`: LFM3 d1 keeps the caller's text from spelling special tokens.
//!
//! Every string and every object key of the template input goes through
//! [`Spellings::d1_break`], which inserts U+200B after the first character
//! of each CONTROL token spelling it finds, so the prompt's tokenizer (which
//! parses special tokens) reads them as text
//! (`tools/server/server-decision.cpp:606-661`, applied at `:1157-1159`).
//! [`Escaped`] is the escaped template input the renderer reads.

use std::collections::HashSet;
use std::sync::Arc;

use crate::ir::decision::{
    ContentPart, DecisionContent, DecisionContentView, DecisionError, JsonNumber, JsonView,
    OrderedJson,
};

/// U+200B ZERO WIDTH SPACE, what `d1_break` inserts.
const ZWSP: &str = "\u{200b}";

/// A vocabulary's CONTROL spellings, as `d1_break` tries them.
#[derive(Debug, Clone)]
pub(super) struct Spellings {
    /// Longest first; equal lengths keep token-id order (a stable sort).
    longest_first: Vec<String>,
    /// Which bytes start a spelling.
    first_bytes: [bool; 256],
}

impl Spellings {
    /// The spellings `d1_break` tries, from the vocabulary's CONTROL
    /// spellings in token-id order (`init_quoted_spellings`,
    /// `server-decision.cpp:213-230`): those of 2 bytes or more, longest
    /// first.
    ///
    /// # Errors
    ///
    /// [`DecisionError::UnsupportedVocabulary`] for a spelling that is not
    /// UTF-8, is a single character, or holds U+200B: breaking those cannot
    /// end (a single character, or a spelling holding the inserted U+200B,
    /// matches again after its break) or breaks a character apart. d1-2B's
    /// 124 spellings are ASCII, 7 bytes or longer.
    pub(super) fn new(control: &[Vec<u8>]) -> Result<Self, DecisionError> {
        let mut longest_first = Vec::with_capacity(control.len());
        for bytes in control.iter().filter(|bytes| bytes.len() >= 2) {
            let text =
                std::str::from_utf8(bytes).map_err(|_| DecisionError::UnsupportedVocabulary)?;
            if text.chars().nth(1).is_none() || text.contains(ZWSP) {
                return Err(DecisionError::UnsupportedVocabulary);
            }
            longest_first.push(text.to_string());
        }
        longest_first.sort_by_key(|spelling| std::cmp::Reverse(spelling.len()));
        let mut first_bytes = [false; 256];
        for spelling in &longest_first {
            first_bytes[usize::from(spelling.as_bytes()[0])] = true;
        }
        Ok(Self {
            longest_first,
            first_bytes,
        })
    }

    /// `text` with U+200B after the first character of every spelling in
    /// it, repeated until a pass finds none (`d1_break`,
    /// `server-decision.cpp:630-661`).
    ///
    /// Each pass scans left to right and breaks the longest spelling that
    /// starts at a position, then continues after it, so a spelling that
    /// starts inside one just broken is broken by the next pass. That ends:
    /// a break splits its own occurrence and, with the spellings [`new`]
    /// admits, creates none.
    ///
    /// [`new`]: Self::new
    pub(super) fn d1_break(&self, text: &str) -> String {
        let mut current = text.to_string();
        while let Some(next) = self.break_once(&current) {
            current = next;
        }
        current
    }

    /// One pass of [`Self::d1_break`]; `None` when it finds no spelling.
    fn break_once(&self, text: &str) -> Option<String> {
        let bytes = text.as_bytes();
        let mut out: Option<String> = None;
        let mut copied = 0; // bytes of `text` already in `out`
        let mut i = 0;
        while i < bytes.len() {
            let hit = self.first_bytes[usize::from(bytes[i])]
                .then(|| {
                    self.longest_first
                        .iter()
                        .find(|spelling| bytes[i..].starts_with(spelling.as_bytes()))
                })
                .flatten();
            let Some(spelling) = hit else {
                i += 1;
                continue;
            };
            // A spelling starts with a whole character and `text` is UTF-8,
            // so `i` and `i + first` are character boundaries.
            let first = spelling.chars().next().map_or(0, char::len_utf8);
            let out = out.get_or_insert_with(|| String::with_capacity(text.len() + 16));
            out.push_str(&text[copied..i + first]);
            out.push_str(ZWSP);
            out.push_str(&text[i + first..i + spelling.len()]);
            i += spelling.len();
            copied = i;
        }
        out.map(|mut out| {
            out.push_str(&text[copied..]);
            out
        })
    }

    #[cfg(test)]
    pub(super) fn longest_first(&self) -> &[String] {
        &self.longest_first
    }
}

/// A value of the escaped template input: JSON, its strings and keys run
/// through `d1_break`, numbers as the template engine reads them.
///
/// Containers hold their items behind `Arc`, so the renderer can hand any
/// part to the template without copying it.
#[derive(Clone, PartialEq)]
pub(super) enum Escaped {
    Null,
    Bool(bool),
    /// An integer: the engine reads every JSON integer as an `int64_t`.
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Arc<Escaped>>),
    /// Members in input order, keys unique.
    Object(Vec<(String, Arc<Escaped>)>),
}

/// Kind and size only: the value is the caller's content.
impl std::fmt::Debug for Escaped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Null => f.write_str("Null"),
            Self::Bool(_) => f.write_str("Bool"),
            Self::Int(_) => f.write_str("Int"),
            Self::Float(_) => f.write_str("Float"),
            Self::Str(text) => write!(f, "Str({} bytes)", text.len()),
            Self::Array(items) => write!(f, "Array({} items)", items.len()),
            Self::Object(members) => write!(f, "Object({} members)", members.len()),
        }
    }
}

impl Escaped {
    /// An object of the template's own structure (the question, an
    /// option): its keys are escaped too, as upstream escapes the whole
    /// input, and cannot collide (`d1_break` only inserts U+200B, which
    /// none of them holds, so it keeps distinct keys distinct).
    pub(super) fn structure(spellings: &Spellings, members: Vec<(&str, Arc<Escaped>)>) -> Self {
        Self::Object(
            members
                .into_iter()
                .map(|(key, value)| (spellings.d1_break(key), value))
                .collect(),
        )
    }

    /// An escaped string.
    pub(super) fn text(spellings: &Spellings, text: &str) -> Self {
        Self::Str(spellings.d1_break(text))
    }

    /// Escaped content: text, JSON, or `null` for none.
    ///
    /// # Errors
    ///
    /// [`DecisionError::EscapedKeyCollision`] when escaping makes two keys
    /// of one object equal (upstream keeps the last value); and
    /// [`DecisionError::NumberOutOfRange`] for an integer above
    /// `i64::MAX`, which upstream's engine reads wrapped to a negative
    /// number. `part` names where the content is.
    pub(super) fn content(
        spellings: &Spellings,
        content: Option<&DecisionContent>,
        part: ContentPart,
    ) -> Result<Self, DecisionError> {
        match content.map(DecisionContent::view) {
            None => Ok(Self::Null),
            Some(DecisionContentView::Text(text)) => Ok(Self::text(spellings, text)),
            Some(DecisionContentView::Json(value)) => Self::json(spellings, value, part),
        }
    }

    fn json(
        spellings: &Spellings,
        value: &OrderedJson,
        part: ContentPart,
    ) -> Result<Self, DecisionError> {
        Ok(match value.view() {
            JsonView::Null => Self::Null,
            JsonView::Bool(value) => Self::Bool(value),
            JsonView::Number(JsonNumber::I64(value)) => Self::Int(value),
            JsonView::Number(JsonNumber::U64(_)) => {
                return Err(DecisionError::NumberOutOfRange { part })
            }
            JsonView::Number(JsonNumber::F64(value)) => Self::Float(value),
            JsonView::String(text) => Self::text(spellings, text),
            JsonView::Array(items) => Self::Array(
                items
                    .iter()
                    .map(|item| Self::json(spellings, item, part).map(Arc::new))
                    .collect::<Result<_, _>>()?,
            ),
            JsonView::Object(members) => {
                let mut seen = HashSet::with_capacity(members.len());
                let mut escaped = Vec::with_capacity(members.len());
                for (key, item) in members {
                    let key = spellings.d1_break(key);
                    if !seen.insert(key.clone()) {
                        return Err(DecisionError::EscapedKeyCollision { part });
                    }
                    escaped.push((key, Arc::new(Self::json(spellings, item, part)?)));
                }
                Self::Object(escaped)
            }
        })
    }
}
