//! `d1_break`: LFM3 d1 keeps the caller's text from spelling special tokens.
//!
//! Every string and every object key of the template input goes through
//! [`Spellings::d1_break`], which inserts U+200B after the first character
//! of each CONTROL token spelling it finds, so the prompt's tokenizer (which
//! parses special tokens) reads them as text
//! (`tools/server/server-decision.cpp:606-661`, applied at `:1157-1159`).
//! [`Escaped`] is the escaped template input the renderer reads.
//!
//! # Bounded work
//!
//! Upstream compares every spelling, longest first, at every position whose
//! byte starts one: a vocabulary of many long spellings sharing a prefix
//! costs `positions × spellings × length`. Here the spellings form a byte
//! trie, and the longest one starting at a position is found in one walk of
//! at most [`MAX_SPELLING_BYTES`] steps, which is the same choice (two
//! spellings of one length that both match are the same text). The
//! spellings themselves are capped ([`MAX_SPELLINGS`], [`MAX_SPELLING_BYTES`],
//! [`MAX_SPELLINGS_BYTES`]), and so is the work each string may take, all
//! passes together ([`ESCAPE_STEPS_PER_BYTE`] per byte of text, plus
//! [`ESCAPE_STEPS_PER_STRING`]). Past either, the vocabulary is refused
//! for the request, before anything is tokenized.

use std::collections::HashSet;
use std::sync::Arc;

use crate::ir::decision::{
    ContentPart, DecisionContent, DecisionContentView, DecisionError, JsonNumber, JsonView,
    OrderedJson,
};

/// U+200B ZERO WIDTH SPACE, what `d1_break` inserts.
const ZWSP: &str = "\u{200b}";

/// Most CONTROL spellings of 2 bytes or more a vocabulary may have: a round
/// ceiling well above d1-2B's 124 and d1-omni-2B's 126, to be raised with
/// evidence when a profile needs more.
pub(super) const MAX_SPELLINGS: usize = 4096;

/// Longest CONTROL spelling, in bytes: a round ceiling well above d1-2B's
/// longest (`<|img_row_10_col_10|>`, 21 bytes).
pub(super) const MAX_SPELLING_BYTES: usize = 256;

/// All CONTROL spellings together, in bytes: a round ceiling well above
/// d1-2B's 2,238 (its 124 spellings), to be raised with evidence when a
/// profile needs more.
pub(super) const MAX_SPELLINGS_BYTES: usize = 64 << 10;

/// Trie steps escaping may take per byte of a string, all passes together.
/// d1-2B's spellings all start with `<` and none starts inside another, so
/// a string costs at most 2 passes of 1 step per byte plus at most 21 per
/// `<`: 64 is far above that, and far below what an unbounded vocabulary
/// could cost.
pub(super) const ESCAPE_STEPS_PER_BYTE: usize = 64;

/// Trie steps escaping may take per string on top of
/// [`ESCAPE_STEPS_PER_BYTE`], for short strings.
pub(super) const ESCAPE_STEPS_PER_STRING: usize = 1024;

/// The trie steps escaping `text` may take, all passes together.
pub(super) fn escape_budget(text: &str) -> usize {
    ESCAPE_STEPS_PER_BYTE
        .saturating_mul(text.len())
        .saturating_add(ESCAPE_STEPS_PER_STRING)
}

/// A vocabulary's CONTROL spellings, as `d1_break` tries them.
#[derive(Debug, Clone)]
pub(super) struct Spellings {
    /// One node per byte of the spellings: the root, then each prefix.
    nodes: Vec<Node>,
}

/// A trie node: its children by byte, and the length in bytes of the first
/// character of the spelling that ends here, if one does.
#[derive(Debug, Clone, Default)]
struct Node {
    children: Vec<(u8, u32)>,
    first_char: Option<u8>,
}

impl Node {
    fn child(&self, byte: u8) -> Option<usize> {
        self.children
            .binary_search_by_key(&byte, |&(b, _)| b)
            .ok()
            .map(|index| self.children[index].1 as usize)
    }
}

impl Spellings {
    /// The spellings `d1_break` tries, from the vocabulary's CONTROL
    /// spellings in token-id order (`init_quoted_spellings`,
    /// `server-decision.cpp:213-230`): those of 2 bytes or more.
    ///
    /// # Errors
    ///
    /// [`DecisionError::UnsupportedVocabulary`] for a spelling that is not
    /// UTF-8, is a single character, or holds U+200B (breaking those cannot
    /// end: a single character, or a spelling holding the inserted U+200B,
    /// matches again after its break; or it breaks a character apart), and
    /// for spellings past [`MAX_SPELLINGS`], [`MAX_SPELLING_BYTES`] or
    /// [`MAX_SPELLINGS_BYTES`]. d1-2B's 124 spellings are ASCII, 7 to 21
    /// bytes.
    pub(super) fn new(control: &[Vec<u8>]) -> Result<Self, DecisionError> {
        let refuse = || DecisionError::UnsupportedVocabulary;
        let mut count = 0usize;
        let mut total = 0usize;
        let mut nodes = vec![Node::default()];
        for bytes in control.iter().filter(|bytes| bytes.len() >= 2) {
            count += 1;
            total += bytes.len();
            if count > MAX_SPELLINGS
                || bytes.len() > MAX_SPELLING_BYTES
                || total > MAX_SPELLINGS_BYTES
            {
                return Err(refuse());
            }
            let text = std::str::from_utf8(bytes).map_err(|_| refuse())?;
            let first = text.chars().next().map_or(0, char::len_utf8);
            if text.chars().nth(1).is_none() || text.contains(ZWSP) {
                return Err(refuse());
            }
            let mut node = 0;
            for &byte in bytes {
                node = match nodes[node].child(byte) {
                    Some(child) => child,
                    None => {
                        let child = nodes.len();
                        nodes.push(Node::default());
                        let children = &mut nodes[node].children;
                        let at = children.partition_point(|&(b, _)| b < byte);
                        children.insert(at, (byte, child as u32));
                        child
                    }
                };
            }
            // A repeated spelling is the same text: the first one stands.
            nodes[node]
                .first_char
                .get_or_insert(u8::try_from(first).expect("a character is at most 4 bytes"));
        }
        Ok(Self { nodes })
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
    /// # Errors
    ///
    /// [`DecisionError::EscapeWorkExceeded`] when the passes would take more
    /// than [`ESCAPE_STEPS_PER_BYTE`] trie steps per byte of `text` plus
    /// [`ESCAPE_STEPS_PER_STRING`].
    ///
    /// [`new`]: Self::new
    pub(super) fn d1_break(&self, text: &str) -> Result<String, DecisionError> {
        self.d1_break_within(text, &mut escape_budget(text))
    }

    /// [`Self::d1_break`] and the trie steps it took: its whole budget when
    /// it refuses.
    #[cfg(test)]
    pub(super) fn d1_break_counted(&self, text: &str) -> (Result<String, DecisionError>, usize) {
        let budget = escape_budget(text);
        let mut steps = budget;
        let result = self.d1_break_within(text, &mut steps);
        (result, budget - steps)
    }

    /// [`Self::d1_break`] within `steps` trie steps, which it takes from.
    fn d1_break_within(&self, text: &str, steps: &mut usize) -> Result<String, DecisionError> {
        let mut current = text.to_string();
        while let Some(next) = self.break_once(&current, steps)? {
            current = next;
        }
        Ok(current)
    }

    /// One pass of [`Self::d1_break`]; `None` when it finds no spelling.
    /// Every trie step (one byte looked up) takes one of `steps`.
    fn break_once(&self, text: &str, steps: &mut usize) -> Result<Option<String>, DecisionError> {
        let bytes = text.as_bytes();
        let mut out: Option<String> = None;
        let mut copied = 0; // bytes of `text` already in `out`
        let mut i = 0;
        while i < bytes.len() {
            let Some((len, first)) = self.longest_at(&bytes[i..], steps)? else {
                i += 1;
                continue;
            };
            // A spelling starts with a whole character and `text` is UTF-8,
            // so `i` and `i + first` are character boundaries.
            let out = out.get_or_insert_with(|| String::with_capacity(text.len() + 16));
            out.push_str(&text[copied..i + first]);
            out.push_str(ZWSP);
            out.push_str(&text[i + first..i + len]);
            i += len;
            copied = i;
        }
        Ok(out.map(|mut out| {
            out.push_str(&text[copied..]);
            out
        }))
    }

    /// The length of the longest spelling `text` starts with, and the length
    /// of its first character.
    fn longest_at(
        &self,
        text: &[u8],
        steps: &mut usize,
    ) -> Result<Option<(usize, usize)>, DecisionError> {
        let mut node = 0;
        let mut longest = None;
        for (depth, &byte) in text.iter().enumerate() {
            *steps = steps
                .checked_sub(1)
                .ok_or(DecisionError::EscapeWorkExceeded)?;
            let Some(child) = self.nodes[node].child(byte) else {
                break;
            };
            node = child;
            if let Some(first) = self.nodes[node].first_char {
                longest = Some((depth + 1, usize::from(first)));
            }
        }
        Ok(longest)
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
    pub(super) fn structure(
        spellings: &Spellings,
        members: Vec<(&str, Arc<Escaped>)>,
    ) -> Result<Self, DecisionError> {
        Ok(Self::Object(
            members
                .into_iter()
                .map(|(key, value)| Ok((spellings.d1_break(key)?, value)))
                .collect::<Result<_, DecisionError>>()?,
        ))
    }

    /// An escaped string.
    pub(super) fn text(spellings: &Spellings, text: &str) -> Result<Self, DecisionError> {
        Ok(Self::Str(spellings.d1_break(text)?))
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
            Some(DecisionContentView::Text(text)) => Self::text(spellings, text),
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
            JsonView::String(text) => Self::text(spellings, text)?,
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
                    let key = spellings.d1_break(key)?;
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
