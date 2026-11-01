//! The request body as JSON: validated whole, read only as deep as the
//! transport looks.
//!
//! A SystemOne request nests its content inside the request's own objects,
//! so the body cannot go through [`OrderedJson::parse`] in one piece: content
//! may nest up to [`MAX_JSON_DEPTH`] levels counting from itself (LFM3 d1
//! accepts instructions 127 deep), which puts the body itself deeper than
//! that. llama.cpp's JSON parser has no depth limit either; its decision code
//! checks the depth of each part of the prompt later. So the transport reads
//! the body in two steps:
//!
//! 1. [`read`] checks the whole body against the grammar
//!    [`OrderedJson::parse`] applies (nlohmann::json's, which llama.cpp
//!    parses requests with), without a depth limit and without recursing,
//!    and refuses repeated keys and numbers that overflow `f64`. It keeps the
//!    containers down to [`READ_DEPTH`] as [`Value`]s, each with the byte
//!    range of its text, and skips what lies deeper.
//! 2. The request reader ([`super::request`]) walks those values, and hands
//!    the text of each piece of content to [`OrderedJson::parse`], which is
//!    the only way to build an [`OrderedJson`] and checks its depth from the
//!    content itself.
//!
//! The grammar here is [`OrderedJson::parse`]'s, so within its own limits
//! (depth, size and value ceilings) the two accept exactly the same texts.
//! Where a text has more than one problem, the one reported can differ, on
//! purpose: a repeated key is reported here only once the whole text is
//! known to be JSON, so `[{"a":1,"a":2},]` is a syntax error here and a
//! repeated key there; and among several repeats this module names the
//! first in the text, while [`OrderedJson::parse`] names one in the first
//! object to close. The tests compare the two on a corpus and on generated
//! and mutated texts, and pin these differences.
//!
//! [`OrderedJson`]: xybrid_core::ir::OrderedJson
//! [`OrderedJson::parse`]: xybrid_core::ir::OrderedJson::parse
//! [`MAX_JSON_DEPTH`]: xybrid_core::ir::decision::MAX_JSON_DEPTH

use std::collections::HashSet;
use std::ops::Range;
use xybrid_core::ir::JsonErrorKind;

/// Containers down to this depth (the body's object is 1) are kept as
/// [`Value`]s; their items and members are kept too, but a container below
/// it is only checked and located ([`Node::Unread`]).
///
/// The deepest the transport looks is into a content part of a chat message
/// in a state that wraps its messages: body (1), state (2), `messages` (3),
/// a message (4), its `content` (5), a part (6), whose members (7) it reads
/// by kind and, for `type`, by string value. Everything else it reads is
/// shallower.
pub(super) const READ_DEPTH: usize = 6;

/// Why a body is not one JSON value the transport reads.
///
/// Offsets are byte offsets into the body; nothing quotes the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BodyError {
    /// The body breaks the JSON grammar, has a number that overflows
    /// `f64` ([`JsonErrorKind::NumberOutOfRange`]), or (only once the rest
    /// is known to be JSON) an object that repeats a key
    /// ([`JsonErrorKind::DuplicateKey`], at the repeat).
    Json { kind: JsonErrorKind, offset: usize },
    /// The body holds more values than the transport reads.
    TooManyValues { max: usize },
}

/// One value of the body: where its text is, and what it is.
pub(super) struct Value {
    /// The value's text in the body, without surrounding whitespace.
    pub(super) span: Range<usize>,
    pub(super) node: Node,
}

/// What a [`Value`] is, read as far as [`READ_DEPTH`].
pub(super) enum Node {
    /// `null`.
    Null,
    /// `true`, `false` or a number.
    Scalar,
    /// A string, unescaped.
    String(String),
    /// An array's items, in order.
    Array(Vec<Value>),
    /// An object's members, in order; keys are unique.
    Object(Vec<(String, Value)>),
    /// An array or object below [`READ_DEPTH`]: checked, not kept.
    Unread,
}

impl Value {
    /// `true` for `null`.
    pub(super) fn is_null(&self) -> bool {
        matches!(self.node, Node::Null)
    }

    /// The text of a string.
    pub(super) fn as_str(&self) -> Option<&str> {
        match &self.node {
            Node::String(text) => Some(text),
            _ => None,
        }
    }

    /// The items of an array within [`READ_DEPTH`].
    pub(super) fn as_array(&self) -> Option<&[Value]> {
        match &self.node {
            Node::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The members of an object within [`READ_DEPTH`].
    pub(super) fn as_object(&self) -> Option<&[(String, Value)]> {
        match &self.node {
            Node::Object(members) => Some(members),
            _ => None,
        }
    }

    /// The value of member `key` of an object within [`READ_DEPTH`].
    pub(super) fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }
}

/// Reads a body that must be one JSON value, with at most `max_values`
/// values (every scalar, array and object counts as one; keys do not).
///
/// The grammar is [`OrderedJson::parse`]'s (RFC 8259 as nlohmann::json
/// applies it, a leading UTF-8 byte-order mark skipped), with no depth
/// limit. Grammar errors and numbers that overflow `f64` are reported at the
/// first one in the text; a repeated key only once the whole body is known
/// to be JSON, so a body that is not JSON is always refused as such.
///
/// [`OrderedJson::parse`]: xybrid_core::ir::OrderedJson::parse
pub(super) fn read(text: &str, max_values: usize) -> Result<Value, BodyError> {
    let mut scanner = Scanner {
        text,
        bytes: text.as_bytes(),
        pos: 0,
        values: 0,
        max_values,
        first_repeat: None,
    };
    // nlohmann::json skips a leading byte-order mark, and so does
    // `OrderedJson::parse`.
    if text.starts_with('\u{feff}') {
        scanner.pos = '\u{feff}'.len_utf8();
    }
    scanner.skip_whitespace();
    let value = scanner.document()?;
    scanner.skip_whitespace();
    if scanner.pos < scanner.bytes.len() {
        return Err(scanner.error(JsonErrorKind::TrailingCharacters));
    }
    if let Some(offset) = scanner.first_repeat {
        return Err(BodyError::Json {
            kind: JsonErrorKind::DuplicateKey,
            offset,
        });
    }
    Ok(value)
}

struct Scanner<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    values: usize,
    max_values: usize,
    /// Offset of the first repeated key in the text, reported once the
    /// whole body is known to be JSON.
    first_repeat: Option<usize>,
}

/// How a value starts: whole (a scalar), or as a container to fill.
enum Start {
    Scalar(Node),
    Array,
    Object,
}

/// A container being filled, innermost last on the scanner's stack.
enum Open {
    /// An array within [`READ_DEPTH`].
    Array { start: usize, items: Vec<Value> },
    /// An object within [`READ_DEPTH`]; `key` names the member being read.
    Object {
        start: usize,
        members: Vec<(String, Value)>,
        key_offsets: Vec<usize>,
        key: String,
        key_offset: usize,
    },
    /// An array below [`READ_DEPTH`]: its items are checked, not kept.
    UnreadArray { start: usize },
    /// An object below [`READ_DEPTH`]: its keys are kept until it closes,
    /// to find a repeat.
    UnreadObject { start: usize, keys: HashSet<String> },
}

impl Scanner<'_> {
    fn error(&self, kind: JsonErrorKind) -> BodyError {
        self.error_at(kind, self.pos)
    }

    fn error_at(&self, kind: JsonErrorKind, offset: usize) -> BodyError {
        BodyError::Json { kind, offset }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// JSON whitespace only: space, tab, line feed, carriage return.
    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.pos += 1;
        }
    }

    fn skip_digits(&mut self) {
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
    }

    /// Notes a repeated key at `offset`; the first in the text is reported.
    fn repeat_at(&mut self, offset: usize) {
        self.first_repeat = Some(self.first_repeat.map_or(offset, |first| first.min(offset)));
    }

    /// The body's value. Containers being filled wait on `open`; a value is
    /// at depth `open.len() + 1`.
    fn document(&mut self) -> Result<Value, BodyError> {
        let mut open: Vec<Open> = Vec::new();
        loop {
            let depth = open.len() + 1;
            let start = self.pos;
            let mut value = match self.start()? {
                Start::Scalar(node) => Value {
                    span: start..self.pos,
                    node,
                },
                Start::Array if self.peek() == Some(b']') => {
                    self.pos += 1;
                    let node = if depth <= READ_DEPTH {
                        Node::Array(Vec::new())
                    } else {
                        Node::Unread
                    };
                    Value {
                        span: start..self.pos,
                        node,
                    }
                }
                Start::Array => {
                    open.push(if depth <= READ_DEPTH {
                        Open::Array {
                            start,
                            items: Vec::new(),
                        }
                    } else {
                        Open::UnreadArray { start }
                    });
                    continue;
                }
                Start::Object if self.peek() == Some(b'}') => {
                    self.pos += 1;
                    let node = if depth <= READ_DEPTH {
                        Node::Object(Vec::new())
                    } else {
                        Node::Unread
                    };
                    Value {
                        span: start..self.pos,
                        node,
                    }
                }
                Start::Object => {
                    let (key, key_offset) = self.member_key()?;
                    open.push(if depth <= READ_DEPTH {
                        Open::Object {
                            start,
                            members: Vec::new(),
                            key_offsets: Vec::new(),
                            key,
                            key_offset,
                        }
                    } else {
                        Open::UnreadObject {
                            start,
                            keys: HashSet::from([key]),
                        }
                    });
                    continue;
                }
            };
            // `value` is whole: add it to its container, and close every
            // container it completes.
            loop {
                let Some(container) = open.pop() else {
                    return Ok(value);
                };
                match container {
                    Open::Array { start, mut items } => {
                        items.push(value);
                        if self.separator(b']')? {
                            open.push(Open::Array { start, items });
                            break;
                        }
                        value = Value {
                            span: start..self.pos,
                            node: Node::Array(items),
                        };
                    }
                    Open::Object {
                        start,
                        mut members,
                        mut key_offsets,
                        key,
                        key_offset,
                    } => {
                        members.push((key, value));
                        key_offsets.push(key_offset);
                        if self.separator(b'}')? {
                            let (key, key_offset) = self.member_key()?;
                            open.push(Open::Object {
                                start,
                                members,
                                key_offsets,
                                key,
                                key_offset,
                            });
                            break;
                        }
                        if let Some(index) = first_repeated_key(&members) {
                            self.repeat_at(key_offsets[index]);
                        }
                        value = Value {
                            span: start..self.pos,
                            node: Node::Object(members),
                        };
                    }
                    Open::UnreadArray { start } => {
                        if self.separator(b']')? {
                            open.push(Open::UnreadArray { start });
                            break;
                        }
                        value = Value {
                            span: start..self.pos,
                            node: Node::Unread,
                        };
                    }
                    Open::UnreadObject { start, mut keys } => {
                        if self.separator(b'}')? {
                            let (key, key_offset) = self.member_key()?;
                            if !keys.insert(key) {
                                self.repeat_at(key_offset);
                            }
                            open.push(Open::UnreadObject { start, keys });
                            break;
                        }
                        value = Value {
                            span: start..self.pos,
                            node: Node::Unread,
                        };
                    }
                }
            }
        }
    }

    /// Starts a value: a scalar is read whole, an array or an object is
    /// opened (its bracket and the whitespace after it read).
    fn start(&mut self) -> Result<Start, BodyError> {
        match self.peek() {
            None => return Err(self.error(JsonErrorKind::UnexpectedEnd)),
            Some(b'n' | b't' | b'f' | b'"' | b'[' | b'{' | b'-' | b'0'..=b'9') => {}
            Some(_) => return Err(self.error(JsonErrorKind::ExpectedValue)),
        }
        self.values += 1;
        if self.values > self.max_values {
            return Err(BodyError::TooManyValues {
                max: self.max_values,
            });
        }
        let node = match self.peek() {
            Some(b'[') => {
                self.pos += 1;
                self.skip_whitespace();
                return Ok(Start::Array);
            }
            Some(b'{') => {
                self.pos += 1;
                self.skip_whitespace();
                return Ok(Start::Object);
            }
            Some(b'n') => {
                self.literal(b"null")?;
                Node::Null
            }
            Some(b't') => {
                self.literal(b"true")?;
                Node::Scalar
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Node::Scalar
            }
            Some(b'"') => Node::String(self.string()?),
            _ => {
                self.number()?;
                Node::Scalar
            }
        };
        Ok(Start::Scalar(node))
    }

    fn literal(&mut self, word: &[u8]) -> Result<(), BodyError> {
        if self.bytes[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(())
        } else {
            Err(self.error(JsonErrorKind::InvalidLiteral))
        }
    }

    /// After an item or member: `,` (true) or the closing bracket (false).
    fn separator(&mut self, close: u8) -> Result<bool, BodyError> {
        self.skip_whitespace();
        match self.peek() {
            Some(b',') => {
                self.pos += 1;
                self.skip_whitespace();
                Ok(true)
            }
            Some(byte) if byte == close => {
                self.pos += 1;
                Ok(false)
            }
            None => Err(self.error(JsonErrorKind::UnexpectedEnd)),
            Some(_) => Err(self.error(JsonErrorKind::ExpectedCommaOrEnd)),
        }
    }

    /// A member's key and its colon, with the whitespace after it; returns
    /// the key and the byte offset of its opening quote.
    fn member_key(&mut self) -> Result<(String, usize), BodyError> {
        match self.peek() {
            Some(b'"') => {}
            None => return Err(self.error(JsonErrorKind::UnexpectedEnd)),
            Some(_) => return Err(self.error(JsonErrorKind::ExpectedKey)),
        }
        let key_offset = self.pos;
        let key = self.string()?;
        self.skip_whitespace();
        match self.peek() {
            Some(b':') => self.pos += 1,
            None => return Err(self.error(JsonErrorKind::UnexpectedEnd)),
            Some(_) => return Err(self.error(JsonErrorKind::ExpectedColon)),
        }
        self.skip_whitespace();
        Ok((key, key_offset))
    }

    /// A string starting at its opening quote, unescaped.
    fn string(&mut self) -> Result<String, BodyError> {
        self.pos += 1; // '"'
        let mut out = String::new();
        loop {
            // Copy a run of plain characters. The run stops on ASCII bytes
            // only, so both ends are character boundaries.
            let run = self.pos;
            while let Some(&byte) = self.bytes.get(self.pos) {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            out.push_str(&self.text[run..self.pos]);
            match self.peek() {
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => self.escape(&mut out)?,
                Some(_) => return Err(self.error(JsonErrorKind::ControlCharacter)),
                None => return Err(self.error(JsonErrorKind::UnexpectedEnd)),
            }
        }
    }

    /// One escape sequence starting at its backslash.
    fn escape(&mut self, out: &mut String) -> Result<(), BodyError> {
        let start = self.pos;
        self.pos += 1; // '\'
        let unescaped = match self.peek() {
            None => return Err(self.error(JsonErrorKind::UnexpectedEnd)),
            Some(b'u') => {
                self.pos += 1;
                return self.unicode_escape(start, out);
            }
            Some(b'"') => '"',
            Some(b'\\') => '\\',
            Some(b'/') => '/',
            Some(b'b') => '\u{8}',
            Some(b'f') => '\u{c}',
            Some(b'n') => '\n',
            Some(b'r') => '\r',
            Some(b't') => '\t',
            Some(_) => return Err(self.error_at(JsonErrorKind::InvalidEscape, start)),
        };
        self.pos += 1;
        out.push(unescaped);
        Ok(())
    }

    /// Unescapes a `\u` escape whose backslash is at `start` into `out`,
    /// taking the escape that completes a surrogate pair with it.
    fn unicode_escape(&mut self, start: usize, out: &mut String) -> Result<(), BodyError> {
        let first = self.hex4(start)?;
        let code = match first {
            0xD800..=0xDBFF => {
                let low_start = self.pos;
                if !self.bytes[self.pos..].starts_with(b"\\u") {
                    return Err(self.error_at(JsonErrorKind::UnpairedSurrogate, start));
                }
                self.pos += 2;
                let low = self.hex4(low_start)?;
                if !(0xDC00..=0xDFFF).contains(&low) {
                    return Err(self.error_at(JsonErrorKind::UnpairedSurrogate, start));
                }
                0x10000 + ((first - 0xD800) << 10) + (low - 0xDC00)
            }
            0xDC00..=0xDFFF => {
                return Err(self.error_at(JsonErrorKind::UnpairedSurrogate, start));
            }
            code => code,
        };
        // Surrogates are handled above, so every code here is a scalar value.
        let unescaped = char::from_u32(code)
            .ok_or_else(|| self.error_at(JsonErrorKind::InvalidEscape, start))?;
        out.push(unescaped);
        Ok(())
    }

    /// The four hex digits of the `\u` escape whose backslash is at `start`.
    fn hex4(&mut self, start: usize) -> Result<u32, BodyError> {
        let mut code = 0;
        for _ in 0..4 {
            let digit = match self.peek() {
                None => return Err(self.error(JsonErrorKind::UnexpectedEnd)),
                Some(byte) => char::from(byte)
                    .to_digit(16)
                    .ok_or_else(|| self.error_at(JsonErrorKind::InvalidEscape, start))?,
            };
            code = code * 16 + digit;
            self.pos += 1;
        }
        Ok(code)
    }

    /// A number: checked against the grammar and the `f64` range, not kept
    /// (content is read again by `OrderedJson::parse`).
    fn number(&mut self) -> Result<(), BodyError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if let Some(b'0'..=b'9') = self.peek() {
                    return Err(self.error_at(JsonErrorKind::InvalidNumber, start));
                }
            }
            Some(b'1'..=b'9') => self.skip_digits(),
            None => return Err(self.error(JsonErrorKind::UnexpectedEnd)),
            Some(_) => return Err(self.error_at(JsonErrorKind::InvalidNumber, start)),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error_at(JsonErrorKind::InvalidNumber, start));
            }
            self.skip_digits();
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.pos += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error_at(JsonErrorKind::InvalidNumber, start));
            }
            self.skip_digits();
        }
        // An integer that fits i64 or u64 is finite as an f64 too, and an
        // integer beyond both is a float to nlohmann::json; only a value
        // past the f64 range is refused, as nlohmann refuses it ("number
        // overflow").
        match self.text[start..self.pos].parse::<f64>() {
            Ok(value) if value.is_finite() => Ok(()),
            Ok(_) => Err(self.error_at(JsonErrorKind::NumberOutOfRange, start)),
            // Unreachable for a literal that passed the grammar above.
            Err(_) => Err(self.error_at(JsonErrorKind::InvalidNumber, start)),
        }
    }
}

/// Index of the first member whose key an earlier member already used.
fn first_repeated_key(members: &[(String, Value)]) -> Option<usize> {
    if members.len() < 2 {
        return None;
    }
    let mut seen = HashSet::with_capacity(members.len());
    members
        .iter()
        .position(|(key, _)| !seen.insert(key.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_core::ir::decision::JsonLimits;
    use xybrid_core::ir::{DecisionError, OrderedJson};

    const LOTS: usize = 1 << 20;

    fn read_text(text: &str) -> Result<Value, BodyError> {
        read(text, LOTS)
    }

    /// Why `read` and `OrderedJson::parse` disagree about `text` under a
    /// value ceiling of `limit`, beyond what the module docs allow; `None`
    /// when they agree.
    ///
    /// They must accept the same texts. A text that is not JSON is refused
    /// by both at the same first problem, kind and offset, unless
    /// `OrderedJson::parse` met a repeated key before it: then this module
    /// reports the first syntax problem (or the value ceiling) instead, which
    /// comes after that object closed. A text that is JSON but repeats a key
    /// is refused by both for a repeat, this module naming the first in the
    /// text, so never a later one.
    fn disagreement(text: &str, limit: usize) -> Option<String> {
        const REPEAT: JsonErrorKind = JsonErrorKind::DuplicateKey;
        let ours = read(text, limit).map(|_| ());
        let limits = JsonLimits::default()
            .with_max_bytes(usize::MAX)
            .with_max_nodes(limit);
        let core = OrderedJson::parse_with_limits(text, &limits).map(|_| ());
        let agree = match (&ours, &core) {
            (Ok(()), Ok(())) => true,
            (
                Err(BodyError::TooManyValues { max }),
                Err(DecisionError::JsonTooManyNodes { max: core_max }),
            ) => max == core_max,
            (
                Err(BodyError::Json { kind, offset }),
                Err(DecisionError::InvalidJson {
                    kind: core_kind,
                    offset: core_offset,
                }),
            ) if *core_kind != REPEAT => kind == core_kind && offset == core_offset,
            (
                Err(BodyError::Json { kind, offset }),
                Err(DecisionError::InvalidJson {
                    kind: REPEAT,
                    offset: core_offset,
                }),
            ) if *kind == REPEAT => offset <= core_offset,
            (
                Err(BodyError::Json { offset, .. }),
                Err(DecisionError::InvalidJson {
                    kind: REPEAT,
                    offset: core_offset,
                }),
            ) => offset > core_offset,
            (
                Err(BodyError::TooManyValues { .. }),
                Err(DecisionError::InvalidJson { kind: REPEAT, .. }),
            ) => true,
            _ => false,
        };
        (!agree).then(|| format!("{text:?} under {limit} values: ours {ours:?}, core {core:?}"))
    }

    /// Texts within `OrderedJson::parse`'s depth and size limits, valid and
    /// not, covering every rule of the grammar.
    const CORPUS: &[&str] = &[
        // Values and whitespace.
        "null",
        "true",
        "false",
        "0",
        "-0",
        "1",
        "-1",
        "1.5",
        "-0.0",
        "1e5",
        "1E+5",
        "1e-5",
        "12.5e3",
        "\"\"",
        "\"a\"",
        "[]",
        "{}",
        " \t\r\n[ 1 , 2 ]\n",
        "{\"a\" : [ ] }",
        "\u{feff}{}",
        "\u{feff} [1]",
        // Numbers at and past the integer and float ranges.
        "9223372036854775807",
        "9223372036854775808",
        "18446744073709551615",
        "18446744073709551616",
        "-9223372036854775808",
        "-9223372036854775809",
        "1.7976931348623157e308",
        "1.7976931348623159e308",
        "1e400",
        "-1e400",
        "1e-400",
        "123456789012345678901234567890",
        // Malformed numbers.
        "01",
        "-01",
        "+1",
        ".5",
        "1.",
        "1.e5",
        "1e",
        "1e+",
        "-",
        "-a",
        "0x10",
        "1.5.5",
        "NaN",
        "Infinity",
        "-Infinity",
        // Literals.
        "nul",
        "nulll",
        "tru",
        "True",
        "fals",
        "none",
        // Strings and escapes.
        "\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"",
        "\"\\u0041\\u00e9\\u4e2d\"",
        "\"\\uD83D\\uDE00\"",
        "\"\\ud83d\\ude00\"",
        "\"\\uD83D\"",
        "\"\\uD83Dx\"",
        "\"\\uD83D\\u0041\"",
        "\"\\uDE00\"",
        "\"\\uD83D\\uDE0\"",
        "\"\\u12\"",
        "\"\\u12G4\"",
        "\"\\x\"",
        "\"\\",
        "\"\\u",
        "\"abc",
        "\"a\u{1}b\"",
        "\"a\tb\"",
        "\"é中😀\"",
        "\"\u{0}\"",
        "\"\\u0000\"",
        // Arrays and objects.
        "[",
        "]",
        "[1,]",
        "[,1]",
        "[1 2]",
        "[1,,2]",
        "{",
        "}",
        "{\"a\"}",
        "{\"a\":}",
        "{\"a\" 1}",
        "{a:1}",
        "{\"a\":1,}",
        "{,\"a\":1}",
        "{\"a\":1 \"b\":2}",
        "{1:2}",
        "[{\"a\":[{\"b\":{}}]}]",
        "[[[[[[[[[[1]]]]]]]]]]",
        // Repeated keys, raw and escaped, at several depths.
        "{\"a\":1,\"a\":2}",
        "{\"a\":1,\"\\u0061\":2}",
        "{\"a\":{\"b\":1,\"b\":2}}",
        "[[[[[[[[{\"k\":1,\"k\":2}]]]]]]]]",
        "{\"a\":1,\"b\":{\"c\":1,\"c\":2},\"a\":3}",
        // Trailing data and empty input.
        "",
        " ",
        "{} {}",
        "[1]]",
        "1 2",
        "\"a\"\"b\"",
        "nullx",
    ];

    #[test]
    fn the_grammar_matches_ordered_json() {
        for text in CORPUS {
            for limit in [LOTS, 3] {
                if let Some(disagreement) = disagreement(text, limit) {
                    panic!("{disagreement}");
                }
            }
        }
    }

    /// A deterministic generator (splitmix64), so every run sees the same
    /// texts and a failure replays.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            usize::try_from(self.next() % u64::try_from(n).unwrap()).unwrap()
        }
    }

    /// Texts to mutate: JSON with nested containers, every kind of number
    /// and escape, odd keys, whitespace and a byte-order mark, and two that
    /// repeat keys.
    const SEEDS: &[&str] = &[
        r#"{"state": {"b": [1, -0, 2.5e-3, {"k": null}], "a": "x\n\"\\\u00e9\ud83d\ude00"}, "questions": {"q": {"type": "choice", "instructions": [true, false], "criteria": {"a": 1.0, "b": [{}]}}}}"#,
        r#"[0, -1, 1.5, 1e10, -1E-5, 0.000123, 9223372036854775807, 18446744073709551615, 18446744073709551616, 1.7976931348623157e308, 5e-324]"#,
        r#"{"a": {"b": {"c": {"d": [[[[{"e": "f", "g": [null]}]]]]}}}}"#,
        r#"{"esc": "\/\b\f\n\r\t\u0000\u001f\"\\", "uni": "\u4e2d\uD83D\uDE00é中"}"#,
        "\u{feff} {\"bom\": true} ",
        r#"{"a": 1, "b": {"c": 1, "c": 2}, "a": 3}"#,
        r#"[{"a":1,"a":2},[1,2,{"z":[]}],{"k":{"k":{"k":1}}}]"#,
        r#"{"":"","\u0000":0,"a b":[ ],"x":{ }}"#,
        "[\n  1,\n  {\n    \"key\" : \"value\"\n  }\n]\n",
        r#""a string with \"quotes\" and \\u0041""#,
        "123",
        "-0.0e-0",
        "true",
        "null",
    ];

    /// Bytes a mutation inserts or writes: JSON's structure, number and
    /// literal characters, whitespace and control characters.
    const BYTES: &[u8] = b"{}[]\",:\\/-+.eE0123456789 \t\n\rtrufalsenu\x00\x01\x1f\x7f";

    /// Snippets a mutation inserts: escapes (whole, cut short, or half a
    /// surrogate pair), a multi-byte character, an overflowing number,
    /// separators and an empty key.
    const SNIPPETS: &[&str] = &[
        "\\u", "\\ud83d", "\\udc00", "\\u12g4", "é", "1e999", "-", "\"\":", ",,", "[{", "}]", "\\x",
    ];

    /// The `"..."` spans followed by `:`: the keys, as far as a quick scan
    /// can tell.
    fn keys(bytes: &[u8]) -> Vec<std::ops::Range<usize>> {
        let mut keys = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            if bytes[at] != b'"' {
                at += 1;
                continue;
            }
            let start = at;
            at += 1;
            while at < bytes.len() && bytes[at] != b'"' {
                at += if bytes[at] == b'\\' { 2 } else { 1 };
            }
            let end = (at + 1).min(bytes.len());
            let after = bytes[end..]
                .iter()
                .position(|byte| !byte.is_ascii_whitespace());
            if after.is_some_and(|skip| bytes[end + skip] == b':') {
                keys.push(start..end);
            }
            at = end;
        }
        keys
    }

    /// One to three edits of `seed`: a byte deleted, inserted, replaced,
    /// swapped, a slice repeated, the text cut short, a snippet inserted, or
    /// one key copied over another (a repeat, when both share an object).
    /// `None` when the edits leave invalid UTF-8.
    fn mutate(rng: &mut Rng, seed: &str) -> Option<String> {
        let mut bytes = seed.as_bytes().to_vec();
        for _ in 0..1 + rng.below(3) {
            let len = bytes.len();
            let at = rng.below(len + 1);
            match rng.below(8) {
                0 if at < len => {
                    bytes.remove(at);
                }
                1 => bytes.insert(at, BYTES[rng.below(BYTES.len())]),
                2 if at < len => bytes[at] = BYTES[rng.below(BYTES.len())],
                3 if at < len => {
                    let end = (at + 1 + rng.below(8)).min(len);
                    let slice = bytes[at..end].to_vec();
                    bytes.splice(at..at, slice);
                }
                4 => bytes.truncate(at),
                5 if at < len => {
                    let other = rng.below(len);
                    bytes.swap(at, other);
                }
                6 => {
                    let snippet = SNIPPETS[rng.below(SNIPPETS.len())];
                    bytes.splice(at..at, snippet.bytes());
                }
                7 => {
                    let keys = keys(&bytes);
                    if keys.len() >= 2 {
                        let from = keys[rng.below(keys.len())].clone();
                        let to = keys[rng.below(keys.len())].clone();
                        let key = bytes[from].to_vec();
                        bytes.splice(to, key);
                    }
                }
                _ => {}
            }
        }
        String::from_utf8(bytes).ok()
    }

    #[test]
    fn generated_and_mutated_texts_match_ordered_json() {
        let mut rng = Rng(0x5eed_d1d1);
        let (mut accepted, mut refused, mut repeats, mut ceilings) = (0, 0, 0, 0);
        let mut disagreements = Vec::new();
        let texts = SEEDS.iter().map(|seed| Some(seed.to_string())).chain(
            SEEDS
                .iter()
                .cycle()
                .take(SEEDS.len() * 4_000)
                .map(|seed| mutate(&mut rng, seed)),
        );
        for text in texts.flatten() {
            for limit in [LOTS, 6] {
                if let Some(disagreement) = disagreement(&text, limit) {
                    disagreements.push(disagreement);
                }
                match read(&text, limit) {
                    Ok(_) => accepted += 1,
                    Err(BodyError::Json {
                        kind: JsonErrorKind::DuplicateKey,
                        ..
                    }) => repeats += 1,
                    Err(BodyError::TooManyValues { .. }) => ceilings += 1,
                    Err(BodyError::Json { .. }) => refused += 1,
                }
            }
        }
        assert!(
            disagreements.is_empty(),
            "{} disagreements, the first:\n{}",
            disagreements.len(),
            disagreements[..disagreements.len().min(5)].join("\n")
        );
        eprintln!(
            "{accepted} accepted, {refused} refused, {repeats} repeated keys, {ceilings} value \
             ceilings"
        );
        // The mutations reach every outcome, often.
        for (outcome, count) in [
            ("accepted", accepted),
            ("refused", refused),
            ("repeated keys", repeats),
            ("value ceilings", ceilings),
        ] {
            assert!(count >= 500, "only {count} {outcome}");
        }
    }

    #[test]
    fn a_repeated_key_waits_for_the_rest_of_the_text() {
        // `OrderedJson::parse` reports a repeat when its object closes; this
        // module once the whole text is JSON, so a later syntax problem
        // comes first here. Among repeats, this module names the first in
        // the text, `OrderedJson::parse` one in the first object to close.
        let cases = [
            (
                r#"[{"a":1,"a":2},]"#,
                (JsonErrorKind::ExpectedValue, 15),
                (JsonErrorKind::DuplicateKey, 8),
            ),
            (
                r#"{"a":1,"a":2}x"#,
                (JsonErrorKind::TrailingCharacters, 13),
                (JsonErrorKind::DuplicateKey, 7),
            ),
            (
                r#"{"a":1,"a":2,"b":{"c":1,"c":2}}"#,
                (JsonErrorKind::DuplicateKey, 7),
                (JsonErrorKind::DuplicateKey, 24),
            ),
        ];
        for (text, (kind, offset), (core_kind, core_offset)) in cases {
            assert_eq!(
                read_text(text).err(),
                Some(BodyError::Json { kind, offset })
            );
            assert!(matches!(
                OrderedJson::parse(text),
                Err(DecisionError::InvalidJson { kind, offset })
                    if kind == core_kind && offset == core_offset
            ));
            assert_eq!(disagreement(text, LOTS), None);
        }
    }

    #[test]
    fn values_keep_their_spans_kinds_and_order() {
        let text = " {\"b\": [1, \"x\\n\", null], \"a\": {\"k\": true}, \"\\u0063\": -0.5} ";
        let value = read_text(text).unwrap();
        assert_eq!(&text[value.span.clone()], text.trim());
        let members = value.as_object().unwrap();
        let keys: Vec<&str> = members.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keys, ["b", "a", "c"]);

        let items = value.get("b").and_then(Value::as_array).unwrap();
        assert!(matches!(items[0].node, Node::Scalar));
        assert_eq!(items[1].as_str(), Some("x\n"));
        assert_eq!(&text[items[1].span.clone()], "\"x\\n\"");
        assert!(items[2].is_null());

        let inner = value.get("a").unwrap();
        assert_eq!(&text[inner.span.clone()], "{\"k\": true}");
        assert!(matches!(
            inner.get("k").map(|k| &k.node),
            Some(Node::Scalar)
        ));
        assert_eq!(&text[value.get("c").unwrap().span.clone()], "-0.5");
        assert!(value.get("missing").is_none());
    }

    #[test]
    fn containers_below_the_read_depth_are_located_not_kept() {
        // Arrays at depths 1 to 5, an object at 6 whose members are at 7.
        let text = "[[[[[{\"k\": [1, {\"x\": 2}], \"s\": \"v\"}]]]]]";
        let value = read_text(text).unwrap();
        let mut level = &value;
        for _ in 1..READ_DEPTH {
            level = &level.as_array().unwrap()[0];
        }
        // `level` is the object at depth 6: kept, with its members.
        assert!(level.as_object().is_some());
        let unread = level.get("k").unwrap();
        assert!(matches!(unread.node, Node::Unread));
        assert!(unread.as_array().is_none());
        assert_eq!(&text[unread.span.clone()], "[1, {\"x\": 2}]");
        // Members at depth 7 are kept as scalars.
        assert_eq!(level.get("s").and_then(Value::as_str), Some("v"));
    }

    #[test]
    fn repeated_keys_are_found_at_any_depth_once_the_text_is_json() {
        let deep = format!("{}{{\"k\":1,\"k\":2}}{}", "[".repeat(200), "]".repeat(200));
        let offset = deep.find("\"k\":2").unwrap();
        assert_eq!(
            read_text(&deep).err(),
            Some(BodyError::Json {
                kind: JsonErrorKind::DuplicateKey,
                offset
            })
        );
        // The first repeat in the text, though an inner object closes first.
        let text = "{\"a\":1,\"a\":2,\"b\":{\"c\":1,\"c\":2}}";
        assert_eq!(
            read_text(text).err(),
            Some(BodyError::Json {
                kind: JsonErrorKind::DuplicateKey,
                offset: 7
            })
        );
        // A grammar error anywhere wins over a repeated key before it.
        let text = "{\"a\":1,\"a\":2,\"b\":[1,]}";
        assert_eq!(
            read_text(text).err(),
            Some(BodyError::Json {
                kind: JsonErrorKind::ExpectedValue,
                offset: text.find(']').unwrap()
            })
        );
    }

    #[test]
    fn nesting_has_no_depth_limit_and_no_recursion() {
        // Far deeper than any recursive reader could go on a test thread.
        let depth = 200_000;
        let text = format!("{}{}", "[".repeat(depth), "]".repeat(depth));
        let value = read_text(&text).unwrap();
        assert_eq!(value.span, 0..text.len());
        let text = format!("{{\"a\":{}1{}}}", "[".repeat(depth), "]".repeat(depth));
        assert!(read_text(&text).is_ok());
        let text = format!("{}1{}", "[".repeat(depth), "]".repeat(depth - 1));
        assert_eq!(
            read_text(&text).err(),
            Some(BodyError::Json {
                kind: JsonErrorKind::UnexpectedEnd,
                offset: text.len()
            })
        );
    }

    #[test]
    fn the_value_ceiling_counts_every_value() {
        assert!(read("[1,2,3]", 4).is_ok());
        assert_eq!(
            read("[1,2,3,4]", 4).err(),
            Some(BodyError::TooManyValues { max: 4 })
        );
        // Keys do not count.
        assert!(read("{\"a\":1,\"b\":2}", 3).is_ok());
    }
}
