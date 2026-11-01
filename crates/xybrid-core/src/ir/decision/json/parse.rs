//! JSON text to [`OrderedJson`], written by hand.
//!
//! serde_json's own parser cannot give what decision content needs, even
//! through a visitor that keeps member order:
//!
//! - it reads `-0` as the float `-0.0`; nlohmann::json, which llama.cpp
//!   parses requests with, reads the integer 0;
//! - its default float parsing is best-effort, while `f64::from_str` (used
//!   here) is correctly rounded like nlohmann's `strtod`;
//! - it refuses the leading byte-order mark nlohmann skips, and its
//!   recursion limit counts nesting differently from llama.cpp's
//!   [`MAX_JSON_DEPTH`];
//! - its behavior moves with features any crate in the build can turn on
//!   (`arbitrary_precision`, `float_roundtrip`, `unbounded_depth`).
//!
//! The parser keeps the containers it is filling on its own heap stack
//! rather than recursing, so its native stack use does not grow with
//! nesting: a debug build on a 512 KiB thread (an iOS secondary thread, a
//! GCD queue) parses and refuses deep text alike.

use super::{JsonLimits, JsonNumber, Node, OrderedJson, MAX_JSON_DEPTH};
use crate::ir::decision::{DecisionError, JsonErrorKind};
use std::collections::HashSet;

/// Parses one JSON text under `limits`.
pub(super) fn parse(text: &str, limits: &JsonLimits) -> Result<OrderedJson, DecisionError> {
    if text.len() > limits.max_bytes {
        return Err(DecisionError::JsonTooLong {
            len: text.len(),
            max: limits.max_bytes,
        });
    }
    let mut parser = Parser {
        text,
        bytes: text.as_bytes(),
        pos: 0,
        nodes: 0,
        max_nodes: limits.max_nodes,
    };
    // nlohmann::json skips a leading UTF-8 byte-order mark, so a text
    // llama.cpp accepts is accepted here too.
    if text.starts_with('\u{feff}') {
        parser.pos = '\u{feff}'.len_utf8();
    }
    parser.skip_whitespace();
    let value = parser.document()?;
    parser.skip_whitespace();
    if parser.pos < parser.bytes.len() {
        return Err(parser.error(JsonErrorKind::TrailingCharacters));
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    nodes: usize,
    max_nodes: usize,
}

/// How a value starts: whole (a scalar), or as a container to fill.
enum Start {
    Scalar(OrderedJson),
    Array,
    Object,
}

/// A container being filled, innermost last on the parser's stack.
enum Open {
    Array(Vec<OrderedJson>),
    Object {
        members: Vec<(String, OrderedJson)>,
        /// Byte offset of each member's key, for a repeated-key error.
        key_offsets: Vec<usize>,
        /// The key of the member whose value is being parsed.
        key: String,
        key_offset: usize,
    },
}

impl Parser<'_> {
    fn error(&self, kind: JsonErrorKind) -> DecisionError {
        self.error_at(kind, self.pos)
    }

    fn error_at(&self, kind: JsonErrorKind, offset: usize) -> DecisionError {
        DecisionError::InvalidJson { kind, offset }
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

    /// The document's value. Containers being filled wait on `open`; a
    /// value is at depth `open.len() + 1`.
    fn document(&mut self) -> Result<OrderedJson, DecisionError> {
        let mut open: Vec<Open> = Vec::new();
        loop {
            let mut value = match self.start(open.len() + 1)? {
                Start::Scalar(value) => value,
                Start::Array if self.peek() == Some(b']') => {
                    self.pos += 1;
                    OrderedJson(Node::Array(Vec::new()))
                }
                Start::Array => {
                    open.push(Open::Array(Vec::new()));
                    continue;
                }
                Start::Object if self.peek() == Some(b'}') => {
                    self.pos += 1;
                    OrderedJson(Node::Object(Vec::new()))
                }
                Start::Object => {
                    let (key, key_offset) = self.member_key()?;
                    open.push(Open::Object {
                        members: Vec::new(),
                        key_offsets: Vec::new(),
                        key,
                        key_offset,
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
                    Open::Array(mut items) => {
                        items.push(value);
                        if self.separator(b']')? {
                            open.push(Open::Array(items));
                            break;
                        }
                        value = OrderedJson(Node::Array(items));
                    }
                    Open::Object {
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
                                members,
                                key_offsets,
                                key,
                                key_offset,
                            });
                            break;
                        }
                        if let Some(index) = first_repeated_key(&members) {
                            return Err(
                                self.error_at(JsonErrorKind::DuplicateKey, key_offsets[index])
                            );
                        }
                        value = OrderedJson(Node::Object(members));
                    }
                }
            }
        }
    }

    /// Starts the value at `depth`: a scalar is read whole, an array or an
    /// object is opened (its bracket and the whitespace after it read).
    fn start(&mut self, depth: usize) -> Result<Start, DecisionError> {
        match self.peek() {
            None => return Err(self.error(JsonErrorKind::UnexpectedEnd)),
            Some(b'n' | b't' | b'f' | b'"' | b'[' | b'{' | b'-' | b'0'..=b'9') => {}
            Some(_) => return Err(self.error(JsonErrorKind::ExpectedValue)),
        }
        if depth > MAX_JSON_DEPTH {
            return Err(DecisionError::JsonTooDeep {
                max: MAX_JSON_DEPTH,
            });
        }
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            return Err(DecisionError::JsonTooManyNodes {
                max: self.max_nodes,
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
                Node::Bool(true)
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Node::Bool(false)
            }
            Some(b'"') => Node::String(self.string()?),
            _ => Node::Number(self.number()?),
        };
        Ok(Start::Scalar(OrderedJson(node)))
    }

    fn literal(&mut self, word: &[u8]) -> Result<(), DecisionError> {
        if self.bytes[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(())
        } else {
            Err(self.error(JsonErrorKind::InvalidLiteral))
        }
    }

    /// After an item or member: `,` (true) or the closing bracket (false).
    fn separator(&mut self, close: u8) -> Result<bool, DecisionError> {
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
    fn member_key(&mut self) -> Result<(String, usize), DecisionError> {
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

    /// A string starting at the opening quote, unescaped.
    fn string(&mut self) -> Result<String, DecisionError> {
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
    fn escape(&mut self, out: &mut String) -> Result<(), DecisionError> {
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
    fn unicode_escape(&mut self, start: usize, out: &mut String) -> Result<(), DecisionError> {
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
    fn hex4(&mut self, start: usize) -> Result<u32, DecisionError> {
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

    /// A number, classified as nlohmann::json classifies it.
    fn number(&mut self) -> Result<JsonNumber, DecisionError> {
        let start = self.pos;
        let mut integer = true;
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
            integer = false;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error_at(JsonErrorKind::InvalidNumber, start));
            }
            self.skip_digits();
        }
        if let Some(b'e' | b'E') = self.peek() {
            integer = false;
            self.pos += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error_at(JsonErrorKind::InvalidNumber, start));
            }
            self.skip_digits();
        }
        let literal = &self.text[start..self.pos];
        if integer {
            // nlohmann tries strtoll/strtoull first and falls back to a
            // float on overflow; `-0` parses as the integer 0 there too.
            if let Ok(value) = literal.parse::<i64>() {
                return Ok(JsonNumber::I64(value));
            }
            if let Ok(value) = literal.parse::<u64>() {
                return Ok(JsonNumber::U64(value));
            }
        }
        match literal.parse::<f64>() {
            Ok(value) if value.is_finite() => Ok(JsonNumber::F64(value)),
            Ok(_) => Err(self.error_at(JsonErrorKind::NumberOutOfRange, start)),
            // Unreachable for a literal that passed the grammar above.
            Err(_) => Err(self.error_at(JsonErrorKind::InvalidNumber, start)),
        }
    }
}

/// Index of the first member whose key an earlier member already used.
fn first_repeated_key(members: &[(String, OrderedJson)]) -> Option<usize> {
    if members.len() < 2 {
        return None;
    }
    let mut seen = HashSet::with_capacity(members.len());
    members
        .iter()
        .position(|(key, _)| !seen.insert(key.as_str()))
}
