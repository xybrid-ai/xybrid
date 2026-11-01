//! Local stand-in for the contract-1 decision IR (P4 spike only).
//!
//! `xybrid_core::ir::decision` does not exist yet: P1 owns it, and its freeze
//! waits for this spike's report. This module mirrors the agreed public shape
//! (framework v2 §2 with the v2.1 corrections) closely enough that the facade
//! conversions in [`super`] are written against the real API:
//!
//! - sealed types: constructors and accessors only;
//! - [`OrderedJson`] is opaque, built only by [`OrderedJson::parse`] and read
//!   through [`JsonView`]; it keeps object order, refuses duplicate keys and
//!   non-finite numbers, and classifies numbers as v2.1 C3 says (an integer
//!   literal that fits `i64` is `I64`, else one that fits `u64` is `U64`, else
//!   `F64`; a fraction or exponent is `F64`);
//! - a JSON `null` at an optional field becomes `None`, so
//!   `Some(Json(null))` cannot be built;
//! - `Debug` prints counts, lengths and kinds, never ids, keys, text or JSON.
//!
//! Replace it with the real IR once P1 lands. Nothing outside the facade's
//! decision module may depend on it.

use std::collections::HashSet;
use std::fmt;

// ============================================================================
// Errors
// ============================================================================

/// Why a decision request or result was refused.
///
/// Messages name fields, indexes and counts only. `field` is a path the
/// facade builds from static field names and indexes (for example
/// `questions[2].options[30].description`), never from caller content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionError {
    /// A JSON text did not parse into an [`OrderedJson`].
    InvalidJson {
        field: String,
        offset: usize,
        reason: JsonErrorReason,
    },
    /// The request carries no question.
    NoQuestions,
    /// A question id is empty.
    EmptyQuestionId { question: usize },
    /// A question id repeats an earlier one.
    DuplicateQuestionId { question: usize, first: usize },
    /// A choice option key is empty.
    EmptyOptionKey { question: usize, option: usize },
    /// A choice option key repeats an earlier one in the same question.
    DuplicateOptionKey {
        question: usize,
        option: usize,
        first: usize,
    },
    /// A choice question offers no option.
    NoOptions { question: usize },
    /// A score question offers no level.
    NoLevels { question: usize },
    /// A count or size ceiling from [`StructuralLimits`] was exceeded.
    LimitExceeded {
        what: &'static str,
        count: usize,
        limit: usize,
    },
    /// A result broke a [`DecisionAnswer`] invariant.
    InvalidAnswer { answer: usize, reason: &'static str },
}

impl fmt::Display for DecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson {
                field,
                offset,
                reason,
            } => write!(f, "{field}: invalid JSON at byte {offset}: {reason}"),
            Self::NoQuestions => write!(f, "the request has no question"),
            Self::EmptyQuestionId { question } => write!(f, "questions[{question}].id is empty"),
            Self::DuplicateQuestionId { question, first } => {
                write!(f, "questions[{question}].id repeats questions[{first}].id")
            }
            Self::EmptyOptionKey { question, option } => {
                write!(f, "questions[{question}].options[{option}].key is empty")
            }
            Self::DuplicateOptionKey {
                question,
                option,
                first,
            } => write!(
                f,
                "questions[{question}].options[{option}].key repeats options[{first}].key"
            ),
            Self::NoOptions { question } => write!(f, "questions[{question}] has no option"),
            Self::NoLevels { question } => write!(f, "questions[{question}] has no level"),
            Self::LimitExceeded { what, count, limit } => {
                write!(f, "{what}: {count} exceeds the limit of {limit}")
            }
            Self::InvalidAnswer { answer, reason } => write!(f, "answers[{answer}]: {reason}"),
        }
    }
}

impl std::error::Error for DecisionError {}

/// Why a JSON text was refused. Content-free by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonErrorReason {
    UnexpectedEnd,
    ExpectedValue,
    ExpectedColon,
    ExpectedCommaOrEnd,
    ExpectedKey,
    InvalidNumber,
    NonFiniteNumber,
    InvalidEscape,
    LoneSurrogate,
    ControlCharacter,
    DuplicateKey,
    TooDeep,
    TooManyNodes,
    TooLarge,
    TrailingCharacters,
}

impl fmt::Display for JsonErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnexpectedEnd => "unexpected end of input",
            Self::ExpectedValue => "expected a value",
            Self::ExpectedColon => "expected ':'",
            Self::ExpectedCommaOrEnd => "expected ',' or a closing bracket",
            Self::ExpectedKey => "expected an object key",
            Self::InvalidNumber => "invalid number",
            Self::NonFiniteNumber => "number does not fit a finite float",
            Self::InvalidEscape => "invalid escape",
            Self::LoneSurrogate => "unpaired UTF-16 surrogate escape",
            Self::ControlCharacter => "unescaped control character in a string",
            Self::DuplicateKey => "duplicate object key",
            Self::TooDeep => "nesting exceeds the depth limit",
            Self::TooManyNodes => "too many values",
            Self::TooLarge => "text exceeds the size limit",
            Self::TrailingCharacters => "characters after the value",
        })
    }
}

/// A refused JSON text: where and why, never what.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonError {
    pub offset: usize,
    pub reason: JsonErrorReason,
}

// ============================================================================
// Structural limits
// ============================================================================

/// Ceilings [`validate_structure`] and [`OrderedJson::parse`] enforce.
#[derive(Clone, Debug)]
pub struct StructuralLimits {
    /// Upstream llama.cpp's nesting limit.
    pub max_json_depth: usize,
    pub max_json_bytes: usize,
    pub max_json_nodes: usize,
    pub max_questions: usize,
    pub max_options: usize,
    pub max_levels: usize,
}

impl Default for StructuralLimits {
    fn default() -> Self {
        Self {
            max_json_depth: 128,
            max_json_bytes: 1 << 20,
            max_json_nodes: 1 << 16,
            max_questions: 64,
            max_options: 1024,
            max_levels: 1024,
        }
    }
}

// ============================================================================
// OrderedJson
// ============================================================================

/// A JSON number with its integer/float identity kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JsonNumber {
    I64(i64),
    U64(u64),
    F64(f64),
}

#[derive(Clone, PartialEq)]
enum Node {
    Null,
    Bool(bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<Node>),
    Object(Vec<(String, Node)>),
}

/// Order-preserving, validated JSON value. Opaque: build it with
/// [`OrderedJson::parse`], read it with [`OrderedJson::view`].
#[derive(Clone, PartialEq)]
pub struct OrderedJson(Node);

/// Read-only view of one [`OrderedJson`] node.
#[derive(Clone, Copy)]
pub struct JsonView<'a>(&'a Node);

/// The kind of a JSON node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonKind {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
}

impl OrderedJson {
    /// Parse JSON text, keeping object order and number identity.
    ///
    /// # Errors
    /// Malformed text, a duplicate key, a number that overflows `f64`, or a
    /// depth, node or size ceiling.
    pub fn parse(text: &str, limits: &StructuralLimits) -> Result<Self, JsonError> {
        if text.len() > limits.max_json_bytes {
            return Err(JsonError {
                offset: limits.max_json_bytes,
                reason: JsonErrorReason::TooLarge,
            });
        }
        let mut parser = Parser {
            bytes: text.as_bytes(),
            pos: 0,
            nodes: 0,
            limits,
        };
        parser.skip_ws();
        let node = parser.value(1)?;
        parser.skip_ws();
        if parser.pos != parser.bytes.len() {
            return Err(parser.error(JsonErrorReason::TrailingCharacters));
        }
        Ok(Self(node))
    }

    /// The root node.
    pub fn view(&self) -> JsonView<'_> {
        JsonView(&self.0)
    }

    /// Compact JSON text: no whitespace, object order kept, integers as
    /// integers and floats with a `.` or exponent, so the text parses back to
    /// an equal value.
    pub fn to_json_text(&self) -> String {
        let mut out = String::new();
        write_node(&self.0, &mut out);
        out
    }

    fn is_null(&self) -> bool {
        matches!(self.0, Node::Null)
    }

    fn into_string(self) -> Result<String, Self> {
        match self.0 {
            Node::String(text) => Ok(text),
            other => Err(Self(other)),
        }
    }

    fn depth_and_nodes(&self) -> (usize, usize) {
        fn walk(node: &Node, depth: usize, max: &mut usize, count: &mut usize) {
            *max = (*max).max(depth);
            *count += 1;
            match node {
                Node::Array(items) => items.iter().for_each(|n| walk(n, depth + 1, max, count)),
                Node::Object(entries) => entries
                    .iter()
                    .for_each(|(_, n)| walk(n, depth + 1, max, count)),
                _ => {}
            }
        }
        let (mut max, mut count) = (0, 0);
        walk(&self.0, 1, &mut max, &mut count);
        (max, count)
    }
}

impl fmt::Debug for OrderedJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (depth, nodes) = self.depth_and_nodes();
        f.debug_struct("OrderedJson")
            .field("kind", &self.view().kind())
            .field("depth", &depth)
            .field("nodes", &nodes)
            .finish()
    }
}

impl<'a> JsonView<'a> {
    pub fn kind(self) -> JsonKind {
        match self.0 {
            Node::Null => JsonKind::Null,
            Node::Bool(_) => JsonKind::Bool,
            Node::Number(_) => JsonKind::Number,
            Node::String(_) => JsonKind::String,
            Node::Array(_) => JsonKind::Array,
            Node::Object(_) => JsonKind::Object,
        }
    }

    pub fn as_bool(self) -> Option<bool> {
        match self.0 {
            Node::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_number(self) -> Option<JsonNumber> {
        match self.0 {
            Node::Number(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_str(self) -> Option<&'a str> {
        match self.0 {
            Node::String(value) => Some(value),
            _ => None,
        }
    }

    /// Array items in order, or `None` for a non-array.
    pub fn items(self) -> Option<Vec<JsonView<'a>>> {
        match self.0 {
            Node::Array(items) => Some(items.iter().map(JsonView).collect()),
            _ => None,
        }
    }

    /// Object entries in document order, or `None` for a non-object.
    pub fn entries(self) -> Option<Vec<(&'a str, JsonView<'a>)>> {
        match self.0 {
            Node::Object(entries) => Some(
                entries
                    .iter()
                    .map(|(key, node)| (key.as_str(), JsonView(node)))
                    .collect(),
            ),
            _ => None,
        }
    }
}

struct Parser<'t> {
    bytes: &'t [u8],
    pos: usize,
    nodes: usize,
    limits: &'t StructuralLimits,
}

impl Parser<'_> {
    fn error(&self, reason: JsonErrorReason) -> JsonError {
        JsonError {
            offset: self.pos,
            reason,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.bytes.get(self.pos) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Result<u8, JsonError> {
        self.bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| self.error(JsonErrorReason::UnexpectedEnd))
    }

    fn literal(&mut self, word: &[u8], node: Node) -> Result<Node, JsonError> {
        if self.bytes[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(node)
        } else {
            Err(self.error(JsonErrorReason::ExpectedValue))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Node, JsonError> {
        if depth > self.limits.max_json_depth {
            return Err(self.error(JsonErrorReason::TooDeep));
        }
        self.nodes += 1;
        if self.nodes > self.limits.max_json_nodes {
            return Err(self.error(JsonErrorReason::TooManyNodes));
        }
        match self.peek()? {
            b'n' => self.literal(b"null", Node::Null),
            b't' => self.literal(b"true", Node::Bool(true)),
            b'f' => self.literal(b"false", Node::Bool(false)),
            b'"' => self.string().map(Node::String),
            b'[' => self.array(depth),
            b'{' => self.object(depth),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.error(JsonErrorReason::ExpectedValue)),
        }
    }

    fn array(&mut self, depth: usize) -> Result<Node, JsonError> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek()? == b']' {
            self.pos += 1;
            return Ok(Node::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek()? {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    return Ok(Node::Array(items));
                }
                _ => return Err(self.error(JsonErrorReason::ExpectedCommaOrEnd)),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Node, JsonError> {
        self.pos += 1;
        let mut entries: Vec<(String, Node)> = Vec::new();
        let mut seen = HashSet::new();
        self.skip_ws();
        if self.peek()? == b'}' {
            self.pos += 1;
            return Ok(Node::Object(entries));
        }
        loop {
            self.skip_ws();
            if self.peek()? != b'"' {
                return Err(self.error(JsonErrorReason::ExpectedKey));
            }
            let key_offset = self.pos;
            let key = self.string()?;
            if !seen.insert(key.clone()) {
                return Err(JsonError {
                    offset: key_offset,
                    reason: JsonErrorReason::DuplicateKey,
                });
            }
            self.skip_ws();
            if self.peek()? != b':' {
                return Err(self.error(JsonErrorReason::ExpectedColon));
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.skip_ws();
            match self.peek()? {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    return Ok(Node::Object(entries));
                }
                _ => return Err(self.error(JsonErrorReason::ExpectedCommaOrEnd)),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let digits = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.error(JsonErrorReason::UnexpectedEnd))?;
        let text =
            std::str::from_utf8(digits).map_err(|_| self.error(JsonErrorReason::InvalidEscape))?;
        let value = u32::from_str_radix(text, 16)
            .map_err(|_| self.error(JsonErrorReason::InvalidEscape))?;
        self.pos += 4;
        Ok(value)
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let start = self.pos;
            while let Some(&byte) = self.bytes.get(self.pos) {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            // The input is a `&str` and the run stops only at ASCII bytes, so
            // the slice is valid UTF-8.
            out.push_str(
                std::str::from_utf8(&self.bytes[start..self.pos])
                    .expect("a run between ASCII delimiters of a &str is valid UTF-8"),
            );
            match self.peek()? {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let escape = self.peek()?;
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let high = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&high) {
                                if !self.bytes[self.pos..].starts_with(b"\\u") {
                                    return Err(self.error(JsonErrorReason::LoneSurrogate));
                                }
                                self.pos += 2;
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(self.error(JsonErrorReason::LoneSurrogate));
                                }
                                0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&high) {
                                return Err(self.error(JsonErrorReason::LoneSurrogate));
                            } else {
                                high
                            };
                            out.push(
                                char::from_u32(code)
                                    .ok_or_else(|| self.error(JsonErrorReason::InvalidEscape))?,
                            );
                        }
                        _ => return Err(self.error(JsonErrorReason::InvalidEscape)),
                    }
                }
                _ => return Err(self.error(JsonErrorReason::ControlCharacter)),
            }
        }
    }

    fn number(&mut self) -> Result<Node, JsonError> {
        let start = self.pos;
        let digits = |parser: &mut Self| {
            let from = parser.pos;
            while let Some(b'0'..=b'9') = parser.bytes.get(parser.pos) {
                parser.pos += 1;
            }
            parser.pos - from
        };
        if self.bytes[self.pos] == b'-' {
            self.pos += 1;
        }
        let int_start = self.pos;
        let int_digits = digits(self);
        if int_digits == 0 || (int_digits > 1 && self.bytes[int_start] == b'0') {
            return Err(self.error(JsonErrorReason::InvalidNumber));
        }
        let mut is_float = false;
        if self.bytes.get(self.pos) == Some(&b'.') {
            self.pos += 1;
            is_float = true;
            if digits(self) == 0 {
                return Err(self.error(JsonErrorReason::InvalidNumber));
            }
        }
        if let Some(b'e' | b'E') = self.bytes.get(self.pos) {
            self.pos += 1;
            is_float = true;
            if let Some(b'+' | b'-') = self.bytes.get(self.pos) {
                self.pos += 1;
            }
            if digits(self) == 0 {
                return Err(self.error(JsonErrorReason::InvalidNumber));
            }
        }
        let text =
            std::str::from_utf8(&self.bytes[start..self.pos]).expect("a JSON number is ASCII");
        if !is_float {
            // v2.1 C3: fits i64 -> I64, else fits u64 -> U64, else F64.
            if let Ok(value) = text.parse::<i64>() {
                return Ok(Node::Number(JsonNumber::I64(value)));
            }
            if let Ok(value) = text.parse::<u64>() {
                return Ok(Node::Number(JsonNumber::U64(value)));
            }
        }
        let value: f64 = text
            .parse()
            .map_err(|_| self.error(JsonErrorReason::InvalidNumber))?;
        if !value.is_finite() {
            return Err(JsonError {
                offset: start,
                reason: JsonErrorReason::NonFiniteNumber,
            });
        }
        Ok(Node::Number(JsonNumber::F64(value)))
    }
}

fn write_node(node: &Node, out: &mut String) {
    use std::fmt::Write as _;
    match node {
        Node::Null => out.push_str("null"),
        Node::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Node::Number(JsonNumber::I64(value)) => write!(out, "{value}").expect("write to String"),
        Node::Number(JsonNumber::U64(value)) => write!(out, "{value}").expect("write to String"),
        // `{:?}` prints the shortest round-trip form and always a `.` or an
        // exponent (`1.0`, `1e300`, `-0.0`), so the text parses back as F64.
        Node::Number(JsonNumber::F64(value)) => write!(out, "{value:?}").expect("write to String"),
        Node::String(text) => write_string(text, out),
        Node::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_node(item, out);
            }
            out.push(']');
        }
        Node::Object(entries) => {
            out.push('{');
            for (index, (key, value)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_node(value, out);
            }
            out.push('}');
        }
    }
}

fn write_string(text: &str, out: &mut String) {
    use std::fmt::Write as _;
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ch if (ch as u32) < 0x20 => {
                write!(out, "\\u{:04x}", ch as u32).expect("write to String")
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

// ============================================================================
// Request
// ============================================================================

#[derive(Clone, PartialEq)]
enum ContentRepr {
    Text(String),
    /// Never a JSON null or string: those become `None` and `Text`.
    Json(OrderedJson),
}

/// Text or a non-string, non-null JSON value.
#[derive(Clone, PartialEq)]
pub struct DecisionContent(ContentRepr);

/// Read-only view of a [`DecisionContent`].
#[derive(Clone, Copy)]
pub enum DecisionContentView<'a> {
    Text(&'a str),
    Json(&'a OrderedJson),
}

impl DecisionContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self(ContentRepr::Text(text.into()))
    }

    /// Normalize a JSON value at an optional field: `null` is absent, a string
    /// is text, anything else stays JSON.
    pub fn from_json(value: OrderedJson) -> Option<Self> {
        if value.is_null() {
            return None;
        }
        Some(Self(match value.into_string() {
            Ok(text) => ContentRepr::Text(text),
            Err(json) => ContentRepr::Json(json),
        }))
    }

    pub fn view(&self) -> DecisionContentView<'_> {
        match &self.0 {
            ContentRepr::Text(text) => DecisionContentView::Text(text),
            ContentRepr::Json(json) => DecisionContentView::Json(json),
        }
    }
}

impl fmt::Debug for DecisionContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ContentRepr::Text(text) => f.debug_struct("Text").field("bytes", &text.len()).finish(),
            ContentRepr::Json(json) => json.fmt(f),
        }
    }
}

/// Reserved for media state (O2). Uninhabited: v1 has no media.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionMedia {}

/// What the questions are asked about.
#[derive(Clone, PartialEq)]
pub struct DecisionState {
    body: Option<DecisionContent>,
    media: Vec<DecisionMedia>,
}

impl DecisionState {
    /// `None` is a null state; `Some(text(""))` is an empty one.
    pub fn new(body: Option<DecisionContent>) -> Self {
        Self {
            body,
            media: Vec::new(),
        }
    }

    pub fn body(&self) -> Option<&DecisionContent> {
        self.body.as_ref()
    }

    pub fn media(&self) -> &[DecisionMedia] {
        &self.media
    }
}

impl fmt::Debug for DecisionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionState")
            .field("body", &self.body)
            .field("media", &self.media.len())
            .finish()
    }
}

/// How a question is put to the model.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionInstructions {
    Content(DecisionContent),
}

/// One choice option. The key may affect inference.
#[derive(Clone, PartialEq)]
pub struct DecisionOption {
    key: String,
    description: Option<DecisionContent>,
}

impl DecisionOption {
    pub fn new(key: impl Into<String>, description: Option<DecisionContent>) -> Self {
        Self {
            key: key.into(),
            description,
        }
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn description(&self) -> Option<&DecisionContent> {
        self.description.as_ref()
    }
}

impl fmt::Debug for DecisionOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionOption")
            .field("key_bytes", &self.key.len())
            .field("description", &self.description)
            .finish()
    }
}

/// The question types.
#[non_exhaustive]
#[derive(Clone, PartialEq)]
pub enum DecisionQuestionKind {
    Choice {
        options: Vec<DecisionOption>,
    },
    YesNo {
        yes: Option<DecisionContent>,
        no: Option<DecisionContent>,
    },
    /// Level `i` has key `"i"`; `None` renders as JSON null.
    Score {
        levels: Vec<Option<DecisionContent>>,
    },
}

impl fmt::Debug for DecisionQuestionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Choice { options } => f
                .debug_struct("Choice")
                .field("options", &options.len())
                .finish(),
            Self::YesNo { yes, no } => f
                .debug_struct("YesNo")
                .field("yes", yes)
                .field("no", no)
                .finish(),
            Self::Score { levels } => f
                .debug_struct("Score")
                .field("levels", &levels.len())
                .finish(),
        }
    }
}

/// One question.
#[derive(Clone, PartialEq)]
pub struct DecisionQuestion {
    id: String,
    instructions: Option<DecisionInstructions>,
    kind: DecisionQuestionKind,
}

impl DecisionQuestion {
    pub fn new(
        id: impl Into<String>,
        instructions: Option<DecisionInstructions>,
        kind: DecisionQuestionKind,
    ) -> Self {
        Self {
            id: id.into(),
            instructions,
            kind,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn instructions(&self) -> Option<&DecisionInstructions> {
        self.instructions.as_ref()
    }

    pub fn kind(&self) -> &DecisionQuestionKind {
        &self.kind
    }
}

impl fmt::Debug for DecisionQuestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionQuestion")
            .field("id_bytes", &self.id.len())
            .field("instructions", &self.instructions)
            .field("kind", &self.kind)
            .finish()
    }
}

/// A state and the questions asked about it, in order.
#[derive(Clone, PartialEq)]
pub struct DecisionRequest {
    state: DecisionState,
    questions: Vec<DecisionQuestion>,
}

impl DecisionRequest {
    pub fn new(state: DecisionState, questions: Vec<DecisionQuestion>) -> Self {
        Self { state, questions }
    }

    pub fn state(&self) -> &DecisionState {
        &self.state
    }

    pub fn questions(&self) -> &[DecisionQuestion] {
        &self.questions
    }
}

impl fmt::Debug for DecisionRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionRequest")
            .field("state", &self.state)
            .field("questions", &self.questions)
            .finish()
    }
}

/// Structure checks that hold for every protocol (contract 1).
///
/// # Errors
/// The first [`DecisionError`] found, in request order.
pub fn validate_structure(
    request: &DecisionRequest,
    limits: &StructuralLimits,
) -> Result<(), DecisionError> {
    let questions = request.questions();
    if questions.is_empty() {
        return Err(DecisionError::NoQuestions);
    }
    if questions.len() > limits.max_questions {
        return Err(DecisionError::LimitExceeded {
            what: "questions",
            count: questions.len(),
            limit: limits.max_questions,
        });
    }
    let mut ids = std::collections::HashMap::new();
    for (index, question) in questions.iter().enumerate() {
        if question.id().is_empty() {
            return Err(DecisionError::EmptyQuestionId { question: index });
        }
        if let Some(first) = ids.insert(question.id(), index) {
            return Err(DecisionError::DuplicateQuestionId {
                question: index,
                first,
            });
        }
        match question.kind() {
            DecisionQuestionKind::Choice { options } => {
                if options.is_empty() {
                    return Err(DecisionError::NoOptions { question: index });
                }
                if options.len() > limits.max_options {
                    return Err(DecisionError::LimitExceeded {
                        what: "options",
                        count: options.len(),
                        limit: limits.max_options,
                    });
                }
                let mut keys = std::collections::HashMap::new();
                for (option_index, option) in options.iter().enumerate() {
                    if option.key().is_empty() {
                        return Err(DecisionError::EmptyOptionKey {
                            question: index,
                            option: option_index,
                        });
                    }
                    if let Some(first) = keys.insert(option.key(), option_index) {
                        return Err(DecisionError::DuplicateOptionKey {
                            question: index,
                            option: option_index,
                            first,
                        });
                    }
                }
            }
            DecisionQuestionKind::YesNo { .. } => {}
            DecisionQuestionKind::Score { levels } => {
                if levels.is_empty() {
                    return Err(DecisionError::NoLevels { question: index });
                }
                if levels.len() > limits.max_levels {
                    return Err(DecisionError::LimitExceeded {
                        what: "levels",
                        count: levels.len(),
                        limit: limits.max_levels,
                    });
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// Result
// ============================================================================

/// Where a scored key came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionOrigin {
    Caller,
    Fixed,
}

/// Which statistic a score answer reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreStatistic {
    Median,
    Expected,
}

/// One key of an answer's distribution.
#[derive(Clone, PartialEq)]
pub struct DecisionScore {
    key: String,
    raw_score: f32,
    probability: f64,
    origin: DecisionOrigin,
}

impl DecisionScore {
    pub fn new(
        key: impl Into<String>,
        raw_score: f32,
        probability: f64,
        origin: DecisionOrigin,
    ) -> Self {
        Self {
            key: key.into(),
            raw_score,
            probability,
            origin,
        }
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    /// The model's score; `-inf` is valid when another key is finite.
    pub fn raw_score(&self) -> f32 {
        self.raw_score
    }

    pub fn probability(&self) -> f64 {
        self.probability
    }

    pub fn origin(&self) -> DecisionOrigin {
        self.origin
    }
}

impl fmt::Debug for DecisionScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionScore")
            .field("key_bytes", &self.key.len())
            .field("raw_score", &self.raw_score)
            .field("probability", &self.probability)
            .field("origin", &self.origin)
            .finish()
    }
}

/// The answer's value.
#[non_exhaustive]
#[derive(Clone, PartialEq)]
pub enum DecisionValue {
    Choice {
        key: String,
    },
    YesNo {
        p_yes: f64,
    },
    Score {
        value: f64,
        statistic: ScoreStatistic,
    },
}

impl fmt::Debug for DecisionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Choice { key } => f
                .debug_struct("Choice")
                .field("key_bytes", &key.len())
                .finish(),
            Self::YesNo { p_yes } => f.debug_struct("YesNo").field("p_yes", p_yes).finish(),
            Self::Score { value, statistic } => f
                .debug_struct("Score")
                .field("value", value)
                .field("statistic", statistic)
                .finish(),
        }
    }
}

/// One question's answer.
#[derive(Clone, PartialEq)]
pub struct DecisionAnswer {
    question_id: String,
    distribution: Vec<DecisionScore>,
    value: DecisionValue,
    confidence: Option<f64>,
}

fn is_probability(p: f64) -> bool {
    p.is_finite() && (0.0..=1.0).contains(&p)
}

impl DecisionAnswer {
    /// Build an answer, checking what every protocol guarantees: a non-empty
    /// distribution with unique keys, probabilities in `[0, 1]`, no NaN or
    /// `+inf` raw score, a choice value naming a distribution key.
    ///
    /// # Errors
    /// [`DecisionError::InvalidAnswer`] naming the broken invariant; `answer`
    /// is the caller's index.
    pub fn new(
        answer: usize,
        question_id: impl Into<String>,
        distribution: Vec<DecisionScore>,
        value: DecisionValue,
        confidence: Option<f64>,
    ) -> Result<Self, DecisionError> {
        let invalid = |reason| DecisionError::InvalidAnswer { answer, reason };
        if distribution.is_empty() {
            return Err(invalid("the distribution is empty"));
        }
        let mut keys = HashSet::new();
        for score in &distribution {
            if !keys.insert(score.key.as_str()) {
                return Err(invalid("a distribution key repeats"));
            }
            if score.raw_score.is_nan() || score.raw_score == f32::INFINITY {
                return Err(invalid("a raw score is NaN or +inf"));
            }
            if !is_probability(score.probability) {
                return Err(invalid("a probability is outside [0, 1]"));
            }
        }
        if distribution
            .iter()
            .all(|s| s.raw_score == f32::NEG_INFINITY)
        {
            return Err(invalid("every raw score is -inf"));
        }
        match &value {
            DecisionValue::Choice { key } if !keys.contains(key.as_str()) => {
                return Err(invalid("the choice is not a distribution key"))
            }
            DecisionValue::YesNo { p_yes } if !is_probability(*p_yes) => {
                return Err(invalid("p_yes is outside [0, 1]"))
            }
            DecisionValue::Score { value, .. } if !value.is_finite() => {
                return Err(invalid("the score is not finite"))
            }
            _ => {}
        }
        if confidence.is_some_and(|c| !is_probability(c)) {
            return Err(invalid("the confidence is outside [0, 1]"));
        }
        Ok(Self {
            question_id: question_id.into(),
            distribution,
            value,
            confidence,
        })
    }

    pub fn question_id(&self) -> &str {
        &self.question_id
    }

    pub fn distribution(&self) -> &[DecisionScore] {
        &self.distribution
    }

    pub fn value(&self) -> &DecisionValue {
        &self.value
    }

    pub fn confidence(&self) -> Option<f64> {
        self.confidence
    }
}

impl fmt::Debug for DecisionAnswer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionAnswer")
            .field("question_id_bytes", &self.question_id.len())
            .field("distribution", &self.distribution.len())
            .field("value", &self.value)
            .field("confidence", &self.confidence)
            .finish()
    }
}

/// Content-free work counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionUsage {
    passes: u32,
    evaluated_tokens: u64,
}

impl DecisionUsage {
    pub fn new(passes: u32, evaluated_tokens: u64) -> Self {
        Self {
            passes,
            evaluated_tokens,
        }
    }

    pub fn passes(&self) -> u32 {
        self.passes
    }

    pub fn evaluated_tokens(&self) -> u64 {
        self.evaluated_tokens
    }
}

/// Every answer, in request order, and the work it took.
#[derive(Clone, PartialEq)]
pub struct Decisions {
    answers: Vec<DecisionAnswer>,
    usage: DecisionUsage,
}

impl Decisions {
    pub fn new(answers: Vec<DecisionAnswer>, usage: DecisionUsage) -> Self {
        Self { answers, usage }
    }

    pub fn answers(&self) -> &[DecisionAnswer] {
        &self.answers
    }

    pub fn usage(&self) -> DecisionUsage {
        self.usage
    }
}

impl fmt::Debug for Decisions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Decisions")
            .field("answers", &self.answers)
            .field("usage", &self.usage)
            .finish()
    }
}

// ============================================================================
// Stub evaluation (stands in for P5's prepare/finish)
// ============================================================================

/// Deterministic fake decision run, standing in for P5/P7 until they exist.
///
/// Raw scores come from a fixed formula over option positions, never from
/// content. Probabilities follow the d1 rule (`exp((s - max) / T)` in option
/// order, normalized in f64), so the result carries realistic float64 values
/// for the FFI round trip. A score question's first level scores `-inf` to
/// exercise the C4 wire form.
pub fn stub_decide(request: &DecisionRequest) -> Decisions {
    const TEMPERATURE: f64 = 1.337_398_051_738_739_f64;
    let distribution = |scores: Vec<(String, f32)>| -> (Vec<DecisionScore>, Vec<f64>) {
        let max = scores
            .iter()
            .map(|(_, s)| f64::from(*s))
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores
            .iter()
            .map(|(_, s)| ((f64::from(*s) - max) / TEMPERATURE).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let probabilities: Vec<f64> = weights.iter().map(|w| w / total).collect();
        let out = scores
            .into_iter()
            .zip(&probabilities)
            .map(|((key, raw), p)| DecisionScore::new(key, raw, *p, DecisionOrigin::Caller))
            .collect();
        (out, probabilities)
    };
    let raw = |position: usize| -> f32 { ((position * 7919) % 23) as f32 * 0.37 - 3.0 };

    let mut answers = Vec::new();
    let mut evaluated_tokens = 0u64;
    for (index, question) in request.questions().iter().enumerate() {
        let answer = match question.kind() {
            DecisionQuestionKind::Choice { options } => {
                let (scores, probabilities) = distribution(
                    options
                        .iter()
                        .enumerate()
                        .map(|(i, o)| (o.key().to_string(), raw(i + index)))
                        .collect(),
                );
                let (best, confidence) = probabilities.iter().enumerate().fold(
                    (0, f64::NEG_INFINITY),
                    |(bi, bp), (i, p)| if *p > bp { (i, *p) } else { (bi, bp) },
                );
                let key = scores[best].key().to_string();
                DecisionAnswer::new(
                    index,
                    question.id(),
                    scores,
                    DecisionValue::Choice { key },
                    Some(confidence),
                )
            }
            DecisionQuestionKind::YesNo { .. } => {
                let (scores, probabilities) = distribution(vec![
                    ("true".to_string(), raw(index + 1)),
                    ("false".to_string(), raw(index + 2)),
                ]);
                DecisionAnswer::new(
                    index,
                    question.id(),
                    scores,
                    DecisionValue::YesNo {
                        p_yes: probabilities[0],
                    },
                    None,
                )
            }
            DecisionQuestionKind::Score { levels } => {
                let (scores, probabilities) = distribution(
                    (0..levels.len())
                        .map(|i| {
                            let s = if i == 0 && levels.len() > 1 {
                                f32::NEG_INFINITY
                            } else {
                                raw(i + index)
                            };
                            (i.to_string(), s)
                        })
                        .collect(),
                );
                let mut cumulative = 0.0;
                let median = probabilities
                    .iter()
                    .position(|p| {
                        cumulative += p;
                        cumulative >= 0.5
                    })
                    .unwrap_or(probabilities.len() - 1);
                let confidence = probabilities.iter().copied().fold(0.0, f64::max);
                DecisionAnswer::new(
                    index,
                    question.id(),
                    scores,
                    DecisionValue::Score {
                        value: median as f64,
                        statistic: ScoreStatistic::Median,
                    },
                    Some(confidence),
                )
            }
        };
        evaluated_tokens += 1000 + 17 * index as u64;
        answers.push(answer.expect("stub_decide builds answers that satisfy every invariant"));
    }
    let passes = u32::try_from(answers.len()).expect("question count is bounded by the limits");
    Decisions::new(answers, DecisionUsage::new(passes, evaluated_tokens))
}
