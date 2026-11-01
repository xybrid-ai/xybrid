//! [`OrderedJson`]: JSON content that keeps key order and number identity.
//!
//! A decision model's prompt renders the caller's JSON as it was written:
//! object keys in document order, `1` distinct from `1.0`. `serde_json::Value`
//! keeps neither (its maps are sorted unless a crate-wide feature says
//! otherwise, and it reads `-0` as a float), so decision content is parsed
//! straight from text into [`OrderedJson`] (`parse.rs`), and nothing else can
//! build one. `write.rs` turns a value back into compact JSON text.

mod parse;
mod write;

#[cfg(test)]
mod tests;

use super::DecisionError;
use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// Deepest a JSON value may nest, counting the value itself.
///
/// A scalar, `[]` and `{}` have depth 1, `[[]]` depth 2 and `{"a": [1]}`
/// depth 3. llama.cpp's decision code walks request JSON with this limit
/// (`RECURSIVE_LIMIT` in `tools/server/server-decision.cpp`): it refuses a
/// value 128 levels below the value it starts from, so a value it starts
/// from directly may be at most 128 deep. It starts from the template input
/// that wraps the request's fields, so a protocol's own limit for a field
/// can be lower.
pub const MAX_JSON_DEPTH: usize = 128;

/// Default [`JsonLimits::max_bytes`]: 1 MiB.
pub const DEFAULT_MAX_JSON_BYTES: usize = 1 << 20;

/// Default [`JsonLimits::max_nodes`]: 65,536 values.
pub const DEFAULT_MAX_JSON_NODES: usize = 1 << 16;

/// Ceilings for parsing one JSON text.
///
/// They bound what one text can make the parser allocate. The defaults
/// ([`DEFAULT_MAX_JSON_BYTES`], [`DEFAULT_MAX_JSON_NODES`]) are ceilings
/// against oversized input, far above what one decision prompt reads; a
/// protocol's token budget is the tighter bound and is checked by the
/// protocol. Depth is always capped at [`MAX_JSON_DEPTH`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct JsonLimits {
    /// Most bytes of JSON text accepted.
    pub max_bytes: usize,
    /// Most values accepted. Every scalar, array and object counts as one;
    /// object keys do not count.
    pub max_nodes: usize,
}

impl Default for JsonLimits {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_JSON_BYTES,
            max_nodes: DEFAULT_MAX_JSON_NODES,
        }
    }
}

impl JsonLimits {
    /// Override the byte ceiling.
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Override the value ceiling.
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }
}

/// A JSON number, classified the way llama.cpp's JSON library
/// (nlohmann::json) classifies it.
///
/// A literal without a fraction or an exponent is an integer:
/// [`I64`](Self::I64) when it fits, else [`U64`](Self::U64) when it fits,
/// else (overflow) [`F64`](Self::F64). A literal with a fraction or an
/// exponent is always `F64`, so `1` and `1.0` stay different values. `-0` is
/// the integer 0; `-0.0` is the float negative zero. nlohmann::json stores a
/// non-negative integer as unsigned where this type says `I64`; the values,
/// and what they render as, are the same.
///
/// A number read from an [`OrderedJson`] is finite and canonical (an
/// integer that fits `i64` is `I64`). Equality is identity: integers compare
/// by value (`I64(5) == U64(5)`), floats by bit pattern (`0.0 != -0.0`), and
/// an integer never equals a float.
///
/// `Debug` prints the variant only: a number in a request is caller content.
#[derive(Clone, Copy)]
pub enum JsonNumber {
    /// An integer literal that fits `i64`.
    I64(i64),
    /// An integer literal above `i64::MAX` that fits `u64`.
    U64(u64),
    /// A literal with a fraction or an exponent, or an integer literal
    /// outside both integer ranges.
    F64(f64),
}

impl JsonNumber {
    fn integer(self) -> Option<i128> {
        match self {
            Self::I64(value) => Some(value.into()),
            Self::U64(value) => Some(value.into()),
            Self::F64(_) => None,
        }
    }
}

impl PartialEq for JsonNumber {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::F64(a), Self::F64(b)) => a.to_bits() == b.to_bits(),
            (a, b) => matches!((a.integer(), b.integer()), (Some(a), Some(b)) if a == b),
        }
    }
}

impl Eq for JsonNumber {}

impl fmt::Debug for JsonNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::I64(_) => "I64",
            Self::U64(_) => "U64",
            Self::F64(_) => "F64",
        })
    }
}

/// The kind of a JSON value, for messages that must not quote it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JsonKind {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool,
    /// A number.
    Number,
    /// A string.
    String,
    /// An array.
    Array,
    /// An object.
    Object,
}

impl JsonKind {
    /// The kind's JSON name: `"null"`, `"boolean"`, `"number"`, `"string"`,
    /// `"array"` or `"object"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool => "boolean",
            Self::Number => "number",
            Self::String => "string",
            Self::Array => "array",
            Self::Object => "object",
        }
    }
}

impl fmt::Display for JsonKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An order-preserving, validated JSON value.
///
/// Built only by [`OrderedJson::parse`] (or [`str::parse`]), never through
/// `serde_json::Value`, so every value satisfies, by construction:
///
/// - no object repeats a key;
/// - every number is finite;
/// - nesting is at most [`MAX_JSON_DEPTH`] levels;
/// - its size fits the [`JsonLimits`] it was parsed with.
///
/// Object members keep their document order, numbers their [`JsonNumber`]
/// identity, and strings their exact text after unescaping. Read a value
/// through [`OrderedJson::view`] or the `as_*` accessors, and take it apart
/// without copying through the `into_*` methods; every part is itself a
/// valid value. Its serde form, in every format, is one string holding its
/// compact JSON text ([`OrderedJson::to_json_text`]).
///
/// Equality is structural, with [`JsonNumber`]'s identity rules, and
/// counts member order: `{"a":1,"b":2}` and `{"b":2,"a":1}` differ.
///
/// `Debug` prints the kind and a length, never the value.
///
/// # Example
///
/// ```
/// use xybrid_core::ir::{JsonNumber, JsonView, OrderedJson};
///
/// let value = OrderedJson::parse(r#"{"b": 1, "a": [1.0, -0, 18446744073709551615]}"#)?;
/// let JsonView::Object(members) = value.view() else { unreachable!() };
/// let keys: Vec<&str> = members.iter().map(|(key, _)| key.as_str()).collect();
/// assert_eq!(keys, ["b", "a"]);
/// assert_eq!(value.to_json_text(), r#"{"b":1,"a":[1.0,0,18446744073709551615]}"#);
///
/// let numbers: Vec<JsonNumber> = value.get("a").and_then(|a| a.as_array()).unwrap()
///     .iter().filter_map(|item| item.as_number()).collect();
/// assert_eq!(numbers, [JsonNumber::F64(1.0), JsonNumber::I64(0), JsonNumber::U64(u64::MAX)]);
///
/// assert!(OrderedJson::parse(r#"{"a": 1, "a": 2}"#).is_err());
/// # Ok::<(), xybrid_core::ir::DecisionError>(())
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct OrderedJson(Node);

#[derive(Clone, PartialEq, Eq)]
enum Node {
    Null,
    Bool(bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<OrderedJson>),
    Object(Vec<(String, OrderedJson)>),
}

/// A borrowed view of one [`OrderedJson`] value.
///
/// `Debug` prints the kind and a length, never the value.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum JsonView<'a> {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A finite number.
    Number(JsonNumber),
    /// A string, unescaped.
    String(&'a str),
    /// An array's items, in order.
    Array(&'a [OrderedJson]),
    /// An object's members, in document order, keys unique.
    Object(&'a [(String, OrderedJson)]),
}

impl JsonView<'_> {
    /// The kind of the value.
    pub fn kind(&self) -> JsonKind {
        match self {
            Self::Null => JsonKind::Null,
            Self::Bool(_) => JsonKind::Bool,
            Self::Number(_) => JsonKind::Number,
            Self::String(_) => JsonKind::String,
            Self::Array(_) => JsonKind::Array,
            Self::Object(_) => JsonKind::Object,
        }
    }

    /// Kind and length only: the value is caller content.
    fn redacted(&self, name: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = f.debug_struct(name);
        out.field("kind", &self.kind().as_str());
        match self {
            Self::String(text) => out.field("bytes", &text.len()),
            Self::Array(items) => out.field("items", &items.len()),
            Self::Object(members) => out.field("members", &members.len()),
            Self::Null | Self::Bool(_) | Self::Number(_) => &mut out,
        };
        out.finish()
    }
}

impl fmt::Debug for JsonView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.redacted("JsonView", f)
    }
}

impl OrderedJson {
    /// Parses JSON text with [`JsonLimits::default`].
    ///
    /// The grammar is RFC 8259's, as nlohmann::json applies it: no comments,
    /// no trailing commas, no leading zeros, only the four JSON whitespace
    /// characters, strings with every control character escaped and
    /// surrogate escapes in pairs, and a leading UTF-8 byte-order mark
    /// skipped. Numbers are classified as [`JsonNumber`] describes, with
    /// correctly rounded floats. On top of the grammar, an object may not
    /// repeat a key, a float may not overflow to infinity, and the value
    /// may not nest deeper than [`MAX_JSON_DEPTH`].
    ///
    /// # Errors
    ///
    /// [`DecisionError::InvalidJson`] with the byte offset of the problem,
    /// [`DecisionError::JsonTooDeep`], or a ceiling of the limits
    /// ([`DecisionError::JsonTooLong`], [`DecisionError::JsonTooManyNodes`]).
    /// Messages never quote the text.
    pub fn parse(text: &str) -> Result<Self, DecisionError> {
        Self::parse_with_limits(text, &JsonLimits::default())
    }

    /// Parses JSON text as [`OrderedJson::parse`] does, under `limits`.
    ///
    /// # Errors
    ///
    /// As for [`OrderedJson::parse`].
    pub fn parse_with_limits(text: &str, limits: &JsonLimits) -> Result<Self, DecisionError> {
        parse::parse(text, limits)
    }

    /// A borrowed view of the value, for matching on its kind.
    pub fn view(&self) -> JsonView<'_> {
        match &self.0 {
            Node::Null => JsonView::Null,
            Node::Bool(value) => JsonView::Bool(*value),
            Node::Number(number) => JsonView::Number(*number),
            Node::String(text) => JsonView::String(text),
            Node::Array(items) => JsonView::Array(items),
            Node::Object(members) => JsonView::Object(members),
        }
    }

    /// The kind of the value.
    pub fn kind(&self) -> JsonKind {
        self.view().kind()
    }

    /// `true` for `null`.
    pub fn is_null(&self) -> bool {
        matches!(self.0, Node::Null)
    }

    /// The value of a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self.0 {
            Node::Bool(value) => Some(value),
            _ => None,
        }
    }

    /// The value of a number.
    pub fn as_number(&self) -> Option<JsonNumber> {
        match self.0 {
            Node::Number(number) => Some(number),
            _ => None,
        }
    }

    /// The text of a string.
    pub fn as_str(&self) -> Option<&str> {
        match &self.0 {
            Node::String(text) => Some(text),
            _ => None,
        }
    }

    /// The items of an array.
    pub fn as_array(&self) -> Option<&[OrderedJson]> {
        match &self.0 {
            Node::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The members of an object, in document order.
    pub fn as_object(&self) -> Option<&[(String, OrderedJson)]> {
        match &self.0 {
            Node::Object(members) => Some(members),
            _ => None,
        }
    }

    /// The value of an object's member `key`. `None` for a missing key or a
    /// value that is not an object.
    pub fn get(&self, key: &str) -> Option<&OrderedJson> {
        self.as_object()?
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// The text of a string, or the value back when it is not one.
    ///
    /// # Errors
    ///
    /// Returns `self` unchanged when the value is not a string.
    pub fn into_string(self) -> Result<String, Self> {
        match self.0 {
            Node::String(text) => Ok(text),
            node => Err(Self(node)),
        }
    }

    /// The items of an array, or the value back when it is not one.
    ///
    /// # Errors
    ///
    /// Returns `self` unchanged when the value is not an array.
    pub fn into_array(self) -> Result<Vec<OrderedJson>, Self> {
        match self.0 {
            Node::Array(items) => Ok(items),
            node => Err(Self(node)),
        }
    }

    /// The members of an object, in document order, or the value back when
    /// it is not one.
    ///
    /// # Errors
    ///
    /// Returns `self` unchanged when the value is not an object.
    pub fn into_object(self) -> Result<Vec<(String, OrderedJson)>, Self> {
        match self.0 {
            Node::Object(members) => Ok(members),
            node => Err(Self(node)),
        }
    }

    /// How deep the value nests, counting itself: 1 for a scalar, `[]` or
    /// `{}`. At most [`MAX_JSON_DEPTH`].
    pub fn depth(&self) -> usize {
        let children = match &self.0 {
            Node::Array(items) => items.iter().map(Self::depth).max(),
            Node::Object(members) => members.iter().map(|(_, value)| value.depth()).max(),
            _ => None,
        };
        1 + children.unwrap_or(0)
    }

    /// How many values the value holds, counting itself: every scalar,
    /// array and object is one; object keys do not count.
    pub fn node_count(&self) -> usize {
        1 + match &self.0 {
            Node::Array(items) => items.iter().map(Self::node_count).sum(),
            Node::Object(members) => members.iter().map(|(_, value)| value.node_count()).sum(),
            _ => 0,
        }
    }

    /// The value as compact JSON text, in serde_json's compact style.
    ///
    /// No whitespace; members in document order; strings escape `"`, `\`
    /// and control characters only (`\b`, `\t`, `\n`, `\f`, `\r`, else
    /// `\u00XX` in lowercase hex), so other characters stay raw UTF-8;
    /// numbers exactly as serde_json writes them (integers in decimal,
    /// floats with the shortest digits that read back as the same `f64`,
    /// always with a `.` or an exponent, so `1.0` stays a float and `-0.0`
    /// stays `-0.0`). Parsing the text gives back an equal value, under a
    /// byte ceiling that admits it: compact text can be longer than the text
    /// a value was parsed from (`1E15` is written `1000000000000000.0`).
    ///
    /// The float spelling is serde_json's and can change with serde_json
    /// (its 1.0.151 formatter writes `1e+300` where earlier releases wrote
    /// `1e300`); the value it reads back as cannot.
    ///
    /// This is not llama.cpp's `tojson` template filter, which a protocol
    /// renders with its own separators and float notation.
    pub fn to_json_text(&self) -> String {
        let mut text = String::new();
        write::value(self, &mut text);
        text
    }

    /// The length in bytes of [`OrderedJson::to_json_text`], computed
    /// without building the text.
    pub fn json_text_len(&self) -> usize {
        let mut counter = write::Counter::default();
        write::value(self, &mut counter);
        counter.bytes()
    }
}

impl FromStr for OrderedJson {
    type Err = DecisionError;

    /// Parses with [`JsonLimits::default`], as [`OrderedJson::parse`] does.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text)
    }
}

impl fmt::Debug for OrderedJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.view().redacted("OrderedJson", f)
    }
}

/// One string holding the compact JSON text, in every serde format. That
/// keeps key order and number identity exactly, and works the same in
/// self-describing formats (JSON) and in formats that are not (bincode).
impl Serialize for OrderedJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_json_text())
    }
}

/// Reads the string [`Serialize`] writes and parses it with the default
/// value ceiling ([`DEFAULT_MAX_JSON_NODES`]) and no byte ceiling; invalid
/// JSON text is refused with a content-free message.
///
/// The text is already in memory, and compact text can be longer than the
/// text a value was parsed from, so a byte ceiling would refuse values the
/// defaults accepted. The value ceiling bounds the tree a decoded blob can
/// build. A value parsed under a larger value ceiling serializes, but does
/// not deserialize: serde gives no way to pass limits in.
impl<'de> Deserialize<'de> for OrderedJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(JsonTextVisitor)
    }
}

struct JsonTextVisitor;

impl Visitor<'_> for JsonTextVisitor {
    type Value = OrderedJson;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a string holding JSON text")
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<OrderedJson, E> {
        let limits = JsonLimits::default().with_max_bytes(usize::MAX);
        OrderedJson::parse_with_limits(text, &limits).map_err(E::custom)
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for OrderedJson {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "OrderedJson".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "description": "A JSON value as compact JSON text; object keys keep their order."
        })
    }
}
