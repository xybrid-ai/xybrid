//! [`OrderedJson`] to compact JSON text, in serde_json's compact style.
//!
//! One writer drives both [`OrderedJson::to_json_text`] and
//! [`OrderedJson::json_text_len`] through [`Sink`], so the length always
//! matches the text. Numbers go through serde_json's own formatter, so
//! their text is serde_json's to the digit; nothing is parsed through
//! serde_json.

use super::{JsonNumber, Node, OrderedJson};
use serde_json::ser::{CompactFormatter, Formatter};
use std::io;

/// Where compact text goes: a `String`, or a byte [`Counter`].
pub(super) trait Sink {
    fn push_str(&mut self, text: &str);
}

impl Sink for String {
    fn push_str(&mut self, text: &str) {
        String::push_str(self, text);
    }
}

/// Counts the bytes of the text instead of keeping it.
#[derive(Default)]
pub(super) struct Counter(usize);

impl Counter {
    pub(super) fn bytes(&self) -> usize {
        self.0
    }
}

impl Sink for Counter {
    fn push_str(&mut self, text: &str) {
        self.0 += text.len();
    }
}

pub(super) fn value(value: &OrderedJson, out: &mut impl Sink) {
    match &value.0 {
        Node::Null => out.push_str("null"),
        Node::Bool(true) => out.push_str("true"),
        Node::Bool(false) => out.push_str("false"),
        Node::Number(number) => self::number(*number, out),
        Node::String(text) => string(text, out),
        Node::Array(items) => {
            out.push_str("[");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(",");
                }
                self::value(item, out);
            }
            out.push_str("]");
        }
        Node::Object(members) => {
            out.push_str("{");
            for (index, (key, item)) in members.iter().enumerate() {
                if index > 0 {
                    out.push_str(",");
                }
                string(key, out);
                out.push_str(":");
                self::value(item, out);
            }
            out.push_str("}");
        }
    }
}

/// serde_json's escapes for U+0000 to U+001F: the short forms where JSON
/// has one, lowercase `\u00XX` otherwise.
const CONTROL_ESCAPES: [&str; 32] = [
    "\\u0000", "\\u0001", "\\u0002", "\\u0003", "\\u0004", "\\u0005", "\\u0006", "\\u0007", "\\b",
    "\\t", "\\n", "\\u000b", "\\f", "\\r", "\\u000e", "\\u000f", "\\u0010", "\\u0011", "\\u0012",
    "\\u0013", "\\u0014", "\\u0015", "\\u0016", "\\u0017", "\\u0018", "\\u0019", "\\u001a",
    "\\u001b", "\\u001c", "\\u001d", "\\u001e", "\\u001f",
];

/// A JSON string: `"` and `\` and control characters escaped, everything
/// else raw UTF-8.
fn string(text: &str, out: &mut impl Sink) {
    out.push_str("\"");
    let mut run = 0;
    for (index, byte) in text.bytes().enumerate() {
        let escape = match byte {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            0x00..=0x1f => CONTROL_ESCAPES[usize::from(byte)],
            _ => continue,
        };
        // Escaped bytes are ASCII, so `run..index` is on character boundaries.
        out.push_str(&text[run..index]);
        out.push_str(escape);
        run = index + 1;
    }
    out.push_str(&text[run..]);
    out.push_str("\"");
}

/// A number exactly as serde_json writes it, through its own compact
/// formatter: integers in decimal; finite floats with the shortest digits
/// that read back as the same `f64` (ties to the even digit), in fixed
/// notation for decimal exponents -5 to 15 with a `.0` after an integral
/// value (`100.0`, `0.00001`), as `d.ddde±x` otherwise (`1e+16`, `1.5e-7`).
/// A float always has a `.` or an `e`, so it never reads back as an
/// integer.
///
/// A non-finite float is written as `null`, as serde_json does; numbers from
/// [`OrderedJson::parse`] are always finite.
pub(super) fn number(number: JsonNumber, out: &mut impl Sink) {
    let mut text = NumberText::default();
    let mut formatter = CompactFormatter;
    let written = match number {
        JsonNumber::I64(value) => formatter.write_i64(&mut text, value),
        JsonNumber::U64(value) => formatter.write_u64(&mut text, value),
        JsonNumber::F64(value) if value.is_finite() => formatter.write_f64(&mut text, value),
        JsonNumber::F64(_) => {
            out.push_str("null");
            return;
        }
    };
    written.expect("a number's text fits NumberText");
    out.push_str(text.as_str());
}

/// A fixed buffer for the text of one number, so writing one does not
/// allocate. 40 bytes hold any `i64`, `u64` or `f64` text (at most 24).
struct NumberText {
    bytes: [u8; 40],
    len: usize,
}

impl Default for NumberText {
    fn default() -> Self {
        Self {
            bytes: [0; 40],
            len: 0,
        }
    }
}

impl NumberText {
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).expect("serde_json writes numbers in ASCII")
    }
}

impl io::Write for NumberText {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let end = self.len + bytes.len();
        let slot = self
            .bytes
            .get_mut(self.len..end)
            .ok_or(io::ErrorKind::WriteZero)?;
        slot.copy_from_slice(bytes);
        self.len = end;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
