//! llama.cpp's `tojson` filter: how the d1 template serializes JSON content.
//!
//! The template prints a JSON state with `tojson(indent=2)` and other JSON
//! content with `tojson`. llama.cpp's own Jinja engine implements the filter
//! (`common/jinja/value.cpp:250-277`) over its own serializer
//! (`value_to_json_internal`, `:1627-1706`) and float notation (`float_repr`,
//! `:1555-1612`): Python's `json.dumps` layout, members in input order, no
//! HTML escaping, floats as Python's `repr`, integers as integers. This is a
//! port of those, over [`Escaped`].

use std::fmt::Write as _;

use super::escape::Escaped;

/// The `tojson` arguments the d1 template uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct JsonStyle {
    /// `indent`: `None` writes one line; `Some(n)` a newline before each
    /// item and `n` spaces per level.
    pub(super) indent: Option<usize>,
}

impl JsonStyle {
    /// The item separator: `", "` on one line, `","` with an indent.
    fn item_separator(self) -> &'static str {
        if self.indent.is_some() {
            ","
        } else {
            ", "
        }
    }

    /// The key separator, `": "` either way.
    fn key_separator(self) -> &'static str {
        ": "
    }
}

/// `value` as llama.cpp's `tojson` writes it, with `ensure_ascii` false, or
/// `None` when that is longer than `cap` bytes. The text never grows past
/// `cap`: writing stops where it would.
pub(super) fn to_json(value: &Escaped, style: JsonStyle, cap: usize) -> Option<String> {
    let mut out = Sink::new(cap);
    write_value(value, 0, style, &mut out);
    out.finish()
}

/// Text that refuses to grow past a cap: once a write would pass it, the
/// sink keeps what it has, ignores every later write, and is full.
pub(super) struct Sink {
    text: String,
    cap: usize,
    full: bool,
}

impl Sink {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            text: String::new(),
            cap,
            full: false,
        }
    }

    pub(super) fn push_str(&mut self, text: &str) {
        if self.full || self.text.len() + text.len() > self.cap {
            self.full = true;
            return;
        }
        self.text.push_str(text);
    }

    fn push(&mut self, c: char) {
        self.push_str(c.encode_utf8(&mut [0; 4]));
    }

    fn push_spaces(&mut self, count: usize) {
        if self.full || self.text.len() + count > self.cap {
            self.full = true;
            return;
        }
        self.text.extend(std::iter::repeat_n(' ', count));
    }

    /// Whether a write was refused.
    pub(super) fn is_full(&self) -> bool {
        self.full
    }

    /// The text, or `None` when a write was refused.
    pub(super) fn finish(self) -> Option<String> {
        (!self.full).then_some(self.text)
    }
}

impl std::fmt::Write for Sink {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        self.push_str(text);
        if self.full {
            Err(std::fmt::Error)
        } else {
            Ok(())
        }
    }
}

/// `value_to_json_internal`: `level` is the depth of `value`.
fn write_value(value: &Escaped, level: usize, style: JsonStyle, out: &mut Sink) {
    if out.is_full() {
        return;
    }
    match value {
        Escaped::Null => out.push_str("null"),
        Escaped::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Escaped::Int(value) => {
            let _ = write!(out, "{value}");
        }
        Escaped::Float(value) => out.push_str(&float_json(*value)),
        Escaped::Str(text) => write_string(text, out),
        Escaped::Array(items) => {
            out.push('[');
            write_items(items.iter(), level, style, out, |item, out| {
                write_value(item, level + 1, style, out)
            });
            out.push(']');
        }
        Escaped::Object(members) => {
            out.push('{');
            write_items(members.iter(), level, style, out, |(key, item), out| {
                write_string(key, out);
                out.push_str(style.key_separator());
                write_value(item, level + 1, style, out);
            });
            out.push('}');
        }
    }
}

/// The items of an array or object at `level`: nothing when there are none;
/// otherwise each on its own line, indented one level deeper, when there is
/// an indent.
fn write_items<T>(
    items: impl ExactSizeIterator<Item = T>,
    level: usize,
    style: JsonStyle,
    out: &mut Sink,
    mut write_item: impl FnMut(T, &mut Sink),
) {
    let count = items.len();
    if count == 0 {
        return;
    }
    let newline = |out: &mut Sink| {
        if style.indent.is_some() {
            out.push('\n');
        }
    };
    let pad = |out: &mut Sink, level: usize| {
        if let Some(indent) = style.indent {
            out.push_spaces(indent * level);
        }
    };
    newline(out);
    for (index, item) in items.enumerate() {
        if out.is_full() {
            return;
        }
        pad(out, level + 1);
        write_item(item, out);
        if index + 1 < count {
            out.push_str(style.item_separator());
        }
        newline(out);
    }
    pad(out, level);
}

/// A JSON string: `"` and `\` escaped, the short escapes for backspace,
/// form feed, newline, carriage return and tab, `\u00XX` (lowercase) for
/// the other control characters, everything else raw UTF-8.
fn write_string(text: &str, out: &mut Sink) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
        if out.is_full() {
            return;
        }
    }
    out.push('"');
}

/// A float as Python's `json.dumps` writes it (`float_json`,
/// `value.cpp:1615-1623`): `NaN`, `Infinity`, `-Infinity`, else
/// [`float_repr`].
pub(super) fn float_json(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        if value < 0.0 { "-Infinity" } else { "Infinity" }.to_string()
    } else {
        float_repr(value)
    }
}

/// A float as Python's `repr` writes it: the shortest digits that read back
/// as the same value, in fixed notation for `1e-4 <= |value| < 1e16` (with
/// at least one decimal) and `d.ddde+XX` otherwise.
///
/// This is llama.cpp's algorithm (`float_repr`, `value.cpp:1555-1612`), not
/// Rust's shortest formatting, which can pick a different digit when two
/// shortest candidates are equally close (`562949953421312.25` is
/// `...312.2` here and in Python, `...312.3` in Rust): it tries `%.*e` from
/// 15 significant digits up (from 1 for subnormals) and keeps the first that
/// reads back, bumping the last digit at a power of two. Rust's
/// fixed-precision formatting rounds ties to even, as glibc's `printf` does.
pub(super) fn float_repr(value: f64) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "nan"
        } else if value < 0.0 {
            "-inf"
        } else {
            "inf"
        }
        .to_string();
    }
    let magnitude = value.abs();
    let is_power_of_two = is_power_of_two(magnitude);

    // DBL_DIG - 1: a normal double has at most one decimal of up to 15
    // significant digits that reads back as it.
    let mut precision: usize = if magnitude < f64::MIN_POSITIVE { 0 } else { 14 };
    let (digits, exponent) = loop {
        let formatted = format!("{magnitude:.precision$e}");
        let (mantissa, exponent) = formatted
            .split_once('e')
            .expect("exponent formatting writes an `e`");
        let mut digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
        let exponent: i32 = exponent.parse().expect("an exponent is an integer");
        let scale = exponent - precision as i32;
        let reads_back = |digits: &str| format!("{digits}e{scale}").parse::<f64>() == Ok(magnitude);
        // 17 significant digits always read back.
        if precision == 16 || reads_back(&digits) {
            break (digits, exponent);
        }
        // At a power of two the gap below is half the gap above: the
        // nearest digits can miss while the next ones up read back. A last
        // 9 is never bumped (it would carry); no power of two needs it.
        if is_power_of_two && !digits.ends_with('9') {
            let last = digits.pop().expect("at least one digit");
            digits.push(char::from(last as u8 + 1));
            if reads_back(&digits) {
                break (digits, exponent);
            }
        }
        precision += 1;
    };
    let digits = match digits.trim_end_matches('0') {
        "" => "0",
        trimmed => trimmed,
    };

    let count = digits.len() as i32;
    let point = exponent + 1; // digits before the decimal point
    let mut out = String::with_capacity(32);
    if value.is_sign_negative() {
        out.push('-');
    }
    if point > -4 && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.push_str(&"0".repeat((-point) as usize));
            out.push_str(digits);
        } else if point >= count {
            out.push_str(digits);
            out.push_str(&"0".repeat((point - count) as usize));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if count > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
    out
}

/// Whether `frexp(magnitude)` has the fraction 0.5: a power of two, normal
/// or subnormal.
fn is_power_of_two(magnitude: f64) -> bool {
    const FRACTION: u64 = (1 << 52) - 1;
    let bits = magnitude.to_bits();
    let fraction = bits & FRACTION;
    if bits >> 52 == 0 {
        fraction.is_power_of_two()
    } else {
        fraction == 0
    }
}
