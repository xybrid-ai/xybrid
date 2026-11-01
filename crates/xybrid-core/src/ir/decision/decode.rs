//! The decode ceilings of the module docs ("Decoding is bounded"), checked
//! while deserializing, before anything they bound is allocated.
//!
//! Every string and sequence field of the decision types deserializes
//! through this module (`#[serde(deserialize_with = ...)]`), in every
//! format, so the boundary that decodes an envelope never builds an
//! oversized decision payload before a check runs, nested occurrences
//! included. [`OrderedJson`](super::OrderedJson)'s own `Deserialize` applies
//! [`MAX_JSON_TEXT_BYTES`] and the default value ceiling.
//!
//! A sequence is refused when its declared length (serde's `size_hint`,
//! which bincode reports exactly) is over its ceiling, before any element is
//! read; otherwise its elements are counted as they arrive and the one past
//! the ceiling is refused. A string is measured before it is copied.
//! Refusals are fixed texts naming the kind of field and its ceiling, never
//! content.
//!
//! What a format allocates before a visitor runs is outside what a visitor
//! can bound. serde_json unescapes a string into a buffer before handing it
//! over, at most as long as the input itself. bincode decoding from a byte
//! slice hands strings over borrowed, so these checks bound it; its
//! `with_limit` does not, because bincode 1.3.x replaces the limit with
//! none for slices. bincode decoding from an `io::Read` reads a string's
//! declared length into its own buffer before any visitor runs, so there
//! only `with_limit` with `deserialize_from` bounds that allocation.

use super::{
    DecisionAnswer, DecisionLevel, DecisionOption, DecisionQuestion, DecisionScore,
    DEFAULT_MAX_OPTIONS, DEFAULT_MAX_QUESTIONS, DEFAULT_MAX_REQUEST_BYTES,
};
use serde::de::{self, Deserialize, Deserializer, SeqAccess, Visitor};
use std::fmt;
use std::marker::PhantomData;

/// Most questions in a request, and answers in a result.
pub(super) const MAX_QUESTIONS: usize = DEFAULT_MAX_QUESTIONS;

/// Most options of a choice, levels of a score, and entries of a
/// distribution.
pub(super) const MAX_ALTERNATIVES: usize = DEFAULT_MAX_OPTIONS;

/// Most bytes of one string field.
pub(super) const MAX_STRING_BYTES: usize = DEFAULT_MAX_REQUEST_BYTES;

/// Most bytes of an `OrderedJson`'s text: 2 MiB.
///
/// The serde form is a value's compact text, which can be longer than the
/// text it was parsed from: only a number held as a float (an integer past
/// 64 bits included) grows when written back, by at most 14 bytes (`1e15`
/// becomes `1000000000000000.0`), while strings, 64-bit integers and
/// literals keep or lose length and whitespace goes. A value
/// [`OrderedJson::parse`](super::OrderedJson::parse) accepts under the
/// default limits (1 MiB, 65,536 values) therefore writes at most
/// 1,048,576 + 14 × 65,536 = 1,966,080 bytes, and always decodes again.
pub(super) const MAX_JSON_TEXT_BYTES: usize = 2 << 20;

/// A string field, at most [`MAX_STRING_BYTES`].
pub(super) fn string<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    deserializer.deserialize_str(BoundedString)
}

struct BoundedString;

impl BoundedString {
    fn check<E: de::Error>(len: usize) -> Result<(), E> {
        if len > MAX_STRING_BYTES {
            Err(E::custom(format_args!(
                "a decision string field is longer than {MAX_STRING_BYTES} bytes"
            )))
        } else {
            Ok(())
        }
    }

    fn utf8<E: de::Error>() -> E {
        E::custom("a decision string field is not UTF-8")
    }
}

impl Visitor<'_> for BoundedString {
    type Value = String;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a string of at most {MAX_STRING_BYTES} bytes")
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<String, E> {
        Self::check(text.len())?;
        Ok(text.to_owned())
    }

    fn visit_string<E: de::Error>(self, text: String) -> Result<String, E> {
        Self::check(text.len())?;
        Ok(text)
    }

    fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<String, E> {
        Self::check(bytes.len())?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| Self::utf8())
    }

    fn visit_byte_buf<E: de::Error>(self, bytes: Vec<u8>) -> Result<String, E> {
        Self::check(bytes.len())?;
        String::from_utf8(bytes).map_err(|_| Self::utf8())
    }
}

/// A sequence of at most `max` elements, refused as "more than `max` `what`".
fn sequence<'de, D, T>(deserializer: D, max: usize, what: &'static str) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer.deserialize_seq(BoundedSequence {
        max,
        what,
        element: PhantomData,
    })
}

struct BoundedSequence<T> {
    max: usize,
    what: &'static str,
    element: PhantomData<T>,
}

impl<'de, T: Deserialize<'de>> Visitor<'de> for BoundedSequence<T> {
    type Value = Vec<T>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a sequence of at most {} {}", self.max, self.what)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<T>, A::Error> {
        let too_many = || de::Error::custom(format_args!("more than {} {}", self.max, self.what));
        // A declared length over the ceiling is refused before anything is
        // read or allocated (bincode declares every length up front).
        let declared = seq.size_hint();
        if declared.is_some_and(|len| len > self.max) {
            return Err(too_many());
        }
        let mut items = Vec::with_capacity(declared.unwrap_or(0));
        while let Some(item) = seq.next_element()? {
            if items.len() == self.max {
                return Err(too_many());
            }
            items.push(item);
        }
        Ok(items)
    }
}

/// A request's questions, at most [`MAX_QUESTIONS`].
pub(super) fn questions<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<DecisionQuestion>, D::Error> {
    sequence(deserializer, MAX_QUESTIONS, "questions")
}

/// A result's answers, at most [`MAX_QUESTIONS`].
pub(super) fn answers<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<DecisionAnswer>, D::Error> {
    sequence(deserializer, MAX_QUESTIONS, "answers")
}

/// A choice's options, at most [`MAX_ALTERNATIVES`].
pub(super) fn options<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<DecisionOption>, D::Error> {
    sequence(deserializer, MAX_ALTERNATIVES, "options")
}

/// A score's levels, at most [`MAX_ALTERNATIVES`].
pub(super) fn levels<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<DecisionLevel>, D::Error> {
    sequence(deserializer, MAX_ALTERNATIVES, "levels")
}

/// An answer's distribution, at most [`MAX_ALTERNATIVES`] entries.
pub(super) fn distribution<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<DecisionScore>, D::Error> {
    sequence(deserializer, MAX_ALTERNATIVES, "distribution entries")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::decision::{
        DecisionContent, DecisionOrigin, DecisionQuestionKind, DecisionRequest, DecisionState,
        DecisionUsage, DecisionValue, Decisions, JsonLimits, OrderedJson, DEFAULT_MAX_JSON_BYTES,
        DEFAULT_MAX_JSON_NODES,
    };
    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::fmt::Debug;
    use std::time::{Duration, Instant};

    /// Content every refused value carries: no refusal may quote it.
    const CANARY: &str = "canary-7f3a9c";

    /// The allocator of xybrid-core's whole unit-test binary (a binary has
    /// one): it records the largest single allocation a thread makes inside
    /// `watched`, and otherwise passes every call to the system allocator.
    struct Probe;

    thread_local! {
        static LARGEST: Cell<Option<usize>> = const { Cell::new(None) };
    }

    fn record(size: usize) {
        // A const-initialized `Cell` needs no allocation and no destructor,
        // so this is safe to run inside the allocator.
        let _ = LARGEST.try_with(|largest| {
            if let Some(max) = largest.get() {
                largest.set(Some(max.max(size)));
            }
        });
    }

    // SAFETY: every call goes to the system allocator unchanged; `record`
    // only reads and writes a thread-local `Cell`.
    unsafe impl GlobalAlloc for Probe {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record(layout.size());
            System.alloc(layout)
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record(layout.size());
            System.alloc_zeroed(layout)
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record(new_size);
            System.realloc(ptr, layout, new_size)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout)
        }
    }

    #[global_allocator]
    static PROBE: Probe = Probe;

    /// What `run` returns, and the largest single allocation it made.
    fn watched<T>(run: impl FnOnce() -> T) -> (T, usize) {
        LARGEST.with(|largest| largest.set(Some(0)));
        let result = run();
        let largest = LARGEST.with(|largest| largest.replace(None)).unwrap_or(0);
        (result, largest)
    }

    /// `value` decoded from its JSON and from its bincode.
    fn both<T: Serialize + DeserializeOwned>(value: &T) -> [(&'static str, Result<T, String>); 2] {
        let json = serde_json::to_string(value).unwrap();
        let bytes = bincode::serialize(value).unwrap();
        [
            (
                "json",
                serde_json::from_str(&json).map_err(|error| error.to_string()),
            ),
            (
                "bincode",
                bincode::deserialize(&bytes).map_err(|error| error.to_string()),
            ),
        ]
    }

    fn decodes<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
        for (format, back) in both(value) {
            match back {
                Ok(back) => assert!(back == *value, "{format}: decoded something else"),
                Err(error) => panic!("{format}: {error}"),
            }
        }
    }

    /// `value` is refused in both formats for `reason`, quoting none of its
    /// content.
    fn refused<T: Serialize + DeserializeOwned + Debug>(value: &T, reason: &str) {
        for (format, back) in both(value) {
            let error = back.expect_err(format);
            assert!(error.contains(reason), "{format}: {error}");
            assert!(!error.contains(CANARY), "{format}: {error}");
        }
    }

    fn yes_no(id: impl Into<String>) -> DecisionQuestion {
        DecisionQuestion::new(id, None, DecisionQuestionKind::yes_no(None, None))
    }

    fn choice(options: usize) -> DecisionQuestion {
        let options = (0..options)
            .map(|i| DecisionOption::new(format!("{CANARY}-{i}"), None))
            .collect();
        DecisionQuestion::new("q", None, DecisionQuestionKind::choice(options))
    }

    fn score(levels: usize) -> DecisionQuestion {
        let levels = vec![DecisionLevel::new(Some(DecisionContent::text(CANARY))); levels];
        DecisionQuestion::new("q", None, DecisionQuestionKind::score(levels))
    }

    fn request(questions: Vec<DecisionQuestion>) -> DecisionRequest {
        DecisionRequest::new(DecisionState::default(), questions)
    }

    /// A valid choice answer over `entries` equally likely keys.
    fn uniform(entries: usize) -> DecisionAnswer {
        let p = 1.0 / entries as f64;
        let distribution = (0..entries)
            .map(|i| DecisionScore::new(format!("{CANARY}-{i}"), 0.0, p, DecisionOrigin::Caller))
            .collect();
        let chosen = DecisionValue::choice(format!("{CANARY}-0"));
        DecisionAnswer::new("q", distribution, chosen, None).unwrap()
    }

    #[test]
    fn questions_and_answers_stop_at_16() {
        let questions = |n: usize| (0..n).map(|i| yes_no(format!("{CANARY}-{i}"))).collect();
        decodes(&request(questions(16)));
        refused(&request(questions(17)), "more than 16 questions");

        let answers = |n: usize| Decisions::new(vec![uniform(2); n], DecisionUsage::new(1, 1));
        decodes(&answers(16));
        refused(&answers(17), "more than 16 answers");
    }

    #[test]
    fn options_levels_and_distributions_stop_at_255() {
        decodes(&request(vec![choice(255)]));
        refused(&request(vec![choice(256)]), "more than 255 options");
        decodes(&request(vec![score(255)]));
        refused(&request(vec![score(256)]), "more than 255 levels");

        let result = |n: usize| Decisions::new(vec![uniform(n)], DecisionUsage::new(1, 1));
        decodes(&result(255));
        refused(&result(256), "more than 255 distribution entries");
    }

    #[test]
    fn every_string_field_stops_at_1_mib() {
        let at = "x".repeat(MAX_STRING_BYTES);
        let over = CANARY.repeat(MAX_STRING_BYTES / CANARY.len() + 1);
        assert!(over.len() > MAX_STRING_BYTES);
        let too_long = "longer than 1048576 bytes";

        // A question id, an option key and text content.
        let requests = |text: &str| {
            let option = DecisionOption::new(text, None);
            [
                request(vec![yes_no(text)]),
                request(vec![DecisionQuestion::new(
                    "q",
                    None,
                    DecisionQuestionKind::choice(vec![option]),
                )]),
                DecisionRequest::new(DecisionState::text(text), vec![yes_no("q")]),
            ]
        };
        for value in requests(&at) {
            decodes(&value);
        }
        for value in requests(&over) {
            refused(&value, too_long);
        }

        // An answer's question id, a distribution key and a chosen key.
        let one = |key: &str| vec![DecisionScore::new(key, 0.0, 1.0, DecisionOrigin::Caller)];
        decodes(&DecisionAnswer::new(&at, one("a"), DecisionValue::choice("a"), None).unwrap());
        refused(
            &DecisionAnswer::new(&over, one("a"), DecisionValue::choice("a"), None).unwrap(),
            too_long,
        );
        decodes(&one(&at)[0]);
        refused(&one(&over)[0], too_long);
        decodes(&DecisionValue::choice(&at));
        refused(&DecisionValue::choice(&over), too_long);
    }

    #[test]
    fn json_text_stops_at_2_mib_and_65536_values() {
        // The longest compact text the default limits allow: 1 MiB of text
        // holding one long string and 65,534 × `1e15`, each written back as
        // `1000000000000000.0`, 14 bytes longer. It decodes.
        let floats = DEFAULT_MAX_JSON_NODES - 2;
        let filler = DEFAULT_MAX_JSON_BYTES - 4 - 5 * floats;
        let text = format!("[\"{}\"{}]", "x".repeat(filler), ",1e15".repeat(floats));
        assert_eq!(text.len(), DEFAULT_MAX_JSON_BYTES);
        let value = OrderedJson::parse(&text).unwrap();
        assert_eq!(value.node_count(), DEFAULT_MAX_JSON_NODES);
        assert_eq!(value.json_text_len(), 1_966_052);
        decodes(&value);

        // Built under larger limits: 2 MiB of text decodes, and one byte
        // more does not, on its own or as content; nor does one more value.
        let roomy = JsonLimits::default()
            .with_max_bytes(4 << 20)
            .with_max_nodes(1 << 20);
        let parse = |text: String| OrderedJson::parse_with_limits(&text, &roomy).unwrap();
        // `["…"]`: an array, which content keeps as JSON.
        let sized = |len: usize| {
            let body = &CANARY.repeat(len / CANARY.len() + 1)[..len - 4];
            parse(format!("[\"{body}\"]"))
        };
        decodes(&sized(MAX_JSON_TEXT_BYTES));
        let too_long = "at most 2097152 are accepted";
        refused(&sized(MAX_JSON_TEXT_BYTES + 1), too_long);
        let content = DecisionContent::from_json(sized(MAX_JSON_TEXT_BYTES + 1));
        refused(&DecisionState::new(content), too_long);

        let values = parse(format!(
            "[\"{CANARY}\"{}]",
            ",0".repeat(DEFAULT_MAX_JSON_NODES - 1)
        ));
        assert_eq!(values.node_count(), DEFAULT_MAX_JSON_NODES + 1);
        refused(&values, "more than 65536 values");
    }

    /// `bytes` with the eight-byte little-endian length at `at` set to `len`.
    fn declaring(mut bytes: Vec<u8>, at: usize, len: u64) -> Vec<u8> {
        bytes[at..at + 8].copy_from_slice(&len.to_le_bytes());
        bytes
    }

    /// `bytes` are refused for `reason` at once, quoting no content and
    /// allocating nothing over 4 KiB at a time.
    fn refused_at_once<T: DeserializeOwned + Debug>(bytes: &[u8], reason: &str) {
        let start = Instant::now();
        let (result, largest) = watched(|| bincode::deserialize::<T>(bytes));
        let elapsed = start.elapsed();
        let error = result.expect_err(reason).to_string();
        assert!(error.contains(reason), "{error}");
        assert!(!error.contains(CANARY), "{error}");
        assert!(
            largest < 4096,
            "{reason}: allocated {largest} bytes at once"
        );
        assert!(
            elapsed < Duration::from_secs(1),
            "{reason}: took {elapsed:?}"
        );
    }

    #[test]
    fn declared_length_bombs_fail_fast_without_allocating() {
        let huge = 1u64 << 62;
        let state = bincode::serialize(&DecisionState::default()).unwrap().len();
        let tail = |value: &DecisionRequest| {
            let bytes = bincode::serialize(value).unwrap();
            let at = bytes.len() - 8;
            declaring(bytes, at, huge)
        };

        // Sequences: the declared length is refused before an element is read.
        let bomb = tail(&request(vec![]));
        refused_at_once::<DecisionRequest>(&bomb, "more than 16 questions");
        let bomb = tail(&request(vec![choice(0)]));
        refused_at_once::<DecisionRequest>(&bomb, "more than 255 options");
        let bomb = tail(&request(vec![score(0)]));
        refused_at_once::<DecisionRequest>(&bomb, "more than 255 levels");
        let none = bincode::serialize(&Decisions::new(vec![], DecisionUsage::new(0, 0))).unwrap();
        refused_at_once::<Decisions>(&declaring(none, 0, huge), "more than 16 answers");
        let one = bincode::serialize(&Decisions::new(vec![uniform(1)], DecisionUsage::new(1, 1)));
        // The answers' length (8), then the question id's length (8) and "q".
        let bomb = declaring(one.unwrap(), 17, huge);
        refused_at_once::<Decisions>(&bomb, "more than 255 distribution entries");

        // A string or a JSON text declaring more than the input holds ends
        // at the input: bincode checks a slice's length before reading.
        let bytes = bincode::serialize(&request(vec![yes_no("q")])).unwrap();
        refused_at_once::<DecisionRequest>(&declaring(bytes, state + 8, huge), "io error");
        let json = |text: &str| {
            let roomy = JsonLimits::default().with_max_bytes(4 << 20);
            let value = OrderedJson::parse_with_limits(text, &roomy).unwrap();
            bincode::serialize(&DecisionState::new(DecisionContent::from_json(value))).unwrap()
        };
        // The body's option tag (1) and variant index (4), then the length.
        refused_at_once::<DecisionState>(&declaring(json("[1]"), 5, huge), "io error");

        // One that is there but over its ceiling is refused on its length,
        // before it is copied or parsed: bincode hands it over borrowed.
        let long = bincode::serialize(&request(vec![yes_no(CANARY.repeat(90_000))])).unwrap();
        refused_at_once::<DecisionRequest>(&long, "longer than 1048576 bytes");
        let long = json(&format!("[\"{}\"]", CANARY.repeat(170_000)));
        refused_at_once::<DecisionState>(&long, "at most 2097152 are accepted");
    }
}
