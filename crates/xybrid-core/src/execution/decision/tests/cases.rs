//! The conformance cases of `integration-tests/fixtures/decision/`, read
//! for the tests.
//!
//! Case files are read with [`OrderedJson`]: object members keep their
//! order (the answers of a case are in request order, its probabilities in
//! option order), and every float is `str::parse::<f64>` of its literal, so
//! the 17-digit expectations come back exactly (the fixture README's
//! comparison rule). serde_json is not used.
//!
//! [`request`] turns a case's `request_text` into a [`DecisionRequest`] for
//! the requests this crate's protocol is meant to see. It is a test
//! transport, not the SystemOne one (the CLI's): it reads the outer
//! document itself and parses each content value (state, instructions,
//! descriptions, levels) with [`OrderedJson::parse`], so the 128-level
//! ceiling applies to each value, not the document; anything a transport
//! would refuse is an `Err` naming why.

use std::path::{Path, PathBuf};

use crate::ir::decision::{
    DecisionContent, DecisionLevel, DecisionOption, DecisionQuestion, DecisionQuestionKind,
    DecisionRequest, DecisionState, JsonLimits, JsonNumber, JsonView, OrderedJson,
};

/// The fixture directory, `integration-tests/fixtures/decision`.
pub(super) fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration-tests/fixtures/decision")
}

/// One case file.
pub(super) struct Case {
    pub(super) id: String,
    pub(super) json: OrderedJson,
}

impl Case {
    /// The value at `path` (object keys, or array indexes as decimal
    /// strings), if any.
    pub(super) fn at(&self, path: &[&str]) -> Option<&OrderedJson> {
        at(&self.json, path)
    }

    /// The string at `path`.
    pub(super) fn str_at(&self, path: &[&str]) -> Option<&str> {
        self.at(path).and_then(OrderedJson::as_str)
    }

    /// The case's expectation for `stage`: `"match"`, …, or `refuse:<kind>`.
    pub(super) fn expect(&self, stage: &str) -> Option<String> {
        let value = self.at(&["xybrid", stage])?;
        match value.view() {
            JsonView::String(text) => Some(text.to_string()),
            JsonView::Object(_) => value
                .get("refuse")
                .and_then(OrderedJson::as_str)
                .map(|kind| format!("refuse:{kind}")),
            _ => None,
        }
    }

    pub(super) fn request_text(&self) -> &str {
        self.str_at(&["request_text"])
            .expect("every case has a request_text")
    }
}

/// The value at `path` below `value`.
pub(super) fn at<'a>(value: &'a OrderedJson, path: &[&str]) -> Option<&'a OrderedJson> {
    path.iter()
        .try_fold(value, |value, step| match value.view() {
            JsonView::Object(_) => value.get(step),
            JsonView::Array(items) => items.get(step.parse::<usize>().ok()?),
            _ => None,
        })
}

/// Every case, sorted by id.
pub(super) fn all() -> Vec<Case> {
    let dir = fixtures_dir().join("cases");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", dir.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
            let json = OrderedJson::parse_with_limits(
                &text,
                &JsonLimits::default().with_max_bytes(8 << 20),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let id = path.file_stem().unwrap().to_string_lossy().to_string();
            assert_eq!(
                at(&json, &["id"]).and_then(OrderedJson::as_str),
                Some(id.as_str())
            );
            Case { id, json }
        })
        .collect()
}

/// A JSON file of the fixture directory (`oracle.json`, `artifacts.json`).
pub(super) fn fixture_file(name: &str) -> OrderedJson {
    let path = fixtures_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    OrderedJson::parse_with_limits(&text, &JsonLimits::default().with_max_bytes(8 << 20))
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// An integer of a case.
pub(super) fn int(value: &OrderedJson) -> i64 {
    match value.as_number() {
        Some(JsonNumber::I64(value)) => value,
        _ => panic!("not an integer: {value:?}"),
    }
}

/// An f64 of a case: a number, or `"NaN"`, `"Infinity"`, `"-Infinity"`.
pub(super) fn f64_of(value: &OrderedJson) -> f64 {
    match value.view() {
        JsonView::Number(JsonNumber::F64(value)) => value,
        JsonView::Number(JsonNumber::I64(value)) => value as f64,
        JsonView::String("NaN") => f64::NAN,
        JsonView::String("Infinity") => f64::INFINITY,
        JsonView::String("-Infinity") => f64::NEG_INFINITY,
        _ => panic!("not an f64: {value:?}"),
    }
}

/// An f32 of a case: its bit pattern, `"0x3fab2fdc"`.
pub(super) fn f32_of(value: &OrderedJson) -> f32 {
    let text = value.as_str().expect("an f32 is a hex string");
    let hex = text.strip_prefix("0x").expect("an f32 starts with 0x");
    f32::from_bits(u32::from_str_radix(hex, 16).expect("hex bits"))
}

/// The items of an array of a case.
pub(super) fn items(value: &OrderedJson) -> &[OrderedJson] {
    value.as_array().expect("an array")
}

// ---------------------------------------------------------------------------
// The test transport.

/// The request of a SystemOne body, or why a transport refuses it.
pub(super) fn request(text: &str) -> Result<DecisionRequest, String> {
    let top = members(text)?;
    let mut state = None;
    let mut questions = None;
    for (key, raw) in top {
        match key.as_str() {
            "state" => state = Some(raw),
            "questions" => questions = Some(raw),
            "model" => {}
            other => return Err(format!("transport: unknown field {other:?}")),
        }
    }
    let state = state.ok_or("transport: no state")?;
    let state = DecisionState::from_json(content(state)?);
    let questions = questions.ok_or("transport: no questions")?;
    let questions = members(questions)?
        .into_iter()
        .map(|(id, raw)| question(id, raw))
        .collect::<Result<Vec<_>, _>>()?;
    if questions.is_empty() {
        return Err("transport: no questions".into());
    }
    Ok(DecisionRequest::new(state, questions))
}

fn question(id: String, text: &str) -> Result<DecisionQuestion, String> {
    let mut kind = None;
    let mut instructions = None;
    let mut criteria = None;
    for (key, raw) in members(text)? {
        match key.as_str() {
            "type" => {
                kind = Some(
                    content(raw)?
                        .into_string()
                        .map_err(|_| "type: not a string")?,
                )
            }
            "instructions" => instructions = Some(raw),
            "criteria" => criteria = Some(raw),
            other => return Err(format!("transport: unknown question field {other:?}")),
        }
    }
    let instructions = match instructions {
        Some(raw) => DecisionContent::from_json(content(raw)?),
        None => None,
    };
    let kind = match kind.as_deref() {
        Some("choice") => {
            let raw = criteria.ok_or("choice: no criteria")?;
            let options = members(raw)?
                .into_iter()
                .map(|(key, raw)| {
                    Ok(DecisionOption::new(
                        key,
                        DecisionContent::from_json(content(raw)?),
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            DecisionQuestionKind::choice(options)
        }
        Some("noul") => {
            let (mut yes, mut no) = (None, None);
            if let Some(raw) = criteria.filter(|raw| raw.trim() != "null") {
                for (key, raw) in members(raw)? {
                    let description = DecisionContent::from_json(content(raw)?);
                    match key.as_str() {
                        "true" => yes = description,
                        "false" => no = description,
                        other => return Err(format!("transport: unknown criterion {other:?}")),
                    }
                }
            }
            DecisionQuestionKind::yes_no(yes, no)
        }
        Some("score") => {
            let raw = criteria.ok_or("score: no criteria")?;
            let levels = array_items(raw)?
                .into_iter()
                .map(|raw| {
                    Ok(DecisionLevel::new(DecisionContent::from_json(content(
                        raw,
                    )?)))
                })
                .collect::<Result<Vec<_>, String>>()?;
            DecisionQuestionKind::score(levels)
        }
        _ => return Err("transport: no known type".into()),
    };
    Ok(DecisionQuestion::new(
        id,
        instructions.map(Into::into),
        kind,
    ))
}

/// One content value, parsed on its own.
fn content(raw: &str) -> Result<OrderedJson, String> {
    OrderedJson::parse(raw).map_err(|error| format!("content: {error}"))
}

/// The members of the object `text` holds: each key unescaped, each value
/// as its raw text.
fn members(text: &str) -> Result<Vec<(String, &str)>, String> {
    let mut scanner = Scanner::new(text);
    let members = scanner.object()?;
    scanner.end()?;
    members
        .into_iter()
        .map(|(key, value)| {
            let key = OrderedJson::parse(key)
                .ok()
                .and_then(|key| key.into_string().ok())
                .ok_or("transport: a bad key")?;
            Ok((key, value))
        })
        .collect()
}

/// The raw items of the array `text` holds.
fn array_items(text: &str) -> Result<Vec<&str>, String> {
    let mut scanner = Scanner::new(text);
    let items = scanner.array()?;
    scanner.end()?;
    Ok(items)
}

/// Finds the extent of JSON values without building them, so an outer
/// document can be split into its content values.
struct Scanner<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }

    fn bytes(&self) -> &'a [u8] {
        self.text.as_bytes()
    }

    fn space(&mut self) {
        while matches!(
            self.bytes().get(self.pos),
            Some(b' ' | b'\t' | b'\n' | b'\r')
        ) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> bool {
        self.space();
        if self.bytes().get(self.pos) == Some(&byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn end(&mut self) -> Result<(), String> {
        self.space();
        if self.pos == self.text.len() {
            Ok(())
        } else {
            Err("transport: trailing data".into())
        }
    }

    /// `{ "key": value, ... }`: raw key literals and raw values.
    fn object(&mut self) -> Result<Vec<(&'a str, &'a str)>, String> {
        if !self.eat(b'{') {
            return Err("transport: not an object".into());
        }
        let mut members = Vec::new();
        if self.eat(b'}') {
            return Ok(members);
        }
        loop {
            self.space();
            let key = self.string()?;
            if !self.eat(b':') {
                return Err("transport: no colon".into());
            }
            let value = self.value()?;
            members.push((key, value));
            if self.eat(b',') {
                continue;
            }
            if self.eat(b'}') {
                return Ok(members);
            }
            return Err("transport: bad object".into());
        }
    }

    /// `[ value, ... ]`: raw values.
    fn array(&mut self) -> Result<Vec<&'a str>, String> {
        if !self.eat(b'[') {
            return Err("transport: not an array".into());
        }
        let mut items = Vec::new();
        if self.eat(b']') {
            return Ok(items);
        }
        loop {
            items.push(self.value()?);
            if self.eat(b',') {
                continue;
            }
            if self.eat(b']') {
                return Ok(items);
            }
            return Err("transport: bad array".into());
        }
    }

    /// A string literal, quotes included.
    fn string(&mut self) -> Result<&'a str, String> {
        let start = self.pos;
        if self.bytes().get(self.pos) != Some(&b'"') {
            return Err("transport: not a string".into());
        }
        self.pos += 1;
        while let Some(&byte) = self.bytes().get(self.pos) {
            self.pos += 1;
            match byte {
                b'\\' => self.pos += 1,
                b'"' => return Ok(&self.text[start..self.pos]),
                _ => {}
            }
        }
        Err("transport: unterminated string".into())
    }

    /// Any value's raw text.
    fn value(&mut self) -> Result<&'a str, String> {
        self.space();
        let start = self.pos;
        match self.bytes().get(self.pos) {
            Some(b'"') => {
                self.string()?;
            }
            Some(b'{') => {
                self.object()?;
            }
            Some(b'[') => {
                self.array()?;
            }
            Some(_) => {
                while let Some(&byte) = self.bytes().get(self.pos) {
                    if matches!(byte, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                        break;
                    }
                    self.pos += 1;
                }
            }
            None => return Err("transport: no value".into()),
        }
        Ok(&self.text[start..self.pos])
    }
}
