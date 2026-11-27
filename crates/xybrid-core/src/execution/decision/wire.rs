//! Strict, ordered Liquid SystemOne JSON transport decoding.
//!
//! Wrappers are sliced before content is parsed. Consequently a field's depth
//! is independent of the number of transport objects surrounding it.

use super::liquid::{validate_wire_request, DecisionProtocolError as Error};
use crate::ir::decision::{
    DecisionContent, DecisionLevel, DecisionOption, DecisionQuestion, DecisionQuestionKind,
    DecisionRequest, DecisionState, JsonLimits, OrderedJson,
};
use std::collections::HashSet;

/// Maximum raw request size, checked before parsing or allocation.
pub const MAX_RAW_BYTES: usize = 8 << 20;

/// Decodes the Liquid wire format, never guessing between it and serialized IR.
///
/// Duplicate keys and unknown fields are refused without quoting their values.
/// The optional `model` field is validated as JSON and ignored.
pub fn decode(text: &str) -> Result<DecisionRequest, Error> {
    if text.len() > MAX_RAW_BYTES {
        return Err(Error::BudgetExceeded);
    }
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let body = object(text, Error::InvalidRequest)?;
    let mut state = None;
    let mut questions = None;
    for (key, raw) in body {
        match key.as_str() {
            "state" => state = Some(DecisionState::from_json(content_json(raw)?)),
            "questions" => questions = Some(decode_questions(raw)?),
            "model" => {
                content_json(raw)?;
            }
            "files" | "images" | "videos" => return Err(Error::MediaUnsupported),
            _ => return Err(Error::UnknownField),
        }
    }
    let request = DecisionRequest::new(
        state.ok_or(Error::MissingState)?,
        questions.ok_or(Error::InvalidQuestions)?,
    );
    validate_wire_request(&request)?;
    Ok(request)
}

fn content_json(raw: &str) -> Result<OrderedJson, Error> {
    OrderedJson::parse_with_limits(raw, &JsonLimits::default().with_max_bytes(MAX_RAW_BYTES))
        .map_err(Into::into)
}

fn content(raw: &str) -> Result<Option<DecisionContent>, Error> {
    Ok(DecisionContent::from_json(content_json(raw)?))
}

fn decode_questions(raw: &str) -> Result<Vec<DecisionQuestion>, Error> {
    let fields = object(raw, Error::InvalidQuestions)?;
    if fields.is_empty() {
        return Err(Error::InvalidQuestions);
    }
    if fields.len() > 16 {
        return Err(Error::BudgetExceeded);
    }
    let mut questions = Vec::with_capacity(fields.len());
    for (id, raw) in fields {
        if id.is_empty() {
            return Err(Error::EmptyQuestionId);
        }
        let fields = object(raw, Error::InvalidQuestion)?;
        let mut instructions = None;
        let mut kind = None;
        let mut criteria = "null";
        for (key, value) in fields {
            match key.as_str() {
                "instructions" => instructions = content(value)?,
                "type" => kind = content_json(value)?.into_string().ok(),
                "criteria" => criteria = value,
                _ => return Err(Error::UnknownField),
            }
        }
        let instructions = instructions.ok_or(Error::MissingInstructions)?;
        let kind = match kind.as_deref() {
            Some("choice") => {
                let fields = object(criteria, Error::InvalidCriteria)?;
                if fields.is_empty() {
                    return Err(Error::InvalidCriteria);
                }
                if fields.len() > 255 {
                    return Err(Error::TooManyOptions);
                }
                let mut options = Vec::with_capacity(fields.len());
                for (key, raw) in fields {
                    if key.is_empty() {
                        return Err(Error::EmptyOptionKey);
                    }
                    options.push(DecisionOption::new(key, content(raw)?));
                }
                DecisionQuestionKind::choice(options)
            }
            Some("noul") => {
                let mut yes = None;
                let mut no = None;
                if criteria.trim() != "null" {
                    for (key, raw) in object(criteria, Error::InvalidCriteria)? {
                        match key.as_str() {
                            "true" => yes = content(raw)?,
                            "false" => no = content(raw)?,
                            _ => return Err(Error::UnknownField),
                        }
                    }
                }
                DecisionQuestionKind::yes_no(yes, no)
            }
            Some("score") => {
                let values = array(criteria)?;
                if !(2..=10).contains(&values.len()) {
                    return Err(Error::LevelCount);
                }
                let levels = values
                    .into_iter()
                    .map(|raw| content(raw).map(DecisionLevel::new))
                    .collect::<Result<_, _>>()?;
                DecisionQuestionKind::score(levels)
            }
            _ => return Err(Error::InvalidQuestionType),
        };
        questions.push(DecisionQuestion::new(id, Some(instructions.into()), kind));
    }
    Ok(questions)
}

// This scanner only slices wrappers; OrderedJson validates every raw content
// value. The stack is bounded even for malformed or unused transport fields.
struct Scanner<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }
    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }
    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }
    fn take(&mut self, byte: u8) -> Result<(), Error> {
        self.whitespace();
        if self.peek() != Some(byte) {
            return Err(Error::InvalidJson);
        }
        self.pos += 1;
        Ok(())
    }
    fn string_end(&mut self) -> Result<(), Error> {
        self.take(b'"')?;
        loop {
            match self.peek() {
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(());
                }
                Some(b'\\') => {
                    self.pos += 1;
                    if self.peek().is_none() {
                        return Err(Error::InvalidJson);
                    }
                    self.pos += 1;
                }
                Some(0..=31) | None => return Err(Error::InvalidJson),
                Some(_) => self.pos += 1,
            }
        }
    }
    fn raw_value(&mut self) -> Result<&'a str, Error> {
        self.whitespace();
        let start = self.pos;
        match self.peek() {
            Some(b'"') => self.string_end()?,
            Some(b'{' | b'[') => {
                let mut stack = Vec::with_capacity(16);
                loop {
                    match self.peek() {
                        Some(b'"') => self.string_end()?,
                        Some(b'{' | b'[') => {
                            if stack.len() == 132 {
                                return Err(Error::JsonTooDeep);
                            }
                            stack.push(self.peek().ok_or(Error::InvalidJson)?);
                            self.pos += 1;
                        }
                        Some(b'}' | b']') => {
                            let expected = if self.peek() == Some(b'}') {
                                b'{'
                            } else {
                                b'['
                            };
                            if stack.pop() != Some(expected) {
                                return Err(Error::InvalidJson);
                            }
                            self.pos += 1;
                            if stack.is_empty() {
                                break;
                            }
                        }
                        None => return Err(Error::InvalidJson),
                        Some(_) => self.pos += 1,
                    }
                }
            }
            Some(b'}' | b']' | b',') | None => return Err(Error::InvalidJson),
            Some(_) => {
                while !matches!(
                    self.peek(),
                    None | Some(b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
                ) {
                    self.pos += 1;
                }
            }
        }
        Ok(&self.text[start..self.pos])
    }
    fn end(mut self) -> Result<(), Error> {
        self.whitespace();
        if self.pos != self.text.len() {
            return Err(Error::InvalidJson);
        }
        Ok(())
    }
}

fn object(text: &str, shape_error: Error) -> Result<Vec<(String, &str)>, Error> {
    let mut scanner = Scanner::new(text);
    scanner.whitespace();
    if scanner.peek() != Some(b'{') {
        content_json(text)?;
        return Err(shape_error);
    }
    scanner.take(b'{')?;
    let mut fields = Vec::new();
    let mut keys = HashSet::new();
    scanner.whitespace();
    if scanner.peek() == Some(b'}') {
        scanner.pos += 1;
        scanner.end()?;
        return Ok(fields);
    }
    loop {
        scanner.whitespace();
        let start = scanner.pos;
        scanner.string_end()?;
        let key: String =
            serde_json::from_str(&text[start..scanner.pos]).map_err(|_| Error::InvalidJson)?;
        if !keys.insert(key.clone()) {
            return Err(Error::DuplicateKey);
        }
        scanner.take(b':')?;
        let raw = scanner.raw_value()?;
        fields.push((key, raw));
        if fields.len() > 65536 {
            return Err(Error::BudgetExceeded);
        }
        scanner.whitespace();
        match scanner.peek() {
            Some(b',') => {
                scanner.pos += 1;
            }
            Some(b'}') => {
                scanner.pos += 1;
                break;
            }
            _ => return Err(Error::InvalidJson),
        }
    }
    scanner.end()?;
    Ok(fields)
}

fn array(text: &str) -> Result<Vec<&str>, Error> {
    let mut scanner = Scanner::new(text);
    scanner.whitespace();
    if scanner.peek() != Some(b'[') {
        content_json(text)?;
        return Err(Error::InvalidCriteria);
    }
    scanner.take(b'[')?;
    let mut values = Vec::new();
    scanner.whitespace();
    if scanner.peek() == Some(b']') {
        scanner.pos += 1;
        scanner.end()?;
        return Ok(values);
    }
    loop {
        values.push(scanner.raw_value()?);
        if values.len() > 65536 {
            return Err(Error::BudgetExceeded);
        }
        scanner.whitespace();
        match scanner.peek() {
            Some(b',') => {
                scanner.pos += 1;
            }
            Some(b']') => {
                scanner.pos += 1;
                break;
            }
            _ => return Err(Error::InvalidJson),
        }
    }
    scanner.end()?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_order_and_null_empty_and_numeric_identity() {
        let request = decode(r#"{"state":{"z":1,"a":[1.0,-0.0,9007199254740993]},"model":"ignored","questions":{"b":{"type":"choice","instructions":"","criteria":{"z":null,"a":""}},"a":{"type":"noul","instructions":false}}}"#).unwrap();
        assert_eq!(request.questions()[0].id(), "b");
        assert_eq!(request.questions()[1].id(), "a");
        assert_eq!(
            request
                .state()
                .body()
                .unwrap()
                .as_json()
                .unwrap()
                .to_json_text(),
            r#"{"z":1,"a":[1.0,-0.0,9007199254740993]}"#
        );
        let options = request.questions()[0].kind().as_choice().unwrap();
        assert!(options[0].description().is_none());
        assert_eq!(options[1].description().unwrap().as_text(), Some(""));
    }

    #[test]
    fn refuses_duplicates_at_every_layer_and_unknown_fields() {
        for text in [
            r#"{"state":null,"state":null,"questions":{}}"#,
            r#"{"state":{"a":0,"a":1},"questions":{}}"#,
            r#"{"state":null,"questions":{"q":{},"q":{}}}"#,
            r#"{"state":null,"questions":{"q":{"type":"noul","type":"score"}}}"#,
            r#"{"state":null,"questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":1,"a":2}}}}"#,
        ] {
            assert_eq!(decode(text).unwrap_err(), Error::DuplicateKey);
        }
        assert_eq!(
            decode(r#"{"unknown":"PRIVATE"}"#).unwrap_err(),
            Error::UnknownField
        );
        assert_eq!(
            decode(r#"{"files":[]}"#).unwrap_err(),
            Error::MediaUnsupported
        );
    }

    #[test]
    fn field_depth_ignores_transport_wrappers() {
        for depth in [126, 127, 128] {
            let value = format!("{}0{}", "[".repeat(depth - 1), "]".repeat(depth - 1));
            let text = format!(
                r#"{{"state":null,"questions":{{"q":{{"instructions":{value},"type":"noul"}}}}}}"#
            );
            let request = decode(&text).unwrap();
            assert_eq!(
                super::super::liquid::validate_request(&request).is_ok(),
                depth <= 127,
                "depth {depth}"
            );
        }
    }

    #[test]
    fn oversized_transport_is_refused_before_parsing() {
        assert_eq!(
            decode(&" ".repeat(MAX_RAW_BYTES + 1)).unwrap_err(),
            Error::BudgetExceeded
        );
    }

    #[test]
    fn syntax_failures_do_not_include_content() {
        for text in [
            "{\"PRIVATE\":",
            "{\"state\":null,}",
            "[0] junk",
            "{\"state\": [1,]}",
        ] {
            let error = decode(text).unwrap_err();
            assert!(!format!("{error:?} {error}").contains("PRIVATE"));
        }
    }
}
