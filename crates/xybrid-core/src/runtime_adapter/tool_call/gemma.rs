use crate::runtime_adapter::AdapterError;
use serde_json::{Map, Number, Value};

use super::cursor::{is_identifier_start, Cursor, ParseError, ParsedCall};
use super::{
    parse_tool_calls, tool_response_content, GEMMA_TOOL_RESPONSE_END, GEMMA_TOOL_RESPONSE_START,
};

const GEMMA_STRING_DELIMITER: &str = "<|\"|>";
const GEMMA_MAX_DEPTH: usize = 16;

pub(super) fn compose_gemma_continuation(
    base_prompt: &str,
    prior_assistant_text: &str,
    responses: &[Value],
) -> Result<String, AdapterError> {
    // `response:NAME{...}` needs the tool name. Prefer the response
    // element's own `name` field (`ToolCallResult` always carries it); fall
    // back to zipping against the prior turn's calls in order for callers
    // that packed bare `{content}` objects.
    let call_names = parse_tool_calls(prior_assistant_text)
        .into_iter()
        .map(|call| call.function.name)
        .collect::<Vec<_>>();
    let mut continuation = String::new();
    continuation.push_str(base_prompt);
    // Gemma-4 models are trained to emit a bare `<|tool_response>` opener
    // right after their call block and stop there — the runtime is expected
    // to fill in the response. Our blocks below bring their own opener, so
    // a trailing dangling opener in the prior text would double the marker.
    let prior_trimmed = prior_assistant_text.trim_end();
    continuation.push_str(
        prior_trimmed
            .strip_suffix(GEMMA_TOOL_RESPONSE_START)
            .unwrap_or(prior_trimmed),
    );

    for (index, response) in responses.iter().enumerate() {
        let content = tool_response_content(response, index)?;
        let name = response
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| call_names.get(index).cloned())
            .ok_or_else(|| {
                AdapterError::InvalidInput(format!(
                    "tool response at index {index} names no tool: include a name field or \
                     answer the prior turn's calls in order"
                ))
            })?;
        continuation.push_str(GEMMA_TOOL_RESPONSE_START);
        continuation.push_str("response:");
        continuation.push_str(&name);
        continuation.push('{');
        format_gemma_response_body(content, &mut continuation);
        continuation.push('}');
        continuation.push_str(GEMMA_TOOL_RESPONSE_END);
    }

    Ok(continuation)
}

impl Cursor<'_> {
    fn parse_complete_gemma_call(mut self) -> Result<ParsedCall, ParseError> {
        self.expect_str("call:")?;
        self.skip_ws();
        let name = self.parse_identifier(true)?;
        self.skip_ws();
        let arguments = self.parse_gemma_arguments()?;
        self.skip_ws();
        if self.is_eof() {
            Ok(ParsedCall {
                name,
                arguments: Value::Object(arguments).to_string(),
            })
        } else {
            Err(ParseError)
        }
    }

    fn parse_gemma_arguments(&mut self) -> Result<Map<String, Value>, ParseError> {
        self.parse_gemma_object(0)
    }

    fn parse_gemma_value(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.skip_ws();
        if self.starts_with(GEMMA_STRING_DELIMITER) {
            return self.parse_gemma_string().map(Value::String);
        }

        match self.peek_char() {
            Some('{') => {
                if depth >= GEMMA_MAX_DEPTH {
                    return Err(ParseError);
                }
                self.parse_gemma_object(depth + 1).map(Value::Object)
            }
            Some('[') => {
                if depth >= GEMMA_MAX_DEPTH {
                    return Err(ParseError);
                }
                self.parse_gemma_list(depth + 1)
            }
            Some('-') => self.parse_gemma_number(),
            Some(ch) if ch.is_ascii_digit() => self.parse_gemma_number(),
            Some(ch) if is_identifier_start(ch) => self.parse_gemma_literal(),
            _ => Err(ParseError),
        }
    }

    fn parse_gemma_object(&mut self, depth: usize) -> Result<Map<String, Value>, ParseError> {
        if depth > GEMMA_MAX_DEPTH {
            return Err(ParseError);
        }
        self.expect_char('{')?;
        let mut map = Map::new();
        self.skip_ws();
        if self.consume_char('}') {
            return Ok(map);
        }

        loop {
            self.skip_ws();
            let key = self.parse_identifier(false)?;
            self.skip_ws();
            self.expect_char(':')?;
            self.skip_ws();
            let value = self.parse_gemma_value(depth)?;
            map.insert(key, value);
            self.skip_ws();

            if self.consume_char(',') {
                continue;
            }
            self.expect_char('}')?;
            return Ok(map);
        }
    }

    fn parse_gemma_list(&mut self, depth: usize) -> Result<Value, ParseError> {
        if depth > GEMMA_MAX_DEPTH {
            return Err(ParseError);
        }
        self.expect_char('[')?;
        let mut values = Vec::new();
        self.skip_ws();
        if self.consume_char(']') {
            return Ok(Value::Array(values));
        }

        loop {
            values.push(self.parse_gemma_value(depth)?);
            self.skip_ws();
            if self.consume_char(',') {
                continue;
            }
            self.expect_char(']')?;
            return Ok(Value::Array(values));
        }
    }

    fn parse_gemma_string(&mut self) -> Result<String, ParseError> {
        self.expect_str(GEMMA_STRING_DELIMITER)?;
        let start = self.pos;
        let offset = self.input[self.pos..]
            .find(GEMMA_STRING_DELIMITER)
            .ok_or(ParseError)?;
        let value = self.input[start..start + offset].to_string();
        self.pos = start + offset + GEMMA_STRING_DELIMITER.len();
        Ok(value)
    }

    fn parse_gemma_number(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        if self.consume_char('-') && self.is_eof() {
            self.pos = start;
            return Err(ParseError);
        }

        if self.consume_digits() == 0 {
            self.pos = start;
            return Err(ParseError);
        }

        let is_float = if self.consume_char('.') {
            if self.consume_digits() == 0 {
                self.pos = start;
                return Err(ParseError);
            }
            true
        } else {
            false
        };

        let literal = &self.input[start..self.pos];
        if is_float {
            let number = literal.parse::<f64>().map_err(|_| ParseError)?;
            Number::from_f64(number)
                .map(Value::Number)
                .ok_or(ParseError)
        } else {
            literal
                .parse::<i64>()
                .map(Number::from)
                .map(Value::Number)
                .map_err(|_| ParseError)
        }
    }

    fn parse_gemma_literal(&mut self) -> Result<Value, ParseError> {
        match self.parse_identifier(false)?.as_str() {
            "true" => Ok(Value::Bool(true)),
            "false" => Ok(Value::Bool(false)),
            "None" | "null" => Ok(Value::Null),
            _ => Err(ParseError),
        }
    }
}

pub(super) fn parse_gemma_block(content: &str) -> Option<ParsedCall> {
    if content.is_empty() {
        return None;
    }

    Cursor::new(content).parse_complete_gemma_call().ok()
}

fn format_gemma_response_body(value: &Value, out: &mut String) {
    if let Value::Object(map) = value {
        format_gemma_object_entries(map, out);
    } else {
        out.push_str("value:");
        format_gemma_value(value, out);
    }
}

fn format_gemma_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => out.push_str(&value.to_string()),
        Value::String(value) => {
            out.push_str(GEMMA_STRING_DELIMITER);
            out.push_str(value);
            out.push_str(GEMMA_STRING_DELIMITER);
        }
        Value::Array(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                format_gemma_value(value, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            format_gemma_object_entries(map, out);
            out.push('}');
        }
    }
}

fn format_gemma_object_entries(map: &Map<String, Value>, out: &mut String) {
    let mut keys = map.keys().collect::<Vec<_>>();
    keys.sort();

    for (index, key) in keys.into_iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(key);
        out.push(':');
        if let Some(value) = map.get(key) {
            format_gemma_value(value, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::ToolCall;
    use crate::runtime_adapter::tool_call::{
        compose_tool_continuation, parse_tool_calls, strip_tool_calls, GEMMA_TOOL_CALL_END,
        GEMMA_TOOL_CALL_START,
    };

    fn gemma_wrapped(content: &str) -> String {
        format!("{GEMMA_TOOL_CALL_START}{content}{GEMMA_TOOL_CALL_END}")
    }

    fn arguments(call: &ToolCall) -> Value {
        serde_json::from_str(&call.function.arguments).unwrap()
    }

    #[test]
    fn parse_gemma_single_call_with_string_arg() {
        let calls = parse_tool_calls(&gemma_wrapped(
            r#"call:get_temperature{room:<|"|>bedroom<|"|>}"#,
        ));

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].function.name, "get_temperature");
        assert_eq!(calls[0].function.arguments, r#"{"room":"bedroom"}"#);
    }

    #[test]
    fn parse_gemma_scalars_when_int_float_negative_bools_and_none() {
        let calls = parse_tool_calls(&gemma_wrapped(
            "call:check{i:2,f:3.5,neg:-7,t:true,b:false,none:None,n:null}",
        ));

        assert_eq!(
            arguments(&calls[0]),
            serde_json::json!({
                "i": 2,
                "f": 3.5,
                "neg": -7,
                "t": true,
                "b": false,
                "none": null,
                "n": null
            })
        );
    }

    #[test]
    fn parse_gemma_nested_map_arg() {
        let calls = parse_tool_calls(&gemma_wrapped(
            r#"call:set_config{config:{mode:<|"|>eco<|"|>,limit:2}}"#,
        ));

        assert_eq!(
            arguments(&calls[0]),
            serde_json::json!({"config": {"mode": "eco", "limit": 2}})
        );
    }

    #[test]
    fn parse_gemma_nested_list_arg_including_strings() {
        let calls = parse_tool_calls(&gemma_wrapped(
            r#"call:plan{items:[<|"|>alpha<|"|>,[<|"|>beta<|"|>,<|"|>gamma<|"|>],3]}"#,
        ));

        assert_eq!(
            arguments(&calls[0]),
            serde_json::json!({"items": ["alpha", ["beta", "gamma"], 3]})
        );
    }

    #[test]
    fn parse_gemma_two_back_to_back_blocks() {
        let text = format!(
            "{}{}",
            gemma_wrapped("call:first{}"),
            gemma_wrapped("call:second{x:2}")
        );

        let calls = parse_tool_calls(&text);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].function.name, "first");
        assert_eq!(calls[0].function.arguments, "{}");
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(calls[1].function.name, "second");
    }

    #[test]
    fn parse_gemma_blocks_when_surrounded_by_prose() {
        let text = format!(
            "before {} between {} after",
            gemma_wrapped("call:first{x:1}"),
            gemma_wrapped("call:second{y:2}")
        );

        let calls = parse_tool_calls(&text);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "first");
        assert_eq!(calls[1].function.name, "second");
    }

    #[test]
    fn parse_gemma_ignores_unclosed_tail_and_strip_removes_it() {
        let text = r#"visible <|tool_call>call:get_temperature{room:<|"|>bedroom"#;

        assert!(parse_tool_calls(text).is_empty());
        assert_eq!(strip_tool_calls(text), "visible");
    }

    #[test]
    fn parse_gemma_skips_content_without_call_prefix() {
        let calls = parse_tool_calls(&gemma_wrapped("note:get_temperature{}"));

        assert!(calls.is_empty());
    }

    #[test]
    fn parse_gemma_skips_depth_bomb_and_keeps_sibling() {
        let mut nested = r#"<|"|>leaf<|"|>"#.to_string();
        for _ in 0..17 {
            nested = format!("{{a:{nested}}}");
        }
        let text = format!(
            "{}{}",
            gemma_wrapped(&format!("call:bomb{{x:{nested}}}")),
            gemma_wrapped("call:ok{x:1}")
        );

        let calls = parse_tool_calls(&text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].function.name, "ok");
    }

    #[test]
    fn parse_gemma_string_preserves_structural_chars() {
        let calls = parse_tool_calls(&gemma_wrapped(
            r#"call:echo{text:<|"|>{brace}, colon: comma, [list]<|"|>}"#,
        ));

        assert_eq!(
            arguments(&calls[0]),
            serde_json::json!({"text": "{brace}, colon: comma, [list]"})
        );
    }

    #[test]
    fn compose_gemma_exact_prompt_when_object_response() {
        let base = "<|turn>user\nhi<turn|>\n<|turn>model\n";
        let prior = gemma_wrapped(r#"call:get_temperature{room:<|"|>bedroom<|"|>}"#);

        let result = compose_tool_continuation(
            base,
            &prior,
            r#"[{"content":{"temperature_c":17.5,"room":"bedroom"}}]"#,
        )
        .unwrap();

        assert_eq!(
            result,
            format!(
                "{base}{prior}<|tool_response>response:get_temperature{{room:<|\"|>bedroom<|\"|>,temperature_c:17.5}}<tool_response|>"
            )
        );
    }

    #[test]
    fn compose_gemma_wraps_non_object_string_response() {
        let base = "<|turn>user\nhi<turn|>\n<|turn>model\n";
        let prior = gemma_wrapped("call:say{}");

        let result = compose_tool_continuation(base, &prior, r#"[{"content":"done"}]"#).unwrap();

        assert_eq!(
            result,
            format!(
                "{base}{prior}<|tool_response>response:say{{value:<|\"|>done<|\"|>}}<tool_response|>"
            )
        );
    }

    #[test]
    fn compose_gemma_preserves_response_order() {
        let base = "<|turn>user\nhi<turn|>\n<|turn>model\n";
        let prior = format!(
            "{}{}",
            gemma_wrapped("call:first{}"),
            gemma_wrapped("call:second{}")
        );

        let result = compose_tool_continuation(
            base,
            &prior,
            r#"[{"content":{"ok":true}},{"content":null}]"#,
        )
        .unwrap();

        assert_eq!(
            result,
            format!(
                "{base}{prior}<|tool_response>response:first{{ok:true}}<tool_response|><|tool_response>response:second{{value:None}}<tool_response|>"
            )
        );
    }

    #[test]
    fn compose_gemma_trims_dangling_response_opener_from_prior() {
        let prior = r#"<|tool_call>call:get_temperature{room:<|"|>bedroom<|"|>}<tool_call|><|tool_response>"#;
        let result = compose_tool_continuation(
            "<|turn>model\n",
            prior,
            r#"[{"name":"get_temperature","content":{"temperature_c":17.5}}]"#,
        )
        .unwrap();

        assert_eq!(
            result,
            "<|turn>model\n<|tool_call>call:get_temperature{room:<|\"|>bedroom<|\"|>}<tool_call|><|tool_response>response:get_temperature{temperature_c:17.5}<tool_response|>"
        );
        assert_eq!(
            result.matches(GEMMA_TOOL_RESPONSE_START).count(),
            1,
            "the dangling opener from the prior turn must not double"
        );
    }

    #[test]
    fn strip_removes_dangling_gemma_response_opener() {
        let text = r#"ok<|tool_call>call:f{x:1}<tool_call|><|tool_response>"#;
        assert_eq!(strip_tool_calls(text), "ok");
    }

    #[test]
    fn strip_removes_complete_gemma_response_block() {
        let text = "before <|tool_response>response:f{ok:true}<tool_response|> after";
        assert_eq!(strip_tool_calls(text), "before  after");
    }
}
