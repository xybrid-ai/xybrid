//! JSON-Schema → GBNF grammar conversion for constrained LLM decoding.
//!
//! The local llama.cpp backend can constrain token sampling to a
//! [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
//! grammar via `llama_sampler_init_grammar`. That guarantees the model can only
//! emit text the grammar accepts — the foundation for reliable on-device data
//! extraction and structured output from small models, which otherwise drift
//! off valid JSON.
//!
//! llama.cpp ships a C++ `json_schema_to_grammar`, but it lives in the `common`
//! library which this build does not link, so the conversion is done here in
//! Rust against a deliberate **subset** of JSON Schema:
//!
//! - `type`: `object`, `array`, `string`, `integer`, `number`, `boolean`,
//!   `null` — or an array of those (e.g. `["string", "null"]` for nullable)
//! - `object`: `properties` (all emitted, in map order), nested objects
//! - `array`: `items`
//! - `enum`: scalar variants (string / number / boolean / null)
//!
//! Advanced constructs (`$ref`, `oneOf`/`anyOf`/`allOf`, `pattern`, numeric or
//! length bounds, tuple-form arrays) are intentionally unsupported — they would
//! require linking llama.cpp's `common`. See `.context/gbnf-design.md`.
//!
//! # Examples
//!
//! ```
//! use serde_json::json;
//! use xybrid_core::runtime_adapter::grammar::json_schema_to_gbnf;
//!
//! let gbnf = json_schema_to_gbnf(&json!({
//!     "type": "object",
//!     "properties": { "name": { "type": "string" }, "age": { "type": "integer" } }
//! }))
//! .unwrap();
//! assert!(gbnf.starts_with("root ::="));
//! ```

use serde_json::Value;

/// Error raised when a JSON schema cannot be converted to a GBNF grammar.
#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    /// The schema uses a construct this converter does not implement (e.g.
    /// `$ref`, `oneOf`, an unknown `type`).
    #[error("unsupported JSON schema construct: {0}")]
    Unsupported(String),

    /// The schema is structurally invalid (e.g. not an object, missing
    /// `items` on an array).
    #[error("invalid JSON schema: {0}")]
    Invalid(String),
}

/// Shared terminal rules appended to every generated grammar.
///
/// The grammar emits **compact** JSON (no inter-token whitespace). This is
/// deliberate: an optional-whitespace rule like `[ \t\n]*` lets a greedy model
/// emit whitespace indefinitely instead of committing to the next structural
/// token, burning the whole token budget on newlines. Forbidding inter-token
/// whitespace removes that trap; the output is still valid (minified) JSON.
/// Unused terminals are harmless to llama.cpp's grammar parser.
const TERMINALS: &str = "\
string ::= \"\\\"\" ( [^\"\\\\] | \"\\\\\" [\"\\\\/bfnrt] | \"\\\\u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )* \"\\\"\"\n\
integer ::= \"-\"? ( \"0\" | [1-9] [0-9]* )\n\
number ::= \"-\"? ( \"0\" | [1-9] [0-9]* ) ( \".\" [0-9]+ )? ( [eE] [-+]? [0-9]+ )?\n\
boolean ::= \"true\" | \"false\"\n\
null ::= \"null\"\n";

/// Convert a JSON Schema (subset) into a GBNF grammar string suitable for
/// `GenerationConfig::with_grammar` / `llama_sampler_init_grammar`.
///
/// The returned grammar's entry rule is named `root`.
///
/// # Errors
///
/// Returns [`GrammarError::Invalid`] if `schema` is structurally malformed, or
/// [`GrammarError::Unsupported`] if it uses a construct outside the supported
/// subset (see the module docs).
pub fn json_schema_to_gbnf(schema: &Value) -> Result<String, GrammarError> {
    let mut builder = GrammarBuilder::default();
    let root = builder.convert(schema)?;
    Ok(builder.finish(&root))
}

/// Accumulates generated rules and hands out unique rule names.
#[derive(Default)]
struct GrammarBuilder {
    rules: Vec<(String, String)>,
    counter: usize,
}

impl GrammarBuilder {
    /// Register a rule body and return its generated name.
    fn add_rule(&mut self, body: String) -> String {
        let name = format!("rule{}", self.counter);
        self.counter += 1;
        self.rules.push((name.clone(), body));
        name
    }

    /// Convert one schema node, returning the GBNF rule/terminal name that
    /// matches it.
    fn convert(&mut self, schema: &Value) -> Result<String, GrammarError> {
        let obj = schema
            .as_object()
            .ok_or_else(|| GrammarError::Invalid("schema node must be an object".to_string()))?;

        // `enum` short-circuits `type` — match any of the literal variants.
        if let Some(variants) = obj.get("enum") {
            let variants = variants
                .as_array()
                .ok_or_else(|| GrammarError::Invalid("`enum` must be an array".to_string()))?;
            return self.convert_enum(variants);
        }

        match obj.get("type") {
            Some(Value::String(ty)) => self.convert_type(ty, obj),
            // Array-valued `type` (e.g. `["string", "null"]` for a nullable
            // field) becomes an alternation over the listed types.
            Some(Value::Array(types)) => {
                if types.is_empty() {
                    return Err(GrammarError::Invalid(
                        "`type` array must list at least one type".to_string(),
                    ));
                }
                let alts = types
                    .iter()
                    .map(|t| {
                        let ty = t.as_str().ok_or_else(|| {
                            GrammarError::Invalid(
                                "`type` array entries must be strings".to_string(),
                            )
                        })?;
                        self.convert_type(ty, obj)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if alts.len() == 1 {
                    return Ok(alts.into_iter().next().expect("len checked == 1"));
                }
                Ok(self.add_rule(alts.join(" | ")))
            }
            Some(_) => Err(GrammarError::Invalid(
                "`type` must be a string or an array of strings".to_string(),
            )),
            None => Err(GrammarError::Unsupported(
                "schema without `type` or `enum`".to_string(),
            )),
        }
    }

    /// Convert a single named `type` against its schema node.
    fn convert_type(
        &mut self,
        ty: &str,
        obj: &serde_json::Map<String, Value>,
    ) -> Result<String, GrammarError> {
        match ty {
            "string" => Ok("string".to_string()),
            "integer" => Ok("integer".to_string()),
            "number" => Ok("number".to_string()),
            "boolean" => Ok("boolean".to_string()),
            "null" => Ok("null".to_string()),
            "object" => self.convert_object(obj),
            "array" => self.convert_array(obj),
            other => Err(GrammarError::Unsupported(format!("type `{other}`"))),
        }
    }

    fn convert_enum(&mut self, variants: &[Value]) -> Result<String, GrammarError> {
        if variants.is_empty() {
            return Err(GrammarError::Invalid(
                "`enum` must list at least one variant".to_string(),
            ));
        }
        let alts = variants
            .iter()
            .map(json_literal)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.add_rule(alts.join(" | ")))
    }

    fn convert_object(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> Result<String, GrammarError> {
        let props = match obj.get("properties") {
            Some(p) => p.as_object().ok_or_else(|| {
                GrammarError::Invalid("`properties` must be an object".to_string())
            })?,
            // No declared properties → match an empty JSON object.
            None => return Ok(self.add_rule("\"{}\"".to_string())),
        };
        if props.is_empty() {
            return Ok(self.add_rule("\"{}\"".to_string()));
        }

        let mut parts = vec!["\"{\"".to_string()];
        for (i, (key, subschema)) in props.iter().enumerate() {
            let value_rule = self.convert(subschema)?;
            if i > 0 {
                parts.push("\",\"".to_string());
            }
            parts.push(format!("{} \":\" {value_rule}", gbnf_json_key(key)));
        }
        parts.push("\"}\"".to_string());
        Ok(self.add_rule(parts.join(" ")))
    }

    fn convert_array(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> Result<String, GrammarError> {
        let items = obj
            .get("items")
            .ok_or_else(|| GrammarError::Unsupported("array without `items`".to_string()))?;
        let item_rule = self.convert(items)?;
        Ok(self.add_rule(format!(
            "\"[\" ( {item_rule} ( \",\" {item_rule} )* )? \"]\""
        )))
    }

    /// Emit the full grammar text with `root` first, then generated rules, then
    /// the shared terminals.
    fn finish(&self, root: &str) -> String {
        let mut out = format!("root ::= {root}\n");
        for (name, body) in &self.rules {
            out.push_str(&format!("{name} ::= {body}\n"));
        }
        out.push_str(TERMINALS);
        out
    }
}

/// GBNF literal matching the JSON-quoted form of an object key, e.g.
/// `merchant` → `"\"merchant\""`.
///
/// The key is JSON-encoded first (quotes + JSON escapes, including `\u00XX`
/// for control characters) so the grammar matches the *escaped* form the model
/// must emit — never raw control characters, which would be invalid JSON.
/// That JSON text is then escaped once more for the GBNF literal syntax.
fn gbnf_json_key(key: &str) -> String {
    let json = serde_json::to_string(key).expect("serializing a &str to JSON string cannot fail");
    let mut out = String::with_capacity(json.len() + 8);
    out.push('"');
    for ch in json.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// GBNF literal matching a single scalar `enum` variant in its JSON form.
fn json_literal(v: &Value) -> Result<String, GrammarError> {
    match v {
        Value::String(s) => Ok(gbnf_json_key(s)),
        Value::Number(n) => Ok(format!("\"{n}\"")),
        Value::Bool(b) => Ok(format!("\"{b}\"")),
        Value::Null => Ok("\"null\"".to_string()),
        _ => Err(GrammarError::Unsupported(
            "enum variants must be scalars (string / number / boolean / null)".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_emits_root_keys_and_terminals() {
        let gbnf = json_schema_to_gbnf(&json!({
            "type": "object",
            "properties": {
                "merchant": { "type": "string" },
                "total": { "type": "number" }
            }
        }))
        .unwrap();

        assert!(gbnf.starts_with("root ::="));
        assert!(gbnf.contains("\\\"merchant\\\""));
        assert!(gbnf.contains("\\\"total\\\""));
        // Shared terminals are always present.
        assert!(gbnf.contains("string ::="));
        assert!(gbnf.contains("number ::="));
        // Compact JSON: no inter-token whitespace rule (avoids the greedy
        // whitespace-loop trap).
        assert!(!gbnf.contains("ws ::="));
    }

    #[test]
    fn nested_object_and_array() {
        let gbnf = json_schema_to_gbnf(&json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": { "name": { "type": "string" } }
                    }
                }
            }
        }))
        .unwrap();

        // Array produces the bracketed repetition shape.
        assert!(gbnf.contains("\"[\""));
        assert!(gbnf.contains("\\\"name\\\""));
    }

    #[test]
    fn string_enum_becomes_alternation() {
        let gbnf = json_schema_to_gbnf(&json!({
            "type": "object",
            "properties": {
                "status": { "enum": ["open", "closed"] }
            }
        }))
        .unwrap();

        assert!(gbnf.contains("\\\"open\\\""));
        assert!(gbnf.contains("\\\"closed\\\""));
        assert!(gbnf.contains(" | "));
    }

    #[test]
    fn top_level_scalar_root() {
        let gbnf = json_schema_to_gbnf(&json!({ "type": "string" })).unwrap();
        assert!(gbnf.starts_with("root ::= string"));
    }

    #[test]
    fn unknown_type_is_unsupported() {
        let err = json_schema_to_gbnf(&json!({ "type": "tuple" })).unwrap_err();
        assert!(matches!(err, GrammarError::Unsupported(_)));
    }

    #[test]
    fn array_without_items_is_unsupported() {
        let err = json_schema_to_gbnf(&json!({ "type": "array" })).unwrap_err();
        assert!(matches!(err, GrammarError::Unsupported(_)));
    }

    #[test]
    fn non_object_schema_is_invalid() {
        let err = json_schema_to_gbnf(&json!("nope")).unwrap_err();
        assert!(matches!(err, GrammarError::Invalid(_)));
    }

    #[test]
    fn nullable_type_array_becomes_alternation() {
        let gbnf = json_schema_to_gbnf(&json!({
            "type": "object",
            "properties": {
                "nickname": { "type": ["string", "null"] }
            }
        }))
        .unwrap();
        assert!(gbnf.contains("string | null"));
    }

    #[test]
    fn empty_or_non_string_type_array_is_invalid() {
        let err = json_schema_to_gbnf(&json!({ "type": [] })).unwrap_err();
        assert!(matches!(err, GrammarError::Invalid(_)));
        let err = json_schema_to_gbnf(&json!({ "type": [42] })).unwrap_err();
        assert!(matches!(err, GrammarError::Invalid(_)));
    }

    #[test]
    fn non_object_properties_is_invalid() {
        let err = json_schema_to_gbnf(&json!({
            "type": "object",
            "properties": true
        }))
        .unwrap_err();
        assert!(matches!(err, GrammarError::Invalid(_)));
    }

    #[test]
    fn control_chars_in_keys_match_json_escaped_form() {
        let gbnf = json_schema_to_gbnf(&json!({
            "type": "object",
            "properties": {
                "a\nb": { "type": "string" }
            }
        }))
        .unwrap();
        // The grammar must match the JSON-escaped key (`a\nb` with a
        // backslash-n), never a raw newline, which would be invalid JSON.
        assert!(gbnf.contains("a\\\\nb"));
        assert!(!gbnf.contains("a\nb"));
    }

    #[test]
    fn string_terminal_accepts_unicode_escapes() {
        let gbnf = json_schema_to_gbnf(&json!({ "type": "string" })).unwrap();
        assert!(gbnf.contains("[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]"));
    }
}
