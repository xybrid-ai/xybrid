//! Conditional routing for pipeline stages.
//!
//! This module provides expression evaluation for `when` clauses in stage configurations.
//! Expressions reference previous stage outputs using dot notation:
//!
//! ```yaml
//! when: "intent.output.intent == 'weather'"
//! when: "asr.output.confidence > 0.8"
//! when: "classifier.output.label in ['positive', 'neutral']"
//! ```
//!
//! ## Supported Expressions
//!
//! - **Comparison**: `==`, `!=`, `>`, `<`, `>=`, `<=`
//! - **Logical**: `and`, `or`, `not`
//! - **Membership**: `in` (for arrays)
//! - **Existence**: `exists()`, `is_empty()`
//! - **String matching**: `contains()`, `starts_with()`, `ends_with()`

use serde_json::Value;
use std::collections::HashMap;

/// Stage output context for condition evaluation.
#[derive(Debug, Clone, Default)]
pub struct StageOutputContext {
    /// Outputs from completed stages, keyed by stage ID.
    outputs: HashMap<String, Value>,
}

impl StageOutputContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            outputs: HashMap::new(),
        }
    }

    /// Add a stage output to the context.
    pub fn add_output(&mut self, stage_id: &str, output: Value) {
        self.outputs.insert(stage_id.to_string(), output);
    }

    /// Get a stage output by ID.
    pub fn get_output(&self, stage_id: &str) -> Option<&Value> {
        self.outputs.get(stage_id)
    }

    /// Resolve a path like "intent.output.intent" to a value.
    pub fn resolve_path(&self, path: &str) -> Option<&Value> {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() {
            return None;
        }

        // First part is the stage ID
        let stage_id = parts[0];
        let mut current = self.outputs.get(stage_id)?;

        // Navigate through the rest of the path
        for part in &parts[1..] {
            current = match current {
                Value::Object(map) => map.get(*part)?,
                Value::Array(arr) => {
                    // Support numeric indexing for arrays
                    let index: usize = part.parse().ok()?;
                    arr.get(index)?
                }
                _ => return None,
            };
        }

        Some(current)
    }

    /// Check if a stage output exists.
    pub fn has_output(&self, stage_id: &str) -> bool {
        self.outputs.contains_key(stage_id)
    }
}

/// Result of condition evaluation.
#[derive(Debug, Clone)]
pub enum ConditionResult {
    /// Condition evaluated to true - stage should execute.
    True,
    /// Condition evaluated to false - stage should be skipped.
    False,
    /// Condition could not be evaluated (missing data, parse error).
    Error(String),
}

impl ConditionResult {
    /// Check if the condition is satisfied.
    pub fn is_satisfied(&self) -> bool {
        matches!(self, ConditionResult::True)
    }

    /// Check if there was an error.
    pub fn is_error(&self) -> bool {
        matches!(self, ConditionResult::Error(_))
    }
}

/// Condition evaluator for `when` clauses.
pub struct ConditionEvaluator;

impl ConditionEvaluator {
    /// Evaluate a condition expression against stage outputs.
    pub fn evaluate(expr: &str, context: &StageOutputContext) -> ConditionResult {
        let expr = expr.trim();

        // Empty expression is always true
        if expr.is_empty() {
            return ConditionResult::True;
        }

        // Handle logical operators (lowest precedence)
        if let Some(result) = Self::try_logical_expr(expr, context) {
            return result;
        }

        // Handle comparison operators
        if let Some(result) = Self::try_comparison_expr(expr, context) {
            return result;
        }

        // Handle function calls
        if let Some(result) = Self::try_function_expr(expr, context) {
            return result;
        }

        // Handle 'in' operator
        if let Some(result) = Self::try_in_expr(expr, context) {
            return result;
        }

        // Handle boolean path (e.g., "stage.output.is_valid")
        if let Some(result) = Self::try_boolean_path(expr, context) {
            return result;
        }

        ConditionResult::Error(format!("Unable to parse expression: {}", expr))
    }

    /// Try to evaluate logical expressions (and, or, not).
    fn try_logical_expr(expr: &str, context: &StageOutputContext) -> Option<ConditionResult> {
        // Handle 'not' prefix
        if let Some(inner) = expr.strip_prefix("not ") {
            let result = Self::evaluate(inner.trim(), context);
            return Some(match result {
                ConditionResult::True => ConditionResult::False,
                ConditionResult::False => ConditionResult::True,
                ConditionResult::Error(e) => ConditionResult::Error(e),
            });
        }

        // Handle 'and' (with parentheses awareness)
        if let Some((left, right)) = Self::split_binary_op(expr, " and ") {
            let left_result = Self::evaluate(left, context);
            let right_result = Self::evaluate(right, context);
            return Some(match (&left_result, &right_result) {
                (ConditionResult::True, ConditionResult::True) => ConditionResult::True,
                (ConditionResult::Error(e), _) | (_, ConditionResult::Error(e)) => {
                    ConditionResult::Error(e.clone())
                }
                _ => ConditionResult::False,
            });
        }

        // Handle 'or' (with parentheses awareness)
        if let Some((left, right)) = Self::split_binary_op(expr, " or ") {
            let left_result = Self::evaluate(left, context);
            let right_result = Self::evaluate(right, context);
            return Some(match (&left_result, &right_result) {
                (ConditionResult::True, _) | (_, ConditionResult::True) => ConditionResult::True,
                (ConditionResult::Error(e), _) | (_, ConditionResult::Error(e)) => {
                    ConditionResult::Error(e.clone())
                }
                _ => ConditionResult::False,
            });
        }

        None
    }

    /// Try to evaluate comparison expressions.
    fn try_comparison_expr(expr: &str, context: &StageOutputContext) -> Option<ConditionResult> {
        // Order matters: check multi-char operators first
        let operators = ["==", "!=", ">=", "<=", ">", "<"];

        for op in operators {
            if let Some((left, right)) = Self::split_binary_op(expr, op) {
                let left_val = Self::resolve_value(left.trim(), context);
                let right_val = Self::resolve_value(right.trim(), context);

                let (left_val, right_val) = match (left_val, right_val) {
                    (Some(l), Some(r)) => (l, r),
                    _ => {
                        return Some(ConditionResult::Error(format!(
                            "Cannot resolve values in: {}",
                            expr
                        )));
                    }
                };

                let result = match op {
                    "==" => Self::values_equal(&left_val, &right_val),
                    "!=" => !Self::values_equal(&left_val, &right_val),
                    ">" => Self::compare_values(&left_val, &right_val) == Some(std::cmp::Ordering::Greater),
                    "<" => Self::compare_values(&left_val, &right_val) == Some(std::cmp::Ordering::Less),
                    ">=" => matches!(
                        Self::compare_values(&left_val, &right_val),
                        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                    ),
                    "<=" => matches!(
                        Self::compare_values(&left_val, &right_val),
                        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                    ),
                    _ => false,
                };

                return Some(if result {
                    ConditionResult::True
                } else {
                    ConditionResult::False
                });
            }
        }

        None
    }

    /// Try to evaluate 'in' expressions.
    fn try_in_expr(expr: &str, context: &StageOutputContext) -> Option<ConditionResult> {
        if let Some((left, right)) = Self::split_binary_op(expr, " in ") {
            let left_val = Self::resolve_value(left.trim(), context)?;
            let right_val = Self::resolve_value(right.trim(), context)?;

            let result = match right_val {
                Value::Array(arr) => arr.iter().any(|v| Self::values_equal(&left_val, v)),
                Value::String(s) => {
                    if let Value::String(needle) = &left_val {
                        s.contains(needle.as_str())
                    } else {
                        false
                    }
                }
                _ => false,
            };

            return Some(if result {
                ConditionResult::True
            } else {
                ConditionResult::False
            });
        }

        None
    }

    /// Try to evaluate function expressions.
    fn try_function_expr(expr: &str, context: &StageOutputContext) -> Option<ConditionResult> {
        // exists(path)
        if let Some(inner) = Self::extract_function_arg(expr, "exists") {
            let exists = context.resolve_path(inner).is_some();
            return Some(if exists {
                ConditionResult::True
            } else {
                ConditionResult::False
            });
        }

        // is_empty(path)
        if let Some(inner) = Self::extract_function_arg(expr, "is_empty") {
            let value = context.resolve_path(inner);
            let is_empty = match value {
                None => true,
                Some(Value::Null) => true,
                Some(Value::String(s)) => s.is_empty(),
                Some(Value::Array(a)) => a.is_empty(),
                Some(Value::Object(o)) => o.is_empty(),
                _ => false,
            };
            return Some(if is_empty {
                ConditionResult::True
            } else {
                ConditionResult::False
            });
        }

        // contains(path, value)
        if let Some((path, needle)) = Self::extract_two_args(expr, "contains") {
            let value = context.resolve_path(path)?;
            let needle_val = Self::parse_literal(needle)?;

            let contains = match value {
                Value::String(s) => {
                    if let Value::String(n) = &needle_val {
                        s.contains(n.as_str())
                    } else {
                        false
                    }
                }
                Value::Array(arr) => arr.iter().any(|v| Self::values_equal(v, &needle_val)),
                _ => false,
            };

            return Some(if contains {
                ConditionResult::True
            } else {
                ConditionResult::False
            });
        }

        // starts_with(path, prefix)
        if let Some((path, prefix)) = Self::extract_two_args(expr, "starts_with") {
            let value = context.resolve_path(path)?;
            let prefix_val = Self::parse_literal(prefix)?;

            let starts = match (value, &prefix_val) {
                (Value::String(s), Value::String(p)) => s.starts_with(p.as_str()),
                _ => false,
            };

            return Some(if starts {
                ConditionResult::True
            } else {
                ConditionResult::False
            });
        }

        // ends_with(path, suffix)
        if let Some((path, suffix)) = Self::extract_two_args(expr, "ends_with") {
            let value = context.resolve_path(path)?;
            let suffix_val = Self::parse_literal(suffix)?;

            let ends = match (value, &suffix_val) {
                (Value::String(s), Value::String(p)) => s.ends_with(p.as_str()),
                _ => false,
            };

            return Some(if ends {
                ConditionResult::True
            } else {
                ConditionResult::False
            });
        }

        None
    }

    /// Try to evaluate a path as a boolean value.
    fn try_boolean_path(expr: &str, context: &StageOutputContext) -> Option<ConditionResult> {
        let value = context.resolve_path(expr)?;
        let is_truthy = match value {
            Value::Bool(b) => *b,
            Value::Null => false,
            Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        };

        Some(if is_truthy {
            ConditionResult::True
        } else {
            ConditionResult::False
        })
    }

    /// Split expression on a binary operator, respecting parentheses.
    fn split_binary_op<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
        let mut depth = 0;
        let mut bracket_depth = 0;

        for (i, c) in expr.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                _ => {}
            }

            if depth == 0 && bracket_depth == 0 && expr[i..].starts_with(op) {
                return Some((&expr[..i], &expr[i + op.len()..]));
            }
        }

        None
    }

    /// Resolve a value from either a path or a literal.
    fn resolve_value(expr: &str, context: &StageOutputContext) -> Option<Value> {
        let expr = expr.trim();

        // Try as literal first
        if let Some(val) = Self::parse_literal(expr) {
            return Some(val);
        }

        // Try as path
        context.resolve_path(expr).cloned()
    }

    /// Parse a literal value (string, number, boolean, array).
    fn parse_literal(expr: &str) -> Option<Value> {
        let expr = expr.trim();

        // String literal
        if (expr.starts_with('"') && expr.ends_with('"'))
            || (expr.starts_with('\'') && expr.ends_with('\''))
        {
            return Some(Value::String(expr[1..expr.len() - 1].to_string()));
        }

        // Boolean literals
        if expr == "true" {
            return Some(Value::Bool(true));
        }
        if expr == "false" {
            return Some(Value::Bool(false));
        }

        // Null
        if expr == "null" || expr == "nil" {
            return Some(Value::Null);
        }

        // Number
        if let Ok(n) = expr.parse::<i64>() {
            return Some(Value::Number(n.into()));
        }
        if let Ok(n) = expr.parse::<f64>() {
            return Some(serde_json::Number::from_f64(n).map(Value::Number)?);
        }

        // Array literal
        if expr.starts_with('[') && expr.ends_with(']') {
            let inner = &expr[1..expr.len() - 1];
            let items: Vec<Value> = inner
                .split(',')
                .filter_map(|s| Self::parse_literal(s.trim()))
                .collect();
            return Some(Value::Array(items));
        }

        None
    }

    /// Check if two values are equal.
    fn values_equal(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Number(a), Value::Number(b)) => {
                a.as_f64().zip(b.as_f64()).map(|(a, b)| (a - b).abs() < f64::EPSILON).unwrap_or(false)
            }
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Null, Value::Null) => true,
            (Value::Array(a), Value::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| Self::values_equal(x, y))
            }
            _ => false,
        }
    }

    /// Compare two values for ordering.
    fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
        match (a, b) {
            (Value::Number(a), Value::Number(b)) => {
                let a = a.as_f64()?;
                let b = b.as_f64()?;
                a.partial_cmp(&b)
            }
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    /// Extract the argument from a function call like "exists(path)".
    fn extract_function_arg<'a>(expr: &'a str, func_name: &str) -> Option<&'a str> {
        let prefix = format!("{}(", func_name);
        if expr.starts_with(&prefix) && expr.ends_with(')') {
            Some(&expr[prefix.len()..expr.len() - 1])
        } else {
            None
        }
    }

    /// Extract two arguments from a function call like "contains(path, value)".
    fn extract_two_args<'a>(expr: &'a str, func_name: &str) -> Option<(&'a str, &'a str)> {
        let prefix = format!("{}(", func_name);
        if expr.starts_with(&prefix) && expr.ends_with(')') {
            let inner = &expr[prefix.len()..expr.len() - 1];
            let comma_pos = inner.find(',')?;
            Some((inner[..comma_pos].trim(), inner[comma_pos + 1..].trim()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_context() -> StageOutputContext {
        let mut ctx = StageOutputContext::new();
        ctx.add_output(
            "intent",
            json!({
                "output": {
                    "intent": "weather",
                    "confidence": 0.95,
                    "entities": ["location", "time"]
                }
            }),
        );
        ctx.add_output(
            "asr",
            json!({
                "output": {
                    "text": "What's the weather like?",
                    "confidence": 0.88
                }
            }),
        );
        ctx.add_output(
            "classifier",
            json!({
                "output": {
                    "label": "positive",
                    "scores": [0.8, 0.15, 0.05]
                }
            }),
        );
        ctx
    }

    #[test]
    fn test_equality_comparison() {
        let ctx = test_context();

        let result = ConditionEvaluator::evaluate("intent.output.intent == 'weather'", &ctx);
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate("intent.output.intent == 'music'", &ctx);
        assert!(!result.is_satisfied());

        let result = ConditionEvaluator::evaluate("intent.output.intent != 'music'", &ctx);
        assert!(result.is_satisfied());
    }

    #[test]
    fn test_numeric_comparison() {
        let ctx = test_context();

        let result = ConditionEvaluator::evaluate("intent.output.confidence > 0.9", &ctx);
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate("intent.output.confidence < 0.9", &ctx);
        assert!(!result.is_satisfied());

        let result = ConditionEvaluator::evaluate("asr.output.confidence >= 0.88", &ctx);
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate("asr.output.confidence <= 0.88", &ctx);
        assert!(result.is_satisfied());
    }

    #[test]
    fn test_logical_operators() {
        let ctx = test_context();

        // and
        let result = ConditionEvaluator::evaluate(
            "intent.output.intent == 'weather' and intent.output.confidence > 0.9",
            &ctx,
        );
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate(
            "intent.output.intent == 'weather' and intent.output.confidence > 0.99",
            &ctx,
        );
        assert!(!result.is_satisfied());

        // or
        let result = ConditionEvaluator::evaluate(
            "intent.output.intent == 'music' or intent.output.intent == 'weather'",
            &ctx,
        );
        assert!(result.is_satisfied());

        // not
        let result = ConditionEvaluator::evaluate("not intent.output.intent == 'music'", &ctx);
        assert!(result.is_satisfied());
    }

    #[test]
    fn test_in_operator() {
        let ctx = test_context();

        let result = ConditionEvaluator::evaluate(
            "classifier.output.label in ['positive', 'neutral']",
            &ctx,
        );
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate(
            "classifier.output.label in ['negative', 'neutral']",
            &ctx,
        );
        assert!(!result.is_satisfied());

        // String contains
        let result = ConditionEvaluator::evaluate("'location' in intent.output.entities", &ctx);
        assert!(result.is_satisfied());
    }

    #[test]
    fn test_function_exists() {
        let ctx = test_context();

        let result = ConditionEvaluator::evaluate("exists(intent.output.intent)", &ctx);
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate("exists(intent.output.nonexistent)", &ctx);
        assert!(!result.is_satisfied());
    }

    #[test]
    fn test_function_is_empty() {
        let mut ctx = StageOutputContext::new();
        ctx.add_output(
            "test",
            json!({
                "empty_string": "",
                "empty_array": [],
                "non_empty": "hello"
            }),
        );

        let result = ConditionEvaluator::evaluate("is_empty(test.empty_string)", &ctx);
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate("is_empty(test.empty_array)", &ctx);
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate("is_empty(test.non_empty)", &ctx);
        assert!(!result.is_satisfied());
    }

    #[test]
    fn test_function_contains() {
        let ctx = test_context();

        let result =
            ConditionEvaluator::evaluate("contains(asr.output.text, 'weather')", &ctx);
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate("contains(asr.output.text, 'music')", &ctx);
        assert!(!result.is_satisfied());
    }

    #[test]
    fn test_function_starts_ends_with() {
        let ctx = test_context();

        let result = ConditionEvaluator::evaluate("starts_with(asr.output.text, 'What')", &ctx);
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate("ends_with(asr.output.text, '?')", &ctx);
        assert!(result.is_satisfied());
    }

    #[test]
    fn test_boolean_path() {
        let mut ctx = StageOutputContext::new();
        ctx.add_output(
            "validation",
            json!({
                "is_valid": true,
                "is_spam": false
            }),
        );

        let result = ConditionEvaluator::evaluate("validation.is_valid", &ctx);
        assert!(result.is_satisfied());

        let result = ConditionEvaluator::evaluate("validation.is_spam", &ctx);
        assert!(!result.is_satisfied());
    }

    #[test]
    fn test_empty_expression() {
        let ctx = test_context();
        let result = ConditionEvaluator::evaluate("", &ctx);
        assert!(result.is_satisfied());
    }

    #[test]
    fn test_missing_path() {
        let ctx = test_context();
        let result = ConditionEvaluator::evaluate("nonexistent.path == 'value'", &ctx);
        assert!(result.is_error());
    }

    #[test]
    fn test_path_resolution() {
        let ctx = test_context();

        // Nested path
        assert_eq!(
            ctx.resolve_path("intent.output.intent"),
            Some(&Value::String("weather".to_string()))
        );

        // Array access
        assert_eq!(
            ctx.resolve_path("intent.output.entities.0"),
            Some(&Value::String("location".to_string()))
        );
    }
}
