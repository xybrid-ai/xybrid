//! Policy Engine module - Enforces data-handling and routing rules before inference stages run.
//!
//! A policy bundle is a small, declarative rule set (YAML or JSON) that the
//! orchestration authority evaluates once per stage decision, against the
//! *actual* input envelope and one live device-metrics snapshot. The outcome
//! constrains routing:
//!
//! - `deny`        — cloud execution is forbidden; the stage must run locally.
//! - `route_cloud` — cloud execution is preferred when it is permitted and a
//!   cloud leg exists (alias: `prefer_cloud`).
//! - `redact`      — the input would need a transform before leaving the
//!   device; transforms are not implemented, so this also forces local
//!   execution.
//! - `allow`       — stop evaluating; route normally.
//!
//! Rules are evaluated in order and the **first matching rule wins**.
//!
//! # Expression language
//!
//! Each rule has exactly one expression of the form `<lhs> <op> <rhs>`, or the
//! bare literal `true` / `false`.
//!
//! | Left operand              | Operators                      | Right operand                                                                        |
//! |---------------------------|--------------------------------|--------------------------------------------------------------------------------------|
//! | `input.kind`              | `==` `!=`                      | `"audio"` `"text"` `"embedding"` `"image"` `"multipart"` (legacy alias `"audioraw"`) |
//! | `input.text`              | `contains` `matches` `==` `!=` | `"literal"` (`matches` takes a regex)                                                |
//! | `input.text_len`          | `==` `!=` `<` `<=` `>` `>=`    | number (Unicode scalar values)                                                       |
//! | `metrics.battery_level`   | `==` `!=` `<` `<=` `>` `>=`    | number, 0–100                                                                        |
//! | `metrics.cpu_pct`         | `==` `!=` `<` `<=` `>` `>=`    | number, 0–100                                                                        |
//! | `metrics.memory_pressure` | `==` `!=`                      | `"unknown"` `"normal"` `"warn"` `"critical"`                                         |
//! | `metrics.thermal_state`   | `==` `!=`                      | `"normal"` `"warm"` `"hot"` `"critical"`                                             |
//!
//! String comparisons on enum-like operands are case-insensitive. `contains`
//! is a case-sensitive substring test; use `matches` with `(?i)` for
//! case-insensitive matching. Inside a quoted literal `\"` and `\\` are
//! decoded; any other backslash sequence is kept verbatim, so regex escapes
//! such as `\d` pass through unchanged. Text rules inspect only the input `Text`
//! payload — never system prompts or metadata — and never match non-text
//! input. A missing CPU reading never matches. Unsupported operands,
//! operators, literals, or actions are rejected when the bundle is loaded,
//! not silently ignored at evaluation time.
//!
//! # Bundle shape
//!
//! ```yaml
//! version: "1.0.0"          # optional
//! signature: "unsigned"     # optional, not verified
//! rules:                    # optional, evaluated first, in order
//!   - id: keep_audio_on_device
//!     expression: 'input.kind == "audio"'
//!     action: deny
//! deny_cloud_if:            # optional shorthand: each entry is a deny rule
//!   - 'input.text matches "(?i)password"'
//! route_cloud_if:           # optional shorthand: each entry is a route_cloud rule
//!   - "metrics.battery_level < 25"
//! ```
//!
//! Shorthand rules are appended after `rules` in the order shown, so an
//! earlier explicit `allow` or `route_cloud` can stop a later shorthand
//! denial — the priority of `deny_cloud_if` over `route_cloud_if` is an
//! ordering convention, not a global override.

use crate::context::DeviceMetrics;
use crate::ir::{Envelope, EnvelopeKind};
use regex::Regex;
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

/// Routing preference expressed by a matched rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolicyRoute {
    /// No preference; the routing ladder decides.
    #[default]
    Unspecified,
    /// Prefer the cloud leg when cloud execution is permitted and available.
    PreferCloud,
}

/// Result of a policy evaluation.
#[derive(Debug, Clone)]
pub struct PolicyResult {
    /// `false` means cloud execution is forbidden for this input.
    pub allowed: bool,
    /// Human-readable explanation of the decision.
    pub reason: Option<String>,
    /// Identifiers of `redact` rules that matched. Non-empty means the input
    /// would need a transform before leaving the device; since transforms are
    /// not implemented, a non-empty list forces local execution.
    pub transforms_applied: Vec<String>,
    /// Routing preference expressed by the matched rule, if any.
    pub route: PolicyRoute,
}

impl PolicyResult {
    /// Create a new PolicyResult with no routing preference.
    pub fn new(allowed: bool, reason: Option<String>) -> Self {
        Self {
            allowed,
            reason,
            transforms_applied: Vec::new(),
            route: PolicyRoute::Unspecified,
        }
    }

    /// Create an allowed result.
    pub fn allow(reason: Option<String>) -> Self {
        Self::new(true, reason)
    }

    /// Create a denied result: cloud execution is forbidden.
    pub fn deny(reason: String) -> Self {
        Self::new(false, Some(reason))
    }

    /// Create an allowed result that prefers the cloud leg.
    pub fn prefer_cloud(reason: String) -> Self {
        let mut result = Self::new(true, Some(reason));
        result.route = PolicyRoute::PreferCloud;
        result
    }

    /// True when the matched rule prefers cloud execution.
    pub fn prefers_cloud(&self) -> bool {
        self.route == PolicyRoute::PreferCloud
    }

    /// True when the stage must not leave the device: cloud was denied, or
    /// the input requires a transform that no component can apply.
    pub fn requires_local(&self) -> bool {
        !self.allowed || !self.transforms_applied.is_empty()
    }
}

/// Action taken when a rule's expression matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyAction {
    /// Stop evaluating and allow normal routing.
    Allow,
    /// Forbid cloud execution for this input.
    Deny,
    /// Prefer cloud execution when it is permitted and available.
    RouteCloud,
    /// Require a transform before the input may leave the device.
    Redact,
}

impl PolicyAction {
    /// Canonical lowercase name used in policy files.
    pub fn as_str(&self) -> &'static str {
        match self {
            PolicyAction::Allow => "allow",
            PolicyAction::Deny => "deny",
            PolicyAction::RouteCloud => "route_cloud",
            PolicyAction::Redact => "redact",
        }
    }
}

impl fmt::Display for PolicyAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PolicyAction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "allow" => Ok(PolicyAction::Allow),
            "deny" => Ok(PolicyAction::Deny),
            "route_cloud" | "prefer_cloud" => Ok(PolicyAction::RouteCloud),
            "redact" => Ok(PolicyAction::Redact),
            other => Err(format!(
                "unknown policy action '{other}' (expected allow, deny, route_cloud, or redact)"
            )),
        }
    }
}

/// Individual policy rule.
#[derive(Debug, Clone)]
pub struct PolicyRule {
    /// Unique identifier, surfaced in decision reasons and telemetry.
    pub id: String,
    /// Source text of the expression (see the module docs for the grammar).
    pub expression: String,
    /// Action taken when the expression matches.
    pub action: PolicyAction,
}

/// Policy bundle containing rules and metadata.
#[derive(Debug, Clone)]
pub struct PolicyBundle {
    pub version: String,
    pub rules: Vec<PolicyRule>,
    /// Carried through from the bundle; not verified.
    pub signature: String,
}

/// Policy Engine trait for evaluating policies.
pub trait PolicyEngine {
    /// Parse and activate a policy bundle (YAML or JSON bytes).
    ///
    /// The whole bundle is validated and compiled before it replaces the
    /// active one; on error the previously loaded policy stays in effect.
    fn load_policies(&mut self, bundle_bytes: Vec<u8>) -> Result<(), String>;

    /// Evaluate policy conditions for a stage against the actual input and
    /// the supplied device metrics.
    fn evaluate(&self, stage: &str, envelope: &Envelope, metrics: &DeviceMetrics) -> PolicyResult;

    /// Apply redaction transforms to an envelope. Returns whether the
    /// envelope changed. Not implemented: always `false`, and never relied
    /// on to authorize cloud transmission.
    fn redact(&self, envelope: &mut Envelope) -> bool;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compiled expressions
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lhs {
    InputKind,
    InputText,
    InputTextLen,
    BatteryLevel,
    CpuPct,
    MemoryPressure,
    ThermalState,
}

impl Lhs {
    fn parse(token: &str) -> Result<Self, String> {
        match token {
            "input.kind" => Ok(Lhs::InputKind),
            "input.text" => Ok(Lhs::InputText),
            "input.text_len" => Ok(Lhs::InputTextLen),
            "metrics.battery_level" => Ok(Lhs::BatteryLevel),
            "metrics.cpu_pct" => Ok(Lhs::CpuPct),
            "metrics.memory_pressure" => Ok(Lhs::MemoryPressure),
            "metrics.thermal_state" => Ok(Lhs::ThermalState),
            other => Err(format!(
                "unknown operand '{other}' (expected input.kind, input.text, input.text_len, \
                 metrics.battery_level, metrics.cpu_pct, metrics.memory_pressure, or \
                 metrics.thermal_state)"
            )),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Lhs::InputKind => "input.kind",
            Lhs::InputText => "input.text",
            Lhs::InputTextLen => "input.text_len",
            Lhs::BatteryLevel => "metrics.battery_level",
            Lhs::CpuPct => "metrics.cpu_pct",
            Lhs::MemoryPressure => "metrics.memory_pressure",
            Lhs::ThermalState => "metrics.thermal_state",
        }
    }

    fn is_numeric(&self) -> bool {
        matches!(self, Lhs::InputTextLen | Lhs::BatteryLevel | Lhs::CpuPct)
    }

    /// Accepted literals for enum-like operands (lowercase).
    fn enum_values(&self) -> Option<&'static [&'static str]> {
        match self {
            Lhs::InputKind => Some(&["audio", "text", "embedding", "image", "multipart"]),
            Lhs::MemoryPressure => Some(&["unknown", "normal", "warn", "critical"]),
            Lhs::ThermalState => Some(&["normal", "warm", "hot", "critical"]),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Contains,
    Matches,
}

impl Op {
    fn parse(token: &str) -> Result<Self, String> {
        match token {
            "==" => Ok(Op::Eq),
            "!=" => Ok(Op::Ne),
            "<" => Ok(Op::Lt),
            "<=" => Ok(Op::Le),
            ">" => Ok(Op::Gt),
            ">=" => Ok(Op::Ge),
            "contains" => Ok(Op::Contains),
            "matches" => Ok(Op::Matches),
            other => Err(format!(
                "unknown operator '{other}' (expected ==, !=, <, <=, >, >=, contains, or matches)"
            )),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Op::Eq => "==",
            Op::Ne => "!=",
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Gt => ">",
            Op::Ge => ">=",
            Op::Contains => "contains",
            Op::Matches => "matches",
        }
    }

    fn is_equality(&self) -> bool {
        matches!(self, Op::Eq | Op::Ne)
    }

    fn is_numeric(&self) -> bool {
        matches!(self, Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge)
    }
}

#[derive(Debug, Clone)]
enum Rhs {
    Str(String),
    Num(f64),
    Regex(Regex),
}

#[derive(Debug, Clone)]
enum Expr {
    Literal(bool),
    Compare { lhs: Lhs, op: Op, rhs: Rhs },
}

enum RhsToken {
    Quoted(String),
    Bare(String),
}

/// Split off the first whitespace-delimited token; the remainder is
/// left-trimmed.
fn split_head(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    match s.find(char::is_whitespace) {
        Some(idx) => (&s[..idx], s[idx..].trim_start()),
        None => (s, ""),
    }
}

/// Parse a double-quoted, backslash-escaped string literal at the start of
/// `s`. Returns the decoded value and the remainder after the closing quote.
fn parse_quoted(s: &str) -> Result<(String, &str), String> {
    let mut chars = s.char_indices();
    match chars.next() {
        Some((_, '"')) => {}
        _ => return Err("expected a double-quoted string".to_string()),
    }
    let mut out = String::new();
    while let Some((idx, ch)) = chars.next() {
        match ch {
            '\\' => match chars.next() {
                Some((_, '"')) => out.push('"'),
                Some((_, '\\')) => out.push('\\'),
                // Any other backslash sequence is preserved verbatim so regex
                // escapes (`\d`, `\s`, `\.`) work without double-escaping.
                Some((_, other)) => {
                    out.push('\\');
                    out.push(other);
                }
                None => return Err("unterminated string literal".to_string()),
            },
            '"' => return Ok((out, &s[idx + ch.len_utf8()..])),
            other => out.push(other),
        }
    }
    Err("unterminated string literal".to_string())
}

fn tokenize(expression: &str) -> Result<(&str, &str, RhsToken), String> {
    let (lhs, rest) = split_head(expression);
    if rest.is_empty() {
        return Err(format!(
            "expected '<operand> <operator> <value>' or a bare true/false, got '{expression}'"
        ));
    }
    let (op, rest) = split_head(rest);
    if rest.is_empty() {
        return Err("missing right operand".to_string());
    }
    if rest.starts_with('"') {
        let (value, remainder) = parse_quoted(rest)?;
        if !remainder.trim().is_empty() {
            return Err(format!(
                "unexpected trailing input '{}' after string literal",
                remainder.trim()
            ));
        }
        Ok((lhs, op, RhsToken::Quoted(value)))
    } else {
        let (value, remainder) = split_head(rest);
        if !remainder.is_empty() {
            return Err(format!("unexpected trailing input '{remainder}'"));
        }
        Ok((lhs, op, RhsToken::Bare(value.to_string())))
    }
}

/// Compile one expression, enforcing operand/operator/literal compatibility.
fn compile_expression(expression: &str) -> Result<Expr, String> {
    let trimmed = expression.trim();
    match trimmed {
        "" => return Err("expression is empty".to_string()),
        "true" => return Ok(Expr::Literal(true)),
        "false" => return Ok(Expr::Literal(false)),
        _ => {}
    }

    let (lhs_token, op_token, rhs_token) = tokenize(trimmed)?;
    let lhs = Lhs::parse(lhs_token)?;
    let op = Op::parse(op_token)?;

    let rhs = if lhs.is_numeric() {
        if !op.is_numeric() {
            return Err(format!(
                "operator '{}' is not valid for numeric operand '{}'",
                op.as_str(),
                lhs.as_str()
            ));
        }
        let RhsToken::Bare(raw) = rhs_token else {
            return Err(format!(
                "operand '{}' must be compared against a number, not a quoted string",
                lhs.as_str()
            ));
        };
        let number: f64 = raw
            .parse()
            .map_err(|_| format!("'{raw}' is not a number"))?;
        if !number.is_finite() {
            return Err(format!("'{raw}' is not a finite number"));
        }
        Rhs::Num(number)
    } else if let Some(allowed) = lhs.enum_values() {
        if !op.is_equality() {
            return Err(format!(
                "operator '{}' is not valid for operand '{}' (use == or !=)",
                op.as_str(),
                lhs.as_str()
            ));
        }
        let RhsToken::Quoted(raw) = rhs_token else {
            return Err(format!(
                "operand '{}' must be compared against a quoted string",
                lhs.as_str()
            ));
        };
        let mut normalized = raw.to_ascii_lowercase();
        // Legacy label from the MVP policy format.
        if lhs == Lhs::InputKind && normalized == "audioraw" {
            normalized = "audio".to_string();
        }
        if !allowed.contains(&normalized.as_str()) {
            let expected = if lhs == Lhs::InputKind {
                format!("{} (legacy alias: audioraw)", allowed.join(", "))
            } else {
                allowed.join(", ")
            };
            return Err(format!(
                "'{raw}' is not a valid value for '{}' (expected one of: {expected})",
                lhs.as_str(),
            ));
        }
        Rhs::Str(normalized)
    } else {
        debug_assert_eq!(lhs, Lhs::InputText);
        let RhsToken::Quoted(raw) = rhs_token else {
            return Err(format!(
                "operand '{}' must be compared against a quoted string",
                lhs.as_str()
            ));
        };
        match op {
            Op::Matches => {
                Rhs::Regex(Regex::new(&raw).map_err(|e| format!("invalid regex '{raw}': {e}"))?)
            }
            Op::Eq | Op::Ne | Op::Contains => Rhs::Str(raw),
            other => {
                return Err(format!(
                "operator '{}' is not valid for operand '{}' (use ==, !=, contains, or matches)",
                other.as_str(),
                lhs.as_str()
            ))
            }
        }
    };

    Ok(Expr::Compare { lhs, op, rhs })
}

fn kind_name(kind: &EnvelopeKind) -> &'static str {
    match kind {
        EnvelopeKind::Audio(_) => "audio",
        EnvelopeKind::Text(_) => "text",
        EnvelopeKind::Embedding(_) => "embedding",
        EnvelopeKind::Image { .. } => "image",
        EnvelopeKind::MultiPart(_) => "multipart",
    }
}

fn battery_level(metrics: &DeviceMetrics) -> f64 {
    metrics
        .resource
        .battery_pct
        .map(f64::from)
        .unwrap_or_else(|| f64::from(metrics.capabilities.battery_level()))
}

fn compare_enum(actual: &str, op: Op, rhs: &Rhs) -> bool {
    let Rhs::Str(expected) = rhs else {
        return false;
    };
    let equal = actual.eq_ignore_ascii_case(expected);
    match op {
        Op::Eq => equal,
        Op::Ne => !equal,
        _ => false,
    }
}

fn compare_text(actual: &str, op: Op, rhs: &Rhs) -> bool {
    match (op, rhs) {
        (Op::Contains, Rhs::Str(needle)) => actual.contains(needle.as_str()),
        (Op::Matches, Rhs::Regex(regex)) => regex.is_match(actual),
        (Op::Eq, Rhs::Str(expected)) => actual == expected,
        (Op::Ne, Rhs::Str(expected)) => actual != expected,
        _ => false,
    }
}

fn compare_number(actual: f64, op: Op, rhs: &Rhs) -> bool {
    let Rhs::Num(expected) = rhs else {
        return false;
    };
    match op {
        Op::Eq => actual == *expected,
        Op::Ne => actual != *expected,
        Op::Lt => actual < *expected,
        Op::Le => actual <= *expected,
        Op::Gt => actual > *expected,
        Op::Ge => actual >= *expected,
        _ => false,
    }
}

impl Expr {
    fn evaluate(&self, envelope: &Envelope, metrics: &DeviceMetrics) -> bool {
        match self {
            Expr::Literal(value) => *value,
            Expr::Compare { lhs, op, rhs } => match lhs {
                Lhs::InputKind => compare_enum(kind_name(&envelope.kind), *op, rhs),
                Lhs::MemoryPressure => {
                    compare_enum(metrics.resource.memory_pressure.as_str(), *op, rhs)
                }
                Lhs::ThermalState => {
                    compare_enum(metrics.resource.thermal_state.as_str(), *op, rhs)
                }
                Lhs::InputText => envelope
                    .as_text()
                    .map(|text| compare_text(text, *op, rhs))
                    .unwrap_or(false),
                Lhs::InputTextLen => envelope
                    .as_text()
                    .map(|text| compare_number(text.chars().count() as f64, *op, rhs))
                    .unwrap_or(false),
                Lhs::BatteryLevel => compare_number(battery_level(metrics), *op, rhs),
                Lhs::CpuPct => metrics
                    .resource
                    .cpu_pct
                    .map(|pct| compare_number(f64::from(pct), *op, rhs))
                    .unwrap_or(false),
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bundle parsing
// ─────────────────────────────────────────────────────────────────────────────

fn string_field(value: &serde_yaml::Value, key: &str) -> Result<String, String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("'{key}' must be a string"))
}

fn expression_value(value: &serde_yaml::Value, location: &str) -> Result<String, String> {
    match value {
        serde_yaml::Value::String(s) => Ok(s.clone()),
        serde_yaml::Value::Bool(b) => Ok(b.to_string()),
        _ => Err(format!("{location}: expression must be a string")),
    }
}

fn mapping_key<'a>(value: &'a serde_yaml::Value, location: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{location}: keys must be strings"))
}

/// Parse and compile a complete bundle. Pure: nothing is activated until the
/// caller swaps the result in.
fn parse_bundle(bytes: &[u8]) -> Result<(PolicyBundle, Vec<Expr>), String> {
    let value: serde_yaml::Value = serde_yaml::from_slice(bytes)
        .map_err(|e| format!("failed to parse policy bundle as YAML or JSON: {e}"))?;
    let map = value
        .as_mapping()
        .ok_or_else(|| "policy bundle must be a mapping at the top level".to_string())?;

    let mut version = "1.0.0".to_string();
    let mut signature = "unsigned".to_string();

    for (key, field) in map {
        match mapping_key(key, "policy bundle")? {
            "version" => version = string_field(field, "version")?,
            "signature" => signature = string_field(field, "signature")?,
            "rules" | "deny_cloud_if" | "route_cloud_if" => {}
            other => {
                return Err(format!(
                    "unknown policy bundle key '{other}' (expected version, signature, rules, \
                     deny_cloud_if, or route_cloud_if)"
                ))
            }
        }
    }

    let mut raw_rules: Vec<(String, String, PolicyAction)> = Vec::new();

    if let Some(rules) = map.get("rules") {
        let sequence = rules
            .as_sequence()
            .ok_or_else(|| "'rules' must be a sequence".to_string())?;
        for (idx, item) in sequence.iter().enumerate() {
            let location = format!("rules[{idx}]");
            let rule = item
                .as_mapping()
                .ok_or_else(|| format!("{location} must be a mapping"))?;
            for key in rule.keys() {
                match mapping_key(key, &location)? {
                    "id" | "expression" | "action" => {}
                    other => {
                        return Err(format!(
                            "{location}: unknown key '{other}' (expected id, expression, action)"
                        ))
                    }
                }
            }
            let id = rule
                .get("id")
                .ok_or_else(|| format!("{location}: missing required field 'id'"))
                .and_then(|v| string_field(v, &format!("{location}.id")))?;
            let expression = rule
                .get("expression")
                .ok_or_else(|| format!("rule '{id}': missing required field 'expression'"))
                .and_then(|v| expression_value(v, &format!("rule '{id}'")))?;
            let action = rule
                .get("action")
                .ok_or_else(|| format!("rule '{id}': missing required field 'action'"))
                .and_then(|v| string_field(v, &format!("rule '{id}'.action")))
                .and_then(|s| {
                    s.parse::<PolicyAction>()
                        .map_err(|e| format!("rule '{id}': {e}"))
                })?;
            raw_rules.push((id, expression, action));
        }
    }

    for (key, action) in [
        ("deny_cloud_if", PolicyAction::Deny),
        ("route_cloud_if", PolicyAction::RouteCloud),
    ] {
        if let Some(entries) = map.get(key) {
            let sequence = entries
                .as_sequence()
                .ok_or_else(|| format!("'{key}' must be a sequence of expressions"))?;
            for (idx, item) in sequence.iter().enumerate() {
                let location = format!("{key}[{idx}]");
                let expression = expression_value(item, &location)?;
                raw_rules.push((format!("{key}_{idx}"), expression, action));
            }
        }
    }

    if raw_rules.is_empty() {
        return Err("policy bundle contains no rules".to_string());
    }

    let mut seen = HashSet::new();
    for (id, _, _) in &raw_rules {
        if !seen.insert(id.as_str()) {
            return Err(format!("duplicate policy rule id '{id}'"));
        }
    }

    let mut rules = Vec::with_capacity(raw_rules.len());
    let mut compiled = Vec::with_capacity(raw_rules.len());
    for (id, expression, action) in raw_rules {
        let expr = compile_expression(&expression).map_err(|e| format!("rule '{id}': {e}"))?;
        rules.push(PolicyRule {
            id,
            expression,
            action,
        });
        compiled.push(expr);
    }

    Ok((
        PolicyBundle {
            version,
            rules,
            signature,
        },
        compiled,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Default engine
// ─────────────────────────────────────────────────────────────────────────────

/// Default implementation of [`PolicyEngine`].
///
/// Expressions are compiled when a bundle is loaded; evaluation is a linear
/// scan over the compiled rules with first-match-wins semantics.
pub struct DefaultPolicyEngine {
    bundle: Option<PolicyBundle>,
    /// Compiled form of `bundle.rules`, index-aligned.
    compiled: Vec<Expr>,
}

impl DefaultPolicyEngine {
    /// Create an engine with no bundle loaded (allow-all).
    pub fn new() -> Self {
        Self {
            bundle: None,
            compiled: Vec::new(),
        }
    }

    /// Create an engine with the default (empty, allow-all) bundle.
    ///
    /// Callers wanting stricter behaviour should `load_policies` an explicit
    /// bundle.
    pub fn with_default_policy() -> Self {
        Self {
            bundle: Some(PolicyBundle {
                version: "0.1.0".to_string(),
                rules: vec![],
                signature: "default_mvp".to_string(),
            }),
            compiled: Vec::new(),
        }
    }

    /// The currently active bundle, if any.
    pub fn bundle(&self) -> Option<&PolicyBundle> {
        self.bundle.as_ref()
    }
}

impl Default for DefaultPolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyEngine for DefaultPolicyEngine {
    fn load_policies(&mut self, bundle_bytes: Vec<u8>) -> Result<(), String> {
        let (bundle, compiled) = parse_bundle(&bundle_bytes)?;
        self.bundle = Some(bundle);
        self.compiled = compiled;
        Ok(())
    }

    fn evaluate(&self, _stage: &str, envelope: &Envelope, metrics: &DeviceMetrics) -> PolicyResult {
        let Some(ref bundle) = self.bundle else {
            return PolicyResult::allow(Some("no policy loaded".to_string()));
        };

        for (rule, expr) in bundle.rules.iter().zip(&self.compiled) {
            if !expr.evaluate(envelope, metrics) {
                continue;
            }
            match rule.action {
                PolicyAction::Allow => {
                    return PolicyResult::allow(Some(format!(
                        "policy rule '{}' allowed: {}",
                        rule.id, rule.expression
                    )));
                }
                PolicyAction::Deny => {
                    return PolicyResult::deny(format!(
                        "Policy rule '{}' matched: {}",
                        rule.id, rule.expression
                    ));
                }
                PolicyAction::RouteCloud => {
                    return PolicyResult::prefer_cloud(format!(
                        "policy rule '{}' prefers cloud: {}",
                        rule.id, rule.expression
                    ));
                }
                PolicyAction::Redact => {
                    let mut result =
                        PolicyResult::allow(Some(format!("Rule '{}' requires redaction", rule.id)));
                    result.transforms_applied.push(rule.id.clone());
                    return result;
                }
            }
        }

        PolicyResult::allow(Some("all policy checks passed".to_string()))
    }

    fn redact(&self, _envelope: &mut Envelope) -> bool {
        // Redaction transforms are not implemented. Callers must treat a
        // non-empty `transforms_applied` as "stay local", never as
        // "redacted, safe to send".
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{MemoryPressure, ResourceSnapshot, ThermalState};
    use crate::ir::{Envelope, EnvelopeKind};

    fn text(value: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(value.to_string()))
    }

    fn audio() -> Envelope {
        Envelope::new(EnvelopeKind::Audio(vec![0, 1, 2]))
    }

    fn embedding() -> Envelope {
        Envelope::new(EnvelopeKind::Embedding(vec![0.1, 0.2]))
    }

    fn multipart() -> Envelope {
        Envelope::new(EnvelopeKind::MultiPart(vec![text("a"), text("b")]))
    }

    fn metrics() -> DeviceMetrics {
        DeviceMetrics::default()
    }

    fn metrics_with(
        battery_pct: Option<u8>,
        cpu_pct: Option<f32>,
        memory_pressure: MemoryPressure,
        thermal_state: ThermalState,
    ) -> DeviceMetrics {
        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.battery_pct = battery_pct;
        snapshot.cpu_pct = cpu_pct;
        snapshot.memory_pressure = memory_pressure;
        snapshot.thermal_state = thermal_state;
        DeviceMetrics::default().with_live_snapshot(snapshot)
    }

    fn engine(bundle: &str) -> DefaultPolicyEngine {
        let mut engine = DefaultPolicyEngine::new();
        engine
            .load_policies(bundle.as_bytes().to_vec())
            .unwrap_or_else(|e| panic!("bundle should load: {e}\n{bundle}"));
        engine
    }

    fn load_error(bundle: &str) -> String {
        let mut engine = DefaultPolicyEngine::new();
        engine
            .load_policies(bundle.as_bytes().to_vec())
            .expect_err("bundle should be rejected")
    }

    fn deny_if(expression: &str) -> DefaultPolicyEngine {
        engine(&format!("deny_cloud_if:\n  - '{expression}'\n"))
    }

    fn matches(expression: &str, envelope: &Envelope, metrics: &DeviceMetrics) -> bool {
        !deny_if(expression)
            .evaluate("stage", envelope, metrics)
            .allowed
    }

    // ── defaults ────────────────────────────────────────────────────────────

    #[test]
    fn default_policy_allows_everything() {
        let engine = DefaultPolicyEngine::with_default_policy();
        for envelope in [text("Text"), audio(), embedding(), multipart()] {
            let result = engine.evaluate("stage", &envelope, &metrics());
            assert!(result.allowed);
            assert!(!result.requires_local());
            assert!(!result.prefers_cloud());
            assert_eq!(result.reason.as_deref(), Some("all policy checks passed"));
        }
    }

    #[test]
    fn no_policy_loaded_allows() {
        let engine = DefaultPolicyEngine::new();
        let result = engine.evaluate("stage", &text("hello"), &metrics());
        assert!(result.allowed);
        assert_eq!(result.reason.as_deref(), Some("no policy loaded"));
        assert!(engine.bundle().is_none());
    }

    // ── input.kind ──────────────────────────────────────────────────────────

    #[test]
    fn input_kind_matches_variant_not_payload() {
        assert!(matches(
            r#"input.kind == "text""#,
            &text("audio"),
            &metrics()
        ));
        assert!(!matches(
            r#"input.kind == "audio""#,
            &text("audio"),
            &metrics()
        ));
        assert!(matches(r#"input.kind == "audio""#, &audio(), &metrics()));
        assert!(matches(
            r#"input.kind == "embedding""#,
            &embedding(),
            &metrics()
        ));
        assert!(matches(
            r#"input.kind == "multipart""#,
            &multipart(),
            &metrics()
        ));
        assert!(matches(r#"input.kind != "text""#, &audio(), &metrics()));
        assert!(!matches(r#"input.kind != "text""#, &text("x"), &metrics()));
    }

    #[test]
    fn input_kind_is_case_insensitive_and_accepts_legacy_audioraw() {
        assert!(matches(r#"input.kind == "TEXT""#, &text("x"), &metrics()));
        assert!(matches(r#"input.kind == "AudioRaw""#, &audio(), &metrics()));
        assert!(!matches(
            r#"input.kind == "AudioRaw""#,
            &text("x"),
            &metrics()
        ));
    }

    #[test]
    fn input_kind_rejects_unknown_values_at_load() {
        let err = load_error("deny_cloud_if:\n  - 'input.kind == \"SensitiveData\"'\n");
        assert!(err.contains("SensitiveData"), "{err}");
        assert!(err.contains("deny_cloud_if_0"), "{err}");
        assert!(err.contains("legacy alias: audioraw"), "{err}");
    }

    // ── input.text / input.text_len ─────────────────────────────────────────

    #[test]
    fn input_text_contains_is_case_sensitive_substring() {
        assert!(matches(
            r#"input.text contains "secret""#,
            &text("my secret"),
            &metrics()
        ));
        assert!(!matches(
            r#"input.text contains "secret""#,
            &text("my SECRET"),
            &metrics()
        ));
        assert!(!matches(
            r#"input.text contains "secret""#,
            &audio(),
            &metrics()
        ));
    }

    #[test]
    fn input_text_matches_uses_regex() {
        let m = metrics();
        assert!(matches(
            r#"input.text matches "(?i)password|ssn""#,
            &text("My PASSWORD"),
            &m
        ));
        assert!(!matches(r#"input.text matches "^\d+$""#, &text("abc"), &m));
        assert!(matches(r#"input.text matches "^\d+$""#, &text("123"), &m));
        assert!(!matches(r#"input.text matches ".*""#, &audio(), &m));
    }

    #[test]
    fn input_text_equality_is_exact() {
        assert!(matches(r#"input.text == "hi""#, &text("hi"), &metrics()));
        assert!(!matches(r#"input.text == "hi""#, &text("Hi"), &metrics()));
        assert!(matches(r#"input.text != "hi""#, &text("Hi"), &metrics()));
        assert!(!matches(r#"input.text != "hi""#, &audio(), &metrics()));
    }

    #[test]
    fn input_text_escaped_literals() {
        let m = metrics();
        assert!(matches(
            r#"input.text contains "say \"hi\"""#,
            &text(r#"please say "hi" now"#),
            &m
        ));
        assert!(matches(
            r#"input.text contains "a\\b""#,
            &text(r"x a\b y"),
            &m
        ));
        // Unknown escapes pass through verbatim, so regex escapes need no
        // double-escaping and a literal `\n` sequence stays two characters.
        assert!(matches(
            r#"input.text matches "^\d{3}-\d{4}$""#,
            &text("555-1234"),
            &m
        ));
        assert!(matches(
            r#"input.text matches "v1\.2""#,
            &text("release v1.2"),
            &m
        ));
        assert!(!matches(
            r#"input.text matches "v1\.2""#,
            &text("release v1x2"),
            &m
        ));
        assert!(matches(
            r#"input.text contains "line\nbreak""#,
            &text(r"line\nbreak"),
            &m
        ));
        assert!(!matches(
            r#"input.text contains "line\nbreak""#,
            &text("line\nbreak"),
            &m
        ));
    }

    #[test]
    fn input_text_len_counts_unicode_scalars() {
        let m = metrics();
        assert!(matches("input.text_len == 5", &text("héllo"), &m));
        assert!(matches("input.text_len == 2", &text("😀😀"), &m));
        assert!(matches("input.text_len > 800", &text(&"x".repeat(801)), &m));
        assert!(!matches(
            "input.text_len > 800",
            &text(&"x".repeat(800)),
            &m
        ));
        assert!(matches(
            "input.text_len >= 800",
            &text(&"x".repeat(800)),
            &m
        ));
        assert!(matches("input.text_len < 3", &text("ab"), &m));
        assert!(matches("input.text_len <= 2", &text("ab"), &m));
        assert!(matches("input.text_len != 2", &text("abc"), &m));
        assert!(!matches("input.text_len >= 0", &audio(), &m));
    }

    // ── metrics.* ───────────────────────────────────────────────────────────

    #[test]
    fn battery_prefers_live_snapshot_then_capabilities() {
        let live = metrics_with(Some(10), None, MemoryPressure::Normal, ThermalState::Normal);
        assert!(matches("metrics.battery_level < 25", &text("x"), &live));
        assert!(!matches("metrics.battery_level >= 25", &text("x"), &live));

        // No live reading: fall back to capabilities (default 100).
        let fallback = metrics();
        assert!(matches(
            "metrics.battery_level == 100",
            &text("x"),
            &fallback
        ));
        assert!(!matches(
            "metrics.battery_level < 25",
            &text("x"),
            &fallback
        ));
    }

    #[test]
    fn cpu_missing_never_matches() {
        let missing = metrics_with(None, None, MemoryPressure::Normal, ThermalState::Normal);
        assert!(!matches("metrics.cpu_pct > 0", &text("x"), &missing));
        assert!(!matches("metrics.cpu_pct <= 100", &text("x"), &missing));
        let hot = metrics_with(
            None,
            Some(96.5),
            MemoryPressure::Normal,
            ThermalState::Normal,
        );
        assert!(matches("metrics.cpu_pct > 95", &text("x"), &hot));
        assert!(matches("metrics.cpu_pct >= 96.5", &text("x"), &hot));
        assert!(!matches("metrics.cpu_pct < 96.5", &text("x"), &hot));
    }

    #[test]
    fn memory_pressure_and_thermal_state_compare_case_insensitively() {
        let m = metrics_with(None, None, MemoryPressure::Critical, ThermalState::Hot);
        assert!(matches(
            r#"metrics.memory_pressure == "CRITICAL""#,
            &text("x"),
            &m
        ));
        assert!(matches(
            r#"metrics.memory_pressure != "normal""#,
            &text("x"),
            &m
        ));
        assert!(matches(r#"metrics.thermal_state == "hot""#, &text("x"), &m));
        assert!(!matches(
            r#"metrics.thermal_state == "critical""#,
            &text("x"),
            &m
        ));
        let unknown = metrics_with(None, None, MemoryPressure::Unknown, ThermalState::Normal);
        assert!(matches(
            r#"metrics.memory_pressure == "unknown""#,
            &text("x"),
            &unknown
        ));
    }

    // ── literals, actions, ordering ─────────────────────────────────────────

    #[test]
    fn boolean_literals() {
        assert!(matches("true", &audio(), &metrics()));
        assert!(!matches("false", &audio(), &metrics()));
        // Unquoted YAML booleans are accepted too.
        let engine = engine("deny_cloud_if:\n  - true\n");
        assert!(!engine.evaluate("s", &text("x"), &metrics()).allowed);
    }

    #[test]
    fn first_matching_rule_wins() {
        let allow_first = engine(
            r#"
rules:
  - id: allow_all
    expression: "true"
    action: allow
  - id: deny_all
    expression: "true"
    action: deny
"#,
        );
        let result = allow_first.evaluate("s", &text("x"), &metrics());
        assert!(result.allowed);
        assert!(result.reason.unwrap().contains("allow_all"));

        let deny_first = engine(
            r#"
rules:
  - id: deny_all
    expression: "true"
    action: deny
  - id: allow_all
    expression: "true"
    action: allow
"#,
        );
        let result = deny_first.evaluate("s", &text("x"), &metrics());
        assert!(!result.allowed);
        assert!(result.reason.unwrap().contains("deny_all"));
    }

    #[test]
    fn route_cloud_and_alias_prefer_cloud() {
        for action in ["route_cloud", "prefer_cloud"] {
            let engine = engine(&format!(
                "rules:\n  - id: offload\n    expression: \"true\"\n    action: {action}\n"
            ));
            let result = engine.evaluate("s", &text("x"), &metrics());
            assert!(result.allowed);
            assert!(result.prefers_cloud());
            assert!(!result.requires_local());
            assert!(result.reason.unwrap().contains("offload"));
            assert_eq!(
                engine.bundle().unwrap().rules[0].action,
                PolicyAction::RouteCloud
            );
        }
        let shorthand = engine("route_cloud_if:\n  - \"true\"\n");
        assert!(shorthand
            .evaluate("s", &text("x"), &metrics())
            .prefers_cloud());
    }

    #[test]
    fn redact_requires_local() {
        let engine = engine(
            "rules:\n  - id: scrub\n    expression: 'input.kind == \"text\"'\n    action: redact\n",
        );
        let result = engine.evaluate("s", &text("x"), &metrics());
        assert!(result.allowed);
        assert_eq!(result.transforms_applied, vec!["scrub".to_string()]);
        assert!(result.requires_local());
        assert!(!result.prefers_cloud());
    }

    #[test]
    fn shorthand_rules_follow_explicit_rules_in_order() {
        // Explicit allow stops evaluation before the shorthand denial.
        let engine_allow = engine(
            r#"
rules:
  - id: allow_text
    expression: 'input.kind == "text"'
    action: allow
deny_cloud_if:
  - "true"
"#,
        );
        assert!(engine_allow.evaluate("s", &text("x"), &metrics()).allowed);
        assert!(!engine_allow.evaluate("s", &audio(), &metrics()).allowed);

        // deny_cloud_if is appended before route_cloud_if.
        let engine_both = engine("deny_cloud_if:\n  - \"true\"\nroute_cloud_if:\n  - \"true\"\n");
        let result = engine_both.evaluate("s", &text("x"), &metrics());
        assert!(!result.allowed);
        assert!(!result.prefers_cloud());
        let ids: Vec<_> = engine_both
            .bundle()
            .unwrap()
            .rules
            .iter()
            .map(|r| r.id.as_str())
            .collect();
        assert_eq!(ids, vec!["deny_cloud_if_0", "route_cloud_if_0"]);
    }

    #[test]
    fn json_bundles_parse_like_yaml() {
        let engine = engine(
            r#"{
  "version": "0.1.0",
  "signature": "test",
  "deny_cloud_if": ["input.kind == \"AudioRaw\""],
  "rules": [{"id": "long", "expression": "input.text_len > 3", "action": "route_cloud"}]
}"#,
        );
        let bundle = engine.bundle().unwrap();
        assert_eq!(bundle.version, "0.1.0");
        assert_eq!(bundle.signature, "test");
        assert!(!engine.evaluate("s", &audio(), &metrics()).allowed);
        assert!(engine
            .evaluate("s", &text("long text"), &metrics())
            .prefers_cloud());
        assert!(engine.evaluate("s", &text("hi"), &metrics()).allowed);
    }

    // ── load-time rejection ────────────────────────────────────────────────

    #[test]
    fn rejects_unknown_operand_operator_and_type_mismatches() {
        let cases = [
            ("metrics.network_rtt > 300", "unknown operand"),
            ("input.kind ~= \"text\"", "unknown operator"),
            (
                "metrics.battery_level contains \"x\"",
                "not valid for numeric operand",
            ),
            (
                "metrics.battery_level < \"low\"",
                "must be compared against a number",
            ),
            ("metrics.battery_level < abc", "is not a number"),
            ("metrics.battery_level < inf", "not a finite number"),
            ("metrics.cpu_pct > NaN", "not a finite number"),
            (
                "input.kind == text",
                "must be compared against a quoted string",
            ),
            ("input.kind < \"text\"", "use == or !="),
            ("metrics.memory_pressure == \"high\"", "not a valid value"),
            ("metrics.thermal_state == \"boiling\"", "not a valid value"),
            ("input.text > \"x\"", "use ==, !=, contains, or matches"),
            ("input.text matches \"(unclosed\"", "invalid regex"),
            (
                "input.text contains \"unterminated",
                "unterminated string literal",
            ),
            (
                "input.text contains \"a\" extra",
                "unexpected trailing input",
            ),
            ("input.text_len > 1 2", "unexpected trailing input"),
            ("input.kind", "expected '<operand> <operator> <value>'"),
            ("input.kind ==", "missing right operand"),
            ("", "expression is empty"),
        ];
        for (expression, expected) in cases {
            let err = load_error(&format!("deny_cloud_if:\n  - '{expression}'\n"));
            assert!(
                err.contains(expected),
                "expression `{expression}` should fail with `{expected}`, got: {err}"
            );
            assert!(err.contains("rule 'deny_cloud_if_0'"), "{err}");
        }
    }

    #[test]
    fn rejects_malformed_bundles() {
        let cases = [
            ("- just\n- a\n- list\n", "must be a mapping at the top level"),
            ("version: \"1\"\n", "contains no rules"),
            ("unexpected: 1\n", "unknown policy bundle key 'unexpected'"),
            ("rules: notalist\n", "'rules' must be a sequence"),
            ("deny_cloud_if: \"true\"\n", "'deny_cloud_if' must be a sequence"),
            ("deny_cloud_if:\n  - 5\n", "expression must be a string"),
            ("rules:\n  - \"true\"\n", "rules[0] must be a mapping"),
            (
                "rules:\n  - expression: \"true\"\n    action: deny\n",
                "missing required field 'id'",
            ),
            ("rules:\n  - id: r\n    action: deny\n", "missing required field 'expression'"),
            ("rules:\n  - id: r\n    expression: \"true\"\n", "missing required field 'action'"),
            (
                "rules:\n  - id: r\n    expression: \"true\"\n    action: explode\n",
                "unknown policy action 'explode'",
            ),
            (
                "rules:\n  - id: r\n    expression: \"true\"\n    action: deny\n    extra: 1\n",
                "unknown key 'extra'",
            ),
            (
                "rules:\n  - id: dup\n    expression: \"true\"\n    action: deny\n  - id: dup\n    expression: \"true\"\n    action: allow\n",
                "duplicate policy rule id 'dup'",
            ),
            (
                "rules:\n  - id: deny_cloud_if_0\n    expression: \"true\"\n    action: allow\ndeny_cloud_if:\n  - \"true\"\n",
                "duplicate policy rule id 'deny_cloud_if_0'",
            ),
            ("version: 1\n", "'version' must be a string"),
            ("{not valid yaml: [", "failed to parse policy bundle"),
        ];
        for (bundle, expected) in cases {
            let err = load_error(bundle);
            assert!(
                err.contains(expected),
                "bundle should fail with `{expected}`, got: {err}\n{bundle}"
            );
        }
    }

    #[test]
    fn failed_reload_keeps_previous_policy() {
        let mut engine = engine("deny_cloud_if:\n  - 'input.kind == \"text\"'\n");
        assert!(!engine.evaluate("s", &text("x"), &metrics()).allowed);

        let err = engine
            .load_policies(b"deny_cloud_if:\n  - 'metrics.network_rtt > 300'\n".to_vec())
            .expect_err("invalid reload must fail");
        assert!(err.contains("unknown operand"), "{err}");

        // Still the old bundle, still denying text.
        assert!(!engine.evaluate("s", &text("x"), &metrics()).allowed);
        assert_eq!(engine.bundle().unwrap().rules[0].id, "deny_cloud_if_0");
    }

    #[test]
    fn successful_reload_replaces_policy() {
        let mut engine = engine("deny_cloud_if:\n  - 'input.kind == \"text\"'\n");
        engine
            .load_policies(b"route_cloud_if:\n  - \"true\"\n".to_vec())
            .expect("valid reload");
        let result = engine.evaluate("s", &text("x"), &metrics());
        assert!(result.allowed);
        assert!(result.prefers_cloud());
    }

    #[test]
    fn redact_is_a_no_op() {
        let engine = DefaultPolicyEngine::with_default_policy();
        let mut envelope = text("keep me");
        assert!(!engine.redact(&mut envelope));
        assert_eq!(envelope.as_text(), Some("keep me"));
    }

    #[test]
    fn policy_action_round_trips() {
        for action in [
            PolicyAction::Allow,
            PolicyAction::Deny,
            PolicyAction::RouteCloud,
            PolicyAction::Redact,
        ] {
            assert_eq!(action.as_str().parse::<PolicyAction>().unwrap(), action);
            assert_eq!(action.to_string(), action.as_str());
        }
        assert_eq!(
            "PREFER_CLOUD".parse::<PolicyAction>().unwrap(),
            PolicyAction::RouteCloud
        );
        assert!("maybe".parse::<PolicyAction>().is_err());
    }
}
