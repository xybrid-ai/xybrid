//! Policy Engine module - Enforces data-handling and routing rules before inference stages run.
//!
//! The Policy Engine ensures that privacy, latency, and cost constraints are respected at runtime
//! by evaluating allow/deny conditions per stage and optionally applying redaction transforms.

use crate::context::DeviceMetrics;
use crate::ir::{Envelope, EnvelopeKind};

/// Result of a policy evaluation.
#[derive(Debug, Clone)]
pub struct PolicyResult {
    pub allowed: bool,
    pub reason: Option<String>,
    pub transforms_applied: Vec<String>,
}

impl PolicyResult {
    /// Create a new PolicyResult.
    pub fn new(allowed: bool, reason: Option<String>) -> Self {
        Self {
            allowed,
            reason,
            transforms_applied: Vec::new(),
        }
    }

    /// Create an allowed result.
    pub fn allow(reason: Option<String>) -> Self {
        Self::new(true, reason)
    }

    /// Create a denied result.
    pub fn deny(reason: String) -> Self {
        Self::new(false, Some(reason))
    }
}

/// Policy bundle containing rules and metadata.
#[derive(Debug, Clone)]
pub struct PolicyBundle {
    pub version: String,
    pub rules: Vec<PolicyRule>,
    pub signature: String,
}

/// Individual policy rule.
#[derive(Debug, Clone)]
pub struct PolicyRule {
    pub id: String,
    pub expression: String, // CEL or mini-DSL
    pub action: String,     // "allow" | "deny" | "redact"
}

/// Policy Engine trait for evaluating policies.
pub trait PolicyEngine {
    /// Load and cache signed policy files.
    fn load_policies(&mut self, bundle_bytes: Vec<u8>) -> Result<(), String>;

    /// Evaluate policy conditions for a stage.
    fn evaluate(&self, stage: &str, envelope: &Envelope, metrics: &DeviceMetrics) -> PolicyResult;

    /// Apply redaction transforms to an envelope.
    fn redact(&self, envelope: &mut Envelope) -> bool;
}

/// Default implementation of PolicyEngine for MVP.
///
/// Currently supports a single expression form: `input.kind == "<value>"`.
/// The legacy `metrics.network_rtt`/`metrics.battery` comparisons were
/// dropped together with the speculative routing scalars; if device-state
/// rules are needed in the future they should target real signals on
/// `metrics.capabilities` / `metrics.resource`.
pub struct DefaultPolicyEngine {
    bundle: Option<PolicyBundle>,
}

impl DefaultPolicyEngine {
    /// Create a new DefaultPolicyEngine instance.
    pub fn new() -> Self {
        Self { bundle: None }
    }

    /// Create a new instance with the default policy bundle.
    ///
    /// The default bundle is currently empty (allow-all). Callers wanting
    /// stricter behaviour should `load_policies` an explicit bundle.
    pub fn with_default_policy() -> Self {
        let mut engine = Self::new();
        let default_bundle = PolicyBundle {
            version: "0.1.0".to_string(),
            rules: vec![],
            signature: "default_mvp".to_string(),
        };
        engine.bundle = Some(default_bundle);
        engine
    }

    /// Evaluate a single expression against the context.
    fn evaluate_expression(
        &self,
        expression: &str,
        envelope: &Envelope,
        _metrics: &DeviceMetrics,
    ) -> bool {
        // MVP: Simple expression evaluation
        // Supports:
        // - input.kind == "value"

        let expr = expression.trim();

        // Check for equality comparisons: input.kind == "value"
        if expr.contains("input.kind ==") {
            let parts: Vec<&str> = expr.split("==").collect();
            if parts.len() == 2 {
                let value = parts[1].trim().trim_matches('"').trim();

                // Direct comparison against metadata label if provided.
                if let Some(label) = envelope.get_metadata("kind_label") {
                    if label == value {
                        return true;
                    }
                }

                // Compare against the textual payload for text envelopes.
                if let EnvelopeKind::Text(text) = &envelope.kind {
                    if text == value {
                        return true;
                    }
                }

                // Compare against the high-level variant name (Audio/Text/Embedding).
                if envelope.kind_str() == value {
                    return true;
                }

                // Backwards compatibility: treat legacy labels as variant aliases.
                match &envelope.kind {
                    EnvelopeKind::Audio(_) => {
                        if value.eq_ignore_ascii_case("audioraw")
                            || value.eq_ignore_ascii_case("audio")
                        {
                            return true;
                        }
                    }
                    EnvelopeKind::Text(_) => {
                        if value.eq_ignore_ascii_case("text") {
                            return true;
                        }
                    }
                    EnvelopeKind::Embedding(_) => {
                        if value.eq_ignore_ascii_case("embedding") {
                            return true;
                        }
                    }
                }
            }
        }

        // If we can't parse it, default to false (no match → no deny).
        false
    }
}

impl Default for DefaultPolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyEngine for DefaultPolicyEngine {
    fn load_policies(&mut self, bundle_bytes: Vec<u8>) -> Result<(), String> {
        // Try to parse as YAML first
        let yaml_result: Result<serde_yaml::Value, _> = serde_yaml::from_slice(&bundle_bytes);

        if let Ok(yaml_value) = yaml_result {
            self.parse_yaml_policy(yaml_value)?;
            return Ok(());
        }

        // Try to parse as JSON
        let json_result: Result<serde_json::Value, _> = serde_json::from_slice(&bundle_bytes);

        if let Ok(json_value) = json_result {
            self.parse_json_policy(json_value)?;
            return Ok(());
        }

        Err("Failed to parse policy bundle as YAML or JSON".to_string())
    }

    fn evaluate(&self, _stage: &str, envelope: &Envelope, metrics: &DeviceMetrics) -> PolicyResult {
        // If no policy bundle is loaded, default to allow
        let Some(ref bundle) = self.bundle else {
            return PolicyResult::allow(Some("no policy loaded".to_string()));
        };

        // Evaluate each rule
        for rule in &bundle.rules {
            let matches = self.evaluate_expression(&rule.expression, envelope, metrics);

            if matches {
                match rule.action.as_str() {
                    "deny" => {
                        let reason =
                            format!("Policy rule '{}' matched: {}", rule.id, rule.expression);
                        return PolicyResult::deny(reason);
                    }
                    "redact" => {
                        // For redact, we'll mark it in the result but still allow
                        // The actual redaction is applied by the redact() method
                        let mut result = PolicyResult::allow(Some(format!(
                            "Rule '{}' requires redaction",
                            rule.id
                        )));
                        result.transforms_applied.push(rule.id.clone());
                        return result;
                    }
                    _ => {
                        // "allow" or unknown action - continue to next rule
                    }
                }
            }
        }

        // No denying rules matched, allow
        PolicyResult::allow(Some("all policy checks passed".to_string()))
    }

    fn redact(&self, _envelope: &mut Envelope) -> bool {
        // MVP: Simple redaction - just log for now
        // TODO: Implement actual redaction transforms (text filtering, truncation, etc.)
        // For now, this is a no-op but returns false to indicate no changes
        false
    }
}

impl DefaultPolicyEngine {
    /// Parse a YAML policy structure.
    fn parse_yaml_policy(&mut self, value: serde_yaml::Value) -> Result<(), String> {
        let mut rules = Vec::new();
        let mut version = "1.0.0".to_string();
        let mut signature = "unsigned".to_string();

        // Parse version if present
        if let Some(v) = value.get("version").and_then(|v| v.as_str()) {
            version = v.to_string();
        }

        // Parse signature if present
        if let Some(s) = value.get("signature").and_then(|s| s.as_str()) {
            signature = s.to_string();
        }

        // Parse deny_cloud_if rules (MVP format)
        if let Some(deny_rules) = value.get("deny_cloud_if").and_then(|v| v.as_sequence()) {
            for (idx, rule_value) in deny_rules.iter().enumerate() {
                if let Some(expr) = rule_value.as_str() {
                    rules.push(PolicyRule {
                        id: format!("deny_rule_{}", idx),
                        expression: expr.to_string(),
                        action: "deny".to_string(),
                    });
                }
            }
        }

        // Parse rules array if present (more structured format)
        if let Some(rules_array) = value.get("rules").and_then(|v| v.as_sequence()) {
            for rule_value in rules_array {
                if let Some(id) = rule_value.get("id").and_then(|v| v.as_str()) {
                    let expression = rule_value
                        .get("expression")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let action = rule_value
                        .get("action")
                        .and_then(|v| v.as_str())
                        .unwrap_or("deny")
                        .to_string();

                    rules.push(PolicyRule {
                        id: id.to_string(),
                        expression,
                        action,
                    });
                }
            }
        }

        if rules.is_empty() {
            return Err("No valid rules found in policy bundle".to_string());
        }

        self.bundle = Some(PolicyBundle {
            version,
            rules,
            signature,
        });

        Ok(())
    }

    /// Parse a JSON policy structure.
    fn parse_json_policy(&mut self, value: serde_json::Value) -> Result<(), String> {
        let mut rules = Vec::new();
        let mut version = "1.0.0".to_string();
        let mut signature = "unsigned".to_string();

        // Parse version if present
        if let Some(v) = value.get("version").and_then(|v| v.as_str()) {
            version = v.to_string();
        }

        // Parse signature if present
        if let Some(s) = value.get("signature").and_then(|s| s.as_str()) {
            signature = s.to_string();
        }

        // Parse deny_cloud_if rules (MVP format)
        if let Some(deny_rules) = value.get("deny_cloud_if").and_then(|v| v.as_array()) {
            for (idx, rule_value) in deny_rules.iter().enumerate() {
                if let Some(expr) = rule_value.as_str() {
                    rules.push(PolicyRule {
                        id: format!("deny_rule_{}", idx),
                        expression: expr.to_string(),
                        action: "deny".to_string(),
                    });
                }
            }
        }

        // Parse rules array if present (more structured format)
        if let Some(rules_array) = value.get("rules").and_then(|v| v.as_array()) {
            for rule_value in rules_array {
                if let Some(id) = rule_value.get("id").and_then(|v| v.as_str()) {
                    let expression = rule_value
                        .get("expression")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let action = rule_value
                        .get("action")
                        .and_then(|v| v.as_str())
                        .unwrap_or("deny")
                        .to_string();

                    rules.push(PolicyRule {
                        id: id.to_string(),
                        expression,
                        action,
                    });
                }
            }
        }

        if rules.is_empty() {
            return Err("No valid rules found in policy bundle".to_string());
        }

        self.bundle = Some(PolicyBundle {
            version,
            rules,
            signature,
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Envelope, EnvelopeKind};

    fn text_envelope(value: &str) -> Envelope {
        Envelope::new(EnvelopeKind::Text(value.to_string()))
    }

    fn audio_envelope(bytes: &[u8]) -> Envelope {
        Envelope::new(EnvelopeKind::Audio(bytes.to_vec()))
    }

    #[test]
    fn test_default_policy_allows_text() {
        let engine = DefaultPolicyEngine::with_default_policy();
        let envelope = text_envelope("Text");
        let metrics = DeviceMetrics::default();

        let result = engine.evaluate("test_stage", &envelope, &metrics);
        assert!(result.allowed);
    }

    #[test]
    fn test_default_policy_allows_audio_raw() {
        let engine = DefaultPolicyEngine::with_default_policy();
        let envelope = audio_envelope(&[0, 1, 2]);
        let metrics = DeviceMetrics::default();

        let result = engine.evaluate("test_stage", &envelope, &metrics);
        assert!(result.allowed);
        assert!(result.reason.is_some());
        assert!(result.reason.unwrap().contains("all policy checks passed"));
    }

    #[test]
    fn test_load_yaml_policy_input_kind_rule() {
        let yaml_content = r#"
version: "0.1.0"
deny_cloud_if:
  - input.kind == "SensitiveData"
signature: "test"
"#;

        let mut engine = DefaultPolicyEngine::new();
        let result = engine.load_policies(yaml_content.as_bytes().to_vec());
        assert!(result.is_ok());

        let envelope = text_envelope("SensitiveData");
        let metrics = DeviceMetrics::default();

        let policy_result = engine.evaluate("test", &envelope, &metrics);
        assert!(!policy_result.allowed);
    }

    #[test]
    fn test_load_json_policy_input_kind_rule() {
        let json_content = r#"{
            "version": "0.1.0",
            "deny_cloud_if": [
                "input.kind == \"AudioRaw\""
            ],
            "signature": "test"
        }"#;

        let mut engine = DefaultPolicyEngine::new();
        let result = engine.load_policies(json_content.as_bytes().to_vec());
        assert!(result.is_ok());

        let envelope = audio_envelope(&[1, 2, 3]);
        let metrics = DeviceMetrics::default();

        let policy_result = engine.evaluate("test", &envelope, &metrics);
        assert!(!policy_result.allowed);
    }

    #[test]
    fn test_no_policy_allows() {
        let engine = DefaultPolicyEngine::new();
        let envelope = text_envelope("Text");
        let metrics = DeviceMetrics::default();

        let result = engine.evaluate("test_stage", &envelope, &metrics);
        assert!(result.allowed);
    }
}
