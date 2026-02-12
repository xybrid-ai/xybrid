//! Telemetry events module - Basic event logging (v0.0.5).
//!
//! The Telemetry module collects and exports metrics, traces, and logs for monitoring
//! orchestrator performance, routing decisions, and system health.
//!
//! For MVP, this implements simple structured JSON logging to stdout following
//! OpenTelemetry JSON format. Future versions will integrate with full OpenTelemetry
//! SDK for distributed tracing and metrics export.
//!
//! # Log Levels
//!
//! Use `set_global_log_level()` to control both telemetry and library logging:
//! - `LogLevel::Quiet`: Errors only, minimal output
//! - `LogLevel::Normal`: Clean output - no telemetry JSON, no library logs (default)
//! - `LogLevel::Verbose`: Show telemetry events (INFO+), library warnings
//! - `LogLevel::VeryVerbose`: All telemetry (DEBUG+), all library logs

use serde_json::json;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// Global Log Level Configuration
// =============================================================================

/// Global log level for controlling verbosity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Errors only, suppress most output
    Quiet = 0,
    /// Info level and above (default)
    Normal = 1,
    /// All telemetry including debug
    Verbose = 2,
    /// All telemetry + library debug logs (very noisy)
    VeryVerbose = 3,
}

impl LogLevel {
    /// Convert to telemetry Severity threshold.
    ///
    /// By default (Normal), all telemetry JSON is suppressed to keep CLI output clean.
    /// Use -v to see telemetry events, -vv for everything including library debug logs.
    pub fn to_min_severity(&self) -> Severity {
        match self {
            // Quiet and Normal: Suppress all telemetry JSON for clean CLI output
            // Only actual errors (not telemetry) will be shown
            LogLevel::Quiet | LogLevel::Normal => Severity::Error,
            // Verbose: Show INFO and above telemetry
            LogLevel::Verbose => Severity::Info,
            // VeryVerbose: Show all telemetry including DEBUG
            LogLevel::VeryVerbose => Severity::Debug,
        }
    }

    /// Convert to llama.cpp verbosity level.
    /// 0 = silent, 1 = errors, 2 = warnings, 3 = info, 4 = debug
    pub fn to_llamacpp_verbosity(&self) -> i32 {
        match self {
            LogLevel::Quiet => 0,
            LogLevel::Normal => 0,  // Suppress library logs at normal level
            LogLevel::Verbose => 2, // Show warnings and errors
            LogLevel::VeryVerbose => 4, // Show all library logs
        }
    }

    /// Create from integer value.
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => LogLevel::Quiet,
            1 => LogLevel::Normal,
            2 => LogLevel::Verbose,
            _ => LogLevel::VeryVerbose,
        }
    }
}

// Global log level storage
static GLOBAL_LOG_LEVEL: AtomicU8 = AtomicU8::new(1); // Default: Normal

/// Set the global log level for all xybrid logging.
///
/// This affects:
/// - Telemetry event filtering (JSON logs)
/// - llama.cpp/ggml library logging
///
/// # Example
///
/// ```rust
/// use xybrid_core::telemetry::{set_global_log_level, LogLevel};
///
/// // Quiet mode - errors only
/// set_global_log_level(LogLevel::Quiet);
///
/// // Verbose mode - all debug logs
/// set_global_log_level(LogLevel::Verbose);
/// ```
pub fn set_global_log_level(level: LogLevel) {
    GLOBAL_LOG_LEVEL.store(level as u8, Ordering::SeqCst);

    // Also update llama.cpp verbosity if available
    #[cfg(feature = "llm-llamacpp")]
    {
        crate::runtime_adapter::llama_cpp::llama_log_set_verbosity(level.to_llamacpp_verbosity());
    }
}

/// Get the current global log level.
pub fn get_global_log_level() -> LogLevel {
    LogLevel::from_u8(GLOBAL_LOG_LEVEL.load(Ordering::SeqCst))
}

/// Check if the given severity should be logged at the current global level.
pub fn should_log(severity: Severity) -> bool {
    let level = get_global_log_level();
    severity >= level.to_min_severity()
}

/// Telemetry event types.
#[derive(Debug, Clone, PartialEq)]
pub enum TelemetryEvent {
    StageStart,
    StageComplete,
    StageError,
    PolicyEvaluation,
    RoutingDecision,
    ExecutionStart,
    ExecutionComplete,
    ExecutionError,
    ControlSync,
}

impl TelemetryEvent {
    pub fn as_str(&self) -> &'static str {
        match self {
            TelemetryEvent::StageStart => "stage_start",
            TelemetryEvent::StageComplete => "stage_complete",
            TelemetryEvent::StageError => "stage_error",
            TelemetryEvent::PolicyEvaluation => "policy_evaluation",
            TelemetryEvent::RoutingDecision => "routing_decision",
            TelemetryEvent::ExecutionStart => "execution_start",
            TelemetryEvent::ExecutionComplete => "execution_complete",
            TelemetryEvent::ExecutionError => "execution_error",
            TelemetryEvent::ControlSync => "control_sync",
        }
    }
}

/// Telemetry severity level.
///
/// Levels are ordered from most verbose (Debug) to least verbose (Error).
/// When filtering, a min_severity of Info means Debug logs are excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Debug => "DEBUG",
            Severity::Info => "INFO",
            Severity::Warn => "WARN",
            Severity::Error => "ERROR",
        }
    }

    /// Convert from integer level (for FFI compatibility).
    /// 0 = Debug, 1 = Info, 2 = Warn, 3 = Error
    pub fn from_level(level: u8) -> Self {
        match level {
            0 => Severity::Debug,
            1 => Severity::Info,
            2 => Severity::Warn,
            _ => Severity::Error,
        }
    }

    /// Convert to integer level.
    pub fn to_level(&self) -> u8 {
        *self as u8
    }
}

/// Structured telemetry log entry following OpenTelemetry JSON format.
#[derive(Debug, Clone)]
pub struct TelemetryEntry {
    pub timestamp: u64,
    pub severity: Severity,
    pub event: TelemetryEvent,
    pub message: String,
    pub attributes: serde_json::Value,
}

impl TelemetryEntry {
    /// Create a new telemetry entry.
    pub fn new(
        severity: Severity,
        event: TelemetryEvent,
        message: String,
        attributes: serde_json::Value,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            timestamp,
            severity,
            event,
            message,
            attributes,
        }
    }

    /// Convert to OpenTelemetry JSON format.
    pub fn to_json(&self) -> String {
        let json_obj = json!({
            "timestamp": self.timestamp,
            "severity": self.severity.as_str(),
            "event": self.event.as_str(),
            "message": self.message,
            "attributes": self.attributes
        });
        serde_json::to_string(&json_obj).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Telemetry collector for observability data.
///
/// For MVP, this emits structured JSON logs to stdout.
/// Future versions will support multiple exporters (file, network, etc.).
///
/// # Filtering
///
/// The `min_severity` field controls which events are emitted:
/// - `Severity::Debug`: All events (most verbose)
/// - `Severity::Info`: Info, Warn, Error (default)
/// - `Severity::Warn`: Warn and Error only
/// - `Severity::Error`: Errors only (least verbose)
pub struct Telemetry {
    enabled: bool,
    min_severity: Severity,
}

impl Telemetry {
    /// Creates a new telemetry instance with default settings (Info level).
    pub fn new() -> Self {
        Self {
            enabled: true,
            min_severity: Severity::Info,
        }
    }

    /// Creates a new telemetry instance with enabled/disabled state.
    pub fn with_enabled(enabled: bool) -> Self {
        Self {
            enabled,
            min_severity: Severity::Info,
        }
    }

    /// Creates a new telemetry instance with custom minimum severity.
    pub fn with_min_severity(min_severity: Severity) -> Self {
        Self {
            enabled: true,
            min_severity,
        }
    }

    /// Set the minimum severity level for emitting events.
    pub fn set_min_severity(&mut self, min_severity: Severity) {
        self.min_severity = min_severity;
    }

    /// Get the current minimum severity level.
    pub fn min_severity(&self) -> Severity {
        self.min_severity
    }

    /// Emit a telemetry entry if it meets the minimum severity threshold.
    ///
    /// The effective threshold is the stricter of:
    /// - The instance's `min_severity`
    /// - The global log level's severity threshold
    pub fn emit(&self, entry: TelemetryEntry) {
        if !self.enabled {
            return;
        }

        // Use the stricter of instance min_severity and global log level
        let global_min = get_global_log_level().to_min_severity();
        let effective_min = if self.min_severity > global_min {
            self.min_severity
        } else {
            global_min
        };

        if entry.severity >= effective_min {
            println!("{}", entry.to_json());
        }
    }

    /// Log a stage start event.
    pub fn log_stage_start(&self, stage_name: &str) {
        let entry = TelemetryEntry::new(
            Severity::Info,
            TelemetryEvent::StageStart,
            format!("Stage '{}' started", stage_name),
            json!({
                "stage": stage_name
            }),
        );
        self.emit(entry);
    }

    /// Log a stage completion event.
    pub fn log_stage_complete(
        &self,
        stage_name: &str,
        target: &str,
        latency_ms: u32,
        additional_attrs: Option<serde_json::Value>,
    ) {
        let mut attrs = json!({
            "stage": stage_name,
            "target": target,
            "latency_ms": latency_ms
        });

        if let Some(extra) = additional_attrs {
            if let Some(attrs_obj) = attrs.as_object_mut() {
                if let Some(extra_obj) = extra.as_object() {
                    for (k, v) in extra_obj {
                        attrs_obj.insert(k.clone(), v.clone());
                    }
                }
            }
        }

        let entry = TelemetryEntry::new(
            Severity::Info,
            TelemetryEvent::StageComplete,
            format!(
                "Stage '{}' completed on {} in {}ms",
                stage_name, target, latency_ms
            ),
            attrs,
        );
        self.emit(entry);
    }

    /// Log a stage error event.
    pub fn log_stage_error(&self, stage_name: &str, error: &str) {
        let entry = TelemetryEntry::new(
            Severity::Error,
            TelemetryEvent::StageError,
            format!("Stage '{}' failed: {}", stage_name, error),
            json!({
                "stage": stage_name,
                "error": error
            }),
        );
        self.emit(entry);
    }

    /// Log a policy evaluation event.
    pub fn log_policy_evaluation(&self, stage_name: &str, allowed: bool, reason: Option<&str>) {
        let mut attrs = json!({
            "stage": stage_name,
            "allowed": allowed
        });

        if let Some(reason_str) = reason {
            attrs
                .as_object_mut()
                .unwrap()
                .insert("reason".to_string(), json!(reason_str));
        }

        let entry = TelemetryEntry::new(
            Severity::Debug,
            TelemetryEvent::PolicyEvaluation,
            format!(
                "Policy evaluation for '{}': {}",
                stage_name,
                if allowed { "allowed" } else { "denied" }
            ),
            attrs,
        );
        self.emit(entry);
    }

    /// Log a routing decision event.
    pub fn log_routing_decision(&self, stage_name: &str, target: &str, reason: &str) {
        let entry = TelemetryEntry::new(
            Severity::Info,
            TelemetryEvent::RoutingDecision,
            format!(
                "Routing decision for '{}': {} ({})",
                stage_name, target, reason
            ),
            json!({
                "stage": stage_name,
                "target": target,
                "reason": reason
            }),
        );
        self.emit(entry);
    }

    /// Log an execution start event.
    pub fn log_execution_start(&self, stage_name: &str, target: &str) {
        let entry = TelemetryEntry::new(
            Severity::Debug,
            TelemetryEvent::ExecutionStart,
            format!("Execution started for '{}' on {}", stage_name, target),
            json!({
                "stage": stage_name,
                "target": target
            }),
        );
        self.emit(entry);
    }

    /// Log an execution completion event.
    pub fn log_execution_complete(&self, stage_name: &str, target: &str, execution_time_ms: u32) {
        let entry = TelemetryEntry::new(
            Severity::Debug,
            TelemetryEvent::ExecutionComplete,
            format!(
                "Execution completed for '{}' on {} in {}ms",
                stage_name, target, execution_time_ms
            ),
            json!({
                "stage": stage_name,
                "target": target,
                "execution_time_ms": execution_time_ms
            }),
        );
        self.emit(entry);
    }

    /// Log an execution error event.
    pub fn log_execution_error(&self, stage_name: &str, target: &str, error: &str) {
        let entry = TelemetryEntry::new(
            Severity::Error,
            TelemetryEvent::ExecutionError,
            format!(
                "Execution failed for '{}' on {}: {}",
                stage_name, target, error
            ),
            json!({
                "stage": stage_name,
                "target": target,
                "error": error
            }),
        );
        self.emit(entry);
    }

    /// Log a control sync event with custom severity and attributes.
    pub fn log_control_sync_event(
        &self,
        severity: Severity,
        action: &str,
        attributes: serde_json::Value,
    ) {
        let mut attrs = json!({
            "action": action
        });

        if let Some(attrs_obj) = attrs.as_object_mut() {
            if let Some(extra) = attributes.as_object() {
                for (key, value) in extra {
                    attrs_obj.insert(key.clone(), value.clone());
                }
            }
        }

        let entry = TelemetryEntry::new(
            severity,
            TelemetryEvent::ControlSync,
            format!("control_sync {}", action),
            attrs,
        );
        self.emit(entry);
    }

    /// Log bootstrap start event.
    pub fn log_bootstrap_start(&self) {
        let entry = TelemetryEntry::new(
            Severity::Info,
            TelemetryEvent::StageStart,
            "Bootstrap started".to_string(),
            json!({
                "bootstrap": true
            }),
        );
        self.emit(entry);
    }

    /// Log bootstrap completion event.
    pub fn log_bootstrap_complete(&self) {
        let entry = TelemetryEntry::new(
            Severity::Info,
            TelemetryEvent::StageComplete,
            "Bootstrap completed".to_string(),
            json!({
                "bootstrap": true
            }),
        );
        self.emit(entry);
    }

    /// Log a custom event with arbitrary attributes.
    pub fn log_custom(
        &self,
        severity: Severity,
        event: TelemetryEvent,
        message: String,
        attributes: serde_json::Value,
    ) {
        let entry = TelemetryEntry::new(severity, event, message, attributes);
        self.emit(entry);
    }

    /// Check if telemetry is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable telemetry.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_creation() {
        let telemetry = Telemetry::new();
        assert!(telemetry.is_enabled());
    }

    #[test]
    fn test_telemetry_disabled() {
        let telemetry = Telemetry::with_enabled(false);
        assert!(!telemetry.is_enabled());
    }

    #[test]
    fn test_log_stage_start() {
        let telemetry = Telemetry::new();
        telemetry.log_stage_start("test_stage");
    }

    #[test]
    fn test_log_stage_complete() {
        let telemetry = Telemetry::new();
        telemetry.log_stage_complete("test_stage", "local", 100, None);
    }

    #[test]
    fn test_log_policy_evaluation() {
        let telemetry = Telemetry::new();
        telemetry.log_policy_evaluation("test_stage", true, Some("policy passed"));
        telemetry.log_policy_evaluation("test_stage", false, None);
    }

    #[test]
    fn test_log_routing_decision() {
        let telemetry = Telemetry::new();
        telemetry.log_routing_decision("test_stage", "local", "high_latency");
    }

    #[test]
    fn test_telemetry_entry_json_format() {
        let entry = TelemetryEntry::new(
            Severity::Info,
            TelemetryEvent::StageComplete,
            "Test message".to_string(),
            json!({"test": "value"}),
        );

        let json_str = entry.to_json();
        assert!(json_str.contains("\"timestamp\""));
        assert!(json_str.contains("\"severity\":\"INFO\""));
        assert!(json_str.contains("\"event\":\"stage_complete\""));
        assert!(json_str.contains("\"message\":\"Test message\""));
        assert!(json_str.contains("\"attributes\""));
    }

    #[test]
    fn test_telemetry_entry_timestamp() {
        let entry1 = TelemetryEntry::new(
            Severity::Info,
            TelemetryEvent::StageStart,
            "Test".to_string(),
            json!({}),
        );

        std::thread::sleep(std::time::Duration::from_millis(1));

        let entry2 = TelemetryEntry::new(
            Severity::Info,
            TelemetryEvent::StageStart,
            "Test".to_string(),
            json!({}),
        );

        assert!(entry2.timestamp >= entry1.timestamp);
    }

    #[test]
    fn test_telemetry_disabled_no_output() {
        let mut telemetry = Telemetry::new();
        telemetry.set_enabled(false);
        telemetry.log_stage_start("test");
    }
}
