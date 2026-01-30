//! Telemetry module - Observability, metrics collection, and session tracking.
//!
//! This module provides:
//! - `Telemetry` - Basic event logging (v0.0.5)
//! - `SessionMetrics` - Session-based aggregation (v0.0.6)
//! - `ApiCallMetric` - Per-model usage tracking (v0.0.6)
//! - `TelemetryExport` - Export format for web backend (v0.0.6)

mod events;
mod session;

pub use events::{
    get_global_log_level, set_global_log_level, should_log, LogLevel, Severity, Telemetry,
    TelemetryEntry, TelemetryEvent,
};
pub use session::{
    ApiCallMetric, ErrorCategory, ErrorSummary, InferenceStats, SessionManager, SessionMetrics,
    TelemetryExport,
};
