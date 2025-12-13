//! Session-based telemetry aggregation (v0.0.6).
//!
//! This module provides session-level metrics tracking and aggregation for
//! usage analytics, billing, and dashboard visualization.

use crate::device::HardwareCapabilities;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Error category for debugging and analytics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Bundle/model file issues
    ModelLoading,
    /// Input format/conversion errors
    Preprocessing,
    /// Runtime execution errors
    Inference,
    /// Output format errors
    Postprocessing,
    /// Registry/cloud connectivity issues
    Network,
    /// GPU/NPU initialization failures
    Hardware,
    /// Out of memory conditions
    Memory,
    /// Unknown/uncategorized errors
    Unknown,
}

impl ErrorCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCategory::ModelLoading => "model_loading",
            ErrorCategory::Preprocessing => "preprocessing",
            ErrorCategory::Inference => "inference",
            ErrorCategory::Postprocessing => "postprocessing",
            ErrorCategory::Network => "network",
            ErrorCategory::Hardware => "hardware",
            ErrorCategory::Memory => "memory",
            ErrorCategory::Unknown => "unknown",
        }
    }

    /// Categorize an error message into a category.
    pub fn from_error(error: &str) -> Self {
        let error_lower = error.to_lowercase();
        if error_lower.contains("load")
            || error_lower.contains("bundle")
            || error_lower.contains("model not found")
        {
            ErrorCategory::ModelLoading
        } else if error_lower.contains("preprocess")
            || error_lower.contains("input")
            || error_lower.contains("format")
        {
            ErrorCategory::Preprocessing
        } else if error_lower.contains("inference")
            || error_lower.contains("execute")
            || error_lower.contains("onnx")
        {
            ErrorCategory::Inference
        } else if error_lower.contains("postprocess") || error_lower.contains("output") {
            ErrorCategory::Postprocessing
        } else if error_lower.contains("network")
            || error_lower.contains("connection")
            || error_lower.contains("registry")
            || error_lower.contains("http")
        {
            ErrorCategory::Network
        } else if error_lower.contains("gpu")
            || error_lower.contains("metal")
            || error_lower.contains("cuda")
            || error_lower.contains("npu")
        {
            ErrorCategory::Hardware
        } else if error_lower.contains("memory") || error_lower.contains("oom") {
            ErrorCategory::Memory
        } else {
            ErrorCategory::Unknown
        }
    }
}

/// Error summary for aggregated reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    /// Error category
    pub category: ErrorCategory,
    /// Error message
    pub message: String,
    /// Number of occurrences
    pub count: u64,
    /// First occurrence timestamp (ms since epoch)
    pub first_seen: u64,
    /// Last occurrence timestamp (ms since epoch)
    pub last_seen: u64,
    /// Model ID if applicable
    pub model_id: Option<String>,
}

/// Per-model API call metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCallMetric {
    /// Model identifier
    pub model_id: String,
    /// Model version
    pub version: String,
    /// Total number of calls
    pub call_count: u64,
    /// Total latency across all calls (ms)
    pub total_latency_ms: u64,
    /// Average latency (ms)
    pub avg_latency_ms: u64,
    /// Number of errors
    pub error_count: u64,
    /// Last call timestamp (ms since epoch)
    pub last_called: u64,
}

impl ApiCallMetric {
    pub fn new(model_id: String, version: String) -> Self {
        Self {
            model_id,
            version,
            call_count: 0,
            total_latency_ms: 0,
            avg_latency_ms: 0,
            error_count: 0,
            last_called: 0,
        }
    }

    /// Record a successful call.
    pub fn record_call(&mut self, latency_ms: u64) {
        self.call_count += 1;
        self.total_latency_ms += latency_ms;
        self.avg_latency_ms = self.total_latency_ms / self.call_count;
        self.last_called = current_timestamp_ms();
    }

    /// Record an error.
    pub fn record_error(&mut self) {
        self.error_count += 1;
        self.last_called = current_timestamp_ms();
    }
}

/// Inference statistics for detailed performance analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    /// Model identifier
    pub model_id: String,
    /// Tokens processed per second (for text models)
    pub tokens_per_second: Option<f32>,
    /// Audio samples processed per second (for audio models)
    pub samples_per_second: Option<f32>,
    /// Peak memory usage during inference (MB)
    pub memory_peak_mb: u64,
    /// Execution target (local, cloud, fallback)
    pub execution_target: String,
    /// Hardware used (cpu, gpu, npu)
    pub hardware_used: String,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// Session metrics aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    /// Unique session identifier
    pub session_id: String,
    /// Device identifier (persistent across sessions)
    pub device_id: String,
    /// Session start timestamp (ms since epoch)
    pub started_at: u64,
    /// Session end timestamp (ms since epoch, None if still active)
    pub ended_at: Option<u64>,
    /// SDK version
    pub sdk_version: String,

    // Aggregated metrics
    /// Total number of inference calls
    pub total_inferences: u64,
    /// Total latency across all inferences (ms)
    pub total_latency_ms: u64,
    /// Average latency per inference (ms)
    pub avg_latency_ms: u64,
    /// List of models used in this session
    pub models_used: Vec<String>,
    /// Total error count
    pub error_count: u64,

    // Per-model breakdown
    /// Metrics by model
    pub by_model: HashMap<String, ApiCallMetric>,
    /// Error summaries
    pub errors: Vec<ErrorSummary>,

    // Device snapshot at session start
    /// Hardware capabilities snapshot
    pub hardware_capabilities: Option<HardwareCapabilities>,
}

impl SessionMetrics {
    /// Create a new session.
    pub fn new(device_id: String) -> Self {
        Self {
            session_id: Uuid::new_v4().to_string(),
            device_id,
            started_at: current_timestamp_ms(),
            ended_at: None,
            sdk_version: env!("CARGO_PKG_VERSION").to_string(),
            total_inferences: 0,
            total_latency_ms: 0,
            avg_latency_ms: 0,
            models_used: Vec::new(),
            error_count: 0,
            by_model: HashMap::new(),
            errors: Vec::new(),
            hardware_capabilities: None,
        }
    }

    /// Set hardware capabilities snapshot.
    pub fn set_hardware_capabilities(&mut self, caps: HardwareCapabilities) {
        self.hardware_capabilities = Some(caps);
    }

    /// Record a successful inference call.
    pub fn record_inference(&mut self, model_id: &str, version: &str, latency_ms: u64) {
        self.total_inferences += 1;
        self.total_latency_ms += latency_ms;
        self.avg_latency_ms = self.total_latency_ms / self.total_inferences;

        // Track unique models
        if !self.models_used.contains(&model_id.to_string()) {
            self.models_used.push(model_id.to_string());
        }

        // Update per-model metrics
        let key = format!("{}@{}", model_id, version);
        let metric = self
            .by_model
            .entry(key)
            .or_insert_with(|| ApiCallMetric::new(model_id.to_string(), version.to_string()));
        metric.record_call(latency_ms);
    }

    /// Record an error.
    pub fn record_error(&mut self, model_id: Option<&str>, error: &str) {
        self.error_count += 1;
        let category = ErrorCategory::from_error(error);
        let now = current_timestamp_ms();

        // Find existing error summary or create new one
        if let Some(summary) = self
            .errors
            .iter_mut()
            .find(|e| e.category == category && e.message == error)
        {
            summary.count += 1;
            summary.last_seen = now;
        } else {
            self.errors.push(ErrorSummary {
                category,
                message: error.to_string(),
                count: 1,
                first_seen: now,
                last_seen: now,
                model_id: model_id.map(|s| s.to_string()),
            });
        }

        // Update per-model error count
        if let Some(mid) = model_id {
            if let Some(metric) = self.by_model.values_mut().find(|m| m.model_id == mid) {
                metric.record_error();
            }
        }
    }

    /// End the session.
    pub fn end_session(&mut self) {
        self.ended_at = Some(current_timestamp_ms());
    }

    /// Check if session is active.
    pub fn is_active(&self) -> bool {
        self.ended_at.is_none()
    }

    /// Get session duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        let end = self.ended_at.unwrap_or_else(current_timestamp_ms);
        end.saturating_sub(self.started_at)
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Convert to pretty JSON string.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Telemetry export format for web backend ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryExport {
    /// Export format version
    pub version: String,
    /// Session information
    pub session: SessionInfo,
    /// Hardware capabilities
    pub hardware: Option<HardwareInfo>,
    /// Aggregated metrics
    pub metrics: MetricsSummary,
    /// Error summaries
    pub errors: Vec<ErrorSummary>,
}

/// Session information for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub device_id: String,
    pub platform: String,
    pub sdk_version: String,
    pub started_at: String, // ISO 8601
    pub ended_at: Option<String>,
}

/// Hardware info for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub has_gpu: bool,
    pub gpu_type: String,
    pub has_npu: bool,
    pub npu_type: String,
    pub memory_total_mb: u64,
    pub battery_level: u8,
    pub thermal_state: String,
    pub platform: String,
}

/// Metrics summary for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_inferences: u64,
    pub total_latency_ms: u64,
    pub avg_latency_ms: u64,
    pub error_count: u64,
    pub by_model: Vec<ApiCallMetric>,
}

impl TelemetryExport {
    /// Create export from session metrics.
    pub fn from_session(session: &SessionMetrics) -> Self {
        let hardware = session.hardware_capabilities.as_ref().map(|caps| HardwareInfo {
            has_gpu: caps.has_gpu,
            gpu_type: caps.gpu_type.as_str().to_string(),
            has_npu: caps.has_npu,
            npu_type: caps.npu_type.as_str().to_string(),
            memory_total_mb: caps.memory_total_mb,
            battery_level: caps.battery_level,
            thermal_state: caps.thermal_state.as_str().to_string(),
            platform: caps.platform.as_str().to_string(),
        });

        let platform = session
            .hardware_capabilities
            .as_ref()
            .map(|c| c.platform.as_str().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        Self {
            version: "2.0".to_string(),
            session: SessionInfo {
                session_id: session.session_id.clone(),
                device_id: session.device_id.clone(),
                platform,
                sdk_version: session.sdk_version.clone(),
                started_at: timestamp_to_iso8601(session.started_at),
                ended_at: session.ended_at.map(timestamp_to_iso8601),
            },
            hardware,
            metrics: MetricsSummary {
                total_inferences: session.total_inferences,
                total_latency_ms: session.total_latency_ms,
                avg_latency_ms: session.avg_latency_ms,
                error_count: session.error_count,
                by_model: session.by_model.values().cloned().collect(),
            },
            errors: session.errors.clone(),
        }
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Convert to pretty JSON string.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Global session manager for singleton access.
pub struct SessionManager {
    current_session: Arc<Mutex<Option<SessionMetrics>>>,
    device_id: String,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new(device_id: String) -> Self {
        Self {
            current_session: Arc::new(Mutex::new(None)),
            device_id,
        }
    }

    /// Start a new session.
    pub fn start_session(&self) -> String {
        let mut session = self.current_session.lock().unwrap();
        let new_session = SessionMetrics::new(self.device_id.clone());
        let session_id = new_session.session_id.clone();
        *session = Some(new_session);
        session_id
    }

    /// Get current session metrics (clone).
    pub fn get_session(&self) -> Option<SessionMetrics> {
        let session = self.current_session.lock().unwrap();
        session.clone()
    }

    /// Record an inference call.
    pub fn record_inference(&self, model_id: &str, version: &str, latency_ms: u64) {
        if let Ok(mut session) = self.current_session.lock() {
            if let Some(ref mut s) = *session {
                s.record_inference(model_id, version, latency_ms);
            }
        }
    }

    /// Record an error.
    pub fn record_error(&self, model_id: Option<&str>, error: &str) {
        if let Ok(mut session) = self.current_session.lock() {
            if let Some(ref mut s) = *session {
                s.record_error(model_id, error);
            }
        }
    }

    /// Set hardware capabilities for current session.
    pub fn set_hardware_capabilities(&self, caps: HardwareCapabilities) {
        if let Ok(mut session) = self.current_session.lock() {
            if let Some(ref mut s) = *session {
                s.set_hardware_capabilities(caps);
            }
        }
    }

    /// End current session and return export.
    pub fn end_session(&self) -> Option<TelemetryExport> {
        let mut session = self.current_session.lock().unwrap();
        if let Some(ref mut s) = *session {
            s.end_session();
            Some(TelemetryExport::from_session(s))
        } else {
            None
        }
    }

    /// Export current session without ending it.
    pub fn export_session(&self) -> Option<TelemetryExport> {
        let session = self.current_session.lock().unwrap();
        session.as_ref().map(TelemetryExport::from_session)
    }

    /// Reset session (start fresh).
    pub fn reset(&self) {
        let mut session = self.current_session.lock().unwrap();
        *session = Some(SessionMetrics::new(self.device_id.clone()));
    }
}

/// Get current timestamp in milliseconds since epoch.
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Convert timestamp to ISO 8601 string.
fn timestamp_to_iso8601(timestamp_ms: u64) -> String {
    // Simple conversion - in production use chrono crate
    let secs = timestamp_ms / 1000;
    let ms = timestamp_ms % 1000;
    format!(
        "{}Z",
        chrono::DateTime::from_timestamp(secs as i64, (ms * 1_000_000) as u32)
            .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S%.3f").to_string())
            .unwrap_or_else(|| format!("{}.{:03}", secs, ms))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_metrics_new() {
        let session = SessionMetrics::new("device-123".to_string());
        assert!(!session.session_id.is_empty());
        assert_eq!(session.device_id, "device-123");
        assert!(session.is_active());
        assert_eq!(session.total_inferences, 0);
    }

    #[test]
    fn test_record_inference() {
        let mut session = SessionMetrics::new("device-123".to_string());
        session.record_inference("wav2vec2", "1.0", 250);
        session.record_inference("wav2vec2", "1.0", 350);

        assert_eq!(session.total_inferences, 2);
        assert_eq!(session.total_latency_ms, 600);
        assert_eq!(session.avg_latency_ms, 300);
        assert_eq!(session.models_used, vec!["wav2vec2"]);
    }

    #[test]
    fn test_record_error() {
        let mut session = SessionMetrics::new("device-123".to_string());
        session.record_error(Some("wav2vec2"), "Network connection failed");
        session.record_error(Some("wav2vec2"), "Network connection failed");
        session.record_error(None, "Model not found");

        assert_eq!(session.error_count, 3);
        assert_eq!(session.errors.len(), 2); // Two unique errors
        assert_eq!(session.errors[0].count, 2); // First error occurred twice
    }

    #[test]
    fn test_error_category() {
        assert_eq!(
            ErrorCategory::from_error("Failed to load model"),
            ErrorCategory::ModelLoading
        );
        assert_eq!(
            ErrorCategory::from_error("Network connection refused"),
            ErrorCategory::Network
        );
        assert_eq!(
            ErrorCategory::from_error("GPU initialization failed"),
            ErrorCategory::Hardware
        );
        assert_eq!(
            ErrorCategory::from_error("Out of memory"),
            ErrorCategory::Memory
        );
        assert_eq!(
            ErrorCategory::from_error("Something weird happened"),
            ErrorCategory::Unknown
        );
    }

    #[test]
    fn test_session_end() {
        let mut session = SessionMetrics::new("device-123".to_string());
        assert!(session.is_active());

        session.end_session();
        assert!(!session.is_active());
        assert!(session.ended_at.is_some());
    }

    #[test]
    fn test_telemetry_export() {
        let mut session = SessionMetrics::new("device-123".to_string());
        session.record_inference("wav2vec2", "1.0", 250);
        session.end_session();

        let export = TelemetryExport::from_session(&session);
        assert_eq!(export.version, "2.0");
        assert_eq!(export.session.device_id, "device-123");
        assert_eq!(export.metrics.total_inferences, 1);
    }

    #[test]
    fn test_session_manager() {
        let manager = SessionManager::new("device-123".to_string());
        let session_id = manager.start_session();
        assert!(!session_id.is_empty());

        manager.record_inference("wav2vec2", "1.0", 250);

        let session = manager.get_session().unwrap();
        assert_eq!(session.total_inferences, 1);

        let export = manager.end_session().unwrap();
        assert_eq!(export.metrics.total_inferences, 1);
    }

    #[test]
    fn test_api_call_metric() {
        let mut metric = ApiCallMetric::new("wav2vec2".to_string(), "1.0".to_string());
        metric.record_call(100);
        metric.record_call(200);
        metric.record_error();

        assert_eq!(metric.call_count, 2);
        assert_eq!(metric.total_latency_ms, 300);
        assert_eq!(metric.avg_latency_ms, 150);
        assert_eq!(metric.error_count, 1);
    }
}
