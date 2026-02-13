//! Pipeline execution result types for FFI bindings.
//!
//! These types mirror the Flutter SDK's result types and are designed to be
//! FFI-safe for use across platform bindings (Flutter, Kotlin, Swift).
//!
//! # Example
//!
//! ```rust
//! use xybrid_sdk::pipeline::{FfiPipelineExecutionResult, FfiStageExecutionResult};
//!
//! // Create a successful result
//! let result = FfiPipelineExecutionResult::success(
//!     "text",
//!     Some("Hello, world!".to_string()),
//!     None,
//!     150,
//!     "whisper-tiny",
//! );
//! assert!(result.success);
//! assert_eq!(result.text, Some("Hello, world!".to_string()));
//!
//! // Create an error result
//! let error_result = FfiPipelineExecutionResult::error("Model not found", "whisper-tiny");
//! assert!(!error_result.success);
//! assert_eq!(error_result.error, Some("Model not found".to_string()));
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// Stage Execution Result (FFI-safe)
// ============================================================================

/// Result of a single stage execution (FFI-safe).
///
/// This struct is designed for use in platform bindings (Flutter, Kotlin, Swift)
/// where simple, serializable types are required.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::pipeline::FfiStageExecutionResult;
///
/// let stage = FfiStageExecutionResult {
///     stage_id: "asr".to_string(),
///     executed: true,
///     skip_reason: None,
///     target: Some("device".to_string()),
///     latency_ms: 45,
/// };
/// assert!(stage.executed);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FfiStageExecutionResult {
    /// Stage identifier
    pub stage_id: String,
    /// Whether the stage was executed
    pub executed: bool,
    /// Reason for skipping (if not executed)
    pub skip_reason: Option<String>,
    /// Execution target (e.g., "device", "cloud")
    pub target: Option<String>,
    /// Execution latency in milliseconds
    pub latency_ms: u32,
}

impl FfiStageExecutionResult {
    /// Create a new stage result for an executed stage.
    ///
    /// # Arguments
    ///
    /// * `stage_id` - Stage identifier
    /// * `target` - Execution target (e.g., "device", "cloud")
    /// * `latency_ms` - Execution latency in milliseconds
    pub fn executed(
        stage_id: impl Into<String>,
        target: impl Into<String>,
        latency_ms: u32,
    ) -> Self {
        Self {
            stage_id: stage_id.into(),
            executed: true,
            skip_reason: None,
            target: Some(target.into()),
            latency_ms,
        }
    }

    /// Create a new stage result for a skipped stage.
    ///
    /// # Arguments
    ///
    /// * `stage_id` - Stage identifier
    /// * `reason` - Reason for skipping
    pub fn skipped(stage_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            stage_id: stage_id.into(),
            executed: false,
            skip_reason: Some(reason.into()),
            target: None,
            latency_ms: 0,
        }
    }
}

// ============================================================================
// Pipeline Execution Result (FFI-safe)
// ============================================================================

/// Result from pipeline execution (FFI-safe).
///
/// This struct is designed for use in platform bindings (Flutter, Kotlin, Swift)
/// where simple, serializable types are required. It supports multiple output
/// types (text, embedding, audio) and includes detailed stage execution results.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::pipeline::{FfiPipelineExecutionResult, FfiStageExecutionResult};
///
/// // Successful text output
/// let result = FfiPipelineExecutionResult::success(
///     "text",
///     Some("Transcribed text".to_string()),
///     None,
///     250,
///     "whisper-tiny",
/// );
/// assert!(result.success);
/// assert_eq!(result.output_type, "text");
///
/// // Error result
/// let error = FfiPipelineExecutionResult::error("Connection timeout", "gpt-4o-mini");
/// assert!(!error.success);
/// assert!(error.error.is_some());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FfiPipelineExecutionResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Output type (e.g., "text", "embedding", "audio", "none")
    pub output_type: String,
    /// Text output (for ASR/NLP)
    pub text: Option<String>,
    /// Embedding output
    pub embedding: Option<Vec<f32>>,
    /// Audio output bytes length (actual bytes handled separately in FFI)
    pub audio_bytes_len: Option<usize>,
    /// Execution latency in milliseconds
    pub latency_ms: u32,
    /// Model ID used
    pub model_id: String,
    /// Stage results (for multi-stage pipelines)
    pub stages: Vec<FfiStageExecutionResult>,
    /// Total stages executed
    pub stages_executed: u32,
    /// Stages skipped (due to conditions)
    pub stages_skipped: u32,
}

impl FfiPipelineExecutionResult {
    /// Create a successful pipeline result.
    ///
    /// # Arguments
    ///
    /// * `output_type` - Type of output ("text", "embedding", "audio")
    /// * `text` - Text output (for ASR/NLP pipelines)
    /// * `embedding` - Embedding output (for embedding pipelines)
    /// * `latency_ms` - Total execution latency
    /// * `model_id` - ID of the model used
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_sdk::pipeline::FfiPipelineExecutionResult;
    ///
    /// // Text output from ASR
    /// let result = FfiPipelineExecutionResult::success(
    ///     "text",
    ///     Some("Hello world".to_string()),
    ///     None,
    ///     150,
    ///     "whisper-tiny",
    /// );
    ///
    /// // Embedding output
    /// let result = FfiPipelineExecutionResult::success(
    ///     "embedding",
    ///     None,
    ///     Some(vec![0.1, 0.2, 0.3]),
    ///     50,
    ///     "bge-small",
    /// );
    /// ```
    pub fn success(
        output_type: &str,
        text: Option<String>,
        embedding: Option<Vec<f32>>,
        latency_ms: u32,
        model_id: &str,
    ) -> Self {
        Self {
            success: true,
            error: None,
            output_type: output_type.to_string(),
            text,
            embedding,
            audio_bytes_len: None,
            latency_ms,
            model_id: model_id.to_string(),
            stages: Vec::new(),
            stages_executed: 1,
            stages_skipped: 0,
        }
    }

    /// Create a successful result with audio output.
    ///
    /// # Arguments
    ///
    /// * `audio_len` - Length of the audio bytes
    /// * `latency_ms` - Total execution latency
    /// * `model_id` - ID of the model used
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_sdk::pipeline::FfiPipelineExecutionResult;
    ///
    /// let result = FfiPipelineExecutionResult::success_audio(48000, 350, "kokoro-82m");
    /// assert_eq!(result.output_type, "audio");
    /// assert_eq!(result.audio_bytes_len, Some(48000));
    /// ```
    pub fn success_audio(audio_len: usize, latency_ms: u32, model_id: &str) -> Self {
        Self {
            success: true,
            error: None,
            output_type: "audio".to_string(),
            text: None,
            embedding: None,
            audio_bytes_len: Some(audio_len),
            latency_ms,
            model_id: model_id.to_string(),
            stages: Vec::new(),
            stages_executed: 1,
            stages_skipped: 0,
        }
    }

    /// Create an error pipeline result.
    ///
    /// # Arguments
    ///
    /// * `message` - Error message describing what went wrong
    /// * `model_id` - ID of the model that failed
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_sdk::pipeline::FfiPipelineExecutionResult;
    ///
    /// let result = FfiPipelineExecutionResult::error(
    ///     "Model file not found",
    ///     "whisper-tiny",
    /// );
    /// assert!(!result.success);
    /// assert_eq!(result.error, Some("Model file not found".to_string()));
    /// ```
    pub fn error(message: &str, model_id: &str) -> Self {
        Self {
            success: false,
            error: Some(message.to_string()),
            output_type: "none".to_string(),
            text: None,
            embedding: None,
            audio_bytes_len: None,
            latency_ms: 0,
            model_id: model_id.to_string(),
            stages: Vec::new(),
            stages_executed: 0,
            stages_skipped: 0,
        }
    }

    /// Add stage results to the pipeline result.
    ///
    /// This method updates `stages_executed` and `stages_skipped` counts
    /// based on the provided stage results.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_sdk::pipeline::{FfiPipelineExecutionResult, FfiStageExecutionResult};
    ///
    /// let mut result = FfiPipelineExecutionResult::success(
    ///     "text",
    ///     Some("output".to_string()),
    ///     None,
    ///     100,
    ///     "model",
    /// );
    ///
    /// result.with_stages(vec![
    ///     FfiStageExecutionResult::executed("stage1", "device", 50),
    ///     FfiStageExecutionResult::skipped("stage2", "condition not met"),
    /// ]);
    ///
    /// assert_eq!(result.stages_executed, 1);
    /// assert_eq!(result.stages_skipped, 1);
    /// ```
    pub fn with_stages(&mut self, stages: Vec<FfiStageExecutionResult>) {
        self.stages_executed = stages.iter().filter(|s| s.executed).count() as u32;
        self.stages_skipped = stages.iter().filter(|s| !s.executed).count() as u32;
        self.stages = stages;
    }

    /// Check if the result contains text output.
    pub fn has_text(&self) -> bool {
        self.text.is_some()
    }

    /// Check if the result contains embedding output.
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Check if the result contains audio output.
    pub fn has_audio(&self) -> bool {
        self.audio_bytes_len.is_some()
    }
}

impl Default for FfiPipelineExecutionResult {
    fn default() -> Self {
        Self {
            success: false,
            error: None,
            output_type: "none".to_string(),
            text: None,
            embedding: None,
            audio_bytes_len: None,
            latency_ms: 0,
            model_id: String::new(),
            stages: Vec::new(),
            stages_executed: 0,
            stages_skipped: 0,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // FfiStageExecutionResult Tests
    // ========================================================================

    #[test]
    fn test_stage_execution_result_executed() {
        let stage = FfiStageExecutionResult::executed("asr", "device", 45);
        assert_eq!(stage.stage_id, "asr");
        assert!(stage.executed);
        assert_eq!(stage.target, Some("device".to_string()));
        assert_eq!(stage.latency_ms, 45);
        assert!(stage.skip_reason.is_none());
    }

    #[test]
    fn test_stage_execution_result_skipped() {
        let stage = FfiStageExecutionResult::skipped("llm", "model too large");
        assert_eq!(stage.stage_id, "llm");
        assert!(!stage.executed);
        assert_eq!(stage.skip_reason, Some("model too large".to_string()));
        assert!(stage.target.is_none());
        assert_eq!(stage.latency_ms, 0);
    }

    #[test]
    fn test_stage_execution_result_default() {
        let stage = FfiStageExecutionResult::default();
        assert!(stage.stage_id.is_empty());
        assert!(!stage.executed);
        assert!(stage.skip_reason.is_none());
        assert!(stage.target.is_none());
        assert_eq!(stage.latency_ms, 0);
    }

    #[test]
    fn test_stage_execution_result_serialization() {
        let stage = FfiStageExecutionResult::executed("asr", "cloud", 100);
        let json = serde_json::to_string(&stage).unwrap();
        assert!(json.contains("\"stage_id\":\"asr\""));
        assert!(json.contains("\"executed\":true"));
        assert!(json.contains("\"target\":\"cloud\""));
        assert!(json.contains("\"latency_ms\":100"));

        let deserialized: FfiStageExecutionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(stage, deserialized);
    }

    // ========================================================================
    // FfiPipelineExecutionResult Tests
    // ========================================================================

    #[test]
    fn test_pipeline_execution_result_success_text() {
        let result = FfiPipelineExecutionResult::success(
            "text",
            Some("Hello world".to_string()),
            None,
            150,
            "whisper-tiny",
        );
        assert!(result.success);
        assert!(result.error.is_none());
        assert_eq!(result.output_type, "text");
        assert_eq!(result.text, Some("Hello world".to_string()));
        assert!(result.embedding.is_none());
        assert!(result.audio_bytes_len.is_none());
        assert_eq!(result.latency_ms, 150);
        assert_eq!(result.model_id, "whisper-tiny");
        assert_eq!(result.stages_executed, 1);
        assert_eq!(result.stages_skipped, 0);
    }

    #[test]
    fn test_pipeline_execution_result_success_embedding() {
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = FfiPipelineExecutionResult::success(
            "embedding",
            None,
            Some(embedding.clone()),
            50,
            "bge-small",
        );
        assert!(result.success);
        assert_eq!(result.output_type, "embedding");
        assert!(result.text.is_none());
        assert_eq!(result.embedding, Some(embedding));
        assert!(result.has_embedding());
        assert!(!result.has_text());
        assert!(!result.has_audio());
    }

    #[test]
    fn test_pipeline_execution_result_success_audio() {
        let result = FfiPipelineExecutionResult::success_audio(48000, 350, "kokoro-82m");
        assert!(result.success);
        assert_eq!(result.output_type, "audio");
        assert!(result.text.is_none());
        assert!(result.embedding.is_none());
        assert_eq!(result.audio_bytes_len, Some(48000));
        assert!(result.has_audio());
        assert!(!result.has_text());
        assert!(!result.has_embedding());
    }

    #[test]
    fn test_pipeline_execution_result_error() {
        let result = FfiPipelineExecutionResult::error("Model not found", "unknown-model");
        assert!(!result.success);
        assert_eq!(result.error, Some("Model not found".to_string()));
        assert_eq!(result.output_type, "none");
        assert!(result.text.is_none());
        assert!(result.embedding.is_none());
        assert!(result.audio_bytes_len.is_none());
        assert_eq!(result.latency_ms, 0);
        assert_eq!(result.model_id, "unknown-model");
        assert_eq!(result.stages_executed, 0);
        assert_eq!(result.stages_skipped, 0);
    }

    #[test]
    fn test_pipeline_execution_result_with_stages() {
        let mut result = FfiPipelineExecutionResult::success(
            "text",
            Some("output".to_string()),
            None,
            200,
            "pipeline-model",
        );

        result.with_stages(vec![
            FfiStageExecutionResult::executed("stage1", "device", 50),
            FfiStageExecutionResult::executed("stage2", "cloud", 100),
            FfiStageExecutionResult::skipped("stage3", "condition not met"),
        ]);

        assert_eq!(result.stages.len(), 3);
        assert_eq!(result.stages_executed, 2);
        assert_eq!(result.stages_skipped, 1);
    }

    #[test]
    fn test_pipeline_execution_result_default() {
        let result = FfiPipelineExecutionResult::default();
        assert!(!result.success);
        assert!(result.error.is_none());
        assert_eq!(result.output_type, "none");
        assert!(result.text.is_none());
        assert!(result.embedding.is_none());
        assert!(result.audio_bytes_len.is_none());
        assert_eq!(result.latency_ms, 0);
        assert!(result.model_id.is_empty());
        assert!(result.stages.is_empty());
        assert_eq!(result.stages_executed, 0);
        assert_eq!(result.stages_skipped, 0);
    }

    #[test]
    fn test_pipeline_execution_result_serialization() {
        let result = FfiPipelineExecutionResult::success(
            "text",
            Some("Hello".to_string()),
            None,
            100,
            "model-id",
        );
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"output_type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
        assert!(json.contains("\"model_id\":\"model-id\""));
        assert!(json.contains("\"latency_ms\":100"));

        let deserialized: FfiPipelineExecutionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);
    }

    #[test]
    fn test_pipeline_execution_result_error_serialization() {
        let result = FfiPipelineExecutionResult::error("Connection failed", "remote-model");
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"success\":false"));
        assert!(json.contains("\"error\":\"Connection failed\""));
        assert!(json.contains("\"output_type\":\"none\""));

        let deserialized: FfiPipelineExecutionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);
    }
}
