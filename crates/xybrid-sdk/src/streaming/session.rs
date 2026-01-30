//! FFI-safe streaming session types for platform bindings.
//!
//! These types are designed to be wrapped by platform-specific FFI layers
//! (flutter_rust_bridge for Flutter, UniFFI for Kotlin/Swift) without modification.
//!
//! # Design Notes
//!
//! - All types use simple, FFI-compatible field types (primitives, String, Option)
//! - No `#[frb]` or UniFFI attributes - those are added by the binding layer
//! - Serializable via serde for JSON transport across FFI boundaries
//! - Matches the Flutter `streaming.rs` types for easy migration

use serde::{Deserialize, Serialize};

/// Streaming session state (FFI-safe).
///
/// Represents the current state of a streaming ASR session.
/// Used for tracking session lifecycle across FFI boundaries.
///
/// # Example
///
/// ```
/// use xybrid_sdk::streaming::FfiStreamState;
///
/// let state = FfiStreamState::Idle;
/// assert_eq!(serde_json::to_string(&state).unwrap(), "\"Idle\"");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FfiStreamState {
    /// Session created but not started
    Idle,
    /// Actively receiving audio
    Streaming,
    /// Processing final audio
    Finalizing,
    /// Session completed
    Completed,
    /// Error occurred
    Error,
}

impl Default for FfiStreamState {
    fn default() -> Self {
        Self::Idle
    }
}

impl FfiStreamState {
    /// Check if the session is in an active state (can receive audio).
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Idle | Self::Streaming)
    }

    /// Check if the session has finished (completed or error).
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Completed | Self::Error)
    }

    /// Convert to a lowercase string representation for JSON APIs.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Streaming => "streaming",
            Self::Finalizing => "finalizing",
            Self::Completed => "completed",
            Self::Error => "error",
        }
    }
}

/// Partial transcription result (FFI-safe).
///
/// Represents an intermediate transcription result from streaming ASR.
/// These results may change as more audio is processed.
///
/// # Example
///
/// ```
/// use xybrid_sdk::streaming::FfiPartialResult;
///
/// let result = FfiPartialResult {
///     text: "Hello".to_string(),
///     is_stable: false,
///     chunk_index: 5,
/// };
/// assert!(!result.is_stable);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiPartialResult {
    /// Current partial transcription text
    pub text: String,
    /// Whether this result is stable (unlikely to change with more audio)
    pub is_stable: bool,
    /// Chunk sequence number (increments with each audio chunk)
    pub chunk_index: u64,
}

impl Default for FfiPartialResult {
    fn default() -> Self {
        Self {
            text: String::new(),
            is_stable: false,
            chunk_index: 0,
        }
    }
}

impl FfiPartialResult {
    /// Create a new partial result.
    pub fn new(text: impl Into<String>, is_stable: bool, chunk_index: u64) -> Self {
        Self {
            text: text.into(),
            is_stable,
            chunk_index,
        }
    }

    /// Create a stable (finalized) partial result.
    pub fn stable(text: impl Into<String>, chunk_index: u64) -> Self {
        Self {
            text: text.into(),
            is_stable: true,
            chunk_index,
        }
    }

    /// Create an unstable (intermediate) partial result.
    pub fn unstable(text: impl Into<String>, chunk_index: u64) -> Self {
        Self {
            text: text.into(),
            is_stable: false,
            chunk_index,
        }
    }
}

/// Stream session statistics (FFI-safe).
///
/// Provides statistics about a streaming session including samples processed,
/// audio duration, and current state.
///
/// # Example
///
/// ```
/// use xybrid_sdk::streaming::{FfiStreamState, FfiStreamStats};
///
/// let stats = FfiStreamStats::default();
/// assert_eq!(stats.state, FfiStreamState::Idle);
/// assert_eq!(stats.samples_received, 0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiStreamStats {
    /// Current state of the stream
    pub state: FfiStreamState,
    /// Number of audio samples received
    pub samples_received: u64,
    /// Number of audio samples processed
    pub samples_processed: u64,
    /// Number of audio chunks processed
    pub chunks_processed: u64,
    /// Total audio duration in milliseconds
    pub audio_duration_ms: u64,
}

impl Default for FfiStreamStats {
    fn default() -> Self {
        Self {
            state: FfiStreamState::Idle,
            samples_received: 0,
            samples_processed: 0,
            chunks_processed: 0,
            audio_duration_ms: 0,
        }
    }
}

/// Streaming session configuration (FFI-safe).
///
/// Configuration for creating a streaming ASR session.
/// Matches the Flutter `StreamingConfig` type for easy migration.
///
/// # Example
///
/// ```
/// use xybrid_sdk::streaming::FfiStreamingConfig;
///
/// // Create default config
/// let config = FfiStreamingConfig::default();
/// assert!(!config.enable_vad);
/// assert!((config.vad_threshold - 0.5).abs() < f32::EPSILON);
///
/// // Create config with VAD
/// let vad_config = FfiStreamingConfig::with_vad("/path/to/model", 0.6);
/// assert!(vad_config.enable_vad);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiStreamingConfig {
    /// Model directory path (containing model_metadata.json)
    pub model_dir: String,
    /// Enable VAD (Voice Activity Detection) for smart chunking
    pub enable_vad: bool,
    /// VAD threshold (0.0-1.0, default: 0.5)
    pub vad_threshold: f32,
    /// Language hint (e.g., "en")
    pub language: Option<String>,
}

impl Default for FfiStreamingConfig {
    fn default() -> Self {
        Self {
            model_dir: String::new(),
            enable_vad: false,
            vad_threshold: 0.5,
            language: Some("en".to_string()),
        }
    }
}

impl FfiStreamingConfig {
    /// Create a new streaming config with the given model directory.
    pub fn new(model_dir: impl Into<String>) -> Self {
        Self {
            model_dir: model_dir.into(),
            ..Default::default()
        }
    }

    /// Create a streaming config with VAD enabled.
    pub fn with_vad(model_dir: impl Into<String>, vad_threshold: f32) -> Self {
        Self {
            model_dir: model_dir.into(),
            enable_vad: true,
            vad_threshold,
            language: Some("en".to_string()),
        }
    }

    /// Set the language hint.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set VAD parameters.
    pub fn with_vad_settings(mut self, enable: bool, threshold: f32) -> Self {
        self.enable_vad = enable;
        self.vad_threshold = threshold;
        self
    }
}

impl FfiStreamStats {
    /// Create new stats in the given state.
    pub fn new(state: FfiStreamState) -> Self {
        Self {
            state,
            ..Default::default()
        }
    }

    /// Create stats with audio metrics.
    pub fn with_audio(
        state: FfiStreamState,
        samples_received: u64,
        samples_processed: u64,
        chunks_processed: u64,
        audio_duration_ms: u64,
    ) -> Self {
        Self {
            state,
            samples_received,
            samples_processed,
            chunks_processed,
            audio_duration_ms,
        }
    }

    /// Calculate the processing ratio (processed / received).
    /// Returns 1.0 if no samples received.
    pub fn processing_ratio(&self) -> f64 {
        if self.samples_received == 0 {
            1.0
        } else {
            self.samples_processed as f64 / self.samples_received as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_state_default() {
        let state = FfiStreamState::default();
        assert_eq!(state, FfiStreamState::Idle);
    }

    #[test]
    fn test_stream_state_is_active() {
        assert!(FfiStreamState::Idle.is_active());
        assert!(FfiStreamState::Streaming.is_active());
        assert!(!FfiStreamState::Finalizing.is_active());
        assert!(!FfiStreamState::Completed.is_active());
        assert!(!FfiStreamState::Error.is_active());
    }

    #[test]
    fn test_stream_state_is_finished() {
        assert!(!FfiStreamState::Idle.is_finished());
        assert!(!FfiStreamState::Streaming.is_finished());
        assert!(!FfiStreamState::Finalizing.is_finished());
        assert!(FfiStreamState::Completed.is_finished());
        assert!(FfiStreamState::Error.is_finished());
    }

    #[test]
    fn test_stream_state_as_str() {
        assert_eq!(FfiStreamState::Idle.as_str(), "idle");
        assert_eq!(FfiStreamState::Streaming.as_str(), "streaming");
        assert_eq!(FfiStreamState::Finalizing.as_str(), "finalizing");
        assert_eq!(FfiStreamState::Completed.as_str(), "completed");
        assert_eq!(FfiStreamState::Error.as_str(), "error");
    }

    #[test]
    fn test_stream_state_serialization() {
        let state = FfiStreamState::Streaming;
        let json = serde_json::to_string(&state).unwrap();
        assert_eq!(json, "\"Streaming\"");

        let deserialized: FfiStreamState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, state);
    }

    #[test]
    fn test_partial_result_default() {
        let result = FfiPartialResult::default();
        assert!(result.text.is_empty());
        assert!(!result.is_stable);
        assert_eq!(result.chunk_index, 0);
    }

    #[test]
    fn test_partial_result_new() {
        let result = FfiPartialResult::new("hello", true, 5);
        assert_eq!(result.text, "hello");
        assert!(result.is_stable);
        assert_eq!(result.chunk_index, 5);
    }

    #[test]
    fn test_partial_result_stable() {
        let result = FfiPartialResult::stable("world", 10);
        assert_eq!(result.text, "world");
        assert!(result.is_stable);
        assert_eq!(result.chunk_index, 10);
    }

    #[test]
    fn test_partial_result_unstable() {
        let result = FfiPartialResult::unstable("foo", 3);
        assert_eq!(result.text, "foo");
        assert!(!result.is_stable);
        assert_eq!(result.chunk_index, 3);
    }

    #[test]
    fn test_partial_result_serialization() {
        let result = FfiPartialResult::new("test", true, 7);
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"text\":\"test\""));
        assert!(json.contains("\"is_stable\":true"));
        assert!(json.contains("\"chunk_index\":7"));

        let deserialized: FfiPartialResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.text, result.text);
        assert_eq!(deserialized.is_stable, result.is_stable);
        assert_eq!(deserialized.chunk_index, result.chunk_index);
    }

    #[test]
    fn test_stream_stats_default() {
        let stats = FfiStreamStats::default();
        assert_eq!(stats.state, FfiStreamState::Idle);
        assert_eq!(stats.samples_received, 0);
        assert_eq!(stats.samples_processed, 0);
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.audio_duration_ms, 0);
    }

    #[test]
    fn test_stream_stats_new() {
        let stats = FfiStreamStats::new(FfiStreamState::Streaming);
        assert_eq!(stats.state, FfiStreamState::Streaming);
        assert_eq!(stats.samples_received, 0);
    }

    #[test]
    fn test_stream_stats_with_audio() {
        let stats = FfiStreamStats::with_audio(
            FfiStreamState::Streaming,
            16000,
            15000,
            5,
            1000,
        );
        assert_eq!(stats.state, FfiStreamState::Streaming);
        assert_eq!(stats.samples_received, 16000);
        assert_eq!(stats.samples_processed, 15000);
        assert_eq!(stats.chunks_processed, 5);
        assert_eq!(stats.audio_duration_ms, 1000);
    }

    #[test]
    fn test_stream_stats_processing_ratio() {
        // No samples - should return 1.0
        let stats = FfiStreamStats::default();
        assert!((stats.processing_ratio() - 1.0).abs() < f64::EPSILON);

        // With samples
        let stats = FfiStreamStats::with_audio(
            FfiStreamState::Streaming,
            100,
            80,
            1,
            10,
        );
        assert!((stats.processing_ratio() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stream_stats_serialization() {
        let stats = FfiStreamStats::with_audio(
            FfiStreamState::Completed,
            1000,
            1000,
            10,
            5000,
        );
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"state\":\"Completed\""));
        assert!(json.contains("\"samples_received\":1000"));
        assert!(json.contains("\"audio_duration_ms\":5000"));

        let deserialized: FfiStreamStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.state, stats.state);
        assert_eq!(deserialized.samples_received, stats.samples_received);
        assert_eq!(deserialized.audio_duration_ms, stats.audio_duration_ms);
    }

    // ========================================================================
    // FfiStreamingConfig tests (US-011)
    // ========================================================================

    #[test]
    fn test_streaming_config_default() {
        let config = FfiStreamingConfig::default();
        assert!(config.model_dir.is_empty());
        assert!(!config.enable_vad);
        assert!((config.vad_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.language, Some("en".to_string()));
    }

    #[test]
    fn test_streaming_config_new() {
        let config = FfiStreamingConfig::new("/path/to/model");
        assert_eq!(config.model_dir, "/path/to/model");
        assert!(!config.enable_vad);
        assert!((config.vad_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.language, Some("en".to_string()));
    }

    #[test]
    fn test_streaming_config_with_vad() {
        let config = FfiStreamingConfig::with_vad("/path/to/model", 0.7);
        assert_eq!(config.model_dir, "/path/to/model");
        assert!(config.enable_vad);
        assert!((config.vad_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.language, Some("en".to_string()));
    }

    #[test]
    fn test_streaming_config_with_language() {
        let config = FfiStreamingConfig::new("/path/to/model")
            .with_language("de");
        assert_eq!(config.language, Some("de".to_string()));
    }

    #[test]
    fn test_streaming_config_with_vad_settings() {
        let config = FfiStreamingConfig::new("/path/to/model")
            .with_vad_settings(true, 0.8);
        assert!(config.enable_vad);
        assert!((config.vad_threshold - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_streaming_config_serialization() {
        let config = FfiStreamingConfig {
            model_dir: "/test/model".to_string(),
            enable_vad: true,
            vad_threshold: 0.6,
            language: Some("fr".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"model_dir\":\"/test/model\""));
        assert!(json.contains("\"enable_vad\":true"));
        assert!(json.contains("\"vad_threshold\":0.6"));
        assert!(json.contains("\"language\":\"fr\""));

        let deserialized: FfiStreamingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_dir, config.model_dir);
        assert_eq!(deserialized.enable_vad, config.enable_vad);
        assert!((deserialized.vad_threshold - config.vad_threshold).abs() < f32::EPSILON);
        assert_eq!(deserialized.language, config.language);
    }
}
