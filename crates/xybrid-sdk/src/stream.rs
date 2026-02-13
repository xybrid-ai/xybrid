//! Streaming inference for xybrid-sdk.
//!
//! This module provides `XybridStream` for real-time streaming ASR inference.
//! It wraps the core `StreamSession` with a simpler API.

use crate::model::SdkError;
use std::path::Path;
use std::sync::{Arc, RwLock};
use xybrid_core::streaming::{
    PartialResult as CorePartialResult, StreamConfig as CoreStreamConfig, StreamSession,
    StreamState as CoreStreamState, StreamStats as CoreStreamStats,
};

/// Current state of a streaming session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
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

impl From<CoreStreamState> for StreamState {
    fn from(state: CoreStreamState) -> Self {
        match state {
            CoreStreamState::Idle => StreamState::Idle,
            CoreStreamState::Streaming => StreamState::Streaming,
            CoreStreamState::Finalizing => StreamState::Finalizing,
            CoreStreamState::Completed => StreamState::Completed,
            CoreStreamState::Error => StreamState::Error,
        }
    }
}

/// Partial transcription result from streaming.
#[derive(Debug, Clone)]
pub struct PartialResult {
    /// Partial transcription text
    pub text: String,
    /// Whether this segment is stable (finalized)
    pub is_stable: bool,
    /// Chunk sequence number
    pub chunk_index: u64,
    /// Audio duration covered by this result
    pub audio_duration_ms: u64,
}

impl From<CorePartialResult> for PartialResult {
    fn from(result: CorePartialResult) -> Self {
        Self {
            text: result.text,
            is_stable: result.is_stable,
            chunk_index: result.chunk_sequence,
            audio_duration_ms: result.audio_duration.as_millis() as u64,
        }
    }
}

/// Final transcription result from streaming.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Final transcription text
    pub text: String,
    /// Total audio duration in milliseconds
    pub duration_ms: u64,
    /// Number of chunks processed
    pub chunks_processed: u64,
}

/// Statistics for a streaming session.
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Current session state
    pub state: StreamState,
    /// Total samples received
    pub samples_received: u64,
    /// Total samples processed
    pub samples_processed: u64,
    /// Number of chunks processed
    pub chunks_processed: u64,
    /// Current transcript length
    pub transcript_length: usize,
    /// Total audio duration processed (milliseconds)
    pub audio_duration_ms: u64,
}

impl From<CoreStreamStats> for StreamStats {
    fn from(stats: CoreStreamStats) -> Self {
        Self {
            state: stats.state.into(),
            samples_received: stats.samples_received,
            samples_processed: stats.samples_processed,
            chunks_processed: stats.chunks_processed,
            transcript_length: stats.transcript_length,
            audio_duration_ms: stats.audio_duration.as_millis() as u64,
        }
    }
}

/// Internal handle for the stream session.
struct StreamHandle {
    session: StreamSession,
    model_id: String,
}

/// Real-time streaming session for ASR.
///
/// Created from `XybridModel::stream()`. Provides a simple API for
/// feeding audio samples and getting transcription results.
///
/// # Example
///
/// ```ignore
/// let stream = model.stream(StreamConfig::with_vad())?;
///
/// // Feed audio samples (16kHz mono f32)
/// stream.feed(&audio_chunk_1)?;
/// stream.feed(&audio_chunk_2)?;
///
/// // Get partial result
/// if let Some(partial) = stream.partial_result() {
///     println!("Partial: {} (stable: {})", partial.text, partial.is_stable);
/// }
///
/// // End stream and get final transcript
/// let result = stream.flush()?;
/// println!("Final: {}", result.text);
///
/// // Reset for new utterance
/// stream.reset()?;
/// ```
pub struct XybridStream {
    handle: Arc<RwLock<StreamHandle>>,
}

impl XybridStream {
    /// Create a new streaming session.
    ///
    /// This is typically called by `XybridModel::stream()`, not directly.
    pub(crate) fn new<P: AsRef<Path>>(
        model_dir: P,
        config: CoreStreamConfig,
        model_id: &str,
    ) -> Result<Self, SdkError> {
        let session = StreamSession::new(model_dir, config)
            .map_err(|e| SdkError::LoadError(format!("Failed to create stream session: {}", e)))?;

        Ok(Self {
            handle: Arc::new(RwLock::new(StreamHandle {
                session,
                model_id: model_id.to_string(),
            })),
        })
    }

    /// Get the current session state.
    pub fn state(&self) -> StreamState {
        self.handle
            .read()
            .map(|h| h.session.state().into())
            .unwrap_or(StreamState::Error)
    }

    /// Get session statistics.
    pub fn stats(&self) -> StreamStats {
        self.handle
            .read()
            .map(|h| h.session.stats().into())
            .unwrap_or_else(|_| StreamStats {
                state: StreamState::Error,
                samples_received: 0,
                samples_processed: 0,
                chunks_processed: 0,
                transcript_length: 0,
                audio_duration_ms: 0,
            })
    }

    /// Check if VAD is enabled and active.
    pub fn has_vad(&self) -> bool {
        self.handle
            .read()
            .map(|h| h.session.has_vad())
            .unwrap_or(false)
    }

    /// Get the model ID this stream is using.
    pub fn model_id(&self) -> String {
        self.handle
            .read()
            .map(|h| h.model_id.clone())
            .unwrap_or_default()
    }

    /// Feed audio samples to the stream.
    ///
    /// # Arguments
    ///
    /// * `samples` - PCM audio samples (16kHz mono float32, -1.0 to 1.0)
    ///
    /// # Returns
    ///
    /// Optional partial result if transcription is ready.
    pub fn feed(&self, samples: &[f32]) -> Result<Option<PartialResult>, SdkError> {
        let mut handle = self
            .handle
            .write()
            .map_err(|_| SdkError::InferenceError("Failed to acquire stream lock".to_string()))?;

        handle
            .session
            .feed(samples)
            .map_err(|e| SdkError::InferenceError(format!("Feed failed: {}", e)))?;

        // Get partial result if available
        Ok(handle.session.partial_result().map(|p| p.into()))
    }

    /// Get the current partial result without feeding new samples.
    pub fn partial_result(&self) -> Option<PartialResult> {
        self.handle
            .read()
            .ok()
            .and_then(|h| h.session.partial_result().map(|p| p.into()))
    }

    /// End the stream and get the final transcription.
    ///
    /// This processes any remaining audio and returns the complete transcript.
    /// After calling this, you must call `reset()` to reuse the stream.
    pub fn flush(&self) -> Result<TranscriptionResult, SdkError> {
        let mut handle = self
            .handle
            .write()
            .map_err(|_| SdkError::InferenceError("Failed to acquire stream lock".to_string()))?;

        let text = handle
            .session
            .flush()
            .map_err(|e| SdkError::InferenceError(format!("Flush failed: {}", e)))?;

        let stats = handle.session.stats();

        Ok(TranscriptionResult {
            text,
            duration_ms: stats.audio_duration.as_millis() as u64,
            chunks_processed: stats.chunks_processed,
        })
    }

    /// Reset the stream for a new utterance.
    ///
    /// This clears the audio buffer and transcript accumulator while
    /// keeping the loaded model. Use this to start a new transcription
    /// session without reloading the model.
    pub fn reset(&self) -> Result<(), SdkError> {
        let mut handle = self
            .handle
            .write()
            .map_err(|_| SdkError::InferenceError("Failed to acquire stream lock".to_string()))?;

        handle.session.reset();
        Ok(())
    }
}

// Make XybridStream cloneable (shares the handle)
impl Clone for XybridStream {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_stream_state_conversion() {
        assert_eq!(StreamState::from(CoreStreamState::Idle), StreamState::Idle);
        assert_eq!(
            StreamState::from(CoreStreamState::Streaming),
            StreamState::Streaming
        );
        assert_eq!(
            StreamState::from(CoreStreamState::Completed),
            StreamState::Completed
        );
    }

    #[test]
    fn test_partial_result_conversion() {
        let core = CorePartialResult {
            text: "hello".to_string(),
            confidence: Some(0.9),
            is_stable: true,
            audio_duration: Duration::from_millis(1500),
            chunk_sequence: 5,
        };

        let result: PartialResult = core.into();
        assert_eq!(result.text, "hello");
        assert!(result.is_stable);
        assert_eq!(result.chunk_index, 5);
        assert_eq!(result.audio_duration_ms, 1500);
    }
}
