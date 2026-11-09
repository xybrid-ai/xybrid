//! Streaming inference for xybrid-sdk.
//!
//! This module provides `XybridStream` for real-time streaming ASR inference.
//! It wraps the core `StreamSession` with a simpler API.

use crate::model::SdkError;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use xybrid_core::execution::TemplateExecutor;
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
    /// The most recent actual decoder execution. Cached reads keep this reference.
    pub decoder_observation: Option<crate::ObservationRef>,
}

impl From<CorePartialResult> for PartialResult {
    fn from(result: CorePartialResult) -> Self {
        Self {
            text: result.text,
            is_stable: result.is_stable,
            chunk_index: result.chunk_sequence,
            audio_duration_ms: result.audio_duration.as_millis() as u64,
            decoder_observation: None,
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
    /// The workflow covering this utterance.
    pub observation: Option<crate::ObservationRef>,
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
    observation: Arc<crate::observability::AsrObservation>,
}

/// Real-time streaming session for ASR.
///
/// Created from `XybridModel::stream()`. Provides a simple API for
/// feeding audio samples and getting transcription results.
///
/// # Example
///
/// ```no_run
/// # use xybrid_sdk::{XybridModel, StreamConfig};
/// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
/// # let model: XybridModel = unimplemented!();
/// # let audio_chunk_1: Vec<f32> = vec![];
/// # let audio_chunk_2: Vec<f32> = vec![];
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
/// # Ok(())
/// # }
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
        executor: Arc<Mutex<TemplateExecutor>>,
        options: Option<&crate::observability::ObservationOptions>,
        metadata: Option<&xybrid_core::execution::ModelMetadata>,
    ) -> Result<Self, SdkError> {
        let mut session = StreamSession::with_executor(model_dir, config, executor)
            .map_err(|e| SdkError::load_src("Failed to create stream session", e))?;

        let observation = crate::observability::AsrObservation::for_model(
            options,
            metadata.map(|metadata| (model_id, metadata)),
        );
        session.set_decoder_observer(observation.clone());
        Ok(Self {
            handle: Arc::new(RwLock::new(StreamHandle {
                session,
                model_id: model_id.to_string(),
                observation,
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

    /// Pay the model's cold-start cost now, before real audio arrives.
    ///
    /// The first inference of a session lazily loads weights and pays
    /// first-run costs; calling this right after creating the stream (e.g.
    /// on a background thread while the microphone spins up) moves that
    /// cost off the first visible partial. No-op once audio has been fed.
    ///
    /// # Errors
    ///
    /// Returns an inference error if the warm-up run fails; the session
    /// stays usable.
    pub fn warmup(&self) -> Result<(), SdkError> {
        let mut handle = self
            .handle
            .write()
            .map_err(|_| SdkError::inference("Failed to acquire stream lock"))?;

        let (_context, scope) = handle.observation.warmup_span();
        let result = handle
            .session
            .warmup()
            .map_err(|e| SdkError::inference_src("Warm-up failed", e));
        scope.end(&result);
        result
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
            .map_err(|_| SdkError::inference("Failed to acquire stream lock"))?;

        let _context = handle.observation.enter_feed();
        if let Err(e) = handle.session.feed(samples) {
            // The utterance stays open, but its workflow ends failed at flush:
            // this chunk's audio is gone.
            let error = SdkError::inference_src("Feed failed", e);
            handle.observation.record_feed_failure(&error);
            return Err(error);
        }

        // Get partial result if available
        Ok(handle.session.partial_result().map(|p| {
            let mut partial = PartialResult::from(p);
            partial.decoder_observation = handle.observation.decoder_reference();
            partial
        }))
    }

    /// Get the current partial result without feeding new samples.
    pub fn partial_result(&self) -> Option<PartialResult> {
        self.handle.read().ok().and_then(|h| {
            h.session.partial_result().map(|p| {
                let mut partial = PartialResult::from(p);
                partial.decoder_observation = h.observation.decoder_reference();
                partial
            })
        })
    }

    /// End the stream and get the final transcription.
    ///
    /// This processes any remaining audio and returns the complete transcript.
    /// After calling this, you must call `reset()` to reuse the stream.
    pub fn flush(&self) -> Result<TranscriptionResult, SdkError> {
        let mut handle = self
            .handle
            .write()
            .map_err(|_| SdkError::inference("Failed to acquire stream lock"))?;

        let _context = handle.observation.enter();
        let text = handle
            .session
            .flush()
            .map_err(|e| SdkError::inference_src("Flush failed", e));
        let observation = handle.observation.finish(&text);
        let text = text?;

        let stats = handle.session.stats();

        Ok(TranscriptionResult {
            text,
            duration_ms: stats.audio_duration.as_millis() as u64,
            chunks_processed: stats.chunks_processed,
            observation,
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
            .map_err(|_| SdkError::inference("Failed to acquire stream lock"))?;

        handle.session.reset();
        handle.observation.reset();
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

    /// A failed feed has already lost its chunk (the core extracts it before
    /// decoding), so the utterance's workflow ends failed even when the final
    /// flush succeeds. The stream keeps accepting audio, and `reset` starts
    /// the next utterance clean.
    #[test]
    fn a_lost_chunk_fails_the_utterance_even_when_flush_succeeds() {
        use crate::observability::test_support::{attr, memory_session};
        use crate::observability::{semconv, ObservationOptions};
        use xybrid_core::execution::ModelMetadata;

        let dir = tempfile::tempdir().unwrap();
        // No weights on disk: a decode fails after its chunk left the buffer.
        let metadata = ModelMetadata::onnx("asr", "1", "missing.onnx");
        std::fs::write(
            dir.path().join("model_metadata.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        let mut config = CoreStreamConfig::default();
        // Without overlap, one whole chunk leaves nothing for flush to decode.
        config.buffer_config.overlap_secs = 0.0;
        let (session, memory, _provider) = memory_session();
        let options = ObservationOptions::default().with_session(&session);
        let executor = Arc::new(Mutex::new(TemplateExecutor::with_base_path(
            dir.path().to_str().unwrap(),
        )));
        let stream = XybridStream::new(
            dir.path(),
            config,
            "asr",
            executor,
            Some(&options),
            Some(&metadata),
        )
        .unwrap();

        // One 5 s chunk at 16 kHz is extracted, and its decode fails.
        assert!(stream.feed(&[0.0; 80_000]).is_err());
        // The utterance stays open and the stream keeps accepting audio.
        assert!(matches!(stream.feed(&[]), Ok(None)));
        let lost = stream
            .flush()
            .expect("nothing is left to decode")
            .observation
            .expect("the utterance is observed");
        assert_eq!(
            attr(&memory.by_ref(&lost), semconv::attr::ERROR_TYPE).as_deref(),
            Some("inference_error"),
            "a lost chunk fails the utterance"
        );

        stream.reset().unwrap();
        assert!(matches!(stream.feed(&[]), Ok(None)));
        let clean = stream.flush().unwrap().observation.unwrap();
        assert_ne!(clean, lost);
        assert_eq!(
            attr(&memory.by_ref(&clean), semconv::attr::ERROR_TYPE),
            None,
            "reset forgets the discarded utterance's failure"
        );
    }
}
