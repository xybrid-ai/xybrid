//! StreamSession for real-time ASR inference.
//!
//! Provides a high-level API for streaming audio transcription.
//! Uses `TemplateExecutor` as the unified execution backend, supporting
//! any ASR model defined via `model_metadata.json`.

use super::audio_buffer::{AudioBuffer, AudioBufferConfig, AudioChunk};
use crate::audio::samples_to_wav;
use crate::audio::vad::{VadConfig, VadSession};
use crate::execution_template::{ExecutionTemplate, ModelMetadata};
use crate::ir::{Envelope, EnvelopeKind};
use crate::template_executor::TemplateExecutor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Error type for streaming operations.
#[derive(Debug)]
pub enum StreamError {
    /// Model loading failed
    ModelLoadError(String),
    /// Inference failed
    InferenceError(String),
    /// Invalid state for operation
    InvalidState(String),
    /// Configuration error
    ConfigError(String),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            StreamError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            StreamError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
            StreamError::ConfigError(msg) => write!(f, "Config error: {}", msg),
        }
    }
}

impl std::error::Error for StreamError {}

/// Result type for streaming operations.
pub type StreamResult<T> = Result<T, StreamError>;

/// VAD (Voice Activity Detection) configuration for smart chunking.
#[derive(Debug, Clone)]
pub struct VadStreamConfig {
    /// Enable VAD-based chunking (splits at speech boundaries)
    pub enabled: bool,
    /// Path to VAD model directory (required when VAD is enabled)
    pub model_dir: Option<String>,
    /// VAD threshold (0.0-1.0, default: 0.5)
    pub threshold: f32,
    /// Minimum silence duration (frames) before splitting (default: 8, ~256ms)
    pub min_silence_frames: usize,
    /// Padding frames to include before/after speech (default: 2, ~64ms)
    pub padding_frames: usize,
}

impl Default for VadStreamConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: None,
            threshold: 0.5,
            min_silence_frames: 8,
            padding_frames: 2,
        }
    }
}

impl VadStreamConfig {
    /// Create VAD config with custom model path.
    pub fn with_model(model_dir: impl Into<String>) -> Self {
        Self {
            enabled: true,
            model_dir: Some(model_dir.into()),
            ..Default::default()
        }
    }

    /// Create enabled VAD config with default model.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }
}

/// Configuration for streaming session.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Audio buffer configuration (can be auto-detected from model)
    pub buffer_config: AudioBufferConfig,
    /// Minimum audio duration before processing (seconds)
    pub min_chunk_secs: f32,
    /// Enable partial results during streaming
    pub enable_partial_results: bool,
    /// Language hint (passed to model if supported)
    pub language: Option<String>,
    /// VAD configuration for smart chunking
    pub vad: VadStreamConfig,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_config: AudioBufferConfig::default(),
            min_chunk_secs: 1.0, // At least 1 second before processing
            enable_partial_results: true,
            language: Some("en".to_string()),
            vad: VadStreamConfig::default(),
        }
    }
}

impl StreamConfig {
    /// Create config with VAD enabled.
    pub fn with_vad() -> Self {
        Self {
            vad: VadStreamConfig::enabled(),
            ..Default::default()
        }
    }

    /// Create config with VAD and custom model path.
    pub fn with_vad_model(model_dir: impl Into<String>) -> Self {
        Self {
            vad: VadStreamConfig::with_model(model_dir),
            ..Default::default()
        }
    }
}

/// Current state of the stream session.
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

/// Partial transcription result.
#[derive(Debug, Clone)]
pub struct PartialResult {
    /// Partial transcription text
    pub text: String,
    /// Confidence score (0.0 - 1.0, if available)
    pub confidence: Option<f32>,
    /// Whether this is a stable (final) segment
    pub is_stable: bool,
    /// Audio duration covered by this result
    pub audio_duration: Duration,
    /// Chunk sequence number
    pub chunk_sequence: u64,
}

/// Accumulated transcription across chunks.
#[derive(Debug, Default)]
struct TranscriptAccumulator {
    /// Segments from completed chunks
    segments: Vec<String>,
    /// Current partial (unstable) text
    current_partial: Option<String>,
    /// Total audio duration processed
    total_duration: Duration,
}

impl TranscriptAccumulator {
    fn new() -> Self {
        Self::default()
    }

    fn add_segment(&mut self, text: String, duration: Duration) {
        if !text.trim().is_empty() {
            self.segments.push(text.trim().to_string());
        }
        self.total_duration += duration;
        self.current_partial = None;
    }

    fn set_partial(&mut self, text: String) {
        self.current_partial = Some(text);
    }

    fn get_full_text(&self) -> String {
        let mut parts = self.segments.clone();
        if let Some(ref partial) = self.current_partial {
            if !partial.trim().is_empty() {
                parts.push(partial.trim().to_string());
            }
        }
        parts.join(" ")
    }

    fn get_stable_text(&self) -> String {
        self.segments.join(" ")
    }

    fn reset(&mut self) {
        self.segments.clear();
        self.current_partial = None;
        self.total_duration = Duration::ZERO;
    }
}

/// Streaming ASR session.
///
/// Manages continuous audio streaming with chunked transcription.
/// Uses `TemplateExecutor` for unified model execution, supporting
/// both ONNX (Wav2Vec2) and Candle (Whisper) backends via `model_metadata.json`.
///
/// # Example
///
/// ```ignore
/// use xybrid_core::streaming::{StreamSession, StreamConfig};
///
/// // Create session from model directory (backend auto-detected)
/// let config = StreamConfig::default();
/// let mut session = StreamSession::new("test_models/whisper-tiny-candle", config)?;
///
/// // Feed audio chunks
/// session.feed(&audio_samples)?;
///
/// // Get final transcript
/// let transcript = session.flush()?;
/// ```
pub struct StreamSession {
    /// Model directory path
    model_dir: PathBuf,
    /// Loaded model metadata
    metadata: ModelMetadata,
    /// Template executor for inference
    executor: TemplateExecutor,
    /// Configuration
    config: StreamConfig,
    /// Audio buffer
    buffer: AudioBuffer,
    /// Transcript accumulator
    transcript: TranscriptAccumulator,
    /// Current state
    state: StreamState,
    /// Last error message
    last_error: Option<String>,
    /// Callback for partial results
    on_partial: Option<Arc<dyn Fn(PartialResult) + Send + Sync>>,
    /// VAD session for smart chunking (optional)
    vad: Option<VadSession>,
    /// Accumulated samples for VAD-based segmentation
    vad_buffer: Vec<f32>,
    /// Current speech segment start (in samples) for VAD mode
    vad_speech_start: Option<usize>,
}

impl StreamSession {
    /// Create a new streaming session from a model directory.
    ///
    /// The model directory must contain `model_metadata.json` which defines
    /// the execution template (ONNX or Candle) and preprocessing steps.
    /// The backend is automatically determined from the metadata.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to model directory containing `model_metadata.json`
    /// * `config` - Stream configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Whisper model (Candle backend, auto-detected)
    /// let session = StreamSession::new("test_models/whisper-tiny-candle", config)?;
    ///
    /// // Wav2Vec2 model (ONNX backend, auto-detected)
    /// let session = StreamSession::new("test_models/wav2vec2-base-960h", config)?;
    /// ```
    pub fn new<P: AsRef<Path>>(model_dir: P, config: StreamConfig) -> StreamResult<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        // Validate model directory exists
        if !model_dir.exists() {
            return Err(StreamError::ConfigError(format!(
                "Model directory does not exist: {:?}",
                model_dir
            )));
        }

        // Load model metadata
        let metadata_path = model_dir.join("model_metadata.json");
        if !metadata_path.exists() {
            return Err(StreamError::ConfigError(format!(
                "model_metadata.json not found in {:?}",
                model_dir
            )));
        }

        let metadata_str = std::fs::read_to_string(&metadata_path)
            .map_err(|e| StreamError::ConfigError(format!("Failed to read metadata: {}", e)))?;

        let metadata: ModelMetadata = serde_json::from_str(&metadata_str)
            .map_err(|e| StreamError::ConfigError(format!("Failed to parse metadata: {}", e)))?;

        // Create executor with model directory as base path
        let executor =
            TemplateExecutor::with_base_path(model_dir.to_str().unwrap_or("."));

        // Infer optimal buffer config from model type
        let buffer_config = Self::infer_buffer_config(&metadata, &config);
        let buffer = AudioBuffer::with_config(buffer_config);

        // Initialize VAD if enabled
        let vad = if config.vad.enabled {
            let vad_model_dir = match &config.vad.model_dir {
                Some(dir) => dir.clone(),
                None => {
                    eprintln!("[StreamSession] Warning: VAD enabled but no model_dir specified. VAD disabled.");
                    return Ok(Self {
                        model_dir,
                        metadata,
                        executor,
                        config,
                        buffer,
                        transcript: TranscriptAccumulator::new(),
                        state: StreamState::Idle,
                        last_error: None,
                        on_partial: None,
                        vad: None,
                        vad_buffer: Vec::new(),
                        vad_speech_start: None,
                    });
                }
            };

            let vad_config = VadConfig {
                threshold: config.vad.threshold,
                min_silence_frames: config.vad.min_silence_frames,
                padding_frames: config.vad.padding_frames,
                ..VadConfig::default()
            };

            match VadSession::new(&vad_model_dir, vad_config) {
                Ok(vad) => Some(vad),
                Err(e) => {
                    eprintln!("[StreamSession] Warning: Failed to initialize VAD: {}. Falling back to fixed chunking.", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            model_dir,
            metadata,
            executor,
            config,
            buffer,
            transcript: TranscriptAccumulator::new(),
            state: StreamState::Idle,
            last_error: None,
            on_partial: None,
            vad,
            vad_buffer: Vec::new(),
            vad_speech_start: None,
        })
    }

    /// Infer optimal buffer configuration from model metadata.
    fn infer_buffer_config(metadata: &ModelMetadata, config: &StreamConfig) -> AudioBufferConfig {
        // Check if this is a Whisper model (Candle)
        let is_whisper = match &metadata.execution_template {
            ExecutionTemplate::CandleModel { model_type, .. } => {
                model_type.as_deref() == Some("whisper")
            }
            _ => false,
        };

        if is_whisper {
            // Whisper: Use 5s chunks for responsive streaming (max 30s supported)
            // This gives partial results every ~5 seconds while speaking
            AudioBufferConfig {
                sample_rate: 16000,
                chunk_duration_secs: 5.0,
                overlap_secs: 0.5, // Small overlap for continuity
                max_buffer_secs: config.buffer_config.max_buffer_secs,
            }
        } else {
            // Default/Wav2Vec2: shorter chunks
            AudioBufferConfig {
                sample_rate: 16000,
                chunk_duration_secs: 5.0,
                overlap_secs: config.buffer_config.overlap_secs,
                max_buffer_secs: config.buffer_config.max_buffer_secs,
            }
        }
    }

    /// Get the model ID from metadata.
    pub fn model_id(&self) -> &str {
        &self.metadata.model_id
    }

    /// Set callback for partial results.
    pub fn on_partial<F>(&mut self, callback: F)
    where
        F: Fn(PartialResult) + Send + Sync + 'static,
    {
        self.on_partial = Some(Arc::new(callback));
    }

    /// Feed audio samples into the stream.
    ///
    /// # Arguments
    ///
    /// * `samples` - PCM audio samples (f32, mono, 16kHz)
    pub fn feed(&mut self, samples: &[f32]) -> StreamResult<()> {
        // Transition to streaming state
        match self.state {
            StreamState::Idle => self.state = StreamState::Streaming,
            StreamState::Streaming => {}
            StreamState::Finalizing | StreamState::Completed => {
                return Err(StreamError::InvalidState(
                    "Cannot feed after stream ended".to_string(),
                ));
            }
            StreamState::Error => {
                return Err(StreamError::InvalidState(format!(
                    "Session in error state: {:?}",
                    self.last_error
                )));
            }
        }

        // Push samples to buffer
        self.buffer.push(samples);

        // Process chunks if ready and partial results enabled
        if self.config.enable_partial_results {
            self.process_ready_chunks()?;
        }

        Ok(())
    }

    /// Get current partial result without processing new chunks.
    pub fn partial_result(&self) -> Option<PartialResult> {
        if !self.config.enable_partial_results {
            return None;
        }

        let text = self.transcript.get_full_text();
        if text.is_empty() {
            return None;
        }

        Some(PartialResult {
            text,
            confidence: None,
            is_stable: false,
            audio_duration: self.transcript.total_duration,
            chunk_sequence: self.buffer.stats().chunks_extracted,
        })
    }

    /// End the stream and get final transcription.
    ///
    /// This processes any remaining audio and returns the complete transcript.
    pub fn flush(&mut self) -> StreamResult<String> {
        // Transition to finalizing
        match self.state {
            StreamState::Idle => {
                // Nothing to flush
                self.state = StreamState::Completed;
                return Ok(String::new());
            }
            StreamState::Streaming => {
                self.state = StreamState::Finalizing;
            }
            StreamState::Finalizing => {}
            StreamState::Completed => {
                return Ok(self.transcript.get_stable_text());
            }
            StreamState::Error => {
                return Err(StreamError::InvalidState(format!(
                    "Session in error state: {:?}",
                    self.last_error
                )));
            }
        }

        // Mark buffer as ended
        self.buffer.end_stream();

        // Process all remaining audio
        self.process_all_remaining()?;

        // Mark completed
        self.state = StreamState::Completed;

        Ok(self.transcript.get_stable_text())
    }

    /// Reset the session for reuse.
    pub fn reset(&mut self) {
        self.buffer.reset();
        self.transcript.reset();
        self.state = StreamState::Idle;
        self.last_error = None;
        // Reset VAD state
        if let Some(ref mut vad) = self.vad {
            vad.reset();
        }
        self.vad_buffer.clear();
        self.vad_speech_start = None;
    }

    /// Check if VAD is enabled and active.
    pub fn has_vad(&self) -> bool {
        self.vad.is_some()
    }

    /// Get current session state.
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Get buffer statistics.
    pub fn stats(&self) -> StreamStats {
        let buffer_stats = self.buffer.stats();
        StreamStats {
            state: self.state,
            samples_received: buffer_stats.total_received,
            samples_processed: buffer_stats.total_processed,
            chunks_processed: buffer_stats.chunks_extracted,
            transcript_length: self.transcript.get_full_text().len(),
            audio_duration: self.transcript.total_duration,
        }
    }

    /// Process ready chunks from buffer.
    fn process_ready_chunks(&mut self) -> StreamResult<()> {
        while self.buffer.has_chunk_ready() {
            if let Some(chunk) = self.buffer.extract_chunk(false) {
                self.process_chunk(chunk)?;
            }
        }
        Ok(())
    }

    /// Process all remaining audio (for flush).
    fn process_all_remaining(&mut self) -> StreamResult<()> {
        // First process any full chunks
        self.process_ready_chunks()?;

        // Process remaining audio in chunks (may need multiple iterations
        // if more than chunk_duration_secs of audio remains)
        while self.buffer.has_audio() {
            if let Some(chunk) = self.buffer.extract_chunk(true) {
                self.process_chunk(chunk)?;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Process a single audio chunk through TemplateExecutor.
    fn process_chunk(&mut self, chunk: AudioChunk) -> StreamResult<()> {
        // Skip very short chunks (less than min_chunk_secs)
        let min_samples =
            (self.config.min_chunk_secs * self.buffer.config().sample_rate as f32) as usize;
        if chunk.samples.len() < min_samples && !chunk.is_final {
            return Ok(());
        }

        // Convert samples to WAV bytes
        let wav_bytes = samples_to_wav(&chunk.samples, self.buffer.config().sample_rate);

        // Create envelope with audio data
        let envelope = Envelope::new(EnvelopeKind::Audio(wav_bytes));

        // Execute through TemplateExecutor (handles ONNX, Candle, etc.)
        let output = self
            .executor
            .execute(&self.metadata, &envelope)
            .map_err(|e| StreamError::InferenceError(format!("Execution failed: {}", e)))?;

        // Extract text from output
        let text = match output.kind {
            EnvelopeKind::Text(text) => text,
            _ => {
                return Err(StreamError::InferenceError(
                    "Model did not return text output".to_string(),
                ));
            }
        };

        // Update transcript
        self.transcript.add_segment(text.clone(), chunk.duration());

        // Fire callback if set
        if let Some(ref callback) = self.on_partial {
            let result = PartialResult {
                text,
                confidence: None,
                is_stable: chunk.is_final,
                audio_duration: chunk.duration(),
                chunk_sequence: chunk.sequence,
            };
            callback(result);
        }

        Ok(())
    }
}

/// Session statistics.
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Current state
    pub state: StreamState,
    /// Total samples received
    pub samples_received: u64,
    /// Total samples processed
    pub samples_processed: u64,
    /// Number of chunks processed
    pub chunks_processed: u64,
    /// Current transcript length
    pub transcript_length: usize,
    /// Total audio duration processed
    pub audio_duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_defaults() {
        let config = StreamConfig::default();
        assert_eq!(config.buffer_config.sample_rate, 16000);
        assert!(config.enable_partial_results);
    }

    #[test]
    fn test_transcript_accumulator() {
        let mut acc = TranscriptAccumulator::new();

        acc.add_segment("Hello".to_string(), Duration::from_secs(1));
        acc.add_segment("world".to_string(), Duration::from_secs(1));

        assert_eq!(acc.get_stable_text(), "Hello world");
        assert_eq!(acc.get_full_text(), "Hello world");

        acc.set_partial("testing".to_string());
        assert_eq!(acc.get_full_text(), "Hello world testing");
        assert_eq!(acc.get_stable_text(), "Hello world");
    }

    #[test]
    fn test_transcript_accumulator_reset() {
        let mut acc = TranscriptAccumulator::new();
        acc.add_segment("Hello".to_string(), Duration::from_secs(1));
        acc.reset();

        assert_eq!(acc.get_stable_text(), "");
        assert_eq!(acc.total_duration, Duration::ZERO);
    }
}
