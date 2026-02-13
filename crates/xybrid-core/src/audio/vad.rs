//! Voice Activity Detection (VAD) module.
//!
//! Provides voice activity detection using the Silero VAD model via ONNX runtime.
//! Used for detecting speech segments in audio streams.
//!
//! # Architecture
//!
//! Silero VAD is a stateful model that processes audio in frames:
//! - 16kHz: 512 samples per frame (32ms), 64 sample context
//! - 8kHz: 256 samples per frame (32ms), 32 sample context
//!
//! The model outputs a probability (0.0-1.0) indicating voice activity.
//!
//! # Example
//!
//! ```ignore
//! use xybrid_core::audio::vad::{VadSession, VadConfig};
//!
//! let config = VadConfig::default();
//! let mut vad = VadSession::new("/path/to/silero-vad", config)?;
//!
//! // Process audio frames
//! for chunk in audio_chunks {
//!     let prob = vad.process(&chunk)?;
//!     if prob > 0.5 {
//!         println!("Speech detected!");
//!     }
//! }
//! ```

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::path::Path;
use thiserror::Error;

/// VAD-specific errors.
#[derive(Error, Debug)]
pub enum VadError {
    #[error("Model load error: {0}")]
    ModelLoadError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("Invalid audio: {0}")]
    AudioError(String),
}

/// Result type for VAD operations.
pub type VadResult<T> = Result<T, VadError>;

/// Supported sample rates for Silero VAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VadSampleRate {
    /// 8kHz sample rate (256 samples/frame, 32 context)
    Rate8k,
    /// 16kHz sample rate (512 samples/frame, 64 context)
    #[default]
    Rate16k,
}

impl VadSampleRate {
    /// Get the numeric sample rate.
    pub fn as_hz(&self) -> i64 {
        match self {
            VadSampleRate::Rate8k => 8000,
            VadSampleRate::Rate16k => 16000,
        }
    }

    /// Get the frame size in samples.
    pub fn frame_size(&self) -> usize {
        match self {
            VadSampleRate::Rate8k => 256,
            VadSampleRate::Rate16k => 512,
        }
    }

    /// Get the context size in samples.
    pub fn context_size(&self) -> usize {
        match self {
            VadSampleRate::Rate8k => 32,
            VadSampleRate::Rate16k => 64,
        }
    }

    /// Get frame duration in milliseconds.
    pub fn frame_duration_ms(&self) -> f32 {
        (self.frame_size() as f32 / self.as_hz() as f32) * 1000.0
    }
}

/// Configuration for VAD session.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Sample rate (8kHz or 16kHz)
    pub sample_rate: VadSampleRate,

    /// Threshold for speech detection (0.0-1.0)
    /// Default: 0.5
    pub threshold: f32,

    /// Minimum speech duration in frames before triggering speech start
    /// Default: 1 (immediate)
    pub min_speech_frames: usize,

    /// Minimum silence duration in frames before triggering speech end
    /// Default: 8 (~256ms at 16kHz)
    pub min_silence_frames: usize,

    /// Apply padding to detected segments (frames before/after)
    /// Default: 2 (~64ms at 16kHz)
    pub padding_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: VadSampleRate::Rate16k,
            threshold: 0.5,
            min_speech_frames: 1,
            min_silence_frames: 8,
            padding_frames: 2,
        }
    }
}

impl VadConfig {
    /// Create config for real-time streaming (more responsive).
    pub fn streaming() -> Self {
        Self {
            sample_rate: VadSampleRate::Rate16k,
            threshold: 0.5,
            min_speech_frames: 1,
            min_silence_frames: 4, // ~128ms
            padding_frames: 1,
        }
    }

    /// Create config for batch processing (more accurate).
    pub fn batch() -> Self {
        Self {
            sample_rate: VadSampleRate::Rate16k,
            threshold: 0.5,
            min_speech_frames: 2,
            min_silence_frames: 16, // ~512ms
            padding_frames: 4,
        }
    }
}

/// Internal state for Silero VAD model.
struct VadState {
    /// Hidden state tensor [2, 1, 128]
    state: Vec<f32>,
    /// Context window (previous frame's tail)
    context: Vec<f32>,
}

impl VadState {
    fn new(context_size: usize) -> Self {
        Self {
            state: vec![0.0; 2 * 128], // [2, 1, 128] flattened
            context: vec![0.0; context_size],
        }
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
    }
}

/// Voice Activity Detection result for a single frame.
#[derive(Debug, Clone)]
pub struct VadFrame {
    /// Voice activity probability (0.0-1.0)
    pub probability: f32,
    /// Whether speech is detected (probability > threshold)
    pub is_speech: bool,
    /// Frame index
    pub frame_index: u64,
}

/// Speech segment detected by VAD.
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Start time in milliseconds
    pub start_ms: f32,
    /// End time in milliseconds
    pub end_ms: f32,
    /// Start frame index
    pub start_frame: u64,
    /// End frame index
    pub end_frame: u64,
    /// Average probability during segment
    pub avg_probability: f32,
}

/// VAD session state for tracking speech segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadSessionState {
    /// No speech detected
    Silence,
    /// Potential speech start (accumulating min_speech_frames)
    SpeechPending,
    /// Active speech
    Speech,
    /// Potential speech end (accumulating min_silence_frames)
    SilencePending,
}

/// Voice Activity Detection session.
///
/// Wraps the Silero VAD ONNX model and provides stateful inference
/// for detecting speech segments in audio streams.
pub struct VadSession {
    /// ONNX session
    session: Session,
    /// Configuration
    config: VadConfig,
    /// Model state
    state: VadState,
    /// Current session state
    session_state: VadSessionState,
    /// Frame counter
    frame_count: u64,
    /// Consecutive speech frames
    speech_frames: usize,
    /// Consecutive silence frames
    silence_frames: usize,
    /// Current segment start frame (if in speech)
    segment_start: Option<u64>,
    /// Probabilities for current segment
    segment_probs: Vec<f32>,
}

impl VadSession {
    /// Create a new VAD session from a model directory.
    ///
    /// The directory should contain `model.onnx` (Silero VAD model).
    pub fn new<P: AsRef<Path>>(model_dir: P, config: VadConfig) -> VadResult<Self> {
        let model_path = model_dir.as_ref().join("model.onnx");

        if !model_path.exists() {
            return Err(VadError::ModelLoadError(format!(
                "Model file not found: {:?}",
                model_path
            )));
        }

        let session = Session::builder()
            .map_err(|e| {
                VadError::ModelLoadError(format!("Failed to create session builder: {}", e))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                VadError::ModelLoadError(format!("Failed to set optimization level: {}", e))
            })?
            .with_intra_threads(1)
            .map_err(|e| VadError::ModelLoadError(format!("Failed to set threads: {}", e)))?
            .commit_from_file(&model_path)
            .map_err(|e| VadError::ModelLoadError(format!("Failed to load model: {}", e)))?;

        let state = VadState::new(config.sample_rate.context_size());

        Ok(Self {
            session,
            config,
            state,
            session_state: VadSessionState::Silence,
            frame_count: 0,
            speech_frames: 0,
            silence_frames: 0,
            segment_start: None,
            segment_probs: Vec::new(),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Get current session state.
    pub fn session_state(&self) -> VadSessionState {
        self.session_state
    }

    /// Get frame count processed.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Check if currently in speech.
    pub fn is_speech(&self) -> bool {
        matches!(
            self.session_state,
            VadSessionState::Speech | VadSessionState::SilencePending
        )
    }

    /// Reset the session state (keep model loaded).
    pub fn reset(&mut self) {
        self.state.reset();
        self.session_state = VadSessionState::Silence;
        self.frame_count = 0;
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.segment_start = None;
        self.segment_probs.clear();
    }

    /// Process a single frame of audio and return VAD result.
    ///
    /// Frame size must match the configured sample rate:
    /// - 16kHz: 512 samples
    /// - 8kHz: 256 samples
    pub fn process_frame(&mut self, samples: &[f32]) -> VadResult<VadFrame> {
        let frame_size = self.config.sample_rate.frame_size();
        let context_size = self.config.sample_rate.context_size();

        if samples.len() != frame_size {
            return Err(VadError::AudioError(format!(
                "Expected {} samples, got {}",
                frame_size,
                samples.len()
            )));
        }

        // Build input with context: [context | frame]
        let mut input = Vec::with_capacity(context_size + frame_size);
        input.extend_from_slice(&self.state.context);
        input.extend_from_slice(samples);

        // Update context for next frame
        self.state
            .context
            .copy_from_slice(&samples[frame_size - context_size..]);

        // Run inference
        let probability = self.run_inference(&input)?;

        // Update state machine
        let is_speech = probability > self.config.threshold;
        self.update_state_machine(is_speech, probability);

        let frame = VadFrame {
            probability,
            is_speech,
            frame_index: self.frame_count,
        };

        self.frame_count += 1;

        Ok(frame)
    }

    /// Process multiple frames and return speech segments.
    ///
    /// This is useful for batch processing entire audio files.
    pub fn process_audio(&mut self, samples: &[f32]) -> VadResult<Vec<SpeechSegment>> {
        let frame_size = self.config.sample_rate.frame_size();
        let mut segments = Vec::new();

        // Process all complete frames
        let num_frames = samples.len() / frame_size;
        for i in 0..num_frames {
            let start = i * frame_size;
            let end = start + frame_size;
            let frame = &samples[start..end];

            let _result = self.process_frame(frame)?;

            // Check for completed segments
            if let Some(segment) = self.check_segment_complete() {
                segments.push(segment);
            }
        }

        Ok(segments)
    }

    /// Flush any pending speech segment.
    ///
    /// Call this when audio stream ends to get the final segment.
    pub fn flush(&mut self) -> Option<SpeechSegment> {
        if self.is_speech() && self.segment_start.is_some() {
            let segment = self.create_segment(self.frame_count);
            self.segment_start = None;
            self.segment_probs.clear();
            self.session_state = VadSessionState::Silence;
            return Some(segment);
        }
        None
    }

    /// Run ONNX inference.
    fn run_inference(&mut self, input: &[f32]) -> VadResult<f32> {
        let frame_size = self.config.sample_rate.frame_size();
        let context_size = self.config.sample_rate.context_size();
        let input_size = frame_size + context_size;

        // Create input tensors
        let input_tensor = Value::from_array(
            ndarray::Array2::from_shape_vec((1, input_size), input.to_vec()).map_err(|e| {
                VadError::InferenceError(format!("Failed to create input array: {}", e))
            })?,
        )
        .map_err(|e| VadError::InferenceError(format!("Failed to create input tensor: {}", e)))?;

        let sr_tensor = Value::from_array(ndarray::Array::from_elem(
            (),
            self.config.sample_rate.as_hz(),
        ))
        .map_err(|e| VadError::InferenceError(format!("Failed to create sr tensor: {}", e)))?;

        let state_tensor = Value::from_array(
            ndarray::Array3::from_shape_vec((2, 1, 128), self.state.state.clone()).map_err(
                |e| VadError::InferenceError(format!("Failed to create state array: {}", e)),
            )?,
        )
        .map_err(|e| VadError::InferenceError(format!("Failed to create state tensor: {}", e)))?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input_tensor,
                "sr" => sr_tensor,
                "state" => state_tensor,
            ])
            .map_err(|e| VadError::InferenceError(format!("Inference failed: {}", e)))?;

        // Extract output probability
        let output = outputs
            .get("output")
            .ok_or_else(|| VadError::InferenceError("Missing 'output' in results".into()))?;
        let (_, output_data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::InferenceError(format!("Failed to extract output: {}", e)))?;
        let probability = output_data.first().copied().unwrap_or(0.0);

        // Update hidden state
        let state_out = outputs
            .get("stateN")
            .ok_or_else(|| VadError::InferenceError("Missing 'stateN' in results".into()))?;
        let (_, state_data) = state_out
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::InferenceError(format!("Failed to extract state: {}", e)))?;
        self.state.state = state_data.to_vec();

        Ok(probability)
    }

    /// Update the state machine based on frame result.
    fn update_state_machine(&mut self, is_speech: bool, probability: f32) {
        match self.session_state {
            VadSessionState::Silence => {
                if is_speech {
                    self.speech_frames = 1;
                    if self.config.min_speech_frames <= 1 {
                        self.session_state = VadSessionState::Speech;
                        self.segment_start = Some(
                            self.frame_count
                                .saturating_sub(self.config.padding_frames as u64),
                        );
                        self.segment_probs.push(probability);
                    } else {
                        self.session_state = VadSessionState::SpeechPending;
                    }
                }
            }
            VadSessionState::SpeechPending => {
                if is_speech {
                    self.speech_frames += 1;
                    if self.speech_frames >= self.config.min_speech_frames {
                        self.session_state = VadSessionState::Speech;
                        self.segment_start = Some(self.frame_count.saturating_sub(
                            (self.speech_frames + self.config.padding_frames) as u64,
                        ));
                        self.segment_probs.push(probability);
                    }
                } else {
                    // Reset - not enough consecutive speech
                    self.session_state = VadSessionState::Silence;
                    self.speech_frames = 0;
                }
            }
            VadSessionState::Speech => {
                self.segment_probs.push(probability);
                if !is_speech {
                    self.silence_frames = 1;
                    if self.config.min_silence_frames <= 1 {
                        self.session_state = VadSessionState::Silence;
                    } else {
                        self.session_state = VadSessionState::SilencePending;
                    }
                }
            }
            VadSessionState::SilencePending => {
                self.segment_probs.push(probability);
                if is_speech {
                    // Back to speech
                    self.session_state = VadSessionState::Speech;
                    self.silence_frames = 0;
                } else {
                    self.silence_frames += 1;
                    if self.silence_frames >= self.config.min_silence_frames {
                        // Segment complete
                        self.session_state = VadSessionState::Silence;
                        self.silence_frames = 0;
                    }
                }
            }
        }
    }

    /// Check if a segment just completed.
    fn check_segment_complete(&mut self) -> Option<SpeechSegment> {
        if self.session_state == VadSessionState::Silence
            && self.segment_start.is_some()
            && !self.segment_probs.is_empty()
        {
            let segment = self.create_segment(
                self.frame_count - self.config.min_silence_frames as u64
                    + self.config.padding_frames as u64,
            );
            self.segment_start = None;
            self.segment_probs.clear();
            return Some(segment);
        }
        None
    }

    /// Create a speech segment from current state.
    fn create_segment(&self, end_frame: u64) -> SpeechSegment {
        let start_frame = self.segment_start.unwrap_or(0);
        let frame_ms = self.config.sample_rate.frame_duration_ms();
        let avg_prob = if self.segment_probs.is_empty() {
            0.0
        } else {
            self.segment_probs.iter().sum::<f32>() / self.segment_probs.len() as f32
        };

        SpeechSegment {
            start_ms: start_frame as f32 * frame_ms,
            end_ms: end_frame as f32 * frame_ms,
            start_frame,
            end_frame,
            avg_probability: avg_prob,
        }
    }
}

/// Simple energy-based VAD (fallback when ONNX model not available).
///
/// Uses RMS energy threshold for basic voice activity detection.
pub struct SimpleVad {
    /// Energy threshold (0.0-1.0)
    threshold: f32,
    /// Smoothing factor for energy
    smoothing: f32,
    /// Current smoothed energy
    current_energy: f32,
}

impl SimpleVad {
    /// Create a new simple VAD with threshold.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            smoothing: 0.1,
            current_energy: 0.0,
        }
    }

    /// Process samples and return if speech is detected.
    pub fn is_speech(&mut self, samples: &[f32]) -> bool {
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        self.current_energy = self.current_energy * (1.0 - self.smoothing) + rms * self.smoothing;
        self.current_energy > self.threshold
    }

    /// Get current energy level.
    pub fn energy(&self) -> f32 {
        self.current_energy
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.current_energy = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_sample_rate() {
        assert_eq!(VadSampleRate::Rate16k.frame_size(), 512);
        assert_eq!(VadSampleRate::Rate16k.context_size(), 64);
        assert_eq!(VadSampleRate::Rate8k.frame_size(), 256);
        assert_eq!(VadSampleRate::Rate8k.context_size(), 32);
    }

    #[test]
    fn test_vad_config_defaults() {
        let config = VadConfig::default();
        assert_eq!(config.sample_rate, VadSampleRate::Rate16k);
        assert_eq!(config.threshold, 0.5);
    }

    #[test]
    fn test_simple_vad() {
        let mut vad = SimpleVad::new(0.01);

        // Silence
        let silence = vec![0.0f32; 512];
        assert!(!vad.is_speech(&silence));

        // Loud signal
        let loud = vec![0.5f32; 512];
        assert!(vad.is_speech(&loud));
    }

    #[test]
    fn test_vad_state() {
        let state = VadState::new(64);
        assert_eq!(state.state.len(), 2 * 128);
        assert_eq!(state.context.len(), 64);
    }
}
