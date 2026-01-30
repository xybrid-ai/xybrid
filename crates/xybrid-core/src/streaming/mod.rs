//! Streaming inference module for real-time ASR.
//!
//! This module provides `StreamSession` for continuous audio streaming
//! with chunked transcription support for ASR models.
//!
//! # Architecture
//!
//! ```text
//! Audio Chunks → [AudioBuffer] → [TemplateExecutor] → [Transcript Aggregation] → Text
//!     ↑                                                         ↓
//!   feed()                                              partial results
//!     ↑                                                         ↓
//!  flush() ─────────────────────────────────────────────→ final text
//! ```
//!
//! # Unified Backend
//!
//! StreamSession uses `TemplateExecutor` internally, which automatically
//! selects the appropriate backend based on `model_metadata.json`:
//!
//! - `execution_template.type: "CandleModel"` → Candle (Whisper)
//! - `execution_template.type: "SimpleMode"` → ONNX (Wav2Vec2)
//!
//! # Example
//!
//! ```ignore
//! use xybrid_core::streaming::{StreamSession, StreamConfig};
//!
//! // Create session - backend auto-detected from model_metadata.json
//! let config = StreamConfig::default();
//! let mut session = StreamSession::new("/path/to/whisper-model", config)?;
//!
//! // Feed audio chunks as they arrive
//! session.feed(&audio_chunk_1)?;
//! session.feed(&audio_chunk_2)?;
//!
//! // Get partial result anytime
//! if let Some(partial) = session.partial_result() {
//!     println!("Partial: {}", partial.text);
//! }
//!
//! // Flush to get final transcription
//! let final_result = session.flush()?;
//! println!("Final: {}", final_result);
//! ```

mod audio_buffer;
mod session;

pub use audio_buffer::{AudioBuffer, AudioBufferConfig, AudioBufferStats, AudioChunk};
pub use session::{
    PartialResult, StreamConfig, StreamError, StreamResult, StreamSession, StreamState,
    StreamStats, VadStreamConfig,
};

pub mod manager;
pub use manager::{StreamManager, StreamManagerConfig};
