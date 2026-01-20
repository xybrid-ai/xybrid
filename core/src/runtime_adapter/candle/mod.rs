//! Candle Runtime backend module.
//!
//! This module provides Candle-based inference for models using the
//! HuggingFace Candle framework. Candle is a pure Rust ML framework
//! with native support for Whisper, LLaMA, and other transformer models.
//!
//! # Features
//!
//! - **Pure Rust**: No external C/C++ dependencies (unlike ONNX Runtime)
//! - **Native Whisper**: Uses candle-transformers Whisper implementation
//! - **Hardware Acceleration**: Metal (macOS/iOS), CUDA (Linux/Windows)
//! - **SafeTensors**: Loads models in HuggingFace SafeTensors format
//!
//! # Feature Flags
//!
//! - `candle`: Enable Candle backend (CPU)
//! - `candle-metal`: Enable Metal acceleration (Apple Silicon)
//! - `candle-cuda`: Enable CUDA acceleration (NVIDIA GPUs)
//!
//! # Example
//!
//! ```rust,ignore
//! use xybrid_core::runtime_adapter::candle::CandleRuntimeAdapter;
//! use xybrid_core::runtime_adapter::RuntimeAdapter;
//!
//! let mut adapter = CandleRuntimeAdapter::new()?;
//! adapter.load_model("/path/to/whisper-tiny")?;
//! let output = adapter.execute(&audio_envelope)?;
//! ```

mod adapter;
mod backend;
mod device;
mod model;
mod runtime; // New runtime implementation
mod whisper;

pub use adapter::CandleRuntimeAdapter;
pub use backend::CandleBackend;
pub use device::{select_device, DeviceSelection};
pub use model::{load_candle_model, CandleModel, CandleModelType, ModelError, ModelResult};
pub use runtime::CandleRuntime;
pub use whisper::{Task, WhisperConfig, WhisperModel, WhisperSize};
