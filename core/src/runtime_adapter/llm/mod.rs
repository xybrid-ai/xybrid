//! LLM Runtime backend module.
//!
//! This module provides local LLM inference through the RuntimeAdapter interface.
//! It uses a pluggable backend architecture, with mistral.rs as the default.
//!
//! # Features
//!
//! - **Pure Rust**: Default backend (mistral.rs) is pure Rust
//! - **GGUF Support**: Load quantized models in GGUF format
//! - **Hardware Acceleration**: Metal (macOS/iOS), CUDA (Linux/Windows)
//! - **Pluggable Backend**: Swap out mistral.rs for llama-cpp-rs, etc.
//!
//! # Feature Flags
//!
//! - `local-llm`: Enable local LLM backend (CPU)
//! - `local-llm-metal`: Enable Metal acceleration (Apple Silicon)
//! - `local-llm-cuda`: Enable CUDA acceleration (NVIDIA GPUs)
//!
//! # Architecture
//!
//! ```text
//! LlmRuntimeAdapter (RuntimeAdapter impl)
//!     │
//!     └── LlmBackend (trait - swappable)
//!             │
//!             ├── MistralBackend (default)
//!             └── (future: LlamaCppBackend, etc.)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use xybrid_core::runtime_adapter::llm::LlmRuntimeAdapter;
//! use xybrid_core::runtime_adapter::RuntimeAdapter;
//! use xybrid_core::ir::{Envelope, EnvelopeKind};
//!
//! let mut adapter = LlmRuntimeAdapter::new()?;
//! adapter.load_model("path/to/qwen2.5-1.5b.gguf")?;
//!
//! let input = Envelope::new(EnvelopeKind::Text("Hello!".to_string()));
//! let output = adapter.execute(&input)?;
//!
//! if let EnvelopeKind::Text(response) = output.kind {
//!     println!("Response: {}", response);
//! }
//! ```

mod adapter;
mod backend;
mod config;
mod mistral;

pub use adapter::LlmRuntimeAdapter;
pub use backend::{ChatMessage, GenerationOutput, LlmBackend, LlmResult};
pub use config::{GenerationConfig, LlmConfig};
pub use mistral::MistralBackend;
