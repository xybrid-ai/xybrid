//! LLM (Large Language Model) abstraction module.
//!
//! This module provides a unified interface for LLM completions that abstracts
//! away the execution source (gateway, local model, or direct API).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Client Application                       │
//! │                                                             │
//! │   llm::complete(request) ──────────────────────────────────┤
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     LLM Router                              │
//! │  ┌──────────────┬──────────────┬──────────────────────────┐ │
//! │  │   Gateway    │    Local     │      Direct API          │ │
//! │  │  (default)   │   (ONNX)     │   (fallback/debug)       │ │
//! │  └──────────────┴──────────────┴──────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//!          │                │                    │
//!          ▼                ▼                    ▼
//!    Xybrid Gateway    Local Model         OpenAI/Anthropic
//!    (OpenAI-compat)    (on-device)          (direct)
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use xybrid_core::llm::{Llm, CompletionRequest};
//!
//! // Create client (routes through gateway by default)
//! let llm = Llm::new()?;
//!
//! // Simple completion
//! let response = llm.complete(CompletionRequest::new("Hello, world!"))?;
//! println!("{}", response.text);
//!
//! // With options
//! let response = llm.complete(
//!     CompletionRequest::new("Explain Rust in one sentence")
//!         .with_system("You are a helpful programming tutor.")
//!         .with_max_tokens(100)
//! )?;
//! ```

mod client;
mod completion;
mod config;
mod error;

pub use client::Llm;
pub use completion::{CompletionRequest, CompletionResponse, Message, Role, Usage};
pub use config::{LlmConfig, LlmBackend};
pub use error::LlmError;
