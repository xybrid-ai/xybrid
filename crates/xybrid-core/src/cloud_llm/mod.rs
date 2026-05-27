//! Cloud LLM API integration module.
//!
//! This module provides unified access to cloud LLM APIs (OpenAI, Anthropic)
//! through a consistent interface. It handles authentication, request formatting,
//! response parsing, and error handling for each provider.
//!
//! ## Supported Providers
//!
//! - **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo
//! - **Anthropic**: Claude 3.5, Claude 3 (Opus, Sonnet, Haiku)
//!
//! ## Example
//!
//! This module is crate-private (`pub(crate)`) — the snippet below is for
//! internal reference only and is not compiled as a doctest.
//!
//! ```text
//! use xybrid_core::cloud_llm::{LlmClient, LlmRequest};
//! use xybrid_core::pipeline::IntegrationProvider;
//!
//! let client = LlmClient::new(IntegrationProvider::OpenAI)?;
//! let response = client.complete(LlmRequest {
//!     prompt: "Hello, world!".to_string(),
//!     system: Some("You are a helpful assistant.".to_string()),
//!     max_tokens: Some(100),
//!     temperature: Some(0.7),
//!     ..Default::default()
//! })?;
//! println!("Response: {}", response.text);
//! ```

mod client;
mod error;
mod request;
mod response;

pub use client::LlmClient;
pub use error::LlmError;
pub use request::{LlmRequest, Message, Role};
pub use response::LlmResponse;
