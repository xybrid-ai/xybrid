//! Cloud client module for third-party API integrations.
//!
//! This module provides a unified interface for cloud API calls that abstracts
//! away the execution method (gateway or direct API).
//!
//! ## Architecture
//!
//! ```text
//! +-------------------------------------------------------------+
//! |                    Client Application                       |
//! |                                                             |
//! |   cloud::complete(request) ---------------------------------|
//! +-------------------------------------------------------------+
//!                              |
//!                              v
//! +-------------------------------------------------------------+
//! |                     Cloud Router                            |
//! |  +----------------------------+---------------------------+ |
//! |  |         Gateway            |         Direct            | |
//! |  |        (default)           |      (dev/testing)        | |
//! |  +----------------------------+---------------------------+ |
//! +-------------------------------------------------------------+
//!          |                                   |
//!          v                                   v
//!    Xybrid Gateway                     OpenAI/Anthropic
//!    (OpenAI-compat)                       (direct)
//! ```
//!
//! ## Usage
//!
//! ```no_run
//! # fn _example() -> Result<(), Box<dyn std::error::Error>> {
//! use xybrid_core::cloud::{Cloud, CompletionRequest};
//!
//! // Create client (routes through gateway by default)
//! let cloud = Cloud::new()?;
//!
//! // Simple completion
//! let response = cloud.complete(CompletionRequest::new("Hello, world!"))?;
//! println!("{}", response.text);
//!
//! // With options
//! let response = cloud.complete(
//!     CompletionRequest::new("Explain Rust in one sentence")
//!         .with_system("You are a helpful programming tutor.")
//!         .with_max_tokens(100)
//! )?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! ## Note
//!
//! For local/on-device inference, use `target: device` in your pipeline YAML,
//! which routes to [`crate::execution::TemplateExecutor`] instead.

mod client;
mod completion;
mod config;
mod error;

pub(crate) use client::parse_gateway_usage;
pub use client::Cloud;
pub use completion::{CompletionRequest, CompletionResponse, Message, Role, Usage};
pub use config::{
    has_xybrid_api_key, set_xybrid_api_key, xybrid_api_key, CloudBackend, CloudConfig,
};
pub use error::CloudError;
