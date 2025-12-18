//! Xybrid Gateway - OpenAI-Compatible Cloud Adapter
//!
//! The Gateway provides a unified, secure endpoint for cloud AI services.
//! It speaks OpenAI's API format and routes requests to the appropriate backend.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Xybrid Gateway                                   │
//! │                  https://api.xybrid.dev/v1                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
//! │  │   /chat/     │    │  /embeddings │    │   /audio/    │               │
//! │  │ completions  │    │              │    │ transcriptions│               │
//! │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
//! │         │                   │                   │                        │
//! │         ▼                   ▼                   ▼                        │
//! │  ┌─────────────────────────────────────────────────────────────┐        │
//! │  │                    Request Router                            │        │
//! │  │  • API Key Validation     • Rate Limiting                    │        │
//! │  │  • Usage Tracking         • Request Logging                  │        │
//! │  │  • Safety Filtering       • Cost Metering                    │        │
//! │  └─────────────────────────────────────────────────────────────┘        │
//! │                              │                                           │
//! │         ┌────────────────────┼────────────────────┐                     │
//! │         │                    │                    │                     │
//! │         ▼                    ▼                    ▼                     │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
//! │  │   OpenAI     │    │  Anthropic   │    │   Groq       │               │
//! │  │   Backend    │    │   Backend    │    │   Backend    │               │
//! │  └──────────────┘    └──────────────┘    └──────────────┘               │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Endpoints
//!
//! ### Chat Completions
//! `POST /v1/chat/completions`
//!
//! OpenAI-compatible chat completions endpoint. Supports:
//! - Standard messages format (system, user, assistant)
//! - Streaming responses (SSE)
//! - Function calling / Tools
//! - Multiple model backends
//!
//! ### Embeddings (Future)
//! `POST /v1/embeddings`
//!
//! ### Audio Transcriptions (Future)
//! `POST /v1/audio/transcriptions`
//!
//! ## Security
//!
//! The Gateway provides several security benefits:
//!
//! 1. **API Key Protection**: Users don't expose their provider API keys to clients
//! 2. **Usage Control**: Rate limiting and quota management
//! 3. **Content Safety**: Optional content filtering and moderation
//! 4. **Audit Logging**: Complete request/response logging for compliance
//! 5. **Cost Management**: Per-user cost tracking and limits
//!
//! ## Authentication
//!
//! Requests use Bearer token authentication:
//! ```text
//! Authorization: Bearer xybrid_sk_...
//! ```
//!
//! Users can either:
//! - Use Xybrid-managed API keys (we handle provider keys)
//! - Bring their own provider keys (stored securely, used per-request)

mod api;
mod config;
mod error;
mod models;

pub use api::{
    ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatChunkChoice, ChatDelta, ChatMessage, ErrorResponse, MessageRole, Usage,
};
pub use config::{GatewayConfig, ProviderCredentials, RateLimitConfig};
pub use error::GatewayError;
pub use models::{ModelInfo, ModelList, ModelListEntry, ModelProvider};
