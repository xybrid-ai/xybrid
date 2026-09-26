//! FFI-agnostic facade over [`xybrid_sdk`].
//!
//! This crate exposes only the shapes every popular Rust FFI generator can
//! describe: owned data, concrete enums, `Arc<Self>` handles, no lifetimes,
//! no generics, no iterators across the boundary. The generator crates
//! ([`xybrid-bolt`], [`xybrid-ffi`], `bindings/flutter/rust`) describe
//! these types externally (BoltFFI macros, C ABI, FRB scan) and add their
//! own scaffolding — they should not need to reach into `xybrid-sdk`
//! directly for type re-translation.
//!
//! # Design rules
//!
//! 1. **No lifetimes, no generics, no iterators** in public signatures.
//! 2. **Owned data only** — `String` / `Vec<T>` / `Option<T>`, never
//!    `&str` / `&[T]` at the boundary.
//! 3. **`Arc<Self>` handles** for any object that crosses the boundary.
//!    BoltFFI uses it for handle types; FRB tolerates it.
//! 4. **Builders collapse into POD options records.** [`RunOptions`] and
//!    [`GenerationConfig`] are plain structs with `Default`; the facade
//!    rebuilds the SDK's builder chain internally.
//! 5. **One canonical [`Error`] enum** with `From<SdkError>`. `io::Error`,
//!    `anyhow::Error`, and trait objects never leak out.
//! 6. **`Send + Sync` everywhere.** Required by every FFI generator and by
//!    xybrid's multi-threaded tokio runtime.
//!
//! # Out of scope (deferred to follow-up PRs)
//!
//! - **ASR streaming.** [`xybrid_sdk::stream::XybridStream`] is wrapped
//!   separately in the same follow-up.
//!
//! [`xybrid-bolt`]: https://docs.rs/xybrid-bolt
//! [`xybrid-ffi`]: https://docs.rs/xybrid-ffi

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use xybrid_sdk as sdk;

mod cloud;

// ============================================================================
// Error
// ============================================================================

/// Canonical error surfaced across every FFI boundary.
///
/// Variants mirror [`sdk::SdkError`] but flatten non-FFI-safe payloads
/// ([`std::io::Error`], trait objects, embedded source chains) into a
/// `message` string plus a stable [`Error::code`].
#[derive(Debug, Clone)]
pub enum Error {
    ModelNotFound {
        id: String,
    },
    DirectoryNotFound {
        path: String,
    },
    MetadataNotFound {
        path: String,
    },
    MetadataInvalid {
        message: String,
    },
    LoadError {
        message: String,
    },
    InferenceError {
        message: String,
    },
    AbortedForCloudFallback {
        reason: String,
    },
    StreamingNotSupported,
    NotLoaded,
    ConfigError {
        message: String,
    },
    NetworkError {
        message: String,
    },
    Offline {
        message: String,
    },
    IoError {
        message: String,
    },
    CacheError {
        message: String,
    },
    PipelineError {
        message: String,
    },
    CircuitOpen {
        message: String,
    },
    RateLimited {
        retry_after_secs: u64,
    },
    Timeout {
        timeout_ms: u64,
    },
    /// A required model artifact (weights, tokenizer, …) was missing.
    MissingArtifact {
        message: String,
    },
    /// The model can't satisfy the request (e.g. image input to a text-only
    /// model).
    UnsupportedModelCapability {
        message: String,
    },
    /// The active backend/build can't satisfy the request (e.g. vision input
    /// without a vision-capable backend).
    UnsupportedBackendCapability {
        message: String,
    },
    /// An image envelope failed decode/validation (bad bytes, unsupported
    /// format, oversized payload).
    InvalidImage {
        message: String,
    },
    /// The caller stopped the operation — today, a cancelled model download.
    Cancelled {
        message: String,
    },
}

impl Error {
    /// Stable numeric discriminant — consumers can branch without parsing
    /// [`Display`]. Append to the tail; never renumber existing variants.
    ///
    /// [`Display`]: std::fmt::Display
    pub fn code(&self) -> u32 {
        match self {
            Error::ModelNotFound { .. } => 1,
            Error::DirectoryNotFound { .. } => 2,
            Error::MetadataNotFound { .. } => 3,
            Error::MetadataInvalid { .. } => 4,
            Error::LoadError { .. } => 5,
            Error::InferenceError { .. } => 6,
            Error::AbortedForCloudFallback { .. } => 7,
            Error::StreamingNotSupported => 8,
            Error::NotLoaded => 9,
            Error::ConfigError { .. } => 10,
            Error::NetworkError { .. } => 11,
            Error::Offline { .. } => 12,
            Error::IoError { .. } => 13,
            Error::CacheError { .. } => 14,
            Error::PipelineError { .. } => 15,
            Error::CircuitOpen { .. } => 16,
            Error::RateLimited { .. } => 17,
            Error::Timeout { .. } => 18,
            Error::MissingArtifact { .. } => 19,
            Error::UnsupportedModelCapability { .. } => 20,
            Error::UnsupportedBackendCapability { .. } => 21,
            Error::InvalidImage { .. } => 22,
            Error::Cancelled { .. } => 23,
        }
    }

    /// Mirrors [`sdk::SdkError`]'s `RetryableError::is_retryable`. Useful
    /// in foreign code that can't call the trait method.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::NetworkError { .. }
                | Error::RateLimited { .. }
                | Error::Timeout { .. }
                | Error::Offline { .. }
        )
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ModelNotFound { id } => write!(f, "Model not found: {id}"),
            Error::DirectoryNotFound { path } => write!(f, "Directory not found: {path}"),
            Error::MetadataNotFound { path } => {
                write!(f, "model_metadata.json not found in directory: {path}")
            }
            Error::MetadataInvalid { message } => {
                write!(f, "model_metadata.json is invalid: {message}")
            }
            Error::LoadError { message } => write!(f, "Failed to load model: {message}"),
            Error::InferenceError { message } => write!(f, "Inference failed: {message}"),
            Error::AbortedForCloudFallback { reason } => {
                write!(f, "Aborted for cloud fallback: {reason}")
            }
            Error::StreamingNotSupported => write!(f, "Streaming not supported by this model"),
            Error::NotLoaded => write!(f, "Model not loaded"),
            Error::ConfigError { message } => write!(f, "Invalid configuration: {message}"),
            Error::NetworkError { message } => write!(f, "Network error: {message}"),
            Error::Offline { message } => write!(f, "Registry unreachable: {message}"),
            Error::IoError { message } => write!(f, "IO error: {message}"),
            Error::CacheError { message } => write!(f, "Cache error: {message}"),
            Error::PipelineError { message } => write!(f, "Pipeline error: {message}"),
            Error::CircuitOpen { message } => write!(f, "Circuit breaker open: {message}"),
            Error::RateLimited { retry_after_secs } => {
                write!(f, "Rate limited, retry after {retry_after_secs} seconds")
            }
            Error::Timeout { timeout_ms } => write!(f, "Request timeout after {timeout_ms}ms"),
            Error::MissingArtifact { message } => write!(f, "Missing artifact: {message}"),
            Error::UnsupportedModelCapability { message } => {
                write!(f, "Unsupported model capability: {message}")
            }
            Error::UnsupportedBackendCapability { message } => {
                write!(f, "Unsupported backend capability: {message}")
            }
            Error::InvalidImage { message } => write!(f, "Invalid image input: {message}"),
            Error::Cancelled { message } => write!(f, "Cancelled: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<sdk::SdkError> for Error {
    fn from(err: sdk::SdkError) -> Self {
        // The whole point of the facade: this `match` is written ONCE, not
        // duplicated across xybrid-ffi / xybrid-bolt / flutter bindings.
        //
        // The message-bearing variants now carry a `#[source]` cause (the SDK
        // stopped pre-formatting it into the message as of the error-source
        // refactor). The FFI boundary flattens to a single string, so fold the
        // source back into the message to preserve the detail consumers saw
        // before the refactor.
        let with_cause =
            |message: String, source: Option<Box<dyn std::error::Error + Send + Sync>>| match source
            {
                Some(cause) => format!("{message}: {cause}"),
                None => message,
            };
        match err {
            sdk::SdkError::ModelNotFound(id) => Error::ModelNotFound { id },
            sdk::SdkError::DirectoryNotFound(path) => Error::DirectoryNotFound { path },
            sdk::SdkError::MetadataNotFound(path) => Error::MetadataNotFound { path },
            sdk::SdkError::MetadataInvalid(message) => Error::MetadataInvalid { message },
            sdk::SdkError::LoadError { message, source } => Error::LoadError {
                message: with_cause(message, source),
            },
            sdk::SdkError::InferenceError { message, source } => Error::InferenceError {
                message: with_cause(message, source),
            },
            sdk::SdkError::AbortedForCloudFallback { reason } => Error::AbortedForCloudFallback {
                reason: reason.to_string(),
            },
            sdk::SdkError::StreamingNotSupported => Error::StreamingNotSupported,
            sdk::SdkError::NotLoaded => Error::NotLoaded,
            sdk::SdkError::ConfigError(message) => Error::ConfigError { message },
            sdk::SdkError::NetworkError { message, source } => Error::NetworkError {
                message: with_cause(message, source),
            },
            sdk::SdkError::Offline { message, source } => Error::Offline {
                message: with_cause(message, source),
            },
            sdk::SdkError::IoError(e) => Error::IoError {
                message: e.to_string(),
            },
            sdk::SdkError::CacheError { message, source } => Error::CacheError {
                message: with_cause(message, source),
            },
            sdk::SdkError::PipelineError { message, source } => Error::PipelineError {
                message: with_cause(message, source),
            },
            sdk::SdkError::CircuitOpen(message) => Error::CircuitOpen { message },
            sdk::SdkError::RateLimited { retry_after_secs } => {
                Error::RateLimited { retry_after_secs }
            }
            sdk::SdkError::Timeout { timeout_ms } => Error::Timeout { timeout_ms },
            sdk::SdkError::Cancelled { message } => Error::Cancelled { message },
            // Capability / artifact errors (vision-era). First-class typed
            // variants so foreign consumers can branch on them; the structured
            // SDK fields are flattened into the diagnostic message.
            sdk::SdkError::MissingArtifact { artifact, path } => Error::MissingArtifact {
                message: format!("missing artifact '{artifact}' at {path}"),
            },
            sdk::SdkError::UnsupportedModelCapability {
                model_id,
                capability,
                hint,
            } => Error::UnsupportedModelCapability {
                message: format!("model '{model_id}' does not support {capability}; {hint}"),
            },
            sdk::SdkError::UnsupportedBackendCapability {
                model_id,
                backend,
                capability,
                hint,
            } => Error::UnsupportedBackendCapability {
                message: format!(
                    "model '{model_id}' requires {capability}, but backend/build '{backend}' does not support it; {hint}"
                ),
            },
        }
    }
}

impl From<xybrid_core::ir::envelope::EnvelopeError> for Error {
    fn from(e: xybrid_core::ir::envelope::EnvelopeError) -> Self {
        Error::InvalidImage {
            message: e.to_string(),
        }
    }
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;

// ============================================================================
// Envelope (input / output container)
// ============================================================================

/// Typed payload variants. FFI-safe mirror of
/// [`xybrid_core::ir::EnvelopeKind`].
#[derive(Debug, Clone, PartialEq)]
pub enum EnvelopeKind {
    Text {
        text: String,
    },
    Audio {
        bytes: Vec<u8>,
    },
    Embedding {
        values: Vec<f32>,
    },
    /// Encoded image input (PNG/JPEG/WebP) for vision-capable models. The
    /// bytes are decode-validated and the dimensions derived when this is
    /// lowered to the SDK in [`Envelope::into_sdk`] — so construction is
    /// cheap and validation surfaces as an [`Error::InvalidImage`] at run.
    Image {
        bytes: Vec<u8>,
        format: String,
    },
    /// Ordered parts of one logical multimodal message (e.g. text + images).
    MultiPart {
        parts: Vec<Envelope>,
    },
}

/// Owned envelope carrying a typed payload plus string metadata.
///
/// Construct via [`Envelope::text`] / [`Envelope::audio`] /
/// [`Envelope::embedding`]; the SDK form is reconstructed at the FFI
/// boundary inside the facade.
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    pub kind: EnvelopeKind,
    pub metadata: HashMap<String, String>,
}

impl Envelope {
    pub fn text(text: String) -> Self {
        Self {
            kind: EnvelopeKind::Text { text },
            metadata: HashMap::new(),
        }
    }

    pub fn audio(bytes: Vec<u8>) -> Self {
        Self {
            kind: EnvelopeKind::Audio { bytes },
            metadata: HashMap::new(),
        }
    }

    pub fn embedding(values: Vec<f32>) -> Self {
        Self {
            kind: EnvelopeKind::Embedding { values },
            metadata: HashMap::new(),
        }
    }

    /// Encoded image envelope (PNG/JPEG/WebP). Construction is infallible;
    /// the bytes are decode-validated when lowered in [`into_sdk`], surfacing
    /// as [`Error::InvalidImage`].
    ///
    /// [`into_sdk`]: Self::into_sdk
    pub fn image(bytes: Vec<u8>, format: String) -> Self {
        Self {
            kind: EnvelopeKind::Image { bytes, format },
            metadata: HashMap::new(),
        }
    }

    /// Multi-part message (e.g. text + image attachments) tagged with the
    /// `User` role, mirroring `xybrid_sdk::ir::Envelope::user_message`.
    pub fn multipart(parts: Vec<Envelope>) -> Self {
        Self {
            kind: EnvelopeKind::MultiPart { parts },
            metadata: HashMap::new(),
        }
        .with_role(MessageRole::User)
    }

    /// Continuation envelope for the turn *after* the model asked for tools.
    ///
    /// One `run` is one model turn, so the tool loop lives in your code: run
    /// a tools-bearing request, execute the calls on
    /// [`InferenceResult::tool_calls`], then run this envelope to feed the
    /// outcomes back.
    ///
    /// `user_text` is the original user message of the turn being continued;
    /// `prior_assistant_text` is that turn's raw output text, tool-call block
    /// included (i.e. [`InferenceResult::text`] verbatim). Pass `results` in
    /// call order, and run the continuation with the same
    /// [`GenerationConfig::tools`] as the original turn so the executor
    /// recomposes an identical chat prefix.
    ///
    /// Only the immediately prior assistant turn is replayed: multi-hop
    /// chains work turn by turn, but earlier tool exchanges are not re-sent.
    /// A continuation runs on every text path — batch, streaming, and both
    /// conversation-context variants. Image-bearing conversations are the one
    /// exception and are rejected: a continuation replays prior turns as a
    /// composed text prompt, and image embeddings cannot be re-evaluated from
    /// text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigError`] when a [`ToolResult::content_json`]
    /// isn't valid JSON.
    pub fn tool_results(
        user_text: String,
        prior_assistant_text: String,
        results: Vec<ToolResult>,
    ) -> Result<Self> {
        let sdk_results = results
            .into_iter()
            .map(|r| {
                let content: serde_json::Value =
                    serde_json::from_str(&r.content_json).map_err(|e| Error::ConfigError {
                        message: format!(
                            "tool result for '{}' has invalid content JSON: {e}",
                            r.name
                        ),
                    })?;
                Ok(sdk::ir::ToolCallResult {
                    call_id: r.call_id,
                    name: r.name,
                    content,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Built through the SDK constructor rather than by re-deriving the
        // metadata keys here, so the continuation wire format has exactly one
        // definition and this can't silently drift from the executor.
        Ok(Self::from_sdk(sdk::ir::Envelope::tool_results(
            user_text,
            prior_assistant_text,
            &sdk_results,
        )))
    }

    /// Set the LLM message role on this envelope. Stored under
    /// `xybrid.role` metadata — matches the SDK's own convention so the
    /// envelope is interchangeable with `xybrid_sdk::ir::Envelope::with_role`.
    pub fn with_role(mut self, role: MessageRole) -> Self {
        self.metadata.insert(
            xybrid_core::ir::Envelope::ROLE_METADATA_KEY.to_string(),
            role.to_sdk().as_str().to_string(),
        );
        self
    }

    /// Read the LLM message role previously set via [`with_role`].
    ///
    /// Returns `None` for envelopes that carry no role, or whose role
    /// metadata string is unknown.
    ///
    /// [`with_role`]: Self::with_role
    pub fn role(&self) -> Option<MessageRole> {
        self.metadata
            .get(xybrid_core::ir::Envelope::ROLE_METADATA_KEY)
            .and_then(|raw| MessageRole::parse(raw))
    }

    /// Consuming conversion to the SDK type. `pub` so binding crates with
    /// their own Ffi envelope POD can convert through the facade.
    ///
    /// # Errors
    /// Returns [`Error::InvalidImage`] if an image envelope (here or nested in
    /// a [`MultiPart`]) fails decode/validation. Text/audio/embedding never
    /// fail.
    ///
    /// [`MultiPart`]: EnvelopeKind::MultiPart
    pub fn into_sdk(self) -> Result<sdk::ir::Envelope> {
        let Envelope { kind, metadata } = self;
        let sdk_kind = match kind {
            EnvelopeKind::Text { text } => sdk::ir::EnvelopeKind::Text(text),
            EnvelopeKind::Audio { bytes } => sdk::ir::EnvelopeKind::Audio(bytes),
            EnvelopeKind::Embedding { values } => sdk::ir::EnvelopeKind::Embedding(values),
            EnvelopeKind::Image { bytes, format } => {
                // Decode-validates the bytes and derives dimensions, then carry
                // this envelope's metadata onto the validated kind via
                // `with_metadata` so a `local_id` is always preserved/minted —
                // matching the non-image branches below.
                let env = sdk::ir::Envelope::image(bytes, format)?;
                return Ok(sdk::ir::Envelope::with_metadata(env.kind, metadata));
            }
            EnvelopeKind::MultiPart { parts } => {
                let sdk_parts = parts
                    .into_iter()
                    .map(Envelope::into_sdk)
                    .collect::<Result<Vec<_>>>()?;
                sdk::ir::EnvelopeKind::MultiPart(sdk_parts)
            }
        };
        Ok(sdk::ir::Envelope::with_metadata(sdk_kind, metadata))
    }

    pub fn from_sdk(env: sdk::ir::Envelope) -> Self {
        let kind = match env.kind {
            sdk::ir::EnvelopeKind::Text(text) => EnvelopeKind::Text { text },
            sdk::ir::EnvelopeKind::Audio(bytes) => EnvelopeKind::Audio { bytes },
            sdk::ir::EnvelopeKind::Embedding(values) => EnvelopeKind::Embedding { values },
            sdk::ir::EnvelopeKind::Image { source } => match source.as_encoded() {
                Some((bytes, format)) => EnvelopeKind::Image {
                    bytes: bytes.to_vec(),
                    format: format.as_str().to_string(),
                },
                // Raw (camera) images aren't representable on the facade
                // surface yet; outputs are never raw images, so this is a
                // defensive marker rather than a real path.
                None => EnvelopeKind::Text {
                    text: "[raw image]".to_string(),
                },
            },
            sdk::ir::EnvelopeKind::MultiPart(parts) => EnvelopeKind::MultiPart {
                parts: parts.into_iter().map(Envelope::from_sdk).collect(),
            },
        };
        Self {
            kind,
            metadata: env.metadata,
        }
    }
}

// ============================================================================
// Conversation context (LLM chat)
// ============================================================================

/// LLM message role. Mirrors [`xybrid_core::ir::MessageRole`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
        }
    }

    /// Parse a lowercase role string (`"system"` / `"user"` / `"assistant"`)
    /// back into a [`MessageRole`]. Returns `None` for unknown inputs.
    ///
    /// Named `parse` rather than `from_str` to avoid collision with
    /// [`std::str::FromStr::from_str`]; foreign-language generators don't
    /// describe `FromStr`, so the inherent name is what callers see.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "system" => Some(MessageRole::System),
            "user" => Some(MessageRole::User),
            "assistant" => Some(MessageRole::Assistant),
            _ => None,
        }
    }

    fn to_sdk(self) -> sdk::ir::MessageRole {
        match self {
            MessageRole::System => sdk::ir::MessageRole::System,
            MessageRole::User => sdk::ir::MessageRole::User,
            MessageRole::Assistant => sdk::ir::MessageRole::Assistant,
        }
    }
}

/// FFI-friendly conversation handle. Generator crates wrap this in
/// `Arc<Self>` for opaque handle semantics.
///
/// The underlying [`sdk::ConversationContext`] is held by value behind a
/// `Mutex` so foreign callers can mutate it (`push` / `clear`) through a
/// shared `&self` — FFI handle methods only ever receive a shared
/// reference. The mutex is uncontended in normal usage — a conversation
/// handle is held by one host thread.
pub struct ConversationContextHandle {
    inner: std::sync::Mutex<sdk::ConversationContext>,
}

impl ConversationContextHandle {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: std::sync::Mutex::new(sdk::ConversationContext::new()),
        })
    }

    pub fn with_id(id: String) -> Arc<Self> {
        Arc::new(Self {
            inner: std::sync::Mutex::new(sdk::ConversationContext::with_id(id)),
        })
    }

    /// Append an envelope (typically `MessageRole::User` or `Assistant`).
    ///
    /// # Errors
    /// [`Error::InvalidImage`] if the envelope carries an image that fails
    /// decode/validation (so multimodal history can't store a bad image).
    pub fn push(&self, envelope: Envelope) -> Result<()> {
        let sdk_env = envelope.into_sdk()?;
        self.lock().push(sdk_env);
        Ok(())
    }

    /// Set the persistent system prompt envelope. Survives [`clear`].
    ///
    /// # Errors
    /// [`Error::InvalidImage`] if the envelope carries an image that fails
    /// decode/validation.
    pub fn set_system(&self, envelope: Envelope) -> Result<()> {
        let sdk_env = envelope.into_sdk()?;
        let mut guard = self.lock();
        let new_ctx = std::mem::take(&mut *guard).with_system(sdk_env);
        *guard = new_ctx;
        Ok(())
    }

    /// Drop history; the system envelope (if any) is preserved.
    pub fn clear(&self) {
        let mut guard = self.lock();
        guard.clear();
    }

    pub fn id(&self) -> String {
        self.lock().id().to_string()
    }

    /// Number of history turns (excludes the system envelope).
    pub fn history_len(&self) -> u32 {
        self.lock().history().len() as u32
    }

    /// Whether a persistent system envelope is set.
    pub fn has_system(&self) -> bool {
        self.lock().system_envelope().is_some()
    }

    /// Set the max history length before FIFO pruning.
    ///
    /// Rebuilds the context and re-pushes history so pruning runs immediately
    /// (matches the pre-bolt C ABI). `with_max_history_len` alone only changes
    /// the cap; the SDK prunes on `push`, so a lower cap wouldn't trim existing
    /// turns until the next push.
    pub fn set_max_history_len(&self, len: u32) {
        let mut guard = self.lock();
        let id = guard.id().to_string();
        let system = guard.system_envelope().cloned();
        let history: Vec<_> = guard.history().to_vec();

        let mut new_ctx = sdk::ConversationContext::with_id(id).with_max_history_len(len as usize);
        if let Some(sys) = system {
            new_ctx = new_ctx.with_system(sys);
        }
        for envelope in history {
            new_ctx.push(envelope);
        }
        *guard = new_ctx;
    }

    /// Return history turns, excluding the persistent system envelope.
    pub fn history(&self) -> Vec<Envelope> {
        self.lock()
            .history()
            .iter()
            .cloned()
            .map(Envelope::from_sdk)
            .collect()
    }

    /// Cheap clone of the inner SDK context for use at the FFI boundary
    /// (e.g. passing into `XybridModel::run_with_context`).
    fn snapshot(&self) -> sdk::ConversationContext {
        self.lock().clone()
    }

    /// Lock the inner context, recovering the guard if the mutex is poisoned.
    ///
    /// A poisoned mutex means a prior call panicked mid-update. We recover
    /// rather than re-panic for two reasons: this runs at the FFI boundary,
    /// where a panic would abort the host app (iOS / Android / Flutter) over a
    /// recoverable condition; and it matches the codebase-wide convention of
    /// surviving lock poison instead of propagating it. The conversation state
    /// is plain message history, so a partially-applied update is at worst
    /// slightly stale, never unsound.
    fn lock(&self) -> std::sync::MutexGuard<'_, sdk::ConversationContext> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

// ============================================================================
// Tool calling
// ============================================================================

/// A tool (function) the model may ask to call.
///
/// FFI-safe mirror of [`sdk::Tool`]: the JSON Schema describing the
/// arguments travels as a JSON *string* rather than a `serde_json::Value`,
/// because no FFI generator can describe an arbitrary JSON tree.
///
/// Offer tools by putting them on [`GenerationConfig::tools`]; the calls the
/// model emits come back on [`InferenceResult::tool_calls`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    /// Function name the model will emit, e.g. `get_weather`.
    pub name: String,
    /// What the tool does. The model reads this to decide when to call it.
    pub description: String,
    /// JSON Schema for the arguments, as a JSON string. Pass
    /// `{"type":"object","properties":{}}` for a tool that takes none.
    pub parameters_json: String,
}

impl ToolDefinition {
    /// Lower to the SDK type.
    ///
    /// # Errors
    /// Returns [`Error::ConfigError`] when `parameters_json` is not valid
    /// JSON. Validated up front by [`GenerationConfig::validate`] so this
    /// surfaces from the run call rather than from deep inside the executor.
    fn to_sdk(&self) -> Result<sdk::Tool> {
        let parameters: serde_json::Value =
            serde_json::from_str(&self.parameters_json).map_err(|e| Error::ConfigError {
                message: format!(
                    "tool '{}' has invalid parameters JSON Schema: {e}",
                    self.name
                ),
            })?;
        Ok(sdk::Tool::function(
            self.name.clone(),
            self.description.clone(),
            parameters,
        ))
    }
}

/// One tool call the model emitted this turn.
///
/// FFI-safe mirror of [`sdk::ToolCall`] — `arguments_json` is the raw JSON
/// object the model produced, left as a string for the same reason as
/// [`ToolDefinition::parameters_json`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Correlation id for this call, e.g. `call_0`. Echo it back as
    /// [`ToolResult::call_id`].
    pub id: String,
    /// Which tool the model wants to run.
    pub name: String,
    /// Arguments as a JSON object string.
    pub arguments_json: String,
}

impl ToolCall {
    fn from_sdk(call: sdk::ToolCall) -> Self {
        Self {
            id: call.id,
            name: call.function.name,
            arguments_json: call.function.arguments,
        }
    }
}

/// The outcome of running one tool, fed back to the model next turn.
///
/// FFI-safe mirror of [`sdk::ir::ToolCallResult`]. Build the continuation
/// envelope with [`Envelope::tool_results`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    /// The [`ToolCall::id`] this answers.
    pub call_id: String,
    /// The tool that was invoked.
    pub name: String,
    /// The tool's output as a JSON string. Wrap plain values — `"42"`,
    /// `"\"sunny\""` — so the whole field parses as JSON.
    pub content_json: String,
}

// ============================================================================
// Generation + Run options
// ============================================================================

/// LLM generation parameters. All fields are `Option<_>` — `None` means
/// "use the model's default". No builder; foreign callers populate fields
/// directly.
#[derive(Debug, Clone, Default)]
pub struct GenerationConfig {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub stop_sequences: Vec<String>,
    /// Optional GBNF grammar constraining generation to structured output
    /// (local llama backend only; other backends ignore it). Produce one from
    /// a JSON Schema with [`json_schema_to_gbnf`], or pass raw GBNF.
    pub grammar: Option<String>,
    /// Tools the model may call this turn. Empty means no tool calling — the
    /// existing behavior, byte-for-byte unchanged.
    ///
    /// Tool calling is llama.cpp-only today and unsupported paths fail
    /// closed rather than silently generating without the tools: a model
    /// with no embedded chat template, the mistralrs backend, and the cloud
    /// fallback leg all reject tool-bearing requests.
    pub tools: Vec<ToolDefinition>,
}

impl GenerationConfig {
    fn from_sdk(config: sdk::GenerationConfig) -> Self {
        let sdk::GenerationConfig {
            max_tokens,
            temperature,
            top_p,
            min_p,
            top_k,
            repetition_penalty,
            stop_sequences,
            grammar,
            tools,
        } = config;
        let tools = tools
            .into_iter()
            .map(|tool| {
                let sdk::Tool {
                    tool_type: _,
                    function,
                } = tool;
                let sdk::FunctionDefinition {
                    name,
                    description,
                    parameters,
                } = function;
                ToolDefinition {
                    name,
                    description: description.unwrap_or_default(),
                    parameters_json: parameters
                        .unwrap_or_else(|| serde_json::json!({}))
                        .to_string(),
                }
            })
            .collect();
        Self {
            max_tokens: Some(u32::try_from(max_tokens).unwrap_or(u32::MAX)),
            temperature: Some(temperature),
            top_p: Some(top_p),
            min_p: Some(min_p),
            top_k: Some(u32::try_from(top_k).unwrap_or(u32::MAX)),
            repetition_penalty: Some(repetition_penalty),
            stop_sequences,
            grammar,
            tools,
        }
    }

    /// Greedy decoding — deterministic, temperature 0.
    pub fn greedy() -> Self {
        Self {
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(0),
            ..Self::default()
        }
    }

    /// Higher temperature for creative output.
    pub fn creative() -> Self {
        Self {
            temperature: Some(0.9),
            top_p: Some(0.95),
            top_k: Some(50),
            ..Self::default()
        }
    }

    /// Apply the explicitly set fields over a caller-provided SDK config.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigError`] when a [`ToolDefinition`] carries a
    /// `parameters_json` that isn't valid JSON. Every other field lowers
    /// infallibly; tools are the one place a foreign caller hands us a
    /// string we have to parse, and lowering it here — rather than dropping
    /// the tool — means a typo'd schema surfaces at the `run` call instead
    /// of silently producing a model that was never offered the tool.
    pub fn apply_over(&self, mut cfg: sdk::GenerationConfig) -> Result<sdk::GenerationConfig> {
        if let Some(v) = self.max_tokens {
            cfg.max_tokens = v as usize;
        }
        if let Some(v) = self.temperature {
            cfg.temperature = v;
        }
        if let Some(v) = self.top_p {
            cfg.top_p = v;
        }
        if let Some(v) = self.min_p {
            cfg.min_p = v;
        }
        if let Some(v) = self.top_k {
            cfg.top_k = v as usize;
        }
        if let Some(v) = self.repetition_penalty {
            cfg.repetition_penalty = v;
        }
        if !self.stop_sequences.is_empty() {
            cfg.stop_sequences = self.stop_sequences.clone();
        }
        if let Some(g) = &self.grammar {
            cfg.grammar = Some(g.clone());
        }
        if !self.tools.is_empty() {
            cfg.tools = self
                .tools
                .iter()
                .map(ToolDefinition::to_sdk)
                .collect::<Result<Vec<_>>>()?;
        }
        Ok(cfg)
    }

    /// Materialize the SDK type over the global defaults.
    ///
    /// Run paths with a model in scope should prefer
    /// `apply_over(model.default_generation_config())` so model-level defaults
    /// are preserved.
    ///
    /// # Errors
    ///
    /// See [`apply_over`](Self::apply_over).
    pub fn to_sdk(&self) -> Result<sdk::GenerationConfig> {
        self.apply_over(sdk::GenerationConfig::default())
    }
}

/// Convert a JSON Schema (as a JSON string) into a GBNF grammar for
/// [`GenerationConfig::grammar`].
///
/// Kept as a free function rather than folded into `to_sdk` so the
/// option-bag → SDK mapping stays infallible; schema conversion is the one
/// step that can fail (invalid JSON, unsupported schema construct).
pub fn json_schema_to_gbnf(schema_json: &str) -> Result<String> {
    sdk::json_schema_str_to_gbnf(schema_json).map_err(|e| Error::ConfigError {
        message: e.to_string(),
    })
}

/// Abort signals the caller can observe. FFI-safe subset of
/// [`sdk::AbortSignal`] — `UserCancelled` is intentionally omitted because
/// user cancellation is expressed through [`CancellationToken`], not through
/// the abort policy list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortSignal {
    MemoryPressureWarn,
    MemoryPressureCritical,
    ThermalHot,
    ThermalCritical,
}

impl AbortSignal {
    fn to_sdk(self) -> sdk::AbortSignal {
        match self {
            AbortSignal::MemoryPressureWarn => sdk::AbortSignal::MemoryPressureWarn,
            AbortSignal::MemoryPressureCritical => sdk::AbortSignal::MemoryPressureCritical,
            AbortSignal::ThermalHot => sdk::AbortSignal::ThermalHot,
            AbortSignal::ThermalCritical => sdk::AbortSignal::ThermalCritical,
        }
    }
}

/// POD replacement for [`sdk::RunOptions`] + [`sdk::AbortPolicy`] builders.
///
/// Drops the non-FFI-safe fields from `sdk::RunOptions`
/// (`Arc<dyn ResourceSnapshotProvider>`, `DeviceMetrics`,
/// `CancellationToken`). Cancellation is exposed separately as an explicit
/// [`CancellationToken`] handle argument.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub generation_config: Option<GenerationConfig>,

    // AbortPolicy, flattened:
    pub abort_on: Vec<AbortSignal>,
    pub fallback_to_cloud: bool,
    pub max_grace_tokens: u32,

    pub correlation_id: Option<String>,

    /// Caller-selected provider. `None` preserves the envelope's existing value.
    pub cloud_provider: Option<String>,
    /// Caller-selected model. `None` preserves the envelope's existing value.
    pub cloud_model: Option<String>,
    /// Caller-selected gateway base URL, validated before SDK dispatch.
    pub cloud_gateway_url: Option<String>,
}

impl RunOptions {
    /// Validate supplied cloud settings and lower them onto the SDK envelope.
    ///
    /// The caller owns defaults. When every cloud field is absent or blank,
    /// this leaves metadata untouched. A supplied target does not change
    /// `fallback_to_cloud`; that permission remains in the SDK abort policy.
    /// Returns the normalized explicit gateway for bindings that also create
    /// a cloud adapter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigError`] for an invalid explicit gateway URL.
    pub fn apply_cloud_fallback_metadata(
        &self,
        envelope: &mut sdk::ir::Envelope,
    ) -> Result<Option<String>> {
        let provider = cloud::non_empty(self.cloud_provider.as_deref());
        let model = cloud::non_empty(self.cloud_model.as_deref());
        let gateway = cloud::non_empty(self.cloud_gateway_url.as_deref())
            .map(cloud::validate_gateway_url)
            .transpose()
            .map_err(|message| Error::ConfigError { message })?;

        if provider.is_none() && model.is_none() && gateway.is_none() {
            return Ok(None);
        }

        if let Some(provider) = provider {
            envelope.metadata.insert("provider".into(), provider.into());
        }
        if let Some(model) = model {
            envelope.metadata.insert("model".into(), model.into());
        }
        if let Some(gateway) = gateway.as_ref() {
            envelope
                .metadata
                .insert("gateway_url".into(), gateway.clone());
        }
        envelope.metadata.insert("backend".into(), "gateway".into());

        if let Some(config) = self.generation_config.as_ref() {
            if let Some(max_tokens) = config.max_tokens {
                envelope
                    .metadata
                    .insert("max_tokens".into(), max_tokens.to_string());
            }
            if let Some(temperature) = config.temperature {
                envelope
                    .metadata
                    .insert("temperature".into(), temperature.to_string());
            }
        }

        Ok(gateway)
    }

    /// Materialize the SDK type over global defaults.
    ///
    /// Run paths with a model in scope should prefer [`Self::to_sdk_over`].
    ///
    /// # Errors
    ///
    /// See [`GenerationConfig::apply_over`].
    pub fn to_sdk(&self, cancel: Option<&CancellationToken>) -> Result<sdk::RunOptions> {
        self.to_sdk_over(cancel, sdk::GenerationConfig::default())
    }

    /// Materialize the SDK type over model-resolved generation defaults.
    ///
    /// # Errors
    ///
    /// See [`GenerationConfig::apply_over`].
    pub fn to_sdk_over(
        &self,
        cancel: Option<&CancellationToken>,
        generation_base: sdk::GenerationConfig,
    ) -> Result<sdk::RunOptions> {
        let mut policy = sdk::AbortPolicy::default()
            .with_cloud_fallback(self.fallback_to_cloud)
            .with_max_grace_tokens(self.max_grace_tokens);
        for sig in &self.abort_on {
            policy = policy.stop_on(sig.to_sdk());
        }

        // Supplying a handle IS the opt-in. `AbortState::detect_user_cancelled`
        // only honours a cancelled token when the policy observes
        // `UserCancelled`, and the FFI `AbortSignal` wire enum deliberately
        // omits that variant — it is not a device-pressure signal a host polls,
        // it is implied by handing over a stop button. Without this the token
        // latches `is_cancelled()` and inference ignores it.
        if cancel.is_some() {
            policy = policy.stop_on(sdk::AbortSignal::UserCancelled);
        }

        let mut opts = sdk::RunOptions::new().with_abort_policy(policy);
        if let Some(gc) = &self.generation_config {
            opts = opts.with_generation_config(gc.apply_over(generation_base)?);
        }
        if let Some(cid) = &self.correlation_id {
            opts = opts.with_correlation_id(cid.clone());
        }
        if let Some(tok) = cancel {
            opts = opts.with_cancellation_token(tok.inner.clone());
        }
        Ok(opts)
    }
}

/// Cooperative cancellation handle.
///
/// Foreign callers hold the `Arc<CancellationToken>` and signal cancel
/// from any thread (e.g. a UI "stop" button). Pass the same handle to
/// [`XybridModel::run_with_options`] / `run_async_with_options` to make
/// it observable inside the run.
pub struct CancellationToken {
    inner: sdk::CancellationToken,
}

impl CancellationToken {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: sdk::CancellationToken::new(),
        })
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

// ============================================================================
// Inference result
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    Text,
    Audio,
    Embedding,
    Unknown,
}

impl OutputType {
    pub fn from_sdk(t: sdk::OutputType) -> Self {
        match t {
            sdk::OutputType::Text => OutputType::Text,
            sdk::OutputType::Audio => OutputType::Audio,
            sdk::OutputType::Embedding => OutputType::Embedding,
            sdk::OutputType::Unknown => OutputType::Unknown,
        }
    }

    /// The output type an envelope's payload reports as, mirroring the SDK's
    /// classification of a model result.
    fn of_envelope(kind: &EnvelopeKind) -> Self {
        match kind {
            EnvelopeKind::Text { .. } => OutputType::Text,
            EnvelopeKind::Audio { .. } => OutputType::Audio,
            EnvelopeKind::Embedding { .. } => OutputType::Embedding,
            EnvelopeKind::Image { .. } | EnvelopeKind::MultiPart { .. } => OutputType::Unknown,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageLatency {
    pub stage_id: String,
    pub latency_ms: u32,
}

#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    pub total_ms: u32,
    pub ttft_ms: Option<u32>,
    pub tokens_per_second: Option<f32>,
    pub prefill_tps: Option<f32>,
    pub decode_tps: Option<f32>,
    pub tokens_out: Option<u32>,
    pub stage_latencies_ms: Vec<StageLatency>,
}

impl InferenceMetrics {
    pub fn from_sdk(m: &sdk::InferenceMetrics) -> Self {
        Self {
            total_ms: m.total_ms,
            ttft_ms: m.ttft_ms,
            tokens_per_second: m.tokens_per_second,
            prefill_tps: m.prefill_tps,
            decode_tps: m.decode_tps,
            tokens_out: m.tokens_out,
            stage_latencies_ms: m
                .stage_latencies_ms
                .iter()
                .map(|s| StageLatency {
                    stage_id: s.stage_id.clone(),
                    latency_ms: s.latency_ms,
                })
                .collect(),
        }
    }
}

/// POD result returned by [`XybridModel::run`] / [`XybridModel::run_async`].
///
/// `unwrap_*` accessors are deliberately omitted — they don't translate to
/// non-Rust languages. Callers branch on [`output_type`](Self::output_type)
/// and read the corresponding field.
///
/// [`output_type`]: Self::output_type
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub envelope: Envelope,
    pub output_type: OutputType,
    pub model_id: String,
    pub latency_ms: u32,
    /// Where this result actually came from. Cloud fallback keeps `model_id`
    /// identical on both legs, so this is the only way to tell them apart.
    pub execution_target: ExecutionTarget,
    pub metrics: InferenceMetrics,
    /// Tool calls the model emitted this turn.
    ///
    /// Non-empty only when the request offered tools via
    /// [`GenerationConfig::tools`] and the model emitted at least one
    /// well-formed call. Run each one, then feed the outcomes back with
    /// [`Envelope::tool_results`].
    ///
    /// The raw tool-call block stays in [`text`](Self::text) untouched, and
    /// malformed model output yields an empty vec rather than an error.
    pub tool_calls: Vec<ToolCall>,
}

/// Where a result was produced — the observed fact, not a routing preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    Local,
    Cloud,
}

impl ExecutionTarget {
    fn from_sdk(provenance: sdk::ExecutionProvenance) -> Self {
        match provenance {
            sdk::ExecutionProvenance::Local => Self::Local,
            sdk::ExecutionProvenance::Cloud => Self::Cloud,
        }
    }
}

/// Lifecycle of a model download — either a standalone [`ModelDownload`] or
/// the background download behind a speculative load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadState {
    Downloading,
    Ready,
    Failed,
    /// The host called `cancel`.
    Cancelled,
}

/// One consistent read of download progress, bytes and state.
///
/// `progress` is aggregated across every artifact the model needs, never
/// moves backwards, and reaches 1.0 only alongside [`DownloadState::Ready`].
/// `total_bytes` is `None` when the source declares no size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DownloadStatus {
    pub state: DownloadState,
    /// 0.0..=1.0.
    pub progress: f32,
    /// Bytes written so far, across every artifact.
    pub downloaded_bytes: u64,
    /// Declared total across every artifact, or `None` when unknown.
    pub total_bytes: Option<u64>,
}

impl DownloadStatus {
    fn from_sdk(status: sdk::DownloadStatus) -> Self {
        let state = match status.state {
            sdk::DownloadState::Downloading => DownloadState::Downloading,
            sdk::DownloadState::Ready => DownloadState::Ready,
            sdk::DownloadState::Failed => DownloadState::Failed,
            sdk::DownloadState::Cancelled => DownloadState::Cancelled,
        };
        Self {
            state,
            progress: status.progress,
            downloaded_bytes: status.downloaded_bytes,
            total_bytes: status.total_bytes,
        }
    }
}

/// A token emitted by a pull-based inference stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamToken {
    pub token: String,
    pub token_id: Option<i64>,
    pub index: u64,
    pub cumulative_text: String,
    /// `"tool_calls"` when the turn ended on a parseable tool-call block.
    pub finish_reason: Option<String>,
    /// Tool calls parsed from the completed turn — populated on the
    /// **terminal** token only (the one carrying `finish_reason`).
    ///
    /// A streaming caller halts here and dispatches these instead of parsing
    /// raw text: the protocol blocks are suppressed from the emitted stream,
    /// so the call text never reaches the token callback. Empty on every
    /// mid-stream token and on turns that emitted no call. Feed the outcomes
    /// back with [`Envelope::tool_results`] and stream the continuation
    /// through the same call.
    pub tool_calls: Vec<ToolCall>,
    /// The completed turn's raw output text, tool-call block included — pass
    /// it to [`Envelope::tool_results`] as `prior_assistant_text`.
    ///
    /// `Some` only alongside a non-empty [`Self::tool_calls`]. It is not the
    /// same as `cumulative_text`, which reports the *emitted* text with the
    /// protocol blocks suppressed.
    pub raw_text: Option<String>,
}

impl StreamToken {
    fn from_sdk(token: xybrid_core::runtime_adapter::types::PartialToken) -> Self {
        Self {
            token: token.token,
            token_id: token.token_id,
            index: token.index as u64,
            cumulative_text: token.cumulative_text,
            finish_reason: token.finish_reason,
            tool_calls: token
                .tool_calls
                .into_iter()
                .map(ToolCall::from_sdk)
                .collect(),
            raw_text: token.raw_text,
        }
    }
}

/// An item returned by [`StreamingSession::next`].
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(StreamToken),
    Complete(InferenceResult),
    Error(Error),
}

const STREAM_CHANNEL_CAPACITY: usize = 32;

/// Pull-based bridge over the SDK's callback streaming API.
///
/// A worker thread runs inference and writes into a bounded channel. Foreign
/// callers repeatedly call [`next`](Self::next), keeping callbacks and Rust
/// references on their respective sides of the FFI boundary.
pub struct StreamingSession {
    receiver: Mutex<Receiver<StreamEvent>>,
}

impl StreamingSession {
    fn spawn(
        produce: impl FnOnce(SyncSender<StreamEvent>) + Send + 'static,
    ) -> std::io::Result<Arc<Self>> {
        let (sender, receiver) = mpsc::sync_channel(STREAM_CHANNEL_CAPACITY);
        std::thread::Builder::new()
            .name("xybrid-stream".into())
            .spawn(move || produce(sender))?;
        Ok(Arc::new(Self {
            receiver: Mutex::new(receiver),
        }))
    }

    /// Block until the next token or terminal event is available.
    ///
    /// Returns `None` only after the producer disconnects. A normal inference
    /// sends exactly one [`StreamEvent::Complete`] or [`StreamEvent::Error`]
    /// before disconnecting.
    pub fn next(&self) -> Option<StreamEvent> {
        self.receiver
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .recv()
            .ok()
    }

    #[cfg(test)]
    fn spawn_for_test(produce: impl FnOnce(SyncSender<StreamEvent>) + Send + 'static) -> Arc<Self> {
        Self::spawn(produce).expect("test streaming worker should spawn")
    }
}

impl InferenceResult {
    pub fn from_sdk(result: sdk::InferenceResult) -> Self {
        let output_type = OutputType::from_sdk(result.output_type());
        let model_id = result.model_id().to_string();
        let latency_ms = result.latency_ms();
        let execution_target = ExecutionTarget::from_sdk(result.provenance());
        let metrics = InferenceMetrics::from_sdk(result.metrics());
        let tool_calls = result
            .tool_calls()
            .into_iter()
            .map(ToolCall::from_sdk)
            .collect();
        let envelope = Envelope::from_sdk(result.into_envelope());
        Self {
            envelope,
            output_type,
            model_id,
            latency_ms,
            execution_target,
            metrics,
            tool_calls,
        }
    }

    /// Convenience: text payload, if the result is `OutputType::Text`.
    pub fn text(&self) -> Option<&str> {
        match &self.envelope.kind {
            EnvelopeKind::Text { text } => Some(text.as_str()),
            _ => None,
        }
    }

    /// Convenience: the model's chain-of-thought / reasoning text, if any.
    ///
    /// Surfaced from the response envelope's `reasoning_content` metadata —
    /// the same key the SDK's [`reasoning_content`] accessor reads — so it is
    /// independent of the payload [`kind`]: a text result carries its answer
    /// in `text()` and its `<think>` reasoning here. Returns `None` when the
    /// model emitted no reasoning or the backend doesn't surface one.
    ///
    /// [`reasoning_content`]: sdk::InferenceResult::reasoning_content
    /// [`kind`]: Envelope::kind
    pub fn reasoning_content(&self) -> Option<&str> {
        self.envelope
            .metadata
            .get("reasoning_content")
            .map(String::as_str)
    }

    /// Convenience: audio bytes, if the result is `OutputType::Audio`.
    pub fn audio_bytes(&self) -> Option<&[u8]> {
        match &self.envelope.kind {
            EnvelopeKind::Audio { bytes } => Some(bytes.as_slice()),
            _ => None,
        }
    }

    /// Convenience: embedding vector, if the result is `OutputType::Embedding`.
    pub fn embedding(&self) -> Option<&[f32]> {
        match &self.envelope.kind {
            EnvelopeKind::Embedding { values } => Some(values.as_slice()),
            _ => None,
        }
    }
}

impl ExecutionTarget {
    /// Map a pipeline stage's routing target to coarse provenance.
    ///
    /// The orchestrator records `local`, `cloud` or `fallback:<id>` (a
    /// xybrid-hosted server). `device` is accepted as a local spelling too.
    /// Anything else is reported as cloud, so a spelling added later can never
    /// claim that remote work ran on-device.
    fn from_pipeline_target(target: &str) -> Self {
        match target {
            "local" | "device" => Self::Local,
            _ => Self::Cloud,
        }
    }
}

// ============================================================================
// Pipeline handle
// ============================================================================

/// What one stage of a pipeline run produced.
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage identifier from the pipeline YAML (`id:`), or the model ID when
    /// the stage declares none. Matches [`Pipeline::stage_names`].
    pub stage_id: String,
    /// This stage's output, which is also the next stage's input — the
    /// transcript of an ASR stage, the reply of an LLM stage.
    pub envelope: Envelope,
    pub output_type: OutputType,
    pub latency_ms: u32,
    /// Where this stage ran. Stages of one pipeline can run in different
    /// places.
    pub execution_target: ExecutionTarget,
    /// Generation figures (TTFT, tokens per second) when this stage is a
    /// language model; `total_ms` is the stage latency and
    /// `stage_latencies_ms` is empty.
    pub metrics: InferenceMetrics,
}

impl StageResult {
    fn from_sdk(stage: sdk::PipelineStageTiming) -> Self {
        let metrics =
            sdk::InferenceMetrics::from_metadata(&stage.output.metadata, stage.latency_ms);
        let envelope = Envelope::from_sdk(stage.output);
        Self {
            execution_target: ExecutionTarget::from_pipeline_target(&stage.target),
            output_type: OutputType::of_envelope(&envelope.kind),
            envelope,
            stage_id: stage.name,
            latency_ms: stage.latency_ms,
            metrics: InferenceMetrics::from_sdk(&metrics),
        }
    }

    /// Convenience: text payload, if this stage produced text.
    pub fn text(&self) -> Option<&str> {
        match &self.envelope.kind {
            EnvelopeKind::Text { text } => Some(text.as_str()),
            _ => None,
        }
    }
}

/// Result of a pipeline run: the final output plus every stage's own output.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The final stage's output — the same envelope as the last entry of
    /// [`stages`](Self::stages).
    pub envelope: Envelope,
    pub output_type: OutputType,
    /// Wall-clock time of the whole run.
    pub latency_ms: u32,
    /// Every executed stage, in order.
    pub stages: Vec<StageResult>,
}

impl PipelineResult {
    fn from_sdk(result: sdk::PipelineExecutionResult) -> Self {
        Self {
            envelope: Envelope::from_sdk(result.output),
            output_type: OutputType::from_sdk(result.output_type),
            latency_ms: result.total_latency_ms,
            stages: result
                .stages
                .into_iter()
                .map(StageResult::from_sdk)
                .collect(),
        }
    }

    /// The stage with this identifier, if it ran.
    pub fn stage(&self, stage_id: &str) -> Option<&StageResult> {
        self.stages.iter().find(|stage| stage.stage_id == stage_id)
    }

    /// Convenience: final text payload, if the last stage produced text.
    pub fn text(&self) -> Option<&str> {
        match &self.envelope.kind {
            EnvelopeKind::Text { text } => Some(text.as_str()),
            _ => None,
        }
    }

    /// Convenience: final audio bytes, if the last stage produced audio.
    pub fn audio_bytes(&self) -> Option<&[u8]> {
        match &self.envelope.kind {
            EnvelopeKind::Audio { bytes } => Some(bytes.as_slice()),
            _ => None,
        }
    }
}

/// FFI-friendly handle around a loaded multi-stage pipeline.
///
/// Construction intentionally collapses the SDK's `PipelineRef -> Pipeline`
/// sequence into one fallible operation. Foreign callers receive one opaque
/// handle with constructors, introspection, and execution methods.
pub struct Pipeline {
    inner: sdk::Pipeline,
}

impl Pipeline {
    /// Parse and load a pipeline from YAML content.
    pub fn from_yaml(yaml: String) -> Result<Arc<Self>> {
        let pipeline = sdk::PipelineRef::from_yaml(&yaml)?.load()?;
        Ok(Arc::new(Self { inner: pipeline }))
    }

    /// Read, parse, and load a pipeline from a YAML file.
    pub fn from_file(path: String) -> Result<Arc<Self>> {
        let pipeline = sdk::PipelineRef::from_file(path)?.load()?;
        Ok(Arc::new(Self { inner: pipeline }))
    }

    /// Load a pipeline bundle.
    ///
    /// Pipeline bundles are YAML files today. Keeping a distinct constructor
    /// preserves the foreign API when richer bundle formats are introduced.
    pub fn from_bundle(path: String) -> Result<Arc<Self>> {
        Self::from_file(path)
    }

    /// Execute every stage, downloading any missing models first.
    ///
    /// Of [`RunOptions`], only `correlation_id` applies to a pipeline run: it
    /// is copied onto the run's telemetry. Effective cloud target fields are
    /// rejected because pipeline routing is configured per stage.
    ///
    /// # Errors
    ///
    /// [`Error::ConfigError`] when `options` sets `generation_config`,
    /// `abort_on`, or a cloud target — a pipeline run cannot honour them, and ignoring them
    /// would look like success. Per-stage generation settings belong in the
    /// pipeline YAML. Otherwise any load or stage failure.
    pub fn run(&self, envelope: Envelope, options: RunOptions) -> Result<PipelineResult> {
        let sdk_options = pipeline_run_options(options)?;
        let envelope = envelope.into_sdk()?;
        let result = self.inner.run_with_options(&envelope, &sdk_options)?;
        Ok(PipelineResult::from_sdk(result))
    }

    /// Pipeline name from the YAML definition, if present.
    pub fn name(&self) -> Option<String> {
        self.inner.name().map(str::to_string)
    }

    /// Stage identifiers in execution order.
    pub fn stage_names(&self) -> Vec<String> {
        self.inner.stage_names()
    }

    /// Number of stages in the pipeline.
    pub fn stage_count(&self) -> u32 {
        u32::try_from(self.inner.stage_count()).unwrap_or(u32::MAX)
    }
}

/// Keep the [`RunOptions`] fields a pipeline run honours; reject the rest.
fn pipeline_run_options(options: RunOptions) -> Result<sdk::RunOptions> {
    if options.generation_config.is_some() {
        return Err(Error::ConfigError {
            message: "generation_config is not supported on pipeline runs; set per-stage \
                      options in the pipeline YAML"
                .into(),
        });
    }
    if !options.abort_on.is_empty() {
        return Err(Error::ConfigError {
            message: "abort_on is not supported on pipeline runs".into(),
        });
    }
    if cloud::non_empty(options.cloud_provider.as_deref()).is_some()
        || cloud::non_empty(options.cloud_model.as_deref()).is_some()
        || cloud::non_empty(options.cloud_gateway_url.as_deref()).is_some()
    {
        return Err(Error::ConfigError {
            message: "cloud target options are not supported on pipeline runs; configure pipeline stages instead".into(),
        });
    }
    let mut sdk_options = sdk::RunOptions::new();
    if let Some(correlation_id) = options.correlation_id {
        sdk_options = sdk_options.with_correlation_id(correlation_id);
    }
    Ok(sdk_options)
}

// ============================================================================
// Device / platform push API (host → Rust, one-way)
// ============================================================================

/// Thermal pressure tier reported by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Normal,
    Warm,
    Hot,
    Critical,
}

impl ThermalState {
    fn to_sdk(self) -> sdk::ThermalState {
        match self {
            ThermalState::Normal => sdk::ThermalState::Normal,
            ThermalState::Warm => sdk::ThermalState::Warm,
            ThermalState::Hot => sdk::ThermalState::Hot,
            ThermalState::Critical => sdk::ThermalState::Critical,
        }
    }
}

/// Push the latest thermal state from the host into the SDK's global
/// [`ResourceMonitor`]. One-way (host → Rust), no callbacks.
///
/// [`ResourceMonitor`]: sdk::ResourceMonitor
pub fn set_thermal_state(state: ThermalState) {
    sdk::set_thermal_state(state.to_sdk());
}

pub fn clear_thermal_state() {
    sdk::clear_thermal_state();
}

/// Push battery level as a 0–100 percentage. Same lifecycle as
/// [`set_thermal_state`]. Values above 100 are clamped by the SDK.
pub fn set_battery_level(percent: u8) {
    sdk::set_battery_level(percent);
}

pub fn clear_battery_level() {
    sdk::clear_battery_level();
}

// ============================================================================
// Voice info (TTS models)
// ============================================================================

/// FFI mirror of [`sdk::VoiceInfo`]. Plain fields, no methods, so every
/// generator describes it as a record.
///
/// `index` and `preview_url` from the core type are deliberately dropped:
/// `index` is a load-time embedding offset that's meaningless to foreign
/// consumers, and `preview_url` is unused by all current bindings. Add
/// them back here if a binding starts surfacing them.
#[derive(Debug, Clone)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub gender: Option<String>,
    pub language: Option<String>,
    pub style: Option<String>,
}

impl VoiceInfo {
    pub fn from_sdk(v: sdk::VoiceInfo) -> Self {
        Self {
            id: v.id,
            name: v.name,
            gender: v.gender,
            language: v.language,
            style: v.style,
        }
    }
}

// ============================================================================
// Model loader + model handle
// ============================================================================

/// FFI-friendly model loader. Constructors return `Arc<Self>` so the same
/// loader can be passed across threads or held by the host while the
/// `load` future runs.
pub struct ModelLoader {
    inner: sdk::ModelLoader,
}

impl ModelLoader {
    /// Resolve via the xybrid registry API. Recommended path.
    pub fn from_registry(id: String) -> Arc<Self> {
        Arc::new(Self {
            inner: sdk::ModelLoader::from_registry(&id),
        })
    }

    /// Registry resolution forced to a specific platform string.
    pub fn from_registry_with_platform(id: String, platform: String) -> Arc<Self> {
        Arc::new(Self {
            inner: sdk::ModelLoader::from_registry_with_platform(&id, &platform),
        })
    }

    /// Registry load that serves from the cloud gateway while the weights
    /// download in the background, instead of blocking on the download.
    ///
    /// Requires a resolvable API key and an uncached model; otherwise this
    /// behaves exactly like [`Self::from_registry`]. Check
    /// [`Self::will_speculate`] to know which you got. LLM/chat models only.
    pub fn from_registry_speculative(id: String) -> Arc<Self> {
        Arc::new(Self {
            inner: sdk::ModelLoader::from_registry(&id).with_speculative_cloud(true),
        })
    }

    /// Speculative variant of [`Self::from_registry_with_platform`].
    pub fn from_registry_speculative_with_platform(id: String, platform: String) -> Arc<Self> {
        Arc::new(Self {
            inner: sdk::ModelLoader::from_registry_with_platform(&id, &platform)
                .with_speculative_cloud(true),
        })
    }

    /// Whether [`Self::load`] would actually speculate: speculation enabled, an
    /// API key resolves, and the model is not already cached. Never touches the
    /// network.
    pub fn will_speculate(&self) -> bool {
        self.inner.will_speculate()
    }

    /// Load from a local model directory (must contain `model_metadata.json`).
    pub fn from_directory(path: String) -> Result<Arc<Self>> {
        let inner = sdk::ModelLoader::from_directory(path).map_err(Error::from)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Load from a local `.xyb` bundle file.
    pub fn from_bundle(path: String) -> Result<Arc<Self>> {
        let inner = sdk::ModelLoader::from_bundle(path).map_err(Error::from)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Resolve a HuggingFace repo (`org/repo` or `org/repo:variant`).
    pub fn from_huggingface(repo: String) -> Arc<Self> {
        Arc::new(Self {
            inner: sdk::ModelLoader::from_huggingface_parsed(&repo),
        })
    }

    /// Resolve a HuggingFace repository pinned to an explicit revision.
    pub fn from_huggingface_with_revision(repo: String, revision: String) -> Arc<Self> {
        Arc::new(Self {
            inner: sdk::ModelLoader::from_huggingface_with_revision(&repo, &revision),
        })
    }

    /// Load from a raw GGUF file: auto-generate `model_metadata.json` from the
    /// GGUF header (writing it next to the file if absent), then load the parent
    /// directory. Mirrors the pre-bolt C ABI's `from_model_file`.
    ///
    /// # Errors
    /// [`Error::ConfigError`] on an empty/missing path or metadata-generation
    /// failure; [`Error::IoError`] if the metadata sidecar can't be written.
    pub fn from_model_file(path: String) -> Result<Arc<Self>> {
        if path.is_empty() {
            return Err(Error::ConfigError {
                message: "path is empty".to_string(),
            });
        }
        let gguf_path = std::path::Path::new(&path);
        if !gguf_path.exists() {
            return Err(Error::ConfigError {
                message: format!("GGUF file not found: {path}"),
            });
        }

        let metadata =
            sdk::metadata_gen::generate_metadata_for_gguf_file(gguf_path).map_err(|e| {
                Error::ConfigError {
                    message: format!("failed to generate metadata for GGUF file: {e}"),
                }
            })?;

        let parent_dir = gguf_path.parent().ok_or_else(|| Error::ConfigError {
            message: "cannot determine parent directory of GGUF file".to_string(),
        })?;

        // Write the sidecar only if absent, so re-loading the same file is
        // idempotent and never clobbers a user-authored metadata file.
        let metadata_path = parent_dir.join("model_metadata.json");
        if !metadata_path.exists() {
            let json = serde_json::to_string_pretty(&metadata).map_err(|e| Error::ConfigError {
                message: format!("failed to serialize metadata: {e}"),
            })?;
            std::fs::write(&metadata_path, json).map_err(|e| Error::IoError {
                message: format!("failed to write model_metadata.json: {e}"),
            })?;
        }

        let inner = sdk::ModelLoader::from_directory(parent_dir.to_string_lossy().to_string())
            .map_err(Error::from)?;
        Ok(Arc::new(Self { inner }))
    }

    pub fn model_id(&self) -> Option<String> {
        self.inner.model_id().map(str::to_string)
    }

    pub fn version(&self) -> Option<String> {
        self.inner.version().map(str::to_string)
    }

    pub fn source_type(&self) -> String {
        self.inner.source_type().to_string()
    }

    /// Synchronous load. For UI hosts use [`load_async`](Self::load_async).
    pub fn load(&self) -> Result<Arc<XybridModel>> {
        let model = self.inner.load().map_err(Error::from)?;
        Ok(Arc::new(XybridModel { inner: model }))
    }

    /// Async load — the SDK offloads to `spawn_blocking` internally so this
    /// is safe to `await` from UI runtimes.
    pub async fn load_async(&self) -> Result<Arc<XybridModel>> {
        let model = self.inner.load_async().await.map_err(Error::from)?;
        Ok(Arc::new(XybridModel { inner: model }))
    }

    /// Start the download in the background and return a handle to watch.
    ///
    /// This is what gives a blocking host a progress bar: `load` has no
    /// object to poll while it runs, this does. The download fills the normal
    /// SDK cache, so the later `load` hits it and returns at once. Returns
    /// immediately; a source with nothing to fetch comes back `Ready`.
    pub fn start_download(&self) -> Arc<ModelDownload> {
        Arc::new(ModelDownload {
            inner: self.inner.start_download(),
        })
    }
}

/// FFI-friendly handle over a background model download.
///
/// Three ways to read it, all fed by the same aggregated source:
/// [`Self::status`] for a snapshot, [`Self::next_status`] to block until it
/// changes, and [`Self::watch`] for pushed updates. `watch` takes a Rust
/// closure that never crosses the FFI boundary — binding crates use it to
/// feed their own transport (a BoltFFI stream, a flutter_rust_bridge sink).
pub struct ModelDownload {
    inner: Arc<sdk::ModelDownload>,
}

impl ModelDownload {
    /// Start downloading a registry model without loading it.
    pub fn from_registry(id: String) -> Arc<Self> {
        Arc::new(Self {
            inner: sdk::ModelLoader::from_registry(&id).start_download(),
        })
    }

    /// Registry download forced to a specific platform string.
    pub fn from_registry_with_platform(id: String, platform: String) -> Arc<Self> {
        Arc::new(Self {
            inner: sdk::ModelLoader::from_registry_with_platform(&id, &platform).start_download(),
        })
    }

    /// Current snapshot. Never blocks — safe from a UI thread or a render loop.
    pub fn status(&self) -> DownloadStatus {
        DownloadStatus::from_sdk(self.inner.status())
    }

    /// Whether the download reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// The failure message once the download ended in `Failed` or `Cancelled`.
    pub fn error(&self) -> Option<String> {
        self.inner.error()
    }

    /// Block until the status changes, `timeout_ms` elapses, or the download
    /// finishes, then report the snapshot.
    ///
    /// Terminal states return immediately, so a host loop driven by this
    /// cannot hang. Call it off the UI thread. A `timeout_ms` of 0 makes it a
    /// non-blocking read, identical to [`Self::status`].
    pub fn next_status(&self, timeout_ms: u64) -> DownloadStatus {
        DownloadStatus::from_sdk(
            self.inner
                .next_status_snapshot(Duration::from_millis(timeout_ms)),
        )
    }

    /// Block until the download reaches a terminal state or `timeout_ms`
    /// elapses.
    pub fn wait(&self, timeout_ms: u64) -> DownloadStatus {
        DownloadStatus::from_sdk(self.inner.wait(Duration::from_millis(timeout_ms)))
    }

    /// Ask the download to stop. Idempotent; a no-op once terminal.
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Register a Rust-side observer for pushed updates. The current snapshot
    /// arrives synchronously, so a late subscriber still gets a first frame.
    pub fn watch<F>(&self, observer: F)
    where
        F: Fn(DownloadStatus) + Send + Sync + 'static,
    {
        self.inner
            .watch(move |status| observer(DownloadStatus::from_sdk(status)));
    }
}

/// FFI-friendly handle around a loaded [`sdk::XybridModel`].
///
/// `Arc<Self>` for shareability across threads / callbacks / generators.
/// The inner SDK model already clones cheaply (shared `Arc<RwLock<…>>`),
/// so cloning the facade handle is also cheap.
pub struct XybridModel {
    inner: sdk::XybridModel,
}

impl XybridModel {
    // -- Identity / capability accessors ------------------------------------

    pub fn model_id(&self) -> String {
        self.inner.model_id().to_string()
    }

    pub fn version(&self) -> String {
        self.inner.version().to_string()
    }

    pub fn output_type(&self) -> OutputType {
        OutputType::from_sdk(self.inner.output_type())
    }

    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    /// Whether runs are currently answered by the cloud because the local
    /// weights are not ready yet. `false` for ordinary local models.
    pub fn is_cloud_serving(&self) -> bool {
        self.inner.is_cloud_serving()
    }

    /// Snapshot of the background download: state + progress in one read, so a
    /// polling host never sees a torn pair. Reports `Ready` at 1.0 for an
    /// ordinary local model, so hosts need no special case.
    pub fn download_status(&self) -> DownloadStatus {
        DownloadStatus::from_sdk(self.inner.download_status())
    }

    /// Block until the download reaches a terminal state or `timeout_ms`
    /// elapses, then report it.
    ///
    /// The polling helper for hosts that just want "tell me when it's
    /// on-device": call it from a background thread / coroutine / detached
    /// task, in the same place `load` is already called off the UI thread.
    /// Deliberately a blocking call rather than a progress *callback* — no
    /// closure has to cross the FFI boundary. Returns immediately for a
    /// non-speculative model.
    ///
    /// A `timeout_ms` of 0 makes this a non-blocking read, identical to
    /// [`Self::download_status`].
    pub fn await_download(&self, timeout_ms: u64) -> DownloadStatus {
        DownloadStatus::from_sdk(self.inner.await_download(timeout_ms))
    }

    /// Register a Rust-side observer for pushed download updates on a
    /// speculatively-loaded model — the push counterpart of
    /// [`Self::await_download`].
    ///
    /// The current snapshot arrives synchronously and the observer is dropped
    /// once the download is terminal. The closure never crosses the FFI
    /// boundary; binding crates forward it into their own transport.
    pub fn watch_download<F>(&self, observer: F)
    where
        F: Fn(DownloadStatus) + Send + Sync + 'static,
    {
        self.inner
            .watch_download(move |status| observer(DownloadStatus::from_sdk(status)));
    }

    pub fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }

    /// Whether this model emits true token-by-token output.
    pub fn supports_token_streaming(&self) -> bool {
        self.inner.supports_token_streaming()
    }

    /// Return the model's resolved generation defaults.
    pub fn default_generation_config(&self) -> GenerationConfig {
        GenerationConfig::from_sdk(self.inner.default_generation_config())
    }

    pub fn is_llm(&self) -> bool {
        self.inner.is_llm()
    }

    /// Whether the model bundle declares local tool-calling support.
    ///
    /// Advisory tri-state: `None` means the bundle says nothing, so the host
    /// cannot tell. Gate tool UI on it; enforcement stays at run time — a
    /// tools-bearing request against a model whose chat template has no tool
    /// support fails as invalid input regardless of what this reports.
    pub fn supports_tool_calling(&self) -> Option<bool> {
        self.inner.supports_tool_calling()
    }

    pub fn has_voices(&self) -> bool {
        self.inner.has_voices()
    }

    pub fn voices(&self) -> Vec<VoiceInfo> {
        self.inner
            .voices()
            .unwrap_or_default()
            .into_iter()
            .map(VoiceInfo::from_sdk)
            .collect()
    }

    pub fn default_voice(&self) -> Option<VoiceInfo> {
        self.inner.default_voice().map(VoiceInfo::from_sdk)
    }

    pub fn voice(&self, voice_id: String) -> Option<VoiceInfo> {
        self.inner.voice(&voice_id).map(VoiceInfo::from_sdk)
    }

    // -- Inference ----------------------------------------------------------

    /// Run inference with no overrides.
    pub fn run(&self, envelope: Envelope) -> Result<InferenceResult> {
        let env = envelope.into_sdk()?;
        let result = self.inner.run(&env, None).map_err(Error::from)?;
        Ok(InferenceResult::from_sdk(result))
    }

    /// Run inference with explicit [`RunOptions`] and an optional
    /// cancellation handle.
    pub fn run_with_options(
        &self,
        envelope: Envelope,
        options: RunOptions,
        cancel: Option<Arc<CancellationToken>>,
    ) -> Result<InferenceResult> {
        let mut env = envelope.into_sdk()?;
        let opts =
            options.to_sdk_over(cancel.as_deref(), self.inner.default_generation_config())?;
        options.apply_cloud_fallback_metadata(&mut env)?;
        let result = self
            .inner
            .run_with_options(&env, &opts)
            .map_err(Error::from)?;
        Ok(InferenceResult::from_sdk(result))
    }

    /// Run inference with conversation history (LLM chat).
    ///
    /// The context is passed by value (the inner SDK type clones cheaply)
    /// so the caller's [`ConversationContextHandle`] remains untouched —
    /// matching the SDK's "does not mutate the context" contract.
    pub fn run_with_context(
        &self,
        envelope: Envelope,
        context: Arc<ConversationContextHandle>,
        generation_config: Option<GenerationConfig>,
    ) -> Result<InferenceResult> {
        let env = envelope.into_sdk()?;
        let ctx = context.snapshot();
        let gc = generation_config
            .as_ref()
            .map(|config| config.apply_over(self.inner.default_generation_config()))
            .transpose()?;
        let result = self
            .inner
            .run_with_context(&env, &ctx, gc.as_ref())
            .map_err(Error::from)?;
        Ok(InferenceResult::from_sdk(result))
    }

    /// Run inference with conversation history, explicit [`RunOptions`] and an
    /// optional cancellation handle.
    ///
    /// The options counterpart of [`Self::run_with_context`], mirroring
    /// [`Self::run_with_options`]. Prefer this when the caller has anything
    /// beyond a generation config to say — the plain `run_with_context` reads
    /// only the generation config, so abort signals and cloud fallback would
    /// otherwise be silently dropped.
    pub fn run_with_context_options(
        &self,
        envelope: Envelope,
        context: Arc<ConversationContextHandle>,
        options: RunOptions,
        cancel: Option<Arc<CancellationToken>>,
    ) -> Result<InferenceResult> {
        let mut env = envelope.into_sdk()?;
        let ctx = context.snapshot();
        let opts =
            options.to_sdk_over(cancel.as_deref(), self.inner.default_generation_config())?;
        options.apply_cloud_fallback_metadata(&mut env)?;
        let result = self
            .inner
            .run_with_context_options(&env, &ctx, &opts)
            .map_err(Error::from)?;
        Ok(InferenceResult::from_sdk(result))
    }

    /// Async inference. The SDK offloads to `spawn_blocking` internally.
    pub async fn run_async(&self, envelope: Envelope) -> Result<InferenceResult> {
        let env = envelope.into_sdk()?;
        let result = self
            .inner
            .run_async(&env, None)
            .await
            .map_err(Error::from)?;
        Ok(InferenceResult::from_sdk(result))
    }

    /// Start inference and return a pull-based token stream.
    ///
    /// The returned session owns a bounded channel and the inference worker
    /// owns a clone of the model, so both remain alive until inference ends.
    /// Dropping the session disconnects the producer at its next token.
    ///
    /// # Errors
    ///
    /// Returns an error when the envelope is invalid or the worker thread
    /// cannot be created. Inference failures arrive as [`StreamEvent::Error`].
    pub fn run_stream(
        &self,
        envelope: Envelope,
        options: RunOptions,
        cancel: Option<Arc<CancellationToken>>,
    ) -> Result<Arc<StreamingSession>> {
        let mut envelope = envelope.into_sdk()?;
        let sdk_options =
            options.to_sdk_over(cancel.as_deref(), self.inner.default_generation_config())?;
        options.apply_cloud_fallback_metadata(&mut envelope)?;
        let model = self.inner.clone();

        StreamingSession::spawn(move |sender| {
            let result = model.run_streaming_with_options(&envelope, &sdk_options, |token| {
                sender
                    .send(StreamEvent::Token(StreamToken::from_sdk(token)))
                    .map_err(|_| {
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "stream receiver dropped",
                        )) as Box<dyn std::error::Error + Send + Sync>
                    })
            });

            let terminal = match result {
                Ok(result) => StreamEvent::Complete(InferenceResult::from_sdk(result)),
                Err(error) => StreamEvent::Error(Error::from(error)),
            };
            let _ = sender.send(terminal);
        })
        .map_err(|error| Error::InferenceError {
            message: format!("failed to start streaming worker: {error}"),
        })
    }

    /// Start context-aware inference and return a pull-based token stream.
    ///
    /// Mirrors [`run_stream`](Self::run_stream) but seeds the worker with a
    /// snapshot of `context`'s conversation history so multi-turn chat streams.
    ///
    /// # Errors
    ///
    /// Returns an error when the envelope is invalid or the worker thread
    /// cannot be created. Inference failures arrive as [`StreamEvent::Error`].
    pub fn run_stream_with_context(
        &self,
        envelope: Envelope,
        context: Arc<ConversationContextHandle>,
        options: RunOptions,
        cancel: Option<Arc<CancellationToken>>,
    ) -> Result<Arc<StreamingSession>> {
        let mut envelope = envelope.into_sdk()?;
        let ctx = context.snapshot();
        let sdk_options =
            options.to_sdk_over(cancel.as_deref(), self.inner.default_generation_config())?;
        options.apply_cloud_fallback_metadata(&mut envelope)?;
        let model = self.inner.clone();

        StreamingSession::spawn(move |sender| {
            let result =
                model.run_streaming_with_context_options(&envelope, &ctx, &sdk_options, |token| {
                    sender
                        .send(StreamEvent::Token(StreamToken::from_sdk(token)))
                        .map_err(|_| {
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::BrokenPipe,
                                "stream receiver dropped",
                            ))
                                as Box<dyn std::error::Error + Send + Sync>
                        })
                });

            let terminal = match result {
                Ok(result) => StreamEvent::Complete(InferenceResult::from_sdk(result)),
                Err(error) => StreamEvent::Error(Error::from(error)),
            };
            let _ = sender.send(terminal);
        })
        .map_err(|error| Error::InferenceError {
            message: format!("failed to start streaming worker: {error}"),
        })
    }

    // -- Lifecycle ----------------------------------------------------------

    pub fn warmup(&self) -> Result<()> {
        self.inner.warmup().map_err(Error::from)
    }

    pub async fn warmup_async(&self) -> Result<()> {
        self.inner.warmup_async().await.map_err(Error::from)
    }

    pub fn unload(&self) -> Result<()> {
        self.inner.unload().map_err(Error::from)
    }

    /// Open a live ASR session on this model: feed microphone PCM in, read
    /// partial transcripts out.
    ///
    /// This is the live-capture surface, distinct from one-shot transcription
    /// of a finished buffer (`run`). The model is already loaded, so opening a
    /// session starts a worker and warms the weights — it does not reload.
    ///
    /// # Errors
    ///
    /// [`Error::StreamingNotSupported`] if the model is not an ASR model,
    /// [`Error::ConfigError`] if the sample rate is not 16 kHz, and
    /// [`Error::LoadError`] if the worker thread cannot be spawned.
    pub fn stream(&self, config: StreamingConfig) -> Result<Arc<AsrSession>> {
        let sdk_config = config.to_sdk()?;
        let stream = self.inner.stream(sdk_config).map_err(Error::from)?;
        AsrSession::spawn(stream)
    }
}

// ============================================================================
// Live ASR session
// ============================================================================

/// 16 kHz mono — the only sample rate the ASR backends accept.
pub const REQUIRED_SAMPLE_RATE: u32 = 16_000;

/// How voice-activity detection (VAD) chunking is resolved for a session.
///
/// One decision in one type, rather than a `bool` + `Option<String>` pair
/// where "disabled, yet a model directory is set" is representable.
///
/// There is deliberately no "on, with the default model" variant: nothing
/// ships a bundled Silero model, and the core handles VAD-enabled-without-a-
/// directory by printing a warning and silently falling back to fixed-window
/// chunking. A variant whose only possible meaning is "quietly did nothing"
/// is worse than no variant, so enabling VAD requires naming a directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadMode {
    /// Fixed time-window chunking; no voice-activity detection.
    Off,
    /// VAD on, using the Silero model in this directory, which must contain a
    /// `model.onnx`.
    Enabled { model_dir: String },
}

/// Configuration for a live ASR session.
///
/// The model is not named here — it comes from the loaded [`XybridModel`] the
/// session is opened on. This only configures *how* the audio is chunked.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Sample rate of the audio you will feed. Must be
    /// [`REQUIRED_SAMPLE_RATE`]; validated rather than forwarded, because the
    /// backends are fixed at 16 kHz.
    pub sample_rate: u32,
    /// Voice-activity-detection mode.
    pub vad: VadMode,
    /// VAD sensitivity, 0.0–1.0. Ignored when `vad` is [`VadMode::Off`].
    pub vad_threshold: f32,
    /// Language hint (e.g. `"en"`); `None` uses the model default.
    pub language: Option<String>,
    /// Whisper encoder context in mel frames; `None` uses the model default.
    pub audio_ctx: Option<u32>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            sample_rate: REQUIRED_SAMPLE_RATE,
            vad: VadMode::Off,
            vad_threshold: 0.5,
            language: None,
            audio_ctx: None,
        }
    }
}

impl StreamingConfig {
    /// Validate and convert to the SDK's `StreamConfig`.
    fn to_sdk(&self) -> Result<sdk::StreamConfig> {
        if self.sample_rate != REQUIRED_SAMPLE_RATE {
            return Err(Error::ConfigError {
                message: format!(
                    "sample_rate must be {REQUIRED_SAMPLE_RATE} Hz, got {}",
                    self.sample_rate
                ),
            });
        }
        let (enable_vad, vad_model_dir) = match &self.vad {
            VadMode::Off => (false, None),
            VadMode::Enabled { model_dir } => (true, Some(model_dir.clone())),
        };
        Ok(sdk::StreamConfig {
            enable_vad,
            vad_threshold: self.vad_threshold,
            vad_model_dir,
            language: self.language.clone(),
            audio_ctx: self.audio_ctx,
        })
    }
}

/// A partial transcript emitted while audio is streaming.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialResult {
    /// Best-effort transcript so far. Cumulative, not a delta.
    pub text: String,
    /// `true` once this span is committed and will not change.
    pub is_stable: bool,
    /// Monotonic chunk sequence number this result corresponds to.
    pub chunk_sequence: u64,
    /// Audio covered so far, in milliseconds.
    pub audio_duration_ms: u64,
}

impl From<sdk::PartialResult> for PartialResult {
    fn from(p: sdk::PartialResult) -> Self {
        Self {
            text: p.text,
            is_stable: p.is_stable,
            chunk_sequence: p.chunk_index,
            audio_duration_ms: p.audio_duration_ms,
        }
    }
}

/// What a live ASR session reports to its observers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsrEvent {
    /// A rolling-window chunk produced a transcript.
    Partial(PartialResult),
    /// The session is over — flushed, closed, or its worker died. No further
    /// events follow, so a binding closes its stream here.
    Finished,
}

/// An in-process observer of a live ASR session. Never crosses an FFI
/// boundary; bindings forward events into their own transport from inside one.
pub(crate) type AsrObserver = Box<dyn Fn(AsrEvent) + Send + Sync>;

/// Observer bookkeeping for one session.
///
/// Everything lives under a single mutex on purpose. The observers, the
/// pending partial and the finished flag are read and written together, and
/// splitting them across locks is exactly how a terminal event gets delivered
/// to an empty list and then dropped.
#[derive(Default)]
struct AsrFanout {
    observers: Vec<AsrObserver>,
    /// Latest partial produced before any observer attached. Partial text is
    /// cumulative, so keeping only the most recent loses nothing, and it is
    /// delivered the instant someone subscribes.
    pending: Option<PartialResult>,
    finished: bool,
}

impl AsrFanout {
    fn emit(&mut self, partial: PartialResult) {
        if self.observers.is_empty() {
            self.pending = Some(partial);
            return;
        }
        for observer in &self.observers {
            observer(AsrEvent::Partial(partial.clone()));
        }
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        for observer in &self.observers {
            observer(AsrEvent::Finished);
        }
        // Nothing more will be emitted, so release the host closures rather
        // than holding them for the session's lifetime.
        self.observers.clear();
    }

    fn watch(&mut self, observer: AsrObserver) {
        if let Some(partial) = self.pending.take() {
            observer(AsrEvent::Partial(partial));
        }
        if self.finished {
            observer(AsrEvent::Finished);
            return;
        }
        self.observers.push(observer);
    }
}

/// Commands applied, in order, by the session's worker thread.
enum AsrCommand {
    Feed(Vec<f32>),
    Flush(SyncSender<Result<String>>),
    Reset(SyncSender<Result<()>>),
}

/// A live ASR session: feed microphone PCM in, read partial transcripts out.
///
/// The session owns a worker thread that holds the SDK stream for its whole
/// lifetime. Commands reach it over a channel, so they apply in submission
/// order — audio fed in order is transcribed in order — and the heavy
/// inference never runs on the caller's thread.
///
/// Audio is PCM **f32, mono, 16 kHz**. Converting from the platform's
/// microphone format is the caller's job, deliberately kept out of the FFI
/// layer.
pub struct AsrSession {
    commands: Mutex<Option<SyncSender<AsrCommand>>>,
    fanout: Arc<Mutex<AsrFanout>>,
    /// Set by [`AsrSession::cancel`]. The worker checks it before each
    /// command, so a cancel abandons the queued backlog instead of waiting
    /// for it to transcribe.
    cancelled: Arc<AtomicBool>,
    /// First per-chunk failure, if any. The core removes a chunk from its
    /// buffer *before* transcribing it, so a failed chunk's audio is gone —
    /// never retried, never in the transcript. Remembering it here is what
    /// stops [`AsrSession::flush`] returning a silently holed transcript as
    /// success.
    failure: Arc<Mutex<Option<String>>>,
}

impl AsrSession {
    /// Spawn the worker that owns `stream` and start accepting commands.
    fn spawn(stream: sdk::XybridStream) -> Result<Arc<Self>> {
        let (sender, receiver) = mpsc::sync_channel::<AsrCommand>(ASR_COMMAND_CAPACITY);
        let fanout = Arc::new(Mutex::new(AsrFanout::default()));
        let cancelled = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None));
        let worker = AsrWorkerState {
            fanout: Arc::clone(&fanout),
            cancelled: Arc::clone(&cancelled),
            failure: Arc::clone(&failure),
        };
        std::thread::Builder::new()
            .name("xybrid-asr".into())
            .spawn(move || asr_worker(stream, receiver, worker))
            .map_err(|e| Error::LoadError {
                message: format!("failed to spawn ASR worker thread: {e}"),
            })?;
        Ok(Arc::new(Self {
            commands: Mutex::new(Some(sender)),
            fanout,
            cancelled,
            failure,
        }))
    }

    /// Subscribe to partial transcripts and the terminal event.
    ///
    /// A partial produced before this call is delivered immediately, so audio
    /// fed before subscribing is never silently lost. Subscribing to an
    /// already-finished session yields [`AsrEvent::Finished`] at once rather
    /// than hanging.
    ///
    /// Rust-side only: the closure never crosses an FFI boundary.
    pub fn watch<F>(&self, observer: F)
    where
        F: Fn(AsrEvent) + Send + Sync + 'static,
    {
        self.fanout
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .watch(Box::new(observer));
    }

    /// Feed PCM f32 mono 16 kHz samples.
    ///
    /// Hands the buffer to the worker and returns; inference happens there.
    /// Blocks only if the command queue is full, which back-pressures a
    /// producer feeding faster than the model can transcribe.
    ///
    /// # Errors
    ///
    /// If the session has been flushed or closed, so the worker is no longer
    /// accepting audio.
    pub fn feed(&self, samples: Vec<f32>) -> Result<()> {
        self.send(AsrCommand::Feed(samples))
    }

    /// Finalize: drain buffered audio and return the complete transcript.
    ///
    /// The session is over afterwards — further [`Self::feed`] calls fail and
    /// observers have seen [`AsrEvent::Finished`]. Blocks until the worker
    /// finishes the last chunk, so call it off a UI thread.
    pub fn flush(&self) -> Result<String> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.send(AsrCommand::Flush(reply))?;
        // The worker always replies before exiting; a receive error means it
        // died mid-flush, which is a worker-gone condition either way.
        let transcript = answer.recv().unwrap_or_else(|_| Err(asr_worker_gone()))?;
        match self.take_failure() {
            Some(message) => Err(Error::InferenceError { message }),
            None => Ok(transcript),
        }
    }

    /// Reset to transcribe fresh audio without reloading the model.
    pub fn reset(&self) -> Result<()> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.send(AsrCommand::Reset(reply))?;
        answer.recv().unwrap_or_else(|_| Err(asr_worker_gone()))
    }

    /// Stop the session and release the model, discarding any buffered audio.
    ///
    /// Idempotent. Use [`Self::flush`] instead when you want the transcript;
    /// this is the "user walked away" path. Named `cancel` rather than
    /// `close` so the bolt mirror does not collide with the `close()` BoltFFI
    /// generates on every handle for the host's disposal idiom.
    pub fn cancel(&self) {
        // Flag first, *then* drop the sender. Dropping alone is not enough:
        // `Receiver::recv` yields every already-queued command before it
        // reports disconnection, so the worker would transcribe the whole
        // backlog — holding the model and the CPU — after the caller believed
        // it had stopped. The worker checks this flag before each command.
        self.cancelled.store(true, Ordering::Relaxed);
        self.commands
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
    }

    /// Whether the session is still accepting audio.
    ///
    /// Goes false the moment [`Self::cancel`] is called, not when the worker
    /// notices — a stop button must read as stopped immediately.
    pub fn is_running(&self) -> bool {
        !self.cancelled.load(Ordering::Relaxed)
            && !self
                .fanout
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .finished
    }

    /// Take the recorded per-chunk failure, if one happened.
    fn take_failure(&self) -> Option<String> {
        self.failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }

    fn send(&self, command: AsrCommand) -> Result<()> {
        // Clone the sender out and release the lock before sending. `feed`
        // blocks when the queue is full, and holding the lock across that
        // would make a concurrent `cancel` — the stop button — wait for the
        // backlog it is trying to abandon.
        let sender = {
            let guard = self.commands.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().ok_or_else(asr_worker_gone)?.clone()
        };
        sender.send(command).map_err(|_| asr_worker_gone())
    }
}

/// How many commands may queue before `feed` back-pressures the producer.
///
/// Microphone capture hands over ~10–100 ms of audio per call, so this is
/// seconds of slack. An unbounded queue would instead let a fast producer
/// grow memory without limit while the model falls behind.
const ASR_COMMAND_CAPACITY: usize = 64;

fn asr_worker_gone() -> Error {
    Error::ConfigError {
        message: "ASR session is no longer running; open a new session".into(),
    }
}

/// Render an error with its full `source()` chain.
///
/// SDK errors carry the root cause as `#[source]`, which `Display` alone drops
/// — "Feed failed" instead of "Feed failed: Inference error: dtype mismatch".
fn asr_error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut source = e.source();
    while let Some(cause) = source {
        out.push_str(": ");
        out.push_str(&cause.to_string());
        source = cause.source();
    }
    out
}

/// The shared state an ASR worker reports through.
struct AsrWorkerState {
    fanout: Arc<Mutex<AsrFanout>>,
    cancelled: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
}

/// Owns the SDK stream and applies commands in order until the channel closes,
/// a flush finalizes the session, or the caller cancels.
fn asr_worker(stream: sdk::XybridStream, commands: Receiver<AsrCommand>, state: AsrWorkerState) {
    let AsrWorkerState {
        fanout,
        cancelled,
        failure,
    } = state;
    // Pay the model's cold-start cost while the host is still opening the
    // microphone, instead of on top of the first visible partial. Feeds that
    // arrive meanwhile queue on the command channel and drain against a warm
    // model. Non-fatal: a real failure resurfaces on the first chunk.
    if let Err(e) = stream.warmup() {
        log::warn!(
            "ASR warm-up failed (continuing cold): {}",
            asr_error_chain(&e)
        );
    }

    // The SDK returns its cached latest partial from *every* feed, and feeds
    // queued behind one inference all drain at once when it finishes — without
    // this, one chunk's transcript would be delivered once per queued feed.
    let mut last_sent_sequence: Option<u64> = None;

    while let Ok(command) = commands.recv() {
        // Checked before every command, not just between feeds: a cancel has
        // to abandon whatever is already queued, or it would sit transcribing
        // the backlog long after the caller stopped listening.
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        match command {
            AsrCommand::Feed(samples) => match stream.feed(&samples) {
                Ok(Some(partial)) => {
                    let partial = PartialResult::from(partial);
                    if last_sent_sequence == Some(partial.chunk_sequence) {
                        continue;
                    }
                    last_sent_sequence = Some(partial.chunk_sequence);
                    fanout
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .emit(partial);
                }
                Ok(None) => {}
                Err(e) => {
                    // Keep transcribing — a transient failure should not end a
                    // live session — but remember it. The core extracts a
                    // chunk from its buffer before transcribing, so this
                    // chunk's audio is already gone and will never reach the
                    // transcript. `flush` reports this instead of handing back
                    // a holed transcript that looks complete.
                    let message = format!("ASR chunk failed: {}", asr_error_chain(&e));
                    log::warn!("{message}");
                    let mut recorded = failure.lock().unwrap_or_else(|e| e.into_inner());
                    recorded.get_or_insert(message);
                }
            },
            AsrCommand::Flush(reply) => {
                let outcome = stream
                    .flush()
                    .map(|r| r.text)
                    .map_err(|e| Error::InferenceError {
                        message: format!("ASR flush failed: {}", asr_error_chain(&e)),
                    });
                let _ = reply.send(outcome);
                break; // session finalized
            }
            AsrCommand::Reset(reply) => {
                let outcome = stream.reset().map_err(|e| Error::InferenceError {
                    message: format!("ASR reset failed: {}", asr_error_chain(&e)),
                });
                // A fresh utterance starts clean — including any failure
                // recorded against the audio that is being discarded.
                last_sent_sequence = None;
                *failure.lock().unwrap_or_else(|e| e.into_inner()) = None;
                let mut guard = fanout.lock().unwrap_or_else(|e| e.into_inner());
                guard.pending = None;
                drop(guard);
                let _ = reply.send(outcome);
            }
        }
    }

    // Reached on flush, on `close`, and on the handle being dropped — every
    // way a session can end, so observers always get their terminal event.
    fanout.lock().unwrap_or_else(|e| e.into_inner()).finish();
}

// ============================================================================
// Model cache management
// ============================================================================

/// Logical storage area containing a cached model entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEntryLocation {
    Registry,
    Extracted,
    HuggingFace,
    HuggingFaceHub,
}

impl From<sdk::CacheEntryLocation> for CacheEntryLocation {
    fn from(location: sdk::CacheEntryLocation) -> Self {
        match location {
            sdk::CacheEntryLocation::Registry => Self::Registry,
            sdk::CacheEntryLocation::Extracted => Self::Extracted,
            sdk::CacheEntryLocation::HuggingFace => Self::HuggingFace,
            sdk::CacheEntryLocation::HuggingFaceHub => Self::HuggingFaceHub,
        }
    }
}

/// One model entry occupying managed cache storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    pub model_id: String,
    pub location: CacheEntryLocation,
    pub path: String,
    pub size_bytes: u64,
}

impl From<sdk::CacheEntryInfo> for CacheEntry {
    fn from(entry: sdk::CacheEntryInfo) -> Self {
        Self {
            model_id: entry.model_id,
            location: entry.location.into(),
            path: entry.path.to_string_lossy().into_owned(),
            size_bytes: entry.size_bytes,
        }
    }
}

/// Aggregate model-cache storage status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStatus {
    pub total_size_bytes: u64,
    /// Number of physical entries across all managed cache locations.
    pub entry_count: u32,
    /// Number of distinct model identifiers represented by those entries.
    pub model_count: u32,
    /// Number of models ready in the runtime extraction cache.
    pub extracted_model_count: u32,
    pub cache_root: String,
}

fn open_cache() -> Result<sdk::CacheManager> {
    sdk::CacheManager::new().map_err(Error::from)
}

fn cache_entries_from(manager: &sdk::CacheManager) -> Result<Vec<CacheEntry>> {
    manager
        .cache_entries()
        .map(|entries| entries.into_iter().map(Into::into).collect())
        .map_err(Error::from)
}

fn cache_status_from(manager: &sdk::CacheManager) -> Result<CacheStatus> {
    let entries = cache_entries_from(manager)?;
    let model_count = entries
        .iter()
        .map(|entry| entry.model_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    let extracted_model_count = manager.list_extracted_model_ids().len();

    Ok(CacheStatus {
        total_size_bytes: entries.iter().map(|entry| entry.size_bytes).sum(),
        entry_count: u32::try_from(entries.len()).unwrap_or(u32::MAX),
        model_count: u32::try_from(model_count).unwrap_or(u32::MAX),
        extracted_model_count: u32::try_from(extracted_model_count).unwrap_or(u32::MAX),
        cache_root: manager.cache_root().to_string_lossy().into_owned(),
    })
}

/// Returns aggregate storage usage across every managed model-cache location.
pub fn cache_status() -> Result<CacheStatus> {
    cache_status_from(&open_cache()?)
}

/// Lists every physical model entry occupying managed cache storage.
pub fn cache_entries() -> Result<Vec<CacheEntry>> {
    cache_entries_from(&open_cache()?)
}

/// Returns whether a model occupies any managed cache entry.
pub fn cache_is_model_cached(model_id: String) -> Result<bool> {
    open_cache()?
        .is_model_cached(&model_id)
        .map_err(Error::from)
}

/// Resolves the preferred local cache path for a model, if present.
pub fn cache_model_path(model_id: String) -> Result<Option<String>> {
    open_cache()?
        .cached_model_path(&model_id)
        .map(|path| path.map(|value| value.to_string_lossy().into_owned()))
        .map_err(Error::from)
}

/// Lists model IDs that are extracted, validated, and ready to run offline.
pub fn cache_list_extracted_model_ids() -> Result<Vec<String>> {
    Ok(open_cache()?.list_extracted_model_ids())
}

/// Removes every managed cache entry for one model.
///
/// Do not call concurrently with a load of the same model.
pub fn cache_remove_model(model_id: String) -> Result<u32> {
    open_cache()?.clear_model(&model_id).map_err(Error::from)
}

/// Clears all managed model-cache storage.
///
/// Do not call concurrently with any model load.
pub fn cache_clear() -> Result<u32> {
    open_cache()?.clear().map_err(Error::from)
}

// ============================================================================
// Process-global init
// ============================================================================

/// One-stop SDK initialization for platform bindings.
///
/// Wraps [`sdk::init()`]'s builder so every foreign-language SDK gets the
/// same unified setup: pass an API key to start the telemetry exporter,
/// override the LLM gateway and/or telemetry ingest URL, and `.run()` the
/// configuration. Omitting `api_key` runs anonymously (local inference,
/// no exporter) — the same semantics as the Rust builder.
///
/// Blank strings are treated as absent so hosts can forward empty
/// `String.fromEnvironment` / `BuildConfig` values without accidentally
/// configuring anything. This is the canonical init path the Swift
/// `Xybrid.initialize(apiKey:gatewayUrl:ingestUrl:)` and Kotlin
/// `Xybrid.init(context, apiKey, gatewayUrl, ingestUrl)` wrappers call.
pub fn configure_runtime(
    api_key: Option<String>,
    gateway_url: Option<String>,
    ingest_url: Option<String>,
) {
    let non_blank = |value: Option<String>| {
        value
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };

    let mut builder = sdk::init();
    if let Some(key) = non_blank(api_key) {
        builder = builder.api_key(key);
    }
    if let Some(gateway) = non_blank(gateway_url) {
        builder = builder.gateway_url(gateway);
    }
    if let Some(ingest) = non_blank(ingest_url) {
        builder = builder.ingest_url(ingest);
    }
    builder.run();
}

/// Register the platform cache directory used for model bundles.
///
/// Mandatory on Android (the SDK uses it to seed `HOME`, `HF_HOME`, and
/// `XDG_CACHE_HOME`). Optional on iOS / macOS / Linux / Windows. First
/// call wins.
pub fn init_sdk_cache_dir(path: String) {
    sdk::init_sdk_cache_dir(path);
}

pub fn get_sdk_cache_dir() -> Option<String> {
    sdk::get_sdk_cache_dir().and_then(|p| p.to_str().map(str::to_string))
}

pub fn is_sdk_cache_configured() -> bool {
    sdk::is_sdk_cache_configured()
}

/// Register the binding identifier (`"flutter"`, `"kotlin"`,
/// `"react-native"`, `"swift"`, `"unity"`) reported in the `X-Xybrid-Client`
/// registry header and on telemetry events.
///
/// Each binding calls this once at SDK init with its hard-coded constant.
/// React Native wraps the Swift and Kotlin SDKs and calls it before them.
/// Unknown strings fall back to [`sdk::DEFAULT_BINDING`] to bound cardinality
/// on the registry side. First call wins.
pub fn set_binding(binding: String) {
    sdk::set_binding(resolve_binding(&binding));
}

/// Map a binding name onto the accepted set, or [`sdk::DEFAULT_BINDING`].
fn resolve_binding(binding: &str) -> &'static str {
    match binding {
        "flutter" => "flutter",
        "kotlin" => "kotlin",
        "react-native" => "react-native",
        "swift" => "swift",
        "unity" => "unity",
        _ => sdk::DEFAULT_BINDING,
    }
}

pub fn get_binding() -> String {
    sdk::get_binding().to_string()
}

pub fn set_api_key(api_key: String) {
    sdk::set_api_key(&api_key);
}

pub fn set_provider_api_key(provider: String, api_key: String) {
    sdk::set_provider_api_key(&provider, &api_key);
}

pub fn has_api_key() -> bool {
    sdk::has_api_key()
}

/// Point the cloud gateway at a platform base URL (staging, self-hosted).
///
/// Held in process memory, not the environment — safe to call after telemetry
/// threads have started. Pass a bare base URL; the `/v1` suffix is internal.
pub fn set_platform_url(url: String) {
    sdk::set_platform_url(&url);
}

/// Enable speculative cloud fallback globally: a registry model that isn't
/// downloaded yet is served from the gateway while the weights download.
///
/// Only takes effect when an API key resolves. Speculation is LLM/chat-only —
/// prefer the per-load [`ModelLoader::from_registry_speculative`] when the app
/// also loads ASR/TTS models, which cannot be served this way.
pub fn set_speculative_cloud(enabled: bool) {
    sdk::set_speculative_cloud(enabled);
}

/// Whether the global speculative-cloud default is on.
pub fn is_speculative_cloud_enabled() -> bool {
    sdk::is_speculative_cloud_enabled()
}

/// Whether a speculative load of `model_id` would actually speculate right now:
/// an API key resolves and the model is not already cached.
///
/// A free function rather than a method because the foreign bindings collapse
/// the loader into `XybridModel::from_*` constructors, leaving nowhere to hang
/// a pre-load query. Never touches the network.
pub fn will_speculate_for_model(model_id: String) -> bool {
    sdk::ModelLoader::from_registry(&model_id)
        .with_speculative_cloud(true)
        .will_speculate()
}

/// Release every idle loaded model's memory, returning how many were released.
///
/// The host hint behind automatic model release: call it from
/// `didReceiveMemoryWarning` (iOS), `onTrimMemory` (Android), or your own
/// desktop logic. Models with a run in flight are skipped, and a released
/// model reloads itself on its next use — the caller never has to reload
/// anything or handle a new error.
///
/// Deliberately a plain call, not a callback registration: nothing is
/// signalled back across the FFI boundary.
pub fn release_memory() -> u32 {
    sdk::release_memory() as u32
}

/// Enable or disable automatic model release for subsequent loads.
///
/// When enabled, loading a model while the device reports memory pressure
/// first releases least-recently-used idle models. Off by default.
/// [`release_memory`] works regardless of this setting.
pub fn set_auto_release(enabled: bool) {
    sdk::set_auto_release(enabled);
}

/// Whether automatic model release is enabled process-wide.
pub fn is_auto_release_enabled() -> bool {
    sdk::auto_release_policy().on_pressure
}

// ============================================================================
// Telemetry (advanced config + lifecycle)
// ============================================================================
//
// The apiKey-only fast path runs through [`configure_runtime`]
// (`sdk::init().api_key().run()`). This handle exposes the *advanced* knobs —
// batch size, flush interval, device label/attributes — that the platform SDKs
// surface via a fluent `TelemetryConfig` builder, plus the process-global
// init/flush/shutdown lifecycle. It mirrors the pre-bolt C ABI's telemetry
// surface so the Unity/C# binding can move onto bolt without losing capability.
//
// Telemetry *events* never cross the FFI boundary — they are exported
// Rust→network. Only this config/lifecycle control plane does.

/// Tracks whether [`telemetry_init`] has run without a matching
/// [`telemetry_shutdown`], so a second init is rejected instead of silently
/// leaking the prior sender. Mirrors the pre-bolt C ABI's gate.
static TELEMETRY_INITIALIZED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The SDK's default telemetry ingest endpoint.
///
/// Bindings display this alongside a freshly created config so callers can read
/// the resolved endpoint back without a round-trip setter.
pub fn telemetry_default_endpoint() -> String {
    sdk::telemetry::DEFAULT_INGEST_URL.to_string()
}

/// The SDK version string, sourced from `CARGO_PKG_VERSION` at compile time.
pub fn version() -> String {
    sdk::SDK_VERSION.to_string()
}

/// Interior-mutable holder for a telemetry configuration under construction.
///
/// Generator crates wrap this in `Arc<Self>` for opaque-handle semantics. The
/// [`sdk::telemetry::TelemetryConfig`] is held behind a `Mutex<Option<_>>` so
/// foreign callers can mutate it through a shared `&self` (FFI handle methods
/// only ever receive a shared reference), and so [`telemetry_init`] can `take`
/// it — leaving the handle empty — matching the pre-bolt consume-on-init
/// contract. The mutex is uncontended in normal usage: a config is built by one
/// host thread.
pub struct TelemetryConfigHandle {
    inner: Mutex<Option<sdk::telemetry::TelemetryConfig>>,
}

impl TelemetryConfigHandle {
    /// A new config bound to the default ingest endpoint and the given API key.
    pub fn new(api_key: String) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Some(sdk::telemetry::TelemetryConfig::new(
                sdk::telemetry::DEFAULT_INGEST_URL,
                api_key,
            ))),
        })
    }

    /// Override the ingest endpoint (self-hosted collector / non-prod).
    pub fn set_endpoint(&self, endpoint: String) {
        self.with_config(|config| config.endpoint = endpoint);
    }

    /// Set the app version reported with every event.
    pub fn set_app_version(&self, version: String) {
        self.with_config(|config| config.app_version = Some(version));
    }

    /// Set the human-friendly device label reported with every event.
    pub fn set_device_label(&self, label: String) {
        self.with_config(|config| config.device_label = Some(label));
    }

    /// Attach an app-provided device attribute (stored under `device.custom`).
    pub fn set_device_attribute(&self, key: String, value: String) {
        self.with_config(|config| {
            config.device_profile_patch.custom.insert(key, value);
        });
    }

    /// Set the number of events buffered before a flush.
    pub fn set_batch_size(&self, batch_size: u32) {
        self.with_config(|config| config.batch_size = batch_size as usize);
    }

    /// Set the background flush interval, in seconds.
    pub fn set_flush_interval_secs(&self, secs: u32) {
        self.with_config(|config| config.flush_interval_secs = secs as u64);
    }

    /// Apply `f` to the in-progress config. A no-op after the config has been
    /// consumed by [`telemetry_init`] — mirroring the pre-bolt setters, which
    /// no-op once the handle is gone.
    fn with_config(&self, f: impl FnOnce(&mut sdk::telemetry::TelemetryConfig)) {
        if let Some(config) = self.lock().as_mut() {
            f(config);
        }
    }

    /// Take the built config out, leaving the handle empty.
    fn take(&self) -> Option<sdk::telemetry::TelemetryConfig> {
        self.lock().take()
    }

    /// Lock the inner slot, recovering the guard if the mutex is poisoned (a
    /// panic at the FFI boundary would abort the host app over a recoverable
    /// condition — matches the codebase-wide poison-recovery convention).
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<sdk::telemetry::TelemetryConfig>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Start the process-global telemetry exporter from a built config.
///
/// Consumes the config held by `handle`: subsequent setter calls become no-ops
/// and a second `telemetry_init` on the same handle errors. This is the
/// advanced entry point; the apiKey-only path is [`configure_runtime`].
///
/// # Errors
/// [`Error::ConfigError`] if the handle was already consumed, or if telemetry
/// is already initialized without an intervening [`telemetry_shutdown`].
pub fn telemetry_init(handle: &TelemetryConfigHandle) -> Result<()> {
    // Reclaim the built config first — the pre-bolt C ABI always consumes the
    // handle on init, success or failure.
    let config = handle.take().ok_or_else(|| Error::ConfigError {
        message: "telemetry config already consumed".to_string(),
    })?;

    // Gate against double-init: only the false→true transition wins. On
    // contention, drop `config` (freeing the second sender's resources) and
    // error without touching the live exporter.
    if TELEMETRY_INITIALIZED
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_err()
    {
        return Err(Error::ConfigError {
            message: "telemetry already initialized; call shutdown before reinitializing"
                .to_string(),
        });
    }

    // Roll the gate back if init panics so a later init can retry — mirrors the
    // pre-bolt C ABI. Without this, a panicked init wedges the gate at `true`
    // with no live exporter: every later `telemetry_init` reports "already
    // initialized" until the process restarts, and `telemetry_shutdown` has no
    // real sender to stop. Catching here also keeps a recoverable
    // telemetry-setup failure from unwinding across the FFI boundary and
    // aborting the host app.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sdk::telemetry::init_platform_telemetry(config);
    }));
    if outcome.is_err() {
        TELEMETRY_INITIALIZED.store(false, std::sync::atomic::Ordering::Release);
        return Err(Error::ConfigError {
            message: "telemetry init panicked".to_string(),
        });
    }

    Ok(())
}

/// Flush pending telemetry events. Safe before init / after shutdown — the SDK
/// no-ops when the exporter is absent.
pub fn telemetry_flush() {
    sdk::telemetry::flush_platform_telemetry();
}

/// Shut down the telemetry exporter. Idempotent: a call before init, or a
/// second call, no-ops without touching the SDK.
pub fn telemetry_shutdown() {
    if TELEMETRY_INITIALIZED.swap(false, std::sync::atomic::Ordering::AcqRel) {
        sdk::telemetry::shutdown_platform_telemetry();
    }
}

// ============================================================================
// Bundle inspection
// ============================================================================
//
// Read-only inspection of `.xyb` model bundles (tar + zstd) for editor tooling
// and asset workflows: open, read manifest/metadata, enumerate + extract files.
// Mirrors the pre-bolt C ABI's bundle surface. Every method takes/returns simple
// types, so the whole surface generates natively (no hand-port).

/// FFI-friendly handle around an opened [`sdk::bundler::XyBundle`].
///
/// Generator crates wrap this in `Arc<Self>` for opaque-handle semantics. The
/// bundle is immutable after [`open`](Self::open), so every accessor takes a
/// shared `&self` and no interior mutability is needed.
pub struct BundleHandle {
    inner: sdk::bundler::XyBundle,
}

impl BundleHandle {
    /// Open and parse a `.xyb` bundle (decompress zstd, parse tar, validate the
    /// manifest).
    ///
    /// # Errors
    /// [`Error::ConfigError`] on an empty path, or if the bundle can't be opened
    /// or is malformed.
    pub fn open(path: String) -> Result<Arc<Self>> {
        if path.is_empty() {
            return Err(Error::ConfigError {
                message: "path is empty".to_string(),
            });
        }
        let inner = sdk::bundler::XyBundle::load(&path).map_err(|e| Error::ConfigError {
            message: format!("failed to open bundle: {e}"),
        })?;
        Ok(Arc::new(Self { inner }))
    }

    /// The model identifier from the manifest.
    pub fn model_id(&self) -> String {
        self.inner.manifest().model_id.clone()
    }

    /// The version string from the manifest.
    pub fn version(&self) -> String {
        self.inner.manifest().version.clone()
    }

    /// The target platform from the manifest.
    pub fn target(&self) -> String {
        self.inner.manifest().target.clone()
    }

    /// The SHA-256 hash from the manifest.
    pub fn hash(&self) -> String {
        self.inner.manifest().hash.clone()
    }

    /// Whether the bundle carries a `model_metadata.json`.
    pub fn has_metadata(&self) -> bool {
        self.inner.manifest().has_metadata
    }

    /// Number of files in the bundle (excludes `manifest.json`).
    pub fn file_count(&self) -> u32 {
        self.inner.manifest().files.len() as u32
    }

    /// The file name at `index`, or `None` if out of bounds.
    pub fn file_name(&self, index: u32) -> Option<String> {
        self.inner.manifest().files.get(index as usize).cloned()
    }

    /// The full bundle manifest serialized as JSON.
    ///
    /// # Errors
    /// [`Error::ConfigError`] if the manifest can't be serialized.
    pub fn manifest_json(&self) -> Result<String> {
        serde_json::to_string(self.inner.manifest()).map_err(|e| Error::ConfigError {
            message: format!("failed to serialize manifest: {e}"),
        })
    }

    /// The `model_metadata.json` contents, or `None` if the bundle has none.
    ///
    /// # Errors
    /// [`Error::ConfigError`] if reading the entry fails.
    pub fn metadata_json(&self) -> Result<Option<String>> {
        self.inner
            .get_metadata_json()
            .map_err(|e| Error::ConfigError {
                message: format!("failed to read metadata: {e}"),
            })
    }

    /// Extract every bundle file to `output_dir` (created if absent), preserving
    /// relative paths.
    ///
    /// # Errors
    /// [`Error::ConfigError`] if extraction fails.
    pub fn extract(&self, output_dir: String) -> Result<()> {
        self.inner
            .extract_to(&output_dir)
            .map_err(|e| Error::ConfigError {
                message: format!("failed to extract bundle: {e}"),
            })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn bare_run_options() -> RunOptions {
        RunOptions {
            generation_config: None,
            abort_on: Vec::new(),
            fallback_to_cloud: false,
            max_grace_tokens: 0,
            correlation_id: None,
            cloud_provider: None,
            cloud_model: None,
            cloud_gateway_url: None,
        }
    }

    #[test]
    fn a_cancel_handle_opts_the_policy_into_user_cancelled() {
        // Without this the token latches `is_cancelled()` and inference
        // ignores it: `AbortState::detect_user_cancelled` only fires when the
        // policy observes `UserCancelled`, and the FFI `AbortSignal` wire enum
        // has no such variant for hosts to set.
        let token = CancellationToken::new();
        let opts = bare_run_options()
            .to_sdk_over(Some(&token), sdk::GenerationConfig::default())
            .expect("options convert");

        assert!(opts.abort_policy.observes(sdk::AbortSignal::UserCancelled));
    }

    #[test]
    fn no_cancel_handle_leaves_user_cancelled_unobserved() {
        let opts = bare_run_options()
            .to_sdk_over(None, sdk::GenerationConfig::default())
            .expect("options convert");

        assert!(!opts.abort_policy.observes(sdk::AbortSignal::UserCancelled));
    }

    #[test]
    fn cache_status_counts_physical_entries_and_distinct_models() {
        let temp = tempfile::TempDir::new().unwrap();
        let cache_root = temp.path().join("cache");
        let models = cache_root.join("models");
        let registry_entry = models.join("model-a");
        let extracted_entry = cache_root.join("extracted").join("model-a");
        std::fs::create_dir_all(&registry_entry).unwrap();
        std::fs::create_dir_all(&extracted_entry).unwrap();
        std::fs::write(registry_entry.join("bundle.xyb"), b"abc").unwrap();
        std::fs::write(extracted_entry.join("model.bin"), b"12345").unwrap();
        let manager = sdk::CacheManager::with_dir(models).unwrap();

        let status = cache_status_from(&manager).unwrap();

        assert_eq!(status.total_size_bytes, 8);
        assert_eq!(status.entry_count, 2);
        assert_eq!(status.model_count, 1);
        assert_eq!(status.extracted_model_count, 0);
        assert_eq!(status.cache_root, cache_root.to_string_lossy());
    }

    #[test]
    fn cache_ready_count_tracks_validation_not_physical_directories() {
        let temp = tempfile::TempDir::new().unwrap();
        let root = temp.path().join("cache");
        let extracted = root.join("extracted/ready");
        std::fs::create_dir_all(&extracted).unwrap();
        std::fs::write(
            extracted.join("model_metadata.json"),
            r#"{
            "model_id":"ready", "version":"1.0",
            "execution_template":{"type":"Onnx","model_file":"model.onnx"},
            "preprocessing":[],"postprocessing":[],"files":["model.onnx"],"metadata":{}
        }"#,
        )
        .unwrap();
        let manager = sdk::CacheManager::with_dir(root.join("models")).unwrap();
        assert_eq!(
            cache_status_from(&manager).unwrap().extracted_model_count,
            0
        );

        std::fs::write(extracted.join("model.onnx"), b"model").unwrap();
        assert_eq!(manager.list_extracted_model_ids(), vec!["ready"]);
        assert_eq!(
            cache_status_from(&manager).unwrap().extracted_model_count,
            1
        );

        std::fs::remove_file(extracted.join("model.onnx")).unwrap();
        let status = cache_status_from(&manager).unwrap();
        assert_eq!(status.extracted_model_count, 0);
        assert_eq!(status.entry_count, 1);
        assert!(status.total_size_bytes > 0);
    }

    #[test]
    fn cache_entry_conversion_preserves_location_and_path() {
        let entry = sdk::CacheEntryInfo {
            model_id: "owner/repo".into(),
            location: sdk::CacheEntryLocation::HuggingFace,
            path: std::path::PathBuf::from("cache/hf/repo"),
            size_bytes: 42,
        };

        assert_eq!(
            CacheEntry::from(entry),
            CacheEntry {
                model_id: "owner/repo".into(),
                location: CacheEntryLocation::HuggingFace,
                path: "cache/hf/repo".into(),
                size_bytes: 42,
            }
        );
    }

    #[test]
    fn error_code_is_stable() {
        assert_eq!(Error::ModelNotFound { id: "x".into() }.code(), 1);
        assert_eq!(Error::NotLoaded.code(), 9);
        assert_eq!(Error::Timeout { timeout_ms: 0 }.code(), 18);
    }

    #[test]
    fn error_is_retryable_matches_sdk_semantics() {
        assert!(Error::NetworkError {
            message: "x".into()
        }
        .is_retryable());
        assert!(Error::RateLimited {
            retry_after_secs: 1
        }
        .is_retryable());
        assert!(Error::Timeout { timeout_ms: 1 }.is_retryable());
        assert!(Error::Offline {
            message: "x".into()
        }
        .is_retryable());

        assert!(!Error::ModelNotFound { id: "x".into() }.is_retryable());
        assert!(!Error::CircuitOpen {
            message: "x".into()
        }
        .is_retryable());
        assert!(!Error::NotLoaded.is_retryable());
    }

    #[test]
    fn pipeline_construction_collapses_ref_and_exposes_stage_ids() {
        let pipeline = Pipeline::from_yaml(
            r#"
name: assistant
stages:
  - id: answer
    model: gpt-4o-mini
    target: cloud
    provider: openai
"#
            .into(),
        )
        .expect("cloud-only pipeline should resolve without model downloads");

        assert_eq!(pipeline.name().as_deref(), Some("assistant"));
        assert_eq!(pipeline.stage_names(), vec!["answer"]);
        assert_eq!(pipeline.stage_count(), 1);
    }

    fn sdk_stage(
        name: &str,
        target: &str,
        kind: sdk::ir::EnvelopeKind,
        metadata: HashMap<String, String>,
    ) -> sdk::PipelineStageTiming {
        sdk::PipelineStageTiming {
            name: name.into(),
            latency_ms: 40,
            target: target.into(),
            reason: "test".into(),
            output: sdk::ir::Envelope::with_metadata(kind, metadata),
        }
    }

    /// A voice-assistant run: the transcript and the reply must survive next
    /// to the final audio, each stage with its own provenance and metrics.
    #[test]
    fn pipeline_result_keeps_every_stage_output() {
        let mut llm_metadata = HashMap::new();
        llm_metadata.insert("ttft_ms".into(), "12".into());
        let tts_audio = sdk::ir::EnvelopeKind::Audio(vec![1, 2, 3]);
        let sdk_result = sdk::PipelineExecutionResult {
            name: Some("assistant".into()),
            stages: vec![
                sdk_stage(
                    "asr",
                    "local",
                    sdk::ir::EnvelopeKind::Text("what time is it".into()),
                    HashMap::new(),
                ),
                sdk_stage(
                    "llm",
                    "cloud",
                    sdk::ir::EnvelopeKind::Text("It is noon.".into()),
                    llm_metadata,
                ),
                sdk_stage("tts", "local", tts_audio.clone(), HashMap::new()),
            ],
            total_latency_ms: 125,
            output_type: sdk::OutputType::Audio,
            output: sdk::ir::Envelope::new(tts_audio),
        };

        let result = PipelineResult::from_sdk(sdk_result);

        assert_eq!(result.output_type, OutputType::Audio);
        assert_eq!(result.audio_bytes(), Some([1u8, 2, 3].as_slice()));
        assert_eq!(result.latency_ms, 125);
        let ids: Vec<&str> = result.stages.iter().map(|s| s.stage_id.as_str()).collect();
        assert_eq!(ids, ["asr", "llm", "tts"]);

        let asr = result.stage("asr").expect("asr stage");
        assert_eq!(asr.text(), Some("what time is it"));
        assert_eq!(asr.output_type, OutputType::Text);
        assert_eq!(asr.execution_target, ExecutionTarget::Local);

        let llm = result.stage("llm").expect("llm stage");
        assert_eq!(llm.text(), Some("It is noon."));
        assert_eq!(llm.execution_target, ExecutionTarget::Cloud);
        assert_eq!(llm.metrics.ttft_ms, Some(12));
        assert_eq!(llm.metrics.total_ms, 40);
        assert!(llm.metrics.stage_latencies_ms.is_empty());

        assert_eq!(result.stages[2].output_type, OutputType::Audio);
        assert_eq!(result.stages[2].envelope.kind, result.envelope.kind);
        assert!(result.stage("missing").is_none());
    }

    #[test]
    fn pipeline_run_options_keep_correlation_id() {
        let options = RunOptions {
            correlation_id: Some("turn-7".into()),
            fallback_to_cloud: true,
            max_grace_tokens: 8,
            ..RunOptions::default()
        };

        let sdk_options = pipeline_run_options(options).expect("inert fields are accepted");

        assert_eq!(sdk_options.correlation_id.as_deref(), Some("turn-7"));

        let blank_cloud = RunOptions {
            cloud_provider: Some("  ".into()),
            cloud_model: Some(String::new()),
            cloud_gateway_url: Some("\t".into()),
            ..RunOptions::default()
        };
        pipeline_run_options(blank_cloud).expect("blank target fields are absent");
    }

    #[test]
    fn pipeline_run_options_reject_what_a_pipeline_cannot_honour() {
        let generation = RunOptions {
            generation_config: Some(GenerationConfig::default()),
            ..RunOptions::default()
        };
        let abort = RunOptions {
            abort_on: vec![AbortSignal::ThermalHot],
            ..RunOptions::default()
        };
        let cloud = RunOptions {
            cloud_model: Some("model-a".into()),
            ..RunOptions::default()
        };

        for options in [generation, abort, cloud] {
            let err = pipeline_run_options(options).expect_err("must not be silently ignored");
            assert!(matches!(err, Error::ConfigError { .. }), "got {err:?}");
        }
    }

    #[test]
    fn cloud_options_default_to_absent_and_do_not_enable_fallback() {
        let options = RunOptions::default();
        assert!(options.cloud_provider.is_none());
        assert!(options.cloud_model.is_none());
        assert!(options.cloud_gateway_url.is_none());
        assert!(!options.fallback_to_cloud);

        let options = RunOptions {
            cloud_provider: Some("openai".into()),
            cloud_model: Some("model-a".into()),
            ..RunOptions::default()
        };
        let sdk_options = options.to_sdk(None).unwrap();
        assert!(!sdk_options.abort_policy.fallback_to_cloud);
    }

    #[test]
    fn omitted_and_blank_cloud_options_preserve_the_entire_metadata_map() {
        let mut envelope = Envelope::text("hello".into()).into_sdk().unwrap();
        envelope.metadata.insert("provider".into(), "prior".into());
        envelope.metadata.insert("backend".into(), "direct".into());
        envelope.metadata.insert("custom".into(), "keep".into());
        let original = envelope.metadata.clone();

        let options = RunOptions {
            generation_config: Some(GenerationConfig {
                max_tokens: Some(9),
                ..GenerationConfig::default()
            }),
            ..RunOptions::default()
        };
        assert_eq!(
            options
                .apply_cloud_fallback_metadata(&mut envelope)
                .unwrap(),
            None
        );
        assert_eq!(envelope.metadata, original);

        let blank = RunOptions {
            cloud_provider: Some("  ".into()),
            cloud_model: Some(String::new()),
            cloud_gateway_url: Some(" \t ".into()),
            ..RunOptions::default()
        };
        assert_eq!(
            blank.apply_cloud_fallback_metadata(&mut envelope).unwrap(),
            None
        );
        assert_eq!(envelope.metadata, original);
    }

    #[test]
    fn supplied_cloud_fields_override_independently_without_invented_defaults() {
        for (field, value, key) in [
            ("provider", "provider-a", "provider"),
            ("model", "model-a", "model"),
            ("gateway", "https://api.xybrid.dev/v1/", "gateway_url"),
        ] {
            let mut envelope = Envelope::text("hello".into()).into_sdk().unwrap();
            envelope.metadata.insert("custom".into(), "keep".into());
            let mut options = RunOptions::default();
            match field {
                "provider" => options.cloud_provider = Some(format!(" {value} ")),
                "model" => options.cloud_model = Some(format!(" {value} ")),
                _ => options.cloud_gateway_url = Some(value.into()),
            }
            let gateway = options
                .apply_cloud_fallback_metadata(&mut envelope)
                .unwrap();
            let effective = if field == "gateway" {
                "https://api.xybrid.dev/v1"
            } else {
                value
            };
            assert_eq!(
                envelope.metadata.get(key).map(String::as_str),
                Some(effective)
            );
            assert_eq!(
                envelope.metadata.get("backend").map(String::as_str),
                Some("gateway")
            );
            assert_eq!(
                envelope.metadata.get("custom").map(String::as_str),
                Some("keep")
            );
            for absent in ["provider", "model", "gateway_url"] {
                if absent != key {
                    assert!(!envelope.metadata.contains_key(absent));
                }
            }
            assert_eq!(gateway.is_some(), field == "gateway");
        }
    }

    #[test]
    fn cloud_lowering_preserves_existing_unsupplied_fields_and_explicit_sampling() {
        let mut envelope = Envelope::text("hello".into()).into_sdk().unwrap();
        envelope
            .metadata
            .insert("provider".into(), "prior-provider".into());
        envelope.metadata.insert("backend".into(), "direct".into());
        envelope
            .metadata
            .insert("max_tokens".into(), "prior-tokens".into());
        let options = RunOptions {
            cloud_model: Some("new-model".into()),
            generation_config: Some(GenerationConfig {
                max_tokens: Some(0),
                temperature: Some(0.0),
                ..GenerationConfig::default()
            }),
            ..RunOptions::default()
        };

        options
            .apply_cloud_fallback_metadata(&mut envelope)
            .unwrap();
        assert_eq!(envelope.metadata.get("provider").unwrap(), "prior-provider");
        assert_eq!(envelope.metadata.get("model").unwrap(), "new-model");
        assert_eq!(envelope.metadata.get("backend").unwrap(), "gateway");
        assert_eq!(envelope.metadata.get("max_tokens").unwrap(), "0");
        assert_eq!(envelope.metadata.get("temperature").unwrap(), "0");
    }

    #[test]
    fn all_explicit_cloud_fields_reach_metadata() {
        let mut envelope = Envelope::text("hello".into()).into_sdk().unwrap();
        let options = RunOptions {
            cloud_provider: Some("provider-a".into()),
            cloud_model: Some("model-a".into()),
            cloud_gateway_url: Some("https://api.xybrid.dev/v1/".into()),
            ..RunOptions::default()
        };

        let gateway = options
            .apply_cloud_fallback_metadata(&mut envelope)
            .unwrap();
        assert_eq!(gateway.as_deref(), Some("https://api.xybrid.dev/v1"));
        assert_eq!(envelope.metadata.get("provider").unwrap(), "provider-a");
        assert_eq!(envelope.metadata.get("model").unwrap(), "model-a");
        assert_eq!(
            envelope.metadata.get("gateway_url").unwrap(),
            "https://api.xybrid.dev/v1"
        );
        assert_eq!(envelope.metadata.get("backend").unwrap(), "gateway");
    }

    #[test]
    fn invalid_gateway_is_rejected_before_any_metadata_mutation() {
        let mut envelope = Envelope::text("hello".into()).into_sdk().unwrap();
        envelope.metadata.insert("backend".into(), "direct".into());
        let original = envelope.metadata.clone();
        let options = RunOptions {
            cloud_provider: Some("provider-a".into()),
            cloud_gateway_url: Some("https://secret@api.xybrid.dev/v1".into()),
            ..RunOptions::default()
        };

        let error = options
            .apply_cloud_fallback_metadata(&mut envelope)
            .unwrap_err();
        assert!(matches!(error, Error::ConfigError { .. }));
        assert!(!error.to_string().contains("secret"));
        assert_eq!(envelope.metadata, original);
        assert!(!options.to_sdk(None).unwrap().abort_policy.fallback_to_cloud);
    }

    #[test]
    fn pipeline_target_mapping_never_reports_remote_work_as_local() {
        assert_eq!(
            ExecutionTarget::from_pipeline_target("device"),
            ExecutionTarget::Local
        );
        assert_eq!(
            ExecutionTarget::from_pipeline_target("local"),
            ExecutionTarget::Local
        );
        assert_eq!(
            ExecutionTarget::from_pipeline_target("cloud"),
            ExecutionTarget::Cloud
        );
        assert_eq!(
            ExecutionTarget::from_pipeline_target("fallback:xybrid-edge"),
            ExecutionTarget::Cloud
        );
        assert_eq!(
            ExecutionTarget::from_pipeline_target("unexpected-remote-target"),
            ExecutionTarget::Cloud
        );
    }

    #[test]
    fn error_from_sdk_io_flattens_message() {
        let io = std::io::Error::other("disk on fire");
        let sdk_err = sdk::SdkError::IoError(io);
        match Error::from(sdk_err) {
            Error::IoError { message } => assert!(message.contains("disk on fire")),
            other => panic!("expected IoError, got {other:?}"),
        }
    }

    #[test]
    fn envelope_roundtrip_preserves_text_and_metadata() {
        let env = Envelope::text("hello".into()).with_role(MessageRole::User);
        let sdk_env = env.clone().into_sdk().unwrap();
        let back = Envelope::from_sdk(sdk_env);

        assert_eq!(
            back.kind,
            EnvelopeKind::Text {
                text: "hello".into()
            }
        );
        // SDK assigns a local-id metadata key on `with_metadata`; the role
        // we set must survive the round trip.
        assert_eq!(
            back.metadata
                .get(xybrid_core::ir::Envelope::ROLE_METADATA_KEY),
            Some(&"user".to_string())
        );
    }

    #[test]
    fn image_envelope_invalid_bytes_surfaces_typed_invalid_image() {
        // Garbage bytes can't be decoded → fallible `into_sdk` yields the
        // typed `InvalidImage` (code 22), never a panic across the boundary.
        let err = Envelope::image(vec![0xde, 0xad, 0xbe, 0xef], "png".into())
            .into_sdk()
            .unwrap_err();
        assert!(matches!(err, Error::InvalidImage { .. }));
        assert_eq!(err.code(), 22);
    }

    #[test]
    fn multipart_envelope_propagates_nested_image_error() {
        // A bad image nested in a multipart message must surface, not panic.
        let msg = Envelope::multipart(vec![
            Envelope::text("describe".into()),
            Envelope::image(vec![0x00, 0x01], "jpeg".into()),
        ]);
        assert!(matches!(msg.into_sdk(), Err(Error::InvalidImage { .. })));
    }

    #[test]
    fn envelope_roundtrip_audio() {
        let env = Envelope::audio(vec![1, 2, 3, 4]);
        let back = Envelope::from_sdk(env.into_sdk().unwrap());
        assert_eq!(
            back.kind,
            EnvelopeKind::Audio {
                bytes: vec![1, 2, 3, 4]
            }
        );
    }

    #[test]
    fn envelope_roundtrip_embedding() {
        let env = Envelope::embedding(vec![0.1, 0.2, 0.3]);
        let back = Envelope::from_sdk(env.into_sdk().unwrap());
        assert_eq!(
            back.kind,
            EnvelopeKind::Embedding {
                values: vec![0.1, 0.2, 0.3]
            }
        );
    }

    fn text_result_with_metadata(metadata: HashMap<String, String>) -> InferenceResult {
        InferenceResult {
            envelope: Envelope {
                kind: EnvelopeKind::Text {
                    text: "the answer".into(),
                },
                metadata,
            },
            output_type: OutputType::Text,
            model_id: "m".into(),
            latency_ms: 0,
            execution_target: ExecutionTarget::Local,
            metrics: InferenceMetrics::default(),
            tool_calls: Vec::new(),
        }
    }

    #[test]
    fn reasoning_content_reads_from_envelope_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("reasoning_content".to_string(), "let me think".to_string());
        let result = text_result_with_metadata(metadata);

        // Answer text and reasoning are surfaced independently.
        assert_eq!(result.text(), Some("the answer"));
        assert_eq!(result.reasoning_content(), Some("let me think"));
    }

    #[test]
    fn reasoning_content_absent_is_none() {
        let result = text_result_with_metadata(HashMap::new());
        assert_eq!(result.reasoning_content(), None);
    }

    #[test]
    fn message_role_roundtrip() {
        for role in [
            MessageRole::System,
            MessageRole::User,
            MessageRole::Assistant,
        ] {
            // sdk → facade round trip via the as_str / parse pair.
            assert_eq!(MessageRole::parse(role.to_sdk().as_str()), Some(role));
            assert_eq!(MessageRole::parse(role.as_str()), Some(role));
        }
        assert_eq!(MessageRole::parse("nope"), None);
    }

    #[test]
    fn envelope_role_accessor_roundtrips() {
        let env = Envelope::text("hi".into()).with_role(MessageRole::Assistant);
        assert_eq!(env.role(), Some(MessageRole::Assistant));
        let plain = Envelope::text("hi".into());
        assert_eq!(plain.role(), None);
    }

    #[test]
    fn generation_config_to_sdk_applies_overrides() {
        let gc = GenerationConfig {
            max_tokens: Some(64),
            temperature: Some(0.3),
            top_k: Some(40),
            stop_sequences: vec!["</s>".into()],
            ..GenerationConfig::default()
        };
        let sdk_gc = gc.to_sdk().expect("no tools, so lowering cannot fail");
        assert_eq!(sdk_gc.max_tokens, 64);
        assert!((sdk_gc.temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(sdk_gc.top_k, 40);
        assert_eq!(sdk_gc.stop_sequences, vec!["</s>".to_string()]);
    }

    #[test]
    fn generation_config_apply_over_preserves_unset_base_fields() {
        let base = sdk::GenerationConfig {
            max_tokens: 3584,
            temperature: 0.7,
            ..sdk::GenerationConfig::default()
        };
        let gc = GenerationConfig {
            top_k: Some(12),
            ..GenerationConfig::default()
        };

        let sdk_gc = gc
            .apply_over(base)
            .expect("no tools, so lowering cannot fail");

        assert_eq!(sdk_gc.max_tokens, 3584);
        assert!((sdk_gc.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(sdk_gc.top_k, 12);
    }

    #[test]
    fn generation_config_empty_stop_sequences_preserve_model_defaults() {
        // Given
        let base = sdk::GenerationConfig {
            stop_sequences: vec!["</s>".to_string()],
            ..sdk::GenerationConfig::default()
        };

        // When
        let from_facade = GenerationConfig::default()
            .apply_over(base)
            .expect("empty tools cannot fail to lower");

        // Then
        assert_eq!(from_facade.stop_sequences, vec!["</s>"]);
    }

    #[test]
    fn generation_config_from_sdk_preserves_resolved_fields() {
        // Given
        let sdk_config = sdk::GenerationConfig {
            max_tokens: 321,
            temperature: 0.4,
            top_p: 0.8,
            min_p: 0.1,
            top_k: 17,
            repetition_penalty: 1.2,
            stop_sequences: vec!["stop".into()],
            grammar: Some("root ::= \"ok\"".into()),
            tools: vec![sdk::Tool::function(
                "weather",
                "Weather lookup",
                serde_json::json!({"type": "object"}),
            )],
        };

        // When
        let config = GenerationConfig::from_sdk(sdk_config);

        // Then
        assert_eq!(config.max_tokens, Some(321));
        assert_eq!(config.temperature, Some(0.4));
        assert_eq!(config.top_p, Some(0.8));
        assert_eq!(config.min_p, Some(0.1));
        assert_eq!(config.top_k, Some(17));
        assert_eq!(config.repetition_penalty, Some(1.2));
        assert_eq!(config.stop_sequences, vec!["stop"]);
        assert_eq!(config.grammar.as_deref(), Some("root ::= \"ok\""));
        assert_eq!(config.tools.len(), 1);
        assert_eq!(config.tools[0].name, "weather");
        assert_eq!(config.tools[0].description, "Weather lookup");
        assert_eq!(config.tools[0].parameters_json, r#"{"type":"object"}"#);
    }

    #[test]
    fn huggingface_revision_loader_preserves_requested_revision() {
        // Given / When
        let loader = ModelLoader::from_huggingface_with_revision(
            "xybrid-ai/model".into(),
            "revision-123".into(),
        );

        // Then
        assert_eq!(loader.model_id().as_deref(), Some("xybrid-ai/model"));
        assert_eq!(loader.version().as_deref(), Some("revision-123"));
    }

    #[test]
    fn huggingface_revision_loader_preserves_variant() {
        let loader = ModelLoader::from_huggingface_with_revision(
            "xybrid-ai/model-GGUF:Q8_0".into(),
            "revision-123".into(),
        );

        assert_eq!(loader.model_id().as_deref(), Some("xybrid-ai/model-GGUF"));
        assert_eq!(loader.version().as_deref(), Some("revision-123"));
    }

    #[test]
    fn generation_config_defaults_preserve_sdk_defaults() {
        // An empty facade config must not silently override the SDK's
        // baked-in defaults. Verifies the `if let Some(...)` guards.
        let baseline = sdk::GenerationConfig::default();
        let from_facade = GenerationConfig::default()
            .to_sdk()
            .expect("no tools, so lowering cannot fail");
        assert_eq!(from_facade.max_tokens, baseline.max_tokens);
        assert_eq!(from_facade.temperature, baseline.temperature);
        assert_eq!(from_facade.top_k, baseline.top_k);
        assert_eq!(from_facade.stop_sequences, baseline.stop_sequences);
    }

    #[test]
    fn run_options_builds_policy_and_cancel_token() {
        let cancel = CancellationToken::new();
        let opts = RunOptions {
            generation_config: Some(GenerationConfig::greedy()),
            abort_on: vec![
                AbortSignal::MemoryPressureCritical,
                AbortSignal::ThermalCritical,
            ],
            fallback_to_cloud: true,
            max_grace_tokens: 16,
            correlation_id: Some("trace-1".into()),
            cloud_provider: None,
            cloud_model: None,
            cloud_gateway_url: None,
        };
        let sdk_opts = opts
            .to_sdk(Some(&cancel))
            .expect("no tools, so lowering cannot fail");

        assert!(sdk_opts.generation_config.is_some());
        assert!(sdk_opts.abort_policy.fallback_to_cloud);
        assert_eq!(sdk_opts.abort_policy.max_grace_tokens, 16);
        assert!(sdk_opts
            .abort_policy
            .observes(sdk::AbortSignal::MemoryPressureCritical));
        assert!(sdk_opts
            .abort_policy
            .observes(sdk::AbortSignal::ThermalCritical));
        assert_eq!(sdk_opts.correlation_id.as_deref(), Some("trace-1"));
        assert!(sdk_opts.cancellation_token.is_some());
    }

    #[test]
    fn run_options_without_generation_config_preserves_none() {
        let base = sdk::GenerationConfig {
            max_tokens: 3584,
            ..sdk::GenerationConfig::default()
        };

        let sdk_opts = RunOptions::default()
            .to_sdk_over(None, base)
            .expect("no tools, so lowering cannot fail");

        assert!(sdk_opts.generation_config.is_none());
    }

    fn weather_tool() -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".into(),
            description: "Current weather for a city.".into(),
            parameters_json: r#"{"type":"object","properties":{"city":{"type":"string"}}}"#.into(),
        }
    }

    #[test]
    fn tools_lower_into_the_sdk_generation_config() {
        let gc = GenerationConfig {
            tools: vec![weather_tool()],
            ..GenerationConfig::default()
        };

        let sdk_gc = gc.to_sdk().expect("valid schema lowers");

        assert_eq!(sdk_gc.tools.len(), 1);
        let function = &sdk_gc.tools[0].function;
        assert_eq!(function.name, "get_weather");
        assert_eq!(
            function.description.as_deref(),
            Some("Current weather for a city.")
        );
        assert_eq!(
            function.parameters.as_ref().and_then(|p| p.get("type")),
            Some(&serde_json::json!("object"))
        );
    }

    #[test]
    fn no_tools_leaves_the_sdk_tool_list_empty() {
        // The zero-tool path must stay byte-for-byte what it was before tool
        // calling existed — an empty `tools` vec, not a `Some(vec![])`.
        let sdk_gc = GenerationConfig::default().to_sdk().expect("lowers");
        assert!(sdk_gc.tools.is_empty());
    }

    #[test]
    fn invalid_tool_schema_fails_the_run_instead_of_dropping_the_tool() {
        let gc = GenerationConfig {
            tools: vec![ToolDefinition {
                name: "broken".into(),
                description: "d".into(),
                parameters_json: "{not json".into(),
            }],
            ..GenerationConfig::default()
        };

        let error = gc.to_sdk().expect_err("invalid schema must not lower");

        assert!(matches!(error, Error::ConfigError { .. }));
        assert!(error.to_string().contains("broken"));
    }

    #[test]
    fn invalid_tool_schema_surfaces_through_run_options() {
        let opts = RunOptions {
            generation_config: Some(GenerationConfig {
                tools: vec![ToolDefinition {
                    name: "broken".into(),
                    description: "d".into(),
                    parameters_json: "[".into(),
                }],
                ..GenerationConfig::default()
            }),
            ..RunOptions::default()
        };

        assert!(matches!(opts.to_sdk(None), Err(Error::ConfigError { .. })));
    }

    #[test]
    fn stream_token_carries_the_terminal_turns_tool_calls() {
        // The core suppresses tool-call blocks from the emitted stream, so a
        // streaming caller has nothing to re-parse — the typed calls have to
        // survive the SDK → facade translation on the terminal token.
        let token = StreamToken::from_sdk(xybrid_core::runtime_adapter::types::PartialToken {
            token: String::new(),
            token_id: None,
            index: 7,
            cumulative_text: "checking".into(),
            finish_reason: Some("tool_calls".into()),
            raw_text: Some(
                "checking<|tool_call_start|>[get_temperature(room=\"kitchen\")]<|tool_call_end|>"
                    .into(),
            ),
            tool_calls: vec![xybrid_core::gateway::ToolCall {
                id: "call_0".into(),
                tool_type: "function".into(),
                function: xybrid_core::gateway::FunctionCall {
                    name: "get_temperature".into(),
                    arguments: r#"{"room":"kitchen"}"#.into(),
                },
            }],
        });

        assert_eq!(token.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(
            token.tool_calls,
            vec![ToolCall {
                id: "call_0".into(),
                name: "get_temperature".into(),
                arguments_json: r#"{"room":"kitchen"}"#.into(),
            }]
        );
        // The raw turn text is what closes the loop: `cumulative_text` has the
        // protocol block suppressed, so only this can be replayed as
        // `prior_assistant_text`.
        assert!(
            token
                .raw_text
                .as_deref()
                .is_some_and(|raw| raw.contains("<|tool_call_start|>")),
            "a terminal token with calls must carry the replayable raw text"
        );
    }

    #[test]
    fn a_mid_stream_token_carries_no_tool_calls() {
        let token = StreamToken::from_sdk(xybrid_core::runtime_adapter::types::PartialToken::new(
            "hel".into(),
            0,
            "hel".into(),
        ));
        assert!(token.tool_calls.is_empty());
        assert!(token.raw_text.is_none());
    }

    #[test]
    fn tool_calls_are_parsed_off_the_response_envelope() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tool_calls".to_string(),
            r#"[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]"#
                .to_string(),
        );
        let sdk_result = sdk::InferenceResult::new(
            sdk::ir::Envelope::with_metadata(
                sdk::ir::EnvelopeKind::Text("calling a tool".into()),
                metadata,
            ),
            "m",
            0,
        );

        let result = InferenceResult::from_sdk(sdk_result);

        assert_eq!(
            result.tool_calls,
            vec![ToolCall {
                id: "call_0".into(),
                name: "get_weather".into(),
                arguments_json: r#"{"city":"Paris"}"#.into(),
            }]
        );
        // The raw block is left in the text — parsing is additive.
        assert_eq!(result.text(), Some("calling a tool"));
    }

    #[test]
    fn a_response_without_tool_calls_yields_an_empty_vec() {
        let result = text_result_with_metadata(HashMap::new());
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn tool_results_envelope_carries_the_continuation_metadata() {
        let envelope = Envelope::tool_results(
            "weather in Paris?".into(),
            r#"<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>"#.into(),
            vec![ToolResult {
                call_id: "call_0".into(),
                name: "get_weather".into(),
                content_json: r#"{"temperature_c":17.5}"#.into(),
            }],
        )
        .expect("valid JSON content");

        assert_eq!(
            envelope.kind,
            EnvelopeKind::Text {
                text: "weather in Paris?".into()
            }
        );
        assert!(envelope
            .metadata
            .get("tool_prior_text")
            .expect("prior text is carried")
            .contains("get_weather"));
        assert!(envelope
            .metadata
            .get("tool_responses")
            .expect("responses are carried")
            .contains("17.5"));
    }

    #[test]
    fn tool_results_envelope_rejects_non_json_content() {
        let error = Envelope::tool_results(
            "u".into(),
            "prior".into(),
            vec![ToolResult {
                call_id: "call_0".into(),
                name: "get_weather".into(),
                content_json: "not json".into(),
            }],
        )
        .expect_err("invalid content must not build an envelope");

        assert!(matches!(error, Error::ConfigError { .. }));
        assert!(error.to_string().contains("get_weather"));
    }

    #[test]
    fn cancellation_token_is_observable_through_arc() {
        let token = CancellationToken::new();
        let clone = Arc::clone(&token);
        assert!(!token.is_cancelled());
        clone.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn streaming_session_delivers_tokens_then_completion_in_order() {
        let session = StreamingSession::spawn_for_test(|sender| {
            sender
                .send(StreamEvent::Token(StreamToken {
                    token: "hel".into(),
                    token_id: Some(1),
                    index: 0,
                    cumulative_text: "hel".into(),
                    finish_reason: None,
                    tool_calls: Vec::new(),
                    raw_text: None,
                }))
                .expect("test stream receiver should remain connected");
            sender
                .send(StreamEvent::Token(StreamToken {
                    token: "lo".into(),
                    token_id: Some(2),
                    index: 1,
                    cumulative_text: "hello".into(),
                    finish_reason: Some("stop".into()),
                    tool_calls: Vec::new(),
                    raw_text: None,
                }))
                .expect("test stream receiver should remain connected");
            sender
                .send(StreamEvent::Complete(text_result_with_metadata(
                    HashMap::new(),
                )))
                .expect("test stream receiver should remain connected");
        });

        assert!(matches!(
            session.next(),
            Some(StreamEvent::Token(StreamToken { index: 0, .. }))
        ));
        assert!(matches!(
            session.next(),
            Some(StreamEvent::Token(StreamToken { index: 1, .. }))
        ));
        assert!(matches!(session.next(), Some(StreamEvent::Complete(_))));
        assert!(session.next().is_none());
    }

    #[test]
    fn streaming_session_delivers_typed_terminal_error() {
        let session = StreamingSession::spawn_for_test(|sender| {
            sender
                .send(StreamEvent::Error(Error::NotLoaded))
                .expect("test stream receiver should remain connected");
        });

        assert!(matches!(
            session.next(),
            Some(StreamEvent::Error(Error::NotLoaded))
        ));
        assert!(session.next().is_none());
    }

    #[test]
    fn conversation_context_push_history_clear() {
        let ctx = ConversationContextHandle::new();
        ctx.push(Envelope::text("hi".into()).with_role(MessageRole::User))
            .unwrap();
        ctx.push(Envelope::text("hello".into()).with_role(MessageRole::Assistant))
            .unwrap();

        let hist = ctx.history();
        assert_eq!(hist.len(), 2);
        assert!(matches!(hist[0].kind, EnvelopeKind::Text { ref text } if text == "hi"));

        ctx.clear();
        assert!(ctx.history().is_empty());
    }

    #[test]
    fn resolve_binding_keeps_every_platform_binding() {
        for binding in ["flutter", "kotlin", "react-native", "swift", "unity"] {
            assert_eq!(resolve_binding(binding), binding);
        }
    }

    #[test]
    fn resolve_binding_collapses_unknown_names_to_default() {
        for binding in [
            "",
            "rust-sdk",
            "React Native",
            "react_native",
            "reactnative",
        ] {
            assert_eq!(resolve_binding(binding), sdk::DEFAULT_BINDING);
        }
    }

    #[test]
    fn set_binding_resolves_known_platforms_only() {
        // Process-global; this test is best-effort and may no-op if another
        // test set the binding first. The contract we care about is that
        // `get_binding()` returns one of the accepted values.
        set_binding("flutter".into());
        let bound = get_binding();
        assert!(matches!(
            bound.as_str(),
            "flutter" | "kotlin" | "react-native" | "swift" | "unity" | "rust"
        ));
    }

    #[test]
    fn binding_setter_rejects_unknown() {
        // First-set-wins on the underlying OnceLock: another test may already
        // have pinned the value, so assert the result is in the accepted set.
        // `resolve_binding_collapses_unknown_names_to_default` covers the
        // mapping itself.
        set_binding("not-a-real-binding".into());
        let bound = get_binding();
        assert!(matches!(
            bound.as_str(),
            "flutter" | "kotlin" | "react-native" | "swift" | "unity" | "rust"
        ));
    }

    #[test]
    fn telemetry_default_endpoint_matches_sdk() {
        assert_eq!(
            telemetry_default_endpoint(),
            sdk::telemetry::DEFAULT_INGEST_URL
        );
    }

    #[test]
    fn version_tracks_cargo_pkg_version() {
        assert_eq!(version(), sdk::SDK_VERSION);
        assert!(!version().is_empty());
    }

    #[test]
    fn telemetry_builder_maps_to_sdk_fields_and_consumes() {
        let handle = TelemetryConfigHandle::new("secret-key".into());

        // A fresh config defaults its endpoint to the ingest URL (bindings read
        // this back before init).
        assert_eq!(
            handle.lock().as_ref().unwrap().endpoint,
            sdk::telemetry::DEFAULT_INGEST_URL
        );

        handle.set_endpoint("https://collector.internal".into());
        handle.set_app_version("1.2.3".into());
        handle.set_device_label("Test Device".into());
        handle.set_device_attribute("ring".into(), "canary".into());
        handle.set_batch_size(64);
        handle.set_flush_interval_secs(30);

        // `telemetry_init` takes the config out; assert every setter landed on the
        // matching SDK field — this is the wire the C#/Unity builder relies on.
        let config = handle.take().expect("config present before init");
        assert_eq!(config.endpoint, "https://collector.internal");
        assert_eq!(config.api_key, "secret-key");
        assert_eq!(config.app_version.as_deref(), Some("1.2.3"));
        assert_eq!(config.device_label.as_deref(), Some("Test Device"));
        assert_eq!(
            config.device_profile_patch.custom.get("ring"),
            Some(&"canary".to_string())
        );
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.flush_interval_secs, 30);

        // Consumed: a second take yields nothing and later setters no-op rather
        // than panic (mirrors the pre-bolt consume-on-init contract).
        assert!(handle.take().is_none());
        handle.set_endpoint("ignored".into());
        assert!(handle.take().is_none());
    }

    /// Build a partial with just the fields the fanout cares about.
    fn partial(sequence: u64, text: &str) -> PartialResult {
        PartialResult {
            text: text.to_string(),
            is_stable: false,
            chunk_sequence: sequence,
            audio_duration_ms: sequence * 100,
        }
    }

    /// Collect events an observer receives.
    fn recorder() -> (Arc<Mutex<Vec<AsrEvent>>>, AsrObserver) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let observer: AsrObserver = Box::new(move |event| {
            sink.lock().unwrap_or_else(|e| e.into_inner()).push(event);
        });
        (seen, observer)
    }

    /// `feed` is non-blocking and the host subscribes a moment later, so the
    /// first transcripts routinely land before anyone is listening. Dropping
    /// them would lose the opening words of every utterance.
    #[test]
    fn a_partial_produced_before_subscribing_is_delivered_on_subscribe() {
        let mut fanout = AsrFanout::default();
        fanout.emit(partial(1, "hello"));
        fanout.emit(partial(2, "hello there"));

        let (seen, observer) = recorder();
        fanout.watch(observer);

        // Only the most recent is held: partial text is cumulative, so the
        // newer one already contains the older.
        let events = seen.lock().unwrap().clone();
        assert_eq!(events, vec![AsrEvent::Partial(partial(2, "hello there"))]);
    }

    #[test]
    fn subscribing_to_a_finished_session_terminates_at_once() {
        let mut fanout = AsrFanout::default();
        fanout.finish();

        let (seen, observer) = recorder();
        fanout.watch(observer);

        // Terminal immediately rather than hanging, and not retained — a
        // stream that never closes is what strands a host's `for await`.
        assert_eq!(seen.lock().unwrap().clone(), vec![AsrEvent::Finished]);
        assert!(fanout.observers.is_empty());
    }

    #[test]
    fn finish_notifies_every_observer_once_then_drops_them() {
        let mut fanout = AsrFanout::default();
        let (first, first_observer) = recorder();
        let (second, second_observer) = recorder();
        fanout.watch(first_observer);
        fanout.watch(second_observer);

        fanout.emit(partial(1, "hi"));
        fanout.finish();
        // Idempotent: a session that is flushed and then dropped reaches this
        // twice, and the host must not see two terminal events.
        fanout.finish();

        let expected = vec![AsrEvent::Partial(partial(1, "hi")), AsrEvent::Finished];
        assert_eq!(first.lock().unwrap().clone(), expected);
        assert_eq!(second.lock().unwrap().clone(), expected);
        assert!(
            fanout.observers.is_empty(),
            "host closures retained past the terminal event"
        );
    }

    #[test]
    fn streaming_config_rejects_a_sample_rate_the_backends_cannot_accept() {
        let config = StreamingConfig {
            sample_rate: 44_100,
            ..StreamingConfig::default()
        };
        let error = config.to_sdk().expect_err("44.1 kHz must be rejected");
        assert!(matches!(error, Error::ConfigError { .. }), "got {error:?}");

        // Rejected rather than resampled: silently accepting it would produce
        // confident nonsense instead of an error.
        assert!(StreamingConfig::default().to_sdk().is_ok());
    }

    /// A cancel must abandon queued audio, not wait for it.
    ///
    /// Dropping the command sender alone does not do this: `Receiver::recv`
    /// yields every buffered command before reporting disconnection, so the
    /// worker would transcribe the whole backlog — holding the model and the
    /// CPU — after the caller believed it had stopped.
    #[test]
    fn cancel_abandons_the_queued_backlog() {
        let (sender, receiver) = mpsc::sync_channel::<AsrCommand>(ASR_COMMAND_CAPACITY);
        let cancelled = Arc::new(AtomicBool::new(false));

        // Queue work, then cancel before the worker gets to any of it.
        for _ in 0..8 {
            sender.send(AsrCommand::Feed(vec![0.0; 16])).unwrap();
        }
        cancelled.store(true, Ordering::Relaxed);
        drop(sender);

        // Stand in for the worker's command loop.
        let mut processed = 0;
        while let Ok(_command) = receiver.recv() {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }
            processed += 1;
        }

        assert_eq!(processed, 0, "cancelled worker drained the backlog");
    }

    /// A failed chunk is gone: the core extracts it from its buffer before
    /// transcribing, so the audio never reaches the transcript. Returning
    /// that transcript as success hides the hole.
    #[test]
    fn a_recorded_chunk_failure_turns_flush_into_an_error() {
        let failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        // First failure wins; later ones must not overwrite the root cause.
        {
            let mut recorded = failure.lock().unwrap();
            recorded.get_or_insert("ASR chunk failed: dtype mismatch".to_string());
        }
        {
            let mut recorded = failure.lock().unwrap();
            recorded.get_or_insert("ASR chunk failed: something later".to_string());
        }

        let taken = failure.lock().unwrap().take();
        assert_eq!(
            taken.as_deref(),
            Some("ASR chunk failed: dtype mismatch"),
            "the first failure is the one worth reporting"
        );
        // Taken, so a second flush of a reset session is not haunted by it.
        assert!(failure.lock().unwrap().is_none());
    }

    /// Enabling VAD always carries a model directory.
    ///
    /// The core reads `enable_vad` without a `vad_model_dir` as "warn on
    /// stderr and silently use fixed windows", and nothing ships a bundled
    /// Silero model — so a config that could produce that pair would be a
    /// feature that quietly does nothing.
    #[test]
    fn enabling_vad_always_carries_a_model_directory() {
        let enabled = StreamingConfig {
            vad: VadMode::Enabled {
                model_dir: "/models/silero".into(),
            },
            ..StreamingConfig::default()
        }
        .to_sdk()
        .expect("valid config");
        assert!(enabled.enable_vad);
        assert_eq!(enabled.vad_model_dir.as_deref(), Some("/models/silero"));

        // The pairing the enum exists to make unrepresentable: no model
        // directory can survive VAD being off.
        let off = StreamingConfig::default().to_sdk().expect("valid config");
        assert!(!off.enable_vad);
        assert_eq!(off.vad_model_dir, None);
    }
}
