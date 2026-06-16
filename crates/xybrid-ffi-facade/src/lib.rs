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
//! - **LLM token streaming.** The SDK exposes callback-based streaming
//!   APIs (`run_streaming`, `run_stream`); a channel-style facade wrapper
//!   will land next, once the binding crates start consuming this surface.
//! - **ASR streaming.** [`xybrid_sdk::stream::XybridStream`] is wrapped
//!   separately in the same follow-up.
//! - **Pipelines.** `xybrid-sdk` already exports POD-friendly
//!   [`FfiPipelineExecutionResult`] / [`FfiStageExecutionResult`]; the
//!   binding crates can re-export those directly. A dedicated facade for
//!   pipelines is a separate concern.
//!
//! [`xybrid-bolt`]: https://docs.rs/xybrid-bolt
//! [`xybrid-ffi`]: https://docs.rs/xybrid-ffi
//! [`FfiPipelineExecutionResult`]: xybrid_sdk::FfiPipelineExecutionResult
//! [`FfiStageExecutionResult`]: xybrid_sdk::FfiStageExecutionResult

use std::collections::HashMap;
use std::sync::Arc;

use xybrid_sdk as sdk;

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
}

impl GenerationConfig {
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

    /// Materialize the SDK type. Binding crates wrapping a non-facade POD
    /// (e.g. `FfiGenerationConfig` in the Flutter bindings) call this to
    /// consume the canonical "option overrides → SDK defaults" mapping
    /// instead of duplicating it. `pub` rather than `pub(crate)` for that
    /// reason.
    pub fn to_sdk(&self) -> sdk::GenerationConfig {
        let mut cfg = sdk::GenerationConfig::default();
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
        cfg
    }
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
}

impl RunOptions {
    /// Materialize the SDK type. `pub` so binding crates with their own
    /// run-options POD (e.g. `FfiRunOptions` in the Flutter bindings) can
    /// route through this for the policy-builder assembly without
    /// re-implementing it.
    pub fn to_sdk(&self, cancel: Option<&CancellationToken>) -> sdk::RunOptions {
        let mut policy = sdk::AbortPolicy::default()
            .with_cloud_fallback(self.fallback_to_cloud)
            .with_max_grace_tokens(self.max_grace_tokens);
        for sig in &self.abort_on {
            policy = policy.stop_on(sig.to_sdk());
        }

        let mut opts = sdk::RunOptions::new().with_abort_policy(policy);
        if let Some(gc) = &self.generation_config {
            opts = opts.with_generation_config(gc.to_sdk());
        }
        if let Some(cid) = &self.correlation_id {
            opts = opts.with_correlation_id(cid.clone());
        }
        if let Some(tok) = cancel {
            opts = opts.with_cancellation_token(tok.inner.clone());
        }
        opts
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
    pub metrics: InferenceMetrics,
}

impl InferenceResult {
    pub fn from_sdk(result: sdk::InferenceResult) -> Self {
        let output_type = OutputType::from_sdk(result.output_type());
        let model_id = result.model_id().to_string();
        let latency_ms = result.latency_ms();
        let metrics = InferenceMetrics::from_sdk(result.metrics());
        let envelope = Envelope::from_sdk(result.into_envelope());
        Self {
            envelope,
            output_type,
            model_id,
            latency_ms,
            metrics,
        }
    }

    /// Convenience: text payload, if the result is `OutputType::Text`.
    pub fn text(&self) -> Option<&str> {
        match &self.envelope.kind {
            EnvelopeKind::Text { text } => Some(text.as_str()),
            _ => None,
        }
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

    pub fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }

    pub fn is_llm(&self) -> bool {
        self.inner.is_llm()
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
        let env = envelope.into_sdk()?;
        let opts = options.to_sdk(cancel.as_deref());
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
        let gc = generation_config.as_ref().map(GenerationConfig::to_sdk);
        let result = self
            .inner
            .run_with_context(&env, &ctx, gc.as_ref())
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

/// Register the binding identifier (`"flutter"`, `"kotlin"`, `"swift"`,
/// `"unity"`) reported in the `X-Xybrid-Client` registry header.
///
/// Each generator crate calls this once at SDK init with its hard-coded
/// constant. Unknown strings fall back to [`sdk::DEFAULT_BINDING`] to
/// bound cardinality on the registry side. First call wins.
pub fn set_binding(binding: String) {
    let resolved: &'static str = match binding.as_str() {
        "flutter" => "flutter",
        "kotlin" => "kotlin",
        "swift" => "swift",
        "unity" => "unity",
        _ => sdk::DEFAULT_BINDING,
    };
    sdk::set_binding(resolved);
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
        let sdk_gc = gc.to_sdk();
        assert_eq!(sdk_gc.max_tokens, 64);
        assert!((sdk_gc.temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(sdk_gc.top_k, 40);
        assert_eq!(sdk_gc.stop_sequences, vec!["</s>".to_string()]);
    }

    #[test]
    fn generation_config_defaults_preserve_sdk_defaults() {
        // An empty facade config must not silently override the SDK's
        // baked-in defaults. Verifies the `if let Some(...)` guards.
        let baseline = sdk::GenerationConfig::default();
        let from_facade = GenerationConfig::default().to_sdk();
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
        };
        let sdk_opts = opts.to_sdk(Some(&cancel));

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
    fn cancellation_token_is_observable_through_arc() {
        let token = CancellationToken::new();
        let clone = Arc::clone(&token);
        assert!(!token.is_cancelled());
        clone.cancel();
        assert!(token.is_cancelled());
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
    fn set_binding_resolves_known_platforms_only() {
        // Process-global; this test is best-effort and may no-op if another
        // test set the binding first. The contract we care about is that
        // `get_binding()` returns one of the accepted values.
        set_binding("flutter".into());
        let bound = get_binding();
        assert!(matches!(
            bound.as_str(),
            "flutter" | "kotlin" | "swift" | "unity" | "rust"
        ));
    }

    #[test]
    fn binding_setter_rejects_unknown() {
        // First-set-wins on the underlying OnceLock means we can only
        // verify the resolution helper indirectly via `get_binding()`.
        // The match arm in `set_binding` collapses unknowns to
        // DEFAULT_BINDING, which is `"rust"`; any other test that ran
        // first may already have pinned the value, so we just assert
        // the result is in the accepted set.
        set_binding("not-a-real-binding".into());
        let bound = get_binding();
        assert!(matches!(
            bound.as_str(),
            "flutter" | "kotlin" | "swift" | "unity" | "rust"
        ));
    }
}
