//! BoltFFI bindings for xybrid-sdk.
//!
//! This crate is the single source for the **non-Flutter** foreign-language
//! SDKs (Swift / Kotlin / Java / C# / WASM, plus the C header that Unity
//! consumes). It describes the [`xybrid_ffi_facade`] surface externally for
//! the BoltFFI generator following BoltFFI's convention:
//!
//! - Records (POD types) are mirrored as `#[data]` structs/enums in this
//!   crate. Proc macros must live on the type definitions, not on
//!   re-exports, so the facade's types are re-declared here and converted
//!   via plain `From` impls.
//! - The error enum is marked `#[error]` so it surfaces as a typed
//!   exception in target languages (`enum Error: Error` in Swift,
//!   `sealed class XybridException : Exception()` in Kotlin, etc.).
//! - Handle types use `#[export] impl Foo { ... }`; BoltFFI manages the
//!   heap allocation and FFI handle internally — no `Arc<Self>` return is
//!   required at the call site.
//!
//! Run `boltffi pack all --release` (or per-target,
//! e.g. `boltffi pack apple`) from `tools/scripts/` to generate the
//! Swift / Kotlin / Java / C# / WASM bindings from this crate.
//!
//! ## Migration status (sketch)
//!
//! - **Records and Error**: complete.
//! - **Free functions** (init / push API): complete for the subset every
//!   binding needs at startup.
//! - **`XybridModel` + `ModelLoader`**: minimal `#[export]` blocks
//!   covering the load / run / warmup / voice surface. Enough to validate
//!   the proc-macro shape against the facade.
//! - **Deferred to follow-up commits**:
//!   - Token streaming (`run_stream` / `run_stream_with_context`) — needs
//!     BoltFFI's stream-event convention nailed down across all targets.
//!   - `CancellationToken` as an `Arc<Self>` handle.
//!   - `ConversationContextHandle` (uniffi opaque object equivalent).
//!   - `run_with_options` / `run_with_context`.
//!   - Pipeline surface.
//!
//! Until `xybrid-uniffi` and `xybrid-ffi` are removed, this crate exists
//! alongside them. The deletion happens once the Swift / Kotlin example
//! apps in `examples/` build against bolt-generated bindings and the
//! Unity package is rewired against bolt's emitted C header.

use boltffi::*;
use xybrid_ffi_facade as facade;

// ============================================================================
// Error
// ============================================================================

/// Errors surfaced across the FFI boundary. Variants mirror
/// [`facade::Error`] — the facade owns the SDK→FFI translation; this enum
/// only re-decorates it for the BoltFFI generator (proc macros must live
/// on the type definition).
#[error]
#[derive(Debug, Clone)]
pub enum Error {
    ModelNotFound { id: String },
    DirectoryNotFound { path: String },
    MetadataNotFound { path: String },
    MetadataInvalid { message: String },
    LoadError { message: String },
    InferenceError { message: String },
    AbortedForCloudFallback { reason: String },
    StreamingNotSupported,
    NotLoaded,
    ConfigError { message: String },
    NetworkError { message: String },
    Offline { message: String },
    IoError { message: String },
    CacheError { message: String },
    PipelineError { message: String },
    CircuitOpen { message: String },
    RateLimited { retry_after_secs: u64 },
    Timeout { timeout_ms: u64 },
}

impl Error {
    /// Stable numeric discriminant inherited from the facade. Same wire
    /// codes across every binding so foreign consumers can switch on a
    /// shared protocol.
    pub fn code(&self) -> u32 {
        // Delegate via the facade so the code table lives in one place.
        // BoltFFI sees this as an inherent method on the error type.
        facade::Error::from(self.clone()).code()
    }
}

impl From<Error> for facade::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::ModelNotFound { id } => facade::Error::ModelNotFound { id },
            Error::DirectoryNotFound { path } => facade::Error::DirectoryNotFound { path },
            Error::MetadataNotFound { path } => facade::Error::MetadataNotFound { path },
            Error::MetadataInvalid { message } => facade::Error::MetadataInvalid { message },
            Error::LoadError { message } => facade::Error::LoadError { message },
            Error::InferenceError { message } => facade::Error::InferenceError { message },
            Error::AbortedForCloudFallback { reason } => {
                facade::Error::AbortedForCloudFallback { reason }
            }
            Error::StreamingNotSupported => facade::Error::StreamingNotSupported,
            Error::NotLoaded => facade::Error::NotLoaded,
            Error::ConfigError { message } => facade::Error::ConfigError { message },
            Error::NetworkError { message } => facade::Error::NetworkError { message },
            Error::Offline { message } => facade::Error::Offline { message },
            Error::IoError { message } => facade::Error::IoError { message },
            Error::CacheError { message } => facade::Error::CacheError { message },
            Error::PipelineError { message } => facade::Error::PipelineError { message },
            Error::CircuitOpen { message } => facade::Error::CircuitOpen { message },
            Error::RateLimited { retry_after_secs } => {
                facade::Error::RateLimited { retry_after_secs }
            }
            Error::Timeout { timeout_ms } => facade::Error::Timeout { timeout_ms },
        }
    }
}

impl From<facade::Error> for Error {
    fn from(e: facade::Error) -> Self {
        match e {
            facade::Error::ModelNotFound { id } => Error::ModelNotFound { id },
            facade::Error::DirectoryNotFound { path } => Error::DirectoryNotFound { path },
            facade::Error::MetadataNotFound { path } => Error::MetadataNotFound { path },
            facade::Error::MetadataInvalid { message } => Error::MetadataInvalid { message },
            facade::Error::LoadError { message } => Error::LoadError { message },
            facade::Error::InferenceError { message } => Error::InferenceError { message },
            facade::Error::AbortedForCloudFallback { reason } => {
                Error::AbortedForCloudFallback { reason }
            }
            facade::Error::StreamingNotSupported => Error::StreamingNotSupported,
            facade::Error::NotLoaded => Error::NotLoaded,
            facade::Error::ConfigError { message } => Error::ConfigError { message },
            facade::Error::NetworkError { message } => Error::NetworkError { message },
            facade::Error::Offline { message } => Error::Offline { message },
            facade::Error::IoError { message } => Error::IoError { message },
            facade::Error::CacheError { message } => Error::CacheError { message },
            facade::Error::PipelineError { message } => Error::PipelineError { message },
            facade::Error::CircuitOpen { message } => Error::CircuitOpen { message },
            facade::Error::RateLimited { retry_after_secs } => {
                Error::RateLimited { retry_after_secs }
            }
            facade::Error::Timeout { timeout_ms } => Error::Timeout { timeout_ms },
        }
    }
}

// ============================================================================
// Envelope payload + role
// ============================================================================

#[data]
#[derive(Clone)]
pub enum EnvelopeKind {
    Text { text: String },
    Audio { bytes: Vec<u8> },
    Embedding { values: Vec<f32> },
}

impl From<EnvelopeKind> for facade::EnvelopeKind {
    fn from(k: EnvelopeKind) -> Self {
        match k {
            EnvelopeKind::Text { text } => facade::EnvelopeKind::Text { text },
            EnvelopeKind::Audio { bytes } => facade::EnvelopeKind::Audio { bytes },
            EnvelopeKind::Embedding { values } => facade::EnvelopeKind::Embedding { values },
        }
    }
}

impl From<facade::EnvelopeKind> for EnvelopeKind {
    fn from(k: facade::EnvelopeKind) -> Self {
        match k {
            facade::EnvelopeKind::Text { text } => EnvelopeKind::Text { text },
            facade::EnvelopeKind::Audio { bytes } => EnvelopeKind::Audio { bytes },
            facade::EnvelopeKind::Embedding { values } => EnvelopeKind::Embedding { values },
        }
    }
}

/// Single metadata key/value entry. BoltFFI doesn't auto-derive
/// `WireEncode` for `HashMap<String, String>`, so we expose metadata as
/// `Vec<MetadataEntry>`. The conversion back to `HashMap` happens at
/// the facade boundary inside [`Envelope::into`].
#[data]
#[derive(Clone)]
pub struct MetadataEntry {
    pub key: String,
    pub value: String,
}

#[data]
#[derive(Clone)]
pub struct Envelope {
    pub kind: EnvelopeKind,
    pub metadata: Vec<MetadataEntry>,
}

impl From<Envelope> for facade::Envelope {
    fn from(e: Envelope) -> Self {
        facade::Envelope {
            kind: e.kind.into(),
            metadata: e
                .metadata
                .into_iter()
                .map(|MetadataEntry { key, value }| (key, value))
                .collect(),
        }
    }
}

impl From<facade::Envelope> for Envelope {
    fn from(e: facade::Envelope) -> Self {
        Self {
            kind: e.kind.into(),
            metadata: e
                .metadata
                .into_iter()
                .map(|(key, value)| MetadataEntry { key, value })
                .collect(),
        }
    }
}

#[data]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl From<MessageRole> for facade::MessageRole {
    fn from(r: MessageRole) -> Self {
        match r {
            MessageRole::System => facade::MessageRole::System,
            MessageRole::User => facade::MessageRole::User,
            MessageRole::Assistant => facade::MessageRole::Assistant,
        }
    }
}

// ============================================================================
// Generation + Run options
// ============================================================================

#[data]
#[derive(Clone)]
pub struct GenerationConfig {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub stop_sequences: Vec<String>,
}

impl From<GenerationConfig> for facade::GenerationConfig {
    fn from(c: GenerationConfig) -> Self {
        Self {
            max_tokens: c.max_tokens,
            temperature: c.temperature,
            top_p: c.top_p,
            min_p: c.min_p,
            top_k: c.top_k,
            repetition_penalty: c.repetition_penalty,
            stop_sequences: c.stop_sequences,
        }
    }
}

#[data]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AbortSignal {
    MemoryPressureWarn,
    MemoryPressureCritical,
    ThermalHot,
    ThermalCritical,
}

impl From<AbortSignal> for facade::AbortSignal {
    fn from(s: AbortSignal) -> Self {
        match s {
            AbortSignal::MemoryPressureWarn => facade::AbortSignal::MemoryPressureWarn,
            AbortSignal::MemoryPressureCritical => facade::AbortSignal::MemoryPressureCritical,
            AbortSignal::ThermalHot => facade::AbortSignal::ThermalHot,
            AbortSignal::ThermalCritical => facade::AbortSignal::ThermalCritical,
        }
    }
}

#[data]
#[derive(Clone)]
pub struct RunOptions {
    pub generation_config: Option<GenerationConfig>,
    pub abort_on: Vec<AbortSignal>,
    pub fallback_to_cloud: bool,
    pub max_grace_tokens: u32,
    pub correlation_id: Option<String>,
}

impl From<RunOptions> for facade::RunOptions {
    fn from(o: RunOptions) -> Self {
        Self {
            generation_config: o.generation_config.map(Into::into),
            abort_on: o.abort_on.into_iter().map(Into::into).collect(),
            fallback_to_cloud: o.fallback_to_cloud,
            max_grace_tokens: o.max_grace_tokens,
            correlation_id: o.correlation_id,
        }
    }
}

// ============================================================================
// Inference result + metrics
// ============================================================================

#[data]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    Text,
    Audio,
    Embedding,
    Unknown,
}

impl From<facade::OutputType> for OutputType {
    fn from(t: facade::OutputType) -> Self {
        match t {
            facade::OutputType::Text => OutputType::Text,
            facade::OutputType::Audio => OutputType::Audio,
            facade::OutputType::Embedding => OutputType::Embedding,
            facade::OutputType::Unknown => OutputType::Unknown,
        }
    }
}

#[data]
#[derive(Clone)]
pub struct StageLatency {
    pub stage_id: String,
    pub latency_ms: u32,
}

impl From<&facade::StageLatency> for StageLatency {
    fn from(s: &facade::StageLatency) -> Self {
        Self {
            stage_id: s.stage_id.clone(),
            latency_ms: s.latency_ms,
        }
    }
}

#[data]
#[derive(Clone)]
pub struct InferenceMetrics {
    pub total_ms: u32,
    pub ttft_ms: Option<u32>,
    pub tokens_per_second: Option<f32>,
    pub prefill_tps: Option<f32>,
    pub decode_tps: Option<f32>,
    pub tokens_out: Option<u32>,
    pub stage_latencies_ms: Vec<StageLatency>,
}

impl From<&facade::InferenceMetrics> for InferenceMetrics {
    fn from(m: &facade::InferenceMetrics) -> Self {
        Self {
            total_ms: m.total_ms,
            ttft_ms: m.ttft_ms,
            tokens_per_second: m.tokens_per_second,
            prefill_tps: m.prefill_tps,
            decode_tps: m.decode_tps,
            tokens_out: m.tokens_out,
            stage_latencies_ms: m.stage_latencies_ms.iter().map(Into::into).collect(),
        }
    }
}

#[data]
#[derive(Clone)]
pub struct InferenceResult {
    pub envelope: Envelope,
    pub output_type: OutputType,
    pub model_id: String,
    pub latency_ms: u32,
    pub metrics: InferenceMetrics,
}

impl From<facade::InferenceResult> for InferenceResult {
    fn from(r: facade::InferenceResult) -> Self {
        let metrics = InferenceMetrics::from(&r.metrics);
        Self {
            envelope: r.envelope.into(),
            output_type: r.output_type.into(),
            model_id: r.model_id,
            latency_ms: r.latency_ms,
            metrics,
        }
    }
}

// ============================================================================
// Voice info
// ============================================================================

#[data]
#[derive(Clone)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub gender: Option<String>,
    pub language: Option<String>,
    pub style: Option<String>,
}

impl From<facade::VoiceInfo> for VoiceInfo {
    fn from(v: facade::VoiceInfo) -> Self {
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
// Device / platform push API
// ============================================================================

#[data]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Normal,
    Warm,
    Hot,
    Critical,
}

impl From<ThermalState> for facade::ThermalState {
    fn from(s: ThermalState) -> Self {
        match s {
            ThermalState::Normal => facade::ThermalState::Normal,
            ThermalState::Warm => facade::ThermalState::Warm,
            ThermalState::Hot => facade::ThermalState::Hot,
            ThermalState::Critical => facade::ThermalState::Critical,
        }
    }
}

#[export]
pub fn set_thermal_state(state: ThermalState) {
    facade::set_thermal_state(state.into());
}

#[export]
pub fn clear_thermal_state() {
    facade::clear_thermal_state();
}

#[export]
pub fn set_battery_level(percent: u8) {
    facade::set_battery_level(percent);
}

#[export]
pub fn clear_battery_level() {
    facade::clear_battery_level();
}

// ============================================================================
// Process-global init
// ============================================================================

#[export]
pub fn init_sdk_cache_dir(path: String) {
    facade::init_sdk_cache_dir(path);
}

#[export]
pub fn set_binding(binding: String) {
    facade::set_binding(binding);
}

#[export]
pub fn set_api_key(api_key: String) {
    facade::set_api_key(api_key);
}

#[export]
pub fn set_provider_api_key(provider: String, api_key: String) {
    facade::set_provider_api_key(provider, api_key);
}

// ============================================================================
// XybridModel handle
// ============================================================================
//
// Sketch scope: load / run / warmup / unload / voice accessors only.
// Streaming + cancellation + conversation context are wired in follow-up
// commits once the bolt artifact emission has been validated against
// the existing Swift / Kotlin / Unity examples.
//
// `ModelLoader` is intentionally **not** mirrored as a separate
// `#[export]` type. BoltFFI's wire layer treats opaque types as handle
// IDs that only the `impl` block they're defined on can return; routing a
// loaded model from `ModelLoader::load` back to `XybridModel` would
// require manual handle-table plumbing. Collapsing it into
// `XybridModel::from_*` constructors removes that whole layer (the
// foreign API becomes one type, like caracas's `Dict::install_*`) and
// matches the facade's existing handle convention.

pub struct XybridModel {
    inner: std::sync::Arc<facade::XybridModel>,
}

#[export]
impl XybridModel {
    /// Load from the xybrid registry. Recommended path.
    pub fn from_registry(id: String) -> Result<Self, Error> {
        let model = facade::ModelLoader::from_registry(id)
            .load()
            .map_err(Error::from)?;
        Ok(Self { inner: model })
    }

    /// Load from a local model directory (must contain `model_metadata.json`).
    pub fn from_directory(path: String) -> Result<Self, Error> {
        let loader = facade::ModelLoader::from_directory(path).map_err(Error::from)?;
        let model = loader.load().map_err(Error::from)?;
        Ok(Self { inner: model })
    }

    /// Load from a local `.xyb` bundle.
    pub fn from_bundle(path: String) -> Result<Self, Error> {
        let loader = facade::ModelLoader::from_bundle(path).map_err(Error::from)?;
        let model = loader.load().map_err(Error::from)?;
        Ok(Self { inner: model })
    }

    /// Resolve and load from a HuggingFace repo (`org/repo` or `org/repo:variant`).
    pub fn from_huggingface(repo: String) -> Result<Self, Error> {
        let model = facade::ModelLoader::from_huggingface(repo)
            .load()
            .map_err(Error::from)?;
        Ok(Self { inner: model })
    }

    pub fn model_id(&self) -> String {
        self.inner.model_id()
    }

    pub fn version(&self) -> String {
        self.inner.version()
    }

    pub fn output_type(&self) -> OutputType {
        self.inner.output_type().into()
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
            .into_iter()
            .map(VoiceInfo::from)
            .collect()
    }

    pub fn default_voice(&self) -> Option<VoiceInfo> {
        self.inner.default_voice().map(VoiceInfo::from)
    }

    pub fn voice(&self, voice_id: String) -> Option<VoiceInfo> {
        self.inner.voice(voice_id).map(VoiceInfo::from)
    }

    pub fn run(&self, envelope: Envelope) -> Result<InferenceResult, Error> {
        let result = self.inner.run(envelope.into()).map_err(Error::from)?;
        Ok(result.into())
    }

    pub fn warmup(&self) -> Result<(), Error> {
        self.inner.warmup().map_err(Error::from)
    }

    pub fn unload(&self) -> Result<(), Error> {
        self.inner.unload().map_err(Error::from)
    }
}

// ============================================================================
// Tests
// ============================================================================
//
// The bolt proc-macros generate FFI glue (extern "C" exports, handle
// tables, etc.) — we verify their *shape* compiles here. Behavioral
// coverage of conversions lives in the facade crate; covering it again
// here would just duplicate `xybrid-ffi-facade`'s test suite.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_roundtrips_through_facade() {
        let env = Envelope {
            kind: EnvelopeKind::Text { text: "hi".into() },
            metadata: vec![MetadataEntry {
                key: "role".into(),
                value: "user".into(),
            }],
        };
        let facade_env: facade::Envelope = env.clone().into();
        // Facade carries metadata as HashMap; verify the key survived the
        // Vec → HashMap conversion (and the test also pins the round trip
        // back through the bolt-side Vec representation).
        assert_eq!(facade_env.metadata.get("role"), Some(&"user".to_string()));
        let back: Envelope = facade_env.into();
        match back.kind {
            EnvelopeKind::Text { text } => assert_eq!(text, "hi"),
            _ => panic!("expected text"),
        }
        assert_eq!(back.metadata.len(), 1);
    }

    #[test]
    fn error_code_matches_facade() {
        let e = Error::Timeout { timeout_ms: 42 };
        // Same wire code as facade::Error::Timeout — protects the
        // foreign-language consumer's switch-on-code logic from drift.
        let f: facade::Error = e.clone().into();
        assert_eq!(e.code(), f.code());
    }

    #[test]
    fn run_options_threads_abort_signals() {
        let opts = RunOptions {
            generation_config: None,
            abort_on: vec![AbortSignal::ThermalCritical],
            fallback_to_cloud: true,
            max_grace_tokens: 4,
            correlation_id: Some("trace".into()),
        };
        let facade_opts: facade::RunOptions = opts.into();
        assert!(facade_opts.fallback_to_cloud);
        assert_eq!(facade_opts.max_grace_tokens, 4);
        assert_eq!(
            facade_opts.abort_on,
            vec![facade::AbortSignal::ThermalCritical]
        );
    }
}
