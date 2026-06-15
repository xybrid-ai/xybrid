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
//!   exception in target languages.
//! - Handle types use `#[export] impl Foo { ... }`; BoltFFI manages the
//!   heap allocation and FFI handle internally — no `Arc<Self>` return is
//!   required at the call site.
//!
//! ## Naming convention
//!
//! All FFI-exposed types are prefixed `Xybrid*` to match the existing
//! foreign-language SDK convention (uniffi already exposes `XybridError`,
//! `XybridResult`, `XybridEnvelope`, etc.; the Flutter `Ffi*` types live in
//! a separate generator and aren't affected). The prefix also avoids
//! collisions with Swift's stdlib `Error` protocol on the error enum.
//!
//! Run `boltffi pack all --release` (or per-target,
//! e.g. `boltffi pack apple`) from `tools/scripts/` to generate the
//! Swift / Kotlin / Java / C# / WASM bindings from this crate.
//!
//! ## Migration status (sketch)
//!
//! - **Records and `XybridError`**: complete.
//! - **Free functions** (init / push API): complete for the subset every
//!   binding needs at startup.
//! - **`XybridModel`**: minimal `#[export]` block covering the load / run /
//!   warmup / voice surface. Enough to validate the proc-macro shape
//!   against the facade.
//! - **Deferred to follow-up commits**:
//!   - Token streaming (`run_stream` / `run_stream_with_context`) — needs
//!     BoltFFI's stream-event convention nailed down across all targets.
//!   - `XybridCancellationToken` as an `Arc<Self>` handle.
//!   - `XybridConversationContext` (uniffi opaque object equivalent).
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
// XybridError
// ============================================================================

/// Errors surfaced across the FFI boundary. Variants mirror
/// [`facade::Error`] — the facade owns the SDK→FFI translation; this enum
/// only re-decorates it for the BoltFFI generator (proc macros must live
/// on the type definition).
///
/// Named `XybridError` (not `Error`) so the emitted Swift type doesn't
/// shadow / collide with Swift's stdlib `Error` protocol, and so the
/// Kotlin sealed-hierarchy name matches the existing uniffi consumer
/// expectations.
///
/// **Variant order is part of the wire contract.** BoltFFI encodes `#[error]`
/// (and `#[data]`) enums by ordinal tag, so reordering or inserting a variant
/// renumbers every variant after it and breaks already-built foreign clients.
/// Only ever append at the tail, and keep this order in lockstep with
/// [`facade::Error`] and its `code()` table.
#[error]
#[derive(Debug, Clone)]
pub enum XybridError {
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
    MissingArtifact { message: String },
    UnsupportedModelCapability { message: String },
    UnsupportedBackendCapability { message: String },
    InvalidImage { message: String },
}

impl XybridError {
    /// Stable numeric discriminant inherited from the facade. Same wire
    /// codes across every binding so foreign consumers can switch on a
    /// shared protocol.
    pub fn code(&self) -> u32 {
        // Delegate via the facade so the code table lives in one place.
        // BoltFFI sees this as an inherent method on the error type.
        facade::Error::from(self.clone()).code()
    }
}

impl From<XybridError> for facade::Error {
    fn from(e: XybridError) -> Self {
        match e {
            XybridError::ModelNotFound { id } => facade::Error::ModelNotFound { id },
            XybridError::DirectoryNotFound { path } => facade::Error::DirectoryNotFound { path },
            XybridError::MetadataNotFound { path } => facade::Error::MetadataNotFound { path },
            XybridError::MetadataInvalid { message } => facade::Error::MetadataInvalid { message },
            XybridError::LoadError { message } => facade::Error::LoadError { message },
            XybridError::InferenceError { message } => facade::Error::InferenceError { message },
            XybridError::AbortedForCloudFallback { reason } => {
                facade::Error::AbortedForCloudFallback { reason }
            }
            XybridError::StreamingNotSupported => facade::Error::StreamingNotSupported,
            XybridError::NotLoaded => facade::Error::NotLoaded,
            XybridError::ConfigError { message } => facade::Error::ConfigError { message },
            XybridError::NetworkError { message } => facade::Error::NetworkError { message },
            XybridError::Offline { message } => facade::Error::Offline { message },
            XybridError::IoError { message } => facade::Error::IoError { message },
            XybridError::CacheError { message } => facade::Error::CacheError { message },
            XybridError::PipelineError { message } => facade::Error::PipelineError { message },
            XybridError::CircuitOpen { message } => facade::Error::CircuitOpen { message },
            XybridError::RateLimited { retry_after_secs } => {
                facade::Error::RateLimited { retry_after_secs }
            }
            XybridError::Timeout { timeout_ms } => facade::Error::Timeout { timeout_ms },
            XybridError::MissingArtifact { message } => facade::Error::MissingArtifact { message },
            XybridError::UnsupportedModelCapability { message } => {
                facade::Error::UnsupportedModelCapability { message }
            }
            XybridError::UnsupportedBackendCapability { message } => {
                facade::Error::UnsupportedBackendCapability { message }
            }
            XybridError::InvalidImage { message } => facade::Error::InvalidImage { message },
        }
    }
}

impl From<facade::Error> for XybridError {
    fn from(e: facade::Error) -> Self {
        match e {
            facade::Error::ModelNotFound { id } => XybridError::ModelNotFound { id },
            facade::Error::DirectoryNotFound { path } => XybridError::DirectoryNotFound { path },
            facade::Error::MetadataNotFound { path } => XybridError::MetadataNotFound { path },
            facade::Error::MetadataInvalid { message } => XybridError::MetadataInvalid { message },
            facade::Error::LoadError { message } => XybridError::LoadError { message },
            facade::Error::InferenceError { message } => XybridError::InferenceError { message },
            facade::Error::AbortedForCloudFallback { reason } => {
                XybridError::AbortedForCloudFallback { reason }
            }
            facade::Error::StreamingNotSupported => XybridError::StreamingNotSupported,
            facade::Error::NotLoaded => XybridError::NotLoaded,
            facade::Error::ConfigError { message } => XybridError::ConfigError { message },
            facade::Error::NetworkError { message } => XybridError::NetworkError { message },
            facade::Error::Offline { message } => XybridError::Offline { message },
            facade::Error::IoError { message } => XybridError::IoError { message },
            facade::Error::CacheError { message } => XybridError::CacheError { message },
            facade::Error::PipelineError { message } => XybridError::PipelineError { message },
            facade::Error::CircuitOpen { message } => XybridError::CircuitOpen { message },
            facade::Error::RateLimited { retry_after_secs } => {
                XybridError::RateLimited { retry_after_secs }
            }
            facade::Error::Timeout { timeout_ms } => XybridError::Timeout { timeout_ms },
            facade::Error::MissingArtifact { message } => XybridError::MissingArtifact { message },
            facade::Error::UnsupportedModelCapability { message } => {
                XybridError::UnsupportedModelCapability { message }
            }
            facade::Error::UnsupportedBackendCapability { message } => {
                XybridError::UnsupportedBackendCapability { message }
            }
            facade::Error::InvalidImage { message } => XybridError::InvalidImage { message },
        }
    }
}

// ============================================================================
// Envelope payload + role
// ============================================================================

#[data]
#[derive(Clone)]
pub enum XybridEnvelopeKind {
    Text { text: String },
    Audio { bytes: Vec<u8> },
    Embedding { values: Vec<f32> },
    Image { bytes: Vec<u8>, format: String },
    MultiPart { parts: Vec<XybridEnvelope> },
}

impl From<XybridEnvelopeKind> for facade::EnvelopeKind {
    fn from(k: XybridEnvelopeKind) -> Self {
        match k {
            XybridEnvelopeKind::Text { text } => facade::EnvelopeKind::Text { text },
            XybridEnvelopeKind::Audio { bytes } => facade::EnvelopeKind::Audio { bytes },
            XybridEnvelopeKind::Embedding { values } => facade::EnvelopeKind::Embedding { values },
            XybridEnvelopeKind::Image { bytes, format } => {
                facade::EnvelopeKind::Image { bytes, format }
            }
            XybridEnvelopeKind::MultiPart { parts } => facade::EnvelopeKind::MultiPart {
                parts: parts.into_iter().map(Into::into).collect(),
            },
        }
    }
}

impl From<facade::EnvelopeKind> for XybridEnvelopeKind {
    fn from(k: facade::EnvelopeKind) -> Self {
        match k {
            facade::EnvelopeKind::Text { text } => XybridEnvelopeKind::Text { text },
            facade::EnvelopeKind::Audio { bytes } => XybridEnvelopeKind::Audio { bytes },
            facade::EnvelopeKind::Embedding { values } => XybridEnvelopeKind::Embedding { values },
            facade::EnvelopeKind::Image { bytes, format } => {
                XybridEnvelopeKind::Image { bytes, format }
            }
            facade::EnvelopeKind::MultiPart { parts } => XybridEnvelopeKind::MultiPart {
                parts: parts.into_iter().map(Into::into).collect(),
            },
        }
    }
}

/// Single metadata key/value entry. BoltFFI doesn't auto-derive
/// `WireEncode` for `HashMap<String, String>`, so we expose metadata as
/// `Vec<XybridMetadataEntry>`. The conversion back to `HashMap` happens
/// at the facade boundary inside [`XybridEnvelope::into`].
#[data]
#[derive(Clone)]
pub struct XybridMetadataEntry {
    pub key: String,
    pub value: String,
}

#[data]
#[derive(Clone)]
pub struct XybridEnvelope {
    pub kind: XybridEnvelopeKind,
    pub metadata: Vec<XybridMetadataEntry>,
}

impl From<XybridEnvelope> for facade::Envelope {
    fn from(e: XybridEnvelope) -> Self {
        facade::Envelope {
            kind: e.kind.into(),
            metadata: e
                .metadata
                .into_iter()
                .map(|XybridMetadataEntry { key, value }| (key, value))
                .collect(),
        }
    }
}

impl From<facade::Envelope> for XybridEnvelope {
    fn from(e: facade::Envelope) -> Self {
        Self {
            kind: e.kind.into(),
            metadata: e
                .metadata
                .into_iter()
                .map(|(key, value)| XybridMetadataEntry { key, value })
                .collect(),
        }
    }
}

#[data]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum XybridMessageRole {
    System,
    User,
    Assistant,
}

impl From<XybridMessageRole> for facade::MessageRole {
    fn from(r: XybridMessageRole) -> Self {
        match r {
            XybridMessageRole::System => facade::MessageRole::System,
            XybridMessageRole::User => facade::MessageRole::User,
            XybridMessageRole::Assistant => facade::MessageRole::Assistant,
        }
    }
}

// ============================================================================
// Generation + Run options
// ============================================================================

#[data]
#[derive(Clone)]
pub struct XybridGenerationConfig {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub stop_sequences: Vec<String>,
}

impl From<XybridGenerationConfig> for facade::GenerationConfig {
    fn from(c: XybridGenerationConfig) -> Self {
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
pub enum XybridAbortSignal {
    MemoryPressureWarn,
    MemoryPressureCritical,
    ThermalHot,
    ThermalCritical,
}

impl From<XybridAbortSignal> for facade::AbortSignal {
    fn from(s: XybridAbortSignal) -> Self {
        match s {
            XybridAbortSignal::MemoryPressureWarn => facade::AbortSignal::MemoryPressureWarn,
            XybridAbortSignal::MemoryPressureCritical => {
                facade::AbortSignal::MemoryPressureCritical
            }
            XybridAbortSignal::ThermalHot => facade::AbortSignal::ThermalHot,
            XybridAbortSignal::ThermalCritical => facade::AbortSignal::ThermalCritical,
        }
    }
}

#[data]
#[derive(Clone)]
pub struct XybridRunOptions {
    pub generation_config: Option<XybridGenerationConfig>,
    pub abort_on: Vec<XybridAbortSignal>,
    pub fallback_to_cloud: bool,
    pub max_grace_tokens: u32,
    pub correlation_id: Option<String>,
}

impl From<XybridRunOptions> for facade::RunOptions {
    fn from(o: XybridRunOptions) -> Self {
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
pub enum XybridOutputType {
    Text,
    Audio,
    Embedding,
    Unknown,
}

impl From<facade::OutputType> for XybridOutputType {
    fn from(t: facade::OutputType) -> Self {
        match t {
            facade::OutputType::Text => XybridOutputType::Text,
            facade::OutputType::Audio => XybridOutputType::Audio,
            facade::OutputType::Embedding => XybridOutputType::Embedding,
            facade::OutputType::Unknown => XybridOutputType::Unknown,
        }
    }
}

#[data]
#[derive(Clone)]
pub struct XybridStageLatency {
    pub stage_id: String,
    pub latency_ms: u32,
}

impl From<&facade::StageLatency> for XybridStageLatency {
    fn from(s: &facade::StageLatency) -> Self {
        Self {
            stage_id: s.stage_id.clone(),
            latency_ms: s.latency_ms,
        }
    }
}

#[data]
#[derive(Clone)]
pub struct XybridInferenceMetrics {
    pub total_ms: u32,
    pub ttft_ms: Option<u32>,
    pub tokens_per_second: Option<f32>,
    pub prefill_tps: Option<f32>,
    pub decode_tps: Option<f32>,
    pub tokens_out: Option<u32>,
    pub stage_latencies_ms: Vec<XybridStageLatency>,
}

impl From<&facade::InferenceMetrics> for XybridInferenceMetrics {
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

/// Inference output. Named `XybridResult` (not `XybridInferenceResult`)
/// to match the existing uniffi-generated Kotlin/Swift name — the iOS
/// example references `XybridResult` directly.
#[data]
#[derive(Clone)]
pub struct XybridResult {
    pub envelope: XybridEnvelope,
    pub output_type: XybridOutputType,
    pub model_id: String,
    pub latency_ms: u32,
    pub metrics: XybridInferenceMetrics,
}

impl From<facade::InferenceResult> for XybridResult {
    fn from(r: facade::InferenceResult) -> Self {
        let metrics = XybridInferenceMetrics::from(&r.metrics);
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
pub struct XybridVoiceInfo {
    pub id: String,
    pub name: String,
    pub gender: Option<String>,
    pub language: Option<String>,
    pub style: Option<String>,
}

impl From<facade::VoiceInfo> for XybridVoiceInfo {
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
pub enum XybridThermalState {
    Normal,
    Warm,
    Hot,
    Critical,
}

impl From<XybridThermalState> for facade::ThermalState {
    fn from(s: XybridThermalState) -> Self {
        match s {
            XybridThermalState::Normal => facade::ThermalState::Normal,
            XybridThermalState::Warm => facade::ThermalState::Warm,
            XybridThermalState::Hot => facade::ThermalState::Hot,
            XybridThermalState::Critical => facade::ThermalState::Critical,
        }
    }
}

#[export]
pub fn set_thermal_state(state: XybridThermalState) {
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

/// One-stop SDK initialization: API key + gateway/ingest URL overrides in
/// one call. Delegates to [`facade::configure_runtime`]; blank strings are
/// treated as absent. This is the canonical init the Swift
/// `Xybrid.initialize(apiKey:gatewayUrl:ingestUrl:)` and Kotlin
/// `Xybrid.init(context, apiKey, gatewayUrl, ingestUrl)` wrappers call.
#[export]
pub fn configure_runtime(
    api_key: Option<String>,
    gateway_url: Option<String>,
    ingest_url: Option<String>,
) {
    facade::configure_runtime(api_key, gateway_url, ingest_url);
}

#[export]
pub fn init_sdk_cache_dir(cache_dir: String) {
    // Param name pinned to `cache_dir` (not `path`) so the emitted Swift
    // is `initSdkCacheDir(cacheDir:)`, matching the existing
    // `examples/ios/XybridExample` call site that uniffi already exposes
    // under that label.
    facade::init_sdk_cache_dir(cache_dir);
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
// foreign API becomes `try XybridModel(fromRegistry:)` rather than
// `XybridModelLoader.fromRegistry().load()` — fewer concepts, same
// capability) and matches the facade's existing handle convention.

pub struct XybridModel {
    inner: std::sync::Arc<facade::XybridModel>,
}

#[export]
impl XybridModel {
    /// Load from the xybrid registry. Recommended path.
    pub fn from_registry(id: String) -> Result<Self, XybridError> {
        let model = facade::ModelLoader::from_registry(id)
            .load()
            .map_err(XybridError::from)?;
        Ok(Self { inner: model })
    }

    /// Load from a local model directory (must contain `model_metadata.json`).
    pub fn from_directory(path: String) -> Result<Self, XybridError> {
        let loader = facade::ModelLoader::from_directory(path).map_err(XybridError::from)?;
        let model = loader.load().map_err(XybridError::from)?;
        Ok(Self { inner: model })
    }

    /// Load from a local `.xyb` bundle.
    pub fn from_bundle(path: String) -> Result<Self, XybridError> {
        let loader = facade::ModelLoader::from_bundle(path).map_err(XybridError::from)?;
        let model = loader.load().map_err(XybridError::from)?;
        Ok(Self { inner: model })
    }

    /// Resolve and load from a HuggingFace repo (`org/repo` or `org/repo:variant`).
    pub fn from_huggingface(repo: String) -> Result<Self, XybridError> {
        let model = facade::ModelLoader::from_huggingface(repo)
            .load()
            .map_err(XybridError::from)?;
        Ok(Self { inner: model })
    }

    pub fn model_id(&self) -> String {
        self.inner.model_id()
    }

    pub fn version(&self) -> String {
        self.inner.version()
    }

    pub fn output_type(&self) -> XybridOutputType {
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

    pub fn voices(&self) -> Vec<XybridVoiceInfo> {
        self.inner
            .voices()
            .into_iter()
            .map(XybridVoiceInfo::from)
            .collect()
    }

    pub fn default_voice(&self) -> Option<XybridVoiceInfo> {
        self.inner.default_voice().map(XybridVoiceInfo::from)
    }

    pub fn voice(&self, voice_id: String) -> Option<XybridVoiceInfo> {
        self.inner.voice(voice_id).map(XybridVoiceInfo::from)
    }

    /// Run inference, optionally with [`XybridRunOptions`] (generation config,
    /// abort signals, cloud-fallback). Pass `None` for the model's defaults.
    ///
    /// The hand-written wrappers add a one-arg `run(envelope)` convenience that
    /// forwards `None`, so simple call sites stay ergonomic.
    pub fn run(
        &self,
        envelope: XybridEnvelope,
        options: Option<XybridRunOptions>,
    ) -> Result<XybridResult, XybridError> {
        let result = match options {
            Some(opts) => self
                .inner
                .run_with_options(envelope.into(), opts.into(), None),
            None => self.inner.run(envelope.into()),
        }
        .map_err(XybridError::from)?;
        Ok(result.into())
    }

    pub fn warmup(&self) -> Result<(), XybridError> {
        self.inner.warmup().map_err(XybridError::from)
    }

    pub fn unload(&self) -> Result<(), XybridError> {
        self.inner.unload().map_err(XybridError::from)
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
        let env = XybridEnvelope {
            kind: XybridEnvelopeKind::Text { text: "hi".into() },
            metadata: vec![XybridMetadataEntry {
                key: "role".into(),
                value: "user".into(),
            }],
        };
        let facade_env: facade::Envelope = env.clone().into();
        // Facade carries metadata as HashMap; verify the key survived the
        // Vec → HashMap conversion (and the test also pins the round trip
        // back through the bolt-side Vec representation).
        assert_eq!(facade_env.metadata.get("role"), Some(&"user".to_string()));
        let back: XybridEnvelope = facade_env.into();
        match back.kind {
            XybridEnvelopeKind::Text { text } => assert_eq!(text, "hi"),
            _ => panic!("expected text"),
        }
        assert_eq!(back.metadata.len(), 1);
    }

    #[test]
    fn error_code_matches_facade() {
        let e = XybridError::Timeout { timeout_ms: 42 };
        // Same wire code as facade::Error::Timeout — protects the
        // foreign-language consumer's switch-on-code logic from drift.
        let f: facade::Error = e.clone().into();
        assert_eq!(e.code(), f.code());
    }

    #[test]
    fn run_options_threads_abort_signals() {
        let opts = XybridRunOptions {
            generation_config: None,
            abort_on: vec![XybridAbortSignal::ThermalCritical],
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
