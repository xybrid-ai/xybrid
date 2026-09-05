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
//! - **`XybridModel`**: load / run / pull-stream / conversation-context runs /
//!   warmup / voice surface.
//! - **Tool calling**: `XybridToolDefinition` on the generation config,
//!   `XybridToolCall` on the result, and `tool_results_envelope` for the
//!   continuation turn. One `run` is one model turn — the loop lives in the
//!   caller's code, not behind a cross-boundary callback.
//! - **`XybridConversationContext`**: opaque handle (new / with_id / push /
//!   set_system / clear / id) feeding `run_with_context` and
//!   `run_stream_with_context`.
//! - **Deferred to follow-up commits**:
//!   - `XybridCancellationToken` as an `Arc<Self>` handle.
//!   - Pipeline surface.
//!
//! This is now the sole native binding crate: `xybrid-uniffi` and the
//! pre-bolt `xybrid-ffi` C ABI have both been removed, and every foreign SDK
//! (Swift / Kotlin / Unity C# / …) rides bolt via [`xybrid_ffi_facade`].

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
        let mut metadata: Vec<_> = e
            .metadata
            .into_iter()
            .map(|(key, value)| XybridMetadataEntry { key, value })
            .collect();
        metadata.sort_unstable_by(|left, right| left.key.cmp(&right.key));
        Self {
            kind: e.kind.into(),
            metadata,
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
// Tool calling
// ============================================================================

/// A tool (function) the model may ask to call.
///
/// `parameters_json` is the JSON Schema for the arguments, carried as a JSON
/// string because no binding generator can describe an arbitrary JSON tree.
#[data]
#[derive(Clone)]
pub struct XybridToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters_json: String,
}

impl From<XybridToolDefinition> for facade::ToolDefinition {
    fn from(t: XybridToolDefinition) -> Self {
        Self {
            name: t.name,
            description: t.description,
            parameters_json: t.parameters_json,
        }
    }
}

impl From<facade::ToolDefinition> for XybridToolDefinition {
    fn from(t: facade::ToolDefinition) -> Self {
        Self {
            name: t.name,
            description: t.description,
            parameters_json: t.parameters_json,
        }
    }
}

/// One tool call the model emitted, from [`XybridResult::tool_calls`].
#[data]
#[derive(Clone)]
pub struct XybridToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

impl From<facade::ToolCall> for XybridToolCall {
    fn from(c: facade::ToolCall) -> Self {
        Self {
            id: c.id,
            name: c.name,
            arguments_json: c.arguments_json,
        }
    }
}

/// The outcome of running one tool, fed back with [`tool_results_envelope`].
#[data]
#[derive(Clone)]
pub struct XybridToolResult {
    /// The [`XybridToolCall::id`] this answers.
    pub call_id: String,
    pub name: String,
    /// The tool's output as a JSON string.
    pub content_json: String,
}

impl From<XybridToolResult> for facade::ToolResult {
    fn from(r: XybridToolResult) -> Self {
        Self {
            call_id: r.call_id,
            name: r.name,
            content_json: r.content_json,
        }
    }
}

/// Build the continuation envelope for the turn after the model asked for
/// tools.
///
/// One `run` is one model turn, so the loop lives in your code: run a
/// tools-bearing request, execute every [`XybridToolCall`] it returns, then
/// run this envelope to feed the outcomes back. Pass the same tools on the
/// continuation's [`XybridGenerationConfig`] as on the original turn.
///
/// A free function rather than a constructor because `XybridEnvelope` is a
/// `#[data]` record, not a handle type — records carry no methods across the
/// generated bindings.
#[export]
pub fn tool_results_envelope(
    user_text: String,
    prior_assistant_text: String,
    results: Vec<XybridToolResult>,
) -> Result<XybridEnvelope, XybridError> {
    facade::Envelope::tool_results(
        user_text,
        prior_assistant_text,
        results.into_iter().map(Into::into).collect(),
    )
    .map(Into::into)
    .map_err(XybridError::from)
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
    /// Optional GBNF grammar constraining generation to structured output
    /// (local llama backend only). Produce one from a JSON Schema with
    /// [`json_schema_to_gbnf`], or pass raw GBNF. Appended last: `#[data]`
    /// PODs serialize by field order across the FFI boundary.
    pub grammar: Option<String>,
    /// Tools the model may call this turn. Empty means no tool calling —
    /// existing behavior, unchanged. Appended after `grammar` for the same
    /// field-order reason.
    ///
    /// Tool calling is llama.cpp-only today; unsupported paths (no embedded
    /// chat template, the mistralrs backend, the cloud fallback leg) reject
    /// tool-bearing requests rather than quietly generating without them.
    pub tools: Vec<XybridToolDefinition>,
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
            grammar: c.grammar,
            tools: c.tools.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<facade::GenerationConfig> for XybridGenerationConfig {
    fn from(config: facade::GenerationConfig) -> Self {
        let facade::GenerationConfig {
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
        Self {
            max_tokens,
            temperature,
            top_p,
            min_p,
            top_k,
            repetition_penalty,
            stop_sequences,
            grammar,
            tools: tools.into_iter().map(Into::into).collect(),
        }
    }
}

/// Convert a JSON Schema (as a JSON string) into a GBNF grammar for
/// [`XybridGenerationConfig::grammar`]. Fails on invalid JSON or schema
/// constructs outside the supported subset.
#[export]
pub fn json_schema_to_gbnf(schema_json: String) -> Result<String, XybridError> {
    facade::json_schema_to_gbnf(&schema_json).map_err(XybridError::from)
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
    /// Where the answer actually came from. Cloud fallback keeps `model_id`
    /// identical on both legs, so this is the only way to tell them apart.
    pub execution_target: XybridExecutionTarget,
    pub metrics: XybridInferenceMetrics,
    /// Tool calls the model emitted this turn. Empty unless the request
    /// offered tools via [`XybridGenerationConfig::tools`].
    /// `#[data]` PODs serialize by field order across the FFI boundary.
    pub tool_calls: Vec<XybridToolCall>,
    /// Model reasoning emitted separately from the final answer text.
    /// Appended last because `#[data]` fields serialize in declaration order.
    pub reasoning_content: Option<String>,
}

/// Where a result was produced — observed fact, not a routing preference.
#[data]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XybridExecutionTarget {
    Local,
    Cloud,
}

impl From<facade::ExecutionTarget> for XybridExecutionTarget {
    fn from(target: facade::ExecutionTarget) -> Self {
        match target {
            facade::ExecutionTarget::Local => Self::Local,
            facade::ExecutionTarget::Cloud => Self::Cloud,
        }
    }
}

/// Lifecycle of the background download behind a speculative load.
#[data]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XybridDownloadState {
    Downloading,
    Ready,
    /// Download failed; the cloud keeps serving and `isLoaded` never flips.
    Failed,
}

/// Download progress + state in one consistent read.
#[data]
#[derive(Clone, Debug, PartialEq)]
pub struct XybridDownloadStatus {
    pub state: XybridDownloadState,
    /// 0.0..=1.0.
    pub progress: f32,
}

impl From<facade::DownloadStatus> for XybridDownloadStatus {
    fn from(status: facade::DownloadStatus) -> Self {
        let state = match status.state {
            facade::DownloadState::Downloading => XybridDownloadState::Downloading,
            facade::DownloadState::Ready => XybridDownloadState::Ready,
            facade::DownloadState::Failed => XybridDownloadState::Failed,
        };
        Self {
            state,
            progress: status.progress,
        }
    }
}

impl From<facade::InferenceResult> for XybridResult {
    fn from(r: facade::InferenceResult) -> Self {
        let metrics = XybridInferenceMetrics::from(&r.metrics);
        let reasoning_content = r.envelope.metadata.get("reasoning_content").cloned();
        Self {
            envelope: r.envelope.into(),
            output_type: r.output_type.into(),
            model_id: r.model_id,
            latency_ms: r.latency_ms,
            execution_target: r.execution_target.into(),
            metrics,
            tool_calls: r.tool_calls.into_iter().map(Into::into).collect(),
            reasoning_content,
        }
    }
}

// ============================================================================
// Pull-based token streaming
// ============================================================================

#[data]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XybridStreamEventKind {
    Token,
    Complete,
}

#[data]
#[derive(Clone)]
pub struct XybridStreamToken {
    pub token: String,
    pub token_id: Option<i64>,
    pub index: u64,
    pub cumulative_text: String,
    /// `"tool_calls"` when the turn ended on a parseable tool-call block.
    pub finish_reason: Option<String>,
    /// Tool calls parsed from the completed turn — populated on the
    /// **terminal** token only (the one carrying `finish_reason`).
    ///
    /// Tool-call blocks are suppressed from the emitted stream, so there is
    /// nothing in the token text to parse: a streaming caller halts here,
    /// runs the tools, then continues the turn by streaming a
    /// [`tool_results_envelope`] through the same call. Empty on every
    /// mid-stream token and on turns that emitted no call.
    pub tool_calls: Vec<XybridToolCall>,
    /// The completed turn's raw output text, tool-call block included — pass
    /// it to [`tool_results_envelope`] as `prior_assistant_text`.
    ///
    /// Present only alongside a non-empty [`Self::tool_calls`]. Not the same
    /// as `cumulative_text`, which reports the *emitted* text with the
    /// protocol blocks suppressed — which is why this field exists at all.
    pub raw_text: Option<String>,
}

impl From<facade::StreamToken> for XybridStreamToken {
    fn from(token: facade::StreamToken) -> Self {
        Self {
            token: token.token,
            token_id: token.token_id,
            index: token.index,
            cumulative_text: token.cumulative_text,
            finish_reason: token.finish_reason,
            tool_calls: token.tool_calls.into_iter().map(Into::into).collect(),
            raw_text: token.raw_text,
        }
    }
}

/// One pull from a streaming inference session.
///
/// This is a flat record instead of a data-carrying enum because the pinned
/// C# generator cannot lower that enum shape reliably. `kind` selects the one
/// populated payload: `token` for `Token`, none for `Complete`. A `Complete`
/// event is followed by [`XybridModel::stream_result`] to retrieve the final
/// result. Inference failures are returned as typed [`XybridError`] values by
/// [`XybridModel::stream_next`].
#[data]
#[derive(Clone)]
pub struct XybridStreamEvent {
    pub kind: XybridStreamEventKind,
    pub token: Option<XybridStreamToken>,
}

impl From<facade::StreamEvent> for XybridStreamEvent {
    fn from(event: facade::StreamEvent) -> Self {
        match event {
            facade::StreamEvent::Token(token) => Self {
                kind: XybridStreamEventKind::Token,
                token: Some(token.into()),
            },
            facade::StreamEvent::Complete(_) => Self {
                kind: XybridStreamEventKind::Complete,
                token: None,
            },
            facade::StreamEvent::Error(_) => {
                unreachable!("stream errors are returned before event conversion")
            }
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
// Model cache management
// ============================================================================

#[data]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum XybridCacheEntryLocation {
    Registry,
    Extracted,
    HuggingFace,
    HuggingFaceHub,
}

impl From<facade::CacheEntryLocation> for XybridCacheEntryLocation {
    fn from(location: facade::CacheEntryLocation) -> Self {
        match location {
            facade::CacheEntryLocation::Registry => Self::Registry,
            facade::CacheEntryLocation::Extracted => Self::Extracted,
            facade::CacheEntryLocation::HuggingFace => Self::HuggingFace,
            facade::CacheEntryLocation::HuggingFaceHub => Self::HuggingFaceHub,
        }
    }
}

#[data]
#[derive(Clone)]
pub struct XybridCacheEntry {
    pub model_id: String,
    pub location: XybridCacheEntryLocation,
    pub path: String,
    pub size_bytes: u64,
}

impl From<facade::CacheEntry> for XybridCacheEntry {
    fn from(entry: facade::CacheEntry) -> Self {
        Self {
            model_id: entry.model_id,
            location: entry.location.into(),
            path: entry.path,
            size_bytes: entry.size_bytes,
        }
    }
}

#[data]
#[derive(Clone)]
pub struct XybridCacheStatus {
    pub total_size_bytes: u64,
    pub entry_count: u32,
    pub model_count: u32,
    pub extracted_model_count: u32,
    pub cache_root: String,
}

impl From<facade::CacheStatus> for XybridCacheStatus {
    fn from(status: facade::CacheStatus) -> Self {
        Self {
            total_size_bytes: status.total_size_bytes,
            entry_count: status.entry_count,
            model_count: status.model_count,
            extracted_model_count: status.extracted_model_count,
            cache_root: status.cache_root,
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

/// Initialize the platform-native `log` backend exactly once per process.
///
/// Mirrors the Flutter binding's `ensure_native_logging` (see
/// `bindings/flutter/rust/src/api/mod.rs`): without a registered logger every
/// `log::warn!` in the SDK — telemetry send failures and registry failovers
/// in particular — is silently discarded on device. Called from the
/// process-global init entry points the Swift/Kotlin wrappers hit during
/// `Xybrid.initialize` / `Xybrid.init`. No-op on desktop targets, where the
/// host process owns logger setup.
fn ensure_native_logging() {
    static LOGGING_INIT: std::sync::Once = std::sync::Once::new();
    LOGGING_INIT.call_once(|| {
        #[cfg(target_os = "android")]
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(log::LevelFilter::Info)
                .with_tag("xybrid"),
        );
        #[cfg(target_os = "ios")]
        {
            // Errors only if a logger is already registered — fine to ignore.
            let _ = oslog::OsLogger::new("dev.xybrid.sdk")
                .level_filter(log::LevelFilter::Info)
                .init();
        }
    });
}

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
    ensure_native_logging();
    facade::configure_runtime(api_key, gateway_url, ingest_url);
}

#[export]
pub fn init_sdk_cache_dir(cache_dir: String) {
    ensure_native_logging();
    // Param name pinned to `cache_dir` (not `path`) so the emitted Swift
    // is `initSdkCacheDir(cacheDir:)`, matching the existing
    // `examples/ios/XybridExample` call site that uniffi already exposes
    // under that label.
    facade::init_sdk_cache_dir(cache_dir);
}

/// Returns aggregate storage usage across every managed model-cache location.
#[export]
pub fn cache_status() -> Result<XybridCacheStatus, XybridError> {
    facade::cache_status()
        .map(Into::into)
        .map_err(XybridError::from)
}

/// Lists every physical model entry occupying managed cache storage.
#[export]
pub fn cache_entries() -> Result<Vec<XybridCacheEntry>, XybridError> {
    facade::cache_entries()
        .map(|entries| entries.into_iter().map(Into::into).collect())
        .map_err(XybridError::from)
}

/// Returns whether a model occupies any managed cache entry.
#[export]
pub fn cache_is_model_cached(model_id: String) -> Result<bool, XybridError> {
    facade::cache_is_model_cached(model_id).map_err(XybridError::from)
}

/// Resolves the preferred local cache path for a model, if present.
#[export]
pub fn cache_model_path(model_id: String) -> Result<Option<String>, XybridError> {
    facade::cache_model_path(model_id).map_err(XybridError::from)
}

/// Lists model IDs extracted, validated, and ready to run offline.
#[export]
pub fn cache_list_extracted_model_ids() -> Result<Vec<String>, XybridError> {
    facade::cache_list_extracted_model_ids().map_err(XybridError::from)
}

/// Reports a configuration error until persistent cache retention is supported.
#[export]
pub fn cache_clean_expired() -> Result<u32, XybridError> {
    facade::cache_clean_expired().map_err(XybridError::from)
}

/// Removes every managed cache entry for one model.
///
/// Do not call concurrently with a load of the same model.
#[export]
pub fn cache_remove_model(model_id: String) -> Result<u32, XybridError> {
    facade::cache_remove_model(model_id).map_err(XybridError::from)
}

/// Clears all managed model-cache storage.
///
/// Do not call concurrently with any model load.
#[export]
pub fn cache_clear() -> Result<u32, XybridError> {
    facade::cache_clear().map_err(XybridError::from)
}

#[export]
pub fn set_binding(binding: String) {
    ensure_native_logging();
    facade::set_binding(binding);
}

#[export]
pub fn set_api_key(api_key: String) {
    ensure_native_logging();
    facade::set_api_key(api_key);
}

#[export]
pub fn set_provider_api_key(provider: String, api_key: String) {
    ensure_native_logging();
    facade::set_provider_api_key(provider, api_key);
}

/// Point the cloud gateway at a platform base URL (staging, self-hosted).
/// Pass a bare base URL — the `/v1` suffix is applied internally.
#[export]
pub fn set_platform_url(url: String) {
    ensure_native_logging();
    facade::set_platform_url(url);
}

/// Enable speculative cloud fallback globally: a registry model that isn't
/// downloaded yet is served from the gateway while the weights download.
///
/// LLM/chat only — prefer `XybridModel.fromRegistrySpeculative` when the app
/// also loads ASR/TTS models, which cannot be served this way.
#[export]
pub fn set_speculative_cloud(enabled: bool) {
    ensure_native_logging();
    facade::set_speculative_cloud(enabled);
}

/// Whether a Xybrid gateway API key is resolvable (in-memory or env).
#[export]
pub fn has_api_key() -> bool {
    facade::has_api_key()
}

/// Whether the global speculative-cloud default is on.
#[export]
pub fn is_speculative_cloud_enabled() -> bool {
    facade::is_speculative_cloud_enabled()
}

/// Whether `XybridModel::from_registry_speculative(model_id)` would actually
/// speculate: an API key resolves and the model is not already cached.
///
/// Lets the hand-written Swift/Kotlin loader facades answer "will this
/// speculate?" before loading. Never touches the network.
#[export]
pub fn will_speculate_for_model(model_id: String) -> bool {
    facade::will_speculate_for_model(model_id)
}

/// The SDK version string (tracks `CARGO_PKG_VERSION`).
#[export]
pub fn version() -> String {
    facade::version()
}

/// Release every idle loaded model's memory; returns how many were released.
///
/// Call this from the platform's low-memory hook (`didReceiveMemoryWarning`
/// on iOS, `onTrimMemory` on Android). Models with a run in flight are
/// skipped, and a released model reloads itself on next use — no reload call,
/// no new error to handle.
#[export]
pub fn release_memory() -> u32 {
    facade::release_memory()
}

/// Enable or disable automatic model release for subsequent loads.
///
/// When enabled, loading a model under device memory pressure first releases
/// least-recently-used idle models. Off by default; [`release_memory`] works
/// either way.
#[export]
pub fn set_auto_release(enabled: bool) {
    facade::set_auto_release(enabled);
}

/// Whether automatic model release is enabled process-wide.
#[export]
pub fn is_auto_release_enabled() -> bool {
    facade::is_auto_release_enabled()
}

// ============================================================================
// XybridModel handle
// ============================================================================
//
// Scope: load / run / pull-stream / warmup / unload / voice accessors.
// Cancellation and conversation context remain follow-up work.
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
    streams: std::sync::Mutex<std::collections::HashMap<u64, std::sync::Arc<StreamEntry>>>,
    next_stream_id: std::sync::atomic::AtomicU64,
}

struct StreamEntry {
    session: std::sync::Arc<facade::StreamingSession>,
    result: std::sync::Mutex<Option<facade::InferenceResult>>,
}

impl XybridModel {
    fn new(inner: std::sync::Arc<facade::XybridModel>) -> Self {
        Self {
            inner,
            streams: std::sync::Mutex::new(std::collections::HashMap::new()),
            next_stream_id: std::sync::atomic::AtomicU64::new(1),
        }
    }
}

#[export]
impl XybridModel {
    /// Load from the xybrid registry. Recommended path.
    pub fn from_registry(id: String) -> Result<Self, XybridError> {
        let model = facade::ModelLoader::from_registry(id)
            .load()
            .map_err(XybridError::from)?;
        Ok(Self::new(model))
    }

    /// Load from the registry, serving from the cloud gateway while the weights
    /// download in the background.
    ///
    /// Returns almost immediately instead of blocking on the download. Requires
    /// a resolvable API key and an uncached model; otherwise it behaves exactly
    /// like `from_registry`. Poll `download_status` for progress and
    /// `is_cloud_serving` to know which leg is answering. LLM/chat models only.
    pub fn from_registry_speculative(id: String) -> Result<Self, XybridError> {
        let model = facade::ModelLoader::from_registry_speculative(id)
            .load()
            .map_err(XybridError::from)?;
        Ok(Self::new(model))
    }

    /// Load from a local model directory (must contain `model_metadata.json`).
    pub fn from_directory(path: String) -> Result<Self, XybridError> {
        let loader = facade::ModelLoader::from_directory(path).map_err(XybridError::from)?;
        let model = loader.load().map_err(XybridError::from)?;
        Ok(Self::new(model))
    }

    /// Load from a local `.xyb` bundle.
    pub fn from_bundle(path: String) -> Result<Self, XybridError> {
        let loader = facade::ModelLoader::from_bundle(path).map_err(XybridError::from)?;
        let model = loader.load().map_err(XybridError::from)?;
        Ok(Self::new(model))
    }

    /// Resolve and load from a HuggingFace repo (`org/repo` or `org/repo:variant`).
    pub fn from_huggingface(repo: String) -> Result<Self, XybridError> {
        let model = facade::ModelLoader::from_huggingface(repo)
            .load()
            .map_err(XybridError::from)?;
        Ok(Self::new(model))
    }

    /// Resolve and load a HuggingFace repository pinned to a revision.
    pub fn from_huggingface_with_revision(
        repo: String,
        revision: String,
    ) -> Result<Self, XybridError> {
        let model = facade::ModelLoader::from_huggingface_with_revision(repo, revision)
            .load()
            .map_err(XybridError::from)?;
        Ok(Self::new(model))
    }

    /// Load from a raw GGUF file, auto-generating `model_metadata.json` from the
    /// GGUF header (written next to the file if absent).
    pub fn from_model_file(path: String) -> Result<Self, XybridError> {
        let loader = facade::ModelLoader::from_model_file(path).map_err(XybridError::from)?;
        let model = loader.load().map_err(XybridError::from)?;
        Ok(Self::new(model))
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

    /// Whether runs are currently answered by the cloud because the local
    /// weights are not ready yet. `false` for ordinary local models.
    pub fn is_cloud_serving(&self) -> bool {
        self.inner.is_cloud_serving()
    }

    /// Download progress + state in one read — poll this to drive a progress
    /// bar. Reports `Ready` at 1.0 for an ordinary local model, so hosts need
    /// no special case.
    pub fn download_status(&self) -> XybridDownloadStatus {
        self.inner.download_status().into()
    }

    /// Block until the download finishes or `timeout_ms` elapses, then report
    /// the status. Call it off the UI thread (the same place `from_registry` is
    /// already called). `timeout_ms = 0` makes it a non-blocking read.
    pub fn await_download(&self, timeout_ms: u64) -> XybridDownloadStatus {
        self.inner.await_download(timeout_ms).into()
    }

    pub fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }

    /// Whether this model emits true token-by-token output.
    pub fn supports_token_streaming(&self) -> bool {
        self.inner.supports_token_streaming()
    }

    /// Return the model's resolved generation defaults.
    pub fn default_generation_config(&self) -> XybridGenerationConfig {
        self.inner.default_generation_config().into()
    }

    pub fn is_llm(&self) -> bool {
        self.inner.is_llm()
    }

    /// Whether the model bundle declares local tool-calling support.
    ///
    /// Advisory tri-state: `null` means the bundle says nothing, so the host
    /// cannot tell. Gate tool UI on it; enforcement stays at run time — a
    /// tools-bearing request against a model whose chat template has no tool
    /// support fails as invalid input regardless of what this reports.
    pub fn supports_tool_calling(&self) -> Option<bool> {
        self.inner.supports_tool_calling()
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

    /// Start token streaming and return a model-scoped session identifier.
    ///
    /// The identifier remains valid until the final result is taken, an error
    /// is returned, or [`Self::stream_close`] is called.
    pub fn run_stream(
        &self,
        envelope: XybridEnvelope,
        options: Option<XybridRunOptions>,
    ) -> Result<u64, XybridError> {
        let session = self
            .inner
            .run_stream(
                envelope.into(),
                options.map(Into::into).unwrap_or_default(),
                None,
            )
            .map_err(XybridError::from)?;
        let stream_id = self
            .next_stream_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                stream_id,
                std::sync::Arc::new(StreamEntry {
                    session,
                    result: std::sync::Mutex::new(None),
                }),
            );
        Ok(stream_id)
    }

    /// Block until the next item for `stream_id` is ready.
    pub fn stream_next(&self, stream_id: u64) -> Result<XybridStreamEvent, XybridError> {
        let entry = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&stream_id)
            .cloned()
            .ok_or_else(|| XybridError::InferenceError {
                message: "unknown streaming session".into(),
            })?;
        match entry.session.next() {
            Some(facade::StreamEvent::Complete(result)) => {
                *entry
                    .result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(result);
                Ok(XybridStreamEvent {
                    kind: XybridStreamEventKind::Complete,
                    token: None,
                })
            }
            Some(facade::StreamEvent::Error(error)) => {
                self.stream_close(stream_id);
                Err(error.into())
            }
            Some(event @ facade::StreamEvent::Token(_)) => Ok(event.into()),
            None => {
                self.stream_close(stream_id);
                Err(XybridError::InferenceError {
                    message: "stream ended without a terminal event".into(),
                })
            }
        }
    }

    /// Take the final result after receiving a `Complete` event.
    pub fn stream_result(&self, stream_id: u64) -> Result<XybridResult, XybridError> {
        let entry = self
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&stream_id)
            .cloned()
            .ok_or_else(|| XybridError::InferenceError {
                message: "unknown streaming session".into(),
            })?;
        let result = entry
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .ok_or_else(|| XybridError::InferenceError {
                message: "streaming result is not ready".into(),
            })?;
        self.stream_close(stream_id);
        Ok(result.into())
    }

    /// Forget a streaming session.
    pub fn stream_close(&self, stream_id: u64) {
        self.streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&stream_id);
    }

    /// Run inference seeded with a conversation `context` (multi-turn chat).
    ///
    /// Only the generation config from `options` is applied — abort signals and
    /// cloud fallback are not wired on the context path (matches the facade's
    /// `run_with_context`).
    pub fn run_with_context(
        &self,
        envelope: XybridEnvelope,
        context: &XybridConversationContext,
        options: Option<XybridRunOptions>,
    ) -> Result<XybridResult, XybridError> {
        let generation_config = options
            .and_then(|opts| opts.generation_config)
            .map(Into::into);
        let result = self
            .inner
            .run_with_context(envelope.into(), context.inner.clone(), generation_config)
            .map_err(XybridError::from)?;
        Ok(result.into())
    }

    /// Start context-aware token streaming; returns a model-scoped session id.
    /// The pull protocol is identical to [`Self::run_stream`]
    /// (`stream_next` / `stream_result` / `stream_close`).
    pub fn run_stream_with_context(
        &self,
        envelope: XybridEnvelope,
        context: &XybridConversationContext,
        options: Option<XybridRunOptions>,
    ) -> Result<u64, XybridError> {
        let session = self
            .inner
            .run_stream_with_context(
                envelope.into(),
                context.inner.clone(),
                options.map(Into::into).unwrap_or_default(),
                None,
            )
            .map_err(XybridError::from)?;
        let stream_id = self
            .next_stream_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                stream_id,
                std::sync::Arc::new(StreamEntry {
                    session,
                    result: std::sync::Mutex::new(None),
                }),
            );
        Ok(stream_id)
    }

    pub fn warmup(&self) -> Result<(), XybridError> {
        self.inner.warmup().map_err(XybridError::from)
    }

    pub fn unload(&self) -> Result<(), XybridError> {
        self.inner.unload().map_err(XybridError::from)
    }
}

/// Opaque handle for multi-turn conversation history.
///
/// Build it up with [`push`](Self::push) / [`set_system`](Self::set_system),
/// then pass it to [`XybridModel::run_with_context`] /
/// [`XybridModel::run_stream_with_context`]. Wraps the facade's
/// interior-mutable, thread-safe `ConversationContextHandle`.
pub struct XybridConversationContext {
    inner: std::sync::Arc<facade::ConversationContextHandle>,
}

#[export]
impl XybridConversationContext {
    /// Create an empty conversation context (fresh id).
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            inner: facade::ConversationContextHandle::new(),
        }
    }

    /// Create a context with a caller-supplied id (for telemetry correlation
    /// across turns).
    pub fn with_id(id: String) -> Self {
        Self {
            inner: facade::ConversationContextHandle::with_id(id),
        }
    }

    /// Append a turn — typically a user or assistant message envelope.
    pub fn push(&self, envelope: XybridEnvelope) -> Result<(), XybridError> {
        self.inner.push(envelope.into()).map_err(XybridError::from)
    }

    /// Set the persistent system-prompt envelope (survives [`clear`](Self::clear)).
    pub fn set_system(&self, envelope: XybridEnvelope) -> Result<(), XybridError> {
        self.inner
            .set_system(envelope.into())
            .map_err(XybridError::from)
    }

    /// Drop the history; the system envelope (if any) is preserved.
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// The context id.
    pub fn id(&self) -> String {
        self.inner.id()
    }

    /// Number of history turns (excludes the system envelope).
    pub fn history_len(&self) -> u32 {
        self.inner.history_len()
    }

    /// Return history turns, excluding the persistent system envelope.
    pub fn history(&self) -> Vec<XybridEnvelope> {
        self.inner.history().into_iter().map(Into::into).collect()
    }

    /// Whether a persistent system-prompt envelope is set.
    pub fn has_system(&self) -> bool {
        self.inner.has_system()
    }

    /// Set the max history length before FIFO pruning.
    pub fn set_max_history_len(&self, len: u32) {
        self.inner.set_max_history_len(len);
    }
}

// ============================================================================
// Telemetry (advanced config + lifecycle)
// ============================================================================
//
// Mirrors the pre-bolt C ABI's telemetry surface. The apiKey-only fast path is
// [`configure_runtime`]; this is the advanced builder (batch size, flush
// interval, device label/attributes) plus the init/flush/shutdown lifecycle.
// Telemetry *events* never cross the FFI — only this config/lifecycle control
// plane does. Every setter takes simple scalars/strings, so the whole surface
// generates natively (no hand-port needed).

/// The SDK's default telemetry ingest endpoint (for display alongside a config).
#[export]
pub fn telemetry_default_endpoint() -> String {
    facade::telemetry_default_endpoint()
}

/// Flush pending telemetry events. Safe before init / after shutdown.
#[export]
pub fn telemetry_flush() {
    facade::telemetry_flush();
}

/// Shut down the telemetry exporter. Idempotent.
#[export]
pub fn telemetry_shutdown() {
    facade::telemetry_shutdown();
}

/// Advanced telemetry configuration builder.
///
/// Create with [`new`](Self::new), tune via the setters, then hand to
/// [`telemetry_init`]. Wraps the facade's interior-mutable, thread-safe
/// `TelemetryConfigHandle`.
pub struct XybridTelemetryConfig {
    inner: std::sync::Arc<facade::TelemetryConfigHandle>,
}

#[export]
impl XybridTelemetryConfig {
    /// A new config bound to the default ingest endpoint and the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            inner: facade::TelemetryConfigHandle::new(api_key),
        }
    }

    /// Override the ingest endpoint (self-hosted collector / non-prod).
    pub fn set_endpoint(&self, endpoint: String) {
        self.inner.set_endpoint(endpoint);
    }

    /// Set the app version reported with every event.
    pub fn set_app_version(&self, version: String) {
        self.inner.set_app_version(version);
    }

    /// Set the human-friendly device label reported with every event.
    pub fn set_device_label(&self, label: String) {
        self.inner.set_device_label(label);
    }

    /// Attach an app-provided device attribute (stored under `device.custom`).
    pub fn set_device_attribute(&self, key: String, value: String) {
        self.inner.set_device_attribute(key, value);
    }

    /// Set the number of events buffered before a flush.
    pub fn set_batch_size(&self, batch_size: u32) {
        self.inner.set_batch_size(batch_size);
    }

    /// Set the background flush interval, in seconds.
    pub fn set_flush_interval_secs(&self, secs: u32) {
        self.inner.set_flush_interval_secs(secs);
    }

    /// Start the process-global telemetry exporter from this config.
    ///
    /// Consumes the config: subsequent setters no-op and a second `init` on the
    /// same handle errors. Modeled as a method (not a free `telemetry_init`)
    /// because boltffi 0.25.3 drops free functions that take a handle
    /// parameter, but lowers a handle self-method fine (same reason the
    /// generated `run` lives on `XybridModel`).
    ///
    /// # Errors
    /// Errors if this config was already consumed, or if telemetry is already
    /// initialized without an intervening [`telemetry_shutdown`].
    pub fn init(&self) -> Result<(), XybridError> {
        facade::telemetry_init(&self.inner).map_err(XybridError::from)
    }
}

// ============================================================================
// Bundle inspection
// ============================================================================
//
// Read-only inspection of `.xyb` model bundles for editor tooling / asset
// workflows. Mirrors the pre-bolt C ABI's bundle surface; every method takes or
// returns simple types, so it generates natively (no hand-port).

/// An opened `.xyb` model bundle.
///
/// Create with [`open`](Self::open); read the manifest/metadata, enumerate
/// files, and [`extract`](Self::extract). Wraps the facade's immutable
/// `BundleHandle`.
pub struct XybridBundle {
    inner: std::sync::Arc<facade::BundleHandle>,
}

#[export]
impl XybridBundle {
    /// Open and parse a `.xyb` bundle (decompress zstd, parse tar, validate the
    /// manifest).
    pub fn open(path: String) -> Result<Self, XybridError> {
        let inner = facade::BundleHandle::open(path).map_err(XybridError::from)?;
        Ok(Self { inner })
    }

    /// The model identifier from the manifest.
    pub fn model_id(&self) -> String {
        self.inner.model_id()
    }

    /// The version string from the manifest.
    pub fn version(&self) -> String {
        self.inner.version()
    }

    /// The target platform from the manifest.
    pub fn target(&self) -> String {
        self.inner.target()
    }

    /// The SHA-256 hash from the manifest.
    pub fn hash(&self) -> String {
        self.inner.hash()
    }

    /// Whether the bundle carries a `model_metadata.json`.
    pub fn has_metadata(&self) -> bool {
        self.inner.has_metadata()
    }

    /// Number of files in the bundle (excludes `manifest.json`).
    pub fn file_count(&self) -> u32 {
        self.inner.file_count()
    }

    /// The file name at `index`, or `None` if out of bounds.
    pub fn file_name(&self, index: u32) -> Option<String> {
        self.inner.file_name(index)
    }

    /// The full bundle manifest serialized as JSON.
    pub fn manifest_json(&self) -> Result<String, XybridError> {
        self.inner.manifest_json().map_err(XybridError::from)
    }

    /// The `model_metadata.json` contents, or `None` if the bundle has none.
    pub fn metadata_json(&self) -> Result<Option<String>, XybridError> {
        self.inner.metadata_json().map_err(XybridError::from)
    }

    /// Extract every bundle file to `output_dir` (created if absent).
    pub fn extract(&self, output_dir: String) -> Result<(), XybridError> {
        self.inner.extract(output_dir).map_err(XybridError::from)
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
    fn cache_records_cross_from_facade_without_losing_storage_semantics() {
        let entry = XybridCacheEntry::from(facade::CacheEntry {
            model_id: "owner/repo".into(),
            location: facade::CacheEntryLocation::HuggingFace,
            path: "/cache/hf/repo".into(),
            size_bytes: 2048,
        });
        let status = XybridCacheStatus::from(facade::CacheStatus {
            total_size_bytes: 2048,
            entry_count: 1,
            model_count: 1,
            extracted_model_count: 0,
            cache_root: "/cache".into(),
        });

        assert_eq!(entry.model_id, "owner/repo");
        assert!(entry.location == XybridCacheEntryLocation::HuggingFace);
        assert_eq!(entry.path, "/cache/hf/repo");
        assert_eq!(entry.size_bytes, 2048);
        assert_eq!(status.total_size_bytes, 2048);
        assert_eq!(status.entry_count, 1);
        assert_eq!(status.model_count, 1);
        assert_eq!(status.extracted_model_count, 0);
        assert_eq!(status.cache_root, "/cache");
    }

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

    #[test]
    fn stream_token_event_flattens_for_the_wire() {
        let event = XybridStreamEvent::from(facade::StreamEvent::Token(facade::StreamToken {
            token: "hi".into(),
            token_id: Some(7),
            index: 3,
            cumulative_text: "say hi".into(),
            finish_reason: None,
            tool_calls: Vec::new(),
            raw_text: None,
        }));

        assert_eq!(event.kind, XybridStreamEventKind::Token);
        let token = event.token.expect("token event should carry a token");
        assert_eq!(token.token, "hi");
        assert_eq!(token.index, 3);
        assert!(token.tool_calls.is_empty());
    }

    #[test]
    fn terminal_stream_token_carries_tool_calls_across_the_boundary() {
        // Foreign callers dispatch on this instead of parsing token text —
        // the call blocks never reach the stream.
        let event = XybridStreamEvent::from(facade::StreamEvent::Token(facade::StreamToken {
            token: String::new(),
            token_id: None,
            index: 9,
            cumulative_text: "checking".into(),
            finish_reason: Some("tool_calls".into()),
            raw_text: Some("checking<|tool_call_start|>[x()]<|tool_call_end|>".into()),
            tool_calls: vec![facade::ToolCall {
                id: "call_0".into(),
                name: "get_temperature".into(),
                arguments_json: r#"{"room":"kitchen"}"#.into(),
            }],
        }));

        let token = event.token.expect("token event should carry a token");
        assert_eq!(token.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(token.tool_calls.len(), 1);
        assert_eq!(token.tool_calls[0].name, "get_temperature");
        assert!(
            token.raw_text.is_some(),
            "the replayable raw text must cross too"
        );
    }

    #[test]
    fn generation_config_crosses_from_facade_with_resolved_values() {
        let config = facade::GenerationConfig {
            max_tokens: Some(128),
            temperature: Some(0.25),
            top_p: Some(0.75),
            min_p: Some(0.05),
            top_k: Some(32),
            repetition_penalty: Some(1.1),
            stop_sequences: vec!["</s>".into(), "END".into()],
            grammar: Some("root ::= \"ok\"".into()),
            tools: vec![facade::ToolDefinition {
                name: "weather".into(),
                description: "Weather lookup".into(),
                parameters_json: r#"{"type":"object"}"#.into(),
            }],
        };

        let wire = XybridGenerationConfig::from(config);

        assert_eq!(wire.max_tokens, Some(128));
        assert_eq!(wire.temperature, Some(0.25));
        assert_eq!(wire.top_p, Some(0.75));
        assert_eq!(wire.min_p, Some(0.05));
        assert_eq!(wire.top_k, Some(32));
        assert_eq!(wire.repetition_penalty, Some(1.1));
        assert_eq!(wire.stop_sequences, vec!["</s>", "END"]);
        assert_eq!(wire.grammar.as_deref(), Some("root ::= \"ok\""));
        assert_eq!(wire.tools.len(), 1);
        assert_eq!(wire.tools[0].name, "weather");
        assert_eq!(wire.tools[0].description, "Weather lookup");
        assert_eq!(wire.tools[0].parameters_json, r#"{"type":"object"}"#);
    }

    #[test]
    fn result_conversion_exposes_reasoning_content() {
        // Given
        let mut envelope = facade::Envelope::text("answer".into());
        envelope
            .metadata
            .insert("reasoning_content".into(), "reasoning".into());
        let result = facade::InferenceResult {
            envelope,
            output_type: facade::OutputType::Text,
            model_id: "model".into(),
            latency_ms: 1,
            execution_target: facade::ExecutionTarget::Local,
            metrics: facade::InferenceMetrics {
                total_ms: 1,
                ttft_ms: None,
                tokens_per_second: None,
                prefill_tps: None,
                decode_tps: None,
                tokens_out: None,
                stage_latencies_ms: Vec::new(),
            },
            tool_calls: Vec::new(),
        };

        // When
        let wire = XybridResult::from(result);

        // Then
        assert_eq!(wire.reasoning_content.as_deref(), Some("reasoning"));
        assert!(wire
            .envelope
            .metadata
            .iter()
            .any(|entry| { entry.key == "reasoning_content" && entry.value == "reasoning" }));
    }

    #[test]
    fn envelope_metadata_order_is_stable_across_wire_conversions() {
        // Given
        let envelope = facade::Envelope {
            kind: facade::EnvelopeKind::Text {
                text: "hello".into(),
            },
            metadata: [
                ("zeta".to_string(), "last".to_string()),
                ("alpha".to_string(), "first".to_string()),
            ]
            .into_iter()
            .collect(),
        };

        // When
        let first = XybridEnvelope::from(envelope.clone());
        let second = XybridEnvelope::from(envelope);

        // Then
        assert_eq!(
            first
                .metadata
                .iter()
                .map(|entry| entry.key.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
        assert_eq!(
            first
                .metadata
                .iter()
                .map(|entry| (entry.key.as_str(), entry.value.as_str()))
                .collect::<Vec<_>>(),
            second
                .metadata
                .iter()
                .map(|entry| (entry.key.as_str(), entry.value.as_str()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn conversation_context_history_crosses_the_wire() {
        // Given
        let context = XybridConversationContext::new();
        context
            .push(XybridEnvelope {
                kind: XybridEnvelopeKind::Text {
                    text: "hello".into(),
                },
                metadata: Vec::new(),
            })
            .expect("test envelope should be accepted");

        // When
        let history = context.history();

        // Then
        assert_eq!(history.len(), 1);
        assert!(matches!(
            &history[0].kind,
            XybridEnvelopeKind::Text { text } if text == "hello"
        ));
    }
}
