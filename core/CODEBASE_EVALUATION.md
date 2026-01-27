# Codebase Evaluation: `xybrid-core` v2

**Evaluation Date**: 2026-01-25
**Evaluator**: Architectural Review for Release
**Lens**: Clean Library Architecture (ports/adapters, dependency inversion, testability)

---

## 1. Executive Summary

The `xybrid-core` crate has matured significantly since the last evaluation. The execution layer architecture is well-defined with clear separation between Orchestrator → Executor → TemplateExecutor. However, several architectural concerns remain that should be addressed before a stable release:

### Key Wins Since Last Evaluation
- ✅ Removed legacy `registry` code from core (moved to SDK)
- ✅ `TemplateExecutor` now cleanly handles metadata-driven execution
- ✅ Authority-based orchestration (`LocalAuthority`, `RemoteAuthority`)
- ✅ Feature-gated backends (Candle, Mistral, LlamaCpp)
- ✅ Good separation of streaming concerns into `streaming/` module

### Critical Issues Remaining
1. **Excessive Public API Surface**: ~1133 `pub` items across 115 files - too many contracts
2. **Error Type Proliferation**: 21+ error types with inconsistent patterns
3. **Dependency Violations**: `execution/executor.rs` directly imports backend implementations
4. **`anyhow` Leakage**: 4 files still use `anyhow` (should be typed errors at boundaries)
5. **Large Files**: `executor.rs` (1036 LOC), `orchestrator.rs` (1020 LOC) still monolithic

---

## 2. Boundary Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PUBLIC API SURFACE                             │
│                                                                             │
│  prelude::*                                                                 │
│  ├── Envelope, EnvelopeKind          (ir)                                   │
│  ├── TemplateExecutor, ModelMetadata (execution)                            │
│  ├── AdapterError, AdapterResult     (runtime_adapter)                      │
│  ├── DeviceMetrics, StageDescriptor  (context)                              │
│  ├── HardwareCapabilities            (device)                               │
│  └── CacheProvider, *CacheProvider   (cache_provider)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               CORE DOMAIN                                   │
│  (Pure logic, no I/O, no backend deps)                                      │
│                                                                             │
│  ir/                    Envelope system (data flow)                         │
│  context/               DeviceMetrics, StageDescriptor                      │
│  execution/template/    Model metadata schema (JSON types)                  │
│  pipeline/              Pipeline DSL types & config                         │
│  device/types           Hardware capability types                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ORCHESTRATION LAYER                              │
│  (Coordination, no direct backend access)                                   │
│                                                                             │
│  orchestrator/          Policy, routing, authority                          │
│    ├── authority/       OrchestrationAuthority trait + impls                │
│    ├── policy_engine    PolicyEngine trait + DefaultPolicyEngine            │
│    └── routing_engine   RoutingEngine trait + DefaultRoutingEngine          │
│                                                                             │
│  executor               Stage execution, adapter registry                   │
│    └── (⚠️ VIOLATION: imports Cloud, TemplateExecutor directly)              │
│                                                                             │
│  execution/executor     TemplateExecutor - metadata-driven inference        │
│    └── (⚠️ VIOLATION: imports OnnxRuntime, CandleRuntime directly)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RUNTIME LAYER                                  │
│  (Ports = traits, Adapters = implementations)                               │
│                                                                             │
│  runtime_adapter/                                                           │
│    ├── traits.rs        ModelRuntime, InferenceBackend (PORTS)              │
│    ├── mod.rs           RuntimeAdapter, RuntimeAdapterExt (PORTS)           │
│    │                                                                        │
│    ├── onnx/            OnnxRuntimeAdapter, OnnxBackend (ADAPTER)           │
│    ├── coreml/          CoreMLRuntimeAdapter (ADAPTER, platform-gated)      │
│    ├── candle/          CandleRuntimeAdapter (ADAPTER, feature-gated)       │
│    ├── mistral/         MistralBackend (ADAPTER, feature-gated)             │
│    └── llama_cpp/       LlamaCppBackend (ADAPTER, feature-gated)            │
│                                                                             │
│  pipeline/runner        PipelineRunner (orchestrates DSL pipelines)         │
│  streaming/             StreamManager, AudioBuffer                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UTILITIES LAYER                                │
│  (Cross-cutting, no domain knowledge)                                       │
│                                                                             │
│  audio/                 Audio processing (mel, format, vad)                 │
│  bundler                .xyb bundle management                              │
│  http/                  RetryPolicy, CircuitBreaker                         │
│  telemetry/             Metrics, session tracking                           │
│  tracing                Distributed tracing                                 │
│  phonemizer             Text-to-phoneme (TTS preprocessing)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INTEGRATION LAYER                                │
│  (External service clients)                                                 │
│                                                                             │
│  cloud/                 Third-party LLM APIs (OpenAI, Anthropic)            │
│  gateway/               OpenAI-compatible gateway types                     │
│  cloud_llm/ (internal)  Cloud LLM provider implementations                  │
│  tts/                   Voice embedding utilities                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dependency Flow (Ideal vs Actual)

**Ideal**: `Core Domain` ← `Orchestration` ← `Runtime` ← `Utilities` (inner never imports outer)

**Actual Violations**:
| Location | Violation | Severity |
|----------|-----------|----------|
| `execution/executor.rs:11` | Imports `OnnxRuntime`, `CandleRuntime` directly | HIGH |
| `execution/executor.rs:22` | Imports `LlmRuntimeAdapter` directly | MEDIUM |
| `executor.rs:27-28` | Imports `Cloud`, `CloudConfig` directly | HIGH |
| `orchestrator.rs:49-54` | Imports `ControlSync`, `EventBus`, `Telemetry` | LOW (cross-cutting) |

---

## 3. Public API Audit

### Current State: ~1133 `pub` items (TOO MANY)

### Proposed Public API Surface (Minimal)

```rust
// ============================================================================
// TIER 1: Core Types (always public)
// ============================================================================
pub mod ir {
    pub struct Envelope { ... }
    pub enum EnvelopeKind { Audio, Text, Embedding, Tokens }
    pub struct AudioSamples { ... }
}

pub mod error {
    // NEW: Single unified error type
    pub enum XybridError {
        Inference(InferenceError),
        Pipeline(PipelineError),
        Io(std::io::Error),
        Config(ConfigError),
    }
}

// ============================================================================
// TIER 2: Execution API (main user entry points)
// ============================================================================
pub mod execution {
    pub struct TemplateExecutor { ... }
    pub struct ModelMetadata { ... }
    pub enum ExecutionTemplate { ... }
    pub enum PreprocessingStep { ... }
    pub enum PostprocessingStep { ... }
    // Voice types for TTS
    pub struct VoiceConfig { ... }
    pub struct VoiceInfo { ... }
}

// ============================================================================
// TIER 3: Pipeline API (DSL users)
// ============================================================================
pub mod pipeline {
    pub struct PipelineConfig { ... }
    pub struct PipelineRunner { ... }
    pub struct StageConfig { ... }
    pub enum ExecutionTarget { ... }
    pub enum IntegrationProvider { ... }
}

// ============================================================================
// TIER 4: Extension Points (advanced users implementing custom backends)
// ============================================================================
pub mod runtime {
    // Traits for user implementations
    pub trait RuntimeAdapter: Send + Sync { ... }
    pub trait InferenceBackend: Send + Sync { ... }

    // Built-in adapters (re-exports)
    pub use crate::runtime_adapter::onnx::OnnxRuntimeAdapter;
    #[cfg(feature = "candle")]
    pub use crate::runtime_adapter::candle::CandleRuntimeAdapter;
}

// ============================================================================
// TIER 5: Orchestration API (framework users)
// ============================================================================
pub mod orchestrator {
    pub struct Orchestrator { ... }
    pub trait OrchestrationAuthority { ... }
    pub struct LocalAuthority { ... }
    pub struct RemoteAuthority { ... }
}

// ============================================================================
// TIER 6: Utilities (optional, for advanced use cases)
// ============================================================================
pub mod device {
    pub fn detect_capabilities() -> HardwareCapabilities;
    pub struct HardwareCapabilities { ... }
}

pub mod bundler {
    pub struct XyBundle { ... }
    pub struct BundleManifest { ... }
}

pub mod cache {
    pub trait CacheProvider { ... }
}
```

### Items to Demote to `pub(crate)`

| Module | Current Public | Should Be |
|--------|---------------|-----------|
| `runtime_adapter::tensor_utils` | `pub mod` | `pub(crate)` |
| `runtime_adapter::metadata_driven` | `pub mod` | `pub(crate)` |
| `audio::mel::*` | Various `pub fn` | `pub(crate)` |
| `phonemizer::*` | `pub struct/enum` | `pub(crate)` |
| `preprocessing::*` | `pub fn` | `pub(crate)` |
| `telemetry::events::*` | ~25 `pub` items | `pub(crate)` |
| `streaming::session::*` | ~22 `pub` items | `pub(crate)` |
| `gateway::api::*` | ~21 `pub` items | Keep (gateway types) |
| `context::StageDescriptor::options` | `pub` | `pub(crate)` |

---

## 4. Dependency Boundary Violations

### HIGH Priority

#### 4.1 `execution/executor.rs` imports backend implementations

```rust
// Line 11-12: Direct ONNX import in core execution
use crate::runtime_adapter::onnx::{ONNXSession, OnnxRuntime};

// Line 19: Conditional Candle import
#[cfg(feature = "candle")]
use crate::runtime_adapter::candle::CandleRuntime;

// Line 22: LLM adapter import
use crate::runtime_adapter::llm::{LlmConfig, LlmRuntimeAdapter};
```

**Problem**: Core execution logic should not know about specific backends.

**Fix**: Use `ModelRuntime` trait exclusively, inject runtime via constructor.

#### 4.2 `executor.rs` imports Cloud client

```rust
// Line 27-28: Cloud client in mid-level executor
use crate::cloud::{Cloud, CloudBackend, CloudConfig, CompletionRequest};
```

**Problem**: `Executor` (adapter registry) should not know about cloud integration.

**Fix**: Move cloud execution to a separate `CloudAdapter` implementing `RuntimeAdapter`.

### MEDIUM Priority

#### 4.3 `orchestrator.rs` imports internal infrastructure

```rust
// Line 49-55: Cross-cutting concerns mixed with orchestration
use crate::control_sync::ControlSync;
use crate::event_bus::{EventBus, OrchestratorEvent};
use crate::telemetry::Telemetry;
```

**Problem**: These could be injected rather than directly instantiated.

**Fix**: Accept these as constructor parameters with defaults.

---

## 5. Error Leakage Analysis

### Current Error Types (21+)

| Error Type | Module | Pattern |
|------------|--------|---------|
| `AdapterError` | `runtime_adapter` | `thiserror`, good |
| `BackendError` | `runtime_adapter/inference_backend` | `thiserror`, good |
| `ExecutorError` | `executor` | `thiserror`, good |
| `OrchestratorError` | `orchestrator` | Manual impl, **inconsistent** |
| `PipelineRunnerError` | `pipeline/runner` | ? |
| `ResolutionError` | `pipeline/resolver` | ? |
| `CloudError` | `cloud/error` | `thiserror`, good |
| `BundlerError` | `bundler` | `thiserror`, good |
| `EnvelopeError` | `ir/envelope` | `thiserror`, good |
| `AudioFormatError` | `audio/format` | ? |
| `PhonemizeError` | `phonemizer` | `thiserror`, internal |
| `VoiceError` | `tts/voice_embedding` | ? |
| `StreamManagerError` | `streaming/manager` | ? |
| `ControlSyncError` | `control_sync` | ? |
| `GatewayError` | `gateway/error` | ? |
| `LlmError` | `cloud_llm/error` | internal |

### `anyhow` Leakage (4 files)

```
core/src/runtime_adapter/candle/whisper.rs  <- anyhow in internal code (OK if mapped)
core/src/orchestrator/bootstrap.rs          <- anyhow in bootstrap (should be typed)
core/src/runtime_adapter/candle/README.md   <- documentation only
core/src/control_sync.rs                    <- anyhow in internal code
```

### Proposed: Unified Public Error Type

```rust
// src/error.rs
use thiserror::Error;

/// The canonical error type for xybrid-core public API.
#[derive(Error, Debug)]
pub enum XybridError {
    /// Model inference failed
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),

    /// Pipeline execution failed
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// Model or file not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// Invalid configuration
    #[error("Config error: {0}")]
    Config(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Inference-specific errors
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Backend error: {0}")]
    Backend(String),
}

/// Pipeline-specific errors
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Stage failed: {stage} - {reason}")]
    StageFailed { stage: String, reason: String },
    #[error("Invalid target: {0}")]
    InvalidTarget(String),
    #[error("Provider error: {0}")]
    Provider(String),
}
```

**Migration**: Internal errors map to `XybridError` at module boundaries.

---

## 6. Extension Points (Traits)

### Current Traits

| Trait | Location | Object-Safe? | Purpose |
|-------|----------|--------------|---------|
| `RuntimeAdapter` | `runtime_adapter/mod.rs` | ✅ Yes | Model loading & execution |
| `RuntimeAdapterExt` | `runtime_adapter/mod.rs` | ✅ Yes | Multi-model management |
| `ModelRuntime` | `runtime_adapter/traits.rs` | ✅ Yes | TemplateExecutor runtime |
| `InferenceBackend` | `runtime_adapter/inference_backend.rs` | ✅ Yes | Low-level backend |
| `LlmBackend` | `runtime_adapter/llm.rs` | ✅ Yes | LLM-specific backend |
| `CandleModel` | `runtime_adapter/candle/model.rs` | ❓ Check | Candle model interface |
| `OrchestrationAuthority` | `orchestrator/authority/mod.rs` | ✅ Yes | Routing/policy decisions |
| `PolicyEngine` | `orchestrator/policy_engine.rs` | ✅ Yes | Policy evaluation |
| `RoutingEngine` | `orchestrator/routing_engine.rs` | ✅ Yes | Route selection |
| `CacheProvider` | `cache_provider.rs` | ✅ Yes | Cache abstraction |
| `DeviceAdapter` | `device_adapter.rs` | ❓ Check | Device interface |
| `RetryableError` | `http/retry.rs` | ✅ Yes | Error retry marker |

### Recommendations

#### Keep Public (User Extension Points)
- `RuntimeAdapter` - Core extension point for custom backends
- `CacheProvider` - Simple, stable, useful for testing
- `OrchestrationAuthority` - Advanced but important for custom routing

#### Make Internal (`pub(crate)`)
- `RuntimeAdapterExt` - Complex, internal optimization
- `ModelRuntime` - Internal to TemplateExecutor
- `InferenceBackend` - Low-level, internal
- `LlmBackend` - Backend-specific, internal
- `PolicyEngine` - Most users use default
- `RoutingEngine` - Most users use default
- `DeviceAdapter` - Internal infrastructure

#### Ergonomic Improvements

```rust
// Current: Users must implement many methods
pub trait RuntimeAdapter: Send + Sync {
    fn name(&self) -> &str;
    fn supported_formats(&self) -> Vec<&'static str>;
    fn load_model(&mut self, path: &str) -> AdapterResult<()>;
    fn execute(&self, input: &Envelope) -> AdapterResult<Envelope>;
}

// Proposed: Provide default implementations where sensible
pub trait RuntimeAdapter: Send + Sync {
    fn name(&self) -> &str;

    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["onnx"] // Default for most adapters
    }

    fn load_model(&mut self, path: &str) -> Result<(), XybridError>;
    fn execute(&self, input: &Envelope) -> Result<Envelope, XybridError>;

    // Optional: warm-up for GPU backends
    fn warmup(&mut self) -> Result<(), XybridError> { Ok(()) }
}
```

---

## 7. Feature Flags Analysis

### Current Feature Flags (11)

| Feature | Purpose | Dependencies | Clean? |
|---------|---------|--------------|--------|
| `default` | `ort-download` | - | ✅ |
| `ort-download` | ONNX prebuilt binaries | `ort/download-binaries` | ✅ |
| `ort-dynamic` | Android .so loading | `ort/load-dynamic` | ✅ |
| `ort-coreml` | Apple Neural Engine | `ort/coreml` | ✅ |
| `candle` | Pure Rust ML | candle-*, safetensors, hf-hub | ✅ |
| `candle-metal` | Candle + Metal | `candle`, metal flags | ✅ |
| `candle-cuda` | Candle + CUDA | `candle`, cuda flags | ✅ |
| `llm-mistral` | Mistral.rs | `mistralrs` | ✅ |
| `llm-mistral-metal` | Mistral + Metal | `llm-mistral`, metal | ✅ |
| `llm-mistral-cuda` | Mistral + CUDA | `llm-mistral`, cuda | ✅ |
| `llm-llamacpp` | Llama.cpp | (build deps) | ⚠️ Empty |

### Issues

1. **`llm-llamacpp` is empty**: Feature flag exists but gates nothing in Cargo.toml
2. **`cfg` scattering**: Platform checks spread across files instead of isolated

### Recommendations

```rust
// runtime_adapter/mod.rs - CURRENT (scattered cfg)
#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub mod coreml;

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
pub mod llm;

// PROPOSED: Isolate in feature modules
// runtime_adapter/mod.rs - CLEANER
pub mod onnx;           // Always available

#[cfg(feature = "backend-coreml")]
pub mod coreml;

#[cfg(feature = "backend-candle")]
pub mod candle;

#[cfg(feature = "backend-llm")]
pub mod llm;
```

---

## 8. Test Architecture

### Current State

Tests are co-located with modules (`#[cfg(test)] mod tests`), which is fine for unit tests but lacks separation.

### Proposed Three-Layer Scheme

```
core/
├── src/                          # Source code
├── tests/                        # Integration tests
│   ├── pipeline_execution.rs     # Full pipeline tests
│   ├── bundle_roundtrip.rs       # Bundle create/extract
│   └── feature_gated/
│       ├── candle_whisper.rs     # #[cfg(feature = "candle")]
│       └── mistral_inference.rs  # #[cfg(feature = "llm-mistral")]
│
└── src/testing/                  # Test infrastructure
    ├── mocks.rs                  # MockRuntimeAdapter (exists, good!)
    ├── fixtures.rs               # Test model fixtures (exists, good!)
    └── fakes.rs                  # NEW: Fakes for traits
```

#### Layer 1: Core Unit Tests (No I/O, No Backends)

```rust
// src/ir/envelope.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_roundtrip() {
        let env = Envelope::new(EnvelopeKind::Text("hello".into()));
        let bytes = env.to_bytes();
        let restored = Envelope::from_bytes(&bytes).unwrap();
        assert_eq!(env.kind, restored.kind);
    }
}
```

#### Layer 2: Backend Conformance Tests

```rust
// tests/conformance/runtime_adapter.rs
use xybrid_core::runtime::{RuntimeAdapter, InferenceBackend};
use xybrid_core::testing::fixtures::*;

/// All RuntimeAdapter implementations must pass these tests
fn adapter_conformance<A: RuntimeAdapter>(adapter: &mut A) {
    // 1. Name is non-empty
    assert!(!adapter.name().is_empty());

    // 2. Supported formats is non-empty
    assert!(!adapter.supported_formats().is_empty());

    // 3. Execute before load returns error
    let result = adapter.execute(&test_envelope());
    assert!(matches!(result, Err(XybridError::Inference(_))));

    // 4. Load then execute succeeds
    adapter.load_model(&test_model_path()).unwrap();
    let output = adapter.execute(&test_envelope()).unwrap();
    assert!(matches!(output.kind, EnvelopeKind::Embedding(_)));
}

#[test]
fn onnx_adapter_conformance() {
    let mut adapter = OnnxRuntimeAdapter::new();
    adapter_conformance(&mut adapter);
}

#[test]
#[cfg(feature = "candle")]
fn candle_adapter_conformance() {
    let mut adapter = CandleRuntimeAdapter::new();
    adapter_conformance(&mut adapter);
}
```

#### Layer 3: Feature-Gated Integration Tests

```rust
// tests/feature_gated/candle_whisper.rs
#![cfg(feature = "candle")]

use xybrid_core::execution::TemplateExecutor;
use std::path::PathBuf;

#[test]
#[ignore] // Requires model download
fn whisper_transcription_e2e() {
    let model_dir = PathBuf::from("fixtures/whisper-tiny-candle");
    if !model_dir.exists() {
        eprintln!("Skipping: model not downloaded");
        return;
    }

    let mut executor = TemplateExecutor::new(model_dir.to_str().unwrap());
    // ... full transcription test
}
```

---

## 9. Refactoring Roadmap

### P0: Critical for Release (Complexity: M-L)

| # | Task | Files | Complexity | Breaking? |
|---|------|-------|------------|-----------|
| 1 | **Create unified `XybridError`** | New `error.rs`, update all modules | L | Yes* |
| 2 | **Fix `execution/executor.rs` violations** | 1 file | M | No |
| 3 | **Remove `anyhow` from public paths** | 4 files | S | No |
| 4 | **Demote internal `pub` to `pub(crate)`** | ~20 modules | M | Yes* |

*Breaking changes can be mitigated with re-exports for deprecation period

### P1: Important for Stability (Complexity: M) ✅ COMPLETED

| # | Task | Files | Complexity | Status |
|---|------|-------|------------|--------|
| 5 | **Extract `CloudAdapter` from `executor.rs`** | 2 files | M | ✅ Done |
| 6 | **Consolidate `OrchestratorError` to use `thiserror`** | 1 file | S | ✅ Done |
| 7 | **Create `tests/conformance/` structure** | New files | M | ✅ Done |
| 8 | **Document `llm-llamacpp` feature flag** | `Cargo.toml` | S | ✅ Done |

**P1 Changes Summary:**
- Created `runtime_adapter/cloud/mod.rs` with `CloudRuntimeAdapter` implementing `RuntimeAdapter` trait
- Refactored `executor.rs` to delegate cloud execution to `CloudRuntimeAdapter` via enriched envelope metadata
- Converted `OrchestratorError` from manual `impl Display/Error` to `#[derive(Error)]` with `thiserror`
- Created `tests/conformance_runtime_adapter.rs` with 18 conformance tests for adapter implementations
- Added documentation comments to `llm-llamacpp` feature explaining it's a marker feature for build.rs gating

### P2: Nice to Have (Complexity: S-M)

| # | Task | Files | Complexity | Breaking? |
|---|------|-------|------------|-----------|
| 9 | **Split `orchestrator.rs` into submodules** | 1 large → 4 small | M | No |
| 10 | **Add `RuntimeAdapter::warmup()` default** | 1 trait | S | No |
| 11 | **Document public API with examples** | All `pub` items | M | No |
| 12 | **Consolidate platform `cfg` into feature modules** | ~10 files | M | No |

---

## 10. Suggested Module Layout (Post-Refactor)

```
core/src/
├── lib.rs                    # Minimal: just pub mod declarations
├── prelude.rs                # pub use of common types
├── error.rs                  # NEW: XybridError, InferenceError, PipelineError
│
├── ir/                       # Intermediate representation
│   ├── mod.rs
│   └── envelope.rs
│
├── execution/                # Metadata-driven execution
│   ├── mod.rs
│   ├── executor.rs           # TemplateExecutor
│   ├── template/             # Model metadata schema
│   │   ├── mod.rs
│   │   ├── metadata.rs
│   │   ├── steps.rs
│   │   └── voice.rs
│   ├── preprocessing/        # pub(crate)
│   ├── postprocessing/       # pub(crate)
│   └── modes/                # pub(crate)
│
├── runtime/                  # NEW: Cleaner runtime module
│   ├── mod.rs                # Re-exports RuntimeAdapter trait + impls
│   ├── adapter.rs            # RuntimeAdapter, RuntimeAdapterExt traits
│   ├── onnx/                 # ONNX backend (always available)
│   ├── coreml/               # CoreML backend (platform-gated)
│   ├── candle/               # Candle backend (feature-gated)
│   └── llm/                  # LLM backends (feature-gated)
│
├── orchestrator/             # High-level coordination
│   ├── mod.rs
│   ├── authority/
│   ├── policy.rs             # PolicyEngine (was policy_engine.rs)
│   ├── routing.rs            # RoutingEngine (was routing_engine.rs)
│   └── events.rs             # OrchestratorEvent (from event_bus.rs)
│
├── pipeline/                 # Pipeline DSL
│   ├── mod.rs
│   ├── config.rs
│   ├── runner.rs
│   └── ...
│
├── device/                   # Hardware detection
│   └── ...
│
├── cloud/                    # External API clients
│   └── ...
│
├── bundler.rs                # .xyb bundle management
├── cache_provider.rs         # CacheProvider trait
│
└── internal/                 # NEW: Explicitly internal modules
    ├── audio/                # Audio processing (was pub)
    ├── phonemizer/           # TTS phonemization (was pub)
    ├── streaming/            # Stream management
    ├── telemetry/            # Metrics
    └── tracing/              # Distributed tracing
```

---

## 11. Top 2 Refactors: Code Diffs

### Refactor 1: Unified Error Type (P0)

**New file: `src/error.rs`**

```rust
//! Unified error types for xybrid-core public API.

use thiserror::Error;

/// The canonical error type for xybrid-core.
///
/// All public API methods return `Result<T, XybridError>`.
#[derive(Error, Debug)]
pub enum XybridError {
    /// Model inference failed
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),

    /// Pipeline execution failed
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// Resource not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// Invalid configuration
    #[error("Configuration error: {0}")]
    Config(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Errors during model inference.
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Preprocessing failed: {0}")]
    Preprocessing(String),

    #[error("Postprocessing failed: {0}")]
    Postprocessing(String),
}

/// Errors during pipeline execution.
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Stage '{stage}' failed: {reason}")]
    StageFailed { stage: String, reason: String },

    #[error("Invalid target: {0}")]
    InvalidTarget(String),

    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Policy denied: {0}")]
    PolicyDenied(String),
}

/// Result type alias for xybrid-core.
pub type XybridResult<T> = Result<T, XybridError>;

// ─────────────────────────────────────────────────────────────────────────────
// Conversions from internal errors
// ─────────────────────────────────────────────────────────────────────────────

impl From<crate::runtime_adapter::AdapterError> for XybridError {
    fn from(e: crate::runtime_adapter::AdapterError) -> Self {
        use crate::runtime_adapter::AdapterError::*;
        match e {
            ModelNotFound(s) => XybridError::NotFound(s),
            ModelNotLoaded(s) => XybridError::Inference(InferenceError::ModelNotLoaded(s)),
            InvalidInput(s) => XybridError::Inference(InferenceError::InvalidInput(s)),
            InferenceFailed(s) => XybridError::Inference(InferenceError::Backend(s)),
            IOError(e) => XybridError::Io(e),
            SerializationError(s) => XybridError::Serialization(s),
            RuntimeError(s) => XybridError::Inference(InferenceError::Backend(s)),
        }
    }
}
```

**Update `src/lib.rs`**:

```diff
+ pub mod error;
+ pub use error::{XybridError, XybridResult, InferenceError, PipelineError};
```

### Refactor 2: Fix Dependency Violation in `execution/executor.rs` (P0)

**Before** (`src/execution/executor.rs:10-24`):

```rust
use crate::ir::Envelope;
use crate::runtime_adapter::onnx::{ONNXSession, OnnxRuntime};  // VIOLATION
use crate::runtime_adapter::{AdapterError, ModelRuntime};
use crate::tracing as xybrid_trace;

#[cfg(feature = "candle")]
use crate::runtime_adapter::candle::CandleRuntime;  // VIOLATION

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::runtime_adapter::llm::{LlmConfig, LlmRuntimeAdapter};  // VIOLATION

#[cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))]
use crate::runtime_adapter::RuntimeAdapter;
```

**After**:

```rust
use crate::ir::Envelope;
use crate::runtime_adapter::{AdapterError, ModelRuntime};
use crate::tracing as xybrid_trace;

// Runtime is now injected, not created internally
// See TemplateExecutor::with_runtime() for custom runtimes
```

**Update `TemplateExecutor` construction** (`src/execution/executor.rs`):

```rust
impl TemplateExecutor {
    /// Creates a new TemplateExecutor with default ONNX runtime.
    pub fn new(base_path: &str) -> Self {
        Self {
            base_path: base_path.to_string(),
            runtime: None,  // Lazy initialization
            runtime_kind: RuntimeKind::Onnx,
        }
    }

    /// Creates a TemplateExecutor with a custom runtime.
    ///
    /// Use this to inject alternative backends:
    /// ```rust,ignore
    /// let runtime = Box::new(MyCustomRuntime::new());
    /// let executor = TemplateExecutor::with_runtime("models/", runtime);
    /// ```
    pub fn with_runtime(base_path: &str, runtime: Box<dyn ModelRuntime>) -> Self {
        Self {
            base_path: base_path.to_string(),
            runtime: Some(runtime),
            runtime_kind: RuntimeKind::Custom,
        }
    }

    // Factory methods for specific runtimes (feature-gated)
    #[cfg(feature = "candle")]
    pub fn with_candle(base_path: &str) -> Self {
        use crate::runtime_adapter::candle::CandleRuntime;
        Self {
            base_path: base_path.to_string(),
            runtime: Some(Box::new(CandleRuntime::new())),
            runtime_kind: RuntimeKind::Candle,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RuntimeKind {
    Onnx,
    Candle,
    Custom,
}
```

---

## 12. Migration Notes for Breaking Changes

### Error Type Migration

```rust
// Before
use xybrid_core::runtime_adapter::AdapterError;

fn do_inference() -> Result<(), AdapterError> { ... }

// After
use xybrid_core::error::XybridError;

fn do_inference() -> Result<(), XybridError> { ... }

// Or use the alias
use xybrid_core::XybridResult;

fn do_inference() -> XybridResult<()> { ... }
```

### Public API Deprecation Period

For items being moved to `pub(crate)`, provide temporary re-exports with deprecation warnings:

```rust
// src/lib.rs (during deprecation period)
#[deprecated(since = "0.1.0", note = "Use xybrid_core::execution::preprocessing instead")]
pub mod preprocessing {
    pub use crate::execution::preprocessing::*;
}
```

---

## 13. Principles Checklist

| Principle | Status | Notes |
|-----------|--------|-------|
| **R1.1** Anything `pub` is a contract | ⚠️ | Too many `pub` items |
| **R1.2** Prefer `pub(crate)` by default | ❌ | Need aggressive demotion |
| **R1.3** Few intention-revealing modules | ⚠️ | Good structure, too granular exports |
| **R2.1** Core has no backend knowledge | ❌ | `execution/executor.rs` violations |
| **R2.2** Backends depend on core | ✅ | Correct direction |
| **R2.3** Feature flags as boundaries | ✅ | Well-implemented |
| **R3.1** Ports as traits in core | ✅ | `RuntimeAdapter`, `ModelRuntime` |
| **R3.2** Adapters in backend modules | ✅ | Clean separation |
| **R3.3** Object-safe extension points | ✅ | All key traits are `dyn`-compatible |
| **R4.1** Invariants in types | ✅ | Good use of enums |
| **R4.2** Separate internal/public repr | ⚠️ | Some leakage |
| **R4.3** Canonical error at boundary | ❌ | 21+ error types |
| **R5.1** Core testable with fakes | ✅ | `MockRuntimeAdapter` exists |
| **R5.2** Backend conformance tests | ❌ | Need to create |
| **R5.3** Feature-gated integration tests | ⚠️ | Partial |
| **R6.1** Heavy modules isolated | ✅ | Backends well-isolated |
| **R6.2** Few rich traits | ⚠️ | Some trait soup in orchestrator |
| **R6.3** No lifetimes in public API | ✅ | Clean |
| **R7.1** Consistent naming | ⚠️ | Error types inconsistent |
| **R7.2** Consistent ownership | ✅ | `Arc` at boundaries |
| **R7.3** Consistent async pattern | ✅ | Sync API + async wrappers |

---

*Evaluated for release readiness*
*Next review: After P0 tasks completed*
