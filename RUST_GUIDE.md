# Xybrid Rust Coding Guide

This guide establishes the core values, architectural principles, and coding conventions for the xybrid codebase. It is intended for AI coding agents and human contributors alike.

---

## Core Values

### 1. Correctness Over Cleverness

Write code that is obviously correct rather than code that is cleverly compact. Prefer explicit patterns that future maintainers can understand at a glance.

```rust
// GOOD: Explicit and readable
let audio_samples = match envelope.kind {
    EnvelopeKind::Audio(bytes) => decode_audio(&bytes)?,
    _ => return Err(AdapterError::InvalidInput("Expected audio".into())),
};

// AVOID: Clever but obscure
let audio_samples = envelope.kind.as_audio().ok_or("Expected audio")?.pipe(decode_audio)?;
```

### 2. Fail Fast, Fail Loudly

Validate inputs early. Return meaningful errors. Never silently swallow failures or continue with invalid state.

```rust
// GOOD: Validate early with context
pub fn execute(&mut self, metadata: &ModelMetadata, input: &Envelope) -> AdapterResult<Envelope> {
    if metadata.files.is_empty() {
        return Err(AdapterError::InvalidInput(
            "ModelMetadata must specify at least one file".into()
        ));
    }
    // ... proceed with valid state
}

// AVOID: Silent fallbacks that hide bugs
let files = metadata.files.first().unwrap_or(&"default.onnx".to_string());
```

### 3. Zero-Cost Abstractions

Leverage Rust's type system to catch errors at compile time. Use enums over stringly-typed APIs. Use newtypes to distinguish semantically different values.

```rust
// GOOD: Type-safe enum
pub enum ExecutionTarget {
    Device,
    Cloud,
    Edge,
    Auto,
}

// AVOID: Stringly-typed
fn execute(target: &str) // "device", "cloud", etc.
```

### 4. Composition Over Inheritance

Build functionality through composition of small, focused types. Use traits for behavior abstraction, not for building deep hierarchies.

### 5. Explicit Dependencies

All dependencies should be visible in function signatures. Avoid hidden global state. Prefer dependency injection over singletons.

---

## Architecture Principles

### Layered Execution Model

Xybrid uses a three-layer execution architecture. Each layer has distinct responsibilities:

```
┌─────────────────────────────────────────────────────────────┐
│                       Orchestrator                          │
│   Policy evaluation, routing decisions, stream management   │
├─────────────────────────────────────────────────────────────┤
│                         Executor                            │
│   Adapter registry, target selection, model loading         │
├─────────────────────────────────────────────────────────────┤
│                     TemplateExecutor                        │
│   Preprocessing, model inference, postprocessing            │
└─────────────────────────────────────────────────────────────┘
```

**Rules:**
- Higher layers may call lower layers, never the reverse
- Each layer owns its error types
- Data flows down via `Envelope`, results flow up via `Result<Envelope, LayerError>`

### Metadata-Driven Execution

**All model execution MUST go through the xybrid execution system via `model_metadata.json`.**

```rust
// CORRECT: Use TemplateExecutor + model_metadata.json
let metadata: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
let mut executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
let output = executor.execute(&metadata, &input)?;

// WRONG: Direct ONNX inference (bypasses preprocessing/postprocessing)
let session = Session::builder()?.commit_from_file(&model_path)?;
let outputs = session.run(ort::inputs![...])?;
```

**Why:** Direct ONNX calls bypass critical preprocessing (phonemization, mel spectrogram, tokenization) and postprocessing (CTC decode, audio encoding) that models require. The execution system provides a uniform interface and enables pipeline composition.

### Envelope-Based Data Flow

All data between pipeline stages flows through the `Envelope` type:

```rust
pub struct Envelope {
    pub kind: EnvelopeKind,
    pub metadata: HashMap<String, String>,
}

pub enum EnvelopeKind {
    Audio(Vec<u8>),       // Raw audio bytes (WAV, PCM)
    Text(String),         // Text content
    Embedding(Vec<f32>),  // Numeric embeddings
}
```

**Rules:**
- Functions that process pipeline data take `&Envelope` and return `Result<Envelope, _>`
- Use `metadata` for passing auxiliary information (sample rate, language, etc.)
- Prefer `EnvelopeKind` variants over raw types in function signatures

### Trait-Based Runtime Abstraction

Runtime backends (ONNX, Candle, CoreML) implement the `ModelRuntime` trait:

```rust
pub trait ModelRuntime: Send + Sync {
    fn name(&self) -> &str;
    fn supported_formats(&self) -> &[&str];
    fn load(&self, path: &Path) -> AdapterResult<Box<dyn Any>>;
    fn execute(&self, model: &dyn Any, input: &Envelope) -> AdapterResult<Envelope>;
}
```

**Rules:**
- New backends MUST implement `ModelRuntime`
- Backend-specific code lives in `runtime_adapter/<backend>/`
- Feature-gate optional backends (e.g., `#[cfg(feature = "candle")]`)

---

## Error Handling

### Two-Level Strategy

1. **Library code** uses `thiserror` for structured, typed errors:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AdapterError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type AdapterResult<T> = Result<T, AdapterError>;
```

2. **Application code** (CLI, examples) uses `anyhow` for convenience:

```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let config = std::fs::read_to_string("config.json")
        .context("Failed to read config file")?;
    // ...
}
```

### Error Type Conventions

| Context | Error Type | Result Alias |
|---------|-----------|--------------|
| Adapter/Runtime | `AdapterError` | `AdapterResult<T>` |
| Orchestrator | `OrchestratorError` | `OrchestratorResult<T>` |
| Envelope/IR | `EnvelopeError` | `EnvelopeResult<T>` |
| Application | `anyhow::Error` | `anyhow::Result<T>` |

### Error Messages

- Start with lowercase (Rust convention)
- Be specific: include the problematic value when possible
- Provide context: what was the code trying to do?

```rust
// GOOD
Err(AdapterError::ModelNotFound(format!(
    "model file '{}' not found in bundle at '{}'",
    filename, bundle_path.display()
)))

// AVOID
Err(AdapterError::ModelNotFound("not found".into()))
```

---

## Module Organization

### Directory Structure

```
crate/src/
├── lib.rs              # Public API re-exports
├── module.rs           # Simple modules
└── complex_module/     # Multi-file modules
    ├── mod.rs          # Module root, re-exports
    ├── types.rs        # Type definitions
    ├── traits.rs       # Trait definitions
    └── impl.rs         # Implementations
```

### Visibility Rules

1. **Default to private.** Only expose what's needed.
2. **Use `pub(crate)`** for crate-internal sharing.
3. **Re-export from `lib.rs`** for clean public API.

```rust
// lib.rs - Clean public API
pub mod orchestrator;
pub mod executor;
pub mod template_executor;
pub mod ir;

pub use ir::{Envelope, EnvelopeKind};
pub use template_executor::TemplateExecutor;
```

### Feature Flags

Use feature flags for optional functionality:

```toml
# Cargo.toml
[features]
default = ["onnx-runtime"]
onnx-runtime = []
candle = ["dep:candle-core", "dep:candle-nn"]
candle-metal = ["candle", "candle-core/metal"]
```

```rust
// Conditional compilation
#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
use crate::runtime_adapter::candle::CandleRuntime;
```

---

## Naming Conventions

### Types and Traits

| Kind | Convention | Examples |
|------|------------|----------|
| Structs | `PascalCase` | `TemplateExecutor`, `ModelMetadata` |
| Enums | `PascalCase` | `EnvelopeKind`, `ExecutionTarget` |
| Traits | `PascalCase`, noun/adjective | `ModelRuntime`, `PolicyEngine` |
| Error types | `<Context>Error` | `AdapterError`, `OrchestratorError` |
| Result aliases | `<Context>Result<T>` | `AdapterResult<T>` |

### Functions and Methods

| Kind | Convention | Examples |
|------|------------|----------|
| Constructors | `new()`, `with_*()` | `new()`, `with_base_path()` |
| Builders | `with_*()` | `with_target()`, `with_provider()` |
| Getters | noun (no `get_` prefix) | `name()`, `metadata()` |
| Predicates | `is_*()`, `has_*()` | `is_cloud()`, `has_gpu()` |
| Conversions | `to_*()`, `into_*()`, `as_*()` | `to_bytes()`, `into_inner()` |
| Fallible ops | verb | `execute()`, `load()`, `parse()` |

### Variables

- `snake_case` for all variables
- Descriptive names: `audio_samples`, not `as` or `samples`
- No Hungarian notation

---

## Common Patterns

### Constructor Pattern

```rust
impl TemplateExecutor {
    /// Create a new executor with default settings.
    pub fn new() -> Self {
        Self {
            base_path: None,
            session_cache: HashMap::new(),
        }
    }

    /// Create an executor with a base path for resolving relative model paths.
    pub fn with_base_path(path: &str) -> Self {
        Self {
            base_path: Some(PathBuf::from(path)),
            session_cache: HashMap::new(),
        }
    }
}
```

### Builder Pattern (for complex configuration)

```rust
pub struct StageDescriptor {
    pub name: String,
    pub bundle_path: Option<String>,
    pub target: Option<ExecutionTarget>,
}

impl StageDescriptor {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bundle_path: None,
            target: None,
        }
    }

    pub fn with_bundle_path(mut self, path: impl Into<String>) -> Self {
        self.bundle_path = Some(path.into());
        self
    }

    pub fn with_target(mut self, target: ExecutionTarget) -> Self {
        self.target = Some(target);
        self
    }
}
```

### Result Propagation

```rust
// Use ? for early return on error
fn process(input: &Envelope) -> AdapterResult<Envelope> {
    let audio = input.audio_bytes()
        .ok_or_else(|| AdapterError::InvalidInput("Expected audio input".into()))?;

    let samples = decode_audio(audio)?;
    let processed = apply_preprocessing(samples)?;

    Ok(Envelope::from_embedding(processed))
}
```

### Enum Dispatch

```rust
impl PreprocessingStep {
    pub fn apply(&self, input: &Envelope) -> AdapterResult<PreprocessedData> {
        match self {
            PreprocessingStep::AudioDecode { sample_rate, channels } => {
                decode_audio(input, *sample_rate, *channels)
            }
            PreprocessingStep::Phonemize { tokens_file, backend } => {
                phonemize(input, tokens_file, backend)
            }
            PreprocessingStep::Tokenize { vocab_file } => {
                tokenize(input, vocab_file)
            }
        }
    }
}
```

---

## Testing

### Test Organization

```rust
// Unit tests in the same file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_roundtrip() {
        let original = Envelope::from_text("hello");
        let bytes = original.to_bytes().unwrap();
        let restored = Envelope::from_bytes(&bytes).unwrap();
        assert_eq!(original.kind, restored.kind);
    }
}
```

### Test Fixtures

Use the `testing` module for shared fixtures:

```rust
use xybrid_core::testing::model_fixtures;

#[test]
fn test_tts_execution() {
    let model_dir = model_fixtures::require_model("kitten-tts-nano");
    // ... test with real model
}
```

### Integration Tests

Place in `integration-tests/` crate or `tests/` directory:

```rust
// integration-tests/tests/tts_pipeline.rs
use xybrid_core::*;

#[test]
fn test_full_tts_pipeline() {
    // End-to-end test
}
```

---

## Documentation

### Module Documentation

```rust
//! # Template Executor
//!
//! Metadata-driven model execution without hard-coding model-specific logic.
//!
//! ## Overview
//!
//! The `TemplateExecutor` interprets `model_metadata.json` to run:
//! - Preprocessing (tokenization, mel spectrogram, phonemization)
//! - Model inference (ONNX, Candle)
//! - Postprocessing (CTC decode, audio encoding)
//!
//! ## Example
//!
//! ```rust,no_run
//! use xybrid_core::template_executor::TemplateExecutor;
//!
//! let mut executor = TemplateExecutor::with_base_path("models/whisper");
//! let output = executor.execute(&metadata, &input)?;
//! ```
```

### Type Documentation

```rust
/// A typed container for data flowing through the inference pipeline.
///
/// `Envelope` wraps different data types (audio, text, embeddings) with
/// optional metadata for passing auxiliary information between stages.
///
/// # Serialization
///
/// Envelopes can be serialized to bytes (bincode) or JSON:
/// - `to_bytes()` / `from_bytes()` - Efficient binary format
/// - `to_json()` / `from_json()` - Human-readable, for debugging
pub struct Envelope {
    /// The payload type and data.
    pub kind: EnvelopeKind,
    /// Auxiliary key-value metadata.
    pub metadata: HashMap<String, String>,
}
```

### Function Documentation

```rust
/// Execute model inference using the provided metadata configuration.
///
/// # Arguments
///
/// * `metadata` - Model configuration from `model_metadata.json`
/// * `input` - Input data wrapped in an `Envelope`
///
/// # Returns
///
/// The inference result as an `Envelope`, or an error if execution fails.
///
/// # Errors
///
/// Returns `AdapterError::InvalidInput` if the input type doesn't match
/// what the model expects (e.g., audio for a text model).
pub fn execute(&mut self, metadata: &ModelMetadata, input: &Envelope) -> AdapterResult<Envelope>
```

---

## Anti-Patterns to Avoid

### 1. Stringly-Typed APIs

```rust
// AVOID
fn route(target: &str) -> Result<()> {
    match target {
        "device" => ...,
        "cloud" => ...,
        _ => Err("unknown target"),
    }
}

// PREFER
fn route(target: ExecutionTarget) -> Result<()> {
    match target {
        ExecutionTarget::Device => ...,
        ExecutionTarget::Cloud => ...,
    }
}
```

### 2. Unwrap in Library Code

```rust
// AVOID
let value = map.get("key").unwrap();

// PREFER
let value = map.get("key")
    .ok_or_else(|| AdapterError::InvalidInput("missing 'key'".into()))?;
```

### 3. Ignoring Results

```rust
// AVOID
let _ = file.write_all(data);

// PREFER
file.write_all(data)?;
// or if truly ignorable:
let _ = file.write_all(data); // Intentionally ignored because...
```

### 4. Deep Nesting

```rust
// AVOID
if let Some(a) = opt_a {
    if let Some(b) = a.get_b() {
        if let Some(c) = b.get_c() {
            // ...
        }
    }
}

// PREFER
let a = opt_a.ok_or(Error::MissingA)?;
let b = a.get_b().ok_or(Error::MissingB)?;
let c = b.get_c().ok_or(Error::MissingC)?;
// ...
```

### 5. Premature Abstraction

```rust
// AVOID: Creating traits for single implementations
trait Processor {
    fn process(&self, input: &Envelope) -> Result<Envelope>;
}

struct AudioProcessor;
impl Processor for AudioProcessor { ... }

// PREFER: Direct implementation until you have multiple implementations
fn process_audio(input: &Envelope) -> Result<Envelope> { ... }
```

### 6. God Objects

Split large structs into focused components:

```rust
// AVOID
struct InferenceEngine {
    policy: PolicyConfig,
    routing: RoutingConfig,
    preprocessing: Vec<Step>,
    model: Model,
    postprocessing: Vec<Step>,
    cache: Cache,
    telemetry: Telemetry,
    // ... 20 more fields
}

// PREFER: Layered architecture
struct Orchestrator { policy: PolicyEngine, routing: RoutingEngine, executor: Executor }
struct Executor { adapters: Vec<RuntimeAdapter>, loader: ModelLoader }
struct TemplateExecutor { session_cache: SessionCache, preprocessors: Preprocessors }
```

---

## Performance Guidelines

### 1. Avoid Unnecessary Allocations

```rust
// AVOID: Allocates on every call
fn get_name(&self) -> String {
    self.name.clone()
}

// PREFER: Return reference when possible
fn name(&self) -> &str {
    &self.name
}
```

### 2. Use Cow for Flexible Ownership

```rust
use std::borrow::Cow;

fn process(input: Cow<'_, str>) -> Result<String> {
    if needs_modification(&input) {
        Ok(modify(input.into_owned()))
    } else {
        Ok(input.into_owned())
    }
}
```

### 3. Cache Expensive Operations

```rust
pub struct TemplateExecutor {
    /// Cache ONNX sessions to avoid reloading
    session_cache: HashMap<PathBuf, Arc<Session>>,
}
```

### 4. Use Appropriate Collection Types

| Use Case | Type |
|----------|------|
| Key-value lookup | `HashMap` |
| Ordered key-value | `BTreeMap` |
| Unique items | `HashSet` |
| Small fixed-size | `[T; N]` or `SmallVec` |
| Growable sequence | `Vec` |

---

## Async Guidelines

### When to Use Async

- Network I/O (HTTP requests, cloud APIs)
- File I/O in hot paths
- Concurrent operations

### When NOT to Use Async

- CPU-bound computation (use threads instead)
- Simple synchronous operations
- Internal library code that doesn't do I/O

### Async Conventions

```rust
// Use tokio as the async runtime
use tokio::fs;

// Async functions return impl Future
pub async fn fetch_model(&self, model_id: &str) -> Result<PathBuf> {
    let response = self.client.get(url).await?;
    // ...
}

// Use spawn_blocking for CPU-bound work
let result = tokio::task::spawn_blocking(move || {
    expensive_computation(&data)
}).await?;
```

---

## Platform-Specific Code

### Conditional Compilation

```rust
// Platform modules
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod apple;

#[cfg(target_os = "android")]
pub mod android;

// Platform-specific implementations
#[cfg(target_os = "macos")]
fn detect_gpu() -> Option<GpuInfo> {
    // Metal detection
}

#[cfg(not(target_os = "macos"))]
fn detect_gpu() -> Option<GpuInfo> {
    // Fallback
}
```

### Feature-Gated Dependencies

```toml
[target.'cfg(any(target_os = "macos", target_os = "ios"))'.dependencies]
metal = "0.27"
```

---

## Checklist for Code Review

Before submitting code, verify:

- [ ] Uses `TemplateExecutor` for model inference (not raw ONNX)
- [ ] Data flows through `Envelope` types
- [ ] Errors use `thiserror` in library code
- [ ] No `unwrap()` or `expect()` in library code (use `?` instead)
- [ ] Public items have doc comments
- [ ] Feature flags used for optional dependencies
- [ ] Tests cover the happy path and at least one error case
- [ ] No stringly-typed APIs (use enums)
- [ ] Functions validate inputs early
- [ ] No premature abstractions

---

## Quick Reference

### Common Imports

```rust
// Error handling
use anyhow::{Context, Result};  // Application code
use thiserror::Error;            // Library code

// Core types
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::execution_template::ModelMetadata;
use xybrid_core::template_executor::TemplateExecutor;
use xybrid_core::runtime_adapter::{AdapterError, AdapterResult};

// Serialization
use serde::{Deserialize, Serialize};

// Collections
use std::collections::HashMap;
use std::path::{Path, PathBuf};
```

### Running Tests

```bash
# All tests
cargo test

# Specific crate
cargo test -p xybrid-core

# With features
cargo test -p xybrid-core --features candle

# Single test
cargo test test_name

# With output
cargo test -- --nocapture
```

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# With features
cargo build --features candle,candle-metal

# Check without building
cargo check
```
