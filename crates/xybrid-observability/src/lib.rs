//! Portable OpenTelemetry instrumentation for AI workloads.
//!
//! `xybrid-observability` turns inference requests, streaming, retries and
//! fallbacks, workflows, tool calls, model loading, feedback, and evaluations
//! into OpenTelemetry traces, metrics, and log events that follow the
//! [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/tree/95aff9cdf605bbc7d7e8f7c4745a693bd0c4eea8).
//! It knows nothing about any particular runtime: llama.cpp, ONNX Runtime,
//! Transformers.js, LiteRT, Apple Foundation Models, or a remote API client all
//! report through the same [`Observer`].
//!
//! The crate never installs global providers. Pass your own
//! tracer/meter/logger providers to [`Observer::builder`], or (feature `otlp`)
//! build an [`otlp::ObservabilityPipeline`] that you own and shut down.
//!
//! ```
//! use xybrid_observability::{
//!     InferenceRequest, InputMessage, Location, Observer, OutputMessage, StartOptions, Usage,
//! };
//!
//! let observer = Observer::disabled(); // or Observer::builder()...build()
//! let op = observer.start_inference(
//!     InferenceRequest::new("chat", "llama.cpp")
//!         .with_model("qwen2.5-0.5b")
//!         .with_location(Location::Local)
//!         .streaming(true),
//!     StartOptions::default(),
//! );
//! op.record_input_messages(&[InputMessage::text("user", "Hi")]);
//! op.record_chunk();
//! op.record_usage(&Usage::tokens(Some(3), Some(5)));
//! op.record_output_messages(&[OutputMessage::assistant_text("Hello!", Some("stop"))]);
//! op.end();
//! ```
//!
//! Content (prompts, outputs, tool payloads) is captured on spans by default
//! once an observer is wired; see [`ContentPolicy`] for metadata-only mode,
//! redaction hooks, and size limits. The behavioral contract shared with the
//! JavaScript package is in `CONTRACT.md`.

#![warn(missing_docs)]

pub mod clock;
pub mod content;
pub mod context;
pub mod error;
mod observer;
mod operation;
pub mod semconv;
pub mod types;

#[cfg(feature = "otlp")]
pub mod otlp;

pub use clock::Clock;
#[cfg(feature = "test-util")]
pub use clock::ManualClock;
pub use content::{
    ContentCapture, ContentPolicy, InputMessage, MediaMetadata, Modality, OutputMessage, Part,
    Redactor, SubstringRedactor,
};
pub use context::{ObservationContext, ObservationRef};
pub use error::{Error, Result};
pub use observer::{Observer, ObserverBuilder};
pub use operation::{
    AttemptOperation, InferenceOperation, ModelLoadOperation, SpanOperation, ToolOperation,
    WorkflowOperation,
};
pub use types::{
    AttemptInfo, AttemptReason, ErrorInfo, Evaluation, Feedback, InferenceRequest, Location,
    ModelLoadInfo, PromptRef, ResourceSample, StartOptions, ToolCallInfo, TraceMode, Usage,
};

/// Re-exported so adapters can attach extension attributes without a direct
/// `opentelemetry` dependency.
pub use opentelemetry::KeyValue;
