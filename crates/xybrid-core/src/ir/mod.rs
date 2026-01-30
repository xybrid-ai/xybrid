//! Intermediate Representation (IR) module - Data serialization layer.
//!
//! This module defines the Envelope IR, which is the serialization layer that
//! defines how data flows between pipeline stages. Envelopes encapsulate typed
//! payloads (audio, text, embeddings) and can be serialized for storage or
//! transmission between local processes or over HTTP to cloud endpoints.

pub mod envelope;

pub use envelope::{AudioSamples, Envelope, EnvelopeKind};
