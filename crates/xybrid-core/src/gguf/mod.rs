//! Typed GGUF header scanning and decision-model classification.
//!
//! # Contents
//!
//! - [`GgufHeader`]: a complete, typed read of a GGUF file's header — the
//!   version, every metadata key in file order with its GGUF type and value
//!   (arrays typed element by element), and every tensor's entry. Pure Rust:
//!   no llama.cpp, no allocation a file's declared sizes could inflate
//!   beyond the bytes actually read, and a ceiling on the bytes read
//!   ([`GgufLimits`]).
//! - [`classify`]: whether the GGUF is a decision model, and of which
//!   implemented profile ([`DecisionClassification`]).
//! - [`native_value`], [`native_f32`], [`effective_f32`]: a value as
//!   llama.cpp's metadata accessors write it, which is what llama.cpp's
//!   decision code reads (temperatures included), and
//!   `verify_native_metadata` (feature `llm-llamacpp`), which checks a
//!   loaded `xybrid_llama::LlamaModel` against the header of its file: a
//!   scalar consistency check (llama.cpp's string view of every scalar
//!   key), not a file identity, since arrays and tensor data are not
//!   compared.
//! - [`admit_gguf`]: the load-boundary check that a model's GGUF is what its
//!   metadata declares: a decision model only as the `GgufDecision` scorer
//!   of its protocol, an ordinary model only as anything else
//!   ([`GgufAdmission`], [`AdmittedDecisionModel`]).
//!
//! # When to use
//!
//! Before any native load. A decision GGUF must never be loaded as an
//! ordinary chat, vision or label model, so every loading boundary
//! (bundles, sidecars, cached models, standalone files, auto-generated
//! metadata) reads the header and classifies it before it allocates a
//! context, and refuses anything that is not [`NotDecision`] or a
//! [`Supported`] profile it implements. [`admit_gguf`] does both for a
//! model's metadata. Reading a header opens and reads the file (only the
//! header: a few MB for a 125,000-token vocabulary); there are no other
//! side effects.
//!
//! llama.cpp's own metadata accessors cannot replace the scan: they skip
//! array-valued keys and stringify scalars, so they cannot prove a key
//! absent. They remain the source of *effective* values, though: llama.cpp
//! reads a decision temperature as its six-decimal string, so the
//! temperature a d1 model runs at is [`effective_f32`] of the stored `f32`,
//! and `verify_native_metadata` checks that llama.cpp's strings are the
//! ones this header predicts.
//!
//! ```no_run
//! use xybrid_core::gguf::{classify, DecisionClassification, GgufHeader};
//!
//! # fn main() -> Result<(), xybrid_core::gguf::GgufError> {
//! let header = GgufHeader::read_file("d1-2B-Q4_K_M.gguf")?;
//! match classify(&header)? {
//!     DecisionClassification::NotDecision => { /* load as before */ }
//!     DecisionClassification::Supported(profile) => println!("decision model: {profile}"),
//!     other => println!("refused: {other:?}"),
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # The classification rule
//!
//! This is the one place the rule is written down; P6's metadata
//! generation and every load boundary call [`classify`] rather than
//! re-deriving it. The test vectors are in `classify.rs`. In order:
//!
//! 1. Only a complete header is classified: [`GgufHeader`] exists only for
//!    a scan that read every key and tensor entry, so a truncated,
//!    malformed or oversized file is an error, never [`NotDecision`].
//! 2. `general.architecture` (`<arch>` below) must be a UTF-8 string;
//!    otherwise an error.
//! 3. A `<x>.decision.type` key with `<x>` other than `<arch>` is an error:
//!    a decision model labelled for another architecture.
//! 4. Without `<arch>.decision.type`: any other `<arch>.decision.*` key is
//!    an error (decision keys without a type); with none,
//!    [`NotDecision`].
//! 5. `<arch>.decision.type` of any type but a UTF-8 string (an array, a
//!    number) is an error, not absence.
//! 6. A decision type other than `lfm3-d1` is [`Unsupported`]: for
//!    `lfm3-d1-omni`, which a later profile admits with its own pins,
//!    [`UnsupportedReason::ProtocolNotYetAdmitted`]; for any other,
//!    [`UnsupportedReason::Protocol`].
//! 7. `lfm3-d1` is the `lfm3-d1/text/v1` profile
//!    ([`DecisionProfileId::Lfm3D1Text`]) when:
//!    - `<arch>` is `lfm3`, else [`UnsupportedReason::Architecture`];
//!    - `lfm3.attention.causal` is the boolean `false` (another type is an
//!      error; `true` or absent, which llama.cpp reads as causal, is
//!      [`UnsupportedReason::CausalAttention`]);
//!    - `tokenizer.chat_template.systemone` is a string (another type is
//!      an error; absent is [`UnsupportedReason::MissingTemplate`]);
//!    - every other `lfm3.decision.*` key is a temperature,
//!      `lfm3.decision.temperature.<name>`, of type `f32` (another type is
//!      an error: a restriction of this profile, since llama.cpp would read
//!      any numeric type through its string accessor) whose effective value
//!      is finite and positive (otherwise an error: llama.cpp refuses such
//!      a model at load). `temperature.image`, which llama.cpp uses for
//!      requests with media, is allowed and unused. Any other decision key,
//!      such as `score_bias`, is [`UnsupportedReason::DecisionKey`].
//!
//!    The architecture is checked first, since the profile's keys are named
//!    after it; then every type and temperature error is reported before
//!    the attention, template and decision-key checks, so a malformed file
//!    is an error even where it is also unsupported.
//!
//! Classification reads facts; it does not check that llama.cpp can load
//! the file. After a load, `xybrid_llama::MemorylessContext::new` asserts
//! the same profile through llama.cpp's view of the model and checks the
//! context llama.cpp built (no memory, non-causal attention).
//!
//! [`NotDecision`]: DecisionClassification::NotDecision
//! [`Supported`]: DecisionClassification::Supported
//! [`Unsupported`]: DecisionClassification::Unsupported

mod admission;
mod classify;
mod error;
mod header;
mod native;

#[cfg(test)]
mod testing;
#[cfg(test)]
mod tests;

pub use admission::{admit_gguf, AdmittedDecisionModel, GgufAdmission};
pub use classify::{classify, DecisionClassification, DecisionProfileId, UnsupportedReason};
pub use error::GgufError;
pub use header::{GgufArray, GgufHeader, GgufLimits, GgufTensorInfo, GgufType, GgufValue};
#[cfg(feature = "llm-llamacpp")]
pub use native::verify_native_metadata;
pub use native::{effective_f32, native_f32, native_value};
