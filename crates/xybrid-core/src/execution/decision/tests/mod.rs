//! Tests of the LFM3 d1 protocol: unit tests per step (fake vocabularies,
//! synthetic headers, plans and readouts), and the conformance cases of
//! `integration-tests/fixtures/decision/` (`conformance`, and `model` on the
//! real d1-2B vocabulary).

mod cases;
mod conformance;
mod escape;
mod fake;
mod finish;
mod json;
mod labels;
#[cfg(feature = "llm-llamacpp")]
mod model;
#[cfg(feature = "llm-llamacpp")]
mod prepare;
mod protocol;
