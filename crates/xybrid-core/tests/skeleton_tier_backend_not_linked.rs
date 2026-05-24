//! Phase 4 smoke test: when only the skeleton tier (`llm-llamacpp`) is
//! active, `LlamaCppBackend::new()` must surface
//! `AdapterError::BackendNotLinked { backend: "llamacpp" }` rather than
//! attempting to call llama.cpp (which isn't linked).

#![cfg(all(feature = "llm-llamacpp", not(feature = "llm-llamacpp-runtime")))]

use xybrid_core::runtime_adapter::llama_cpp::LlamaCppBackend;
use xybrid_core::runtime_adapter::AdapterError;

#[test]
fn new_returns_backend_not_linked_in_skeleton_tier() {
    match LlamaCppBackend::new() {
        Ok(_) => panic!("skeleton tier must refuse to construct"),
        Err(AdapterError::BackendNotLinked { backend }) => {
            assert_eq!(backend, "llamacpp");
        }
        Err(other) => panic!("expected BackendNotLinked, got {other:?}"),
    }
}
