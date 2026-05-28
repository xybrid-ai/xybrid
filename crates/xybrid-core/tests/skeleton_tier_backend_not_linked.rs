//! Phase 4 smoke test: when only the skeleton tier (`llm-llamacpp`) is
//! active, `LlamaCppBackend` may be constructed but all fallible runtime
//! paths must surface `AdapterError::BackendNotLinked { backend: "llamacpp" }`
//! rather than attempting to call llama.cpp (which isn't linked).

#![cfg(all(feature = "llm-llamacpp", not(feature = "llm-llamacpp-runtime")))]

use xybrid_core::runtime_adapter::llama_cpp::LlamaCppBackend;
use xybrid_core::runtime_adapter::llm::{ChatMessage, GenerationConfig, LlmBackend, LlmConfig};
use xybrid_core::runtime_adapter::{AdapterError, AdapterResult};

#[test]
fn new_and_default_construct_in_skeleton_tier() {
    let _backend = LlamaCppBackend::new().expect("skeleton backend should construct");
    let _default = LlamaCppBackend;
    let _default: LlamaCppBackend = default_backend();
}

#[test]
fn runtime_methods_return_backend_not_linked_in_skeleton_tier() {
    let mut backend = LlamaCppBackend::new().expect("skeleton backend should construct");
    let llm_config = LlmConfig::new("missing.gguf");
    let generation_config = GenerationConfig::default();
    let messages = [ChatMessage::user("hello")];

    assert_backend_not_linked(backend.load(&llm_config));
    assert!(!backend.is_loaded());
    assert_backend_not_linked(backend.generate(&messages, &generation_config));
    assert_backend_not_linked(backend.generate_raw("hello", &generation_config));
    assert_backend_not_linked(backend.generate_streaming(
        &messages,
        &generation_config,
        Box::new(|_| Ok(())),
    ));
    assert!(backend.unload().is_ok());
}

fn assert_backend_not_linked<T>(result: AdapterResult<T>) {
    match result {
        Ok(_) => panic!("expected BackendNotLinked"),
        Err(AdapterError::BackendNotLinked { backend }) => {
            assert_eq!(backend, "llamacpp");
        }
        Err(other) => panic!("expected BackendNotLinked, got {other:?}"),
    }
}

fn default_backend<T: Default>() -> T {
    T::default()
}
