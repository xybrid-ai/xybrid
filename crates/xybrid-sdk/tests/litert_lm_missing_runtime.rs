//! A `.litertlm` model in a build that has the `llm-litert` feature, on a
//! host without the LiteRT-LM runtime library.
//!
//! The library is opened at run time, so its absence is found when the
//! model first runs, not when its directory is loaded. What the caller gets
//! then must be the same typed refusal whether the run was streamed or not.
//!
//! One test, in a binary of its own: it points the library variable at a
//! file that does not exist, and no other test may read the environment
//! while it does.

#![cfg(feature = "llm-litert")]

use serde_json::json;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::litert_lm::LIBRARY_ENV;
use xybrid_core::runtime_adapter::GenerationConfig;
use xybrid_sdk::model::{ModelLoader, SdkError};

const MODEL_FILE: &str = "model.litertlm";

#[test]
fn a_missing_runtime_library_is_the_same_typed_refusal_streamed_or_not() {
    std::env::set_var(LIBRARY_ENV, "/nonexistent/liblitert-lm-for-this-test.so");
    let dir = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(
        dir.path().join("model_metadata.json"),
        json!({
            "model_id": "no-runtime",
            "version": "1.0",
            "execution_template": { "type": "LiteRtLm", "model_file": MODEL_FILE },
            "preprocessing": [],
            "postprocessing": [],
            "files": [MODEL_FILE],
            "metadata": { "task": "text-generation" },
        })
        .to_string(),
    )
    .expect("metadata");
    std::fs::write(dir.path().join(MODEL_FILE), b"never opened").expect("a file");

    // Loading reads the metadata; the runtime is not opened yet.
    let model = ModelLoader::from_directory(dir.path())
        .expect("a loader is built from the model directory")
        .load()
        .expect("the directory loads without the runtime");
    let input = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
    let config = GenerationConfig::greedy();

    let batch = model
        .run(&input, Some(&config))
        .expect_err("there is no runtime to run on");
    let mut tokens = 0usize;
    let streamed = model
        .run_streaming(&input, Some(&config), |_token| {
            tokens += 1;
            Ok(())
        })
        .expect_err("there is no runtime to run on");

    assert_eq!(tokens, 0);
    for err in [&batch, &streamed] {
        match err {
            SdkError::UnsupportedBackendCapability {
                model_id: _,
                backend,
                capability,
                hint,
            } => {
                assert_eq!(backend, "litert-lm");
                assert!(capability.contains("runtime library"), "{capability}");
                assert!(
                    hint.contains(LIBRARY_ENV),
                    "the hint says what to set: {hint}"
                );
                assert!(
                    hint.contains("liblitert-lm-for-this-test"),
                    "the hint names the file that was tried: {hint}"
                );
            }
            other => panic!("expected a missing backend capability, got {other:?}"),
        }
    }
    assert_eq!(batch.to_string(), streamed.to_string());
}
