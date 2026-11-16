//! `.litertlm` models through the public SDK surface, on a scripted
//! LiteRT-LM runtime: no model and no native library.
//!
//! `FakeRuntime::at` puts a fake behind the model file of a directory the
//! SDK loads, so these run anywhere, CI included. They cover what the SDK
//! adds on top of the backend: which error a caller gets, streamed or not,
//! and whether a pipeline stage streams. `litert_lm_gemma4.rs` covers the
//! real model.
//!
//! ```bash
//! cargo test -p xybrid-sdk --features llm-litert,test-util --test litert_lm_fake
//! ```

#![cfg(all(feature = "llm-litert", feature = "test-util"))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use httpmock::prelude::*;
use serde_json::json;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::litert_lm::testing::{FakeModel, FakeRuntime, Script, Step};
use xybrid_core::runtime_adapter::types::{PartialToken, StreamingCallback};
use xybrid_core::runtime_adapter::GenerationConfig;
use xybrid_sdk::model::{ModelLoader, SdkError, XybridModel};
use xybrid_sdk::run_options::RunOptions;
use xybrid_sdk::Xybrid;

const MODEL_FILE: &str = "model.litertlm";

fn metadata(model_id: &str) -> String {
    json!({
        "model_id": model_id,
        "version": "1.0",
        "execution_template": {
            "type": "LiteRtLm",
            "model_file": MODEL_FILE,
            "context_length": 4096,
        },
        "preprocessing": [],
        "postprocessing": [],
        "files": [MODEL_FILE],
        "metadata": { "task": "vlm" },
    })
    .to_string()
}

/// Write a model directory at `dir` and put a scripted runtime behind its
/// model file.
fn stage(dir: &Path, model_id: &str, fake_model: FakeModel) -> FakeRuntime {
    std::fs::create_dir_all(dir).expect("the model directory");
    std::fs::write(dir.join("model_metadata.json"), metadata(model_id)).expect("metadata");
    let model_path = dir.join(MODEL_FILE);
    std::fs::write(&model_path, b"not a model: a scripted runtime stands in").expect("a file");
    let fake = FakeRuntime::at(&model_path);
    fake.set_model(fake_model)
        .set_render_prefix("<|turn>user\n")
        .set_usage(24, 6);
    fake
}

/// A loaded model on a scripted runtime.
struct Loaded {
    fake: FakeRuntime,
    model: XybridModel,
    _dir: tempfile::TempDir,
}

fn load(fake_model: FakeModel) -> Loaded {
    let dir = tempfile::tempdir().expect("a temporary directory");
    // The fake is found by the exact path the executor opens.
    let root = dir.path().canonicalize().expect("a canonical path");
    let fake = stage(&root, "scripted-litert", fake_model);
    let model = ModelLoader::from_directory(&root)
        .expect("a loader is built from the model directory")
        .load()
        .expect("the model loads");
    Loaded {
        fake,
        model,
        _dir: dir,
    }
}

fn content(text: &str) -> Step {
    Step::text(
        json!({"role": "assistant", "content": [{"type": "text", "text": text}]}).to_string(),
    )
}

fn text(text: &str) -> Envelope {
    Envelope::new(EnvelopeKind::Text(text.to_string()))
}

fn with_image(text: &str) -> Envelope {
    let image = image::RgbImage::from_pixel(16, 16, image::Rgb([220, 20, 20]));
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("the test image encodes");
    let image = Envelope::image(encoded.into_inner(), "png").expect("a valid image");
    Envelope::user_message(text, vec![image]).expect("a multipart input")
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

#[test]
fn a_litertlm_directory_loads_as_a_streaming_chat_model() {
    let loaded = load(FakeModel::default());
    loaded.fake.script(
        "hi",
        Script::new([content("Hello"), content(" there."), Step::End]),
    );
    assert!(loaded.model.is_llm());
    assert!(loaded.model.supports_streaming());
    let config = GenerationConfig::greedy();

    let batch = loaded
        .model
        .run(&text("script:hi Say hello."), Some(&config))
        .expect("the run succeeds");
    let mut emitted = String::new();
    let streamed = loaded
        .model
        .run_streaming(&text("script:hi Say hello."), Some(&config), |token| {
            emitted.push_str(&token.token);
            Ok(())
        })
        .expect("the streaming run succeeds");

    assert_eq!(batch.text(), Some("Hello there."));
    assert_eq!(streamed.text(), Some("Hello there."));
    assert_eq!(emitted, "Hello there.");
    assert_eq!(loaded.fake.turns().len(), 2);
}

// ---------------------------------------------------------------------------
// The same refusal, streamed or not
// ---------------------------------------------------------------------------

fn run_both(
    model: &XybridModel,
    input: &Envelope,
    config: &GenerationConfig,
) -> (SdkError, SdkError) {
    let batch = model
        .run(input, Some(config))
        .expect_err("the batch run is refused");
    let mut tokens = 0usize;
    let streamed = model
        .run_streaming(input, Some(config), |_token| {
            tokens += 1;
            Ok(())
        })
        .expect_err("the streamed run is refused");
    assert_eq!(tokens, 0, "a refused run delivers no token");
    (batch, streamed)
}

/// Something the backend cannot do is a typed refusal however the run was
/// started. A caller must not have to parse an inference error's text to
/// tell "this build cannot" from "this run failed".
#[test]
fn a_capability_the_backend_lacks_is_the_same_error_streamed_or_not() {
    let loaded = load(FakeModel::default());
    let grammar = GenerationConfig::greedy().with_grammar("root ::= \"yes\" | \"no\"");

    let (batch, streamed) = run_both(&loaded.model, &text("Is water wet?"), &grammar);

    for err in [&batch, &streamed] {
        match err {
            SdkError::UnsupportedBackendCapability {
                backend,
                capability,
                ..
            } => {
                assert_eq!(backend, "litert-lm");
                assert!(capability.contains("grammar"), "{capability}");
            }
            other => panic!("expected a missing backend capability, got {other:?}"),
        }
    }
    assert_eq!(batch.to_string(), streamed.to_string());
    assert!(
        loaded.fake.turns().is_empty(),
        "nothing reached the runtime"
    );
}

#[test]
fn a_capability_the_model_lacks_is_the_same_error_streamed_or_not() {
    // No vision encoder in the file.
    let loaded = load(FakeModel::default());

    let (batch, streamed) = run_both(
        &loaded.model,
        &with_image("What is this?"),
        &GenerationConfig::greedy(),
    );

    for err in [&batch, &streamed] {
        assert!(
            matches!(err, SdkError::UnsupportedModelCapability { .. }),
            "expected a missing model capability, got {err:?}"
        );
    }
    assert_eq!(batch.to_string(), streamed.to_string());
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

const PIPELINE_MODEL_ID: &str = "scripted-litert-stage";

/// One SDK cache for the whole test binary: `init_sdk_cache_dir` is a
/// process-global `OnceLock`.
fn shared_cache_root() -> &'static Path {
    static CACHE: OnceLock<(tempfile::TempDir, PathBuf)> = OnceLock::new();
    let (_, root) = CACHE.get_or_init(|| {
        let temp = tempfile::tempdir().expect("temp cache dir");
        let root = temp.path().canonicalize().expect("a canonical path");
        xybrid_sdk::init_sdk_cache_dir(root.join("models"));
        (temp, root)
    });
    root
}

/// The model staged into `<cache>/extracted/<id>/`, where the registry
/// client finds it without downloading.
fn staged_pipeline_model() -> &'static FakeRuntime {
    static STAGED: OnceLock<FakeRuntime> = OnceLock::new();
    STAGED.get_or_init(|| {
        // `extracted/` sits beside the `models/` directory the SDK was given.
        let dir = shared_cache_root()
            .join("extracted")
            .join(PIPELINE_MODEL_ID);
        stage(&dir, PIPELINE_MODEL_ID, FakeModel::default())
    })
}

/// Loopback registry stub: the model is already in the SDK cache, so the
/// download URL is never followed.
fn registry_resolve_mock(server: &MockServer) -> httpmock::Mock<'_> {
    server.mock(|when, then| {
        when.method(GET)
            .path_contains(format!("/v1/models/{PIPELINE_MODEL_ID}/resolve"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "mask": PIPELINE_MODEL_ID,
                "platform": "test",
                "resolved": {
                    "hf_repo": "example/scripted-litert",
                    "file": MODEL_FILE,
                    "download_url": "http://127.0.0.1:9/never-followed.litertlm",
                    "format": "litertlm",
                    "quantization": "mixed",
                    "size_bytes": 1,
                    "sha256": ""
                }
            }));
    })
}

/// A pipeline of one `.litertlm` stage streams: the caller's token callback
/// runs while the model generates, not once at the end or never.
#[test]
fn a_single_litertlm_stage_streams_in_a_pipeline() {
    let fake = staged_pipeline_model();
    fake.script(
        "pipeline",
        Script::new([
            content("One"),
            content(" two"),
            content(" three."),
            Step::End,
        ]),
    );
    let server = MockServer::start();
    let _registry = registry_resolve_mock(&server);
    let yaml = format!(
        "name: litert-streaming\nregistry: {base}\nstages:\n  - id: llm\n    model: {PIPELINE_MODEL_ID}\n    target: device\n",
        base = server.base_url()
    );
    let tokens: Arc<Mutex<Vec<PartialToken>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&tokens);
    let callback: StreamingCallback<'_> = Box::new(move |token| {
        sink.lock().expect("the token log").push(token);
        Ok(())
    });

    let result = Xybrid::run_pipeline_streaming_with_options(
        &yaml,
        &text("script:pipeline Count to three."),
        &RunOptions::default(),
        callback,
    )
    .expect("the pipeline runs");

    assert_eq!(result.text(), Some("One two three."));
    let tokens = tokens.lock().expect("the token log");
    let pieces: Vec<&str> = tokens
        .iter()
        .map(|token| token.token.as_str())
        .filter(|piece| !piece.is_empty())
        .collect();
    assert_eq!(
        pieces,
        ["One", " two", " three."],
        "the stage streamed token by token"
    );
    assert!(
        tokens.last().is_some_and(PartialToken::is_final),
        "the stream ends with a terminal token"
    );
}
