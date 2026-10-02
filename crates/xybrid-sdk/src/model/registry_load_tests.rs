//! Registry loads with speculation requested.
//!
//! The extracted cache answers first, and an uncached model is resolved once:
//! only a variant the registry describes as a chat model is served from the
//! cloud while it downloads. Each test runs against a local mock registry and
//! a temporary cache, and counts the requests it receives.

use super::*;
use httpmock::prelude::*;
use serde_json::{json, Value};

/// The bytes every test variant downloads as its model file.
const WEIGHTS: &[u8] = b"stand-in model weights";

/// Upper bound on waiting for a background download. It only bounds a hang:
/// the wait returns as soon as the download finishes.
const DOWNLOAD_TIMEOUT_MS: u64 = 60_000;

fn metadata_with_template(id: &str, template: Value) -> Value {
    json!({
        "model_id": id,
        "version": "1.0",
        "execution_template": template,
        "files": ["model.gguf"],
        "metadata": {}
    })
}

fn gguf_metadata(id: &str) -> Value {
    metadata_with_template(id, json!({ "type": "Gguf", "model_file": "model.gguf" }))
}

/// A resolve response for `id` whose variant downloads `file` from `server`.
fn resolve_body(
    server: &MockServer,
    id: &str,
    file: &str,
    passthrough: bool,
    model_metadata: Option<Value>,
) -> Value {
    let mut resolved = json!({
        "hf_repo": format!("xybrid-ai/{id}"),
        "file": file,
        "download_url": server.url(format!("/{file}")),
        "format": "gguf",
        "quantization": "q4_k_m",
        "size_bytes": WEIGHTS.len(),
        "sha256": "",
        "passthrough": passthrough,
    });
    if let Some(model_metadata) = model_metadata {
        resolved["model_metadata"] = model_metadata;
    }
    json!({ "mask": id, "platform": "universal", "resolved": resolved })
}

fn mock_resolve<'a>(server: &'a MockServer, id: &str, body: Value) -> httpmock::Mock<'a> {
    server.mock(|when, then| {
        when.method(GET).path(format!("/v1/models/{id}/resolve"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(body);
    })
}

fn mock_file<'a>(server: &'a MockServer, file: &str, body: &[u8]) -> httpmock::Mock<'a> {
    server.mock(|when, then| {
        when.method(GET).path(format!("/{file}"));
        then.status(200).body(body);
    })
}

/// A client for the registry at `url`, caching under `temp`.
fn client(url: &str, temp: &tempfile::TempDir) -> Arc<RegistryClient> {
    Arc::new(
        RegistryClient::with_url(url)
            .unwrap()
            .with_cache_dir(temp.path().join("cache")),
    )
}

/// What `ModelLoader::load` does for `id` when speculation is enabled and an
/// API key is set.
fn load_with_speculation(
    client: Arc<RegistryClient>,
    id: &str,
    progress: impl Fn(DownloadStatus),
) -> SdkResult<XybridModel> {
    ModelLoader::from_registry(id).load_from_registry_client(client, id, None, true, progress)
}

#[test]
fn a_cached_model_loads_without_the_registry_even_when_it_is_down() {
    let registry = MockServer::start();
    // Every route fails, as an unreachable registry would.
    let any_request = registry.mock(|_, then| {
        then.status(503);
    });
    let temp = tempfile::TempDir::new().unwrap();
    let client = client(&registry.base_url(), &temp);
    let dir = client.extraction_dir("cached-chat");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("model_metadata.json"),
        gguf_metadata("cached-chat").to_string(),
    )
    .unwrap();
    std::fs::write(dir.join("model.gguf"), WEIGHTS).unwrap();

    let model = load_with_speculation(client, "cached-chat", |_| {})
        .expect("a cached model must load without the registry");

    assert_eq!(
        any_request.hits(),
        0,
        "a cached load must not call the registry"
    );
    assert!(
        model.speculative.is_none() && !model.is_cloud_serving(),
        "a cached model must run on the device, so no run can reach the gateway"
    );
    assert!(model.is_loaded());
}

#[test]
fn an_uncached_chat_model_speculates_after_one_resolve_and_downloads_that_variant() {
    let registry = MockServer::start();
    let resolve = mock_resolve(
        &registry,
        "chat",
        resolve_body(
            &registry,
            "chat",
            "model.gguf",
            true,
            Some(gguf_metadata("chat")),
        ),
    );
    let download = mock_file(&registry, "model.gguf", WEIGHTS);
    let temp = tempfile::TempDir::new().unwrap();

    let model = load_with_speculation(client(&registry.base_url(), &temp), "chat", |_| {})
        .expect("an uncached chat model must load");

    // Set when the placeholder is built, so this holds however far the
    // background download has got.
    assert!(
        model.speculative.is_some(),
        "a chat model must be served from the cloud while it downloads"
    );
    let status = model.await_download(DOWNLOAD_TIMEOUT_MS);
    assert_eq!(status.state, DownloadState::Ready, "{status:?}");
    assert_eq!(
        status.downloaded_bytes,
        WEIGHTS.len() as u64,
        "the background download must report its progress"
    );
    assert!(
        model.is_loaded() && !model.is_cloud_serving(),
        "the downloaded weights must take over from the cloud"
    );
    resolve.assert_hits(1);
    download.assert_hits(1);
}

/// Before this check every uncached registry model was speculated, so a TTS or
/// ASR model got a chat placeholder and its requests went to the gateway's
/// chat endpoint.
#[test]
fn a_passthrough_variant_without_chat_metadata_loads_normally() {
    let not_chat = [
        (
            "speech-recognition",
            Some(metadata_with_template(
                "speech-recognition",
                json!({ "type": "GgmlWhisper", "model_file": "model.gguf" }),
            )),
            true,
        ),
        (
            "onnx",
            Some(metadata_with_template(
                "onnx",
                json!({ "type": "Onnx", "model_file": "model.gguf" }),
            )),
            true,
        ),
        // These download, then fail to load exactly as they would with
        // speculation off: the model they describe cannot run here.
        ("missing-metadata", None, false),
        ("malformed-metadata", Some(json!({ "model_id": 7 })), false),
        (
            "unknown-template",
            Some(metadata_with_template(
                "unknown-template",
                json!({ "type": "ChoiceScorer", "scorer": {} }),
            )),
            false,
        ),
    ];

    for (id, metadata, loads) in not_chat {
        let registry = MockServer::start();
        let resolve = mock_resolve(
            &registry,
            id,
            resolve_body(&registry, id, "model.gguf", true, metadata),
        );
        let download = mock_file(&registry, "model.gguf", WEIGHTS);
        let temp = tempfile::TempDir::new().unwrap();
        let progress = Mutex::new(Vec::new());

        let outcome = load_with_speculation(client(&registry.base_url(), &temp), id, |status| {
            progress.lock().unwrap().push(status);
        });

        match outcome {
            Ok(model) => {
                assert!(loads, "{id}: expected the load to fail");
                assert!(
                    model.speculative.is_none() && model.is_loaded(),
                    "{id}: must load on the device, not behind a cloud placeholder"
                );
                let last = *progress
                    .lock()
                    .unwrap()
                    .last()
                    .expect("no progress reported");
                assert_eq!(last.state, DownloadState::Ready, "{id}: {last:?}");
                assert_eq!(last.downloaded_bytes, WEIGHTS.len() as u64, "{id}");
            }
            Err(err) => assert!(!loads, "{id}: expected a local model, got {err}"),
        }
        // One resolve: the load downloaded the variant it had already resolved.
        resolve.assert_hits(1);
        download.assert_hits(1);
    }
}

#[test]
fn a_bundle_is_not_speculated_even_when_it_holds_a_chat_model() {
    let temp = tempfile::TempDir::new().unwrap();
    let staged = temp.path().join("staged");
    std::fs::create_dir_all(&staged).unwrap();
    std::fs::write(
        staged.join("model_metadata.json"),
        gguf_metadata("bundled-chat").to_string(),
    )
    .unwrap();
    std::fs::write(staged.join("model.gguf"), WEIGHTS).unwrap();
    let mut bundle = xybrid_core::bundler::XyBundle::new("bundled-chat", "1.0", "universal");
    bundle.add_file(staged.join("model_metadata.json")).unwrap();
    bundle.add_file(staged.join("model.gguf")).unwrap();
    let bundle_path = temp.path().join("bundled-chat.xyb");
    bundle.write(&bundle_path).unwrap();
    let bundle_bytes = std::fs::read(&bundle_path).unwrap();

    let registry = MockServer::start();
    // Inline chat metadata on a bundle does not count: the bundle's own
    // metadata is only known once it is downloaded.
    let resolve = mock_resolve(
        &registry,
        "bundled-chat",
        resolve_body(
            &registry,
            "bundled-chat",
            "bundled-chat.xyb",
            false,
            Some(gguf_metadata("bundled-chat")),
        ),
    );
    let download = mock_file(&registry, "bundled-chat.xyb", &bundle_bytes);

    let model = load_with_speculation(client(&registry.base_url(), &temp), "bundled-chat", |_| {})
        .expect("a bundled model must load normally");

    assert!(model.speculative.is_none() && model.is_loaded());
    resolve.assert_hits(1);
    download.assert_hits(1);
}

#[test]
fn a_failed_resolve_fails_the_load_instead_of_returning_a_placeholder() {
    let registry = MockServer::start();
    let resolve = registry.mock(|when, then| {
        when.method(GET).path("/v1/models/missing/resolve");
        then.status(404);
    });
    let temp = tempfile::TempDir::new().unwrap();

    let outcome = load_with_speculation(client(&registry.base_url(), &temp), "missing", |_| {});

    let Err(err) = outcome else {
        panic!("a failed resolve must not produce a model");
    };
    assert!(matches!(err, SdkError::ModelNotFound(_)), "{err:?}");
    resolve.assert_hits(1);
}

#[test]
fn an_unreachable_registry_fails_an_uncached_speculative_load() {
    let temp = tempfile::TempDir::new().unwrap();

    // Nothing listens on the discard port, so the connection is refused.
    let outcome = load_with_speculation(client("http://127.0.0.1:9", &temp), "offline", |_| {});

    let Err(err) = outcome else {
        panic!("an unreachable registry must not produce a model");
    };
    assert!(matches!(err, SdkError::Offline { .. }), "{err:?}");
}
