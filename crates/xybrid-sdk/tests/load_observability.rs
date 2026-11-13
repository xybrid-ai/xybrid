//! Model-free exported load attributes, with real host boundary measurements.
//! This separate test binary isolates the process-wide SDK/cache configuration.
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanProcessor};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use xybrid_core::bundler::XyBundle;
use xybrid_core::device::ResourceTelemetryMode;
use xybrid_core::execution::ModelMetadata;
use xybrid_core::tracing::{model_load_stage, RecorderScope};
use xybrid_sdk::observability::{
    self, semconv, ObservabilityConfig, ObservationOptions, Observer, StageCapture,
};
use xybrid_sdk::{DownloadState, ModelLoader, PipelineRef};

#[derive(Clone, Debug, Default)]
struct Memory(Arc<Mutex<Vec<SpanData>>>);

impl SpanProcessor for Memory {
    fn on_start(&self, _: &mut opentelemetry_sdk::trace::Span, _: &opentelemetry::Context) {}
    fn on_end(&self, span: SpanData) {
        self.0.lock().unwrap().push(span);
    }
    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }
    fn shutdown_with_timeout(&self, _: Duration) -> OTelSdkResult {
        Ok(())
    }
}

fn attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|a| a.key.as_str() == key)
        .map(|a| a.value.to_string())
}

#[test]
fn loads_export_resources_conversation_declared_size_and_actual_cache_state() {
    let cache = tempfile::tempdir().unwrap();
    xybrid_sdk::init_sdk_cache_dir(cache.path().join("models"));
    let registry = httpmock::MockServer::start();
    std::env::set_var("XYBRID_REGISTRY_URL", registry.base_url());
    std::env::set_var("XYBRID_RESOURCE_TELEMETRY", "boundary");
    xybrid_sdk::set_api_key("test-speculative-key");

    let memory = Memory::default();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(memory.clone())
        .build();
    let observer = Observer::builder().tracer_provider(&provider).build();
    observability::init_observability(
        ObservabilityConfig::with_observer(observer.clone())
            .resource_telemetry(ResourceTelemetryMode::Boundary),
    )
    .unwrap();
    let options = ObservationOptions::default().with_conversation_id("load-conversation");
    let workflow = observability::start_workflow("loads", &options);
    let scope = workflow.enter();

    let mut local = ModelMetadata::onnx("load-fixture", "1", "model.onnx");
    local
        .metadata
        .insert("size_bytes".into(), serde_json::json!(4));
    let mut speculative = serde_json::to_value(&local).unwrap();
    speculative["model_id"] = "spec-fixture".into();
    speculative["execution_template"] =
        serde_json::json!({"type": "Gguf", "model_file": "model.gguf"});
    speculative["files"] = serde_json::json!(["model.gguf"]);
    for (id, file, metadata) in [
        (
            "load-fixture",
            "model.onnx",
            serde_json::to_value(&local).unwrap(),
        ),
        ("spec-fixture", "model.gguf", speculative),
    ] {
        registry.mock(|when, then| {
            when.path(format!("/v1/models/{id}/resolve"));
            then.status(200).json_body(
                serde_json::json!({"mask": id, "platform": "universal", "resolved": {
                    "hf_repo": format!("test/{id}"), "file": file,
                    "download_url": registry.url(format!("/{file}")),
                    "format": if id == "load-fixture" { "onnx" } else { "gguf" },
                    "quantization": "none", "size_bytes": 4, "sha256": "",
                    "passthrough": true, "model_metadata": metadata
                }}),
            );
        });
        registry.mock(|when, then| {
            when.path(format!("/{file}"));
            then.status(200).body("test");
        });
    }
    let first = ModelLoader::from_registry("load-fixture")
        .with_speculative_cloud(false)
        .observation(options.clone())
        .load()
        .unwrap();
    let first_ref = first.load_observation_ref().unwrap();
    let cached = ModelLoader::from_registry("load-fixture")
        .with_speculative_cloud(false)
        .observation(options.clone())
        .load()
        .unwrap();
    let cached_ref = cached.load_observation_ref().unwrap();

    // A verified cached payload remains a hit when extraction is still needed.
    let bundle_directory = tempfile::tempdir().unwrap();
    let mut bundled = local.clone();
    bundled.model_id = "bundle-fixture".into();
    std::fs::write(
        bundle_directory.path().join("model_metadata.json"),
        serde_json::to_vec(&bundled).unwrap(),
    )
    .unwrap();
    std::fs::write(bundle_directory.path().join("model.onnx"), b"test").unwrap();
    let mut bundle = XyBundle::new("bundle-fixture", "1", "universal");
    bundle
        .add_file(bundle_directory.path().join("model_metadata.json"))
        .unwrap();
    bundle
        .add_file(bundle_directory.path().join("model.onnx"))
        .unwrap();
    let bundle_path = bundle_directory.path().join("bundle.xyb");
    bundle.write(&bundle_path).unwrap();
    let bundle_bytes = std::fs::read(&bundle_path).unwrap();
    registry.mock(|when, then| {
        when.path("/v1/models/bundle-fixture/resolve");
        then.status(200).json_body(serde_json::json!({
            "mask": "bundle-fixture", "platform": "universal", "resolved": {
                "hf_repo": "test/bundle-fixture", "file": "bundle.xyb",
                "download_url": registry.url("/must-not-download"),
                "format": "onnx", "quantization": "none",
                "size_bytes": bundle_bytes.len(),
                "sha256": format!("{:x}", Sha256::digest(&bundle_bytes))
            }
        }));
    });
    let client =
        xybrid_sdk::registry_client::RegistryClient::with_url(registry.base_url()).unwrap();
    let resolved = client.resolve("bundle-fixture", None).unwrap();
    let payload_cache = client.get_cache_path(&resolved);
    std::fs::create_dir_all(payload_cache.parent().unwrap()).unwrap();
    std::fs::write(&payload_cache, &bundle_bytes).unwrap();
    assert!(client.resolve_offline("bundle-fixture").is_none());
    let bundled = ModelLoader::from_registry("bundle-fixture")
        .with_speculative_cloud(false)
        .observation(options.clone())
        .load()
        .unwrap();
    let bundled_ref = bundled.load_observation_ref().unwrap();

    // Exercise the core weight-load hook with actual boundary sampling and known metadata.
    {
        let recorder = StageCapture::new(observer.clone(), workflow.context())
            .for_model("load-fixture", Some(&local));
        let _recorder = RecorderScope::install(Arc::new(recorder));
        drop(model_load_stage("ort"));
        {
            let _load = model_load_stage("ort");
            xybrid_core::tracing::mark_stage_failed();
        }
    }
    let pipeline = PipelineRef::from_yaml(&format!(
        "name: preload\nregistry: {}\nstages:\n  - id: local\n    model: load-fixture\n    target: device\n", registry.base_url()
    )).unwrap().load().unwrap();
    pipeline.load_models().unwrap();

    let background = ModelLoader::from_registry("spec-fixture")
        .with_speculative_cloud(true)
        .observation(options)
        .load()
        .unwrap();
    assert!(background.is_speculative());
    assert_eq!(
        background.await_download(10_000).state,
        DownloadState::Ready
    );
    assert!(!background.is_cloud_serving());
    drop(scope);
    workflow.end();
    let spans = memory.0.lock().unwrap();
    let by_ref = |span_id: &str| {
        spans
            .iter()
            .find(|s| s.span_context.span_id().to_string() == span_id)
            .unwrap()
    };
    assert_eq!(
        attr(by_ref(&first_ref.span_id), semconv::attr::MODEL_CACHE_HIT).as_deref(),
        Some("false")
    );
    assert_eq!(
        attr(by_ref(&cached_ref.span_id), semconv::attr::MODEL_CACHE_HIT).as_deref(),
        Some("true")
    );
    assert_eq!(
        attr(by_ref(&bundled_ref.span_id), semconv::attr::MODEL_CACHE_HIT).as_deref(),
        Some("true"),
        "fresh extraction must preserve the verified bundle-cache hit"
    );
    let loads: Vec<_> = spans
        .iter()
        .filter(|s| attr(s, semconv::attr::OPERATION_NAME).as_deref() == Some("load_model"))
        .collect();
    assert_eq!(
        loads.len(),
        8,
        "explicit, cached, cached bundle, successful/failed lazy hooks, preload, speculative caller and background"
    );
    for load in loads {
        for key in [
            semconv::attr::MODEL_SIZE_BYTES,
            semconv::attr::MODEL_CACHE_HIT,
            semconv::attr::MODEL_FORMAT,
            semconv::attr::RUNTIME_NAME,
        ] {
            assert!(
                load.attributes
                    .iter()
                    .filter(|a| a.key.as_str() == key)
                    .count()
                    <= 1,
                "duplicate exported {key} on {}",
                load.name
            );
        }
        assert_eq!(
            attr(load, semconv::attr::CONVERSATION_ID).as_deref(),
            Some("load-conversation")
        );
        assert_eq!(
            attr(load, semconv::attr::MODEL_SIZE_BYTES).as_deref(),
            Some("4"),
            "model declaration must override archive bytes on {}",
            load.name
        );
        assert!(
            attr(load, semconv::attr::RESOURCE_CPU_UTILIZATION).is_some(),
            "load missing measured CPU: {}",
            load.name
        );
    }
    let speculative_load = spans
        .iter()
        .find(|s| attr(s, "xybrid.load.trigger").as_deref() == Some("speculative_background"))
        .unwrap();
    assert_eq!(
        attr(speculative_load, semconv::attr::MODEL_CACHE_HIT).as_deref(),
        Some("false")
    );
    drop(spans);
    // Resource-off mode and an incomplete weight directory export no invented values.
    std::env::set_var("XYBRID_RESOURCE_TELEMETRY", "off");
    observability::init_observability(ObservabilityConfig::with_observer(observer)).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let missing = ModelMetadata::onnx("missing-size", "1", "missing.onnx");
    std::fs::write(
        directory.path().join("model_metadata.json"),
        serde_json::to_vec(&missing).unwrap(),
    )
    .unwrap();
    let missing_model = ModelLoader::from_directory(directory.path())
        .unwrap()
        .load()
        .unwrap();
    let missing_ref = missing_model.load_observation_ref().unwrap();
    let spans = memory.0.lock().unwrap();
    let missing = spans
        .iter()
        .find(|s| s.span_context.span_id().to_string() == missing_ref.span_id)
        .unwrap();
    for key in [
        semconv::attr::RESOURCE_CPU_UTILIZATION,
        semconv::attr::MODEL_SIZE_BYTES,
        semconv::attr::MODEL_CACHE_HIT,
    ] {
        assert_eq!(attr(missing, key), None, "unknown/disabled {key}");
    }
    drop(spans);
    observability::shutdown_observability(Duration::from_secs(1)).unwrap();
    provider.shutdown().unwrap();
}
