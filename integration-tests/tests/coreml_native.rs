#![cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]

use std::path::PathBuf;
use std::sync::Arc;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::runtime_adapter::{CoreMLRuntimeAdapter, RuntimeAdapter, RuntimeAdapterExt};

fn model_path() -> PathBuf {
    const FIXTURE: &str = "integration-tests/fixtures/coreml/xybrid_linear.mlpackage";

    let cargo_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/coreml/xybrid_linear.mlpackage");
    if cargo_path.exists() {
        return cargo_path;
    }

    std::env::var_os("TEST_SRCDIR")
        .map(PathBuf::from)
        .map(|runfiles| runfiles.join("_main").join(FIXTURE))
        .unwrap_or(cargo_path)
}

#[test]
fn native_coreml_runs_a_real_model_end_to_end() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = model_path();
    let mut adapter = CoreMLRuntimeAdapter::new();
    adapter.load_model(&model_path.to_string_lossy())?;

    let metadata = adapter.get_metadata("xybrid_linear")?;
    assert_eq!(metadata.input_schema.get("input"), Some(&vec![4]));
    assert_eq!(metadata.output_schema.get("scores"), Some(&vec![3]));

    let output = adapter.execute(&Envelope::new(EnvelopeKind::Embedding(vec![
        1.0, 2.0, 3.0, 4.0,
    ])))?;
    let EnvelopeKind::Embedding(scores) = output.kind else {
        panic!("expected an embedding output");
    };
    assert_eq!(scores.len(), 3);
    for (actual, expected) in scores.iter().zip([0.25, 1.5, 8.0]) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    Ok(())
}

#[test]
fn native_coreml_rejects_the_wrong_tensor_shape() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = model_path();
    let mut adapter = CoreMLRuntimeAdapter::new();
    adapter.load_model(&model_path.to_string_lossy())?;

    let error = adapter
        .execute(&Envelope::new(EnvelopeKind::Embedding(vec![1.0, 2.0])))
        .expect_err("shape mismatch must fail before prediction");
    assert!(error.to_string().contains("expects 4 input values"));
    Ok(())
}

#[test]
fn native_coreml_serializes_concurrent_predictions_safely() -> Result<(), Box<dyn std::error::Error>>
{
    let model_path = model_path();
    let mut adapter = CoreMLRuntimeAdapter::new();
    adapter.load_model(&model_path.to_string_lossy())?;
    let adapter = Arc::new(adapter);

    let workers = (0..8)
        .map(|_| {
            let adapter = Arc::clone(&adapter);
            std::thread::spawn(move || {
                adapter.execute(&Envelope::new(EnvelopeKind::Embedding(vec![
                    1.0, 2.0, 3.0, 4.0,
                ])))
            })
        })
        .collect::<Vec<_>>();

    for worker in workers {
        let output = worker.join().expect("prediction thread must not panic")?;
        assert_eq!(output.kind, EnvelopeKind::Embedding(vec![0.25, 1.5, 8.0]));
    }
    Ok(())
}
