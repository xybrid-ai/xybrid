use integration_tests::fixtures;
use xybrid_core::pipeline::PipelineConfig;

#[test]
fn test_downloaded_models_existence() {
    let models_dir = fixtures::models_dir();

    // Skip entirely if model files aren't downloaded (CI doesn't download them)
    let wav2vec2 = models_dir.join("wav2vec2-base-960h/model.onnx");
    if !wav2vec2.exists() {
        eprintln!("Skipping: model fixtures not downloaded (run integration-tests/download.sh)");
        return;
    }

    // If wav2vec2 is present, check the rest
    assert!(
        models_dir.join("kitten-tts/model.fp16.onnx").exists(),
        "kitten-tts model.fp16.onnx missing"
    );
}

#[tokio::test]
async fn test_pipeline_loading() {
    let pipeline_path = fixtures::pipelines_dir().join("simple_integration.yaml");
    let yaml = std::fs::read_to_string(pipeline_path).expect("Failed to read pipeline YAML");

    let pipeline = PipelineConfig::from_yaml(&yaml).expect("Failed to parse pipeline");
    assert_eq!(pipeline.stages.len(), 1);

    // In a real integration test we would run this, but for now we just verify we can see the context
    // and that the environment is set up correctly.

    println!("Pipeline loaded successfully");
}
