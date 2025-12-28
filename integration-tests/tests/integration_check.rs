use integration_tests::fixtures;
use xybrid_core::pipeline::{PipelineConfig, PipelineRunner, RunnerConfig};
use xybrid_core::context::DeviceMetrics;

#[test]
fn test_downloaded_models_existence() {
    let models_dir = fixtures::models_dir();
    assert!(models_dir.exists(), "Models directory should exist (run xtask setup-test-env)");
    
    // Check specific models
    assert!(models_dir.join("wav2vec2-base-960h/1.0/model.bin").exists());
    assert!(models_dir.join("gpt-4o-mini/1.0/model.bin").exists());
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
