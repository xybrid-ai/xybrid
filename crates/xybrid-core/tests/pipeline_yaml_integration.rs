//! Integration tests for YAML pipeline execution with cross-layer support.
//!
//! These tests verify that pipelines defined in YAML format correctly execute
//! across different layers (device, cloud, integration).
//!
//! The cross-layer pipeline is Xybrid's core feature:
//! ASR (device) -> LLM (integration) -> TTS (device)
//!
//! NOTE: Most tests are ignored — they reference `target: integration` which
//! is not yet a valid ExecutionTarget variant.

use xybrid_core::context::DeviceMetrics;
use xybrid_core::pipeline::{IntegrationProvider, PipelineRunner, RunnerConfig};

/// Test parsing YAML pipeline with integration stage.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_yaml_pipeline_with_integration_stage_parses() {
    let yaml = r#"
name: "Voice Assistant Pipeline"
version: "1.0"

input:
  type: audio
  sample_rate: 16000
  channels: 1

stages:
  - id: asr
    model: wav2vec2-base-960h
    version: "1.0"
    target: device

  - id: llm
    model: gpt-4o-mini
    target: integration
    provider: openai
    options:
      temperature: 0.7
      max_tokens: 500
      system_prompt: "You are a helpful voice assistant."

  - id: tts
    model: kokoro-82m
    version: "0.1"
    target: device
"#;

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(yaml);
    assert!(
        pipeline.is_ok(),
        "Failed to parse YAML: {:?}",
        pipeline.err()
    );

    let pipeline = pipeline.unwrap();
    assert_eq!(pipeline.stages.len(), 3);

    // Verify ASR stage (device)
    let asr = &pipeline.stages[0];
    assert_eq!(asr.id, "asr");
    assert_eq!(asr.model, "wav2vec2-base-960h");
    assert!(matches!(
        asr.target,
        xybrid_core::pipeline::ExecutionTarget::Device
    ));

    // Verify LLM stage (integration)
    let llm = &pipeline.stages[1];
    assert_eq!(llm.id, "llm");
    assert_eq!(llm.model, "gpt-4o-mini");
    assert!(matches!(
        llm.target,
        xybrid_core::pipeline::ExecutionTarget::Cloud
    ));
    assert_eq!(llm.provider, Some(IntegrationProvider::OpenAI));

    // Verify TTS stage (device)
    let tts = &pipeline.stages[2];
    assert_eq!(tts.id, "tts");
    assert_eq!(tts.model, "kokoro-82m");
    assert!(matches!(
        tts.target,
        xybrid_core::pipeline::ExecutionTarget::Device
    ));
}

/// Test parsing YAML pipeline with Anthropic provider.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_yaml_pipeline_with_anthropic_provider() {
    let yaml = r#"
name: "Anthropic Pipeline"
version: "1.0"

input:
  type: text

stages:
  - id: llm
    model: claude-3-5-sonnet-20241022
    target: integration
    provider: anthropic
    options:
      temperature: 0.8
      max_tokens: 1000
      system_prompt: "You are Claude."
"#;

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(yaml);
    assert!(pipeline.is_ok());

    let pipeline = pipeline.unwrap();
    let llm = &pipeline.stages[0];
    assert_eq!(llm.provider, Some(IntegrationProvider::Anthropic));
    assert_eq!(llm.model, "claude-3-5-sonnet-20241022");
}

/// Test that integration stages are correctly identified in StageDescriptor.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_stage_descriptor_integration_identification() {
    use xybrid_core::context::StageDescriptor;
    use xybrid_core::pipeline::ExecutionTarget;

    // Test is_cloud() with provider
    let stage = StageDescriptor::new("llm").with_provider(IntegrationProvider::OpenAI);
    assert!(stage.is_cloud());
    assert!(!stage.is_device());

    // Test is_device() without provider
    let device_stage = StageDescriptor::new("asr");
    assert!(!device_stage.is_cloud());
    assert!(device_stage.is_device());

    // Test with explicit target but no provider
    let mut explicit_device = StageDescriptor::new("tts");
    explicit_device.target = Some(ExecutionTarget::Device);
    assert!(!explicit_device.is_cloud());
    assert!(explicit_device.is_device());
}

/// Test full cross-layer pipeline execution (device stages use mock execution).
/// Note: Integration stages require actual API keys and network access.
/// This test verifies the pipeline structure and routing, not actual LLM calls.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_cross_layer_pipeline_structure() {
    let yaml = r#"
name: "Cross-Layer Voice Assistant"
version: "1.0"

input:
  type: text

stages:
  - id: process
    model: test-model
    target: device

  - id: llm
    model: gpt-4o-mini
    target: integration
    provider: openai
    options:
      temperature: 0.7
      max_tokens: 100
      system_prompt: "Echo back what you receive."
"#;

    let config = RunnerConfig {
        metrics: DeviceMetrics {
            network_rtt: 100,
            battery: 80,
            temperature: 25.0,
        },
        ..Default::default()
    };

    let mut runner = PipelineRunner::with_config(config);
    runner.register_local_model("test-model", true);
    runner.register_integration(IntegrationProvider::OpenAI, true);

    // Parse the pipeline to verify structure
    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(yaml).unwrap();
    assert_eq!(pipeline.stages.len(), 2);

    // The first stage should be device target
    assert!(matches!(
        pipeline.stages[0].target,
        xybrid_core::pipeline::ExecutionTarget::Device
    ));

    // The second stage should be integration target with OpenAI provider
    assert!(matches!(
        pipeline.stages[1].target,
        xybrid_core::pipeline::ExecutionTarget::Cloud
    ));
    assert_eq!(
        pipeline.stages[1].provider,
        Some(IntegrationProvider::OpenAI)
    );
}

/// Test StageOptions accessor methods for integration configuration.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_stage_options_accessors() {
    use xybrid_core::pipeline::StageOptions;

    let mut options = StageOptions::new();
    options.set("temperature", 0.7f64);
    options.set("max_tokens", 500u32);
    options.set("system_prompt", "You are helpful.".to_string());
    options.set("timeout_ms", 30000u64);

    assert_eq!(options.temperature(), Some(0.7));
    assert_eq!(options.max_tokens(), Some(500));
    assert_eq!(
        options.system_prompt(),
        Some("You are helpful.".to_string())
    );
    assert_eq!(options.timeout_ms(), Some(30000));
}

/// Test that pipeline runner correctly converts StageConfig to StageDescriptor with provider info.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_stage_config_to_descriptor_preserves_provider() {
    let yaml = r#"
name: "Provider Test"
version: "1.0"

input:
  type: text

stages:
  - id: llm
    model: gpt-4o-mini
    target: integration
    provider: openai
    options:
      temperature: 0.5
"#;

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(yaml).unwrap();
    let stage_config = &pipeline.stages[0];

    // Verify StageConfig has provider
    assert_eq!(stage_config.provider, Some(IntegrationProvider::OpenAI));
    assert_eq!(stage_config.model, "gpt-4o-mini");
    assert!(matches!(
        stage_config.target,
        xybrid_core::pipeline::ExecutionTarget::Cloud
    ));

    // Verify options
    let options = &stage_config.options;
    assert_eq!(options.temperature(), Some(0.5));
}

/// Test pipeline with multiple provider types.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_yaml_pipeline_multiple_providers() {
    let yaml = r#"
name: "Multi-Provider Pipeline"
version: "1.0"

input:
  type: text

stages:
  - id: openai_stage
    model: gpt-4o-mini
    target: integration
    provider: openai
    options:
      temperature: 0.7

  - id: anthropic_stage
    model: claude-3-haiku-20240307
    target: integration
    provider: anthropic
    options:
      temperature: 0.5
"#;

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(yaml).unwrap();
    assert_eq!(pipeline.stages.len(), 2);

    assert_eq!(
        pipeline.stages[0].provider,
        Some(IntegrationProvider::OpenAI)
    );
    assert_eq!(
        pipeline.stages[1].provider,
        Some(IntegrationProvider::Anthropic)
    );
}

/// Test the complete voice assistant YAML configuration matches expected structure.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_voice_assistant_yaml_structure() {
    // This is the canonical voice assistant pipeline structure
    let yaml = r#"
name: "Voice Assistant (wav2vec2 -> openai -> kokoro)"

registry: "http://localhost:8080"

stages:
  - id: asr
    model: wav2vec2-base-960h
    version: "1.0"
    target: device

  - id: llm
    model: gpt-4o-mini
    target: integration
    provider: openai
    options:
      temperature: 0.7
      max_tokens: 500
      system_prompt: "You are a helpful voice assistant. Keep responses concise and conversational."

  - id: tts
    model: kokoro-82m
    version: "0.1"
    target: device

input:
  type: audio
  sample_rate: 16000
  channels: 1

metrics:
  network_rtt: 100
  battery: 80
  temperature: 25.0

availability:
  "wav2vec2-base-960h": true
  "kokoro-82m": true
"#;

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(yaml).unwrap();

    // Verify name
    assert_eq!(
        pipeline.name,
        Some("Voice Assistant (wav2vec2 -> openai -> kokoro)".to_string())
    );

    // Verify registry
    assert_eq!(pipeline.registry, Some("http://localhost:8080".to_string()));

    // Verify 3 stages
    assert_eq!(pipeline.stages.len(), 3);

    // Stage 1: ASR (device)
    let asr = &pipeline.stages[0];
    assert_eq!(asr.id, "asr");
    assert!(matches!(
        asr.target,
        xybrid_core::pipeline::ExecutionTarget::Device
    ));
    assert_eq!(asr.provider, None);

    // Stage 2: LLM (integration)
    let llm = &pipeline.stages[1];
    assert_eq!(llm.id, "llm");
    assert!(matches!(
        llm.target,
        xybrid_core::pipeline::ExecutionTarget::Cloud
    ));
    assert_eq!(llm.provider, Some(IntegrationProvider::OpenAI));
    assert_eq!(
        llm.options.system_prompt(),
        Some(
            "You are a helpful voice assistant. Keep responses concise and conversational."
                .to_string()
        )
    );

    // Stage 3: TTS (device)
    let tts = &pipeline.stages[2];
    assert_eq!(tts.id, "tts");
    assert!(matches!(
        tts.target,
        xybrid_core::pipeline::ExecutionTarget::Device
    ));
    assert_eq!(tts.provider, None);
}

// ============================================================================
// Tests for actual demo YAML files in xybrid_flutter/example/assets/pipelines/
// ============================================================================

/// Test that voice-assistant.yaml parses correctly.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_demo_voice_assistant_yaml_parses() {
    let yaml_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../xybrid_flutter/example/assets/pipelines/voice-assistant.yaml"
    );
    let yaml = std::fs::read_to_string(yaml_path).expect("Failed to read voice-assistant.yaml");

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(&yaml);
    assert!(
        pipeline.is_ok(),
        "Failed to parse voice-assistant.yaml: {:?}",
        pipeline.err()
    );

    let pipeline = pipeline.unwrap();
    assert_eq!(pipeline.stages.len(), 3);

    // Verify cross-layer structure: device -> integration -> device
    assert!(matches!(
        pipeline.stages[0].target,
        xybrid_core::pipeline::ExecutionTarget::Device
    ));
    assert!(matches!(
        pipeline.stages[1].target,
        xybrid_core::pipeline::ExecutionTarget::Cloud
    ));
    assert!(matches!(
        pipeline.stages[2].target,
        xybrid_core::pipeline::ExecutionTarget::Device
    ));

    // Verify LLM provider
    assert_eq!(
        pipeline.stages[1].provider,
        Some(IntegrationProvider::OpenAI)
    );
}

/// Test that voice-assistant-a.yaml parses correctly.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_demo_voice_assistant_a_yaml_parses() {
    let yaml_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../xybrid_flutter/example/assets/pipelines/voice-assistant-a.yaml"
    );
    let yaml = std::fs::read_to_string(yaml_path).expect("Failed to read voice-assistant-a.yaml");

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(&yaml);
    assert!(
        pipeline.is_ok(),
        "Failed to parse voice-assistant-a.yaml: {:?}",
        pipeline.err()
    );

    let pipeline = pipeline.unwrap();
    assert_eq!(pipeline.stages.len(), 3);
    assert_eq!(
        pipeline.stages[1].provider,
        Some(IntegrationProvider::OpenAI)
    );
}

/// Test that voice-assistant-b.yaml parses correctly.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_demo_voice_assistant_b_yaml_parses() {
    let yaml_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../xybrid_flutter/example/assets/pipelines/voice-assistant-b.yaml"
    );
    let yaml = std::fs::read_to_string(yaml_path).expect("Failed to read voice-assistant-b.yaml");

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(&yaml);
    assert!(
        pipeline.is_ok(),
        "Failed to parse voice-assistant-b.yaml: {:?}",
        pipeline.err()
    );

    let pipeline = pipeline.unwrap();
    assert_eq!(pipeline.stages.len(), 3);
    assert_eq!(
        pipeline.stages[1].provider,
        Some(IntegrationProvider::Anthropic)
    );
}

/// Test that hiiipe.yaml parses correctly.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_demo_hiiipe_yaml_parses() {
    let yaml_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../xybrid_flutter/example/assets/pipelines/hiiipe.yaml"
    );
    let yaml = std::fs::read_to_string(yaml_path).expect("Failed to read hiiipe.yaml");

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(&yaml);
    assert!(
        pipeline.is_ok(),
        "Failed to parse hiiipe.yaml: {:?}",
        pipeline.err()
    );

    let pipeline = pipeline.unwrap();
    // hiiipe.yaml has 2 stages (ASR + TTS, no LLM)
    assert_eq!(pipeline.stages.len(), 2);
}

/// Test that speech-to-text.yaml parses correctly.
#[test]
#[ignore = "target: integration not yet a valid ExecutionTarget variant"]
fn test_demo_speech_to_text_yaml_parses() {
    let yaml_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../xybrid_flutter/example/assets/pipelines/speech-to-text.yaml"
    );
    let yaml = std::fs::read_to_string(yaml_path).expect("Failed to read speech-to-text.yaml");

    let pipeline = xybrid_core::pipeline::PipelineConfig::from_yaml(&yaml);
    assert!(
        pipeline.is_ok(),
        "Failed to parse speech-to-text.yaml: {:?}",
        pipeline.err()
    );

    let pipeline = pipeline.unwrap();
    // speech-to-text.yaml has 1 stage (ASR only)
    assert_eq!(pipeline.stages.len(), 1);
}
