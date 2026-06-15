use ndarray::{ArrayD, IxDyn};
use xybrid_core::{
    execution::{
        ExecutionTemplate, ImageNormalizePreset, ImageResizeMode, ModelMetadata, PreprocessingStep,
        VisionPreprocessingPreset,
    },
    runtime_adapter::{BackendResult, VisionEmbeddings, VisionEncoder},
};

fn metadata_json(vision_encoder: &str, files: &str) -> String {
    format!(
        r#"{{
            "model_id": "gemma-3n-e2b",
            "version": "1.0",
            "execution_template": {{
                "type": "Gguf",
                "model_file": "model.gguf"
            }},
            "vision_encoder": {vision_encoder},
            "files": {files},
            "metadata": {{ "task": "vlm" }}
        }}"#
    )
}

#[test]
fn model_metadata_accepts_and_validates_vision_encoder_block() {
    let json = metadata_json(
        r#"{
            "file": "mmproj.gguf",
            "preprocessing_preset": "gemma3_vision",
            "image_size": 896,
            "patch_size": 14
        }"#,
        r#"["model.gguf", "mmproj.gguf"]"#,
    );

    let metadata: ModelMetadata = serde_json::from_str(&json).unwrap();
    metadata.validate().unwrap();

    let vision = metadata.vision_encoder.as_ref().unwrap();
    assert_eq!(vision.file, "mmproj.gguf");
    assert_eq!(
        vision.preprocessing_preset,
        VisionPreprocessingPreset::Gemma3Vision
    );
    assert_eq!(vision.image_size, 896);
    assert_eq!(vision.patch_size, Some(14));
}

#[test]
fn model_metadata_accepts_vision_language_template_with_encoder_block() {
    let json = r#"{
        "model_id": "gemma-3n-e2b",
        "version": "1.0",
        "execution_template": {
            "type": "VisionLanguage",
            "model_file": "model.gguf",
            "chat_template": "chat_template.json",
            "context_length": 8192
        },
        "vision_encoder": {
            "file": "mmproj.gguf",
            "preprocessing_preset": "gemma3_vision",
            "image_size": 896,
            "patch_size": 14
        },
        "files": ["model.gguf", "chat_template.json", "mmproj.gguf"],
        "metadata": { "task": "vlm" }
    }"#;

    let metadata: ModelMetadata = serde_json::from_str(json).unwrap();
    metadata.validate().unwrap();

    match &metadata.execution_template {
        ExecutionTemplate::VisionLanguage {
            model_file,
            chat_template,
            context_length,
            ..
        } => {
            assert_eq!(model_file, "model.gguf");
            assert_eq!(chat_template.as_deref(), Some("chat_template.json"));
            assert_eq!(*context_length, 8192);
        }
        other => panic!("expected VisionLanguage template, got {other:?}"),
    }
}

#[test]
fn vision_language_template_requires_encoder_block() {
    let json = r#"{
        "model_id": "gemma-3n-e2b",
        "version": "1.0",
        "execution_template": {
            "type": "VisionLanguage",
            "model_file": "model.gguf"
        },
        "files": ["model.gguf"],
        "metadata": { "task": "vlm" }
    }"#;

    let metadata: ModelMetadata = serde_json::from_str(json).unwrap();
    let err = metadata.validate().unwrap_err();

    assert!(err.contains("VisionLanguage"));
    assert!(err.contains("vision_encoder"));
}

#[test]
fn vision_encoder_validation_requires_sibling_file_in_files_array() {
    let json = metadata_json(
        r#"{
            "file": "mmproj.gguf",
            "preprocessing_preset": "clip_vit",
            "image_size": 224
        }"#,
        r#"["model.gguf"]"#,
    );

    let metadata: ModelMetadata = serde_json::from_str(&json).unwrap();
    let err = metadata.validate().unwrap_err();

    assert!(err.contains("mmproj.gguf"));
    assert!(err.contains("files"));
}

#[test]
fn vision_encoder_rejects_unknown_preprocessing_preset() {
    let json = metadata_json(
        r#"{
            "file": "mmproj.gguf",
            "preprocessing_preset": "unknown_vlm",
            "image_size": 224
        }"#,
        r#"["model.gguf", "mmproj.gguf"]"#,
    );

    let err = serde_json::from_str::<ModelMetadata>(&json).unwrap_err();

    assert!(err.to_string().contains("unknown_vlm"));
}

#[test]
fn vision_presets_resolve_to_image_preprocessing_chains() {
    let gemma = VisionPreprocessingPreset::Gemma3Vision.resolve_steps(896);
    assert!(matches!(gemma[0], PreprocessingStep::ImageDecode { .. }));
    assert!(matches!(
        gemma[1],
        PreprocessingStep::ImageResize {
            width: 896,
            height: 896,
            mode: ImageResizeMode::Letterbox,
            ..
        }
    ));
    assert!(matches!(
        gemma[2],
        PreprocessingStep::ImageNormalize {
            preset: ImageNormalizePreset::SigLip,
            ..
        }
    ));

    let gemma4 = VisionPreprocessingPreset::Gemma4Vision.resolve_steps(224);
    assert!(matches!(gemma4[0], PreprocessingStep::ImageDecode { .. }));
    assert!(matches!(
        gemma4[1],
        PreprocessingStep::ImageResize {
            width: 224,
            height: 224,
            mode: ImageResizeMode::Letterbox,
            ..
        }
    ));
    assert!(matches!(
        &gemma4[2],
        PreprocessingStep::ImageNormalize {
            preset: ImageNormalizePreset::Custom { mean, std },
            ..
        } if mean == &[0.0, 0.0, 0.0] && std == &[1.0, 1.0, 1.0]
    ));

    let clip = VisionPreprocessingPreset::ClipVit.resolve_steps(224);
    assert!(matches!(clip[0], PreprocessingStep::ImageDecode { .. }));
    assert!(matches!(
        clip[1],
        PreprocessingStep::ImageResize {
            width: 224,
            height: 224,
            ..
        }
    ));
    assert!(matches!(
        clip[2],
        PreprocessingStep::ImageNormalize {
            preset: ImageNormalizePreset::Clip,
            ..
        }
    ));
}

#[test]
fn vision_encoder_trait_exports_embedding_style_backend_contract() {
    struct StubVisionEncoder;

    impl VisionEncoder for StubVisionEncoder {
        fn encode(&mut self, image_tensor: ArrayD<f32>) -> BackendResult<VisionEmbeddings> {
            assert_eq!(image_tensor.shape(), &[1, 3, 2, 2]);
            Ok(VisionEmbeddings {
                placeholder_tokens: vec![32000, 32001],
                embeddings: ArrayD::zeros(IxDyn(&[2, 4])),
            })
        }
    }

    let mut encoder = StubVisionEncoder;
    let result = encoder.encode(ArrayD::zeros(IxDyn(&[1, 3, 2, 2]))).unwrap();

    assert_eq!(result.placeholder_tokens, vec![32000, 32001]);
    assert_eq!(result.embeddings.shape(), &[2, 4]);
}
