#![cfg(feature = "vision")]

use ndarray::Array4;
use xybrid_core::{
    execution::{ModelMetadata, PreprocessingStep, TemplateExecutor},
    ir::{Envelope, EnvelopeKind},
    testing::model_fixtures,
};

fn encoded_png_and_flattened_nchw() -> (Vec<u8>, Vec<f32>) {
    let mut image = image::RgbImage::new(256, 256);
    let mut tensor = Array4::<f32>::zeros((1, 3, 256, 256));

    for y in 0..256 {
        for x in 0..256 {
            let r = ((x * 3 + y) % 256) as u8;
            let g = ((x + y * 5) % 256) as u8;
            let b = ((x * 7 + y * 11) % 256) as u8;

            image.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            tensor[[0, 0, y, x]] = r as f32 / 255.0;
            tensor[[0, 1, y, x]] = g as f32 / 255.0;
            tensor[[0, 2, y, x]] = b as f32 / 255.0;
        }
    }

    let mut encoded = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("test image encodes");

    (encoded.into_inner(), tensor.iter().copied().collect())
}

fn load_mobilenet_metadata(model_dir: &std::path::Path) -> ModelMetadata {
    let metadata_content = std::fs::read_to_string(model_dir.join("model_metadata.json")).unwrap();
    serde_json::from_str(&metadata_content).unwrap()
}

fn legacy_flattened_metadata(mut metadata: ModelMetadata) -> ModelMetadata {
    metadata.preprocessing = vec![
        PreprocessingStep::Reshape {
            shape: vec![1, 3, 256, 256],
        },
        PreprocessingStep::CenterCrop {
            width: 224,
            height: 224,
        },
        PreprocessingStep::Normalize {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        },
    ];
    metadata
}

fn topk_pairs(envelope: Envelope) -> Vec<(usize, f32)> {
    match envelope.kind {
        EnvelopeKind::Embedding(values) => values
            .chunks_exact(2)
            .map(|pair| (pair[0] as usize, pair[1]))
            .collect(),
        other => panic!("expected embedding TopK output, got {other:?}"),
    }
}

#[test]
fn mobilenet_image_metadata_matches_legacy_flattened_predictions() {
    let Some(model_dir) = model_fixtures::model_or_skip("mobilenet") else {
        return;
    };

    let image_metadata = load_mobilenet_metadata(&model_dir);
    let legacy_metadata = legacy_flattened_metadata(image_metadata.clone());
    let (encoded_png, flattened_nchw) = encoded_png_and_flattened_nchw();

    let mut image_executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let image_output = image_executor
        .execute(
            &image_metadata,
            &Envelope::image(encoded_png, "png").unwrap(),
            None,
        )
        .unwrap();

    let mut legacy_executor = TemplateExecutor::with_base_path(model_dir.to_str().unwrap());
    let legacy_output = legacy_executor
        .execute(
            &legacy_metadata,
            &Envelope::new(EnvelopeKind::Embedding(flattened_nchw)),
            None,
        )
        .unwrap();

    let image_topk = topk_pairs(image_output);
    let legacy_topk = topk_pairs(legacy_output);

    assert_eq!(
        image_topk
            .iter()
            .map(|(class_id, _)| *class_id)
            .collect::<Vec<_>>(),
        legacy_topk
            .iter()
            .map(|(class_id, _)| *class_id)
            .collect::<Vec<_>>()
    );

    for ((image_class, image_score), (legacy_class, legacy_score)) in
        image_topk.iter().zip(legacy_topk.iter())
    {
        assert_eq!(image_class, legacy_class);
        assert!(
            (image_score - legacy_score).abs() < 1e-5,
            "class {image_class}: image score {image_score} != legacy score {legacy_score}"
        );
    }
}
