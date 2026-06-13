#![cfg(all(feature = "vision", feature = "llm-llamacpp"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use xybrid_core::testing::model_fixtures;

const MODEL_ID: &str = "gemma-4-e2b";
const MODEL_FILE: &str = "gemma-4-E2B-it-Q8_0.gguf";
const MMPROJ_FILE: &str = "mmproj-gemma-4-E2B-it-Q8_0.gguf";

fn xybrid_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_xybrid"))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn encoded_test_image(width: u32, height: u32) -> Vec<u8> {
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        width,
        height,
        image::Rgb([17, 34, 51]),
    ));
    let mut encoded = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("test image encodes");
    encoded.into_inner()
}

fn link_or_copy_fixture_file(source: &Path, destination: &Path) {
    if std::fs::hard_link(source, destination).is_ok() {
        return;
    }

    #[cfg(unix)]
    if std::os::unix::fs::symlink(source, destination).is_ok() {
        return;
    }

    std::fs::copy(source, destination)
        .unwrap_or_else(|err| panic!("failed to materialize {}: {err}", source.display()));
}

fn materialize_fast_fixture(source_dir: &Path) -> tempfile::TempDir {
    let temp_dir = tempfile::tempdir().expect("create temporary VLM fixture");

    let metadata_path = source_dir.join("model_metadata.json");
    let mut metadata: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&metadata_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", metadata_path.display())),
    )
    .unwrap_or_else(|err| panic!("failed to parse {}: {err}", metadata_path.display()));
    metadata["execution_template"]["generation_params"]["max_tokens"] = serde_json::json!(24);
    std::fs::write(
        temp_dir.path().join("model_metadata.json"),
        serde_json::to_vec_pretty(&metadata).expect("serialize fixture metadata"),
    )
    .expect("write temporary fixture metadata");

    link_or_copy_fixture_file(
        &source_dir.join(MODEL_FILE),
        &temp_dir.path().join(MODEL_FILE),
    );
    link_or_copy_fixture_file(
        &source_dir.join(MMPROJ_FILE),
        &temp_dir.path().join(MMPROJ_FILE),
    );

    temp_dir
}

#[test]
fn run_input_image_captions_gemma_4_e2b_fixture() {
    let Some(source_dir) = model_fixtures::model_or_skip(MODEL_ID) else {
        return;
    };
    if !source_dir.join(MODEL_FILE).exists() || !source_dir.join(MMPROJ_FILE).exists() {
        eprintln!(
            "Skipping: {} requires both {} and {}",
            MODEL_ID, MODEL_FILE, MMPROJ_FILE
        );
        return;
    }

    let model_dir = materialize_fast_fixture(&source_dir);
    let temp_dir = tempfile::tempdir().expect("create CLI test temp dir");
    let image_path = temp_dir.path().join("fixture.png");
    let output_path = temp_dir.path().join("caption.txt");
    std::fs::write(&image_path, encoded_test_image(32, 32)).expect("write fixture image");

    let output = Command::new(xybrid_bin())
        .current_dir(workspace_root())
        .args(["run", "--directory"])
        .arg(model_dir.path())
        .args(["--input-text", "Describe the image in one short sentence."])
        .arg("--input-image")
        .arg(&image_path)
        .arg("--output")
        .arg(&output_path)
        .output()
        .expect("run xybrid CLI VLM command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "xybrid run VLM command failed with exit code {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        stdout,
        stderr
    );

    let caption = std::fs::read_to_string(&output_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", output_path.display()));
    assert!(
        !caption.trim().is_empty(),
        "CLI VLM caption output must not be empty\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );
}
