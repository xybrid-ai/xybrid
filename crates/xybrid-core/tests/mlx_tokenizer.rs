//! Byte-exact parity test for the MLX tokenizer loader.
//!
//! Only compiled on Apple targets with the `llm-mlx` feature on, since the
//! `xybrid_core::runtime_adapter::mlx` module itself is gated that way.
//! The fixture is a Qwen-family `tokenizer.json` checked into the repo at
//! `tests/fixtures/qwen_tokenizer.json`; the expected token sequence below was
//! captured from the reference HuggingFace `tokenizers` Python binding
//! (`Tokenizer.from_file(...).encode("Hello, world!").ids`).

#![cfg(all(feature = "llm-mlx", any(target_os = "macos", target_os = "ios")))]

use std::path::PathBuf;

use xybrid_core::runtime_adapter::mlx::tokenizer;

/// Reference token IDs for "Hello, world!" under the Qwen tokenizer.
const EXPECTED_IDS: &[u32] = &[9707, 11, 1879, 0];

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qwen_tokenizer.json")
}

#[test]
fn qwen_tokenizer_byte_exact_parity() {
    let path = fixture_path();
    let tok = tokenizer::load_from_file(&path)
        .unwrap_or_else(|e| panic!("failed to load fixture at {}: {e}", path.display()));

    let enc = tok
        .encode("Hello, world!", false)
        .expect("tokenizer encode should succeed");

    assert_eq!(
        enc.get_ids(),
        EXPECTED_IDS,
        "Qwen tokenizer output drifted from captured reference — \
         either the fixture changed or the tokenizer library changed behavior"
    );
}
