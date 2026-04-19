//! MLX LLM integration tests (US-014).
//!
//! These tests require:
//! - An Apple host (macOS or iOS simulator).
//! - The `llm-mlx-runtime` feature enabled (which forwards
//!   `xybrid-mlx/bindings` → `mlx-c-sys/bindings` and links
//!   `vendor/mlx-apple/mlx.xcframework`).
//! - A real Qwen 3.5 MLX bundle staged at `$XYBRID_MLX_QWEN_DIR` (the test
//!   harness skips with a clear message when the env var is unset; CI sets
//!   it after fetching the bundle from HuggingFace).
//!
//! Run with:
//! ```bash
//! # Fetch the xcframework first (US-001/US-002 pipeline):
//! ./tools/scripts/fetch-mlx-xcframework.sh
//! # Fetch a Qwen 3.5 MLX bundle (any MLX-LM community bundle works):
//! export XYBRID_MLX_QWEN_DIR=/path/to/Qwen3.5-0.5B-Instruct-mlx
//! cargo test -p xybrid-core --features llm-mlx-runtime --test mlx_llm_chat
//! ```
//!
//! When the harness runs without the env var set the tests SKIP (`return`)
//! rather than fail — this keeps the macOS CI runner happy when the bundle
//! hasn't been staged yet.

#![cfg(all(
    feature = "llm-mlx-runtime",
    any(target_os = "macos", target_os = "ios")
))]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use xybrid_core::runtime_adapter::llm::{ChatMessage, GenerationConfig, LlmBackend};
use xybrid_core::runtime_adapter::mlx::{
    generate::{self, GenerateParams},
    sampler::Sampler,
    MlxLlmAdapter, MlxLlmConfig,
};
use xybrid_core::runtime_adapter::types::{PartialToken, StreamingCallback};

fn bundle_dir() -> Option<PathBuf> {
    std::env::var_os("XYBRID_MLX_QWEN_DIR").map(PathBuf::from)
}

fn short_greedy_config() -> GenerationConfig {
    // 32 tokens matches the PRD's "32-token completion" and is short enough
    // to keep CI runtime bounded. Greedy (temperature=0) removes sampling
    // noise from the correctness checks; the streaming-vs-batched parity
    // test switches to a seeded sampler.
    GenerationConfig {
        max_tokens: 32,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        stop_sequences: Vec::new(),
    }
}

#[test]
fn four_turn_chat_produces_coherent_output() {
    let Some(dir) = bundle_dir() else {
        eprintln!("SKIP: XYBRID_MLX_QWEN_DIR not set");
        return;
    };

    let adapter = MlxLlmAdapter::load(&dir, &MlxLlmConfig::default()).expect("load qwen3 bundle");

    // 4-turn: system + user/assistant/user/assistant/user — the model
    // completes turn #4's assistant response.
    let messages = vec![
        ChatMessage::system("You are a concise, helpful assistant."),
        ChatMessage::user("Say 'apple'."),
        ChatMessage::assistant("apple"),
        ChatMessage::user("Now say 'banana'."),
        ChatMessage::assistant("banana"),
        ChatMessage::user("What came after apple?"),
    ];
    let out =
        LlmBackend::generate(&adapter, &messages, &short_greedy_config()).expect("generate ok");

    // Non-empty output + no template leakage (the assistant turn must not
    // contain the turn markers that the chat template introduced).
    assert!(!out.text.trim().is_empty(), "empty output");
    assert!(
        !out.text.contains("<|im_start|>"),
        "template leakage: {}",
        out.text
    );
    assert!(
        !out.text.contains("<|im_end|>"),
        "template leakage: {}",
        out.text
    );
    assert!(out.tokens_generated > 0);
    assert!(out.tokens_per_second > 0.0);
    assert!(
        matches!(out.finish_reason.as_str(), "stop" | "length"),
        "unexpected finish_reason: {}",
        out.finish_reason
    );
}

#[test]
fn streaming_matches_non_streaming_for_same_seed() {
    let Some(dir) = bundle_dir() else {
        eprintln!("SKIP: XYBRID_MLX_QWEN_DIR not set");
        return;
    };

    let adapter = MlxLlmAdapter::load(&dir, &MlxLlmConfig::default()).expect("load qwen3 bundle");

    // Use a sampling config (temperature > 0) so the seeded sampler
    // actually matters; with greedy decoding the two paths would trivially
    // agree.
    let cfg = GenerationConfig {
        max_tokens: 16,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        min_p: 0.0,
        repetition_penalty: 1.0,
        stop_sequences: Vec::new(),
    };

    let prompt = "Write a single English sentence about oceans.";
    const SEED: u64 = 0x5eed_c0de;

    // Batched path.
    let batched = generate::generate_tokens(
        &adapter,
        prompt,
        GenerateParams::new(&cfg).with_sampler(Sampler::seeded(SEED)),
        None,
    )
    .expect("batched");

    // Streaming path with the SAME seed + SAME prompt. The callback
    // records per-token ids so we can assert parity.
    let recorded: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
    let rec_for_cb = recorded.clone();
    let cb: StreamingCallback<'_> = Box::new(move |t: PartialToken| {
        if let Some(id) = t.token_id {
            rec_for_cb.lock().unwrap().push(id);
        }
        Ok(())
    });
    let streamed = generate::generate_tokens(
        &adapter,
        prompt,
        GenerateParams::new(&cfg).with_sampler(Sampler::seeded(SEED)),
        Some(cb),
    )
    .expect("streamed");

    // Parity: the token counts and the final text must match for the same
    // seed + prompt. The callback must have fired at least once per
    // generated token.
    assert_eq!(batched.text, streamed.text, "streaming text diverged");
    assert_eq!(batched.tokens_generated, streamed.tokens_generated);
    let recorded_ids = recorded.lock().unwrap();
    assert!(
        recorded_ids.len() >= streamed.tokens_generated,
        "callback fired {} times for {} tokens",
        recorded_ids.len(),
        streamed.tokens_generated
    );
    assert!(!recorded_ids.is_empty(), "callback never fired");
}
