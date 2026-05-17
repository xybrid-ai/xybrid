//! MLX LLM integration tests (US-014).
//!
//! These tests require:
//! - Apple Silicon macOS.
//! - The `llm-mlx-runtime` feature enabled (which forwards
//!   `xybrid-mlx/bindings` → `mlx-c-sys/bindings` and links
//!   `vendor/mlx-apple/mlx.xcframework`).
//! - Optional real MLX bundles staged at `$XYBRID_MLX_QWEN_DIR` or
//!   `$XYBRID_MLX_QWEN_4B_DIR`, `$XYBRID_MLX_GEMMA_DIR`,
//!   `$XYBRID_MLX_LFM_DIR`, and `$XYBRID_MLX_LFM25_DIR`. Real-fixture tests
//!   skip with a clear message when their env var is unset; synthetic
//!   Qwen/Gemma/LFM fixtures always run when the linked runtime is available.
//!
//! Run with:
//! ```bash
//! # Materialize the xcframework first from pinned source on Apple Silicon, or
//! # use fetch when vendor/mlx-apple/UPSTREAM_VERSIONS.txt has a download pin:
//! ./tools/scripts/build-local-mlx-xcframework.sh
//! ./tools/scripts/fetch-mlx-xcframework.sh
//! # Optionally fetch real MLX bundles:
//! export XYBRID_MLX_QWEN_4B_DIR=/path/to/qwen3-4b-mlx
//! export XYBRID_MLX_GEMMA_DIR=/path/to/gemma4-mlx
//! export XYBRID_MLX_LFM_DIR=/path/to/lfm2-mlx
//! export XYBRID_MLX_LFM25_DIR=/path/to/lfm2.5-mlx
//! cargo test -p xybrid-core --no-default-features --features llm-mlx-runtime --test mlx_llm_chat
//! ```
//!
//! When the harness runs without the env var set the tests SKIP (`return`)
//! rather than fail — this keeps the macOS CI runner happy when the bundle
//! hasn't been staged yet.

#![cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use xybrid_core::runtime_adapter::llm::{ChatMessage, GenerationConfig, LlmBackend};
use xybrid_core::runtime_adapter::mlx::{
    generate::{self, GenerateParams},
    sampler::Sampler,
    MlxLlmAdapter, MlxLlmConfig,
};
use xybrid_core::runtime_adapter::types::{PartialToken, StreamingCallback};

fn bundle_dir(env_var: &str) -> Option<PathBuf> {
    std::env::var_os(env_var).map(PathBuf::from)
}

fn qwen_bundle_dir() -> Option<PathBuf> {
    bundle_dir("XYBRID_MLX_QWEN_DIR").or_else(|| bundle_dir("XYBRID_MLX_QWEN_4B_DIR"))
}

fn mlx_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("MLX integration test lock poisoned")
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
fn synthetic_qwen_bundle_runs_runtime_forward_without_external_fixture() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());

    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic qwen3");
    let out = LlmBackend::generate_raw(
        &adapter,
        "Hello",
        &GenerationConfig {
            max_tokens: 1,
            ..short_greedy_config()
        },
    )
    .expect("synthetic qwen generation");

    assert_eq!(out.tokens_generated, 1);
    assert!(out.tokens_per_second.is_finite());
    assert!(out.tokens_per_second > 0.0);
    assert!(matches!(out.finish_reason.as_str(), "stop" | "length"));
}

#[test]
fn synthetic_qwen_sharded_bundle_runs_runtime_forward_without_external_fixture() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());
    convert_single_weights_to_indexed_shard(tmp.path());

    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load sharded qwen3");
    let out = LlmBackend::generate_raw(
        &adapter,
        "Hello",
        &GenerationConfig {
            max_tokens: 1,
            ..short_greedy_config()
        },
    )
    .expect("synthetic sharded qwen generation");

    assert_eq!(out.tokens_generated, 1);
    assert!(out.tokens_per_second.is_finite());
    assert!(out.tokens_per_second > 0.0);
    assert!(matches!(out.finish_reason.as_str(), "stop" | "length"));
}

#[test]
fn synthetic_gemma_bundle_runs_runtime_forward_without_external_fixture() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_gemma_bundle(tmp.path());

    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic gemma4");
    let out = LlmBackend::generate_raw(
        &adapter,
        "Hello",
        &GenerationConfig {
            max_tokens: 1,
            ..short_greedy_config()
        },
    )
    .expect("synthetic gemma generation");

    assert_eq!(out.tokens_generated, 1);
    assert!(out.tokens_per_second.is_finite());
    assert!(out.tokens_per_second > 0.0);
    assert!(matches!(out.finish_reason.as_str(), "stop" | "length"));
}

#[test]
fn synthetic_lfm_bundle_runs_runtime_forward_without_external_fixture() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_lfm_bundle(tmp.path());

    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic lfm3");
    let out = LlmBackend::generate_raw(
        &adapter,
        "Hello",
        &GenerationConfig {
            max_tokens: 1,
            ..short_greedy_config()
        },
    )
    .expect("synthetic lfm generation");

    assert_eq!(out.tokens_generated, 1);
    assert!(out.tokens_per_second.is_finite());
    assert!(out.tokens_per_second > 0.0);
    assert!(matches!(out.finish_reason.as_str(), "stop" | "length"));
}

#[test]
fn real_gemma_bundle_generates_when_staged() {
    let _guard = mlx_test_lock();
    let Some(dir) = bundle_dir("XYBRID_MLX_GEMMA_DIR") else {
        eprintln!("SKIP: XYBRID_MLX_GEMMA_DIR not set");
        return;
    };

    let adapter = MlxLlmAdapter::load(&dir, &MlxLlmConfig::default()).expect("load gemma bundle");
    let config = GenerationConfig {
        max_tokens: 8,
        ..short_greedy_config()
    };
    let out = if adapter.chat_template().is_some() {
        LlmBackend::generate(
            &adapter,
            &[ChatMessage::user("Write one short sentence about apples.")],
            &config,
        )
    } else {
        LlmBackend::generate_raw(&adapter, "Write one short sentence about apples.", &config)
    }
    .expect("gemma generate ok");

    assert!(!out.text.trim().is_empty(), "empty Gemma output");
    assert!(out.tokens_generated > 0);
    assert!(out.tokens_per_second > 0.0);
    assert!(
        matches!(out.finish_reason.as_str(), "stop" | "length"),
        "unexpected finish_reason: {}",
        out.finish_reason
    );
}

#[test]
fn real_lfm_bundle_generates_when_staged() {
    let _guard = mlx_test_lock();
    let Some(dir) = bundle_dir("XYBRID_MLX_LFM_DIR") else {
        eprintln!("SKIP: XYBRID_MLX_LFM_DIR not set");
        return;
    };

    assert_lfm_bundle_generates(&dir, "lfm");
}

#[test]
fn real_lfm25_bundle_generates_when_staged() {
    let _guard = mlx_test_lock();
    let Some(dir) = bundle_dir("XYBRID_MLX_LFM25_DIR") else {
        eprintln!("SKIP: XYBRID_MLX_LFM25_DIR not set");
        return;
    };

    assert_lfm_bundle_generates(&dir, "lfm2.5");
}

fn assert_lfm_bundle_generates(dir: &Path, label: &str) {
    let adapter = MlxLlmAdapter::load(dir, &MlxLlmConfig::default()).expect("load lfm bundle");
    let config = GenerationConfig {
        max_tokens: 8,
        ..short_greedy_config()
    };
    let out = if adapter.chat_template().is_some() {
        LlmBackend::generate(
            &adapter,
            &[ChatMessage::user("Write one short sentence about oceans.")],
            &config,
        )
    } else {
        LlmBackend::generate_raw(&adapter, "Write one short sentence about oceans.", &config)
    }
    .expect("lfm generate ok");

    assert!(!out.text.trim().is_empty(), "empty {label} output");
    assert!(out.tokens_generated > 0);
    assert!(out.tokens_per_second > 0.0);
    assert!(
        matches!(out.finish_reason.as_str(), "stop" | "length"),
        "unexpected finish_reason: {}",
        out.finish_reason
    );
}

#[test]
fn four_turn_chat_produces_coherent_output() {
    let _guard = mlx_test_lock();
    let Some(dir) = qwen_bundle_dir() else {
        eprintln!("SKIP: neither XYBRID_MLX_QWEN_DIR nor XYBRID_MLX_QWEN_4B_DIR is set");
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
    let _guard = mlx_test_lock();
    let Some(dir) = qwen_bundle_dir() else {
        eprintln!("SKIP: neither XYBRID_MLX_QWEN_DIR nor XYBRID_MLX_QWEN_4B_DIR is set");
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

struct OwnedTensor {
    name: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn write_synthetic_qwen_bundle(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create bundle dir");

    let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qwen_tokenizer.json");
    std::fs::copy(&tok_src, dir.join("tokenizer.json")).expect("copy tokenizer");

    let tokenizer = tokenizers::Tokenizer::from_file(&tok_src).expect("load tokenizer");
    let vocab_size = tokenizer.get_vocab_size(true);

    const HIDDEN: usize = 16;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const INTERMEDIATE: usize = 32;

    let cfg = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": vocab_size,
        "max_position_embeddings": 128,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1.0e-6,
        "tie_word_embeddings": true,
        "head_dim": HEAD_DIM
    });
    std::fs::write(dir.join("config.json"), cfg.to_string()).expect("write config");

    let mut tensors = Vec::new();
    push_tensor(
        &mut tensors,
        "model.embed_tokens.weight",
        &[vocab_size, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.input_layernorm.weight",
        &[HIDDEN],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.self_attn.q_proj.weight",
        &[HEADS * HEAD_DIM, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.self_attn.k_proj.weight",
        &[KV_HEADS * HEAD_DIM, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.self_attn.v_proj.weight",
        &[KV_HEADS * HEAD_DIM, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.self_attn.o_proj.weight",
        &[HIDDEN, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.self_attn.q_norm.weight",
        &[HEAD_DIM],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.self_attn.k_norm.weight",
        &[HEAD_DIM],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.post_attention_layernorm.weight",
        &[HIDDEN],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.mlp.gate_proj.weight",
        &[INTERMEDIATE, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.mlp.up_proj.weight",
        &[INTERMEDIATE, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.mlp.down_proj.weight",
        &[HIDDEN, INTERMEDIATE],
        0.0,
    );
    push_tensor(&mut tensors, "model.norm.weight", &[HIDDEN], 1.0);

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    tensor.shape.clone(),
                    &tensor.bytes,
                )
                .expect("tensor view"),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors"))
        .expect("write safetensors");
}

fn convert_single_weights_to_indexed_shard(dir: &Path) {
    let single = dir.join("model.safetensors");
    let shard_name = "model-00001-of-00002.safetensors";
    let shard = dir.join(shard_name);
    std::fs::rename(&single, &shard).expect("rename single weights to shard");

    let bytes = std::fs::read(&shard).expect("read shard");
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).expect("read shard metadata");
    let weight_map: serde_json::Map<String, serde_json::Value> = meta
        .tensors()
        .into_keys()
        .map(|name| (name, serde_json::Value::String(shard_name.to_string())))
        .collect();
    let index = serde_json::json!({
        "metadata": {},
        "weight_map": weight_map,
    });
    std::fs::write(dir.join("model.safetensors.index.json"), index.to_string())
        .expect("write safetensors index");
}

fn write_synthetic_gemma_bundle(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create bundle dir");

    let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qwen_tokenizer.json");
    std::fs::copy(&tok_src, dir.join("tokenizer.json")).expect("copy tokenizer");

    let tokenizer = tokenizers::Tokenizer::from_file(&tok_src).expect("load tokenizer");
    let vocab_size = tokenizer.get_vocab_size(true);

    const HIDDEN: usize = 16;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const INTERMEDIATE: usize = 32;
    const LAYERS: usize = 2;

    let cfg = serde_json::json!({
        "model_type": "gemma4",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": vocab_size,
        "max_position_embeddings": 128,
        "rope_theta": 1_000_000.0,
        "rope_local_base_freq": 10_000.0,
        "rms_norm_eps": 1.0e-6,
        "tie_word_embeddings": true,
        "head_dim": HEAD_DIM,
        "sliding_window": 2,
        "sliding_window_pattern": 2,
        "query_pre_attn_scalar": HEAD_DIM
    });
    std::fs::write(dir.join("config.json"), cfg.to_string()).expect("write config");

    let mut tensors = Vec::new();
    push_tensor(
        &mut tensors,
        "model.embed_tokens.weight",
        &[vocab_size, HIDDEN],
        0.0,
    );
    for layer in 0..LAYERS {
        let base = format!("model.layers.{layer}");
        push_tensor(
            &mut tensors,
            &format!("{base}.input_layernorm.weight"),
            &[HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.self_attn.q_proj.weight"),
            &[HEADS * HEAD_DIM, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.self_attn.k_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.self_attn.v_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.self_attn.o_proj.weight"),
            &[HIDDEN, HEADS * HEAD_DIM],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.self_attn.q_norm.weight"),
            &[HEAD_DIM],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.self_attn.k_norm.weight"),
            &[HEAD_DIM],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.post_attention_layernorm.weight"),
            &[HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.pre_feedforward_layernorm.weight"),
            &[HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.post_feedforward_layernorm.weight"),
            &[HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.mlp.up_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
            0.0,
        );
        push_tensor(
            &mut tensors,
            &format!("{base}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
            0.0,
        );
    }
    push_tensor(&mut tensors, "model.norm.weight", &[HIDDEN], 0.0);

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    tensor.shape.clone(),
                    &tensor.bytes,
                )
                .expect("tensor view"),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors"))
        .expect("write safetensors");
}

fn write_synthetic_lfm_bundle(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create bundle dir");

    let tok_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qwen_tokenizer.json");
    std::fs::copy(&tok_src, dir.join("tokenizer.json")).expect("copy tokenizer");

    let tokenizer = tokenizers::Tokenizer::from_file(&tok_src).expect("load tokenizer");
    let vocab_size = tokenizer.get_vocab_size(true);

    const HIDDEN: usize = 16;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const INTERMEDIATE: usize = 32;
    const KERNEL: usize = 3;

    let cfg = serde_json::json!({
        "model_type": "lfm3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": vocab_size,
        "max_position_embeddings": 128,
        "rope_theta": 1_000_000.0,
        "norm_eps": 1.0e-5,
        "tie_word_embeddings": true,
        "head_dim": HEAD_DIM,
        "conv_L_cache": KERNEL,
        "conv_bias": false,
        "layer_types": ["conv", "full_attention"]
    });
    std::fs::write(dir.join("config.json"), cfg.to_string()).expect("write config");

    let mut tensors = Vec::new();
    push_tensor(
        &mut tensors,
        "model.embed_tokens.weight",
        &[vocab_size, HIDDEN],
        0.0,
    );

    push_tensor(
        &mut tensors,
        "model.layers.0.operator_norm.weight",
        &[HIDDEN],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.conv.in_proj.weight",
        &[HIDDEN * 3, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.conv.conv.weight",
        &[HIDDEN, KERNEL, 1],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.conv.out_proj.weight",
        &[HIDDEN, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.ffn_norm.weight",
        &[HIDDEN],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.feed_forward.w1.weight",
        &[INTERMEDIATE, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.feed_forward.w2.weight",
        &[HIDDEN, INTERMEDIATE],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.0.feed_forward.w3.weight",
        &[INTERMEDIATE, HIDDEN],
        0.0,
    );

    push_tensor(
        &mut tensors,
        "model.layers.1.operator_norm.weight",
        &[HIDDEN],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.self_attn.q_proj.weight",
        &[HEADS * HEAD_DIM, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.self_attn.k_proj.weight",
        &[KV_HEADS * HEAD_DIM, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.self_attn.v_proj.weight",
        &[KV_HEADS * HEAD_DIM, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.self_attn.out_proj.weight",
        &[HIDDEN, HEADS * HEAD_DIM],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.self_attn.q_layernorm.weight",
        &[HEAD_DIM],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.self_attn.k_layernorm.weight",
        &[HEAD_DIM],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.ffn_norm.weight",
        &[HIDDEN],
        1.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.feed_forward.w1.weight",
        &[INTERMEDIATE, HIDDEN],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.feed_forward.w2.weight",
        &[HIDDEN, INTERMEDIATE],
        0.0,
    );
    push_tensor(
        &mut tensors,
        "model.layers.1.feed_forward.w3.weight",
        &[INTERMEDIATE, HIDDEN],
        0.0,
    );
    push_tensor(&mut tensors, "model.embedding_norm.weight", &[HIDDEN], 1.0);

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    tensor.shape.clone(),
                    &tensor.bytes,
                )
                .expect("tensor view"),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors"))
        .expect("write safetensors");
}

fn push_tensor(tensors: &mut Vec<OwnedTensor>, name: &str, shape: &[usize], fill: f32) {
    let elems: usize = shape.iter().product();
    let mut bytes = Vec::with_capacity(elems * std::mem::size_of::<f32>());
    for _ in 0..elems {
        bytes.extend_from_slice(&fill.to_le_bytes());
    }
    tensors.push(OwnedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        bytes,
    });
}
