//! MLX LLM integration tests (US-014).
//!
//! These tests require:
//! - Apple Silicon macOS.
//! - The `llm-mlx-runtime` feature enabled (which forwards
//!   `xybrid-mlx/bindings` → `xybrid-mlx-sys/bindings` and links
//!   `vendor/mlx-apple/mlx.xcframework`).
//! - Optional real MLX bundles staged at `$XYBRID_MLX_QWEN_DIR` or
//!   `$XYBRID_MLX_QWEN_4B_DIR`, `$XYBRID_MLX_QWEN_4BIT_DIR`,
//!   `$XYBRID_MLX_GEMMA_DIR`,
//!   `$XYBRID_MLX_GEMMA_4BIT_DIR`, `$XYBRID_MLX_LFM_DIR`,
//!   `$XYBRID_MLX_LFM25_DIR`, and `$XYBRID_MLX_LFM25_4BIT_DIR`. Real-fixture tests
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
//! export XYBRID_MLX_QWEN_4BIT_DIR=/path/to/qwen3-4b-4bit-mlx
//! export XYBRID_MLX_GEMMA_DIR=/path/to/gemma4-mlx
//! export XYBRID_MLX_GEMMA_4BIT_DIR=/path/to/gemma-4-e2b-it-4bit
//! export XYBRID_MLX_LFM_DIR=/path/to/lfm2-mlx
//! export XYBRID_MLX_LFM25_DIR=/path/to/lfm2.5-mlx
//! export XYBRID_MLX_LFM25_4BIT_DIR=/path/to/LFM2.5-1.2B-Instruct-MLX-4bit
//! cargo test -p xybrid-core --no-default-features --features llm-mlx-runtime --test mlx_llm_chat
//! ```
//!
//! When the harness runs without the env var set the tests SKIP (`return`)
//! rather than fail, so a run without staged bundles stays green. A lane that
//! does stage bundles (test-mlx.yml) lists their variables in
//! `XYBRID_MLX_REQUIRE`, which turns a missing one into a failure.

#![cfg(all(
    feature = "llm-mlx-runtime",
    target_os = "macos",
    target_arch = "aarch64"
))]

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use xybrid_core::abort::{AbortReason, CloudFallbackAbort};
use xybrid_core::runtime_adapter::llm::{ChatMessage, GenerationConfig, LlmBackend};
use xybrid_core::runtime_adapter::mlx::{
    generate::{self, GenerateParams},
    kv_cache::{KvCache, KV_CACHE_PAGE_TOKENS},
    sampler::Sampler,
    MlxLlmAdapter, MlxLlmConfig, MlxLlmError,
};
use xybrid_core::runtime_adapter::types::{PartialToken, StreamingCallback};
use xybrid_core::runtime_adapter::AdapterError;

/// The bundle staged through `env_var`, or `None` for the test to skip.
///
/// A lane that stages bundles names their variables in `XYBRID_MLX_REQUIRE`
/// (comma-separated). A mistyped variable or a dropped step then fails there
/// instead of passing with every real-weight test skipped.
fn bundle_dir(env_var: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(env_var).map(PathBuf::from);
    let required = std::env::var("XYBRID_MLX_REQUIRE").unwrap_or_default();
    if dir.is_none() && required.split(',').any(|name| name.trim() == env_var) {
        panic!("{env_var} is listed in XYBRID_MLX_REQUIRE but is not set");
    }
    dir
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
        seed: None,
        stop_sequences: Vec::new(),
        ..Default::default()
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
fn synthetic_qwen_generation_config_seed_reproduces_token_streams() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());

    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic qwen3");
    let prompt = "Hello";
    let cfg = sampling_seed_config(0x5eed_c0de);

    let first = streamed_token_ids(&adapter, prompt, &cfg);
    let second = streamed_token_ids(&adapter, prompt, &cfg);

    assert_eq!(first, second);
    assert!(!first.is_empty(), "seeded generation emitted no tokens");
}

#[test]
fn synthetic_qwen_generation_config_seed_changes_token_streams() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());

    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic qwen3");
    let prompt = "Hello";

    let first = streamed_token_ids(&adapter, prompt, &sampling_seed_config(1));
    let second = streamed_token_ids(&adapter, prompt, &sampling_seed_config(999_999));

    assert_ne!(
        first, second,
        "different MLX seeds should diverge on synthetic uniform logits"
    );
}

#[test]
fn synthetic_qwen_streaming_callback_abort_preserves_cloud_fallback_marker() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());

    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic qwen3");
    let seen_tokens = Arc::new(Mutex::new(0usize));
    let seen_tokens_for_cb = seen_tokens.clone();
    let cb: StreamingCallback<'_> = Box::new(move |_token: PartialToken| {
        *seen_tokens_for_cb.lock().unwrap() += 1;
        Err(Box::new(CloudFallbackAbort::new(AbortReason::StressMemory)))
    });

    let err = generate::generate_tokens(
        &adapter,
        "Hello",
        GenerateParams::new(&GenerationConfig {
            max_tokens: 4,
            ..short_greedy_config()
        }),
        Some(cb),
    )
    .expect_err("streaming callback abort must stop MLX generation");
    let err = AdapterError::from(err);

    assert_eq!(
        err.cloud_fallback_abort_reason(),
        Some(AbortReason::StressMemory),
        "MLX callback errors must preserve the typed CloudFallbackAbort marker: {err:?}"
    );
    assert_eq!(
        *seen_tokens.lock().unwrap(),
        1,
        "callback should abort after the first generated token"
    );
}

#[derive(Debug)]
struct AbortAfterTokens {
    limit: usize,
}

impl fmt::Display for AbortAfterTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "abort after {} streamed tokens", self.limit)
    }
}

impl Error for AbortAfterTokens {}

#[test]
fn synthetic_qwen_streaming_abort_after_n_tokens_preserves_error_and_resets_state() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());

    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic qwen3");
    let cfg = GenerationConfig {
        max_tokens: 8,
        ..short_greedy_config()
    };
    let prompt = "Hello";
    const ABORT_AFTER: usize = 3;

    let seen_tokens = Arc::new(Mutex::new(0usize));
    let seen_tokens_for_cb = seen_tokens.clone();
    let cb: StreamingCallback<'_> = Box::new(move |_token: PartialToken| {
        let mut seen = seen_tokens_for_cb.lock().unwrap();
        *seen += 1;
        if *seen >= ABORT_AFTER {
            return Err(Box::new(AbortAfterTokens { limit: ABORT_AFTER }));
        }
        Ok(())
    });

    let err = generate::generate_tokens(&adapter, prompt, GenerateParams::new(&cfg), Some(cb))
        .expect_err("streaming callback abort must stop MLX generation");
    match err {
        MlxLlmError::StreamingCallback(source) => {
            assert!(
                source.downcast_ref::<AbortAfterTokens>().is_some(),
                "callback source was not preserved: {source}"
            );
        }
        other => panic!("expected streaming callback error, got {other:?}"),
    }
    assert!(
        *seen_tokens.lock().unwrap() <= ABORT_AFTER + 1,
        "abort should stop promptly after {ABORT_AFTER} tokens"
    );

    let after_abort = generate::generate_tokens(&adapter, prompt, GenerateParams::new(&cfg), None)
        .expect("same adapter rerun after callback abort");
    assert!(
        after_abort.tokens_generated > 0,
        "post-abort rerun should generate at least one token"
    );
}

/// Tool calling and grammar-constrained decoding are llama.cpp-only. A request
/// that carries either fails on MLX before any token is generated, on every
/// generation entry point, instead of answering without them.
#[test]
fn synthetic_qwen_refuses_tools_and_grammar_on_every_entry_point() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());
    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic qwen3");

    let tool = xybrid_core::gateway::Tool::function(
        "lookup_weather",
        "Look up the weather",
        serde_json::json!({ "type": "object", "properties": {} }),
    );
    let with_tools = short_greedy_config().with_tools([tool]);
    let with_grammar = short_greedy_config().with_grammar("root ::= \"ok\"");
    let messages = [ChatMessage::user("Hello")];

    for (config, refusal) in [
        (&with_tools, "does not support tool calling"),
        (
            &with_grammar,
            "does not support grammar-constrained generation",
        ),
    ] {
        let streamed = Arc::new(Mutex::new(0usize));
        let counting = || -> StreamingCallback<'static> {
            let streamed = streamed.clone();
            Box::new(move |_token: PartialToken| {
                *streamed.lock().unwrap() += 1;
                Ok(())
            })
        };
        let results = [
            LlmBackend::generate(&adapter, &messages, config).map(drop),
            LlmBackend::generate_raw(&adapter, "Hello", config).map(drop),
            LlmBackend::generate_streaming(&adapter, &messages, config, counting()).map(drop),
            LlmBackend::generate_raw_streaming(&adapter, "Hello", config, counting()).map(drop),
        ];
        for result in results {
            let err = result.expect_err("MLX must refuse the request");
            assert!(
                matches!(err, AdapterError::InvalidInput(_)),
                "expected an invalid-input refusal, got {err:?}"
            );
            let message = err.to_string();
            assert!(message.contains(refusal), "{message}");
            assert!(
                message.contains("backend: llamacpp"),
                "the refusal names the backend that can serve the request: {message}"
            );
        }
        assert_eq!(*streamed.lock().unwrap(), 0, "nothing may be generated");
    }
}

/// A streamed chat turn delivers text deltas, then one empty terminal token
/// that carries `finish_reason` and the final text: the contract the llama.cpp
/// backend has, so a consumer needs no per-backend handling. The batch call
/// returns the same text.
#[test]
fn synthetic_qwen_chat_stream_ends_with_one_terminal_token() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());
    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::default()).expect("load synthetic qwen3");
    let config = GenerationConfig {
        max_tokens: 8,
        ..short_greedy_config()
    };
    let messages = [ChatMessage::user("Hello")];

    let seen: Arc<Mutex<Vec<PartialToken>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_for_cb = seen.clone();
    let cb: StreamingCallback<'_> = Box::new(move |token: PartialToken| {
        seen_for_cb.lock().unwrap().push(token);
        Ok(())
    });
    let streamed = LlmBackend::generate_streaming(&adapter, &messages, &config, cb)
        .expect("synthetic qwen chat stream");

    let seen = seen.lock().unwrap();
    let (terminal, deltas) = seen.split_last().expect("a terminal token");
    assert!(
        terminal.token.is_empty(),
        "the terminal token carries no text"
    );
    assert_eq!(
        terminal.finish_reason.as_deref(),
        Some(streamed.finish_reason.as_str())
    );
    assert_eq!(terminal.cumulative_text, streamed.text);
    for (index, token) in seen.iter().enumerate() {
        assert_eq!(token.index, index, "chunk indices count emitted chunks");
    }
    for delta in deltas {
        assert!(
            delta.finish_reason.is_none(),
            "only the terminal token ends the turn"
        );
        assert!(!delta.token.is_empty(), "a delta carries text");
    }
    let streamed_text: String = deltas.iter().map(|t| t.token.as_str()).collect();
    assert_eq!(
        streamed_text.trim(),
        streamed.text,
        "the deltas add up to the answer"
    );

    let batch =
        LlmBackend::generate(&adapter, &messages, &config).expect("synthetic qwen chat batch");
    assert_eq!(batch.text, streamed.text);
    assert_eq!(batch.finish_reason, streamed.finish_reason);
}

/// The SDK loads a model on one thread and runs it on others (`spawn_blocking`
/// for its async entry points), and an application can hold several models.
/// MLX keeps its default streams per thread, so each case needs proof.
fn thread_test_config() -> GenerationConfig {
    GenerationConfig {
        max_tokens: 24,
        ..short_greedy_config()
    }
}

const THREAD_PROMPT_A: &str = "The ocean is";
const THREAD_PROMPT_B: &str = "A list of three fruits:";

fn generate_on_thread(
    adapter: &Arc<MlxLlmAdapter>,
    prompt: &'static str,
) -> std::thread::JoinHandle<Vec<i64>> {
    let adapter = adapter.clone();
    std::thread::spawn(move || streamed_token_ids(&adapter, prompt, &thread_test_config()))
}

/// Weights materialized on the loading thread generate on another thread,
/// with nothing generated on the loading thread first.
fn assert_generates_off_the_loading_thread(dir: &Path) {
    let adapter =
        Arc::new(MlxLlmAdapter::load(dir, &MlxLlmConfig::default()).expect("load bundle"));

    let moved = generate_on_thread(&adapter, THREAD_PROMPT_A)
        .join()
        .expect("generation on another thread");
    assert!(!moved.is_empty());

    let home = streamed_token_ids(&adapter, THREAD_PROMPT_A, &thread_test_config());
    assert_eq!(moved, home, "a model generates the same on any thread");

    let again = generate_on_thread(&adapter, THREAD_PROMPT_A)
        .join()
        .expect("generation on a third thread");
    assert_eq!(again, home);
}

/// Two adapters generate at the same time, each producing what it produces
/// alone.
fn assert_two_models_generate_at_once(dir_a: &Path, dir_b: &Path) {
    let a =
        Arc::new(MlxLlmAdapter::load(dir_a, &MlxLlmConfig::default()).expect("load first bundle"));
    let b =
        Arc::new(MlxLlmAdapter::load(dir_b, &MlxLlmConfig::default()).expect("load second bundle"));
    let alone_a = streamed_token_ids(&a, THREAD_PROMPT_A, &thread_test_config());
    let alone_b = streamed_token_ids(&b, THREAD_PROMPT_B, &thread_test_config());
    assert!(!alone_a.is_empty() && !alone_b.is_empty());

    for round in 0..4 {
        let first = generate_on_thread(&a, THREAD_PROMPT_A);
        let second = generate_on_thread(&b, THREAD_PROMPT_B);
        assert_eq!(
            first.join().expect("first concurrent generation"),
            alone_a,
            "round {round}: first model beside another"
        );
        assert_eq!(
            second.join().expect("second concurrent generation"),
            alone_b,
            "round {round}: second model beside another"
        );
    }
}

/// MLX bytes in use and pooled, in that order.
fn mlx_memory() -> (usize, usize) {
    (
        xybrid_mlx::active_memory().expect("active memory"),
        xybrid_mlx::cache_memory().expect("cache memory"),
    )
}

/// Dropping a model gives its memory back: nothing of it stays in use, and
/// nothing of it stays in MLX's pool waiting for a generation that may never
/// come.
fn assert_dropped_model_returns_its_memory(dir: &Path) {
    xybrid_mlx::clear_cache().expect("clear cache");
    let (active_before, pooled_before) = mlx_memory();

    let adapter = MlxLlmAdapter::load(dir, &MlxLlmConfig::default()).expect("load bundle");
    let (active_loaded, pooled_loaded) = mlx_memory();
    let tokens = streamed_token_ids(&adapter, THREAD_PROMPT_A, &thread_test_config());
    assert!(!tokens.is_empty());
    let (active_generated, pooled_generated) = mlx_memory();
    drop(adapter);
    let (active_after, pooled_after) = mlx_memory();

    eprintln!(
        "mlx memory (active, pooled): before=({active_before}, {pooled_before}) \
         loaded=({active_loaded}, {pooled_loaded}) \
         generated=({active_generated}, {pooled_generated}) \
         dropped=({active_after}, {pooled_after})"
    );
    assert!(
        active_loaded > active_before,
        "a loaded model holds MLX memory"
    );
    // A thread keeps a few constant scalars alive; they are not the model.
    assert!(
        active_after <= active_before + 64,
        "the dropped model still holds {} bytes",
        active_after - active_before
    );
    assert!(
        pooled_after <= pooled_before,
        "MLX still pools {} bytes of the dropped model",
        pooled_after - pooled_before
    );
}

#[test]
fn synthetic_qwen_dropped_model_returns_its_memory() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());

    assert_dropped_model_returns_its_memory(tmp.path());
}

#[test]
fn real_qwen_dropped_model_returns_its_memory_when_staged() {
    let Some(dir) = qwen_bundle_dir() else {
        eprintln!("SKIP: XYBRID_MLX_QWEN_DIR not set");
        return;
    };
    let _guard = mlx_test_lock();

    assert_dropped_model_returns_its_memory(&dir);
}

#[test]
fn synthetic_qwen_generates_off_the_loading_thread() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(tmp.path());

    assert_generates_off_the_loading_thread(tmp.path());
}

#[test]
fn synthetic_qwen_generates_beside_another_model() {
    let _guard = mlx_test_lock();
    let first = tempfile::tempdir().expect("tempdir");
    let second = tempfile::tempdir().expect("tempdir");
    write_synthetic_qwen_bundle(first.path());
    write_synthetic_qwen_bundle(second.path());

    assert_two_models_generate_at_once(first.path(), second.path());
}

#[test]
fn real_qwen_generates_off_the_loading_thread_when_staged() {
    let Some(dir) = qwen_bundle_dir() else {
        eprintln!("SKIP: XYBRID_MLX_QWEN_DIR not set");
        return;
    };
    let _guard = mlx_test_lock();

    assert_generates_off_the_loading_thread(&dir);
}

#[test]
fn real_qwen_generates_beside_another_model_when_staged() {
    let Some(dir) = qwen_bundle_dir() else {
        eprintln!("SKIP: XYBRID_MLX_QWEN_DIR not set");
        return;
    };
    let _guard = mlx_test_lock();

    // Two adapters over the same bundle are two independent models in memory.
    assert_two_models_generate_at_once(&dir, &dir);
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
fn synthetic_gemma_long_context_crosses_sliding_mask_and_cache_page_boundaries() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    const CONTEXT: usize = 8192;
    const PROMPT_TOKENS: usize = 4208;
    write_synthetic_gemma_bundle_with_tiny_tokenizer(tmp.path(), CONTEXT);

    let prompt = long_wordlevel_prompt(PROMPT_TOKENS);
    let tokenizer = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json"))
        .expect("load tiny tokenizer");
    let prompt_token_count = tokenizer
        .encode(prompt.as_str(), false)
        .expect("encode long prompt")
        .get_ids()
        .len();
    assert!(prompt_token_count >= 4000);
    assert_eq!(prompt_token_count, PROMPT_TOKENS);

    let adapter =
        MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::new(CONTEXT)).expect("load long gemma4");
    let cfg = GenerationConfig {
        max_tokens: 32,
        ..short_greedy_config()
    };
    let out = LlmBackend::generate_raw(&adapter, &prompt, &cfg).expect("long-context generation");

    assert!(out.tokens_generated > 0);
    assert!(out.tokens_generated <= 32);
    assert!(matches!(out.finish_reason.as_str(), "stop" | "length"));

    let total_tokens = prompt_token_count + out.tokens_generated;
    let cache = KvCache::new(2, CONTEXT);
    let final_page = cache.page_index(total_tokens - 1);
    assert!(
        final_page >= 16,
        "expected to cross at least 16 {KV_CACHE_PAGE_TOKENS}-token page boundaries, got final page {final_page}"
    );
    let mut previous = 0usize;
    for pos in 0..total_tokens {
        let page = cache.page_index(pos);
        assert!(page >= previous, "KV page index regressed at token {pos}");
        previous = page;
    }
}

#[test]
fn synthetic_gemma_shared_sliding_kv_prefill_uses_current_source_kv() {
    let _guard = mlx_test_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    const CONTEXT: usize = 64;
    const PROMPT_TOKENS: usize = 8;
    write_synthetic_gemma_shared_sliding_bundle(tmp.path(), CONTEXT);

    let prompt = long_wordlevel_prompt(PROMPT_TOKENS);
    let tokenizer = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json"))
        .expect("load tiny tokenizer");
    let prompt_token_count = tokenizer
        .encode(prompt.as_str(), false)
        .expect("encode shared-kv prompt")
        .get_ids()
        .len();
    assert_eq!(prompt_token_count, PROMPT_TOKENS);

    let adapter = MlxLlmAdapter::load(tmp.path(), &MlxLlmConfig::new(CONTEXT))
        .expect("load shared-kv gemma4");
    let out = LlmBackend::generate_raw(
        &adapter,
        &prompt,
        &GenerationConfig {
            max_tokens: 1,
            ..short_greedy_config()
        },
    )
    .expect("shared sliding-kv generation");

    assert_eq!(out.tokens_generated, 1);
    assert!(out.tokens_per_second.is_finite());
    assert!(out.tokens_per_second > 0.0);
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
fn gemma4_quantized_mlx_generates_when_staged() {
    let _guard = mlx_test_lock();
    let Some(dir) = bundle_dir("XYBRID_MLX_GEMMA_4BIT_DIR") else {
        eprintln!("SKIP: XYBRID_MLX_GEMMA_4BIT_DIR not set");
        return;
    };

    assert_quantized_bundle_generates(&dir, "gemma4 4bit", "Say hello in one sentence");
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

#[test]
fn lfm25_quantized_mlx_generates_when_staged() {
    let _guard = mlx_test_lock();
    let Some(dir) = bundle_dir("XYBRID_MLX_LFM25_4BIT_DIR") else {
        eprintln!("SKIP: XYBRID_MLX_LFM25_4BIT_DIR not set");
        return;
    };

    assert_quantized_bundle_generates(&dir, "lfm2.5 4bit", "Say hello in one sentence");
}

#[test]
fn qwen35_quantized_mlx_generates_when_staged() {
    let _guard = mlx_test_lock();
    let Some(dir) = bundle_dir("XYBRID_MLX_QWEN_4BIT_DIR") else {
        eprintln!("SKIP: XYBRID_MLX_QWEN_4BIT_DIR not set");
        return;
    };

    assert_quantized_bundle_generates(&dir, "qwen3 4bit", "Write one short sentence about apples.");
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

    eprintln!("{label} output: {:?}", out.text);
    assert!(!out.text.trim().is_empty(), "empty {label} output");
    assert!(out.tokens_generated > 0);
    assert!(out.tokens_per_second > 0.0);
    assert!(
        matches!(out.finish_reason.as_str(), "stop" | "length"),
        "unexpected finish_reason: {}",
        out.finish_reason
    );
}

fn assert_quantized_bundle_generates(dir: &Path, label: &str, prompt: &str) {
    let adapter = MlxLlmAdapter::load(dir, &MlxLlmConfig::default())
        .unwrap_or_else(|e| panic!("load {label} bundle: {e}"));
    let config = GenerationConfig {
        max_tokens: 8,
        ..short_greedy_config()
    };
    let result = if adapter.chat_template().is_some() {
        LlmBackend::generate(&adapter, &[ChatMessage::user(prompt)], &config)
    } else {
        LlmBackend::generate_raw(&adapter, prompt, &config)
    };
    let out = match result {
        Ok(out) => out,
        Err(err) => {
            let msg = err.to_string();
            assert!(
                !msg.contains("unsupported tensor dtype U32"),
                "{label} hit the old U32 dtype failure: {msg}"
            );
            assert!(
                !msg.contains("unsupported MLX quantization"),
                "{label} hit the old unsupported-quantization failure: {msg}"
            );
            panic!("{label} generate failed: {msg}");
        }
    };

    eprintln!("{label} output: {:?}", out.text);
    assert!(!out.text.trim().is_empty(), "empty {label} output");
    if prompt.to_ascii_lowercase().contains("hello") {
        assert!(
            out.text.to_ascii_lowercase().contains("hello"),
            "{label} did not produce a semantic hello response: {:?}",
            out.text
        );
    }
    assert!(out.tokens_generated > 0);
    assert!(out.tokens_generated <= 8);
    assert!(out.tokens_per_second > 0.0);
    assert!(
        matches!(out.finish_reason.as_str(), "stop" | "length"),
        "unexpected {label} finish_reason: {}",
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
        seed: None,
        stop_sequences: Vec::new(),
        ..Default::default()
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

#[test]
fn real_qwen_abort_then_rerun_matches_fresh_adapter_when_staged() {
    let _guard = mlx_test_lock();
    let Some(dir) = qwen_bundle_dir() else {
        eprintln!("SKIP: neither XYBRID_MLX_QWEN_DIR nor XYBRID_MLX_QWEN_4B_DIR is set");
        return;
    };

    let cfg = GenerationConfig {
        max_tokens: 8,
        ..short_greedy_config()
    };
    let prompt = "Reply with a short phrase about clean runtime state.";

    let adapter = MlxLlmAdapter::load(&dir, &MlxLlmConfig::default()).expect("load qwen3 bundle");
    let cb: StreamingCallback<'_> = Box::new(move |_token: PartialToken| {
        Err(Box::new(CloudFallbackAbort::new(AbortReason::StressMemory)))
    });
    let err = generate::generate_tokens(&adapter, prompt, GenerateParams::new(&cfg), Some(cb))
        .expect_err("streaming callback abort must stop MLX generation");
    let err = AdapterError::from(err);
    assert_eq!(
        err.cloud_fallback_abort_reason(),
        Some(AbortReason::StressMemory),
        "MLX streaming abort must keep the cloud-fallback marker"
    );

    let after_abort = generate::generate_tokens(&adapter, prompt, GenerateParams::new(&cfg), None)
        .expect("same adapter rerun after abort");
    let fresh_adapter =
        MlxLlmAdapter::load(&dir, &MlxLlmConfig::default()).expect("load fresh qwen3 bundle");
    let fresh = generate::generate_tokens(&fresh_adapter, prompt, GenerateParams::new(&cfg), None)
        .expect("fresh adapter generation");

    assert_eq!(
        after_abort.text, fresh.text,
        "abort must not leak KV/decoder state into the next generation"
    );
    assert_eq!(after_abort.tokens_generated, fresh.tokens_generated);
    assert!(
        after_abort.tokens_generated > 0,
        "post-abort rerun should generate at least one token"
    );
}

fn sampling_seed_config(seed: u64) -> GenerationConfig {
    GenerationConfig {
        max_tokens: 12,
        temperature: 0.8,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        stop_sequences: Vec::new(),
        seed: Some(seed),
        ..Default::default()
    }
}

fn streamed_token_ids(
    adapter: &MlxLlmAdapter,
    prompt: &str,
    config: &GenerationConfig,
) -> Vec<i64> {
    let recorded: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
    let rec_for_cb = recorded.clone();
    let cb: StreamingCallback<'_> = Box::new(move |t: PartialToken| {
        if let Some(id) = t.token_id {
            rec_for_cb.lock().unwrap().push(id);
        }
        Ok(())
    });

    generate::generate_tokens(adapter, prompt, GenerateParams::new(config), Some(cb))
        .expect("streamed seeded synthetic generation");
    let ids = recorded.lock().unwrap().clone();
    ids
}

fn long_wordlevel_prompt(tokens: usize) -> String {
    std::iter::repeat_n("a", tokens)
        .collect::<Vec<_>>()
        .join(" ")
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

fn write_synthetic_gemma_bundle_with_tiny_tokenizer(dir: &Path, max_position_embeddings: usize) {
    write_synthetic_gemma_bundle_with_tiny_tokenizer_options(
        dir,
        max_position_embeddings,
        128,
        2,
        None,
    );
}

fn write_synthetic_gemma_shared_sliding_bundle(dir: &Path, max_position_embeddings: usize) {
    write_synthetic_gemma_bundle_with_tiny_tokenizer_options(
        dir,
        max_position_embeddings,
        3,
        4,
        Some(1),
    );
}

fn write_synthetic_gemma_bundle_with_tiny_tokenizer_options(
    dir: &Path,
    max_position_embeddings: usize,
    sliding_window: usize,
    sliding_window_pattern: usize,
    num_kv_shared_layers: Option<usize>,
) {
    std::fs::create_dir_all(dir).expect("create bundle dir");
    write_tiny_wordlevel_tokenizer(dir);

    const HIDDEN: usize = 8;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 4;
    const INTERMEDIATE: usize = 16;
    const LAYERS: usize = 2;
    const VOCAB_SIZE: usize = 8;

    let mut cfg = serde_json::json!({
        "model_type": "gemma4",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": max_position_embeddings,
        "rope_theta": 1_000_000.0,
        "rope_local_base_freq": 10_000.0,
        "rms_norm_eps": 1.0e-6,
        "tie_word_embeddings": true,
        "head_dim": HEAD_DIM,
        "sliding_window": sliding_window,
        "sliding_window_pattern": sliding_window_pattern,
        "query_pre_attn_scalar": HEAD_DIM
    });
    if let Some(shared_layers) = num_kv_shared_layers {
        cfg["num_kv_shared_layers"] = serde_json::json!(shared_layers);
    }
    std::fs::write(dir.join("config.json"), cfg.to_string()).expect("write config");

    let mut tensors = Vec::new();
    push_tensor(
        &mut tensors,
        "model.embed_tokens.weight",
        &[VOCAB_SIZE, HIDDEN],
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

fn write_tiny_wordlevel_tokenizer(dir: &Path) {
    let vocab: HashMap<&str, usize> = [
        ("[UNK]", 0),
        ("a", 1),
        ("b", 2),
        ("c", 3),
        ("d", 4),
        ("e", 5),
        ("f", 6),
        ("g", 7),
    ]
    .into_iter()
    .collect();
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "[UNK]"
        }
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).expect("write tokenizer");
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
