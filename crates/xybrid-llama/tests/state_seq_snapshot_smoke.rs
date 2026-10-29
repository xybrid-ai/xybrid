//! Per-sequence state snapshots restore a cached prefix exactly.
//!
//! `LlamaContext::state_seq_save` / `state_seq_restore` are the prefix-reuse
//! path for models whose cache cannot be truncated by position (recurrent /
//! hybrid, such as LFM2): prefill a shared prefix once, save it, and restore
//! it before each request instead of prefilling it again.
//!
//! Each model below prefills a system-prompt prefix, then answers a question
//! greedily from it. The snapshot path saves the prefix, answers a different
//! question so both the KV cache and any recurrent state move on, restores the
//! snapshot and asks the first question again: the answer must match token for
//! token. It runs on lfm2.5-350m (hybrid) and qwen2.5-0.5b (plain KV cache).
//! It also checks the input llama.cpp would mishandle comes back as an error:
//! a seq_id the context does not hold (-1 would address the whole cache, and
//! restoring into an id outside [-1, 256) hits an assert that aborts the
//! process), and a snapshot from another context. Restoring a snapshot of an
//! empty sequence must empty it, which llama.cpp alone does not do.
//!
//! Run via:
//!
//! ```sh
//! XYBRID_LFM2_GGUF=~/.xybrid/cache/extracted/lfm2.5-350m/LFM2.5-350M-Q4_K_M.gguf \
//! XYBRID_QWEN_GGUF=~/.xybrid/cache/extracted/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//!   cargo test -p xybrid-llama --features bindings \
//!     --test state_seq_snapshot_smoke -- --nocapture --ignored
//! ```
//!
//! `#[ignore]` because it requires the GGUFs on disk. A missing model skips
//! its test.

#![cfg(feature = "bindings")]

use std::error::Error;
use std::fmt;
use std::path::PathBuf;

use xybrid_llama::{backend_init, generate_streaming, LlamaContext, LlamaError, LlamaModel};

const PREFIX: &str = "<|im_start|>system\nYou are Jarv, the ship's AI. Answer in one short \
                      sentence.<|im_end|>\n";
const QUESTION: &str = "<|im_start|>user\nName three colours of a sunset.<|im_end|>\n\
                        <|im_start|>assistant\n";
const OTHER_QUESTION: &str = "<|im_start|>user\nCount from one to ten in words.<|im_end|>\n\
                              <|im_start|>assistant\n";
const ANSWER_TOKENS: usize = 24;

/// `$env_var` first, then each path under `$HOME`.
fn locate_gguf(env_var: &str, home_relative: &[&str]) -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var(env_var) {
        let p = PathBuf::from(shellexpand_tilde(&env_path));
        if p.exists() {
            return Some(p);
        }
    }
    let home = PathBuf::from(std::env::var("HOME").ok()?);
    home_relative
        .iter()
        .map(|rel| home.join(rel))
        .find(|p| p.exists())
}

fn shellexpand_tilde(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    s.to_string()
}

/// Returned by the prefill callback to stop on the first sampled token.
#[derive(Debug)]
struct PrefillDone;

impl fmt::Display for PrefillDone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("prefill done")
    }
}

impl Error for PrefillDone {}

/// Decode `tokens` at `n_past` without adding anything after them.
///
/// The native loop samples a token before the callback runs but only decodes
/// it after, so stopping in the callback leaves exactly `tokens` in the cache.
/// An end-of-generation first token ends the loop undecoded as well.
fn prefill(ctx: &LlamaContext, model: &LlamaModel, tokens: &[i32], n_past: usize) {
    let result = generate_streaming(
        ctx,
        model,
        tokens,
        1,
        0.0,
        1.0,
        0.0,
        0,
        1.0,
        &[],
        None,
        |_token_id, _text| Err(PrefillDone.into()),
        n_past,
    );
    match result {
        Err(LlamaError::StreamingCallbackAborted(err)) if err.is::<PrefillDone>() => {}
        Ok(_) => {}
        Err(other) => panic!("prefill at n_past={n_past} failed: {other:?}"),
    }
}

/// Greedy answer to `tokens`, decoded after the `n_past` tokens in the cache.
fn answer(ctx: &LlamaContext, model: &LlamaModel, tokens: &[i32], n_past: usize) -> Vec<i32> {
    generate_streaming(
        ctx,
        model,
        tokens,
        ANSWER_TOKENS,
        0.0,
        1.0,
        0.0,
        0,
        1.0,
        &[],
        None,
        |_token_id, _text| Ok(()),
        n_past,
    )
    .unwrap_or_else(|e| panic!("generation at n_past={n_past} failed: {e:?}"))
    .0
}

fn snapshot_restores_the_prefix(gguf: PathBuf, expect_recurrent: bool) {
    backend_init();
    let model = LlamaModel::load(gguf.to_str().unwrap(), 0).expect("model must load");
    assert_eq!(model.has_recurrent_state(), expect_recurrent);
    let ctx = LlamaContext::new(&model, 2048, 0, 0, false).expect("context creation");

    let prefix = model
        .tokenize_special(PREFIX, true)
        .expect("tokenize prefix");
    let question = model
        .tokenize_special(QUESTION, false)
        .expect("tokenize question");
    let other = model
        .tokenize_special(OTHER_QUESTION, false)
        .expect("tokenize other question");
    let n_prefix = prefix.len();

    // Reference: the same decode calls as the snapshot path, minus the
    // snapshot, so the outputs can be compared token for token.
    ctx.kv_cache_clear();
    prefill(&ctx, &model, &prefix, 0);
    let reference = answer(&ctx, &model, &question, n_prefix);
    assert!(!reference.is_empty(), "the reference answer is empty");

    ctx.kv_cache_clear();
    prefill(&ctx, &model, &prefix, 0);
    let snapshot = ctx.state_seq_save(0).expect("save the prefix");
    assert!(snapshot.size_bytes() > 0);

    // Move the sequence past the prefix so the restore has state to replace.
    let other_answer = answer(&ctx, &model, &other, n_prefix);
    assert_ne!(
        other_answer, reference,
        "the two questions must lead to different answers for the restore to be tested"
    );

    for round in 1..=2 {
        ctx.state_seq_restore(&snapshot, 0)
            .unwrap_or_else(|e| panic!("restore {round} failed: {e:?}"));
        let restored = answer(&ctx, &model, &question, n_prefix);
        assert_eq!(
            restored, reference,
            "restore {round}: the answer differs from a fresh prefill of the prefix"
        );
    }

    // The context holds one sequence, so 0 is the only valid id. Without the
    // shim's check, -1 would address the whole cache and restoring into -2
    // or 256 would abort the process.
    for seq_id in [-2, -1, 1, 64, 256] {
        assert!(
            ctx.state_seq_save(seq_id).is_err(),
            "save of seq_id {seq_id} must fail"
        );
        assert!(
            ctx.state_seq_restore(&snapshot, seq_id).is_err(),
            "restore into seq_id {seq_id} must fail"
        );
    }

    // Snapshots only restore into the context that saved them.
    let other_ctx = LlamaContext::new(&model, 2048, 0, 0, false).expect("second context");
    assert!(matches!(
        other_ctx.state_seq_restore(&snapshot, 0),
        Err(LlamaError::InvalidInput(_))
    ));

    // The snapshot still restores after the rejected calls.
    ctx.state_seq_restore(&snapshot, 0)
        .expect("restore after the rejected calls");
    assert_eq!(answer(&ctx, &model, &question, n_prefix), reference);

    // An empty snapshot empties the sequence: a fresh prefill from position
    // 0 then gives the reference again. Left-over cells would make that
    // decode fail (plain KV) or attend to stale tokens (hybrid).
    ctx.kv_cache_clear();
    let empty = ctx.state_seq_save(0).expect("save an empty sequence");
    prefill(&ctx, &model, &prefix, 0);
    answer(&ctx, &model, &other, n_prefix);
    ctx.state_seq_restore(&empty, 0)
        .expect("restore the empty snapshot");
    prefill(&ctx, &model, &prefix, 0);
    assert_eq!(
        answer(&ctx, &model, &question, n_prefix),
        reference,
        "restoring an empty snapshot left state behind"
    );
}

#[test]
#[ignore = "requires lfm2.5-350m GGUF cached locally"]
fn lfm2_350m_snapshot_restores_the_prefix() {
    let Some(gguf) = locate_gguf(
        "XYBRID_LFM2_GGUF",
        &[".xybrid/cache/extracted/lfm2.5-350m/LFM2.5-350M-Q4_K_M.gguf"],
    ) else {
        eprintln!("SKIP: lfm2.5-350m GGUF not found at $XYBRID_LFM2_GGUF or ~/.xybrid/cache");
        return;
    };
    snapshot_restores_the_prefix(gguf, true);
}

#[test]
#[ignore = "requires qwen2.5-0.5b-instruct GGUF cached locally"]
fn qwen_05b_snapshot_restores_the_prefix() {
    let Some(gguf) = locate_gguf(
        "XYBRID_QWEN_GGUF",
        &[
            ".xybrid/cache/extracted/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            ".xybrid/cache/models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ],
    ) else {
        eprintln!(
            "SKIP: qwen2.5-0.5b-instruct GGUF not found at $XYBRID_QWEN_GGUF or ~/.xybrid/cache"
        );
        return;
    };
    snapshot_restores_the_prefix(gguf, false);
}
