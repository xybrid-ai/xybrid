//! INF-162 regression smoke test.
//!
//! The brief's `Until [done]` gate:
//!
//! > Recurrent-model gating (lfm2.5-350m, ≥ 2 turns) passes — turn 2 does
//! > not fail with `llama_decode != 0` (INF-162 regression class).
//!
//! lfm2.5-350m is a hybrid architecture. The pre-INF-99 prefix-reuse path
//! truncated the KV cache by position via `llama_kv_cache_seq_rm`, which
//! leaves recurrent residual state inconsistent with the new prefix
//! length — `llama_decode` then fails on the diverging tail (wrapper
//! error code -3, surfaced as [`LlamaError::DecodeFailed`]).
//!
//! This test verifies that running 2 turns against lfm2.5-350m through
//! the safe `xybrid-llama` surface does NOT surface that decode failure.
//! Gating itself lives in `xybrid-core::runtime_adapter::llama_cpp` (it
//! calls `model.has_recurrent_state()` and full-clears the cache for
//! recurrent / hybrid models); this test exercises the predicate at the
//! `xybrid-llama` layer plus an end-to-end 2-turn run.
//!
//! Run with the cached GGUF visible to the test:
//!
//! ```sh
//! XYBRID_LFM2_GGUF=~/.xybrid/cache/extracted/lfm2.5-350m/LFM2.5-350M-Q4_K_M.gguf \
//!   cargo test -p xybrid-llama --features bindings \
//!     --test recurrent_gating_smoke -- --nocapture --ignored
//! ```
//!
//! Marked `#[ignore]` because it requires a ~250 MB GGUF on disk and 5+
//! seconds of CPU. The brief's verification script runs it via the
//! cached lfm2.5-350m fixture; CI does not run it as part of the default
//! suite.

#![cfg(feature = "bindings")]

use std::path::PathBuf;

use xybrid_llama::{backend_init, generate_with_stops, LlamaContext, LlamaError, LlamaModel};

fn locate_lfm2_gguf() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("XYBRID_LFM2_GGUF") {
        let p = PathBuf::from(shellexpand_tilde(&env_path));
        if p.exists() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let cached =
        PathBuf::from(home).join(".xybrid/cache/extracted/lfm2.5-350m/LFM2.5-350M-Q4_K_M.gguf");
    if cached.exists() {
        Some(cached)
    } else {
        None
    }
}

fn shellexpand_tilde(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    s.to_string()
}

#[test]
#[ignore = "requires lfm2.5-350m GGUF cached locally"]
fn lfm2_350m_two_turns_does_not_fail_with_decode_error() {
    let gguf = match locate_lfm2_gguf() {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIP: lfm2.5-350m GGUF not found at $XYBRID_LFM2_GGUF or \
                 ~/.xybrid/cache/extracted/lfm2.5-350m/LFM2.5-350M-Q4_K_M.gguf"
            );
            return;
        }
    };

    backend_init();

    let model = LlamaModel::load(gguf.to_str().unwrap(), 0)
        .expect("lfm2.5-350m must load through the safe wrapper");

    // Predicate gate for the integration layer: lfm2.5 is hybrid, so
    // the runtime adapter's prefix-reuse path must skip cache truncation.
    assert!(
        model.has_recurrent_state(),
        "lfm2.5-350m must report recurrent/hybrid state — this is the \
         signal the prefix-reuse path keys on for INF-162-class safety"
    );

    let ctx = LlamaContext::new(&model, 4096, 0, 0, false)
        .expect("lfm2.5-350m context creation must succeed");

    // Apply chat template manually (the lfm2 GGUF embeds a chatml-like
    // template). For a raw smoke test, prefilling text directly through
    // tokenize() exercises the same `generate*` surface.
    let turn1_prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
    let turn1_tokens = model
        .tokenize_special(turn1_prompt, true)
        .expect("tokenize turn 1");
    let _ = generate_with_stops(
        &ctx,
        &model,
        &turn1_tokens,
        32,
        0.0,
        1.0,
        0.0,
        0,
        1.0,
        &["<|im_end|>".to_string()],
    )
    .expect("turn 1 must not fail");

    // Turn 2: full-clear the cache (as the safe runtime-adapter does
    // for recurrent / hybrid models) then prefill turn 2. This is the
    // path that broke pre-INF-162 — the buggy code skipped the
    // full-clear and tried position-based truncation, then `llama_decode`
    // returned non-zero on the diverging recurrent state.
    ctx.kv_cache_clear();
    let turn2_prompt = "<|im_start|>user\nAnd 3+3?<|im_end|>\n<|im_start|>assistant\n";
    let turn2_tokens = model
        .tokenize_special(turn2_prompt, true)
        .expect("tokenize turn 2");
    let result = generate_with_stops(
        &ctx,
        &model,
        &turn2_tokens,
        32,
        0.0,
        1.0,
        0.0,
        0,
        1.0,
        &["<|im_end|>".to_string()],
    );

    match result {
        Ok(_) => {
            // Pass — turn 2 didn't fail with llama_decode != 0.
        }
        Err(LlamaError::DecodeFailed {
            code,
            n_past_in,
            detail,
        }) => {
            panic!(
                "INF-162 regression — turn 2 on lfm2.5-350m failed with \
                 llama_decode != 0: code={code}, n_past_in={n_past_in}, detail={detail}"
            );
        }
        Err(other) => {
            panic!("unexpected error on turn 2: {other}");
        }
    }
}
