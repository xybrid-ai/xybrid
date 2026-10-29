//! A streaming callback that stops generation must surface as
//! `LlamaError::StreamingCallbackAborted` on every token, the first few
//! included.
//!
//! llama.cpp reports a callback stop as `-n_generated`, so a stop on tokens
//! 1–4 returns the same value as `generate_streaming`'s hard error codes
//! (-1..=-4), and a stop on token 5 the same value as the current-logits
//! variant's -5. This drives the real native loop and stops it on each of
//! those tokens and on two past them, so the return-value interpretation is
//! checked against llama.cpp itself rather than a simulated return code.
//!
//! Run via:
//!
//! ```sh
//! XYBRID_QWEN_GGUF=~/.xybrid/cache/extracted/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//!   cargo test -p xybrid-llama --features bindings \
//!     --test callback_stop_boundaries -- --nocapture --ignored
//! ```
//!
//! `#[ignore]` because it requires a ~490 MB GGUF on disk.

#![cfg(feature = "bindings")]

use std::error::Error;
use std::fmt;
use std::path::PathBuf;

use xybrid_llama::{backend_init, generate_streaming, LlamaContext, LlamaError, LlamaModel};

/// `$XYBRID_QWEN_GGUF` first, then the extracted registry cache, then the
/// older models cache.
fn locate_qwen_gguf() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("XYBRID_QWEN_GGUF") {
        let p = PathBuf::from(shellexpand_tilde(&env_path));
        if p.exists() {
            return Some(p);
        }
    }
    let home = PathBuf::from(std::env::var("HOME").ok()?);
    [
        ".xybrid/cache/extracted/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ".xybrid/cache/models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ]
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

/// The error the test callback returns. It carries the token it stopped on,
/// so the assertion can tell the original error from a rebuilt one.
#[derive(Debug)]
struct StopAt(usize);

impl fmt::Display for StopAt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "callback stopped generation on token {}", self.0)
    }
}

impl Error for StopAt {}

#[test]
#[ignore = "requires qwen2.5-0.5b-instruct GGUF cached locally"]
fn callback_stop_surfaces_as_callback_error_on_every_token() {
    let gguf = match locate_qwen_gguf() {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIP: qwen2.5-0.5b-instruct GGUF not found at $XYBRID_QWEN_GGUF or \
                 ~/.xybrid/cache/{{extracted,models}}"
            );
            return;
        }
    };

    backend_init();
    let model = LlamaModel::load(gguf.to_str().unwrap(), 0)
        .expect("qwen2.5-0.5b must load through the safe wrapper");
    let ctx = LlamaContext::new(&model, 2048, 0, 0, false)
        .expect("qwen2.5-0.5b context creation must succeed");

    // Greedy decoding of a long counting answer: no end-of-generation token
    // can end the loop before the callback stops it.
    let prompt = "<|im_start|>user\nCount from one to forty in words, separated by commas.\
                  <|im_end|>\n<|im_start|>assistant\n";
    let tokens = model
        .tokenize_special(prompt, true)
        .expect("tokenize prompt");

    for stop_at in [1, 2, 3, 4, 5, 6, 8] {
        ctx.kv_cache_clear();
        let mut calls = 0;
        let result = generate_streaming(
            &ctx,
            &model,
            &tokens,
            32,
            0.0,
            1.0,
            0.0,
            0,
            1.0,
            &[],
            None,
            |_token_id, _text| {
                calls += 1;
                if calls == stop_at {
                    Err(StopAt(stop_at).into())
                } else {
                    Ok(())
                }
            },
            0,
        );

        match result {
            Err(LlamaError::StreamingCallbackAborted(err)) => {
                let stopped = err.downcast_ref::<StopAt>().unwrap_or_else(|| {
                    panic!("token {stop_at}: the callback's own error must come back, got: {err}")
                });
                assert_eq!(stopped.0, stop_at);
            }
            Err(other) => {
                panic!("token {stop_at}: expected StreamingCallbackAborted, got {other:?}")
            }
            Ok((output, _)) => panic!(
                "token {stop_at}: generation finished after {} tokens without the callback \
                 stopping it",
                output.len()
            ),
        }
        assert_eq!(
            calls, stop_at,
            "the loop must stop on the token whose callback failed"
        );
    }
}
