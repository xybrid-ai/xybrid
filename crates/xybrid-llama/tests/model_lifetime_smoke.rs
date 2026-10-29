//! A llama context keeps the model it was created from loaded.
//!
//! The native context keeps reading its model, so `LlamaContext` shares the
//! model's native handle: dropping the `LlamaModel` first must leave the
//! context usable. Generation must also refuse a model the context was not
//! created from, even a second load of the same file.
//!
//! Run via:
//!
//! ```sh
//! XYBRID_LFM2_GGUF=~/.xybrid/cache/extracted/lfm2.5-350m/LFM2.5-350M-Q4_K_M.gguf \
//!   cargo test -p xybrid-llama --features bindings \
//!     --test model_lifetime_smoke -- --nocapture --ignored
//! ```
//!
//! `#[ignore]` because it requires a GGUF on disk. A missing model skips its
//! test.

#![cfg(feature = "bindings")]

use std::path::PathBuf;

use xybrid_llama::{backend_init, generate_with_stops, LlamaContext, LlamaError, LlamaModel};

const PROMPT: &str = "<|im_start|>user\nName one colour of a sunset.<|im_end|>\n\
                      <|im_start|>assistant\n";

/// `XYBRID_LFM2_GGUF` first, then the SDK cache's LFM2.5-350M.
fn locate_lfm2() -> Option<String> {
    let path = match std::env::var("XYBRID_LFM2_GGUF") {
        Ok(env_path) => PathBuf::from(shellexpand_tilde(&env_path)),
        Err(_) => PathBuf::from(std::env::var("HOME").ok()?)
            .join(".xybrid/cache/extracted/lfm2.5-350m/LFM2.5-350M-Q4_K_M.gguf"),
    };
    if path.exists() {
        path.to_str().map(str::to_string)
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

fn greedy(ctx: &LlamaContext, model: &LlamaModel, prompt: &[i32]) -> Result<Vec<i32>, LlamaError> {
    generate_with_stops(ctx, model, prompt, 4, 0.0, 1.0, 0.0, 1, 1.0, &[], None)
}

#[test]
#[ignore = "requires a GGUF on disk"]
fn a_context_stays_usable_after_its_model_is_dropped() {
    let Some(path) = locate_lfm2() else {
        eprintln!("skipping: no LFM2 GGUF (set XYBRID_LFM2_GGUF)");
        return;
    };
    backend_init();
    let model = LlamaModel::load(&path, 0).expect("load the GGUF");
    let ctx = LlamaContext::new(&model, 512, 2, 0, false).expect("create a context");
    let prompt = model
        .tokenize_special(PROMPT, true)
        .expect("tokenize the prompt");
    let answer = greedy(&ctx, &model, &prompt).expect("generate before the drop");
    assert!(!answer.is_empty(), "the model generated nothing");

    drop(model);

    // Saving and restoring the sequence read the model's hyperparameters
    // through the context.
    let snapshot = ctx.state_seq_save(0).expect("save the sequence");
    ctx.kv_cache_clear();
    ctx.state_seq_restore(&snapshot, 0)
        .expect("restore the sequence");
    assert!(ctx.n_ctx() >= 512, "n_ctx is {}", ctx.n_ctx());
}

#[test]
#[ignore = "requires a GGUF on disk"]
fn generation_refuses_another_load_of_the_same_file() {
    let Some(path) = locate_lfm2() else {
        eprintln!("skipping: no LFM2 GGUF (set XYBRID_LFM2_GGUF)");
        return;
    };
    backend_init();
    let model = LlamaModel::load(&path, 0).expect("load the GGUF");
    let other = LlamaModel::load(&path, 0).expect("load the GGUF again");
    let ctx = LlamaContext::new(&model, 512, 2, 0, false).expect("create a context");
    let prompt = model
        .tokenize_special(PROMPT, true)
        .expect("tokenize the prompt");

    match greedy(&ctx, &other, &prompt) {
        Err(LlamaError::InvalidInput(message)) => {
            assert!(message.contains("different model"), "{message}");
        }
        result => panic!("expected InvalidInput, got {result:?}"),
    }
    greedy(&ctx, &model, &prompt).expect("the context's own model still generates");
}
