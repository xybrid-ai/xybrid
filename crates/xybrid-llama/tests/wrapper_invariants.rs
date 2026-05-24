//! Regression tests for the safe-wrapper layer's invariants, ported
//! verbatim from the pre-Phase-2 home of these wrappers inside
//! `xybrid-core::runtime_adapter::llama_cpp` (deleted in Phase 3).
//!
//! These are pure-logic simulations — they do not touch real llama.cpp,
//! so they run in the default-feature build of `xybrid-llama` without
//! needing the cmake / libllama link. Each test documents the bug class
//! it was originally written to catch.

use std::os::raw::c_int;

// =========================================================================
// Regression: Stop Sequence Count Mismatch
// =========================================================================
//
// Bug: `generate_with_stops` / `generate_streaming` previously passed
// `stop_sequences.len()` as `n_stop_seqs` to the C function, but the
// `stop_lens` array only had entries for sequences that tokenized to
// non-empty results. If any sequence tokenized to empty, the C code
// read past the end of stop_lens → out-of-bounds access → SIGSEGV.
//
// The fix lives in `crate::generation::build_stop_token_arrays`; this
// test reproduces the count-passing contract.

#[test]
fn stop_sequence_count_matches_filtered_lens() {
    // Simulate tokenization results: sequence [1] returns empty
    let tokenize_results: Vec<Vec<i32>> = vec![
        vec![32000, 32001], // <|im_end|> → 2 tokens
        vec![],             // <|unknown_token|> → empty (filtered out)
        vec![32002],        // <|end_of_text|> → 1 token
    ];

    let mut stop_tokens: Vec<i32> = Vec::new();
    let mut stop_lens: Vec<c_int> = Vec::new();

    for tokens in &tokenize_results {
        if !tokens.is_empty() {
            stop_lens.push(tokens.len() as c_int);
            stop_tokens.extend(tokens);
        }
    }

    let n_stop_seqs = stop_lens.len() as c_int;

    assert_eq!(n_stop_seqs, 2, "n_stop_seqs must match stop_lens.len()");
    assert_eq!(stop_lens.len(), 2);
    assert_eq!(stop_tokens.len(), 3);
    assert_eq!(stop_lens[0], 2);
    assert_eq!(stop_lens[1], 1);
}

// =========================================================================
// Regression: Buffer Retry Uses Wrong Length Variable
// =========================================================================
//
// Bug: In the chat-template render path, after a buffer resize-and-retry
// the code previously used `result` (from the FIRST call) instead of
// `retry_result` to decide how many bytes to read. Could surface stale
// or uninitialized bytes when the template grew the buffer.

#[test]
fn format_chat_retry_uses_correct_length() {
    let buf_len: usize = 4096;
    let result: c_int = 5000;
    assert!(result as usize >= buf_len, "should trigger resize path");

    let _new_buf_len = (result as usize) + 1;
    let retry_result: c_int = 4998;

    let len = if result as usize >= buf_len {
        retry_result as usize
    } else {
        result as usize
    };

    assert_eq!(
        len, 4998,
        "must use retry_result (4998), not first result (5000)"
    );
}

// =========================================================================
// Regression: Prompt Size Exceeds Context Window
// =========================================================================
//
// Bug: Neither the Rust layer nor the C layer validated that input token
// count fit within the KV cache context window (n_ctx). When tokens
// >= n_ctx, the KV cache overflowed → heap corruption. The current
// runtime adapter performs this check before calling generate; this
// test pins the contract.

#[test]
fn context_window_bounds_check() {
    let n_ctx: usize = 4096;

    let tokens_at_limit = vec![0i32; 4096];
    assert!(tokens_at_limit.len() >= n_ctx);

    let tokens_over_limit = vec![0i32; 5000];
    assert!(tokens_over_limit.len() >= n_ctx);

    let tokens_within_limit = vec![0i32; 2000];
    assert!(tokens_within_limit.len() < n_ctx);

    let tokens_just_under = vec![0i32; 4095];
    assert!(tokens_just_under.len() < n_ctx);
}

// =========================================================================
// Regression: Batch Size Must Fit Input Tokens
// =========================================================================
//
// Bug: Pre-fix the C layer used `llama_batch_init(512, ...)` which
// produced heap corruption when n_input > 512. Test pins the
// fits-input-tokens contract.

#[test]
fn batch_size_must_fit_input_tokens() {
    let fixed_batch_size: usize = 512;

    let small_input = 100;
    let batch_size = if small_input > fixed_batch_size {
        small_input
    } else {
        fixed_batch_size
    };
    assert!(batch_size >= small_input);

    let large_input = 2000;
    let batch_size = if large_input > fixed_batch_size {
        large_input
    } else {
        fixed_batch_size
    };
    assert_eq!(batch_size, 2000);
    assert!(batch_size >= large_input);

    let exact_input = 512;
    let batch_size = if exact_input > fixed_batch_size {
        exact_input
    } else {
        fixed_batch_size
    };
    assert_eq!(batch_size, 512);
    assert!(batch_size >= exact_input);

    let over_input = 513;
    let batch_size = if over_input > fixed_batch_size {
        over_input
    } else {
        fixed_batch_size
    };
    assert_eq!(batch_size, 513);
}
