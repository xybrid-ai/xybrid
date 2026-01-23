/**
 * llama_wrapper.cpp - Minimal C++ wrapper for llama.cpp FFI
 *
 * This provides the `_c` suffixed functions that our Rust FFI bindings expect.
 * Uses the modern llama.cpp API (llama_model_* functions).
 */

#include "llama.h"
#include <stdlib.h>
#include <string.h>

extern "C" {

// =============================================================================
// Backend Management
// =============================================================================

void llama_backend_init_c(void) {
    llama_backend_init();
}

void llama_backend_free_c(void) {
    llama_backend_free();
}

// =============================================================================
// Model Loading (using new API)
// =============================================================================

llama_model* llama_load_model_from_file_c(
    const char* path_model,
    int n_gpu_layers
) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;

    return llama_model_load_from_file(path_model, params);
}

void llama_free_model_c(llama_model* model) {
    if (model) {
        llama_model_free(model);
    }
}

// =============================================================================
// Context Management (using new API)
// =============================================================================

llama_context* llama_new_context_with_model_c(
    llama_model* model,
    int n_ctx
) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(n_ctx);
    params.n_batch = 512;  // Default batch size
    params.n_threads = 4;  // Reasonable default for mobile
    params.n_threads_batch = 4;

    return llama_init_from_model(model, params);
}

void llama_free_c(llama_context* ctx) {
    if (ctx) {
        llama_free(ctx);
    }
}

// =============================================================================
// Tokenization (using new vocab API)
// =============================================================================

int llama_tokenize_c(
    const llama_model* model,
    const char* text,
    int text_len,
    int32_t* tokens,
    int n_tokens_max,
    bool add_special,
    bool parse_special
) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llama_tokenize(
        vocab,
        text,
        text_len,
        tokens,
        n_tokens_max,
        add_special,
        parse_special
    );
}

int llama_token_to_piece_c(
    const llama_model* model,
    int32_t token,
    char* buf,
    int length,
    int lstrip,
    bool special
) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llama_token_to_piece(vocab, token, buf, length, lstrip, special);
}

// =============================================================================
// Special Tokens (using new vocab API)
// =============================================================================

int32_t llama_token_bos_c(const llama_model* model) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llama_vocab_bos(vocab);
}

int32_t llama_token_eos_c(const llama_model* model) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llama_vocab_eos(vocab);
}

int32_t llama_token_nl_c(const llama_model* model) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llama_vocab_nl(vocab);
}

int llama_n_vocab_c(const llama_model* model) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llama_vocab_n_tokens(vocab);
}

int llama_n_ctx_c(const llama_context* ctx) {
    return static_cast<int>(llama_n_ctx(ctx));
}

// =============================================================================
// Generation (low-level)
// =============================================================================

int llama_decode_c(llama_context* ctx, const llama_batch* batch) {
    return llama_decode(ctx, *batch);
}

float* llama_get_logits_c(llama_context* ctx) {
    return llama_get_logits(ctx);
}

// =============================================================================
// Chat Template
// =============================================================================

int llama_chat_apply_template_c(
    const char* tmpl,
    const llama_chat_message* chat,
    size_t n_msg,
    bool add_ass,
    char* buf,
    int length
) {
    return llama_chat_apply_template(
        tmpl,
        chat,
        n_msg,
        add_ass,
        buf,
        length
    );
}

// =============================================================================
// Batch Management
// =============================================================================

llama_batch llama_batch_init_c(int n_tokens, int embd, int n_seq_max) {
    return llama_batch_init(n_tokens, embd, n_seq_max);
}

void llama_batch_free_c(llama_batch batch) {
    llama_batch_free(batch);
}

} // extern "C"
