/**
 * wrapper.h — umbrella header for bindgen.
 *
 * Pulls in the upstream llama.cpp public surface (`llama.h`) so bindgen
 * can see the types our `_c`-suffixed shim functions reference, then
 * declares the shim's C-callable surface so bindgen emits bindings for
 * those too.
 *
 * Phase 5 of the `llamacpp-crate-split` epic: replaces the prior
 * hand-written 26-symbol `extern "C" {}` block in
 * `src/lib.rs::bindings` with bindgen output. The allowlist in
 * `build.rs` (`llama_.*`) keeps the generated surface focused on the
 * llama.cpp world; `ggml_*` is intentionally NOT allowlisted because
 * no `xybrid-llama` / `xybrid-core` consumer references a `ggml_*`
 * symbol directly (the upstream call into ggml only happens inside
 * `wrapper.cpp`, which builds against the C++ headers separately).
 */

#ifndef XYBRID_LLAMA_CPP_SYS_WRAPPER_H
#define XYBRID_LLAMA_CPP_SYS_WRAPPER_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * `_c`-suffixed wrapper surface from wrapper.cpp.
 *
 * These are first-party C-callable functions that adapt the C++ llama.cpp
 * API (which moves/breaks across versions) to a stable C ABI consumed by
 * xybrid-llama. Bindgen processes these declarations alongside the
 * llama.cpp public symbols pulled in via <llama.h>.
 * ---------------------------------------------------------------------- */

/* Backend lifecycle */
void llama_backend_init_c(void);
void llama_backend_free_c(void);

/* Log verbosity control (xybrid-side filter sitting in front of
   llama.cpp's own ggml_log_callback). */
void llama_log_set_verbosity_c(int level);
int  llama_log_get_verbosity_c(void);

/* Model lifecycle */
void* llama_load_model_from_file_c(const char* path_model, int n_gpu_layers);
void  llama_free_model_c(void* model);

/* Context lifecycle */
void* llama_new_context_with_model_c(
    void* model,
    int n_ctx,
    int n_threads,
    int n_batch,
    bool flash_attn);
void  llama_free_c(void* ctx);
void  llama_kv_cache_clear_c(void* ctx);
int   llama_kv_cache_seq_rm_c(void* ctx, int seq_id, int p_keep);

/* Tokenization */
int  llama_tokenize_c(
    const void* model,
    const char* text,
    int text_len,
    int* tokens,
    int n_tokens_max,
    bool add_special,
    bool parse_special);

int  llama_token_to_piece_c(
    const void* model,
    int token,
    char* buf,
    int length,
    int lstrip,
    bool special);

/* Special tokens / vocabulary */
int  llama_token_bos_c(const void* model);
int  llama_token_eos_c(const void* model);
bool llama_vocab_is_eog_c(const void* model, int token);

/* Model metadata */
const char* llama_model_chat_template_c(const void* model);
int  llama_n_vocab_c(const void* model);
int  llama_n_ctx_c(const void* ctx);
bool llama_model_is_recurrent_c(const void* model);
bool llama_model_has_recurrent_state_c(const void* model);

/* Low-level generation primitives (carried in wrapper.cpp's exports
   even though xybrid-llama drives generation via the higher-level
   `llama_generate*_c` entry points below). */
int   llama_decode_c(void* ctx, const void* batch);
float* llama_get_logits_c(void* ctx);

/* Chat template formatting */
int  llama_chat_apply_template_c(
    const char* tmpl,
    const void* chat,
    size_t n_msg,
    bool add_ass,
    char* buf,
    int length);

int  llama_format_chat_with_model_c(
    const void* model,
    const char* const* roles,
    const char* const* contents,
    size_t n_msg,
    char* buf,
    int buf_size);

/* Streaming-callback token type (matches Rust's `TokenCallback`). */
typedef int (*llama_token_callback_c)(int token_id, const char* token_text, void* user_data);

/* Autoregressive generation with stop-sequence support */
int  llama_generate_c(
    void* ctx,
    const void* model,
    const int* input_tokens,
    int n_input,
    int* output_tokens,
    int max_tokens,
    float temperature,
    float top_p,
    float min_p,
    int top_k,
    float repeat_penalty,
    unsigned int seed,
    const int* stop_seqs,
    const int* stop_lens,
    int n_stop_seqs);

/* Streaming variant with per-token callback + KV-cache prefix-reuse */
int  llama_generate_streaming_c(
    void* ctx,
    const void* model,
    const int* input_tokens,
    int n_input,
    int* output_tokens,
    int max_tokens,
    float temperature,
    float top_p,
    float min_p,
    int top_k,
    float repeat_penalty,
    unsigned int seed,
    const int* stop_seqs,
    const int* stop_lens,
    int n_stop_seqs,
    llama_token_callback_c callback,
    void* user_data,
    int n_past_in);

#ifdef __cplusplus
}
#endif

#endif /* XYBRID_LLAMA_CPP_SYS_WRAPPER_H */
