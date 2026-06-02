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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "llama.h"
#ifdef XYBRID_LLAMA_VISION
#include "mtmd.h"
#include "mtmd-helper.h"
#endif

#ifdef __cplusplus
extern "C" {
#define XYBRID_LLAMA_MODEL llama_model
#define XYBRID_LLAMA_CONTEXT llama_context
#define XYBRID_LLAMA_SAMPLER llama_sampler
#define XYBRID_CONST_PTR
#else
#define XYBRID_LLAMA_MODEL void
#define XYBRID_LLAMA_CONTEXT void
#define XYBRID_LLAMA_SAMPLER void
#define XYBRID_CONST_PTR const
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
XYBRID_LLAMA_MODEL* llama_load_model_from_file_c(const char* path_model, int n_gpu_layers);
void  llama_free_model_c(XYBRID_LLAMA_MODEL* model);

/* Context lifecycle */
XYBRID_LLAMA_CONTEXT* llama_new_context_with_model_c(
    XYBRID_LLAMA_MODEL* model,
    int n_ctx,
    int n_threads,
    int n_batch,
    bool flash_attn);
void  llama_free_c(XYBRID_LLAMA_CONTEXT* ctx);
void  llama_kv_cache_clear_c(XYBRID_LLAMA_CONTEXT* ctx);
int   llama_kv_cache_seq_rm_c(XYBRID_LLAMA_CONTEXT* ctx, int seq_id, int p_keep);

/* Tokenization */
int  llama_tokenize_c(
    const XYBRID_LLAMA_MODEL* model,
    const char* text,
    int text_len,
    int32_t* tokens,
    int n_tokens_max,
    bool add_special,
    bool parse_special);

int  llama_token_to_piece_c(
    const XYBRID_LLAMA_MODEL* model,
    int32_t token,
    char* buf,
    int length,
    int lstrip,
    bool special);

/* Special tokens / vocabulary */
int32_t llama_token_bos_c(const XYBRID_LLAMA_MODEL* model);
int32_t llama_token_eos_c(const XYBRID_LLAMA_MODEL* model);
int32_t llama_token_nl_c(const XYBRID_LLAMA_MODEL* model);
bool llama_vocab_is_eog_c(const XYBRID_LLAMA_MODEL* model, int32_t token);

/* Model metadata */
const char* llama_model_chat_template_c(const XYBRID_LLAMA_MODEL* model);
int  llama_n_vocab_c(const XYBRID_LLAMA_MODEL* model);
int  llama_n_ctx_c(const XYBRID_LLAMA_CONTEXT* ctx);
bool llama_model_is_recurrent_c(const XYBRID_LLAMA_MODEL* model);
bool llama_model_has_recurrent_state_c(const XYBRID_LLAMA_MODEL* model);

/* Low-level generation primitives (carried in wrapper.cpp's exports
   even though xybrid-llama drives generation via the higher-level
   `llama_generate*_c` entry points below). */
int   llama_decode_c(XYBRID_LLAMA_CONTEXT* ctx, const llama_batch* batch);
float* llama_get_logits_c(XYBRID_LLAMA_CONTEXT* ctx);

/* Chat template formatting */
int  llama_chat_apply_template_c(
    const char* tmpl,
    const llama_chat_message* chat,
    size_t n_msg,
    bool add_ass,
    char* buf,
    int length);

int  llama_format_chat_with_model_c(
    const XYBRID_LLAMA_MODEL* model,
    const char* XYBRID_CONST_PTR* roles,
    const char* XYBRID_CONST_PTR* contents,
    size_t n_msg,
    char* buf,
    int buf_size);

/* Batch and sampler helpers exported by wrapper.cpp for low-level callers. */
llama_batch llama_batch_init_c(int n_tokens, int embd, int n_seq_max);
void llama_batch_free_c(llama_batch batch);

XYBRID_LLAMA_SAMPLER* llama_sampler_chain_create_c(
    float temperature,
    float top_p,
    float min_p,
    int top_k,
    float repeat_penalty,
    int penalty_last_n,
    uint32_t seed);
void llama_sampler_free_c(XYBRID_LLAMA_SAMPLER* smpl);

/* Streaming-callback token type (matches Rust's `TokenCallback`). */
typedef int (*llama_token_callback_c)(int32_t token_id, const char* token_text, void* user_data);

/* Autoregressive generation with stop-sequence support */
int  llama_generate_c(
    XYBRID_LLAMA_CONTEXT* ctx,
    const XYBRID_LLAMA_MODEL* model,
    const int32_t* input_tokens,
    int n_input,
    int32_t* output_tokens,
    int max_tokens,
    float temperature,
    float top_p,
    float min_p,
    int top_k,
    float repeat_penalty,
    uint32_t seed,
    const int32_t* stop_seqs,
    const int* stop_lens,
    int n_stop_seqs);

/* Streaming variant with per-token callback + KV-cache prefix-reuse */
int  llama_generate_streaming_c(
    XYBRID_LLAMA_CONTEXT* ctx,
    const XYBRID_LLAMA_MODEL* model,
    const int32_t* input_tokens,
    int n_input,
    int32_t* output_tokens,
    int max_tokens,
    float temperature,
    float top_p,
    float min_p,
    int top_k,
    float repeat_penalty,
    uint32_t seed,
    const int32_t* stop_seqs,
    const int* stop_lens,
    int n_stop_seqs,
    llama_token_callback_c callback,
    void* user_data,
    int n_past_in);

/* Continue generation from logits already present in the context. Used
   after mtmd helper eval prefills text/image chunks. */
int  llama_generate_from_current_logits_c(
    XYBRID_LLAMA_CONTEXT* ctx,
    const XYBRID_LLAMA_MODEL* model,
    int32_t* output_tokens,
    int max_tokens,
    float temperature,
    float top_p,
    float min_p,
    int top_k,
    float repeat_penalty,
    uint32_t seed,
    const int32_t* stop_seqs,
    const int* stop_lens,
    int n_stop_seqs,
    llama_token_callback_c callback,
    void* user_data,
    int n_past);

#ifdef XYBRID_LLAMA_VISION
/* mtmd vision-language projector lifecycle */
mtmd_context* mtmd_init_from_file_c(
    const char* mmproj_fname,
    const XYBRID_LLAMA_MODEL* text_model,
    bool use_gpu,
    bool warmup,
    int n_threads,
    bool flash_attn);
void mtmd_free_c(mtmd_context* ctx);

/* mtmd bitmap helpers */
mtmd_bitmap* mtmd_bitmap_init_from_buf_c(
    mtmd_context* ctx,
    const unsigned char* buf,
    size_t len);
/* Build an mtmd image bitmap directly from tightly-packed RGB pixels.
 * `data` must point to exactly nx * ny * 3 bytes in RGBRGB... order
 * (no row-stride padding, no alpha). Wraps the packed-RGB ctor so raw
 * camera frames skip the per-frame JPEG encode/decode round-trip. */
mtmd_bitmap* mtmd_bitmap_init_rgb_c(
    uint32_t nx,
    uint32_t ny,
    const unsigned char* data);
void mtmd_bitmap_free_c(mtmd_bitmap* bitmap);
uint32_t mtmd_bitmap_get_nx_c(const mtmd_bitmap* bitmap);
uint32_t mtmd_bitmap_get_ny_c(const mtmd_bitmap* bitmap);
size_t mtmd_bitmap_get_n_bytes_c(const mtmd_bitmap* bitmap);
const char* mtmd_bitmap_get_id_c(const mtmd_bitmap* bitmap);
void mtmd_bitmap_set_id_c(mtmd_bitmap* bitmap, const char* id);

/* mtmd chunk helpers */
mtmd_input_chunks* mtmd_input_chunks_init_c(void);
size_t mtmd_input_chunks_size_c(const mtmd_input_chunks* chunks);
const mtmd_input_chunk* mtmd_input_chunks_get_c(
    const mtmd_input_chunks* chunks,
    size_t idx);
void mtmd_input_chunks_free_c(mtmd_input_chunks* chunks);
int mtmd_input_chunk_get_type_c(const mtmd_input_chunk* chunk);
const int32_t* mtmd_input_chunk_get_tokens_text_c(
    const mtmd_input_chunk* chunk,
    size_t* n_tokens_output);
const mtmd_image_tokens* mtmd_input_chunk_get_tokens_image_c(
    const mtmd_input_chunk* chunk);
size_t mtmd_input_chunk_get_n_tokens_c(const mtmd_input_chunk* chunk);
int32_t mtmd_input_chunk_get_n_pos_c(const mtmd_input_chunk* chunk);
size_t mtmd_image_tokens_get_n_tokens_c(const mtmd_image_tokens* image_tokens);
int32_t mtmd_image_tokens_get_n_pos_c(const mtmd_image_tokens* image_tokens);
struct mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos_c(
    const mtmd_image_tokens* image_tokens,
    int32_t pos_0,
    size_t i);
size_t mtmd_helper_get_n_tokens_c(const mtmd_input_chunks* chunks);
int32_t mtmd_helper_get_n_pos_c(const mtmd_input_chunks* chunks);
int32_t mtmd_tokenize_c(
    mtmd_context* ctx,
    mtmd_input_chunks* output,
    const char* text,
    bool add_special,
    bool parse_special,
    const mtmd_bitmap** bitmaps,
    size_t n_bitmaps);
int32_t mtmd_helper_eval_chunks_c(
    mtmd_context* ctx,
    XYBRID_LLAMA_CONTEXT* lctx,
    const mtmd_input_chunks* chunks,
    int32_t n_past,
    int32_t seq_id,
    int32_t n_batch,
    bool logits_last,
    int32_t* new_n_past);
#endif

#ifdef __cplusplus
#undef XYBRID_CONST_PTR
#undef XYBRID_LLAMA_SAMPLER
#undef XYBRID_LLAMA_CONTEXT
#undef XYBRID_LLAMA_MODEL
}
#else
#undef XYBRID_CONST_PTR
#undef XYBRID_LLAMA_SAMPLER
#undef XYBRID_LLAMA_CONTEXT
#undef XYBRID_LLAMA_MODEL
#endif

#endif /* XYBRID_LLAMA_CPP_SYS_WRAPPER_H */
