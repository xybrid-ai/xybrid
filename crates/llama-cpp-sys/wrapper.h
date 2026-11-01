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
 *
 * A C++ exception must not cross into the caller: the Rust side cannot catch
 * it and aborts. The shims documented as "guarded" catch every exception
 * llama.cpp raises (in practice std::bad_alloc) and report it as a status.
 * ---------------------------------------------------------------------- */

/* Statuses of the guarded shims that return a result >= 0 (a length, a
   count, attribute bits). A failure is never reported as an absence. */
#define LLAMA_XYBRID_ABSENT (-1) /* no such key, template or token */
#define LLAMA_XYBRID_FAILED (-2) /* llama.cpp raised an exception, or an argument was NULL */
/* llama_decode_c's status when llama.cpp raised an exception; none of
   llama_decode's own statuses. */
#define LLAMA_XYBRID_DECODE_EXCEPTION (-1000)

/* Fault injection for tests: the next call of the guarded shim named
   `shim_name` (e.g. "llama_decode_c") throws std::bad_alloc inside its
   exception boundary, as an allocation failure inside llama.cpp would. NULL
   disarms. Nothing calls this outside tests. */
void llama_xybrid_fail_next_c(const char* shim_name);

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
/* Guarded; an exception leaves the memory as llama.cpp left it. */
void  llama_kv_cache_clear_c(XYBRID_LLAMA_CONTEXT* ctx);
/* Guarded: 0, -1 when the context has no memory, LLAMA_XYBRID_FAILED. */
int   llama_kv_cache_seq_rm_c(XYBRID_LLAMA_CONTEXT* ctx, int seq_id, int p_keep);

/* Per-sequence state snapshots (KV cache plus recurrent state). Each returns
   a byte count; 0 means failure, including a seq_id outside
   [0, n_seq_max). A failed set leaves the sequence empty. Only pass set
   bytes from get_data on the same model: llama.cpp does not validate them. */
size_t llama_state_seq_get_size_c(XYBRID_LLAMA_CONTEXT* ctx, int seq_id);
size_t llama_state_seq_get_data_c(XYBRID_LLAMA_CONTEXT* ctx, uint8_t* dst, size_t size, int seq_id);
size_t llama_state_seq_set_data_c(XYBRID_LLAMA_CONTEXT* ctx, const uint8_t* src, size_t size, int seq_id);

/* Tokenization. Guarded: an exception returns INT32_MIN, which llama.cpp
   itself returns when the result does not fit an int32_t. */
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
bool llama_vocab_get_add_bos_c(const XYBRID_LLAMA_MODEL* model);

/* Model metadata */
const char* llama_model_chat_template_c(const XYBRID_LLAMA_MODEL* model);
int  llama_n_vocab_c(const XYBRID_LLAMA_MODEL* model);
int  llama_n_ctx_c(const XYBRID_LLAMA_CONTEXT* ctx);
bool llama_model_is_recurrent_c(const XYBRID_LLAMA_MODEL* model);
bool llama_model_has_recurrent_state_c(const XYBRID_LLAMA_MODEL* model);
/* Guarded: 1 for an encoder-decoder model, 0 otherwise, LLAMA_XYBRID_FAILED. */
int32_t llama_model_has_encoder_c(const XYBRID_LLAMA_MODEL* model);

/* The GGUF metadata llama.cpp loaded with the model: every scalar key, its
   value written as a string (floats through std::to_string, so with six
   decimals). Array-valued keys are not visible. Guarded. The accessors
   return the length of the whole value or key, excluding the terminating
   NUL, and write at most buf_size - 1 bytes of it plus a NUL;
   LLAMA_XYBRID_ABSENT when the key or index does not exist. buf may be NULL
   when buf_size is 0, to probe the length. */
int32_t llama_model_meta_count_c(const XYBRID_LLAMA_MODEL* model);
int32_t llama_model_meta_key_by_index_c(
    const XYBRID_LLAMA_MODEL* model, int32_t i, char* buf, size_t buf_size);
int32_t llama_model_meta_val_str_by_index_c(
    const XYBRID_LLAMA_MODEL* model, int32_t i, char* buf, size_t buf_size);
int32_t llama_model_meta_val_str_c(
    const XYBRID_LLAMA_MODEL* model, const char* key, char* buf, size_t buf_size);

/* The chat template stored as `tokenizer.chat_template.<name>`. Guarded:
   returns its length and points *out at it (owned by the model), or
   LLAMA_XYBRID_ABSENT when the model has none by that name. */
int32_t llama_model_chat_template_named_c(
    const XYBRID_LLAMA_MODEL* model, const char* name, const char** out);

/* A vocabulary entry's attributes (a LLAMA_TOKEN_ATTR_* bit set) and its
   stored text (length returned, *out pointed at it, owned by the model).
   Guarded; LLAMA_XYBRID_ABSENT for an id outside the vocabulary. */
int32_t llama_vocab_get_attr_c(const XYBRID_LLAMA_MODEL* model, int32_t token);
int32_t llama_vocab_get_text_c(const XYBRID_LLAMA_MODEL* model, int32_t token, const char** out);

/* A context for one-shot readouts: one sequence, the given context, batch and
   microbatch sizes, at most n_outputs_max outputs (0 = n_batch, llama.cpp's
   default) and one per sequence, no embeddings, and the model's own attention
   type. Guarded; returns NULL on failure. Whether it has memory is the
   model's choice: check it. */
XYBRID_LLAMA_CONTEXT* llama_new_readout_context_c(
    XYBRID_LLAMA_MODEL* model,
    uint32_t n_ctx,
    uint32_t n_batch,
    uint32_t n_ubatch,
    uint32_t n_outputs_max,
    int n_threads,
    bool flash_attn);

/* The sizes and attention a context actually got, which can differ from the
   ones requested. has_memory is false when llama.cpp runs the model without
   a KV cache or recurrent state (it then decodes with the encoder). */
struct llama_xybrid_context_facts {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    bool has_memory;
    bool causal_attn;
};
/* Guarded: 0 and *out filled, or LLAMA_XYBRID_FAILED. */
int32_t llama_context_facts_c(
    const XYBRID_LLAMA_CONTEXT* ctx, struct llama_xybrid_context_facts* out);

/* The logits row of batch entry i from the last decode, owned by the context
   until its next decode. Guarded; NULL on failure. Only pass an entry the
   last successful decode flagged for output: for any other, llama.cpp
   returns NULL when built with NDEBUG and aborts the process when built
   without it. */
float* llama_get_logits_ith_c(XYBRID_LLAMA_CONTEXT* ctx, int32_t i);

/* Low-level generation primitives (carried in wrapper.cpp's exports
   even though xybrid-llama drives generation via the higher-level
   `llama_generate*_c` entry points below). Guarded: llama_decode's status,
   -1 for a NULL argument, LLAMA_XYBRID_DECODE_EXCEPTION for an exception
   (the context's state is then whatever llama.cpp left). */
int   llama_decode_c(XYBRID_LLAMA_CONTEXT* ctx, const llama_batch* batch);
/* Guarded; NULL on failure. */
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
    const XYBRID_LLAMA_MODEL* model,
    float temperature,
    float top_p,
    float min_p,
    int top_k,
    float repeat_penalty,
    int penalty_last_n,
    uint32_t seed,
    const char* grammar,
    const char* grammar_root);
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
    const char* grammar,
    const char* grammar_root,
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
    const char* grammar,
    const char* grammar_root,
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
    const char* grammar,
    const char* grammar_root,
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
    bool flash_attn,
    int image_max_tokens);
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
