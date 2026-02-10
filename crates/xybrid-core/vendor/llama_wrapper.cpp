/**
 * llama_wrapper.cpp - Minimal C++ wrapper for llama.cpp FFI
 *
 * This provides the `_c` suffixed functions that our Rust FFI bindings expect.
 * Uses the modern llama.cpp API (llama_model_* functions).
 */

#include "llama.h"
#include "ggml.h"
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>
#include <atomic>

// =============================================================================
// Log Verbosity Control
// =============================================================================

// Log levels matching Rust's Severity enum
// 0 = Silent (no logs), 1 = Error, 2 = Warn, 3 = Info, 4 = Debug
static std::atomic<int> g_log_verbosity{0};  // Default: silent (suppress all library logs)

// Custom log callback that filters based on verbosity level
static void xybrid_log_callback(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;  // unused

    int verbosity = g_log_verbosity.load(std::memory_order_relaxed);

    // Map ggml log levels to our verbosity levels
    // GGML_LOG_LEVEL_NONE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4
    int required_verbosity;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            required_verbosity = 1;  // Errors need verbosity >= 1
            break;
        case GGML_LOG_LEVEL_WARN:
            required_verbosity = 2;  // Warnings need verbosity >= 2
            break;
        case GGML_LOG_LEVEL_INFO:
            required_verbosity = 3;  // Info needs verbosity >= 3
            break;
        case GGML_LOG_LEVEL_DEBUG:
        default:
            required_verbosity = 4;  // Debug needs verbosity >= 4
            break;
    }

    // Only print if verbosity is high enough
    if (verbosity >= required_verbosity) {
        fputs(text, stderr);
    }
}

extern "C" {

// =============================================================================
// Log Control
// =============================================================================

/**
 * Set the verbosity level for llama.cpp/ggml logging.
 *
 * @param level 0 = silent, 1 = errors only, 2 = +warnings, 3 = +info, 4 = +debug
 */
void llama_log_set_verbosity_c(int level) {
    g_log_verbosity.store(level, std::memory_order_relaxed);
}

/**
 * Get the current verbosity level.
 */
int llama_log_get_verbosity_c(void) {
    return g_log_verbosity.load(std::memory_order_relaxed);
}

// =============================================================================
// Backend Management
// =============================================================================

void llama_backend_init_c(void) {
    // Install our custom log callback BEFORE backend init
    // This suppresses the verbose Metal/tensor loading logs
    // Use llama_log_set which internally sets ggml_log_set as well
    llama_log_set(xybrid_log_callback, nullptr);

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
    int n_ctx,
    int n_threads,
    int n_batch,
    bool flash_attn
) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(n_ctx);
    params.n_batch = static_cast<uint32_t>(n_batch > 0 ? n_batch : 512);
    params.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // Use provided thread count, or fall back to hardware concurrency
    int actual_threads = n_threads > 0 ? n_threads : std::thread::hardware_concurrency();
    if (actual_threads == 0) actual_threads = 4;  // Fallback if detection fails
    params.n_threads = static_cast<uint32_t>(actual_threads);
    params.n_threads_batch = static_cast<uint32_t>(actual_threads);

    return llama_init_from_model(model, params);
}

void llama_free_c(llama_context* ctx) {
    if (ctx) {
        llama_free(ctx);
    }
}

void llama_kv_cache_clear_c(llama_context* ctx) {
    if (ctx) {
        // Use the new memory API: get memory and clear it
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_clear(mem, true);  // Clear data buffers too
        }
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

/**
 * Check if a token is an end-of-generation token.
 *
 * Unlike llama_token_eos_c() which returns the primary EOS token,
 * this checks ALL end-of-generation tokens registered in the model vocabulary.
 * Modern models have multiple EOG tokens:
 *   - Llama 3: <|eot_id|> (128009) + <|end_of_text|> (128001)
 *   - Gemma: <end_of_turn> (107)
 *   - Qwen: <|im_end|> + <|endoftext|>
 *
 * @return true if the token is any end-of-generation token
 */
bool llama_vocab_is_eog_c(const llama_model* model, int32_t token) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llama_vocab_is_eog(vocab, token);
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

/**
 * Format chat messages using the model's built-in chat template.
 *
 * @param model     The llama model (for extracting chat template metadata)
 * @param roles     Array of role strings ("user", "assistant", "system")
 * @param contents  Array of content strings
 * @param n_msg     Number of messages
 * @param buf       Output buffer for formatted prompt
 * @param buf_size  Size of output buffer
 * @return          Length of formatted prompt, or negative on error
 */
int llama_format_chat_with_model_c(
    const llama_model* model,
    const char** roles,
    const char** contents,
    size_t n_msg,
    char* buf,
    int buf_size
) {
    if (!model || !roles || !contents || n_msg == 0) {
        return -1;
    }

    // Build llama_chat_message array
    std::vector<llama_chat_message> messages(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        messages[i].role = roles[i];
        messages[i].content = contents[i];
    }

    // Get model's chat template from metadata
    // Pass nullptr to use model's built-in template
    int result = llama_chat_apply_template(
        nullptr,  // Use model's default template
        messages.data(),
        n_msg,
        true,     // add_ass: add assistant start tag
        buf,
        buf_size
    );

    return result;
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

// =============================================================================
// Sampler Management
// =============================================================================

llama_sampler* llama_sampler_chain_create_c(
    float temperature,
    float top_p,
    float min_p,
    int top_k,
    float repeat_penalty,
    int penalty_last_n,
    uint32_t seed
) {
    // Create sampler chain with default params
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(chain_params);

    // Add samplers in order: penalties -> top_k -> top_p -> min_p -> temp -> dist
    // Repetition penalty must come first to modify logits before sampling
    if (repeat_penalty != 1.0f && penalty_last_n > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            penalty_last_n,    // penalty_last_n: how many tokens to consider
            repeat_penalty,    // penalty_repeat: > 1.0 penalizes repetition
            0.0f,              // penalty_freq: frequency penalty (disabled)
            0.0f               // penalty_present: presence penalty (disabled)
        ));
    }

    if (top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    }
    // min_p: prune tokens with probability < min_p * max_probability.
    // More adaptive than top_p — aggressive when confident, permissive when uncertain.
    if (min_p > 0.0f && min_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
    }
    if (temperature > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    } else {
        // Greedy decoding when temperature is 0
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }

    return chain;
}

void llama_sampler_free_c(llama_sampler* smpl) {
    if (smpl) {
        llama_sampler_free(smpl);
    }
}

// =============================================================================
// Stop Sequence Checking
// =============================================================================

/**
 * Check if the generated tokens end with any of the stop sequences.
 *
 * @param output_tokens Generated tokens so far
 * @param n_generated   Number of generated tokens
 * @param stop_seqs     Array of stop sequences (flattened token IDs)
 * @param stop_lens     Length of each stop sequence
 * @param n_stop_seqs   Number of stop sequences
 * @return true if a stop sequence was matched
 */
static bool check_stop_sequences(
    const int32_t* output_tokens,
    int n_generated,
    const int32_t* stop_seqs,
    const int* stop_lens,
    int n_stop_seqs
) {
    if (!stop_seqs || !stop_lens || n_stop_seqs <= 0 || n_generated <= 0) {
        return false;
    }

    int seq_offset = 0;
    for (int s = 0; s < n_stop_seqs; s++) {
        int seq_len = stop_lens[s];

        // Check if we have enough tokens to match this stop sequence
        if (seq_len > 0 && n_generated >= seq_len) {
            bool match = true;
            for (int i = 0; i < seq_len; i++) {
                if (output_tokens[n_generated - seq_len + i] != stop_seqs[seq_offset + i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return true;
            }
        }
        seq_offset += seq_len;
    }
    return false;
}

// =============================================================================
// Generation Loop
// =============================================================================

/**
 * Generate tokens from input tokens using autoregressive decoding.
 *
 * @param ctx         The llama context
 * @param model       The llama model (for EOS token)
 * @param input_tokens Input token array
 * @param n_input     Number of input tokens
 * @param output_tokens Output buffer for generated tokens
 * @param max_tokens  Maximum tokens to generate
 * @param temperature Sampling temperature (0 = greedy)
 * @param top_p       Top-p (nucleus) sampling threshold
 * @param min_p       Min-p sampling threshold (0.0 = disabled, 0.05 = default)
 * @param top_k       Top-k sampling (0 = disabled)
 * @param repeat_penalty Repetition penalty (1.0 = disabled, > 1.0 = penalize)
 * @param seed        Random seed for sampling
 * @param stop_seqs   Flattened array of stop sequence token IDs (can be NULL)
 * @param stop_lens   Length of each stop sequence (can be NULL)
 * @param n_stop_seqs Number of stop sequences (0 if none)
 * @return Number of tokens generated, or negative on error
 */
int llama_generate_c(
    llama_context* ctx,
    const llama_model* model,
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
    int n_stop_seqs
) {
    if (!ctx || !model || !input_tokens || !output_tokens || n_input <= 0 || max_tokens <= 0) {
        return -1;
    }

    // Validate input tokens fit within context window.
    // n_input must be strictly less than n_ctx to leave room for at least 1 generated token.
    const int n_ctx = llama_n_ctx(ctx);
    if (n_input >= n_ctx) {
        fprintf(stderr, "llama_generate_c: input tokens (%d) >= context window (%d)\n", n_input, n_ctx);
        return -4;  // Input too long for context window
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Create sampler chain with repetition penalty
    // penalty_last_n = 64 is a reasonable default (consider last 64 tokens for penalty)
    llama_sampler* sampler = llama_sampler_chain_create_c(
        temperature, top_p, min_p, top_k, repeat_penalty, 64, seed
    );
    if (!sampler) {
        return -2;
    }

    // Use a batch sized to n_batch (the decode limit per call, default 512).
    // Input tokens are processed in chunks of n_batch to avoid exceeding
    // llama_decode's per-call limit.
    const int n_batch = llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch > 0 ? n_batch : 512, 0, 1);

    int n_generated = 0;
    int n_cur = 0;  // Current position in context

    // Pre-allocate candidates buffer once — reused every token.
    // Avoids ~128KB alloc/free per token (n_vocab * sizeof(llama_token_data)).
    llama_token_data* candidates_data = new llama_token_data[n_vocab];

    // Process input tokens in chunks of n_batch.
    // Previously tried to decode all tokens in one call, which failed when
    // n_input > n_batch (default 512).
    for (int chunk_start = 0; chunk_start < n_input; chunk_start += n_batch) {
        int chunk_end = chunk_start + n_batch;
        if (chunk_end > n_input) chunk_end = n_input;

        batch.n_tokens = 0;
        for (int i = chunk_start; i < chunk_end; i++) {
            batch.token[batch.n_tokens] = input_tokens[i];
            batch.pos[batch.n_tokens] = n_cur;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            // Only request logits for the very last input token
            batch.logits[batch.n_tokens] = (i == n_input - 1) ? 1 : 0;
            batch.n_tokens++;
            n_cur++;
        }

        if (llama_decode(ctx, batch) != 0) {
            delete[] candidates_data;
            llama_batch_free(batch);
            llama_sampler_free(sampler);
            return -3;
        }
    }

    // Generation loop
    while (n_generated < max_tokens) {
        // Get logits for the last token
        float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!logits) {
            break;
        }

        // Fill candidates from logits (reuses pre-allocated buffer)
        llama_token_data_array candidates;
        for (int i = 0; i < n_vocab; i++) {
            candidates_data[i].id = i;
            candidates_data[i].logit = logits[i];
            candidates_data[i].p = 0.0f;
        }

        candidates.data = candidates_data;
        candidates.size = n_vocab;
        candidates.selected = -1;
        candidates.sorted = false;

        // Apply sampler chain to get next token
        llama_sampler_apply(sampler, &candidates);

        llama_token new_token = candidates.data[candidates.selected].id;

        // Accept token in sampler (for repetition penalty etc)
        llama_sampler_accept(sampler, new_token);

        // Store generated token
        output_tokens[n_generated] = new_token;
        n_generated++;

        // Check for end-of-generation (covers ALL EOG tokens, not just primary EOS).
        // Llama 3: <|eot_id|> + <|end_of_text|>, Gemma: <end_of_turn>, Qwen: <|im_end|>, etc.
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Check for stop sequences
        if (check_stop_sequences(output_tokens, n_generated, stop_seqs, stop_lens, n_stop_seqs)) {
            break;
        }

        // Prepare batch for next token
        batch.n_tokens = 0;
        batch.token[0] = new_token;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;  // Request logits for this token
        batch.n_tokens = 1;
        n_cur++;

        // Decode the new token
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
    }

    delete[] candidates_data;
    llama_batch_free(batch);
    llama_sampler_free(sampler);

    return n_generated;
}

// =============================================================================
// Streaming Generation
// =============================================================================

/**
 * Callback type for streaming token generation.
 *
 * @param token_id   The raw token ID
 * @param token_text The decoded token text (null-terminated)
 * @param user_data  User-provided context pointer
 * @return 0 to continue, non-zero to stop generation
 */
typedef int (*token_callback_t)(int32_t token_id, const char* token_text, void* user_data);

/**
 * Generate tokens with streaming callback.
 *
 * Same as llama_generate_c but calls the callback for each generated token.
 * If the callback returns non-zero, generation stops early.
 *
 * @param ctx         The llama context
 * @param model       The llama model
 * @param input_tokens Input token array
 * @param n_input     Number of input tokens
 * @param output_tokens Output buffer for generated tokens
 * @param max_tokens  Maximum tokens to generate
 * @param temperature Sampling temperature (0 = greedy)
 * @param top_p       Top-p (nucleus) sampling threshold
 * @param min_p       Min-p sampling threshold (0.0 = disabled, 0.05 = default)
 * @param top_k       Top-k sampling (0 = disabled)
 * @param repeat_penalty Repetition penalty (1.0 = disabled)
 * @param seed        Random seed for sampling
 * @param stop_seqs   Flattened array of stop sequence token IDs (can be NULL)
 * @param stop_lens   Length of each stop sequence (can be NULL)
 * @param n_stop_seqs Number of stop sequences (0 if none)
 * @param callback    Callback function called for each token (can be NULL)
 * @param user_data   User data passed to callback
 * @return Number of tokens generated, or negative on error
 */
int llama_generate_streaming_c(
    llama_context* ctx,
    const llama_model* model,
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
    token_callback_t callback,
    void* user_data
) {
    if (!ctx || !model || !input_tokens || !output_tokens || n_input <= 0 || max_tokens <= 0) {
        return -1;
    }

    // Validate input tokens fit within context window.
    const int n_ctx = llama_n_ctx(ctx);
    if (n_input >= n_ctx) {
        fprintf(stderr, "llama_generate_streaming_c: input tokens (%d) >= context window (%d)\n", n_input, n_ctx);
        return -4;  // Input too long for context window
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Create sampler chain with repetition penalty
    llama_sampler* sampler = llama_sampler_chain_create_c(
        temperature, top_p, min_p, top_k, repeat_penalty, 64, seed
    );
    if (!sampler) {
        return -2;
    }

    // Use a batch sized to n_batch (the decode limit per call, default 512).
    const int n_batch = llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch > 0 ? n_batch : 512, 0, 1);

    int n_generated = 0;
    int n_cur = 0;  // Current position in context
    bool stopped_by_callback = false;

    // Pre-allocate candidates buffer once — reused every token.
    llama_token_data* candidates_data = new llama_token_data[n_vocab];

    // Process input tokens in chunks of n_batch.
    for (int chunk_start = 0; chunk_start < n_input; chunk_start += n_batch) {
        int chunk_end = chunk_start + n_batch;
        if (chunk_end > n_input) chunk_end = n_input;

        batch.n_tokens = 0;
        for (int i = chunk_start; i < chunk_end; i++) {
            batch.token[batch.n_tokens] = input_tokens[i];
            batch.pos[batch.n_tokens] = n_cur;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = (i == n_input - 1) ? 1 : 0;
            batch.n_tokens++;
            n_cur++;
        }

        if (llama_decode(ctx, batch) != 0) {
            delete[] candidates_data;
            llama_batch_free(batch);
            llama_sampler_free(sampler);
            return -3;
        }
    }

    // Buffer for token text conversion (1024 to handle merged/multi-byte tokens)
    char token_buf[1024];

    // Generation loop
    while (n_generated < max_tokens) {
        // Get logits for the last token
        float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!logits) {
            break;
        }

        // Fill candidates from logits (reuses pre-allocated buffer)
        llama_token_data_array candidates;
        for (int i = 0; i < n_vocab; i++) {
            candidates_data[i].id = i;
            candidates_data[i].logit = logits[i];
            candidates_data[i].p = 0.0f;
        }

        candidates.data = candidates_data;
        candidates.size = n_vocab;
        candidates.selected = -1;
        candidates.sorted = false;

        // Apply sampler chain to get next token
        llama_sampler_apply(sampler, &candidates);

        llama_token new_token = candidates.data[candidates.selected].id;

        // Accept token in sampler (for repetition penalty etc)
        llama_sampler_accept(sampler, new_token);

        // Store generated token
        output_tokens[n_generated] = new_token;
        n_generated++;

        // Check for end-of-generation BEFORE emitting to callback.
        // This prevents EOG tokens (e.g. <|im_end|>, <end_of_turn>, <|eot_id|>)
        // from leaking into the output as literal text.
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Call streaming callback if provided
        if (callback) {
            // Convert token to text
            int len = llama_token_to_piece_c(model, new_token, token_buf, sizeof(token_buf) - 1, 0, true);
            if (len > 0) {
                token_buf[len] = '\0';
            } else {
                token_buf[0] = '\0';
            }

            // Call callback - if it returns non-zero, stop generation
            int cb_result = callback(new_token, token_buf, user_data);
            if (cb_result != 0) {
                stopped_by_callback = true;
                break;
            }
        }

        // Check for stop sequences
        if (check_stop_sequences(output_tokens, n_generated, stop_seqs, stop_lens, n_stop_seqs)) {
            break;
        }

        // Prepare batch for next token
        batch.n_tokens = 0;
        batch.token[0] = new_token;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;
        n_cur++;

        // Decode the new token
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
    }

    delete[] candidates_data;
    llama_batch_free(batch);
    llama_sampler_free(sampler);

    // Return negative if stopped by callback (to distinguish from normal completion)
    // The absolute value is still the number of tokens generated
    return stopped_by_callback ? -n_generated : n_generated;
}

} // extern "C"
