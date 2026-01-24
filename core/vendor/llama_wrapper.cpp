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

// =============================================================================
// Sampler Management
// =============================================================================

llama_sampler* llama_sampler_chain_create_c(
    float temperature,
    float top_p,
    int top_k,
    uint32_t seed
) {
    // Create sampler chain with default params
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(chain_params);

    // Add samplers in order: top_k -> top_p -> temp -> dist
    if (top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
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
 * @param top_k       Top-k sampling (0 = disabled)
 * @param seed        Random seed for sampling
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
    int top_k,
    uint32_t seed
) {
    if (!ctx || !model || !input_tokens || !output_tokens || n_input <= 0 || max_tokens <= 0) {
        return -1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const llama_token eos_token = llama_vocab_eos(vocab);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Create sampler chain
    llama_sampler* sampler = llama_sampler_chain_create_c(temperature, top_p, top_k, seed);
    if (!sampler) {
        return -2;
    }

    // Create batch for decoding
    llama_batch batch = llama_batch_init(512, 0, 1);

    int n_generated = 0;
    int n_cur = 0;  // Current position in context

    // First, process all input tokens
    batch.n_tokens = 0;
    for (int i = 0; i < n_input; i++) {
        batch.token[batch.n_tokens] = input_tokens[i];
        batch.pos[batch.n_tokens] = n_cur;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        // Only request logits for the last input token
        batch.logits[batch.n_tokens] = (i == n_input - 1) ? 1 : 0;
        batch.n_tokens++;
        n_cur++;
    }

    // Decode input tokens
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        llama_sampler_free(sampler);
        return -3;
    }

    // Generation loop
    while (n_generated < max_tokens) {
        // Get logits for the last token
        float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!logits) {
            break;
        }

        // Create token data array for sampling
        llama_token_data_array candidates;
        llama_token_data* candidates_data = new llama_token_data[n_vocab];

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
        delete[] candidates_data;

        // Accept token in sampler (for repetition penalty etc)
        llama_sampler_accept(sampler, new_token);

        // Store generated token
        output_tokens[n_generated] = new_token;
        n_generated++;

        // Check for EOS
        if (new_token == eos_token) {
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

    llama_batch_free(batch);
    llama_sampler_free(sampler);

    return n_generated;
}

} // extern "C"
