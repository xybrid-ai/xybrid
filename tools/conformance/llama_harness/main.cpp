// Same-artifact (G2) harness for final-position label scoring.
//
// Scores one prompt the way xybrid-llama's LlamaContext::score_final_position
// does, through llama.h only: same context settings, a cleared memory, the
// prompt decoded in chunks of llama_n_batch(ctx) with output on the last
// token, and the log-sum-exp of the final row summed in double in index
// order. Linked against the llama.cpp libraries cargo links into
// xybrid-llama, its output must equal the Rust result bit for bit.
//
// Usage: llama_harness MODEL N_CTX N_BATCH N_THREADS < REQUEST
//   REQUEST is two lines of whitespace-separated token ids: the prompt, then
//   the labels. The model is loaded on the CPU (no GPU layers) and flash
//   attention is off, matching the test that drives this harness.
// Output: "logsumexp BITS" then "labels BITS BITS ...", each BITS being the
//   IEEE-754 bit pattern of an f32, in decimal.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> read_ids(std::istream & in) {
    std::string line;
    std::vector<int32_t> ids;
    if (!std::getline(in, line)) {
        return ids;
    }
    std::istringstream stream(line);
    int32_t id;
    while (stream >> id) {
        ids.push_back(id);
    }
    return ids;
}

uint32_t bits(float value) {
    uint32_t out;
    std::memcpy(&out, &value, sizeof out);
    return out;
}

void quiet(ggml_log_level, const char *, void *) {}

int fail(const char * message) {
    std::fprintf(stderr, "llama_harness: %s\n", message);
    return 1;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL N_CTX N_BATCH N_THREADS < REQUEST\n", argv[0]);
        return 2;
    }
    const int n_ctx = std::atoi(argv[2]);
    const int n_batch = std::atoi(argv[3]);
    const int n_threads = std::atoi(argv[4]);
    const std::vector<int32_t> tokens = read_ids(std::cin);
    const std::vector<int32_t> labels = read_ids(std::cin);
    if (n_ctx <= 0 || n_threads <= 0 || tokens.empty() || labels.empty()) {
        return fail("needs a positive N_CTX and N_THREADS, a prompt line and a labels line");
    }

    llama_log_set(quiet, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        return fail("cannot load the model");
    }

    // The settings xybrid-llama's context creation applies.
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(n_ctx);
    context_params.n_batch = static_cast<uint32_t>(n_batch > 0 ? n_batch : 512);
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.n_threads = n_threads;
    context_params.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) {
        return fail("cannot create the context");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const size_t window = llama_n_ctx(ctx);
    const auto in_vocab = [n_vocab](int32_t id) { return id >= 0 && id < n_vocab; };
    if (tokens.size() >= window || !std::all_of(tokens.begin(), tokens.end(), in_vocab) ||
        !std::all_of(labels.begin(), labels.end(), in_vocab)) {
        return fail("the prompt does not fit, or an id is outside the vocabulary");
    }

    llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        return fail("the context has no memory: llama.cpp runs this model as an encoder");
    }
    llama_memory_clear(memory, true);
    // llama.cpp's own limit per decode call, which xybrid-llama mirrors.
    const size_t chunk = llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(static_cast<int32_t>(chunk), 0, 1);
    for (size_t start = 0; start < tokens.size(); start += chunk) {
        const size_t end = std::min(tokens.size(), start + chunk);
        batch.n_tokens = 0;
        for (size_t i = start; i < end; ++i) {
            const int32_t j = batch.n_tokens++;
            batch.token[j] = tokens[i];
            batch.pos[j] = static_cast<llama_pos>(i);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = i + 1 == tokens.size();
        }
        if (llama_decode(ctx, batch) != 0) {
            return fail("llama_decode failed");
        }
    }

    const float * row = llama_get_logits(ctx);
    if (!row) {
        return fail("llama_get_logits returned no logits");
    }
    float max = -INFINITY;
    for (int32_t i = 0; i < n_vocab; ++i) {
        max = std::fmax(max, row[i]);
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum += std::exp(static_cast<double>(row[i]) - static_cast<double>(max));
    }
    const float logsumexp = static_cast<float>(static_cast<double>(max) + std::log(sum));

    std::printf("logsumexp %u\nlabels", bits(logsumexp));
    for (const int32_t id : labels) {
        std::printf(" %u", bits(row[id]));
    }
    std::printf("\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
