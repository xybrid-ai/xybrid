// decision-oracle: llama.cpp's own SystemOne decision code, run outside llama-server and observed.
//
// It links tools/server/server-decision.cpp of the pinned llama.cpp, unmodified, from upstream's
// server-context library (see CMakeLists.txt), and runs each request the way llama-server's
// POST /v1/systemone handler does (post_systemone in tools/server/server-context.cpp):
//
//   parse     common_json::parse, parse_questions, parse_state, prepare_state, the media check
//   prepare   fill_task for every question and variant, in request order
//   evaluate  one decode per task, as a server slot runs a decision prompt: the whole prompt in one
//             batch of sequence 0 at positions 0..n-1, output on the last token only, then the
//             logits of the labels and the max of each label group (send_decision); or synthetic
//             logits given with the request
//   finalize  format_answer for every question, then the usage
//
// It writes every intermediate value as JSON: the rendered prompt, its tokens, the output index,
// the label codes and token groups, the temperature, the logits and the answers. An exception
// stops the request at its stage, with the status llama-server would answer. test_access.h lists
// the private members it reads.
//
// Usage:
//   decision-oracle run --input IN.json --output OUT.json -- <llama-server arguments>
//   decision-oracle run --input IN.json --output OUT.json --vocab-only MODEL.gguf
//   decision-oracle variant --base IN.gguf --output OUT.gguf [--set KEY=f32:VALUE|KEY=str:TEXT]...
//                           [--remove KEY]... [--token-type ID=TYPE]...
//
// README.md documents the input and output formats.

#include "test_access.h"

#include "server-common.h"
#include "server-decision.h"
#include "server-task.h"

#include "arg.h"
#include "common.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

//
// JSON output: compact, f32 as bit patterns, f64 with 17 significant digits
//

std::string quote(const std::string & s) {
    return json::make(s).dump();
}

std::string f32_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    char buf[16];
    snprintf(buf, sizeof(buf), "0x%08x", bits);
    return buf;
}

float f32_from_bits(const std::string & text) {
    if (text.size() != 10 || text.compare(0, 2, "0x") != 0) {
        throw std::runtime_error("an f32 must be written as its bit pattern, 0x%08x: " + text);
    }
    const uint32_t bits = (uint32_t) std::stoul(text.substr(2), nullptr, 16);
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// a JSON number, or a string for a value JSON cannot hold; an integral value keeps a ".0",
// so that it reads back as a float
std::string f64_json(double v) {
    if (std::isnan(v)) {
        return "\"NaN\"";
    }
    if (std::isinf(v)) {
        return v > 0 ? "\"Infinity\"" : "\"-Infinity\"";
    }
    char buf[40];
    snprintf(buf, sizeof(buf), "%.17g", v);
    std::string out = buf;
    if (out.find_first_of(".e") == std::string::npos) {
        out += ".0";
    }
    return out;
}

struct json_writer {
    std::string out;
    std::vector<bool> has_items;
    bool after_key = false;

    void item() {
        if (after_key) {
            after_key = false;
            return;
        }
        if (!has_items.empty()) {
            if (has_items.back()) {
                out += ',';
            }
            has_items.back() = true;
        }
    }
    void raw(const std::string & text) { item(); out += text; }
    void str(const std::string & s) { raw(quote(s)); }
    void num(int64_t v) { raw(std::to_string(v)); }
    void boolean(bool v) { raw(v ? "true" : "false"); }
    void key(const std::string & k) { item(); out += quote(k); out += ':'; after_key = true; }
    void begin_object() { item(); out += '{'; has_items.push_back(false); }
    void end_object() { out += '}'; has_items.pop_back(); }
    void begin_array() { item(); out += '['; has_items.push_back(false); }
    void end_array() { out += ']'; has_items.pop_back(); }

    void f32_array(const std::vector<float> & vals) {
        begin_array();
        for (const float v : vals) {
            str(f32_bits(v));
        }
        end_array();
    }
    template <typename T>
    void int_array(const std::vector<T> & vals) {
        begin_array();
        for (const T v : vals) {
            num((int64_t) v);
        }
        end_array();
    }
    void str_array(const std::vector<std::string> & vals) {
        begin_array();
        for (const auto & v : vals) {
            str(v);
        }
        end_array();
    }

    // a value from upstream: floats get 17 significant digits, everything else upstream's own dump
    void value(const json & j) {
        if (j.is_object()) {
            begin_object();
            for (const auto & [k, v] : j.items()) {
                key(k);
                value(v);
            }
            end_object();
        } else if (j.is_array()) {
            begin_array();
            for (const auto & v : j) {
                value(v);
            }
            end_array();
        } else if (j.is_number_float()) {
            raw(f64_json(j.get<double>()));
        } else {
            raw(j.dump());
        }
    }
};

//
// refusals
//

struct refusal {
    std::string question; // the question being processed, empty for the request as a whole
    std::string exception;
    int         status = 500;
    std::string type;
    std::string message;
};

// the status ex_wrapper (tools/server/server.cpp) gives each exception, through format_error_response
refusal classify(const std::exception_ptr & ep, const std::string & question) {
    try {
        std::rethrow_exception(ep);
    } catch (const std::invalid_argument & e) {
        return {question, "invalid_argument", 400, "invalid_request_error", e.what()};
    } catch (const common_json_error & e) {
        return {question, "common_json_error", 400, "invalid_request_error", e.what()};
    } catch (const std::runtime_error & e) {
        return {question, "runtime_error", 500, "server_error", e.what()};
    } catch (const std::logic_error & e) {
        return {question, "logic_error", 500, "server_error", e.what()};
    } catch (const std::exception & e) {
        return {question, "exception", 500, "server_error", e.what()};
    } catch (...) {
        return {question, "unknown", 500, "server_error", "unknown error"};
    }
}

void write_refusal(json_writer & w, const refusal & r) {
    w.key("refusal");
    w.begin_object();
    if (!r.question.empty()) {
        w.key("question");
        w.str(r.question);
    }
    w.key("exception");
    w.str(r.exception);
    w.key("status");
    w.num(r.status);
    w.key("type");
    w.str(r.type);
    w.key("message");
    w.str(r.message);
    w.end_object();
}

const char * question_type_name(server_decision_question_type type) {
    switch (type) {
        case SERVER_DECISION_QUESTION_CHOICE: return "choice";
        case SERVER_DECISION_QUESTION_SCORE:  return "score";
        case SERVER_DECISION_QUESTION_NOUL:   return "noul";
    }
    return "";
}

std::string meta_str(const llama_model * model, const std::string & key) {
    std::vector<char> buf(1 << 16);
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf.data(), buf.size());
    return n < 0 ? std::string() : std::string(buf.data());
}

//
// the model, as llama-server loads it
//

struct oracle_model {
    common_params          params;
    common_init_result_ptr init;              // the server's path: model and context
    llama_model *          vocab_model = nullptr; // --vocab-only: metadata and vocabulary, no context
    llama_model *          model = nullptr;
    llama_context *        ctx   = nullptr;
    const llama_vocab *    vocab = nullptr;
    server_decision_context decision;

    ~oracle_model() {
        if (vocab_model != nullptr) {
            llama_model_free(vocab_model);
        }
    }
};

void write_facts(json_writer & w, const oracle_model & m) {
    w.key("facts");
    w.begin_object();

    const std::string arch = meta_str(m.model, "general.architecture");
    w.key("architecture");
    w.str(arch);
    w.key("decision_type");
    w.str(meta_str(m.model, arch + ".decision.type"));

    // every "<arch>.decision.*" key, as the string accessor gives it
    const std::string prefix = arch + ".decision.";
    w.key("decision_metadata");
    w.begin_object();
    for (int32_t i = 0; i < llama_model_meta_count(m.model); i++) {
        char key[256];
        if (llama_model_meta_key_by_index(m.model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix)) {
            continue;
        }
        std::vector<char> val(1 << 16);
        if (llama_model_meta_val_str_by_index(m.model, i, val.data(), val.size()) < 0) {
            continue;
        }
        w.key(key);
        w.str(val.data());
    }
    w.end_object();

    // the tables init() built from them
    w.key("temperatures");
    w.begin_object();
    for (const auto & [name, temp] : decision_oracle::temperatures(m.decision)) {
        w.key(name);
        w.str(f32_bits(temp));
    }
    w.end_object();
    w.key("score_bias");
    w.begin_object();
    for (const auto & [n, bias] : decision_oracle::score_bias(m.decision)) {
        w.key(std::to_string(n));
        w.f32_array(bias);
    }
    w.end_object();

    const char * tmpl = llama_model_chat_template(m.model, "systemone");
    w.key("template");
    if (tmpl == nullptr) {
        w.raw("null");
    } else {
        w.str(tmpl);
    }
    w.key("quoted_spellings");
    w.str_array(decision_oracle::quoted_spellings(m.decision));
    w.key("n_vocab");
    w.num(llama_vocab_n_tokens(m.vocab));
    w.key("add_bos");
    w.boolean(llama_vocab_get_add_bos(m.vocab));

    if (m.ctx != nullptr) {
        w.key("context");
        w.begin_object();
        w.key("n_ctx");
        w.num(llama_n_ctx(m.ctx));
        w.key("n_batch");
        w.num(llama_n_batch(m.ctx));
        w.key("n_ubatch");
        w.num(llama_n_ubatch(m.ctx));
        w.key("n_seq_max");
        w.num(llama_n_seq_max(m.ctx));
        w.key("n_outputs_max");
        w.num(m.params.n_outputs_max);
        w.key("n_outputs_max_per_seq");
        w.num(m.params.n_outputs_max_per_seq);
        w.key("n_threads");
        w.num(llama_n_threads(m.ctx));
        w.key("n_threads_batch");
        w.num(llama_n_threads_batch(m.ctx));
        w.key("flash_attn");
        w.str(m.params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED ? "on" : m.params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED ? "off" : "auto");
        w.key("causal_attn");
        w.boolean(llama_get_causal_attn(m.ctx));
        w.key("memory");
        w.boolean(llama_get_memory(m.ctx) != nullptr);
        w.end_object();
    }
    w.end_object();
}

//
// one request
//

struct oracle_case {
    std::string id;
    std::string request_text;
    bool has_synthetic = false;
    std::vector<std::vector<float>> synthetic; // per task, the logits of its labels
    bool has_media = false;                    // passed to format_answer, as if files were sent
};

// the checks a slot makes before it reads a prompt it cannot split (update_slots, server-context.cpp),
// then the decode and the readout of send_decision
bool evaluate_task(const oracle_model & m, const server_task & task, std::vector<float> & select, refusal & err) {
    const llama_tokens & tokens = task.tokens.get_tokens();
    const int32_t n_tokens = (int32_t) tokens.size();
    const int32_t n_ubatch = (int32_t) llama_n_ubatch(m.ctx);
    const int32_t n_ctx    = (int32_t) llama_n_ctx(m.ctx) / std::max(1, m.params.n_parallel);

    err.exception = "slot";
    if (n_tokens > n_ubatch) {
        err.status  = 500;
        err.type    = "server_error";
        err.message = string_format("input (%d tokens) is too large to process. increase the physical batch size (current batch size: %d)", n_tokens, n_ubatch);
        return false;
    }
    if (n_tokens > n_ctx) {
        err.status  = 400;
        err.type    = "exceed_context_size_error";
        err.message = string_format("input (%d tokens) is larger than the max context size (%d tokens). skipping", n_tokens, n_ctx);
        return false;
    }

    common_batch batch(m.ctx);
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.add(tokens[i], i, 0, false);
    }
    batch.set_output(n_tokens - 1, true);

    const int32_t ret = llama_process(m.ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (ret != 0) {
        err.status  = 500;
        err.type    = "server_error";
        err.message = ret == -1 ? "Invalid input batch." : ret < -1 ? "Compute error." : string_format("decode returned %d", ret);
        return false;
    }
    llama_synchronize(m.ctx);

    const float * logits = llama_get_logits_ith(m.ctx, n_tokens - 1);
    if (logits == nullptr) {
        err.status  = 500;
        err.type    = "server_error";
        err.message = "failed to get logits";
        return false;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(m.vocab);
    select.clear();
    for (const llama_token label : task.decision.labels) {
        GGML_ASSERT(label >= 0 && label < n_vocab);
        select.push_back(logits[label]);
    }
    return true;
}

// send_decision: each output is the max of its label group
std::vector<float> group_scores(const server_task & task, const std::vector<float> & select) {
    if (task.decision.label_groups.empty()) {
        return select;
    }
    std::vector<float> out;
    size_t i = 0;
    for (const int32_t n : task.decision.label_groups) {
        GGML_ASSERT(n > 0 && i + n <= select.size());
        out.push_back(*std::max_element(select.begin() + i, select.begin() + i + n));
        i += n;
    }
    return out;
}

struct prepared_task {
    size_t question;
    size_t variant;
    server_task task;
    std::string prompt;
    std::vector<std::string> codes;
    std::vector<llama_tokens> groups;
    float temperature;
};

void run_case(json_writer & w, oracle_model & m, const oracle_case & c) {
    w.begin_object();
    w.key("id");
    w.str(c.id);

    // parse
    json body;
    std::vector<server_decision_question> questions;
    std::vector<raw_buffer> files;
    json state;
    {
        w.key("parse");
        w.begin_object();
        try {
            body      = json::parse(c.request_text);
            questions = m.decision.parse_questions(body);
            state     = m.decision.prepare_state(m.decision.parse_state(body, files));
        } catch (...) {
            write_refusal(w, classify(std::current_exception(), ""));
            w.end_object();
            w.end_object();
            return;
        }
        // no --mmproj: the server answers 501
        if (!files.empty()) {
            write_refusal(w, refusal{"", "not_supported", 501, "not_supported_error",
                "This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`"});
            w.end_object();
            w.end_object();
            return;
        }
        w.key("questions");
        w.begin_array();
        for (const auto & q : questions) {
            w.begin_object();
            w.key("id");
            w.str(q.id);
            w.key("type");
            w.str(question_type_name(q.type));
            w.key("keys");
            w.begin_array();
            for (const auto & opt : q.options) {
                w.str(opt.key);
            }
            w.end_array();
            w.end_object();
        }
        w.end_array();
        w.end_object();
    }

    // prepare
    std::vector<prepared_task> tasks;
    {
        w.key("prepare");
        w.begin_object();
        const mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
        for (size_t iq = 0; iq < questions.size(); iq++) {
            const auto & question = questions[iq];
            for (size_t variant = 0; variant < m.decision.n_variants(question); variant++) {
                prepared_task pt{iq, variant, server_task(SERVER_TASK_TYPE_DECISION), {}, {}, {}, 0.0f};
                try {
                    m.decision.fill_task(state, questions, question, variant, files, nullptr, init_opt, pt.task);
                    pt.prompt = decision_oracle::render(m.decision, state, questions, question, variant, files.size());
                    decision_oracle::d1_labels(m.decision, question, pt.codes, pt.groups);
                    pt.temperature = decision_oracle::get_temperature(m.decision, question);
                } catch (...) {
                    write_refusal(w, classify(std::current_exception(), question.id));
                    w.end_object();
                    w.end_object();
                    return;
                }
                tasks.push_back(std::move(pt));
            }
        }

        w.key("tasks");
        w.begin_array();
        for (const auto & pt : tasks) {
            const auto & question = questions[pt.question];
            const llama_tokens & tokens = pt.task.tokens.get_tokens();

            // the prompt written here is the one fill_task tokenized
            if (common_tokenize(m.vocab, pt.prompt, false, true) != tokens) {
                throw std::runtime_error("case " + c.id + ": the rendered prompt does not give the tokens of fill_task");
            }
            // and the groups are the ones it gave the task
            llama_tokens flat;
            std::vector<int32_t> sizes;
            for (const auto & g : pt.groups) {
                flat.insert(flat.end(), g.begin(), g.end());
                sizes.push_back((int32_t) g.size());
            }
            if (flat != pt.task.decision.labels || sizes != pt.task.decision.label_groups || pt.codes.size() != question.options.size()) {
                throw std::runtime_error("case " + c.id + ": d1_labels does not give the labels of fill_task");
            }

            w.begin_object();
            w.key("question");
            w.str(question.id);
            w.key("type");
            w.str(question_type_name(question.type));
            w.key("variant");
            w.num((int64_t) pt.variant);
            w.key("prompt");
            w.str(pt.prompt);
            w.key("token_ids");
            w.int_array(tokens);
            w.key("output_index");
            w.num((int64_t) tokens.size() - 1);
            w.key("select");
            w.int_array(pt.task.decision.labels);
            w.key("label_groups");
            w.int_array(pt.task.decision.label_groups);
            w.key("keys");
            w.begin_array();
            for (const auto & opt : question.options) {
                w.str(opt.key);
            }
            w.end_array();
            w.key("codes");
            w.str_array(pt.codes);
            w.key("temperature");
            w.str(f32_bits(pt.temperature));
            w.key("no_batching");
            w.boolean(pt.task.decision.no_batching);
            w.end_object();
        }
        w.end_array();
        w.end_object();
    }

    // evaluate: the logits of the labels of every task, from the model or given with the case
    std::vector<std::vector<float>> selects;
    std::vector<std::vector<float>> scores;
    if (!c.has_synthetic && m.ctx == nullptr) {
        // --vocab-only and no logits: prepare only
        w.end_object();
        return;
    }
    {
        w.key("evaluate");
        w.begin_object();
        if (c.has_synthetic && c.synthetic.size() != tasks.size()) {
            throw std::runtime_error("case " + c.id + ": synthetic logits for " + std::to_string(c.synthetic.size()) + " tasks, the request has " + std::to_string(tasks.size()));
        }
        for (size_t i = 0; i < tasks.size(); i++) {
            std::vector<float> select;
            if (c.has_synthetic) {
                select = c.synthetic[i];
                if (select.size() != tasks[i].task.decision.labels.size()) {
                    throw std::runtime_error("case " + c.id + ": task " + std::to_string(i) + " has " + std::to_string(tasks[i].task.decision.labels.size()) + " labels");
                }
            } else {
                refusal err;
                err.question = questions[tasks[i].question].id;
                if (!evaluate_task(m, tasks[i].task, select, err)) {
                    write_refusal(w, err);
                    w.end_object();
                    w.end_object();
                    return;
                }
            }
            scores.push_back(group_scores(tasks[i].task, select));
            selects.push_back(std::move(select));
        }
        w.key("source");
        w.str(c.has_synthetic ? "synthetic" : "model");
        w.key("tasks");
        w.begin_array();
        for (size_t i = 0; i < tasks.size(); i++) {
            w.begin_object();
            w.key("select_logits");
            w.f32_array(selects[i]);
            w.key("scores");
            w.f32_array(scores[i]);
            w.end_object();
        }
        w.end_array();
        w.end_object();
    }

    // finalize
    {
        w.key("finalize");
        w.begin_object();
        std::vector<json> answers;
        size_t i_task  = 0;
        int64_t n_tokens = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> q_scores;
            for (size_t variant = 0; variant < m.decision.n_variants(question); variant++) {
                q_scores.push_back(scores[i_task]);
                n_tokens += tasks[i_task].task.n_tokens();
                i_task++;
            }
            try {
                answers.push_back(m.decision.format_answer(question, q_scores, c.has_media));
            } catch (...) {
                write_refusal(w, classify(std::current_exception(), question.id));
                w.end_object();
                w.end_object();
                return;
            }
        }
        w.key("answers");
        w.begin_object();
        for (size_t iq = 0; iq < questions.size(); iq++) {
            w.key(questions[iq].id);
            w.value(answers[iq]);
        }
        w.end_object();
        w.key("usage");
        w.begin_object();
        w.key("input_tokens");
        w.num(n_tokens);
        w.key("output_tokens");
        w.num(0);
        w.end_object();
        w.end_object();
    }

    w.end_object();
}

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void write_file(const std::string & path, const std::string & text) {
    std::ofstream f(path, std::ios::binary);
    if (!f || !(f << text)) {
        throw std::runtime_error("cannot write " + path);
    }
}

std::vector<oracle_case> read_cases(const std::string & path) {
    const json input = json::parse(read_file(path));
    std::vector<oracle_case> cases;
    for (const auto & item : input.at("cases")) {
        oracle_case c;
        c.id           = item.at("id").get<std::string>();
        c.request_text = item.at("request_text").get<std::string>();
        c.has_media    = item.contains("has_media") && item.at("has_media").get<bool>();
        if (item.contains("select_logits") && !item.at("select_logits").is_null()) {
            c.has_synthetic = true;
            for (const auto & task : item.at("select_logits")) {
                std::vector<float> vals;
                for (const auto & v : task) {
                    vals.push_back(f32_from_bits(v.get<std::string>()));
                }
                c.synthetic.push_back(vals);
            }
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

int usage() {
    fprintf(stderr,
        "usage:\n"
        "  decision-oracle run --input IN.json --output OUT.json -- <llama-server arguments>\n"
        "  decision-oracle run --input IN.json --output OUT.json --vocab-only MODEL.gguf\n"
        "  decision-oracle variant --base IN.gguf --output OUT.gguf [--set KEY=f32:VALUE|KEY=str:TEXT]... [--remove KEY]...\n"
        "                          [--token-type ID=TYPE]...\n");
    return 2;
}

int cmd_run(int argc, char ** argv) {
    std::string input;
    std::string output;
    std::string vocab_only;
    int i_server = argc;
    for (int i = 0; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--") {
            i_server = i + 1;
            break;
        }
        if (i + 1 >= argc) {
            return usage();
        }
        if (arg == "--input") {
            input = argv[++i];
        } else if (arg == "--output") {
            output = argv[++i];
        } else if (arg == "--vocab-only") {
            vocab_only = argv[++i];
        } else {
            return usage();
        }
    }
    if (input.empty() || output.empty() || (vocab_only.empty() == (i_server >= argc))) {
        return usage();
    }
    const auto cases = read_cases(input);

    oracle_model m;
    if (!vocab_only.empty()) {
        llama_backend_init();
        llama_model_params mparams = llama_model_default_params();
        mparams.vocab_only = true;
        m.vocab_model = llama_model_load_from_file(vocab_only.c_str(), mparams);
        if (m.vocab_model == nullptr) {
            fprintf(stderr, "decision-oracle: cannot load %s\n", vocab_only.c_str());
            return 1;
        }
        m.model = m.vocab_model;
    } else {
        // llama-server's own argument parser and model setup (server.cpp, server_context_impl::load_model)
        std::vector<char *> server_argv = {argv[-1]};
        for (int i = i_server; i < argc; i++) {
            server_argv.push_back(argv[i]);
        }
        if (!common_params_parse((int) server_argv.size(), server_argv.data(), m.params, LLAMA_EXAMPLE_SERVER)) {
            return 1;
        }
        if (m.params.n_parallel != 1 || !m.params.mmproj.path.empty() || m.params.embedding) {
            fprintf(stderr, "decision-oracle: pass -np 1, and neither --mmproj nor --embedding\n");
            return 2;
        }
        llama_backend_init();
        llama_numa_init(m.params.numa);
        // server_output_limits: no mmproj, no embeddings
        const auto limits = common_speculative_get_output_limits(
                m.params.n_batch, m.params.n_parallel, common_speculative_n_max(&m.params.speculative));
        m.params.n_outputs_max         = std::max<int32_t>(1, limits.total);
        m.params.n_outputs_max_per_seq = std::max<int32_t>(1, limits.per_seq);

        m.init  = common_init_from_params(m.params);
        m.model = m.init->model();
        m.ctx   = m.init->context();
        if (m.model == nullptr || m.ctx == nullptr) {
            fprintf(stderr, "decision-oracle: cannot load the model\n");
            return 1;
        }
    }
    m.vocab = llama_model_get_vocab(m.model);

    json_writer w;
    w.begin_object();
    // llama-server fails to start when init() throws: no request is answered
    try {
        m.decision.init(m.model);
    } catch (...) {
        w.key("load");
        w.begin_object();
        write_refusal(w, classify(std::current_exception(), ""));
        w.end_object();
        w.key("cases");
        w.begin_array();
        w.end_array();
        w.end_object();
        w.out += '\n';
        write_file(output, w.out);
        return 0;
    }
    if (m.decision.type == COMMON_DECISION_TYPE_NONE) {
        fprintf(stderr, "decision-oracle: not a decision model\n");
        return 1;
    }
    write_facts(w, m);
    w.key("cases");
    w.begin_array();
    for (const auto & c : cases) {
        run_case(w, m, c);
    }
    w.end_array();
    w.end_object();
    w.out += '\n';
    write_file(output, w.out);
    return 0;
}

// a GGUF with the metadata of another and none of its tensors, with keys set or removed and token
// types changed: enough for --vocab-only, which reads the vocabulary and the metadata only
int cmd_variant(int argc, char ** argv) {
    std::string base;
    std::string output;
    std::vector<std::pair<std::string, std::string>> sets;
    std::vector<std::string> removes;
    std::vector<std::pair<int64_t, int32_t>> token_types;
    for (int i = 0; i + 1 < argc; i += 2) {
        const std::string arg = argv[i];
        const std::string val = argv[i + 1];
        const size_t eq = val.find('=');
        if (arg == "--base") {
            base = val;
        } else if (arg == "--output") {
            output = val;
        } else if (arg == "--set" && eq != std::string::npos) {
            sets.emplace_back(val.substr(0, eq), val.substr(eq + 1));
        } else if (arg == "--remove") {
            removes.push_back(val);
        } else if (arg == "--token-type" && eq != std::string::npos) {
            token_types.emplace_back(std::stoll(val.substr(0, eq)), (int32_t) std::stol(val.substr(eq + 1)));
        } else {
            return usage();
        }
    }
    if (base.empty() || output.empty() || argc % 2 != 0) {
        return usage();
    }
    gguf_init_params params = {/* no_alloc = */ true, /* ctx = */ nullptr};
    gguf_context * src = gguf_init_from_file(base.c_str(), params);
    if (src == nullptr) {
        fprintf(stderr, "decision-oracle: cannot read %s\n", base.c_str());
        return 1;
    }
    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    for (const auto & key : removes) {
        if (gguf_remove_key(dst, key.c_str()) < 0) {
            fprintf(stderr, "decision-oracle: no key %s\n", key.c_str());
            return 1;
        }
    }
    for (const auto & [key, typed] : sets) {
        if (string_starts_with(typed, "f32:")) {
            gguf_set_val_f32(dst, key.c_str(), std::stof(typed.substr(4)));
        } else if (string_starts_with(typed, "str:")) {
            gguf_set_val_str(dst, key.c_str(), typed.substr(4).c_str());
        } else {
            fprintf(stderr, "decision-oracle: a value is f32:<number> or str:<text>: %s\n", typed.c_str());
            return 1;
        }
    }
    if (!token_types.empty()) {
        const char * key = "tokenizer.ggml.token_type";
        const int64_t id = gguf_find_key(dst, key);
        if (id < 0 || gguf_get_arr_type(dst, id) != GGUF_TYPE_INT32) {
            fprintf(stderr, "decision-oracle: no int32 array %s\n", key);
            return 1;
        }
        const size_t n = gguf_get_arr_n(dst, id);
        const int32_t * data = (const int32_t *) gguf_get_arr_data(dst, id);
        std::vector<int32_t> types(data, data + n);
        for (const auto & [token, type] : token_types) {
            if (token < 0 || (size_t) token >= n) {
                fprintf(stderr, "decision-oracle: no token %lld\n", (long long) token);
                return 1;
            }
            types[token] = type;
        }
        gguf_set_arr_data(dst, key, GGUF_TYPE_INT32, types.data(), types.size());
    }
    const bool ok = gguf_write_to_file(dst, output.c_str(), false);
    gguf_free(dst);
    gguf_free(src);
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();
    if (argc < 2) {
        return usage();
    }
    try {
        const std::string cmd = argv[1];
        if (cmd == "run") {
            return cmd_run(argc - 2, argv + 2);
        }
        if (cmd == "variant") {
            return cmd_variant(argc - 2, argv + 2);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "decision-oracle: %s\n", e.what());
        return 1;
    }
    return usage();
}
