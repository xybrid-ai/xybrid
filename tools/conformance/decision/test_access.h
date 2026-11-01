// Test access to private members of llama.cpp's server_decision_context.
//
// The decision oracle observes intermediate values that upstream keeps private:
// the rendered prompt, the label codes, the effective temperature and the tables
// they come from. Upstream code is neither edited nor recompiled for this: the
// oracle links the server-context library as llama-server does, and this header
// takes pointers to the members below through explicit template instantiation,
// where C++ does not apply access checks ([temp.spec.general]: "The usual access
// checking rules do not apply to names in a declaration of an explicit
// instantiation"). Pointers to members only; nothing is copied or redefined, so
// a call runs the upstream function itself.
//
// Exposed, and nothing else:
//   render            the prompt text of one question, as fill_task() renders it
//   d1_labels         the label code text and token group of each option
//   get_temperature   the temperature format_answer() uses without media
//   temperatures      the "<arch>.decision.temperature.*" table read by init()
//   score_bias        the "<arch>.decision.score_bias.*" table read by init()
//   quoted_spellings  the special-token texts d1_break() breaks, longest first
//
// A signature change upstream fails to compile here, which is the point: update
// the declarations below together with the llama.cpp pin.
#pragma once

#include "server-decision.h"

#include <map>
#include <string>
#include <vector>

namespace decision_oracle {

template <typename Tag, typename Tag::type Member>
struct expose {
    friend typename Tag::type member(Tag) { return Member; }
};

struct render_tag {
    using type = std::string (server_decision_context::*)(
            const json &, const std::vector<server_decision_question> &, const server_decision_question &,
            size_t, size_t, bool) const;
    friend type member(render_tag);
};
template struct expose<render_tag, &server_decision_context::render>;

struct d1_labels_tag {
    using type = void (server_decision_context::*)(
            const server_decision_question &, std::vector<std::string> &, std::vector<llama_tokens> &) const;
    friend type member(d1_labels_tag);
};
template struct expose<d1_labels_tag, &server_decision_context::d1_labels>;

struct get_temperature_tag {
    using type = float (server_decision_context::*)(const server_decision_question &) const;
    friend type member(get_temperature_tag);
};
template struct expose<get_temperature_tag, &server_decision_context::get_temperature>;

struct temperatures_tag {
    using type = std::map<std::string, float> server_decision_context::*;
    friend type member(temperatures_tag);
};
template struct expose<temperatures_tag, &server_decision_context::temperatures>;

struct score_bias_tag {
    using type = std::map<size_t, std::vector<float>> server_decision_context::*;
    friend type member(score_bias_tag);
};
template struct expose<score_bias_tag, &server_decision_context::score_bias>;

struct quoted_spellings_tag {
    using type = std::vector<std::string> server_decision_context::*;
    friend type member(quoted_spellings_tag);
};
template struct expose<quoted_spellings_tag, &server_decision_context::quoted_spellings>;

inline std::string render(const server_decision_context & ctx, const json & state,
                          const std::vector<server_decision_question> & questions,
                          const server_decision_question & question, size_t variant, size_t n_images) {
    return (ctx.*member(render_tag{}))(state, questions, question, variant, n_images, false);
}

inline void d1_labels(const server_decision_context & ctx, const server_decision_question & question,
                      std::vector<std::string> & texts, std::vector<llama_tokens> & groups) {
    (ctx.*member(d1_labels_tag{}))(question, texts, groups);
}

inline float get_temperature(const server_decision_context & ctx, const server_decision_question & question) {
    return (ctx.*member(get_temperature_tag{}))(question);
}

inline const std::map<std::string, float> & temperatures(const server_decision_context & ctx) {
    return ctx.*member(temperatures_tag{});
}

inline const std::map<size_t, std::vector<float>> & score_bias(const server_decision_context & ctx) {
    return ctx.*member(score_bias_tag{});
}

inline const std::vector<std::string> & quoted_spellings(const server_decision_context & ctx) {
    return ctx.*member(quoted_spellings_tag{});
}

} // namespace decision_oracle
