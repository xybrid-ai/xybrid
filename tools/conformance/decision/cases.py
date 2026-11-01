"""The decision conformance case set: requests and xybrid's expectations.

oracle.py runs every case through the upstream oracle and the stock llama-server and writes
integration-tests/fixtures/decision/cases/<id>.json. A case here holds only what is decided by
hand: the request, what it covers, what xybrid must do, and for the answer-arithmetic cases
the synthetic scores. Everything upstream does with it is observed, never written here.

Stdlib only; imported by oracle.py.
"""

import json

# A request answered by every stage, read on the model.
MATCH = {"parse": "match", "prepare": "match", "evaluate": "reference", "finalize": "match"}
# The same with synthetic scores in place of the model's logits.
SYNTH = {"parse": "match", "prepare": "match", "evaluate": "given", "finalize": "match"}
STAGES = ("load", "parse", "prepare", "evaluate", "finalize")


def refuse(stage, kind, base=MATCH):
    """xybrid refuses at `stage`: earlier stages as in `base`, later ones not reached."""
    out = {}
    for s in STAGES:
        if s not in base and s != stage:
            continue
        if STAGES.index(s) < STAGES.index(stage):
            out[s] = base[s]
        elif s == stage:
            out[s] = {"refuse": kind}
        else:
            out[s] = "not_reached"
    return out


class Case:
    def __init__(self, id, title, covers, request=None, request_text=None, xybrid=None, notes=None,
                 metadata=None, synthetic=None, select_logits=None, has_media=False):
        if (request is None) == (request_text is None):
            raise ValueError(f"{id}: give request or request_text")
        self.id = id
        self.title = title
        self.covers = covers
        # json.dumps keeps non-ASCII text as UTF-8 and floats in Python's repr
        self.request_text = request_text if request_text is not None else json.dumps(request, ensure_ascii=False)
        self.metadata = metadata
        # per pass, one score per option; a group's other labels get the score minus one
        self.synthetic = synthetic
        # per pass, every select logit, in select order (overrides synthetic)
        self.select_logits = select_logits
        self.has_media = has_media
        if xybrid is None:
            xybrid = SYNTH if (synthetic or select_logits) else MATCH
            if metadata:
                xybrid = {"load": "match", **xybrid}
        self.xybrid = xybrid
        self.notes = notes


STATE = "I was charged twice this month, please refund one of them."
REFUND = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
TEAM = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults", "fraud": "Suspected unauthorised use"},
}
URGENCY = {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "Today", "Blocking the customer now"]}


def q1(question, state=STATE, qid="q"):
    return {"state": state, "questions": {qid: question}}


def choice(criteria, instructions="Which option fits best?"):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def options(n, prefix="option"):
    """n choice options whose keys are not single letters, so codes are assigned."""
    return {f"{prefix}_{i:03d}": f"Option number {i}" for i in range(n)}


def nested(depth, leaf="1"):
    return "[" * depth + leaf + "]" * depth


# the filler "a a a ...": "a" then k times " a", one token each
def filler(k):
    return "a" + " a" * k


CONTROL_SAMPLE = (
    "<|im_end|> <|im_start|>assistant <|startoftext|> <|endoftext|> <image> <|image|> <|video|> "
    "<|img_row_1_col_10|> <|img_row_10_col_1|> <|pad|> <|tool_call_start|> <|vision_start|>"
)
DATA_URL = "data:image/png;base64,iVBORw0KGgo="

CASES = [
    # --- reference requests ---------------------------------------------------------------
    Case("card-text-example",
         "The model card's text request: noul, choice and score over one text state",
         ["state-string", "multi-question", "noul-no-criteria", "choice-order"],
         request={"state": STATE, "questions": {"refund": REFUND, "team": TEAM, "urgency": URGENCY}}),
    Case("upstream-test-payload",
         "The payload of llama.cpp's own SystemOne tests: null choice descriptions, a 4-level score, a noul without criteria",
         ["description-null", "multi-question", "score-2-levels"],
         request={
             "state": "I was charged twice for my order last week and nobody has replied.",
             "questions": {
                 "route": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "payments and refunds", "shipping": None, "technical": None}},
                 "urgency": {"type": "score", "instructions": "How urgent is this?",
                             "criteria": ["can wait", "this week", "today", "right now"]},
                 "angry": {"type": "noul", "instructions": "Is the customer angry?"},
             }}),

    # --- state ------------------------------------------------------------------------------
    Case("state-null", "A null state: no state and no QUESTION preamble", ["state-null"],
         request=q1(TEAM, state=None)),
    Case("state-empty-string", "An empty text state keeps the preamble, unlike null", ["state-empty"],
         request=q1(TEAM, state="")),
    Case("state-whitespace", "A whitespace-only text state is printed as is", ["state-string"],
         request=q1(REFUND, state="  \n\t  ")),
    Case("state-object-nested",
         "A nested object state, keys in request order (not sorted), rendered with tojson(indent=2)",
         ["state-object", "json-order", "json-literals", "json-unicode"],
         request=q1(REFUND, state={
             "zeta": "last letter first",
             "ticket": {"id": 4471, "items": [{"sku": "B-2", "qty": 2, "price": 19.99}, {"sku": "A-1", "qty": 1, "price": 5.0}],
                        "customer": {"name": "Zoë Ångström", "tier": "pro"}},
             "alpha": [True, False, None],
             "empty_object": {},
             "empty_array": [],
             "note": "Charged twice.",
         })),
    Case("state-array", "An array state of plain values", ["state-array", "json-literals"],
         request=q1(REFUND, state=["first message", 2, 3.5, True, None, {"k": "v"}, []])),
    Case("state-chat-messages",
         "A state of chat messages without media: kept as JSON, text parts in place",
         ["state-messages", "state-array"],
         request=q1(REFUND, state=[
             {"role": "user", "content": "I was charged twice."},
             {"role": "assistant", "content": [{"type": "text", "text": "Sorry to hear that."}]},
             {"role": "user", "content": [{"type": "text", "text": "Please refund one."}, {"type": "text", "text": "Thanks."}]},
         ])),
    Case("state-wrapped-messages",
         "An object state with a messages array: the messages are rebuilt, other keys kept",
         ["state-messages", "state-object"],
         request=q1(REFUND, state={"channel": "email", "messages": [{"role": "user", "content": [{"type": "text", "text": "Refund please."}]}], "priority": 2})),
    Case("state-numbers",
         "Number formatting in tojson: float repr, exponents, -0.0, integers, int/float identity",
         ["json-float", "json-int"],
         request_text='{"state": {"one": 1, "one_float": 1.0, "neg_zero": -0.0, "tenth": 0.1, "tiny": 1e-7, "small": 0.0001, '
                      '"e15": 1e15, "e16": 1e16, "pi": 3.141592653589793, "third": 0.3333333333333333, "max": 1.7976931348623157e308, '
                      '"min_sub": 5e-324, "exp_int": 1E2, "neg": -12.5, "i64_max": 9223372036854775807, '
                      '"i64_min": -9223372036854775808, "above_u64": 18446744073709551616, "list": [0, -0, 0.0, 2.50]}, '
                      '"questions": {"q": {"type": "noul", "instructions": {"threshold": 0.75, "count": 3, "ratio": 2.0, "big": 1e300}}}}'),
    Case("state-u64-above-i64",
         "Integers above i64::MAX and below 2^64: upstream renders them wrapped to negative numbers",
         ["json-int", "divergence"],
         request_text='{"state": {"u64_max": 18446744073709551615, "two_63": 9223372036854775808}, '
                      '"questions": {"q": {"type": "noul", "instructions": "Is the account flagged?"}}}',
         xybrid=refuse("prepare", "number_out_of_range"),
         notes="The Jinja engine reads every JSON integer with get<int64_t>(), so 18446744073709551615 prints as -1 and "
               "9223372036854775808 as -9223372036854775808. xybrid must not reproduce a wrong number; it refuses."),
    Case("state-number-overflow",
         "A number beyond f64: upstream's JSON parser refuses the request",
         ["json-float", "refusal-parse"],
         request_text='{"state": {"huge": 1e400}, "questions": {"q": {"type": "noul", "instructions": "x"}}}',
         xybrid=refuse("parse", "number_out_of_range")),
    Case("state-string-escapes",
         "Quotes, backslashes, control characters and non-ASCII text: raw in a text state, escaped by tojson in a JSON one",
         ["json-escapes", "json-unicode", "state-string"],
         request={"state": "Quote \" backslash \\ newline\nend tab\tend bell\u0007 nul\u0000 é 中文 😀   ​",
                  "questions": {"q": {"type": "noul", "instructions": {"text": "Quote \" backslash \\ newline\nend tab\tend bell\u0007 nul\u0000 é 中文 😀   ​ \u007f"}}}}),
    Case("state-unicode-escapes",
         "\\u escapes in the request text, a surrogate pair among them, decode to the same UTF-8",
         ["json-unicode", "json-escapes"],
         request_text='{"state": "caf\\u00e9 \\ud83d\\ude00 \\u4e2d\\u6587 \\/ \\b\\f", "questions": {"q": {"type": "noul", '
                      '"instructions": {"caf\\u00e9": "\\ud83d\\ude00"}}}}'),

    # --- instructions ------------------------------------------------------------------------
    Case("instructions-object", "Object instructions: compact tojson, keys in request order", ["instructions-structured", "json-order"],
         request=q1({"type": "noul", "instructions": {"task": "Is this a refund request?", "labels": ["yes", "no"], "strict": True, "weight": 0.5}})),
    Case("instructions-array", "Array instructions: compact tojson", ["instructions-structured"],
         request=q1({"type": "noul", "instructions": ["Is", "this", "a refund", 1, None]})),
    Case("instructions-scalars",
         "Number, boolean and empty-string instructions are accepted: tojson for the first ones, nothing for the last",
         ["instructions-structured", "json-literals", "multi-question"],
         request={"state": STATE, "questions": {
             "int": {"type": "noul", "instructions": 42},
             "float": {"type": "noul", "instructions": 0.5},
             "false": {"type": "noul", "instructions": False},
             "true": {"type": "noul", "instructions": True},
             "empty": {"type": "noul", "instructions": ""},
         }}),
    Case("instructions-null", "Null instructions are refused like missing ones", ["refusal-parse"],
         request=q1({"type": "noul", "instructions": None}), xybrid=refuse("parse", "missing_instructions")),
    Case("instructions-missing", "Missing instructions are refused", ["refusal-parse"],
         request=q1({"type": "noul"}), xybrid=refuse("parse", "missing_instructions")),

    # --- choice -----------------------------------------------------------------------------
    Case("choice-null-descriptions", "Null descriptions print the key, underscores as spaces", ["description-null"],
         request=q1(choice({"needs_human": None, "auto_reply": None, "spam": None, "two__underscores_": None}))),
    Case("choice-falsy-descriptions",
         "Every falsy description (empty string, false, 0, 0.0, [], {}) prints the key; truthy ones print themselves",
         ["description-falsy", "description-structured"],
         request=q1(choice({"empty_string": "", "false_value": False, "zero": 0, "zero_float": 0.0, "empty_list": [], "empty_object": {},
                            "null_value": None, "truthy": "kept", "true_value": True, "space": " "}))),
    Case("choice-structured-descriptions", "Object, array and number descriptions print with compact tojson",
         ["description-structured", "json-float"],
         request=q1(choice({"billing": {"team": "billing", "sla_hours": 24}, "technical": ["app", "site", 2], "fraud": 3.5,
                            "other": -1, "nested": {"a": {"b": [1.0, None]}}}))),
    Case("choice-letter-keys", "Keys that are all single ASCII letters are the codes, in request order", ["choice-letter-keys"],
         request=q1(choice({"B": "Billing", "A": "Account", "C": "Cancellation"}))),
    Case("choice-lowercase-keys", "Lowercase single-letter keys are codes too", ["choice-letter-keys", "choice-lowercase-keys"],
         request=q1(choice({"y": "Yes, refund", "n": "No refund", "m": "Maybe later"}))),
    Case("choice-mixed-case-keys", "Mixed-case single-letter keys", ["choice-letter-keys", "choice-lowercase-keys"],
         request=q1(choice({"a": None, "B": None, "c": None, "D": None}))),
    Case("choice-52-letter-keys", "All 52 ASCII letters as keys: more than 26 options, and the keys are still the codes",
         ["choice-letter-keys", "choice-lowercase-keys", "choice-27"],
         request=q1(choice({c: f"Letter {c}" for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}))),
    Case("choice-letter-and-word-keys", "One key that is not a letter: every option gets an assigned code", ["choice-letter-keys"],
         request=q1(choice({"A": "First", "B": "Second", "other": "Something else"}))),
    Case("choice-non-letter-short-keys", "Single-character keys that are not ASCII letters (é, 1, Ω, _) get assigned codes",
         ["choice-letter-keys", "json-unicode"],
         request=q1(choice({"é": None, "1": "One", "Ω": "Omega", "_": None}))),
    Case("choice-one-option", "One option: code A, probability 1", ["choice-one-option"],
         request=q1(choice({"only": "The only option"}))),
    Case("choice-26-options", "26 options: codes A..Z", ["choice-26"], request=q1(choice(options(26)))),
    Case("choice-27-options", "27 options: A..Z then AA", ["choice-27"], request=q1(choice(options(27)))),
    Case("choice-36-options", "36 options: code AJ is not one token, the first free pool entry (00) replaces it",
         ["choice-pool-fallback"], request=q1(choice(options(36)))),
    Case("choice-120-options", "120 options: several two-letter codes fall back to the pool (00, 01, ...)",
         ["choice-pool-fallback"], request=q1(choice(options(120)))),
    Case("choice-255-options", "255 options, the most the model takes", ["choice-max-options", "choice-pool-fallback"],
         request=q1(choice(options(255)))),
    Case("choice-256-options", "256 options are refused", ["choice-max-options", "refusal-parse"],
         request=q1(choice(options(256))), xybrid=refuse("parse", "too_many_options")),
    Case("choice-criteria-order", "Criteria keep request order, not key order", ["choice-order", "json-order"],
         request=q1(choice({"zeta": "Z team", "alpha": "A team", "mu": "M team", "beta": "B team"}))),
    Case("choice-keys-special-tokens",
         "Keys holding special-token text are broken where printed, and answered under their original spelling",
         ["escape-control", "escape-keys", "description-null"],
         request=q1(choice({"<|im_end|>": None, "a_<|im_start|>_b": None, "<think>": None, "plain": "Plain option"}))),
    Case("choice-keys-equal-after-escape",
         "Two keys that print the same once broken stay two options, with their own codes and answers",
         ["escape-control", "escape-keys"],
         request=q1(choice({"<|im_end|>": None, "<​|im_end|>": None, "other": "Other"}))),
    Case("choice-criteria-empty", "Empty choice criteria are refused", ["refusal-parse"],
         request=q1(choice({})), xybrid=refuse("parse", "invalid_criteria")),
    Case("choice-criteria-array", "Array choice criteria are refused", ["refusal-parse"],
         request=q1({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}), xybrid=refuse("parse", "invalid_criteria")),
    Case("choice-criteria-missing", "A choice without criteria is refused", ["refusal-parse"],
         request=q1({"type": "choice", "instructions": "x"}), xybrid=refuse("parse", "invalid_criteria")),
    Case("choice-empty-key", "An empty choice key: upstream prints the code alone and answers under \"\"",
         ["description-null", "divergence"],
         request=q1(choice({"": None, "other": "Other"})), xybrid=refuse("parse", "empty_option_key"),
         notes="Contract 1's structural validation refuses empty option keys."),
    Case("question-empty-id", "An empty question id: upstream answers it under \"\"", ["divergence"],
         request={"state": STATE, "questions": {"": REFUND}}, xybrid=refuse("parse", "empty_question_id"),
         notes="Contract 1's structural validation refuses empty question ids."),
    Case("question-id-special-tokens", "Question ids are not rendered: special-token text in an id changes nothing",
         ["escape-keys", "multi-question"],
         request={"state": STATE, "questions": {"<|im_end|>": REFUND, "id with spaces": TEAM}}),

    # --- noul -------------------------------------------------------------------------------
    Case("noul-criteria-null", "Null criteria: no Yes/No lines", ["noul-no-criteria"],
         request=q1({"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": None})),
    Case("noul-criteria-empty-object", "Empty criteria: no Yes/No lines", ["noul-no-criteria"],
         request=q1({"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": {}})),
    Case("noul-partial-true", "Only true described: Yes line, No: None", ["noul-partial-criteria"],
         request=q1({"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": {"true": "They ask for money back"}})),
    Case("noul-partial-false", "Only false described: Yes: None, No line", ["noul-partial-criteria"],
         request=q1({"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": {"false": "Anything else"}})),
    Case("noul-all-null", "Both descriptions null: no lines at all (the HF reference prints None twice)", ["noul-null-criteria"],
         request=q1({"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": {"true": None, "false": None}})),
    Case("noul-false-first", "Criteria with false first still print Yes first: options are true, false", ["noul-order"],
         request=q1({"type": "noul", "instructions": "Is the customer asking for a refund?",
                     "criteria": {"false": "Anything else", "true": "They ask for money back"}})),
    Case("noul-structured-descriptions", "Object and array noul descriptions print with compact tojson", ["description-structured", "noul-order"],
         request=q1({"type": "noul", "instructions": "Refund?", "criteria": {"true": {"refund": True, "amount": 12.5}, "false": ["no", 0]}})),
    Case("noul-empty-string-description", "An empty-string description is not null: it prints, and the other prints None",
         ["noul-partial-criteria", "description-falsy"],
         request=q1({"type": "noul", "instructions": "Refund?", "criteria": {"true": ""}})),
    Case("noul-yes-no-aliases", "yes/no keys are not read for lfm3-d1: upstream ignores them and prints no lines",
         ["noul-partial-criteria", "divergence"],
         request=q1({"type": "noul", "instructions": "Refund?", "criteria": {"yes": "They ask for money back", "no": "Anything else"}}),
         xybrid=refuse("parse", "unknown_field"),
         notes="Upstream drops descriptions given under yes/no without a word; xybrid refuses criteria keys other than true and false."),
    Case("noul-extra-criteria-key", "A criteria key other than true/false is ignored upstream", ["noul-partial-criteria", "divergence"],
         request=q1({"type": "noul", "instructions": "Refund?", "criteria": {"true": "Money back", "maybe": "Unclear"}}),
         xybrid=refuse("parse", "unknown_field")),
    Case("noul-criteria-array", "Array noul criteria are refused", ["refusal-parse"],
         request=q1({"type": "noul", "instructions": "Refund?", "criteria": ["yes", "no"]}), xybrid=refuse("parse", "invalid_criteria")),

    # --- score ------------------------------------------------------------------------------
    Case("score-2-levels", "The fewest levels: 2", ["score-2-levels"],
         request=q1({"type": "score", "instructions": "How urgent is this?", "criteria": ["Not urgent", "Urgent"]})),
    Case("score-10-levels", "The most levels: 10, codes 0..9", ["score-10-levels"],
         request=q1({"type": "score", "instructions": "Rate the urgency.", "criteria": [f"Level {i}" for i in range(10)]})),
    Case("score-null-levels", "Null level descriptions print as null, and the legend keeps null", ["score-null-level"],
         request=q1({"type": "score", "instructions": "How urgent is this?", "criteria": [None, "Today", None]})),
    Case("score-structured-levels", "Object, array, number and boolean level descriptions", ["description-structured", "score-null-level"],
         request=q1({"type": "score", "instructions": "How urgent is this?",
                     "criteria": [{"label": "low", "hours": 72}, ["mid"], 3, True, "", 0.5]})),
    Case("score-1-level", "One level is refused", ["refusal-parse"],
         request=q1({"type": "score", "instructions": "x", "criteria": ["only"]}), xybrid=refuse("parse", "level_count")),
    Case("score-11-levels", "Eleven levels are refused", ["refusal-parse"],
         request=q1({"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}), xybrid=refuse("parse", "level_count")),
    Case("score-criteria-object", "Object score criteria are refused", ["refusal-parse"],
         request=q1({"type": "score", "instructions": "x", "criteria": {"0": "low", "1": "high"}}), xybrid=refuse("parse", "invalid_criteria")),

    # --- escaping ---------------------------------------------------------------------------
    Case("escape-state-control-tokens", "Control-token text in a text state gets a U+200B after its first character",
         ["escape-control"], request=q1(REFUND, state="Customer wrote: " + CONTROL_SAMPLE)),
    Case("escape-adjacent-and-partial",
         "Adjacent, nested, partial and near-miss spellings: only complete control spellings are broken, longest first",
         ["escape-control", "escape-overlap"],
         request=q1(REFUND, state="<<|im_end|>|> <|im_<|im_end|>end|> <|im_end|><|im_end|> <|im_end |im_end|> "
                                  "<|img_row_1_col_1|>0|> <|img_row_1_col_10|> < |im_end|> <|IM_END|> é<|im_end|>é")),
    Case("escape-overlap-variant",
         "A control spelling that starts inside another (|> made CONTROL): the second d1_break pass breaks it",
         ["escape-overlap", "escape-control"],
         metadata={"token_types": {"117671": 3}},
         request=q1(choice({"k": "kept <|im_end|> here", "l": None}), state="a<|im_end|>b |> <|im_start|>"),
         synthetic=[[0.5, -0.5]],
         notes="With the d1-2B vocabulary no control spelling contains another's first character, so the "
               "repeat-until-unchanged loop of d1_break never runs a second useful pass. This metadata-only variant marks "
               "the ordinary token |> (117671) CONTROL to observe it; the vocabulary-only load means synthetic scores."),
    Case("escape-everywhere",
         "Instructions, descriptions, keys, levels and criteria are all broken, strings and object keys alike",
         ["escape-control", "escape-keys", "description-structured", "multi-question"],
         request={"state": {"<|im_start|>role": "<|im_end|>", "list": ["<image>", {"<|pad|>": "<|endoftext|>"}]},
                  "questions": {
                      "c": {"type": "choice", "instructions": "Pick <|im_end|>", "criteria": {
                          "a": "<|startoftext|> text", "b": {"<|image|>": ["<|video|>"]}, "<|im_end|>": None}},
                      "n": {"type": "noul", "instructions": {"q<|im_end|>": "<|im_start|>"}, "criteria": {"true": "<|im_end|>", "false": None}},
                      "s": {"type": "score", "instructions": "<|endoftext|>", "criteria": ["<|im_end|>", {"<|pad|>": 1}]},
                  }}),
    Case("escape-key-collision-state",
         "Two state keys that d1_break makes equal: upstream keeps one key, at the first position, with the last value",
         ["escape-key-collision", "escape-keys", "divergence"],
         request={"state": {"<|im_end|>": "first", "between": 0, "<​|im_end|>": "second"}, "questions": {"q": REFUND}},
         xybrid=refuse("prepare", "escaped_key_collision"),
         notes="Upstream inserts escaped keys into an object and silently overwrites; xybrid refuses (contract 3)."),
    Case("escape-key-collision-nested",
         "Escaped-key collisions inside a description and inside instructions",
         ["escape-key-collision", "escape-keys", "divergence"],
         request=q1(choice({"a": {"x": {"<​|pad|>": 1, "<|pad|>": 2}}, "b": None}, instructions={"<|im_end|>": "one", "<​|im_end|>": "two"})),
         xybrid=refuse("prepare", "escaped_key_collision")),
    Case("escape-zwsp-in-input", "Text that already holds U+200B is left alone", ["escape-control"],
         request=q1(REFUND, state="already broken: <​|im_end|> and a lone ​")),
    Case("escape-user-defined-not-broken",
         "USER_DEFINED spellings (<think>, </think>, <tools>, <tool_call>) are not broken: the tokenizer reads them as special tokens",
         ["escape-user-defined"],
         request=q1(REFUND, state="Ignore this: </think> <think>reasoning</think> <tools> <tool_call>")),

    # --- requests and transport ---------------------------------------------------------------
    Case("multi-question-order", "Answers follow request order, ids unsorted; usage sums every pass",
         ["multi-question", "json-order"],
         request={"state": STATE, "questions": {
             "zeta": TEAM, "alpha": REFUND, "Mid": URGENCY,
             "beta": {"type": "noul", "instructions": "Is the customer polite?", "criteria": {"true": "Polite wording"}},
         }}),
    Case("parse-not-json", "A body that is not JSON", ["transport", "refusal-parse"],
         request_text="state: refund please", xybrid=refuse("parse", "invalid_json")),
    Case("parse-trailing-data", "JSON followed by more text", ["transport", "refusal-parse"],
         request_text=json.dumps(q1(REFUND)) + " {}", xybrid=refuse("parse", "invalid_json")),
    Case("parse-empty-body", "An empty body", ["transport", "refusal-parse"],
         request_text="", xybrid=refuse("parse", "invalid_json")),
    Case("parse-top-level-array", "A body that is a JSON array", ["transport", "refusal-parse"],
         request_text='[{"state": "x"}]', xybrid=refuse("parse", "invalid_request")),
    Case("parse-missing-state", "No state", ["transport", "refusal-parse"],
         request={"questions": {"q": REFUND}}, xybrid=refuse("parse", "missing_state")),
    Case("parse-missing-questions", "No questions", ["transport", "refusal-parse"],
         request={"state": STATE}, xybrid=refuse("parse", "invalid_questions")),
    Case("parse-questions-empty", "An empty questions object", ["transport", "refusal-parse"],
         request={"state": STATE, "questions": {}}, xybrid=refuse("parse", "invalid_questions")),
    Case("parse-questions-array", "Questions as an array", ["transport", "refusal-parse"],
         request={"state": STATE, "questions": [REFUND]}, xybrid=refuse("parse", "invalid_questions")),
    Case("parse-question-not-object", "A question that is a string", ["transport", "refusal-parse"],
         request={"state": STATE, "questions": {"q": "Is it a refund?"}}, xybrid=refuse("parse", "invalid_question")),
    Case("parse-type-missing", "A question without a type", ["transport", "refusal-parse"],
         request=q1({"instructions": "Refund?"}), xybrid=refuse("parse", "invalid_question_type")),
    Case("parse-type-not-string", "A type that is a number reads as missing", ["transport", "refusal-parse"],
         request=q1({"type": 1, "instructions": "Refund?"}), xybrid=refuse("parse", "invalid_question_type")),
    Case("parse-type-unknown", "An unknown type", ["transport", "refusal-parse"],
         request=q1({"type": "multiple_choice", "instructions": "Refund?"}), xybrid=refuse("parse", "invalid_question_type")),
    Case("parse-duplicate-question-ids", "The same question id twice: upstream keeps the last question at the first position",
         ["transport", "divergence"],
         request_text='{"state": "' + STATE + '", "questions": {"q": {"type": "noul", "instructions": "First?"}, '
                      '"other": {"type": "noul", "instructions": "Other?"}, "q": {"type": "noul", "instructions": "Second?"}}}',
         xybrid=refuse("parse", "duplicate_key"),
         notes="OrderedJson refuses duplicate keys (contract 1); upstream's parser keeps the last value."),
    Case("parse-duplicate-criteria-keys", "The same choice key twice: upstream keeps the last description",
         ["transport", "divergence"],
         request_text='{"state": "' + STATE + '", "questions": {"q": {"type": "choice", "instructions": "Team?", '
                      '"criteria": {"billing": "First", "technical": "Tech", "billing": "Second"}}}}',
         xybrid=refuse("parse", "duplicate_key")),
    Case("parse-duplicate-state-keys", "The same key twice inside the state: upstream renders the last value",
         ["transport", "divergence", "state-object"],
         request_text='{"state": {"amount": 10, "currency": "EUR", "amount": 20}, "questions": {"q": {"type": "noul", "instructions": "Refund?"}}}',
         xybrid=refuse("parse", "duplicate_key")),
    Case("parse-unknown-top-level-field", "An unknown top-level field: upstream ignores it", ["transport", "divergence"],
         request={"state": STATE, "questions": {"q": REFUND}, "temperature": 0.1},
         xybrid=refuse("parse", "unknown_field"),
         notes="Contract 4: a top-level model field is ignored, other unknown fields are refused."),
    Case("parse-model-field", "A top-level model field is ignored by both", ["transport"],
         request={"model": "d1-2B", "state": STATE, "questions": {"q": REFUND}}),
    Case("parse-question-unknown-field", "multiple: true (not served by llama.cpp v0.8.0) is ignored upstream",
         ["transport", "divergence"],
         request=q1({"type": "choice", "instructions": "Which teams?", "multiple": True,
                     "criteria": {"billing": "Charges", "technical": "Faults"}}),
         xybrid=refuse("parse", "unknown_field"),
         notes="Upstream answers a single choice to a request for several without a word."),
    Case("parse-files-data-url", "An image data URL in files: no mmproj, so the server answers 501", ["media", "refusal-parse"],
         request={"state": STATE, "files": [DATA_URL], "questions": {"q": REFUND}}, xybrid=refuse("parse", "media_unsupported")),
    Case("parse-images-http-url", "An http URL in images is refused", ["media", "refusal-parse"],
         request={"state": STATE, "images": ["http://example.com/cat.png"], "questions": {"q": REFUND}},
         xybrid=refuse("parse", "media_unsupported")),
    Case("parse-videos", "videos is refused", ["media", "refusal-parse"],
         request={"state": STATE, "videos": ["data:video/mp4;base64,AAAA"], "questions": {"q": REFUND}},
         xybrid=refuse("parse", "media_unsupported")),
    Case("parse-message-image-part", "An image_url part in the state's messages becomes a file: 501", ["media", "state-messages", "refusal-parse"],
         request={"state": [{"role": "user", "content": [{"type": "text", "text": "See photo"},
                                                         {"type": "image_url", "image_url": {"url": DATA_URL}}]}],
                  "questions": {"q": REFUND}},
         xybrid=refuse("parse", "media_unsupported")),
    Case("parse-files-empty", "An empty files array: upstream has nothing to load and answers", ["media", "divergence"],
         request={"state": STATE, "files": [], "questions": {"q": REFUND}}, xybrid=refuse("parse", "media_unsupported"),
         notes="Contract 4 refuses the files field in v1, empty or not."),
    Case("parse-files-null", "A null files field: upstream ignores it", ["media", "divergence"],
         request={"state": STATE, "files": None, "questions": {"q": REFUND}}, xybrid=refuse("parse", "media_unsupported"),
         notes="Contract 4 refuses the files field in v1, null or not."),
    Case("parse-spoken-instructions", "Audio instructions need an audio model", ["media", "refusal-parse"],
         request=q1({"type": "noul", "instructions": {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}}),
         xybrid=refuse("parse", "spoken_instructions")),

    # --- depth and budget ---------------------------------------------------------------------
    Case("depth-instructions-at-limit", "126 nested arrays around a value in the instructions: the deepest upstream renders",
         ["json-depth", "instructions-structured"],
         request_text='{"state": null, "questions": {"q": {"type": "noul", "instructions": %s}}}' % nested(126)),
    Case("depth-instructions-over-limit", "127 nested arrays in the instructions are refused", ["json-depth", "refusal-prepare"],
         request_text='{"state": null, "questions": {"q": {"type": "noul", "instructions": %s}}}' % nested(127),
         xybrid=refuse("prepare", "json_too_deep")),
    Case("depth-state-empty-at-limit", "127 nested empty arrays in the state: an empty innermost array is one level shallower",
         ["json-depth", "state-array"],
         request_text='{"state": %s, "questions": {"q": {"type": "noul", "instructions": "x"}}}' % nested(127, leaf="")),
    Case("depth-state-empty-over-limit", "128 nested empty arrays in the state are refused", ["json-depth", "refusal-prepare"],
         request_text='{"state": %s, "questions": {"q": {"type": "noul", "instructions": "x"}}}' % nested(128, leaf=""),
         xybrid=refuse("prepare", "json_too_deep")),
    Case("depth-description-at-limit", "124 nested arrays in a choice description: descriptions sit two levels deeper",
         ["json-depth", "description-structured"],
         request_text='{"state": null, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": %s, "b": null}}}}' % nested(124)),
    Case("depth-description-over-limit", "125 nested arrays in a choice description are refused", ["json-depth", "refusal-prepare"],
         request_text='{"state": null, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": %s, "b": null}}}}' % nested(125),
         xybrid=refuse("prepare", "json_too_deep")),
    Case("budget-at-limit", "A prompt of exactly 4096 tokens, the budget of the settings, is evaluated", ["budget"],
         request=q1({"type": "noul", "instructions": "x"}, state=filler(4072))),
    Case("budget-over-limit", "A prompt of 4097 tokens is refused", ["budget", "refusal-evaluate"],
         request=q1({"type": "noul", "instructions": "x"}, state=filler(4073)), xybrid=refuse("prepare", "budget_exceeded"),
         notes="Upstream refuses when the slot reads the prompt; xybrid checks budgets before any native call."),
    Case("budget-second-question-over", "One question over budget refuses the whole request, after the first was evaluated",
         ["budget", "multi-question", "refusal-evaluate"],
         request={"state": filler(4060), "questions": {"short": {"type": "noul", "instructions": "x"},
                                                        "long": choice({"a": "First", "b": "Second", "c": "Third"})}},
         xybrid=refuse("prepare", "budget_exceeded")),

    # --- answer arithmetic (synthetic scores) ---------------------------------------------------
    Case("math-reference-choice", "Choice: probabilities exp((s - max) / T) in f64; confidence is max p", ["math-reference"],
         request=q1(choice({"a": "A", "b": "B", "c": "C"})), synthetic=[[4.0, 1.5, -2.0]]),
    Case("math-reference-score", "Score: the median level, confidence max p", ["math-reference"],
         request=q1({"type": "score", "instructions": "x", "criteria": ["low", "mid", "high"]}), synthetic=[[0.3, 2.0, 1.1]]),
    Case("math-reference-noul", "Noul: p(true), no confidence", ["math-reference"],
         request=q1({"type": "noul", "instructions": "x"}), synthetic=[[2.5, 0.25]]),
    Case("math-median-b3", "Logits [0, 1e-8]: the f64 distribution puts the median at level 1; rounding to f32 first would give 0",
         ["math-median-b3"],
         request=q1({"type": "score", "instructions": "x", "criteria": ["no", "yes"]}), synthetic=[[0.0, 1e-8]]),
    Case("math-median-exact-half", "Equal logits: cumulative mass reaches exactly 0.5 at level 0, the median", ["math-median-b3", "math-tie"],
         request=q1({"type": "score", "instructions": "x", "criteria": ["no", "yes"]}), synthetic=[[1.0, 1.0]]),
    Case("math-median-not-mode", "The median level differs from the mode; confidence is the mode's p, not the median's",
         ["math-median-not-mode"],
         request=q1({"type": "score", "instructions": "x", "criteria": ["0", "1", "2", "3", "4"]}), synthetic=[[2.0, 1.4, -3.0, -3.0, 2.3]]),
    Case("math-choice-tie", "Tied maxima: the first one wins", ["math-tie"],
         request=q1(choice({"a": "A", "b": "B", "c": "C", "d": "D"})), synthetic=[[1.0, 3.0, 3.0, 2.0]]),
    Case("math-noul-tie", "Equal yes and no: 0.5", ["math-tie"],
         request=q1({"type": "noul", "instructions": "x"}), synthetic=[[0.75, 0.75]]),
    Case("math-underflow", "A spread wide enough that exp underflows to exactly 0", ["math-underflow"],
         request={"state": STATE, "questions": {
             "c": choice({"a": "A", "b": "B", "c": "C"}),
             "s": {"type": "score", "instructions": "x", "criteria": ["0", "1", "2", "3"]}}},
         synthetic=[[0.0, -800.0, -1600.0], [-1000.0, 30.0, -2000.0, 29.0]]),
    Case("math-neg-inf-some", "-inf scores next to finite ones get probability 0", ["math-neg-inf"],
         request={"state": STATE, "questions": {
             "c": choice({"a": "A", "b": "B", "c": "C"}),
             "n": {"type": "noul", "instructions": "x"}}},
         synthetic=[[1.0, "-inf", 0.5], ["-inf", 3.0]]),
    Case("math-neg-inf-all", "Every score -inf: upstream answers NaN probabilities", ["math-neg-inf", "math-non-finite", "divergence"],
         request=q1(choice({"a": "A", "b": "B"})), synthetic=[["-inf", "-inf"]],
         xybrid=refuse("finalize", "non_finite_scores", SYNTH),
         notes="Contract 3: -inf is allowed only if one offered score is finite."),
    Case("math-nan", "A NaN score: upstream refuses", ["math-non-finite", "refusal-finalize"],
         request=q1(choice({"a": "A", "b": "B"})), synthetic=[["nan", 1.0]], xybrid=refuse("finalize", "non_finite_scores", SYNTH)),
    Case("math-pos-inf", "A +inf score: upstream refuses", ["math-non-finite", "refusal-finalize"],
         request=q1({"type": "score", "instructions": "x", "criteria": ["0", "1"]}), synthetic=[[0.0, "inf"]],
         xybrid=refuse("finalize", "non_finite_scores", SYNTH)),
    Case("math-nan-alias", "NaN in a group's second label: upstream's max skips it and answers", ["math-non-finite", "math-alias-max", "divergence"],
         request=q1(choice({"a": "A", "b": "B"})), select_logits=[[1.0, "nan", 0.5, -0.5]],
         xybrid=refuse("finalize", "non_finite_scores", SYNTH),
         notes="std::max_element ignores a NaN that is not first in its group, so upstream's result depends on the label "
               "order. xybrid refuses any NaN among the select logits."),
    Case("math-nan-primary", "NaN as a group's first label: upstream's max keeps it, then refuses", ["math-non-finite", "math-alias-max"],
         request=q1(choice({"a": "A", "b": "B"})), select_logits=[["nan", 1.0, 0.5, -0.5]],
         xybrid=refuse("finalize", "non_finite_scores", SYNTH)),
    Case("math-pos-inf-alias", "+inf in a group's second label becomes the group's score: refused", ["math-non-finite", "math-alias-max"],
         request=q1(choice({"a": "A", "b": "B"})), select_logits=[[1.0, "inf", 0.5, -0.5]],
         xybrid=refuse("finalize", "non_finite_scores", SYNTH)),
    Case("math-alias-max", "A group's score is its largest logit, here the space-prefixed alias", ["math-alias-max"],
         request={"state": STATE, "questions": {"c": choice({"a": "A", "b": "B"}), "n": {"type": "noul", "instructions": "x"}}},
         select_logits=[[1.0, 2.5, 2.0, 0.0], [0.5, 1.5, 0.25, 3.0, -1.0]]),
    Case("math-one-option", "One option: p = 1 and confidence 1 whatever the score", ["choice-one-option", "math-reference"],
         request=q1(choice({"only": "Only"})), synthetic=[[-7.25]]),

    # --- temperature and decision metadata (metadata-only variants, synthetic scores) ------------
    Case("math-temperature-buckets-choice",
         "Bucket temperatures choice.2, .3_5, .6_10 and .11 win over the type temperature",
         ["math-temperature-bucket"],
         metadata={"set": {"lfm3.decision.temperature.choice.2": "f32:0.5", "lfm3.decision.temperature.choice.3_5": "f32:2.0",
                           "lfm3.decision.temperature.choice.6_10": "f32:3.25", "lfm3.decision.temperature.choice.11": "f32:0.125"}},
         request={"state": STATE, "questions": {
             "two": choice(options(2)), "four": choice(options(4)), "six": choice(options(6)), "eleven": choice(options(11))}},
         synthetic=[[1.0, 0.0], [1.0, 0.0, -1.0, 0.5], [1.0, 0.0, -1.0, 0.5, 2.0, -2.0],
                    [1.0, 0.0, -1.0, 0.5, 2.0, -2.0, 0.25, 0.75, 1.5, -0.5, 0.125]]),
    Case("math-temperature-buckets-score-noul", "Bucket temperatures for score and noul questions",
         ["math-temperature-bucket"],
         metadata={"set": {"lfm3.decision.temperature.score.2": "f32:0.5", "lfm3.decision.temperature.score.3_5": "f32:2.0",
                           "lfm3.decision.temperature.score.6_10": "f32:4.0", "lfm3.decision.temperature.noul.2": "f32:0.25"}},
         request={"state": STATE, "questions": {
             "s2": {"type": "score", "instructions": "x", "criteria": ["0", "1"]},
             "s4": {"type": "score", "instructions": "x", "criteria": ["0", "1", "2", "3"]},
             "s10": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(10)]},
             "n": {"type": "noul", "instructions": "x"}}},
         synthetic=[[0.0, 1.0], [0.0, 1.0, 2.0, 0.5], [0.0, 1.0, 2.0, 0.5, -1.0, 3.0, 2.5, 1.5, 0.25, -0.5], [1.0, 0.0]]),
    Case("math-temperature-bucket-fallback", "Without its bucket key a question falls back to the type temperature",
         ["math-temperature-bucket"],
         metadata={"set": {"lfm3.decision.temperature.choice.3_5": "f32:2.0"}},
         request={"state": STATE, "questions": {"two": choice(options(2)), "four": choice(options(4))}},
         synthetic=[[1.0, 0.0], [1.0, 0.0, -1.0, 0.5]]),
    Case("math-temperature-default", "No temperature key at all: T = 1", ["math-temperature-default"],
         metadata={"remove": ["lfm3.decision.temperature.choice", "lfm3.decision.temperature.noul", "lfm3.decision.temperature.score"]},
         request={"state": STATE, "questions": {"c": choice({"a": "A", "b": "B", "c": "C"}), "n": {"type": "noul", "instructions": "x"},
                                               "s": {"type": "score", "instructions": "x", "criteria": ["0", "1", "2"]}}},
         synthetic=[[4.0, 1.5, -2.0], [2.5, 0.25], [0.3, 2.0, 1.1]]),
    Case("math-temperature-tiny", "A temperature of 1e-5 survives the 6-decimal string round trip as 0.000010",
         ["math-temperature-bucket", "math-underflow"],
         metadata={"set": {"lfm3.decision.temperature.choice": "f32:0.00001"}},
         request=q1(choice({"a": "A", "b": "B"})), synthetic=[[1.0, 0.99]]),
    Case("math-temperature-rounds-to-zero", "A temperature of 4e-7 reads back as 0.000000: upstream refuses to load the model",
         ["refusal-load", "math-temperature-bucket"],
         metadata={"set": {"lfm3.decision.temperature.choice": "f32:0.0000004"}},
         request=q1(choice({"a": "A", "b": "B"})), synthetic=[[1.0, 0.0]],
         xybrid=refuse("load", "invalid_metadata", {"load": "match", **SYNTH})),
    Case("math-temperature-negative", "A negative temperature: upstream refuses to load the model", ["refusal-load"],
         metadata={"set": {"lfm3.decision.temperature.noul": "f32:-1.0"}},
         request=q1({"type": "noul", "instructions": "x"}), synthetic=[[1.0, 0.0]],
         xybrid=refuse("load", "invalid_metadata", {"load": "match", **SYNTH})),
    Case("math-score-bias", "score_bias.3.* is added to the scores of 3-level questions before the softmax",
         ["math-score-bias", "divergence"],
         metadata={"set": {"lfm3.decision.score_bias.3.0": "f32:0.5", "lfm3.decision.score_bias.3.2": "f32:-0.25"}},
         request={"state": STATE, "questions": {"s3": {"type": "score", "instructions": "x", "criteria": ["0", "1", "2"]},
                                               "s2": {"type": "score", "instructions": "x", "criteria": ["0", "1"]}}},
         synthetic=[[0.3, 2.0, 1.1], [0.0, 0.25]],
         xybrid=refuse("load", "unsupported_metadata", {"load": "match", **SYNTH}),
         notes="Contract 3: unsupported optional decision metadata such as score_bias is refused, never ignored."),
    Case("math-media-temperature", "With media, the image temperature replaces the question's", ["math-media-temperature"],
         metadata={"set": {"lfm3.decision.temperature.image": "f32:2.5"}},
         request=q1(choice({"a": "A", "b": "B", "c": "C"})), synthetic=[[4.0, 1.5, -2.0]], has_media=True,
         xybrid={"load": "match", "parse": "match", "prepare": "match", "evaluate": "given", "finalize": "out_of_scope"},
         notes="The text profile has no media; recorded for O2. format_answer ran with has_media = true, no media was sent."),
    Case("math-media-temperature-unused", "The image temperature is unused without media", ["math-media-temperature"],
         metadata={"set": {"lfm3.decision.temperature.image": "f32:2.5"}},
         request=q1(choice({"a": "A", "b": "B", "c": "C"})), synthetic=[[4.0, 1.5, -2.0]],
         notes="An image temperature cannot change a text answer, so a text-profile model carrying one still loads."),
]
