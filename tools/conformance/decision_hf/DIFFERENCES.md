# Where LiquidAI's d1 code and llama.cpp's server disagree (LFM3 d1, text)

The decision profile `lfm3-d1/text/v1` has two references:

- **HF**: LiquidAI's own code for `LiquidAI/d1-2B` at
  `10e7671d44e4b37700020b1eed3f594c5f2af0b4` (artifact `d1-2b-hf`): `prompt.py`,
  `api.py`, `attention.py`, `runner.py`, `calibration.py`, run by
  [`run_reference.py`](run_reference.py) one question per sequence
  (`set_size = 1`). Line numbers below refer to that revision
  (`https://huggingface.co/LiquidAI/d1-2B/blob/10e7671d44e4b37700020b1eed3f594c5f2af0b4/<file>`).
- **upstream**: llama.cpp v0.8.0 (`vendor/llama-cpp` at
  `8642541b255bfb5dc39cd351a178be09013a55d6`), `tools/server/server-decision.cpp`
  behind `POST /v1/systemone`, on the d1-2B GGUFs. Line numbers refer to
  `vendor/llama-cpp/`.

xybrid follows upstream (contract 3); HF is the independent numerical reference
for G1. Where the two disagree, G1 compares only what they share, and the
answer for xybrid is upstream's (or a refusal; see each case's `xybrid` block).

Every row names a case in
[`integration-tests/fixtures/decision/hf/cases/`](../../../integration-tests/fixtures/decision/hf/cases).
**HF observed** is that case's observation in `hf/observations/` (or
`hf/packed/`, `hf/synthetic/`). **upstream observed** is what the stock
`llama-server` built from `vendor/llama-cpp` answered on the same request on
this machine on the F16 GGUF, the G1 artifact (contract C10)
(`hf/crosscheck/d1-2b-f16.cpu-t4-fa-off.json`, settings `cpu-t4-fa-off`; the
BF16 run of the same cases is a report and agrees on every row). That is a
black-box check: the upstream oracle (`tools/conformance/decision`) observes
the stages themselves.

"gap" below is the largest difference of max-pooled logit differences
(s_i − s_top) between the two sides. The server returns probabilities only,
so its gaps are recovered as T·log(p_i/p_top). Where both sides read the same
prompt (the pilot cases and C16's two `hf-diff-*` additions; section 6) the F16
GGUF stays within 0.0036 of HF (the BF16 GGUF within 0.027), so larger gaps on
a case mean different prompts, not numerics. Which cases bound G1 is P11's
allowlist (contract C16), not these names.

## 1. Every request: evaluation and numerics

| # | Topic | HF | upstream | Case | Observed |
|---|---|---|---|---|---|
| 1.1 | **Question packing** | Serves `d1_attention: "set"` with `d1_set_size: 16` (`config.json:118-120`): up to 16 questions after one state in one sequence (`runner.py:164-174`, `attention.py:1-21`). Full-attention layers let the state see every question; the short-convolution and sliding-window layers carry earlier questions into later ones. | One pass per question, never batched: lfm3 d1 sets `no_batching` (`server-decision.cpp:1236-1237`). | `hf-diff-packing` (5 questions), `hf-card-three` | HF packed vs alone (`hf/packed/`): logits move up to 1.36, probabilities up to 0.113; usage 178 vs 254 tokens. HF alone vs upstream: gap 0.0036, numerical noise; read from `observations/`, it is in P11's allowlist (C16). G1 must use the `set_size = 1` observations only. |
| 1.2 | **Readout quantity** | Reads the vocabulary log-softmax at the answer slot in float32 (logits − logsumexp, `runner.py:357-358`) and max-pools log-probabilities (`prompt.py:392`). | Reads the raw logits of the label tokens and max-pools them (`send_decision`, `server-context.cpp:2526-2546`). | every case | The shift is the same for every token at the slot, so the winners and the option softmax agree up to float32 rounding of `logit − lse` (measured ≤ 9.5e-7). On the 44 evaluated questions, softmaxing the log-probability maxima instead of the logit maxima moves a probability by at most 1.9e-9. The observations carry both: `hf.evaluate` holds raw logits (upstream's quantity), `detail.evaluate.raw_logprobs` what HF softmaxes. |
| 1.3 | **Temperature value** | `calibration.json` `default` 1.337398453966105 as an f64, for every type (`temperatures` is empty; `calibration.py:60-61`). | The GGUF stores f32 `0x3fab2fdf` (1.3373984098, the nearest f32). Upstream reads it through `llama_model_meta_val_str` (`std::to_string`, so "1.337398") and `strtof`, giving `0x3fab2fdc` (1.3373980522) (`server-decision.cpp:58-73`). | every case; `hf/synthetic/median.json` `end_to_end` | Relative difference 3.0e-7. On the 44 evaluated questions it moves a probability by at most 1.5e-7, but near one half it flips a discrete answer: the synthetic scores `[0, -6, 0.0149775...]` give median level 1 (HF, p₂ = 0.4999999955) against level 2 (upstream's rule, p₂ = 0.5000000001). G1 compares logit gaps, which no temperature enters. |
| 1.4 | **Softmax operation order** | `exp(s/T − max(s/T))` with numpy's vectorized `exp`, f64 (`prompt.py:393-395`). | `exp((s − max)/T)` with glibc `exp`, f64 (`server-decision.cpp:1681-1687`). | every case | Last-bit differences. Exact answer arithmetic is checked against upstream by the oracle (G0), not by G1. |
| 1.5 | **Score median** | Renormalizes as it accumulates: `mass += p / sum(p)` (`api.py:67-78`). | Accumulates `p` as it is (`server-decision.cpp:1731-1740`). | `hf/synthetic/median.json` `renormalization` | Differs only when the probabilities do not sum to exactly 1 and the cumulative mass sits within about 1e-16 of one half. Three float64-score cases from HF's own `readouts` show it (renormalized and plain rules give different levels), and two hand-built vectors fall on either side (levels 2 vs 1 and 1 vs 2). A float32 score step moves the mass by about 1e-7, so model logits almost never land there. |
| 1.6 | **Forward precision** | Float32 throughout (PyTorch, MKL), over the checkpoint's bf16 weights widened exactly (README.md, "Precision"). | ggml on the CPU rounds each matmul's activations to the weight type: bf16 for the BF16 GGUF, f16 for the F16 GGUF (`ggml/src/ggml-cpu/ggml-cpu.c:222-227, 395-399`). | the pilot cases (section 6) | F16 GGUF (the G1 artifact) vs HF: gap ≤ 0.0033, \|Δp\| ≤ 0.00049. BF16 GGUF vs HF (a report): gap ≤ 0.027, \|Δp\| ≤ 0.0042. HF's own bf16 forward moves logits by up to 0.11. Section 6. |
| 1.7 | **Usage** | `input_tokens` sums each sequence's tokens; packed, the state counts once (`runner.py:192-194`). | Sums every pass's tokens (`server-context.cpp:6674`). | every case | Equal at `set_size = 1` on all 24 pilot cases and C16's two additions. |

## 2. Prompt construction

| # | Topic | HF | upstream | Case | Observed |
|---|---|---|---|---|---|
| 2.1 | **Choice keys as codes** | Strips each key and takes the keys as codes if every one is then a single Unicode letter (`str(x).strip()`, `str.isalpha`; `prompt.py:166-167`). | Takes the keys as codes only if every one is exactly one byte that `std::isalpha` accepts (ASCII; `server-decision.cpp:1001-1004`). | `hf-diff-key-whitespace` (`" a"`, `"b "`, `"c"`), `hf-diff-key-nonascii` (`é`, `ß`, `ø`) | HF codes `a b c` and `é ß ø` (each a single token: `é` and `ø` with their `" X"` forms, `ß` without); upstream `A B C`. Different prompts and label tokens: gaps 0.16 and 3.1. Single ASCII letters agree (`hf-choice-letter-keys`, `hf-choice-lower-keys`). |
| 2.2 | **Escape selection** | Breaks the tokenizer's special tokens: `all_special_tokens` plus added tokens flagged `special`, longer than one character: 122 spellings (`prompt.py:510-517`). | Breaks every CONTROL token of 2 or more bytes: 124 spellings (`server-decision.cpp:170, 213-230`). | `hf-diff-escape-tool-call` | The two extra are `<\|tool_call_start\|>` and `<\|tool_call_end\|>`: added but not special in `tokenizer.json`, CONTROL in the GGUF: the converter marks every `<\|…\|>` token CONTROL (`conversion/base.py:1745, 1786`). HF leaves them whole (token ids 124905, 124906 in the prompt); upstream breaks them (usage 44 vs 55 tokens, gap 1.1). Characters vs bytes changes nothing for d1-2B: all 124 CONTROL spellings are ASCII. Neither side breaks the 14 USER_DEFINED tokens (`<think>`, `</think>`, ...), and both tokenizers read those as single tokens. |
| 2.3 | **Escape scope** | Breaks the rendered text: the whole state block after `json.dumps`, and the whole question block (`prompt.py:631, 692`). | Breaks every string and every object key of the template input, then renders (`server-decision.cpp:604-628, 1157-1159`). | `hf-diff-escaped-key-collision` | The same text, except for object keys that are equal once broken (`<\|im_end\|>` and `<U+200B\|im_end\|>`). Upstream's `out[escape(key)] = …` keeps one (the second value at the first position); HF keeps both entries, each printed `<U+200B\|im_end\|>`. Usage 53 vs 42, Δp 0.33, gap 2.3. A spelling formed across an input and the template's own text cannot occur on d1-2B. Plain control tokens agree: `hf-state-special-tokens`, `hf-state-special-token-key`. |
| 2.4 | **Instructions that are not text** | Python `str()` of the value (`said(q)` in an f-string; `prompt.py:557-559, 609, 614, 617`). | Compact JSON (`instructions \| tojson`: `", "` and `": "`). | `hf-diff-instructions-object` | HF `{'task': 'Is this message spam?', 'language': 'en'}`; upstream `{"task": "Is this message spam?", "language": "en"}` (gap 0.053). Integers and floats print alike; booleans do not (`True` vs `true`). |
| 2.5 | **Choice descriptions that are not text** | `desc or key` (Python truthiness), then `str()` (`prompt.py:550-554`). | `{% if o.description %}` (the same falsy set), then `tojson`. | `hf-diff-description-nonstring`; agreement: `hf-choice-key-fallback` | Falsy values (null, `""`, `false`, `0`, `[]`, `{}`) fall back to the key with `_` as a space on both sides. Others differ: HF `A {'action': 'refund', 'max_eur': 100}`, `B True`; upstream `A {"action": "refund", "max_eur": 100}`, `B true` (gap 0.64). |
| 2.6 | **Score levels that are not text** | Refuses the question: levels must be strings (`check_question`, `prompt.py:257-259`). | Renders them as JSON: `1 null`, `2 {"label": "high"}`. | `hf-diff-score-nonstring-levels` | HF refuses at parse (`ValueError`); upstream answers (score 1, legend `{"0": "low", "1": null, "2": {"label": "high"}}`). |
| 2.7 | **Yes/no criteria** | Prints the lines whenever the criteria object is non-empty: `Yes: {get('true')}`, `No: {get('false')}`, so a missing or null value prints `None` (`prompt.py:612-613`). | Prints them only if `true` or `false` holds a non-null description. A missing one prints `None`. lfm3-d1 reads no `yes`/`no` aliases (`server-decision.cpp:338-348`). | `hf-diff-noul-null-criteria` (`{"true": null, "false": null}`), `hf-diff-noul-alias-criteria` (`{"yes": …, "no": …}`); agreement: `hf-noul-criteria`, `hf-noul-criteria-partial` | HF adds `\nYes: None\nNo: None`; upstream adds nothing (usage 51 vs 43, gap 0.6). |
| 2.8 | **Empty or blank instructions** | Refuses them: `"instructions" is the question as text` (`prompt.py:253-254`). | Accepts them: only missing or null instructions are refused (`server-decision.cpp:272-277`). | `hf-diff-instructions-empty`, `hf-diff-instructions-blank` | HF refuses at parse; upstream answers (noul 0.84 and 0.83). |
| 2.9 | **Integers past int64 in JSON** | Python integers: every digit (`json.dumps`, `prompt.py:500`). | The template reads integers as `int64` (`common/jinja/value.cpp:1449-1450`): an unsigned value above 2⁶³−1 wraps (2⁶⁴−1 prints `-1`, 2⁶³ prints `-9223372036854775808`). One past 2⁶⁴ was parsed as a double and prints as one (`1.2345678901234568e+29`). | `hf-diff-state-big-integer` | Different prompts (usage 74 vs 69) and a different answer: Δp 0.66, gap 7.9. |
| 2.10 | **What counts as spoken instructions** | Instructions that are an object whose `type` is `image`, `audio` or `video` are media (`prompt.py:277-279`, `media.py:24`); this runner refuses those. Any other object is written with `str()`. | lfm3-d1 refuses `type: "input_audio"` (`server-decision.cpp:237-239, 281-283`) and writes any other object as JSON. | `hf-diff-spoken-input-audio` | HF writes `{'type': 'input_audio', 'input_audio': {...}}` as the question and answers; upstream refuses (400, `spoken instructions need a model with audio input`). |
| 2.11 | **Media parts in a chat-message state** | Renders the state's JSON as given. | Takes `image_url`, `input_audio` and `video_url` parts out of a messages state and loads them as media (`server-decision.cpp:438-477`). Without an mmproj it refuses (`server-context.cpp:6616-6619`). | `hf-diff-chat-media-part` | HF answers over the JSON text (data URL included); upstream 501 `This server does not support image input for decisions`. |

## 3. Requests the two sides read differently

| # | Topic | HF | upstream | Case | Observed |
|---|---|---|---|---|---|
| 3.1 | **`"multiple": true`** | Answers each option yes or no, with its own prompt and the `multiple` calibration table (`prompt.py:602-606`, `api.py:125-128`). | Ignores the field (`parse_questions` reads `type`, `instructions` and `criteria` only; `server-decision.cpp:285-310`) and answers a single choice. | `hf-diff-choice-multiple` | HF `choices: ["billing", "delivery"]` (0.93, 0.97); upstream `choice: "delivery"` (0.63). |
| 3.2 | **Typed questions** (`text`, `number`, `date`, estimates, `box`, `boxes`) | Answers them (`api.py:91-122`). | Refuses: `"type" must be one of: choice, score, noul` (`server-decision.cpp:352-354`). | `hf-diff-typed-text` | HF reads `ACME SA` (found 0.99); upstream 400. |
| 3.3 | **Option count** | No limit but the code pool: 26 + 100 + 26 + 200 + 676 codes (`prompt.py:176-184`). | At most 255 (`n_options_max`, `server-decision.cpp:166, 376-378`). | `hf-diff-choice-256` | HF answers (n137 at 0.969, 1,064 tokens); upstream 400 `too many options (256), this model supports at most 255`. |
| 3.4 | **An empty questions object** | Accepts it and answers nothing. | Refuses it: `"questions" must be a non-empty object` (`server-decision.cpp:254-256`). | `hf-diff-questions-empty` | HF `answers: {}`; upstream 400. |
| 3.5 | **JSON that is not JSON** | Takes Python objects. Its callers' `json.loads` reads `NaN`, `Infinity` and lone surrogate escapes; `json.dumps` writes `NaN` and `Infinity` back. | nlohmann's parser (via `common_json`) refuses both (`server-context.cpp:6612`). | `hf-diff-state-nan`, `hf-diff-lone-surrogate` | HF renders `"amount": NaN` and answers; for the lone surrogate the tokenizer raises `TypeError` at prepare. Upstream 400 on both (`invalid literal`; `surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF`). |

## 4. Media (outside `lfm3-d1/text/v1`)

| # | Topic | HF | upstream | Case | Observed |
|---|---|---|---|---|---|
| 4.1 | **Picture size** | Caps every picture at `d1_vision_max_pixels` = 1,638,400 px, bicubic, before the processor (`config.json:121`, `runner.py:114-115`, `media.py:64-78`). | Keeps the mmproj's processor limit (16,777,216 px), capped at 4,096 image tokens (about 4.2 MP at 1,024 px a token) and at `n_ubatch / 2` tokens (`tools/mtmd/clip.cpp:1926-1935`, `server-context.cpp:1382-1386`). | none (text profile) | Not run. Pictures above about 1.64 MP are read at different resolutions; O2 needs its own cases. |

## 5. Checked and in agreement

On these the HF observation and the server agree at the numerical noise
level (gap ≤ 0.0036 on F16) with equal token counts, so the prompts and
tokenizations are the same:

- Prompt layout for text, JSON object, JSON array, null and empty states
  (`hf-state-*`), including the `<|startoftext|>` opening and the
  `<think></think>` lead. A null state drops the `QUESTION:` preamble; an
  empty one keeps it.
- JSON rendering: indent 2, `","` and `": "`, keys in request order, Python
  float repr (`-0.0`, `1e-05`, `1e+16`), non-ASCII text as is
  (`hf-state-json-object`).
- Codes `A`..`Z`, then `AA`, `AB`, …, with the pool's `00` for `AJ` at 36
  options (`hf-choice-27`, `hf-choice-36`); groups `[code, " code"]`; yes/no
  groups `yes`/`Yes` and `no`/`No`/`NO` (`YES` is two tokens); score digits.
- Falsy descriptions fall back to the key with `_` as a space
  (`hf-choice-key-fallback`).
- Control tokens in state text, JSON keys and JSON values are broken with
  U+200B (`hf-state-special-tokens`, `hf-state-special-token-key`).
- Repeated keys keep the last value at the first position on both sides
  (`hf-diff-duplicate-keys`: gap 0.0007). The xybrid profile refuses them.
- Answers: a choice's first maximum with confidence = max p; yes/no = p(true);
  a score's median with confidence = max p.
- A budget-sized state (`hf-state-long`, 3,605 tokens) reads alike.

## 6. Numbers behind the G1 note

The pilot cases (`hf-*` without `hf-diff-*`: 24 cases, 26 questions) were sent
to the stock `llama-server` (`hf/crosscheck/`) and compared with the HF
observations (`hf-cpu-f32-t1`). They are a historical selection by name, kept
as these summaries were first made: the G1 allowlist is P11's (contract C16),
not this subset. `d1-2b-f16` is the G1 artifact (contract C10); the other rows
are reports:

| upstream artifact, settings | role | max \|Δp\| | max gap | answers that differ | equal token counts | server seconds a case: median; 3,605 tokens |
|---|---|---|---|---|---|---|
| `d1-2b-f16`, `cpu-t4-fa-off` | G1 counterpart | 0.00049 | 0.0033 | 0 of 26 | 24 of 24 | 2.2; 172 |
| `d1-2b-bf16`, `cpu-t4-fa-off` | report | 0.0042 | 0.027 | 0 of 26 | 24 of 24 | 2.2; 173 |
| `d1-2b-bf16`, `cpu-t4-fa-on` | report | 0.0031 | 0.031 | 0 of 26 | 24 of 24 | 2.2; 137 |
| `d1-2b-q4km`, `cpu-t4-fa-off` | report (G3) | 0.097 | 0.87 | 0 of 26 | 24 of 24 | 1.4; 109 |

C16's allowlist also takes `hf-diff-packing` (from `observations/`) and
`hf-diff-duplicate-keys` (reference to reference). With them (26 cases, 32
questions) `d1-2b-f16` gives a largest gap of 0.0036 and \|Δp\| of 0.00049, and
BF16 0.027 and 0.0042, still with every answer and token count equal.

The answers and gaps above are identical across two runs of these binaries.
The server timings are not: the same long pass took 124 to 159 s in an
earlier run on this shared 4-vCPU machine. At one thread, the reference takes
a median of about 6 s a case without the parity proof and 18.6 s with it,
and 86 s and 224 s for the 3,605-token case.

The `hf-diff-*` cases on `d1-2b-f16`, where both sides answered (the BF16 run
gives each gap within a few percent):

| Case | equal token counts | max \|Δp\| | gap | same answer |
|---|---|---|---|---|
| `hf-diff-key-whitespace` | yes | 0.00006 | 0.16 | yes |
| `hf-diff-key-nonascii` | yes | 0.0005 | 3.1 | yes |
| `hf-diff-escape-tool-call` | no (44, 55) | 0.019 | 1.1 | yes |
| `hf-diff-escaped-key-collision` | no (53, 42) | 0.33 | 2.3 | yes |
| `hf-diff-instructions-object` | yes | 0.001 | 0.053 | yes |
| `hf-diff-description-nonstring` | yes | 0.005 | 0.64 | yes |
| `hf-diff-noul-null-criteria`, `hf-diff-noul-alias-criteria` | no (51, 43) | 0.0012 | 0.59 | yes |
| `hf-diff-state-big-integer` | no (74, 69) | 0.66 | 7.9 | **no** |
| `hf-diff-duplicate-keys` (agreement; in C16's allowlist) | yes | < 0.00001 | 0.0007 | yes |
| `hf-diff-packing` at `set_size = 1` (agreement; in C16's allowlist) | yes | 0.0005 | 0.0036 | yes |

So the F16 GGUF is the right G1 counterpart on CPU: ggml rounds a matmul's
activations to the weight type (row 1.6). The smallest prompt-induced gap here
(0.053, a quote style) is about 15 times F16's numerical noise (0.0036), but
only twice BF16's.
