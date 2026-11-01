# Upstream decision oracle (llama.cpp, LFM3 d1)

Writes the decision conformance cases,
[`integration-tests/fixtures/decision/cases/`](../../../integration-tests/fixtures/decision),
from llama.cpp's own decision code: the oracle side of gates G0 (exact
preparation) and the answer-arithmetic contract. Each case records what the
pinned upstream routines did with a SystemOne request, stage by stage, and the
stock `llama-server` built from the same sources answers the same request as a
black-box check. The fixture README defines the case format.

Nothing here is part of xybrid's cargo or Bazel build, and nothing here is
copied into xybrid: it exists so that xybrid's own implementation is compared
with upstream, never with a port of it.

| File | Role |
|---|---|
| `CMakeLists.txt` | Builds `vendor/llama-cpp` with its own CMake, then `decision-oracle` against upstream's `server-context` library |
| `oracle.cpp` | `decision-oracle`: runs the upstream routines on each request and prints every intermediate value |
| `test_access.h` | The test-access shim: the six private members of `server_decision_context` the oracle reads |
| `cases.py` | The case set: requests, coverage tags, xybrid's expectations, synthetic scores |
| `oracle.py` | Runs the oracle and the server on every case, writes the case files and `oracle.json` |
| `build.sh`, `clean.sh` | Build; list or delete `build/` and `work/` |

## What runs, and what is the oracle's own code

`server-decision.cpp` is compiled exactly as for `llama-server`: the oracle links
the same `server-context` static library target, built from the pinned
submodule with no patch. Per request, the oracle calls what the
`POST /v1/systemone` handler (`post_systemone`,
`tools/server/server-context.cpp`) calls, in the same order:

- parse: `common_json::parse`, `parse_questions`, `parse_state`,
  `prepare_state`; with files and no `--mmproj`, the handler's 501.
- prepare: `fill_task` per question and variant; then, through the shim,
  `render` (the prompt `fill_task` tokenized, checked by re-tokenizing it and
  comparing with the task's tokens), `d1_labels` (codes, checked against the
  task's labels and group sizes) and `get_temperature`.
- evaluate: as a server slot runs a decision prompt it cannot split. The
  model and context come from `common_params_parse(..., LLAMA_EXAMPLE_SERVER)`
  on the same arguments `llama-server` gets, the server's output limits
  (`common_speculative_get_output_limits`) and `common_init_from_params`. The
  batch is a `common_batch` of the whole prompt (sequence 0, positions 0..n-1,
  output on the last token), run by `llama_process(DECODE)`; the readout is
  `llama_get_logits_ith(ctx, n - 1)` at the task's labels.
- finalize: `format_answer` per question; usage sums the tasks' tokens.

Three small pieces mirror server code the oracle cannot call, because it lives
inside the server's slot loop or HTTP layer. The black-box server check covers
each: every model-backed case reads group maxima, the budget cases hit the size
checks, and the refusal cases the status mapping.

| Mirror | Upstream |
|---|---|
| The prompt-size checks before decoding (`n_tokens > n_ubatch`: 500, `> n_ctx`: 400) and their messages | `update_slots`, the `!slot.can_split()` branch (`server-context.cpp:4058`) |
| Each group's score is the max of its label logits | `send_decision` (`server-context.cpp:2526`) |
| An exception's HTTP status: `invalid_argument` and JSON errors 400, others 500 | `ex_wrapper` (`server.cpp:61`) |

### The test-access shim

`test_access.h` reads private members of `server_decision_context` without
editing or recompiling upstream: explicit template instantiation may name
private members (C++ applies no access check there), which yields pointers to
members; calling through them runs the upstream function itself. It exposes
exactly:

| Member | Used for |
|---|---|
| `render` | the exact prompt bytes |
| `d1_labels` | each option's label text (the choice code) |
| `get_temperature` | the temperature `format_answer` uses without media |
| `temperatures`, `score_bias` | the tables `init` read from the metadata (`oracle.json`) |
| `quoted_spellings` | the CONTROL spellings `d1_break` breaks, longest first (`oracle.json`) |

A changed signature upstream fails to compile here; see "When the llama.cpp
pin moves".

## Build

Needs CMake 3.14+, a C++17 compiler, ninja (or make), Python 3.9+ and the
submodule. On Amazon Linux 2023:

```bash
sudo dnf install -y gcc gcc-c++ cmake ninja-build
git submodule update --init vendor/llama-cpp
tools/conformance/decision/build.sh        # prints .../build/bin
```

`build.sh` configures `CMakeLists.txt` into `build/` and builds two targets,
`decision-oracle` and `llama-server`, CPU only: no GPU backend, no OpenMP, no
`-march=native`, and on x86_64 the x86-64-v3 flags xybrid's own natives use
(`crates/llama-cpp-sys/build.rs`). Everything comes from the vendored tree
(ggml, llama, common, mtmd, the server, cpp-httplib, nlohmann/json); the server
is built without its web UI and without OpenSSL, so the build needs no network.
On a 4-vCPU VM a clean build takes about 5.5 minutes, the oracle alone a few
seconds.

## Run

The model is the `d1-2b-q4km` artifact of
[`artifacts.json`](../../../integration-tests/fixtures/decision/artifacts.json)
(1.1 GB). `oracle.py` checks its size and sha256 before use; `--download`
fetches it into `work/models/` (or `$XYBRID_DECISION_MODELS`).

```bash
python3 tools/conformance/decision/oracle.py --download          # writes cases/ and oracle.json
python3 tools/conformance/decision/oracle.py --check             # writes nothing; fails on any difference
python3 tools/conformance/decision/oracle.py --only choice-36-options
```

A run builds one metadata-only GGUF per distinct `metadata` edit (in
`work/run/`), runs every case through `decision-oracle`, starts `llama-server`
with the settings' arguments on a free local port, sends it every request it
can be given as is (no synthetic scores, no metadata edits), and fails if any
answer, usage or error differs from the oracle's. On 4 vCPUs it takes about 12
minutes, half in the oracle's forward passes and half in the server's; the two
4096-token prompts dominate. `--settings cpu-t4-fa-on` evaluates with flash
attention on (the files then record that settings id), and `--out DIR` writes
somewhere other than the committed fixtures.

Two runs give byte-identical files, raw logits included: the CPU kernels are
deterministic at a fixed thread count, which is what makes `--check` useful.

`--check` needs the reference platform for the logits: the same CPU kernels
(this build's flags), 4 threads and the same compiler give bit-identical
results; another machine can differ in the last bits of `raw_scores`, which is
what G2/G3 measure, not what this check is for.

### `decision-oracle` directly

```bash
build/bin/decision-oracle run --input IN.json --output OUT.json -- -m MODEL.gguf -c 4096 -b 4096 -ub 4096 -np 1 -t 4 -tb 4 -fa off
build/bin/decision-oracle run --input IN.json --output OUT.json --vocab-only MODEL.gguf
build/bin/decision-oracle variant --base MODEL.gguf --output VARIANT.gguf --set KEY=f32:VALUE --remove KEY --token-type ID=TYPE
```

`IN.json` is `{"cases": [{"id", "request_text", "select_logits"?, "has_media"?}]}`;
`select_logits` (per pass, every label logit as an f32 bit pattern) replaces
the model's. Without a model context (`--vocab-only`) and without
`select_logits`, a case stops after `prepare`. `OUT.json` holds `facts` (the
artifact's decision metadata, temperature tables, template, CONTROL spellings,
vocabulary size, the context's effective parameters) and, per case, the
`parse`, `prepare`, `evaluate` and `finalize` objects that `oracle.py` turns
into the case format, or the refusal that stopped it. When `init` itself
throws, `OUT.json` holds `load` instead. `variant` writes a GGUF with the
metadata of another, edited, and no tensors: enough for `--vocab-only`.

## Add a case

Add it to `cases.py` (request, title, coverage tags from the fixture README,
xybrid's expectation per stage, notes for any divergence), run `oracle.py`,
and review the new file: the prompt, the groups and the answers are upstream's,
the expectations are yours. `oracle.py` refuses an expectation that contradicts
upstream (`match` where upstream refused, a refusal later than upstream's).

## When the llama.cpp pin moves

1. Update the submodule, then `build.sh`. A compile error in `test_access.h`
   or `oracle.cpp` means an upstream signature changed: update the declaration,
   then check that the routine still means the same thing.
2. Compare the three mirrored pieces above with the new `server-context.cpp`
   and `server.cpp`, and the order of calls in `post_systemone`.
3. Run `oracle.py`, then review `git diff integration-tests/fixtures/decision`:
   prompt bytes, token ids, codes and answers. Every difference is an upstream
   behavior change xybrid has to follow or refuse; record it in the PR.
4. Model-backed answers move whenever the logits do (a kernel change); that
   alone is expected, a changed prompt or code is not.
