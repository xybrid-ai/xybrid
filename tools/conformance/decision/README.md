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
| `CMakeLists.txt` | Builds `vendor/llama-cpp` with its own CMake, CPU backend only, then `decision-oracle` against upstream's `server-context` library |
| `identity.cmake` | Stamps each build's identity into `decision-oracle` and `llama-server` |
| `oracle.cpp` | `decision-oracle`: runs the upstream routines on each request and prints every intermediate value |
| `test_access.h` | The test-access shim: the six private members of `server_decision_context` the oracle reads |
| `cases.py` | The case set: requests, coverage tags, xybrid's expectations, synthetic scores |
| `oracle.py` | Runs the oracle and the server on every case; writes the case files and `oracle.json`, checks them, or collects observations for other case files and artifacts |
| `build.sh`, `clean.sh` | Build; list or delete `build/` and `work/` |
| `NOTICE` | The llama.cpp material repeated here, and its MIT license |

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

`NOTICE` lists these fragments with upstream's license.

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

### CPU only, three ways

The settings ids `cpu-t4-fa-off` and `cpu-t4-fa-on` mean the ggml CPU backend
and nothing else. A collector for another backend gets settings ids of its own.

1. **Build.** `CMakeLists.txt` turns every other ggml backend off (Metal, CUDA,
   HIP, MUSA, Vulkan, SYCL, OpenCL, CANN, RPC, WebGPU, zDNN, VirtGPU, Hexagon,
   ZenDNN, OpenVINO, ET), the CPU backend's alternative matmul libraries too
   (BLAS, Accelerate, KleidiAI), and dynamic backends, native tuning and
   OpenMP. A value set otherwise, on the command line or in a reused cache,
   stops the configuration instead of being overridden in silence; after
   configuring, any `ggml-*` backend target but `ggml-cpu` is an error too.
2. **Arguments.** Both executables get `-ngl 0 --device none -fit off` with the
   settings' context, threads and flash attention, and run in an environment
   without `LLAMA_*`, `GGML_*` or `MTMD_*` variables, which would otherwise set
   options behind the arguments' back.
3. **Run time.** `decision-oracle` refuses to start unless every ggml device it
   can reach is a CPU device, and to run unless the loaded model has no offload
   device; it records both, with the effective context, which `oracle.py`
   checks against `artifacts.json`. `oracle.py` runs `llama-server
   --list-devices` before starting the server and refuses anything but
   `(none)`.

### Build identity

`identity.cmake` runs on every build and stamps the same JSON into both
executables:

- the configured llama.cpp directory;
- its commit, read with `git -C <dir>` only if `<dir>` is the top of its own
  checkout, so an uninitialized submodule directory does not report the
  enclosing repository's commit;
- whether that checkout has local changes;
- the sha256 of the oracle's sources (`CMakeLists.txt`, `identity.cmake`,
  `oracle.cpp`, `test_access.h`) and of the upstream files it runs or mirrors;
- the build profile: compiler and version, build type, flags, every backend
  option;
- a fingerprint of all of that, the directory excepted.

`decision-oracle --identity` prints it, and `build/bin/oracle-identity.json`
holds a copy.

The identity rides into `llama-server` as one inert data object added to its
sources (`target_sources` in `CMakeLists.txt`). No upstream code changes and
nothing reads the object at run time, but the stamped `llama-server` is **not
byte-identical to a plain upstream build** of the same commit; reproducibility
claims about its bytes must say so.

## Build

Needs CMake 3.14+, a C++17 compiler, ninja (or make), Python 3.9+ and the
submodule. On Amazon Linux 2023:

```bash
sudo dnf install -y gcc gcc-c++ cmake ninja-build
git submodule update --init vendor/llama-cpp
tools/conformance/decision/build.sh        # prints .../build/bin
```

`build.sh` configures `CMakeLists.txt` into `build/` and builds two targets,
`decision-oracle` and `llama-server`. The CPU kernels are xybrid's own natives'
(`crates/llama-cpp-sys/build.rs`): no `-march=native`, and on x86_64 the
x86-64-v3 flags. Everything comes from the vendored tree (ggml, llama, common,
mtmd, the server, cpp-httplib, nlohmann/json); the server is built without its
web UI and without OpenSSL, so the build needs no network. On a 4-vCPU VM a
clean build takes about 5 minutes, the oracle alone a few seconds. A build
directory configured by an earlier version of this file may hold upstream's
default `GGML_ACCELERATE=ON`: start from an empty one.

## Run

The model is an artifact of
[`artifacts.json`](../../../integration-tests/fixtures/decision/artifacts.json),
`d1-2b-q4km` (1.1 GB) for the case set. Its default place is where
`integration-tests/download.sh` puts it,
`integration-tests/fixtures/models/<artifact>/<file>`; `oracle.py` checks its
size and sha256 against `artifacts.json`, and that `models.json`, when it lists
the artifact, pins the same URL, file and sha256. `--download` fetches a missing
one.

```bash
integration-tests/download.sh d1-2b-q4km                         # or oracle.py --download
python3 tools/conformance/decision/oracle.py                     # writes cases/ and oracle.json
python3 tools/conformance/decision/oracle.py --check             # writes nothing; fails on any difference
python3 tools/conformance/decision/oracle.py --only choice-36-options --out /tmp/cases
```

A run builds one metadata-only GGUF per distinct `metadata` edit, runs every
case through `decision-oracle`, starts `llama-server` with the same arguments on
a free local port, sends it every request it can be given as is (no synthetic
scores, no metadata edits), and fails if any answer, usage or error differs from
the oracle's. On 4 vCPUs it takes about 13 minutes, half in the oracle's forward
passes and half in the server's; the two 4096-token prompts dominate.
`--settings cpu-t4-fa-on` evaluates with flash attention on, and `--out DIR`
writes somewhere other than the committed fixtures.

### Official runs

`oracle.py` reads the identity of the executables it is given and produces
official observations only if:

- both carry the same identity;
- it names a git checkout of llama.cpp at the pinned commit
  (`PINNED_LLAMA_CPP` in `oracle.py`), with no local changes;
- the repository's `vendor/llama-cpp` gitlink agrees;
- the oracle's sources on disk are the ones the executables were built from.

Otherwise it stops and says why. `--dev` runs anyway, whatever the
executables: the records say
`"collector": {"official": false, "developer_run": true, "problems": [...]}`, and
a developer run does not write the committed fixtures.

`oracle.json` records the collector, the build identity, the settings as they
ran (server arguments, effective context, devices) and the sha256 of every
recipe file (`CMakeLists.txt`, `identity.cmake`, `oracle.cpp`, `test_access.h`,
`build.sh`, `cases.py`, `oracle.py`).

`--check` regenerates in memory and fails unless:

- every case file is identical;
- the committed `oracle.json` is official;
- its recipe hashes, build identity (the directory excepted), settings, llama.cpp
  commit and artifact facts are those of the run.

The platform (CPU model, libc) is reported, not compared. Two runs on the same
build give byte-identical files, raw logits included: the CPU kernels are
deterministic at a fixed thread count. Another compiler or flags make another
build, whose logits can differ in the last bits; that is what G2/G3 measure, not
what this check is for. `--check --dev` compares the observations from a
different build and reports the identity differences as warnings.

### Collecting observations for other cases and artifacts

```bash
python3 tools/conformance/decision/oracle.py --collect integration-tests/fixtures/decision/hf/cases \
    --artifact d1-2b-f16 --settings cpu-t4-fa-off --expect-cases 46 --out /tmp/f16-observations --download
```

`--collect DIR` runs a directory of schema-v1 case files on `--artifact` (any
id of `artifacts.json`, size and sha256 checked) under `--settings`, sends the
sendable requests to `llama-server`, and writes observations to `--out`. It is
the path for evidence other gates rely on, so it fails closed:

- **Before anything runs:**
  - `DIR` must exist and hold `*.json` files.
  - `--out` must be a new or empty directory, neither `DIR` nor inside the
    committed fixtures.
  - `--expect-cases N`, when given, must equal the number of case files, or of
    `--only` ids.
  - Every `--only` id must be a valid case there.
  - `--collect` refuses `--check`.

  A corpus that holds any invalid file is not sound, whatever `--only`
  selects: every invalid file still gets its `invalid` receipt, so an `--only`
  run over that corpus is `incomplete` and exits 1 even when every selected case
  is observed. Its counts include those files: `corpus.selected` in
  `collection.json` and the printed summary count the invalid files among the
  selected ones, selected or not.
- **Every file is read as a strict envelope:**
  - a JSON object with no key twice and no `NaN`/`Infinity`;
  - `schema_version` the integer 1, `profile` `lfm3-d1/text/v1`, `id` matching
    the schema's `^[a-z0-9]+(-[a-z0-9]+)*$`;
  - `request_text` a string with UTF-8 bytes;
  - `has_media` a boolean, `metadata` the typed edits of the variant command;
  - synthetic `upstream.evaluate.raw_scores` lists of f32 bit patterns.

  The request text itself is opaque: malformed or duplicate-key requests are
  test data and go out byte for byte.
- **A file that fails** (as does every file sharing its id with another) gets
  an `invalid` receipt with the reason and is never run. A case the oracle
  cannot run gets an `error` receipt. Either makes the collection `incomplete`
  and the exit status non-zero, as does a disagreeing server. Every file is
  written inside `--out` or not at all.

It writes, all inside `--out`:

- `<id>.<sha256 of the request bytes>.json` per observed case. It holds:
  - the `upstream` stages as in a case file, `server` included;
  - `case_file_sha256`, the digest of the whole case file;
  - `request_sha256`, the join key;
  - `inputs`, the identity of the case: the request digest, the media flag,
    the metadata edits, the synthetic scores, and the artifact with its sha256
    and the sha256 of the metadata-only variant built from it;
  - `inputs_sha256`, its canonical digest;
  - the collector, the build fingerprint, and the digest of the recipe.

  `inputs_sha256` identifies the case, not the computation. What the
  computation ran on is recorded once per run, in `collection.json`:
  - `settings`: the settings tuple, with the server arguments, the effective
    context and the devices; flash attention alone changes the logits;
  - `build`: the build identity;
  - `platform`.

  An observation names only its settings id, or `null` when the model did not
  evaluate it: synthetic scores (a metadata-only variant always has them), or a
  refusal before evaluation. Evidence must bind both: the case through
  `inputs_sha256`, the computation through those run-level blocks.
- `collection.json`, the receipt:
  - `status` (`complete` or `incomplete`);
  - the corpus counts;
  - the artifact;
  - `recipe`, the sha256 of every file that makes an observation (the four
    compiled sources, `build.sh`, `cases.py`, `oracle.py`), and its digest;
  - the collector, the build identity, and the settings as they ran;
  - every case file with its `file_sha256`, its status, and for a selected
    case its request and inputs digests.

`--dev` collections, like `--dev` generations, say
`"official": false, "developer_run": true`. Cases with metadata edits run on a
metadata-only copy of the chosen artifact, so they are evaluated only from
synthetic scores.

`tests/` holds stdlib tests of these rules, which need neither a model nor a
build:

```bash
python3 -m unittest discover -s tools/conformance/decision/tests
```

### `decision-oracle` directly

```bash
build/bin/decision-oracle run --input IN.json --output OUT.json -- -m MODEL.gguf -c 4096 -b 4096 -ub 4096 -np 1 -t 4 -tb 4 -fa off -ngl 0 --device none -fit off
build/bin/decision-oracle run --input IN.json --output OUT.json --vocab-only MODEL.gguf
build/bin/decision-oracle variant --base MODEL.gguf --output VARIANT.gguf --set KEY=f32:VALUE --remove KEY --token-type ID=TYPE
build/bin/decision-oracle --identity
```

`IN.json` is `{"cases": [{"id", "request_text", "select_logits"?, "has_media"?}]}`;
`select_logits` (per pass, every label logit as an f32 bit pattern) replaces
the model's. Without a model context (`--vocab-only`) and without
`select_logits`, a case stops after `prepare`. `OUT.json` holds:

- `identity`;
- `facts`: the artifact's decision metadata, temperature tables, template,
  CONTROL spellings, vocabulary size, the devices, and the context's effective
  parameters;
- per case, the `parse`, `prepare`, `evaluate` and `finalize` objects that
  `oracle.py` turns into the case format, or the refusal that stopped it.

When `init` itself throws, `OUT.json` holds `load` instead. `variant` writes a
GGUF with the metadata of another, edited, and no tensors: enough for
`--vocab-only`.

## Add a case

Add it to `cases.py` (request, title, coverage tags from the fixture README,
xybrid's expectation per stage, notes for any divergence), run `oracle.py`,
and review the new file: the prompt, the groups and the answers are upstream's,
the expectations are yours. `oracle.py` refuses an expectation that contradicts
upstream (`match` where upstream refused, a refusal later than upstream's).

## When the llama.cpp pin moves

1. Update the submodule and `PINNED_LLAMA_CPP` together, then `build.sh`. A
   compile error in `test_access.h` or `oracle.cpp` means an upstream signature
   changed: update the declaration, then check that the routine still means the
   same thing.
2. Compare the three mirrored pieces above with the new `server-context.cpp`
   and `server.cpp`, and the order of calls in `post_systemone`; update the
   line numbers in `NOTICE`.
3. Run `oracle.py`, then review `git diff integration-tests/fixtures/decision`:
   prompt bytes, token ids, codes and answers. Every difference is an upstream
   behavior change xybrid has to follow or refuse; record it in the PR.
4. Model-backed answers move whenever the logits do (a kernel change); that
   alone is expected, a changed prompt or code is not.
