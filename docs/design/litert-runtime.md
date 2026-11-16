# LiteRT in Xybrid: architecture proposal

**Status:** The first slice is implemented on the branch
`feat/litert-lm-gemma4` and is not merged or shipped. The section "What was
built" below records it. Sections 1 to 12 are the proposal as it stood before
the build and are superseded where the two differ.

**Prepared:** 15 November 2026 against `master` at `77648247` (workspace
0.18.0). Upstream facts were read the same day from the LiteRT and LiteRT-LM
repositories, their release assets and the package registries.

**Versions:** this is version 5 of the proposal. Versions 2 and 3 narrowed
version 1, which covered LiteRT-LM, the tensor runtime and the browser on
every binding, to one Android slice. Version 4 rewrote sections 1, 4.3, 5, 9,
10 and 11 once the maintainer had named the use case, Gemma 4 on mobile.
Version 5 corrects three things version 4 overstated and adds section 4.4, on
open upstream defects. Section 2 records what changed between versions and
why.

Claims marked **(verified)** were checked against the code, the upstream
source or the shipped binary. Claims marked **(unverified)** were not.

## What was built (15 November 2026)

The maintainer asked for the build before the experiment in section 5, with
text, image input and tools all required and the Android emulator as the test
target. The device experiment was not run. Everything below is CPU.

### Scope

Gemma 4 E2B (`gemma-4-E2B-it.litertlm`) through the existing `run*` surface:

| Capability | State |
|---|---|
| Text chat, batch and streaming, system prompt, multi-turn | works |
| Image input: one or several per message, and in earlier turns | works |
| Tool calling: call, streamed call, continuation with results | works |
| Thinking, reported as `reasoning_content` | works |
| Stop sequences, output budget, repetition penalty | works |
| A single `.litertlm` stage in a streaming pipeline | streams |
| Cancel from the streaming callback | works when a visible token arrives; not before (see "Open") |
| Grammar or JSON Schema output, raw prompts | refused with a typed error, streamed or not |
| `min_p` | not applied (the runtime has no such filter) |
| Seeded sampling | bindings crate only (`Sampling::with_seed`); `GenerationConfig` has no seed to carry |
| Audio input, GPU, NPU, speculative decoding | not attempted |

### Where it lives

- `crates/xybrid-litert-lm`: the bindings. Run-time loading, engine,
  conversation, and the stream bridge. No xybrid dependencies. Its
  `test-util` feature adds `testing::FakeRuntime`, a scripted runtime that
  lives in the test process.
- `crates/xybrid-core/src/runtime_adapter/litert_lm`: the `LlmBackend`.
  `messages.rs` turns xybrid messages into the runtime's JSON, `chunks.rs`
  turns stream chunks back into the text the model emitted.
- Feature `llm-litert` in core, sdk, cli and bolt; Bazel flag
  `--//:llm_litert=true`. Off everywhere by default.
- Fixture `integration-tests/fixtures/models/gemma-4-e2b-litert`, and
  `tests/litert_lm_gemma4.rs` in core (19 tests) and sdk (5 tests), which
  need the model and the runtime library.
- Tests that need neither, on the scripted runtime: the bindings'
  `tests/fake_runtime.rs`, the backend's `fake_tests.rs`, and
  `tests/litert_lm_fake.rs` in core and sdk. `tests/litert_lm_missing_runtime.rs`
  in the sdk covers a host without the library.

### Verified

| Where | What ran | Result |
|---|---|---|
| Mac, M4 Max, CPU | bindings: 19 unit and 27 scripted-runtime tests; 6 tests on the real library. Under Miri: the 27, and 18 of the 19 (the other opens a library file, which Miri cannot do) | pass |
| Mac, M4 Max, CPU | core: 1,729 unit tests with llama.cpp and LiteRT-LM compiled together, 57 of them the backend's (19 on the scripted runtime); 8 executor tests on the scripted runtime; 19 Gemma 4 tests. SDK: 5 tests without a model, 5 Gemma 4 tests. `xybrid run -d` with text and with an image | pass |
| Mac, llama.cpp models | the paths shared with GGUF models: LFM2-VL 450M (3 tests, one of them new: an image request keeps its system prompt), FunctionGemma streaming (2 tests) and the two FunctionGemma tool-calling examples | pass |
| Android 15 emulator, arm64, CPU | native test binaries run with `adb shell`: the bindings' 19, 27 and 6; core's 57 backend tests, 8 executor tests and 19 Gemma 4 tests; the SDK's 5 tests without a model and 5 Gemma 4 tests | pass |
| Bazel on the Mac | analysis of the workspace with the flag off and on; the bindings built and tested; `xybrid-core` compiled with the flag on | pass |

These are the results on the final code (16 November 2026).

Not verified: the new CI steps, which have not run on CI; the Bazel tests
with the flag on, which need remote execution; any binding inside an app
(Kotlin, Swift, Flutter, React Native); a physical phone; GPU; and the
Android AAR built with the flag (Bazel's Android targets build on remote
execution only, and this machine has no credentials for it).

### What the runtime turned out to do (v0.19.1, Gemma 4 E2B, CPU)

- **One sampler.** The CPU path implements top-p only; greedy and top-k
  return `UNIMPLEMENTED`. Greedy is sent as top-p over one candidate, and is
  repeatable.
- **Stream chunks are JSON messages.** Answer text arrives under `content`,
  a parsed call under `tool_calls`, and with one setting the call's raw text
  under `channels.tool_call`. Reasoning arrives twice in one chunk, under
  `channels.thought` and `reasoning_content`.
- **A tool call that was cut off comes back as answer text.** When the output
  budget ends inside a call block, no parsed call is reported; the runtime
  hands the unfinished block over under `content`, opening marker included
  (read in upstream's `internal_callback_util.cc`, and played on the scripted
  runtime; not provoked on the real model).
- **An error chunk is final but does not mean the runtime stopped.** Upstream
  marks every error final, including a tool call it could not parse, which it
  reports while decoding continues. Its own wait gives up after ten minutes
  (read in upstream's source; not provoked on the real model).
- **The finish reason is not reported.** A turn ends the same way at its
  natural end, at the output budget and when the context window fills. The
  backend infers `length` from the token counts.
- **An oversize prompt is refused cleanly** with `INVALID_ARGUMENT`, and the
  engine stays usable. The backend reports it as `InvalidInput`.
- **The file's context limit is fixed** at 4,096 tokens for this file.
- **The model file says it does not support function calling**, yet calls
  work. The backend goes by the rendered prompt, not by that field.
- **Images in the conversation's starting history are processed**, so
  resending history keeps earlier images.
- **Caches of about 1 GB** are written beside the model on first load unless
  a cache directory is set (`XYBRID_LITERT_LM_CACHE_DIR`).
- **Android emulator on Apple silicon:** upstream's arm64 library uses SME
  when the CPU reports it, and the emulator then faults. It needs
  `-qemu -append arm64.nosme`.

### Where the build departs from sections 6 and 7

- **Tools and images are in the first slice.** They go through the runtime's
  own chat interface, which applies the model's template, so no model-specific
  prompt code was added. Tool results continue through one new optional
  `LlmBackend` method that takes them as data (the idea of R6 in 6.3); the
  default is unchanged, so llama.cpp behaves as before.
- **Memory safety does not rest on the join.** 6.4 relied on wait and delete
  blocking until callbacks had returned. Callback data is now an id looked up
  in a table, so a late callback finds nothing. The fake-runtime tests cover a
  runtime that calls back after everything was freed.
- **`min_p` is not refused.** 6.4 refused a sampled request that carried it.
  Every sampled request carries the default 0.05, so that rule refused the
  ordinary case. It is documented as not applied. This is decision 11.8, and
  it is still the maintainer's to make.
- **Most of the core refactors in 6.3 were not done.** The feature gates
  were widened to name `llm-litert` (158 sites) instead of R1's umbrella
  feature; the six copies of adapter loading remain (R2a, R2b); and
  `.litertlm` models take the vision-language dispatch path. Of R5 only
  `is_chat_llm()` exists, and the registry's chat check does not use it; the
  pipeline's streaming fast path names `LiteRtLm` beside `Gguf` and still
  leaves `VisionLanguage` bundles on the batch path. R4 is a function a
  caller can ask (`litert_lm::runtime_available`), not a probe behind the
  feature list. R7 is done for the two capability refusals, which now keep
  their type on streaming calls; other adapter errors still become
  `InferenceError` there. R8 is open. R0 is done for the error messages; the
  docs still list `TfLite` as a usable template.
- **No companion AAR, identity library or release check.** The loader
  accepts any library that reports C API major 1. Nothing checks which
  runtime release it is, so the state fix in `e1f86b05` is not required.
- **Not tested:** the isolation, copy and lifecycle-soak tests of section 7,
  which target the GPU defects in 4.4.

### Defects found after the first build, and their fixes

The first build passed its tests. Reading it again against upstream's source,
and walking through what a caller can do with it, found the defects below.
Each is fixed, with a test that needs no model, except the one under "Open".

| Defect | Fix |
|---|---|
| `Library::from_api` was safe but trusted every function pointer it was given | It is `unsafe`, and its docs state the contract of the table |
| A stream handle lost with `mem::forget` left a conversation that could be used, and freed, while the runtime was still generating | The conversation owns its stream's state: it refuses other calls until the stream is finished, and dropping it tears the stream down |
| A final chunk with an error, or a wait that failed, was treated as proof that the runtime was at rest | A stream that ended with an error is cancelled before its conversation is deleted. When the wait fails, the conversation and its engine are leaked and refuse further work |
| A stalled engine still let its other conversations in | Every conversation checks its engine first, and their teardown no longer enters the runtime |
| Finished streams kept their queues until the conversation was dropped | A stream is retired when it is torn down |
| A tool call cut off by the output budget was closed with a made-up end marker and reported as a call | A call enters the reply's text only when the runtime reports it parsed. An unfinished block stays unterminated, and the turn ends as `length` |
| The arguments reported came from re-parsing the raw call text, which can differ from what the runtime parsed (it keeps the first of two equal keys; xybrid's parser keeps the last) | The runtime's parsed call is written out, read back with xybrid's parser, and refused if the two differ |
| Tool results sent after a conversation that contains an image were dropped, and the turn answered the original question again | Refused with `InvalidInput`, as the one-shot path already did. This is the shared vision-language path, so it applies to llama.cpp models too |
| A one-shot image request lost its `system_prompt` | The system prompt is sent as the first message. Shared path, so llama.cpp models too |
| A teardown error replaced the error that had stopped the turn, a cloud-fallback abort included | The turn's own error is returned; the teardown error is logged |
| Rendering a prompt from a token callback blocked for good | Rendering no longer takes the generation lock |
| A stop sequence inside reasoning did not end the turn | Stop sequences are matched in reasoning too, as on llama.cpp |
| A single `.litertlm` stage in a streaming pipeline ran as a batch, and the token callback never fired | The pipeline's streaming fast path takes `LiteRtLm` |
| Capability refusals lost their type on streaming calls | `UnsupportedBackendCapability` and `UnsupportedModelCapability` are kept, for every backend |
| An SDK test read its own crate's feature, which Bazel sets on core only, so it would fail with the flag on | It asks core, as the loader does |
| CI compiled the feature in core only, and ran only the adapter's unit tests | `clippy-llm` and `test-llm` cover core, sdk, cli and bolt with the feature; `bazel.yml` runs the inline tests with the flag on. None of these has run on CI yet |
| The CLI recorded the runtime as `litertlm`, the SDK as `litert-lm` | Both record `litert-lm` |
| The docs said a missing library fails the load, and listed seeded sampling | Corrected: the directory loads and the first run is refused; seeding is in the bindings crate only |

Upstream's source turned up one more: a tool
call the runtime cannot parse arrives with the status it uses for a bad
request, and was reported as `InvalidInput`. It is now `InferenceFailed`, and
the runtime's message, which quotes the model's whole reply, goes to the debug
log only.

To test these without a model, the scripted runtime moved out of the
bindings' test file into the crate (`xybrid_litert_lm::testing`, feature
`test-util`). It plays what the real runtime rarely does: a stream that never
ends, a wait that fails, an error while decoding continues, a callback after
everything was freed. Core re-exports it as
`runtime_adapter::litert_lm::testing` under its own `test-util` feature, and
`FakeRuntime::at(path)` puts a fake behind a model file, so the executor and
the SDK run on it unchanged.

### Open

- **A cancel is not seen while nothing visible arrives.** The SDK checks its
  cancellation token and abort policy in the token callback, and the backend
  calls that callback for visible answer text only. While the runtime reads a
  long prompt or an image, thinks, or writes a tool call, a cancel waits for
  the next visible token; a stream that stays silent waits out the timeout
  (`XYBRID_LITERT_LM_STREAM_TIMEOUT_SECS`, 600 s by default) and then the
  cancel grace, with the model's lock held. llama.cpp has the same limit, for
  the same reason. Fixing it needs a cancel signal carried from `RunOptions`
  through the executor to the backend, which is a change to the shared
  streaming path and was left out of this branch.
- **This is the likeliest first production incident:** a Cancel button that
  does nothing during a long image prefill, with later requests queued behind
  it.

### Before this ships to anyone

1. A maintainer review of the branch.
2. The Kotlin SDK in an app: build the AAR with the flag on remote execution,
   package `liblitert-lm.so` with the app, and run the consumer smoke.
3. The device experiment of section 5, which decides GPU and whether this
   beats a smaller GGUF.
4. A cancel that reaches a silent generation ("Open").
5. The companion package and the release check of 6.5, and the open
   decisions in section 11.

## 1. Recommendation

**The use case is decided (15 November 2026): Gemma 4 on mobile.**

On paper this is LiteRT-LM's best case. Gemma 4 is the model family the
runtime is built around, Google publishes the files under Apache-2.0, and its
figures show a phone GPU giving a first token in 0.3 seconds where the CPU
takes 1.8 (section 4.3). Xybrid's documented Gemma 4 path is a 5.5 GB bundle
that runs on CPU only on Android.

In practice the GPU path cannot be trusted yet. At its default settings,
Gemma 4 on LiteRT-LM returns wrong digits, and in one report broken tool
calls, once a conversation passes about 2,048 tokens. That is reported on
Adreno, Mali and Apple GPUs, and Google's maintainers say they have
reproduced it. The setting that fixes it costs speed and memory, and the
published figures were taken without it (section 4.4). So the order is:

1. **Set the limits, the devices and an owner** (section 11).
2. **Check correctness first** (section 5, stage A): Gemma 4 E2B on LiteRT-LM
   on one Android phone, with a short probe on an iPhone. Wrong answers,
   memory growth or state carried between conversations end the experiment
   before any speed is measured.
3. **Then compare, in the configuration that gives correct answers** (stage
   B), with what the existing runtime can do for the same model: a smaller
   GGUF, which is a registry change, and a llama.cpp GPU backend. Where a
   phone has it, Android's built-in AICore is measured too.
4. **If LiteRT-LM still wins, build the Android slice:** Gemma 4 E2B, text
   chat, arm64, loaded at run time from an optional companion AAR (sections 6
   and 7). It sits beside today's GGUF Gemma 4, which keeps images and tools
   until the LiteRT path has them.
5. **iOS is the second slice**, decided by its own measurements.
6. **Not proposed now:** the `.tflite` tensor runtime, the browser, NPU,
   embeddings, structured output and desktop (section 9).
7. **Worth doing whatever the experiment says:** fix the error messages and
   docs that still send LiteRT models to the browser SDK, and remove the six
   copies of LLM adapter loading in the executor (section 6.3, R0 and R2a).
   Also try a Q4 GGUF of Gemma 4 E2B on the existing runtime: most of the
   download saving comes from the smaller quantization, which needs no new
   runtime. A Q4 GGUF is 2.8 GB for text and 3.4 GB with vision; LiteRT-LM's
   file is 2.6 GB.

The project removed LiteRT from the browser five weeks ago to keep one
runtime; a second one still needs a measured reason.

Liquid models stay outside this work. Their `.litertlm` conversions are
community work with known defects, and the current upstream release has a
state bug that affects their architecture (section 4.1).

## 2. How the proposal changed, and why

Each table lists what was wrong in a version, checked against the code or
upstream before it was accepted, and what the next version did about it.

### Version 1 to version 2 (version 1 proposed LiteRT-LM, the tensor runtime and the browser, as add-on packages for every binding)

| Problem | Change |
|---|---|
| In LiteRT-LM v0.19.1 a new conversation on a shared engine continues from the previous one's state for models with non-KV state; fixed on `main` only | Runtime pin must include the fix; isolation is a qualification test; Liquid models are out of the first slice (4.1, 7) |
| "Give llama.cpp a GPU on Android" was listed but not evaluated, and the go/no-go ran on macOS | The experiment runs on Android and includes llama.cpp OpenCL and Vulkan (5) |
| The streaming bridge was safe only on the normal return path, and turning the callback's error into a flag lost the typed cloud-fallback abort | Replaced by an owned stream operation (6.4) |
| The `llm` umbrella feature would not compile alone; Bazel does not expand feature implications | R1 gains a no-backend path and explicit Bazel wiring (6.3) |
| "One function to add a format" was false: several places decide "is this a chat model" on their own | Every site is listed; classification, availability and request validation are separate (6.3) |
| Tool continuation composes a raw prompt in the executor; LiteRT-LM has no equivalent | Tools are deferred behind a spike (6.3, R6) |
| Ignoring `min_p` contradicted "differences fail closed"; raw sessions take no repetition penalty or constraint | A sampling rule on effective parameters; raw generation is not offered (6.4) |
| Bindings expose only GBNF, so a JSON Schema kept inside core would be unreachable | Structured output is deferred (9) |
| Hash-then-`dlopen` conflicts with macOS signing and is not authentication | Integrity is checked at packaging time; platform signing is the load-time control (6.5) |
| "An add-on that carries only the library" is not an installation contract | Only the Android AAR is specified (6.5); the rest is deferred with named gaps (9) |
| The AAR gate's vendor exemption skips alignment, API-level and TLS checks; Android CI runs x86_64 | A per-library manifest with every check; the published AAR includes x86_64 (6.5) |
| Telemetry would report LiteRT as CPU and its runtime as unknown | The backend reports the accelerator it selected (6.6) |

Version 1 also stated five things that were wrong: that `is_llm_template`
decides speculative loads, that the executor refuses `TfLite`, that Flutter
ships a vendored ONNX Runtime, that upstream "removed" LiteRT 2.3.2 binaries,
and that `seed` is an existing generation option. Sections 3 and 4 are
corrected.

### Version 2 to version 3 (version 2 was the narrowed proposal)

| Problem | Change |
|---|---|
| Seeing the final chunk does not prove the callback has returned; on a stall the engine must be kept alive too | Upstream's own join point is used, each callback holds its own reference, and a stall leaks the whole engine family (6.4) |
| No C API call reports the runtime release, so "qualified runtime" could not be enforced | The companion carries an identity the loader reads (6.4, 6.5) |
| Typed refusals were deferred, but the first slice already needs them: streaming turns every capability error into `InferenceError` | Moved into the first slice (6.3, R7) |
| The proposed load spec dropped the vision projector from the cache identity, which an existing regression test protects | Restored (6.3, R2b) |
| History is converted to text before validation, so a non-text message is silently dropped | Validation runs on the original input and history (6.3, R8) |
| Two more classifiers (SDK and CLI observability), a REPL path that assumes GGUF, and a Hugging Face filter that excludes `.litertlm` unconditionally | Added to the inventory (6.3, R5) |
| Companion versions tied to upstream conflict with the workspace version check, and Maven refuses redeploys | A separate version domain (6.5) |
| The gate was undecidable ("user-visible gain"), and refactoring could start on a different binary from the one that ships | Limits are set before measuring; confirmation on the shipping candidate precedes any shared-core change (5, 8) |
| The registry resolver cannot express a (model, runtime, device) qualification | Registry delivery stays deferred until it can (6.6) |
| Mapping details: `TopP` needs both top-k and top-p, penalties below 1 are clamped upstream, thinking is on by default upstream but off for Gemma in Xybrid | Rules added (6.4) |

One change considered for version 2 was not made: using `tokio::sync::mpsc`
in the bridge. The bridge joins a foreign thread to a blocked synchronous
caller in a crate with no Tokio dependency.

### Version 4 to version 5 (version 4 was the Gemma 4 update)

The conclusion stands: run the experiment, and keep the Android slice
conditional on it.

| Problem | Change |
|---|---|
| "The state bug should not apply to Gemma 4" was too strong: 28 of E2B's 35 layers use a 512-token sliding window, the runtime keeps separate local caches for them, and the upstream fix says it does not cover sliding-window caches | Gemma 4 passed as the control in two upstream reports; that is evidence, and the isolation tests are widened to settle it (4.3, 7) |
| "Resending history may be cheap" was unestablished: a new conversation starts with an empty token history, and the skip is disabled with GPU ring buffers | The first slice budgets for the whole history to be read again until measured (4.3, 5) |
| The memory comparison was unfair: llama.cpp memory-maps weights too, the Android figures leave out GPU memory, and the iOS figures do not | Memory is measured the same way for every contender; the claim that a llama.cpp GPU build "would address speed only" is withdrawn (4.3, 12) |
| Benchmark reading: "as low as 0.8 GB" is a best case, and the speculative-decoding rows used other prompt lengths | Corrected (4.3) |
| Open upstream defects on the GPU path: wrong digits past about 2,048 tokens at default precision, memory growth when conversations are recreated, memory that scales with the context limit | New section 4.4; correctness becomes stage A (5); three design rules (6.4); new tests (7) |
| "GPU needs companion libraries" was stale: since v0.19.1 the GPU backend is open source and linked into the library | Corrected, and confirmed in the shipped binary (4.1, 6.5) |
| AICore was only "recorded"; it is the main product alternative on phones that have it, with limits of its own | Measured where available (5, 11, 12) |
| The candidate limits were not decidable, and "runs on a 6 GB phone" had no 6 GB phone behind it | Rewritten (11.2) |
| Android's result should not gate a first look at iOS | An iPhone probe joins stage A (5) |

Checking these also showed that two llama.cpp facts in version 4 had been
read from the submodule's working tree, which is at a different commit from the
one the repository pins. Read at the pinned commit: Gemma 4 audio input carries
no "experimental" label, and llama.cpp already contains speculative decoding
with a Gemma 4 drafter model type and a Hexagon NPU backend (4.3).

## 3. Current architecture (verified)

### 3.1 Two execution layers

- **Orchestrator layer:** `crates/xybrid-core/src/executor.rs` keeps
  `RuntimeAdapter`s by name (`onnx`, `coreml`, `cloud`), registered in
  `orchestrator/bootstrap.rs:147-211`. For a directory with a
  `model_metadata.json` it delegates downward (`executor.rs:341-398`).
- **Metadata layer:** `execution/executor.rs`, `TemplateExecutor`, 7,280
  lines. The SDK's `XybridModel` owns one per loaded model
  (`xybrid-sdk/src/model.rs:3546`).

### 3.2 Format selects the runtime

`ExecutionTemplate` (`execution/template/metadata.rs:27`): "Variants are named
by format, not by runtime implementation". `TfLite` and `LiteRtLm` are still
variants; `LiteRtLm` is documented as legacy.

### 3.3 The LLM seam

`trait LlmBackend` (`runtime_adapter/llm.rs:180`): `load`, `generate`,
`generate_raw`, `render_chat_prompt`, the streaming and multimodal variants,
`supports_vision`, `supports_streaming`, `context_length`,
`last_cached_prefix_len`, `wire_label`.

- One live implementation, `LlamaCppBackend`. `llm-mistral` is a marker
  feature with no dependency.
- Streaming is a borrowed callback on the caller's thread,
  `Box<dyn FnMut(PartialToken) -> Result<(), StreamingError> + Send + 'a>`
  (`types.rs:158`). Returning `Err` stops generation.
  `AdapterError::from_streaming_callback_error` (`runtime_adapter/mod.rs:208`)
  turns a typed abort back into `AbortedForCloudFallback`, which starts cloud
  fallback (`xybrid-sdk/src/model.rs:824`).
- llama.cpp holds its context lock across the user's callback
  (`llama_cpp/mod.rs:304`), so re-entering the same model from a callback
  already deadlocks today.
- Backend choice is `LlmRuntimeAdapter::with_backend_hint` (`llm.rs:476`). It
  calls `Self::new()`, which exists only when a concrete backend feature is on
  (`llm.rs:436`, `:444`).
- The executor repeats "cache key, create adapter, load, cache" six times
  (`executor.rs:1939`, `2077`, `2370`, `2565`, `2681`, `2816`). The key,
  `LlmAdapterCacheKey` (`:73`), holds the model path, chat template, context
  length, backend hint and vision projector path.
- **Places that decide "is this a chat model" independently:**
  the executor's four dispatches (`executor.rs:939`, `1381`, `1588`, `1794`);
  the pipeline's streaming fast path (`xybrid-sdk/src/pipeline/mod.rs:1711`);
  `resolved_is_chat` (`registry_client.rs:2154`), which decides speculative
  loads; `ModelLoader::is_llm_template` (`model.rs:3594`), which decides
  output type and streaming dispatch; `RunSubject::from_metadata`
  (`observability.rs:532`) and the CLI's request classifier
  (`xybrid-cli/src/commands/run/observability.rs:170`), without which a model
  is reported as `predict`; and the REPL's `--model-file` path
  (`repl.rs:169`), which writes GGUF metadata. Flutter, the facade and Bolt
  forward the SDK's answer; Flutter adds its own feature gate
  (`bindings/flutter/rust/src/api/model.rs:1295`).
- The context paths convert history to text and skip anything else
  (`executor.rs:1402`, `:1813`); the guard before them rejects images but not
  audio (`:106`).
- About 155 `cfg(any(feature = "llm-mistral", feature = "llm-llamacpp"))`
  gates in 11 tracked files, 88 of them in `execution/executor.rs`.
- Tool continuation is llama-shaped: the executor renders the chat prompt,
  appends protocol text and calls `generate_raw`
  (`execution/tool_continuation.rs:115`). `MessageRole` has no tool role.
- `GenerationConfig` (`types.rs:384`): `grammar` is GBNF only;
  `with_json_schema` converts and drops the schema. There is no `seed` field.
  Defaults: `temperature` 0.0 (greedy), `min_p` 0.05, `repetition_penalty`
  1.1, applied by llama.cpp over a 64-token window
  (`llama-cpp-sys/wrapper.cpp:1325`).
- Prefix reuse is off for recurrent and hybrid models, LFM2 and Qwen 3.5
  included (`llama_cpp/mod.rs:361`).
- Xybrid turns thinking off for Gemma-family templates
  (`llama_cpp/chat.rs:300`).
- `streaming_postprocess` is gated on `llm-llamacpp` though nothing in it is
  llama-specific. Its filter assumes small chunks: the push that closes a
  reasoning block returns nothing, even if the same chunk holds answer text,
  and it does so before stop sequences are checked
  (`streaming_postprocess.rs:420-431`).
- The SDK's streaming error mapper keeps only the cloud-fallback abort typed;
  every other adapter error becomes `InferenceError` (`model.rs:470`).

### 3.4 The tensor seam

`trait ModelRuntime` (`runtime_adapter/traits.rs:9`) and the executor's
`runtimes` map. `TfLite` is routed to a runtime named `"tflite"`
(`executor.rs:927`) that nothing registers, so it fails with `Runtime 'tflite'
not configured`. The seam is narrower than it looks: `tensor_utils` handles
one `f32` input and one output (`tensor_utils.rs:29`, `:100`); the token-ID
and TTS paths, model-graph stages (`executor.rs:2958`) and autoregressive
sessions (`:3111`) all assume ONNX.

### 3.5 Capabilities, guards and telemetry

- `features::enabled()` lists compiled features and is cached for the life of
  the process (`features.rs:61`); only zzz has a run-time check.
  `ARTIFACT_BACKENDS` (`registry_client.rs:127`) filters what reaches the
  registry, whose resolver receives the platform, the OS and backend names
  and nothing else (`:139`).
- The SDK refuses `TfLite` and `LiteRtLm` at load (`model.rs:3561`) and
  excludes their files from Hugging Face downloads (`:1441-1463`). Both
  messages still say these models "run in the browser SDK", false since
  0.11.0.
- Telemetry assumes the existing runtimes: `local_execution_provider` returns
  `cpu` for an unknown backend (`llm.rs:50`), `span_kind_from_template`
  classifies LiteRT as CPU (`metadata.rs:909`), and the SDK maps unknown
  runtimes to `unknown` (`observability.rs:502`).

### 3.6 How native runtimes ship

| Runtime | Linking | Source |
|---|---|---|
| ONNX Runtime on Android, Linux, Windows | `dlopen` by bare name at run time | Kotlin AAR: vendored `libonnxruntime.so`. Flutter: a Maven dependency (`bindings/flutter/android/build.gradle:97`) |
| ONNX Runtime on Apple | static | vendored |
| llama.cpp | static; shared ggml variants in the arm64 AAR | built by Bazel from the submodule |
| zzz speech engine | static, symbols isolated | pinned prebuilt slices with a verified manifest |

A separately packaged upstream library that Xybrid opens by name is already
how Flutter gets ONNX Runtime on Android. React Native on Android depends on
the Kotlin AAR (`bindings/react-native/android/build.gradle:62`).

## 4. Upstream today

### 4.1 LiteRT-LM v0.19.1 (5 November 2026), Apache-2.0

- C++, Kotlin and Python are marked stable; Swift and JavaScript are early
  preview.
- **C API 1.0.0** in this release, versioned by semver independently of the
  runtime, with `litert_lm_get_api_version` for dynamic loading. 1.0.0 itself
  was a breaking change. `main` is at C API 1.1.0 and runtime 0.20.0.
  **No call reports the runtime release**, on 1.0.0 or on `main`; the binary
  only embeds it as a string.
- **Official prebuilt C API bundle**, `litert_lm_c_api-1.0.0.zip` (SHA-256
  `1d430d80eaf536f696a4922270ba55eb3d112189ac198c3066d487a45e3b4d4c`).
  **Verified by inspecting the binaries:**

  | Slice | Size | Properties |
  |---|---|---|
  | Android arm64 `liblitert-lm.so` | 22.8 MB | exports exactly 220 `litert_lm_*` symbols; needs only system libraries, so libc++ is static; LOAD segments aligned to 16 KB; no `PT_TLS`, no TPREL relocations; built for API 24 with NDK r29; imports `dlopen` |
  | Android x86_64 | 26.9 MB | same dependency set |
  | macOS arm64 `.dylib` | 51.0 MB | exports exactly 220 `litert_lm_*`; `minos 14.0`; ad-hoc signed |
  | Linux x86_64, arm64 | 85 MB each | not inspected |
  | Windows x86_64 | 53 MB | not inspected |

  iOS is a separate dynamic framework (48 MB device slice, iOS 15.0) that
  exports 52,326 symbols.
- **Known defect in this release (verified in the source).** `Reset()` in
  `runtime/executor/llm_litert_compiled_model_executor.cc` resets counters but
  not state buffers. Upstream commit `e1f86b05` (3 November, "Start a new
  sequence from a zeroed state") fixes it on `main` and is not in the v0.19.1
  tag. Until a release carries it, a new conversation on a shared engine
  continues from the previous one for any model with state outside the KV
  cache. The `litert-community/LFM2.5-230M` card tells users to create one
  engine per conversation for this reason. A model whose only state is a
  position-masked KV cache is unaffected; whether Gemma 4's sliding-window
  layers qualify is taken up in 4.3.
- **Liquid conversions are community work with listed defects:** a chat
  template that does not run on the runtime's Jinja engine, metadata added
  after export, a separate GPU export, and a GPU weight-loading issue on
  Android (model cards for `LFM2.5-230M` and `LFM2.5-1.2B-Instruct`).
- **Threading contract**, from `engine.h`: handles are not thread-safe; stream
  callbacks run on library-owned threads shared by all streams of an engine
  (in single-threaded mode, only while the caller drives
  `conversation_wait_until_done`); a callback may call nothing on its engine
  except `*_cancel_process` on a non-final chunk; cancel only requests, and
  callbacks can still run after it returns; a cancelled handle must only be
  deleted; the last error is thread-local; errors after a stream starts arrive
  in chunks, whose text may be NULL; a chunk's text, error and finality are
  separate accessors. The header says these describe current behaviour, not a
  commitment.
- **Where callbacks are joined (verified in the v0.19.1 source, not promised
  by the header).** `Conversation::WaitUntilDone` "simply blocks until the
  pending work completes" with multi-threaded execution, and
  `~SessionAdvanced()` calls `WaitUntilDone()` before releasing the session
  (`runtime/core/session_advanced.cc:487`). So waiting on, or deleting, a
  conversation blocks until its callbacks have finished. Neither takes a
  timeout.
- **API facts that shape the mapping:** sampler types `TopK`, `TopP` (top-k
  first, then top-p) and `Greedy`; no `min_p`; repetition penalty, constraints
  and the per-request output budget attach to a **conversation** only, not to
  a raw session; a repetition penalty below 1 is clamped to 1; no stop
  strings; the engine's token capacity can differ from the value requested and
  reads as 0 before the first prefill; a thinking configuration exists and is
  separate from the output text; `litert_lm_loaded_file_*` reads a file's
  context size, modalities, supported backends and minimum runtime version
  without creating an engine.
- **The GPU backend is inside the library.** Maintainers state that since
  v0.19.1 the GPU accelerators are built from open source (ML Drift,
  Apache-2.0) and linked into the runtime (issue #1050, 11 November). The
  prebuilt Android library bears that out **(verified with `strings`)**: it
  contains the ML Drift and OpenCL code and opens the phone's own
  `libOpenCL.so` at run time. Upstream's Android guide accordingly requires
  the app manifest to declare `libOpenCL.so` and `libvndksupport.so` with
  `<uses-native-library>`. The library also looks for optional plugins by
  name: a GPU sampler (`libLiteRtTopKOpenClSampler.so`) and the Gemma
  constraint provider, which only structured output needs. Whether the GPU
  sampler matters for speed is a stage A question.
- Upstream's published figures for Gemma 4 are in section 4.3, and its open
  defects on the GPU path in section 4.4.

### 4.2 LiteRT 2.3 (tensor runtime) and JavaScript

`libLiteRt` has an ABI-stable C API by design (an append-only function table
and a runtime ABI version). Prebuilt binaries are published for 2.3.0 (Android
arm64 5.5 MB plus a 3.3 MB GPU accelerator) and are absent at the same path
for 2.3.2. `@litertjs/core` is at 2.6.0 and `@litert-lm/core` at 0.19.0 (47 MB
and 123 MB unpacked). Neither is used by the first slice.

### 4.3 Gemma 4, the chosen model

**In Xybrid today (verified).** `gemma-4-e2b` and `gemma-4-e4b` are listed
models. In the test fixture and the vision guide, E2B runs through llama.cpp
as a `VisionLanguage` bundle: a Q8 GGUF (4,967 MB) plus a Q8 vision projector
(557 MB), 5.5 GB in all, with `ram_minimum_mb: 5632` declared in its metadata.
The guide calls it a "heavyweight correctness target" for "8 GB RAM, iPhone 15
Pro+ class mobile devices". It takes text and images, speaks Gemma's tool
protocol, runs on CPU on Android and on Metal on iOS, has thinking turned off,
and reuses a shared prompt prefix between text turns. Xybrid uses neither
audio input nor speculative decoding for it. Which quantization the public
registry serves was not checked.

**On LiteRT-LM (Google's figures, not measured here).**
`litert-community/gemma-4-E2B-it-litert-lm` is the conversion Google publishes
and benchmarks: Apache-2.0, not gated, about 1.4 million downloads in the last
30 days, last updated 11 November. One 2,588 MB file holds text, vision and
audio; a GPU-oriented variant of 2,341 MB sits beside it. It uses a mixed 2-,
4- and 8-bit quantization. The card says the text-only weight footprint "can
be as low as 0.8 GB", with 1.12 GB of embeddings memory-mapped and the vision
and audio encoders loaded on demand. E4B is 3,660 MB.

| Device, backend | Prefill, tokens/s | Decode, tokens/s | Time to first token | Memory as reported |
|---|---|---|---|---|
| Galaxy S26 Ultra, CPU | 557 | 46.9 | 1.8 s | 1,733 MB |
| Galaxy S26 Ultra, GPU | 3,808 | 52.1 | 0.3 s | 676 MB |
| iPhone 17 Pro, CPU | 532 | 25.0 | 1.9 s | 607 MB |
| iPhone 17 Pro, GPU | 2,878 | 56.5 | 0.3 s | 1,450 MB |

Conditions: 1,024 prompt tokens, 256 generated, a 2,048-token context, caches
warm, load time excluded. Three cautions on reading it:

- The Android memory column is CPU memory (peak resident size), which leaves
  out what the GPU holds. The iOS column is labelled "CPU/GPU" and uses a
  different measure. The platforms cannot be compared with each other, and
  the Android GPU figure is a lower bound.
- A separate table reports speculative decoding on the S26 Ultra GPU at 66 to
  92 tokens per second against a 51.5 baseline, on prompts of varying length,
  so it is not comparable with the rows above.
- The benchmark never goes past 1,280 tokens of context. The GPU path starts
  to give wrong answers at about 2,048 (section 4.4).

What this means for the proposal:

- **Model fit is no longer the weak point.** This is the family LiteRT-LM is
  built around, and the licence allows redistribution.
- **Download size favours LiteRT-LM modestly.** Its 2,588 MB includes vision
  and audio. On llama.cpp a Q4_0 GGUF is 2,841 MB for text alone and 3,399 MB
  with a vision projector; Google's quantization-aware Q4_0 build is 3,350 MB
  plus a 987 MB projector. Against the 5.5 GB Q8 bundle, most of the saving
  comes from the quantization, which the existing runtime can have too.
- **Memory is unknown until measured the same way.** llama.cpp memory-maps
  its weights as well, Xybrid's 5,632 MB is a declared minimum and not a
  measurement, and Google's Android figure leaves out GPU memory.
- **Speed is where LiteRT-LM may lead**, on GPU, and only as far as section
  4.4 allows.
- **Whether state can leak between conversations is not settled.** Gemma 4
  passed as the control in two upstream reports of the state bug (issue
  #3165). But 28 of E2B's 35 layers use a 512-token sliding window
  (`config.json` of `google/gemma-4-E2B-it`), the runtime keeps separate
  local caches for such layers (`runtime/executor/litert/state.cc`), and the
  fix commit says sliding-window caches and rewinds to a later position are
  not covered. Section 7 tests those cases directly.
- **Plan for the whole history to be read again on each call.** v0.19.1 can
  skip prompt tokens it has already processed, but a new conversation starts
  with an empty token history, and the skip is disabled when GPU ring buffers
  are in use (`resource_manager.cc`). Stage A measures what a fifth turn
  costs.
- **Audio is not a LiteRT-only feature.** The pinned llama.cpp lists Gemma 4
  with image and audio input. What Xybrid lacks is an audio part in its
  message type, for either runtime.
- **llama.cpp has more to offer for this model than Xybrid uses.** At the
  pinned commit it has an OpenCL backend with Adreno 750 to 840 listed as
  verified, a Hexagon NPU backend for Snapdragon, and speculative decoding
  that includes a Gemma 4 drafter model type. None is built into Xybrid:
  Android is CPU only, and of llama.cpp's `common/` directory, where
  speculative decoding lives, Xybrid compiles only the template code. Each is
  integration work, and none was tried here.
- **Google points production Android apps to AICore.** The model card says
  that on supported devices Gemma 4 is available through Android AICore as
  Gemini Nano, "the recommended path for production applications". It does
  not say which Gemma 4 variant. The developer API is in beta, limits input
  to under 4,000 tokens, applies a per-app quota, and reports the model per
  device as available, downloadable or unavailable. There is no runtime or
  model to ship, and no control over the model version or device coverage.

### 4.4 Open upstream defects on the GPU path

All are open issues in the LiteRT-LM repository, reported by users. None was
reproduced here. Maintainers have replied on all but the last, and on the
first say they reproduced it.

| Defect | Reported on | What it means here |
|---|---|---|
| **Wrong digits past about 2,048 tokens** at the default half-precision activations: copied numbers, dates and IDs come back scrambled while prose reads fine, and one report shows tool calls failing the same way. CPU is correct. (#4001, #3913, #2814) | Adreno 740 and 840, Mali-G715, iPhone and Mac (Metal), Intel Arc; E2B and E4B; releases 0.13 to 0.19.1, the last through the C API | The default GPU configuration cannot ship. Full-precision activations fix the reported cases, at a cost given as 39% more wall time (E2B, iPhone 15 Pro), a first token in 0.66 s against 0.47 s with about 6.9 GB against 5.2 GB of GPU memory (E4B, Adreno 840, 8,192-token limit), or 640 MiB more (E2B, Mac). Google's figures in 4.3 are at half precision |
| One answer still wrong on Metal at full precision, on a short prompt (#3913) | iPhone 15 Pro, E2B, 0.17.1 | A second, unexplained difference on iOS |
| **Memory grows when conversations are closed and recreated**: 67 MB over 49 cycles (#4567) | Android GPU, E2B, 0.19.1 | The first slice creates a conversation per call (6.4), so it would meet this on every turn |
| GPU memory for the context is reserved for the full token limit on every layer, sliding-window layers included: about 2.1 GB more at a 32K limit than at 4K (#4096) | Immortalis-G925, E4B, 0.18.0 | The engine's token limit must be the context Xybrid uses, not what the model supports |
| The general file costs 31 s more start-up and 3.3 GB more GPU memory than the `-gpu` file (#4096) | same | Which file to ship per backend is a stage A question |
| Speculative decoding slows decode from 8.7 to 3.0 tokens per second (#2227) | Pixel 10 Pro (PowerVR), E2B, 0.11.0 | Old and unconfirmed on current releases. Decode on that phone was a sixth of the published S26 Ultra figure even without it, so results depend heavily on the GPU vendor |

Two reporters ran the same weights as GGUF through llama.cpp on the same
phones and got exact output: E4B on the OpenCL backend up to 8,151 tokens
(Adreno 840), and E2B on the Hexagon backend up to 59,750 tokens (Snapdragon 8
Gen 2). Those are their measurements, and they say nothing about speed.

Reporters trace the first defect to position-encoding precision in the
published model files and ask for a re-export
(`google-ai-edge/litert-torch` #1343). Until then the remedy is
`litert_lm_engine_settings_set_activation_data_type`, which the C API exports.

## 5. The experiment that decides (phase 0)

No Xybrid code changes. Everything runs from an ordinary signed APK, because
vendor GPU libraries behave differently under `adb shell`, and an app that
targets API 31 or later must declare `libOpenCL.so` in its manifest to use it.

**Model:** Gemma 4 E2B. LiteRT-LM runs the general file (2,588 MB) and the
`-gpu` file (2,341 MB). llama.cpp runs the Q8 files Xybrid uses today, a Q4_0
GGUF (2,841 MB) and Google's quantization-aware Q4_0 (3,350 MB), so that a
smaller quantization is not mistaken for a runtime gain.

**Set before measuring** (decision 11.2): the limits, the scoring method and
how a mixed result is read. An inconclusive or mixed-device result counts as
stop unless decided otherwise in advance.

**Stage A: is LiteRT-LM correct and stable?** One Android phone, and a short
run of the same tests on an iPhone. Nothing here is a speed test.

- *Answers.* A copy test (numbers, dates and IDs repeated back) at about
  1,500, 2,500, 4,000 and 8,000 tokens of context, on CPU, on GPU at the
  default precision and on GPU at full precision. This finds which
  configuration, if any, is correct on that phone (4.4).
- *Isolation.* The same conversation on a fresh engine and after other
  conversations, including ones longer than the 512-token window, a
  cancelled one, a rewound one and one that shares only its opening (4.3).
- *Lifecycle.* Memory after 200 create-and-delete cycles of a conversation
  (#4567). Whether cancellation reaches quiescence and the next inference
  then succeeds. Whether a wait or a delete joins a callback that is still
  running.
- *Shape.* What stream chunks contain (plain text, or JSON with separate
  reasoning and tool channels). How token counts and the finish reason are
  obtained. Which of the two files starts faster and uses less memory on GPU.
  Whether the optional GPU sampler plugin is needed. What a fifth turn costs
  when the history is resent.

Stop here if no GPU configuration is correct up to the context Xybrid uses
(4,096 tokens in today's Gemma 4 bundle), or if memory grows without bound.
Every later number for LiteRT-LM is taken in the configuration that passed.

**Stage B: is it better than the alternatives?** Same phone, same prompts,
same context limit, every contender held to the same correctness tests.

1. Xybrid as documented: llama.cpp on CPU with the Q8 files.
2. The same with a Q4_0 GGUF, which is only a registry change.
3. llama.cpp with a GPU backend: OpenCL on Adreno (the pinned
   `docs/backend/OPENCL.md` lists Adreno 750 to 840 as verified), Vulkan
   elsewhere. It costs build work in `bazel/ggml`, the AAR's backend loader
   and the gates, but no second format, registry variant or change in
   semantics.
4. AICore's Prompt API, where the phone offers it: latency and answer quality
   on the same prompts, within its 4,000-token input limit.

Stop if a cheaper contender meets the limits.

**Stage C, only if the result can still change.** A second phone with a
different GPU vendor, since reports differ widely by vendor (4.4). LiteRT-LM's
speculative decoding. llama.cpp's Hexagon backend on Snapdragon.

**iOS** gets the stage A tests early, so that a problem there is known before
Android work starts. The full comparison, against Xybrid's llama.cpp on Metal
on the lowest iPhone the Gemma 4 path claims today (iPhone 15 Pro), decides
the second slice and can wait.

**The workload** measures cold start including shader compilation, time to
first visible token on the first and the fifth turn, sustained decode over
several minutes, peak memory and energy. Memory is taken the same way for
every contender: process memory plus graphics memory on Android, physical
footprint on iOS. The harness reproduces each runtime's real cache policy:
Xybrid reuses a shared prompt prefix for Gemma between text turns, and
LiteRT-LM is assumed to read the whole history again until stage A shows
otherwise.

**Confirmation.** Before any shared-core change, stage A and the headline
numbers are repeated on the exact runtime release that would ship, which must
contain `e1f86b05`.

## 6. Architecture for the first slice

### 6.1 Principles

1. **Format selects the runtime.** No hint string, no new `run*` method.
2. **One seam.** LiteRT-LM implements `LlmBackend`. No LiteRT type appears
   above `runtime_adapter/`.
3. **A C ABI boundary, opened at run time.** Never linked statically: the
   library carries its own Rust runtime (`tokenizers` 0.21 against our 0.23),
   abseil and protobuf.
4. **Absent is not broken.** Without the library a build starts normally, does
   not advertise the capability and returns a typed error for a LiteRT model.
5. **Contracts fail closed.** A request the backend cannot honour is refused
   with `UnsupportedBackendCapability` before generation, and that error
   reaches the caller typed.
6. **Qualify exact combinations.** A (model, runtime release, device class)
   triple is supported when it passes section 7, not before, and nothing is
   advertised more broadly than what was qualified.

### 6.2 Units

```
crates/xybrid-litert-lm/                 NEW  loader, private `sys` declarations,
                                              safe Engine and Conversation, the
                                              stream operation. No Xybrid deps.
crates/xybrid-core/src/runtime_adapter/litert_lm/
                                         NEW  impl LlmBackend   [feature llm-litert]
bindings/kotlin (companion)              NEW  AAR that carries the upstream library
```

One crate, not a `-sys` pair: nothing is linked at build time, so there is no
build script or `links` key for a `-sys` crate to own. Declarations are
written by hand against the pinned headers, as in `zzz-sys`. The crate joins
the crates.io publication order ahead of `xybrid-core`
(`release-publish.yml:314`).

### 6.3 Changes to the core, in order

- **R0, independent.** Fix the stale "run in the browser SDK" messages and
  docs.
- **R2a, independent.** One `ensure_llm_adapter` in the executor replaces the
  six copies, keeping today's cache identity exactly, vision projector
  included (the regression test at `executor.rs:5321` protects it).
- **R1. An `llm` umbrella feature**, enabled by each backend feature. It must
  compile with no backend: `with_backend_hint` gets a path that returns a
  typed "no LLM backend in this build" error. Every Bazel `crate_features`
  list is edited by hand, because Bazel does not expand implications
  (`crates/xybrid-cli/BUILD.bazel:28`). `asr-whispercpp` keeps its dependency
  on `llm-llamacpp`; decisions and codec TTS stay gated on `llm-llamacpp`.
- **R2b. One description of an LLM load**, which is the cache identity:

  ```rust
  pub(crate) enum LlmBackendKind { LlamaCpp, Mistral, LiteRtLm }

  pub(crate) struct LlmLoadSpec<'a> {
      pub backend: LlmBackendKind,
      pub model_file: &'a str,
      pub chat_template: Option<&'a str>,
      pub vision_encoder: Option<&'a str>,     // existing cache identity
      pub context_length: Option<usize>,       // None: the file's own limit
      pub accelerator: AcceleratorPreference,  // Auto | Cpu | Gpu
  }
  ```

  It is how the executor's four dispatches stop matching `Gguf`. llama.cpp's
  tuning fields stay where they are; nothing is relocated for this slice.
- **R5. Three questions, three functions**, replacing the scattered matches
  in 3.3:
  - *Is this a chat model?* `ExecutionTemplate::is_chat()`, used by
    `is_llm_template`, the pipeline fast path, `resolved_is_chat` and both
    observability classifiers. The bindings keep forwarding; only Flutter's
    feature gate changes.
  - *Can this build run it right now?* A core function that checks the
    feature and the library's availability. The SDK load guard and the
    Hugging Face filter ask it, instead of refusing by format.
  - *Can this backend honour this request?* See R8.
  The REPL's `--model-file` stays GGUF-only; the first CLI slice takes a model
  directory.
- **R4. An availability probe** for LiteRT-LM in `features.rs`. Discovery is
  decided once, at first use, and frozen, because the list is cached; a host
  that configures a library path must do so before then. Documented and
  tested.
- **R7. Typed refusals reach the caller.** `streaming_execution_error`
  (`model.rs:470`) keeps `UnsupportedBackendCapability` and
  `UnsupportedModelCapability` typed. The facade already has the variants
  (`xybrid-ffi-facade/src/lib.rs:309`), so no binding surface is added.
- **R8. Validate before converting.** The executor checks the original input
  and history against what the backend supports before turning them into
  `ChatMessage`s, so a non-text message is refused, not dropped. This closes
  an existing gap for llama.cpp too (audio in history).
- **R6, not in the first slice. Tool continuation behind the seam.** A trait
  method that takes history and tool results as data. Its default is today's
  raw-prompt composition, so llama.cpp does not change; LiteRT-LM would
  override it with structured messages. It must be spiked against the C API
  before tools are attempted, because it tests whether `LlmBackend` fits a
  conversation engine at all.

### 6.4 LiteRT-LM backend

**Scope of the first slice:** text chat, streaming and batch, system prompt,
multi-turn by resending history, cancellation, stop sequences, greedy and
top-k/top-p sampling, repetition penalty. Everything else is refused with a
typed error: raw prompts, GBNF and JSON Schema, tools, images and audio,
decisions.

**Three rules that follow from upstream's open defects (4.4).**

- *Precision.* On GPU the backend asks for full-precision activations. Half
  precision is allowed only for a (model, device class) pair that passes the
  copy test at its full context limit.
- *Context.* The engine's token limit is the context Xybrid will use, never
  the model's maximum, because GPU memory is reserved for all of it.
- *Conversation lifecycle.* "A new conversation per call" stands only if
  stage A shows that creating and deleting conversations does not grow memory
  on the pinned library. If it does, the slice needs one of two things before
  it ships: conversation reuse (below) brought forward, or an engine that is
  rebuilt after a set number of conversations.

**Loading and identity.** The library handle is an explicit `Arc<Library>`
owned by every engine made from it; a process-wide cache is an optimization
that correctness does not depend on. On Android it is opened by bare name, as
ONNX Runtime is, and resolves inside the signed app package. The library is
never unloaded, and single-threaded execution mode is never enabled.

Three checks, because the C API version says nothing about the release:

1. `litert_lm_get_api_version` reports major 1 and a sufficient minor.
2. The companion's identity (6.5) names a runtime release that this SDK
   version has qualified. This is what guarantees the state fix is present.
3. `loaded_file_min_runtime_version`, when the model declares one, is not
   newer than that release.

**Request mapping.**

| Xybrid | LiteRT-LM | Rule |
|---|---|---|
| chat turn | new conversation: system message, history, then `send_message_stream` | no state is reused between calls in this slice |
| `temperature == 0` | sampler type `Greedy` | |
| `temperature > 0` | `TopP`, carrying both `top_k` and `top_p` | disabled-filter boundaries (`top_k = 0`, `top_p = 1`) are qualified, not assumed |
| `repetition_penalty >= 1` | penalty config on the conversation, window 64 | matches llama.cpp's window |
| `repetition_penalty < 1` | clamped upstream | refused |
| `min_p` | none | irrelevant under greedy, so default requests pass; **with `temperature > 0` and `min_p > 0` the request is refused**, with a hint to set `min_p` to 0 |
| thinking | thinking configuration | set explicitly from the model's reasoning setting, off otherwise, because upstream's default is on and thinking spends the output budget |
| `max_tokens` | per-request output budget on the conversation | |
| context limit | engine capacity, read back after load | Xybrid's contract is kept: an oversize prompt is refused and the output budget clamped. How to do that before the first prefill is a stage A question |
| stop sequences | none | Xybrid's filter, then cancel |
| batch `generate` | the stream operation with a no-op consumer | same stop and limit behaviour as streaming |
| token counts, finish reason | upstream's counters | chunk count is not a token count. After a cancel only "delete" is allowed, so a cancelled run reports the bridge's count, marked approximate. Telling end-of-sequence from the budget is a stage A question; the slice does not ship until it is answered |

The filter's small-chunk assumption (3.3) is fixed in the shared code before
this backend uses it. That changes llama.cpp's path too, which is what the
parity suite is for.

**The stream operation.** One object owns a stream from start to quiescence.

- *Shared state* lives in an `Arc`. One strong reference is handed to the
  library as the callback data. **Each callback invocation takes its own
  strong reference on entry and drops it on exit**, so the state outlives any
  invocation still on the stack. It holds a queue with its own lock and
  condition variable, and atomics for `cancel_issued`, `final_seen` and
  `callback_failed`.
- *Trampoline* (`extern "C"`, body in `catch_unwind`): copy the chunk into an
  owned event carrying text, error and finality **independently**, with a
  timestamp, and push it. It calls no LiteRT function and takes no lock but
  the queue's. A caught panic sets `callback_failed`. The queue is bounded by
  the request's output budget; a chunk beyond it is a protocol violation and
  ends the operation with an error.
- *Caller side*: holds the engine's lock for the whole operation, as
  llama.cpp does, and consumes events on its own thread with an inactivity
  deadline. The queue lock is released before any user code runs. Text goes
  through the filter, then to the user's callback. A callback error is
  **stored as the original error**, delivery stops, and after teardown it
  goes through `from_streaming_callback_error`, so a cloud-fallback abort
  stays typed. A stop sequence ends as a normal stop; an error event ends as
  a native error.
- *Cancel* is issued at most once (compare-and-exchange on `cancel_issued`)
  and only by the caller's thread, when it decides to stop early. No other
  call on that conversation is in progress at that moment, since the stream
  call has returned.
- *Teardown*: a fallible `finish()` that reports teardown errors, with a
  non-panicking `Drop` as the fallback so a panicking user callback still
  tears down. It cancels if the stream is not final, then calls
  `conversation_wait_until_done`, then deletes the conversation, then drops
  the callback-data reference. The wait and the delete are the join: upstream
  blocks there until callbacks have finished (4.1). That is source-level
  behaviour of the pinned release, so section 7 tests it on the real binary.
- *If the stream stalls* (no event before the deadline, and no final chunk
  within a grace period after cancel), nothing is waited on or deleted, since
  neither call has a timeout. The conversation, the shared state **and a
  reference to the engine** are leaked, and the backend is marked unusable: it
  refuses further requests and its unload leaks the engine instead of
  destroying it with a live child. A leak is recoverable; a use-after-free is
  not.
- A start failure is read from the thread-local last error on the same
  thread, immediately.

**Session reuse** is not in this slice unless the lifecycle rule above forces
it. When it is designed, the cache key
must cover everything bound at conversation creation (sampler, tools, system
message), and a stop found after generation finished must invalidate the
conversation, since its hidden suffix is already in the state.

### 6.5 Packaging: one Android companion AAR

- **Artifact.** `ai.xybrid:xybrid-litert-lm`, containing for `arm64-v8a` and
  `x86_64`: `liblitert-lm.so`, any optional plugin stage A shows is needed (the
  GPU backend itself is inside the library), a small identity library (below)
  and every applicable licence file. No JNI code, no
  Kotlin API. `armeabi-v7a` has no slice, so the capability is absent there.
  Its `minSdk` is upstream's (24), so it raises nobody's floor: Kotlin stays
  at 29 and Flutter at 28. Publishing `x86_64` means the emulator gate loads
  the artifact that ships.
- **Manifest.** The companion's `AndroidManifest.xml` declares `libOpenCL.so`
  and `libvndksupport.so` with `<uses-native-library android:required="false">`,
  which upstream requires for GPU, so that manifest merging carries them into
  the app. The consumer test checks the merged manifest.
- **Identity.** The companion includes `libxybrid_litert_lm_info.so`, built by
  us, exporting one data symbol: the runtime release, the upstream tag and the
  SHA-256 of each library beside it. The loader opens it by bare name. It is
  the only way to know which release was loaded (4.1).
- **Consumers.** One Gradle dependency in the app serves all three Android
  SDKs. Kotlin and React Native use the Kotlin AAR's native library; Flutter
  has its own, so `llm-litert` is compiled into both once the slice is
  qualified, and Flutter's precompiled binaries are rebuilt under the
  crate-hash rule (`bindings/flutter/rust/Cargo.toml:3`). The core AAR is
  otherwise unchanged; the 22.8 MB library stays out of it.
- **Versions.** The companion has its own version domain, `<runtime
  release>-<packaging revision>`, outside the workspace version check, with
  its own check. It is published only when its content changes, because Maven
  Central refuses a redeploy (`release-publish.yml:494`). Each SDK release
  states the companion versions it accepts.
- **A per-library manifest**, not the existing vendor exemption (which returns
  before the alignment, API-level and TLS checks,
  `verify_zzz_android.py:630`): origin, runtime and source identity, SHA-256
  before and after packaging, `DT_NEEDED` set, export allowlist, 16 KB LOAD
  alignment, API level, no static TLS, and the exact notice paths and digests
  that must survive into the APK, for the main library and each plugin shipped
  beside it.
- **Consumer tests** on the published AAR, per binding and through each
  binding's real entry path with no manual preload: zip alignment, native
  libraries not extracted, a missing companion giving the typed
  "unavailable" error, one generation on the `x86_64` emulator, and a 16 KB
  page-size image. An arm64 device run with GPU is the qualification step.
- **Supply chain.** The pinned upstream assets are verified by SHA-256 when
  the AAR is built and mirrored in our own storage, with an anonymous
  retrieval check like `natives-verify-anon.sh` and an owner for retention,
  requalification and withdrawal. Load-time trust is the APK signature.
  Building `//c:litert-lm` from the pinned tag in an isolated job stays
  available as the repair path if an upstream binary is unusable; its cost is
  unmeasured.
- **Release integration.** The new checks are named and wired into the gates
  that already block `master` and draft creation; the Gradle-produced Maven
  artifact is gated like the Kotlin one; upstream updates get an owner and,
  later, automation shaped like `llamacpp-update.yml`.

### 6.6 Registry and telemetry

- The backend reports the accelerator it actually selected; that replaces the
  name lookup in `local_execution_provider`. `span_kind`, the SDK's runtime
  mapping and its allowlists learn `litertlm`.
- **Registry delivery stays deferred.** The resolver sends a platform, an OS
  and backend names. It cannot express a GPU class or a runtime release, so
  it cannot keep distribution inside what was qualified: two phones that pass
  must not become support for every Android device with the companion
  installed. Until `xybrid-platform` has a compatibility predicate for that,
  LiteRT models load from a local path or Hugging Face, and no capability
  name is added to `ARTIFACT_BACKENDS`.

## 7. Qualification and tests

- **Scripted fake library**, compiled against the official headers, with
  scheduling under test control: a callback before the start call returns; a
  final chunk with no text; a final chunk with text; an error chunk; a chunk
  carrying text and an error; callbacks after cancel; a panicking consumer; a
  stream that never finishes; a consumer that rejects the last chunk.
  Assertions: destruction order, no LiteRT call from a callback, one cancel
  at most, the typed cloud-fallback error preserved, the engine kept alive on
  a stall. Run under sanitizers. It proves the Rust side only.
- **Real-library join test** on the pinned binary: a callback that blocks
  after the final chunk, with wait and delete called from the owner; they must
  not return first. Also unload after a stall.
- **Backend parity suite** over every `LlmBackend`: batch, streaming, cancel,
  stop sequences including split matches, partial prefixes and a stop inside
  reasoning, limits and the context boundary, multi-turn, refusals. None
  exists today.
- **Isolation test** per model on the real library: the same conversation on
  a fresh engine and after other conversations gives the same output under
  greedy sampling. The earlier conversations include one longer than the
  model's sliding window, a cancelled one, a rewound one and one that shares
  only its opening tokens.
- **Copy test** for every qualified combination of model, device class,
  backend and precision: numbers, dates and IDs repeated back exactly at
  every context length up to the limit (4.4).
- **Lifecycle soak**: memory after 200 conversations created and deleted on
  one engine stays within an agreed bound.
- **A real-model test that cannot skip silently** in the lane that claims
  qualification.
- **Two engines** in one process, and LiteRT-LM beside llama.cpp.

## 8. Phases

| Phase | Outcome | Depends on |
|---|---|---|
| R0, R2a | Stale messages fixed; adapter loading deduplicated | nothing |
| 0a | Decision 11.1 is made (Gemma 4 on mobile); 11.2 to 11.5 remain | nothing |
| 0b | The staged experiment, with a written result: stage A (correct and stable?), then stage B (better than the alternatives?) | 0a |
| 0c | Confirmation on the release that would ship | 0b says proceed; a release containing `e1f86b05` |
| 1 | R1, R2b, R4, R5, R7, R8 | 0c |
| 2 | `xybrid-litert-lm`, the first-slice backend, fake-library and parity suites, CLI on a model directory | 1 |
| 3 | Companion AAR, identity library, manifest, consumer tests, GPU on a device | 2 |
| 4 | Registry variant and capability name | 3 qualified, and a compatibility predicate in `xybrid-platform` |

## 9. Deferred, and what each needs first

| Item | What it needs |
|---|---|
| Tools | R6 spiked against the C API; a choice between LiteRT's tool API and Xybrid's text protocols. Today's GGUF Gemma 4 has tools, so this is parity work, not a new feature |
| JSON Schema and regex output | a portable constraint type across core, facade, Bolt, Flutter and React Native (the parity test will require it), and cloud projection of constraints (`model.rs:1667` omits them) |
| Images | next after text, because today's GGUF Gemma 4 is used as a vision model. The executor's multimodal dispatch is tied to `VisionLanguage`; in a `.litertlm` file the encoders are inside the one file |
| Audio | an audio part in the multimodal message type (`types.rs:243`), for either runtime |
| Session reuse | the cache rules in 6.4. It moves into the first slice if conversations cannot be recreated without memory growth (4.4) |
| Liquid-family models | the isolation test passing on a fixed runtime, and an owner for the conversions |
| iOS (the second slice) | its own comparison with llama.cpp on Metal (5); an explanation for the Metal answer that full precision does not fix (4.4); proof that a framework exporting 52,326 symbols coexists with the static ONNX Runtime; signing and App Store size (48 MB device slice) |
| macOS, Linux, Windows | a signing story (re-signing changes the bytes; Hardened Runtime restricts foreign libraries), restricted DLL search on Windows, the glibc floor against the release check's 2.31 cap, an install location the loader searches |
| Swift, React Native iOS, Flutter beyond Android, Python, Unity packages | one installation contract each: React Native on iOS uses CocoaPods, not SPM; a Python extra cannot add files to the existing wheel, so it is a separate distribution; Unity needs `.meta` files and framework embedding |
| `.tflite` tensor runtime | a typed tensor contract beyond one `f32` input and output, and the ONNX assumptions in 3.4 removed. ONNX Runtime already covers these models, EmbeddingGemma included (`integration-tests/tests/embedding_parity.rs`) |
| Browser | a reason that outweighs #620's choice of one runtime |
| NPU | per-chipset model files and registry variants |

## 10. Risks

1. **The premise may be false.** A smaller GGUF, or llama.cpp with a GPU, may
   be enough. Phase 0 exists to find out cheaply.
2. **The GPU path gives wrong answers today at its default settings**, and
   the remedy costs some of the speed and memory that justify the work (4.4).
3. **Upstream correctness more broadly.** A state bug whose fix is
   unreleased, memory growth across conversations, and a runtime still at
   0.x. More such defects are likely, and each needs a test of ours to catch.
4. **Results depend on the GPU vendor.** The published figures are from one
   Adreno phone and one iPhone; reports from Mali, Immortalis and PowerVR
   phones differ widely (4.4). Two phones that pass do not make Android
   supported.
5. **The teardown relies on upstream behaviour the header does not promise.**
   It is tested on each pinned binary, and a change upstream would be caught
   there, not by the compiler.
6. **The first slice does less than today's Gemma 4.** Text only, where the
   GGUF path has images and tools. Until the LiteRT path catches up, a user
   chooses between the two.
7. **Google recommends AICore, not a bundled runtime, for production Android
   apps** on devices that have it. Part of the audience for "Gemma 4 on
   Android" may be better served there.
8. **Liquid models stay out.** Their conversions are community work, and the
   state bug affects them.
9. **Two formats per model**, with a separate GPU export for some.
10. **A small team.** 111 commits by two people since 1 October, and release
    workflows of 2,683 lines. A second runtime needs a named owner.
11. **Behaviour differs between backends** even after the rules in 6.4.
12. **Churn at Google.** MediaPipe's LLM API is already in maintenance with
    LiteRT-LM as its successor.

## 11. Decisions for a maintainer

1. **Use case and model. Decided on 15 November 2026: Gemma 4 on mobile.**
   This document starts with E2B.
2. **Limits.** Candidates to confirm or replace. They are a starting point,
   not derived from a stated product requirement.

   *Test conditions.* A 4,096-token context (today's Gemma 4 bundle), prompts
   of 1,024 and 3,000 tokens, 256 tokens generated, greedy sampling, ten runs
   each, median and 95th percentile reported.

   *Floors.* Every contender must meet them, and none is traded for speed:
   - every value in the copy test exact, at every prompt length;
   - conversation isolation passes;
   - no crash in 200 consecutive conversations, and memory growth across
     them under an agreed bound (candidate: 50 MB);
   - answer quality on a fixed prompt set no worse than today's Q8 path, by a
     scoring method and tolerance agreed before the run.

   *The win condition,* against the best llama.cpp configuration that meets
   the floors: time to first visible token (95th percentile) at most half as
   long, or peak memory at least 1 GB lower, measured the same way. Without
   that margin the cheaper alternative wins.

   *Reported but not decisive:* sustained decode after five minutes, cold
   start with shader compilation, energy, and download size for text alone
   and with vision.

   Version 4 also proposed "runs on a 6 GB phone" and "download under 3 GB".
   The first means nothing without a 6 GB phone in the test set. The second
   was arbitrary and would have ruled out Google's own GGUF build before it
   was measured.
3. **Devices and budget** for phase 0. Suggested: one recent Snapdragon
   phone, one Pixel (Mali or PowerVR), the lowest-memory Android phone the
   product means to support, and an iPhone 15 Pro.
4. **Owner** of the runtime pin, the mirror and the model variants.
5. **Is AICore acceptable for part of the audience?** If the need is a good
   built-in model on recent Android phones, AICore may meet it with nothing
   to ship. If the need is Gemma 4, the same on every phone and under our
   control, it does not.
6. **Platform order.** Android first and iOS second, as proposed, or iOS in
   the first slice?
7. **First-slice scope.** Is text-only Gemma 4 useful on its own, given that
   today's path is used for images and tools? If not, images join the first
   slice and it grows.
8. **Sampling.** This proposal refuses `min_p` with sampling on LiteRT-LM. The
   alternative, accepting the request and reporting the difference, changes
   semantics silently unless the caller opts in. Note that
   `GenerationConfig::creative()` would be refused on a LiteRT model until
   `min_p` is set to 0.

## 12. Alternatives considered

| Alternative | Assessment |
|---|---|
| A smaller GGUF on llama.cpp (Q4_0 in place of Q8) | A registry change only. Cuts the Gemma 4 E2B download from 5.5 GB to 3.4 GB with its projector, or 2.8 GB for text alone. Its effect on speed, memory and answer quality is unmeasured. Measured in phase 0. |
| llama.cpp with a GPU or NPU backend on Android | The main competitor on speed. Keeps GGUF, the registry and today's semantics. The pinned llama.cpp has OpenCL (Adreno 750 to 840 verified upstream), Vulkan and a Hexagon NPU backend; Xybrid builds none of them for Android. Costs build work and driver coverage. Measured in phase 0. |
| System models (Gemini Nano through AICore, Apple Foundation Models) | On supported Android devices AICore serves Gemma 4, and Google recommends it for production apps. No runtime or weights to ship. In beta, with input under 4,000 tokens and a per-app quota; no control over model version or device coverage, and no path for other models. Measured in phase 0 where a test phone has it. |
| ExecuTorch, MLC | Credible GPU and NPU paths; each brings its own export format and the same maintenance cost. |
| ONNX Runtime GenAI | Not automatic from the existing ONNX Runtime; no confirmed Android path. |
| Cloud routing | Already in the product; cannot be mandatory in an offline-first SDK. |
| Link LiteRT-LM statically | Rejected: two Rust runtimes and duplicate C++ libraries in one image. |
| Third-party Rust crates for LiteRT-LM | Rejected: single-owner crates, most under 500 downloads, in a shipped SDK. |

## Appendix: how the upstream facts were obtained

`litert_lm_c_api-1.0.0.zip` and `CLiteRTLM.xcframework.zip` were downloaded
from the v0.19.1 release and inspected with `unzip -l`, `llvm-objdump -p`,
`llvm-objdump -R`, `llvm-nm -D`, `otool -L`, `otool -l`, `codesign -dv` and
`strings`. The Android API level was read from `.note.android.ident`. LiteRT
prebuilt sizes come from HEAD requests to
`storage.googleapis.com/litert/binaries/2.3.0/`. The state bug was confirmed by
reading `Reset()` at the v0.19.1 tag and the patch in commit `e1f86b05`; the
join behaviour by reading `runtime/conversation/conversation.h` and
`runtime/core/session_advanced.{h,cc}` at the same tag. Nothing was built or
run against a model, and no Xybrid code was changed.

For the Gemma 4 update: sizes come from the Hugging Face file listings of
`litert-community/gemma-4-E2B-it-litert-lm`, `ggml-org/gemma-4-E2B-it-GGUF`
and `google/gemma-4-E2B-it-qat-q4_0-gguf`; benchmark figures from the first
repository's model card; the layer layout from `config.json` in
`google/gemma-4-E2B-it`. The defects in 4.4 were read from LiteRT-LM issues
#4001, #3913, #2814, #4567, #4096, #2227, #3165 and #1050, with their
comments. The GPU backend's presence in the prebuilt Android library was
checked with `strings`. The sliding-window caches and the token-skipping logic
were read in `runtime/executor/litert/state.cc` and
`runtime/framework/resource_management/resource_manager.cc` at the v0.19.1
tag. AICore's limits come from Android's Gemini Nano page and the ML Kit
Prompt API guide. llama.cpp facts were read with `git show` at the commit the
repository pins (`8642541b`), because the submodule's working tree is at a
different commit.
