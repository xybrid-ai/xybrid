# Liquid decision conformance and qualification

The native text/JSON implementation targets `lfm3-d1/text/v1`. Its first
artifact is the immutable d1-2B Q4_K_M entry in
[`artifacts.json`](../../integration-tests/fixtures/decision/artifacts.json).
Running the implementation is distinct from qualifying every SDK package and
device for release. [SDK and CLI examples](../sdk/decisions.md) use a local
directory; registry publication remains a separate platform-owner operation.

## Executable gates

From the repository root:

```sh
python3 tools/conformance/decision/check.py
python3 -m unittest discover -s tools/conformance/decision/tests
cargo test -p integration-tests --test decision_conformance
just integration-tests decision-native
```

The last recipe stages and hashes Q4_K_M, builds the pinned upstream oracle,
collects a fresh official reference with the host compiler and CPU, and runs
the Rust suite with `XYBRID_REQUIRE_MODELS=1`. The matched numerical recipe is
currently Linux x86_64 GNU. It stages the oracle's exact Release engine archives
through `stage_native.py` and `XYBRID_NATIVES_PREBUILT_DIR`; Cargo still builds
the current wrapper and pinned Jinja bridge locally. This explicit prefix takes
precedence over `XYBRID_NATIVES_FORCE_SOURCE`, which prevents automatic published
cache fallback. A source build with a different Cargo/CMake profile is not
assumed to be numerically identical. It requires several GB of disk,
a native C++ toolchain and sufficient RAM for a whole 4096-token attention
pass. Downloading is a preparation step; subsequent model execution is local.

The stage-isolated checks are:

| Check | Inputs and assertions |
|---|---|
| Wire | All 143 cases are accounted for: 98 accepted, 42 parse refusals, three stopped at model admission. Request key order is checked. |
| G0 | Exact prompt bytes, token IDs, label groups, aliases, output index, temperature strings and f32 bits; all 89 accepted preparations, nine preparation refusals and three metadata refusals. Metadata-only vocabulary variants use the pinned oracle's variant command. |
| Finalizer | Upstream selected logits supplied directly to Rust, independently of Rust preparation/evaluation: 82 accepted cases, six non-finite refusals, one media-only result excluded. Ordered f64 probabilities, median, confidence and JSON round trips are checked exactly on Linux x86_64 glibc; other libm configurations require separate qualification. |
| G2 | All 63 model-evaluated cases accepted by this profile (78 whole-prompt passes), supplied upstream tokens and selected IDs, compared bitwise with a fresh matched CPU build. This is not a cross-device tolerance. |
| Lifecycle | A→B→A; 4097-token and invalid-token refusal; pre-cancellation and cancellation during a whole native pass followed by a valid request. |
| SDK | Four representative text, structured-number, question-order and yes/no cases through both sync and async `run`, including concurrent queued callers, with ordered typed results and input/zero-output usage. |
| CLI | Conflicting flags and bounded/redacted stdin errors before model loading; a real request through file, stdin and `--output` compared with the Rust SDK, including ordered f64 answers and usage. |

Every required test name is enumerated in `CI: Decision conformance`, and the
tests assert corpus execution counts. Missing/corrupt models, missing reference
receipts, missing test names and narrowed corpora fail the required lane. Local
native checks can skip when assets are absent, with an explicit message.
`Decision CPU conformance` also gates the llama.cpp update automation. The
workflow has no path filters, so required checks always reach a terminal state.

For a manually collected current-host reference:

```sh
tools/conformance/decision/build.sh
integration-tests/download.sh d1-2b-q4km
python3 tools/conformance/decision/stage_native.py --out /tmp/decision-native
python3 tools/conformance/decision/stage_native.py --out /tmp/decision-native --check
python3 tools/conformance/decision/oracle.py \
  --collect integration-tests/fixtures/decision/hf/cases \
  --artifact d1-2b-q4km --expect-cases 63 --out /tmp/decision-upstream
XYBRID_NATIVES_PREBUILT_DIR=/tmp/decision-native \
  XYBRID_DECISION_REFERENCE_DIR=/tmp/decision-upstream XYBRID_REQUIRE_MODELS=1 \
  cargo test -p integration-tests --features llm-llamacpp \
  --test decision_conformance -- --nocapture --test-threads=1
```

The output directory must be new or empty. The reference collection binds
artifact checksum, native pin, compiler flags, actual context/device settings,
request and corpus digests. Another compiler may differ from the committed
oracle's numerical observations: `oracle.py --check` correctly refuses that
identity change. Never loosen G2's bitwise comparison to hide a mismatch;
first verify that both paths use the same compiler, native library, CPU ISA,
four threads, flash attention off and explicit empty offload-device selection.
The staging receipt binds every copied archive/header and both upstream
executables to the official embedded source/build identity. It rejects missing,
changed or unsupported inputs and is uploaded with each required CI run. It
is comparison evidence, not a native-cache publication or package qualification.

Model admission also bounds native metadata to 8 MiB in aggregate, vocabulary
size to one million tokens, and CONTROL spellings to 256 entries / 64 KiB.
Unbreakable one-character or separator-bearing CONTROL spellings fail closed.
These limits bound metadata processing; they are not measured device-memory
requirements.

## Independent HF reference

[`hf/assets.json`](../../integration-tests/fixtures/decision/hf/assets.json)
pins the checkpoint and every Python/config/tokenizer/calibration/template
companion file by byte length and SHA-256 at the model's exact revision.
[`hf/corpus.json`](../../integration-tests/fixtures/decision/hf/corpus.json)
freezes 63 request inputs before observations: 42 calibration and 21 held-out
cases. They are inputs, not evidence that the HF path agreed with GGUF.

Use Python 3.12.14 and the locked Linux CPU environment:

```sh
python3.12 -m venv tools/conformance/decision/work/hf-env
tools/conformance/decision/work/hf-env/bin/pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r tools/conformance/decision/requirements-hf.txt
python3 tools/conformance/decision/hf_reference.py \
  --model-dir tools/conformance/decision/work/hf-model --stage
tools/conformance/decision/work/hf-env/bin/python \
  tools/conformance/decision/hf_reference.py \
  --model-dir tools/conformance/decision/work/hf-model \
  --out tools/conformance/decision/work/hf-observations
```

Asset staging is the only network phase. Collection checks all hashes and
installed package versions before loading pinned local custom code, sets HF
offline, and explicitly runs `set_size=1`, one question and one batch row per
call, on four CPU threads. Float32 is the default evaluation dtype; the original
checkpoint is BF16. The dtype, attention mode, native PyTorch build, package
versions, inputs, collector digest and results are recorded. An exception is a
failed receipt and a nonzero exit, never an omitted case. `--only ID` requires
the corresponding explicit `--expect-cases N` and produces a partial corpus
collection; it does not establish qualification of the full corpus.

Collect the same inputs with `oracle.py --collect ... --artifact d1-2b-f16`
to investigate HF versus unquantized GGUF independently of Q4 quantization.
Compare preparation semantics before interpreting numerical differences.
Metadata string rounding, selected-token aliases and f64 finalization are
separate from raw model arithmetic.

```sh
python3 tools/conformance/decision/compare_reference.py \
  --hf-dir tools/conformance/decision/work/hf-observations \
  --native-dir /tmp/decision-f16 --expect-cases 63 \
  --out tools/conformance/decision/work/hf-f16-comparison.json
```

The comparison reports every HF refusal, differing prompt/token/label map and
changed choice/median. Numerical maxima are reported only when preparation
matches and are observations, never automatically chosen acceptance limits.
The HF runner subtracts vocabulary-wide logsumexp before its readout; the
collector hooks the output head to retain true raw logits as well as those
normalized log-probabilities.

## Implementation validation summary

Raw collection outputs, token/logit arrays and per-run receipts are not
committed. Use the commands above to reproduce them in the ignored
`tools/conformance/decision/work/` directory or temporary storage. The required
CI lane collects a fresh native reference and uploads its observations,
runtime reports and staging receipt as the `decision-cpu-reference` artifact.
`check.py` verifies the committed fixture inputs and source/artifact identities;
it does not require historical run outputs or establish release qualification.

The initial required native integration run passed all seven tests with zero
skips. G0 matched all 89 accepted preparations, nine preparation refusals and
three load refusals.
G2 matched every selected f32 logit bit across 63 real-model cases and 78
whole-prompt passes, including 4096 tokens. Lifecycle refusal/cancellation
recovery, Rust sync/async/concurrent execution, wire decoding and independent
finalization also passed. All three required CLI tests passed: conflicting
flags and bounded/redacted transport refusals before loading, plus real Q4
file/stdin/output-file parity with the Rust SDK for ordering, f64 values and
usage. These are ten required tests across the two integration targets.

The independent HF collector attempted every one of the 63 frozen inputs:
58 completed and five were explicitly refused by its pinned validator. Twelve
accepted cases had different preparation semantics from the pinned upstream
profile. Among matching preparations, four choices/medians changed against
Q4, including two held-out cases. The one-case/three-question F16 smoke matched
preparation and selected answers; it is deliberately not a full F16 study.
These findings leave P02 open. The comparison command reports observed
numerical maxima for review without promoting them to acceptance limits.

## Host binding verification

The implementation was also exercised through installed/staged Linux native
interfaces, separately from the matched Release G2 engine:

- The installed public Python wheel passed 70 tests with no skips, including
  actual Q4 inference, ordered numeric content, f64 results and usage, in-flight
  cancellation, concurrent disposal/property access, and successful recovery.
  Its packaged native NOTICE and shared-library hash were verified.
- Flutter passed analysis, 51 Dart tests, deterministic regeneration, and a
  required Dart-to-Rust native smoke with no skips. The smoke covers signed
  integer limits, values above 2^53, NUL/signed zero, typed results, refusal,
  cancellation through a still-live caller handle, and recovery.
- The generated and handwritten Unity C# API passed a Linux .NET native smoke
  against the canonically staged library: raw/typed input, all three answer
  kinds, order, f64/usage, cancellation callbacks, refusals and recovery.
  Managed compilation and all 140 native-import checks also passed.
- Pinned Bolt generation checks, Kotlin codec tests and React Native's five
  JavaScript suites passed. Swift/Xcode and physical mobile execution were not
  available on this host.

These shared-library smokes use a source-built CPU Debug/PIC engine; the G2
reference uses its separately identified Release archives. They establish
binding and lifetime behavior, not numerical equivalence across builds or
release/device qualification. Unity Editor/IL2CPP, a deliberately mismatched
Flutter native artifact, and the complete packaged platform matrix remain open.

## Promotion requirements still requiring release evidence

The input-integrity checker is deliberately not named a numerical-provenance
gate. No `provenance.json` with invented tolerances is supplied. Release
qualification requires measured and reviewed bounds with metric, units,
calibration/held-out corpus hashes, source/configuration identities, derivation,
and every changed or ambiguous decision explicitly reported.

The first implementation PR does not itself establish the following:

- HF custom forward versus F16/BF16 GGUF compatibility and frozen G1 limits.
- Quantization-specific G3 limits or calibration quality, including Q8_0.
- Packaged macOS/Metal, Windows, physical iOS and Android correctness,
  cancellation latency, peak memory and thermal/resource limits.
- Every generated-language, Flutter and React Native real-device G4 scenario.
- Anonymous published native-cache retrieval, source/prebuilt equivalence,
  all clean release-candidate package consumers, or primary/fallback registry deployment.

Do not advertise a device/backend because it compiles or inherits native
symbols. Keep browser, legacy d1 models, image, video, audio/Omni, additional
heads and typed pipelines on their separately versioned qualification paths.
The release branch workflow remains the only publication entry point.

## Work-package status against the implementation plan

This table distinguishes code delivered in the implementation PR from the
plan's release acceptance criteria. A binding or build addition does not
qualify a device, package or numerical tolerance. The executable gates above
are the source of test results; release prerequisites below remain open until
their own evidence is reviewed.

| Plan item | Implementation and evidence scope | Remaining acceptance work |
|---|---|---|
| P01 — Protocol and API contract | [Versioned protocol contract](liquid-decision-protocol.md), Decision metadata, ordered request/result types, explicit admission rules and fixture-stage accounting. | Keep the contract and the pinned corpus synchronized for later protocol changes. |
| P02 — Independent references and tolerances | Immutable HF assets, locked CPU environment, frozen calibration/held-out input split, raw-head collection and semantic comparison tooling. | **Incomplete:** resolve documented HF/GGUF semantic differences; complete F16/BF16 and repeated Q4/Q8 studies, compatibility review and derived numerical provenance. No promotion tolerances are supplied. |
| P03 — Native facilities | Bounded metadata/tokenizer APIs, vocabulary-only model handle and pinned Jinja bridge; Cargo/Bazel source inputs updated. | Complete the supported native build matrix and packaged renderer linkage evidence. |
| P04 — Whole-prompt evaluator | Dedicated fixed 4096-token CPU context, selected-logit copying, native cancellation and recovery checks; recorded matched-host G2 passes 63 cases/78 passes. | Independent devices and packaged configurations need separate qualification. |
| P05 — Wire and preparation | Shared ordered decoder, field-depth and byte limits, CONTROL escaping, labels, aliases and temperature handling. | G0 is the exact preparation gate; new templates or vocabularies require new admission evidence. |
| P06 — Finalization | Independent f64 finalizer, ordered ties/median, non-finite handling and exact upstream-score checks. | Qualify other libm/platform configurations independently. Structural probability validation is not a numerical promotion bound. |
| P07 — Core dispatch | Decision template/runtime, cache/lifecycle integration, explicit unsupported-path refusals and pinned local fixture metadata. | Cross-platform lifecycle and packaged regressions remain part of release qualification. |
| P08 — Rust SDK and CLI | Sync/async typed results, local JSON helper/CLI, whole-pass cancellation, route/pipeline refusals and content-free telemetry; real-model Rust sync/async/concurrent and all three CLI checks passed. | Release-candidate consumers and platform behavior require package evidence. |
| P09 — Foreign contract | Facade-owned ordered decision records and appended foreign variants/accessors; legacy record layouts retained. | Foreign compatibility and real-model checks must accompany each shipped generated package. |
| P10 — Swift/Kotlin/Python/Unity | Pinned generation/drift and managed/codec checks pass; an isolated public Python wheel and canonically staged Linux Unity managed/native interface passed real-model smokes. | **Not release-qualified:** all four shipped native interfaces and actual platform consumers; Swift/Xcode, Unity Editor/IL2CPP and physical devices remain open. |
| P11 — Flutter | Typed run-result carrier retains legacy getters; generated drift, 51 Dart tests and a required real Dart/native cancellation/recovery smoke pass. | Complete Dart/native corpus, deliberately mismatched artifact rejection, cargokit/precompiled consumers and mobile qualification. |
| P12 — React Native | Explicit ordered-text transport and native codecs with parity mappings. | Both native codec suites and real iOS/Android bridge scenarios, including cancellation and numeric edge cases. |
| P13 — Required conformance CI | Model-required CPU workflow, exact test-name/count checks, hash verification, cold schedule and update-gate integration. | Independent-reference promotion bounds and packaged foreign-language G4 coverage remain blocked by P02/P14 evidence. |
| P14 — Packaged artifacts/resources | Native build inputs and notices are prepared for the existing packaging paths. | **Unqualified:** packaged matrix, device RAM/thermal/latency/cancellation measurements, advertised memory floors and cache equivalence. |
| P15 — Registry/docs | Protocol-specific capability named (`llamacpp-lfm3-d1-text-v1`) but reported by no build, so registry resolution cannot select a decision artifact while P02 is open; local SDK/binding usage documented. | **Publication deferred:** platform-owner primary/fallback entries and release-candidate resolver/rollback checks depend on qualification. |
| P16 — Release/observe | Existing release workflow remains authoritative. | **Not performed:** versioned package release, anonymous cache retrieval, installed consumers and release telemetry observation. |
