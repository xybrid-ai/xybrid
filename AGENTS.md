# AGENTS.md — guidance for AI coding agents in xybrid

This is the **canonical agent guide**: xybrid's project decisions first, then a
compressed index of the two upstream guides we follow. Where a project decision
here disagrees with the upstream rule indexes below, **the project decision
wins**. (`CLAUDE.md` holds only Claude-specific material and imports this file —
new project decisions belong here, not there.)

Fetch the upstream guides when you need detail on a specific rule:

- **Microsoft Pragmatic Rust (agent edition)** —
  <https://microsoft.github.io/rust-guidelines/agents/all.txt>
- **rust-skills SKILL.md** (leonardomso) —
  <https://github.com/leonardomso/rust-skills/blob/master/SKILL.md>
  (per-rule examples live at `rules/<rule-id>.md` in that repo)

---

## How to work in this repo

Before declaring a task done, run these locally — CI runs them again:

```bash
cargo fmt --all -- --check                              # CI: fmt
cargo clippy --workspace --all-targets -- -D warnings   # CI: clippy
cargo test  --workspace --features ort-download         # CI: test (matrix)
```

`just fmt-check | lint | test | check` are the equivalent recipes. `just fmt`
(no `-check`) writes fixes — run before committing. Match CI's
`--features ort-download` locally so feature-gated paths don't surprise you.

CI gates (`.github/workflows/ci.yml`): `fmt`, `clippy`, `test` (macOS/Linux/Windows),
plus feature-matrix jobs: `check-no-default-features`, `check-platform-macos`,
`check-platform-desktop`, `clippy-llm`, `test-llm`, `test-candle`, `build-cli`,
`api-contract`.

Crate-wide lint opt-outs go in `lib.rs` at the crate root (see e.g.
`crates/xybrid-core/src/lib.rs`). Don't sprinkle `#[allow(...)]` at call sites —
push it to crate level or fix the lint. Never bypass hooks (`--no-verify`).

### Building native bindings / cross-compiled artifacts

**Native artifacts are built by Bazel**, not `xtask`. Bazel brings its own
hermetic toolchains (Rust, Android NDK, clang, Windows SDK), so these are the
same targets the release ships from — do **not** hand-roll `cargo ndk` or raw
`cargo build --target <triple>` to reproduce them:

```bash
bazel build -c opt //bindings/kotlin:xybrid_kotlin_aar        # Android AAR (jniLibs inside)
bazel build --config=ios //bindings/apple:XybridFFI           # Apple XCFramework (macOS host + Xcode)
bazel build --config=macos-metal //crates/xybrid-cli:xybrid   # CLI (see .bazelrc for other configs)
bazel build --config=windows-msvc //...                       # MSVC-ABI Windows, cross-built
```

Use `bazelisk` (reads `.bazelversion`). `just bazel-build | bazel-analyze |
bazel-test` are the shortcuts; each forwards extra Bazel flags. Full setup,
including the Windows MSVC EULA note, is in `CONTRIBUTING.md`.

`xtask` is **not** the native-binding entry point anymore. `build-android`,
`build-xcframework`, `build-uniffi`, `stage-react-native`, `setup-targets`,
`build-all`, and `package` were all removed once Bazel took over. What remains
is Flutter, Unity staging, and dev-env chores:

```bash
cargo xtask build-flutter --platform <linux|macos|windows>  # Flutter native (deliberately cargo)
cargo xtask build-ffi --target <triple> --release           # xybrid-bolt cdylib (Unity native)
cargo xtask build-unity                                     # xybrid-bolt across Unity platforms
cargo xtask deploy-unity-native --lib <path> --target <triple>  # stage a Bazel-built lib into Unity
cargo xtask setup-test-env                                  # dev-env chore
```

Flutter native is cargo **on purpose**: `flutter run` inside the repo goes
through cargokit → cargo, so the contributor path must match. `build-ffi` /
`build-unity` build `xybrid-bolt` (the pre-bolt `xybrid-ffi` C ABI is retired);
`deploy-unity-native` exists so a Bazel-built lib can be staged into
`bindings/unity/Runtime/Plugins/` with the right per-platform `.meta`.

After editing `bindings/flutter/rust`, regenerate the Dart glue with
`flutter_rust_bridge_codegen generate` (CLI version must match the pinned
`flutter_rust_bridge` in `bindings/flutter/rust/Cargo.toml`); `flutter run` then
rebuilds the native lib via cargokit.

Note: `tools/README.md` still documents the pre-Bazel xtask matrix and is stale.

### Prebuilt llama.cpp natives (the cargo fast path)

`crates/llama-cpp-sys/build.rs` does not always run cmake. It resolves the
llama.cpp static archives in this order, falling through on any miss:

1. `XYBRID_NATIVES_PREBUILT_DIR/<target>` — a slice staged by the caller. Our
   CI jobs pull with `tools/scripts/natives-pull.sh` and point at it.
2. A slice named in `crates/llama-cpp-sys/natives-manifest.txt`, downloaded
   over plain HTTPS from `ghcr.io/xybrid-ai/llama-natives` and SHA-256
   verified. Needs no oras, no env var, no cmake — this is what makes an
   **external** `cargo build --features llm-llamacpp` cheap.
3. The cmake source build.

The manifest is GENERATED — `.github/workflows/build-natives.yml` publishes the
slices, then its `publish-manifest` job regenerates the file via
`tools/scripts/natives-manifest.sh` and opens a PR. Never hand-edit it.

Two traps:

- **The manifest goes stale on purpose.** It pins plain hashes of
  `wrapper.cpp`, `wrapper.h`, `build.rs`, and the llama.cpp commit. Touch any
  of them and every row is ignored until CI republishes — that is the guard
  that stops a local edit from linking archives that predate it. A dropped
  fast path after editing `build.rs` is expected, not a bug.
- **Anonymous reachability is not covered by our own CI.** Every job here
  `oras login`s first, so a private package looks healthy internally and 401s
  for everyone outside. `tools/scripts/natives-verify-anon.sh` is the check
  that catches it; it runs credential-free at the end of `build-natives`.

The publisher fingerprint (`natives-fingerprint.sh`) folds in the LOCAL
cmake/cc/NDK versions. That is right for publisher/consumer cache parity and
wrong for distribution, which is why the download path selects by target +
feature set + ABI attributes from the manifest instead of recomputing it.

### Releases

**Releases are cut by branch name, not by hand.** Push a `release/v<version>`
branch and `.github/workflows/release-prep.yml` does the rest (manifest-version
check, artifact builds, `Package.swift` checksum patch, draft GitHub Release, and
the `Release v<version>` PR to master). Don't `gh pr create` a release or run
`just bump-version` on a feature branch and open a PR from it — that bypasses the
pipeline and ships nothing. Never commit `Package.swift useLocalNatives = true`
(local-dev only; breaks remote SPM — run `bindings/apple/scripts/set-natives-mode.sh
--set-remote`). Full ritual + gotchas: **§ Releases** below.

---

# Section 0 — xybrid project decisions

These are the decisions xybrid has already made. They override the upstream
rule indexes in Sections 1–2 wherever the two disagree.

## What xybrid is (read this first)

xybrid is a **local-first execution engine with a platform layer built on top**.
Both planes share one codebase, and the platform plane is **additive** — it
extends the same runtime without replacing the offline path:

1. **Foundation — on-device execution engine.** Model load/run/stream,
   pipelines, hardware acceleration. Zero-config, offline, no account required.
   This is the default path and most of the code today.
2. **Platform / control-plane layer (additive).** Opt-in capabilities layered on
   the engine — the direction being actively built out. When a developer
   authenticates, these light up *on top of* the same local runtime:
   - **Auth / API keys** — `crates/xybrid-core/src/cloud/config.rs`
     (`set_xybrid_api_key`, `XYBRID_API_KEY`). Gates the cloud gateway *and* the
     telemetry exporter. Default gateway: `api.xybrid.dev`.
   - **Cloud routing** — `crates/xybrid-core/src/cloud/` +
     `crates/xybrid-core/src/orchestrator/routing_engine.rs` (local→cloud fallback under device stress).
   - **Telemetry / observability** — `crates/xybrid-core/src/telemetry/`;
     SDK exporter in `crates/xybrid-sdk/src/telemetry.rs`; ingest at `ingest.xybrid.dev`.
   - **Remote routing authority** — `crates/xybrid-core/src/orchestrator/authority/remote.rs`
     (`GET /v1/routing/advice`; partial).
   - **Control sync** — `crates/xybrid-core/src/control_sync.rs` (policy /
     registry refresh; scaffolded, backend not yet wired).

The public README markets the foundation ("offline, no cloud, no API keys")
because that's the zero-config default — the platform layer is what you add on
top once you authenticate. **When touching `xybrid-sdk` or `xybrid-core`, treat
the platform plane as a first-class, additive surface**, not an afterthought:
new SDK entry points should consider whether they extend into it too.

---

## Workspace layout

Cargo workspace, `resolver = "2"`, edition 2021, MSRV not pinned. Members:

| Crate                          | Role                                                       | Layer    |
|--------------------------------|------------------------------------------------------------|----------|
| `crates/xybrid-core`           | ML execution + pipelines; **additive platform plane** (cloud routing, telemetry, control sync) | core lib |
| `crates/xybrid-sdk`            | Public Rust SDK; model load/run/stream + platform init (auth, telemetry) | lib      |
| `crates/xybrid-cli`            | `xybrid` binary                                            | bin      |
| `crates/xybrid-ffi-facade`     | FFI-agnostic POD/Arc facade over the SDK (one canonical translation) | FFI |
| `crates/xybrid-bolt`           | BoltFFI bindings: Swift / Kotlin / Java / C# (Unity) / WASM + C header — the sole native binding crate | FFI |
| `bindings/flutter/rust`        | flutter_rust_bridge wrapper for Dart                       | FFI      |
| `macros`                       | proc-macros (`xybrid-macros`); syn/quote only              | proc     |
| `xtask`                        | build / codegen automation                                 | tool     |
| `integration-tests`            | end-to-end tests with real models & fixtures               | test     |

`xybrid-uniffi` was removed once iOS + Android migrated to `xybrid-bolt`;
the FFI binding crates now route their SDK→foreign-language translation
through `xybrid-ffi-facade` rather than each re-translating SDK types.

The Python SDK (`bindings/python`, not a workspace member) runs on boltffi's
**generated** bindings as of 0.29: `xybrid/_bolt/` is generator output
(`tools/scripts/gen_python_bolt.py`, byte-compared in CI via `--check`), and it
imports a compiled CPython bridge that dlopens the `xybrid-bolt` cdylib. Both
binaries are staged by `tools/scripts/build-python-bolt.sh` and are **build
outputs, never committed** — so wheels are per-interpreter (`cp3XX`) and the
SDK requires Python >= 3.10.

Because the generated package is byte-compared, it carries no hand-written
code. The Pythonic surface (envelope factories, `result.text`, model
properties, typed exceptions) is attached to the generated classes at import by
`xybrid/_sugar.py` and `xybrid/_errors.py`, guarded by `tests/test_sdk.py`. Add
SDK ergonomics there, never in `xybrid/_bolt/`.

**Dependency direction (do not reverse):**

```
xybrid-cli  ──────────────────────► xybrid-sdk ─► xybrid-core
xybrid-bolt ──► xybrid-ffi-facade ─► xybrid-sdk ─► xybrid-core
flutter rust──► xybrid-ffi-facade ─► xybrid-sdk ─► xybrid-core
xtask ────────────────────────────► xybrid-core
integration-tests ────────────────► xybrid-core
```

Workspace **package metadata** is inherited via `[workspace.package]` —
member crates use `version.workspace = true`, `edition.workspace = true`,
etc. Keep that pattern.

Workspace **dependencies** are *not* uniformly inherited today. The root
`[workspace.dependencies]` block exists, but most member crates still pin
versions per-crate (e.g. `serde = "1.0"`, `tokio = { version = "1.0", … }`).
When adding a dep, match the surrounding crate's existing style — don't
unilaterally migrate one crate to `dep.workspace = true` while the rest stay
version-pinned. Full `proj-workspace-deps` migration is a deliberate
refactor, not a drive-by change.

---

## Error handling

Rust-API library crates (`xybrid-core`, `xybrid-sdk`, `xybrid-ffi-facade`)
use **`thiserror`** with a single canonical error enum and a `Result` alias
per crate:

| Crate              | Error type     | Result alias    | Defined in                           |
|--------------------|----------------|-----------------|--------------------------------------|
| `xybrid-core`      | `XybridError`  | `XybridResult`  | `crates/xybrid-core/src/error.rs`    |
| `xybrid-sdk`       | `SdkError`     | `SdkResult`     | `crates/xybrid-sdk/src/model.rs`     |
| `xybrid-ffi-facade`| `Error`        | `Result`        | `crates/xybrid-ffi-facade/src/lib.rs` (one canonical `From<SdkError>`; the FFI generator crates re-decorate it) |

Sub-error enums (`InferenceError`, `PipelineError`, `AdapterError`, …) live
next to the modules that raise them and convert into the canonical type via
`#[from]` / `impl From`. Follow that pattern for new modules — don't invent
parallel top-level error types.

Binaries (`xybrid-cli`, `xtask`) use **`anyhow`** with `.context(...)` at the
boundaries where errors get printed.

`SdkError` implements a `RetryableError` trait (`is_retryable`, `retry_after`)
— preserve those semantics when adding variants. As of today
(`crates/xybrid-sdk/src/model.rs`) the retryable variants are
`NetworkError`, `RateLimited`, `Timeout`, and `Offline`; everything else
(including `CircuitOpen`, `ConfigError`, `ModelNotFound`, `LoadError`,
`InferenceError`, `IoError`, `CacheError`, `PipelineError`, …) is
explicitly **non-retryable**. Read the current `is_retryable` match arm
before changing or extending it — don't infer the rule from the variant name.

Don't use `Box<dyn Error>` in public signatures. Don't `.unwrap()` outside
tests, examples, and clearly-marked invariant checks (use `.expect("...")`
with a message that explains the invariant — rust-skills `err-expect-bugs-only`).

---

## Async runtime

**Tokio**, multi-threaded. Workspace pins:
`tokio = { version = "1.0", features = ["rt", "rt-multi-thread", "sync"] }`.
No async-std, no smol.

Public SDK APIs come in **sync + async pairs**: `load` / `load_async`,
`run` / `run_async`, `warmup` / `warmup_async`, `run_pipeline_async`, etc.
Sync variants block on the runtime internally. **Match this convention** when
adding new SDK entry points — don't break the symmetry.

Inside async code:

- Use `tokio::task::spawn_blocking` for CPU-bound or sync I/O (model loading
  is the canonical example — see `xybrid-sdk` model loader).
- Don't hold `Mutex` / `RwLock` guards across `.await` (rust-skills
  `async-no-lock-await`, `anti-lock-across-await`).
- Channels: `tokio::sync::mpsc` for streaming events; that's the established
  pattern for pipeline event streams (`xybrid-sdk/src/lib.rs`).

Tests that need a runtime use `tokio::runtime::Runtime::new().unwrap().block_on(...)`
today. New async tests may use `#[tokio::test]` — both are accepted.

---

## SDK run surface — capabilities are data, not entry points

The public run surface grows by axes, and axes multiply. The rule:

- **A new capability rides `RunOptions`, `GenerationConfig`, or `Envelope`** —
  never a new `run*` method. Constrained decoding shipped this way
  (`GenerationConfig.grammar` — zero new entry points), and new capabilities
  (tool calling, etc.) must too (`RunOptions`-carried inputs, parsed outputs
  on `InferenceResult`).
- **A new entry point is justified only by a new IO shape** — a different way
  results physically flow out: batch return, per-token callback, async
  `Stream` handle, TTS chunk sink. Each IO shape gets exactly one sync + one
  async form, nothing else.
- **Suffix chains are the anti-pattern.** Names like
  `run_streaming_with_context_options_preempt` are capabilities leaking into
  the namespace. `XybridModel` currently carries 14 `run*` variants for this
  reason, while bolt's entire foreign surface needs exactly one
  (`run(envelope, options)`). The excess is consolidation backlog, **not
  precedent** — don't add variant #15.
- Multi-step orchestration (agent loops, retries, tool-execution cycles)
  lives **outside** `XybridModel` — a free helper or the caller's own loop —
  and FFI bindings get turn-based primitives (parsed results + another `run`
  call), not cross-boundary callbacks.

---

## Testing & mocking

- **Unit tests** inline as `#[cfg(test)] mod tests { use super::*; ... }`.
- **Integration tests** in each crate's `tests/` directory.
- **End-to-end tests** with real models in `/integration-tests/`. Fixtures
  live in `integration-tests/fixtures/{input,models,pipelines}/`. Tests that
  need a downloaded model gate themselves with `fixtures::model_if_available()`
  and skip cleanly if the model isn't present — follow that pattern, don't
  hard-fail on missing assets.
- **HTTP mocking:** `httpmock` (already a dev-dep in `xybrid-sdk`). **No
  `mockall`, `mockito`, or `wiremock`** in this repo today — don't introduce
  another mocking library without discussion.
- **Benchmarks:** `criterion` (dev-dep in `xybrid-core`).
- No `insta` snapshots, no `proptest`. Don't add either casually — they bring
  CI cost and a learning curve.

Run model-gated tests with `just`-recipes under `mod integration-tests` in the
root `justfile`.

---

## Concurrency primitives — when to use what

The workspace is multi-threaded. **Don't use `Rc` or `RefCell`** — they aren't
in use anywhere and they trap you in single-threaded contexts.

| Need                                            | Use                              |
|-------------------------------------------------|----------------------------------|
| Pass data into a function for read-only use     | `&T` (or `&[T]`, `&str`)         |
| Share owned state across threads / async tasks  | `Arc<T>`                         |
| Shared state, mostly reads, some writes         | `Arc<RwLock<T>>` (std)           |
| Shared state, exclusive access each time        | `Arc<Mutex<T>>` (std)            |
| Cross-task message passing                      | `tokio::sync::mpsc`              |
| One-shot reply channel                          | `tokio::sync::oneshot`           |

Use `std::sync::{Mutex, RwLock}` — **not** `parking_lot` (not a dependency).
Public traits that cross task boundaries are bounded `Send + Sync`; this is
established convention for backend / strategy / session traits in
`xybrid-core`. Keep that bound on new traits in the same family.

Prefer borrows over `Arc::clone` when a borrow's lifetime is obviously short
enough. Reach for `Arc` when you're crossing a `spawn` / `spawn_blocking` /
channel boundary, or storing the value behind a trait object.

---

## Releases — never hand-roll one (agents read this)

**Releases are cut by branch name, not by hand.** The entire pipeline keys off a
`release/v<version>` branch. An agent's *only* correct action to start a release
is to create and push that branch — everything else (artifact builds, checksum
patch, draft GitHub Release, the release PR, the tag, and the crates.io / pub.dev
/ Maven Central publishes) is generated by CI. The ritual
(`.github/workflows/release-prep.yml` → `release-publish.yml`):

```bash
git switch -c release/v<version> origin/master
just bump-version <version>     # syncs every manifest: Cargo, pubspec, Unity,
                                # Kotlin, Package.swift sdkVersion, Python pyproject
./bindings/apple/scripts/set-natives-mode.sh --set-remote   # useLocalNatives=false
# fill the CHANGELOG entry for <version> (and bindings/flutter/CHANGELOG.md)
git commit -am "bump: <version>" && git push -u origin release/v<version>
```

Pushing `release/v*` triggers `release-prep.yml`, which:

1. Parses the version **from the branch name** and **fails** unless it matches
   every package manifest (`version-sync.sh --check` must be green).
2. Builds all artifacts, patches `Package.swift`'s `xybridFFIChecksum`, and
   creates a **draft GitHub Release**.
3. **Opens the `Release v<version>` PR to master itself.** On merge,
   `release-publish.yml` tags and publishes.

**Do NOT:**

- run `gh pr create` for a release, or run `just bump-version` on a feature
  branch and open a PR from it — that bypasses the artifact builds, the checksum
  patch, and the draft release, producing a PR that *looks* like a release but
  ships nothing. The branch name (`release/v*`) is the trigger; a PR title is not.
- leave `Package.swift`'s `useLocalNatives = true` in any committed state. It is
  **local-dev only** and breaks remote SPM consumers (the local xcframework is
  not committed). Run `bindings/apple/scripts/set-natives-mode.sh --set-remote`
  before committing — remote (`false`) is the canonical committed state.

`just bump-version` only rewrites the workspace `version`; it does **not** touch
internal path-dep `version = "..."` pins (e.g. `xybrid-sdk`/`xybrid-core` in
sibling `Cargo.toml`s). If those drift, lockfile regen fails under `^0.1.x` —
update them to match and re-run `cargo update -w` when bumping.

---

## Things to leave alone unless explicitly asked

- `rustfmt.toml` is intentionally empty (defaults). Don't add style overrides.
- The `#![allow(clippy::...)]` lists at the top of `xybrid-core/src/lib.rs`
  and `xybrid-sdk/src/lib.rs` exist because the crates are still alpha
  (`0.1.0-beta12`). Fixing those lints crate-wide is fine; **disabling
  individual call-sites** with `#[allow]` is not — push it to crate-level if
  it's project-wide.
- API contract checks (`tools/scripts/api-contract-check.sh`) run in CI as a
  soft warning. If you change a public SDK signature, run it locally.

---

## Open questions (resolve before encoding as rules)

These are genuinely ambiguous in the current code — flag them to a maintainer
rather than picking arbitrarily:

1. **MSRV.** No `rust-version` is pinned in any `Cargo.toml`. Should the
   workspace pin one (e.g. matching what CI's `dtolnay/rust-toolchain@stable`
   resolves to today)?
2. **Async test style.** `runtime.block_on` (current) vs `#[tokio::test]`
   (rust-skills `test-tokio-async`) — both work; no canonical choice yet.
3. **Workspace-level lints.** Only `bindings/flutter/rust` has a `[lints]`
   table. The rust-skills `lint-workspace-lints` / `lint-deny-correctness`
   rules suggest configuring lints workspace-wide; alpha-stage allow-lists in
   each crate make that disruptive today. Worth revisiting post-1.0.
4. **`Box<dyn Trait>` vs `impl Trait` in public APIs.** Trait-object style is
   used widely for backends (`Arc<dyn LlmBackend>` etc.) for plug-in
   replaceability. New code should follow that — but if a single-impl
   internal trait shows up, prefer `impl Trait`.
5. **Naming of streaming/event APIs.** `recv()` (channel-style) vs an
   `EventStream`-newtype wrapper. Current code uses the channel idiom; an
   abstraction layer hasn't been decided.

When you hit one of these, ask in the PR rather than guessing.

---

# Section 1 — Microsoft Pragmatic Rust (rule index)

Each ID below links to the upstream anchor on `microsoft.github.io/rust-guidelines`.
The one-line summary is enough to decide whether to read the full rule. **Bolded
rules are the ones you'll consult most often in this repo.**

> *License: MIT (Microsoft Corporation). Source:*
> `https://microsoft.github.io/rust-guidelines/agents/all.txt`

### AI-friendly design
- `M-DESIGN-FOR-AI` — strong types, thorough docs/examples, testable APIs make agent work tractable.

### Applications
- `M-APP-ERROR` — apps may use `anyhow`/`eyre`; libraries must not (see `M-ERRORS-CANONICAL-STRUCTS`).
- `M-MIMALLOC-APPS` — set `mimalloc` as global allocator in app binaries for free perf.

### Documentation
- **`M-CANONICAL-DOCS`** — public items: summary sentence + `# Examples / Errors / Panics / Safety / Abort` as applicable. No parameter tables.
- `M-DOC-INLINE` — annotate `pub use foo::Foo` with `#[doc(inline)]` for own-crate re-exports (not std/third-party).
- `M-FIRST-DOC-SENTENCE` — first sentence ≤15 words, one line.
- **`M-MODULE-DOCS`** — every public module needs `//!` docs covering contents, when to use, examples, side effects.

### FFI
- **`M-ISOLATE-DLL-STATE`** — only portable (`#[repr(C)]`, no statics/TypeId/non-portable refs) data crosses DLL boundaries. Critical for `xybrid-bolt` (the cdylib crossing DLL boundaries).

### Performance
- `M-HOTPATH` — identify hot paths early, bench with criterion, profile (Intel VTune / Superluminal). Enable `debug = 1` in `[profile.bench]`.
- `M-THROUGHPUT` — items/cycle is the metric; batch, partition, avoid empty cycles and contention.
- **`M-YIELD-POINTS`** — long CPU-bound async tasks must `yield_now().await` every ~10–100μs.

### Safety
- **`M-UNSAFE-IMPLIES-UB`** — `unsafe` is *only* for things where misuse causes UB. Don't use it to flag merely dangerous functions.
- **`M-UNSAFE`** — needs a real reason (novel abstraction, perf, FFI). Document soundness, run Miri, follow unsafe-code guidelines.
- **`M-UNSOUND`** — never acceptable. Safe code that *can* cause UB is a bug, no exceptions.

### Universal
- `M-CONCISE-NAMES` — drop weasel words (`Service`, `Manager`, `Factory`). `BookingDispatcher` > `BookingService`.
- `M-DOCUMENTED-MAGIC` — magic numbers get a comment explaining why; prefer named constants.
- `M-LINT-OVERRIDE-EXPECT` — submodule lint overrides use `#[expect(..., reason = "...")]`, not `#[allow]`.
- `M-LOG-STRUCTURED` — structured events with named properties + message templates (`{{property}}` syntax). Follow OTel semantic conventions. Redact PII.
- **`M-PANIC-IS-STOP`** — panic means *stop the program*. Not for upstream error signaling. Code must be panic-safe.
- **`M-PANIC-ON-BUG`** — detected programming bugs panic; no `Error` variant for "this shouldn't happen".
- `M-PUBLIC-DEBUG` — all public types implement `Debug`; redact sensitive data via custom impl + tested redaction.
- `M-PUBLIC-DISPLAY` — types meant to be read by humans (errors, string wrappers) implement `Display`.
- `M-REGULAR-FN` — free functions over associated functions, except for constructors and trait methods.
- `M-SMALLER-CRATES` — split when in doubt; faster compiles, fewer cycles. Re-export via umbrella crate when useful.
- `M-STATIC-VERIFICATION` — fmt, clippy (`correctness`, `complexity`, `perf`, `style`, `suspicious`, `pedantic` + select `restriction`), cargo-audit, cargo-hack, cargo-udeps, miri.
- `M-UPSTREAM-GUIDELINES` — also follow Rust API Guidelines / Style Guide / Design Patterns. Watch `C-CONV`, `C-GETTER`, `C-COMMON-TRAITS`, `C-CTOR`, `C-FEATURE`.

### Library / building
- `M-FEATURES-ADDITIVE` — features only *add*; no `no-std` (use `std` feature); enabling any combo must compile.
- `M-OOBE` — libraries `cargo build` on Tier-1 platforms with no extra tools/env. You own your dep tree's OOBE-ness.
- `M-SYS-CRATES` — `-sys` crates: build via `cc` in `build.rs`, embed sources, optional tools, support static + `libloading`.

### Library / interoperability
- `M-DONT-LEAK-TYPES` — prefer `std` types in public APIs; leak third-party types only for substantial ecosystem benefit or behind a feature.
- `M-ESCAPE-HATCHES` — native-handle wrappers expose `unsafe fn from_native`, `to_native`, `into_native`.
- **`M-TYPES-SEND`** — public types and async futures should be `Send` for Tokio compatibility.

### Library / resilience
- **`M-AVOID-STATICS`** — `static`/thread-local items secretly duplicate across crate-version boundaries. Don't use them for correctness-relevant state.
- `M-MOCKABLE-SYSCALLS` — I/O, clocks, entropy, anything non-deterministic is mockable via an internal enum dispatch + `test-util` feature.
- `M-NO-GLOB-REEXPORTS` — `pub use foo::{A, B, C}`, not `pub use foo::*` (HAL platform-specific re-exports excepted).
- `M-STRONG-TYPES` — `PathBuf`/`Path` for OS paths, not `String`/`&str`. Use the strongest std type early.
- `M-TEST-UTIL` — testing/mocking functionality gated behind a `test-util` feature.

### Library / UX
- **`M-AVOID-WRAPPERS`** — don't expose `Arc<Mutex<T>>`, `Box<T>`, `Rc<RefCell<T>>` in public APIs; hide them behind clean signatures.
- **`M-DI-HIERARCHY`** — concrete types > generics > `dyn Trait`. Don't translate `IFoo` interfaces from C# verbatim.
- **`M-ERRORS-CANONICAL-STRUCTS`** — errors are structs with `Backtrace` + optional source; expose `is_xxx()` methods over public `ErrorKind`. See xybrid's per-crate error tables in § Error handling above.
- `M-ESSENTIAL-FN-INHERENT` — core functionality is inherent; trait impls forward to inherent methods.
- `M-IMPL-ASREF` — accept `impl AsRef<str>` / `impl AsRef<Path>` / `impl AsRef<[u8]>` for non-owning string/path/byte inputs.
- `M-IMPL-IO` — sans-io: accept `impl std::io::Read`/`Write` (or `futures::io::AsyncRead`) for one-shot init I/O.
- `M-IMPL-RANGEBOUNDS` — range params are `Range<T>` or `impl RangeBounds<T>`, not `(low, high)` tuples.
- **`M-INIT-BUILDER`** — ≥4 init permutations → `FooBuilder` with `Foo::builder()` and chainable methods ending in `.build()`. Required params go to `builder(deps: impl Into<FooDeps>)`.
- `M-INIT-CASCADED` — types with 4+ params group parameters into helper types semantically.
- `M-SERVICES-CLONE` — service types implement `Clone` via `Arc<Inner>` so dependents share a handle.
- **`M-SIMPLE-ABSTRACTIONS`** — service-type generics don't nest visibly. `Foo<Bar<FooBar>>` in a user's field is a smell.

---

# Section 2 — rust-skills (rule index)

179 rules across 14 categories. IDs are descriptive; consult
`github.com/leonardomso/rust-skills/blob/master/rules/<id>.md` for examples.
**Priority order:** apply CRITICAL before HIGH before MEDIUM before LOW.

> *License: MIT. Source:* `https://github.com/leonardomso/rust-skills`

### CRITICAL — Ownership (`own-`)
`own-borrow-over-clone`, `own-slice-over-vec` (`&[T]` not `&Vec<T>`, `&str` not `&String`),
`own-cow-conditional`, `own-arc-shared`, `own-rc-single-thread`,
`own-refcell-interior`, `own-mutex-interior`, `own-rwlock-readers`,
`own-copy-small`, `own-clone-explicit`, `own-move-large`, `own-lifetime-elision`.

### CRITICAL — Errors (`err-`)
`err-thiserror-lib`, `err-anyhow-app`, `err-result-over-panic`,
`err-context-chain`, `err-no-unwrap-prod`, `err-expect-bugs-only` (only for
programming errors, with a message describing the invariant),
`err-question-mark`, `err-from-impl` (`#[from]`), `err-source-chain` (`#[source]`),
`err-lowercase-msg`, `err-doc-errors`, `err-custom-type` (no `Box<dyn Error>` in
public signatures).

### CRITICAL — Memory (`mem-`)
`mem-with-capacity`, `mem-smallvec`, `mem-arrayvec`, `mem-box-large-variant`,
`mem-boxed-slice`, `mem-thinvec`, `mem-clone-from`, `mem-reuse-collections`,
`mem-avoid-format`, `mem-write-over-format`, `mem-arena-allocator`,
`mem-zero-copy`, `mem-compact-string`, `mem-smaller-integers`,
`mem-assert-type-size`.

### HIGH — API design (`api-`)
`api-builder-pattern`, `api-builder-must-use`, `api-newtype-safety`,
`api-typestate`, `api-sealed-trait`, `api-extension-trait`,
`api-parse-dont-validate`, `api-impl-into`, `api-impl-asref`, `api-must-use`,
`api-non-exhaustive`, `api-from-not-into`, `api-default-impl`,
`api-common-traits`, `api-serde-optional`.

### HIGH — Async (`async-`)
`async-tokio-runtime`, **`async-no-lock-await`** (never hold `Mutex`/`RwLock`
guards across `.await`), `async-spawn-blocking` (CPU-bound or sync I/O — e.g.
model loading), `async-tokio-fs`, `async-cancellation-token`,
`async-join-parallel`, `async-try-join`, `async-select-racing`,
`async-bounded-channel`, `async-mpsc-queue`, `async-broadcast-pubsub`,
`async-watch-latest`, `async-oneshot-response`, `async-joinset-structured`,
`async-clone-before-await`.

### HIGH — Compiler optimization (`opt-`)
`opt-inline-small`, `opt-inline-always-rare`, `opt-inline-never-cold`,
`opt-cold-unlikely`, `opt-likely-hint`, `opt-lto-release`, `opt-codegen-units`,
`opt-pgo-profile`, `opt-target-cpu`, `opt-bounds-check` (iterators elide
checks), `opt-simd-portable`, `opt-cache-friendly` (SoA layouts).

### MEDIUM — Naming (`name-`)
Types/variants `UpperCamelCase`; functions/modules `snake_case`; consts
`SCREAMING_SNAKE_CASE`; lifetimes short lowercase (`'a`, `'de`, `'src`); type
params single uppercase. Conversion prefixes: `as_` free, `to_` expensive,
`into_` ownership-transfer. No `get_` for simple getters. Booleans use
`is_`/`has_`/`can_`. Iterators: `iter`/`iter_mut`/`into_iter`. Acronyms as
words (`Uuid`, not `UUID`). No `-rs` crate suffix.

### MEDIUM — Type safety (`type-`)
`type-newtype-ids`, `type-newtype-validated`, `type-enum-states`,
`type-option-nullable`, `type-result-fallible`, `type-phantom-marker`,
`type-never-diverge`, `type-generic-bounds`, `type-no-stringly`,
`type-repr-transparent`.

### MEDIUM — Testing (`test-`)
`test-cfg-test-module`, `test-use-super`, `test-integration-dir`,
`test-descriptive-names`, `test-arrange-act-assert`,
`test-proptest-properties`, `test-mockall-mocking` (note: xybrid uses
`httpmock`, not `mockall` — see § Testing & mocking above), `test-mock-traits`,
`test-fixture-raii`, `test-tokio-async`, `test-should-panic`,
`test-criterion-bench`, `test-doctest-examples`.

### MEDIUM — Documentation (`doc-`)
`doc-all-public`, `doc-module-inner`, `doc-examples-section`,
`doc-errors-section`, `doc-panics-section`, `doc-safety-section`,
`doc-question-mark` (use `?` not `.unwrap()` in examples),
`doc-hidden-setup` (`# ` prefix for example setup),
`doc-intra-links`, `doc-link-types`, `doc-cargo-metadata`.

### MEDIUM — Performance patterns (`perf-`)
`perf-iter-over-index`, `perf-iter-lazy`, `perf-collect-once`,
`perf-entry-api`, `perf-drain-reuse`, `perf-extend-batch`,
`perf-chain-avoid`, `perf-collect-into`, `perf-black-box-bench`,
`perf-release-profile`, `perf-profile-first`.

### LOW — Project structure (`proj-`)
`proj-lib-main-split`, `proj-mod-by-feature`, `proj-flat-small`,
`proj-mod-rs-dir`, `proj-pub-crate-internal`, `proj-pub-super-parent`,
`proj-pub-use-reexport`, `proj-prelude-module`, `proj-bin-dir`,
`proj-workspace-large`, `proj-workspace-deps` (xybrid is partially migrated —
see § Workspace layout above, don't drive-by-convert).

### LOW — Clippy / linting (`lint-`)
`lint-deny-correctness`, `lint-warn-suspicious`, `lint-warn-style`,
`lint-warn-complexity`, `lint-warn-perf`, `lint-pedantic-selective`,
`lint-missing-docs`, `lint-unsafe-doc`, `lint-cargo-metadata`,
`lint-rustfmt-check`, `lint-workspace-lints` (open question — see § Open
questions above).

### REFERENCE — Anti-patterns (`anti-`)
Don't: `anti-unwrap-abuse`, `anti-expect-lazy`, `anti-clone-excessive`,
**`anti-lock-across-await`**, `anti-string-for-str`, `anti-vec-for-slice`,
`anti-index-over-iter`, `anti-panic-expected`, `anti-empty-catch`,
`anti-over-abstraction`, `anti-premature-optimize`, `anti-type-erasure`
(`impl Trait` over `Box<dyn Trait>` when possible), `anti-format-hot-path`,
`anti-collect-intermediate`, `anti-stringly-typed`.

---

## Recommended `Cargo.toml` release profile (from rust-skills)

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true

[profile.bench]
inherits = "release"
debug = true
strip = false

[profile.dev.package."*"]
opt-level = 3  # optimize deps in dev builds
```

Cross-check against xybrid's actual `Cargo.toml` before changing — some of
these (notably `panic = "abort"`) interact with the FFI boundary in ways the
project may have decided against.
