# CLAUDE.md — xybrid project context for AI agents

@AGENTS.md

The file above pulls in the Microsoft Pragmatic Rust Guidelines and the
`rust-skills` ruleset. **This file overrides them where xybrid has already
made a decision.** Read it before writing code.

---

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
| `crates/xybrid-ffi`            | C ABI for Unity / C / C++                                  | FFI      |
| `crates/xybrid-uniffi`         | UniFFI bindings for Swift / Kotlin (Apple/Android SDKs)    | FFI      |
| `bindings/flutter/rust`        | flutter_rust_bridge wrapper for Dart                       | FFI      |
| `macros`                       | proc-macros (`xybrid-macros`); syn/quote only              | proc     |
| `xtask`                        | build / codegen automation                                 | tool     |
| `integration-tests`            | end-to-end tests with real models & fixtures               | test     |

**Dependency direction (do not reverse):**

```
xybrid-cli  ─┐
xybrid-ffi  ─┤
xybrid-uniffi ─┼─► xybrid-sdk ─► xybrid-core
flutter rust──┤
xtask ──────────────────────────► xybrid-core
integration-tests ──────────────► xybrid-core
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

Rust-API library crates (`xybrid-core`, `xybrid-sdk`, `xybrid-uniffi`) use
**`thiserror`** with a single canonical error enum and a `Result` alias per
crate:

| Crate           | Error type     | Result alias    | Defined in                        |
|-----------------|----------------|-----------------|-----------------------------------|
| `xybrid-core`   | `XybridError`  | `XybridResult`  | `crates/xybrid-core/src/error.rs` |
| `xybrid-sdk`    | `SdkError`     | `SdkResult`     | `crates/xybrid-sdk/src/model.rs`  |
| `xybrid-uniffi` | `XybridError`  | —               | `crates/xybrid-uniffi/src/lib.rs` (also derives `uniffi::Error`) |

Sub-error enums (`InferenceError`, `PipelineError`, `AdapterError`, …) live
next to the modules that raise them and convert into the canonical type via
`#[from]` / `impl From`. Follow that pattern for new modules — don't invent
parallel top-level error types.

`xybrid-ffi` is **different**: it's a C-ABI crate and uses opaque handles
plus error strings/codes carried in result structs (see
`crates/xybrid-ffi/src/lib.rs`). Don't bolt a public `thiserror` enum onto
it — match the existing C-ABI pattern when adding new endpoints, and only
surface error info through the documented handle/result conventions.

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
