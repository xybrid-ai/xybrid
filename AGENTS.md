# AGENTS.md — Rust guidance for AI coding agents in xybrid

Project-specific decisions live in [`CLAUDE.md`](./CLAUDE.md) — **CLAUDE.md
wins** when it disagrees with anything below. This file is a compressed index
of the two upstream guides we follow. Fetch them when you need detail on a
specific rule:

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

**Releases are cut by branch name, not by hand.** Push a `release/v<version>`
branch and `.github/workflows/release-prep.yml` does the rest (manifest-version
check, artifact builds, `Package.swift` checksum patch, draft GitHub Release, and
the `Release v<version>` PR to master). Don't `gh pr create` a release or run
`just bump-version` on a feature branch and open a PR from it — that bypasses the
pipeline and ships nothing. Never commit `Package.swift useLocalNatives = true`
(local-dev only; breaks remote SPM — run `bindings/apple/scripts/set-natives-mode.sh
--set-remote`). Full ritual + gotchas: **CLAUDE.md § Releases**.

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
- **`M-ISOLATE-DLL-STATE`** — only portable (`#[repr(C)]`, no statics/TypeId/non-portable refs) data crosses DLL boundaries. Critical for `xybrid-ffi`.

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
- **`M-ERRORS-CANONICAL-STRUCTS`** — errors are structs with `Backtrace` + optional source; expose `is_xxx()` methods over public `ErrorKind`. See xybrid's per-crate error tables in CLAUDE.md.
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
`httpmock`, not `mockall` — see CLAUDE.md), `test-mock-traits`,
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
see CLAUDE.md, don't drive-by-convert).

### LOW — Clippy / linting (`lint-`)
`lint-deny-correctness`, `lint-warn-suspicious`, `lint-warn-style`,
`lint-warn-complexity`, `lint-warn-perf`, `lint-pedantic-selective`,
`lint-missing-docs`, `lint-unsafe-doc`, `lint-cargo-metadata`,
`lint-rustfmt-check`, `lint-workspace-lints` (open question for xybrid — see
CLAUDE.md).

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
these (notably `panic = "abort"`) interact with FFI/UniFFI in ways the
project may have decided against.
