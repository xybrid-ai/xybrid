# Contributing to Xybrid

Thank you for your interest in contributing to Xybrid! This guide will help you get started.

## Where to start

If you're looking for a first task, browse the [`good first issue`](https://github.com/xybrid-ai/xybrid/labels/good%20first%20issue) label — these are scoped to be self-contained, with clear acceptance criteria and no platform-internals knowledge required.

Issues are also grouped by area so you can find ones that match your interest:

- [`area: core`](https://github.com/xybrid-ai/xybrid/labels/area%3A%20core) — `xybrid-core` (execution, audio, runtime adapters)
- [`area: sdk`](https://github.com/xybrid-ai/xybrid/labels/area%3A%20sdk) — `xybrid-sdk` (registry client, high-level Rust API)
- [`area: examples`](https://github.com/xybrid-ai/xybrid/labels/area%3A%20examples) — example apps and runnable demos
- [`area: bindings`](https://github.com/xybrid-ai/xybrid/labels/area%3A%20bindings) — Flutter, Kotlin, Swift, Unity bindings
- [`area: tests`](https://github.com/xybrid-ai/xybrid/labels/area%3A%20tests) — test coverage and the test harness

Medium-difficulty tasks are labeled [`help wanted`](https://github.com/xybrid-ai/xybrid/labels/help%20wanted). If you want to claim an issue, leave a comment so we can avoid duplicate work — no formal assignment process is needed.

## Prerequisites

- **Rust** 1.75+ with `cargo` ([rustup.rs](https://rustup.rs))
- **just** task runner ([github.com/casey/just](https://github.com/casey/just))
- **Git** for version control
- **Flutter** 3.x, **Xcode** 15+, or **Android NDK** (only if working on those bindings)

## Dev Environment Setup

```bash
git clone https://github.com/xybrid-ai/xybrid.git
cd xybrid
cargo build --workspace
cargo test --workspace
```

## Building

```bash
cargo build --workspace                   # Build all packages
cargo build --workspace --release         # Release mode
cargo xtask build-xcframework             # Apple XCFramework (iOS + macOS)
cargo xtask build-android                 # Android .so libraries
cargo xtask build-flutter                 # Flutter native libraries
```

## Testing

```bash
cargo test --workspace                    # Unit tests
cargo test --workspace -- --nocapture     # Tests with output
cargo test --workspace --ignored          # Integration tests (requires model fixtures)
cargo clippy --workspace -- -D warnings   # Lints
cargo fmt --all -- --check                # Format check
```

## PR Process

1. **Fork** the repository on GitHub
2. **Create a branch** from `master`:
   ```bash
   git checkout -b your-feature-name
   ```
3. **Make your changes** — keep commits focused and minimal
4. **Ensure quality checks pass:**
   ```bash
   cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all -- --check
   ```
5. **Push** your branch and open a Pull Request against `master`
6. **Respond to review feedback** — a maintainer will review your PR

### PR Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation if behavior changes
- Follow existing code patterns and conventions

## Code Style

**Rust:** Follow standard Rust conventions (`rustfmt` defaults), use `clippy` with warnings as errors, prefer `thiserror` for error types, and keep functions under 50 lines where practical.

**Bindings:** Follow each platform's standard conventions (Dart style for Flutter, Swift API guidelines, Kotlin coding conventions).

## Adding a Model

1. **Create a model directory:**
   ```
   integration-tests/fixtures/models/your-model/
   ├── model_metadata.json    # Execution configuration
   ├── model.onnx             # ONNX model file
   └── (supporting files)     # vocab, tokens, voices, etc.
   ```

2. **Define `model_metadata.json`** with preprocessing and postprocessing steps:
   ```json
   {
     "model_id": "your-model",
     "version": "1.0",
     "execution_template": { "type": "SimpleMode", "model_file": "model.onnx" },
     "preprocessing": [],
     "postprocessing": [],
     "files": ["model.onnx"]
   }
   ```

3. **Create a test example** in `crates/xybrid-core/examples/`:
   ```rust
   use xybrid_core::execution::{ModelMetadata, TemplateExecutor};
   use xybrid_core::ir::Envelope;

   fn main() -> Result<(), Box<dyn std::error::Error>> {
       let model_dir = "integration-tests/fixtures/models/your-model";
       let metadata: ModelMetadata = serde_json::from_str(
           &std::fs::read_to_string(format!("{model_dir}/model_metadata.json"))?
       )?;
       let mut executor = TemplateExecutor::with_base_path(model_dir);
       let output = executor.execute(&metadata, &Envelope::text("test input"))?;
       println!("{output:?}");
       Ok(())
   }
   ```

4. **Run your example:** `cargo run --example your_model_example`

## Dependencies

How Xybrid selects, declares, and keeps its third-party dependencies up to date.

**Where dependencies are declared**

| Area               | Manifest                                             |
|--------------------|------------------------------------------------------|
| Rust core & SDKs   | `Cargo.toml` (workspace root + per-crate)            |
| Flutter binding    | `bindings/flutter/pubspec.yaml`                      |
| Kotlin / Android   | `bindings/kotlin/build.gradle.kts`                   |
| Unity              | `bindings/unity/package.json`                        |
| Swift              | `Package.swift`                                       |
| Documentation site | `docs/package.json` (pnpm)                           |

Lock files (`Cargo.lock`, `pnpm-lock.yaml`, `pubspec.lock`) are committed so
builds are reproducible.

**Choosing a dependency**

- Prefer the standard library; add a dependency only when it earns its keep.
- Prefer well-maintained, widely-used crates with a license compatible with
  Apache-2.0 (e.g. MIT/Apache-2.0). Avoid copyleft licenses for linked code.
- Match the surrounding crate's existing version-pinning style (see
  `CLAUDE.md`); don't unilaterally migrate conventions.

**Keeping dependencies current**

- [Dependabot](.github/dependabot.yml) opens weekly update PRs for Cargo and
  GitHub Actions dependencies.
- Security advisories across ecosystems are surfaced by Dependabot security
  alerts and the OpenSSF Scorecard / OSV database; known-vulnerable
  dependencies are upgraded promptly.

## Getting Help

- **Questions?** Open a [GitHub Issue](https://github.com/xybrid-ai/xybrid/issues) or ask on [Discord](https://discord.gg/YhFHHkhbad)
- **Bug reports** — use the [bug report template](https://github.com/xybrid-ai/xybrid/issues/new?template=bug_report.md)
- **Feature requests** — use the [feature request template](https://github.com/xybrid-ai/xybrid/issues/new?template=feature_request.md)

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
