# Contributing to Xybrid

Thank you for your interest in contributing to Xybrid! This guide will help you get started.

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
2. **Create a branch** from `main`:
   ```bash
   git checkout -b your-feature-name
   ```
3. **Make your changes** — keep commits focused and minimal
4. **Ensure quality checks pass:**
   ```bash
   cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo fmt --all -- --check
   ```
5. **Push** your branch and open a Pull Request against `main`
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

## Getting Help

- **Questions?** Open a [GitHub Issue](https://github.com/xybrid-ai/xybrid/issues) or ask on [Discord](https://discord.gg/xybrid)
- **Bug reports** — use the [bug report template](https://github.com/xybrid-ai/xybrid/issues/new?template=bug_report.md)
- **Feature requests** — use the [feature request template](https://github.com/xybrid-ai/xybrid/issues/new?template=feature_request.md)

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
