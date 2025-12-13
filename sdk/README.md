# Xybrid SDK

Developer-facing API for hybrid cloud-edge AI inference with declarative routing annotations.

## Overview

The Xybrid SDK provides high-level abstractions and macros for building hybrid inference pipelines. It allows developers to annotate functions with `#[hybrid::route]` to enable automatic orchestrator-based routing between local and cloud execution.

## Usage

### Basic Example

```rust
use xybrid_sdk::hybrid;

#[hybrid::route]
fn asr_stage(input: String) -> String {
    // This function will be routed by the orchestrator
    // based on policy, metrics, and availability
    format!("asr_output: {}", input)
}
```

### Using Common Types

```rust
use xybrid_sdk::prelude::*;

// Use orchestrator directly
let mut orchestrator = Orchestrator::new();
let envelope = Envelope { kind: "AudioRaw".to_string() };
// ...
```

## Macro Status

The `#[hybrid::route]` macro is currently a **placeholder**. It:
- ✅ Compiles and can be used on functions
- ✅ Preserves the function signature and behavior
- ⏳ Does not yet transform functions to use the orchestrator

### Future Implementation

Future versions of the macro will:
1. Extract function metadata (name, parameters, return type)
2. Generate orchestrator calls automatically
3. Wrap function body with `orchestrator.execute_stage()` calls
4. Handle input/output envelope conversion
5. Inject `DeviceMetrics` and `LocalAvailability` handling

## Project Structure

- **`xybrid-sdk`**: Main SDK crate that re-exports `xybrid-core` and provides the macro
- **`xybrid-macros`**: Procedural macro crate implementing `#[hybrid::route]`

## Examples

See `examples/macro_demo.rs` for a complete example of using the macro.

```bash
cargo run --example macro_demo
```

