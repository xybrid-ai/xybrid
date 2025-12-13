# Xybrid Macros

Procedural macro crate for the Xybrid SDK.

## Purpose

This crate provides the `#[hybrid::route]` procedural macro attribute that transforms functions to use the Xybrid orchestrator for hybrid routing.

## Implementation Status

**Current (MVP)**: The macro is a placeholder that:
- Parses the input function
- Passes it through unchanged
- Preserves all attributes, visibility, signature, and body

**Future**: The macro will transform functions to:
- Create `StageDescriptor` from function metadata
- Wrap function calls with `orchestrator.execute_stage()`
- Handle `Envelope` input/output conversion
- Inject policy evaluation and routing logic
- Handle `DeviceMetrics` and `LocalAvailability`

## Usage

The macro is used via the SDK:

```rust
use xybrid_sdk::hybrid;

#[hybrid::route]
fn my_function(input: String) -> String {
    // Implementation
}
```

## Dependencies

- `syn`: For parsing Rust syntax
- `quote`: For generating Rust code
- `proc-macro2`: For procedural macro support

