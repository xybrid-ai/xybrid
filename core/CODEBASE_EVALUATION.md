# Codebase Evaluation: `core`

## 1. Executive Summary

The `core` crate contains the fundamental logic for the Xybrid orchestrator.
**Recent Updates:** The removal of legacy `registry` code has significantly improved the separation of concerns regarding model downloading. The `StageDescriptor` now cleanly separates the *definition* of a stage from the *mechanics* of acquiring its assets.

However, critical architectural issues remain:
1.  **Monolithic Files**: `TemplateExecutor` (`src/template_executor/executor_impl.rs`) is a massive 3400+ LOC file that handles too many disparate responsibilities.
2.  **Flat Module Structure**: While `registry` is gone, other areas like `orchestrator`, `policy`, and `routing` are still loosely organized.

## 2. Issues Identification

### A. High LOC Files (The "God Classes")

| File | LOC | Primary Issue |
|------|-----|---------------|
| `src/template_executor/executor_impl.rs` | ~3433 | **Critical**. Handles metadata interpretation, pipeline flow, generic execution, AND specific runtime implementations (Candle, TTS, etc.). |
| `src/device/capabilities.rs` | ~1367 | **Moderate**. Bundles hardware detection for all platforms (Metal, Vulkan, NPU) into one file. |
| `src/executor.rs` | ~1043 | **improved**. Removal of registry logic helped (~100 LOC reduction), but it still mixes stage execution with adapter selection. |
| `src/execution_template.rs` | ~902 | **Minor**. Large collection of data types. |

### B. Lack of Modularization

`src/lib.rs` shows improved organization but still has room for grouping:

*   **Orchestration Group**: Consolidates `orchestrator`, `policy_engine`, `routing_engine` into `src/orchestrator/`. (Done)
*   **Streaming Group**: Consolidates `streaming`, `stream_manager` into `src/streaming/`. (Done)
*   **Runtime Adapters**: `runtime_adapter` is well structured, but its interaction with `executor` and `template_executor` is confusing.

### C. Redundancy and Cohesion

- **Executor Structure**: The split between `executor.rs` (Stage Executor) and `template_executor` (Model Executor) is clearer now that registry logic is gone, but the terminology overlap is still confusing for new contributors.

## 3. Rust Best Practices & Principles

### Principle 1: Hierarchy over Flatness
*   **Rule**: If a feature requires more than 2 files, it gets its own directory.
*   **Goal**: `src/lib.rs` should contain high-level modules only.

### Principle 2: Separation of Concerns (The "Executor" Rule)
*   **Rule**: Structs should not know about specific sub-implementations unless via a trait.
*   **Application**: `TemplateExecutor` currently imports `Candle` and `TTS` logic directly. This should be inverted using traits.

### Principle 3: Explicit Public API
*   **Rule**: Use `pub(crate)` effectively. Only types strictly needed by the consumer of `core` should be `pub`.

### Principle 4: Small Files
*   **Target**: Files should ideally be under 500 lines.
*   **Hard Limit**: Files over 1000 lines triggers a mandatory "Refactor Review".

## 4. Refactoring Roadmap (Draft)

1.  **Phase 1: Regrouping (Completed)**
    *   Moved `policy_engine.rs`, `routing_engine.rs` -> `src/orchestrator/`.
    *   Moved `stream_manager.rs` -> `src/streaming/manager.rs`.
    *   Updated imports and module structure.

2.  **Phase 2: Breaking the Monolith (High Impact)**
    *   **Refactor `TemplateExecutor`**:
        *   Extract `ModelRuntime` trait. ✅
        *   Move `Candle` logic to `runtime_adapter::candle`. ✅
        *   Extract TTS voice loading to `src/tts/voice_embedding.rs`. ✅
            *Note: TTS is NOT a runtime - it uses ONNX with Phonemize preprocessing.
            The original `runtime_adapter/tts.rs` was removed as it was architecturally incorrect.*
    *   **Refactor `DeviceCapabilities`**:
        *   Split into `device/apple.rs`, `device/android.rs`, etc.
3.  **Phase 3: Cleanup**
    *   Review `executor.rs` for remaining responsibilities now that registry is gone.

---
*Evaluated by Antigravity*
