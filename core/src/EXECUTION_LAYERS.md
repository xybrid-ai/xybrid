# Execution Layer Architecture

This document clarifies the responsibilities of the three execution layers in xybrid-core.

## Layer Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       Orchestrator                          │
│   Policy evaluation, routing decisions, stream management   │
├─────────────────────────────────────────────────────────────┤
│                         Executor                            │
│   Adapter registry, target selection, model loading         │
├─────────────────────────────────────────────────────────────┤
│                     TemplateExecutor                        │
│   Preprocessing, model inference, postprocessing            │
└─────────────────────────────────────────────────────────────┘
```

## 1. Orchestrator (`orchestrator.rs`)

**Purpose**: High-level coordination of hybrid cloud-edge inference pipelines.

**Responsibilities**:
- Evaluate policies (should this request be allowed?)
- Make routing decisions (local vs edge vs cloud)
- Manage streaming state (chunk buffering, real-time processing)
- Emit telemetry events
- Coordinate multi-stage pipelines

**Key Methods**:
- `execute_single_stage()` - Run one stage of a pipeline
- `execute_pipeline()` - Run a complete multi-stage pipeline
- `push_and_execute_stream_chunk()` - Handle streaming audio

**Dependencies**: Executor, PolicyEngine, RoutingEngine, StreamManager, Telemetry

**When to use**: Application entry point for inference requests.

## 2. Executor (`executor.rs`)

**Purpose**: Execute model inference stages using the appropriate runtime adapter.

**Responsibilities**:
- Maintain registry of runtime adapters
- Select adapter based on execution target (device, cloud)
- Load models from .xyb bundles or registry
- Delegate to TemplateExecutor for actual inference
- Handle cloud API calls (OpenAI, Anthropic) for cloud stages

**Key Methods**:
- `execute_stage()` - Execute a single stage with routing
- `execute_local_stage()` - Execute on device using ONNX
- `execute_cloud()` - Execute via cloud API

**Dependencies**: TemplateExecutor, RuntimeAdapter, Cloud

**When to use**: Direct model execution when routing is already decided.

## 3. TemplateExecutor (`template_executor/`)

**Purpose**: Metadata-driven model execution without hard-coding model-specific logic.

**Responsibilities**:
- Interpret `model_metadata.json` configuration
- Run preprocessing steps (mel spectrogram, tokenization, phonemization)
- Execute ONNX models with correct input/output handling
- Run postprocessing steps (CTC decode, argmax, softmax)
- Handle multi-stage execution modes (SingleShot, Autoregressive, WhisperDecoder)
- Cache ONNX sessions for efficiency

**Key Methods**:
- `execute()` - Main entry point: preprocessing → inference → postprocessing
- `run_preprocessing()` - Apply preprocessing steps from metadata
- `run_postprocessing()` - Apply postprocessing steps from metadata

**Dependencies**: ONNXSession, MelConfig, Phonemizer

**When to use**: Low-level model inference given a model directory.

## Data Flow

```
Application Request
        │
        ▼
┌───────────────────┐
│   Orchestrator    │  ← Policy check: allow/deny
│                   │  ← Routing: local/edge/cloud
└───────────────────┘
        │
        ▼
┌───────────────────┐
│     Executor      │  ← Adapter selection
│                   │  ← Model loading (.xyb bundles)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ TemplateExecutor  │  ← Preprocess (mel, tokenize)
│                   │  ← ONNX inference
│                   │  ← Postprocess (decode)
└───────────────────┘
        │
        ▼
    Envelope (output)
```

## When to Use Each Layer

| Use Case | Layer to Call |
|----------|---------------|
| Application inference request | Orchestrator |
| Known target, skip routing | Executor |
| Direct model execution with metadata | TemplateExecutor |
| CLI model testing | TemplateExecutor |
| Streaming audio processing | Orchestrator |
| LLM API integration | Executor |

## Key Differences

| Aspect | Orchestrator | Executor | TemplateExecutor |
|--------|-------------|----------|------------------|
| Scope | Pipeline orchestration | Stage execution | Model inference |
| Routing | Decides route | Follows route | N/A |
| Policy | Evaluates | N/A | N/A |
| Streaming | Manages | N/A | N/A |
| Model loading | N/A | Yes | Yes (cached) |
| Preprocessing | N/A | N/A | Yes |
| Postprocessing | N/A | N/A | Yes |

## Example Usage

### High-level (Application)
```rust
let orchestrator = Orchestrator::new(policy, routing, executor);
let result = orchestrator.execute_single_stage(&stage, input, &metrics).await?;
```

### Mid-level (Direct Execution)
```rust
let executor = Executor::new();
let result = executor.execute_local_stage(&stage, input)?;
```

### Low-level (Model Testing)
```rust
let mut executor = TemplateExecutor::with_base_path("test_models/whisper");
let metadata: ModelMetadata = serde_json::from_str(&config)?;
let output = executor.execute(&metadata, &input)?;
```
