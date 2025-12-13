# Xybrid Core

The Rust orchestrator runtime for hybrid cloud-edge AI inference. This library provides the core components for building applications that intelligently route AI model inference between local devices and cloud services.

## Overview

Xybrid Core implements a policy-driven orchestrator that:
- **Evaluates policies** to enforce data-handling and routing rules
- **Makes routing decisions** based on device metrics, policy results, and model availability
- **Executes inference** on local or cloud targets with fallback handling
- **Manages streaming** data flows for real-time applications
- **Collects telemetry** for observability and performance monitoring
- **Provides event-driven communication** between components

## Architecture

The orchestrator follows a five-stage runtime flow:

1. **Policy Evaluation** - Check if the request is allowed and determine constraints
2. **Routing Decision** - Choose execution target (local, cloud, or fallback)
3. **Execution** - Run the model inference on the selected target
4. **Telemetry** - Emit observability data
5. **Event Broadcasting** - Notify subscribers of pipeline events

## Core Components

### Orchestrator

The main entry point that coordinates pipeline execution.

```rust
use xybrid_core::orchestrator::Orchestrator;
use xybrid_core::context::{Envelope, EnvelopeKind, DeviceMetrics, StageDescriptor};
use xybrid_core::routing_engine::LocalAvailability;

// Create a new orchestrator
let mut orchestrator = Orchestrator::new();

// Execute a single stage
let stage = StageDescriptor { name: "asr".to_string() };
let input = Envelope::new(EnvelopeKind::Audio(vec![0u8; 1600]));
let metrics = DeviceMetrics {
    network_rtt: 100,
    battery: 80,
    temperature: 25.0,
};
let availability = LocalAvailability::new(true);

let result = orchestrator.execute_stage(&stage, &input, &metrics, &availability)?;

// Execute a multi-stage pipeline
let stages = vec![
    StageDescriptor { name: "asr".to_string() },
    StageDescriptor { name: "tts".to_string() },
];
let results = orchestrator.execute_pipeline(&stages, &input, &metrics, &|s| {
    LocalAvailability::new(s == "asr") // ASR available locally, TTS in cloud
})?;
```

### Policy Engine

Evaluates policies to determine if requests are allowed and what constraints apply.

```rust
use xybrid_core::policy_engine::{PolicyEngine, DefaultPolicyEngine};

let mut engine = DefaultPolicyEngine::new();

// Load policies from YAML or JSON
let policy_yaml = r#"
version: "0.1.0"
rules:
  - id: "audio_rule"
    expression: "input.kind == \"AudioRaw\""
    action: "deny"
signature: "test-signature"
"#;

engine.load_policies(policy_yaml.as_bytes().to_vec())?;

// Evaluate policy for a request
let result = engine.evaluate("asr", &input, &metrics);
if !result.allowed {
    println!("Request denied: {:?}", result.reason);
}
```

### Routing Engine

Makes intelligent routing decisions based on device metrics, policy results, and model availability.

```rust
use xybrid_core::routing_engine::{RoutingEngine, DefaultRoutingEngine, LocalAvailability};

let mut routing_engine = DefaultRoutingEngine::new();

// Make a routing decision
let decision = routing_engine.decide(
    "asr",
    &metrics,
    &policy_result,
    &LocalAvailability::new(true)
);

match decision.target {
    RouteTarget::Local => println!("Executing locally"),
    RouteTarget::Cloud => println!("Executing in cloud"),
    RouteTarget::Fallback(id) => println!("Using fallback: {}", id),
}
```

### Executor

Executes model inference on the selected target.

```rust
use xybrid_core::executor::{Executor, DefaultExecutor, RouteTarget};

let executor = DefaultExecutor::new();

// Execute on a specific target
let (output, metadata) = executor.execute(
    "asr",
    &input,
    &RouteTarget::Local
)?;

println!("Execution time: {}ms", metadata.execution_time_ms);
```

### Device Adapter

Collects real-time device metrics for routing decisions.

```rust
use xybrid_core::device_adapter::{DeviceAdapter, LocalDeviceAdapter};

let adapter = LocalDeviceAdapter::new();
let metrics = adapter.collect_metrics();

println!("Network RTT: {}ms", metrics.network_rtt);
println!("Battery: {}%", metrics.battery);
println!("Temperature: {}°C", metrics.temperature);
```

### Registry

Manages storage and retrieval of bundles (policies, models, configs).

```rust
use xybrid_core::registry::{Registry, LocalRegistry};

// Create a registry (defaults to ~/.xybrid/registry)
let mut registry = LocalRegistry::default()?;

// Store a bundle
let bundle_data = b"model binary data".to_vec();
let metadata = registry.store_bundle("model-id", "1.0.0", bundle_data)?;

// Retrieve a bundle
let retrieved = registry.get_bundle("model-id", Some("1.0.0"))?;

// List all bundles
let bundles = registry.list_bundles()?;
```

### Telemetry

Collects and exports structured observability data.

```rust
use xybrid_core::telemetry::Telemetry;

let telemetry = Telemetry::new();

// Log events
telemetry.log_stage_start("asr");
telemetry.log_routing_decision("asr", "local", "low_latency");
telemetry.log_stage_complete("asr", "local", 50, None);
```

### Event Bus

Pub/sub mechanism for event-driven communication.

```rust
use xybrid_core::event_bus::{EventBus, OrchestratorEvent};

let event_bus = EventBus::new();

// Subscribe to events
let subscription = event_bus.subscribe();

// Publish events
event_bus.publish(OrchestratorEvent::StageStart {
    stage_name: "asr".to_string(),
});

// Receive events
if let Ok(event) = subscription.try_recv() {
    match event {
        OrchestratorEvent::StageStart { stage_name } => {
            println!("Stage started: {}", stage_name);
        }
        _ => {}
    }
}
```

### Stream Manager

Manages streaming data flows for real-time applications.

```rust
use xybrid_core::stream_manager::{StreamManager, StreamConfig};

let config = StreamConfig {
    buffer_size: 1000,
    chunk_size: 512,
};

let mut stream_manager = StreamManager::with_config(config);

// Push input chunks
stream_manager.push_input_chunk(input.clone(), false)?;
stream_manager.push_input_chunk(input.clone(), true)?; // last chunk

// Pop output chunks
while let Some(output) = stream_manager.pop_output_chunk()? {
    // Process output chunk
}
```

## Examples

The `examples/` directory contains several examples:

- **`hiiipe.rs`** - Complete Hiiipe pipeline demonstration (ASR → Motivator → TTS)
- **`device_metrics.rs`** - Device metrics collection example
- **`registry_demo.rs`** - Registry bundle storage and retrieval

Run an example:

```bash
cargo run --example hiiipe
cargo run --example device_metrics
cargo run --example registry_demo
```

## Testing

The library includes comprehensive unit tests and integration tests:

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run integration tests
cargo test --test system_validation -- --nocapture
```

## Module Structure

- **`orchestrator`** - Main orchestrator coordination logic
- **`policy_engine`** - Policy evaluation and enforcement
- **`routing_engine`** - Routing decision making
- **`executor`** - Model execution on local/cloud targets
- **`stream_manager`** - Streaming data flow management
- **`telemetry`** - Observability and metrics collection
- **`event_bus`** - Event-driven pub/sub communication
- **`context`** - Shared data structures (Envelope, DeviceMetrics, etc.)
- **`device_adapter`** - Device metrics collection
- **`registry`** - Bundle storage and retrieval
- **`control_sync`** - Control plane synchronization (TODO)

## Dependencies

- `serde` / `serde_json` / `serde_yaml` - Serialization
- `anyhow` - Error handling
- `dirs` - Platform-specific directories

## License

See the main repository LICENSE file.

