//! Criterion bench for local routing feedback.
//!
//! SLO defended:
//!   * LocalAuthority::record_outcome write: <= 1 us
//!
//! Protects the bounded-window write path used by the local authority
//! when the routing engine biases against repeatedly-failing local runs.
//!
//! Run with:
//!   cargo bench -p xybrid-core --bench routing_feedback

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xybrid_core::device::{MemoryPressure, ThermalState};
use xybrid_core::orchestrator::authority::{
    ExecutionOutcome, LocalAuthority, OrchestrationAuthority, OutcomeCategory, SignalContext,
};
use xybrid_core::orchestrator::ResolvedTarget;

fn signal() -> SignalContext {
    SignalContext {
        memory_pressure: MemoryPressure::Warn,
        thermal_state: ThermalState::Normal,
        cpu_bucket: Some(5),
    }
}

fn local_failure_outcome() -> ExecutionOutcome {
    ExecutionOutcome {
        stage_id: "stage".to_string(),
        target: ResolvedTarget::Device,
        latency_ms: 10,
        success: false,
        error: Some("local_failed".to_string()),
        category: Some(OutcomeCategory::HardFail {
            reason: "local_failed".to_string(),
        }),
        model_id: Some("model".to_string()),
        signal_context: Some(signal()),
    }
}

fn bench_record_outcome(c: &mut Criterion) {
    let authority = LocalAuthority::default();
    let outcome = local_failure_outcome();

    c.bench_function("local_authority::record_outcome", |b| {
        b.iter(|| {
            authority.record_outcome(black_box(&outcome));
        })
    });
}

criterion_group!(benches, bench_record_outcome);
criterion_main!(benches);
