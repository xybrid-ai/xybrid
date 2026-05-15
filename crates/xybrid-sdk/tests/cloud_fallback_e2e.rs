//! Cloud-fallback integration tests.
//!
//! Two tiers of coverage:
//!
//! 1. **Compile-time guard** (`cloud_fallback_api_surface_compiles`,
//!    `cloud_fallback_dispatch_with_fake_adapter`): exercises the public API
//!    pieces the demo example uses. Runs in CI; needs no model and no cloud.
//!    Catches accidental rename/shape changes that would silently break the
//!    `cloud_fallback_demo` example.
//!
//! 2. **End-to-end (manual)** (`cloud_fallback_demo_runs_end_to_end` —
//!    `#[ignore]`'d): drives `run_streaming_with_fallback` through a real
//!    cached `qwen2.5-0.5b-instruct` and a mock cloud target. Run with
//!    `cargo test --features platform-macos,dev-tools,llm-llamacpp \
//!     --test cloud_fallback_e2e -- --ignored`.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use xybrid_core::abort::AbortReason as CoreAbortReason;
use xybrid_core::device::{MemoryPressure, ResourceSnapshot, ResourceSnapshotProvider};
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_core::orchestrator::authority::test_seams::{
    FixedResourceProvider, StagedResourceProvider,
};
use xybrid_core::runtime_adapter::types::{GenerationConfig, PartialToken, StreamingCallback};
use xybrid_core::runtime_adapter::{
    AdapterError, AdapterResult, CloudRuntimeAdapter, CloudStreaming,
};
use xybrid_sdk::run_options::{AbortPolicy, AbortSignal, RunOptions};
use xybrid_sdk::SeamInfo;

#[test]
fn cloud_fallback_api_surface_compiles() {
    // Each public API piece the demo example reaches for must remain
    // reachable through the SDK's re-exports.
    let mut crit = ResourceSnapshot::unknown();
    crit.memory_pressure = MemoryPressure::Critical;

    let provider: Arc<dyn ResourceSnapshotProvider> =
        Arc::new(StagedResourceProvider::new(3, crit));

    let _options = RunOptions::new()
        .with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_cloud_fallback(true)
                .with_max_grace_tokens(0),
        )
        .with_resource_provider(provider.clone());

    let _adapter: Box<dyn CloudStreaming> =
        Box::new(CloudRuntimeAdapter::with_gateway("http://example.test"));

    let _seam = SeamInfo {
        reason: CoreAbortReason::StressMemory,
        correlation_id: "run-1".to_string(),
        local_tokens: 3,
        local_latency_ms: 100,
    };

    let _fixed: Arc<dyn ResourceSnapshotProvider> =
        Arc::new(FixedResourceProvider::new(ResourceSnapshot::unknown()));
}

/// Records calls and emits a fixed response as one synthetic token. Stand-in
/// for `CloudRuntimeAdapter` so this test runs without a network.
struct RecordingCloud {
    response: String,
    calls: Mutex<Vec<Envelope>>,
    tokens_emitted: AtomicU32,
}

impl RecordingCloud {
    fn new(response: &str) -> Self {
        Self {
            response: response.to_string(),
            calls: Mutex::new(Vec::new()),
            tokens_emitted: AtomicU32::new(0),
        }
    }
}

impl CloudStreaming for RecordingCloud {
    fn execute_streaming(
        &self,
        input: &Envelope,
        mut on_token: StreamingCallback<'_>,
    ) -> AdapterResult<Envelope> {
        self.calls.lock().unwrap().push(input.clone());
        let token = PartialToken {
            token: self.response.clone(),
            token_id: None,
            index: 0,
            cumulative_text: self.response.clone(),
            finish_reason: Some("stop".to_string()),
        };
        on_token(token).map_err(|e| AdapterError::InferenceFailed(format!("{}", e)))?;
        self.tokens_emitted.fetch_add(1, Ordering::SeqCst);
        Ok(Envelope::new(EnvelopeKind::Text(self.response.clone())))
    }
}

#[derive(Debug)]
struct TimedStagedResourceProvider {
    normal: ResourceSnapshot,
    stressed: ResourceSnapshot,
    normal_reads: usize,
    reads_so_far: AtomicUsize,
    first_stressed_read_at: Mutex<Option<Instant>>,
}

impl TimedStagedResourceProvider {
    fn new(normal_reads: usize, stressed: ResourceSnapshot) -> Self {
        Self {
            normal: ResourceSnapshot::unknown(),
            stressed,
            normal_reads,
            reads_so_far: AtomicUsize::new(0),
            first_stressed_read_at: Mutex::new(None),
        }
    }

    fn first_stressed_read_at(&self) -> Option<Instant> {
        *self.first_stressed_read_at.lock().unwrap()
    }
}

impl ResourceSnapshotProvider for TimedStagedResourceProvider {
    fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
        let n = self.reads_so_far.fetch_add(1, Ordering::SeqCst);
        if n < self.normal_reads {
            self.normal
        } else {
            let mut first_stressed_read_at = self.first_stressed_read_at.lock().unwrap();
            if first_stressed_read_at.is_none() {
                *first_stressed_read_at = Some(Instant::now());
            }
            self.stressed
        }
    }
}

#[test]
fn cloud_fallback_dispatch_with_fake_adapter() {
    // Sanity check that a `CloudStreaming` implementation talks to its
    // callback the way the wrapper expects: one envelope clone in, one
    // PartialToken out, one envelope clone out.
    let cloud = RecordingCloud::new("hello from cloud");
    let envelope = Envelope::new(EnvelopeKind::Text("prompt".to_string()));

    let received: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let received_for_cb = received.clone();
    let cb: StreamingCallback<'_> = Box::new(move |t: PartialToken| {
        received_for_cb.lock().unwrap().push(t.token);
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
    });

    let out = cloud
        .execute_streaming(&envelope, cb)
        .expect("fake cloud should succeed");

    assert_eq!(cloud.calls.lock().unwrap().len(), 1);
    assert_eq!(cloud.tokens_emitted.load(Ordering::SeqCst), 1);
    assert_eq!(received.lock().unwrap().len(), 1);
    assert_eq!(received.lock().unwrap()[0], "hello from cloud");
    match out.kind {
        EnvelopeKind::Text(t) => assert_eq!(t, "hello from cloud"),
        _ => panic!("expected Text envelope back from RecordingCloud"),
    }
}

/// End-to-end exercise of `run_streaming_with_fallback` against a real local
/// LLM and a mock cloud. Excluded from default `cargo test` because it
/// requires `qwen2.5-0.5b-instruct` to be present in the registry cache.
///
/// Run manually:
///
/// ```bash
/// xybrid run --model qwen2.5-0.5b-instruct --input-text "warmup"
/// cargo test --features platform-macos,dev-tools,llm-llamacpp \
///   --test cloud_fallback_e2e -- --ignored
/// ```
#[test]
#[ignore]
fn cloud_fallback_demo_runs_end_to_end() {
    use xybrid_sdk::ModelLoader;

    const FALLBACK_ABORT_LATENCY_BUDGET: Duration = Duration::from_millis(200);

    let model_id = std::env::var("XYBRID_FALLBACK_E2E_MODEL_ID")
        .unwrap_or_else(|_| "qwen2.5-0.5b-instruct".to_string());
    let model = ModelLoader::from_registry(&model_id)
        .load()
        .unwrap_or_else(|err| panic!("{model_id} must be cached for this test: {err}"));

    let mut crit = ResourceSnapshot::unknown();
    crit.memory_pressure = MemoryPressure::Critical;
    let provider = Arc::new(TimedStagedResourceProvider::new(2, crit));

    let cloud = RecordingCloud::new("hello from cloud");

    let options = RunOptions::new()
        .with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_cloud_fallback(true)
                .with_max_grace_tokens(0),
        )
        .with_resource_provider(provider.clone())
        .with_generation_config(GenerationConfig::greedy().with_max_tokens(256));

    let mut envelope = Envelope::new(EnvelopeKind::Text(
        "Write a long numbered list of short reasons adaptive compute keeps mobile LLM apps responsive. Do not stop early.".to_string(),
    ));
    envelope
        .metadata
        .insert("provider".to_string(), "openai".to_string());
    envelope
        .metadata
        .insert("model".to_string(), "gpt-4o-mini".to_string());

    let local_count = Arc::new(AtomicU32::new(0));
    let cloud_count = Arc::new(AtomicU32::new(0));
    let on_cloud = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let captured_seam: Arc<Mutex<Option<SeamInfo>>> = Arc::new(Mutex::new(None));
    let seam_observed_at: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));

    let local_count_cb = local_count.clone();
    let cloud_count_cb = cloud_count.clone();
    let on_cloud_cb = on_cloud.clone();
    let mut on_token =
        move |_: PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            if on_cloud_cb.load(Ordering::SeqCst) {
                cloud_count_cb.fetch_add(1, Ordering::SeqCst);
            } else {
                local_count_cb.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        };

    let on_cloud_seam = on_cloud.clone();
    let captured_seam_for_cb = captured_seam.clone();
    let seam_observed_at_for_cb = seam_observed_at.clone();
    let mut on_seam = move |info: SeamInfo| {
        on_cloud_seam.store(true, Ordering::SeqCst);
        *seam_observed_at_for_cb.lock().unwrap() = Some(Instant::now());
        *captured_seam_for_cb.lock().unwrap() = Some(info);
    };

    let result = model
        .run_streaming_with_fallback(&envelope, &options, &cloud, &mut on_token, &mut on_seam)
        .expect("fallback should succeed against a fake cloud");

    let seam = captured_seam.lock().unwrap().clone();
    assert!(seam.is_some(), "seam should fire when pressure trips");
    let first_stressed_read_at = provider
        .first_stressed_read_at()
        .expect("provider should have returned a stressed snapshot");
    let seam_observed_at = (*seam_observed_at.lock().unwrap())
        .expect("seam callback should record when abort surfaced");
    let abort_latency = seam_observed_at.duration_since(first_stressed_read_at);
    assert!(
        abort_latency <= FALLBACK_ABORT_LATENCY_BUDGET,
        "local abort exceeded one-token low-end budget: {:?} > {:?}",
        abort_latency,
        FALLBACK_ABORT_LATENCY_BUDGET
    );
    assert!(
        local_count.load(Ordering::SeqCst) >= 1,
        "local should emit at least one token"
    );
    assert!(
        cloud_count.load(Ordering::SeqCst) >= 1,
        "cloud should emit at least one token"
    );
    assert_eq!(result.text(), Some("hello from cloud"));
    assert_eq!(cloud.calls.lock().unwrap().len(), 1);
}
