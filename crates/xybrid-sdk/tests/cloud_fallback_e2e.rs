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
//! 2. **End-to-end (manual)** (`#[ignore]`'d): drives
//!    `run_streaming_with_fallback` through a real cached
//!    `qwen2.5-0.5b-instruct` and a mock cloud target.
//!    `cloud_fallback_demo_runs_end_to_end` aborts once pressure trips a few
//!    tokens in; `cloud_fallback_fires_when_pressure_trips_on_an_early_token`
//!    and `user_cancel_on_the_first_token_is_reported_as_a_cancel` abort on
//!    the very first tokens. Run with
//!    `cargo test --features platform-macos,dev-tools,llm-llamacpp \
//!     --test cloud_fallback_e2e -- --ignored`.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
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
use xybrid_sdk::run_options::{AbortPolicy, AbortSignal, CancellationToken, RunOptions};
use xybrid_sdk::{SeamInfo, XybridModel};

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
            tool_calls: Vec::new(),
            raw_text: None,
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

/// The abort check samples resources at most every 100 ms; holding a token
/// callback longer than that guarantees the next token takes a fresh sample.
const SAMPLE_GAP: Duration = Duration::from_millis(150);

/// Reports normal resources until armed, critical memory pressure after.
///
/// The first read is the pre-run gate, which always sees normal resources.
/// When the provider is armed before the run, that read also sleeps past the
/// sampling interval, so the first generated token samples the critical
/// snapshot.
#[derive(Debug, Default)]
struct ArmedResourceProvider {
    armed: AtomicBool,
    reads: AtomicUsize,
}

impl ArmedResourceProvider {
    fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl ResourceSnapshotProvider for ArmedResourceProvider {
    fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
        let armed = self.armed.load(Ordering::SeqCst);
        if self.reads.fetch_add(1, Ordering::SeqCst) == 0 {
            if armed {
                std::thread::sleep(SAMPLE_GAP);
            }
            return ResourceSnapshot::unknown();
        }
        let mut snapshot = ResourceSnapshot::unknown();
        if armed {
            snapshot.memory_pressure = MemoryPressure::Critical;
        }
        snapshot
    }
}

/// Cancels `token` from inside the pre-run gate's resource read. The gate has
/// already checked the token by then, so the run starts and sees the cancel
/// on its first generated token.
#[derive(Debug)]
struct CancelOnFirstRead {
    token: CancellationToken,
    reads: AtomicUsize,
}

impl ResourceSnapshotProvider for CancelOnFirstRead {
    fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
        if self.reads.fetch_add(1, Ordering::SeqCst) == 0 {
            self.token.cancel();
        }
        ResourceSnapshot::unknown()
    }
}

/// Loads the cached local model the manual end-to-end tests run on.
/// `XYBRID_FALLBACK_E2E_MODEL_ID` overrides the default.
fn load_fallback_model() -> XybridModel {
    let model_id = std::env::var("XYBRID_FALLBACK_E2E_MODEL_ID")
        .unwrap_or_else(|_| "qwen2.5-0.5b-instruct".to_string());
    xybrid_sdk::ModelLoader::from_registry(&model_id)
        .load()
        .unwrap_or_else(|err| panic!("{model_id} must be cached for this test: {err}"))
}

/// A prompt whose local answer runs long enough for every abort point these
/// tests pick, plus cloud routing metadata the fake cloud ignores.
fn fallback_envelope() -> Envelope {
    let mut envelope = Envelope::new(EnvelopeKind::Text(
        "Write a long numbered list of short reasons adaptive compute keeps mobile LLM apps responsive. Do not stop early.".to_string(),
    ));
    envelope
        .metadata
        .insert("provider".to_string(), "openai".to_string());
    envelope
        .metadata
        .insert("model".to_string(), "gpt-4o-mini".to_string());
    envelope
}

/// Every message in `err`'s source chain, outermost first.
fn error_chain(err: &dyn std::error::Error) -> String {
    let mut messages = vec![err.to_string()];
    let mut source = err.source();
    while let Some(cause) = source {
        messages.push(cause.to_string());
        source = cause.source();
    }
    messages.join(" <- ")
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
    const FALLBACK_ABORT_LATENCY_BUDGET: Duration = Duration::from_millis(200);

    let model = load_fallback_model();

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

    let envelope = fallback_envelope();

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

/// Memory pressure that trips on one of the first generated tokens must still
/// reach the cloud. llama.cpp reports a callback stop as `-n_generated`, which
/// on tokens 1–4 equals one of its hard error codes; the abort has to be read
/// as the fallback it is, not as a failed decode. Token 8 is the later
/// control.
///
/// Run manually (needs the same cached model as the test above):
///
/// ```bash
/// cargo test -p xybrid-sdk --features dev-tools,llm-llamacpp \
///   --test cloud_fallback_e2e -- --ignored early_token
/// ```
#[test]
#[ignore]
fn cloud_fallback_fires_when_pressure_trips_on_an_early_token() {
    let model = load_fallback_model();

    for abort_at in [1u32, 2, 3, 4, 5, 8] {
        let provider = Arc::new(ArmedResourceProvider::default());
        if abort_at == 1 {
            provider.arm();
        }
        let cloud = RecordingCloud::new("hello from cloud");
        let options = RunOptions::new()
            .with_abort_policy(
                AbortPolicy::default()
                    .stop_on(AbortSignal::MemoryPressureCritical)
                    .with_cloud_fallback(true)
                    .with_max_grace_tokens(0),
            )
            .with_resource_provider(provider.clone())
            .with_generation_config(GenerationConfig::greedy().with_max_tokens(64));

        let local_count = Arc::new(AtomicU32::new(0));
        let on_cloud = Arc::new(AtomicBool::new(false));
        let captured_seam: Arc<Mutex<Option<SeamInfo>>> = Arc::new(Mutex::new(None));

        let local_count_cb = local_count.clone();
        let on_cloud_cb = on_cloud.clone();
        let provider_cb = provider.clone();
        let mut on_token =
            move |_: PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                if !on_cloud_cb.load(Ordering::SeqCst) {
                    let delivered = local_count_cb.fetch_add(1, Ordering::SeqCst) + 1;
                    // Arm on the token before the target, then hold it so the
                    // target token's abort check takes a fresh sample.
                    if delivered + 1 == abort_at {
                        provider_cb.arm();
                        std::thread::sleep(SAMPLE_GAP);
                    }
                }
                Ok(())
            };

        let on_cloud_seam = on_cloud.clone();
        let captured_seam_for_cb = captured_seam.clone();
        let mut on_seam = move |info: SeamInfo| {
            on_cloud_seam.store(true, Ordering::SeqCst);
            *captured_seam_for_cb.lock().unwrap() = Some(info);
        };

        let result = model
            .run_streaming_with_fallback(
                &fallback_envelope(),
                &options,
                &cloud,
                &mut on_token,
                &mut on_seam,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "pressure on token {abort_at} must fall back to the cloud, got: {}",
                    error_chain(&err)
                )
            });

        let seam = captured_seam
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| panic!("token {abort_at}: the seam must fire"));
        assert_eq!(
            seam.local_tokens,
            abort_at - 1,
            "token {abort_at}: the abort must land on the token after the one that armed it"
        );
        assert_eq!(local_count.load(Ordering::SeqCst), abort_at - 1);
        assert_eq!(
            cloud.calls.lock().unwrap().len(),
            1,
            "token {abort_at}: exactly one cloud call"
        );
        assert_eq!(result.text(), Some("hello from cloud"));
    }
}

/// A cancel that lands before the first generated token must come back as a
/// cancel, not as a failed decode, and must not reach the cloud. Same native
/// return-value collision as the test above, reached through the cancel path.
///
/// Run manually (needs the same cached model as the tests above):
///
/// ```bash
/// cargo test -p xybrid-sdk --features dev-tools,llm-llamacpp \
///   --test cloud_fallback_e2e -- --ignored first_token
/// ```
#[test]
#[ignore]
fn user_cancel_on_the_first_token_is_reported_as_a_cancel() {
    let model = load_fallback_model();
    let token = CancellationToken::new();
    let cloud = RecordingCloud::new("hello from cloud");
    let options = RunOptions::new()
        .with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::UserCancelled)
                // Observing a resource signal is what makes the pre-run gate
                // read the provider that cancels the token.
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_cloud_fallback(true)
                .with_max_grace_tokens(0),
        )
        .with_cancellation_token(token.clone())
        .with_resource_provider(Arc::new(CancelOnFirstRead {
            token,
            reads: AtomicUsize::new(0),
        }))
        .with_generation_config(GenerationConfig::greedy().with_max_tokens(64));

    let local_count = Arc::new(AtomicU32::new(0));
    let seam_fired = Arc::new(AtomicBool::new(false));

    let local_count_cb = local_count.clone();
    let mut on_token =
        move |_: PartialToken| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            local_count_cb.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };
    let seam_fired_cb = seam_fired.clone();
    let mut on_seam = move |_: SeamInfo| seam_fired_cb.store(true, Ordering::SeqCst);

    let err = model
        .run_streaming_with_fallback(
            &fallback_envelope(),
            &options,
            &cloud,
            &mut on_token,
            &mut on_seam,
        )
        .expect_err("a cancelled run must fail");

    let chain = error_chain(&err);
    assert!(
        chain.contains("user_cancelled"),
        "the cancel must be named in the error: {chain}"
    );
    assert!(
        !chain.contains("error code"),
        "the cancel was reported as a native failure: {chain}"
    );
    assert_eq!(
        local_count.load(Ordering::SeqCst),
        0,
        "no token may be delivered once the run is cancelled"
    );
    assert!(
        !seam_fired.load(Ordering::SeqCst),
        "a cancel must not open the cloud seam"
    );
    assert!(
        cloud.calls.lock().unwrap().is_empty(),
        "a cancel must not reach the cloud"
    );
}
