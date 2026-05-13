//! Per-run controls for cancellation and cloud-fallback policy.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use xybrid_core::context::DeviceMetrics;
use xybrid_core::device::{
    MemoryPressure, ResourceMonitor, ResourceSnapshot, ResourceSnapshotProvider, ThermalState,
};
use xybrid_core::runtime_adapter::types::GenerationConfig;

const RESOURCE_ABORT_CHECK_INTERVAL: Duration = Duration::from_millis(500);

pub(crate) trait ResourceSnapshotReader: Send + Sync {
    fn current_snapshot(&self, max_age: Duration) -> ResourceSnapshot;
}

impl ResourceSnapshotReader for ResourceMonitor {
    fn current_snapshot(&self, max_age: Duration) -> ResourceSnapshot {
        ResourceMonitor::current_snapshot(self, max_age)
    }
}

/// Bridges any [`ResourceSnapshotProvider`] (the public xybrid-core trait) into
/// this module's internal `ResourceSnapshotReader`. Lets demos and integration
/// tests inject deterministic snapshots via `RunOptions::with_resource_provider`
/// without exposing the SDK's internal trait surface.
struct ProviderReader(Arc<dyn ResourceSnapshotProvider>);

impl ResourceSnapshotReader for ProviderReader {
    fn current_snapshot(&self, max_age: Duration) -> ResourceSnapshot {
        self.0.current_snapshot(max_age)
    }
}

/// A cooperative cancellation token shared between callers and in-flight runs.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Runtime signals that may abort a local run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbortSignal {
    UserCancelled,
    MemoryPressureWarn,
    MemoryPressureCritical,
    ThermalHot,
    ThermalCritical,
}

/// Policy for cooperative local-run abort and optional cloud restart.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbortPolicy {
    /// Signals that should stop the local run.
    #[serde(default)]
    pub stop_on: Vec<AbortSignal>,
    /// Whether the caller permits a cloud restart after local abort.
    #[serde(default)]
    pub fallback_to_cloud: bool,
    /// Tokens to allow after the first stop signal before aborting.
    #[serde(default)]
    pub max_grace_tokens: u32,
}

impl AbortPolicy {
    pub fn stop_on(mut self, signal: AbortSignal) -> Self {
        if !self.stop_on.contains(&signal) {
            self.stop_on.push(signal);
        }
        self
    }

    pub fn with_cloud_fallback(mut self, enabled: bool) -> Self {
        self.fallback_to_cloud = enabled;
        self
    }

    pub fn with_max_grace_tokens(mut self, tokens: u32) -> Self {
        self.max_grace_tokens = tokens;
        self
    }

    pub fn observes(&self, signal: AbortSignal) -> bool {
        self.stop_on.contains(&signal)
    }
}

/// Per-run options accepted by the Rust SDK. Binding-specific adapters can
/// map this to their native option bags without changing the execution core.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub generation_config: Option<GenerationConfig>,
    pub abort_policy: AbortPolicy,
    pub cancellation_token: Option<CancellationToken>,
    pub correlation_id: Option<String>,
    /// Optional [`ResourceSnapshotProvider`] override used by the per-token
    /// abort check. Defaults to the process-wide [`ResourceMonitor`] when
    /// absent. Set this from demos/tests (typically a provider from
    /// `xybrid_core::orchestrator::authority::test_seams`) to drive
    /// deterministic abort behavior.
    pub resource_provider: Option<Arc<dyn ResourceSnapshotProvider>>,
    /// Optional caller-supplied device metrics used by the cloud-fallback
    /// policy check. The `network_rtt` and `battery` legacy scalars only
    /// flow into the policy engine through this seam — `with_live_snapshot`
    /// (the only other source available to the SDK) does not touch them.
    /// Without this, RTT-based or battery-based deny rules in the policy
    /// engine evaluate against `DeviceMetrics::default()` (network_rtt=100,
    /// battery=100) and are unreachable for any real-world value.
    pub device_metrics: Option<DeviceMetrics>,
}

impl RunOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_generation_config(mut self, config: GenerationConfig) -> Self {
        self.generation_config = Some(config);
        self
    }

    pub fn with_abort_policy(mut self, policy: AbortPolicy) -> Self {
        self.abort_policy = policy;
        self
    }

    pub fn with_cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }

    /// Inject a custom [`ResourceSnapshotProvider`] used by the per-token
    /// abort check. Intended for demos and integration tests; production
    /// callers should leave this `None` so the global [`ResourceMonitor`]
    /// supplies real device telemetry.
    pub fn with_resource_provider(mut self, provider: Arc<dyn ResourceSnapshotProvider>) -> Self {
        self.resource_provider = Some(provider);
        self
    }

    /// Supply caller-owned [`DeviceMetrics`] for the cloud-fallback policy
    /// check. Use this when your application has its own platform bridge that
    /// populates `HardwareCapabilities` (battery, thermal) ahead of routing.
    /// When `None`, the SDK falls back to `DeviceMetrics::default()`
    /// overlaid with the live `ResourceSnapshot`.
    pub fn with_device_metrics(mut self, metrics: DeviceMetrics) -> Self {
        self.device_metrics = Some(metrics);
        self
    }
}

/// Reason a run was cooperatively aborted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortReason {
    UserCancelled,
    MemoryPressure(MemoryPressure),
    Thermal(ThermalState),
}

impl AbortReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserCancelled => "user_cancelled",
            Self::MemoryPressure(MemoryPressure::Warn) => "memory_pressure_warn",
            Self::MemoryPressure(MemoryPressure::Critical) => "memory_pressure_critical",
            Self::MemoryPressure(_) => "memory_pressure",
            Self::Thermal(ThermalState::Hot) => "thermal_hot",
            Self::Thermal(ThermalState::Critical) => "thermal_critical",
            Self::Thermal(_) => "thermal",
        }
    }

    pub(crate) fn to_core_abort_reason(&self) -> xybrid_core::abort::AbortReason {
        match self {
            Self::UserCancelled => xybrid_core::abort::AbortReason::UserCancelled,
            Self::MemoryPressure(_) => xybrid_core::abort::AbortReason::StressMemory,
            Self::Thermal(_) => xybrid_core::abort::AbortReason::StressThermal,
        }
    }

    pub(crate) fn into_streaming_error(
        self,
        fallback_to_cloud: bool,
    ) -> Box<dyn std::error::Error + Send + Sync> {
        if fallback_to_cloud {
            return match self {
                Self::UserCancelled => Box::new(Self::UserCancelled),
                reason => Box::new(xybrid_core::abort::CloudFallbackAbort::new(
                    reason.to_core_abort_reason(),
                )),
            };
        }
        Box::new(self)
    }
}

impl std::fmt::Display for AbortReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for AbortReason {}

pub(crate) struct AbortState {
    policy: AbortPolicy,
    token: Option<CancellationToken>,
    active_reason: Option<AbortReason>,
    grace_tokens_seen: u32,
    resource_reader: Arc<dyn ResourceSnapshotReader>,
    last_resource_check_at: Option<Instant>,
}

impl AbortState {
    pub(crate) fn new(options: &RunOptions) -> Self {
        let reader: Arc<dyn ResourceSnapshotReader> = match options.resource_provider.as_ref() {
            Some(provider) => Arc::new(ProviderReader(provider.clone())),
            None => ResourceMonitor::global(),
        };
        Self::with_resource_reader(options, reader)
    }

    pub(crate) fn with_resource_reader(
        options: &RunOptions,
        resource_reader: Arc<dyn ResourceSnapshotReader>,
    ) -> Self {
        Self {
            policy: options.abort_policy.clone(),
            token: options.cancellation_token.clone(),
            active_reason: None,
            grace_tokens_seen: 0,
            resource_reader,
            last_resource_check_at: None,
        }
    }

    pub(crate) fn check_before_token(&mut self) -> Result<(), AbortReason> {
        if let Some(reason) = self.detect_user_cancelled() {
            // User cancellation gets its own fresh grace window. Without
            // this reset, a cancel arriving mid-grace after a prior
            // resource event would inherit the grace tokens already
            // consumed by the resource event and lag by
            // `max_grace_tokens - K` emitted tokens — making the cancel
            // button feel unresponsive on policies that allow a graceful
            // drain. Resetting on the transition keeps the documented
            // grace semantics intact while honouring cancel as a
            // distinct, caller-driven signal.
            if !matches!(self.active_reason, Some(AbortReason::UserCancelled)) {
                self.grace_tokens_seen = 0;
            }
            self.active_reason = Some(reason);
        }

        if let Some(reason) = self.active_reason.clone() {
            if self.grace_tokens_seen >= self.policy.max_grace_tokens {
                return Err(reason);
            }
            self.grace_tokens_seen = self.grace_tokens_seen.saturating_add(1);
            return Ok(());
        }

        if let Some(reason) = self.detect_abort_reason() {
            self.active_reason = Some(reason.clone());
            if self.policy.max_grace_tokens == 0 {
                return Err(reason);
            }
            self.grace_tokens_seen = 1;
        }

        Ok(())
    }

    pub(crate) fn check_before_run(&mut self) -> Result<(), AbortReason> {
        if let Some(reason) = self.detect_abort_reason() {
            self.active_reason = Some(reason.clone());
            return Err(reason);
        }
        Ok(())
    }

    fn detect_user_cancelled(&self) -> Option<AbortReason> {
        (self
            .token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
            && self.policy.observes(AbortSignal::UserCancelled))
        .then_some(AbortReason::UserCancelled)
    }

    fn detect_abort_reason(&mut self) -> Option<AbortReason> {
        if let Some(reason) = self.detect_user_cancelled() {
            return Some(reason);
        }

        let needs_resource_snapshot = self.policy.observes(AbortSignal::MemoryPressureWarn)
            || self.policy.observes(AbortSignal::MemoryPressureCritical)
            || self.policy.observes(AbortSignal::ThermalHot)
            || self.policy.observes(AbortSignal::ThermalCritical);
        if !needs_resource_snapshot {
            return None;
        }

        let now = Instant::now();
        if self
            .last_resource_check_at
            .is_some_and(|last| now.duration_since(last) < RESOURCE_ABORT_CHECK_INTERVAL)
        {
            return None;
        }
        self.last_resource_check_at = Some(now);

        let snapshot = self
            .resource_reader
            .current_snapshot(RESOURCE_ABORT_CHECK_INTERVAL);
        abort_reason_from_snapshot(&self.policy, snapshot)
    }
}

fn abort_reason_from_snapshot(
    policy: &AbortPolicy,
    snapshot: ResourceSnapshot,
) -> Option<AbortReason> {
    if policy.observes(AbortSignal::MemoryPressureCritical)
        && snapshot.memory_pressure == MemoryPressure::Critical
    {
        return Some(AbortReason::MemoryPressure(MemoryPressure::Critical));
    }
    if policy.observes(AbortSignal::MemoryPressureWarn)
        && matches!(
            snapshot.memory_pressure,
            MemoryPressure::Warn | MemoryPressure::Critical
        )
    {
        return Some(AbortReason::MemoryPressure(snapshot.memory_pressure));
    }
    if policy.observes(AbortSignal::ThermalCritical)
        && snapshot.thermal_state == ThermalState::Critical
    {
        return Some(AbortReason::Thermal(ThermalState::Critical));
    }
    if policy.observes(AbortSignal::ThermalHot)
        && matches!(
            snapshot.thermal_state,
            ThermalState::Hot | ThermalState::Critical
        )
    {
        return Some(AbortReason::Thermal(snapshot.thermal_state));
    }
    None
}

/// Streaming-only gate around [`AbortState::check_before_token`]. Only
/// invokes the abort check + the [`AbortReason::into_streaming_error`]
/// conversion when the model actually streams tokens.
///
/// Non-streaming models execute the full local inference and then emit
/// a single synthetic token through the streaming callback. Without this
/// gate, an abort policy that trips while the batch was running would
/// fire at the synthetic-emit boundary, after local already succeeded —
/// silently discarding completed work and triggering an unnecessary
/// cloud retry of the original prompt (privacy + cost surprise). Gating
/// on `supports_streaming` makes the contract honest: cloud fallback
/// only protects mid-stream aborts, which can only happen on a real
/// stream.
pub(crate) fn check_abort_for_streaming(
    supports_streaming: bool,
    abort_state: &mut AbortState,
    fallback_to_cloud: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if !supports_streaming {
        return Ok(());
    }
    if let Err(reason) = abort_state.check_before_token() {
        return Err(reason.into_streaming_error(fallback_to_cloud));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingResourceReader {
        snapshot: ResourceSnapshot,
        reads: AtomicUsize,
    }

    impl CountingResourceReader {
        fn new(snapshot: ResourceSnapshot) -> Self {
            Self {
                snapshot,
                reads: AtomicUsize::new(0),
            }
        }

        fn reads(&self) -> usize {
            self.reads.load(Ordering::SeqCst)
        }
    }

    impl ResourceSnapshotReader for CountingResourceReader {
        fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.snapshot
        }
    }

    #[test]
    fn cancellation_token_is_cooperative() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn abort_state_respects_grace_tokens() {
        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(
                AbortPolicy::default()
                    .stop_on(AbortSignal::UserCancelled)
                    .with_max_grace_tokens(1),
            );
        let mut state = AbortState::new(&options);

        token.cancel();

        assert!(state.check_before_token().is_ok());
        assert_eq!(
            state.check_before_token().expect_err("second token aborts"),
            AbortReason::UserCancelled
        );
    }

    #[test]
    fn abort_state_aborts_before_first_token_without_grace() {
        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(AbortPolicy::default().stop_on(AbortSignal::UserCancelled));
        let mut state = AbortState::new(&options);

        token.cancel();

        assert_eq!(
            state
                .check_before_token()
                .expect_err("first token should abort"),
            AbortReason::UserCancelled
        );
    }

    #[test]
    fn abort_state_checks_cancellation_before_batch_run() {
        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(AbortPolicy::default().stop_on(AbortSignal::UserCancelled));
        let mut state = AbortState::new(&options);

        token.cancel();

        assert_eq!(
            state.check_before_run().expect_err("run should abort"),
            AbortReason::UserCancelled
        );
    }

    #[test]
    fn snapshot_abort_prefers_critical_memory() {
        let policy = AbortPolicy::default()
            .stop_on(AbortSignal::MemoryPressureWarn)
            .stop_on(AbortSignal::MemoryPressureCritical);
        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.memory_pressure = MemoryPressure::Critical;

        assert_eq!(
            abort_reason_from_snapshot(&policy, snapshot),
            Some(AbortReason::MemoryPressure(MemoryPressure::Critical))
        );
    }

    #[test]
    fn sdk_abort_reason_translates_to_core_reason() {
        assert_eq!(
            AbortReason::UserCancelled.to_core_abort_reason(),
            xybrid_core::abort::AbortReason::UserCancelled
        );
        assert_eq!(
            AbortReason::MemoryPressure(MemoryPressure::Warn).to_core_abort_reason(),
            xybrid_core::abort::AbortReason::StressMemory
        );
        assert_eq!(
            AbortReason::Thermal(ThermalState::Hot).to_core_abort_reason(),
            xybrid_core::abort::AbortReason::StressThermal
        );
    }

    #[test]
    fn cloud_fallback_streaming_error_preserves_core_abort_reason() {
        let error =
            AbortReason::MemoryPressure(MemoryPressure::Critical).into_streaming_error(true);

        assert_eq!(
            xybrid_core::abort::cloud_fallback_reason_from_error(error.as_ref()),
            Some(xybrid_core::abort::AbortReason::StressMemory)
        );
    }

    #[test]
    fn user_cancelled_streaming_error_never_uses_cloud_fallback_marker() {
        let error = AbortReason::UserCancelled.into_streaming_error(true);

        assert_eq!(
            xybrid_core::abort::cloud_fallback_reason_from_error(error.as_ref()),
            None
        );
        assert!(error.downcast_ref::<AbortReason>().is_some());
    }

    #[test]
    fn non_fallback_streaming_error_does_not_use_core_abort_marker() {
        let error = AbortReason::Thermal(ThermalState::Hot).into_streaming_error(false);

        assert_eq!(
            xybrid_core::abort::cloud_fallback_reason_from_error(error.as_ref()),
            None
        );
    }

    #[test]
    fn resource_abort_checks_are_throttled_per_state() {
        let options = RunOptions::new()
            .with_abort_policy(AbortPolicy::default().stop_on(AbortSignal::MemoryPressureCritical));
        let reader = Arc::new(CountingResourceReader::new(ResourceSnapshot::default()));
        let mut state = AbortState::with_resource_reader(&options, reader.clone());

        for _ in 0..10 {
            state.check_before_token().unwrap();
        }

        assert_eq!(reader.reads(), 1);
    }

    #[test]
    fn user_cancellation_is_checked_even_when_resource_checks_are_throttled() {
        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(
                AbortPolicy::default()
                    .stop_on(AbortSignal::UserCancelled)
                    .stop_on(AbortSignal::MemoryPressureCritical),
            );
        let reader = Arc::new(CountingResourceReader::new(ResourceSnapshot::default()));
        let mut state = AbortState::with_resource_reader(&options, reader);

        state.check_before_token().unwrap();
        token.cancel();

        assert_eq!(
            state
                .check_before_token()
                .expect_err("second token should observe cancellation"),
            AbortReason::UserCancelled
        );
    }

    #[test]
    fn user_cancellation_overrides_active_resource_abort_grace() {
        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(
                AbortPolicy::default()
                    .stop_on(AbortSignal::UserCancelled)
                    .stop_on(AbortSignal::MemoryPressureCritical)
                    .with_max_grace_tokens(1),
            );
        let snapshot = ResourceSnapshot {
            memory_pressure: MemoryPressure::Critical,
            ..Default::default()
        };
        let reader = Arc::new(CountingResourceReader::new(snapshot));
        let mut state = AbortState::with_resource_reader(&options, reader);

        // Token 1: resource event, grace burned (grace=1, max=1).
        state.check_before_token().unwrap();
        token.cancel();

        // Token 2: cancel transitions reason → UserCancelled, grace resets
        // to 0 so cancel gets its own fresh grace window. With grace=0 and
        // max=1, this token still emits.
        state.check_before_token().unwrap();
        // Token 3: cancel still active, grace exhausted → terminal.
        assert_eq!(
            state.check_before_token().expect_err(
                "cancel must terminate within its own grace window, not inherit resource burn"
            ),
            AbortReason::UserCancelled
        );
    }

    /// Regression for the grace-token-leak bug: with `max_grace_tokens > 1`,
    /// a resource-pressure event that consumes part of the grace budget must
    /// not delay user cancellation by `max_grace_tokens - K` tokens. The
    /// cancel transition resets the grace counter so the cancel pathway
    /// gets its own fresh grace window — the lag is bounded by the
    /// configured grace, not by what the resource event happened to leave
    /// behind.
    #[test]
    fn user_cancellation_resets_grace_on_transition_from_resource_abort() {
        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(
                AbortPolicy::default()
                    .stop_on(AbortSignal::UserCancelled)
                    .stop_on(AbortSignal::MemoryPressureCritical)
                    .with_max_grace_tokens(5),
            );
        let snapshot = ResourceSnapshot {
            memory_pressure: MemoryPressure::Critical,
            ..Default::default()
        };
        let reader = Arc::new(CountingResourceReader::new(snapshot));
        let mut state = AbortState::with_resource_reader(&options, reader);

        // Resource event consumes 3 of 5 grace tokens.
        state.check_before_token().unwrap();
        state.check_before_token().unwrap();
        state.check_before_token().unwrap();

        token.cancel();

        // Cancel resets the grace counter, so it must allow exactly
        // max_grace_tokens (5) more emissions before firing — NOT
        // max_grace_tokens - 3 = 2 (which is what the leak produced).
        for _ in 0..5 {
            state.check_before_token().expect("cancel grace window");
        }
        assert_eq!(
            state
                .check_before_token()
                .expect_err("cancel grace must be exactly max_grace_tokens, not inherited"),
            AbortReason::UserCancelled
        );
    }

    #[test]
    fn abort_policy_serializes_cloud_fallback() {
        let policy = AbortPolicy::default()
            .stop_on(AbortSignal::ThermalHot)
            .with_cloud_fallback(true)
            .with_max_grace_tokens(2);

        let value = serde_json::to_value(&policy).unwrap();

        assert_eq!(value["fallback_to_cloud"], true);
        assert_eq!(value["max_grace_tokens"], 2);
        assert_eq!(value["stop_on"][0], "thermal_hot");
    }

    #[test]
    fn run_options_keeps_string_correlation_id() {
        let options = RunOptions::new().with_correlation_id("run-123");

        assert_eq!(options.correlation_id.as_deref(), Some("run-123"));
    }

    /// Implements the **public** [`ResourceSnapshotProvider`] trait so demos
    /// and integration tests can drive `RunOptions::with_resource_provider`
    /// without touching the SDK's internal `ResourceSnapshotReader`.
    #[derive(Debug)]
    struct FixedProviderForTest(ResourceSnapshot);

    impl ResourceSnapshotProvider for FixedProviderForTest {
        fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
            self.0
        }
    }

    #[test]
    fn abort_state_uses_resource_provider_override_from_run_options() {
        let snapshot = ResourceSnapshot {
            memory_pressure: MemoryPressure::Critical,
            ..Default::default()
        };
        let provider: Arc<dyn ResourceSnapshotProvider> = Arc::new(FixedProviderForTest(snapshot));

        let options = RunOptions::new()
            .with_abort_policy(AbortPolicy::default().stop_on(AbortSignal::MemoryPressureCritical))
            .with_resource_provider(provider);

        let mut state = AbortState::new(&options);

        let err = state
            .check_before_token()
            .expect_err("provider override should drive abort");
        assert_eq!(err, AbortReason::MemoryPressure(MemoryPressure::Critical));
    }

    #[test]
    fn abort_state_falls_back_to_global_monitor_when_no_provider_override() {
        // Without a provider override, AbortState consults the process-wide
        // ResourceMonitor. Real device state is platform-dependent, but the
        // policy here only fires on UserCancelled, so the resource path is
        // unused and the run never aborts. This guards the default branch.
        let options = RunOptions::new()
            .with_abort_policy(AbortPolicy::default().stop_on(AbortSignal::UserCancelled));

        let mut state = AbortState::new(&options);

        state
            .check_before_token()
            .expect("no abort when only UserCancelled is observed and no token is cancelled");
    }

    /// Streaming-only gate must short-circuit when supports_streaming=false,
    /// even if the AbortState would otherwise trip. This is the regression
    /// test for the codex P2 finding: a non-streaming model whose batch
    /// inference happens to overlap with critical memory pressure must not
    /// silently retry the original prompt on cloud after local already
    /// produced output.
    #[test]
    fn check_abort_for_streaming_skips_check_when_model_does_not_stream() {
        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.memory_pressure = MemoryPressure::Critical;
        let reader = Arc::new(CountingResourceReader::new(snapshot));

        let options = RunOptions::new().with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_max_grace_tokens(0),
        );
        let mut state = AbortState::with_resource_reader(&options, reader.clone());

        let result =
            check_abort_for_streaming(/* supports_streaming */ false, &mut state, true);

        assert!(
            result.is_ok(),
            "non-streaming model must not propagate abort even under critical memory pressure"
        );
        assert_eq!(
            reader.reads(),
            0,
            "non-streaming gate must not consult the resource reader at all"
        );
    }

    /// Symmetric coverage: when the model DOES stream, the helper must
    /// still surface the abort as before — the gate is a non-streaming
    /// short-circuit, not a behavior change for streaming.
    #[test]
    fn check_abort_for_streaming_propagates_abort_when_model_streams() {
        let mut snapshot = ResourceSnapshot::unknown();
        snapshot.memory_pressure = MemoryPressure::Critical;
        let reader = Arc::new(CountingResourceReader::new(snapshot));

        let options = RunOptions::new().with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_max_grace_tokens(0),
        );
        let mut state = AbortState::with_resource_reader(&options, reader);

        let err = check_abort_for_streaming(/* supports_streaming */ true, &mut state, true)
            .expect_err("streaming model under critical memory pressure must abort");

        assert_eq!(
            xybrid_core::abort::cloud_fallback_reason_from_error(err.as_ref()),
            Some(xybrid_core::abort::AbortReason::StressMemory),
            "streaming abort must carry the typed CloudFallbackAbort marker so dispatch_after_local can retry"
        );
    }

    /// User cancellation must remain terminal even when the model streams,
    /// per the prior codex HIGH finding. The gate must not regress this.
    #[test]
    fn check_abort_for_streaming_keeps_user_cancellation_terminal() {
        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_abort_policy(AbortPolicy::default().stop_on(AbortSignal::UserCancelled));
        let mut state = AbortState::new(&options);
        token.cancel();

        let err = check_abort_for_streaming(/* supports_streaming */ true, &mut state, true)
            .expect_err("user cancellation must surface as a streaming error");

        assert_eq!(
            xybrid_core::abort::cloud_fallback_reason_from_error(err.as_ref()),
            None,
            "user cancellation must NOT carry the CloudFallbackAbort marker (terminal, no retry)"
        );
    }
}
