//! Registry of live loaded models and the automatic-release (eviction) engine.
//!
//! ## What this module is for
//!
//! Every model loaded through [`crate::ModelLoader`] registers a [`Weak`]
//! reference here, together with the timestamp of its most recent run. That
//! gives the SDK two things it could not do before: a list of what is loaded
//! right now, and a least-recently-used (LRU) order over it. On memory
//! pressure — or when the host app explicitly asks — the engine picks LRU
//! victims and *evicts* them: the model's executor (and with it the LLM
//! adapter, TTS session and vision encoders) is dropped, while the metadata
//! and the on-disk model directory are kept. The next run on an evicted model
//! reloads it from disk transparently; see [`crate::model::LoadState`].
//!
//! ## Division of labour
//!
//! The host decides *when* (it forwards the OS memory warning, or calls
//! [`release_memory`] on its own schedule); the SDK decides *what* (LRU victim
//! selection). There is no background thread in this tier — nothing here polls,
//! sleeps, or wakes the device.
//!
//! ## Side effects
//!
//! Evicting frees memory and makes the next run on that model pay a load. It
//! never touches disk, never cancels an in-flight run (a busy model is skipped
//! via `try_write`), and never turns a successful run into an error.
//!
//! ## Lock ordering
//!
//! The registry mutex is **never** held while a model handle's `RwLock` is
//! taken. Every sweep collects and upgrades the `Weak` handles under the
//! registry lock, releases it, and only then attempts `try_write` on each
//! handle. Do not collapse those two phases.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, TryLockError, Weak};
use std::time::Duration;

use xybrid_core::device::{MemoryPressure, ResourceMonitor, ResourceSnapshotProvider};

use crate::model::{LoadState, ModelHandle};

/// Freshness bound for the first pressure read of a sweep. Short enough that
/// a load which follows another load by more than a moment re-measures, long
/// enough that back-to-back loads share one `sysinfo` refresh.
const PRESSURE_MAX_AGE: Duration = Duration::from_millis(250);

/// Policy controlling when a model may be released automatically.
///
/// Default is **off**: nothing is ever evicted unless the app opts in with
/// [`crate::ModelLoader::with_auto_release`] / [`crate::set_auto_release`], or
/// calls [`release_memory`] explicitly.
///
/// # Examples
/// ```
/// use xybrid_sdk::AutoReleasePolicy;
/// assert!(!AutoReleasePolicy::default().on_pressure);
/// assert!(AutoReleasePolicy::from(true).on_pressure);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AutoReleasePolicy {
    /// Evict least-recently-used models when a new load starts while the
    /// device reports memory pressure.
    pub on_pressure: bool,
}

impl AutoReleasePolicy {
    /// Policy that evicts LRU models under memory pressure.
    pub const fn enabled() -> Self {
        Self { on_pressure: true }
    }

    /// Policy that never evicts automatically. Same as [`Default`].
    pub const fn disabled() -> Self {
        Self { on_pressure: false }
    }
}

impl From<bool> for AutoReleasePolicy {
    fn from(on_pressure: bool) -> Self {
        Self { on_pressure }
    }
}

/// One live model known to the registry.
struct RegisteredModel {
    /// Weak so a registry entry never keeps a dropped model alive. Entries
    /// whose model is gone are purged on the next sweep — there is no
    /// unregister bookkeeping on `Drop`.
    handle: Weak<RwLock<ModelHandle>>,
    model_id: String,
    /// Shared with the `XybridModel` (and all of its clones), stamped on every
    /// run/stream entry point. Milliseconds since the Unix epoch.
    last_accessed: Arc<AtomicU64>,
    /// Advisory size, used only to break ties between models last used in the
    /// same millisecond. `None` when the bundle does not declare one.
    approx_size_mb: Option<u32>,
}

/// Process-wide registry of loaded models.
///
/// `M-AVOID-STATICS` warns that statics silently duplicate across crate-version
/// boundaries. That risk does not apply here: this static lives in exactly one
/// crate, holds no correctness-relevant state (a duplicated registry would
/// evict less, never wrongly), and is never shared across a DLL boundary — the
/// FFI crates call into this one.
static REGISTRY: Mutex<Vec<RegisteredModel>> = Mutex::new(Vec::new());

fn lock_registry() -> MutexGuard<'static, Vec<RegisteredModel>> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Milliseconds since the Unix epoch, saturating to 0 if the clock is before it.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Record a freshly loaded model. Called once per successful load.
pub(crate) fn register(
    model_id: &str,
    handle: &Arc<RwLock<ModelHandle>>,
    last_accessed: &Arc<AtomicU64>,
    approx_size_mb: Option<u32>,
) {
    let mut registry = lock_registry();
    registry.retain(|entry| entry.handle.strong_count() > 0);
    registry.push(RegisteredModel {
        handle: Arc::downgrade(handle),
        model_id: model_id.to_string(),
        last_accessed: Arc::clone(last_accessed),
        approx_size_mb,
    });
}

/// Stamp a model as used right now. Cheap enough to call on every run.
pub(crate) fn touch(last_accessed: &AtomicU64) {
    last_accessed.store(now_ms(), Ordering::Relaxed);
}

/// A registry entry resolved to a live handle, taken while the registry lock
/// was held and used after it was released.
pub(crate) struct Candidate {
    handle: Arc<RwLock<ModelHandle>>,
    model_id: String,
    last_accessed_ms: u64,
    approx_size_mb: Option<u32>,
}

impl Candidate {
    /// Build a candidate directly, for tests that need handles in known states
    /// rather than whatever the process-global registry happens to hold.
    #[cfg(test)]
    pub(crate) fn for_test(
        handle: &Arc<RwLock<ModelHandle>>,
        model_id: &str,
        last_accessed_ms: u64,
    ) -> Self {
        Self {
            handle: Arc::clone(handle),
            model_id: model_id.to_string(),
            last_accessed_ms,
            approx_size_mb: None,
        }
    }
}

/// Sort key placing least-recently-used first, breaking ties toward the larger
/// model (frees more memory for the same disruption).
///
/// Pure, so the ordering can be tested without fabricating live model handles.
fn lru_order_key(
    last_accessed_ms: u64,
    approx_size_mb: Option<u32>,
) -> (u64, std::cmp::Reverse<u32>) {
    (
        last_accessed_ms,
        std::cmp::Reverse(approx_size_mb.unwrap_or(0)),
    )
}

/// Live models, least-recently-used first. Ties break toward the larger model
/// (frees more for the same disruption). Purges entries whose model was dropped.
fn ranked_candidates() -> Vec<Candidate> {
    let mut candidates = {
        // Registry lock is confined to this block: no handle lock is taken
        // while it is held (see the module's lock-ordering note).
        let mut registry = lock_registry();
        registry.retain(|entry| entry.handle.strong_count() > 0);
        registry
            .iter()
            .filter_map(|entry| {
                entry.handle.upgrade().map(|handle| Candidate {
                    handle,
                    model_id: entry.model_id.clone(),
                    last_accessed_ms: entry.last_accessed.load(Ordering::Relaxed),
                    approx_size_mb: entry.approx_size_mb,
                })
            })
            .collect::<Vec<_>>()
    };

    candidates.sort_by_key(|c| lru_order_key(c.last_accessed_ms, c.approx_size_mb));
    candidates
}

/// How many models the policy is willing to evict for this pressure reading.
///
/// Pure function over the two inputs, so the thresholds are unit-testable
/// without a device that can actually be pushed into `Critical`:
///
/// - `Critical` — every loaded model is fair game; reload is transparent.
/// - `Warn` — keep the most-recently-used one, evict the rest, and only when
///   more than one model is loaded (evicting the single loaded model to make
///   room for a load that is about to reload it is churn, not relief).
/// - `Normal` / `Unknown` — evict nothing.
pub(crate) fn eviction_budget(pressure: MemoryPressure, loaded_count: usize) -> usize {
    match pressure {
        MemoryPressure::Critical => loaded_count,
        MemoryPressure::Warn if loaded_count > 1 => loaded_count - 1,
        _ => 0,
    }
}

/// Whether a reading still calls for eviction.
fn pressure_calls_for_eviction(pressure: MemoryPressure) -> bool {
    matches!(pressure, MemoryPressure::Warn | MemoryPressure::Critical)
}

/// Whether this handle currently holds weights, and so is worth counting
/// toward an eviction budget.
///
/// Uses `try_read` so a sweep never blocks on a busy model. A handle whose
/// lock is already held for writing is mid-run, which means it is resident —
/// count it, and let `try_evict_handle` skip it a moment later.
pub(crate) fn holds_weights(handle: &RwLock<ModelHandle>) -> bool {
    match handle.try_read() {
        Ok(guard) => guard.state == LoadState::Loaded,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner().state == LoadState::Loaded,
        Err(TryLockError::WouldBlock) => true,
    }
}

fn try_evict(candidate: &Candidate) -> bool {
    try_evict_handle(&candidate.handle, &candidate.model_id)
}

/// Evict one model if it is idle and currently loaded.
///
/// Returns `false` without waiting when the model is busy. There are two ways
/// to be busy, and the write lock only catches one of them:
///
/// - An in-flight `run` holds the handle's write lock for its whole duration,
///   so `try_write` failing *is* the in-flight-run check.
/// - A live streaming session holds no lock at all — it clones the executor
///   `Arc` and lets go of the handle — so it is invisible to `try_write`.
///   Evicting under one would swap the handle's executor while the session
///   kept the old, still-loaded one alive: no memory actually freed, and a
///   `release_memory()` count that overstates what it did.
pub(crate) fn try_evict_handle(handle: &RwLock<ModelHandle>, model_id: &str) -> bool {
    let mut guard = match handle.try_write() {
        Ok(guard) => guard,
        // A poisoned lock is still ours to take; existing run paths recover
        // the same way rather than failing forever.
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => return false,
    };

    if guard.state != LoadState::Loaded {
        return false;
    }
    if guard.executor_is_shared() {
        log::debug!(
            target: "xybrid_sdk",
            "Skipping auto-release of {model_id}: a streaming session holds its executor"
        );
        return false;
    }
    guard.evict();
    log::info!(
        target: "xybrid_sdk",
        "Auto-released model {model_id} (will reload on next use)"
    );
    true
}

/// Release every idle loaded model, regardless of policy or device pressure.
///
/// This is the host hint: wire it to `didReceiveMemoryWarning` (iOS),
/// `onTrimMemory` (Android), or your own desktop logic. Models currently
/// running are skipped. Released models reload from disk on their next run —
/// callers do not have to reload anything by hand.
///
/// Returns the number of models released.
///
/// # Examples
/// ```
/// // Nothing loaded in this process: nothing to release.
/// assert_eq!(xybrid_sdk::release_memory(), 0);
/// ```
pub fn release_memory() -> usize {
    ranked_candidates()
        .iter()
        .filter(|candidate| try_evict(candidate))
        .count()
}

/// Evict LRU models if `provider` reports memory pressure. Called before a new
/// load when auto-release is enabled.
///
/// Re-reads pressure after each eviction so a sweep stops as soon as the device
/// recovers instead of clearing the whole registry.
pub(crate) fn evict_for_pressure(provider: &dyn ResourceSnapshotProvider) -> usize {
    sweep(ranked_candidates(), provider)
}

/// The sweep itself, over an already-ranked candidate list.
///
/// Split from [`evict_for_pressure`] so the budget accounting can be tested
/// against handles in known states, without going through the process-global
/// registry (which other tests share).
pub(crate) fn sweep(ranked: Vec<Candidate>, provider: &dyn ResourceSnapshotProvider) -> usize {
    // Only models that actually hold weights may count toward the budget or be
    // spent from it. The registry also holds evicted models, explicitly
    // unloaded ones, and speculative placeholders that have never loaded —
    // none of which can be released. Counting them would break the `Warn` rule
    // in both directions: a budget of one spent on an unloadable entry frees
    // nothing, and two entries of which only one is loaded would look like
    // "more than one loaded" and evict the sole loaded model.
    let candidates: Vec<Candidate> = ranked
        .into_iter()
        .filter(|candidate| holds_weights(&candidate.handle))
        .collect();
    let pressure = provider.current_snapshot(PRESSURE_MAX_AGE).memory_pressure;
    let budget = eviction_budget(pressure, candidates.len());
    if budget == 0 {
        return 0;
    }

    let mut evicted = 0;
    for candidate in candidates.iter().take(budget) {
        if try_evict(candidate) {
            evicted += 1;
        }
        let now = provider.current_snapshot(Duration::ZERO).memory_pressure;
        if !pressure_calls_for_eviction(now) {
            break;
        }
    }
    evicted
}

/// Evict LRU models under pressure using the process-wide [`ResourceMonitor`].
pub(crate) fn evict_for_pressure_global() -> usize {
    evict_for_pressure(ResourceMonitor::global().as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_core::device::ResourceSnapshot;

    /// Snapshot provider that replays a scripted sequence of pressure
    /// readings, repeating the last one once the script runs out.
    #[derive(Debug)]
    struct ScriptedPressure {
        readings: Vec<MemoryPressure>,
        calls: AtomicU64,
    }

    impl ScriptedPressure {
        fn new(readings: Vec<MemoryPressure>) -> Self {
            Self {
                readings,
                calls: AtomicU64::new(0),
            }
        }

        fn constant(pressure: MemoryPressure) -> Self {
            Self::new(vec![pressure])
        }
    }

    impl ResourceSnapshotProvider for ScriptedPressure {
        fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
            let index = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
            let pressure = *self
                .readings
                .get(index)
                .or_else(|| self.readings.last())
                .unwrap_or(&MemoryPressure::Unknown);
            ResourceSnapshot {
                memory_pressure: pressure,
                ..ResourceSnapshot::unknown()
            }
        }
    }

    #[test]
    fn critical_pressure_makes_every_model_evictable() {
        assert_eq!(eviction_budget(MemoryPressure::Critical, 3), 3);
        assert_eq!(eviction_budget(MemoryPressure::Critical, 1), 1);
    }

    #[test]
    fn warn_pressure_keeps_the_most_recently_used_model() {
        assert_eq!(eviction_budget(MemoryPressure::Warn, 3), 2);
    }

    #[test]
    fn warn_pressure_never_evicts_the_only_loaded_model() {
        assert_eq!(eviction_budget(MemoryPressure::Warn, 1), 0);
    }

    #[test]
    fn calm_pressure_evicts_nothing() {
        for pressure in [MemoryPressure::Normal, MemoryPressure::Unknown] {
            assert_eq!(eviction_budget(pressure, 4), 0, "{pressure:?}");
        }
        assert_eq!(eviction_budget(MemoryPressure::Critical, 0), 0);
    }

    #[test]
    fn scripted_provider_drives_the_pressure_gate() {
        // No candidates registered in this unit-test process for these ids,
        // so the sweep is a no-op — the point is that a calm reading returns
        // early without consulting the registry order.
        let calm = ScriptedPressure::constant(MemoryPressure::Normal);
        assert_eq!(evict_for_pressure(&calm), 0);
    }

    #[test]
    fn pressure_gate_recognizes_warn_and_critical_only() {
        assert!(pressure_calls_for_eviction(MemoryPressure::Warn));
        assert!(pressure_calls_for_eviction(MemoryPressure::Critical));
        assert!(!pressure_calls_for_eviction(MemoryPressure::Normal));
        assert!(!pressure_calls_for_eviction(MemoryPressure::Unknown));
    }

    #[test]
    fn lru_order_puts_the_stalest_model_first() {
        let mut order = vec![
            lru_order_key(300, None),
            lru_order_key(100, None),
            lru_order_key(200, None),
        ];
        order.sort();
        assert_eq!(
            order,
            vec![
                lru_order_key(100, None),
                lru_order_key(200, None),
                lru_order_key(300, None),
            ]
        );
    }

    #[test]
    fn equally_stale_models_break_the_tie_toward_the_larger_one() {
        let big = lru_order_key(100, Some(4096));
        let small = lru_order_key(100, Some(256));
        let unknown = lru_order_key(100, None);
        assert!(big < small, "larger model should be evicted first");
        assert!(small < unknown, "a known size outranks an undeclared one");
    }

    #[test]
    fn policy_defaults_to_off() {
        assert!(!AutoReleasePolicy::default().on_pressure);
        assert!(!AutoReleasePolicy::disabled().on_pressure);
        assert!(AutoReleasePolicy::enabled().on_pressure);
        assert_eq!(AutoReleasePolicy::from(true), AutoReleasePolicy::enabled());
    }
}
