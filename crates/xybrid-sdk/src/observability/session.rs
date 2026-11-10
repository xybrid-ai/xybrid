//! Application-owned sessions and context captured before execution hops.

use super::*;

struct SessionInner {
    observer: Observer,
    provider_origins: Vec<String>,
    pipeline: Option<Arc<xybrid_observability::otlp::ObservabilityPipeline>>,
    live_trace_interval: Duration,
    live_traced: Mutex<LiveSessions>,
    live_clock: LiveClock,
    retired: std::sync::atomic::AtomicBool,
}

/// Live frame sessions are forgotten this long after their last frame.
const LIVE_SESSION_TTL: Duration = Duration::from_secs(300);

/// The live-frame sampler remembers at most this many frame sessions; a new
/// one beyond it evicts the least recently used.
const MAX_LIVE_SESSIONS: usize = 1024;

/// The frame sessions the live-frame sampler remembers: for each, when its
/// current interval opened (its last traced frame), least recently used
/// first.
///
/// Bounded by [`MAX_LIVE_SESSIONS`] and [`LIVE_SESSION_TTL`], so a caller
/// that uses a fresh id per frame cannot grow it. A lookup is a hash probe;
/// keeping the use order costs `O(log n)` with `n` at most the cap, and
/// expiry and eviction remove entries from the least recently used end, so
/// no frame scans the whole set.
#[derive(Default)]
struct LiveSessions {
    by_id: HashMap<Arc<str>, LiveSession>,
    /// Session ids by their last use, least recent first.
    by_use: std::collections::BTreeMap<u64, Arc<str>>,
    /// Use stamp of the next frame (strictly increasing).
    next_use: u64,
}

struct LiveSession {
    /// When the session's last traced frame opened its interval.
    traced: Instant,
    /// When the session last had a frame.
    used: Instant,
    /// Its key in [`LiveSessions::by_use`].
    use_stamp: u64,
}

impl LiveSessions {
    /// Whether a frame of `id` at `now` opens a traced interval, recording
    /// the frame as the session's latest use.
    fn trace_mode(&mut self, id: &str, now: Instant, interval: Duration) -> TraceMode {
        let stamp = self.next_use;
        self.next_use += 1;
        let mode = match self.by_id.get_key_value(id) {
            Some((key, session)) => {
                let key = key.clone();
                let previous = session.use_stamp;
                let opens = now.saturating_duration_since(session.traced) >= interval;
                self.by_use.remove(&previous);
                self.by_use.insert(stamp, key);
                let session = self.by_id.get_mut(id).expect("the session was just found");
                session.used = now;
                session.use_stamp = stamp;
                if opens {
                    session.traced = now;
                    TraceMode::Default
                } else {
                    TraceMode::Suppress
                }
            }
            None => {
                if self.by_id.len() >= MAX_LIVE_SESSIONS {
                    self.evict_least_recently_used();
                }
                let key: Arc<str> = Arc::from(id);
                self.by_use.insert(stamp, key.clone());
                self.by_id.insert(
                    key,
                    LiveSession {
                        traced: now,
                        used: now,
                        use_stamp: stamp,
                    },
                );
                TraceMode::Default
            }
        };
        // The session of this frame was just used, so it is never expired here.
        self.expire(now);
        mode
    }

    /// Forgets the sessions with no frame for longer than the TTL.
    fn expire(&mut self, now: Instant) {
        while let Some((_, id)) = self.by_use.first_key_value() {
            let used = self.by_id.get(id).map_or(now, |session| session.used);
            if now.saturating_duration_since(used) <= LIVE_SESSION_TTL {
                break;
            }
            self.evict_least_recently_used();
        }
    }

    fn evict_least_recently_used(&mut self) {
        if let Some((_, id)) = self.by_use.pop_first() {
            self.by_id.remove(&id);
        }
    }

    fn len(&self) -> usize {
        self.by_id.len()
    }
}

/// Monotonic time source of the live-frame sampler: the system clock, or in
/// tests a clock the test moves, so which frame opens an interval never
/// depends on how fast the test runs.
enum LiveClock {
    System,
    #[cfg(test)]
    Manual(Arc<Mutex<Instant>>),
}

impl LiveClock {
    fn now(&self) -> Instant {
        match self {
            Self::System => Instant::now(),
            #[cfg(test)]
            Self::Manual(now) => *now
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        }
    }
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        if self.retired.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(pipeline) = self.pipeline.take() {
                // Last operation has released its session. Drain away from
                // inference callbacks and foreign finalizer threads.
                let _ = std::thread::Builder::new()
                    .name("xybrid-observability-retire".into())
                    .spawn(move || {
                        let _ = pipeline.shutdown(Duration::from_secs(1));
                    });
            }
        }
    }
}

/// An observer and, optionally, the export pipeline owned by the SDK.
///
/// Clones share the same session. Borrowed observers leave provider lifecycle
/// to the application; their stats are unavailable and flush/shutdown are no-ops.
/// Operations retain this session even if the process default changes.
///
/// An operation that selects this session samples device resources with its
/// configured [`ObservabilityConfig::resource_telemetry`] mode, whether or not
/// it is the default. Operations that select no session, or a borrowed
/// observer, use the process-wide
/// [`resource_telemetry_mode`](crate::telemetry::resource_telemetry_mode).
/// Either way the mode is fixed when the operation starts.
#[derive(Clone)]
pub struct ObservabilitySession {
    inner: Arc<SessionInner>,
    /// Resource sampling for operations using this handle. `None` (a borrowed
    /// observer has no mode of its own) resolves to the process-wide mode
    /// once, when an operation starts ([`Self::for_operation`]).
    resource_telemetry: Option<ResourceTelemetryMode>,
    /// The session's observer with its content capture capped, when an
    /// operation chose this session over the one it would otherwise use
    /// (see [`resolve_session`]); the session's own observer otherwise.
    capped: Option<Observer>,
    /// Whether this handle was resolved for an operation (never one an
    /// application holds): its content cap is already fixed.
    resolved: bool,
}

impl ObservabilitySession {
    /// Validates configuration and creates a session without installing it.
    ///
    /// # Errors
    /// [`SdkError::ConfigError`] for invalid exporter or device configuration.
    pub fn new(config: ObservabilityConfig) -> SdkResult<Self> {
        let (observer, pipeline) =
            match config.exporter {
                Exporter::Observer(observer) => {
                    if config.device_id.is_some() {
                        return Err(SdkError::ConfigError(
                            "set device.id on the application-owned provider's Resource".into(),
                        ));
                    }
                    (observer, None)
                }
                Exporter::Otlp(mut otlp) => {
                    otlp.resource
                        .attributes
                        .extend(sdk_resource_attributes(config.device_attributes));
                    if let Some(id) = config.device_id {
                        let id = validated_device_id(&id)?;
                        if otlp.resource.attributes.iter().any(|kv| {
                            kv.key.as_str() == "device.id" && kv.value != id.clone().into()
                        }) {
                            return Err(SdkError::ConfigError(
                                "conflicting device.id resource attributes".into(),
                            ));
                        }
                        otlp.resource
                            .attributes
                            .retain(|kv| kv.key.as_str() != "device.id");
                        otlp.resource
                            .attributes
                            .push(KeyValue::new("device.id", id));
                    }
                    if otlp.resource.service_name.is_none() {
                        otlp.resource.service_name = Some("xybrid-app".into());
                    }
                    let pipeline = xybrid_observability::otlp::ObservabilityPipeline::build(*otlp)
                        .map_err(|e| SdkError::ConfigError(format!("observability: {e}")))?;
                    let content = config.content.unwrap_or_default();
                    (pipeline.observer_with(content), Some(Arc::new(pipeline)))
                }
            };
        Ok(Self {
            inner: Arc::new(SessionInner {
                observer,
                provider_origins: config.provider_origins,
                pipeline,
                live_trace_interval: config.live_trace_interval,
                live_traced: Mutex::new(LiveSessions::default()),
                live_clock: LiveClock::System,
                retired: std::sync::atomic::AtomicBool::new(false),
            }),
            resource_telemetry: Some(crate::telemetry::prepare_resource_telemetry(
                config.resource_telemetry,
            )),
            capped: None,
            resolved: false,
        })
    }

    /// Wraps an application-owned observer without taking provider ownership.
    #[must_use]
    pub fn borrowed(observer: Observer) -> Self {
        Self {
            inner: Arc::new(SessionInner {
                observer,
                provider_origins: Vec::new(),
                pipeline: None,
                live_trace_interval: Duration::from_secs(1),
                live_traced: Mutex::new(LiveSessions::default()),
                live_clock: LiveClock::System,
                retired: std::sync::atomic::AtomicBool::new(false),
            }),
            resource_telemetry: None,
            capped: None,
            resolved: false,
        }
    }

    /// The observer used by operations in this session.
    #[must_use]
    pub fn observer(&self) -> &Observer {
        self.capped.as_ref().unwrap_or(&self.inner.observer)
    }

    /// Whether this session owns an export pipeline.
    #[must_use]
    pub fn owns_pipeline(&self) -> bool {
        self.inner.pipeline.is_some()
    }

    /// Per-signal delivery counters; unavailable for borrowed providers.
    ///
    /// They stay readable after [`Self::shutdown`], which settles them: the
    /// final counters, with what the shutdown failed or dropped.
    #[must_use]
    pub fn stats(&self) -> Option<PipelineStats> {
        self.inner.pipeline.as_ref().map(|p| p.stats())
    }

    /// Flushes the owned pipeline up to its acceptance barrier.
    ///
    /// # Errors
    /// Returns timeout or delivery errors from the pipeline.
    pub fn flush(&self, timeout: Duration) -> SdkResult<()> {
        self.inner.pipeline.as_ref().map_or(Ok(()), |p| {
            p.force_flush(timeout)
                .map_err(|e| lifecycle_error(e, timeout))
        })
    }

    /// Async [`Self::flush`], on any executor.
    ///
    /// # Errors
    /// As for [`Self::flush`].
    pub async fn flush_async(&self, timeout: Duration) -> SdkResult<()> {
        match &self.inner.pipeline {
            Some(p) => p
                .force_flush_async(timeout)
                .await
                .map_err(|e| lifecycle_error(e, timeout)),
            None => Ok(()),
        }
    }

    /// Shuts down the owned pipeline, retaining its memoized result.
    ///
    /// # Errors
    /// Returns timeout or delivery errors from the pipeline.
    pub fn shutdown(&self, timeout: Duration) -> SdkResult<()> {
        self.inner.pipeline.as_ref().map_or(Ok(()), |p| {
            p.shutdown(timeout).map_err(|e| lifecycle_error(e, timeout))
        })
    }

    /// Async [`Self::shutdown`], on any executor.
    ///
    /// # Errors
    /// As for [`Self::shutdown`].
    pub async fn shutdown_async(&self, timeout: Duration) -> SdkResult<()> {
        match &self.inner.pipeline {
            Some(p) => p
                .shutdown_async(timeout)
                .await
                .map_err(|e| lifecycle_error(e, timeout)),
            None => Ok(()),
        }
    }

    /// Resource sampling for operations using this handle.
    pub(super) fn resource_telemetry(&self) -> ResourceTelemetryMode {
        self.resource_telemetry
            .unwrap_or_else(crate::telemetry::resource_telemetry_mode)
    }

    /// Starts measuring resources around work of an operation in this session.
    pub(super) fn begin_resources(&self) -> xybrid_core::device::RunGuard {
        xybrid_core::device::ResourceMonitor::global().begin_run(self.resource_telemetry())
    }

    /// This handle with resource sampling fixed for one operation, so a later
    /// process-wide change cannot alter the operation's measurements.
    pub(super) fn for_operation(mut self) -> Self {
        self.resource_telemetry = Some(self.resource_telemetry());
        self.resolved = true;
        self
    }

    /// This handle, chosen by an operation over `floor` (see [`content_floor`]),
    /// with content capture capped by `floor`'s policy: a per-run session or
    /// observer can lower capture, never raise it.
    /// With no floor it keeps its own policy; the same session as the floor
    /// keeps the floor's cap.
    fn capped_by(mut self, floor: Option<&ObservabilitySession>) -> Self {
        let Some(floor) = floor else {
            return self;
        };
        self.capped = if Arc::ptr_eq(&self.inner, &floor.inner) {
            floor.capped.clone()
        } else {
            Some(
                self.observer()
                    .content_capped_by(floor.observer().content_policy()),
            )
        };
        self
    }

    pub(super) fn allows_provider_context(&self, origin: &str) -> bool {
        self.inner
            .provider_origins
            .iter()
            .any(|allowed| allowed == origin)
    }

    #[cfg(test)]
    pub(super) fn is_retired(&self) -> bool {
        self.inner
            .retired
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Drives this session's live-frame sampler from `clock` instead of the
    /// system clock (a test seam; see [`LiveClock`]).
    ///
    /// # Panics
    ///
    /// When the session is already shared: the clock is set right after the
    /// session is built, before any clone or operation holds it.
    #[cfg(test)]
    pub(crate) fn drive_live_clock(&mut self, clock: Arc<Mutex<Instant>>) {
        Arc::get_mut(&mut self.inner)
            .expect("the live clock is set before the session is shared")
            .live_clock = LiveClock::Manual(clock);
    }

    pub(super) fn retire(&self) {
        self.inner
            .retired
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether a frame of live frame session `session` is traced: the first
    /// frame of each interval is, the others are suppressed (and still
    /// counted in metrics).
    pub(super) fn live_trace_mode(&self, session: &str) -> TraceMode {
        let now = self.inner.live_clock.now();
        self.inner
            .live_traced
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .trace_mode(session, now, self.inner.live_trace_interval)
    }

    /// How many frame sessions the live-frame sampler remembers.
    #[cfg(test)]
    pub(crate) fn live_sessions(&self) -> usize {
        self.inner
            .live_traced
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

impl fmt::Debug for ObservabilitySession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObservabilitySession")
            .field("owns_pipeline", &self.owns_pipeline())
            .field("stats", &self.stats())
            .finish()
    }
}

pub(super) struct DefaultSlot {
    pub session: ObservabilitySession,
    pub helper_owned: bool,
}

/// The process default, and the pipeline whose counters [`super::stats`]
/// reports while none is installed.
struct Defaults {
    current: Option<DefaultSlot>,
    /// The pipeline of the helper-owned default that the last
    /// [`super::shutdown_observability`] uninstalled. Its counters, settled
    /// by that shutdown, stay readable until the next installation
    ///.
    retired: Option<Arc<xybrid_observability::otlp::ObservabilityPipeline>>,
}

static DEFAULT: RwLock<Defaults> = RwLock::new(Defaults {
    current: None,
    retired: None,
});

fn defaults_write() -> std::sync::RwLockWriteGuard<'static, Defaults> {
    DEFAULT
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn defaults_read() -> std::sync::RwLockReadGuard<'static, Defaults> {
    DEFAULT
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(super) fn replace_default(
    session: ObservabilitySession,
    helper_owned: bool,
) -> Option<DefaultSlot> {
    // The default's mode becomes the process-wide mode, which operations that
    // select no session use. A borrowed observer configures none: `Off`.
    let mode = session.resource_telemetry.unwrap_or_else(|| {
        crate::telemetry::prepare_resource_telemetry(ResourceTelemetryMode::Off)
    });
    let mut defaults = defaults_write();
    // Under the slot lock, so an operation resolving the default reads the
    // session and the mode it published together (see `default_for_operation`).
    crate::telemetry::set_resource_telemetry_mode(mode);
    // A new default starts fresh counters: the last shutdown's are gone.
    defaults.retired = None;
    defaults.current.replace(DefaultSlot {
        session,
        helper_owned,
    })
}

/// Uninstalls the default. A helper-owned one leaves its pipeline's counters
/// readable through [`default_stats`] until the next installation: its
/// shutdown settles them.
pub(super) fn uninstall_default() -> Option<DefaultSlot> {
    let mut defaults = defaults_write();
    // With nothing installed (a repeated shutdown) the retained counters stay.
    let slot = defaults.current.take()?;
    defaults.retired = slot
        .helper_owned
        .then(|| slot.session.inner.pipeline.clone())
        .flatten();
    Some(slot)
}

/// The default session's counters; with none installed, those of the
/// helper-owned default the last shutdown uninstalled, until the next
/// installation.
pub(super) fn default_stats() -> Option<PipelineStats> {
    let defaults = defaults_read();
    match &defaults.current {
        Some(slot) => slot.session.stats(),
        None => defaults.retired.as_ref().map(|pipeline| pipeline.stats()),
    }
}

/// Installs a default for new root operations and returns the previous session.
///
/// The application owns lifecycle of the session it installs. A previous
/// session created by [`init_observability`] is returned retired: its pipeline
/// drains (bounded, in the background) once the returned handle and every
/// operation pinned to it are dropped, unless the caller shuts it down first.
pub fn install_default_session(session: ObservabilitySession) -> Option<ObservabilitySession> {
    replace_default(session, false).map(|slot| {
        if slot.helper_owned {
            slot.session.retire();
        }
        slot.session
    })
}

/// Retires a replaced helper-owned session. With no operation pinned to it,
/// its pipeline shuts down now (bounded); otherwise it drains in the background
/// once the last pinned operation releases it.
pub(super) fn retire_replaced(previous: DefaultSlot) {
    if !previous.helper_owned {
        return;
    }
    if Arc::strong_count(&previous.session.inner) == 1 {
        if let Err(error) = previous.session.shutdown(Duration::from_secs(1)) {
            log::warn!(target: "xybrid_sdk", "replaced observability session: {error}");
        }
    } else {
        previous.session.retire();
    }
}

/// The default session for new root operations, when configured.
#[must_use]
pub fn default_session() -> Option<ObservabilitySession> {
    defaults_read()
        .current
        .as_ref()
        .map(|slot| slot.session.clone())
}

/// The default session for a new operation, sampling resources with the
/// process-wide mode as of now: the mode the default published, unless
/// `XybridInit::resource_telemetry` has set another since.
fn default_for_operation() -> Option<ObservabilitySession> {
    let defaults = defaults_read();
    defaults.current.as_ref().map(|slot| ObservabilitySession {
        inner: slot.session.inner.clone(),
        resource_telemetry: Some(crate::telemetry::resource_telemetry_mode()),
        capped: None,
        resolved: true,
    })
}

/// The session of the SDK operation running on this thread, if any.
pub(super) fn ambient_session() -> Option<ObservabilitySession> {
    xybrid_core::tracing::current_recorder().and_then(|recorder| {
        recorder
            .as_any()?
            .downcast_ref::<StageRecorder>()
            .map(|r| r.session.clone())
    })
}

/// The session whose content policy caps an explicit per-run session or
/// observer: the first of the enclosing operation's session and the default
/// that exports anything. A session whose observer is disabled records no
/// content to cap by, so it never lifts the default's cap: a per-run session
/// inside an operation that observes nothing is still capped by the default.
fn content_floor() -> Option<ObservabilitySession> {
    let enabled = |session: &ObservabilitySession| session.observer().is_enabled();
    ambient_session()
        .filter(enabled)
        .or_else(|| default_session().filter(enabled))
}

/// The session a new operation is pinned to, with its resource mode fixed:
/// the explicit session or observer, else the enclosing operation's, else the
/// default.
///
/// An explicit session or observer can only lower content capture: its
/// content is capped by the policy of the session the operation would
/// otherwise use (the enclosing operation's, else the default; the first
/// that is enabled, see [`content_floor`]), so a per-run setting that would
/// capture more than that session is ignored. A
/// handle the SDK already resolved for an operation (a live ASR stream's
/// decodes) keeps its cap.
pub(super) fn resolve_session(options: Option<&ObservationOptions>) -> ObservabilitySession {
    let chosen = options.and_then(|o| o.session.clone()).or_else(|| {
        options
            .and_then(|o| o.observer.clone())
            .map(ObservabilitySession::borrowed)
    });
    match chosen {
        Some(session) if session.resolved => session,
        Some(session) => session.capped_by(content_floor().as_ref()).for_operation(),
        None => ambient_session()
            .map(ObservabilitySession::for_operation)
            .or_else(default_for_operation)
            .unwrap_or_else(|| {
                ObservabilitySession::borrowed(Observer::disabled()).for_operation()
            }),
    }
}

/// Context and session captured on the calling thread before a worker hop.
/// Captures explicit suppression even when no trace headers exist.
#[derive(Clone, Debug)]
pub struct CapturedContext {
    parent: Option<ObservationContext>,
    session: ObservabilitySession,
    suppressed: bool,
    conversation_id: Option<String>,
}

impl CapturedContext {
    /// Captures the ambient parent and the operation's original session.
    #[must_use]
    pub fn capture() -> Self {
        let parent = ambient_parent();
        // Only an SDK recorder's explicit suppression carries over; a local
        // collecting recorder (`xybrid_core::tracing::capture()`) has no
        // trace context but suppresses nothing.
        let suppressed = recorder_suppressed();
        Self {
            parent,
            session: resolve_session(None),
            suppressed,
            conversation_id: conversation_id(None),
        }
    }

    /// Installs this context until the returned scope is dropped.
    #[must_use]
    pub fn enter(&self) -> CapturedScope {
        CapturedScope {
            _scope: RecorderScope::install(Arc::new(
                StageRecorder::for_session(
                    self.session.clone(),
                    self.parent.clone(),
                    self.suppressed,
                )
                .with_conversation_id(self.conversation_id.clone()),
            )),
        }
    }
}

/// A thread-local captured context guard. Never hold it across an async await.
pub struct CapturedScope {
    _scope: RecorderScope,
}

impl CapturedScope {
    pub(super) fn for_operation(
        session: ObservabilitySession,
        parent: Option<ObservationContext>,
        suppressed: bool,
        conversation_id: Option<String>,
    ) -> Self {
        Self {
            _scope: RecorderScope::install(Arc::new(
                StageRecorder::for_session(session, parent, suppressed)
                    .with_conversation_id(conversation_id),
            )),
        }
    }
}
