//! Application-owned sessions and context captured before execution hops.

use super::*;

struct SessionInner {
    observer: Observer,
    provider_origins: Vec<String>,
    pipeline: Option<Arc<xybrid_observability::otlp::ObservabilityPipeline>>,
    live_trace_interval: Duration,
    live_traced: Mutex<HashMap<String, Instant>>,
    resource_telemetry: ResourceTelemetryMode,
    retired: std::sync::atomic::AtomicBool,
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
#[derive(Clone)]
pub struct ObservabilitySession {
    inner: Arc<SessionInner>,
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
                live_traced: Mutex::new(HashMap::new()),
                resource_telemetry: config.resource_telemetry,
                retired: std::sync::atomic::AtomicBool::new(false),
            }),
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
                live_traced: Mutex::new(HashMap::new()),
                resource_telemetry: ResourceTelemetryMode::Off,
                retired: std::sync::atomic::AtomicBool::new(false),
            }),
        }
    }

    /// The observer used by operations in this session.
    #[must_use]
    pub fn observer(&self) -> &Observer {
        &self.inner.observer
    }

    /// Whether this session owns an export pipeline.
    #[must_use]
    pub fn owns_pipeline(&self) -> bool {
        self.inner.pipeline.is_some()
    }

    /// Per-signal delivery counters; unavailable for borrowed providers.
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

    pub(super) fn retire(&self) {
        self.inner
            .retired
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub(super) fn live_trace_mode(&self, session: &str) -> TraceMode {
        let now = Instant::now();
        let mut map = self
            .inner
            .live_traced
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.retain(|id, last| {
            id == session || now.duration_since(*last) <= Duration::from_secs(300)
        });
        match map.get(session) {
            Some(last) if now.duration_since(*last) < self.inner.live_trace_interval => {
                TraceMode::Suppress
            }
            _ => {
                map.insert(session.to_owned(), now);
                TraceMode::Default
            }
        }
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
static DEFAULT: RwLock<Option<DefaultSlot>> = RwLock::new(None);

pub(super) fn replace_default(
    session: ObservabilitySession,
    helper_owned: bool,
) -> Option<DefaultSlot> {
    crate::telemetry::activate_resource_telemetry(session.inner.resource_telemetry);
    DEFAULT
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .replace(DefaultSlot {
            session,
            helper_owned,
        })
}

pub(super) fn take_default() -> Option<DefaultSlot> {
    DEFAULT
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
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
    DEFAULT
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map(|slot| slot.session.clone())
}

pub(super) fn ambient_session() -> Option<ObservabilitySession> {
    xybrid_core::tracing::current_recorder().and_then(|recorder| {
        recorder
            .as_any()?
            .downcast_ref::<StageRecorder>()
            .map(|r| r.session.clone())
    })
}

pub(super) fn resolve_session(options: Option<&ObservationOptions>) -> ObservabilitySession {
    options
        .and_then(|o| o.session.clone())
        .or_else(|| {
            options
                .and_then(|o| o.observer.clone())
                .map(ObservabilitySession::borrowed)
        })
        .or_else(ambient_session)
        .or_else(default_session)
        .unwrap_or_else(|| ObservabilitySession::borrowed(Observer::disabled()))
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
        let suppressed = xybrid_core::tracing::current_recorder().is_some() && parent.is_none();
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
