//! The packaged OTLP pipeline (D29–D32): application-owned providers over the
//! package's bounded processors and exporters, on a private transport
//! runtime.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::logs::SdkLoggerProvider;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};

use super::batch::{
    Batcher, FlushTicket, Logs, Losses, PipelineLogProcessor, PipelineSpanProcessor, Spans,
};
use super::config::{self, OtlpConfig};
use super::encode::ResourceInfo;
use super::export::{core_for, ExporterShared, OtlpMetricExporter, SETTLE_GRACE};
use super::runtime::{
    self, after, lock, tokio_after, AbortOnDrop, CompleteOnDrop, Completion, TransportRuntime,
};
use super::stats::{PipelineStats, SignalStats};
use super::transport;
use crate::content::{ContentPolicy, SpanContentEncoding};
use crate::error::{Error, Result};
use crate::observer::Observer;

/// Extra time a synchronous caller waits beyond its timeout for the
/// transport runtime to report (scheduling slack).
const SLACK: Duration = Duration::from_millis(250);

/// How long a shutdown task may run past its deadline: one grace period to
/// settle cancelled work, one to shut the providers down (application span
/// processors included). The first caller waits this long beyond its timeout.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2 * SETTLE_GRACE.as_secs());

fn timeout_error(operation: &'static str, timeout: Duration) -> Error {
    Error::Timeout {
        operation,
        timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
    }
}

fn stopped(operation: &'static str) -> Error {
    Error::Lifecycle {
        operation,
        message: "the transport stopped before the operation completed".into(),
    }
}

/// Whether a provider shutdown in step e left the provider shut down. One the
/// application already shut down is: whatever that shutdown lost is counted.
fn provider_shut_down(result: OTelSdkResult) -> bool {
    matches!(result, Ok(()) | Err(OTelSdkError::AlreadyShutdown))
}

fn delta(now: SignalStats, before: SignalStats) -> Losses {
    Losses {
        failed: now.failed.saturating_sub(before.failed),
        dropped: now.dropped.saturating_sub(before.dropped),
    }
}

/// The first shutdown caller's result once its bounded wait ended: the
/// shutdown's own result if known; otherwise this caller's timeout, stored
/// so that every later call returns exactly what the first one did (D30.5).
fn settle_first(
    completion: &Completion<Result<()>>,
    waited: Option<Result<()>>,
    timeout: Duration,
) -> Result<()> {
    if let Some(result) = waited {
        return result;
    }
    let timed_out = Err(timeout_error("shutdown", timeout));
    if completion.complete(timed_out.clone()) {
        timed_out
    } else {
        completion.get().unwrap_or(timed_out)
    }
}

/// An application-owned OTLP/HTTP pipeline.
///
/// Builds tracer, meter and logger providers that the application owns;
/// nothing is installed globally. Delivery is bounded in memory per signal
/// (item and byte budgets, counted in [`ObservabilityPipeline::stats`]),
/// exports retry within one deadline, batches are split to fit the request
/// body bound, and nothing is spooled to disk. Transport work runs on a
/// private Tokio runtime, so the pipeline works from plain threads and from
/// inside any application runtime.
pub struct ObservabilityPipeline {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
    logger_provider: Option<SdkLoggerProvider>,
    traces: Option<Arc<Batcher<Spans>>>,
    logs: Option<Arc<Batcher<Logs>>>,
    metrics: Option<Arc<ExporterShared>>,
    /// Serializes metrics flushes, so at most one blocking SDK flush is
    /// parked on the transport's blocking pool at a time (the periodic reader
    /// runs collections one at a time anyway).
    meter_gate: Arc<tokio::sync::Mutex<()>>,
    runtime: Arc<TransportRuntime>,
    shutdown: Mutex<Option<Arc<Completion<Result<()>>>>>,
    /// `max_request_body_bytes` and content encoding of the traces export, if
    /// spans are exported.
    span_export: Option<(usize, SpanContentEncoding)>,
}

impl ObservabilityPipeline {
    /// Validates the configuration, then builds providers for every
    /// configured signal and starts the transport runtime. No network I/O
    /// happens until the first export.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when no signal is configured or an option is invalid
    /// (checked before any thread, runtime or connection exists), or
    /// [`Error::Lifecycle`] when the transport cannot start.
    pub fn build(config: OtlpConfig) -> Result<Self> {
        Self::build_configured(config, false, |builder| builder)
    }

    /// Builds the owned pipeline with an application processor for local inspection.
    /// The processor observes the same sampled spans as the transport. If no traces
    /// endpoint is configured, spans stay local while other configured signals export.
    ///
    /// # Errors
    /// Returns the same configuration and lifecycle errors as [`Self::build`].
    pub fn build_with_span_processor<P: opentelemetry_sdk::trace::SpanProcessor + 'static>(
        config: OtlpConfig,
        processor: P,
    ) -> Result<Self> {
        Self::build_configured(config, true, |builder| {
            builder.with_span_processor(processor)
        })
    }

    fn build_configured(
        config: OtlpConfig,
        local_traces: bool,
        decorate: impl FnOnce(
            opentelemetry_sdk::trace::TracerProviderBuilder,
        ) -> opentelemetry_sdk::trace::TracerProviderBuilder,
    ) -> Result<Self> {
        let resolved = config::resolve(&config)?;
        let client = transport::client()?;
        let runtime = TransportRuntime::start()?;
        let limits = resolved.limits;
        let resource = resolved.resource.clone();
        let info = Arc::new(ResourceInfo::from_sdk(&resource));
        let handle = runtime.handle().clone();

        let traces = resolved.traces.as_ref().map(|signal| {
            Batcher::<Spans>::start(
                core_for(&client, signal, &limits),
                limits,
                info.clone(),
                handle.clone(),
                runtime.token(),
            )
        });
        let tracer_provider = if traces.is_some() || local_traces {
            let mut builder = decorate(SdkTracerProvider::builder());
            if let Some(batcher) = &traces {
                builder = builder.with_span_processor(PipelineSpanProcessor {
                    batcher: batcher.clone(),
                });
            }
            Some(
                builder
                    .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
                    .with_resource(resource.clone())
                    .build(),
            )
        } else {
            None
        };

        let logs = resolved.logs.as_ref().map(|signal| {
            Batcher::<Logs>::start(
                core_for(&client, signal, &limits),
                limits,
                info.clone(),
                handle.clone(),
                runtime.token(),
            )
        });
        let logger_provider = logs.as_ref().map(|batcher| {
            SdkLoggerProvider::builder()
                .with_log_processor(PipelineLogProcessor {
                    batcher: batcher.clone(),
                })
                .with_resource(resource.clone())
                .build()
        });

        let metrics = resolved.metrics.as_ref().map(|signal| {
            ExporterShared::new(
                core_for(&client, signal, &limits),
                handle.clone(),
                runtime.token(),
                None,
                limits.metric_export_timeout(),
                limits.max_buffered_bytes,
            )
        });
        let meter_provider = metrics.as_ref().map(|shared| {
            let reader = PeriodicReader::builder(OtlpMetricExporter::new(shared.clone()))
                .with_interval(limits.metric_export_interval)
                .build();
            SdkMeterProvider::builder()
                .with_reader(reader)
                .with_resource(resource.clone())
                .build()
        });

        Ok(Self {
            tracer_provider,
            meter_provider,
            logger_provider,
            traces,
            logs,
            metrics,
            meter_gate: Arc::new(tokio::sync::Mutex::new(())),
            runtime,
            shutdown: Mutex::new(None),
            span_export: resolved.traces.as_ref().map(|traces| {
                (
                    limits.max_request_body_bytes,
                    traces.protocol.span_content_encoding(),
                )
            }),
        })
    }

    /// An observer wired to this pipeline's providers with the default
    /// content policy (environment override honored).
    #[must_use]
    pub fn observer(&self) -> Observer {
        self.observer_with(ContentPolicy::from_env())
    }

    /// An observer wired to this pipeline's providers. With a traces signal,
    /// the content budgets are clamped to fit this pipeline's
    /// `max_request_body_bytes` (or the policy's own bound, default 480 KiB,
    /// if smaller) as its traces protocol writes content: `http/protobuf`
    /// carries twice the content of `http/json` in the same request (416 KiB
    /// against 208 KiB at the default bound). The pipeline's protocol
    /// replaces the policy's own [`ContentPolicy::span_content_encoding`].
    #[must_use]
    pub fn observer_with(&self, content: ContentPolicy) -> Observer {
        let content = match self.span_export {
            Some((bound, encoding)) => {
                let content = content.with_span_content_encoding(encoding);
                if bound < content.max_request_body_bytes() {
                    content.with_max_request_body_bytes(bound)
                } else {
                    content
                }
            }
            None => content,
        };
        let mut builder = Observer::builder().content_policy(content);
        if let Some(p) = &self.tracer_provider {
            builder = builder.tracer_provider(p);
        }
        if let Some(p) = &self.meter_provider {
            builder = builder.meter_provider(p);
        }
        if let Some(p) = &self.logger_provider {
            builder = builder.logger_provider(p);
        }
        builder.build()
    }

    /// The tracer provider, for applications that also create their own spans.
    #[must_use]
    pub fn tracer_provider(&self) -> Option<&SdkTracerProvider> {
        self.tracer_provider.as_ref()
    }

    /// The meter provider.
    #[must_use]
    pub fn meter_provider(&self) -> Option<&SdkMeterProvider> {
        self.meter_provider.as_ref()
    }

    /// The logger provider.
    #[must_use]
    pub fn logger_provider(&self) -> Option<&SdkLoggerProvider> {
        self.logger_provider.as_ref()
    }

    /// Per-signal delivery counters and buffered bytes (zeros for
    /// unconfigured signals).
    #[must_use]
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            traces: self.traces.as_ref().map(|b| b.stats()).unwrap_or_default(),
            logs: self.logs.as_ref().map(|b| b.stats()).unwrap_or_default(),
            metrics: self.metrics.as_ref().map(|m| m.stats()).unwrap_or_default(),
        }
    }

    /// Exports the items pending now (and takes a metrics collection),
    /// waiting at most `timeout`. Items accepted after the call are not
    /// waited for. After shutdown started, returns the shutdown result.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] when the bound is exceeded (exports keep their own
    /// deadlines), [`Error::Delivery`] when a covered item failed or was
    /// dropped.
    pub fn force_flush(&self, timeout: Duration) -> Result<()> {
        let completion = self.start_flush(timeout);
        completion
            .wait_timeout(timeout.saturating_add(SLACK))
            .unwrap_or_else(|| Err(timeout_error("force_flush", timeout)))
    }

    /// Async [`ObservabilityPipeline::force_flush`]; runs on any executor.
    ///
    /// # Errors
    ///
    /// As for [`ObservabilityPipeline::force_flush`].
    pub async fn force_flush_async(&self, timeout: Duration) -> Result<()> {
        let completion = self.start_flush(timeout);
        self.wait_bounded(&completion, timeout.saturating_add(SLACK))
            .await
            .unwrap_or_else(|| Err(timeout_error("force_flush", timeout)))
    }

    /// Closes admission for every signal, drains queued items and takes a
    /// final metrics collection, cancels what is still running at the
    /// deadline, settles the counters, shuts the providers and the transport
    /// down, and only then returns. Every later call returns the first
    /// call's result without new work; a concurrent call waits at most its
    /// own timeout for it.
    ///
    /// Safe to call from inside another Tokio runtime (it never calls
    /// `block_on` on the caller's thread).
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] when work had to be cancelled at the deadline (or
    /// the final metrics collection did not complete by then),
    /// [`Error::Delivery`] when an item pending at the call failed or was
    /// dropped, [`Error::Lifecycle`] when a provider failed to shut down (an
    /// application span processor returned an error or panicked).
    pub fn shutdown(&self, timeout: Duration) -> Result<()> {
        let (completion, first) = self.start_shutdown(timeout);
        let result = if first {
            let waited = completion
                .wait_timeout(timeout.saturating_add(SHUTDOWN_GRACE).saturating_add(SLACK));
            settle_first(&completion, waited, timeout)
        } else {
            completion
                .wait_timeout(timeout)
                .unwrap_or_else(|| Err(timeout_error("shutdown", timeout)))
        };
        if first {
            self.runtime.teardown();
        }
        result
    }

    /// Async [`ObservabilityPipeline::shutdown`]; runs on any executor.
    ///
    /// # Errors
    ///
    /// As for [`ObservabilityPipeline::shutdown`].
    pub async fn shutdown_async(&self, timeout: Duration) -> Result<()> {
        let (completion, first) = self.start_shutdown(timeout);
        let result = if first {
            let waited = self
                .wait_bounded(
                    &completion,
                    timeout.saturating_add(SHUTDOWN_GRACE).saturating_add(SLACK),
                )
                .await;
            settle_first(&completion, waited, timeout)
        } else {
            self.wait_bounded(&completion, timeout)
                .await
                .unwrap_or_else(|| Err(timeout_error("shutdown", timeout)))
        };
        if first {
            self.runtime.teardown();
        }
        result
    }

    /// Whether [`ObservabilityPipeline::shutdown`] has been called (span and
    /// log admission is closed once this returns `true`).
    #[must_use]
    pub fn is_shut_down(&self) -> bool {
        lock(&self.shutdown).is_some()
    }

    /// Waits at most `timeout` for `completion` on any executor, using the
    /// transport runtime's timer.
    async fn wait_bounded(
        &self,
        completion: &Arc<Completion<Result<()>>>,
        timeout: Duration,
    ) -> Option<Result<()>> {
        if let Some(result) = completion.get() {
            return Some(result);
        }
        let bounded = Completion::new();
        let guard = CompleteOnDrop::new(bounded.clone(), None);
        let source = completion.clone();
        runtime::spawn(self.runtime.handle(), async move {
            let result = tokio::time::timeout(timeout, source.wait()).await.ok();
            guard.complete(result);
        });
        match bounded.wait().await {
            Some(result) => Some(result),
            None => completion.get(),
        }
    }

    fn start_flush(&self, timeout: Duration) -> Arc<Completion<Result<()>>> {
        let result = Completion::new();
        if let Some(shutdown) = lock(&self.shutdown).clone() {
            if let Some(known) = shutdown.get() {
                result.complete(known);
                return result;
            }
            let guard =
                CompleteOnDrop::new(result.clone(), Err(timeout_error("force_flush", timeout)));
            runtime::spawn(self.runtime.handle(), async move {
                match tokio::time::timeout(timeout, shutdown.wait()).await {
                    Ok(known) => guard.complete(known),
                    Err(_) => guard.complete(Err(timeout_error("force_flush", timeout))),
                }
            });
            return result;
        }
        // The barrier is taken now, on the caller's thread.
        let tickets: Vec<FlushTicket> = [
            self.traces.as_ref().map(|b| b.flush_ticket()),
            self.logs.as_ref().map(|b| b.flush_ticket()),
        ]
        .into_iter()
        .flatten()
        .collect();
        let metrics = self.metrics.clone();
        let metrics_before = metrics.as_ref().map(|m| m.stats());
        // The collection taken by this call. It runs on the periodic reader
        // (the SDK flush blocks until the reader exported it), parked on the
        // blocking pool behind `meter_gate`; abandoned if the flush times out.
        let collection = self.meter_provider.clone().map(|m| {
            let gate = self.meter_gate.clone();
            AbortOnDrop(runtime::spawn(self.runtime.handle(), async move {
                let permit = gate.lock_owned().await;
                tokio::task::spawn_blocking(move || {
                    let result = m.force_flush();
                    drop(permit);
                    result
                })
                .await
            }))
        });
        let deadline = tokio_after(tokio::time::Instant::now(), timeout);
        let guard = CompleteOnDrop::new(result.clone(), Err(stopped("force_flush")));
        runtime::spawn(self.runtime.handle(), async move {
            let work = async {
                let mut losses = Losses::default();
                let mut closed = false;
                for ticket in tickets {
                    match ticket {
                        FlushTicket::Done(l) => losses.add(l),
                        FlushTicket::Wait(done) => losses.add(done.wait().await),
                        FlushTicket::Closed => closed = true,
                    }
                }
                let metrics_ok = match collection {
                    Some(task) => matches!(task.await, Ok(Ok(Ok(())))),
                    None => true,
                };
                (losses, closed, metrics_ok)
            };
            let outcome = match tokio::time::timeout_at(deadline, work).await {
                Err(_) => Err(timeout_error("force_flush", timeout)),
                Ok((mut losses, closed, metrics_ok)) => {
                    if let (Some(m), Some(before)) = (&metrics, metrics_before) {
                        losses.add(delta(m.stats(), before));
                    }
                    if !losses.is_empty() {
                        Err(Error::Delivery {
                            operation: "force_flush",
                            failed: losses.failed,
                            dropped: losses.dropped,
                        })
                    } else if closed {
                        Err(Error::Lifecycle {
                            operation: "force_flush",
                            message: "a signal was already shut down".into(),
                        })
                    } else if !metrics_ok {
                        Err(Error::Lifecycle {
                            operation: "force_flush",
                            message: "the metrics collection failed".into(),
                        })
                    } else {
                        Ok(())
                    }
                }
            };
            guard.complete(outcome);
        });
        result
    }

    /// Starts the shutdown once; returns its completion and whether this
    /// call started it.
    fn start_shutdown(&self, timeout: Duration) -> (Arc<Completion<Result<()>>>, bool) {
        self.start_shutdown_before_drain(timeout, || {})
    }

    /// The private callback lets the race regression settle a pending export
    /// after entry but before the drain is scheduled, without timing guesses.
    fn start_shutdown_before_drain(
        &self,
        timeout: Duration,
        before_drain: impl FnOnce(),
    ) -> (Arc<Completion<Result<()>>>, bool) {
        let mut slot = lock(&self.shutdown);
        if let Some(existing) = slot.as_ref() {
            return (existing.clone(), false);
        }
        // Capture losses on the caller's thread, before any shutdown work can
        // run. A pending collection may fail before the drain gets scheduled.
        let metrics_before = self.metrics.as_ref().map(|m| m.stats());
        let completion = Completion::new();
        *slot = Some(completion.clone());
        // a. Close admission for every signal at once, before anyone can
        // observe `is_shut_down()`.
        if let Some(batcher) = &self.traces {
            batcher.close();
        }
        if let Some(batcher) = &self.logs {
            batcher.close();
        }
        drop(slot);
        let deadline = after(Instant::now(), timeout);
        if let Some(metrics) = &self.metrics {
            metrics.cap_deadline(deadline);
        }
        let traces = self.traces.clone();
        let logs = self.logs.clone();
        let metrics = self.metrics.clone();
        let meter = self.meter_provider.clone();
        let tracer = self.tracer_provider.clone();
        let logger = self.logger_provider.clone();
        let guard = CompleteOnDrop::new(completion.clone(), Err(stopped("shutdown")));
        before_drain();
        runtime::spawn(self.runtime.handle(), async move {
            let until = tokio::time::Instant::from_std(deadline);
            let settle_until = tokio_after(until, SETTLE_GRACE);
            let mut cancelled = false;
            let mut losses = Losses::default();

            // b. Drain. The final metrics collection runs on the SDK's
            // periodic reader, started by the meter provider's shutdown. That
            // call stops waiting after a fixed 5 s while the reader keeps
            // going (and the SDK folds its timeout into a generic error), so
            // its return value is not the signal: the reader shuts the
            // exporter down right after the final export, and that
            // (`finished`) is what the drain waits for, up to this shutdown's
            // own deadline. A reader that can no longer run (its thread
            // died) is therefore reported as a timeout at the deadline.
            let final_collection = match (meter, &metrics, metrics_before) {
                (Some(meter), Some(shared), Some(before)) => {
                    let task = tokio::task::spawn_blocking(move || {
                        let _ = meter.shutdown();
                    });
                    Some((shared.clone(), before, shared.finished(), task))
                }
                _ => None,
            };
            let signals: Vec<_> = traces
                .as_ref()
                .map(|b| b.shutdown(deadline))
                .into_iter()
                .chain(logs.as_ref().map(|b| b.shutdown(deadline)))
                .collect();

            if let Some((shared, before, finished, task)) = final_collection {
                if tokio::time::timeout_at(until, finished.wait())
                    .await
                    .is_err()
                {
                    // c. The deadline passed: refuse later collections and
                    // abort the export in flight.
                    cancelled = true;
                    shared.close();
                    shared.cancel();
                }
                // d. Settle the counters before computing the result.
                if tokio::time::timeout_at(settle_until, shared.wait_idle())
                    .await
                    .is_err()
                {
                    cancelled = true;
                }
                shared.close();
                cancelled |= shared.cut_at_deadline();
                losses.add(delta(shared.stats(), before));
                // Let the SDK call return so its thread is gone by teardown.
                let _ = tokio::time::timeout_at(settle_until, task).await;
            }
            for signal in signals {
                let result = signal.wait().await;
                cancelled |= result.cancelled;
                losses.add(result.losses);
            }

            // e. Shut the providers down. Our processors have settled and
            // return at once; an application's processor may block, so this
            // never runs on the transport's worker thread, and it is bounded.
            // Our processors fail only with what `cancelled` and `losses`
            // already hold, so a failure the result does not otherwise explain
            // comes from an application processor.
            let mut providers_failed = false;
            if tracer.is_some() || logger.is_some() {
                let provider_until =
                    tokio_after(tokio::time::Instant::now().max(until), SETTLE_GRACE);
                let budget = provider_until.saturating_duration_since(tokio::time::Instant::now());
                let task = tokio::task::spawn_blocking(move || {
                    // The logger carries only our processor: shut it down
                    // first, so an application span processor that panics
                    // cannot skip it.
                    let logs = logger.map(|p| p.shutdown_with_timeout(budget));
                    let traces = tracer.map(|p| p.shutdown_with_timeout(budget));
                    [logs, traces].into_iter().flatten().all(provider_shut_down)
                });
                match tokio::time::timeout_at(provider_until, task).await {
                    Err(_) => cancelled = true,
                    Ok(Ok(shut_down)) => providers_failed = !shut_down,
                    // A processor panicked (or the task was cancelled).
                    Ok(Err(_)) => providers_failed = true,
                }
            }

            let result = if cancelled {
                Err(timeout_error("shutdown", timeout))
            } else if !losses.is_empty() {
                Err(Error::Delivery {
                    operation: "shutdown",
                    failed: losses.failed,
                    dropped: losses.dropped,
                })
            } else if providers_failed {
                // The processor's own error text is not ours to forward: it
                // may carry payloads or credentials.
                Err(Error::Lifecycle {
                    operation: "shutdown",
                    message: "a span processor failed to shut down".into(),
                })
            } else {
                Ok(())
            };
            guard.complete(result);
        });
        (completion, true)
    }
}

impl Drop for ObservabilityPipeline {
    fn drop(&mut self) {
        if lock(&self.shutdown).is_none() {
            // Dropped without shutdown: cancel everything now, no flush.
            if let Some(batcher) = &self.traces {
                batcher.abandon();
            }
            if let Some(batcher) = &self.logs {
                batcher.abandon();
            }
            if let Some(metrics) = &self.metrics {
                metrics.close();
                metrics.cancel();
            }
        }
        self.runtime.teardown();
    }
}

impl fmt::Debug for ObservabilityPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObservabilityPipeline")
            .field("traces", &self.tracer_provider.is_some())
            .field("metrics", &self.meter_provider.is_some())
            .field("logs", &self.logger_provider.is_some())
            .field("stats", &self.stats())
            .field("shut_down", &self.is_shut_down())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_failure_before_the_drain_is_scheduled_is_reported() {
        use std::io::{Read as _, Write as _};
        use std::net::TcpListener;
        use std::sync::mpsc;
        use std::thread;

        use opentelemetry::metrics::MeterProvider as _;

        use super::super::config::{BatchLimits, EnvironmentSource, SignalEndpoint};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1/metrics", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let (received, request_ready) = mpsc::sync_channel(1);
        let (release, failed_response) = mpsc::sync_channel(1);
        let server = thread::spawn(move || {
            // Hold the first export until shutdown has captured its baseline.
            // The final cumulative collection then succeeds, so only the
            // first export can make shutdown report a delivery failure.
            for index in 0..2 {
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut socket = loop {
                    match listener.accept() {
                        Ok((socket, _)) => break socket,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "metrics request {index} absent");
                            thread::sleep(Duration::from_millis(1));
                        }
                        Err(e) => panic!("accept: {e}"),
                    }
                };
                // Accepted sockets inherit O_NONBLOCK on macOS/BSD (not on Linux), which
                // would make the read timeout below a busy WouldBlock instead of a wait.
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = Vec::new();
                let mut buf = [0; 4096];
                loop {
                    let n = socket.read(&mut buf).unwrap();
                    assert_ne!(n, 0, "incomplete metrics request");
                    request.extend_from_slice(&buf[..n]);
                    if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = std::str::from_utf8(&request[..end]).unwrap();
                        let length: usize = headers
                            .lines()
                            .find_map(|line| {
                                let (key, value) = line.split_once(':')?;
                                key.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse().unwrap())
                            })
                            .unwrap();
                        if request.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
                let status = if index == 0 {
                    received.send(()).unwrap();
                    failed_response
                        .recv_timeout(Duration::from_secs(5))
                        .unwrap();
                    "400 Bad Request"
                } else {
                    "200 OK"
                };
                write!(
                    socket,
                    "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                .unwrap();
            }
        });
        let pipeline = Arc::new(
            ObservabilityPipeline::build(
                OtlpConfig::default()
                    .with_environment(EnvironmentSource::Disabled)
                    .with_metrics(Some(SignalEndpoint::new(endpoint)))
                    .with_metric_export_interval(Duration::from_secs(600))
                    .with_batch(BatchLimits {
                        export_timeout: Duration::from_secs(10),
                        ..BatchLimits::default()
                    }),
            )
            .unwrap(),
        );
        let counter = pipeline
            .meter_provider()
            .unwrap()
            .meter("race")
            .u64_counter("calls")
            .build();
        counter.add(1, &[]);
        let flush_pipeline = pipeline.clone();
        let flush = thread::spawn(move || flush_pipeline.force_flush(Duration::from_secs(5)));
        request_ready.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(pipeline.stats().metrics.pending, 1);
        let (completion, first) =
            pipeline.start_shutdown_before_drain(Duration::from_secs(5), || {
                release.send(()).unwrap();
                assert!(matches!(
                    flush.join().unwrap(),
                    Err(Error::Delivery { failed: 1, .. })
                ));
                assert_eq!(pipeline.stats().metrics.pending, 0);
            });
        assert!(first);
        let result = completion.wait_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            matches!(
                result,
                Err(Error::Delivery {
                    operation: "shutdown",
                    failed: 1,
                    dropped: 0,
                })
            ),
            "{result:?}"
        );
        let stats = pipeline.stats().metrics;
        assert_eq!(
            (stats.accepted, stats.exported, stats.failed, stats.pending),
            (2, 1, 1, 0)
        );
        assert!(stats.is_consistent());
        server.join().unwrap();
    }

    #[test]
    fn the_first_callers_timeout_is_the_remembered_shutdown_result() {
        // F9: when the first caller stops waiting before the shutdown task
        // reports, its timeout is stored, so a later fallback (the task
        // dropped by teardown) cannot change what later calls return.
        let completion = Completion::<Result<()>>::new();
        let fallback = CompleteOnDrop::new(completion.clone(), Err(stopped("shutdown")));
        let first = settle_first(&completion, None, Duration::from_millis(50));
        assert!(first.as_ref().is_err_and(Error::is_timeout));
        drop(fallback);
        let later = completion.get().unwrap();
        assert_eq!(
            later.map_err(|e| e.to_string()),
            first.map_err(|e| e.to_string())
        );

        // A result that arrived in time is returned unchanged.
        let completion = Completion::<Result<()>>::new();
        assert!(settle_first(&completion, Some(Ok(())), Duration::from_millis(50)).is_ok());
        // A result that arrived just after the wait ended wins over the
        // caller's timeout (write-once).
        let completion = Completion::<Result<()>>::new();
        completion.complete(Ok(()));
        assert!(settle_first(&completion, None, Duration::from_millis(50)).is_ok());
    }
}
