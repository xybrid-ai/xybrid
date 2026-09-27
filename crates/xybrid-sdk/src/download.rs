//! Download progress as one aggregated, monotonic, throttled signal.
//!
//! Every surface that shows a progress bar reads the same [`DownloadStatus`]:
//! state, a 0.0–1.0 fraction, bytes transferred, and the total when the source
//! declares one. The fraction is aggregated **across all of a model's
//! artifacts** (weights plus companions such as a VLM projector), never moves
//! backwards, and is emitted at most [`MIN_EMIT_INTERVAL`] apart rather than
//! once per 8 KiB chunk.
//!
//! Two things produce a status:
//!
//! - [`ProgressReporter`] — handed to
//!   [`RegistryClient::fetch_extracted`](crate::registry_client::RegistryClient::fetch_extracted)
//!   and friends, it turns per-file byte counts into aggregated statuses.
//! - [`ModelDownload`] — a standalone handle over a background download, for
//!   hosts whose `load` call blocks (every FFI binding). Poll
//!   [`ModelDownload::status`], block on [`ModelDownload::next_status`], or
//!   register an in-process observer with [`ModelDownload::watch`].
//!
//! # Example
//!
//! ```no_run
//! # fn _example() -> Result<(), Box<dyn std::error::Error>> {
//! use std::time::Duration;
//! use xybrid_sdk::{DownloadState, ModelLoader};
//!
//! let loader = ModelLoader::from_registry("qwen3-0.6b");
//! let download = loader.start_download();
//! loop {
//!     let status = download.next_status(Duration::from_millis(250))?;
//!     println!("{}/{:?} bytes", status.downloaded_bytes, status.total_bytes);
//!     if status.state != DownloadState::Downloading {
//!         break;
//!     }
//! }
//! let model = loader.load()?; // cached: returns without touching the network
//! # Ok(())
//! # }
//! ```

use crate::model::{SdkError, SdkResult};
use crate::registry_client::RegistryClient;
use crate::source::ModelSource;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Ceiling for in-flight download progress, in basis points (99.99%).
///
/// 1.0 is reserved for [`DownloadState::Ready`]: checksum verification and
/// extraction still run after the last byte lands, so a bar that hit 100%
/// while those were going would announce a model the caller cannot use yet.
pub(crate) const MAX_IN_FLIGHT_PROGRESS_BP: u32 = 9_999;

/// Shortest gap between two emitted progress updates.
///
/// A 1 GiB download reads ~130,000 chunks; forwarding each one across an FFI
/// boundary costs far more than the bar is worth. ~10 updates a second is
/// smooth to the eye and cheap on every transport.
pub(crate) const MIN_EMIT_INTERVAL: Duration = Duration::from_millis(100);

/// Lifecycle of a model download.
///
/// Also describes the background download behind a speculative load, where
/// `Downloading` means runs are being served from the cloud meanwhile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadState {
    /// Bytes still in flight.
    Downloading,
    /// Every artifact landed, verified and extracted — the model is usable.
    Ready,
    /// The download failed. For a speculative load the cloud keeps serving and
    /// `is_loaded()` will never flip, so surfacing this is the only way a host
    /// can stop waiting.
    Failed,
    /// The caller asked for the download to stop (see [`ModelDownload::cancel`]).
    Cancelled,
}

/// One consistent read of a download's progress and state.
///
/// Taken as a snapshot so a polling host cannot observe a torn pair (for
/// example `Ready` alongside a stale 0.34 progress).
///
/// # Guarantees
///
/// - `progress` and `downloaded_bytes` never decrease for a given download.
///   A retry re-transfers bytes that were already counted, so the bar stalls
///   rather than rewinding.
/// - `progress` stays below 1.0 until `state` is [`DownloadState::Ready`].
/// - `total_bytes` is `None` while the size is unknown (a Hugging Face repo,
///   or a registry entry with `size_bytes = 0`). `progress` then comes from a
///   coarser signal — completed files over total files — and
///   `downloaded_bytes` still counts real bytes.
/// - For a single-file download, the size the server announces fills in (or
///   corrects) `total_bytes` as soon as the response arrives, so it can turn
///   from `None` to `Some` before the first byte. It never changes once bytes
///   are counted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DownloadStatus {
    pub state: DownloadState,
    /// 0.0..=1.0, aggregated across every artifact of the model.
    pub progress: f32,
    /// Running total of bytes written across every artifact.
    pub downloaded_bytes: u64,
    /// Sum of every artifact's declared size (for a single file, the size the
    /// server announced), or `None` when unknown.
    pub total_bytes: Option<u64>,
}

impl Default for DownloadStatus {
    /// A download that has not reported anything yet.
    fn default() -> Self {
        Self {
            state: DownloadState::Downloading,
            progress: 0.0,
            downloaded_bytes: 0,
            total_bytes: None,
        }
    }
}

impl DownloadStatus {
    /// A finished download: 1.0, with `downloaded_bytes` settled on whichever
    /// of the measured and declared counts is larger.
    ///
    /// Taking the larger covers both ways the two can disagree: a throttled
    /// final update may have been dropped, leaving the measured count short of
    /// the declared total; and a stale registry size may be smaller than what
    /// actually landed, in which case settling on the declared total would
    /// rewind the bar at the very last frame.
    pub(crate) fn ready(downloaded_bytes: u64, total_bytes: Option<u64>) -> Self {
        Self {
            state: DownloadState::Ready,
            progress: 1.0,
            downloaded_bytes: downloaded_bytes.max(total_bytes.unwrap_or(0)),
            total_bytes,
        }
    }

    /// Terminal status for a download that stopped without completing.
    pub(crate) fn terminal(state: DownloadState, last: &DownloadStatus) -> Self {
        Self { state, ..*last }
    }
}

/// Turns per-artifact byte counts into aggregated [`DownloadStatus`] updates.
///
/// One reporter covers one logical download — the model's main file plus every
/// companion artifact. The registry client hands it the total up front (it
/// resolves every artifact's `size_bytes` before the first request), which is
/// what makes a single bar across multiple files possible.
///
/// Emission is throttled to [`MIN_EMIT_INTERVAL`] except for the start of a
/// transfer, a newly announced size, artifact completion and the terminal
/// update, which always go out.
pub struct ProgressReporter<'a> {
    /// Byte total the bar is scaled against; `0` while unknown. Atomic because
    /// a single-file download learns its real size from the response headers
    /// (see [`Self::file_size_announced`]).
    total_bytes: AtomicU64,
    /// Bytes belonging to artifacts that already finished.
    completed_bytes: AtomicU64,
    /// Highest byte count ever reported. A retry restarts the current file at
    /// zero; reporting the high-water mark keeps the bar from rewinding.
    reported_bytes: AtomicU64,
    /// Fallback progress in basis points, used when `total_bytes` is unknown.
    fraction_bp: AtomicU32,
    /// How many artifacts this download will fetch, for the coarse
    /// completed-files fallback. `0` means even that is unknown.
    artifact_count: u32,
    /// Artifacts finished so far, feeding the coarse fallback.
    completed_files: AtomicU32,
    cancel: Arc<AtomicBool>,
    last_emit: Mutex<Option<Instant>>,
    /// Borrowed rather than boxed so callers can pass a closure that captures
    /// its environment (a CLI progress bar, a stream sink) without a `'static`
    /// bound leaking into every public `fetch*` signature.
    sink: &'a dyn Fn(DownloadStatus),
}

impl std::fmt::Debug for ProgressReporter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgressReporter")
            .field("total_bytes", &self.total())
            .field("reported_bytes", &self.reported_bytes)
            .finish_non_exhaustive()
    }
}

impl<'a> ProgressReporter<'a> {
    /// Build a reporter over a callback. `total_bytes` of `Some(0)` is treated
    /// as unknown — the registry reports `0` for entries with no declared size.
    pub(crate) fn new(
        total_bytes: Option<u64>,
        artifact_count: usize,
        cancel: Arc<AtomicBool>,
        sink: &'a dyn Fn(DownloadStatus),
    ) -> Self {
        Self {
            total_bytes: AtomicU64::new(total_bytes.unwrap_or(0)),
            completed_bytes: AtomicU64::new(0),
            reported_bytes: AtomicU64::new(0),
            fraction_bp: AtomicU32::new(0),
            artifact_count: artifact_count.try_into().unwrap_or(u32::MAX),
            completed_files: AtomicU32::new(0),
            cancel,
            last_emit: Mutex::new(None),
            sink,
        }
    }

    /// Whether the caller asked for this download to stop.
    ///
    /// Checked inside the read loop, so cancellation takes effect within one
    /// 8 KiB chunk rather than at the end of the file.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Error to return from a download loop that observed cancellation.
    pub(crate) fn cancelled_error() -> SdkError {
        SdkError::Cancelled {
            message: "download cancelled by caller".to_string(),
        }
    }

    /// The byte total, or `None` while it is unknown.
    fn total(&self) -> Option<u64> {
        Some(self.total_bytes.load(Ordering::Relaxed)).filter(|bytes| *bytes > 0)
    }

    /// Announce that a transfer is starting, before any response arrives.
    ///
    /// Connecting, following the redirect and waiting for headers can take
    /// seconds on a slow link. Without this, nothing is emitted until the first
    /// body bytes land, and a host cannot tell a download that is starting
    /// from one that is stuck.
    pub(crate) fn begin_transfer(&self) {
        self.emit(true);
    }

    /// Adopt the size a server announced for the file about to transfer.
    ///
    /// Only for a single-file download, and only before any byte is counted.
    /// There the server's size is the real one, which beats a registry entry
    /// that declares none (`size_bytes = 0`) or a stale one. A multi-file
    /// download keeps its declared sum, since one file's size says nothing
    /// about the rest. And a total never moves once bytes are counted, so the
    /// bar cannot rewind.
    pub(crate) fn file_size_announced(&self, file_bytes: u64) {
        if self.artifact_count != 1
            || file_bytes == 0
            || self.reported_bytes.load(Ordering::Relaxed) > 0
        {
            return;
        }
        if self.total_bytes.swap(file_bytes, Ordering::Relaxed) != file_bytes {
            // Forced, so the host can show the size right away.
            self.emit(true);
        }
    }

    /// Report bytes written so far **for the artifact currently in flight**.
    /// Throttled.
    pub(crate) fn file_bytes(&self, current_file_bytes: u64) {
        let total = self.completed_bytes.load(Ordering::Relaxed) + current_file_bytes;
        self.reported_bytes.fetch_max(total, Ordering::Relaxed);
        self.emit(false);
    }

    /// Fold a finished artifact into the running total and force an update.
    pub(crate) fn finish_file(&self, file_bytes: u64) {
        let total = self
            .completed_bytes
            .fetch_add(file_bytes, Ordering::Relaxed)
            + file_bytes;
        self.reported_bytes.fetch_max(total, Ordering::Relaxed);
        // Keep the coarse signal advancing too. It is ignored while a byte
        // total is known, but when one artifact declares no size there is no
        // usable byte total at all — without this the bar would sit at zero
        // for the whole download even as `downloaded_bytes` climbed.
        self.advance_completed_files();
        self.emit(true);
    }

    /// Move the coarse completed-files fraction on by one artifact.
    fn advance_completed_files(&self) {
        if self.artifact_count == 0 {
            return;
        }
        let completed = self.completed_files.fetch_add(1, Ordering::Relaxed) + 1;
        self.set_fraction(completed as f32 / self.artifact_count as f32);
    }

    /// Set progress directly, for sources that expose no byte totals (Hugging
    /// Face: completed files over total files). Ignored once `total_bytes` is
    /// known, where the byte count is strictly better.
    pub(crate) fn set_fraction(&self, fraction: f32) {
        if self.total().is_some() {
            return;
        }
        let bp = (fraction.clamp(0.0, 1.0) * 10_000.0) as u32;
        self.fraction_bp
            .fetch_max(bp.min(MAX_IN_FLIGHT_PROGRESS_BP), Ordering::Relaxed);
        self.emit(false);
    }

    /// Snapshot without emitting.
    pub(crate) fn snapshot(&self) -> DownloadStatus {
        let downloaded = self.reported_bytes.load(Ordering::Relaxed);
        let total_bytes = self.total();
        let progress = match total_bytes {
            Some(total) => {
                let bp = ((downloaded as f64 / total as f64) * 10_000.0) as u32;
                bp.min(MAX_IN_FLIGHT_PROGRESS_BP) as f32 / 10_000.0
            }
            None => self.fraction_bp.load(Ordering::Relaxed) as f32 / 10_000.0,
        };
        DownloadStatus {
            state: DownloadState::Downloading,
            progress,
            downloaded_bytes: downloaded,
            total_bytes,
        }
    }

    /// Emit the terminal `Ready` update. Called once the last artifact is
    /// verified and extracted, and on a cache hit (nothing to transfer).
    pub(crate) fn finish(&self) {
        let downloaded = self.reported_bytes.load(Ordering::Relaxed);
        (self.sink)(DownloadStatus::ready(downloaded, self.total()));
    }

    fn emit(&self, force: bool) {
        if !force && !self.due() {
            return;
        }
        (self.sink)(self.snapshot());
    }

    /// Whether enough time has passed since the last emitted update.
    fn due(&self) -> bool {
        let now = Instant::now();
        let mut last = self.last_emit.lock().unwrap_or_else(|e| e.into_inner());
        match *last {
            Some(previous) if now.duration_since(previous) < MIN_EMIT_INTERVAL => false,
            _ => {
                *last = Some(now);
                true
            }
        }
    }
}

/// An in-process progress observer. Never crosses an FFI boundary — bindings
/// forward statuses into their own transport from inside one of these.
pub(crate) type DownloadObserver = Box<dyn Fn(DownloadStatus) + Send + Sync>;

/// Shared state between a [`ModelDownload`] handle and its worker thread.
#[derive(Debug)]
struct DownloadCell {
    status: Mutex<DownloadStatus>,
    /// Bumped on every status change so `next_status` can tell a real update
    /// from a spurious wake-up.
    revision: Mutex<u64>,
    changed: Condvar,
    error: Mutex<Option<String>>,
}

/// A model download running in the background, decoupled from loading.
///
/// Exists because every FFI binding's `load` call blocks: there is no object
/// to poll while the weights come down. Start the download, drive a bar off
/// it, then `load()` — which hits the cache and returns immediately.
///
/// Dropping the handle does **not** cancel the download; call
/// [`Self::cancel`] for that. A finished download leaves the model in the
/// normal SDK cache, so a later `load()` is offline-clean.
pub struct ModelDownload {
    cell: Arc<DownloadCell>,
    cancel: Arc<AtomicBool>,
    observers: Arc<Mutex<Vec<DownloadObserver>>>,
}

impl std::fmt::Debug for ModelDownload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelDownload")
            .field("status", &self.status())
            .finish_non_exhaustive()
    }
}

impl ModelDownload {
    /// Start downloading everything `source` needs, on a background thread.
    ///
    /// Returns immediately. Non-registry sources have nothing to fetch and
    /// come back already [`DownloadState::Ready`].
    pub(crate) fn spawn(source: ModelSource) -> Arc<Self> {
        let (id, platform) = match &source {
            ModelSource::Registry { id, platform } => (id.clone(), platform.clone()),
            // Bundles, directories and Hugging Face repos either need no
            // network or have no resolvable size to report against; there is
            // nothing this handle can usefully drive.
            _ => return Arc::new(Self::completed()),
        };

        let download = Arc::new(Self {
            cell: Arc::new(DownloadCell {
                status: Mutex::new(DownloadStatus {
                    state: DownloadState::Downloading,
                    progress: 0.0,
                    downloaded_bytes: 0,
                    total_bytes: None,
                }),
                revision: Mutex::new(0),
                changed: Condvar::new(),
                error: Mutex::new(None),
            }),
            cancel: Arc::new(AtomicBool::new(false)),
            observers: Arc::new(Mutex::new(Vec::new())),
        });

        let worker = Arc::clone(&download);
        let worker_id = id.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("xybrid-download-{id}"))
            .spawn(move || {
                let id = worker_id;
                let publisher = Arc::clone(&worker);
                let outcome = RegistryClient::from_env().and_then(|client| {
                    client.fetch_extracted_cancellable(
                        &id,
                        platform.as_deref(),
                        Arc::clone(&worker.cancel),
                        move |status| publisher.publish(status),
                    )
                });
                match outcome {
                    Ok(_) => worker.publish_terminal(DownloadState::Ready, None),
                    Err(SdkError::Cancelled { message }) => {
                        worker.publish_terminal(DownloadState::Cancelled, Some(message))
                    }
                    Err(err) => {
                        worker.publish_terminal(DownloadState::Failed, Some(err.to_string()))
                    }
                }
            });

        if let Err(err) = spawned {
            // No thread means no download will ever land: fail loudly instead
            // of leaving a bar parked at zero forever.
            log::error!("failed to spawn download thread for '{id}': {err}");
            download.publish_terminal(DownloadState::Failed, Some(err.to_string()));
        }

        download
    }

    /// A handle for something already on disk: `Ready` from the first read.
    fn completed() -> Self {
        Self {
            cell: Arc::new(DownloadCell {
                status: Mutex::new(DownloadStatus::ready(0, None)),
                revision: Mutex::new(1),
                changed: Condvar::new(),
                error: Mutex::new(None),
            }),
            cancel: Arc::new(AtomicBool::new(false)),
            observers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Current snapshot. Never blocks.
    pub fn status(&self) -> DownloadStatus {
        *self.cell.status.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the download reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.status().state != DownloadState::Downloading
    }

    /// The failure message, once the download ended in
    /// [`DownloadState::Failed`] or [`DownloadState::Cancelled`].
    pub fn error(&self) -> Option<String> {
        self.cell
            .error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Block until the status changes, `timeout` elapses, or the download
    /// finishes, then return the current snapshot.
    ///
    /// Returns immediately once the download is terminal, so a `loop` driven
    /// by this cannot hang. Call it off the UI thread.
    ///
    /// # Errors
    ///
    /// Returns the download's own error once it has failed, so a `?`-driven
    /// loop exits on the real cause rather than spinning on a `Failed` state.
    pub fn next_status(&self, timeout: Duration) -> SdkResult<DownloadStatus> {
        let start = *self.cell.revision.lock().unwrap_or_else(|e| e.into_inner());
        let status = self.wait_for_change(start, timeout);
        match status.state {
            DownloadState::Failed => Err(SdkError::network(
                self.error()
                    .unwrap_or_else(|| "model download failed".to_string()),
            )),
            DownloadState::Cancelled => Err(SdkError::Cancelled {
                message: self
                    .error()
                    .unwrap_or_else(|| "download cancelled by caller".to_string()),
            }),
            _ => Ok(status),
        }
    }

    /// [`Self::next_status`] without the error mapping — the snapshot alone,
    /// terminal states included. What the FFI layers forward.
    pub fn next_status_snapshot(&self, timeout: Duration) -> DownloadStatus {
        let start = *self.cell.revision.lock().unwrap_or_else(|e| e.into_inner());
        self.wait_for_change(start, timeout)
    }

    /// Block until the download reaches a terminal state or `timeout` elapses.
    pub fn wait(&self, timeout: Duration) -> DownloadStatus {
        let deadline = Instant::now().checked_add(timeout);
        loop {
            let status = self.status();
            if status.state != DownloadState::Downloading {
                return status;
            }
            let remaining = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                // `Instant + Duration` saturates to `None` past the platform's
                // representable range, and the timeout arrives as an
                // unvalidated `u64` from the bindings. Poll in bounded slices
                // rather than aborting the process.
                None => Duration::from_secs(3600),
            };
            if remaining.is_zero() {
                return status;
            }
            self.next_status_snapshot(remaining.min(Duration::from_secs(3600)));
        }
    }

    /// Register an in-process observer, invoked on every status update.
    ///
    /// Rust-side only: the closure never crosses an FFI boundary. Bindings
    /// use it to feed their own transport (a BoltFFI stream, a
    /// flutter_rust_bridge sink). The current snapshot is delivered
    /// immediately, so a late subscriber is never left without a first frame.
    pub fn watch<F>(&self, observer: F)
    where
        F: Fn(DownloadStatus) + Send + Sync + 'static,
    {
        let observer: DownloadObserver = Box::new(observer);
        // The observer lock is taken *before* the status is read, and
        // [`Self::publish`] delivers its terminal frame and clears the list
        // under that same lock. Reading the status first would leave a window
        // where a download finishing right now publishes to an empty list and
        // clears it, after which this registration is never notified again —
        // the host's stream would then stay open forever.
        let mut observers = self.observers.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = self.status();
        observer(snapshot);
        // A download that already finished gets no further updates, so keeping
        // the observer would leak it for the handle's lifetime.
        if snapshot.state == DownloadState::Downloading {
            observers.push(observer);
        }
    }

    /// Ask the download to stop. Takes effect within one chunk read; the
    /// partial file is discarded so a later attempt starts clean.
    ///
    /// Idempotent, and a no-op once the download is terminal.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    fn wait_for_change(&self, since: u64, timeout: Duration) -> DownloadStatus {
        {
            let status = self.status();
            if status.state != DownloadState::Downloading {
                return status;
            }
        }
        let revision = self.cell.revision.lock().unwrap_or_else(|e| e.into_inner());
        if *revision != since {
            drop(revision);
            return self.status();
        }
        let (_guard, _) = self
            .cell
            .changed
            .wait_timeout(revision, timeout)
            .unwrap_or_else(|e| e.into_inner());
        drop(_guard);
        self.status()
    }

    /// Record a status and wake every waiter + observer.
    fn publish(&self, status: DownloadStatus) {
        {
            let mut current = self.cell.status.lock().unwrap_or_else(|e| e.into_inner());
            // Terminal is final: a late in-flight update from the worker must
            // not walk a finished bar back to `Downloading`.
            if current.state != DownloadState::Downloading {
                return;
            }
            *current = status;
        }
        {
            let mut revision = self.cell.revision.lock().unwrap_or_else(|e| e.into_inner());
            *revision += 1;
        }
        self.cell.changed.notify_all();
        let mut observers = self.observers.lock().unwrap_or_else(|e| e.into_inner());
        for observer in observers.iter() {
            observer(status);
        }
        if status.state != DownloadState::Downloading {
            // Terminal: nothing further will be emitted, so release the host
            // closures here rather than holding them for the handle's
            // lifetime. Done under the same lock `watch` registers through, so
            // an observer arriving concurrently either lands before this and
            // receives the frame, or lands after and reads the terminal status
            // itself.
            observers.clear();
        }
    }

    fn publish_terminal(&self, state: DownloadState, error: Option<String>) {
        if let Some(message) = error {
            *self.cell.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(message);
        }
        let last = self.status();
        let terminal = match state {
            DownloadState::Ready => DownloadStatus::ready(last.downloaded_bytes, last.total_bytes),
            other => DownloadStatus::terminal(other, &last),
        };
        self.publish(terminal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    type Recorder = Box<dyn Fn(DownloadStatus)>;

    /// A reporter writing every emitted status into a shared vector. The sink
    /// is returned alongside because the reporter only borrows it.
    fn recording_reporter() -> (Recorder, Arc<Mutex<Vec<DownloadStatus>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let recorder: Recorder = Box::new(move |status| {
            sink.lock().unwrap_or_else(|e| e.into_inner()).push(status);
        });
        (recorder, seen)
    }

    fn reporter_over<'a>(total: Option<u64>, sink: &'a Recorder) -> ProgressReporter<'a> {
        reporter_over_files(total, 0, sink)
    }

    fn reporter_over_files<'a>(
        total: Option<u64>,
        artifact_count: usize,
        sink: &'a Recorder,
    ) -> ProgressReporter<'a> {
        ProgressReporter::new(
            total,
            artifact_count,
            Arc::new(AtomicBool::new(false)),
            sink.as_ref(),
        )
    }

    #[test]
    fn progress_aggregates_across_artifacts() {
        // Two files, 100 bytes total: finishing the first must read 50%, not
        // 100% — the multi-file reset that used to snap the bar back.
        let (sink, seen) = recording_reporter();
        let reporter = reporter_over(Some(100), &sink);
        reporter.finish_file(50);
        assert!((seen.lock().unwrap().last().unwrap().progress - 0.5).abs() < 1e-3);

        reporter.finish_file(50);
        let last = *seen.lock().unwrap().last().unwrap();
        assert_eq!(last.downloaded_bytes, 100);
        // Still short of 1.0: verification and extraction have not run yet.
        assert!(last.progress < 1.0, "got {}", last.progress);
        assert_eq!(last.state, DownloadState::Downloading);
    }

    #[test]
    fn progress_never_rewinds_on_retry() {
        let (sink, seen) = recording_reporter();
        let reporter = reporter_over(Some(1_000), &sink);
        reporter.file_bytes(400);
        let peak = seen.lock().unwrap().last().unwrap().downloaded_bytes;
        assert_eq!(peak, 400);

        // A retry restarts the file at zero. The bar must stall, not rewind.
        std::thread::sleep(MIN_EMIT_INTERVAL);
        reporter.file_bytes(10);
        let after = *seen.lock().unwrap().last().unwrap();
        assert_eq!(after.downloaded_bytes, 400);
        assert!((after.progress - 0.4).abs() < 1e-3);
    }

    #[test]
    fn updates_are_throttled() {
        let (sink, seen) = recording_reporter();
        let reporter = reporter_over(Some(1_000_000), &sink);
        for chunk in 1..=1_000u64 {
            reporter.file_bytes(chunk * 8_192);
        }
        let count = seen.lock().unwrap().len();
        assert!(
            count <= 3,
            "1000 chunks inside one throttle window emitted {count} updates"
        );
    }

    #[test]
    fn unknown_total_reports_bytes_and_coarse_fraction() {
        let (sink, seen) = recording_reporter();
        let reporter = reporter_over(None, &sink);
        reporter.finish_file(4_096);
        reporter.set_fraction(0.5);
        let last = *seen.lock().unwrap().last().unwrap();
        assert_eq!(last.total_bytes, None);
        assert_eq!(last.downloaded_bytes, 4_096);
        assert!((last.progress - 0.5).abs() < 1e-3, "got {}", last.progress);
    }

    #[test]
    fn an_undeclared_artifact_size_falls_back_to_a_file_count_bar() {
        // A vision model whose projector omits a size has no usable byte
        // total, but it must still get a moving bar rather than a flat zero
        // until the terminal frame.
        let (sink, seen) = recording_reporter();
        let reporter = reporter_over_files(None, 2, &sink);

        reporter.finish_file(1_000);
        let after_first = *seen.lock().unwrap().last().unwrap();
        assert!(
            (after_first.progress - 0.5).abs() < 1e-3,
            "expected a half-way file-count bar, got {}",
            after_first.progress
        );
        assert_eq!(after_first.downloaded_bytes, 1_000);
        assert_eq!(after_first.total_bytes, None);

        reporter.finish_file(500);
        let after_second = *seen.lock().unwrap().last().unwrap();
        assert!(
            after_second.progress < 1.0,
            "in-flight progress must stay below 1.0, got {}",
            after_second.progress
        );
        assert_eq!(after_second.downloaded_bytes, 1_500);
    }

    #[test]
    fn byte_totals_outrank_the_coarse_fraction() {
        let (sink, seen) = recording_reporter();
        let reporter = reporter_over(Some(1_000), &sink);
        reporter.finish_file(100);
        reporter.set_fraction(0.9);
        let last = *seen.lock().unwrap().last().unwrap();
        assert!((last.progress - 0.1).abs() < 1e-3, "got {}", last.progress);
    }

    #[test]
    fn a_single_file_download_adopts_the_announced_size() {
        // A registry entry with `size_bytes = 0` used to leave the bar at zero
        // until the terminal frame. The server's announced size fills it in.
        let (sink, seen) = recording_reporter();
        let reporter = reporter_over_files(None, 1, &sink);
        reporter.file_size_announced(1_000);
        let announced = *seen
            .lock()
            .unwrap()
            .last()
            .expect("adopting a size must emit it");
        assert_eq!(announced.total_bytes, Some(1_000));
        assert_eq!(announced.downloaded_bytes, 0);

        reporter.file_bytes(250);
        let status = reporter.snapshot();
        assert!(
            (status.progress - 0.25).abs() < 1e-3,
            "got {}",
            status.progress
        );
    }

    #[test]
    fn an_announced_size_replaces_a_stale_declared_one() {
        let (sink, _seen) = recording_reporter();
        let reporter = reporter_over_files(Some(900), 1, &sink);
        reporter.file_size_announced(1_000);
        assert_eq!(reporter.snapshot().total_bytes, Some(1_000));
    }

    #[test]
    fn an_announced_size_is_ignored_across_files_and_once_bytes_are_counted() {
        // One file's size says nothing about a multi-file total.
        let (sink, _seen) = recording_reporter();
        let multi = reporter_over_files(None, 2, &sink);
        multi.file_size_announced(1_000);
        assert_eq!(multi.snapshot().total_bytes, None);

        // A resumed retry announces again mid-download. Moving the total then
        // would make the bar jump or rewind.
        let single = reporter_over_files(Some(1_000), 1, &sink);
        single.file_bytes(600);
        single.file_size_announced(2_000);
        assert_eq!(single.snapshot().total_bytes, Some(1_000));
    }

    #[test]
    fn beginning_a_transfer_emits_a_zero_frame() {
        let (sink, seen) = recording_reporter();
        let reporter = reporter_over_files(None, 1, &sink);
        reporter.begin_transfer();
        let frames = seen.lock().unwrap().clone();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].state, DownloadState::Downloading);
        assert_eq!(frames[0].downloaded_bytes, 0);
        assert_eq!(frames[0].progress, 0.0);
    }

    #[test]
    fn finish_emits_ready_at_one() {
        let (sink, seen) = recording_reporter();
        let reporter = reporter_over(Some(2_048), &sink);
        reporter.finish_file(2_048);
        reporter.finish();
        let last = *seen.lock().unwrap().last().unwrap();
        assert_eq!(last.state, DownloadState::Ready);
        assert_eq!(last.progress, 1.0);
        assert_eq!(last.downloaded_bytes, 2_048);
    }

    #[test]
    fn cancel_flag_is_observed_by_the_reporter() {
        let cancel = Arc::new(AtomicBool::new(false));
        let sink: Recorder = Box::new(|_| {});
        let reporter = ProgressReporter::new(None, 0, Arc::clone(&cancel), sink.as_ref());
        assert!(!reporter.is_cancelled());
        cancel.store(true, Ordering::Relaxed);
        assert!(reporter.is_cancelled());
    }

    #[test]
    fn non_registry_sources_report_ready_immediately() {
        let download = ModelDownload::spawn(ModelSource::Directory {
            path: "/tmp/does-not-need-downloading".into(),
        });
        let status = download.status();
        assert_eq!(status.state, DownloadState::Ready);
        assert_eq!(status.progress, 1.0);
        assert!(download.is_finished());
    }

    #[test]
    fn watch_delivers_a_first_frame_then_stops_after_terminal() {
        let download = ModelDownload::spawn(ModelSource::Directory {
            path: "/tmp/does-not-need-downloading".into(),
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        download.watch(move |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        // One immediate snapshot, and no retained observer for a download that
        // is already terminal.
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(download
            .observers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty());
    }

    /// A handle parked in `Downloading`, so a test can drive the terminal
    /// transition by hand rather than through a real registry fetch.
    fn pending_download() -> ModelDownload {
        ModelDownload {
            cell: Arc::new(DownloadCell {
                status: Mutex::new(DownloadStatus::default()),
                revision: Mutex::new(0),
                changed: Condvar::new(),
                error: Mutex::new(None),
            }),
            cancel: Arc::new(AtomicBool::new(false)),
            observers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[test]
    fn watch_holds_the_observer_lock_across_its_status_read() {
        // This ordering is the whole fix for a dropped terminal frame.
        // `publish` delivers the terminal status and clears the list under the
        // observer lock, so `watch` must hold that lock before it reads the
        // status. Reading first leaves a window where a download finishing
        // right now publishes to an empty list and clears it, after which the
        // observer registered a moment later is never notified — the host's
        // `for await` / `collect` loop then never closes and the `load()`
        // after it never runs.
        //
        // Asserted by holding the lock and proving `watch` cannot deliver its
        // first frame until it is released.
        let download = Arc::new(pending_download());
        let guard = download.observers.lock().unwrap_or_else(|e| e.into_inner());

        let subscriber = Arc::clone(&download);
        let delivered = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&delivered);
        let watching = std::thread::spawn(move || {
            subscriber.watch(move |_| flag.store(true, Ordering::Relaxed));
        });

        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !delivered.load(Ordering::Relaxed),
            "watch delivered a frame without holding the observer lock"
        );

        drop(guard);
        watching.join().expect("watching thread panicked");
        assert!(
            delivered.load(Ordering::Relaxed),
            "watch never delivered its first frame"
        );
    }

    #[test]
    fn publish_delivers_the_terminal_frame_then_drops_the_observers() {
        let download = pending_download();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        download.watch(move |status| {
            sink.lock().unwrap_or_else(|e| e.into_inner()).push(status);
        });

        download.publish_terminal(DownloadState::Ready, None);

        let frames = seen.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert_eq!(frames.len(), 2, "expected a first frame and a terminal one");
        assert_eq!(frames[0].state, DownloadState::Downloading);
        assert_eq!(frames[1].state, DownloadState::Ready);
        assert!(
            download
                .observers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty(),
            "host closures retained past the terminal frame"
        );
    }

    #[test]
    fn terminal_frame_never_rewinds_the_measured_byte_count() {
        // A stale registry size under-declares what actually landed. Settling
        // the terminal frame on the declared total would walk the bar
        // backwards at the very last update.
        let overshot = DownloadStatus::ready(120, Some(100));
        assert_eq!(overshot.downloaded_bytes, 120);
        assert_eq!(overshot.total_bytes, Some(100));
        assert_eq!(overshot.progress, 1.0);

        // The other direction still settles on the total: the last in-flight
        // update may have been dropped by the throttle.
        let throttled = DownloadStatus::ready(90, Some(100));
        assert_eq!(throttled.downloaded_bytes, 100);

        // No declared total: report exactly what was measured.
        let unknown = DownloadStatus::ready(64, None);
        assert_eq!(unknown.downloaded_bytes, 64);
    }

    #[test]
    fn next_status_returns_at_once_when_terminal() {
        let download = ModelDownload::spawn(ModelSource::Directory {
            path: "/tmp/does-not-need-downloading".into(),
        });
        let started = Instant::now();
        let status = download
            .next_status(Duration::from_secs(30))
            .expect("a ready download is not an error");
        assert_eq!(status.state, DownloadState::Ready);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "next_status blocked on a finished download"
        );
    }
}
