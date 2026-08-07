//! Live (rolling-window) ASR bindings for Flutter.
//!
//! Wraps the SDK's [`xybrid_sdk::XybridStream`] so a Dart caller can feed
//! microphone PCM and receive partial transcripts as a `Stream`. A session is
//! created from an already-loaded model via `FfiModel::stream` — the model's
//! on-disk location is resolved by the SDK, so the same registry / Hugging Face
//! / bundle / directory model you loaded for batch inference streams with no
//! extra path wrangling. Audio flows *in* via [`FfiStreamSession::feed`];
//! partial transcripts flow *out* via the sink from [`FfiStreamSession::subscribe`].
//!
//! Backend (Whisper / Wav2Vec2) is auto-detected from the model metadata.
//!
//! # Threading model
//!
//! A single worker thread owns the `XybridStream` for its whole lifetime.
//! Commands reach it over a [`tokio::sync::mpsc`] channel, so they are applied
//! in submission order — audio fed in order is transcribed in order. [`feed`]
//! is a cheap, non-blocking channel send (`#[frb(sync)]`); the heavy inference
//! runs on the worker, never on the Dart isolate, so the UI never stalls.
//! (Rolling-window chunking itself lives entirely in `xybrid-core`; this
//! binding is a transport — see the module docs of `xybrid_core::streaming`.)
//!
//! # Audio contract
//!
//! Samples are PCM **f32, mono, 16 kHz**. Conversion from the platform mic
//! format is the caller's responsibility (kept out of FFI deliberately).
//!
//! [`feed`]: FfiStreamSession::feed
//! [`subscribe`]: FfiStreamSession::subscribe

use std::sync::Once;

use flutter_rust_bridge::frb;
use tokio::sync::{mpsc, oneshot};
use xybrid_sdk::{PartialResult as SdkPartialResult, StreamConfig, XybridStream};

use crate::frb_generated::StreamSink;

/// 16 kHz mono — the only sample rate the ASR backends accept.
pub const REQUIRED_SAMPLE_RATE: u32 = 16_000;

static LOG_INIT: Once = Once::new();

/// Install a `log` backend and a panic hook on first use. Idempotent.
///
/// Without this the binding registers no logger, so on Android every `log::*`
/// line across the whole Rust stack is dropped and a panic on a worker thread
/// dies silently. This makes both visible in `logcat` (tag `xybrid`).
fn ensure_logging() {
    LOG_INIT.call_once(|| {
        #[cfg(target_os = "android")]
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(log::LevelFilter::Debug)
                .with_tag("xybrid"),
        );
        // Surface panics that would otherwise vanish on a detached worker thread.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            log::error!("xybrid RUST PANIC: {info}");
            prev(info);
        }));
    });
}

/// How voice-activity detection (VAD) chunking is resolved for a session.
///
/// Models one decision in one type instead of a `bool` + `Option<String>`
/// pair where illegal states (disabled, yet a path is set) are representable.
#[derive(Debug, Clone)]
pub enum FfiVadMode {
    /// Fixed time-window chunking; no voice-activity detection.
    Off,
    /// VAD on, using the bundled default Silero model.
    Default,
    /// VAD on, using a Silero model from this directory.
    Custom {
        /// Directory containing the VAD model.
        model_dir: String,
    },
}

/// Configuration for a live ASR session.
///
/// The model itself is not named here — it comes from the loaded `FfiModel`
/// you call `stream` on. This only configures *how* the audio is chunked.
#[derive(Debug, Clone)]
pub struct FfiStreamingConfig {
    /// PCM sample rate of the audio you will feed. Must be 16 kHz; validated
    /// (and asserted as the contract) rather than forwarded — the backends are
    /// fixed at 16 kHz.
    pub sample_rate: u32,
    /// Voice-activity-detection mode.
    pub vad: FfiVadMode,
    /// Optional language hint (e.g. `"en"`); `None` uses the model default.
    pub language: Option<String>,
}

impl FfiStreamingConfig {
    /// Validate and convert to the SDK's [`StreamConfig`].
    ///
    /// `pub(crate)` so `FfiModel::stream` can build the SDK request without
    /// re-exposing the SDK type at the Dart boundary.
    pub(crate) fn to_sdk(&self) -> Result<StreamConfig, String> {
        if self.sample_rate != REQUIRED_SAMPLE_RATE {
            return Err(format!(
                "sample_rate must be {REQUIRED_SAMPLE_RATE} Hz, got {}",
                self.sample_rate
            ));
        }
        let (enable_vad, vad_model_dir) = match &self.vad {
            FfiVadMode::Off => (false, None),
            FfiVadMode::Default => (true, None),
            FfiVadMode::Custom { model_dir } => (true, Some(model_dir.clone())),
        };
        Ok(StreamConfig {
            enable_vad,
            vad_model_dir,
            language: self.language.clone(),
            ..StreamConfig::default()
        })
    }
}

/// A partial transcript emitted while audio is streaming.
#[derive(Debug, Clone)]
pub struct FfiPartialResult {
    /// Best-effort transcript text so far.
    pub text: String,
    /// `true` once this span is committed and will not change.
    pub is_stable: bool,
    /// Monotonic chunk sequence number this result corresponds to.
    pub chunk_sequence: u64,
    /// Audio covered so far, in milliseconds.
    pub audio_duration_ms: u64,
}

impl From<SdkPartialResult> for FfiPartialResult {
    fn from(p: SdkPartialResult) -> Self {
        Self {
            text: p.text,
            is_stable: p.is_stable,
            chunk_sequence: p.chunk_index,
            audio_duration_ms: p.audio_duration_ms,
        }
    }
}

/// Commands applied, in order, by the session's worker thread.
enum Command {
    /// Register (or replace) the sink that partial transcripts flow into.
    SetSink(StreamSink<FfiPartialResult>),
    /// Feed PCM samples; inference may run and emit a partial.
    Feed(Vec<f32>),
    /// Finalize: drain remaining audio and reply with the full transcript.
    Flush(oneshot::Sender<Result<String, String>>),
    /// Reset for fresh audio without reloading the model.
    Reset(oneshot::Sender<Result<(), String>>),
}

/// A live ASR session usable from Dart.
///
/// Internally just the sending end of the worker's command channel; the
/// `XybridStream` (which transitively owns tokio/executor/model state — none of
/// it portable across a DLL boundary) lives only on the worker thread and never
/// crosses to Dart. Dropping this handle closes the channel, which ends the
/// worker and drops the stream on its own thread.
#[frb(opaque)]
pub struct FfiStreamSession {
    cmd_tx: mpsc::UnboundedSender<Command>,
}

impl FfiStreamSession {
    /// Spawn the worker thread that owns `stream` and start accepting commands.
    ///
    /// `pub(crate)`: not an FFI entry point. Callers reach this through
    /// `FfiModel::stream`, which resolves the model directory for us.
    pub(crate) fn spawn(stream: XybridStream) -> Self {
        ensure_logging();
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<Command>();
        // The model is already loaded inside `stream`, so the worker only owns
        // and drives it — no model load happens here.
        std::thread::Builder::new()
            .name("xybrid-asr".to_string())
            .spawn(move || worker_loop(stream, cmd_rx))
            .expect("failed to spawn xybrid ASR worker thread");
        Self { cmd_tx }
    }

    /// Subscribe to partial transcripts. Call this once, before [`feed`].
    ///
    /// Partials are delivered on this sink as rolling-window chunks complete.
    /// Audio fed before the subscription is processed will not be reported.
    ///
    /// [`feed`]: Self::feed
    pub fn subscribe(&self, sink: StreamSink<FfiPartialResult>) {
        // If the worker is gone the send fails; the Dart stream simply ends.
        let _ = self.cmd_tx.send(Command::SetSink(sink));
    }

    /// Feed PCM f32 mono 16 kHz samples.
    ///
    /// A cheap, ordered, non-blocking channel send — inference happens on the
    /// worker thread. Takes an owned `Vec` because the samples are moved across
    /// the thread boundary; frb hands us an owned buffer already, so this adds
    /// no copy beyond the unavoidable cross-thread handoff.
    ///
    /// # Errors
    ///
    /// If the session has been finalized (after [`flush`]) or otherwise torn
    /// down, so the worker is no longer accepting audio.
    ///
    /// [`flush`]: Self::flush
    #[frb(sync)]
    pub fn feed(&self, samples: Vec<f32>) -> Result<(), String> {
        self.cmd_tx
            .send(Command::Feed(samples))
            .map_err(|_| worker_gone())
    }

    /// Finalize: drain buffered audio and return the complete transcript.
    ///
    /// After this the session is finalized; further [`feed`] calls error.
    ///
    /// # Errors
    ///
    /// If finalization fails in the core, or the worker is already gone.
    ///
    /// [`feed`]: Self::feed
    pub async fn flush(&self) -> Result<String, String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .send(Command::Flush(reply_tx))
            .map_err(|_| worker_gone())?;
        reply_rx.await.map_err(|_| worker_gone())?
    }

    /// Reset the session to transcribe fresh audio without reloading the model.
    ///
    /// # Errors
    ///
    /// If the reset fails in the core, or the worker is already gone.
    pub async fn reset(&self) -> Result<(), String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .send(Command::Reset(reply_tx))
            .map_err(|_| worker_gone())?;
        reply_rx.await.map_err(|_| worker_gone())?
    }
}

/// The error returned when a command cannot reach the worker thread.
fn worker_gone() -> String {
    "ASR session is no longer running; create a new session".to_string()
}

/// Render an error with its full `source()` chain.
///
/// SDK errors carry the root cause as `#[source]`, which `Display` alone
/// drops — "Feed failed" instead of "Feed failed: Inference error: dtype
/// mismatch …". Dart gets one string, so the chain has to be flattened here.
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut source = e.source();
    while let Some(cause) = source {
        out.push_str(": ");
        out.push_str(&cause.to_string());
        source = cause.source();
    }
    out
}

/// Owns the `XybridStream` and applies commands in order until the channel
/// closes (all senders dropped) or a `Flush` finalizes the session.
fn worker_loop(stream: XybridStream, mut cmd_rx: mpsc::UnboundedReceiver<Command>) {
    ensure_logging();
    log::debug!("ASR worker started");

    // Pay the model's cold-start cost (~5 s for whisper-tiny on a Pixel 8)
    // while the app is still spinning up the microphone, instead of on top
    // of the first visible partial. Feeds that arrive meanwhile just queue
    // on the command channel and drain against a warm model.
    let warmup_started = std::time::Instant::now();
    match stream.warmup() {
        Ok(()) => log::info!(
            "ASR model warmed up in {} ms",
            warmup_started.elapsed().as_millis()
        ),
        // Non-fatal: the first real chunk will retry the load and surface
        // any real failure through the partial-event error path.
        Err(e) => log::warn!("ASR warm-up failed (continuing cold): {}", error_chain(&e)),
    }

    let mut sink: Option<StreamSink<FfiPartialResult>> = None;
    // Latest partial produced before a sink is attached. `feed` is `#[frb(sync)]`
    // and fires immediately, but `subscribe` is async, so the first feeds can
    // reach the worker before `SetSink`. Partial text is cumulative, so holding
    // only the most recent partial loses nothing — it is flushed the instant the
    // sink registers. This closes that race; early transcripts are never dropped.
    let mut pending: Option<FfiPartialResult> = None;
    // The SDK returns its cached latest partial from *every* `feed` call, and
    // feeds queued up behind one inference all drain at once when it finishes —
    // without this, one chunk's partial would be delivered to Dart once per
    // queued feed.
    let mut last_sent_seq: Option<u64> = None;

    while let Some(cmd) = cmd_rx.blocking_recv() {
        match cmd {
            Command::SetSink(s) => {
                log::debug!("ASR sink attached");
                if let Some(p) = pending.take() {
                    last_sent_seq = Some(p.chunk_sequence);
                    let _ = s.add(p);
                }
                sink = Some(s);
            }
            Command::Feed(samples) => {
                // Inference happens here, on the worker; `feed` returns the
                // latest partial (if a chunk boundary was crossed).
                match stream.feed(&samples) {
                    Ok(Some(partial)) => {
                        let p = FfiPartialResult::from(partial);
                        if last_sent_seq == Some(p.chunk_sequence) {
                            continue;
                        }
                        log::debug!(
                            "ASR partial seq={} ({} chars)",
                            p.chunk_sequence,
                            p.text.len()
                        );
                        match sink.as_ref() {
                            Some(s) => {
                                last_sent_seq = Some(p.chunk_sequence);
                                let _ = s.add(p);
                            }
                            None => pending = Some(p),
                        }
                    }
                    Ok(None) => {}
                    Err(e) => log::warn!("ASR feed error: {}", error_chain(&e)),
                }
            }
            Command::Flush(reply) => {
                let result = stream
                    .flush()
                    .map(|r| r.text)
                    .map_err(|e| format!("ASR flush failed: {}", error_chain(&e)));
                let _ = reply.send(result);
                break; // session finalized; the worker ends here.
            }
            Command::Reset(reply) => {
                let result = stream.reset().map_err(|e| format!("ASR reset failed: {e}"));
                // A fresh utterance starts clean; drop any stale pending state.
                pending = None;
                last_sent_seq = None;
                let _ = reply.send(result);
            }
        }
    }

    log::debug!("ASR worker stopped");
}
