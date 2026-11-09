//! Shared pieces of the observability benchmarks: local OTLP/HTTP endpoints
//! (one that acknowledges every request at once, one that never answers),
//! representative GenAI content with the CONTRACT.md preflight cost, process
//! RSS, percentiles and the JSON side reports written next to Criterion's.
//!
//! Every bench binary includes this module and uses a different subset.
#![expect(
    dead_code,
    reason = "each bench binary compiles this module and uses a different subset of it"
)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use xybrid_observability::{InputMessage, OutputMessage, Part};

// ---------------------------------------------------------------------------
// Local endpoints
// ---------------------------------------------------------------------------

/// How a local endpoint treats requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Behavior {
    /// Reads each request and answers `200` with an empty body at once
    /// (keep-alive); the body is discarded.
    Blackhole,
    /// Accepts connections and reads what arrives, never answers: every
    /// export waits until its own deadline or the pipeline's shutdown.
    Blocked,
}

/// A local OTLP/HTTP endpoint on `127.0.0.1`, stopped on drop.
pub struct Endpoint {
    pub url: String,
    pub behavior: Behavior,
    requests: Arc<AtomicU64>,
    bytes: Arc<AtomicU64>,
    connections: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Endpoint {
    pub fn start(behavior: Behavior) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a local port");
        let url = format!("http://{}", listener.local_addr().expect("local address"));
        listener
            .set_nonblocking(true)
            .expect("non-blocking listener");
        let requests = Arc::new(AtomicU64::new(0));
        let bytes = Arc::new(AtomicU64::new(0));
        let connections = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let shared = (
            requests.clone(),
            bytes.clone(),
            connections.clone(),
            stop.clone(),
        );
        let thread = thread::spawn(move || {
            let (requests, bytes, connections, stop) = shared;
            let mut handlers = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((socket, _)) => {
                        // BSD and macOS hand out accepted sockets with the
                        // listener's O_NONBLOCK (Linux does not). On a
                        // non-blocking socket the read timeout in `serve` has
                        // no effect and every handler would spin a core.
                        socket.set_nonblocking(false).expect(
                            "make the accepted socket blocking so its read timeout applies",
                        );
                        connections.fetch_add(1, Ordering::Relaxed);
                        let counters = (requests.clone(), bytes.clone(), stop.clone());
                        handlers.push(thread::spawn(move || {
                            let (requests, bytes, stop) = counters;
                            serve(socket, behavior, &requests, &bytes, &stop);
                        }));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => thread::sleep(Duration::from_millis(1)),
                }
            }
            for handler in handlers {
                let _ = handler.join();
            }
        });
        Self {
            url,
            behavior,
            requests,
            bytes,
            connections,
            stop,
            thread: Some(thread),
        }
    }

    /// Requests fully read (both behaviors).
    pub fn requests(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    /// Request body bytes read.
    pub fn body_bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Connections accepted.
    pub fn connections(&self) -> u64 {
        self.connections.load(Ordering::Relaxed)
    }

    pub fn summary(&self) -> Value {
        json!({
            "behavior": format!("{:?}", self.behavior),
            "requests": self.requests(),
            "body_bytes": self.body_bytes(),
            "connections": self.connections(),
        })
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Reads requests off one connection: `Content-Length` framed (the
/// exporter always sends a sized body). A blackhole answers each; a blocked
/// endpoint keeps reading until the client goes away or the server stops.
fn serve(
    mut socket: TcpStream,
    behavior: Behavior,
    requests: &AtomicU64,
    bytes: &AtomicU64,
    stop: &AtomicBool,
) {
    let _ = socket.set_read_timeout(Some(Duration::from_millis(20)));
    let _ = socket.set_nodelay(true);
    let mut buffer: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        // Parse as many complete requests as the buffer holds.
        while let Some(end) = find_header_end(&buffer) {
            let length = content_length(&buffer[..end]);
            if buffer.len() < end + 4 + length {
                break;
            }
            buffer.drain(..end + 4 + length);
            requests.fetch_add(1, Ordering::Relaxed);
            bytes.fetch_add(length as u64, Ordering::Relaxed);
            if behavior == Behavior::Blackhole {
                let response = b"HTTP/1.1 200 OK\r\ncontent-type: application/x-protobuf\r\ncontent-length: 0\r\n\r\n";
                if socket.write_all(response).is_err() {
                    return;
                }
            }
        }
        match socket.read(&mut chunk) {
            Ok(0) => return,
            Ok(n) => buffer.extend_from_slice(&chunk[..n]),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => return,
        }
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|w| w == b"\r\n\r\n")
}

fn content_length(head: &[u8]) -> usize {
    String::from_utf8_lossy(head)
        .split("\r\n")
        .filter_map(|line| line.split_once(':'))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.trim().parse().ok())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Representative content
// ---------------------------------------------------------------------------

/// Prose with the kinds of strings redaction rules target (an e-mail address,
/// a phone number, an API-key prefix), repeated to any size. ASCII, so every
/// byte count below is also a character count.
const PROSE: &str = "Lisbon sits on seven hills above the Tagus estuary, and its \
yellow trams climb past tiled facades towards the castle. Contact \
user@example.com about order 4411 or call +1-555-0100 before noon. The deploy \
key sk-live-0000 must never appear in logs. Pastelarias open early; the \
miradouros fill at sunset. ";

/// `bytes` bytes of representative prose.
pub fn text(bytes: usize) -> String {
    let mut out = String::with_capacity(bytes + PROSE.len());
    while out.len() < bytes {
        out.push_str(PROSE);
    }
    out.truncate(bytes);
    out
}

/// A system prompt of `bytes` bytes as message parts.
pub fn system(bytes: usize) -> Vec<Part> {
    vec![Part::text(text(bytes))]
}

/// One user message of `bytes` bytes.
pub fn single_message(bytes: usize) -> Vec<InputMessage> {
    vec![InputMessage::text("user", text(bytes))]
}

/// A chat history of `messages` messages, alternating `user`/`assistant`,
/// `bytes_each` bytes of text each.
pub fn history(messages: usize, bytes_each: usize) -> Vec<InputMessage> {
    (0..messages)
        .map(|i| {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            InputMessage::text(role, text(bytes_each))
        })
        .collect()
}

/// An assistant answer of `bytes` bytes.
pub fn answer(bytes: usize) -> Vec<OutputMessage> {
    vec![OutputMessage::assistant_text(text(bytes), Some("stop"))]
}

/// Preflight cost of the canonical JSON form (CONTRACT.md "Content
/// pipeline", `preflight_node_bytes` = 16). The benches use it to place
/// workloads just under or just over `max_traversal_bytes`.
pub fn preflight_cost(value: &Value) -> u64 {
    const NODE: u64 = 16;
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => NODE,
        Value::String(s) => NODE + s.len() as u64,
        Value::Array(items) => NODE + items.iter().map(preflight_cost).sum::<u64>(),
        Value::Object(map) => {
            NODE + map
                .iter()
                .map(|(k, v)| NODE + k.len() as u64 + preflight_cost(v))
                .sum::<u64>()
        }
    }
}

/// Preflight cost of any serializable content value.
pub fn cost_of<T: serde::Serialize + ?Sized>(value: &T) -> u64 {
    preflight_cost(&serde_json::to_value(value).expect("content serializes"))
}

/// Compact JSON length (what the 16 KiB attribute budget is measured on).
pub fn json_len<T: serde::Serialize + ?Sized>(value: &T) -> usize {
    serde_json::to_vec(value).expect("content serializes").len()
}

/// The largest history of `bytes_each`-byte messages whose preflight cost
/// stays within `budget` (the 480 KiB traversal budget by default).
pub fn history_within(budget: u64, bytes_each: usize) -> Vec<InputMessage> {
    let one = cost_of(&history(1, bytes_each)) - 16;
    let assistant_extra = cost_of(&history(2, bytes_each)) - 16 - 2 * one;
    // cost(n) = 16 + n * one + floor(n / 2) * (assistant role is longer).
    let mut n = ((budget - 16) / one) as usize;
    while n > 0 && 16 + n as u64 * one + (n / 2) as u64 * assistant_extra > budget {
        n -= 1;
    }
    let messages = history(n, bytes_each);
    debug_assert!(cost_of(&messages) <= budget);
    messages
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Nearest-rank percentile of sorted samples (`q` in 0..=1).
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Distribution summary of samples, in their own unit.
pub fn distribution(mut samples: Vec<f64>) -> Value {
    samples.retain(|v| v.is_finite());
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite samples"));
    if samples.is_empty() {
        return json!({ "n": 0 });
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    json!({
        "n": samples.len(),
        "min": samples[0],
        "p50": percentile(&samples, 0.50),
        "p90": percentile(&samples, 0.90),
        "p95": percentile(&samples, 0.95),
        "p99": percentile(&samples, 0.99),
        "p999": percentile(&samples, 0.999),
        "max": samples[samples.len() - 1],
        "mean": mean,
    })
}

/// Nanoseconds of a duration as `f64`.
pub fn nanos(d: Duration) -> f64 {
    d.as_secs_f64() * 1e9
}

// ---------------------------------------------------------------------------
// Process memory
// ---------------------------------------------------------------------------

/// Current resident set size of this process: `/proc/self/status` on Linux,
/// `proc_pidinfo(PROC_PIDTASKINFO)` on macOS, `None` elsewhere. Neither spawns
/// a process, so sampling every few milliseconds inside a measured burst
/// costs one read or one system call.
pub fn rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        status_kib("VmRSS:").map(|kib| kib * 1024)
    }
    #[cfg(target_os = "macos")]
    {
        macos::resident_bytes()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Peak resident set size of this process so far: Linux `VmHWM`, macOS
/// `getrusage` `ru_maxrss`; `None` elsewhere, where the sampled maximum
/// stands in.
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        status_kib("VmHWM:").map(|kib| kib * 1024)
    }
    #[cfg(target_os = "macos")]
    {
        macos::peak_resident_bytes()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Process memory through libc on macOS, where std exposes neither value.
#[cfg(target_os = "macos")]
mod macos {
    use std::mem::{size_of, MaybeUninit};

    /// `pti_resident_size` of this task.
    pub fn resident_bytes() -> Option<u64> {
        let mut info = MaybeUninit::<libc::proc_taskinfo>::zeroed();
        let size = libc::c_int::try_from(size_of::<libc::proc_taskinfo>()).ok()?;
        // SAFETY: `info` points to `size` writable bytes, the buffer size
        // passed; PROC_PIDTASKINFO writes one `proc_taskinfo` into it and
        // returns the number of bytes written.
        let written = unsafe {
            libc::proc_pidinfo(
                libc::getpid(),
                libc::PROC_PIDTASKINFO,
                0,
                info.as_mut_ptr().cast(),
                size,
            )
        };
        if written != size {
            return None;
        }
        // SAFETY: `proc_taskinfo` is plain integers, so the zeroed value is
        // valid, and the call above filled every byte of it.
        Some(unsafe { info.assume_init() }.pti_resident_size)
    }

    /// `ru_maxrss` of this process: bytes on macOS (kilobytes on Linux).
    pub fn peak_resident_bytes() -> Option<u64> {
        let mut usage = MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: `usage` points to one writable `rusage`, which getrusage
        // fills when it returns 0.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: `rusage` is plain integers (zeroed is valid) and getrusage
        // succeeded.
        u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()
    }
}

#[cfg(target_os = "linux")]
fn status_kib(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|l| l.starts_with(field))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// Samples a value on a background thread and keeps the maximum.
pub struct MaxSampler {
    stop: Arc<AtomicBool>,
    max: Arc<Mutex<Option<u64>>>,
    samples: Arc<AtomicU64>,
    thread: Option<thread::JoinHandle<()>>,
}

impl MaxSampler {
    pub fn start(every: Duration, mut read: impl FnMut() -> Option<u64> + Send + 'static) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let max: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
        let samples = Arc::new(AtomicU64::new(0));
        let shared = (stop.clone(), max.clone(), samples.clone());
        let thread = thread::spawn(move || {
            let (stop, max, samples) = shared;
            loop {
                if let Some(v) = read() {
                    let mut m = max.lock().expect("sampler lock");
                    *m = Some(m.map_or(v, |old: u64| old.max(v)));
                    samples.fetch_add(1, Ordering::Relaxed);
                }
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                thread::sleep(every);
            }
        });
        Self {
            stop,
            max,
            samples,
            thread: Some(thread),
        }
    }

    /// Stops sampling (after one last sample) and returns the maximum and
    /// the number of samples taken.
    pub fn finish(mut self) -> (Option<u64>, u64) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let max = *self.max.lock().expect("sampler lock");
        (max, self.samples.load(Ordering::Relaxed))
    }
}

/// How often RSS is sampled during a burst: one read or system call per
/// sample on Linux and macOS (`rss_bytes`), so every 2 ms.
pub fn rss_sample_interval() -> Duration {
    Duration::from_millis(2)
}

// ---------------------------------------------------------------------------
// Side reports
// ---------------------------------------------------------------------------

/// Whether Criterion runs in full benchmark mode (`cargo bench`), as opposed
/// to `cargo test --benches` (each benchmark once) or `--list`.
pub fn bench_mode() -> bool {
    std::env::args().any(|a| a == "--bench")
}

/// Whether the arguments only ask for the benchmark list.
pub fn list_only() -> bool {
    std::env::args().any(|a| a == "--list")
}

/// Whether Criterion's `--quick` was passed (side reports shrink too).
pub fn quick() -> bool {
    std::env::args().any(|a| a == "--quick")
}

/// Directory of the JSON side reports: `XYBRID_OBS_BENCH_DIR`, else
/// `<target>/observability-bench`.
pub fn report_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("XYBRID_OBS_BENCH_DIR") {
        return PathBuf::from(dir);
    }
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    tmp.parent()
        .map_or(tmp.clone(), |target| target.join("observability-bench"))
}

/// Environment block of every side report.
pub fn environment() -> Value {
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "available_parallelism": thread::available_parallelism().map(|n| n.get()).ok(),
        "profile": if cfg!(debug_assertions) { "debug-assertions" } else { "optimized" },
        "crate_version": env!("CARGO_PKG_VERSION"),
        "unix_time_s": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    })
}

/// Writes `report` as `<report_dir>/<name>.json` and returns the path.
pub fn write_report(name: &str, report: &Value) -> PathBuf {
    let dir = report_dir();
    std::fs::create_dir_all(&dir).expect("create the report directory");
    let path = dir.join(format!("{name}.json"));
    let body = serde_json::to_vec_pretty(report).expect("report serializes");
    std::fs::write(&path, body).expect("write the report");
    eprintln!("side report: {}", path.display());
    path
}

/// Pipeline counters as JSON, per signal.
#[cfg(feature = "otlp")]
pub fn stats_json(stats: &xybrid_observability::otlp::PipelineStats) -> Value {
    let signal = |s: &xybrid_observability::otlp::SignalStats| {
        json!({
            "accepted": s.accepted,
            "exported": s.exported,
            "dropped": s.dropped,
            "failed": s.failed,
            "pending": s.pending,
            "consistent": s.is_consistent(),
        })
    };
    json!({
        "traces": signal(&stats.traces),
        "logs": signal(&stats.logs),
        "metrics": signal(&stats.metrics),
    })
}

/// Times `f` once.
pub fn timed<R>(f: impl FnOnce() -> R) -> (R, Duration) {
    let start = Instant::now();
    let r = f();
    (r, start.elapsed())
}
