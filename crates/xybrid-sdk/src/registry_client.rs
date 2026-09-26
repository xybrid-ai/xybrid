//! Registry client for fetching models from registry.xybrid.dev.
//!
//! This module provides:
//! - `RegistryClient`: High-level API for model resolution and download
//! - Mask-based model lookup with platform resolution
//! - SHA256 hash verification
//! - Download progress callbacks
//! - Automatic retry with exponential backoff
//! - Circuit breaker for failing endpoints
//! - **Dual-endpoint failover** (primary: registry.xybrid.dev, fallback: r2.xybrid.dev)
//!
//! # Example
//!
//! ```no_run
//! # fn _example() -> Result<(), Box<dyn std::error::Error>> {
//! use xybrid_sdk::registry_client::RegistryClient;
//!
//! let client = RegistryClient::default_client()?;
//!
//! // List available models
//! let models = client.list_models()?;
//! for model in models {
//!     println!("{}: {} ({})", model.id, model.description, model.task);
//! }
//!
//! // Resolve a model for the current platform
//! let resolved = client.resolve("kokoro-82m", None)?;
//! println!("Download URL: {}", resolved.download_url);
//!
//! // Fetch and cache the bundle
//! let bundle_path = client.fetch("kokoro-82m", None, |status| {
//!     println!("Downloaded: {:.1}%", status.progress * 100.0);
//! })?;
//! # Ok(())
//! # }
//! ```

use crate::cache::CacheManager;
use crate::download::{DownloadStatus, ProgressReporter};
use crate::model::SdkError;
use crate::platform::current_platform;
use crate::source::detect_platform;
use crate::telemetry_optout::is_telemetry_opted_out;
use crate::{get_binding, DEFAULT_BINDING, SDK_VERSION};
use chrono::{DateTime, NaiveDateTime, Utc};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use xybrid_core::http::{CircuitBreaker, CircuitConfig, RetryPolicy};

/// How often a retry backoff wakes to check for cancellation.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(100);

pub const DEFAULT_REGISTRY_URL: &str = "https://registry.xybrid.dev";
pub const FALLBACK_REGISTRY_URL: &str = "https://r2.xybrid.dev";

pub use crate::cache::{CacheEntryInfo, CacheEntryLocation};

/// All registry URLs in priority order.
pub const REGISTRY_URLS: &[&str] = &[DEFAULT_REGISTRY_URL, FALLBACK_REGISTRY_URL];

/// HTTP header carrying anonymous Xybrid SDK client identity for registry calls.
///
/// Set on every metadata request unless [`is_telemetry_opted_out`] is true.
/// See `docs/telemetry/registry.md` for the full schema.
pub const CLIENT_HEADER_NAME: &str = "X-Xybrid-Client";

/// Build the value for the [`CLIENT_HEADER_NAME`] header.
///
/// Returns `None` when the user has opted out via `XYBRID_TELEMETRY_OPTOUT=1`.
/// Callers must skip setting the header when this returns `None`.
///
/// The `binding` argument is sanitized: if it contains any character outside
/// `[a-z0-9_-]`, or is empty, it is replaced with [`DEFAULT_BINDING`] to
/// prevent user-supplied junk from being smuggled into the header value.
///
/// # Format
///
/// `binding={b}; sdk_version={v}; core_version={cv}; platform={p}; backends={list}`
///
/// `backends` is the comma-separated, alphabetical output of
/// [`xybrid_core::features::enabled`].
pub fn build_client_header(binding: &str) -> Option<String> {
    build_client_header_with_optout(binding, is_telemetry_opted_out())
}

/// Pure helper underlying [`build_client_header`].
///
/// Takes the opt-out decision as a parameter so unit tests can exercise both
/// branches without depending on the process-global `OnceLock` cache that
/// [`is_telemetry_opted_out`] keeps.
fn build_client_header_with_optout(binding: &str, opted_out: bool) -> Option<String> {
    if opted_out {
        return None;
    }
    let safe_binding = sanitize_binding(binding);
    let backends = xybrid_core::features::enabled().join(",");
    Some(format!(
        "binding={}; sdk_version={}; core_version={}; platform={}; backends={}",
        safe_binding,
        SDK_VERSION,
        xybrid_core::VERSION,
        current_platform(),
        backends,
    ))
}

/// Header identifying which version of a file a partial download holds, so a
/// resumed request can prove it continues the same bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ResumeValidator {
    ETag(String),
    LastModified(String),
}

impl ResumeValidator {
    /// The strongest validator `response` carries. Weak ETags (`W/"…"`) are
    /// skipped: they promise equivalent content, not identical bytes, and
    /// splicing needs identical bytes.
    fn from_response(response: &ureq::Response) -> Option<Self> {
        if let Some(etag) = response.header("ETag").filter(|tag| !tag.starts_with("W/")) {
            return Some(Self::ETag(etag.to_string()));
        }
        response
            .header("Last-Modified")
            .map(|date| Self::LastModified(date.to_string()))
    }

    /// The header value, as sent back in `If-Range`.
    fn value(&self) -> &str {
        match self {
            Self::ETag(value) | Self::LastModified(value) => value,
        }
    }
}

/// What earlier requests left at a download's destination, and how to
/// continue it.
#[derive(Debug, Default)]
struct PartialFile {
    /// Proves the server still serves the same file. `None` means the bytes on
    /// disk cannot be continued, so the next request starts over.
    validator: Option<ResumeValidator>,
    /// The whole file's size, when the server announced it. Checked after
    /// every response: a `206` may legally cover less than was asked for.
    size: Option<u64>,
}

/// The current run of "network unreachable" errors during a download.
///
/// Kept apart from the retry loop so the patience rule can be tested without
/// waiting out real outages.
#[derive(Debug, Default)]
struct OfflineWindow {
    /// When the run began; `None` while the network is reachable.
    since: Option<Instant>,
    /// Retries taken during the run, which paces the backoff.
    retries: u32,
}

impl OfflineWindow {
    /// The network answered (with bytes, or even with an HTTP error), so the
    /// outage is over. The next one gets a full patience window of its own.
    fn end(&mut self) {
        *self = Self::default();
    }

    /// Record an unreachable-network error at `now`. Returns the retry number
    /// within this outage, or `None` once it has lasted `patience`.
    fn record(&mut self, now: Instant, patience: Duration) -> Option<u32> {
        let since = *self.since.get_or_insert(now);
        if now.saturating_duration_since(since) >= patience {
            return None;
        }
        self.retries += 1;
        Some(self.retries)
    }
}

/// Whether a `206 Partial Content` response continues `partial`: it starts
/// exactly at `offset` and serves the same file, at the same size.
fn continues_partial(response: &ureq::Response, offset: u64, partial: &PartialFile) -> bool {
    let Some((start, total)) = response
        .header("Content-Range")
        .and_then(parse_content_range)
    else {
        return false;
    };
    let same_size = match (total, partial.size) {
        (Some(total), Some(size)) => total == size,
        _ => true,
    };
    start == offset
        && same_size
        && partial.validator.is_some()
        && ResumeValidator::from_response(response) == partial.validator
}

/// Parse `Content-Range: bytes <start>-<end>/<total>` into `(start, total)`.
/// `total` is `None` when the server writes `*` (size unknown).
fn parse_content_range(value: &str) -> Option<(u64, Option<u64>)> {
    let range = value.trim().strip_prefix("bytes ")?;
    let (span, total) = range.split_once('/')?;
    let (start, _end) = span.split_once('-')?;
    let start = start.trim().parse().ok()?;
    let total = match total.trim() {
        "*" => None,
        total => Some(total.parse().ok()?),
    };
    Some((start, total))
}

/// Bytes already on disk at `path`; `0` when there is no file.
fn partial_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
}

/// Sum of every byte the registry says a resolved variant needs — the main
/// file plus each companion artifact (a VLM projector, a draft model, …).
///
/// This is what lets one progress bar span a multi-file model: the total is
/// known before the first request, so finishing file 1 of 2 reads 50% instead
/// of restarting the bar. Entries with no declared size contribute `0`, and a
/// total of `0` is treated as "unknown" by [`ProgressReporter`].
fn total_declared_bytes(resolved: &ResolvedVariant) -> u64 {
    // One undeclared size poisons the whole aggregate: the missing artifact's
    // bytes would push the running count past the "total", pin progress at the
    // in-flight ceiling for the rest of the download, and let the terminal
    // frame report fewer bytes than the bar already showed. Reporting the
    // total as unknown instead downgrades progress to the coarse signal and
    // keeps the byte count exact, which is the honest trade.
    if resolved.size_bytes == 0
        || resolved
            .artifacts
            .iter()
            .any(|artifact| artifact.size_bytes == 0)
    {
        return 0;
    }
    resolved
        .artifacts
        .iter()
        .map(|artifact| artifact.size_bytes)
        .fold(resolved.size_bytes, u64::saturating_add)
}

/// Classify a download URL into the canonical telemetry `source` label
/// emitted on `ModelDownload` events.
///
/// Recognised hosts:
/// - `r2.xybrid.dev` / `*.r2.dev` / `r2.cloudflarestorage.com` → `"r2"`
///   (Xybrid's Cloudflare R2 mirror, fronts both the registry-served
///   bundles and the fallback URL list).
/// - `huggingface.co` / `hf.co` → `"huggingface"` (direct HF pulls
///   used for passthrough variants where the registry forwards the
///   raw upstream URL).
///
/// Anything else passes through as `"other"` so cost attribution still
/// produces a labelled event when a future variant adds a new origin
/// — the analytics backend can then promote the new label into a
/// recognised category without dropping rows in the meantime.
fn classify_download_source(url: &str) -> &'static str {
    let lower = url.to_ascii_lowercase();
    if lower.contains("huggingface.co") || lower.contains("hf.co/") {
        "huggingface"
    } else if lower.contains("r2.xybrid.dev")
        || lower.contains("r2.cloudflarestorage.com")
        || lower.contains(".r2.dev")
    {
        "r2"
    } else {
        "other"
    }
}

fn sanitize_binding(binding: &str) -> &str {
    let valid = !binding.is_empty()
        && binding
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
    if valid {
        binding
    } else {
        DEFAULT_BINDING
    }
}

/// Connection timeout in milliseconds.
const CONNECT_TIMEOUT_MS: u64 = 5000;

/// Request timeout in milliseconds.
const REQUEST_TIMEOUT_MS: u64 = 15000;

/// Longest a model download may go without receiving a byte before the
/// attempt is dropped and retried from where it stopped.
///
/// A stall bound, not a transfer deadline. The old 5-minute limit covered the
/// whole body, so a model bigger than five minutes of the link's bandwidth
/// could never finish (229 MB needs about 6 Mbit/s). A slow link now takes as
/// long as it takes, while a connection that went silent (a Wi-Fi to cellular
/// handoff, a dropped NAT mapping) is abandoned within this window.
const DOWNLOAD_STALL_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a download that has already received bytes waits for the network
/// to come back before giving up.
///
/// While a device is offline every retry fails at once (the DNS lookup), so
/// counting those against the retry budget ended a download about seven
/// seconds into an outage: shorter than a Wi-Fi to cellular handoff, a lift or
/// a tunnel. A download that has not received a byte yet still fails fast, so
/// an app that starts offline hears about it right away.
const DOWNLOAD_OFFLINE_PATIENCE: Duration = Duration::from_secs(120);

/// Longest wait between two retries while offline, so a network that comes
/// back is picked up within seconds.
const DOWNLOAD_OFFLINE_RETRY_CAP: Duration = Duration::from_secs(10);

/// Default retry delay when a 429 response omits or mangles Retry-After.
const DEFAULT_RATE_LIMIT_RETRY_AFTER_SECS: u64 = 60;

/// Cap Retry-After to avoid a registry hint sleeping the client indefinitely.
const MAX_RATE_LIMIT_RETRY_AFTER_SECS: u64 = 5 * 60;

fn rate_limit_retry_after_secs(retry_after: Option<&str>) -> u64 {
    rate_limit_retry_after_secs_at(retry_after, Utc::now())
}

fn rate_limit_retry_after_secs_at(retry_after: Option<&str>, now: DateTime<Utc>) -> u64 {
    let Some(value) = retry_after.map(str::trim).filter(|value| !value.is_empty()) else {
        return DEFAULT_RATE_LIMIT_RETRY_AFTER_SECS;
    };

    if let Ok(seconds) = value.parse::<u64>() {
        return seconds.min(MAX_RATE_LIMIT_RETRY_AFTER_SECS);
    }

    parse_retry_after_http_date(value, now).unwrap_or(DEFAULT_RATE_LIMIT_RETRY_AFTER_SECS)
}

fn parse_retry_after_http_date(value: &str, now: DateTime<Utc>) -> Option<u64> {
    let parsed = DateTime::parse_from_rfc2822(value)
        .map(|date| date.with_timezone(&Utc))
        .or_else(|_| {
            NaiveDateTime::parse_from_str(value, "%a, %d %b %Y %H:%M:%S GMT")
                .map(|date| DateTime::<Utc>::from_naive_utc_and_offset(date, Utc))
        })
        .ok()?;

    let seconds = parsed.signed_duration_since(now).num_seconds().max(0) as u64;
    Some(seconds.min(MAX_RATE_LIMIT_RETRY_AFTER_SECS))
}

/// Registry client for model resolution and download.
pub struct RegistryClient {
    /// Registry URLs in priority order (primary first, then fallbacks)
    api_urls: Vec<String>,
    /// Cache manager for storing downloaded bundles
    cache: CacheManager,
    /// HTTP agent with timeouts configured
    agent: ureq::Agent,
    /// Circuit breakers for each registry URL
    circuits: Vec<Arc<CircuitBreaker>>,
    /// Retry policy for API calls
    retry_policy: RetryPolicy,
    /// Retry policy for model downloads. Only attempts that add no bytes
    /// count against it (see [`Self::download_with_progress`]).
    download_retry_policy: RetryPolicy,
    /// See [`DOWNLOAD_STALL_TIMEOUT`].
    download_stall_timeout: Duration,
    /// See [`DOWNLOAD_OFFLINE_PATIENCE`].
    download_offline_patience: Duration,
    /// Binding identifier reported via the `X-Xybrid-Client` header.
    binding: &'static str,
}

impl RegistryClient {
    /// Create a new registry client with the specified API URLs (primary first).
    ///
    /// The client picks up the process-global binding via [`get_binding`]
    /// (defaulting to [`DEFAULT_BINDING`] when unset). Per-instance overrides
    /// go through [`Self::with_binding`].
    pub fn new(api_urls: Vec<String>) -> Result<Self, SdkError> {
        if api_urls.is_empty() {
            return Err(SdkError::ConfigError(
                "No registry URLs provided".to_string(),
            ));
        }

        // Create HTTP agent with timeouts
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_millis(CONNECT_TIMEOUT_MS))
            .timeout(Duration::from_millis(REQUEST_TIMEOUT_MS))
            .build();

        let cache = CacheManager::new()?;

        // Create circuit breakers for each URL
        let circuits: Vec<Arc<CircuitBreaker>> = api_urls
            .iter()
            .map(|_| Arc::new(CircuitBreaker::new(CircuitConfig::default())))
            .collect();

        debug!(
            "RegistryClient created with {} URLs, cache_dir={}",
            api_urls.len(),
            cache.cache_dir().display()
        );

        Ok(Self {
            api_urls,
            cache,
            agent,
            circuits,
            retry_policy: RetryPolicy::default(),
            // Longer delays than API calls: downloads hit a CDN, not the registry.
            download_retry_policy: RetryPolicy::conservative(),
            download_stall_timeout: DOWNLOAD_STALL_TIMEOUT,
            download_offline_patience: DOWNLOAD_OFFLINE_PATIENCE,
            binding: get_binding(),
        })
    }

    /// Override the binding identifier reported via the `X-Xybrid-Client` header.
    ///
    /// Each platform binding (Flutter, Kotlin, Swift, Unity) calls this with
    /// its own identifier so registry calls are attributed correctly. Defaults
    /// to [`DEFAULT_BINDING`] when not set.
    pub fn with_binding(mut self, binding: &'static str) -> Self {
        self.binding = binding;
        self
    }

    /// Return the binding identifier this client reports.
    pub fn binding(&self) -> &'static str {
        self.binding
    }

    /// Apply the [`CLIENT_HEADER_NAME`] header to a request when telemetry is opted in.
    ///
    /// When [`is_telemetry_opted_out`] is true, returns the request unchanged so
    /// no header is set on the wire.
    fn apply_client_header(&self, req: ureq::Request) -> ureq::Request {
        self.apply_client_header_with_optout(req, is_telemetry_opted_out())
    }

    /// Same as [`Self::apply_client_header`] but takes the opt-out flag explicitly.
    ///
    /// Tests use this to exercise both branches without depending on the
    /// process-global `OnceLock` cache that [`is_telemetry_opted_out`] keeps.
    fn apply_client_header_with_optout(
        &self,
        req: ureq::Request,
        opted_out: bool,
    ) -> ureq::Request {
        match build_client_header_with_optout(self.binding, opted_out) {
            Some(value) => req.set(CLIENT_HEADER_NAME, &value),
            None => req,
        }
    }

    /// Create a new registry client with a single API URL.
    pub fn with_url(api_url: impl Into<String>) -> Result<Self, SdkError> {
        Self::new(vec![api_url.into()])
    }

    /// Create a registry client with default URLs (primary + fallback).
    pub fn default_client() -> Result<Self, SdkError> {
        Self::new(REGISTRY_URLS.iter().map(|s| s.to_string()).collect())
    }

    /// Create a registry client from environment variable or defaults.
    ///
    /// Checks `XYBRID_REGISTRY_URL` environment variable first.
    /// If set, uses only that URL. Otherwise uses default URLs with fallback.
    pub fn from_env() -> Result<Self, SdkError> {
        if let Ok(url) = std::env::var("XYBRID_REGISTRY_URL") {
            // User specified a custom URL, use only that
            Self::with_url(url)
        } else {
            // Use default URLs with fallback
            Self::default_client()
        }
    }

    /// Get the primary API URL.
    pub fn primary_url(&self) -> &str {
        &self.api_urls[0]
    }

    /// Check if any circuit breaker is allowing requests.
    pub fn is_circuit_open(&self) -> bool {
        self.circuits.iter().all(|c| c.is_open())
    }

    /// Reset all circuit breakers to closed state.
    pub fn reset_circuit(&self) {
        for circuit in &self.circuits {
            circuit.reset();
        }
    }

    /// List all available models in the registry.
    ///
    /// Tries primary URL first, falls back to secondary on failure.
    /// Automatically retries on transient failures and respects circuit breaker.
    pub fn list_models(&self) -> Result<Vec<ModelSummary>, SdkError> {
        self.execute_with_fallback(|api_url| {
            let url = format!("{}/v1/models", api_url);
            let req = self.apply_client_header(self.agent.get(&url));
            let response = req.call();
            self.handle_response(response, "list models")
        })
        .and_then(|response| {
            let list_response: ListModelsResponse = response
                .into_json()
                .map_err(|e| SdkError::network_src("Failed to parse response", e))?;
            Ok(list_response.models)
        })
    }

    /// Get detailed information about a specific model.
    ///
    /// Tries primary URL first, falls back to secondary on failure.
    /// Automatically retries on transient failures and respects circuit breaker.
    pub fn get_model(&self, mask: &str) -> Result<ModelDetail, SdkError> {
        self.execute_with_fallback(|api_url| {
            let url = format!("{}/v1/models/{}", api_url, mask);
            let req = self.apply_client_header(self.agent.get(&url));
            let response = req.call();
            self.handle_response_with_404(response, "get model", || {
                SdkError::ModelNotFound(format!("Model '{}' not found", mask))
            })
        })
        .and_then(|response| {
            response
                .into_json()
                .map_err(|e| SdkError::network_src("Failed to parse response", e))
        })
    }

    /// Resolve a model mask to the best variant for the given platform.
    ///
    /// If platform is None, auto-detects the current platform.
    /// Tries primary URL first, falls back to secondary on failure.
    /// Automatically retries on transient failures and respects circuit breaker.
    pub fn resolve(&self, mask: &str, platform: Option<&str>) -> Result<ResolvedVariant, SdkError> {
        let platform = platform.map(String::from).unwrap_or_else(detect_platform);

        self.execute_with_fallback(|api_url| {
            let url = format!(
                "{}/v1/models/{}/resolve?platform={}",
                api_url, mask, platform
            );
            let req = self.apply_client_header(self.agent.get(&url));
            let response = req.call();
            self.handle_response_with_404(response, "resolve model", || {
                SdkError::ModelNotFound(format!(
                    "Model '{}' not found or no compatible variant for platform '{}'",
                    mask, platform
                ))
            })
        })
        .and_then(|response| {
            let resolve_response: ResolveResponse = response
                .into_json()
                .map_err(|e| SdkError::network_src("Failed to parse response", e))?;
            Ok(resolve_response.resolved)
        })
    }

    /// Execute an operation with fallback to secondary URLs.
    ///
    /// Tries each URL in order until one succeeds or all fail.
    fn execute_with_fallback<T, F>(&self, mut operation: F) -> Result<T, SdkError>
    where
        F: FnMut(&str) -> Result<T, SdkError>,
    {
        let mut last_error: Option<SdkError> = None;

        for (idx, api_url) in self.api_urls.iter().enumerate() {
            let circuit = &self.circuits[idx];

            // Skip if circuit is open
            if !circuit.can_execute() {
                debug!("Skipping {} (circuit open)", api_url);
                continue;
            }

            match self.execute_with_retry_for_url(api_url, circuit, &mut operation) {
                Ok(result) => {
                    if idx > 0 {
                        info!("Request succeeded using fallback URL: {}", api_url);
                    }
                    return Ok(result);
                }
                Err(err) => {
                    // Don't try fallback for non-retryable errors (like 404)
                    if !err.is_retryable() {
                        return Err(err);
                    }
                    // warn, not debug: on mobile the default native log level is
                    // Info, and a registry endpoint failing over is exactly the
                    // signal needed to diagnose "models/metrics not working"
                    // reports from devices.
                    warn!("URL {} failed: {}, trying next", api_url, err);
                    last_error = Some(err);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| SdkError::network("All registry URLs failed or circuits open")))
    }

    /// Execute an operation with retry for a specific URL.
    fn execute_with_retry_for_url<T, F>(
        &self,
        api_url: &str,
        circuit: &Arc<CircuitBreaker>,
        operation: &mut F,
    ) -> Result<T, SdkError>
    where
        F: FnMut(&str) -> Result<T, SdkError>,
    {
        let mut last_error: Option<SdkError> = None;

        for attempt in 0..self.retry_policy.max_attempts {
            // Calculate delay for this attempt
            let delay = if let Some(ref err) = last_error {
                err.retry_after()
                    .unwrap_or_else(|| self.retry_policy.delay_for_attempt(attempt))
            } else {
                self.retry_policy.delay_for_attempt(attempt)
            };

            if !delay.is_zero() {
                std::thread::sleep(delay);
            }

            // Check circuit breaker again (might have opened)
            if !circuit.can_execute() {
                return Err(SdkError::CircuitOpen(format!(
                    "Circuit breaker open for {}",
                    api_url
                )));
            }

            match operation(api_url) {
                Ok(result) => {
                    circuit.record_success();
                    return Ok(result);
                }
                Err(err) => {
                    // Offline errors (DNS, connection refused, network I/O) are
                    // not the registry's fault — they represent local
                    // unreachability. Don't count them toward the failure
                    // threshold (otherwise the breaker opens for 30s and the
                    // user sees "circuit open" even after they come back
                    // online), and skip the retry loop within this URL since
                    // backoff won't help a DNS failure. Return immediately and
                    // let `execute_with_fallback` try the next URL.
                    if matches!(&err, SdkError::Offline { .. }) {
                        return Err(err);
                    }

                    circuit.record_failure();

                    // Check for rate limit (opens circuit immediately)
                    if let SdkError::RateLimited { .. } = &err {
                        circuit.record_rate_limited();
                    }

                    // Don't retry non-retryable errors
                    if !err.is_retryable() {
                        return Err(err);
                    }

                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            SdkError::network(format!("All retry attempts exhausted for {}", api_url))
        }))
    }

    /// Handle HTTP response, converting errors appropriately.
    fn handle_response(
        &self,
        response: Result<ureq::Response, ureq::Error>,
        operation: &str,
    ) -> Result<ureq::Response, SdkError> {
        match response {
            Ok(resp) => {
                if resp.status() == 200 {
                    Ok(resp)
                } else {
                    Err(self.response_status_to_error(&resp, operation))
                }
            }
            Err(e) => Err(self.ureq_error_to_sdk_error(e, operation)),
        }
    }

    /// Handle HTTP response with special 404 handling.
    fn handle_response_with_404<F>(
        &self,
        response: Result<ureq::Response, ureq::Error>,
        operation: &str,
        not_found_err: F,
    ) -> Result<ureq::Response, SdkError>
    where
        F: FnOnce() -> SdkError,
    {
        match response {
            Ok(resp) => {
                if resp.status() == 200 {
                    Ok(resp)
                } else if resp.status() == 404 {
                    Err(not_found_err())
                } else {
                    Err(self.response_status_to_error(&resp, operation))
                }
            }
            Err(ureq::Error::Status(404, _)) => Err(not_found_err()),
            Err(e) => Err(self.ureq_error_to_sdk_error(e, operation)),
        }
    }

    /// Convert an HTTP response status and headers to SdkError.
    fn response_status_to_error(&self, response: &ureq::Response, operation: &str) -> SdkError {
        self.status_to_error(response.status(), operation, response.header("Retry-After"))
    }

    /// Convert HTTP status code to SdkError.
    fn status_to_error(&self, status: u16, operation: &str, retry_after: Option<&str>) -> SdkError {
        match status {
            429 => SdkError::RateLimited {
                retry_after_secs: rate_limit_retry_after_secs(retry_after),
            },
            502..=504 => SdkError::network(format!(
                "Registry {} failed with status {} (server error)",
                operation, status
            )),
            400 | 401 | 403 | 422 => SdkError::ConfigError(format!(
                "Registry {} failed with status {} (client error)",
                operation, status
            )),
            _ => SdkError::network(format!("Registry {} returned status {}", operation, status)),
        }
    }

    /// Convert ureq error to SdkError.
    ///
    /// Transport-level failures (DNS, connection refused, low-level I/O) are
    /// reported as `SdkError::Offline` rather than `NetworkError`. They represent
    /// "the local machine cannot reach the registry" — not a registry-side
    /// problem — and the circuit breaker deliberately does not count them
    /// toward the failure threshold (see `execute_with_retry_for_url`).
    fn ureq_error_to_sdk_error(&self, error: ureq::Error, operation: &str) -> SdkError {
        match error {
            ureq::Error::Status(status, response) => {
                self.status_to_error(status, operation, response.header("Retry-After"))
            }
            ureq::Error::Transport(transport) => {
                let kind = transport.kind();
                match kind {
                    ureq::ErrorKind::Dns => SdkError::offline_src(
                        format!("Failed to {} (DNS resolution failed)", operation),
                        transport,
                    ),
                    ureq::ErrorKind::ConnectionFailed => SdkError::offline_src(
                        format!(
                            "Failed to {} (connection refused or host unreachable)",
                            operation
                        ),
                        transport,
                    ),
                    ureq::ErrorKind::Io => SdkError::offline_src(
                        format!("Failed to {} (network I/O error)", operation),
                        transport,
                    ),
                    _ => SdkError::network_src(format!("Failed to {}", operation), transport),
                }
            }
        }
    }

    /// Check if a model is cached locally.
    pub fn is_cached(&self, mask: &str, platform: Option<&str>) -> Result<bool, SdkError> {
        let resolved = self.resolve(mask, platform)?;
        let cache_path = self.get_cache_path(&resolved);

        if !cache_path.exists() {
            return Ok(false);
        }

        // Verify hash if available
        if !resolved.sha256.is_empty() {
            let hash = compute_sha256(&cache_path)?;
            Ok(hash == resolved.sha256)
        } else {
            Ok(true)
        }
    }

    /// Get the local cache path for a resolved variant.
    pub fn get_cache_path(&self, resolved: &ResolvedVariant) -> PathBuf {
        self.cache
            .registry_bundle_path(&resolved.hf_repo, &resolved.file)
    }

    /// Fetch a model bundle, downloading if not cached.
    ///
    /// Returns the path to the extracted bundle directory.
    ///
    /// # Arguments
    ///
    /// * `mask` - Model mask (e.g., "kokoro-82m")
    /// * `platform` - Target platform (None for auto-detect)
    /// * `progress_callback` - Receives a [`DownloadStatus`] (state, fraction,
    ///   bytes) roughly ten times a second while the transfer runs, then once
    ///   more with [`DownloadState::Ready`](crate::DownloadState::Ready).
    pub fn fetch<F>(
        &self,
        mask: &str,
        platform: Option<&str>,
        progress_callback: F,
    ) -> Result<PathBuf, SdkError>
    where
        F: Fn(DownloadStatus),
    {
        self.fetch_cancellable(
            mask,
            platform,
            Arc::new(AtomicBool::new(false)),
            progress_callback,
        )
    }

    /// [`Self::fetch`] with a caller-owned cancellation flag.
    ///
    /// Setting the flag stops the transfer within one chunk read and discards
    /// the partial file; the call returns [`SdkError::Cancelled`].
    pub fn fetch_cancellable<F>(
        &self,
        mask: &str,
        platform: Option<&str>,
        cancel: Arc<AtomicBool>,
        progress_callback: F,
    ) -> Result<PathBuf, SdkError>
    where
        F: Fn(DownloadStatus),
    {
        let resolved = self.resolve(mask, platform)?;
        let reporter = ProgressReporter::new(
            Some(total_declared_bytes(&resolved)),
            1 + resolved.artifacts.len(),
            cancel,
            &progress_callback,
        );
        let path = self.fetch_bundle(mask, &resolved, &reporter)?;
        // Hash verification and extraction run after the last byte, with no
        // chunk loop to check the flag — so re-check here rather than
        // announcing `Ready` for a download the caller already stopped. The
        // bytes stay cached: they are complete and verified, so discarding
        // them would only cost the next attempt.
        if reporter.is_cancelled() {
            return Err(ProgressReporter::cancelled_error());
        }
        reporter.finish();
        Ok(path)
    }

    /// Download (or reuse the cached) `.xyb` bundle for an already-resolved
    /// variant, reporting into `reporter`.
    fn fetch_bundle(
        &self,
        mask: &str,
        resolved: &ResolvedVariant,
        reporter: &ProgressReporter<'_>,
    ) -> Result<PathBuf, SdkError> {
        let cache_path = self.get_cache_path(resolved);

        debug!(
            "Cache check for '{}': path={}, exists={}, sha256_provided={}",
            mask,
            cache_path.display(),
            cache_path.exists(),
            !resolved.sha256.is_empty()
        );

        // Check if already cached with correct hash
        if cache_path.exists() && !resolved.sha256.is_empty() {
            // Try fast path: read cached hash from sidecar file
            let hash = match read_cached_hash(&cache_path) {
                Some(cached_hash) => {
                    debug!("Using cached hash for '{}'", mask);
                    cached_hash
                }
                None => {
                    // Fall back to computing hash (slow for large files)
                    debug!("Computing hash for '{}' (no cached hash found)", mask);
                    let computed = compute_sha256(&cache_path)?;
                    // Cache the hash for next time
                    write_cached_hash(&cache_path, &computed);
                    computed
                }
            };

            debug!(
                "Cache verification for '{}': expected={}, actual={}",
                mask, resolved.sha256, hash
            );
            if hash == resolved.sha256 {
                // Already cached and verified
                info!("Cache hit for '{}' at {}", mask, cache_path.display());
                return Ok(cache_path);
            }
            // Hash mismatch - re-download
            info!("Cache hash mismatch for '{}', re-downloading", mask);
            std::fs::remove_file(&cache_path).ok();
            remove_cached_hash(&cache_path);
        } else if cache_path.exists() {
            info!(
                "Cache exists for '{}' but no sha256 to verify, re-downloading",
                mask
            );
        } else {
            info!(
                "Cache miss for '{}', downloading to {}",
                mask,
                cache_path.display()
            );
        }

        // Create cache directory
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Download from HuggingFace
        info!("Downloading '{}' from {}", mask, resolved.download_url);
        // Time only the wallclock spent inside the download itself so
        // the emitted `ModelDownload.duration_ms` reflects bytes-on-the-
        // wire latency. Hash verification + cache extraction run after
        // this block and have their own (much cheaper) cost; conflating
        // them would smear the network signal that operators actually
        // care about for the cost dashboard.
        let download_started = Instant::now();
        self.download_with_progress(&resolved.download_url, &cache_path, reporter)?;
        let download_duration = download_started.elapsed();

        // Emit a ModelDownload telemetry event for cost accounting. Use
        // the actual on-disk size — `resolved.size_bytes` is the
        // registry-declared expected size, which can drift from what
        // landed if the upstream changed between resolve and fetch. The
        // helper honors XYBRID_TELEMETRY_OPTOUT internally.
        let bytes_downloaded = std::fs::metadata(&cache_path)
            .map(|m| m.len())
            .unwrap_or(resolved.size_bytes);
        reporter.finish_file(bytes_downloaded);
        crate::telemetry::publish_model_download(
            mask,
            bytes_downloaded,
            classify_download_source(&resolved.download_url),
            download_duration.as_millis().min(u32::MAX as u128) as u32,
        );

        // Verify hash and cache it for fast future lookups
        if !resolved.sha256.is_empty() {
            let hash = compute_sha256(&cache_path)?;
            if hash != resolved.sha256 {
                std::fs::remove_file(&cache_path).ok();
                return Err(SdkError::cache(format!(
                    "SHA256 mismatch: expected {}, got {}",
                    resolved.sha256, hash
                )));
            }
            // Cache the verified hash for instant verification next time
            write_cached_hash(&cache_path, &hash);
            info!(
                "Download complete for '{}', SHA256 verified, cached at {}",
                mask,
                cache_path.display()
            );
        } else {
            info!(
                "Download complete for '{}' (no SHA256 verification), cached at {}",
                mask,
                cache_path.display()
            );
        }

        Ok(cache_path)
    }

    /// Fetch a model bundle and extract it, returning the extracted directory path.
    ///
    /// This is the **preferred method** for fetching models, as it returns a ready-to-use
    /// directory containing the model files and `model_metadata.json`.
    ///
    /// Extraction is idempotent: if the bundle was already extracted, returns immediately.
    ///
    /// For **passthrough variants** (e.g., GGUF models hosted on external HuggingFace repos),
    /// the model file is downloaded directly and `model_metadata.json` is written from the
    /// registry response, skipping the .xyb bundle flow entirely.
    ///
    /// # Arguments
    ///
    /// * `mask` - Model mask (e.g., "kokoro-82m")
    /// * `platform` - Target platform (None for auto-detect)
    /// * `progress_callback` - Receives a [`DownloadStatus`] (state, fraction,
    ///   bytes) roughly ten times a second while the transfer runs, then once
    ///   more with [`DownloadState::Ready`](crate::DownloadState::Ready). The
    ///   fraction is aggregated across *every* artifact the model needs, so a
    ///   multi-file model drives one bar rather than restarting per file.
    ///
    /// # Returns
    ///
    /// Path to the extracted directory containing `model_metadata.json` and model files.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn _example() -> Result<(), Box<dyn std::error::Error>> {
    /// # use xybrid_sdk::RegistryClient;
    /// let client = RegistryClient::default_client()?;
    /// let model_dir = client.fetch_extracted("kokoro-82m", None, |status| {
    ///     println!(
    ///         "{} / {:?} bytes ({:.1}%)",
    ///         status.downloaded_bytes,
    ///         status.total_bytes,
    ///         status.progress * 100.0
    ///     );
    /// })?;
    ///
    /// // model_dir now contains model_metadata.json and all model files
    /// let metadata_path = model_dir.join("model_metadata.json");
    /// # Ok(())
    /// # }
    /// ```
    pub fn fetch_extracted<F>(
        &self,
        mask: &str,
        platform: Option<&str>,
        progress_callback: F,
    ) -> Result<PathBuf, SdkError>
    where
        F: Fn(DownloadStatus),
    {
        self.fetch_extracted_cancellable(
            mask,
            platform,
            Arc::new(AtomicBool::new(false)),
            progress_callback,
        )
    }

    /// [`Self::fetch_extracted`] with a caller-owned cancellation flag.
    ///
    /// Setting the flag stops the transfer within one chunk read and discards
    /// the partial file; the call returns [`SdkError::Cancelled`]. This is what
    /// backs [`ModelDownload::cancel`](crate::ModelDownload::cancel).
    pub fn fetch_extracted_cancellable<F>(
        &self,
        mask: &str,
        platform: Option<&str>,
        cancel: Arc<AtomicBool>,
        progress_callback: F,
    ) -> Result<PathBuf, SdkError>
    where
        F: Fn(DownloadStatus),
    {
        // Offline-first: if we already have an extracted copy locally, return it
        // immediately. This avoids hitting the network (and tripping the circuit
        // breaker) when the user is offline with a previously-downloaded model.
        if let Some(extract_dir) = self.resolve_offline(mask) {
            debug!(
                "Using locally extracted model '{}' at {} (skipping registry)",
                mask,
                extract_dir.display()
            );
            // Nothing to transfer, but a host driving a bar still needs the
            // terminal frame — otherwise a cached model looks stuck at 0%.
            progress_callback(DownloadStatus::ready(0, None));
            return Ok(extract_dir);
        }

        // Resolve first to check if passthrough
        let resolved = self.resolve(mask, platform)?;
        let reporter = ProgressReporter::new(
            Some(total_declared_bytes(&resolved)),
            1 + resolved.artifacts.len(),
            cancel,
            &progress_callback,
        );

        let extract_dir = if resolved.passthrough {
            // Passthrough: download raw model file directly, write metadata from registry
            self.fetch_passthrough(mask, &resolved, &reporter)
        } else {
            // Standard flow: download .xyb bundle, then extract
            let xyb_path = self.fetch_bundle(mask, &resolved, &reporter)?;
            self.cache.ensure_extracted(&xyb_path)
        }?;
        // Same reasoning as `fetch_cancellable`: the tail past the last byte
        // has no chunk loop, so a cancel landing there must not report `Ready`.
        if reporter.is_cancelled() {
            return Err(ProgressReporter::cancelled_error());
        }
        reporter.finish();
        Ok(extract_dir)
    }

    /// Fetch a passthrough model: download raw file directly and write metadata from registry.
    ///
    /// For passthrough variants, there is no .xyb bundle. The model file (e.g., a GGUF)
    /// is downloaded directly from the source HuggingFace repo, and `model_metadata.json`
    /// is written from the inline metadata in the registry response.
    fn fetch_passthrough(
        &self,
        mask: &str,
        resolved: &ResolvedVariant,
        reporter: &ProgressReporter<'_>,
    ) -> Result<PathBuf, SdkError> {
        let extract_dir = self.cache.extraction_dir(mask);
        let model_file_path = extract_dir.join(&resolved.file);
        let metadata_path = extract_dir.join("model_metadata.json");

        // Idempotency check: if model file + metadata exist, check cache validity
        if metadata_path.exists() {
            let mut cache_valid =
                self.passthrough_file_is_valid(&model_file_path, &resolved.sha256)?;
            for artifact in &resolved.artifacts {
                let artifact_path = extract_dir.join(&artifact.file);
                cache_valid &= self.passthrough_file_is_valid(&artifact_path, &artifact.sha256)?;
            }

            if cache_valid {
                if resolved.sha256.is_empty()
                    || resolved
                        .artifacts
                        .iter()
                        .any(|artifact| artifact.sha256.is_empty())
                {
                    warn!(
                        "Passthrough cache hit for '{}' (one or more files lack hash verification) at {}",
                        mask,
                        extract_dir.display()
                    );
                } else {
                    info!(
                        "Passthrough cache hit for '{}' at {}",
                        mask,
                        extract_dir.display()
                    );
                }
                return Ok(extract_dir);
            }
            info!(
                "Passthrough cache incomplete or hash mismatch for '{}', re-downloading",
                mask
            );
        }

        // Create extraction directory
        std::fs::create_dir_all(&extract_dir)
            .map_err(|e| SdkError::cache_src("Failed to create extraction directory", e))?;

        self.download_passthrough_file(
            mask,
            &resolved.file,
            &resolved.download_url,
            &model_file_path,
            resolved.size_bytes,
            reporter,
            &resolved.sha256,
        )?;

        // Note: download_passthrough_file already handles telemetry + SHA256 per file.
        // Download additional artifacts (e.g. mmproj for VLM models)
        for artifact in &resolved.artifacts {
            let artifact_path = extract_dir.join(&artifact.file);
            self.download_passthrough_file(
                mask,
                &artifact.file,
                &artifact.download_url,
                &artifact_path,
                artifact.size_bytes,
                reporter,
                &artifact.sha256,
            )?;
        }

        // Write model_metadata.json from registry response
        if let Some(ref metadata) = resolved.model_metadata {
            let metadata_json = serde_json::to_string_pretty(metadata)
                .map_err(|e| SdkError::cache_src("Failed to serialize model metadata", e))?;
            std::fs::write(&metadata_path, metadata_json)
                .map_err(|e| SdkError::cache_src("Failed to write model_metadata.json", e))?;
            info!(
                "Wrote model_metadata.json for passthrough model '{}' at {}",
                mask,
                metadata_path.display()
            );
        } else {
            return Err(SdkError::cache(format!(
                "Passthrough variant for '{}' has no model_metadata in registry response",
                mask
            )));
        }

        Ok(extract_dir)
    }

    fn passthrough_file_is_valid(
        &self,
        file_path: &PathBuf,
        expected_sha256: &str,
    ) -> Result<bool, SdkError> {
        if !file_path.exists() {
            return Ok(false);
        }
        if expected_sha256.is_empty() {
            return Ok(true);
        }
        if let Some(cached_hash) = read_cached_hash(file_path) {
            return Ok(cached_hash == expected_sha256);
        }
        let computed = compute_sha256(file_path)?;
        let matches = computed == expected_sha256;
        if matches {
            write_cached_hash(file_path, &computed);
        }
        Ok(matches)
    }

    fn download_passthrough_file(
        &self,
        mask: &str,
        file_name: &str,
        download_url: &str,
        dest: &PathBuf,
        size_bytes: u64,
        reporter: &ProgressReporter<'_>,
        expected_sha256: &str,
    ) -> Result<(), SdkError> {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        info!(
            "Passthrough download '{}' file '{}' from {}",
            mask, file_name, download_url
        );
        // Same reasoning as the standard fetch path: we time the
        // network transfer alone so the cost dashboard sees a clean
        // bytes-on-the-wire signal.
        let download_started = Instant::now();
        self.download_with_progress(download_url, dest, reporter)?;
        let download_duration = download_started.elapsed();

        let bytes_downloaded = std::fs::metadata(dest)
            .map(|m| m.len())
            .unwrap_or(size_bytes);
        reporter.finish_file(bytes_downloaded);
        crate::telemetry::publish_model_download(
            mask,
            bytes_downloaded,
            classify_download_source(download_url),
            download_duration.as_millis().min(u32::MAX as u128) as u32,
        );

        if !expected_sha256.is_empty() {
            let hash = compute_sha256(dest)?;
            if hash != expected_sha256 {
                std::fs::remove_file(dest).ok();
                return Err(SdkError::cache(format!(
                    "Passthrough SHA256 mismatch for '{}': expected {}, got {}",
                    file_name, expected_sha256, hash
                )));
            }
            write_cached_hash(dest, &hash);
            info!(
                "Passthrough SHA256 verified for '{}' file '{}'",
                mask, file_name
            );
        }

        Ok(())
    }

    /// Check if a model is already extracted and ready to use.
    ///
    /// Returns true if the model has been fetched AND extracted.
    pub fn is_extracted(&self, model_id: &str) -> bool {
        self.cache.is_extracted(model_id)
    }

    /// Get the extraction directory for a model.
    ///
    /// Note: This returns the path even if not yet extracted. Use `is_extracted()` to check.
    pub fn extraction_dir(&self, model_id: &str) -> PathBuf {
        self.cache.extraction_dir(model_id)
    }

    /// Try to locate a ready-to-use model in the local cache without touching the network.
    ///
    /// Returns the path to the extraction directory if a previously-extracted copy of
    /// the model exists. Returns `None` if the model has not been fetched and extracted
    /// on this machine.
    ///
    /// This is the fast path for offline operation. It never calls out to the network,
    /// never trips the circuit breaker, and is safe to call repeatedly. Callers should
    /// prefer this over `resolve()` + `fetch()` when they don't need to check for
    /// registry updates.
    pub fn resolve_offline(&self, mask: &str) -> Option<PathBuf> {
        self.cache.existing_extraction_dir(mask)
    }

    /// List all model IDs that are currently available for offline use.
    ///
    /// These are models that have been downloaded and extracted on this machine.
    /// Never touches the network. Useful for showing "what you can run right now"
    /// in offline error messages and in `xybrid models list` when the registry
    /// is unreachable.
    pub fn list_offline_models(&self) -> Vec<String> {
        self.cache.list_extracted_model_ids()
    }

    /// Download a file with progress tracking, resuming across retries.
    ///
    /// Note: Downloads use a separate retry mechanism because:
    /// 1. HuggingFace is a different endpoint than the registry API
    /// 2. Large file downloads need a stall timeout, not a request deadline
    /// 3. We don't want a failed HuggingFace download to trip the registry circuit breaker
    ///
    /// An interrupted attempt keeps its partial file, and the next one asks
    /// the server for only the missing bytes. Only attempts that add nothing
    /// to the file count against the retry policy, so a download that keeps
    /// moving on a flaky link is not killed after three drops. Once bytes have
    /// arrived, losing the network entirely does not count either: the
    /// download waits up to [`DOWNLOAD_OFFLINE_PATIENCE`] for it to return.
    /// The loop still ends: every retry that is not counted has either grown
    /// the file past its previous best (the file is finite) or falls inside
    /// that bounded wait, and a wait can only restart after the network
    /// answers, which either adds bytes or spends an attempt.
    ///
    /// On failure or cancellation the partial file is removed.
    fn download_with_progress(
        &self,
        url: &str,
        dest: &PathBuf,
        reporter: &ProgressReporter<'_>,
    ) -> Result<(), SdkError> {
        let result = self.download_with_retries(url, dest, reporter);
        if result.is_err() {
            std::fs::remove_file(dest).ok();
        }
        result
    }

    fn download_with_retries(
        &self,
        url: &str,
        dest: &PathBuf,
        reporter: &ProgressReporter<'_>,
    ) -> Result<(), SdkError> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_millis(CONNECT_TIMEOUT_MS))
            .timeout_read(self.download_stall_timeout)
            .timeout_write(self.download_stall_timeout)
            .build();
        let policy = &self.download_retry_policy;
        reporter.begin_transfer();

        let mut partial = PartialFile::default();
        let mut furthest: u64 = 0;
        let mut failed_attempts: u32 = 0;
        let mut offline = OfflineWindow::default();
        let mut next_delay: Option<Duration> = None;

        loop {
            if let Some(delay) = next_delay {
                // Sliced rather than one sleep: a server-supplied `Retry-After`
                // can run to tens of seconds, and a cancel arriving during it
                // would otherwise sit unobserved for that whole interval while
                // the download still reported `Downloading`.
                if !Self::sleep_unless_cancelled(delay, reporter) {
                    return Err(ProgressReporter::cancelled_error());
                }
            }

            let err = match self.try_download(&agent, url, dest, &mut partial, reporter) {
                Ok(()) => return Ok(()),
                Err(err) => err,
            };
            // `Cancelled` is non-retryable, so an aborted download leaves the
            // loop here rather than burning its attempts.
            if !err.is_retryable() {
                return Err(err);
            }
            let on_disk = partial_len(dest);
            let unreachable = matches!(err, SdkError::Offline { .. });
            if on_disk > furthest || !unreachable {
                // Bytes arrived, or the server answered with an error: either
                // way the network is back, so the outage is over.
                offline.end();
            }
            let delay = if on_disk > furthest {
                furthest = on_disk;
                policy.delay_for_attempt(1)
            } else if furthest > 0 && unreachable {
                // The network went away mid-download. Every lookup fails at
                // once until it returns, so wait for it (bounded) instead of
                // spending the retry budget on attempts that cannot succeed.
                let Some(retry) = offline.record(Instant::now(), self.download_offline_patience)
                else {
                    return Err(err);
                };
                policy
                    .delay_for_attempt(retry)
                    .min(DOWNLOAD_OFFLINE_RETRY_CAP)
            } else {
                failed_attempts += 1;
                if failed_attempts >= policy.max_attempts {
                    return Err(err);
                }
                policy.delay_for_attempt(failed_attempts)
            };
            warn!(
                "Download of {} interrupted at {} bytes, retrying: {}",
                url, on_disk, err
            );
            next_delay = Some(err.retry_after().unwrap_or(delay));
        }
    }

    /// Sleep for `delay`, waking every [`CANCEL_POLL_INTERVAL`] to check the
    /// cancellation flag. Returns `false` if the wait was cut short by a
    /// cancel.
    fn sleep_unless_cancelled(delay: Duration, reporter: &ProgressReporter<'_>) -> bool {
        let mut remaining = delay;
        while !remaining.is_zero() {
            if reporter.is_cancelled() {
                return false;
            }
            let slice = remaining.min(CANCEL_POLL_INTERVAL);
            std::thread::sleep(slice);
            remaining -= slice;
        }
        !reporter.is_cancelled()
    }

    /// Download `url` into `dest`, continuing the partial file there when the
    /// server proves it still serves the same one.
    ///
    /// `partial` describes the bytes on disk. It is set from the response that
    /// starts the file and reset whenever those bytes cannot be continued.
    /// Returns once the file is whole; a transfer that breaks off returns its
    /// error with `partial` ready for the next attempt to continue from.
    fn try_download(
        &self,
        agent: &ureq::Agent,
        url: &str,
        dest: &PathBuf,
        partial: &mut PartialFile,
        reporter: &ProgressReporter<'_>,
    ) -> Result<(), SdkError> {
        // One request per pass. A later pass either asks for the rest after a
        // resumed range came back short, or starts over once the server has
        // shown the partial file cannot be continued. Neither can repeat
        // forever: a short range must have added bytes to the file, and the
        // pass after a reset is a full request, which always returns.
        loop {
            if reporter.is_cancelled() {
                return Err(ProgressReporter::cancelled_error());
            }

            // Resume only with a validator from the response that wrote the
            // partial file. Without one there is no way to tell whether the
            // file changed on the server in between, and splicing two versions
            // corrupts the model silently when the registry has no SHA-256.
            let offset = match partial.validator {
                Some(_) => partial_len(dest),
                None => 0,
            };

            let mut request = agent.get(url);
            if offset > 0 {
                request = request.set("Range", &format!("bytes={offset}-"));
                if let Some(validator) = partial.validator.as_ref() {
                    request = request.set("If-Range", validator.value());
                }
            }

            let response = match request.call() {
                Ok(response) => response,
                // The partial is no longer a prefix of the server's file (it
                // shrank or was replaced). Start over.
                Err(ureq::Error::Status(416, _)) if offset > 0 => {
                    *partial = PartialFile::default();
                    continue;
                }
                Err(e) => return Err(self.ureq_error_to_sdk_error(e, "download bundle")),
            };

            let resuming = offset > 0 && response.status() == 206;
            if resuming && !continues_partial(&response, offset, partial) {
                // A range of some other file. Hugging Face's CDN ignores
                // `If-Range`, so this is how a file replaced mid-download
                // shows up. Start over.
                *partial = PartialFile::default();
                continue;
            }
            if response.status() != 200 && !resuming {
                return Err(self.response_status_to_error(&response, "download bundle"));
            }

            // The server's word on the whole file's size. The registry may
            // declare none, or a stale one.
            let file_size = if resuming {
                response
                    .header("Content-Range")
                    .and_then(parse_content_range)
                    .and_then(|(_, total)| total)
            } else {
                response
                    .header("Content-Length")
                    .and_then(|value| value.trim().parse().ok())
            };
            if resuming {
                partial.size = partial.size.or(file_size);
            } else {
                // A full body: whatever was on disk is replaced.
                *partial = PartialFile {
                    validator: ResumeValidator::from_response(&response),
                    size: file_size,
                };
            }
            if let Some(size) = partial.size {
                reporter.file_size_announced(size);
            }

            let (mut file, mut downloaded) = if resuming {
                info!("Resuming download of {} at byte {}", url, offset);
                let file = std::fs::OpenOptions::new().append(true).open(dest)?;
                (file, offset)
            } else {
                (File::create(dest)?, 0)
            };
            let mut reader = response.into_reader();
            let mut buffer = [0u8; 8192];

            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| SdkError::network_src("Read error", e))?;

                if bytes_read == 0 {
                    break;
                }

                file.write_all(&buffer[..bytes_read])?;
                downloaded += bytes_read as u64;

                // Report progress. The reporter throttles and aggregates; this
                // loop just says how many bytes of the current file landed.
                reporter.file_bytes(downloaded);

                // Checked per chunk so a cancel takes effect in milliseconds
                // rather than at the end of a multi-gigabyte file.
                if reporter.is_cancelled() {
                    return Err(ProgressReporter::cancelled_error());
                }
            }

            // With no announced size, the end of the body is all there is to
            // go on. With one, the file must match it exactly: ending early
            // would cache a truncated model as ready.
            let Some(size) = partial.size else {
                return Ok(());
            };
            if downloaded == size {
                return Ok(());
            }
            if downloaded > size {
                // More bytes than the file has: the partial is not the file
                // the server serves. The next attempt starts over.
                *partial = PartialFile::default();
                return Err(SdkError::network(format!(
                    "Download of {} overran its size: {} of {} bytes",
                    url, downloaded, size
                )));
            }
            if !resuming || downloaded == offset {
                // A full response that ended early (possible when chunked
                // framing overrides `Content-Length`), or a range that added
                // nothing. Hand it to the retry policy, which backs off and
                // gives up on attempts that make no headway. Continuing here
                // could spin: a server that ignores `Range` restarts the file
                // from zero on every pass.
                return Err(SdkError::network(format!(
                    "Download of {} stopped at {} of {} bytes",
                    url, downloaded, size
                )));
            }
            // The resumed range ended before the file did, which HTTP allows.
            // Ask for the rest.
            debug!(
                "Range for {} ended at {} of {} bytes, requesting the rest",
                url, downloaded, size
            );
        }
    }

    /// Clear the local cache for a specific model.
    ///
    /// # Returns
    ///
    /// The number of cache roots removed for `mask` across all managed cache
    /// areas (registry bundle, extracted runtime cache, HuggingFace
    /// downloads). Returns `0` when the model was not cached.
    ///
    /// # Concurrency
    ///
    /// Not safe to run concurrently with a load of the same model: it removes
    /// whole cache directories that an in-flight extraction may be writing to.
    pub fn clear_cache(&mut self, mask: &str) -> Result<u32, SdkError> {
        self.cache.clear_model(mask)
    }

    /// Clear the entire model cache.
    ///
    /// # Returns
    ///
    /// The number of cache roots removed across all managed cache areas.
    /// Returns `0` when nothing was cached.
    ///
    /// # Concurrency
    ///
    /// Not safe to run concurrently with any model load: it removes whole
    /// cache directories that in-flight downloads or extractions may be
    /// writing to.
    pub fn clear_all_cache(&mut self) -> Result<u32, SdkError> {
        self.cache.clear()
    }

    /// Get aggregate cache statistics across all managed model cache roots.
    pub fn cache_stats(&self) -> Result<CacheStats, SdkError> {
        let entries = self.cache_entries()?;
        let total_size = entries.iter().map(|entry| entry.size_bytes).sum();

        Ok(CacheStats {
            total_size_bytes: total_size,
            model_count: entries.len(),
            cache_path: self.cache.cache_dir().to_path_buf(),
        })
    }

    /// Return the root directory that owns all managed model cache locations.
    pub fn cache_root(&self) -> PathBuf {
        crate::cache::layout::CacheLayout::from_registry_root(self.cache.cache_dir().to_path_buf())
            .cache_root()
            .to_path_buf()
    }

    /// List cached model entries across all managed cache roots.
    ///
    /// Includes the legacy registry bundle cache and runtime caches such as
    /// extracted bundles and direct Hugging Face downloads.
    pub fn cache_entries(&self) -> Result<Vec<CacheEntryInfo>, SdkError> {
        self.cache.cache_entries()
    }
}

/// Compute SHA256 hash of a file.
fn compute_sha256(path: &PathBuf) -> Result<String, SdkError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Get the path to the cached hash sidecar file.
///
/// For a file like `model.xyb`, returns `model.xyb.sha256`.
/// For a file like `model.gguf`, returns `model.gguf.sha256`.
fn hash_cache_path(file_path: &PathBuf) -> PathBuf {
    let mut sidecar = file_path.as_os_str().to_os_string();
    sidecar.push(".sha256");
    PathBuf::from(sidecar)
}

/// Read cached hash from sidecar file if it exists and is still valid.
///
/// Returns None if:
/// - Sidecar file doesn't exist
/// - Sidecar file is older than the bundle file (bundle was modified)
/// - Sidecar file can't be read
fn read_cached_hash(bundle_path: &PathBuf) -> Option<String> {
    let hash_path = hash_cache_path(bundle_path);

    // Check if sidecar exists
    if !hash_path.exists() {
        return None;
    }

    // Check if bundle is newer than sidecar (invalidates cache)
    let bundle_mtime = std::fs::metadata(bundle_path).ok()?.modified().ok()?;
    let hash_mtime = std::fs::metadata(&hash_path).ok()?.modified().ok()?;
    if bundle_mtime > hash_mtime {
        // Bundle was modified after hash was cached
        return None;
    }

    // Read and validate hash format (64 hex chars)
    let hash = std::fs::read_to_string(&hash_path).ok()?;
    let hash = hash.trim();
    if hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(hash.to_string())
    } else {
        None
    }
}

/// Write hash to sidecar file for fast future lookups.
fn write_cached_hash(bundle_path: &PathBuf, hash: &str) {
    let hash_path = hash_cache_path(bundle_path);
    // Ignore errors - this is just an optimization
    let _ = std::fs::write(&hash_path, hash);
}

/// Remove the cached hash sidecar file.
fn remove_cached_hash(bundle_path: &PathBuf) {
    let hash_path = hash_cache_path(bundle_path);
    let _ = std::fs::remove_file(&hash_path);
}

// ============================================================================
// API Response Types
// ============================================================================

/// Response from GET /v1/models/registry
#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    models: Vec<ModelSummary>,
}

/// Summary of a model in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Model mask ID (e.g., "kokoro-82m")
    pub id: String,
    /// Model family (e.g., "hexgrad", "openai")
    pub family: String,
    /// Task type (e.g., "text-to-speech", "speech-recognition")
    pub task: String,
    /// Number of parameters
    pub parameters: u64,
    /// Human-readable description
    pub description: String,
    /// Available variants (e.g., ["universal"])
    pub variants: Vec<String>,
}

/// Detailed model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetail {
    /// Model mask ID
    pub id: String,
    /// Model family
    pub family: String,
    /// Task type
    pub task: String,
    /// Number of parameters
    pub parameters: u64,
    /// Description
    pub description: String,
    /// Default variant name
    pub default_variant: Option<String>,
    /// Available variants with details
    pub variants: HashMap<String, VariantInfo>,
}

/// Information about a model variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantInfo {
    /// Platform identifier
    pub platform: String,
    /// Model format (e.g., "onnx", "safetensors")
    pub format: String,
    /// Quantization level (e.g., "fp16", "fp32", "int8")
    pub quantization: String,
    /// Bundle size in bytes
    pub size_bytes: u64,
    /// HuggingFace repository
    pub hf_repo: String,
    /// Bundle filename
    pub file: String,
}

/// Response from GET /v1/models/registry/{mask}/resolve
#[derive(Debug, Deserialize)]
struct ResolveResponse {
    #[allow(dead_code)]
    mask: String,
    #[allow(dead_code)]
    platform: String,
    resolved: ResolvedVariant,
}

/// Resolved variant ready for download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedVariant {
    /// HuggingFace repository
    pub hf_repo: String,
    /// Bundle filename (or raw model filename for passthrough)
    pub file: String,
    /// Direct download URL
    pub download_url: String,
    /// Model format
    pub format: String,
    /// Quantization level
    pub quantization: String,
    /// Bundle size in bytes
    pub size_bytes: u64,
    /// SHA256 hash for verification
    pub sha256: String,
    /// Additional files required by this variant, such as VLM mmproj siblings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ResolvedArtifact>,
    /// Whether this is a passthrough variant (direct download, no .xyb bundle)
    #[serde(default)]
    pub passthrough: bool,
    /// Inline model_metadata.json for passthrough variants
    #[serde(default)]
    pub model_metadata: Option<serde_json::Value>,
}

/// Additional resolved artifact required alongside the primary variant file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedArtifact {
    /// Local file path relative to the extracted model directory.
    pub file: String,
    /// Direct download URL.
    pub download_url: String,
    /// Expected size in bytes.
    pub size_bytes: u64,
    /// Expected SHA256 hash. Empty means no hash verification is available.
    #[serde(default)]
    pub sha256: String,
}

/// Aggregate cache statistics across all managed model cache roots.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total size of cached model entries in bytes.
    pub total_size_bytes: u64,
    /// Number of cached model entries across managed cache roots.
    pub model_count: usize,
    /// Path to the legacy registry bundle cache directory.
    pub cache_path: PathBuf,
}

impl CacheStats {
    /// Return the root directory that owns all managed model cache locations.
    pub fn cache_root(&self) -> PathBuf {
        crate::cache::layout::CacheLayout::from_registry_root(self.cache_path.clone())
            .cache_root()
            .to_path_buf()
    }

    /// Get human-readable size.
    pub fn total_size_human(&self) -> String {
        Self::format_size(self.total_size_bytes)
    }

    /// Format a byte size using the cache summary display convention.
    pub fn format_size(bytes: u64) -> String {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::layout::CacheLayout;
    use chrono::TimeZone;
    use std::sync::Mutex;

    fn create_vlm_bundle(temp_dir: &tempfile::TempDir, model_id: &str) -> PathBuf {
        let model_dir = temp_dir.path().join("bundle_model_files");
        std::fs::create_dir_all(&model_dir).unwrap();

        let metadata = format!(
            r#"{{
                "model_id": "{}",
                "version": "1.0",
                "execution_template": {{
                    "type": "VisionLanguage",
                    "model_file": "model.gguf"
                }},
                "vision_encoder": {{
                    "file": "mmproj-model.gguf",
                    "preprocessing_preset": "gemma3_vision",
                    "image_size": 896
                }},
                "preprocessing": [],
                "postprocessing": [],
                "files": ["model.gguf", "mmproj-model.gguf"],
                "metadata": {{ "task": "vlm" }}
            }}"#,
            model_id
        );
        std::fs::write(model_dir.join("model_metadata.json"), &metadata).unwrap();
        std::fs::write(model_dir.join("model.gguf"), b"fake language model").unwrap();
        std::fs::write(
            model_dir.join("mmproj-model.gguf"),
            b"fake vision projector",
        )
        .unwrap();

        let mut bundle = xybrid_core::bundler::XyBundle::new(model_id, "1.0", "universal");
        bundle
            .add_file(model_dir.join("model_metadata.json"))
            .unwrap();
        bundle.add_file(model_dir.join("model.gguf")).unwrap();
        bundle
            .add_file(model_dir.join("mmproj-model.gguf"))
            .unwrap();

        let bundle_path = temp_dir.path().join(format!("{}.xyb", model_id));
        bundle.write(&bundle_path).unwrap();
        bundle_path
    }

    #[test]
    fn classify_download_source_recognises_r2_hosts() {
        // Xybrid's R2 mirror serves both the registry's primary bundle
        // URLs and the `r2.xybrid.dev` fallback list. All three host
        // shapes must label as `"r2"` so cost attribution doesn't split
        // the row across CDN edges.
        assert_eq!(
            classify_download_source("https://r2.xybrid.dev/v1/kokoro/universal.xyb"),
            "r2"
        );
        assert_eq!(
            classify_download_source("https://abcd1234.r2.cloudflarestorage.com/bundles/x.xyb"),
            "r2"
        );
        assert_eq!(
            classify_download_source("https://pub-xxx.r2.dev/x.xyb"),
            "r2"
        );
    }

    #[test]
    fn classify_download_source_recognises_huggingface_hosts() {
        // Passthrough variants resolve to raw HuggingFace download URLs.
        assert_eq!(
            classify_download_source(
                "https://huggingface.co/xybrid-ai/kokoro-82m/resolve/main/model.gguf"
            ),
            "huggingface"
        );
        assert_eq!(
            classify_download_source("https://hf.co/owner/repo/resolve/main/m.gguf"),
            "huggingface"
        );
    }

    #[test]
    fn classify_download_source_falls_back_to_other() {
        // Unknown hosts must still produce a labelled event so a future
        // origin doesn't silently drop attribution rows. The platform
        // can promote `"other"` to a recognised category later.
        assert_eq!(
            classify_download_source("https://cdn.example.com/m.gguf"),
            "other"
        );
        assert_eq!(classify_download_source(""), "other");
    }

    /// Build a resolved variant with the given main + companion sizes.
    fn variant_with_sizes(main: u64, companions: &[u64]) -> ResolvedVariant {
        let artifacts: Vec<serde_json::Value> = companions
            .iter()
            .enumerate()
            .map(|(index, size)| {
                serde_json::json!({
                    "file": format!("companion-{index}.gguf"),
                    "download_url": format!("https://example.com/companion-{index}.gguf"),
                    "size_bytes": size,
                    "sha256": ""
                })
            })
            .collect();
        serde_json::from_value(serde_json::json!({
            "hf_repo": "xybrid-ai/sized",
            "file": "model.gguf",
            "download_url": "https://example.com/model.gguf",
            "format": "gguf",
            "quantization": "q4_k_m",
            "size_bytes": main,
            "sha256": "",
            "passthrough": true,
            "artifacts": artifacts,
        }))
        .expect("test variant should deserialize")
    }

    #[test]
    fn declared_total_is_unknown_unless_every_artifact_publishes_a_size() {
        assert_eq!(total_declared_bytes(&variant_with_sizes(10, &[5, 2])), 17);
        assert_eq!(total_declared_bytes(&variant_with_sizes(10, &[])), 10);

        // A companion with no declared size would otherwise let the running
        // count overshoot the "total": the bar would saturate at the in-flight
        // ceiling and the terminal frame could report fewer bytes than the
        // caller already saw. `0` means unknown, which downgrades progress to
        // the coarse signal but keeps the byte count exact.
        assert_eq!(total_declared_bytes(&variant_with_sizes(10, &[0])), 0);
        assert_eq!(total_declared_bytes(&variant_with_sizes(10, &[5, 0])), 0);
        assert_eq!(total_declared_bytes(&variant_with_sizes(0, &[5])), 0);
    }

    #[test]
    fn retry_backoff_gives_up_promptly_when_cancelled() {
        // A server-supplied `Retry-After` can run to tens of seconds; a cancel
        // arriving during it must not sit unobserved for the whole interval.
        let cancel = Arc::new(AtomicBool::new(true));
        let sink = |_: DownloadStatus| {};
        let reporter = ProgressReporter::new(None, 1, cancel, &sink);

        let started = Instant::now();
        assert!(!RegistryClient::sleep_unless_cancelled(
            Duration::from_secs(30),
            &reporter
        ));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "cancelled backoff waited {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn resolved_variant_preserves_passthrough_sibling_artifacts() {
        let resolved: ResolvedVariant = serde_json::from_str(
            r#"
{
  "hf_repo": "xybrid-ai/vlm",
  "file": "model.gguf",
  "download_url": "https://example.com/model.gguf",
  "format": "gguf",
  "quantization": "q4_k_m",
  "size_bytes": 10,
  "sha256": "model-hash",
  "passthrough": true,
  "artifacts": [
    {
      "file": "mmproj-model.gguf",
      "download_url": "https://example.com/mmproj-model.gguf",
      "size_bytes": 5,
      "sha256": "mmproj-hash"
    }
  ],
  "model_metadata": {
    "model_id": "vlm",
    "version": "1.0",
    "execution_template": { "type": "VisionLanguage", "model_file": "model.gguf" },
    "vision_encoder": {
      "file": "mmproj-model.gguf",
      "preprocessing_preset": "gemma3_vision",
      "image_size": 896
    },
    "files": ["model.gguf", "mmproj-model.gguf"],
    "metadata": {}
  }
}
"#,
        )
        .unwrap();

        let roundtrip = serde_json::to_value(&resolved).unwrap();
        let artifacts = roundtrip
            .get("artifacts")
            .and_then(|value| value.as_array())
            .expect("resolved passthrough variant must keep sibling artifact descriptors");
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0]["file"], "mmproj-model.gguf");
        assert_eq!(
            artifacts[0]["download_url"],
            "https://example.com/mmproj-model.gguf"
        );
        assert_eq!(artifacts[0]["sha256"], "mmproj-hash");
    }

    #[test]
    fn fetch_extracted_passthrough_downloads_sibling_artifacts() {
        use httpmock::prelude::*;
        use sha2::{Digest, Sha256};

        fn sha256_hex(bytes: &[u8]) -> String {
            let mut hasher = Sha256::new();
            hasher.update(bytes);
            format!("{:x}", hasher.finalize())
        }

        let server = MockServer::start();
        let model_bytes = b"main model";
        let mmproj_bytes = b"vision projector";
        let model_hash = sha256_hex(model_bytes);
        let mmproj_hash = sha256_hex(mmproj_bytes);

        let model_mock = server.mock(|when, then| {
            when.method(GET).path("/model.gguf");
            then.status(200).body(model_bytes.as_slice());
        });
        let mmproj_mock = server.mock(|when, then| {
            when.method(GET).path("/mmproj-model.gguf");
            then.status(200).body(mmproj_bytes.as_slice());
        });

        let resolve_body = serde_json::json!({
            "mask": "vlm",
            "platform": "universal",
            "resolved": {
                "hf_repo": "xybrid-ai/vlm",
                "file": "model.gguf",
                "download_url": server.url("/model.gguf"),
                "format": "gguf",
                "quantization": "q4_k_m",
                "size_bytes": model_bytes.len(),
                "sha256": model_hash,
                "passthrough": true,
                "artifacts": [
                    {
                        "file": "mmproj-model.gguf",
                        "download_url": server.url("/mmproj-model.gguf"),
                        "size_bytes": mmproj_bytes.len(),
                        "sha256": mmproj_hash
                    }
                ],
                "model_metadata": {
                    "model_id": "vlm",
                    "version": "1.0",
                    "execution_template": {
                        "type": "VisionLanguage",
                        "model_file": "model.gguf"
                    },
                    "vision_encoder": {
                        "file": "mmproj-model.gguf",
                        "preprocessing_preset": "gemma3_vision",
                        "image_size": 896
                    },
                    "files": ["model.gguf", "mmproj-model.gguf"],
                    "metadata": {}
                }
            }
        });
        let resolve_mock = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models/vlm/resolve")
                .query_param_exists("platform");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(resolve_body);
        });

        let temp_dir = tempfile::TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache").join("models");
        let mut client = RegistryClient::with_url(server.base_url()).unwrap();
        client.cache = CacheManager::with_dir(cache_dir).unwrap();

        let model_dir = client
            .fetch_extracted("vlm", Some("universal"), |_| {})
            .expect("passthrough VLM fetch should materialize all artifacts");

        assert_eq!(
            std::fs::read(model_dir.join("model.gguf")).unwrap(),
            model_bytes
        );
        assert_eq!(
            std::fs::read(model_dir.join("mmproj-model.gguf")).unwrap(),
            mmproj_bytes
        );
        assert!(
            model_dir.join("model_metadata.json").exists(),
            "metadata must be written next to passthrough artifacts"
        );
        assert_eq!(
            read_cached_hash(&model_dir.join("model.gguf")).as_deref(),
            Some(model_hash.as_str())
        );
        assert_eq!(
            read_cached_hash(&model_dir.join("mmproj-model.gguf")).as_deref(),
            Some(mmproj_hash.as_str())
        );

        resolve_mock.assert();
        model_mock.assert();
        mmproj_mock.assert();
    }

    /// The headline bug this surface exists to fix: a two-file model used to
    /// run the bar 0→1 for the weights, then 0→1 again for the projector.
    #[test]
    fn multi_file_progress_is_one_monotonic_bar_over_summed_bytes() {
        use httpmock::prelude::*;
        use sha2::{Digest, Sha256};

        fn sha256_hex(bytes: &[u8]) -> String {
            let mut hasher = Sha256::new();
            hasher.update(bytes);
            format!("{:x}", hasher.finalize())
        }

        let server = MockServer::start();
        let model_bytes = b"main model";
        let mmproj_bytes = b"vision projector";
        let total = (model_bytes.len() + mmproj_bytes.len()) as u64;
        server.mock(|when, then| {
            when.method(GET).path("/model.gguf");
            then.status(200).body(model_bytes.as_slice());
        });
        server.mock(|when, then| {
            when.method(GET).path("/mmproj-model.gguf");
            then.status(200).body(mmproj_bytes.as_slice());
        });
        let resolve_body = serde_json::json!({
            "mask": "vlm",
            "platform": "universal",
            "resolved": {
                "hf_repo": "xybrid-ai/vlm",
                "file": "model.gguf",
                "download_url": server.url("/model.gguf"),
                "format": "gguf",
                "quantization": "q4_k_m",
                "size_bytes": model_bytes.len(),
                "sha256": sha256_hex(model_bytes),
                "passthrough": true,
                "artifacts": [
                    {
                        "file": "mmproj-model.gguf",
                        "download_url": server.url("/mmproj-model.gguf"),
                        "size_bytes": mmproj_bytes.len(),
                        "sha256": sha256_hex(mmproj_bytes)
                    }
                ],
                "model_metadata": {
                    "model_id": "vlm",
                    "version": "1.0",
                    "execution_template": {
                        "type": "VisionLanguage",
                        "model_file": "model.gguf"
                    },
                    "vision_encoder": {
                        "file": "mmproj-model.gguf",
                        "preprocessing_preset": "gemma3_vision",
                        "image_size": 896
                    },
                    "files": ["model.gguf", "mmproj-model.gguf"],
                    "metadata": {}
                }
            }
        });
        server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models/vlm/resolve")
                .query_param_exists("platform");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(resolve_body);
        });

        let temp_dir = tempfile::TempDir::new().unwrap();
        let mut client = RegistryClient::with_url(server.base_url()).unwrap();
        client.cache = CacheManager::with_dir(temp_dir.path().join("cache")).unwrap();

        let seen = std::sync::Mutex::new(Vec::new());
        client
            .fetch_extracted("vlm", Some("universal"), |status| {
                seen.lock().unwrap().push(status);
            })
            .expect("passthrough VLM fetch should materialize all artifacts");

        let seen = seen.into_inner().unwrap();
        assert!(!seen.is_empty(), "no progress was reported at all");

        // Every update names the summed total, so the bar is scaled once.
        assert!(
            seen.iter().all(|status| status.total_bytes == Some(total)),
            "total must span both artifacts: {seen:?}"
        );

        // Monotonic, and no mid-download update claims completion.
        let mut previous = 0;
        for status in &seen {
            assert!(
                status.downloaded_bytes >= previous,
                "bytes rewound: {seen:?}"
            );
            previous = status.downloaded_bytes;
            if status.state == crate::DownloadState::Downloading {
                assert!(
                    status.progress < 1.0,
                    "in-flight update hit 1.0: {status:?}"
                );
            }
        }

        // Finishing the first artifact reads its share of the whole, not 100%.
        let after_first = seen
            .iter()
            .find(|status| status.downloaded_bytes == model_bytes.len() as u64)
            .expect("the first artifact's completion should be reported");
        let expected = model_bytes.len() as f32 / total as f32;
        assert!(
            (after_first.progress - expected).abs() < 1e-3,
            "expected {expected}, got {}",
            after_first.progress
        );

        // Exactly one terminal frame, and it is the last thing emitted.
        let last = seen.last().unwrap();
        assert_eq!(last.state, crate::DownloadState::Ready);
        assert_eq!(last.progress, 1.0);
        assert_eq!(last.downloaded_bytes, total);
    }

    /// Cancelling must stop the transfer and leave no partial file behind, so
    /// a later attempt starts clean rather than resuming into a truncated one.
    #[test]
    fn cancelling_a_fetch_stops_it_and_discards_the_partial_file() {
        use httpmock::prelude::*;
        use std::sync::atomic::{AtomicBool, Ordering};

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/model.gguf");
            then.status(200).body(vec![0u8; 512 * 1024]);
        });
        let resolve_body = serde_json::json!({
            "mask": "slow",
            "platform": "universal",
            "resolved": {
                "hf_repo": "xybrid-ai/slow",
                "file": "model.gguf",
                "download_url": server.url("/model.gguf"),
                "format": "gguf",
                "quantization": "q4_k_m",
                "size_bytes": 512 * 1024,
                "sha256": "",
                "passthrough": true,
                "model_metadata": {
                    "model_id": "slow",
                    "version": "1.0",
                    "execution_template": { "type": "Gguf", "model_file": "model.gguf" },
                    "files": ["model.gguf"],
                    "metadata": {}
                }
            }
        });
        server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models/slow/resolve")
                .query_param_exists("platform");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(resolve_body);
        });

        let temp_dir = tempfile::TempDir::new().unwrap();
        let mut client = RegistryClient::with_url(server.base_url()).unwrap();
        client.cache = CacheManager::with_dir(temp_dir.path().join("cache")).unwrap();

        // Flip the flag from inside the progress callback once bytes are
        // flowing, so cancellation lands mid-transfer rather than on the
        // start frame, before the request goes out.
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        let err = client
            .fetch_extracted_cancellable("slow", Some("universal"), cancel, move |status| {
                if status.downloaded_bytes > 0 {
                    flag.store(true, Ordering::Relaxed);
                }
            })
            .expect_err("a cancelled fetch must not report success");

        assert!(
            matches!(err, SdkError::Cancelled { .. }),
            "expected Cancelled, got {err:?}"
        );
        assert!(
            !err.is_retryable(),
            "retrying would resume what the caller just stopped"
        );
        let partial = client.cache.extraction_dir("slow").join("model.gguf");
        assert!(
            !partial.exists(),
            "partial file survived cancellation at {}",
            partial.display()
        );
    }

    /// One canned reply per connection, for failures `httpmock` cannot stage:
    /// a body cut off mid-transfer, a connection that goes silent.
    struct ScriptedReply {
        /// Status line and headers, without the blank line that ends them.
        head: String,
        body: Vec<u8>,
        /// Keep the connection open this long after the body, sending nothing.
        stall: Option<Duration>,
    }

    impl ScriptedReply {
        fn new(status: &str, headers: &[(&str, String)], body: &[u8]) -> Self {
            let mut head = format!("HTTP/1.1 {status}\r\nConnection: close");
            for (name, value) in headers {
                head.push_str(&format!("\r\n{name}: {value}"));
            }
            Self {
                head,
                body: body.to_vec(),
                stall: None,
            }
        }

        fn then_stall(mut self, stall: Duration) -> Self {
            self.stall = Some(stall);
            self
        }
    }

    /// Serve `replies` in order, one per connection. Returns the download URL
    /// and the request heads received, in arrival order.
    fn scripted_server(replies: Vec<ScriptedReply>) -> (String, Arc<Mutex<Vec<String>>>) {
        scripted_server_at("127.0.0.1:0", replies)
    }

    /// [`scripted_server`] on a given address. The listener closes once the
    /// replies run out, so the address refuses connections from then on.
    fn scripted_server_at(
        addr: &str,
        replies: Vec<ScriptedReply>,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        use std::io::BufRead;

        let listener = std::net::TcpListener::bind(addr).unwrap();
        let url = format!("http://{}/model.gguf", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        std::thread::spawn(move || {
            for (reply, stream) in replies.into_iter().zip(listener.incoming()) {
                let Ok(mut stream) = stream else { return };
                let recorded = Arc::clone(&recorded);
                // One thread per connection, so a stalled reply cannot hold up
                // the retry that follows it.
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut head = String::new();
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                        head.push_str(&line);
                    }
                    recorded.lock().unwrap().push(head.to_ascii_lowercase());
                    let _ = stream.write_all(format!("{}\r\n\r\n", reply.head).as_bytes());
                    let _ = stream.write_all(&reply.body);
                    let _ = stream.flush();
                    if let Some(stall) = reply.stall {
                        std::thread::sleep(stall);
                    }
                });
            }
        });
        (url, requests)
    }

    /// A client whose downloads retry at once and give up on a silent
    /// connection quickly.
    fn fast_retry_client(temp: &Path) -> RegistryClient {
        let mut client = RegistryClient::with_url("http://127.0.0.1:9").unwrap();
        client.cache = CacheManager::with_dir(temp.join("cache")).unwrap();
        client.download_retry_policy = RetryPolicy {
            max_attempts: 3,
            initial_delay_ms: 0,
            max_delay_ms: 0,
            jitter_factor: 0.0,
        };
        client.download_stall_timeout = Duration::from_millis(300);
        client
    }

    /// Distinct bytes, so a spliced or duplicated range cannot compare equal.
    fn model_body(len: u32, seed: u32) -> Vec<u8> {
        (0..len).map(|i| ((i + seed) % 251) as u8).collect()
    }

    fn partial_reply(body: &[u8], from: usize, etag: &str) -> ScriptedReply {
        ScriptedReply::new(
            "206 Partial Content",
            &[
                ("ETag", etag.to_string()),
                (
                    "Content-Range",
                    format!("bytes {from}-{}/{}", body.len() - 1, body.len()),
                ),
                ("Content-Length", (body.len() - from).to_string()),
            ],
            &body[from..],
        )
    }

    /// A 200 that announces all of `body` but sends only its first `sent` bytes.
    fn cut_off_reply(body: &[u8], sent: usize, etag: &str) -> ScriptedReply {
        ScriptedReply::new(
            "200 OK",
            &[
                ("ETag", etag.to_string()),
                ("Content-Length", body.len().to_string()),
            ],
            &body[..sent],
        )
    }

    fn full_reply(body: &[u8], etag: &str) -> ScriptedReply {
        cut_off_reply(body, body.len(), etag)
    }

    #[test]
    fn an_interrupted_download_resumes_where_it_stopped() {
        let body = model_body(20_000, 0);
        let half = body.len() / 2;
        let (url, requests) = scripted_server(vec![
            cut_off_reply(&body, half, "\"v1\""),
            partial_reply(&body, half, "\"v1\""),
        ]);
        let temp = tempfile::TempDir::new().unwrap();
        let mut client = fast_retry_client(temp.path());
        // A drop that still moved the file along must not use up the budget,
        // or a large model on a flaky link would fail after a few drops.
        client.download_retry_policy.max_attempts = 1;
        let dest = temp.path().join("model.gguf");

        let seen = Mutex::new(Vec::new());
        let sink = |status: DownloadStatus| seen.lock().unwrap().push(status);
        // No declared size, like the registry's `lfm2.5-350m` entry.
        let reporter = ProgressReporter::new(None, 1, Arc::new(AtomicBool::new(false)), &sink);

        client
            .download_with_progress(&url, &dest, &reporter)
            .expect("the interrupted download should resume and complete");

        assert_eq!(std::fs::read(&dest).unwrap(), body);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2, "{requests:?}");
        assert!(!requests[0].contains("range:"), "{}", requests[0]);
        assert!(
            requests[1].contains(&format!("range: bytes={half}-")),
            "the retry must ask for only the missing bytes: {}",
            requests[1]
        );
        assert!(requests[1].contains("if-range: \"v1\""), "{}", requests[1]);

        let seen = seen.lock().unwrap();
        assert_eq!(
            seen[0].downloaded_bytes, 0,
            "a start frame must precede the first byte"
        );
        assert!(
            seen.iter()
                .skip(1)
                .all(|status| status.total_bytes == Some(body.len() as u64)),
            "the announced size must stand in for the missing declared one: {seen:?}"
        );
        assert!(
            seen.windows(2)
                .all(|pair| pair[1].downloaded_bytes >= pair[0].downloaded_bytes),
            "bytes rewound: {seen:?}"
        );
    }

    #[test]
    fn a_short_range_is_continued_until_the_file_is_whole() {
        // HTTP lets a server answer `Range: bytes=N-` with less than the rest
        // of the file. Stopping at the end of that range would cache a
        // truncated model as ready. The middle reply also withholds the total
        // (`/*`), so only the size from the first response can catch it.
        let body = model_body(20_000, 0);
        let half = body.len() / 2;
        let chunk_end = half + 1_000;
        let (url, requests) = scripted_server(vec![
            cut_off_reply(&body, half, "\"v1\""),
            ScriptedReply::new(
                "206 Partial Content",
                &[
                    ("ETag", "\"v1\"".to_string()),
                    ("Content-Range", format!("bytes {half}-{}/*", chunk_end - 1)),
                    ("Content-Length", (chunk_end - half).to_string()),
                ],
                &body[half..chunk_end],
            ),
            partial_reply(&body, chunk_end, "\"v1\""),
        ]);
        let temp = tempfile::TempDir::new().unwrap();
        let client = fast_retry_client(temp.path());
        let dest = temp.path().join("model.gguf");
        let sink = |_: DownloadStatus| {};
        let reporter = ProgressReporter::new(None, 1, Arc::new(AtomicBool::new(false)), &sink);

        client
            .download_with_progress(&url, &dest, &reporter)
            .expect("the download should continue past the short range");

        assert_eq!(std::fs::read(&dest).unwrap(), body);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3, "{requests:?}");
        assert!(
            requests[2].contains(&format!("range: bytes={chunk_end}-")),
            "{}",
            requests[2]
        );
    }

    /// A 200 whose chunked body ends before its `Content-Length`: ureq lets
    /// chunked framing win, so the short body arrives without an error.
    fn short_chunked_reply(body: &[u8], sent: usize) -> ScriptedReply {
        let mut chunked = format!("{sent:x}\r\n").into_bytes();
        chunked.extend_from_slice(&body[..sent]);
        chunked.extend_from_slice(b"\r\n0\r\n\r\n");
        ScriptedReply::new(
            "200 OK",
            &[
                ("ETag", "\"v1\"".to_string()),
                ("Content-Length", body.len().to_string()),
                ("Transfer-Encoding", "chunked".to_string()),
            ],
            &chunked,
        )
    }

    #[test]
    fn a_full_response_that_ends_early_goes_through_the_retry_budget() {
        // The server ignores `Range`, so every request restarts the file, and
        // each ends at a different point. Asking for "the rest" right away
        // would loop with no backoff and no limit. Each short response must
        // instead count as an attempt: 5 000 and 15 000 bytes are new highs,
        // 8 000 is not, and with a budget of one the third request is the last.
        let body = model_body(20_000, 0);
        let (url, requests) = scripted_server(vec![
            short_chunked_reply(&body, 5_000),
            short_chunked_reply(&body, 15_000),
            short_chunked_reply(&body, 8_000),
            short_chunked_reply(&body, 12_000),
        ]);
        let temp = tempfile::TempDir::new().unwrap();
        let mut client = fast_retry_client(temp.path());
        client.download_retry_policy.max_attempts = 1;
        let dest = temp.path().join("model.gguf");
        let sink = |_: DownloadStatus| {};
        let reporter = ProgressReporter::new(None, 1, Arc::new(AtomicBool::new(false)), &sink);

        let err = client
            .download_with_progress(&url, &dest, &reporter)
            .expect_err("a file that never arrives whole must fail");

        assert!(err.is_retryable(), "expected a network error, got {err:?}");
        assert_eq!(requests.lock().unwrap().len(), 3);
        assert!(!dest.exists(), "partial file left at {}", dest.display());
    }

    #[test]
    fn a_download_waits_out_a_network_drop_once_bytes_have_arrived() {
        // The first connection delivers half the file, then the address
        // refuses connections, the way a phone that lost Wi-Fi fails every
        // attempt at once. A budget of one would end the download on the
        // first refusal; having received bytes, it must wait instead and
        // resume when the server is back.
        let body = model_body(20_000, 0);
        let half = body.len() / 2;
        let (url, requests) = scripted_server(vec![cut_off_reply(&body, half, "\"v1\"")]);
        let addr = url
            .trim_start_matches("http://")
            .trim_end_matches("/model.gguf")
            .to_string();
        let temp = tempfile::TempDir::new().unwrap();
        let mut client = fast_retry_client(temp.path());
        client.download_retry_policy.max_attempts = 1;
        client.download_retry_policy.initial_delay_ms = 50;
        client.download_offline_patience = Duration::from_secs(10);
        let dest = temp.path().join("model.gguf");

        let restored = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(600));
            scripted_server_at(&addr, vec![partial_reply(&body, half, "\"v1\"")])
        });
        let sink = |_: DownloadStatus| {};
        let reporter = ProgressReporter::new(None, 1, Arc::new(AtomicBool::new(false)), &sink);

        client
            .download_with_progress(&url, &dest, &reporter)
            .expect("the download should wait for the network and resume");

        let (_, resumed_requests) = restored.join().unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), model_body(20_000, 0));
        assert_eq!(requests.lock().unwrap().len(), 1);
        let resumed = resumed_requests.lock().unwrap();
        assert_eq!(resumed.len(), 1, "{resumed:?}");
        assert!(
            resumed[0].contains(&format!("range: bytes={half}-")),
            "{}",
            resumed[0]
        );
    }

    #[test]
    fn a_new_outage_gets_its_own_patience_window() {
        // An outage of nearly the whole window, a brief return (the server
        // answers 502), then a second outage. Carrying the first outage's
        // clock over would end the download the moment the second began.
        let patience = Duration::from_secs(120);
        let start = Instant::now();
        let mut window = OfflineWindow::default();

        assert_eq!(window.record(start, patience), Some(1));
        assert_eq!(
            window.record(start + Duration::from_secs(110), patience),
            Some(2)
        );

        window.end();
        let second = start + Duration::from_secs(115);
        assert_eq!(window.record(second, patience), Some(1));
        assert_eq!(
            window.record(second + Duration::from_secs(60), patience),
            Some(2)
        );
        assert_eq!(
            window.record(second + Duration::from_secs(120), patience),
            None,
            "a single outage is still bounded"
        );
    }

    #[test]
    fn a_download_that_starts_offline_fails_fast() {
        // Nothing listens here. With no byte received yet, refusals count
        // against the budget as before, so an app that starts offline hears
        // about it at once rather than after the offline patience.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let url = format!("http://127.0.0.1:{port}/model.gguf");
        let temp = tempfile::TempDir::new().unwrap();
        let mut client = fast_retry_client(temp.path());
        client.download_offline_patience = Duration::from_secs(60);
        let dest = temp.path().join("model.gguf");
        let sink = |_: DownloadStatus| {};
        let reporter = ProgressReporter::new(None, 1, Arc::new(AtomicBool::new(false)), &sink);

        let started = Instant::now();
        let err = client
            .download_with_progress(&url, &dest, &reporter)
            .expect_err("nothing is listening");

        assert!(
            matches!(err, SdkError::Offline { .. }),
            "expected Offline, got {err:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "waited {:?} before reporting a download that never started",
            started.elapsed()
        );
    }

    #[test]
    fn a_silent_connection_is_dropped_and_resumed() {
        let body = model_body(20_000, 0);
        let half = body.len() / 2;
        let (url, _requests) = scripted_server(vec![
            cut_off_reply(&body, half, "\"v1\"").then_stall(Duration::from_secs(5)),
            partial_reply(&body, half, "\"v1\""),
        ]);
        let temp = tempfile::TempDir::new().unwrap();
        let client = fast_retry_client(temp.path());
        let dest = temp.path().join("model.gguf");
        let sink = |_: DownloadStatus| {};
        let reporter = ProgressReporter::new(None, 1, Arc::new(AtomicBool::new(false)), &sink);

        let started = Instant::now();
        client
            .download_with_progress(&url, &dest, &reporter)
            .expect("the stalled download should resume and complete");

        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "waited {:?} on a silent connection",
            started.elapsed()
        );
    }

    #[test]
    fn a_resume_that_lands_on_a_replaced_file_starts_over() {
        // Hugging Face's CDN answers a range request for a replaced file with
        // a 206 of the new one, ignoring `If-Range`. Appending it would splice
        // two versions into one corrupt model.
        let old = model_body(20_000, 0);
        let new = model_body(20_000, 7);
        let half = old.len() / 2;
        let (url, requests) = scripted_server(vec![
            cut_off_reply(&old, half, "\"v1\""),
            partial_reply(&new, half, "\"v2\""),
            full_reply(&new, "\"v2\""),
        ]);
        let temp = tempfile::TempDir::new().unwrap();
        let client = fast_retry_client(temp.path());
        let dest = temp.path().join("model.gguf");
        let sink = |_: DownloadStatus| {};
        let reporter = ProgressReporter::new(None, 1, Arc::new(AtomicBool::new(false)), &sink);

        client
            .download_with_progress(&url, &dest, &reporter)
            .expect("the download should restart on the new file");

        assert_eq!(std::fs::read(&dest).unwrap(), new);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3, "{requests:?}");
        assert!(!requests[2].contains("range:"), "{}", requests[2]);
    }

    #[test]
    fn a_server_that_ignores_ranges_gets_a_clean_restart() {
        let body = model_body(20_000, 0);
        let (url, _requests) = scripted_server(vec![
            cut_off_reply(&body, body.len() / 2, "\"v1\""),
            full_reply(&body, "\"v1\""),
        ]);
        let temp = tempfile::TempDir::new().unwrap();
        let client = fast_retry_client(temp.path());
        let dest = temp.path().join("model.gguf");
        let sink = |_: DownloadStatus| {};
        let reporter = ProgressReporter::new(None, 1, Arc::new(AtomicBool::new(false)), &sink);

        client
            .download_with_progress(&url, &dest, &reporter)
            .expect("a full 200 should replace the partial file");

        // Replaced, not appended to the first half.
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[test]
    fn a_download_that_never_progresses_gives_up_and_leaves_no_partial() {
        let body = model_body(20_000, 0);
        let (url, requests) =
            scripted_server((0..3).map(|_| cut_off_reply(&body, 0, "\"v1\"")).collect());
        let temp = tempfile::TempDir::new().unwrap();
        let client = fast_retry_client(temp.path());
        let dest = temp.path().join("model.gguf");
        let sink = |_: DownloadStatus| {};
        let reporter = ProgressReporter::new(None, 1, Arc::new(AtomicBool::new(false)), &sink);

        let err = client
            .download_with_progress(&url, &dest, &reporter)
            .expect_err("a server that never sends a byte must not loop forever");

        assert!(err.is_retryable(), "expected a network error, got {err:?}");
        assert_eq!(requests.lock().unwrap().len(), 3);
        assert!(!dest.exists(), "partial file left at {}", dest.display());
    }

    #[test]
    fn content_range_parses_start_and_total() {
        assert_eq!(
            parse_content_range("bytes 1000-1999/229312224"),
            Some((1000, Some(229_312_224)))
        );
        assert_eq!(parse_content_range("bytes 5-9/*"), Some((5, None)));
        assert_eq!(parse_content_range("bytes */100"), None);
        assert_eq!(parse_content_range("items 0-1/2"), None);
    }

    #[test]
    fn fetch_extracted_bundle_repairs_partial_multifile_vlm_extraction() {
        use httpmock::prelude::*;
        use sha2::{Digest, Sha256};

        fn sha256_hex(bytes: &[u8]) -> String {
            let mut hasher = Sha256::new();
            hasher.update(bytes);
            format!("{:x}", hasher.finalize())
        }

        let temp_dir = tempfile::TempDir::new().unwrap();
        let bundle_path = create_vlm_bundle(&temp_dir, "vlm-bundle");
        let bundle_bytes = std::fs::read(&bundle_path).unwrap();
        let bundle_hash = sha256_hex(&bundle_bytes);

        let server = MockServer::start();
        let bundle_mock = server.mock(|when, then| {
            when.method(GET).path("/universal.xyb");
            then.status(200).body(bundle_bytes.clone());
        });
        let resolve_body = serde_json::json!({
            "mask": "vlm-bundle",
            "platform": "universal",
            "resolved": {
                "hf_repo": "xybrid-ai/vlm-bundle",
                "file": "universal.xyb",
                "download_url": server.url("/universal.xyb"),
                "format": "gguf",
                "quantization": "q4_k_m",
                "size_bytes": bundle_bytes.len(),
                "sha256": bundle_hash
            }
        });
        let resolve_mock = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models/vlm-bundle/resolve")
                .query_param_exists("platform");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(resolve_body);
        });

        let cache_dir = temp_dir.path().join("cache").join("models");
        let mut client = RegistryClient::with_url(server.base_url()).unwrap();
        client.cache = CacheManager::with_dir(cache_dir).unwrap();

        let partial_dir = client.cache.extraction_dir("vlm-bundle");
        std::fs::create_dir_all(&partial_dir).unwrap();
        let bundle = xybrid_core::bundler::XyBundle::load(&bundle_path).unwrap();
        let metadata_json = bundle.get_metadata_json().unwrap().unwrap();
        std::fs::write(partial_dir.join("model_metadata.json"), metadata_json).unwrap();

        let model_dir = client
            .fetch_extracted("vlm-bundle", Some("universal"), |_| {})
            .expect("bundle VLM fetch should repair partial extraction and materialize siblings");

        assert_eq!(model_dir, partial_dir);
        assert!(model_dir.join("model_metadata.json").exists());
        assert_eq!(
            std::fs::read(model_dir.join("model.gguf")).unwrap(),
            b"fake language model"
        );
        assert_eq!(
            std::fs::read(model_dir.join("mmproj-model.gguf")).unwrap(),
            b"fake vision projector"
        );
        assert!(client.is_extracted("vlm-bundle"));

        // One resolve, not two: `fetch_extracted` used to resolve, then hand
        // the mask to `fetch`, which resolved again. Building the progress
        // reporter needs every artifact's size up front, so the resolved
        // variant is now threaded straight through to the bundle download.
        resolve_mock.assert_hits(1);
        bundle_mock.assert();
    }

    #[test]
    fn cache_entries_include_extracted_runtime_cache() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let cache_root = temp_dir.path().join("cache");
        let models_dir = cache_root.join("models");
        let extracted_model_dir = cache_root.join("extracted").join("lfm2.5-350m");
        let metadata = br#"{"files":[]}"#;
        let weights = b"fake weights";
        std::fs::create_dir_all(&extracted_model_dir).unwrap();
        std::fs::write(extracted_model_dir.join("model_metadata.json"), metadata).unwrap();
        std::fs::write(extracted_model_dir.join("LFM2.5-350M-Q4_K_M.gguf"), weights).unwrap();

        let mut client = RegistryClient::with_url("https://example.test").unwrap();
        client.cache = CacheManager::with_dir(models_dir).unwrap();

        let entries = client.cache_entries().unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].model_id, "lfm2.5-350m");
        assert_eq!(entries[0].location, CacheEntryLocation::Extracted);
        assert_eq!(entries[0].path, extracted_model_dir);
        assert_eq!(
            entries[0].size_bytes,
            (metadata.len() + weights.len()) as u64
        );

        let stats = client.cache_stats().unwrap();
        assert_eq!(stats.model_count, 1);
        assert_eq!(stats.cache_root(), cache_root);
        assert_eq!(
            stats.total_size_bytes,
            (metadata.len() + weights.len()) as u64
        );
    }

    #[test]
    fn cache_entries_include_all_managed_cache_roots() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let cache_root = temp_dir.path().join("cache");
        let models_dir = cache_root.join("models");
        let registry_model_dir = models_dir.join("registry-model");
        let extracted_model_dir = cache_root.join("extracted").join("runtime-model");
        let hf_model_dir = cache_root.join("hf").join("owner--repo");
        let layout = CacheLayout::from_registry_root(models_dir.clone());
        let hf_hub_model_dir = layout
            .prepare_huggingface_hub_repo_root("owner/repo")
            .unwrap();
        std::fs::create_dir_all(&registry_model_dir).unwrap();
        std::fs::create_dir_all(&extracted_model_dir).unwrap();
        std::fs::create_dir_all(&hf_model_dir).unwrap();
        std::fs::create_dir_all(&hf_hub_model_dir).unwrap();
        std::fs::write(registry_model_dir.join("model.xyb"), b"bundle").unwrap();
        std::fs::write(extracted_model_dir.join("model.gguf"), b"runtime").unwrap();
        std::fs::write(hf_model_dir.join("model.gguf"), b"hf").unwrap();
        std::fs::write(hf_hub_model_dir.join("blob"), b"hub").unwrap();

        let mut client = RegistryClient::with_url("https://example.test").unwrap();
        client.cache = CacheManager::with_dir(models_dir).unwrap();

        let entries = client.cache_entries().unwrap();
        let labels: Vec<_> = entries
            .iter()
            .map(|entry| (entry.model_id.as_str(), entry.location.as_str()))
            .collect();

        assert_eq!(entries.len(), 4);
        assert!(labels.contains(&("registry-model", "models")));
        assert!(labels.contains(&("runtime-model", "extracted")));
        assert!(labels.contains(&("owner--repo", "hf")));
        assert!(labels.contains(&("owner/repo", "hf-hub")));

        let stats = client.cache_stats().unwrap();
        assert_eq!(stats.model_count, 4);
        assert_eq!(stats.total_size_bytes, 18);
    }

    #[test]
    fn cache_entries_include_custom_root_and_legacy_parent_extracted_cache() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let custom_cache = temp_dir.path().join("custom-cache");
        let registry_model_dir = custom_cache.join("registry-model");
        let extracted_model_dir = custom_cache.join("extracted").join("runtime-model");
        let legacy_extracted_model_dir = temp_dir.path().join("extracted").join("legacy-model");
        let hf_model_dir = custom_cache.join("hf").join("owner--repo");
        let layout = CacheLayout::from_registry_root(custom_cache.clone());
        let hf_hub_model_dir = layout
            .prepare_huggingface_hub_repo_root("owner/repo")
            .unwrap();
        std::fs::create_dir_all(&registry_model_dir).unwrap();
        std::fs::create_dir_all(&extracted_model_dir).unwrap();
        std::fs::create_dir_all(&legacy_extracted_model_dir).unwrap();
        std::fs::create_dir_all(&hf_model_dir).unwrap();
        std::fs::create_dir_all(&hf_hub_model_dir).unwrap();
        std::fs::write(registry_model_dir.join("model.xyb"), b"bundle").unwrap();
        std::fs::write(extracted_model_dir.join("model.gguf"), b"runtime").unwrap();
        std::fs::write(legacy_extracted_model_dir.join("model.gguf"), b"legacy").unwrap();
        std::fs::write(hf_model_dir.join("model.gguf"), b"hf").unwrap();
        std::fs::write(hf_hub_model_dir.join("blob"), b"hub").unwrap();

        let mut client = RegistryClient::with_url("https://example.test").unwrap();
        client.cache = CacheManager::with_dir(custom_cache.clone()).unwrap();

        let entries = client.cache_entries().unwrap();
        let labels: Vec<_> = entries
            .iter()
            .map(|entry| (entry.model_id.as_str(), entry.location.as_str()))
            .collect();

        assert_eq!(entries.len(), 5);
        assert!(labels.contains(&("registry-model", "models")));
        assert!(labels.contains(&("runtime-model", "extracted")));
        assert!(labels.contains(&("legacy-model", "extracted")));
        assert!(labels.contains(&("owner--repo", "hf")));
        assert!(labels.contains(&("owner/repo", "hf-hub")));
        assert!(!labels.contains(&("extracted", "models")));
        assert!(!labels.contains(&("hf", "models")));
        assert!(!labels.contains(&("hf-hub", "models")));

        let stats = client.cache_stats().unwrap();
        assert_eq!(stats.model_count, 5);
        assert_eq!(stats.cache_root(), custom_cache);
        assert_eq!(stats.total_size_bytes, 24);
    }

    #[test]
    fn clear_cache_removes_legacy_bundle_from_memory_index() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("models");
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::fs::write(cache_dir.join("test-model@1.0.xyb"), b"bundle").unwrap();

        let mut client = RegistryClient::with_url("https://example.test").unwrap();
        client.cache = CacheManager::with_dir(cache_dir).unwrap();

        let removed = client.clear_cache("test-model").unwrap();

        assert_eq!(removed, 1);
        assert_eq!(client.cache.status().unwrap().total_models, 0);
        assert!(!client.cache.is_cached("test-model"));
    }

    #[test]
    fn cache_stats_root_keeps_custom_non_models_cache_self_contained() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("custom-cache");
        let stats = CacheStats {
            total_size_bytes: 0,
            model_count: 0,
            cache_path: cache_path.clone(),
        };

        assert_eq!(stats.cache_root(), cache_path);
    }

    #[test]
    fn test_default_client() {
        let client = RegistryClient::default_client().unwrap();
        assert_eq!(client.api_urls.len(), 2);
        assert_eq!(client.primary_url(), DEFAULT_REGISTRY_URL);
    }

    #[test]
    fn build_client_header_default_binding_has_all_fields() {
        let header = build_client_header_with_optout("rust", false)
            .expect("header must be built when not opted out");
        assert!(
            header.starts_with("binding=rust;"),
            "header should start with sanitized binding: {}",
            header
        );
        assert!(
            header.contains("sdk_version="),
            "missing sdk_version: {}",
            header
        );
        assert!(
            header.contains("core_version="),
            "missing core_version: {}",
            header
        );
        assert!(
            header.contains(&format!("platform={}", current_platform())),
            "platform mismatch: {}",
            header
        );
        assert!(
            header.contains("backends="),
            "missing backends key: {}",
            header
        );
    }

    #[test]
    fn build_client_header_opt_out_returns_none() {
        // Tests the inner helper directly so it doesn't fight the OnceLock-
        // cached opt-out state owned by `is_telemetry_opted_out` in other tests.
        assert!(build_client_header_with_optout("rust", true).is_none());
    }

    #[test]
    fn build_client_header_malformed_binding_falls_back_to_default() {
        let header = build_client_header_with_optout("flutter; injected", false)
            .expect("header must be built when not opted out");
        assert!(
            header.starts_with("binding=rust;"),
            "malformed binding must collapse to DEFAULT_BINDING: {}",
            header
        );
        assert!(
            !header.contains("injected"),
            "smuggled tokens must not appear in the header: {}",
            header
        );
    }

    #[test]
    fn build_client_header_uppercase_binding_falls_back_to_default() {
        let header = build_client_header_with_optout("FLUTTER", false).unwrap();
        assert!(
            header.starts_with("binding=rust;"),
            "uppercase binding is not in the [a-z0-9_-] allowlist: {}",
            header
        );
    }

    #[test]
    fn build_client_header_empty_binding_falls_back_to_default() {
        let header = build_client_header_with_optout("", false).unwrap();
        assert!(header.starts_with("binding=rust;"));
    }

    #[test]
    fn build_client_header_accepts_known_bindings() {
        for binding in [
            "rust",
            "flutter",
            "kotlin",
            "react-native",
            "swift",
            "unity",
        ] {
            let header = build_client_header_with_optout(binding, false).unwrap();
            let prefix = format!("binding={};", binding);
            assert!(
                header.starts_with(&prefix),
                "binding `{}` should pass sanitization: {}",
                binding,
                header
            );
        }
    }

    #[test]
    fn build_client_header_renders_empty_backends_list_without_panic() {
        // We can't dynamically clear the compiled-in features table at runtime,
        // but we can assert the header always includes the literal `backends=`
        // key and never panics when the value is empty (the join on an empty
        // slice yields ""). When no features are enabled, the header would end
        // with `backends=` — and that is valid output, not a panic surface.
        let header = build_client_header_with_optout("rust", false).unwrap();
        assert!(
            header.contains("backends="),
            "header always carries the backends key: {}",
            header
        );
        // Sanity: the format must not produce the broken `backends=,` shape.
        assert!(
            !header.contains("backends=,"),
            "leading comma in backends list: {}",
            header
        );
    }

    #[test]
    fn sanitize_binding_accepts_alphanumerics_underscore_and_hyphen() {
        assert_eq!(sanitize_binding("rust"), "rust");
        assert_eq!(sanitize_binding("flutter"), "flutter");
        assert_eq!(sanitize_binding("react-native"), "react-native");
        assert_eq!(sanitize_binding("snake_case"), "snake_case");
        assert_eq!(sanitize_binding("v2"), "v2");
    }

    #[test]
    fn sanitize_binding_rejects_invalid_chars() {
        assert_eq!(sanitize_binding(""), DEFAULT_BINDING);
        assert_eq!(sanitize_binding("Flutter"), DEFAULT_BINDING);
        assert_eq!(sanitize_binding("flutter app"), DEFAULT_BINDING);
        assert_eq!(sanitize_binding("flutter;injected"), DEFAULT_BINDING);
        assert_eq!(sanitize_binding("flu/tter"), DEFAULT_BINDING);
    }

    #[test]
    fn test_single_url_client() {
        let client = RegistryClient::with_url("https://custom.example.com").unwrap();
        assert_eq!(client.api_urls.len(), 1);
        assert_eq!(client.primary_url(), "https://custom.example.com");
    }

    #[test]
    fn test_registry_urls_constant() {
        assert_eq!(REGISTRY_URLS.len(), 2);
        assert_eq!(REGISTRY_URLS[0], DEFAULT_REGISTRY_URL);
        assert_eq!(REGISTRY_URLS[1], FALLBACK_REGISTRY_URL);
    }

    #[test]
    fn retry_after_seconds_header_is_used_for_rate_limit_errors() {
        let client = RegistryClient::default_client().unwrap();
        let error = client.status_to_error(429, "list models", Some("120"));

        assert!(matches!(
            error,
            SdkError::RateLimited {
                retry_after_secs: 120
            }
        ));
    }

    #[test]
    fn retry_after_header_is_read_from_ureq_error_response() {
        use httpmock::prelude::*;

        let server = MockServer::start();
        let rate_limited = server.mock(|when, then| {
            when.method(GET).path("/v1/models");
            then.status(429).header("Retry-After", "120");
        });

        let client = RegistryClient::with_url(server.base_url()).unwrap();
        let response = client
            .agent
            .get(&format!("{}/v1/models", server.base_url()))
            .call();

        assert!(matches!(
            client.handle_response(response, "list models"),
            Err(SdkError::RateLimited {
                retry_after_secs: 120
            })
        ));
        rate_limited.assert();
    }

    #[test]
    fn retry_after_http_date_header_is_used_for_rate_limit_errors() {
        let now = Utc.with_ymd_and_hms(2026, 10, 21, 7, 27, 0).unwrap();

        assert_eq!(
            rate_limit_retry_after_secs_at(Some("Wed, 21 Oct 2026 07:28:00 GMT"), now),
            60
        );
    }

    #[test]
    fn retry_after_missing_or_malformed_header_uses_default_delay() {
        let now = Utc.with_ymd_and_hms(2026, 10, 21, 7, 27, 0).unwrap();

        assert_eq!(
            rate_limit_retry_after_secs_at(None, now),
            DEFAULT_RATE_LIMIT_RETRY_AFTER_SECS
        );
        assert_eq!(
            rate_limit_retry_after_secs_at(Some("not a retry date"), now),
            DEFAULT_RATE_LIMIT_RETRY_AFTER_SECS
        );
    }

    #[test]
    fn retry_after_header_is_capped() {
        let now = Utc.with_ymd_and_hms(2026, 10, 21, 7, 27, 0).unwrap();

        assert_eq!(
            rate_limit_retry_after_secs_at(Some("1200"), now),
            MAX_RATE_LIMIT_RETRY_AFTER_SECS
        );
        assert_eq!(
            rate_limit_retry_after_secs_at(Some("Wed, 21 Oct 2026 07:37:00 GMT"), now),
            MAX_RATE_LIMIT_RETRY_AFTER_SECS
        );
    }

    #[test]
    fn test_cache_path() {
        let client = RegistryClient::default_client().unwrap();
        let resolved = ResolvedVariant {
            hf_repo: "xybrid-ai/kokoro-82m".to_string(),
            file: "universal.xyb".to_string(),
            download_url: "https://example.com/bundle.xyb".to_string(),
            format: "onnx".to_string(),
            quantization: "fp16".to_string(),
            size_bytes: 100000,
            sha256: "abc123".to_string(),
            artifacts: Vec::new(),
            passthrough: false,
            model_metadata: None,
        };
        let path = client.get_cache_path(&resolved);
        assert!(path.to_string_lossy().contains("kokoro-82m"));
        assert!(path.to_string_lossy().contains("universal.xyb"));
    }

    #[test]
    fn test_extraction_dir() {
        let client = RegistryClient::default_client().unwrap();
        let dir = client.extraction_dir("test-model");
        assert!(dir.to_string_lossy().contains("extracted"));
        assert!(dir.to_string_lossy().contains("test-model"));
    }

    #[test]
    fn test_is_extracted_false_for_nonexistent() {
        let client = RegistryClient::default_client().unwrap();
        // A random model ID should not be extracted
        assert!(!client.is_extracted("nonexistent-model-12345"));
    }

    #[test]
    fn test_resolve_offline_none_for_nonexistent() {
        // resolve_offline must return None for a model that has never been
        // fetched, and it must do so without touching the network. Using an
        // obviously-bogus mask guarantees the registry would 404 if it were
        // reached.
        let client = RegistryClient::default_client().unwrap();
        assert!(client
            .resolve_offline("definitely-not-a-real-model-xyzzy-42")
            .is_none());
    }

    #[test]
    fn test_resolve_offline_matches_is_extracted() {
        // resolve_offline is a thin Option wrapper over is_extracted: the two
        // must agree on whether a given model is locally available.
        let client = RegistryClient::default_client().unwrap();
        let mask = "nonexistent-model-12345";
        assert_eq!(
            client.resolve_offline(mask).is_some(),
            client.is_extracted(mask)
        );
    }

    #[test]
    fn test_resolve_offline_returns_extraction_dir() {
        // When resolve_offline does return Some, the path must match
        // extraction_dir() so callers can rely on it as a base_path for
        // TemplateExecutor. We verify the shape of the path for a known
        // mask — whether or not the directory physically exists.
        let client = RegistryClient::default_client().unwrap();
        let mask = "some-model";
        let expected = client.extraction_dir(mask);
        if let Some(actual) = client.resolve_offline(mask) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_offline_error_does_not_trip_circuit_breaker() {
        // When the local machine can't reach the registry (DNS/connect-refused),
        // the circuit breaker must NOT open. Opening it for 30s punishes the
        // user even after they come back online and poisons the cached-model
        // path because `can_execute()` would short-circuit before resolve_offline
        // has a chance to run in callers that consult the breaker state.
        let client = RegistryClient::with_url("https://primary.example.invalid").unwrap();
        let circuit = client.circuits[0].clone();
        assert!(circuit.is_closed(), "breaker starts closed");

        let mut op = |_url: &str| -> Result<ureq::Response, SdkError> {
            Err(SdkError::offline("simulated offline"))
        };

        let result =
            client.execute_with_retry_for_url("https://primary.example.invalid", &circuit, &mut op);
        assert!(matches!(result, Err(SdkError::Offline { .. })));
        assert_eq!(
            circuit.failure_count(),
            0,
            "breaker must not count offline errors toward the failure threshold"
        );
        assert!(
            circuit.is_closed(),
            "breaker must stay closed after offline errors"
        );
    }

    #[test]
    fn test_offline_error_short_circuits_retry_loop() {
        // A DNS failure is not going to recover in 2s, 4s, or 8s. The retry
        // loop must bail out after a single attempt rather than grinding
        // through the full exponential-backoff schedule.
        use std::sync::atomic::{AtomicU32, Ordering};

        let client = RegistryClient::with_url("https://primary.example.invalid").unwrap();
        let circuit = client.circuits[0].clone();
        let call_count = AtomicU32::new(0);

        let mut op = |_url: &str| -> Result<ureq::Response, SdkError> {
            call_count.fetch_add(1, Ordering::SeqCst);
            Err(SdkError::offline("simulated offline"))
        };

        let result =
            client.execute_with_retry_for_url("https://primary.example.invalid", &circuit, &mut op);
        assert!(result.is_err());
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "offline errors must not be retried within a single URL"
        );
    }

    #[test]
    fn registry_client_default_binding_is_rust() {
        let client = RegistryClient::default_client().unwrap();
        assert_eq!(client.binding(), DEFAULT_BINDING);
    }

    #[test]
    fn registry_client_with_binding_overrides_default() {
        let client = RegistryClient::default_client()
            .unwrap()
            .with_binding("flutter");
        assert_eq!(client.binding(), "flutter");
    }

    #[test]
    fn apply_client_header_sets_header_when_not_opted_out() {
        // Build a request through the helper that takes opted_out explicitly so
        // the test never touches the OnceLock-cached opt-out state owned by
        // `is_telemetry_opted_out`.
        let client = RegistryClient::with_url("http://127.0.0.1:1").unwrap();
        let req = client.agent.get("http://127.0.0.1:1/v1/models");
        let req = client.apply_client_header_with_optout(req, false);
        let header = req.header(CLIENT_HEADER_NAME);
        assert!(header.is_some(), "header must be set when opt-out is false");
        let value = header.unwrap();
        assert!(value.contains("binding=rust;"), "value: {}", value);
        assert!(value.contains("sdk_version="), "value: {}", value);
        assert!(value.contains("core_version="), "value: {}", value);
        assert!(value.contains("platform="), "value: {}", value);
        assert!(value.contains("backends="), "value: {}", value);
    }

    #[test]
    fn apply_client_header_omits_header_when_opted_out() {
        let client = RegistryClient::with_url("http://127.0.0.1:1").unwrap();
        let req = client.agent.get("http://127.0.0.1:1/v1/models");
        let req = client.apply_client_header_with_optout(req, true);
        assert_eq!(
            req.header(CLIENT_HEADER_NAME),
            None,
            "no header on the wire when telemetry is opted out"
        );
    }

    #[test]
    fn apply_client_header_uses_configured_binding() {
        let client = RegistryClient::with_url("http://127.0.0.1:1")
            .unwrap()
            .with_binding("flutter");
        let req = client.agent.get("http://127.0.0.1:1/v1/models");
        let req = client.apply_client_header_with_optout(req, false);
        let value = req.header(CLIENT_HEADER_NAME).unwrap();
        assert!(
            value.starts_with("binding=flutter;"),
            "configured binding must flow into the header: {}",
            value
        );
    }

    // ------------------------------------------------------------------------
    // Mock-server integration tests for header wiring.
    //
    // These exercise the actual HTTP path through `list_models`, `get_model`,
    // and `resolve` against a local httpmock instance. The opt-in/opt-out
    // matrix is covered by the in-process `apply_client_header_with_optout`
    // tests above (the OnceLock cache in `is_telemetry_opted_out` makes a
    // process-wide flip impractical here).
    // ------------------------------------------------------------------------

    #[test]
    fn metadata_calls_send_x_xybrid_client_header() {
        // Spins up a local httpmock server and exercises all three metadata
        // methods through the real wire path. Each mock matches against the
        // EXACT expected header value, computed from `build_client_header`
        // with the same binding the client is configured with. If the wired
        // header value differs in any field the mock won't match and the
        // request will 404 — so a regression in format or content surfaces
        // as a test failure, not a silent pass.
        use httpmock::prelude::*;

        let expected = build_client_header_with_optout("flutter", false)
            .expect("header must be built when not opted out");

        let server = MockServer::start();

        let list_mock = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models")
                .header(CLIENT_HEADER_NAME, expected.as_str());
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models": []}"#);
        });
        let get_mock = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models/test-model")
                .header(CLIENT_HEADER_NAME, expected.as_str());
            then.status(200).header("content-type", "application/json")
                .body(
                    r#"{"id":"test-model","family":"test","task":"text-generation","parameters":1,"description":"d","default_variant":null,"variants":{}}"#,
                );
        });
        let resolve_mock = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models/test-model/resolve")
                .query_param_exists("platform")
                .header(CLIENT_HEADER_NAME, expected.as_str());
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"mask":"test-model","platform":"x","resolved":{"hf_repo":"o/r","file":"u.xyb","download_url":"https://x","format":"onnx","quantization":"fp32","size_bytes":1,"sha256":""}}"#,
                );
        });

        let client = RegistryClient::with_url(server.base_url())
            .unwrap()
            .with_binding("flutter");

        client.list_models().expect("list_models should succeed");
        client
            .get_model("test-model")
            .expect("get_model should succeed");
        client
            .resolve("test-model", Some("apple-arm64-cpu"))
            .expect("resolve should succeed");

        list_mock.assert();
        get_mock.assert();
        resolve_mock.assert();

        // Independent format check on the expected value to ensure the
        // exact-match assertion above is meaningful (not e.g. the empty string).
        assert!(expected.starts_with("binding=flutter;"), "{}", expected);
        assert!(expected.contains("sdk_version="), "{}", expected);
        assert!(expected.contains("core_version="), "{}", expected);
        assert!(expected.contains("platform="), "{}", expected);
        assert!(expected.contains("backends="), "{}", expected);
    }

    #[test]
    fn metadata_calls_omit_header_when_opt_out_helper_returns_none() {
        // Mock-server companion to `apply_client_header_omits_header_when_opted_out`:
        // verifies that when the helper is invoked with `opted_out=true` (the
        // contract that `build_client_header` honors under
        // XYBRID_TELEMETRY_OPTOUT=1), no X-Xybrid-Client header reaches the
        // wire. We can't flip the process-wide OnceLock cache that
        // `is_telemetry_opted_out` keeps, so this test exercises the wire path
        // through `apply_client_header_with_optout(_, true)` directly. The
        // mock fails the match if the header is present (header(name, "")
        // requires equality, which a missing header satisfies as None).
        use httpmock::prelude::*;

        let server = MockServer::start();

        // A permissive mock: matches any request to /v1/models. We then
        // inspect the mock's hit count to confirm the request reached it
        // (i.e. wasn't rejected) and rely on the absence of any
        // header-asserting mock. To assert no header, we set up a SECOND
        // mock that REQUIRES the header — if that one fires, the wire
        // carried the header and the test fails.
        let permissive = server.mock(|when, then| {
            when.method(GET).path("/v1/models");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models": []}"#);
        });
        let with_header = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models")
                .header_exists(CLIENT_HEADER_NAME);
            then.status(599).body("UNEXPECTED HEADER");
        });

        let client = RegistryClient::with_url(server.base_url()).unwrap();
        let url = format!("{}/v1/models", server.base_url());
        let req = client.apply_client_header_with_optout(client.agent.get(&url), true);
        let response = req.call().expect("request should reach mock server");
        assert_eq!(response.status(), 200);

        permissive.assert();
        assert_eq!(
            with_header.hits(),
            0,
            "X-Xybrid-Client header must NOT be sent when opted out"
        );
    }
}
