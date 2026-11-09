//! The final transport boundary (D24, D26, D28): one HTTP request flow per
//! body, sharing the export's absolute deadline across attempts, backoff,
//! `Retry-After` waits and the response read. Redirects are never followed.
//! A destination's response hook sees every response here, once, and may
//! disable the destination (see [`super::hook`]).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, RETRY_AFTER};
use reqwest::{Client, StatusCode, Url};
use tokio::time::{timeout_at, Instant};
use tokio_util::sync::CancellationToken;

use super::config::{Protocol, ResolvedSignal, RetryLimits};
use super::defaults;
use super::encode::{self, Signal};
use super::hook::{ExportResponse, ResponseAction, ResponseHook};
use super::runtime::tokio_after;
use crate::error::{Error, Result};

/// Builds the shared HTTP client: rustls, HTTP/1.1, no redirects, no proxy
/// discovery, no client-level timeout (each attempt carries its own).
pub(crate) fn client() -> Result<Client> {
    Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .http1_only()
        .pool_max_idle_per_host(2)
        .pool_idle_timeout(Duration::from_secs(30))
        .build()
        .map_err(|_| Error::Lifecycle {
            operation: "build",
            message: "could not build the HTTP client".into(),
        })
}

/// How one body ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Delivery {
    /// A 2xx status; `rejected` items were refused by OTLP partial success.
    Acknowledged { rejected: u64 },
    /// No acknowledgement (status, transport error, deadline, cancellation).
    Failed,
    /// Not attempted: the destination was disabled by its response hook.
    Disabled,
}

/// One signal's destination: the URL, the resolved headers, the retry
/// policy, its response hook and whether that hook disabled it. The only
/// code that builds requests. One is built per pipeline (or exporter set),
/// so a new build starts enabled.
#[derive(Clone, Debug)]
pub(crate) struct Sender {
    client: Client,
    url: Url,
    headers: HeaderMap,
    protocol: Protocol,
    timeout: Duration,
    retry: RetryLimits,
    signal: Signal,
    hook: Option<ResponseHook>,
    disabled: Arc<AtomicBool>,
}

impl Sender {
    pub(crate) fn new(client: Client, resolved: &ResolvedSignal, retry: RetryLimits) -> Self {
        Self {
            client,
            url: resolved.url.clone(),
            headers: resolved.headers.clone(),
            protocol: resolved.protocol,
            timeout: resolved.timeout,
            retry,
            signal: resolved.signal,
            hook: resolved.hook.clone(),
            disabled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Whether the response hook disabled this destination.
    pub(crate) fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Acquire)
    }

    fn disable(&self) {
        if !self.disabled.swap(true, Ordering::AcqRel) {
            log::warn!(
                "xybrid-observability: the {} destination was disabled by its response hook; \
                 pending and later items are dropped",
                self.signal.as_str()
            );
        }
    }

    pub(crate) fn protocol(&self) -> Protocol {
        self.protocol
    }

    pub(crate) fn signal(&self) -> Signal {
        self.signal
    }

    /// Delivers one body, retrying within `deadline`. Cancelling `cancel`
    /// aborts the in-flight request or backoff at once.
    pub(crate) async fn deliver(
        &self,
        body: Vec<u8>,
        deadline: Instant,
        cancel: &CancellationToken,
    ) -> Delivery {
        let signal = self.signal.as_str();
        let content_type = HeaderValue::from_static(self.protocol.content_type());
        let mut retry = 0usize;
        let mut attempted = false;
        loop {
            if self.is_disabled() {
                // Items whose delivery was already attempted fail; untried
                // ones are the caller's to drop.
                return if attempted {
                    Delivery::Failed
                } else {
                    Delivery::Disabled
                };
            }
            let now = Instant::now();
            if cancel.is_cancelled() || now >= deadline {
                return Delivery::Failed;
            }
            // D28.2: each attempt's timeout is min(signal timeout, remaining).
            let attempt_deadline = tokio_after(now, self.timeout).min(deadline);
            // D24: exactly the resolved headers plus the transport-owned
            // Content-Type. (The HTTP client still supplies framing headers
            // and its generic `Accept: */*` when none was configured.)
            let mut headers = self.headers.clone();
            headers.insert(CONTENT_TYPE, content_type.clone());
            let request = match self
                .client
                .post(self.url.clone())
                .headers(headers)
                .body(body.clone())
                .build()
            {
                Ok(request) => request,
                Err(_) => {
                    log::warn!("xybrid-observability: {signal} export request could not be built");
                    return Delivery::Failed;
                }
            };
            attempted = true;
            let sent = cancel
                .run_until_cancelled(timeout_at(attempt_deadline, self.client.execute(request)))
                .await;
            let (retryable, retry_after) = match sent {
                None => return Delivery::Failed,
                Some(Err(_elapsed)) => {
                    log::debug!("xybrid-observability: {signal} export attempt timed out");
                    (true, None)
                }
                Some(Ok(Err(_))) => {
                    // Transport errors never include payloads or headers.
                    log::debug!("xybrid-observability: {signal} export transport error");
                    (true, None)
                }
                Some(Ok(Ok(response))) => {
                    let status = response.status();
                    let wait = retry_after(response.headers());
                    // A hook sees the headers and the bounded body of every
                    // response; without one, only a 2xx body is read.
                    let headers = self.hook.as_ref().map(|_| response.headers().clone());
                    let body = if status.is_success() || headers.is_some() {
                        match cancel
                            .run_until_cancelled(timeout_at(deadline, read_capped(response)))
                            .await
                        {
                            Some(Ok(body)) => body,
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if let (Some(hook), Some(headers)) = (&self.hook, &headers) {
                        let seen = ExportResponse::new(
                            self.signal,
                            self.protocol,
                            status.as_u16(),
                            headers,
                            body.as_deref(),
                        );
                        if hook.call(&seen) == ResponseAction::Disable {
                            self.disable();
                        }
                    }
                    if status.is_success() {
                        let rejected = body.map_or(0, |bytes| {
                            encode::rejected(self.signal, self.protocol, &bytes)
                        });
                        return Delivery::Acknowledged { rejected };
                    }
                    log::warn!(
                        "xybrid-observability: {signal} export failed with status {}",
                        status.as_u16()
                    );
                    if !is_retryable(status) {
                        return Delivery::Failed;
                    }
                    (true, wait)
                }
            };
            if !retryable || retry >= self.retry.max_retries {
                return Delivery::Failed;
            }
            let wait = retry_after.unwrap_or_else(|| backoff(&self.retry, retry));
            if tokio_after(Instant::now(), wait) >= deadline {
                // The wait would reach the deadline: fail now rather than
                // sleep into it.
                return Delivery::Failed;
            }
            if cancel
                .run_until_cancelled(tokio::time::sleep(wait))
                .await
                .is_none()
            {
                return Delivery::Failed;
            }
            retry += 1;
        }
    }
}

fn is_retryable(status: StatusCode) -> bool {
    defaults::RETRYABLE_STATUS.contains(&status.as_u16())
}

/// `min(initial_delay × 2^n, max_delay)`, no jitter.
pub(crate) fn backoff(retry: &RetryLimits, n: usize) -> Duration {
    let factor = 1u32
        .checked_shl(u32::try_from(n).unwrap_or(u32::MAX))
        .unwrap_or(u32::MAX);
    retry
        .initial_delay
        .checked_mul(factor)
        .unwrap_or(retry.max_delay)
        .min(retry.max_delay)
}

/// `Retry-After` as delta-seconds (decimals accepted) or an HTTP-date.
pub(crate) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
        let seconds: f64 = value.parse().ok()?;
        return (seconds.is_finite() && seconds >= 0.0)
            .then(|| Duration::from_secs_f64(seconds.min(1e9)));
    }
    let when = httpdate::parse_http_date(value).ok()?;
    Some(
        when.duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO),
    )
}

/// Reads at most `max_response_body_bytes`; `None` when the body is larger
/// or the read fails (the body is then treated as undecodable).
async fn read_capped(mut response: reqwest::Response) -> Option<Vec<u8>> {
    let mut body = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if body.len() + chunk.len() > defaults::MAX_RESPONSE_BODY_BYTES {
                    return None;
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => return Some(body),
            Err(_) => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_up_to_the_cap() {
        let retry = RetryLimits {
            max_retries: 10,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(1000),
        };
        let waits: Vec<_> = (0..6).map(|n| backoff(&retry, n).as_millis()).collect();
        assert_eq!(waits, [100, 200, 400, 800, 1000, 1000]);
        assert_eq!(backoff(&retry, 200), Duration::from_millis(1000));
    }

    #[test]
    fn retry_after_accepts_seconds_decimals_and_dates() {
        let header = |v: &str| {
            let mut h = HeaderMap::new();
            h.insert(RETRY_AFTER, HeaderValue::from_str(v).unwrap());
            retry_after(&h)
        };
        assert_eq!(header("3"), Some(Duration::from_secs(3)));
        assert_eq!(header("0.5"), Some(Duration::from_millis(500)));
        assert_eq!(header("-1"), None);
        assert_eq!(header("1e3"), None);
        assert_eq!(header("soon"), None);
        let future = SystemTime::now() + Duration::from_secs(120);
        let wait = header(&httpdate::fmt_http_date(future)).unwrap();
        assert!(wait > Duration::from_secs(100) && wait <= Duration::from_secs(120));
        let past = SystemTime::now() - Duration::from_secs(120);
        assert_eq!(header(&httpdate::fmt_http_date(past)), Some(Duration::ZERO));
    }
}
