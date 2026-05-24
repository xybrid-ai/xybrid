//! Regression test for the streaming-trampoline error round-trip path.
//!
//! Moved verbatim (semantics-wise) from
//! `xybrid-core/src/runtime_adapter/llama_cpp/sys.rs` (the prior
//! `streaming_trampoline_preserves_cloud_fallback_abort_marker` test).
//!
//! The original test downcast a `CloudFallbackAbort` type owned by
//! `xybrid-core`. We can't depend on `xybrid-core` here (would cycle), so
//! the test uses a stand-in `MarkerError` type that exercises the
//! same property: any `Box<dyn Error + Send + Sync>` returned from the
//! per-token closure must survive the trampoline's storage round-trip and
//! the trampoline must signal "stop" to the C side within the M-series
//! cancellation budget (≤ 50 ms).
//!
//! `xybrid-core`'s end-to-end variant (which actually downcasts
//! `CloudFallbackAbort`) stays where it is and validates the integration
//! tier; the move here covers the boxed-error preservation contract that
//! `xybrid-llama` is the authority for.

#![cfg(feature = "bindings")]

use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::os::raw::c_void;
use std::time::{Duration, Instant};

// Re-export of the trampoline + its context struct from the crate's
// `generation` module. Tests in `tests/` are out-of-crate consumers, so
// we go through `pub(crate)` ⇒ this exercises an integration-shaped
// path. We surface the trampoline + context via a small `__test_hooks`
// module on the crate so the test isn't tied to internal layout.
use xybrid_llama::__test_hooks::{streaming_trampoline, StreamingContext};

#[derive(Debug)]
struct MarkerError(&'static str);

impl fmt::Display for MarkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for MarkerError {}

type Callback = fn(i32, &str) -> Result<(), Box<dyn Error + Send + Sync>>;

fn abort_callback(_token_id: i32, _text: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    Err(Box::new(MarkerError("marker preserved through trampoline")))
}

#[test]
fn streaming_trampoline_preserves_boxed_error_marker() {
    let mut callback: Callback = abort_callback;
    let mut ctx = StreamingContext {
        callback: &mut callback,
        error: None,
    };
    let token_text = CString::new("token").unwrap();

    let started = Instant::now();
    // SAFETY: ctx is a valid `&mut StreamingContext<Callback>`, the
    // token_text CString lives for the duration of the call, and the
    // callback pointer in ctx is live.
    let stop = unsafe {
        streaming_trampoline::<Callback>(
            42,
            token_text.as_ptr(),
            &mut ctx as *mut StreamingContext<Callback> as *mut c_void,
        )
    };
    let elapsed = started.elapsed();

    assert_eq!(stop, 1, "callback errors must stop the C stream");
    assert!(
        elapsed <= Duration::from_millis(50),
        "llama.cpp trampoline abort exceeded M-series cancellation budget: {:?}",
        elapsed
    );
    let err = ctx.error.take().expect("callback error must be stored");
    let downcast: &MarkerError = err
        .downcast_ref::<MarkerError>()
        .expect("typed marker must survive the trampoline boundary");
    assert_eq!(downcast.0, "marker preserved through trampoline");
}
