// Flutter FFI API module
// Thin wrappers with #[frb] attributes that delegate to xybrid-sdk

/// Binding identifier reported in the `X-Xybrid-Client` registry header
/// for Flutter apps. Routed through `xybrid_sdk::set_binding` at every
/// FFI entry so registry calls are attributed correctly even on
/// platforms that skip `init_sdk_cache_dir` (iOS/macOS), and on entry
/// points the host hits before `init_sdk_cache_dir` is called (the
/// push-state setters in [`device`]).
pub(crate) const FLUTTER_BINDING: &str = "flutter";

/// Install the platform-native `log` backend and a panic logger, once per
/// process.
///
/// Runs from `XybridRustLib.init()` through the `#[frb(init)]` hook
/// [`sdk_client::init_native_logging`], so every app gets native logs from the
/// first call. It used to run only from `initSdkCacheDir`, `setApiKey` and the
/// telemetry setters, and `Xybrid.init` calls `initSdkCacheDir` on Android
/// only: an iOS app without an API key got no `dev.xybrid.sdk` logs at all.
///
/// The panic hook matters because every streaming call runs on a detached
/// worker thread, where a panic would otherwise vanish. No logger on desktop
/// targets, where the host process owns logger setup; the panic hook is
/// installed everywhere.
pub(crate) fn ensure_native_logging() {
    static LOGGING_INIT: std::sync::Once = std::sync::Once::new();
    LOGGING_INIT.call_once(|| {
        #[cfg(target_os = "android")]
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(log::LevelFilter::Info)
                .with_tag("xybrid"),
        );
        #[cfg(target_os = "ios")]
        {
            // Errors only if a logger is already registered — fine to ignore.
            let _ = oslog::OsLogger::new("dev.xybrid.sdk")
                .level_filter(log::LevelFilter::Info)
                .init();
        }
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            log::error!("xybrid RUST PANIC: {info}");
            previous(info);
        }));
    });
}

/// Run `work` on a detached thread, handing a panic's message to
/// `report_panic` instead of letting it end the thread silently.
///
/// Every streaming call works this way: `work` owns a Dart `StreamSink`, and
/// a panic drops it, which closes the Dart stream with no error event. A
/// caller waiting for `Complete` or `Error` then reads the bare close as
/// success. `report_panic` sends the stream's error event instead.
pub(crate) fn spawn_reporting_panics<W, R>(work: W, report_panic: R)
where
    W: FnOnce() + Send + 'static,
    R: FnOnce(String) + Send + 'static,
{
    std::thread::spawn(move || {
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
            report_panic(format!("xybrid panicked: {}", panic_message(&*payload)));
        }
    });
}

/// The message a panic was raised with, when it carried one.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message
    } else {
        "unknown cause"
    }
}

pub mod context;
pub mod device;
pub mod envelope;
pub mod model;
pub mod pipeline;
pub mod result;
pub mod sdk_client;
pub mod streaming;

// Re-export all public types for convenient access
pub use context::{FfiConversationContext, FfiMessageRole};
pub use device::{FfiThermalState, XybridDevice};
pub use envelope::FfiEnvelope;
pub use model::{
    FfiGenerationConfig, FfiModel, FfiModelLoader, FfiRunOptions, FfiStreamEvent, FfiStreamToken,
};
pub use pipeline::FfiPipeline;
pub use result::FfiResult;
pub use streaming::{FfiPartialResult, FfiStreamSession, FfiStreamingConfig, FfiVadMode};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn a_worker_panic_reaches_the_reporter() {
        let (tx, rx) = mpsc::channel();
        spawn_reporting_panics(
            || panic!("tokenizer exploded"),
            move |message| tx.send(message).expect("test receiver is alive"),
        );
        let message = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("a panicking worker must report");
        assert_eq!(message, "xybrid panicked: tokenizer exploded");
    }

    #[test]
    fn a_worker_that_finishes_reports_nothing() {
        let (tx, rx) = mpsc::channel::<String>();
        let (done_tx, done_rx) = mpsc::channel();
        spawn_reporting_panics(
            move || done_tx.send(()).expect("test receiver is alive"),
            move |message| tx.send(message).expect("test receiver is alive"),
        );
        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the worker should run");
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
    }

    #[test]
    fn formatted_panic_messages_are_kept() {
        let payload = std::panic::catch_unwind(|| panic!("bad index {}", 3)).unwrap_err();
        assert_eq!(panic_message(&*payload), "bad index 3");
    }
}
