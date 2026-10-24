//! Mock native calls exercise the safe surface without an engine archive.

use super::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Default)]
pub(crate) struct MockState {
    calls: Mutex<Vec<&'static str>>,
    closed: AtomicUsize,
    cancelled: AtomicBool,
}

impl MockState {
    fn record(&self, call: &'static str) {
        self.calls.lock().unwrap().push(call);
    }
}

thread_local! {
    static STATE: RefCell<Arc<MockState>> = RefCell::new(Arc::default());
}

#[cfg(feature = "android-companion")]
pub(crate) fn recorded_call_count() -> usize {
    let state = state();
    let count = state.calls.lock().unwrap().len();
    count
}

fn state() -> Arc<MockState> {
    STATE.with(|s| Arc::clone(&s.borrow()))
}

pub(crate) fn with_state<T>(run: impl FnOnce(Arc<MockState>) -> T) -> T {
    struct Restore(Arc<MockState>);
    impl Drop for Restore {
        fn drop(&mut self) {
            STATE.with(|slot| *slot.borrow_mut() = Arc::clone(&self.0));
        }
    }
    let next = Arc::new(MockState::default());
    let _restore = Restore(STATE.with(|slot| slot.replace(Arc::clone(&next))));
    run(next)
}

unsafe extern "C" fn abi() -> u32 {
    state().record("abi");
    crate::ZZZ_EMBED_ABI_VERSION
}
unsafe extern "C" fn supports(_: *const c_char) -> i32 {
    state().record("supports");
    1
}
unsafe extern "C" fn defaults(out: *mut EmbedOptions) {
    state().record("defaults");
    // SAFETY: safe wrapper supplies one caller-owned options value.
    unsafe {
        *out = EmbedOptions {
            struct_size: std::mem::size_of::<EmbedOptions>() as u32,
            max_frames_per_chunk: 8,
            max_chunk_tokens: 16,
            temperature: 0.5,
            seed: 0,
        };
    }
}
unsafe fn open_session(out: *mut *mut c_void) -> i32 {
    // SAFETY: open supplies valid output storage, owning the returned box.
    unsafe {
        *out = Box::into_raw(Box::new(state())).cast();
    }
    crate::ZZZ_EMBED_OK
}
unsafe extern "C" fn open(
    _: *const c_char,
    _: *const c_char,
    _: *const c_char,
    _: *const EmbedOptions,
    out: *mut *mut c_void,
    _: *mut EmbedError,
) -> i32 {
    state().record("open");
    // SAFETY: forwarded mock session output follows the same ownership ABI.
    unsafe { open_session(out) }
}
unsafe extern "C" fn open_kitten(
    _: *const KittenOptions,
    out: *mut *mut c_void,
    _: *mut EmbedError,
) -> i32 {
    state().record("open_kitten");
    // SAFETY: forwarded mock session output follows the same ownership ABI.
    unsafe { open_session(out) }
}
unsafe extern "C" fn synthesize(
    session: *mut c_void,
    _: *const c_char,
    _: *const c_char,
    callback: Option<AudioCallback>,
    user: *mut c_void,
    result: *mut EmbedResult,
    _: *mut EmbedError,
) -> i32 {
    // SAFETY: successful mock open allocates this Arc; handle ownership anchors it.
    let state = unsafe { &*session.cast::<Arc<MockState>>() };
    state.record("synthesize");
    state.cancelled.store(false, Ordering::SeqCst);
    let samples = [0.25f32, 0.5, -0.5, -0.25];
    // SAFETY: callback and output storage originate in the safe synchronous call;
    // the stack samples are borrowed only for callback return.
    let cancel = unsafe {
        *result = EmbedResult {
            sample_rate: 24_000,
            channels: 1,
            chunks: 1,
            samples: samples.len() as u64,
            limited_chunks: 0,
        };
        callback.unwrap()(samples.as_ptr(), samples.len(), 24_000, 1, 0, user)
    };
    if cancel != 0 || state.cancelled.load(Ordering::SeqCst) {
        crate::ZZZ_EMBED_CANCELLED
    } else {
        crate::ZZZ_EMBED_OK
    }
}
unsafe extern "C" fn cancel(session: *mut c_void) {
    // SAFETY: cancellation's shared SessionHandle keeps this allocation alive.
    let state = unsafe { &*session.cast::<Arc<MockState>>() };
    state.record("cancel");
    state.cancelled.store(true, Ordering::SeqCst);
}
unsafe extern "C" fn close(session: *mut c_void) {
    // SAFETY: final SessionHandle owns exactly this allocation, closed once.
    let state = unsafe { Box::from_raw(session.cast::<Arc<MockState>>()) };
    state.record("close");
    state.closed.fetch_add(1, Ordering::SeqCst);
}

pub(crate) fn mock_functions() -> Functions {
    Functions {
        abi_version: abi,
        supports_model: supports,
        default_options: defaults,
        open,
        open_kitten,
        synthesize,
        cancel,
        close,
    }
}

pub(crate) fn mock_api() -> Arc<EngineApi> {
    Arc::new(EngineApi {
        functions: mock_functions(),
        #[cfg(all(feature = "android-companion", any(not(feature = "bindings"), test)))]
        _library: None,
    })
}

// Every controlled mock path, including a new cancellation thread, establishes
// its own fake supported facts inside an explicit denied scope. This proves
// that the tests do not depend on host v3 support and that denial is restored.
pub(crate) fn with_supported_mock_cpu<T>(run: impl FnOnce() -> T) -> T {
    crate::cpu::without_v3(|| {
        let result = crate::cpu::with_supported_cpu(run);
        assert!(!crate::cpu::available());
        result
    })
}

#[test]
fn all_eight_calls_use_the_owned_table_and_cancel_keeps_session_alive() {
    use crate::{KittenSession, KittenSettings, PhononSession, PhononSettings, SynthesisStatus};
    use std::path::Path;
    with_supported_mock_cpu(|| {
        with_state(|state| {
            let api = mock_api();
            let weak = Arc::downgrade(&api);
            let (mut phonon, mut kitten) = with_test_api(Ok(api), || {
                assert_eq!(crate::abi_version(), 1);
                assert!(crate::supports_model("phonon"));
                assert_eq!(crate::default_options().max_chunk_tokens, 16);
                let path = Path::new("mock.gguf");
                (
                    PhononSession::open_phonon(path, None, PhononSettings::default()).unwrap(),
                    KittenSession::open_kitten(path, path, path, path, KittenSettings::default())
                        .unwrap(),
                )
            });
            let cancel = phonon.cancellation_handle();
            // A changed global availability result must never replace a session's table.
            with_test_api(Err(UnavailableReason::Unconfigured), || {
                let cancelled = phonon
                    .synthesize_stream_outcome("hello", None, &mut |samples, _| {
                        assert_eq!(samples.len(), 4);
                        cancel.cancel();
                        true
                    })
                    .unwrap();
                assert_eq!(cancelled.status, SynthesisStatus::Cancelled);
                assert_eq!(phonon.synthesize("reuse", None).unwrap().samples.len(), 4);
                assert_eq!(kitten.synthesize("kitten", None).unwrap().samples.len(), 4);
            });
            drop(phonon);
            assert_eq!(state.closed.load(Ordering::SeqCst), 0);
            // Cancellation may be moved to another thread after the session is gone.
            std::thread::spawn(move || {
                with_supported_mock_cpu(|| {
                    cancel.cancel();
                    drop(cancel);
                })
            })
            .join()
            .unwrap();
            assert_eq!(state.closed.load(Ordering::SeqCst), 1);
            assert!(weak.upgrade().is_some());
            drop(kitten);
            assert_eq!(state.closed.load(Ordering::SeqCst), 2);
            assert!(weak.upgrade().is_none());
            let calls = state.calls.lock().unwrap();
            for name in [
                "abi",
                "supports",
                "defaults",
                "open",
                "open_kitten",
                "synthesize",
                "cancel",
                "close",
            ] {
                assert!(calls.contains(&name), "missing {name}");
            }
        })
    });
}

#[test]
fn every_unavailable_state_rejects_before_paths_defaults_or_native_calls() {
    use crate::{KittenSession, KittenSettings, PhononSession, PhononSettings};
    use std::path::Path;
    with_supported_mock_cpu(|| {
        with_state(|state| {
            for reason in [
                UnavailableReason::UnsupportedPlatform,
                UnavailableReason::UnsupportedAbi,
                UnavailableReason::Unconfigured,
                UnavailableReason::UnknownAndroidApi,
                UnavailableReason::UnsupportedAndroidApi,
                UnavailableReason::LibraryNotFound,
                UnavailableReason::GetterNotFound,
                UnavailableReason::GetterRejected,
                UnavailableReason::InvalidTable,
                UnavailableReason::EngineAbiMismatch,
                UnavailableReason::MissingModels,
            ] {
                with_test_api(Err(reason), || {
                    assert_eq!(availability(), Err(reason));
                    assert_eq!(crate::abi_version(), 0);
                    assert!(!crate::supports_model("phonon"));
                    assert_eq!(crate::default_options().max_chunk_tokens, 0);
                    let path = Path::new("invalid\0path");
                    let phonon = PhononSession::open_phonon(
                        path,
                        Some("invalid\0voice"),
                        PhononSettings::default(),
                    );
                    let kitten = KittenSession::open_kitten(
                        path,
                        path,
                        path,
                        path,
                        KittenSettings::default(),
                    );
                    for result in [phonon.map(|_| ()), kitten.map(|_| ())] {
                        assert!(matches!(result, Err(crate::ZzzError::LoadFailed { .. })));
                    }
                });
            }
            assert!(state.calls.lock().unwrap().is_empty());
        })
    });
}
