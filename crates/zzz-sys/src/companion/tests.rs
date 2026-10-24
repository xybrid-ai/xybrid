use super::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(feature = "bindings"))]
use std::sync::OnceLock;

const MOCK_CONTRACT: [u8; 32] = [0xa5; 32];

fn valid_table() -> CompanionApiV1 {
    let f = crate::api::tests::mock_functions();
    CompanionApiV1 {
        struct_size: std::mem::size_of::<CompanionApiV1>() as u32,
        companion_abi: COMPANION_ABI,
        engine_abi: crate::ZZZ_EMBED_ABI_VERSION,
        android_api: ANDROID_API,
        models: SELECTED_MODELS,
        reserved: 0,
        contract_id: MOCK_CONTRACT,
        abi_version: Some(f.abi_version),
        supports_model: Some(f.supports_model),
        default_options: Some(f.default_options),
        open: Some(f.open),
        open_kitten: Some(f.open_kitten),
        synthesize: Some(f.synthesize),
        cancel: Some(f.cancel),
        close: Some(f.close),
    }
}

thread_local! {
    static TABLE: Cell<CompanionApiV1> = Cell::new(valid_table());
    static GETTER_STATUS: Cell<i32> = const { Cell::new(0) };
}

unsafe extern "C" fn getter(abi: u32, size: u32, out: *mut CompanionApiV1) -> i32 {
    assert_eq!(abi, COMPANION_ABI);
    assert_eq!(size as usize, std::mem::size_of::<CompanionApiV1>());
    // SAFETY: load_with passes exactly one caller-owned table of the stated size.
    unsafe {
        *out = TABLE.with(Cell::get);
    }
    GETTER_STATUS.with(Cell::get)
}

struct RestoreTable(CompanionApiV1, i32);
impl Drop for RestoreTable {
    fn drop(&mut self) {
        TABLE.with(|slot| slot.set(self.0));
        GETTER_STATUS.with(|slot| slot.set(self.1));
    }
}
fn table_scope(table: CompanionApiV1, status: i32) -> RestoreTable {
    RestoreTable(
        TABLE.with(|slot| slot.replace(table)),
        GETTER_STATUS.with(|slot| slot.replace(status)),
    )
}

struct FakeLoader {
    api: Option<i32>,
    fail_open: bool,
    fail_getter: bool,
    probes: AtomicUsize,
    opens: AtomicUsize,
    getters: AtomicUsize,
    drops: Arc<AtomicUsize>,
}
impl Default for FakeLoader {
    fn default() -> Self {
        Self {
            api: Some(29),
            fail_open: false,
            fail_getter: false,
            probes: AtomicUsize::new(0),
            opens: AtomicUsize::new(0),
            getters: AtomicUsize::new(0),
            drops: Arc::default(),
        }
    }
}
impl Loader for FakeLoader {
    fn device_api(&self) -> Option<i32> {
        self.probes.fetch_add(1, Ordering::SeqCst);
        self.api
    }
    fn open(&self) -> Result<Library, UnavailableReason> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        if self.fail_open {
            return Err(UnavailableReason::LibraryNotFound);
        }
        Ok(Library {
            #[cfg(all(
                target_os = "android",
                target_arch = "aarch64",
                target_pointer_width = "64"
            ))]
            handle: None,
            drop_count: Arc::clone(&self.drops),
        })
    }
    fn getter(&self, _: &Library) -> Result<Getter, UnavailableReason> {
        self.getters.fetch_add(1, Ordering::SeqCst);
        if self.fail_getter {
            Err(UnavailableReason::GetterNotFound)
        } else {
            Ok(getter)
        }
    }
}

fn load(loader: &FakeLoader) -> Result<Arc<EngineApi>, UnavailableReason> {
    load_with(Platform::AndroidArm64, Some(MOCK_CONTRACT), loader)
}
fn reason(result: Result<Arc<EngineApi>, UnavailableReason>) -> UnavailableReason {
    match result {
        Err(reason) => reason,
        Ok(_) => panic!("must reject"),
    }
}

#[test]
fn unsupported_platform_abi_and_unconfigured_contract_never_probe_or_open() {
    let loader = FakeLoader::default();
    assert_eq!(
        reason(load_with(Platform::OtherOs, Some(MOCK_CONTRACT), &loader)),
        UnavailableReason::UnsupportedPlatform
    );
    assert_eq!(
        reason(load_with(
            Platform::OtherAndroidAbi,
            Some(MOCK_CONTRACT),
            &loader
        )),
        UnavailableReason::UnsupportedAbi
    );
    assert_eq!(
        reason(load_with(
            Platform::AndroidArm64,
            EXPECTED_CONTRACT,
            &loader
        )),
        UnavailableReason::Unconfigured
    );
    assert_eq!(loader.probes.load(Ordering::SeqCst), 0);
    assert_eq!(loader.opens.load(Ordering::SeqCst), 0);
    assert_eq!(loader.getters.load(Ordering::SeqCst), 0);
}

#[test]
fn old_unknown_and_negative_actual_device_apis_never_open_the_companion() {
    for api in [None, Some(-1), Some(21), Some(24), Some(28)] {
        let loader = FakeLoader {
            api,
            ..FakeLoader::default()
        };
        let error = reason(load(&loader));
        assert_eq!(
            error,
            if api.is_none() || api == Some(-1) {
                UnavailableReason::UnknownAndroidApi
            } else {
                UnavailableReason::UnsupportedAndroidApi
            }
        );
        assert_eq!(loader.probes.load(Ordering::SeqCst), 1);
        assert_eq!(loader.opens.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn every_metadata_field_and_null_function_slot_is_checked_before_probes() {
    for i in 0..15 {
        let mut table = valid_table();
        match i {
            0 => table.struct_size -= 1,
            1 => table.companion_abi += 1,
            2 => table.engine_abi += 1,
            3 => table.android_api -= 1,
            4 => table.models |= 4,
            5 => table.reserved = 1,
            6 => table.contract_id[0] ^= 1,
            7 => table.abi_version = None,
            8 => table.supports_model = None,
            9 => table.default_options = None,
            10 => table.open = None,
            11 => table.open_kitten = None,
            12 => table.synthesize = None,
            13 => table.cancel = None,
            14 => table.close = None,
            _ => unreachable!(),
        }
        let _restore = table_scope(table, 0);
        let loader = FakeLoader::default();
        let calls_before = crate::api::tests::recorded_call_count();
        assert_eq!(
            reason(load(&loader)),
            UnavailableReason::InvalidTable,
            "case {i}"
        );
        assert_eq!(crate::api::tests::recorded_call_count(), calls_before);
        assert_eq!(loader.drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn loader_getter_and_handshake_failures_release_only_unaccepted_handles() {
    let loader = FakeLoader {
        fail_open: true,
        ..FakeLoader::default()
    };
    assert_eq!(reason(load(&loader)), UnavailableReason::LibraryNotFound);
    assert_eq!(loader.getters.load(Ordering::SeqCst), 0);
    assert_eq!(loader.drops.load(Ordering::SeqCst), 0);
    let loader = FakeLoader {
        fail_getter: true,
        ..FakeLoader::default()
    };
    assert_eq!(reason(load(&loader)), UnavailableReason::GetterNotFound);
    assert_eq!(loader.drops.load(Ordering::SeqCst), 1);
    let _restore = table_scope(valid_table(), -1);
    let loader = FakeLoader::default();
    assert_eq!(reason(load(&loader)), UnavailableReason::GetterRejected);
    assert_eq!(loader.drops.load(Ordering::SeqCst), 1);
}

unsafe extern "C" fn wrong_abi() -> u32 {
    2
}
unsafe extern "C" fn missing_model(_: *const std::ffi::c_char) -> i32 {
    0
}

#[test]
fn actual_engine_probes_must_agree_with_accepted_metadata() {
    for (table, expected) in [
        (
            CompanionApiV1 {
                abi_version: Some(wrong_abi),
                ..valid_table()
            },
            UnavailableReason::EngineAbiMismatch,
        ),
        (
            CompanionApiV1 {
                supports_model: Some(missing_model),
                ..valid_table()
            },
            UnavailableReason::MissingModels,
        ),
    ] {
        let _restore = table_scope(table, 0);
        let loader = FakeLoader::default();
        assert_eq!(reason(load(&loader)), expected);
        assert_eq!(loader.drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn accepted_library_lives_through_session_and_last_cancellation_owner() {
    use crate::{KittenSession, KittenSettings};
    use std::path::Path;
    crate::api::tests::with_supported_mock_cpu(|| {
        crate::api::tests::with_state(|_| {
            let loader = FakeLoader::default();
            let api = load(&loader).unwrap();
            let mut session = crate::api::with_test_api(Ok(api), || {
                let path = Path::new("mock.gguf");
                KittenSession::open_kitten(path, path, path, path, KittenSettings::default())
                    .unwrap()
            });
            let cancel = session.cancellation_handle();
            assert_eq!(loader.drops.load(Ordering::SeqCst), 0);
            session.synthesize("hello", None).unwrap();
            drop(session);
            assert_eq!(loader.drops.load(Ordering::SeqCst), 0);
            cancel.cancel();
            drop(cancel);
            assert_eq!(loader.drops.load(Ordering::SeqCst), 1);
        })
    });
}

#[cfg(not(feature = "bindings"))]
#[test]
fn simultaneous_first_use_initializes_once_and_caches_success_and_failure() {
    for succeeds in [false, true] {
        let loader = Arc::new(FakeLoader {
            fail_open: !succeeds,
            ..FakeLoader::default()
        });
        let cache = Arc::new(OnceLock::new());
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let loader = Arc::clone(&loader);
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    crate::api::cached_load(&cache, || load(&loader)).is_ok()
                })
            })
            .collect();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), succeeds);
        }
        assert_eq!(loader.probes.load(Ordering::SeqCst), 1);
        assert_eq!(loader.opens.load(Ordering::SeqCst), 1);
        assert_eq!(loader.getters.load(Ordering::SeqCst), usize::from(succeeds));
        // A success remains loaded while cached, without requiring a live session.
        assert_eq!(loader.drops.load(Ordering::SeqCst), 0);
        drop(cache);
        assert_eq!(loader.drops.load(Ordering::SeqCst), usize::from(succeeds));
    }
}

#[test]
fn production_companion_is_unavailable_without_adoption() {
    assert_eq!(EXPECTED_CONTRACT, None);
    assert!(super::load().is_err());
    #[cfg(not(feature = "bindings"))]
    {
        assert!(crate::availability().is_err());
        assert_eq!(crate::abi_version(), 0);
        assert!(!crate::supports_model("phonon"));
    }
}
