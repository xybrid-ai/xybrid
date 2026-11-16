//! A scripted LiteRT-LM runtime that lives in the calling process, for tests.
//!
//! [`FakeRuntime`] implements every C function this crate calls and plays a
//! [`Script`] chosen by the message a test sends: a message that contains
//! `script:<name>` runs the script registered under `<name>`. It needs no
//! model and no native library, so code built on this crate can be tested
//! anywhere, including the paths the real runtime rarely takes: a stream
//! that never ends, a failing wait, a callback after everything was freed.
//!
//! The fake also records what it was asked ([`FakeRuntime::engines`],
//! [`FakeRuntime::turns`]) and what teardown did ([`FakeRuntime::events`]).
//!
//! It models the real runtime only as far as its scripts say. In particular
//! it does not cancel by itself: a script that should end the way the real
//! runtime ends a cancelled turn says so, with [`Step::UntilCancel`] or
//! [`Step::WaitForCancel`] followed by [`Step::cancelled`].
//!
//! Enabled by the `test-util` feature.
//!
//! # Example
//!
//! ```
//! use std::time::Duration;
//! use xybrid_litert_lm::testing::{FakeRuntime, Script, Step};
//! use xybrid_litert_lm::{ConversationConfig, Sampling, TurnOptions};
//!
//! let fake = FakeRuntime::new();
//! fake.script("hello", Script::new([Step::text("Hi"), Step::End]));
//! let engine = fake.engine();
//! let mut conversation = engine
//!     .create_conversation(&ConversationConfig::new(Sampling::greedy()))
//!     .unwrap();
//! let mut stream = conversation
//!     .send_stream("script:hello", &TurnOptions::default())
//!     .unwrap();
//! let first = stream.next_chunk(Duration::from_secs(5)).unwrap();
//! assert_eq!(first.text.as_deref(), Some("Hi"));
//! ```

use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::sys::{self, Status, STATUS_OK};
use crate::{Engine, EngineSettings, Library};

const STATUS_NOT_FOUND: Status = 5;
const STATUS_FAILED_PRECONDITION: Status = 9;

// ---------------------------------------------------------------------------
// What tests script
// ---------------------------------------------------------------------------

/// One thing the fake does while it plays a reply.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// A chunk with text that is not final.
    Text(String),
    /// The clean final chunk: no text, no error.
    End,
    /// A final chunk that carries an error.
    Error(String),
    /// A chunk with every field chosen.
    Chunk {
        /// The chunk's text.
        text: Option<String>,
        /// The chunk's error.
        error: Option<String>,
        /// Whether the chunk is final.
        is_final: bool,
    },
    /// The callback is invoked with a null chunk.
    NullChunk,
    /// The text again and again, a millisecond apart, until a cancel is
    /// requested.
    UntilCancel(String),
    /// Nothing until a cancel is requested.
    WaitForCancel,
    /// Nothing until the test calls [`FakeRuntime::release`].
    WaitForRelease,
    /// The text, this many times.
    Repeat(String, usize),
    /// Counts in [`FakeRuntime::late_callbacks`] that the script got this far.
    MarkLate,
}

impl Step {
    /// A chunk with text that is not final.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// A final chunk that carries an error.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(message.into())
    }

    /// The final chunk the real runtime sends when a turn was cancelled.
    pub fn cancelled() -> Self {
        Self::Error("CANCELLED: Task cancelled".to_string())
    }
}

/// A reply the fake plays, and how the calls around it behave.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Script {
    steps: Vec<Step>,
    inline: bool,
    detached: bool,
    start_status: Option<Status>,
    wait_status: Option<Status>,
}

impl Script {
    /// A reply made of `steps`, played on a thread of the fake's own.
    pub fn new(steps: impl IntoIterator<Item = Step>) -> Self {
        Self {
            steps: steps.into_iter().collect(),
            ..Self::default()
        }
    }

    /// The start call itself fails with `status`; nothing is played.
    pub fn failing_start(status: Status) -> Self {
        Self {
            start_status: Some(status),
            ..Self::default()
        }
    }

    /// Play the whole reply on the caller's thread, inside the start call.
    #[must_use]
    pub fn inline(mut self) -> Self {
        self.inline = true;
        self
    }

    /// Keep playing after wait and delete have returned, the way a runtime
    /// that does not join its callbacks would.
    #[must_use]
    pub fn detached(mut self) -> Self {
        self.detached = true;
        self
    }

    /// `wait_until_done` fails with `status` and does not wait for the reply.
    #[must_use]
    pub fn failing_wait(mut self, status: Status) -> Self {
        self.wait_status = Some(status);
        self
    }
}

/// What the fake's model file declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FakeModel {
    /// The file has a vision encoder.
    pub vision: bool,
    /// The file has an audio encoder.
    pub audio: bool,
    /// The file declares function calling.
    pub function_calling: bool,
    /// The file's context limit; `0` for none declared.
    pub max_context_tokens: u32,
}

impl Default for FakeModel {
    fn default() -> Self {
        Self {
            vision: false,
            audio: false,
            function_calling: false,
            max_context_tokens: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// What the fake records
// ---------------------------------------------------------------------------

/// The settings one engine was created with.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineRequest {
    /// The language model's backend string.
    pub backend: String,
    /// The vision backend string, when one was given.
    pub vision_backend: Option<String>,
    /// `set_max_num_tokens`.
    pub max_num_tokens: Option<i32>,
    /// `set_num_threads`.
    pub num_threads: Option<i32>,
    /// `set_cache_dir`.
    pub cache_dir: Option<String>,
    /// `set_activation_data_type`, as the C enum value.
    pub activation: Option<i32>,
    /// `enable_benchmark` was called.
    pub benchmark: bool,
}

/// The sampler a conversation was created with.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SamplerRequest {
    /// The sampler type, as the C enum value.
    pub sampler_type: i32,
    /// `set_top_k`.
    pub top_k: Option<i32>,
    /// `set_top_p`.
    pub top_p: Option<f32>,
    /// `set_temperature`.
    pub temperature: Option<f32>,
    /// `set_seed`.
    pub seed: Option<i32>,
}

/// One streamed turn: the conversation it ran on and the message sent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnRequest {
    /// The conversation's starting messages (`set_messages`).
    pub history_json: Option<String>,
    /// The conversation's tools (`set_tools`).
    pub tools_json: Option<String>,
    /// The conversation's thinking setting.
    pub thinking: Option<bool>,
    /// The channel raw tool-call text is streamed on, when enabled.
    pub tool_call_channel: Option<String>,
    /// The conversation's sampler.
    pub sampler: Option<SamplerRequest>,
    /// The message sent.
    pub message_json: String,
    /// The turn's output budget.
    pub max_output_tokens: Option<i32>,
    /// The turn's repetition penalty and its window.
    pub repetition_penalty: Option<(f32, i32)>,
    /// The turn's visual token budget.
    pub visual_token_budget: Option<i32>,
}

// ---------------------------------------------------------------------------
// The fake
// ---------------------------------------------------------------------------

struct State {
    path: String,
    scripts: Mutex<BTreeMap<String, Script>>,
    model: Mutex<FakeModel>,
    usage: Mutex<Option<(usize, usize)>>,
    render_prefix: Mutex<String>,
    events: Mutex<Vec<&'static str>>,
    engines: Mutex<Vec<EngineRequest>>,
    turns: Mutex<Vec<TurnRequest>>,
    live_conversations: AtomicUsize,
    late_callbacks: AtomicUsize,
    released: Mutex<bool>,
    release_signal: Condvar,
}

/// Lock a mutex of the fake. Its data stays valid after a panic.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl State {
    fn record(&self, event: &'static str) {
        lock(&self.events).push(event);
    }

    fn wait_for_release(&self) {
        let mut released = lock(&self.released);
        while !*released {
            released = self
                .release_signal
                .wait(released)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// The script a message asks for: the longest registered name `n` for
    /// which the message contains `script:n`.
    fn script_for(&self, message: &str) -> Option<Script> {
        lock(&self.scripts)
            .iter()
            .filter(|(name, _)| message.contains(&format!("script:{name}")))
            .max_by_key(|(name, _)| name.len())
            .map(|(_, script)| script.clone())
    }
}

/// Fakes by the model path that selects them.
static FAKES: Mutex<BTreeMap<String, Arc<State>>> = Mutex::new(BTreeMap::new());
static NEXT_FAKE: AtomicUsize = AtomicUsize::new(1);

fn fake_for(path: *const c_char) -> Option<Arc<State>> {
    if path.is_null() {
        return None;
    }
    // SAFETY: a non-null, NUL-terminated string from the safe layer.
    let path = unsafe { CStr::from_ptr(path) }.to_string_lossy();
    lock(&FAKES).get(path.as_ref()).cloned()
}

/// A scripted runtime. Each one has its own model path, scripts and records,
/// so tests can run in parallel.
pub struct FakeRuntime {
    state: Arc<State>,
}

impl Default for FakeRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FakeRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeRuntime")
            .field("model_path", &self.state.path)
            .finish()
    }
}

impl FakeRuntime {
    /// A fake with no scripts, a text-only model and no usage figures, under
    /// a model path of its own (`fake://<n>`).
    pub fn new() -> Self {
        Self::at(format!(
            "fake://{}",
            NEXT_FAKE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// A fake selected by `model_path`, for code that derives the path
    /// itself (a model directory on disk, say). The file need not exist. A
    /// fake already registered under the path is replaced.
    pub fn at(model_path: impl AsRef<Path>) -> Self {
        let path = model_path.as_ref().to_string_lossy().into_owned();
        let state = Arc::new(State {
            path: path.clone(),
            scripts: Mutex::new(BTreeMap::new()),
            model: Mutex::new(FakeModel::default()),
            usage: Mutex::new(None),
            render_prefix: Mutex::new(String::new()),
            events: Mutex::new(Vec::new()),
            engines: Mutex::new(Vec::new()),
            turns: Mutex::new(Vec::new()),
            live_conversations: AtomicUsize::new(0),
            late_callbacks: AtomicUsize::new(0),
            released: Mutex::new(false),
            release_signal: Condvar::new(),
        });
        lock(&FAKES).insert(path, Arc::clone(&state));
        Self { state }
    }

    /// A library whose functions are the fake's.
    pub fn library(&self) -> Arc<Library> {
        // SAFETY: every function of the table is implemented below under the
        // contract of the C function it stands for: out-parameters are
        // written, strings live as long as their owner, objects are created
        // and destroyed in matching pairs, and stream callbacks get the
        // callback data they were registered with.
        unsafe { Library::from_api(api()) }.expect("the fake reports C API 1.0.0")
    }

    /// The model path that selects this fake.
    pub fn model_path(&self) -> PathBuf {
        PathBuf::from(&self.state.path)
    }

    /// An engine on this fake, with default settings.
    pub fn engine(&self) -> Engine {
        Engine::load(self.library(), &EngineSettings::new(self.model_path()))
            .expect("the fake engine loads")
    }

    /// Register what a message containing `script:<name>` plays.
    pub fn script(&self, name: &str, script: Script) -> &Self {
        lock(&self.state.scripts).insert(name.to_string(), script);
        self
    }

    /// Set what the model file declares.
    pub fn set_model(&self, model: FakeModel) -> &Self {
        *lock(&self.state.model) = model;
        self
    }

    /// Set the token counts a conversation reports. Without this, asking for
    /// them fails, as it does on an engine without benchmarking.
    pub fn set_usage(&self, prompt_tokens: usize, output_tokens: usize) -> &Self {
        *lock(&self.state.usage) = Some((prompt_tokens, output_tokens));
        self
    }

    /// Set the text a rendered prompt starts with. The message follows it.
    pub fn set_render_prefix(&self, prefix: &str) -> &Self {
        *lock(&self.state.render_prefix) = prefix.to_string();
        self
    }

    /// Teardown calls, in the order the fake received them: `cancel`,
    /// `wait`, `conversation_delete`, `engine_delete`, and
    /// `engine_delete_with_live_conversations` for an engine deleted too
    /// early.
    pub fn events(&self) -> Vec<&'static str> {
        lock(&self.state.events).clone()
    }

    /// How often `event` occurs in [`Self::events`].
    pub fn count(&self, event: &str) -> usize {
        self.events().iter().filter(|e| **e == event).count()
    }

    /// The settings of every engine created on this fake, in order.
    pub fn engines(&self) -> Vec<EngineRequest> {
        lock(&self.state.engines).clone()
    }

    /// Every turn started on this fake, in order.
    pub fn turns(&self) -> Vec<TurnRequest> {
        lock(&self.state.turns).clone()
    }

    /// Let scripts waiting at [`Step::WaitForRelease`] carry on.
    pub fn release(&self) {
        *lock(&self.state.released) = true;
        self.state.release_signal.notify_all();
    }

    /// How many [`Step::MarkLate`] steps have been reached.
    pub fn late_callbacks(&self) -> usize {
        self.state.late_callbacks.load(Ordering::SeqCst)
    }
}

impl Drop for FakeRuntime {
    fn drop(&mut self) {
        // Objects still alive keep the state through their own references.
        let mut fakes = lock(&FAKES);
        if fakes
            .get(&self.state.path)
            .is_some_and(|registered| Arc::ptr_eq(registered, &self.state))
        {
            fakes.remove(&self.state.path);
        }
    }
}

/// The fake library, when a [`FakeRuntime`] is registered under
/// `model_path`; `None` otherwise.
///
/// For the one place in an embedder that opens the runtime library: asking
/// here first lets a test put a fake behind a model path while every other
/// path still opens the real library.
pub fn library_for_model(model_path: &Path) -> Option<Arc<Library>> {
    let registered = lock(&FAKES).contains_key(model_path.to_string_lossy().as_ref());
    // SAFETY: as in `FakeRuntime::library`.
    registered.then(|| unsafe { Library::from_api(api()) }.expect("the fake reports C API 1.0.0"))
}

// ---------------------------------------------------------------------------
// The C objects
// ---------------------------------------------------------------------------

struct FakeLoadedFile {
    state: Arc<State>,
}

struct FakeSettings {
    state: Arc<State>,
    request: EngineRequest,
}

struct FakeEngine {
    state: Arc<State>,
}

#[derive(Default)]
struct FakeSessionConfig {
    sampler: Option<SamplerRequest>,
}

#[derive(Default)]
struct FakeThinkingConfig {
    enable: Option<bool>,
}

#[derive(Default)]
struct FakeConversationConfig {
    sampler: Option<SamplerRequest>,
    history_json: Option<String>,
    tools_json: Option<String>,
    thinking: Option<bool>,
    tool_call_channel: Option<String>,
}

#[derive(Default)]
struct FakeRepetitionPenalty {
    penalty: f32,
    window: i32,
}

#[derive(Default)]
struct FakeOptionalArgs {
    max_output_tokens: Option<i32>,
    repetition_penalty: Option<(f32, i32)>,
    visual_token_budget: Option<i32>,
}

struct FakeConversation {
    state: Arc<State>,
    config: FakeConversationConfig,
    cancel: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    /// The running script does not let wait and delete join it.
    detached: AtomicBool,
    /// The status `wait_until_done` fails with, when the script says so.
    wait_status: Mutex<Option<Status>>,
    /// Owns the string the latest render returned.
    rendered: Mutex<CString>,
}

impl FakeConversation {
    /// Wait for the script to finish, as upstream's wait and delete do.
    fn join(&self) {
        if self.detached.load(Ordering::SeqCst) {
            return;
        }
        let worker = lock(&self.worker).take();
        if let Some(worker) = worker {
            // A script that panicked has already failed its test.
            let _ = worker.join();
        }
    }
}

struct FakeBenchmark {
    prompt_tokens: usize,
    output_tokens: usize,
}

struct FakeChunk {
    text: Option<CString>,
    error: Option<CString>,
    is_final: bool,
}

/// Calls the stream callback the safe layer registered.
#[derive(Clone, Copy)]
struct Emitter {
    callback: sys::StreamCallback,
    /// The callback data, kept as an address so the emitter can cross
    /// threads. The fake never looks inside it.
    data: usize,
}

impl Emitter {
    fn send(&self, text: Option<&str>, error: Option<&str>, is_final: bool) {
        let chunk = FakeChunk {
            text: text.map(c_string),
            error: error.map(c_string),
            is_final,
        };
        // SAFETY: the callback and its data are the pair the caller
        // registered; the chunk outlives the call.
        unsafe {
            (self.callback)(
                ptr::without_provenance_mut(self.data),
                (&raw const chunk).cast(),
            );
        }
    }

    fn null_chunk(&self) {
        // SAFETY: as in `send`. A null chunk is the fault being scripted.
        unsafe { (self.callback)(ptr::without_provenance_mut(self.data), ptr::null()) };
    }
}

/// A C string for test text; an interior NUL is replaced.
fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', "\u{fffd}")).expect("interior NULs were replaced")
}

fn play(steps: &[Step], emit: Emitter, state: &State, cancel: &AtomicBool) {
    for step in steps {
        match step {
            Step::Text(text) => emit.send(Some(text), None, false),
            Step::End => emit.send(None, None, true),
            Step::Error(message) => emit.send(None, Some(message), true),
            Step::Chunk {
                text,
                error,
                is_final,
            } => emit.send(text.as_deref(), error.as_deref(), *is_final),
            Step::NullChunk => emit.null_chunk(),
            Step::UntilCancel(text) => {
                while !cancel.load(Ordering::SeqCst) {
                    emit.send(Some(text), None, false);
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            Step::WaitForCancel => {
                while !cancel.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            Step::WaitForRelease => state.wait_for_release(),
            Step::Repeat(text, count) => {
                for _ in 0..*count {
                    emit.send(Some(text), None, false);
                }
            }
            Step::MarkLate => {
                state.late_callbacks.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The C functions
//
// Each takes the pointers the safe layer passes for the C function of the
// same name. Handles are the fake's own boxes, cast to the opaque C types.
// ---------------------------------------------------------------------------

thread_local! {
    /// The calling thread's last error message, as in the real runtime.
    static LAST_ERROR: Cell<&'static CStr> = const { Cell::new(c"") };
}

fn fail(status: Status, message: &'static CStr) -> Status {
    LAST_ERROR.with(|last| last.set(message));
    status
}

fn failure_message(status: Status) -> &'static CStr {
    match status {
        4 => c"DEADLINE_EXCEEDED: scripted failure",
        13 => c"INTERNAL: scripted failure",
        _ => c"scripted failure",
    }
}

/// Copy a string argument; null becomes `None`.
unsafe fn owned(text: *const c_char) -> Option<String> {
    (!text.is_null()).then(|| CStr::from_ptr(text).to_string_lossy().into_owned())
}

macro_rules! boxed {
    ($create:ident, $delete:ident, $sys:ty, $fake:ty) => {
        unsafe extern "C" fn $create(out: *mut *mut $sys) -> Status {
            *out = Box::into_raw(Box::<$fake>::default()).cast();
            STATUS_OK
        }

        unsafe extern "C" fn $delete(object: *mut $sys) {
            if !object.is_null() {
                drop(Box::from_raw(object.cast::<$fake>()));
            }
        }
    };
}

unsafe extern "C" fn get_api_version(
    major: *mut c_int,
    minor: *mut c_int,
    patch: *mut c_int,
) -> Status {
    *major = 1;
    *minor = 0;
    *patch = 0;
    STATUS_OK
}

unsafe extern "C" fn get_last_error_message() -> *const c_char {
    LAST_ERROR.with(|last| last.get().as_ptr())
}

unsafe extern "C" fn set_min_log_level(_level: c_int) -> Status {
    STATUS_OK
}

unsafe extern "C" fn loaded_file_create(
    path: *const c_char,
    out: *mut *mut sys::LoadedFile,
) -> Status {
    let Some(state) = fake_for(path) else {
        *out = ptr::null_mut();
        return fail(
            STATUS_NOT_FOUND,
            c"NOT_FOUND: no fake is registered under this model path",
        );
    };
    *out = Box::into_raw(Box::new(FakeLoadedFile { state })).cast();
    STATUS_OK
}

unsafe extern "C" fn loaded_file_delete(file: *mut sys::LoadedFile) {
    if !file.is_null() {
        drop(Box::from_raw(file.cast::<FakeLoadedFile>()));
    }
}

unsafe extern "C" fn loaded_file_supports_input_modality(
    file: *const sys::LoadedFile,
    modality: c_int,
    out: *mut bool,
) -> Status {
    let file = &*file.cast::<FakeLoadedFile>();
    let model = *lock(&file.state.model);
    *out = match modality {
        sys::MODALITY_VISION => model.vision,
        sys::MODALITY_AUDIO => model.audio,
        _ => modality == 0,
    };
    STATUS_OK
}

unsafe extern "C" fn loaded_file_supports_function_calling(
    file: *const sys::LoadedFile,
    out: *mut bool,
) -> Status {
    let file = &*file.cast::<FakeLoadedFile>();
    *out = lock(&file.state.model).function_calling;
    STATUS_OK
}

unsafe extern "C" fn loaded_file_max_context_tokens(
    file: *const sys::LoadedFile,
    out: *mut u32,
) -> Status {
    let file = &*file.cast::<FakeLoadedFile>();
    *out = lock(&file.state.model).max_context_tokens;
    STATUS_OK
}

unsafe extern "C" fn loaded_file_min_runtime_version(
    _file: *const sys::LoadedFile,
    out: *mut *const c_char,
) -> Status {
    // Legitimately absent: success with a null string.
    *out = ptr::null();
    STATUS_OK
}

unsafe extern "C" fn engine_settings_create(
    model_path: *const c_char,
    backend: *const c_char,
    vision_backend: *const c_char,
    _audio_backend: *const c_char,
    out: *mut *mut sys::EngineSettings,
) -> Status {
    let Some(state) = fake_for(model_path) else {
        *out = ptr::null_mut();
        return fail(
            STATUS_NOT_FOUND,
            c"NOT_FOUND: no fake is registered under this model path",
        );
    };
    let request = EngineRequest {
        backend: owned(backend).unwrap_or_default(),
        vision_backend: owned(vision_backend),
        ..EngineRequest::default()
    };
    *out = Box::into_raw(Box::new(FakeSettings { state, request })).cast();
    STATUS_OK
}

unsafe extern "C" fn engine_settings_delete(settings: *mut sys::EngineSettings) {
    if !settings.is_null() {
        drop(Box::from_raw(settings.cast::<FakeSettings>()));
    }
}

unsafe extern "C" fn engine_settings_set_max_num_tokens(
    settings: *mut sys::EngineSettings,
    max_num_tokens: c_int,
) -> Status {
    (*settings.cast::<FakeSettings>()).request.max_num_tokens = Some(max_num_tokens);
    STATUS_OK
}

unsafe extern "C" fn engine_settings_set_num_threads(
    settings: *mut sys::EngineSettings,
    num_threads: c_int,
) -> Status {
    (*settings.cast::<FakeSettings>()).request.num_threads = Some(num_threads);
    STATUS_OK
}

unsafe extern "C" fn engine_settings_set_cache_dir(
    settings: *mut sys::EngineSettings,
    cache_dir: *const c_char,
) -> Status {
    (*settings.cast::<FakeSettings>()).request.cache_dir = owned(cache_dir);
    STATUS_OK
}

unsafe extern "C" fn engine_settings_set_activation_data_type(
    settings: *mut sys::EngineSettings,
    data_type: c_int,
) -> Status {
    (*settings.cast::<FakeSettings>()).request.activation = Some(data_type);
    STATUS_OK
}

unsafe extern "C" fn engine_settings_enable_benchmark(
    settings: *mut sys::EngineSettings,
) -> Status {
    (*settings.cast::<FakeSettings>()).request.benchmark = true;
    STATUS_OK
}

unsafe extern "C" fn engine_create(
    settings: *const sys::EngineSettings,
    out: *mut *mut sys::Engine,
) -> Status {
    let settings = &*settings.cast::<FakeSettings>();
    lock(&settings.state.engines).push(settings.request.clone());
    *out = Box::into_raw(Box::new(FakeEngine {
        state: Arc::clone(&settings.state),
    }))
    .cast();
    STATUS_OK
}

unsafe extern "C" fn engine_delete(engine: *mut sys::Engine) {
    if engine.is_null() {
        return;
    }
    let engine = Box::from_raw(engine.cast::<FakeEngine>());
    let live = engine.state.live_conversations.load(Ordering::SeqCst);
    engine.state.record(if live == 0 {
        "engine_delete"
    } else {
        "engine_delete_with_live_conversations"
    });
}

unsafe extern "C" fn sampler_params_create(
    sampler_type: c_int,
    out: *mut *mut sys::SamplerParams,
) -> Status {
    *out = Box::into_raw(Box::new(SamplerRequest {
        sampler_type,
        ..SamplerRequest::default()
    }))
    .cast();
    STATUS_OK
}

unsafe extern "C" fn sampler_params_delete(params: *mut sys::SamplerParams) {
    if !params.is_null() {
        drop(Box::from_raw(params.cast::<SamplerRequest>()));
    }
}

unsafe extern "C" fn sampler_params_set_top_k(
    params: *mut sys::SamplerParams,
    top_k: i32,
) -> Status {
    (*params.cast::<SamplerRequest>()).top_k = Some(top_k);
    STATUS_OK
}

unsafe extern "C" fn sampler_params_set_top_p(
    params: *mut sys::SamplerParams,
    top_p: f32,
) -> Status {
    (*params.cast::<SamplerRequest>()).top_p = Some(top_p);
    STATUS_OK
}

unsafe extern "C" fn sampler_params_set_temperature(
    params: *mut sys::SamplerParams,
    temperature: f32,
) -> Status {
    (*params.cast::<SamplerRequest>()).temperature = Some(temperature);
    STATUS_OK
}

unsafe extern "C" fn sampler_params_set_seed(params: *mut sys::SamplerParams, seed: i32) -> Status {
    (*params.cast::<SamplerRequest>()).seed = Some(seed);
    STATUS_OK
}

boxed!(
    session_config_create,
    session_config_delete,
    sys::SessionConfig,
    FakeSessionConfig
);
boxed!(
    thinking_config_create,
    thinking_config_delete,
    sys::ThinkingConfig,
    FakeThinkingConfig
);
boxed!(
    conversation_config_create,
    conversation_config_delete,
    sys::ConversationConfig,
    FakeConversationConfig
);
boxed!(
    repetition_penalty_config_create,
    repetition_penalty_config_delete,
    sys::RepetitionPenaltyConfig,
    FakeRepetitionPenalty
);
boxed!(
    conversation_optional_args_create,
    conversation_optional_args_delete,
    sys::OptionalArgs,
    FakeOptionalArgs
);

unsafe extern "C" fn session_config_set_sampler_params(
    config: *mut sys::SessionConfig,
    params: *const sys::SamplerParams,
) -> Status {
    (*config.cast::<FakeSessionConfig>()).sampler = Some(*params.cast::<SamplerRequest>());
    STATUS_OK
}

unsafe extern "C" fn thinking_config_set_enable_thinking(
    config: *mut sys::ThinkingConfig,
    enable: bool,
) -> Status {
    (*config.cast::<FakeThinkingConfig>()).enable = Some(enable);
    STATUS_OK
}

unsafe extern "C" fn conversation_config_set_session_config(
    config: *mut sys::ConversationConfig,
    session_config: *const sys::SessionConfig,
) -> Status {
    (*config.cast::<FakeConversationConfig>()).sampler =
        (*session_config.cast::<FakeSessionConfig>()).sampler;
    STATUS_OK
}

unsafe extern "C" fn conversation_config_set_tools(
    config: *mut sys::ConversationConfig,
    json: *const c_char,
) -> Status {
    (*config.cast::<FakeConversationConfig>()).tools_json = owned(json);
    STATUS_OK
}

unsafe extern "C" fn conversation_config_set_messages(
    config: *mut sys::ConversationConfig,
    json: *const c_char,
) -> Status {
    (*config.cast::<FakeConversationConfig>()).history_json = owned(json);
    STATUS_OK
}

unsafe extern "C" fn conversation_config_set_thinking_config(
    config: *mut sys::ConversationConfig,
    thinking: *const sys::ThinkingConfig,
) -> Status {
    (*config.cast::<FakeConversationConfig>()).thinking = if thinking.is_null() {
        None
    } else {
        (*thinking.cast::<FakeThinkingConfig>()).enable
    };
    STATUS_OK
}

unsafe extern "C" fn conversation_config_set_stream_tool_calls(
    config: *mut sys::ConversationConfig,
    stream: bool,
    channel_name: *const c_char,
) -> Status {
    (*config.cast::<FakeConversationConfig>()).tool_call_channel =
        if stream { owned(channel_name) } else { None };
    STATUS_OK
}

unsafe extern "C" fn repetition_penalty_config_set_repetition_penalty(
    config: *mut sys::RepetitionPenaltyConfig,
    penalty: f32,
) -> Status {
    (*config.cast::<FakeRepetitionPenalty>()).penalty = penalty;
    STATUS_OK
}

unsafe extern "C" fn repetition_penalty_config_set_window_size(
    config: *mut sys::RepetitionPenaltyConfig,
    window: c_int,
) -> Status {
    (*config.cast::<FakeRepetitionPenalty>()).window = window;
    STATUS_OK
}

unsafe extern "C" fn conversation_optional_args_set_max_output_tokens(
    args: *mut sys::OptionalArgs,
    max_output_tokens: c_int,
) -> Status {
    (*args.cast::<FakeOptionalArgs>()).max_output_tokens = Some(max_output_tokens);
    STATUS_OK
}

unsafe extern "C" fn conversation_optional_args_set_repetition_penalty_config(
    args: *mut sys::OptionalArgs,
    config: *const sys::RepetitionPenaltyConfig,
) -> Status {
    (*args.cast::<FakeOptionalArgs>()).repetition_penalty = if config.is_null() {
        None
    } else {
        let config = &*config.cast::<FakeRepetitionPenalty>();
        Some((config.penalty, config.window))
    };
    STATUS_OK
}

unsafe extern "C" fn conversation_optional_args_set_visual_token_budget(
    args: *mut sys::OptionalArgs,
    budget: c_int,
) -> Status {
    (*args.cast::<FakeOptionalArgs>()).visual_token_budget = Some(budget);
    STATUS_OK
}

unsafe extern "C" fn conversation_create(
    engine: *mut sys::Engine,
    config: *mut sys::ConversationConfig,
    out: *mut *mut sys::Conversation,
) -> Status {
    let engine = &*engine.cast::<FakeEngine>();
    let config = if config.is_null() {
        FakeConversationConfig::default()
    } else {
        let config = &*config.cast::<FakeConversationConfig>();
        FakeConversationConfig {
            sampler: config.sampler,
            history_json: config.history_json.clone(),
            tools_json: config.tools_json.clone(),
            thinking: config.thinking,
            tool_call_channel: config.tool_call_channel.clone(),
        }
    };
    engine
        .state
        .live_conversations
        .fetch_add(1, Ordering::SeqCst);
    *out = Box::into_raw(Box::new(FakeConversation {
        state: Arc::clone(&engine.state),
        config,
        cancel: Arc::new(AtomicBool::new(false)),
        worker: Mutex::new(None),
        detached: AtomicBool::new(false),
        wait_status: Mutex::new(None),
        rendered: Mutex::new(CString::default()),
    }))
    .cast();
    STATUS_OK
}

unsafe extern "C" fn conversation_delete(conversation: *mut sys::Conversation) {
    if conversation.is_null() {
        return;
    }
    let conversation = Box::from_raw(conversation.cast::<FakeConversation>());
    conversation.join();
    conversation
        .state
        .live_conversations
        .fetch_sub(1, Ordering::SeqCst);
    conversation.state.record("conversation_delete");
}

unsafe extern "C" fn conversation_send_message_stream(
    conversation: *mut sys::Conversation,
    message_json: *const c_char,
    _extra_context: *const c_char,
    args: *const sys::OptionalArgs,
    callback: sys::StreamCallback,
    callback_data: *mut c_void,
) -> Status {
    let conversation = &*conversation.cast::<FakeConversation>();
    let message = owned(message_json).unwrap_or_default();
    let args = (!args.is_null()).then(|| &*args.cast::<FakeOptionalArgs>());
    lock(&conversation.state.turns).push(TurnRequest {
        history_json: conversation.config.history_json.clone(),
        tools_json: conversation.config.tools_json.clone(),
        thinking: conversation.config.thinking,
        tool_call_channel: conversation.config.tool_call_channel.clone(),
        sampler: conversation.config.sampler,
        message_json: message.clone(),
        max_output_tokens: args.and_then(|args| args.max_output_tokens),
        repetition_penalty: args.and_then(|args| args.repetition_penalty),
        visual_token_budget: args.and_then(|args| args.visual_token_budget),
    });

    let Some(script) = conversation.state.script_for(&message) else {
        return fail(
            STATUS_NOT_FOUND,
            c"NOT_FOUND: the message names no registered script",
        );
    };
    if let Some(status) = script.start_status {
        return fail(status, failure_message(status));
    }
    conversation
        .detached
        .store(script.detached, Ordering::SeqCst);
    *lock(&conversation.wait_status) = script.wait_status;

    let emit = Emitter {
        callback,
        data: callback_data.addr(),
    };
    if script.inline {
        // The whole reply arrives on the caller's thread, inside this call.
        play(
            &script.steps,
            emit,
            &conversation.state,
            &conversation.cancel,
        );
        return STATUS_OK;
    }
    // A reply left over from an earlier turn has to be over first.
    conversation.join();
    let state = Arc::clone(&conversation.state);
    let cancel = Arc::clone(&conversation.cancel);
    let worker = std::thread::spawn(move || play(&script.steps, emit, &state, &cancel));
    *lock(&conversation.worker) = Some(worker);
    STATUS_OK
}

unsafe extern "C" fn conversation_render_message_to_string(
    conversation: *mut sys::Conversation,
    message_json: *const c_char,
    out: *mut *const c_char,
) -> Status {
    let conversation = &*conversation.cast::<FakeConversation>();
    let rendered = format!(
        "{}{}",
        lock(&conversation.state.render_prefix),
        owned(message_json).unwrap_or_default()
    );
    let mut slot = lock(&conversation.rendered);
    *slot = c_string(&rendered);
    // Owned by the conversation until its next render, as upstream documents.
    *out = slot.as_ptr();
    STATUS_OK
}

unsafe extern "C" fn conversation_cancel_process(conversation: *mut sys::Conversation) -> Status {
    let conversation = &*conversation.cast::<FakeConversation>();
    conversation.cancel.store(true, Ordering::SeqCst);
    conversation.state.record("cancel");
    STATUS_OK
}

unsafe extern "C" fn conversation_wait_until_done(conversation: *mut sys::Conversation) -> Status {
    let conversation = &*conversation.cast::<FakeConversation>();
    let wait_status = *lock(&conversation.wait_status);
    conversation.state.record("wait");
    if let Some(status) = wait_status {
        return fail(status, failure_message(status));
    }
    conversation.join();
    STATUS_OK
}

unsafe extern "C" fn conversation_get_benchmark_info(
    conversation: *mut sys::Conversation,
    out: *mut *mut sys::BenchmarkInfo,
) -> Status {
    let conversation = &*conversation.cast::<FakeConversation>();
    let Some((prompt_tokens, output_tokens)) = *lock(&conversation.state.usage) else {
        *out = ptr::null_mut();
        return fail(
            STATUS_FAILED_PRECONDITION,
            c"FAILED_PRECONDITION: the fake has no usage figures",
        );
    };
    *out = Box::into_raw(Box::new(FakeBenchmark {
        prompt_tokens,
        output_tokens,
    }))
    .cast();
    STATUS_OK
}

unsafe extern "C" fn benchmark_info_delete(info: *mut sys::BenchmarkInfo) {
    if !info.is_null() {
        drop(Box::from_raw(info.cast::<FakeBenchmark>()));
    }
}

unsafe extern "C" fn benchmark_info_get_num_prefill_turns(
    _info: *const sys::BenchmarkInfo,
    out: *mut usize,
) -> Status {
    *out = 1;
    STATUS_OK
}

unsafe extern "C" fn benchmark_info_get_num_decode_turns(
    _info: *const sys::BenchmarkInfo,
    out: *mut usize,
) -> Status {
    *out = 1;
    STATUS_OK
}

unsafe extern "C" fn benchmark_info_get_prefill_token_count_at(
    info: *const sys::BenchmarkInfo,
    _index: usize,
    out: *mut usize,
) -> Status {
    *out = (*info.cast::<FakeBenchmark>()).prompt_tokens;
    STATUS_OK
}

unsafe extern "C" fn benchmark_info_get_decode_token_count_at(
    info: *const sys::BenchmarkInfo,
    _index: usize,
    out: *mut usize,
) -> Status {
    *out = (*info.cast::<FakeBenchmark>()).output_tokens;
    STATUS_OK
}

unsafe extern "C" fn benchmark_info_get_prefill_tokens_per_sec_at(
    _info: *const sys::BenchmarkInfo,
    _index: usize,
    out: *mut f64,
) -> Status {
    *out = 1000.0;
    STATUS_OK
}

unsafe extern "C" fn benchmark_info_get_decode_tokens_per_sec_at(
    _info: *const sys::BenchmarkInfo,
    _index: usize,
    out: *mut f64,
) -> Status {
    *out = 25.0;
    STATUS_OK
}

unsafe extern "C" fn stream_chunk_get_text(
    chunk: *const sys::StreamChunk,
    out: *mut *const c_char,
) -> Status {
    let chunk = &*chunk.cast::<FakeChunk>();
    *out = chunk
        .text
        .as_ref()
        .map_or(ptr::null(), |text| text.as_ptr());
    STATUS_OK
}

unsafe extern "C" fn stream_chunk_is_final(
    chunk: *const sys::StreamChunk,
    out: *mut bool,
) -> Status {
    *out = (*chunk.cast::<FakeChunk>()).is_final;
    STATUS_OK
}

unsafe extern "C" fn stream_chunk_get_error(
    chunk: *const sys::StreamChunk,
    out: *mut *const c_char,
) -> Status {
    let chunk = &*chunk.cast::<FakeChunk>();
    *out = chunk
        .error
        .as_ref()
        .map_or(ptr::null(), |error| error.as_ptr());
    STATUS_OK
}

/// The fake's function table. Every entry is listed, so a function added to
/// [`sys::Api`] does not compile until the fake implements it.
fn api() -> sys::Api {
    sys::Api {
        get_api_version,
        get_last_error_message,
        set_min_log_level,
        loaded_file_create,
        loaded_file_delete,
        loaded_file_supports_input_modality,
        loaded_file_supports_function_calling,
        loaded_file_max_context_tokens,
        loaded_file_min_runtime_version,
        engine_settings_create,
        engine_settings_delete,
        engine_settings_set_max_num_tokens,
        engine_settings_set_num_threads,
        engine_settings_set_cache_dir,
        engine_settings_set_activation_data_type,
        engine_settings_enable_benchmark,
        engine_create,
        engine_delete,
        sampler_params_create,
        sampler_params_delete,
        sampler_params_set_top_k,
        sampler_params_set_top_p,
        sampler_params_set_temperature,
        sampler_params_set_seed,
        session_config_create,
        session_config_delete,
        session_config_set_sampler_params,
        thinking_config_create,
        thinking_config_delete,
        thinking_config_set_enable_thinking,
        conversation_config_create,
        conversation_config_delete,
        conversation_config_set_session_config,
        conversation_config_set_tools,
        conversation_config_set_messages,
        conversation_config_set_thinking_config,
        conversation_config_set_stream_tool_calls,
        repetition_penalty_config_create,
        repetition_penalty_config_delete,
        repetition_penalty_config_set_repetition_penalty,
        repetition_penalty_config_set_window_size,
        conversation_optional_args_create,
        conversation_optional_args_delete,
        conversation_optional_args_set_max_output_tokens,
        conversation_optional_args_set_repetition_penalty_config,
        conversation_optional_args_set_visual_token_budget,
        conversation_create,
        conversation_delete,
        conversation_send_message_stream,
        conversation_render_message_to_string,
        conversation_cancel_process,
        conversation_wait_until_done,
        conversation_get_benchmark_info,
        benchmark_info_delete,
        benchmark_info_get_num_prefill_turns,
        benchmark_info_get_num_decode_turns,
        benchmark_info_get_prefill_token_count_at,
        benchmark_info_get_decode_token_count_at,
        benchmark_info_get_prefill_tokens_per_sec_at,
        benchmark_info_get_decode_tokens_per_sec_at,
        stream_chunk_get_text,
        stream_chunk_is_final,
        stream_chunk_get_error,
    }
}
