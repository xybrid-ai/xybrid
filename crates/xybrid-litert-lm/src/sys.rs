//! Hand-written declarations for the part of the LiteRT-LM C API this crate
//! calls, written against the headers of C API 1.0.0 (LiteRT-LM v0.19.1).
//!
//! Nothing here is linked at build time. [`Api`] is a table of function
//! pointers, filled from a library opened at run time or, in tests, from
//! in-process fakes. Only functions the safe layer uses are declared; adding
//! one means adding a line to the table below.

use std::ffi::{c_char, c_int, c_void};

/// `LiteRtLmStatusCode`. `0` is success; every other value is a failure.
pub type Status = c_int;

/// `kLiteRtLmStatusOk`.
pub const STATUS_OK: Status = 0;
/// `kLiteRtLmStatusUnimplemented`, what an unset fake function returns.
pub const STATUS_UNIMPLEMENTED: Status = 12;

/// `kLiteRtLmSamplerTypeTopP`: top-k filtering, then nucleus sampling.
///
/// It is the only sampler the CPU path of v0.19.1 implements
/// (`runtime/components/sampler_factory.cc`: every other type returns
/// `UNIMPLEMENTED`), so greedy decoding is expressed as top-p with `k = 1`.
pub const SAMPLER_TYPE_TOP_P: c_int = 2;

/// `kLiteRtLmModalityVision`.
pub const MODALITY_VISION: c_int = 1;
/// `kLiteRtLmModalityAudio`.
pub const MODALITY_AUDIO: c_int = 2;

/// `kLiteRtLmActivationDataTypeFloat32`.
pub const ACTIVATION_FLOAT32: c_int = 0;
/// `kLiteRtLmActivationDataTypeFloat16`.
pub const ACTIVATION_FLOAT16: c_int = 1;

/// `kLiteRtLmLogSeverity*` values.
pub const LOG_VERBOSE: c_int = 0;
/// See [`LOG_VERBOSE`].
pub const LOG_INFO: c_int = 2;
/// See [`LOG_VERBOSE`].
pub const LOG_WARNING: c_int = 3;
/// See [`LOG_VERBOSE`].
pub const LOG_ERROR: c_int = 4;
/// See [`LOG_VERBOSE`].
pub const LOG_SILENT: c_int = 1000;

macro_rules! opaque {
    ($($(#[$meta:meta])* $name:ident),* $(,)?) => {
        $(
            $(#[$meta])*
            #[repr(C)]
            pub struct $name {
                _data: [u8; 0],
                _marker: std::marker::PhantomData<(*mut u8, std::marker::PhantomPinned)>,
            }
        )*
    };
}

opaque!(
    /// `LiteRtLmEngine`.
    Engine,
    /// `LiteRtLmEngineSettings`.
    EngineSettings,
    /// `LiteRtLmLoadedFile`.
    LoadedFile,
    /// `LiteRtLmSamplerParams`.
    SamplerParams,
    /// `LiteRtLmSessionConfig`.
    SessionConfig,
    /// `LiteRtLmConversation`.
    Conversation,
    /// `LiteRtLmConversationConfig`.
    ConversationConfig,
    /// `LiteRtLmConversationOptionalArgs`.
    OptionalArgs,
    /// `LiteRtLmThinkingConfig`.
    ThinkingConfig,
    /// `LiteRtLmRepetitionPenaltyConfig`.
    RepetitionPenaltyConfig,
    /// `LiteRtLmBenchmarkInfo`.
    BenchmarkInfo,
    /// `LiteRtLmStreamChunk`. Valid only for the duration of one callback.
    StreamChunk,
);

/// `LiteRtLmStreamCallback`.
pub type StreamCallback =
    unsafe extern "C" fn(callback_data: *mut c_void, chunk: *const StreamChunk);

/// What an unset entry of [`Api::stub`] returns.
pub trait StubReturn {
    /// The value.
    fn stub() -> Self;
}

impl StubReturn for () {
    fn stub() -> Self {}
}

impl StubReturn for Status {
    fn stub() -> Self {
        STATUS_UNIMPLEMENTED
    }
}

impl StubReturn for *const c_char {
    fn stub() -> Self {
        std::ptr::null()
    }
}

macro_rules! litert_lm_api {
    ($( fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty; )*) => {
        /// The C functions this crate calls, as function pointers.
        ///
        /// Field `foo` is the C function `litert_lm_foo`.
        pub struct Api {
            $(
                #[doc = concat!("`litert_lm_", stringify!($name), "`.")]
                pub $name: unsafe extern "C" fn($($arg: $ty),*) -> $ret,
            )*
        }

        impl Api {
            /// Resolve every function from an opened library.
            ///
            /// # Safety
            ///
            /// `lib` must be a LiteRT-LM runtime whose exported functions have
            /// the signatures declared here, which [`crate::Library`] checks
            /// through the library's reported C API version before any other
            /// call is made.
            pub(crate) unsafe fn load(
                lib: &libloading::Library,
            ) -> std::result::Result<Self, (&'static str, libloading::Error)> {
                Ok(Self {
                    $(
                        $name: *lib
                            .get(concat!("litert_lm_", stringify!($name), "\0").as_bytes())
                            .map_err(|err| (concat!("litert_lm_", stringify!($name)), err))?,
                    )*
                })
            }

            /// A table whose every function fails with
            /// [`STATUS_UNIMPLEMENTED`], returns null, or does nothing.
            ///
            /// The starting point for an in-process fake: override the
            /// functions a test needs with struct-update syntax.
            pub fn stub() -> Self {
                $(
                    unsafe extern "C" fn $name($(_: $ty),*) -> $ret {
                        <$ret as StubReturn>::stub()
                    }
                )*
                Self { $($name,)* }
            }
        }
    };
}

litert_lm_api! {
    fn get_api_version(out_major: *mut c_int, out_minor: *mut c_int, out_patch: *mut c_int) -> Status;
    fn get_last_error_message() -> *const c_char;
    fn set_min_log_level(level: c_int) -> Status;

    fn loaded_file_create(path: *const c_char, out: *mut *mut LoadedFile) -> Status;
    fn loaded_file_delete(file: *mut LoadedFile) -> ();
    fn loaded_file_supports_input_modality(file: *const LoadedFile, modality: c_int, out: *mut bool) -> Status;
    fn loaded_file_supports_function_calling(file: *const LoadedFile, out: *mut bool) -> Status;
    fn loaded_file_max_context_tokens(file: *const LoadedFile, out: *mut u32) -> Status;
    fn loaded_file_min_runtime_version(file: *const LoadedFile, out: *mut *const c_char) -> Status;

    fn engine_settings_create(
        model_path: *const c_char,
        backend: *const c_char,
        vision_backend: *const c_char,
        audio_backend: *const c_char,
        out: *mut *mut EngineSettings,
    ) -> Status;
    fn engine_settings_delete(settings: *mut EngineSettings) -> ();
    fn engine_settings_set_max_num_tokens(settings: *mut EngineSettings, max_num_tokens: c_int) -> Status;
    fn engine_settings_set_num_threads(settings: *mut EngineSettings, num_threads: c_int) -> Status;
    fn engine_settings_set_cache_dir(settings: *mut EngineSettings, cache_dir: *const c_char) -> Status;
    fn engine_settings_set_activation_data_type(settings: *mut EngineSettings, data_type: c_int) -> Status;
    fn engine_settings_enable_benchmark(settings: *mut EngineSettings) -> Status;
    fn engine_create(settings: *const EngineSettings, out: *mut *mut Engine) -> Status;
    fn engine_delete(engine: *mut Engine) -> ();

    fn sampler_params_create(sampler_type: c_int, out: *mut *mut SamplerParams) -> Status;
    fn sampler_params_delete(params: *mut SamplerParams) -> ();
    fn sampler_params_set_top_k(params: *mut SamplerParams, top_k: i32) -> Status;
    fn sampler_params_set_top_p(params: *mut SamplerParams, top_p: f32) -> Status;
    fn sampler_params_set_temperature(params: *mut SamplerParams, temperature: f32) -> Status;
    fn sampler_params_set_seed(params: *mut SamplerParams, seed: i32) -> Status;
    fn session_config_create(out: *mut *mut SessionConfig) -> Status;
    fn session_config_delete(config: *mut SessionConfig) -> ();
    fn session_config_set_sampler_params(config: *mut SessionConfig, params: *const SamplerParams) -> Status;

    fn thinking_config_create(out: *mut *mut ThinkingConfig) -> Status;
    fn thinking_config_delete(config: *mut ThinkingConfig) -> ();
    fn thinking_config_set_enable_thinking(config: *mut ThinkingConfig, enable: bool) -> Status;

    fn conversation_config_create(out: *mut *mut ConversationConfig) -> Status;
    fn conversation_config_delete(config: *mut ConversationConfig) -> ();
    fn conversation_config_set_session_config(config: *mut ConversationConfig, session_config: *const SessionConfig) -> Status;
    fn conversation_config_set_tools(config: *mut ConversationConfig, json: *const c_char) -> Status;
    fn conversation_config_set_messages(config: *mut ConversationConfig, json: *const c_char) -> Status;
    fn conversation_config_set_thinking_config(config: *mut ConversationConfig, thinking: *const ThinkingConfig) -> Status;
    fn conversation_config_set_stream_tool_calls(config: *mut ConversationConfig, stream: bool, channel_name: *const c_char) -> Status;

    fn repetition_penalty_config_create(out: *mut *mut RepetitionPenaltyConfig) -> Status;
    fn repetition_penalty_config_delete(config: *mut RepetitionPenaltyConfig) -> ();
    fn repetition_penalty_config_set_repetition_penalty(config: *mut RepetitionPenaltyConfig, penalty: f32) -> Status;
    fn repetition_penalty_config_set_window_size(config: *mut RepetitionPenaltyConfig, window: c_int) -> Status;

    fn conversation_optional_args_create(out: *mut *mut OptionalArgs) -> Status;
    fn conversation_optional_args_delete(args: *mut OptionalArgs) -> ();
    fn conversation_optional_args_set_max_output_tokens(args: *mut OptionalArgs, max_output_tokens: c_int) -> Status;
    fn conversation_optional_args_set_repetition_penalty_config(args: *mut OptionalArgs, config: *const RepetitionPenaltyConfig) -> Status;
    fn conversation_optional_args_set_visual_token_budget(args: *mut OptionalArgs, budget: c_int) -> Status;

    fn conversation_create(engine: *mut Engine, config: *mut ConversationConfig, out: *mut *mut Conversation) -> Status;
    fn conversation_delete(conversation: *mut Conversation) -> ();
    fn conversation_send_message_stream(
        conversation: *mut Conversation,
        message_json: *const c_char,
        extra_context: *const c_char,
        args: *const OptionalArgs,
        callback: StreamCallback,
        callback_data: *mut c_void,
    ) -> Status;
    fn conversation_render_message_to_string(conversation: *mut Conversation, message_json: *const c_char, out: *mut *const c_char) -> Status;
    fn conversation_cancel_process(conversation: *mut Conversation) -> Status;
    fn conversation_wait_until_done(conversation: *mut Conversation) -> Status;
    fn conversation_get_benchmark_info(conversation: *mut Conversation, out: *mut *mut BenchmarkInfo) -> Status;

    fn benchmark_info_delete(info: *mut BenchmarkInfo) -> ();
    fn benchmark_info_get_num_prefill_turns(info: *const BenchmarkInfo, out: *mut usize) -> Status;
    fn benchmark_info_get_num_decode_turns(info: *const BenchmarkInfo, out: *mut usize) -> Status;
    fn benchmark_info_get_prefill_token_count_at(info: *const BenchmarkInfo, index: usize, out: *mut usize) -> Status;
    fn benchmark_info_get_decode_token_count_at(info: *const BenchmarkInfo, index: usize, out: *mut usize) -> Status;
    fn benchmark_info_get_prefill_tokens_per_sec_at(info: *const BenchmarkInfo, index: usize, out: *mut f64) -> Status;
    fn benchmark_info_get_decode_tokens_per_sec_at(info: *const BenchmarkInfo, index: usize, out: *mut f64) -> Status;

    fn stream_chunk_get_text(chunk: *const StreamChunk, out: *mut *const c_char) -> Status;
    fn stream_chunk_is_final(chunk: *const StreamChunk, out: *mut bool) -> Status;
    fn stream_chunk_get_error(chunk: *const StreamChunk, out: *mut *const c_char) -> Status;
}
