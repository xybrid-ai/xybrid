#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t code;
} FfiStatus;

#define FFI_STATUS_OK ((FfiStatus){0})
#define FFI_STATUS_NULL_POINTER ((FfiStatus){1})
#define FFI_STATUS_BUFFER_TOO_SMALL ((FfiStatus){2})
#define FFI_STATUS_INVALID_ARG ((FfiStatus){3})
#define FFI_STATUS_CANCELLED ((FfiStatus){4})
#define FFI_STATUS_INTERNAL_ERROR ((FfiStatus){100})

typedef struct {
    uint8_t *ptr;
    uintptr_t len;
    uintptr_t cap;
    uintptr_t align;
} FfiBuf_u8;

typedef struct {
    uint8_t *ptr;
    uintptr_t len;
    uintptr_t cap;
} FfiString;

typedef struct {
    FfiString message;
} FfiError;

typedef struct {
    const uint8_t *ptr;
    uintptr_t len;
} FfiSpan;

typedef const void *RustFutureHandle;
typedef int8_t StreamPollResult;
typedef int32_t WaitResult;
typedef void (*RustFutureContinuationCallback)(uint64_t callback_data, int8_t poll_result);
typedef void (*StreamContinuationCallback)(uint64_t callback_data, StreamPollResult result);

static inline bool boltffi_atomic_u8_cas(uint8_t *state, uint8_t expected, uint8_t desired) {
    return atomic_compare_exchange_strong_explicit((_Atomic uint8_t *)state, &expected, desired, memory_order_acq_rel, memory_order_acquire);
}

static inline uint64_t boltffi_atomic_u64_exchange(uint64_t *slot, uint64_t value) {
    return atomic_exchange_explicit((_Atomic uint64_t *)slot, value, memory_order_acq_rel);
}

static inline bool boltffi_atomic_u64_cas(uint64_t *slot, uint64_t expected, uint64_t desired) {
    return atomic_compare_exchange_strong_explicit((_Atomic uint64_t *)slot, &expected, desired, memory_order_acq_rel, memory_order_acquire);
}

static inline uint64_t boltffi_atomic_u64_load(uint64_t *slot) {
    return atomic_load_explicit((_Atomic uint64_t *)slot, memory_order_acquire);
}

typedef struct {
    uint64_t handle;
    const void *vtable;
} BoltFFICallbackHandle;

void boltffi_free_string(FfiString string);
void boltffi_free_buf(FfiBuf_u8 buf);
FfiBuf_u8 boltffi_buf_from_bytes(const uint8_t *ptr, uintptr_t len);
FfiBuf_u8 boltffi_buf_with_len(uintptr_t len);
FfiStatus boltffi_last_error_message(FfiString *out);
void boltffi_clear_last_error(void);
typedef uint32_t ___XybridError;
#define XYBRID_ERROR_MODEL_NOT_FOUND ((___XybridError)0)
#define XYBRID_ERROR_DIRECTORY_NOT_FOUND ((___XybridError)1)
#define XYBRID_ERROR_METADATA_NOT_FOUND ((___XybridError)2)
#define XYBRID_ERROR_METADATA_INVALID ((___XybridError)3)
#define XYBRID_ERROR_LOAD_ERROR ((___XybridError)4)
#define XYBRID_ERROR_INFERENCE_ERROR ((___XybridError)5)
#define XYBRID_ERROR_ABORTED_FOR_CLOUD_FALLBACK ((___XybridError)6)
#define XYBRID_ERROR_STREAMING_NOT_SUPPORTED ((___XybridError)7)
#define XYBRID_ERROR_NOT_LOADED ((___XybridError)8)
#define XYBRID_ERROR_CONFIG_ERROR ((___XybridError)9)
#define XYBRID_ERROR_NETWORK_ERROR ((___XybridError)10)
#define XYBRID_ERROR_OFFLINE ((___XybridError)11)
#define XYBRID_ERROR_IO_ERROR ((___XybridError)12)
#define XYBRID_ERROR_CACHE_ERROR ((___XybridError)13)
#define XYBRID_ERROR_PIPELINE_ERROR ((___XybridError)14)
#define XYBRID_ERROR_CIRCUIT_OPEN ((___XybridError)15)
#define XYBRID_ERROR_RATE_LIMITED ((___XybridError)16)
#define XYBRID_ERROR_TIMEOUT ((___XybridError)17)
#define XYBRID_ERROR_MISSING_ARTIFACT ((___XybridError)18)
#define XYBRID_ERROR_UNSUPPORTED_MODEL_CAPABILITY ((___XybridError)19)
#define XYBRID_ERROR_UNSUPPORTED_BACKEND_CAPABILITY ((___XybridError)20)
#define XYBRID_ERROR_INVALID_IMAGE ((___XybridError)21)
typedef uint32_t ___XybridEnvelopeKind;
#define XYBRID_ENVELOPE_KIND_TEXT ((___XybridEnvelopeKind)0)
#define XYBRID_ENVELOPE_KIND_AUDIO ((___XybridEnvelopeKind)1)
#define XYBRID_ENVELOPE_KIND_EMBEDDING ((___XybridEnvelopeKind)2)
#define XYBRID_ENVELOPE_KIND_IMAGE ((___XybridEnvelopeKind)3)
#define XYBRID_ENVELOPE_KIND_MULTI_PART ((___XybridEnvelopeKind)4)
typedef int32_t ___XybridMessageRole;
#define XYBRID_MESSAGE_ROLE_SYSTEM ((___XybridMessageRole)0)
#define XYBRID_MESSAGE_ROLE_USER ((___XybridMessageRole)1)
#define XYBRID_MESSAGE_ROLE_ASSISTANT ((___XybridMessageRole)2)
typedef int32_t ___XybridAbortSignal;
#define XYBRID_ABORT_SIGNAL_MEMORY_PRESSURE_WARN ((___XybridAbortSignal)0)
#define XYBRID_ABORT_SIGNAL_MEMORY_PRESSURE_CRITICAL ((___XybridAbortSignal)1)
#define XYBRID_ABORT_SIGNAL_THERMAL_HOT ((___XybridAbortSignal)2)
#define XYBRID_ABORT_SIGNAL_THERMAL_CRITICAL ((___XybridAbortSignal)3)
typedef int32_t ___XybridOutputType;
#define XYBRID_OUTPUT_TYPE_TEXT ((___XybridOutputType)0)
#define XYBRID_OUTPUT_TYPE_AUDIO ((___XybridOutputType)1)
#define XYBRID_OUTPUT_TYPE_EMBEDDING ((___XybridOutputType)2)
#define XYBRID_OUTPUT_TYPE_UNKNOWN ((___XybridOutputType)3)
typedef int32_t ___XybridExecutionTarget;
#define XYBRID_EXECUTION_TARGET_LOCAL ((___XybridExecutionTarget)0)
#define XYBRID_EXECUTION_TARGET_CLOUD ((___XybridExecutionTarget)1)
typedef int32_t ___XybridDownloadState;
#define XYBRID_DOWNLOAD_STATE_DOWNLOADING ((___XybridDownloadState)0)
#define XYBRID_DOWNLOAD_STATE_READY ((___XybridDownloadState)1)
#define XYBRID_DOWNLOAD_STATE_FAILED ((___XybridDownloadState)2)
typedef int32_t ___XybridStreamEventKind;
#define XYBRID_STREAM_EVENT_KIND_TOKEN ((___XybridStreamEventKind)0)
#define XYBRID_STREAM_EVENT_KIND_COMPLETE ((___XybridStreamEventKind)1)
typedef int32_t ___XybridThermalState;
#define XYBRID_THERMAL_STATE_NORMAL ((___XybridThermalState)0)
#define XYBRID_THERMAL_STATE_WARM ((___XybridThermalState)1)
#define XYBRID_THERMAL_STATE_HOT ((___XybridThermalState)2)
#define XYBRID_THERMAL_STATE_CRITICAL ((___XybridThermalState)3)
void boltffi_release_class_xybrid_bolt_xybrid_model(uint64_t handle);
FfiBuf_u8 boltffi_init_class_xybrid_bolt_xybrid_model_from_registry(const uint8_t *id_ptr, uintptr_t id_len, uint64_t *return_out);
FfiBuf_u8 boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative(const uint8_t *id_ptr, uintptr_t id_len, uint64_t *return_out);
FfiBuf_u8 boltffi_init_class_xybrid_bolt_xybrid_model_from_directory(const uint8_t *path_ptr, uintptr_t path_len, uint64_t *return_out);
FfiBuf_u8 boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle(const uint8_t *path_ptr, uintptr_t path_len, uint64_t *return_out);
FfiBuf_u8 boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface(const uint8_t *repo_ptr, uintptr_t repo_len, uint64_t *return_out);
FfiBuf_u8 boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision(const uint8_t *repo_ptr, uintptr_t repo_len, const uint8_t *revision_ptr, uintptr_t revision_len, uint64_t *return_out);
FfiBuf_u8 boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file(const uint8_t *path_ptr, uintptr_t path_len, uint64_t *return_out);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_model_id(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_version(uint64_t receiver);
___XybridOutputType boltffi_method_class_xybrid_bolt_xybrid_model_output_type(uint64_t receiver);
bool boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded(uint64_t receiver);
bool boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_download_status(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_await_download(uint64_t receiver, uint64_t timeout_ms);
bool boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming(uint64_t receiver);
bool boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config(uint64_t receiver);
bool boltffi_method_class_xybrid_bolt_xybrid_model_is_llm(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling(uint64_t receiver);
bool boltffi_method_class_xybrid_bolt_xybrid_model_has_voices(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_voices(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_default_voice(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_voice(uint64_t receiver, const uint8_t *voice_id_ptr, uintptr_t voice_id_len);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_run(uint64_t receiver, const uint8_t *envelope_ptr, uintptr_t envelope_len, const uint8_t *options_ptr, uintptr_t options_len, FfiBuf_u8 *return_out);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_run_stream(uint64_t receiver, const uint8_t *envelope_ptr, uintptr_t envelope_len, const uint8_t *options_ptr, uintptr_t options_len, uint64_t *return_out);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_stream_next(uint64_t receiver, uint64_t stream_id, FfiBuf_u8 *return_out);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_stream_result(uint64_t receiver, uint64_t stream_id, FfiBuf_u8 *return_out);
FfiStatus boltffi_method_class_xybrid_bolt_xybrid_model_stream_close(uint64_t receiver, uint64_t stream_id);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context(uint64_t receiver, const uint8_t *envelope_ptr, uintptr_t envelope_len, uint64_t context, const uint8_t *options_ptr, uintptr_t options_len, FfiBuf_u8 *return_out);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context(uint64_t receiver, const uint8_t *envelope_ptr, uintptr_t envelope_len, uint64_t context, const uint8_t *options_ptr, uintptr_t options_len, uint64_t *return_out);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_warmup(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_model_unload(uint64_t receiver);
void boltffi_release_class_xybrid_bolt_xybrid_conversation_context(uint64_t handle);
uint64_t boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new(void);
uint64_t boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id(const uint8_t *id_ptr, uintptr_t id_len);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push(uint64_t receiver, const uint8_t *envelope_ptr, uintptr_t envelope_len);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system(uint64_t receiver, const uint8_t *envelope_ptr, uintptr_t envelope_len);
FfiStatus boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id(uint64_t receiver);
uint32_t boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history(uint64_t receiver);
bool boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system(uint64_t receiver);
FfiStatus boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len(uint64_t receiver, uint32_t len);
void boltffi_release_class_xybrid_bolt_xybrid_telemetry_config(uint64_t handle);
uint64_t boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new(const uint8_t *api_key_ptr, uintptr_t api_key_len);
FfiStatus boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint(uint64_t receiver, const uint8_t *endpoint_ptr, uintptr_t endpoint_len);
FfiStatus boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version(uint64_t receiver, const uint8_t *version_ptr, uintptr_t version_len);
FfiStatus boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label(uint64_t receiver, const uint8_t *label_ptr, uintptr_t label_len);
FfiStatus boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute(uint64_t receiver, const uint8_t *key_ptr, uintptr_t key_len, const uint8_t *value_ptr, uintptr_t value_len);
FfiStatus boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size(uint64_t receiver, uint32_t batch_size);
FfiStatus boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs(uint64_t receiver, uint32_t secs);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init(uint64_t receiver);
void boltffi_release_class_xybrid_bolt_xybrid_bundle(uint64_t handle);
FfiBuf_u8 boltffi_init_class_xybrid_bolt_xybrid_bundle_open(const uint8_t *path_ptr, uintptr_t path_len, uint64_t *return_out);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_bundle_version(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_bundle_target(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_bundle_hash(uint64_t receiver);
bool boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata(uint64_t receiver);
uint32_t boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count(uint64_t receiver);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name(uint64_t receiver, uint32_t index);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json(uint64_t receiver, FfiBuf_u8 *return_out);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json(uint64_t receiver, FfiBuf_u8 *return_out);
FfiBuf_u8 boltffi_method_class_xybrid_bolt_xybrid_bundle_extract(uint64_t receiver, const uint8_t *output_dir_ptr, uintptr_t output_dir_len);
FfiBuf_u8 boltffi_function_xybrid_bolt_tool_results_envelope(const uint8_t *user_text_ptr, uintptr_t user_text_len, const uint8_t *prior_assistant_text_ptr, uintptr_t prior_assistant_text_len, const uint8_t *results_ptr, uintptr_t results_len, FfiBuf_u8 *return_out);
FfiBuf_u8 boltffi_function_xybrid_bolt_json_schema_to_gbnf(const uint8_t *schema_json_ptr, uintptr_t schema_json_len, FfiBuf_u8 *return_out);
FfiStatus boltffi_function_xybrid_bolt_set_thermal_state(___XybridThermalState state);
void boltffi_function_xybrid_bolt_clear_thermal_state(void);
FfiStatus boltffi_function_xybrid_bolt_set_battery_level(uint8_t percent);
void boltffi_function_xybrid_bolt_clear_battery_level(void);
FfiStatus boltffi_function_xybrid_bolt_configure_runtime(const uint8_t *api_key_ptr, uintptr_t api_key_len, const uint8_t *gateway_url_ptr, uintptr_t gateway_url_len, const uint8_t *ingest_url_ptr, uintptr_t ingest_url_len);
FfiStatus boltffi_function_xybrid_bolt_init_sdk_cache_dir(const uint8_t *cache_dir_ptr, uintptr_t cache_dir_len);
FfiStatus boltffi_function_xybrid_bolt_set_binding(const uint8_t *binding_ptr, uintptr_t binding_len);
FfiStatus boltffi_function_xybrid_bolt_set_api_key(const uint8_t *api_key_ptr, uintptr_t api_key_len);
FfiStatus boltffi_function_xybrid_bolt_set_provider_api_key(const uint8_t *provider_ptr, uintptr_t provider_len, const uint8_t *api_key_ptr, uintptr_t api_key_len);
FfiStatus boltffi_function_xybrid_bolt_set_platform_url(const uint8_t *url_ptr, uintptr_t url_len);
FfiStatus boltffi_function_xybrid_bolt_set_speculative_cloud(bool enabled);
bool boltffi_function_xybrid_bolt_has_api_key(void);
bool boltffi_function_xybrid_bolt_is_speculative_cloud_enabled(void);
bool boltffi_function_xybrid_bolt_will_speculate_for_model(const uint8_t *model_id_ptr, uintptr_t model_id_len);
FfiBuf_u8 boltffi_function_xybrid_bolt_version(void);
uint32_t boltffi_function_xybrid_bolt_release_memory(void);
FfiStatus boltffi_function_xybrid_bolt_set_auto_release(bool enabled);
bool boltffi_function_xybrid_bolt_is_auto_release_enabled(void);
FfiBuf_u8 boltffi_function_xybrid_bolt_telemetry_default_endpoint(void);
void boltffi_function_xybrid_bolt_telemetry_flush(void);
void boltffi_function_xybrid_bolt_telemetry_shutdown(void);

#ifdef __cplusplus
}
#endif