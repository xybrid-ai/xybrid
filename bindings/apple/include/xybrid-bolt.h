#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdatomic.h>

typedef struct FfiStatus { int32_t code; } FfiStatus;
typedef struct FfiString { uint8_t* ptr; size_t len; size_t cap; } FfiString;
typedef struct FfiBuf_u8 { uint8_t* ptr; size_t len; size_t cap; size_t align; } FfiBuf_u8;
typedef struct FfiError { FfiString message; } FfiError;
typedef struct BoltFFICallbackHandle { uint64_t handle; const void* vtable; } BoltFFICallbackHandle;

static inline bool boltffi_atomic_u8_cas(volatile uint8_t* target, uint8_t expected, uint8_t desired) {
    return atomic_compare_exchange_strong((_Atomic uint8_t*)target, &expected, desired);
}

static inline uint64_t boltffi_atomic_u64_exchange(volatile uint64_t* target, uint64_t value) {
    return atomic_exchange((_Atomic uint64_t*)target, value);
}

static inline bool boltffi_atomic_u64_cas(volatile uint64_t* target, uint64_t expected, uint64_t desired) {
    return atomic_compare_exchange_strong((_Atomic uint64_t*)target, &expected, desired);
}

static inline uint64_t boltffi_atomic_u64_load(const volatile uint64_t* target) {
    return atomic_load_explicit((const _Atomic uint64_t*)target, memory_order_acquire);
}



struct XybridModel;
struct XybridConversationContext;
struct XybridTelemetryConfig;
struct XybridBundle;

typedef int32_t ___XybridMessageRole;
#define ___XybridMessageRole_System 0
#define ___XybridMessageRole_User 1
#define ___XybridMessageRole_Assistant 2
typedef int32_t ___XybridAbortSignal;
#define ___XybridAbortSignal_MemoryPressureWarn 0
#define ___XybridAbortSignal_MemoryPressureCritical 1
#define ___XybridAbortSignal_ThermalHot 2
#define ___XybridAbortSignal_ThermalCritical 3
typedef int32_t ___XybridOutputType;
#define ___XybridOutputType_Text 0
#define ___XybridOutputType_Audio 1
#define ___XybridOutputType_Embedding 2
#define ___XybridOutputType_Unknown 3
typedef int32_t ___XybridExecutionTarget;
#define ___XybridExecutionTarget_Local 0
#define ___XybridExecutionTarget_Cloud 1
typedef int32_t ___XybridDownloadState;
#define ___XybridDownloadState_Downloading 0
#define ___XybridDownloadState_Ready 1
#define ___XybridDownloadState_Failed 2
typedef int32_t ___XybridStreamEventKind;
#define ___XybridStreamEventKind_Token 0
#define ___XybridStreamEventKind_Complete 1
typedef int32_t ___XybridThermalState;
#define ___XybridThermalState_Normal 0
#define ___XybridThermalState_Warm 1
#define ___XybridThermalState_Hot 2
#define ___XybridThermalState_Critical 3
FfiBuf_u8 boltffi_json_schema_to_gbnf(const uint8_t* schema_json, uintptr_t schema_json_len);void boltffi_set_thermal_state(int32_t state);void boltffi_clear_thermal_state(void);void boltffi_set_battery_level(uint8_t percent);void boltffi_clear_battery_level(void);void boltffi_configure_runtime(const uint8_t* api_key, uintptr_t api_key_len, const uint8_t* gateway_url, uintptr_t gateway_url_len, const uint8_t* ingest_url, uintptr_t ingest_url_len);void boltffi_init_sdk_cache_dir(const uint8_t* cache_dir, uintptr_t cache_dir_len);void boltffi_set_binding(const uint8_t* binding, uintptr_t binding_len);void boltffi_set_api_key(const uint8_t* api_key, uintptr_t api_key_len);void boltffi_set_provider_api_key(const uint8_t* provider, uintptr_t provider_len, const uint8_t* api_key, uintptr_t api_key_len);void boltffi_set_platform_url(const uint8_t* url, uintptr_t url_len);void boltffi_set_speculative_cloud(bool enabled);bool boltffi_is_speculative_cloud_enabled(void);bool boltffi_will_speculate_for_model(const uint8_t* model_id, uintptr_t model_id_len);FfiBuf_u8 boltffi_version(void);FfiBuf_u8 boltffi_telemetry_default_endpoint(void);void boltffi_telemetry_flush(void);void boltffi_telemetry_shutdown(void);struct XybridModel * boltffi_xybrid_model_from_registry(const uint8_t* id, uintptr_t id_len);struct XybridModel * boltffi_xybrid_model_from_registry_speculative(const uint8_t* id, uintptr_t id_len);struct XybridModel * boltffi_xybrid_model_from_directory(const uint8_t* path, uintptr_t path_len);struct XybridModel * boltffi_xybrid_model_from_bundle(const uint8_t* path, uintptr_t path_len);struct XybridModel * boltffi_xybrid_model_from_huggingface(const uint8_t* repo, uintptr_t repo_len);struct XybridModel * boltffi_xybrid_model_from_model_file(const uint8_t* path, uintptr_t path_len);void boltffi_xybrid_model_free(struct XybridModel * handle);FfiBuf_u8 boltffi_xybrid_model_model_id(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_version(const struct XybridModel * self);int32_t boltffi_xybrid_model_output_type(const struct XybridModel * self);bool boltffi_xybrid_model_is_loaded(const struct XybridModel * self);bool boltffi_xybrid_model_is_cloud_serving(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_download_status(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_await_download(const struct XybridModel * self, uint64_t timeout_ms);bool boltffi_xybrid_model_supports_streaming(const struct XybridModel * self);bool boltffi_xybrid_model_supports_token_streaming(const struct XybridModel * self);bool boltffi_xybrid_model_is_llm(const struct XybridModel * self);bool boltffi_xybrid_model_has_voices(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_voices(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_default_voice(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_voice(const struct XybridModel * self, const uint8_t* voice_id, uintptr_t voice_id_len);FfiBuf_u8 boltffi_xybrid_model_run(const struct XybridModel * self, const uint8_t* envelope, uintptr_t envelope_len, const uint8_t* options, uintptr_t options_len);FfiBuf_u8 boltffi_xybrid_model_run_stream(const struct XybridModel * self, const uint8_t* envelope, uintptr_t envelope_len, const uint8_t* options, uintptr_t options_len);FfiBuf_u8 boltffi_xybrid_model_stream_next(const struct XybridModel * self, uint64_t stream_id);FfiBuf_u8 boltffi_xybrid_model_stream_result(const struct XybridModel * self, uint64_t stream_id);void boltffi_xybrid_model_stream_close(const struct XybridModel * self, uint64_t stream_id);FfiBuf_u8 boltffi_xybrid_model_run_with_context(const struct XybridModel * self, const uint8_t* envelope, uintptr_t envelope_len, const struct XybridConversationContext * context, const uint8_t* options, uintptr_t options_len);FfiBuf_u8 boltffi_xybrid_model_run_stream_with_context(const struct XybridModel * self, const uint8_t* envelope, uintptr_t envelope_len, const struct XybridConversationContext * context, const uint8_t* options, uintptr_t options_len);FfiBuf_u8 boltffi_xybrid_model_warmup(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_unload(const struct XybridModel * self);
struct XybridConversationContext * boltffi_xybrid_conversation_context_new(void);struct XybridConversationContext * boltffi_xybrid_conversation_context_with_id(const uint8_t* id, uintptr_t id_len);void boltffi_xybrid_conversation_context_free(struct XybridConversationContext * handle);FfiBuf_u8 boltffi_xybrid_conversation_context_push(const struct XybridConversationContext * self, const uint8_t* envelope, uintptr_t envelope_len);FfiBuf_u8 boltffi_xybrid_conversation_context_set_system(const struct XybridConversationContext * self, const uint8_t* envelope, uintptr_t envelope_len);void boltffi_xybrid_conversation_context_clear(const struct XybridConversationContext * self);FfiBuf_u8 boltffi_xybrid_conversation_context_id(const struct XybridConversationContext * self);uint32_t boltffi_xybrid_conversation_context_history_len(const struct XybridConversationContext * self);bool boltffi_xybrid_conversation_context_has_system(const struct XybridConversationContext * self);void boltffi_xybrid_conversation_context_set_max_history_len(const struct XybridConversationContext * self, uint32_t len);
struct XybridTelemetryConfig * boltffi_xybrid_telemetry_config_new(const uint8_t* api_key, uintptr_t api_key_len);void boltffi_xybrid_telemetry_config_free(struct XybridTelemetryConfig * handle);void boltffi_xybrid_telemetry_config_set_endpoint(const struct XybridTelemetryConfig * self, const uint8_t* endpoint, uintptr_t endpoint_len);void boltffi_xybrid_telemetry_config_set_app_version(const struct XybridTelemetryConfig * self, const uint8_t* version, uintptr_t version_len);void boltffi_xybrid_telemetry_config_set_device_label(const struct XybridTelemetryConfig * self, const uint8_t* label, uintptr_t label_len);void boltffi_xybrid_telemetry_config_set_device_attribute(const struct XybridTelemetryConfig * self, const uint8_t* key, uintptr_t key_len, const uint8_t* value, uintptr_t value_len);void boltffi_xybrid_telemetry_config_set_batch_size(const struct XybridTelemetryConfig * self, uint32_t batch_size);void boltffi_xybrid_telemetry_config_set_flush_interval_secs(const struct XybridTelemetryConfig * self, uint32_t secs);FfiBuf_u8 boltffi_xybrid_telemetry_config_init(const struct XybridTelemetryConfig * self);
struct XybridBundle * boltffi_xybrid_bundle_open(const uint8_t* path, uintptr_t path_len);void boltffi_xybrid_bundle_free(struct XybridBundle * handle);FfiBuf_u8 boltffi_xybrid_bundle_model_id(const struct XybridBundle * self);FfiBuf_u8 boltffi_xybrid_bundle_version(const struct XybridBundle * self);FfiBuf_u8 boltffi_xybrid_bundle_target(const struct XybridBundle * self);FfiBuf_u8 boltffi_xybrid_bundle_hash(const struct XybridBundle * self);bool boltffi_xybrid_bundle_has_metadata(const struct XybridBundle * self);uint32_t boltffi_xybrid_bundle_file_count(const struct XybridBundle * self);FfiBuf_u8 boltffi_xybrid_bundle_file_name(const struct XybridBundle * self, uint32_t index);FfiBuf_u8 boltffi_xybrid_bundle_manifest_json(const struct XybridBundle * self);FfiBuf_u8 boltffi_xybrid_bundle_metadata_json(const struct XybridBundle * self);FfiBuf_u8 boltffi_xybrid_bundle_extract(const struct XybridBundle * self, const uint8_t* output_dir, uintptr_t output_dir_len);

void boltffi_free_string(FfiString s);
void boltffi_free_buf(FfiBuf_u8 buf);
FfiStatus boltffi_last_error_message(FfiString *out);
void boltffi_clear_last_error(void);
