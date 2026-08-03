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
typedef int32_t ___XybridThermalState;
#define ___XybridThermalState_Normal 0
#define ___XybridThermalState_Warm 1
#define ___XybridThermalState_Hot 2
#define ___XybridThermalState_Critical 3
FfiBuf_u8 boltffi_json_schema_to_gbnf(const uint8_t* schema_json, uintptr_t schema_json_len);void boltffi_set_thermal_state(int32_t state);void boltffi_clear_thermal_state(void);void boltffi_set_battery_level(uint8_t percent);void boltffi_clear_battery_level(void);void boltffi_configure_runtime(const uint8_t* api_key, uintptr_t api_key_len, const uint8_t* gateway_url, uintptr_t gateway_url_len, const uint8_t* ingest_url, uintptr_t ingest_url_len);void boltffi_init_sdk_cache_dir(const uint8_t* cache_dir, uintptr_t cache_dir_len);void boltffi_set_binding(const uint8_t* binding, uintptr_t binding_len);void boltffi_set_api_key(const uint8_t* api_key, uintptr_t api_key_len);void boltffi_set_provider_api_key(const uint8_t* provider, uintptr_t provider_len, const uint8_t* api_key, uintptr_t api_key_len);struct XybridModel * boltffi_xybrid_model_from_registry(const uint8_t* id, uintptr_t id_len);struct XybridModel * boltffi_xybrid_model_from_directory(const uint8_t* path, uintptr_t path_len);struct XybridModel * boltffi_xybrid_model_from_bundle(const uint8_t* path, uintptr_t path_len);struct XybridModel * boltffi_xybrid_model_from_huggingface(const uint8_t* repo, uintptr_t repo_len);void boltffi_xybrid_model_free(struct XybridModel * handle);FfiBuf_u8 boltffi_xybrid_model_model_id(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_version(const struct XybridModel * self);int32_t boltffi_xybrid_model_output_type(const struct XybridModel * self);bool boltffi_xybrid_model_is_loaded(const struct XybridModel * self);bool boltffi_xybrid_model_supports_streaming(const struct XybridModel * self);bool boltffi_xybrid_model_is_llm(const struct XybridModel * self);bool boltffi_xybrid_model_has_voices(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_voices(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_default_voice(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_voice(const struct XybridModel * self, const uint8_t* voice_id, uintptr_t voice_id_len);FfiBuf_u8 boltffi_xybrid_model_run(const struct XybridModel * self, const uint8_t* envelope, uintptr_t envelope_len, const uint8_t* options, uintptr_t options_len);FfiBuf_u8 boltffi_xybrid_model_warmup(const struct XybridModel * self);FfiBuf_u8 boltffi_xybrid_model_unload(const struct XybridModel * self);

void boltffi_free_string(FfiString s);
void boltffi_free_buf(FfiBuf_u8 buf);
FfiStatus boltffi_last_error_message(FfiString *out);
void boltffi_clear_last_error(void);
