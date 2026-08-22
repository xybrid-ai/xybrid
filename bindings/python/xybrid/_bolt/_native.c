#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "../boltffi.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
typedef void (*boltffi_python_boltffi_free_string_fn)(FfiString);
static boltffi_python_boltffi_free_string_fn boltffi_python_boltffi_free_string = NULL;
typedef void (*boltffi_python_boltffi_free_buf_fn)(FfiBuf_u8);
static boltffi_python_boltffi_free_buf_fn boltffi_python_boltffi_free_buf = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_buf_from_bytes_fn)(const uint8_t *, uintptr_t);
static boltffi_python_boltffi_buf_from_bytes_fn boltffi_python_boltffi_buf_from_bytes = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_buf_with_len_fn)(uintptr_t);
static boltffi_python_boltffi_buf_with_len_fn boltffi_python_boltffi_buf_with_len = NULL;
typedef FfiStatus (*boltffi_python_boltffi_last_error_message_fn)(FfiString *);
static boltffi_python_boltffi_last_error_message_fn boltffi_python_boltffi_last_error_message = NULL;
typedef void (*boltffi_python_boltffi_clear_last_error_fn)(void);
static boltffi_python_boltffi_clear_last_error_fn boltffi_python_boltffi_clear_last_error = NULL;
typedef void (*boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model_fn)(uint64_t);
static boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model_fn boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_fn)(const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative_fn)(const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory_fn)(const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle_fn)(const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_fn)(const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision_fn)(const uint8_t *, uintptr_t, const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file_fn)(const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version = NULL;
typedef ___XybridOutputType (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type = NULL;
typedef bool (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded = NULL;
typedef bool (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download_fn)(uint64_t, uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download = NULL;
typedef bool (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming = NULL;
typedef bool (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config = NULL;
typedef bool (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling = NULL;
typedef bool (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice_fn)(uint64_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_fn)(uint64_t, const uint8_t *, uintptr_t, const uint8_t *, uintptr_t, FfiBuf_u8 *);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_fn)(uint64_t, const uint8_t *, uintptr_t, const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next_fn)(uint64_t, uint64_t, FfiBuf_u8 *);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result_fn)(uint64_t, uint64_t, FfiBuf_u8 *);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result = NULL;
typedef FfiStatus (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close_fn)(uint64_t, uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context_fn)(uint64_t, const uint8_t *, uintptr_t, uint64_t, const uint8_t *, uintptr_t, FfiBuf_u8 *);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context_fn)(uint64_t, const uint8_t *, uintptr_t, uint64_t, const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload = NULL;
typedef void (*boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context_fn)(uint64_t);
static boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context_fn boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context = NULL;
typedef uint64_t (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new_fn)(void);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new = NULL;
typedef uint64_t (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id_fn)(const uint8_t *, uintptr_t);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push_fn)(uint64_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system_fn)(uint64_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system = NULL;
typedef FfiStatus (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id = NULL;
typedef uint32_t (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history = NULL;
typedef bool (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system = NULL;
typedef FfiStatus (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len_fn)(uint64_t, uint32_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len = NULL;
typedef void (*boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config_fn)(uint64_t);
static boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config_fn boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config = NULL;
typedef uint64_t (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new_fn)(const uint8_t *, uintptr_t);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new = NULL;
typedef FfiStatus (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint_fn)(uint64_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint = NULL;
typedef FfiStatus (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version_fn)(uint64_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version = NULL;
typedef FfiStatus (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label_fn)(uint64_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label = NULL;
typedef FfiStatus (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute_fn)(uint64_t, const uint8_t *, uintptr_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute = NULL;
typedef FfiStatus (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size_fn)(uint64_t, uint32_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size = NULL;
typedef FfiStatus (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs_fn)(uint64_t, uint32_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init = NULL;
typedef void (*boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle_fn)(uint64_t);
static boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle_fn boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open_fn)(const uint8_t *, uintptr_t, uint64_t *);
static boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open_fn boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash = NULL;
typedef bool (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata = NULL;
typedef uint32_t (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count_fn)(uint64_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name_fn)(uint64_t, uint32_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json_fn)(uint64_t, FfiBuf_u8 *);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json_fn)(uint64_t, FfiBuf_u8 *);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract_fn)(uint64_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract_fn boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope_fn)(const uint8_t *, uintptr_t, const uint8_t *, uintptr_t, const uint8_t *, uintptr_t, FfiBuf_u8 *);
static boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope_fn boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf_fn)(const uint8_t *, uintptr_t, FfiBuf_u8 *);
static boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf_fn boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf = NULL;
typedef FfiStatus (*boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state_fn)(___XybridThermalState);
static boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state_fn boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state = NULL;
typedef void (*boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state_fn)(void);
static boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state_fn boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state = NULL;
typedef FfiStatus (*boltffi_python_boltffi_function_xybrid_bolt_set_battery_level_fn)(uint8_t);
static boltffi_python_boltffi_function_xybrid_bolt_set_battery_level_fn boltffi_python_boltffi_function_xybrid_bolt_set_battery_level = NULL;
typedef void (*boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level_fn)(void);
static boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level_fn boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level = NULL;
typedef FfiStatus (*boltffi_python_boltffi_function_xybrid_bolt_configure_runtime_fn)(const uint8_t *, uintptr_t, const uint8_t *, uintptr_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_function_xybrid_bolt_configure_runtime_fn boltffi_python_boltffi_function_xybrid_bolt_configure_runtime = NULL;
typedef FfiStatus (*boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir_fn)(const uint8_t *, uintptr_t);
static boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir_fn boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir = NULL;
typedef FfiStatus (*boltffi_python_boltffi_function_xybrid_bolt_set_binding_fn)(const uint8_t *, uintptr_t);
static boltffi_python_boltffi_function_xybrid_bolt_set_binding_fn boltffi_python_boltffi_function_xybrid_bolt_set_binding = NULL;
typedef FfiStatus (*boltffi_python_boltffi_function_xybrid_bolt_set_api_key_fn)(const uint8_t *, uintptr_t);
static boltffi_python_boltffi_function_xybrid_bolt_set_api_key_fn boltffi_python_boltffi_function_xybrid_bolt_set_api_key = NULL;
typedef FfiStatus (*boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key_fn)(const uint8_t *, uintptr_t, const uint8_t *, uintptr_t);
static boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key_fn boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key = NULL;
typedef FfiStatus (*boltffi_python_boltffi_function_xybrid_bolt_set_platform_url_fn)(const uint8_t *, uintptr_t);
static boltffi_python_boltffi_function_xybrid_bolt_set_platform_url_fn boltffi_python_boltffi_function_xybrid_bolt_set_platform_url = NULL;
typedef FfiStatus (*boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud_fn)(bool);
static boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud_fn boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud = NULL;
typedef bool (*boltffi_python_boltffi_function_xybrid_bolt_has_api_key_fn)(void);
static boltffi_python_boltffi_function_xybrid_bolt_has_api_key_fn boltffi_python_boltffi_function_xybrid_bolt_has_api_key = NULL;
typedef bool (*boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled_fn)(void);
static boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled_fn boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled = NULL;
typedef bool (*boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model_fn)(const uint8_t *, uintptr_t);
static boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model_fn boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_function_xybrid_bolt_version_fn)(void);
static boltffi_python_boltffi_function_xybrid_bolt_version_fn boltffi_python_boltffi_function_xybrid_bolt_version = NULL;
typedef FfiBuf_u8 (*boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint_fn)(void);
static boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint_fn boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint = NULL;
typedef void (*boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush_fn)(void);
static boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush_fn boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush = NULL;
typedef void (*boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown_fn)(void);
static boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown_fn boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown = NULL;

#ifdef _WIN32
static HMODULE boltffi_python_library_handle = NULL;
#else
static void *boltffi_python_library_handle = NULL;
#endif

static void boltffi_python_release_host_state(void);
static int boltffi_python_bind_host_state(void);

static void boltffi_python_clear_symbols(void) {
    boltffi_python_boltffi_free_string = NULL;
    boltffi_python_boltffi_free_buf = NULL;
    boltffi_python_boltffi_buf_from_bytes = NULL;
    boltffi_python_boltffi_buf_with_len = NULL;
    boltffi_python_boltffi_last_error_message = NULL;
    boltffi_python_boltffi_clear_last_error = NULL;
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload = NULL;
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len = NULL;
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init = NULL;
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle = NULL;
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json = NULL;
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_set_battery_level = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_configure_runtime = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_set_binding = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_set_api_key = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_set_platform_url = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_has_api_key = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_version = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush = NULL;
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown = NULL;
}

static void boltffi_python_unload_library(void) {
    boltffi_python_clear_symbols();
    if (boltffi_python_library_handle == NULL) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(boltffi_python_library_handle);
#else
    dlclose(boltffi_python_library_handle);
#endif
    boltffi_python_library_handle = NULL;
}

static int boltffi_python_load_library(PyObject *library_path) {
#ifdef _WIN32
    wchar_t *wide_library_path = NULL;
#else
    const char *utf8_library_path = NULL;
    const char *loader_error = NULL;
#endif
    if (!PyUnicode_Check(library_path)) {
        PyErr_SetString(PyExc_TypeError, "expected str library path");
        return 0;
    }
#ifdef _WIN32
    wide_library_path = PyUnicode_AsWideCharString(library_path, NULL);
    if (wide_library_path == NULL) {
        return 0;
    }
    boltffi_python_library_handle = LoadLibraryW(wide_library_path);
    PyMem_Free(wide_library_path);
    if (boltffi_python_library_handle == NULL) {
        PyErr_Format(PyExc_ImportError, "failed to load native library from %S", library_path);
        return 0;
    }
#else
    utf8_library_path = PyUnicode_AsUTF8(library_path);
    if (utf8_library_path == NULL) {
        return 0;
    }
    dlerror();
    boltffi_python_library_handle = dlopen(utf8_library_path, RTLD_NOW | RTLD_LOCAL);
    if (boltffi_python_library_handle == NULL) {
        loader_error = dlerror();
        if (loader_error == NULL) {
            PyErr_Format(PyExc_ImportError, "failed to load native library from %S", library_path);
        } else {
            PyErr_Format(PyExc_ImportError, "failed to load native library from %S: %s", library_path, loader_error);
        }
        return 0;
    }
#endif
    return 1;
}

static int boltffi_python_bind_symbols(void) {
#ifdef _WIN32
    boltffi_python_boltffi_free_string = (boltffi_python_boltffi_free_string_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_free_string");
#else
    boltffi_python_boltffi_free_string = (boltffi_python_boltffi_free_string_fn)dlsym(boltffi_python_library_handle, "boltffi_free_string");
#endif
    if (boltffi_python_boltffi_free_string == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_free_string");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_free_buf = (boltffi_python_boltffi_free_buf_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_free_buf");
#else
    boltffi_python_boltffi_free_buf = (boltffi_python_boltffi_free_buf_fn)dlsym(boltffi_python_library_handle, "boltffi_free_buf");
#endif
    if (boltffi_python_boltffi_free_buf == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_free_buf");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_buf_from_bytes = (boltffi_python_boltffi_buf_from_bytes_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_buf_from_bytes");
#else
    boltffi_python_boltffi_buf_from_bytes = (boltffi_python_boltffi_buf_from_bytes_fn)dlsym(boltffi_python_library_handle, "boltffi_buf_from_bytes");
#endif
    if (boltffi_python_boltffi_buf_from_bytes == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_buf_from_bytes");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_buf_with_len = (boltffi_python_boltffi_buf_with_len_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_buf_with_len");
#else
    boltffi_python_boltffi_buf_with_len = (boltffi_python_boltffi_buf_with_len_fn)dlsym(boltffi_python_library_handle, "boltffi_buf_with_len");
#endif
    if (boltffi_python_boltffi_buf_with_len == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_buf_with_len");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_last_error_message = (boltffi_python_boltffi_last_error_message_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_last_error_message");
#else
    boltffi_python_boltffi_last_error_message = (boltffi_python_boltffi_last_error_message_fn)dlsym(boltffi_python_library_handle, "boltffi_last_error_message");
#endif
    if (boltffi_python_boltffi_last_error_message == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_last_error_message");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_clear_last_error = (boltffi_python_boltffi_clear_last_error_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_clear_last_error");
#else
    boltffi_python_boltffi_clear_last_error = (boltffi_python_boltffi_clear_last_error_fn)dlsym(boltffi_python_library_handle, "boltffi_clear_last_error");
#endif
    if (boltffi_python_boltffi_clear_last_error == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_clear_last_error");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model = (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_release_class_xybrid_bolt_xybrid_model");
#else
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model = (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model_fn)dlsym(boltffi_python_library_handle, "boltffi_release_class_xybrid_bolt_xybrid_model");
#endif
    if (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_release_class_xybrid_bolt_xybrid_model");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_registry");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_registry");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_model_from_registry");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_directory");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_directory");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_model_from_directory");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_model_id");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_model_id");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_model_id");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_version");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_version");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_version");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_output_type");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_output_type");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_output_type");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_download_status");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_download_status");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_download_status");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_await_download");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_await_download");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_await_download");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_is_llm");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_is_llm");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_is_llm");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_has_voices");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_has_voices");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_has_voices");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_voices");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_voices");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_voices");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_default_voice");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_default_voice");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_default_voice");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_voice");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_voice");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_voice");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_run");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_run");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_run");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_run_stream");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_run_stream");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_run_stream");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_stream_next");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_stream_next");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_stream_next");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_stream_result");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_stream_result");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_stream_result");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_stream_close");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_stream_close");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_stream_close");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_warmup");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_warmup");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_warmup");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_unload");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_model_unload");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_model_unload");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context = (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_release_class_xybrid_bolt_xybrid_conversation_context");
#else
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context = (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context_fn)dlsym(boltffi_python_library_handle, "boltffi_release_class_xybrid_bolt_xybrid_conversation_context");
#endif
    if (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_release_class_xybrid_bolt_xybrid_conversation_context");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config = (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_release_class_xybrid_bolt_xybrid_telemetry_config");
#else
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config = (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config_fn)dlsym(boltffi_python_library_handle, "boltffi_release_class_xybrid_bolt_xybrid_telemetry_config");
#endif
    if (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_release_class_xybrid_bolt_xybrid_telemetry_config");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle = (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_release_class_xybrid_bolt_xybrid_bundle");
#else
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle = (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle_fn)dlsym(boltffi_python_library_handle, "boltffi_release_class_xybrid_bolt_xybrid_bundle");
#endif
    if (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_release_class_xybrid_bolt_xybrid_bundle");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_bundle_open");
#else
    boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open = (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open_fn)dlsym(boltffi_python_library_handle, "boltffi_init_class_xybrid_bolt_xybrid_bundle_open");
#endif
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_init_class_xybrid_bolt_xybrid_bundle_open");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_version");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_version");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_version");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_target");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_target");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_target");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_hash");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_hash");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_hash");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_extract");
#else
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract = (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract_fn)dlsym(boltffi_python_library_handle, "boltffi_method_class_xybrid_bolt_xybrid_bundle_extract");
#endif
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_method_class_xybrid_bolt_xybrid_bundle_extract");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope = (boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_tool_results_envelope");
#else
    boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope = (boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_tool_results_envelope");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_tool_results_envelope");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf = (boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_json_schema_to_gbnf");
#else
    boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf = (boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_json_schema_to_gbnf");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_json_schema_to_gbnf");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state = (boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_thermal_state");
#else
    boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state = (boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_thermal_state");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_set_thermal_state");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state = (boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_clear_thermal_state");
#else
    boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state = (boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_clear_thermal_state");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_clear_thermal_state");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_set_battery_level = (boltffi_python_boltffi_function_xybrid_bolt_set_battery_level_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_battery_level");
#else
    boltffi_python_boltffi_function_xybrid_bolt_set_battery_level = (boltffi_python_boltffi_function_xybrid_bolt_set_battery_level_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_battery_level");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_set_battery_level == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_set_battery_level");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level = (boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_clear_battery_level");
#else
    boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level = (boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_clear_battery_level");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_clear_battery_level");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_configure_runtime = (boltffi_python_boltffi_function_xybrid_bolt_configure_runtime_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_configure_runtime");
#else
    boltffi_python_boltffi_function_xybrid_bolt_configure_runtime = (boltffi_python_boltffi_function_xybrid_bolt_configure_runtime_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_configure_runtime");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_configure_runtime == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_configure_runtime");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir = (boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_init_sdk_cache_dir");
#else
    boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir = (boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_init_sdk_cache_dir");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_init_sdk_cache_dir");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_set_binding = (boltffi_python_boltffi_function_xybrid_bolt_set_binding_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_binding");
#else
    boltffi_python_boltffi_function_xybrid_bolt_set_binding = (boltffi_python_boltffi_function_xybrid_bolt_set_binding_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_binding");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_set_binding == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_set_binding");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_set_api_key = (boltffi_python_boltffi_function_xybrid_bolt_set_api_key_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_api_key");
#else
    boltffi_python_boltffi_function_xybrid_bolt_set_api_key = (boltffi_python_boltffi_function_xybrid_bolt_set_api_key_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_api_key");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_set_api_key == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_set_api_key");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key = (boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_provider_api_key");
#else
    boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key = (boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_provider_api_key");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_set_provider_api_key");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_set_platform_url = (boltffi_python_boltffi_function_xybrid_bolt_set_platform_url_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_platform_url");
#else
    boltffi_python_boltffi_function_xybrid_bolt_set_platform_url = (boltffi_python_boltffi_function_xybrid_bolt_set_platform_url_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_platform_url");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_set_platform_url == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_set_platform_url");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud = (boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_speculative_cloud");
#else
    boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud = (boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_set_speculative_cloud");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_set_speculative_cloud");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_has_api_key = (boltffi_python_boltffi_function_xybrid_bolt_has_api_key_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_has_api_key");
#else
    boltffi_python_boltffi_function_xybrid_bolt_has_api_key = (boltffi_python_boltffi_function_xybrid_bolt_has_api_key_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_has_api_key");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_has_api_key == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_has_api_key");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled = (boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_is_speculative_cloud_enabled");
#else
    boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled = (boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_is_speculative_cloud_enabled");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_is_speculative_cloud_enabled");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model = (boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_will_speculate_for_model");
#else
    boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model = (boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_will_speculate_for_model");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_will_speculate_for_model");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_version = (boltffi_python_boltffi_function_xybrid_bolt_version_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_version");
#else
    boltffi_python_boltffi_function_xybrid_bolt_version = (boltffi_python_boltffi_function_xybrid_bolt_version_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_version");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_version == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_version");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint = (boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_telemetry_default_endpoint");
#else
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint = (boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_telemetry_default_endpoint");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_telemetry_default_endpoint");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush = (boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_telemetry_flush");
#else
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush = (boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_telemetry_flush");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_telemetry_flush");
        return 0;
    }
#ifdef _WIN32
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown = (boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown_fn)GetProcAddress(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_telemetry_shutdown");
#else
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown = (boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown_fn)dlsym(boltffi_python_library_handle, "boltffi_function_xybrid_bolt_telemetry_shutdown");
#endif
    if (boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown == NULL) {
        boltffi_python_unload_library();
        PyErr_SetString(PyExc_ImportError, "failed to resolve native symbol " "boltffi_function_xybrid_bolt_telemetry_shutdown");
        return 0;
    }
    return 1;
}

static PyObject *boltffi_python_initialize_loader(PyObject *self, PyObject *library_path) {
    (void)self;
    if (boltffi_python_library_handle != NULL) {
        Py_RETURN_NONE;
    }
    if (!boltffi_python_load_library(library_path)) {
        return NULL;
    }
    if (!boltffi_python_bind_symbols()) {
        return NULL;
    }
    if (!boltffi_python_bind_host_state()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static void boltffi_python_free_module(void *module) {
    (void)module;
    boltffi_python_release_host_state();
    boltffi_python_unload_library();
}

static void boltffi_python_write_u16_le(uint8_t *buffer, uint16_t value) {
    buffer[0] = (uint8_t)(value & 0xffu);
    buffer[1] = (uint8_t)((value >> 8) & 0xffu);
}

static void boltffi_python_write_u32_le(uint8_t *buffer, uint32_t value) {
    buffer[0] = (uint8_t)(value & 0xffu);
    buffer[1] = (uint8_t)((value >> 8) & 0xffu);
    buffer[2] = (uint8_t)((value >> 16) & 0xffu);
    buffer[3] = (uint8_t)((value >> 24) & 0xffu);
}

static void boltffi_python_write_u64_le(uint8_t *buffer, uint64_t value) {
    buffer[0] = (uint8_t)(value & 0xffu);
    buffer[1] = (uint8_t)((value >> 8) & 0xffu);
    buffer[2] = (uint8_t)((value >> 16) & 0xffu);
    buffer[3] = (uint8_t)((value >> 24) & 0xffu);
    buffer[4] = (uint8_t)((value >> 32) & 0xffu);
    buffer[5] = (uint8_t)((value >> 40) & 0xffu);
    buffer[6] = (uint8_t)((value >> 48) & 0xffu);
    buffer[7] = (uint8_t)((value >> 56) & 0xffu);
}

static int boltffi_python_wire_fixed(const uint8_t *payload, Py_ssize_t payload_len, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = PyBytes_FromStringAndSize((const char *)payload, payload_len);
    if (wire == NULL) {
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)payload_len;
    return 1;
}

static int boltffi_python_wire_payload(const uint8_t *payload, Py_ssize_t payload_len, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    if (payload_len < 0 || (uint64_t)payload_len > UINT32_MAX || payload_len > PY_SSIZE_T_MAX - 4) {
        PyErr_SetString(PyExc_OverflowError, "payload is too large");
        return 0;
    }
    Py_ssize_t wire_len = payload_len + 4;
    PyObject *wire = PyBytes_FromStringAndSize(NULL, wire_len);
    if (wire == NULL) {
        return 0;
    }
    uint8_t *bytes = (uint8_t *)PyBytes_AS_STRING(wire);
    boltffi_python_write_u32_le(bytes, (uint32_t)payload_len);
    if (payload_len > 0) {
        memcpy(bytes + 4, payload, (size_t)payload_len);
    }
    *out_wire = wire;
    *out_ptr = bytes;
    *out_len = (uintptr_t)wire_len;
    return 1;
}


static int boltffi_python_wire_string(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    Py_ssize_t len = 0;
    const char *utf8 = PyUnicode_AsUTF8AndSize(value, &len);
    if (utf8 == NULL) {
        return 0;
    }
    return boltffi_python_wire_payload((const uint8_t *)utf8, len, out_wire, out_ptr, out_len);
}



static int boltffi_python_wire_raw(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    Py_buffer view;
    if (PyObject_GetBuffer(value, &view, PyBUF_CONTIG_RO) < 0) {
        return 0;
    }
    int ok = boltffi_python_wire_fixed((const uint8_t *)view.buf, view.len, out_wire, out_ptr, out_len);
    PyBuffer_Release(&view);
    return ok;
}


static PyObject *boltffi_python_wire_codecs = NULL;

static int boltffi_python_validate_codec_bytes(const uint8_t *ptr, uintptr_t len) {
    if (ptr == NULL && len != 0) {
        PyErr_SetString(PyExc_RuntimeError, "native callback argument contains an invalid buffer");
        return 0;
    }
    if (len > PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "native callback argument is too large");
        return 0;
    }
    return 1;
}

static PyObject *boltffi_python_register_wire_codec(PyObject *self, PyObject *args) {
    const char *key = NULL;
    PyObject *callable = NULL;
    (void)self;
    if (!PyArg_ParseTuple(args, "sO", &key, &callable)) {
        return NULL;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "wire codec must be callable");
        return NULL;
    }
    if (boltffi_python_wire_codecs == NULL) {
        boltffi_python_wire_codecs = PyDict_New();
        if (boltffi_python_wire_codecs == NULL) {
            return NULL;
        }
    }
    if (PyDict_SetItemString(boltffi_python_wire_codecs, key, callable) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *boltffi_python_wire_codec(const char *key) {
    PyObject *callable = NULL;
    if (boltffi_python_wire_codecs == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "wire codec registry is not initialized");
        return NULL;
    }
    callable = PyDict_GetItemString(boltffi_python_wire_codecs, key);
    if (callable == NULL) {
        PyErr_Format(PyExc_RuntimeError, "wire codec %s is not registered", key);
        return NULL;
    }
    return callable;
}

static PyObject *boltffi_python_decode_wire_codec(const char *key, const uint8_t *ptr, uintptr_t len) {
    PyObject *callable = NULL;
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_codec_bytes(ptr, len)) {
        return NULL;
    }
    callable = boltffi_python_wire_codec(key);
    if (callable == NULL) {
        return NULL;
    }
    wire = PyBytes_FromStringAndSize((const char *)ptr, (Py_ssize_t)len);
    if (wire == NULL) {
        return NULL;
    }
    result = PyObject_CallOneArg(callable, wire);
    Py_DECREF(wire);
    return result;
}

static int boltffi_python_encode_wire_codec(const char *key, PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *callable = boltffi_python_wire_codec(key);
    PyObject *wire = NULL;
    if (callable == NULL) {
        return 0;
    }
    wire = PyObject_CallOneArg(callable, value);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "wire codec must return bytes");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}


static PyObject *boltffi_python_decode_read_fe83cddcf3822a1d(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_fe83cddcf3822a1d", ptr, len);
}

static PyObject *boltffi_python_decode_read_89cd31291d2aefa4(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_89cd31291d2aefa4", ptr, len);
}

static PyObject *boltffi_python_decode_read_29c0b1cb6cb65e99(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_29c0b1cb6cb65e99", ptr, len);
}

static PyObject *boltffi_python_decode_read_49d0adb26a1528e6(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_49d0adb26a1528e6", ptr, len);
}

static PyObject *boltffi_python_decode_read_bd1359a0ca4e78d7(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_bd1359a0ca4e78d7", ptr, len);
}

static PyObject *boltffi_python_decode_read_74dbe00a1a77ad93(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_74dbe00a1a77ad93", ptr, len);
}

static PyObject *boltffi_python_decode_read_c9bb5dd3c2ec1b2a(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_c9bb5dd3c2ec1b2a", ptr, len);
}

static PyObject *boltffi_python_decode_read_e9a0b9fd71f8c9ff(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_e9a0b9fd71f8c9ff", ptr, len);
}

static PyObject *boltffi_python_decode_read_3cfe09c223256b1b(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_3cfe09c223256b1b", ptr, len);
}

static PyObject *boltffi_python_decode_read_9415281aa52df749(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_9415281aa52df749", ptr, len);
}

static PyObject *boltffi_python_decode_read_c9e5fd91113e36a2(const uint8_t *ptr, uintptr_t len) {
    return boltffi_python_decode_wire_codec("read_c9e5fd91113e36a2", ptr, len);
}


static int boltffi_python_encode_write_c26bffea5b1b16cc(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_c26bffea5b1b16cc", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_766cdeb069dd2b0a(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_766cdeb069dd2b0a", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_23c08924af812de7(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_23c08924af812de7", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_8b5b57b4a65a4084(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_8b5b57b4a65a4084", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_8d84d7157f6e715c(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_8d84d7157f6e715c", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_62eeac930738df49(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_62eeac930738df49", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_922e13039dd3c493(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_922e13039dd3c493", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_45cfac4c89613282(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_45cfac4c89613282", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_736b3e4af7f4fdd8(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_736b3e4af7f4fdd8", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_f1696b1e73a7f219(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_f1696b1e73a7f219", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_15a81e8bd2929d67(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_15a81e8bd2929d67", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_94d2821c547dda88(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_94d2821c547dda88", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_ed06f1a2bac0816e(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_ed06f1a2bac0816e", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_1b888e23ceb4a009(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_1b888e23ceb4a009", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_3f05cdfbd6f68333(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_3f05cdfbd6f68333", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_544f2725dda888e0(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_544f2725dda888e0", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_d82fa724b184c72a(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_d82fa724b184c72a", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_cd5b56c1c6bfc6e0(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_cd5b56c1c6bfc6e0", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_a4eb0446f96b83ef(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_a4eb0446f96b83ef", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_5eb4d1ef4dc0ea3f(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_5eb4d1ef4dc0ea3f", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_a67118e385bc3069(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_a67118e385bc3069", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_73b9be8d33badc3c(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_73b9be8d33badc3c", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_a087f842b9a13bc6(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_a087f842b9a13bc6", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_c0b19b1465c99138(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_c0b19b1465c99138", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_b4a023e995953df2(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_b4a023e995953df2", value, out_wire, out_ptr, out_len);
}

static int boltffi_python_encode_write_83cc917c5525e5c3(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    return boltffi_python_encode_wire_codec("write_83cc917c5525e5c3", value, out_wire, out_ptr, out_len);
}



static uint16_t boltffi_python_read_u16_le(const uint8_t *buffer) {
    return ((uint16_t)buffer[0])
        | ((uint16_t)buffer[1] << 8);
}

static uint32_t boltffi_python_read_u32_le(const uint8_t *buffer) {
    return ((uint32_t)buffer[0])
        | ((uint32_t)buffer[1] << 8)
        | ((uint32_t)buffer[2] << 16)
        | ((uint32_t)buffer[3] << 24);
}

static uint64_t boltffi_python_read_u64_le(const uint8_t *buffer) {
    return ((uint64_t)buffer[0])
        | ((uint64_t)buffer[1] << 8)
        | ((uint64_t)buffer[2] << 16)
        | ((uint64_t)buffer[3] << 24)
        | ((uint64_t)buffer[4] << 32)
        | ((uint64_t)buffer[5] << 40)
        | ((uint64_t)buffer[6] << 48)
        | ((uint64_t)buffer[7] << 56);
}

typedef struct {
    const uint8_t *ptr;
    uintptr_t len;
    uintptr_t offset;
} boltffi_python_wire_reader;

typedef struct {
    uint8_t *ptr;
    uintptr_t len;
    uintptr_t offset;
} boltffi_python_wire_writer;

static int boltffi_python_wire_add(uintptr_t *size, uintptr_t amount) {
    if (amount > (uintptr_t)PY_SSIZE_T_MAX - *size) {
        PyErr_SetString(PyExc_OverflowError, "wire payload is too large");
        return 0;
    }
    *size += amount;
    return 1;
}

static int boltffi_python_wire_reader_read(boltffi_python_wire_reader *reader, uintptr_t count, const uint8_t **out) {
    if (count > reader->len - reader->offset) {
        PyErr_SetString(PyExc_ValueError, "truncated BoltFFI wire bytes");
        return 0;
    }
    *out = reader->ptr + reader->offset;
    reader->offset += count;
    return 1;
}

static int boltffi_python_wire_reader_u8(boltffi_python_wire_reader *reader, uint8_t *out) {
    const uint8_t *bytes = NULL;
    if (!boltffi_python_wire_reader_read(reader, 1, &bytes)) {
        return 0;
    }
    *out = bytes[0];
    return 1;
}

static int boltffi_python_wire_reader_u16(boltffi_python_wire_reader *reader, uint16_t *out) {
    const uint8_t *bytes = NULL;
    if (!boltffi_python_wire_reader_read(reader, 2, &bytes)) {
        return 0;
    }
    *out = boltffi_python_read_u16_le(bytes);
    return 1;
}

static int boltffi_python_wire_reader_u32(boltffi_python_wire_reader *reader, uint32_t *out) {
    const uint8_t *bytes = NULL;
    if (!boltffi_python_wire_reader_read(reader, 4, &bytes)) {
        return 0;
    }
    *out = boltffi_python_read_u32_le(bytes);
    return 1;
}

static int boltffi_python_wire_reader_u64(boltffi_python_wire_reader *reader, uint64_t *out) {
    const uint8_t *bytes = NULL;
    if (!boltffi_python_wire_reader_read(reader, 8, &bytes)) {
        return 0;
    }
    *out = boltffi_python_read_u64_le(bytes);
    return 1;
}

static int boltffi_python_wire_writer_write(boltffi_python_wire_writer *writer, const uint8_t *payload, uintptr_t len) {
    if (len > writer->len - writer->offset) {
        PyErr_SetString(PyExc_RuntimeError, "wire writer overflow");
        return 0;
    }
    if (len != 0) {
        memcpy(writer->ptr + writer->offset, payload, (size_t)len);
    }
    writer->offset += len;
    return 1;
}

static int boltffi_python_wire_writer_u8(boltffi_python_wire_writer *writer, uint8_t value) {
    return boltffi_python_wire_writer_write(writer, &value, 1);
}

static int boltffi_python_wire_writer_u16(boltffi_python_wire_writer *writer, uint16_t value) {
    uint8_t bytes[2];
    boltffi_python_write_u16_le(bytes, value);
    return boltffi_python_wire_writer_write(writer, bytes, 2);
}

static int boltffi_python_wire_writer_u32(boltffi_python_wire_writer *writer, uint32_t value) {
    uint8_t bytes[4];
    boltffi_python_write_u32_le(bytes, value);
    return boltffi_python_wire_writer_write(writer, bytes, 4);
}

static int boltffi_python_wire_writer_u64(boltffi_python_wire_writer *writer, uint64_t value) {
    uint8_t bytes[8];
    boltffi_python_write_u64_le(bytes, value);
    return boltffi_python_wire_writer_write(writer, bytes, 8);
}

static int boltffi_python_validate_owned_memory(FfiBuf_u8 buffer) {
    if (buffer.ptr == NULL && buffer.len != 0) {
        PyErr_SetString(PyExc_RuntimeError, "native function returned invalid buffer");
        return 0;
    }
    if (buffer.len > PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "native buffer is too large");
        return 0;
    }
    return 1;
}

static int boltffi_python_validate_owned_buffer(FfiBuf_u8 buffer) {
    if (!boltffi_python_validate_owned_memory(buffer)) {
        return 0;
    }
    if (buffer.len < 4) {
        PyErr_SetString(PyExc_RuntimeError, "native function returned truncated wire buffer");
        return 0;
    }
    return 1;
}

static int boltffi_python_validate_owned_fixed_buffer(FfiBuf_u8 buffer, uintptr_t expected_len) {
    if (!boltffi_python_validate_owned_memory(buffer)) {
        return 0;
    }
    if (buffer.len != expected_len) {
        PyErr_SetString(PyExc_RuntimeError, "native function returned wrong fixed wire size");
        return 0;
    }
    return 1;
}

static void boltffi_python_release_owned_buffer(FfiBuf_u8 buffer) {
    boltffi_python_boltffi_free_buf(buffer);
}



static PyObject *boltffi_python_decode_owned_utf8(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_buffer(buffer)) {
        goto done;
    }
    uint32_t len = boltffi_python_read_u32_le(buffer.ptr);
    if ((uintptr_t)len > buffer.len - 4) {
        PyErr_SetString(PyExc_RuntimeError, "native function returned truncated string buffer");
        goto done;
    }
    if (len > (uint32_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "native string is too large");
        goto done;
    }
    result = PyUnicode_FromStringAndSize((const char *)(buffer.ptr + 4), (Py_ssize_t)len);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}



static PyObject *boltffi_python_decode_owned_raw_wire(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    result = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}



static int boltffi_python_validate_registered_type_object(PyObject *type_object, const char *type_name) {
    if (!PyType_Check(type_object)) {
        PyErr_Format(PyExc_TypeError, "expected type for %s", type_name);
        return 0;
    }
    return 1;
}

static int boltffi_python_store_registered_type(PyObject **type_slot, PyObject *type_object, const char *type_name) {
    if (!boltffi_python_validate_registered_type_object(type_object, type_name)) {
        return 0;
    }
    Py_INCREF(type_object);
    Py_XDECREF(*type_slot);
    *type_slot = type_object;
    return 1;
}

static int boltffi_python_expect_registered_type(PyObject *type_object, const char *type_name) {
    if (type_object != NULL) {
        return 1;
    }
    PyErr_Format(PyExc_ImportError, "native type %s is not registered", type_name);
    return 0;
}

static int boltffi_python_expect_type_instance(PyObject *value, PyObject *type_object, const char *type_name) {
    int is_instance = 0;
    if (!boltffi_python_expect_registered_type(type_object, type_name)) {
        return 0;
    }
    is_instance = PyObject_IsInstance(value, type_object);
    if (is_instance < 0) {
        return 0;
    }
    if (is_instance == 0) {
        PyErr_Format(PyExc_TypeError, "expected %s", type_name);
        return 0;
    }
    return 1;
}

static PyObject *boltffi_python_get_record_field(PyObject *value, const char *record_name, const char *field_name) {
    PyObject *field_value = PyObject_GetAttrString(value, field_name);
    if (field_value == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError, "%s is missing field %s", record_name, field_name);
    }
    return field_value;
}

static PyObject *boltffi_python_box_registered_record(PyObject *type_object, PyObject *constructor_args, const char *record_name) {
    PyObject *record_value = NULL;
    if (constructor_args == NULL) {
        return NULL;
    }
    if (!boltffi_python_expect_registered_type(type_object, record_name)) {
        Py_DECREF(constructor_args);
        return NULL;
    }
    record_value = PyObject_CallObject(type_object, constructor_args);
    Py_DECREF(constructor_args);
    return record_value;
}


typedef PyObject *(*boltffi_python_load_c_style_enum_member_fn)(PyObject *, Py_ssize_t);

typedef struct boltffi_python_c_style_enum_registration {
    PyObject *type_object;
    Py_ssize_t member_count;
    PyObject **members_by_wire_tag;
} boltffi_python_c_style_enum_registration;

static void boltffi_python_release_registered_enum_members(PyObject **members_by_wire_tag, Py_ssize_t member_count) {
    Py_ssize_t member_index = 0;
    for (member_index = 0; member_index < member_count; member_index += 1) {
        Py_XDECREF(members_by_wire_tag[member_index]);
        members_by_wire_tag[member_index] = NULL;
    }
}

static void boltffi_python_clear_c_style_enum_registration(
    boltffi_python_c_style_enum_registration *registration
) {
    Py_XDECREF(registration->type_object);
    registration->type_object = NULL;
    boltffi_python_release_registered_enum_members(
        registration->members_by_wire_tag,
        registration->member_count
    );
}

static PyObject *boltffi_python_load_c_style_enum_member(
    PyObject *type_object,
    const char *enum_name,
    const char *member_name,
    PyObject *native_value
) {
    PyObject *named_member = NULL;
    PyObject *resolved_member = NULL;
    if (native_value == NULL) {
        return NULL;
    }
    named_member = PyObject_GetAttrString(type_object, member_name);
    if (named_member == NULL) {
        return NULL;
    }
    resolved_member = PyObject_CallOneArg(type_object, native_value);
    if (resolved_member == NULL) {
        Py_DECREF(named_member);
        return NULL;
    }
    if (named_member != resolved_member) {
        PyErr_Format(PyExc_ValueError, "native enum %s member %s has the wrong value", enum_name, member_name);
        Py_DECREF(named_member);
        Py_DECREF(resolved_member);
        return NULL;
    }
    Py_DECREF(resolved_member);
    return named_member;
}

static int boltffi_python_store_c_style_enum_registration(
    boltffi_python_c_style_enum_registration *registration,
    PyObject *type_object,
    const char *enum_name,
    boltffi_python_load_c_style_enum_member_fn load_member
) {
    PyObject **loaded_members = NULL;
    Py_ssize_t member_index = 0;
    if (!boltffi_python_validate_registered_type_object(type_object, enum_name)) {
        return 0;
    }
    loaded_members = PyMem_Calloc((size_t)registration->member_count, sizeof(PyObject *));
    if (loaded_members == NULL) {
        PyErr_NoMemory();
        return 0;
    }
    for (member_index = 0; member_index < registration->member_count; member_index += 1) {
        loaded_members[member_index] = load_member(type_object, member_index);
        if (loaded_members[member_index] == NULL) {
            boltffi_python_release_registered_enum_members(loaded_members, registration->member_count);
            PyMem_Free(loaded_members);
            return 0;
        }
    }
    boltffi_python_clear_c_style_enum_registration(registration);
    if (!boltffi_python_store_registered_type(&registration->type_object, type_object, enum_name)) {
        boltffi_python_release_registered_enum_members(loaded_members, registration->member_count);
        PyMem_Free(loaded_members);
        return 0;
    }
    for (member_index = 0; member_index < registration->member_count; member_index += 1) {
        registration->members_by_wire_tag[member_index] = loaded_members[member_index];
    }
    PyMem_Free(loaded_members);
    return 1;
}

static int boltffi_python_expect_enum_instance(
    PyObject *value,
    const boltffi_python_c_style_enum_registration *registration,
    const char *enum_name
) {
    return boltffi_python_expect_type_instance(value, registration->type_object, enum_name);
}

static PyObject *boltffi_python_box_registered_enum_member(
    const boltffi_python_c_style_enum_registration *registration,
    Py_ssize_t member_index,
    const char *enum_name
) {
    PyObject *member = NULL;
    if (!boltffi_python_expect_registered_type(registration->type_object, enum_name)) {
        return NULL;
    }
    if (member_index < 0 || member_index >= registration->member_count) {
        PyErr_SetString(PyExc_RuntimeError, "native enum member index is invalid");
        return NULL;
    }
    if (registration->members_by_wire_tag[member_index] == NULL) {
        PyErr_Format(PyExc_ImportError, "native enum %s member cache is not initialized", enum_name);
        return NULL;
    }
    member = registration->members_by_wire_tag[member_index];
    Py_INCREF(member);
    return member;
}



static int boltffi_python_parse_bool(PyObject *value, bool *out) {
    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "expected bool");
        return 0;
    }
    *out = value == Py_True;
    return 1;
}

static PyObject *boltffi_python_box_bool(bool value) {
    return PyBool_FromLong(value ? 1 : 0);
}

















static int boltffi_python_parse_u8(PyObject *value, uint8_t *out) {
    unsigned long long parsed = PyLong_AsUnsignedLongLong(value);
    if (parsed == (unsigned long long)-1 && PyErr_Occurred()) {
        return 0;
    }
    if (parsed > UINT8_MAX) {
        PyErr_SetString(PyExc_OverflowError, "expected u8");
        return 0;
    }
    *out = (uint8_t)parsed;
    return 1;
}

static PyObject *boltffi_python_box_u8(uint8_t value) {
    return PyLong_FromUnsignedLong((unsigned long)value);
}


















static int boltffi_python_parse_i32(PyObject *value, int32_t *out) {
    long long parsed = PyLong_AsLongLong(value);
    if (parsed == -1 && PyErr_Occurred()) {
        return 0;
    }
    if (parsed < INT32_MIN || parsed > INT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "expected i32");
        return 0;
    }
    *out = (int32_t)parsed;
    return 1;
}

static PyObject *boltffi_python_box_i32(int32_t value) {
    return PyLong_FromLong((long)value);
}
















static int boltffi_python_parse_u32(PyObject *value, uint32_t *out) {
    unsigned long long parsed = PyLong_AsUnsignedLongLong(value);
    if (parsed == (unsigned long long)-1 && PyErr_Occurred()) {
        return 0;
    }
    if (parsed > UINT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "expected u32");
        return 0;
    }
    *out = (uint32_t)parsed;
    return 1;
}

static PyObject *boltffi_python_box_u32(uint32_t value) {
    return PyLong_FromUnsignedLong((unsigned long)value);
}

















static int boltffi_python_parse_u64(PyObject *value, uint64_t *out) {
    unsigned long long parsed = PyLong_AsUnsignedLongLong(value);
    if (parsed == (unsigned long long)-1 && PyErr_Occurred()) {
        return 0;
    }
    *out = (uint64_t)parsed;
    return 1;
}

static PyObject *boltffi_python_box_u64(uint64_t value) {
    return PyLong_FromUnsignedLongLong((unsigned long long)value);
}









static PyObject *boltffi_python_decode_owned_bool(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_fixed_buffer(buffer, 1)) {
        goto done;
    }
    if (buffer.ptr[0] > 1) {
        PyErr_SetString(PyExc_RuntimeError, "native function returned invalid bool wire value");
        goto done;
    }
    result = boltffi_python_box_bool(buffer.ptr[0] == 1);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}













static PyObject *boltffi_python_decode_owned_optional_bool(FfiBuf_u8 buffer) {
    PyObject *result = NULL;


    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    if (buffer.len < 1) {
        PyErr_SetString(PyExc_RuntimeError, "native function returned truncated optional scalar");
        goto done;
    }
    if (buffer.ptr[0] == 0) {
        if (buffer.len != 1) {
            PyErr_SetString(PyExc_RuntimeError, "native function returned invalid none scalar payload");
            goto done;
        }
        Py_INCREF(Py_None);
        result = Py_None;
        goto done;
    }
    if (buffer.ptr[0] != 1) {
        PyErr_SetString(PyExc_RuntimeError, "native function returned invalid optional scalar tag");
        goto done;
    }
    if (buffer.len != 1 + 1) {
        PyErr_SetString(PyExc_RuntimeError, "native function returned invalid optional scalar payload");
        goto done;
    }

    if (buffer.ptr[1] > 1) {
        PyErr_SetString(PyExc_RuntimeError, "native function returned invalid bool wire value");
        goto done;
    }
    result = boltffi_python_box_bool(buffer.ptr[1] == 1);













done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}




static PyObject *boltffi_python_xybrid_metadata_entry_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_metadata_entry(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_metadata_entry() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_metadata_entry_type, args[0], "XybridMetadataEntry")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_metadata_entry(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    uint8_t *bytes = NULL;
    uintptr_t wire_len = 0;
    boltffi_python_wire_writer writer = {0};
    int ok = 0;
    PyObject *key_value = NULL;
    PyObject *value_value = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_metadata_entry_type, "XybridMetadataEntry")) {
        goto done;
    }
    key_value = boltffi_python_get_record_field(value, "XybridMetadataEntry", "key");
    if (key_value == NULL) {
        goto done;
    }
    value_value = boltffi_python_get_record_field(value, "XybridMetadataEntry", "value");
    if (value_value == NULL) {
        goto done;
    }
    {
        PyObject *field_value = key_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = value_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    wire = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)wire_len);
    if (wire == NULL) {
        goto done;
    }
    bytes = (uint8_t *)PyBytes_AS_STRING(wire);
    writer.ptr = bytes;
    writer.len = wire_len;
    writer.offset = 0;
    {
        PyObject *field_value = key_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = value_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    if (writer.offset != writer.len) {
        PyErr_SetString(PyExc_RuntimeError, "wire writer produced wrong byte count");
        goto done;
    }
    *out_wire = wire;
    *out_ptr = bytes;
    *out_len = wire_len;
    wire = NULL;
    ok = 1;
done:
    Py_XDECREF(wire);
    Py_XDECREF(key_value);
    Py_XDECREF(value_value);
    return ok;
}

static PyObject *boltffi_python_decode_owned_xybrid_metadata_entry_read(boltffi_python_wire_reader *reader) {
    PyObject *result = NULL;
    PyObject *values[2] = {0};
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[0] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[0] == NULL) {
            goto done;
        }
    }
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[1] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[1] == NULL) {
            goto done;
        }
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_metadata_entry_type, "XybridMetadataEntry")) {
        goto done;
    }
    result = PyObject_Vectorcall(boltffi_python_xybrid_metadata_entry_type, values, 2, NULL);
done:
    Py_XDECREF(values[0]);
    Py_XDECREF(values[1]);
    return result;
}

static PyObject *boltffi_python_decode_owned_xybrid_metadata_entry(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    boltffi_python_wire_reader reader = {0};
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    reader.ptr = buffer.ptr;
    reader.len = buffer.len;
    reader.offset = 0;
    result = boltffi_python_decode_owned_xybrid_metadata_entry_read(&reader);
    if (result != NULL && reader.offset != reader.len) {
        Py_CLEAR(result);
        PyErr_SetString(PyExc_ValueError, "trailing BoltFFI wire bytes");
    }
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_envelope_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_envelope(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_envelope() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_envelope_type, args[0], "XybridEnvelope")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_envelope(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_envelope_type, "XybridEnvelope")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridEnvelope._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridEnvelope wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_envelope(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_envelope_type, "XybridEnvelope")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_envelope_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_tool_definition_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_tool_definition(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_tool_definition() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_tool_definition_type, args[0], "XybridToolDefinition")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_tool_definition(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    uint8_t *bytes = NULL;
    uintptr_t wire_len = 0;
    boltffi_python_wire_writer writer = {0};
    int ok = 0;
    PyObject *name_value = NULL;
    PyObject *description_value = NULL;
    PyObject *parameters_json_value = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_tool_definition_type, "XybridToolDefinition")) {
        goto done;
    }
    name_value = boltffi_python_get_record_field(value, "XybridToolDefinition", "name");
    if (name_value == NULL) {
        goto done;
    }
    description_value = boltffi_python_get_record_field(value, "XybridToolDefinition", "description");
    if (description_value == NULL) {
        goto done;
    }
    parameters_json_value = boltffi_python_get_record_field(value, "XybridToolDefinition", "parameters_json");
    if (parameters_json_value == NULL) {
        goto done;
    }
    {
        PyObject *field_value = name_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = description_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = parameters_json_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    wire = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)wire_len);
    if (wire == NULL) {
        goto done;
    }
    bytes = (uint8_t *)PyBytes_AS_STRING(wire);
    writer.ptr = bytes;
    writer.len = wire_len;
    writer.offset = 0;
    {
        PyObject *field_value = name_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = description_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = parameters_json_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    if (writer.offset != writer.len) {
        PyErr_SetString(PyExc_RuntimeError, "wire writer produced wrong byte count");
        goto done;
    }
    *out_wire = wire;
    *out_ptr = bytes;
    *out_len = wire_len;
    wire = NULL;
    ok = 1;
done:
    Py_XDECREF(wire);
    Py_XDECREF(name_value);
    Py_XDECREF(description_value);
    Py_XDECREF(parameters_json_value);
    return ok;
}

static PyObject *boltffi_python_decode_owned_xybrid_tool_definition_read(boltffi_python_wire_reader *reader) {
    PyObject *result = NULL;
    PyObject *values[3] = {0};
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[0] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[0] == NULL) {
            goto done;
        }
    }
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[1] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[1] == NULL) {
            goto done;
        }
    }
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[2] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[2] == NULL) {
            goto done;
        }
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_tool_definition_type, "XybridToolDefinition")) {
        goto done;
    }
    result = PyObject_Vectorcall(boltffi_python_xybrid_tool_definition_type, values, 3, NULL);
done:
    Py_XDECREF(values[0]);
    Py_XDECREF(values[1]);
    Py_XDECREF(values[2]);
    return result;
}

static PyObject *boltffi_python_decode_owned_xybrid_tool_definition(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    boltffi_python_wire_reader reader = {0};
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    reader.ptr = buffer.ptr;
    reader.len = buffer.len;
    reader.offset = 0;
    result = boltffi_python_decode_owned_xybrid_tool_definition_read(&reader);
    if (result != NULL && reader.offset != reader.len) {
        Py_CLEAR(result);
        PyErr_SetString(PyExc_ValueError, "trailing BoltFFI wire bytes");
    }
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_tool_call_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_tool_call(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_tool_call() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_tool_call_type, args[0], "XybridToolCall")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_tool_call(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    uint8_t *bytes = NULL;
    uintptr_t wire_len = 0;
    boltffi_python_wire_writer writer = {0};
    int ok = 0;
    PyObject *id_value = NULL;
    PyObject *name_value = NULL;
    PyObject *arguments_json_value = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_tool_call_type, "XybridToolCall")) {
        goto done;
    }
    id_value = boltffi_python_get_record_field(value, "XybridToolCall", "id");
    if (id_value == NULL) {
        goto done;
    }
    name_value = boltffi_python_get_record_field(value, "XybridToolCall", "name");
    if (name_value == NULL) {
        goto done;
    }
    arguments_json_value = boltffi_python_get_record_field(value, "XybridToolCall", "arguments_json");
    if (arguments_json_value == NULL) {
        goto done;
    }
    {
        PyObject *field_value = id_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = name_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = arguments_json_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    wire = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)wire_len);
    if (wire == NULL) {
        goto done;
    }
    bytes = (uint8_t *)PyBytes_AS_STRING(wire);
    writer.ptr = bytes;
    writer.len = wire_len;
    writer.offset = 0;
    {
        PyObject *field_value = id_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = name_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = arguments_json_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    if (writer.offset != writer.len) {
        PyErr_SetString(PyExc_RuntimeError, "wire writer produced wrong byte count");
        goto done;
    }
    *out_wire = wire;
    *out_ptr = bytes;
    *out_len = wire_len;
    wire = NULL;
    ok = 1;
done:
    Py_XDECREF(wire);
    Py_XDECREF(id_value);
    Py_XDECREF(name_value);
    Py_XDECREF(arguments_json_value);
    return ok;
}

static PyObject *boltffi_python_decode_owned_xybrid_tool_call_read(boltffi_python_wire_reader *reader) {
    PyObject *result = NULL;
    PyObject *values[3] = {0};
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[0] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[0] == NULL) {
            goto done;
        }
    }
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[1] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[1] == NULL) {
            goto done;
        }
    }
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[2] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[2] == NULL) {
            goto done;
        }
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_tool_call_type, "XybridToolCall")) {
        goto done;
    }
    result = PyObject_Vectorcall(boltffi_python_xybrid_tool_call_type, values, 3, NULL);
done:
    Py_XDECREF(values[0]);
    Py_XDECREF(values[1]);
    Py_XDECREF(values[2]);
    return result;
}

static PyObject *boltffi_python_decode_owned_xybrid_tool_call(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    boltffi_python_wire_reader reader = {0};
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    reader.ptr = buffer.ptr;
    reader.len = buffer.len;
    reader.offset = 0;
    result = boltffi_python_decode_owned_xybrid_tool_call_read(&reader);
    if (result != NULL && reader.offset != reader.len) {
        Py_CLEAR(result);
        PyErr_SetString(PyExc_ValueError, "trailing BoltFFI wire bytes");
    }
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_tool_result_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_tool_result(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_tool_result() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_tool_result_type, args[0], "XybridToolResult")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_tool_result(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    uint8_t *bytes = NULL;
    uintptr_t wire_len = 0;
    boltffi_python_wire_writer writer = {0};
    int ok = 0;
    PyObject *call_id_value = NULL;
    PyObject *name_value = NULL;
    PyObject *content_json_value = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_tool_result_type, "XybridToolResult")) {
        goto done;
    }
    call_id_value = boltffi_python_get_record_field(value, "XybridToolResult", "call_id");
    if (call_id_value == NULL) {
        goto done;
    }
    name_value = boltffi_python_get_record_field(value, "XybridToolResult", "name");
    if (name_value == NULL) {
        goto done;
    }
    content_json_value = boltffi_python_get_record_field(value, "XybridToolResult", "content_json");
    if (content_json_value == NULL) {
        goto done;
    }
    {
        PyObject *field_value = call_id_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = name_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = content_json_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    wire = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)wire_len);
    if (wire == NULL) {
        goto done;
    }
    bytes = (uint8_t *)PyBytes_AS_STRING(wire);
    writer.ptr = bytes;
    writer.len = wire_len;
    writer.offset = 0;
    {
        PyObject *field_value = call_id_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = name_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = content_json_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    if (writer.offset != writer.len) {
        PyErr_SetString(PyExc_RuntimeError, "wire writer produced wrong byte count");
        goto done;
    }
    *out_wire = wire;
    *out_ptr = bytes;
    *out_len = wire_len;
    wire = NULL;
    ok = 1;
done:
    Py_XDECREF(wire);
    Py_XDECREF(call_id_value);
    Py_XDECREF(name_value);
    Py_XDECREF(content_json_value);
    return ok;
}

static PyObject *boltffi_python_decode_owned_xybrid_tool_result_read(boltffi_python_wire_reader *reader) {
    PyObject *result = NULL;
    PyObject *values[3] = {0};
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[0] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[0] == NULL) {
            goto done;
        }
    }
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[1] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[1] == NULL) {
            goto done;
        }
    }
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[2] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[2] == NULL) {
            goto done;
        }
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_tool_result_type, "XybridToolResult")) {
        goto done;
    }
    result = PyObject_Vectorcall(boltffi_python_xybrid_tool_result_type, values, 3, NULL);
done:
    Py_XDECREF(values[0]);
    Py_XDECREF(values[1]);
    Py_XDECREF(values[2]);
    return result;
}

static PyObject *boltffi_python_decode_owned_xybrid_tool_result(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    boltffi_python_wire_reader reader = {0};
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    reader.ptr = buffer.ptr;
    reader.len = buffer.len;
    reader.offset = 0;
    result = boltffi_python_decode_owned_xybrid_tool_result_read(&reader);
    if (result != NULL && reader.offset != reader.len) {
        Py_CLEAR(result);
        PyErr_SetString(PyExc_ValueError, "trailing BoltFFI wire bytes");
    }
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_generation_config_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_generation_config(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_generation_config() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_generation_config_type, args[0], "XybridGenerationConfig")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_generation_config(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_generation_config_type, "XybridGenerationConfig")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridGenerationConfig._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridGenerationConfig wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_generation_config(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_generation_config_type, "XybridGenerationConfig")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_generation_config_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_run_options_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_run_options(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_run_options() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_run_options_type, args[0], "XybridRunOptions")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_run_options(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_run_options_type, "XybridRunOptions")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridRunOptions._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridRunOptions wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_run_options(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_run_options_type, "XybridRunOptions")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_run_options_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_stage_latency_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_stage_latency(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_stage_latency() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_stage_latency_type, args[0], "XybridStageLatency")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_stage_latency(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    uint8_t *bytes = NULL;
    uintptr_t wire_len = 0;
    boltffi_python_wire_writer writer = {0};
    int ok = 0;
    PyObject *stage_id_value = NULL;
    PyObject *latency_ms_value = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_stage_latency_type, "XybridStageLatency")) {
        goto done;
    }
    stage_id_value = boltffi_python_get_record_field(value, "XybridStageLatency", "stage_id");
    if (stage_id_value == NULL) {
        goto done;
    }
    latency_ms_value = boltffi_python_get_record_field(value, "XybridStageLatency", "latency_ms");
    if (latency_ms_value == NULL) {
        goto done;
    }
    {
        PyObject *field_value = stage_id_value;
        Py_ssize_t utf8_len = 0;
        if (PyUnicode_AsUTF8AndSize(field_value, &utf8_len) == NULL) {
            goto done;
        }
        if (utf8_len > UINT32_MAX) {
            PyErr_SetString(PyExc_OverflowError, "string field is too large");
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, 4 + (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = latency_ms_value;
        if (!boltffi_python_wire_add(&wire_len, 4)) {
            goto done;
        }
    }
    wire = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)wire_len);
    if (wire == NULL) {
        goto done;
    }
    bytes = (uint8_t *)PyBytes_AS_STRING(wire);
    writer.ptr = bytes;
    writer.len = wire_len;
    writer.offset = 0;
    {
        PyObject *field_value = stage_id_value;
        Py_ssize_t utf8_len = 0;
        const char *utf8 = PyUnicode_AsUTF8AndSize(field_value, &utf8_len);
        if (utf8 == NULL) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)utf8_len)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_write(&writer, (const uint8_t *)utf8, (uintptr_t)utf8_len)) {
            goto done;
        }
    }
    {
        PyObject *field_value = latency_ms_value;
        uint32_t parsed = 0;
        if (!boltffi_python_parse_u32(field_value, &parsed)) {
            goto done;
        }
        if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)parsed)) {
            goto done;
        }
    }
    if (writer.offset != writer.len) {
        PyErr_SetString(PyExc_RuntimeError, "wire writer produced wrong byte count");
        goto done;
    }
    *out_wire = wire;
    *out_ptr = bytes;
    *out_len = wire_len;
    wire = NULL;
    ok = 1;
done:
    Py_XDECREF(wire);
    Py_XDECREF(stage_id_value);
    Py_XDECREF(latency_ms_value);
    return ok;
}

static PyObject *boltffi_python_decode_owned_xybrid_stage_latency_read(boltffi_python_wire_reader *reader) {
    PyObject *result = NULL;
    PyObject *values[2] = {0};
    {
        uint32_t len = 0;
        const uint8_t *bytes = NULL;
        if (!boltffi_python_wire_reader_u32(reader, &len)) {
            goto done;
        }
        if (!boltffi_python_wire_reader_read(reader, len, &bytes)) {
            goto done;
        }
        values[0] = PyUnicode_FromStringAndSize((const char *)bytes, (Py_ssize_t)len);
        if (values[0] == NULL) {
            goto done;
        }
    }
    {
        uint32_t decoded = 0;
        uint32_t bytes = 0;
        if (!boltffi_python_wire_reader_u32(reader, &bytes)) {
            goto done;
        }
        decoded = bytes;
        values[1] = boltffi_python_box_u32(decoded);
        if (values[1] == NULL) {
            goto done;
        }
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_stage_latency_type, "XybridStageLatency")) {
        goto done;
    }
    result = PyObject_Vectorcall(boltffi_python_xybrid_stage_latency_type, values, 2, NULL);
done:
    Py_XDECREF(values[0]);
    Py_XDECREF(values[1]);
    return result;
}

static PyObject *boltffi_python_decode_owned_xybrid_stage_latency(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    boltffi_python_wire_reader reader = {0};
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    reader.ptr = buffer.ptr;
    reader.len = buffer.len;
    reader.offset = 0;
    result = boltffi_python_decode_owned_xybrid_stage_latency_read(&reader);
    if (result != NULL && reader.offset != reader.len) {
        Py_CLEAR(result);
        PyErr_SetString(PyExc_ValueError, "trailing BoltFFI wire bytes");
    }
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_inference_metrics_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_inference_metrics(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_inference_metrics() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_inference_metrics_type, args[0], "XybridInferenceMetrics")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_inference_metrics(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_inference_metrics_type, "XybridInferenceMetrics")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridInferenceMetrics._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridInferenceMetrics wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_inference_metrics(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_inference_metrics_type, "XybridInferenceMetrics")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_inference_metrics_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_result_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_result(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_result() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_result_type, args[0], "XybridResult")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_result(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_result_type, "XybridResult")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridResult._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridResult wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_result(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_result_type, "XybridResult")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_result_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_download_status_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_download_status(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_download_status() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_download_status_type, args[0], "XybridDownloadStatus")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_download_status(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_download_status_type, "XybridDownloadStatus")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridDownloadStatus._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridDownloadStatus wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_download_status(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_download_status_type, "XybridDownloadStatus")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_download_status_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_stream_token_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_stream_token(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_stream_token() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_stream_token_type, args[0], "XybridStreamToken")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_stream_token(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_stream_token_type, "XybridStreamToken")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridStreamToken._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridStreamToken wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_stream_token(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_stream_token_type, "XybridStreamToken")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_stream_token_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_stream_event_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_stream_event(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_stream_event() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_stream_event_type, args[0], "XybridStreamEvent")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_stream_event(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_stream_event_type, "XybridStreamEvent")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridStreamEvent._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridStreamEvent wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_stream_event(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_stream_event_type, "XybridStreamEvent")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_stream_event_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_voice_info_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_voice_info(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_voice_info() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_voice_info_type, args[0], "XybridVoiceInfo")) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static int boltffi_python_wire_xybrid_voice_info(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_voice_info_type, "XybridVoiceInfo")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridVoiceInfo._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridVoiceInfo wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_voice_info(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_voice_info_type, "XybridVoiceInfo")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_voice_info_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}



static int boltffi_python_wire_vec_xybrid_tool_result(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    typedef struct {
        PyObject *wire;
        const uint8_t *ptr;
        uintptr_t len;
    } item_wire;
    PyObject *sequence = NULL;
    item_wire *items = NULL;
    PyObject *wire = NULL;
    boltffi_python_wire_writer writer;
    uintptr_t wire_len = 4;
    Py_ssize_t item_count = 0;
    Py_ssize_t index = 0;
    int ok = 0;
    sequence = PySequence_Fast(value, "expected sequence");
    if (sequence == NULL) {
        return 0;
    }
    item_count = PySequence_Fast_GET_SIZE(sequence);
    if (item_count > UINT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "sequence too large to encode");
        goto done;
    }
    if (item_count > 0) {
        items = PyMem_Calloc((size_t)item_count, sizeof(item_wire));
        if (items == NULL) {
            PyErr_NoMemory();
            goto done;
        }
    }
    for (index = 0; index < item_count; index += 1) {
        if (!boltffi_python_wire_xybrid_tool_result(PySequence_Fast_GET_ITEM(sequence, index), &items[index].wire, &items[index].ptr, &items[index].len)) {
            goto done;
        }
        if (!boltffi_python_wire_add(&wire_len, items[index].len)) {
            goto done;
        }
    }
    wire = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)wire_len);
    if (wire == NULL) {
        goto done;
    }
    writer.ptr = (uint8_t *)PyBytes_AS_STRING(wire);
    writer.len = wire_len;
    writer.offset = 0;
    if (!boltffi_python_wire_writer_u32(&writer, (uint32_t)item_count)) {
        goto done;
    }
    for (index = 0; index < item_count; index += 1) {
        if (!boltffi_python_wire_writer_write(&writer, items[index].ptr, items[index].len)) {
            goto done;
        }
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = wire_len;
    wire = NULL;
    ok = 1;
done:
    Py_XDECREF(wire);
    if (items != NULL) {
        for (index = 0; index < item_count; index += 1) {
            Py_XDECREF(items[index].wire);
        }
        PyMem_Free(items);
    }
    Py_DECREF(sequence);
    return ok;
}

static PyObject *boltffi_python_decode_owned_vec_xybrid_tool_result(FfiBuf_u8 buffer) {
    boltffi_python_wire_reader reader;
    PyObject *result = NULL;
    PyObject *item = NULL;
    uint32_t item_count = 0;
    Py_ssize_t count = 0;
    Py_ssize_t index = 0;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    reader.ptr = buffer.ptr;
    reader.len = buffer.len;
    reader.offset = 0;
    if (!boltffi_python_wire_reader_u32(&reader, &item_count)) {
        goto done;
    }
    if (item_count > (uint32_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "native sequence is too large");
        goto done;
    }
    count = (Py_ssize_t)item_count;
    result = PyList_New(count);
    if (result == NULL) {
        goto done;
    }
    for (index = 0; index < count; index += 1) {
        item = boltffi_python_decode_owned_xybrid_tool_result_read(&reader);
        if (item == NULL) {
            Py_CLEAR(result);
            goto done;
        }
        PyList_SET_ITEM(result, index, item);
        item = NULL;
    }
    if (reader.offset != reader.len) {
        PyErr_SetString(PyExc_ValueError, "trailing BoltFFI wire bytes");
        Py_CLEAR(result);
    }
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}


static PyObject *boltffi_python_xybrid_error_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_error(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_error() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_error_type, args[0], "XybridError")) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static int boltffi_python_wire_xybrid_error(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_error_type, "XybridError")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridError._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridError wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_error(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_error_type, "XybridError")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_error_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}

static PyObject *boltffi_python_xybrid_envelope_kind_type = NULL;

static PyObject *boltffi_python_wrapper_register_xybrid_envelope_kind(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_envelope_kind() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_registered_type(&boltffi_python_xybrid_envelope_kind_type, args[0], "XybridEnvelopeKind")) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static int boltffi_python_wire_xybrid_envelope_kind(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    PyObject *wire = NULL;
    if (!boltffi_python_expect_type_instance(value, boltffi_python_xybrid_envelope_kind_type, "XybridEnvelopeKind")) {
        return 0;
    }
    wire = PyObject_CallMethod(value, "_boltffi_wire", NULL);
    if (wire == NULL) {
        return 0;
    }
    if (!PyBytes_Check(wire)) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_TypeError, "XybridEnvelopeKind._boltffi_wire() must return bytes");
        return 0;
    }
    if (PyBytes_GET_SIZE(wire) > PY_SSIZE_T_MAX) {
        Py_DECREF(wire);
        PyErr_SetString(PyExc_OverflowError, "XybridEnvelopeKind wire payload is too large");
        return 0;
    }
    *out_wire = wire;
    *out_ptr = (const uint8_t *)PyBytes_AS_STRING(wire);
    *out_len = (uintptr_t)PyBytes_GET_SIZE(wire);
    return 1;
}

static PyObject *boltffi_python_decode_owned_xybrid_envelope_kind(FfiBuf_u8 buffer) {
    PyObject *wire = NULL;
    PyObject *result = NULL;
    if (!boltffi_python_validate_owned_memory(buffer)) {
        goto done;
    }
    wire = PyBytes_FromStringAndSize((const char *)buffer.ptr, (Py_ssize_t)buffer.len);
    if (wire == NULL) {
        goto done;
    }
    if (!boltffi_python_expect_registered_type(boltffi_python_xybrid_envelope_kind_type, "XybridEnvelopeKind")) {
        goto done;
    }
    result = PyObject_CallMethod(boltffi_python_xybrid_envelope_kind_type, "_boltffi_from_wire", "O", wire);
done:
    Py_XDECREF(wire);
    boltffi_python_release_owned_buffer(buffer);
    return result;
}

static PyObject *boltffi_python_xybrid_message_role_members_by_wire_tag[3] = {NULL};
static const char *boltffi_python_xybrid_message_role_member_names[3] = {
    "SYSTEM",
    "USER",
    "ASSISTANT"
};
static const ___XybridMessageRole boltffi_python_xybrid_message_role_member_native_values[3] = {
    0,
    1,
    2
};
static boltffi_python_c_style_enum_registration boltffi_python_xybrid_message_role_registration = {
    NULL,
    3,
    boltffi_python_xybrid_message_role_members_by_wire_tag,
};

static PyObject *boltffi_python_load_xybrid_message_role_member(PyObject *type_object, Py_ssize_t member_index) {
    PyObject *native_value = NULL;
    PyObject *member = NULL;
    native_value = boltffi_python_box_i32(boltffi_python_xybrid_message_role_member_native_values[member_index]);
    member = boltffi_python_load_c_style_enum_member(
        type_object,
        "XybridMessageRole",
        boltffi_python_xybrid_message_role_member_names[member_index],
        native_value
    );
    Py_XDECREF(native_value);
    return member;
}

static PyObject *boltffi_python_wrapper_register_xybrid_message_role(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_message_role() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_c_style_enum_registration(
        &boltffi_python_xybrid_message_role_registration,
        args[0],
        "XybridMessageRole",
        boltffi_python_load_xybrid_message_role_member
    )) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static int boltffi_python_parse_xybrid_message_role(PyObject *value, ___XybridMessageRole *out) {
    if (!boltffi_python_expect_enum_instance(value, &boltffi_python_xybrid_message_role_registration, "XybridMessageRole")) {
        return 0;
    }
    return boltffi_python_parse_i32(value, out);
}

static int boltffi_python_xybrid_message_role_native_to_wire_tag(___XybridMessageRole value, int32_t *out) {
    switch (value) {
        case 0:
            *out = 0;
            return 1;
        case 1:
            *out = 1;
            return 1;
        case 2:
            *out = 2;
            return 1;
        default:
            PyErr_SetString(PyExc_ValueError, "invalid XybridMessageRole value");
            return 0;
    }
}

static int boltffi_python_wire_xybrid_message_role(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    ___XybridMessageRole native_value = 0;
    uint8_t bytes[4] = {0};
    if (!boltffi_python_parse_xybrid_message_role(value, &native_value)) {
        return 0;
    }
    boltffi_python_write_u32_le(bytes, (uint32_t)native_value);
    return boltffi_python_wire_fixed(bytes, 4, out_wire, out_ptr, out_len);
}

static PyObject *boltffi_python_box_xybrid_message_role_from_wire_tag(int32_t wire_tag) {
    switch (wire_tag) {
        case 0:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_message_role_registration,
                0,
                "XybridMessageRole"
            );
        case 1:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_message_role_registration,
                1,
                "XybridMessageRole"
            );
        case 2:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_message_role_registration,
                2,
                "XybridMessageRole"
            );
        default:
            PyErr_SetString(PyExc_RuntimeError, "native enum wire tag is invalid");
            return NULL;
    }
}

static PyObject *boltffi_python_box_xybrid_message_role(___XybridMessageRole value) {
    int32_t wire_tag = 0;
    if (!boltffi_python_xybrid_message_role_native_to_wire_tag(value, &wire_tag)) {
        return NULL;
    }
    return boltffi_python_box_xybrid_message_role_from_wire_tag(wire_tag);
}

static PyObject *boltffi_python_decode_owned_xybrid_message_role(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    ___XybridMessageRole native_value = 0;
    if (!boltffi_python_validate_owned_fixed_buffer(buffer, 4)) {
        goto done;
    }
    native_value = (___XybridMessageRole)boltffi_python_read_u32_le(buffer.ptr);
    result = boltffi_python_box_xybrid_message_role(native_value);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}

static PyObject *boltffi_python_xybrid_abort_signal_members_by_wire_tag[4] = {NULL};
static const char *boltffi_python_xybrid_abort_signal_member_names[4] = {
    "MEMORY_PRESSURE_WARN",
    "MEMORY_PRESSURE_CRITICAL",
    "THERMAL_HOT",
    "THERMAL_CRITICAL"
};
static const ___XybridAbortSignal boltffi_python_xybrid_abort_signal_member_native_values[4] = {
    0,
    1,
    2,
    3
};
static boltffi_python_c_style_enum_registration boltffi_python_xybrid_abort_signal_registration = {
    NULL,
    4,
    boltffi_python_xybrid_abort_signal_members_by_wire_tag,
};

static PyObject *boltffi_python_load_xybrid_abort_signal_member(PyObject *type_object, Py_ssize_t member_index) {
    PyObject *native_value = NULL;
    PyObject *member = NULL;
    native_value = boltffi_python_box_i32(boltffi_python_xybrid_abort_signal_member_native_values[member_index]);
    member = boltffi_python_load_c_style_enum_member(
        type_object,
        "XybridAbortSignal",
        boltffi_python_xybrid_abort_signal_member_names[member_index],
        native_value
    );
    Py_XDECREF(native_value);
    return member;
}

static PyObject *boltffi_python_wrapper_register_xybrid_abort_signal(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_abort_signal() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_c_style_enum_registration(
        &boltffi_python_xybrid_abort_signal_registration,
        args[0],
        "XybridAbortSignal",
        boltffi_python_load_xybrid_abort_signal_member
    )) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static int boltffi_python_parse_xybrid_abort_signal(PyObject *value, ___XybridAbortSignal *out) {
    if (!boltffi_python_expect_enum_instance(value, &boltffi_python_xybrid_abort_signal_registration, "XybridAbortSignal")) {
        return 0;
    }
    return boltffi_python_parse_i32(value, out);
}

static int boltffi_python_xybrid_abort_signal_native_to_wire_tag(___XybridAbortSignal value, int32_t *out) {
    switch (value) {
        case 0:
            *out = 0;
            return 1;
        case 1:
            *out = 1;
            return 1;
        case 2:
            *out = 2;
            return 1;
        case 3:
            *out = 3;
            return 1;
        default:
            PyErr_SetString(PyExc_ValueError, "invalid XybridAbortSignal value");
            return 0;
    }
}

static int boltffi_python_wire_xybrid_abort_signal(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    ___XybridAbortSignal native_value = 0;
    uint8_t bytes[4] = {0};
    if (!boltffi_python_parse_xybrid_abort_signal(value, &native_value)) {
        return 0;
    }
    boltffi_python_write_u32_le(bytes, (uint32_t)native_value);
    return boltffi_python_wire_fixed(bytes, 4, out_wire, out_ptr, out_len);
}

static PyObject *boltffi_python_box_xybrid_abort_signal_from_wire_tag(int32_t wire_tag) {
    switch (wire_tag) {
        case 0:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_abort_signal_registration,
                0,
                "XybridAbortSignal"
            );
        case 1:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_abort_signal_registration,
                1,
                "XybridAbortSignal"
            );
        case 2:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_abort_signal_registration,
                2,
                "XybridAbortSignal"
            );
        case 3:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_abort_signal_registration,
                3,
                "XybridAbortSignal"
            );
        default:
            PyErr_SetString(PyExc_RuntimeError, "native enum wire tag is invalid");
            return NULL;
    }
}

static PyObject *boltffi_python_box_xybrid_abort_signal(___XybridAbortSignal value) {
    int32_t wire_tag = 0;
    if (!boltffi_python_xybrid_abort_signal_native_to_wire_tag(value, &wire_tag)) {
        return NULL;
    }
    return boltffi_python_box_xybrid_abort_signal_from_wire_tag(wire_tag);
}

static PyObject *boltffi_python_decode_owned_xybrid_abort_signal(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    ___XybridAbortSignal native_value = 0;
    if (!boltffi_python_validate_owned_fixed_buffer(buffer, 4)) {
        goto done;
    }
    native_value = (___XybridAbortSignal)boltffi_python_read_u32_le(buffer.ptr);
    result = boltffi_python_box_xybrid_abort_signal(native_value);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}

static PyObject *boltffi_python_xybrid_output_type_members_by_wire_tag[4] = {NULL};
static const char *boltffi_python_xybrid_output_type_member_names[4] = {
    "TEXT",
    "AUDIO",
    "EMBEDDING",
    "UNKNOWN"
};
static const ___XybridOutputType boltffi_python_xybrid_output_type_member_native_values[4] = {
    0,
    1,
    2,
    3
};
static boltffi_python_c_style_enum_registration boltffi_python_xybrid_output_type_registration = {
    NULL,
    4,
    boltffi_python_xybrid_output_type_members_by_wire_tag,
};

static PyObject *boltffi_python_load_xybrid_output_type_member(PyObject *type_object, Py_ssize_t member_index) {
    PyObject *native_value = NULL;
    PyObject *member = NULL;
    native_value = boltffi_python_box_i32(boltffi_python_xybrid_output_type_member_native_values[member_index]);
    member = boltffi_python_load_c_style_enum_member(
        type_object,
        "XybridOutputType",
        boltffi_python_xybrid_output_type_member_names[member_index],
        native_value
    );
    Py_XDECREF(native_value);
    return member;
}

static PyObject *boltffi_python_wrapper_register_xybrid_output_type(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_output_type() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_c_style_enum_registration(
        &boltffi_python_xybrid_output_type_registration,
        args[0],
        "XybridOutputType",
        boltffi_python_load_xybrid_output_type_member
    )) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static int boltffi_python_parse_xybrid_output_type(PyObject *value, ___XybridOutputType *out) {
    if (!boltffi_python_expect_enum_instance(value, &boltffi_python_xybrid_output_type_registration, "XybridOutputType")) {
        return 0;
    }
    return boltffi_python_parse_i32(value, out);
}

static int boltffi_python_xybrid_output_type_native_to_wire_tag(___XybridOutputType value, int32_t *out) {
    switch (value) {
        case 0:
            *out = 0;
            return 1;
        case 1:
            *out = 1;
            return 1;
        case 2:
            *out = 2;
            return 1;
        case 3:
            *out = 3;
            return 1;
        default:
            PyErr_SetString(PyExc_ValueError, "invalid XybridOutputType value");
            return 0;
    }
}

static int boltffi_python_wire_xybrid_output_type(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    ___XybridOutputType native_value = 0;
    uint8_t bytes[4] = {0};
    if (!boltffi_python_parse_xybrid_output_type(value, &native_value)) {
        return 0;
    }
    boltffi_python_write_u32_le(bytes, (uint32_t)native_value);
    return boltffi_python_wire_fixed(bytes, 4, out_wire, out_ptr, out_len);
}

static PyObject *boltffi_python_box_xybrid_output_type_from_wire_tag(int32_t wire_tag) {
    switch (wire_tag) {
        case 0:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_output_type_registration,
                0,
                "XybridOutputType"
            );
        case 1:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_output_type_registration,
                1,
                "XybridOutputType"
            );
        case 2:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_output_type_registration,
                2,
                "XybridOutputType"
            );
        case 3:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_output_type_registration,
                3,
                "XybridOutputType"
            );
        default:
            PyErr_SetString(PyExc_RuntimeError, "native enum wire tag is invalid");
            return NULL;
    }
}

static PyObject *boltffi_python_box_xybrid_output_type(___XybridOutputType value) {
    int32_t wire_tag = 0;
    if (!boltffi_python_xybrid_output_type_native_to_wire_tag(value, &wire_tag)) {
        return NULL;
    }
    return boltffi_python_box_xybrid_output_type_from_wire_tag(wire_tag);
}

static PyObject *boltffi_python_decode_owned_xybrid_output_type(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    ___XybridOutputType native_value = 0;
    if (!boltffi_python_validate_owned_fixed_buffer(buffer, 4)) {
        goto done;
    }
    native_value = (___XybridOutputType)boltffi_python_read_u32_le(buffer.ptr);
    result = boltffi_python_box_xybrid_output_type(native_value);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}

static PyObject *boltffi_python_xybrid_execution_target_members_by_wire_tag[2] = {NULL};
static const char *boltffi_python_xybrid_execution_target_member_names[2] = {
    "LOCAL",
    "CLOUD"
};
static const ___XybridExecutionTarget boltffi_python_xybrid_execution_target_member_native_values[2] = {
    0,
    1
};
static boltffi_python_c_style_enum_registration boltffi_python_xybrid_execution_target_registration = {
    NULL,
    2,
    boltffi_python_xybrid_execution_target_members_by_wire_tag,
};

static PyObject *boltffi_python_load_xybrid_execution_target_member(PyObject *type_object, Py_ssize_t member_index) {
    PyObject *native_value = NULL;
    PyObject *member = NULL;
    native_value = boltffi_python_box_i32(boltffi_python_xybrid_execution_target_member_native_values[member_index]);
    member = boltffi_python_load_c_style_enum_member(
        type_object,
        "XybridExecutionTarget",
        boltffi_python_xybrid_execution_target_member_names[member_index],
        native_value
    );
    Py_XDECREF(native_value);
    return member;
}

static PyObject *boltffi_python_wrapper_register_xybrid_execution_target(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_execution_target() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_c_style_enum_registration(
        &boltffi_python_xybrid_execution_target_registration,
        args[0],
        "XybridExecutionTarget",
        boltffi_python_load_xybrid_execution_target_member
    )) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static int boltffi_python_parse_xybrid_execution_target(PyObject *value, ___XybridExecutionTarget *out) {
    if (!boltffi_python_expect_enum_instance(value, &boltffi_python_xybrid_execution_target_registration, "XybridExecutionTarget")) {
        return 0;
    }
    return boltffi_python_parse_i32(value, out);
}

static int boltffi_python_xybrid_execution_target_native_to_wire_tag(___XybridExecutionTarget value, int32_t *out) {
    switch (value) {
        case 0:
            *out = 0;
            return 1;
        case 1:
            *out = 1;
            return 1;
        default:
            PyErr_SetString(PyExc_ValueError, "invalid XybridExecutionTarget value");
            return 0;
    }
}

static int boltffi_python_wire_xybrid_execution_target(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    ___XybridExecutionTarget native_value = 0;
    uint8_t bytes[4] = {0};
    if (!boltffi_python_parse_xybrid_execution_target(value, &native_value)) {
        return 0;
    }
    boltffi_python_write_u32_le(bytes, (uint32_t)native_value);
    return boltffi_python_wire_fixed(bytes, 4, out_wire, out_ptr, out_len);
}

static PyObject *boltffi_python_box_xybrid_execution_target_from_wire_tag(int32_t wire_tag) {
    switch (wire_tag) {
        case 0:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_execution_target_registration,
                0,
                "XybridExecutionTarget"
            );
        case 1:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_execution_target_registration,
                1,
                "XybridExecutionTarget"
            );
        default:
            PyErr_SetString(PyExc_RuntimeError, "native enum wire tag is invalid");
            return NULL;
    }
}

static PyObject *boltffi_python_box_xybrid_execution_target(___XybridExecutionTarget value) {
    int32_t wire_tag = 0;
    if (!boltffi_python_xybrid_execution_target_native_to_wire_tag(value, &wire_tag)) {
        return NULL;
    }
    return boltffi_python_box_xybrid_execution_target_from_wire_tag(wire_tag);
}

static PyObject *boltffi_python_decode_owned_xybrid_execution_target(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    ___XybridExecutionTarget native_value = 0;
    if (!boltffi_python_validate_owned_fixed_buffer(buffer, 4)) {
        goto done;
    }
    native_value = (___XybridExecutionTarget)boltffi_python_read_u32_le(buffer.ptr);
    result = boltffi_python_box_xybrid_execution_target(native_value);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}

static PyObject *boltffi_python_xybrid_download_state_members_by_wire_tag[3] = {NULL};
static const char *boltffi_python_xybrid_download_state_member_names[3] = {
    "DOWNLOADING",
    "READY",
    "FAILED"
};
static const ___XybridDownloadState boltffi_python_xybrid_download_state_member_native_values[3] = {
    0,
    1,
    2
};
static boltffi_python_c_style_enum_registration boltffi_python_xybrid_download_state_registration = {
    NULL,
    3,
    boltffi_python_xybrid_download_state_members_by_wire_tag,
};

static PyObject *boltffi_python_load_xybrid_download_state_member(PyObject *type_object, Py_ssize_t member_index) {
    PyObject *native_value = NULL;
    PyObject *member = NULL;
    native_value = boltffi_python_box_i32(boltffi_python_xybrid_download_state_member_native_values[member_index]);
    member = boltffi_python_load_c_style_enum_member(
        type_object,
        "XybridDownloadState",
        boltffi_python_xybrid_download_state_member_names[member_index],
        native_value
    );
    Py_XDECREF(native_value);
    return member;
}

static PyObject *boltffi_python_wrapper_register_xybrid_download_state(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_download_state() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_c_style_enum_registration(
        &boltffi_python_xybrid_download_state_registration,
        args[0],
        "XybridDownloadState",
        boltffi_python_load_xybrid_download_state_member
    )) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static int boltffi_python_parse_xybrid_download_state(PyObject *value, ___XybridDownloadState *out) {
    if (!boltffi_python_expect_enum_instance(value, &boltffi_python_xybrid_download_state_registration, "XybridDownloadState")) {
        return 0;
    }
    return boltffi_python_parse_i32(value, out);
}

static int boltffi_python_xybrid_download_state_native_to_wire_tag(___XybridDownloadState value, int32_t *out) {
    switch (value) {
        case 0:
            *out = 0;
            return 1;
        case 1:
            *out = 1;
            return 1;
        case 2:
            *out = 2;
            return 1;
        default:
            PyErr_SetString(PyExc_ValueError, "invalid XybridDownloadState value");
            return 0;
    }
}

static int boltffi_python_wire_xybrid_download_state(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    ___XybridDownloadState native_value = 0;
    uint8_t bytes[4] = {0};
    if (!boltffi_python_parse_xybrid_download_state(value, &native_value)) {
        return 0;
    }
    boltffi_python_write_u32_le(bytes, (uint32_t)native_value);
    return boltffi_python_wire_fixed(bytes, 4, out_wire, out_ptr, out_len);
}

static PyObject *boltffi_python_box_xybrid_download_state_from_wire_tag(int32_t wire_tag) {
    switch (wire_tag) {
        case 0:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_download_state_registration,
                0,
                "XybridDownloadState"
            );
        case 1:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_download_state_registration,
                1,
                "XybridDownloadState"
            );
        case 2:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_download_state_registration,
                2,
                "XybridDownloadState"
            );
        default:
            PyErr_SetString(PyExc_RuntimeError, "native enum wire tag is invalid");
            return NULL;
    }
}

static PyObject *boltffi_python_box_xybrid_download_state(___XybridDownloadState value) {
    int32_t wire_tag = 0;
    if (!boltffi_python_xybrid_download_state_native_to_wire_tag(value, &wire_tag)) {
        return NULL;
    }
    return boltffi_python_box_xybrid_download_state_from_wire_tag(wire_tag);
}

static PyObject *boltffi_python_decode_owned_xybrid_download_state(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    ___XybridDownloadState native_value = 0;
    if (!boltffi_python_validate_owned_fixed_buffer(buffer, 4)) {
        goto done;
    }
    native_value = (___XybridDownloadState)boltffi_python_read_u32_le(buffer.ptr);
    result = boltffi_python_box_xybrid_download_state(native_value);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}

static PyObject *boltffi_python_xybrid_stream_event_kind_members_by_wire_tag[2] = {NULL};
static const char *boltffi_python_xybrid_stream_event_kind_member_names[2] = {
    "TOKEN",
    "COMPLETE"
};
static const ___XybridStreamEventKind boltffi_python_xybrid_stream_event_kind_member_native_values[2] = {
    0,
    1
};
static boltffi_python_c_style_enum_registration boltffi_python_xybrid_stream_event_kind_registration = {
    NULL,
    2,
    boltffi_python_xybrid_stream_event_kind_members_by_wire_tag,
};

static PyObject *boltffi_python_load_xybrid_stream_event_kind_member(PyObject *type_object, Py_ssize_t member_index) {
    PyObject *native_value = NULL;
    PyObject *member = NULL;
    native_value = boltffi_python_box_i32(boltffi_python_xybrid_stream_event_kind_member_native_values[member_index]);
    member = boltffi_python_load_c_style_enum_member(
        type_object,
        "XybridStreamEventKind",
        boltffi_python_xybrid_stream_event_kind_member_names[member_index],
        native_value
    );
    Py_XDECREF(native_value);
    return member;
}

static PyObject *boltffi_python_wrapper_register_xybrid_stream_event_kind(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_stream_event_kind() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_c_style_enum_registration(
        &boltffi_python_xybrid_stream_event_kind_registration,
        args[0],
        "XybridStreamEventKind",
        boltffi_python_load_xybrid_stream_event_kind_member
    )) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static int boltffi_python_parse_xybrid_stream_event_kind(PyObject *value, ___XybridStreamEventKind *out) {
    if (!boltffi_python_expect_enum_instance(value, &boltffi_python_xybrid_stream_event_kind_registration, "XybridStreamEventKind")) {
        return 0;
    }
    return boltffi_python_parse_i32(value, out);
}

static int boltffi_python_xybrid_stream_event_kind_native_to_wire_tag(___XybridStreamEventKind value, int32_t *out) {
    switch (value) {
        case 0:
            *out = 0;
            return 1;
        case 1:
            *out = 1;
            return 1;
        default:
            PyErr_SetString(PyExc_ValueError, "invalid XybridStreamEventKind value");
            return 0;
    }
}

static int boltffi_python_wire_xybrid_stream_event_kind(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    ___XybridStreamEventKind native_value = 0;
    uint8_t bytes[4] = {0};
    if (!boltffi_python_parse_xybrid_stream_event_kind(value, &native_value)) {
        return 0;
    }
    boltffi_python_write_u32_le(bytes, (uint32_t)native_value);
    return boltffi_python_wire_fixed(bytes, 4, out_wire, out_ptr, out_len);
}

static PyObject *boltffi_python_box_xybrid_stream_event_kind_from_wire_tag(int32_t wire_tag) {
    switch (wire_tag) {
        case 0:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_stream_event_kind_registration,
                0,
                "XybridStreamEventKind"
            );
        case 1:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_stream_event_kind_registration,
                1,
                "XybridStreamEventKind"
            );
        default:
            PyErr_SetString(PyExc_RuntimeError, "native enum wire tag is invalid");
            return NULL;
    }
}

static PyObject *boltffi_python_box_xybrid_stream_event_kind(___XybridStreamEventKind value) {
    int32_t wire_tag = 0;
    if (!boltffi_python_xybrid_stream_event_kind_native_to_wire_tag(value, &wire_tag)) {
        return NULL;
    }
    return boltffi_python_box_xybrid_stream_event_kind_from_wire_tag(wire_tag);
}

static PyObject *boltffi_python_decode_owned_xybrid_stream_event_kind(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    ___XybridStreamEventKind native_value = 0;
    if (!boltffi_python_validate_owned_fixed_buffer(buffer, 4)) {
        goto done;
    }
    native_value = (___XybridStreamEventKind)boltffi_python_read_u32_le(buffer.ptr);
    result = boltffi_python_box_xybrid_stream_event_kind(native_value);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}

static PyObject *boltffi_python_xybrid_thermal_state_members_by_wire_tag[4] = {NULL};
static const char *boltffi_python_xybrid_thermal_state_member_names[4] = {
    "NORMAL",
    "WARM",
    "HOT",
    "CRITICAL"
};
static const ___XybridThermalState boltffi_python_xybrid_thermal_state_member_native_values[4] = {
    0,
    1,
    2,
    3
};
static boltffi_python_c_style_enum_registration boltffi_python_xybrid_thermal_state_registration = {
    NULL,
    4,
    boltffi_python_xybrid_thermal_state_members_by_wire_tag,
};

static PyObject *boltffi_python_load_xybrid_thermal_state_member(PyObject *type_object, Py_ssize_t member_index) {
    PyObject *native_value = NULL;
    PyObject *member = NULL;
    native_value = boltffi_python_box_i32(boltffi_python_xybrid_thermal_state_member_native_values[member_index]);
    member = boltffi_python_load_c_style_enum_member(
        type_object,
        "XybridThermalState",
        boltffi_python_xybrid_thermal_state_member_names[member_index],
        native_value
    );
    Py_XDECREF(native_value);
    return member;
}

static PyObject *boltffi_python_wrapper_register_xybrid_thermal_state(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_register_xybrid_thermal_state() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (!boltffi_python_store_c_style_enum_registration(
        &boltffi_python_xybrid_thermal_state_registration,
        args[0],
        "XybridThermalState",
        boltffi_python_load_xybrid_thermal_state_member
    )) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static int boltffi_python_parse_xybrid_thermal_state(PyObject *value, ___XybridThermalState *out) {
    if (!boltffi_python_expect_enum_instance(value, &boltffi_python_xybrid_thermal_state_registration, "XybridThermalState")) {
        return 0;
    }
    return boltffi_python_parse_i32(value, out);
}

static int boltffi_python_xybrid_thermal_state_native_to_wire_tag(___XybridThermalState value, int32_t *out) {
    switch (value) {
        case 0:
            *out = 0;
            return 1;
        case 1:
            *out = 1;
            return 1;
        case 2:
            *out = 2;
            return 1;
        case 3:
            *out = 3;
            return 1;
        default:
            PyErr_SetString(PyExc_ValueError, "invalid XybridThermalState value");
            return 0;
    }
}

static int boltffi_python_wire_xybrid_thermal_state(PyObject *value, PyObject **out_wire, const uint8_t **out_ptr, uintptr_t *out_len) {
    ___XybridThermalState native_value = 0;
    uint8_t bytes[4] = {0};
    if (!boltffi_python_parse_xybrid_thermal_state(value, &native_value)) {
        return 0;
    }
    boltffi_python_write_u32_le(bytes, (uint32_t)native_value);
    return boltffi_python_wire_fixed(bytes, 4, out_wire, out_ptr, out_len);
}

static PyObject *boltffi_python_box_xybrid_thermal_state_from_wire_tag(int32_t wire_tag) {
    switch (wire_tag) {
        case 0:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_thermal_state_registration,
                0,
                "XybridThermalState"
            );
        case 1:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_thermal_state_registration,
                1,
                "XybridThermalState"
            );
        case 2:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_thermal_state_registration,
                2,
                "XybridThermalState"
            );
        case 3:
            return boltffi_python_box_registered_enum_member(
                &boltffi_python_xybrid_thermal_state_registration,
                3,
                "XybridThermalState"
            );
        default:
            PyErr_SetString(PyExc_RuntimeError, "native enum wire tag is invalid");
            return NULL;
    }
}

static PyObject *boltffi_python_box_xybrid_thermal_state(___XybridThermalState value) {
    int32_t wire_tag = 0;
    if (!boltffi_python_xybrid_thermal_state_native_to_wire_tag(value, &wire_tag)) {
        return NULL;
    }
    return boltffi_python_box_xybrid_thermal_state_from_wire_tag(wire_tag);
}

static PyObject *boltffi_python_decode_owned_xybrid_thermal_state(FfiBuf_u8 buffer) {
    PyObject *result = NULL;
    ___XybridThermalState native_value = 0;
    if (!boltffi_python_validate_owned_fixed_buffer(buffer, 4)) {
        goto done;
    }
    native_value = (___XybridThermalState)boltffi_python_read_u32_le(buffer.ptr);
    result = boltffi_python_box_xybrid_thermal_state(native_value);
done:
    boltffi_python_release_owned_buffer(buffer);
    return result;
}



static PyObject *boltffi_python_callable_wrapper_boltffi_release_class_xybrid_bolt_xybrid_model(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t handle;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_release() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        return NULL;
    }
    if (!boltffi_python_parse_u64(args[0], &handle)) {
        return NULL;
    }
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_model(handle);
    Py_RETURN_NONE;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *id_wire = NULL;
    const uint8_t *id_ptr = NULL;
    uintptr_t id_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_from_registry() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &id_wire, &id_ptr, &id_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry(id_ptr, id_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(id_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *id_wire = NULL;
    const uint8_t *id_ptr = NULL;
    uintptr_t id_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_from_registry_speculative() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &id_wire, &id_ptr, &id_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative(id_ptr, id_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(id_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *path_wire = NULL;
    const uint8_t *path_ptr = NULL;
    uintptr_t path_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_from_directory() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &path_wire, &path_ptr, &path_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory(path_ptr, path_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(path_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *path_wire = NULL;
    const uint8_t *path_ptr = NULL;
    uintptr_t path_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_from_bundle() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &path_wire, &path_ptr, &path_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle(path_ptr, path_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(path_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *repo_wire = NULL;
    const uint8_t *repo_ptr = NULL;
    uintptr_t repo_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_from_huggingface() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &repo_wire, &repo_ptr, &repo_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface(repo_ptr, repo_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(repo_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *repo_wire = NULL;
    const uint8_t *repo_ptr = NULL;
    uintptr_t repo_len = 0;
    PyObject *revision_wire = NULL;
    const uint8_t *revision_ptr = NULL;
    uintptr_t revision_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_from_huggingface_with_revision() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &repo_wire, &repo_ptr, &repo_len)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[1], &revision_wire, &revision_ptr, &revision_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision(repo_ptr, repo_len, revision_ptr, revision_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(repo_wire);
    Py_XDECREF(revision_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *path_wire = NULL;
    const uint8_t *path_ptr = NULL;
    uintptr_t path_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_from_model_file() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &path_wire, &path_ptr, &path_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file(path_ptr, path_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(path_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_model_id(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_model_id() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_model_id(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_version(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_version() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_version(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_output_type(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_output_type() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_xybrid_output_type(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_output_type(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_is_loaded() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_is_cloud_serving() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_download_status(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_download_status() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_download_status(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_await_download(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    uint64_t timeout_ms;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_await_download() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_parse_u64(args[1], &timeout_ms)) {
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_await_download(receiver, timeout_ms));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_supports_streaming() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_supports_token_streaming() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_default_generation_config() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_is_llm() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_supports_tool_calling() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_optional_bool(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_has_voices() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_voices(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_voices() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voices(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_default_voice() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_voice(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *voice_id_wire = NULL;
    const uint8_t *voice_id_ptr = NULL;
    uintptr_t voice_id_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_voice() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[1], &voice_id_wire, &voice_id_ptr, &voice_id_len)) {
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_voice(receiver, voice_id_ptr, voice_id_len));
done:
    Py_XDECREF(voice_id_wire);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_run(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *envelope_wire = NULL;
    const uint8_t *envelope_ptr = NULL;
    uintptr_t envelope_len = 0;
    PyObject *options_wire = NULL;
    const uint8_t *options_ptr = NULL;
    uintptr_t options_len = 0;
    FfiBuf_u8 return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 3) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_run() takes 3 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[1], &envelope_wire, &envelope_ptr, &envelope_len)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[2], &options_wire, &options_ptr, &options_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run(receiver, envelope_ptr, envelope_len, options_ptr, options_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(return_success);
done:
    Py_XDECREF(envelope_wire);
    Py_XDECREF(options_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *envelope_wire = NULL;
    const uint8_t *envelope_ptr = NULL;
    uintptr_t envelope_len = 0;
    PyObject *options_wire = NULL;
    const uint8_t *options_ptr = NULL;
    uintptr_t options_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 3) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_run_stream() takes 3 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[1], &envelope_wire, &envelope_ptr, &envelope_len)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[2], &options_wire, &options_ptr, &options_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream(receiver, envelope_ptr, envelope_len, options_ptr, options_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(envelope_wire);
    Py_XDECREF(options_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    uint64_t stream_id;
    FfiBuf_u8 return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_stream_next() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_parse_u64(args[1], &stream_id)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next(receiver, stream_id, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(return_success);
done:
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    uint64_t stream_id;
    FfiBuf_u8 return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_stream_result() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_parse_u64(args[1], &stream_id)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result(receiver, stream_id, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(return_success);
done:
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    uint64_t stream_id;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_stream_close() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_parse_u64(args[1], &stream_id)) {
        goto done;
    }
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close(receiver, stream_id);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *envelope_wire = NULL;
    const uint8_t *envelope_ptr = NULL;
    uintptr_t envelope_len = 0;
    uint64_t context;
    PyObject *options_wire = NULL;
    const uint8_t *options_ptr = NULL;
    uintptr_t options_len = 0;
    FfiBuf_u8 return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 4) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_run_with_context() takes 4 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[1], &envelope_wire, &envelope_ptr, &envelope_len)) {
        goto done;
    }
    if (!boltffi_python_parse_u64(args[2], &context)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[3], &options_wire, &options_ptr, &options_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context(receiver, envelope_ptr, envelope_len, context, options_ptr, options_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(return_success);
done:
    Py_XDECREF(envelope_wire);
    Py_XDECREF(options_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *envelope_wire = NULL;
    const uint8_t *envelope_ptr = NULL;
    uintptr_t envelope_len = 0;
    uint64_t context;
    PyObject *options_wire = NULL;
    const uint8_t *options_ptr = NULL;
    uintptr_t options_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 4) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_run_stream_with_context() takes 4 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[1], &envelope_wire, &envelope_ptr, &envelope_len)) {
        goto done;
    }
    if (!boltffi_python_parse_u64(args[2], &context)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[3], &options_wire, &options_ptr, &options_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context(receiver, envelope_ptr, envelope_len, context, options_ptr, options_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(envelope_wire);
    Py_XDECREF(options_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_warmup(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_warmup() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_warmup(receiver);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_unload(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_model_unload() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_model_unload(receiver);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_release_class_xybrid_bolt_xybrid_conversation_context(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t handle;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_release() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        return NULL;
    }
    if (!boltffi_python_parse_u64(args[0], &handle)) {
        return NULL;
    }
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_conversation_context(handle);
    Py_RETURN_NONE;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *result = NULL;
    (void)self;
    if (nargs != 0) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_new() takes 0 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    result = boltffi_python_box_u64(boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new());
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *id_wire = NULL;
    const uint8_t *id_ptr = NULL;
    uintptr_t id_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_with_id() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &id_wire, &id_ptr, &id_len)) {
        goto done;
    }
    result = boltffi_python_box_u64(boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id(id_ptr, id_len));
done:
    Py_XDECREF(id_wire);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *envelope_wire = NULL;
    const uint8_t *envelope_ptr = NULL;
    uintptr_t envelope_len = 0;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_push() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[1], &envelope_wire, &envelope_ptr, &envelope_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push(receiver, envelope_ptr, envelope_len);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(envelope_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *envelope_wire = NULL;
    const uint8_t *envelope_ptr = NULL;
    uintptr_t envelope_len = 0;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_set_system() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[1], &envelope_wire, &envelope_ptr, &envelope_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system(receiver, envelope_ptr, envelope_len);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(envelope_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_clear() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear(receiver);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_id() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_history_len() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_u32(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_history() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_has_system() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    uint32_t len;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_conversation_context_set_max_history_len() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_parse_u32(args[1], &len)) {
        goto done;
    }
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len(receiver, len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t handle;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_telemetry_config_release() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        return NULL;
    }
    if (!boltffi_python_parse_u64(args[0], &handle)) {
        return NULL;
    }
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config(handle);
    Py_RETURN_NONE;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *api_key_wire = NULL;
    const uint8_t *api_key_ptr = NULL;
    uintptr_t api_key_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_telemetry_config_new() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &api_key_wire, &api_key_ptr, &api_key_len)) {
        goto done;
    }
    result = boltffi_python_box_u64(boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new(api_key_ptr, api_key_len));
done:
    Py_XDECREF(api_key_wire);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *endpoint_wire = NULL;
    const uint8_t *endpoint_ptr = NULL;
    uintptr_t endpoint_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_telemetry_config_set_endpoint() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[1], &endpoint_wire, &endpoint_ptr, &endpoint_len)) {
        goto done;
    }
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint(receiver, endpoint_ptr, endpoint_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(endpoint_wire);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *version_wire = NULL;
    const uint8_t *version_ptr = NULL;
    uintptr_t version_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_telemetry_config_set_app_version() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[1], &version_wire, &version_ptr, &version_len)) {
        goto done;
    }
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version(receiver, version_ptr, version_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(version_wire);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *label_wire = NULL;
    const uint8_t *label_ptr = NULL;
    uintptr_t label_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_telemetry_config_set_device_label() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[1], &label_wire, &label_ptr, &label_len)) {
        goto done;
    }
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label(receiver, label_ptr, label_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(label_wire);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *key_wire = NULL;
    const uint8_t *key_ptr = NULL;
    uintptr_t key_len = 0;
    PyObject *value_wire = NULL;
    const uint8_t *value_ptr = NULL;
    uintptr_t value_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 3) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_telemetry_config_set_device_attribute() takes 3 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[1], &key_wire, &key_ptr, &key_len)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[2], &value_wire, &value_ptr, &value_len)) {
        goto done;
    }
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute(receiver, key_ptr, key_len, value_ptr, value_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(key_wire);
    Py_XDECREF(value_wire);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    uint32_t batch_size;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_telemetry_config_set_batch_size() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_parse_u32(args[1], &batch_size)) {
        goto done;
    }
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size(receiver, batch_size);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    uint32_t secs;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_telemetry_config_set_flush_interval_secs() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_parse_u32(args[1], &secs)) {
        goto done;
    }
    boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs(receiver, secs);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_telemetry_config_init() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init(receiver);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_release_class_xybrid_bolt_xybrid_bundle(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t handle;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_release() takes 1 positional argument but %zd were given", nargs);
        return NULL;
    }
    if (boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        return NULL;
    }
    if (!boltffi_python_parse_u64(args[0], &handle)) {
        return NULL;
    }
    boltffi_python_boltffi_release_class_xybrid_bolt_xybrid_bundle(handle);
    Py_RETURN_NONE;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_bundle_open(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *path_wire = NULL;
    const uint8_t *path_ptr = NULL;
    uintptr_t path_len = 0;
    uint64_t return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_open() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &path_wire, &path_ptr, &path_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_init_class_xybrid_bolt_xybrid_bundle_open(path_ptr, path_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_box_u64(return_success);
done:
    Py_XDECREF(path_wire);
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_model_id() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_version(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_version() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_version(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_target(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_target() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_target(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_hash() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_has_metadata() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_file_count() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    result = boltffi_python_box_u32(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count(receiver));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    uint32_t index;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_file_name() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_parse_u32(args[1], &index)) {
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name(receiver, index));
done:
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    FfiBuf_u8 return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_manifest_json() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json(receiver, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(return_success);
done:
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    FfiBuf_u8 return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_metadata_json() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json(receiver, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(return_success);
done:
    Py_XDECREF(error);
    return result;
}

static PyObject *boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint64_t receiver;
    PyObject *output_dir_wire = NULL;
    const uint8_t *output_dir_ptr = NULL;
    uintptr_t output_dir_len = 0;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "_boltffi_xybrid_bundle_extract() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u64(args[0], &receiver)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[1], &output_dir_wire, &output_dir_ptr, &output_dir_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract(receiver, output_dir_ptr, output_dir_len);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(output_dir_wire);
    Py_XDECREF(error);
    return result;
}


static int boltffi_python_bind_host_state(void) {
    return 1;
}

static void boltffi_python_release_host_state(void) {
    Py_CLEAR(boltffi_python_xybrid_metadata_entry_type);
    Py_CLEAR(boltffi_python_xybrid_envelope_type);
    Py_CLEAR(boltffi_python_xybrid_tool_definition_type);
    Py_CLEAR(boltffi_python_xybrid_tool_call_type);
    Py_CLEAR(boltffi_python_xybrid_tool_result_type);
    Py_CLEAR(boltffi_python_xybrid_generation_config_type);
    Py_CLEAR(boltffi_python_xybrid_run_options_type);
    Py_CLEAR(boltffi_python_xybrid_stage_latency_type);
    Py_CLEAR(boltffi_python_xybrid_inference_metrics_type);
    Py_CLEAR(boltffi_python_xybrid_result_type);
    Py_CLEAR(boltffi_python_xybrid_download_status_type);
    Py_CLEAR(boltffi_python_xybrid_stream_token_type);
    Py_CLEAR(boltffi_python_xybrid_stream_event_type);
    Py_CLEAR(boltffi_python_xybrid_voice_info_type);
    Py_CLEAR(boltffi_python_xybrid_error_type);
    Py_CLEAR(boltffi_python_xybrid_envelope_kind_type);
    boltffi_python_clear_c_style_enum_registration(&boltffi_python_xybrid_message_role_registration);
    boltffi_python_clear_c_style_enum_registration(&boltffi_python_xybrid_abort_signal_registration);
    boltffi_python_clear_c_style_enum_registration(&boltffi_python_xybrid_output_type_registration);
    boltffi_python_clear_c_style_enum_registration(&boltffi_python_xybrid_execution_target_registration);
    boltffi_python_clear_c_style_enum_registration(&boltffi_python_xybrid_download_state_registration);
    boltffi_python_clear_c_style_enum_registration(&boltffi_python_xybrid_stream_event_kind_registration);
    boltffi_python_clear_c_style_enum_registration(&boltffi_python_xybrid_thermal_state_registration);

    Py_CLEAR(boltffi_python_wire_codecs);

}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_tool_results_envelope(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *user_text_wire = NULL;
    const uint8_t *user_text_ptr = NULL;
    uintptr_t user_text_len = 0;
    PyObject *prior_assistant_text_wire = NULL;
    const uint8_t *prior_assistant_text_ptr = NULL;
    uintptr_t prior_assistant_text_len = 0;
    PyObject *results_wire = NULL;
    const uint8_t *results_ptr = NULL;
    uintptr_t results_len = 0;
    FfiBuf_u8 return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 3) {
        PyErr_Format(PyExc_TypeError, "tool_results_envelope() takes 3 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &user_text_wire, &user_text_ptr, &user_text_len)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[1], &prior_assistant_text_wire, &prior_assistant_text_ptr, &prior_assistant_text_len)) {
        goto done;
    }
    if (!boltffi_python_wire_vec_xybrid_tool_result(args[2], &results_wire, &results_ptr, &results_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_function_xybrid_bolt_tool_results_envelope(user_text_ptr, user_text_len, prior_assistant_text_ptr, prior_assistant_text_len, results_ptr, results_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_decode_owned_raw_wire(return_success);
done:
    Py_XDECREF(user_text_wire);
    Py_XDECREF(prior_assistant_text_wire);
    Py_XDECREF(results_wire);
    Py_XDECREF(error);
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_json_schema_to_gbnf(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *schema_json_wire = NULL;
    const uint8_t *schema_json_ptr = NULL;
    uintptr_t schema_json_len = 0;
    FfiBuf_u8 return_success;
    FfiBuf_u8 return_error = {0};
    PyObject *error = NULL;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "json_schema_to_gbnf() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &schema_json_wire, &schema_json_ptr, &schema_json_len)) {
        goto done;
    }
    return_error = boltffi_python_boltffi_function_xybrid_bolt_json_schema_to_gbnf(schema_json_ptr, schema_json_len, &return_success);
    if (return_error.len != 0) {
        error = boltffi_python_decode_owned_raw_wire(return_error);
        if (error != NULL) {
            PyErr_SetObject(PyExc_RuntimeError, error);
        }
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(return_success);
done:
    Py_XDECREF(schema_json_wire);
    Py_XDECREF(error);
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_thermal_state(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    ___XybridThermalState state;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "set_thermal_state() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_xybrid_thermal_state(args[0], &state)) {
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_set_thermal_state(state);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_clear_thermal_state(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *result = NULL;
    (void)self;
    if (nargs != 0) {
        PyErr_Format(PyExc_TypeError, "clear_thermal_state() takes 0 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_clear_thermal_state();
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_battery_level(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    uint8_t percent;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "set_battery_level() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_set_battery_level == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_u8(args[0], &percent)) {
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_set_battery_level(percent);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_clear_battery_level(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *result = NULL;
    (void)self;
    if (nargs != 0) {
        PyErr_Format(PyExc_TypeError, "clear_battery_level() takes 0 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_clear_battery_level();
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_configure_runtime(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *api_key_wire = NULL;
    const uint8_t *api_key_ptr = NULL;
    uintptr_t api_key_len = 0;
    PyObject *gateway_url_wire = NULL;
    const uint8_t *gateway_url_ptr = NULL;
    uintptr_t gateway_url_len = 0;
    PyObject *ingest_url_wire = NULL;
    const uint8_t *ingest_url_ptr = NULL;
    uintptr_t ingest_url_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 3) {
        PyErr_Format(PyExc_TypeError, "configure_runtime() takes 3 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_configure_runtime == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_raw(args[0], &api_key_wire, &api_key_ptr, &api_key_len)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[1], &gateway_url_wire, &gateway_url_ptr, &gateway_url_len)) {
        goto done;
    }
    if (!boltffi_python_wire_raw(args[2], &ingest_url_wire, &ingest_url_ptr, &ingest_url_len)) {
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_configure_runtime(api_key_ptr, api_key_len, gateway_url_ptr, gateway_url_len, ingest_url_ptr, ingest_url_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(api_key_wire);
    Py_XDECREF(gateway_url_wire);
    Py_XDECREF(ingest_url_wire);
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_init_sdk_cache_dir(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *cache_dir_wire = NULL;
    const uint8_t *cache_dir_ptr = NULL;
    uintptr_t cache_dir_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "init_sdk_cache_dir() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &cache_dir_wire, &cache_dir_ptr, &cache_dir_len)) {
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_init_sdk_cache_dir(cache_dir_ptr, cache_dir_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(cache_dir_wire);
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_binding(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *binding_wire = NULL;
    const uint8_t *binding_ptr = NULL;
    uintptr_t binding_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "set_binding() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_set_binding == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &binding_wire, &binding_ptr, &binding_len)) {
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_set_binding(binding_ptr, binding_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(binding_wire);
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_api_key(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *api_key_wire = NULL;
    const uint8_t *api_key_ptr = NULL;
    uintptr_t api_key_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "set_api_key() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_set_api_key == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &api_key_wire, &api_key_ptr, &api_key_len)) {
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_set_api_key(api_key_ptr, api_key_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(api_key_wire);
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_provider_api_key(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *provider_wire = NULL;
    const uint8_t *provider_ptr = NULL;
    uintptr_t provider_len = 0;
    PyObject *api_key_wire = NULL;
    const uint8_t *api_key_ptr = NULL;
    uintptr_t api_key_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "set_provider_api_key() takes 2 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &provider_wire, &provider_ptr, &provider_len)) {
        goto done;
    }
    if (!boltffi_python_wire_string(args[1], &api_key_wire, &api_key_ptr, &api_key_len)) {
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_set_provider_api_key(provider_ptr, provider_len, api_key_ptr, api_key_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(provider_wire);
    Py_XDECREF(api_key_wire);
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_platform_url(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *url_wire = NULL;
    const uint8_t *url_ptr = NULL;
    uintptr_t url_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "set_platform_url() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_set_platform_url == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &url_wire, &url_ptr, &url_len)) {
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_set_platform_url(url_ptr, url_len);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    Py_XDECREF(url_wire);
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_speculative_cloud(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    bool enabled;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "set_speculative_cloud() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_parse_bool(args[0], &enabled)) {
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_set_speculative_cloud(enabled);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_has_api_key(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *result = NULL;
    (void)self;
    if (nargs != 0) {
        PyErr_Format(PyExc_TypeError, "has_api_key() takes 0 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_has_api_key == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_function_xybrid_bolt_has_api_key());
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *result = NULL;
    (void)self;
    if (nargs != 0) {
        PyErr_Format(PyExc_TypeError, "is_speculative_cloud_enabled() takes 0 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled());
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_will_speculate_for_model(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *model_id_wire = NULL;
    const uint8_t *model_id_ptr = NULL;
    uintptr_t model_id_len = 0;
    PyObject *result = NULL;
    (void)self;
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "will_speculate_for_model() takes 1 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    if (!boltffi_python_wire_string(args[0], &model_id_wire, &model_id_ptr, &model_id_len)) {
        goto done;
    }
    result = boltffi_python_box_bool(boltffi_python_boltffi_function_xybrid_bolt_will_speculate_for_model(model_id_ptr, model_id_len));
done:
    Py_XDECREF(model_id_wire);
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_version(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *result = NULL;
    (void)self;
    if (nargs != 0) {
        PyErr_Format(PyExc_TypeError, "version() takes 0 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_version == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(boltffi_python_boltffi_function_xybrid_bolt_version());
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_telemetry_default_endpoint(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *result = NULL;
    (void)self;
    if (nargs != 0) {
        PyErr_Format(PyExc_TypeError, "telemetry_default_endpoint() takes 0 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    result = boltffi_python_decode_owned_utf8(boltffi_python_boltffi_function_xybrid_bolt_telemetry_default_endpoint());
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_telemetry_flush(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *result = NULL;
    (void)self;
    if (nargs != 0) {
        PyErr_Format(PyExc_TypeError, "telemetry_flush() takes 0 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_flush();
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}


static PyObject *boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_telemetry_shutdown(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    PyObject *result = NULL;
    (void)self;
    if (nargs != 0) {
        PyErr_Format(PyExc_TypeError, "telemetry_shutdown() takes 0 positional arguments but %zd were given", nargs);
        goto done;
    }
    if (boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown == NULL) {
        PyErr_SetString(PyExc_ImportError, "native library is not initialized");
        goto done;
    }
    boltffi_python_boltffi_function_xybrid_bolt_telemetry_shutdown();
    Py_INCREF(Py_None);
    result = Py_None;
done:
    return result;
}



static PyMethodDef boltffi_python_methods[] = {

    {"_register_wire_codec", (PyCFunction)boltffi_python_register_wire_codec, METH_VARARGS, NULL},

    {"_initialize_loader", (PyCFunction)boltffi_python_initialize_loader, METH_O, NULL},
    {"_register_xybrid_metadata_entry", (PyCFunction)boltffi_python_wrapper_register_xybrid_metadata_entry, METH_FASTCALL, NULL},
    {"_register_xybrid_envelope", (PyCFunction)boltffi_python_wrapper_register_xybrid_envelope, METH_FASTCALL, NULL},
    {"_register_xybrid_tool_definition", (PyCFunction)boltffi_python_wrapper_register_xybrid_tool_definition, METH_FASTCALL, NULL},
    {"_register_xybrid_tool_call", (PyCFunction)boltffi_python_wrapper_register_xybrid_tool_call, METH_FASTCALL, NULL},
    {"_register_xybrid_tool_result", (PyCFunction)boltffi_python_wrapper_register_xybrid_tool_result, METH_FASTCALL, NULL},
    {"_register_xybrid_generation_config", (PyCFunction)boltffi_python_wrapper_register_xybrid_generation_config, METH_FASTCALL, NULL},
    {"_register_xybrid_run_options", (PyCFunction)boltffi_python_wrapper_register_xybrid_run_options, METH_FASTCALL, NULL},
    {"_register_xybrid_stage_latency", (PyCFunction)boltffi_python_wrapper_register_xybrid_stage_latency, METH_FASTCALL, NULL},
    {"_register_xybrid_inference_metrics", (PyCFunction)boltffi_python_wrapper_register_xybrid_inference_metrics, METH_FASTCALL, NULL},
    {"_register_xybrid_result", (PyCFunction)boltffi_python_wrapper_register_xybrid_result, METH_FASTCALL, NULL},
    {"_register_xybrid_download_status", (PyCFunction)boltffi_python_wrapper_register_xybrid_download_status, METH_FASTCALL, NULL},
    {"_register_xybrid_stream_token", (PyCFunction)boltffi_python_wrapper_register_xybrid_stream_token, METH_FASTCALL, NULL},
    {"_register_xybrid_stream_event", (PyCFunction)boltffi_python_wrapper_register_xybrid_stream_event, METH_FASTCALL, NULL},
    {"_register_xybrid_voice_info", (PyCFunction)boltffi_python_wrapper_register_xybrid_voice_info, METH_FASTCALL, NULL},
    {"_register_xybrid_error", (PyCFunction)boltffi_python_wrapper_register_xybrid_error, METH_FASTCALL, NULL},
    {"_register_xybrid_envelope_kind", (PyCFunction)boltffi_python_wrapper_register_xybrid_envelope_kind, METH_FASTCALL, NULL},
    {"_register_xybrid_message_role", (PyCFunction)boltffi_python_wrapper_register_xybrid_message_role, METH_FASTCALL, NULL},
    {"_register_xybrid_abort_signal", (PyCFunction)boltffi_python_wrapper_register_xybrid_abort_signal, METH_FASTCALL, NULL},
    {"_register_xybrid_output_type", (PyCFunction)boltffi_python_wrapper_register_xybrid_output_type, METH_FASTCALL, NULL},
    {"_register_xybrid_execution_target", (PyCFunction)boltffi_python_wrapper_register_xybrid_execution_target, METH_FASTCALL, NULL},
    {"_register_xybrid_download_state", (PyCFunction)boltffi_python_wrapper_register_xybrid_download_state, METH_FASTCALL, NULL},
    {"_register_xybrid_stream_event_kind", (PyCFunction)boltffi_python_wrapper_register_xybrid_stream_event_kind, METH_FASTCALL, NULL},
    {"_register_xybrid_thermal_state", (PyCFunction)boltffi_python_wrapper_register_xybrid_thermal_state, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_release", (PyCFunction)boltffi_python_callable_wrapper_boltffi_release_class_xybrid_bolt_xybrid_model, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_from_registry", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_from_registry_speculative", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_from_directory", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_directory, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_from_bundle", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_from_huggingface", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_from_huggingface_with_revision", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_from_model_file", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_model_id", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_model_id, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_version", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_version, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_output_type", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_output_type, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_is_loaded", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_is_cloud_serving", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_download_status", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_download_status, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_await_download", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_await_download, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_supports_streaming", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_supports_token_streaming", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_default_generation_config", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_is_llm", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_is_llm, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_supports_tool_calling", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_has_voices", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_has_voices, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_voices", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_voices, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_default_voice", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_default_voice, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_voice", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_voice, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_run", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_run, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_run_stream", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_stream_next", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_stream_next, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_stream_result", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_stream_result, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_stream_close", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_stream_close, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_run_with_context", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_run_stream_with_context", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_warmup", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_warmup, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_model_unload", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_model_unload, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_release", (PyCFunction)boltffi_python_callable_wrapper_boltffi_release_class_xybrid_bolt_xybrid_conversation_context, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_new", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_with_id", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_push", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_set_system", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_clear", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_id", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_history_len", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_history", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_has_system", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_conversation_context_set_max_history_len", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_telemetry_config_release", (PyCFunction)boltffi_python_callable_wrapper_boltffi_release_class_xybrid_bolt_xybrid_telemetry_config, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_telemetry_config_new", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_telemetry_config_set_endpoint", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_telemetry_config_set_app_version", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_telemetry_config_set_device_label", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_telemetry_config_set_device_attribute", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_telemetry_config_set_batch_size", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_telemetry_config_set_flush_interval_secs", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_telemetry_config_init", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_release", (PyCFunction)boltffi_python_callable_wrapper_boltffi_release_class_xybrid_bolt_xybrid_bundle, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_open", (PyCFunction)boltffi_python_callable_wrapper_boltffi_init_class_xybrid_bolt_xybrid_bundle_open, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_model_id", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_version", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_version, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_target", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_target, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_hash", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_hash, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_has_metadata", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_file_count", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_file_name", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_manifest_json", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_metadata_json", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json, METH_FASTCALL, NULL},
    {"_boltffi_xybrid_bundle_extract", (PyCFunction)boltffi_python_callable_wrapper_boltffi_method_class_xybrid_bolt_xybrid_bundle_extract, METH_FASTCALL, NULL},
    {"tool_results_envelope", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_tool_results_envelope, METH_FASTCALL, NULL},
    {"json_schema_to_gbnf", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_json_schema_to_gbnf, METH_FASTCALL, NULL},
    {"set_thermal_state", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_thermal_state, METH_FASTCALL, NULL},
    {"clear_thermal_state", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_clear_thermal_state, METH_FASTCALL, NULL},
    {"set_battery_level", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_battery_level, METH_FASTCALL, NULL},
    {"clear_battery_level", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_clear_battery_level, METH_FASTCALL, NULL},
    {"configure_runtime", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_configure_runtime, METH_FASTCALL, NULL},
    {"init_sdk_cache_dir", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_init_sdk_cache_dir, METH_FASTCALL, NULL},
    {"set_binding", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_binding, METH_FASTCALL, NULL},
    {"set_api_key", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_api_key, METH_FASTCALL, NULL},
    {"set_provider_api_key", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_provider_api_key, METH_FASTCALL, NULL},
    {"set_platform_url", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_platform_url, METH_FASTCALL, NULL},
    {"set_speculative_cloud", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_set_speculative_cloud, METH_FASTCALL, NULL},
    {"has_api_key", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_has_api_key, METH_FASTCALL, NULL},
    {"is_speculative_cloud_enabled", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_is_speculative_cloud_enabled, METH_FASTCALL, NULL},
    {"will_speculate_for_model", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_will_speculate_for_model, METH_FASTCALL, NULL},
    {"version", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_version, METH_FASTCALL, NULL},
    {"telemetry_default_endpoint", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_telemetry_default_endpoint, METH_FASTCALL, NULL},
    {"telemetry_flush", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_telemetry_flush, METH_FASTCALL, NULL},
    {"telemetry_shutdown", (PyCFunction)boltffi_python_callable_wrapper_boltffi_function_xybrid_bolt_telemetry_shutdown, METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef boltffi_python_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    NULL,
    -1,
    boltffi_python_methods,
    NULL,
    NULL,
    NULL,
    boltffi_python_free_module
};

PyMODINIT_FUNC PyInit__native(void) {
    PyObject *module = PyModule_Create(&boltffi_python_module);
    if (module == NULL) {
        return NULL;
    }
    return module;
}