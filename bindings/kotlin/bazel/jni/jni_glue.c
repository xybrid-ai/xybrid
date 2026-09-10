#include <jni.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#if defined(__ANDROID__)
#include <pthread.h>
#endif

#include "xybrid_bolt.h"
static void boltffi_jni_throw_runtime(JNIEnv *env, const char *message) {
    jclass exception_class = (*env)->FindClass(env, "java/lang/RuntimeException");
    if (exception_class == NULL) {
        return;
    }
    (*env)->ThrowNew(env, exception_class, message);
    (*env)->DeleteLocalRef(env, exception_class);
}

static void boltffi_jni_throw_illegal_argument(JNIEnv *env, const char *message) {
    jclass exception_class = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
    if (exception_class == NULL) {
        return;
    }
    (*env)->ThrowNew(env, exception_class, message);
    (*env)->DeleteLocalRef(env, exception_class);
}

static void boltffi_jni_throw_status(JNIEnv *env, FfiStatus status) {
    if (status.code != 0) {
        boltffi_jni_throw_runtime(env, "BoltFFI call failed");
    }
}

static void boltffi_jni_throw_error_buffer(JNIEnv *env, FfiBuf_u8 buffer) {
    if (buffer.len > ((uintptr_t)INT32_MAX)) {
        boltffi_free_buf(buffer);
        boltffi_jni_throw_runtime(env, "BoltFFI error buffer was too large");
        return;
    }
    jbyteArray bytes = (*env)->NewByteArray(env, (jsize)buffer.len);
    if (bytes != NULL && buffer.len != 0) {
        (*env)->SetByteArrayRegion(env, bytes, 0, (jsize)buffer.len, (const jbyte *)buffer.ptr);
    }
    boltffi_free_buf(buffer);
    if (bytes == NULL || (*env)->ExceptionCheck(env)) {
        return;
    }
    jclass exception_class = (*env)->FindClass(env, "ai/xybrid/BoltFfiErrorBufferException");
    if (exception_class == NULL) {
        (*env)->DeleteLocalRef(env, bytes);
        return;
    }
    jmethodID constructor = (*env)->GetMethodID(env, exception_class, "<init>", "([B)V");
    if (constructor == NULL) {
        (*env)->DeleteLocalRef(env, exception_class);
        (*env)->DeleteLocalRef(env, bytes);
        return;
    }
    jthrowable exception = (jthrowable)(*env)->NewObject(env, exception_class, constructor, bytes);
    if (exception != NULL) {
        (*env)->Throw(env, exception);
        (*env)->DeleteLocalRef(env, exception);
    }
    (*env)->DeleteLocalRef(env, exception_class);
    (*env)->DeleteLocalRef(env, bytes);
}

static jbyteArray boltffi_jni_buffer_to_byte_array(JNIEnv *env, FfiBuf_u8 buffer) {
    if (buffer.ptr == NULL) {
        if (buffer.len != 0) {
            boltffi_jni_throw_runtime(env, "BoltFFI buffer pointer was null with non-zero length");
            return NULL;
        }
        return (*env)->NewByteArray(env, 0);
    }
    if (buffer.len > (uintptr_t)INT32_MAX) {
        boltffi_free_buf(buffer);
        boltffi_jni_throw_runtime(env, "BoltFFI buffer too large for Java byte array");
        return NULL;
    }
    jbyteArray array = (*env)->NewByteArray(env, (jsize)buffer.len);
    if (array == NULL) {
        boltffi_free_buf(buffer);
        return NULL;
    }
    (*env)->SetByteArrayRegion(env, array, 0, (jsize)buffer.len, (const jbyte *)buffer.ptr);
    boltffi_free_buf(buffer);
    if ((*env)->ExceptionCheck(env)) {
        (*env)->DeleteLocalRef(env, array);
        return NULL;
    }
    return array;
}

static inline jbyteArray boltffi_jni_bytes_to_byte_array(JNIEnv *env, const uint8_t *bytes, uintptr_t len) {
    if (bytes == NULL && len != 0) {
        boltffi_jni_throw_runtime(env, "BoltFFI byte slice pointer was null with non-zero length");
        return NULL;
    }
    if (len > (uintptr_t)INT32_MAX) {
        boltffi_jni_throw_runtime(env, "BoltFFI byte slice too large for Java byte array");
        return NULL;
    }
    jbyteArray array = (*env)->NewByteArray(env, (jsize)len);
    if (array == NULL) {
        return NULL;
    }
    if (len != 0) {
        (*env)->SetByteArrayRegion(env, array, 0, (jsize)len, (const jbyte *)bytes);
    }
    return array;
}

static inline FfiBuf_u8 boltffi_jni_byte_array_to_buffer(JNIEnv *env, jbyteArray array) {
    FfiBuf_u8 empty = {0};
    if (array == NULL) {
        boltffi_jni_throw_runtime(env, "BoltFFI byte array return was null");
        return empty;
    }
    jsize len = (*env)->GetArrayLength(env, array);
    if (len == 0) {
        return empty;
    }
    FfiBuf_u8 buffer = boltffi_buf_with_len((uintptr_t)len);
    if (buffer.ptr == NULL) {
        boltffi_jni_throw_runtime(env, "failed to allocate BoltFFI byte array return");
        return empty;
    }
    (*env)->GetByteArrayRegion(env, array, 0, len, (jbyte *)buffer.ptr);
    return buffer;
}

static bool boltffi_jni_direct_buffer_address(JNIEnv *env, jobject buffer, jlong required_capacity, void **address) {
    if (buffer == NULL) {
        boltffi_jni_throw_illegal_argument(env, "BoltFFI direct buffer argument was null");
        return false;
    }
    if (required_capacity < 0) {
        boltffi_jni_throw_illegal_argument(env, "BoltFFI direct buffer length was negative");
        return false;
    }
    jlong capacity = (*env)->GetDirectBufferCapacity(env, buffer);
    if (capacity < 0) {
        boltffi_jni_throw_illegal_argument(env, "BoltFFI argument was not a direct buffer");
        return false;
    }
    if (capacity < required_capacity) {
        boltffi_jni_throw_illegal_argument(env, "BoltFFI direct buffer capacity was too small");
        return false;
    }
    *address = (*env)->GetDirectBufferAddress(env, buffer);
    if (*address == NULL && required_capacity != 0) {
        boltffi_jni_throw_illegal_argument(env, "BoltFFI direct buffer address was unavailable");
        return false;
    }
    return true;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1release_1class_1xybrid_1bolt_1xybrid_1model(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;

    (void)env;
    boltffi_release_class_xybrid_bolt_xybrid_model(handle);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1model_1from_1registry(JNIEnv *env, jclass cls, jobject id, jint __boltffi_id_len) {
    (void)cls;

    void *__boltffi_id_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, id, (jlong)__boltffi_id_len, &__boltffi_id_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_model_from_registry((const uint8_t *)__boltffi_id_ptr, (uintptr_t)__boltffi_id_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1model_1from_1registry_1speculative(JNIEnv *env, jclass cls, jobject id, jint __boltffi_id_len) {
    (void)cls;

    void *__boltffi_id_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, id, (jlong)__boltffi_id_len, &__boltffi_id_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative((const uint8_t *)__boltffi_id_ptr, (uintptr_t)__boltffi_id_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1model_1from_1directory(JNIEnv *env, jclass cls, jobject path, jint __boltffi_path_len) {
    (void)cls;

    void *__boltffi_path_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, path, (jlong)__boltffi_path_len, &__boltffi_path_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_model_from_directory((const uint8_t *)__boltffi_path_ptr, (uintptr_t)__boltffi_path_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1model_1from_1bundle(JNIEnv *env, jclass cls, jobject path, jint __boltffi_path_len) {
    (void)cls;

    void *__boltffi_path_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, path, (jlong)__boltffi_path_len, &__boltffi_path_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle((const uint8_t *)__boltffi_path_ptr, (uintptr_t)__boltffi_path_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1model_1from_1huggingface(JNIEnv *env, jclass cls, jobject repo, jint __boltffi_repo_len) {
    (void)cls;

    void *__boltffi_repo_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, repo, (jlong)__boltffi_repo_len, &__boltffi_repo_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface((const uint8_t *)__boltffi_repo_ptr, (uintptr_t)__boltffi_repo_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1model_1from_1huggingface_1with_1revision(JNIEnv *env, jclass cls, jobject repo, jint __boltffi_repo_len, jobject revision, jint __boltffi_revision_len) {
    (void)cls;

    void *__boltffi_repo_ptr = NULL;
    void *__boltffi_revision_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, repo, (jlong)__boltffi_repo_len, &__boltffi_repo_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, revision, (jlong)__boltffi_revision_len, &__boltffi_revision_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision((const uint8_t *)__boltffi_repo_ptr, (uintptr_t)__boltffi_repo_len, (const uint8_t *)__boltffi_revision_ptr, (uintptr_t)__boltffi_revision_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1model_1from_1model_1file(JNIEnv *env, jclass cls, jobject path, jint __boltffi_path_len) {
    (void)cls;

    void *__boltffi_path_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, path, (jlong)__boltffi_path_len, &__boltffi_path_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file((const uint8_t *)__boltffi_path_ptr, (uintptr_t)__boltffi_path_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1model_1id(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_model_id(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1version(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_version(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1output_1type(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    ___XybridOutputType __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_output_type(receiver);

    return (jint)__boltffi_result;
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1is_1loaded(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1is_1cloud_1serving(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1download_1status(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_download_status(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1await_1download(JNIEnv *env, jclass cls, jlong receiver, jlong timeout_ms) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_await_download(receiver, timeout_ms);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1supports_1streaming(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1supports_1token_1streaming(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1default_1generation_1config(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1is_1llm(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_is_llm(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1supports_1tool_1calling(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1has_1voices(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_has_voices(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1voices(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_voices(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1default_1voice(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_default_voice(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1voice(JNIEnv *env, jclass cls, jlong receiver, jobject voice_id, jint __boltffi_voice_id_len) {
    (void)cls;

    void *__boltffi_voice_id_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, voice_id, (jlong)__boltffi_voice_id_len, &__boltffi_voice_id_ptr)) {
        goto __boltffi_error;
    }

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_model_voice(receiver, (const uint8_t *)__boltffi_voice_id_ptr, (uintptr_t)__boltffi_voice_id_len);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
__boltffi_error:
    return NULL;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1run(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len, jobject options, jint __boltffi_options_len) {
    (void)cls;

    void *__boltffi_envelope_ptr = NULL;
    void *__boltffi_options_ptr = NULL;
    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    if (!boltffi_jni_direct_buffer_address(env, envelope, (jlong)__boltffi_envelope_len, &__boltffi_envelope_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, options, (jlong)__boltffi_options_len, &__boltffi_options_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_run(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len, (const uint8_t *)__boltffi_options_ptr, (uintptr_t)__boltffi_options_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
__boltffi_error:
    return NULL;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1run_1stream(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len, jobject options, jint __boltffi_options_len) {
    (void)cls;

    void *__boltffi_envelope_ptr = NULL;
    void *__boltffi_options_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, envelope, (jlong)__boltffi_envelope_len, &__boltffi_envelope_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, options, (jlong)__boltffi_options_len, &__boltffi_options_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_run_stream(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len, (const uint8_t *)__boltffi_options_ptr, (uintptr_t)__boltffi_options_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1stream_1next(JNIEnv *env, jclass cls, jlong receiver, jlong stream_id) {
    (void)cls;

    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_stream_next(receiver, stream_id, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1stream_1result(JNIEnv *env, jclass cls, jlong receiver, jlong stream_id) {
    (void)cls;

    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_stream_result(receiver, stream_id, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1stream_1close(JNIEnv *env, jclass cls, jlong receiver, jlong stream_id) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_model_stream_close(receiver, stream_id);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1run_1with_1context(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len, jlong context, jobject options, jint __boltffi_options_len) {
    (void)cls;

    void *__boltffi_envelope_ptr = NULL;
    void *__boltffi_options_ptr = NULL;
    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    if (!boltffi_jni_direct_buffer_address(env, envelope, (jlong)__boltffi_envelope_len, &__boltffi_envelope_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, options, (jlong)__boltffi_options_len, &__boltffi_options_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len, context, (const uint8_t *)__boltffi_options_ptr, (uintptr_t)__boltffi_options_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
__boltffi_error:
    return NULL;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1run_1stream_1with_1context(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len, jlong context, jobject options, jint __boltffi_options_len) {
    (void)cls;

    void *__boltffi_envelope_ptr = NULL;
    void *__boltffi_options_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, envelope, (jlong)__boltffi_envelope_len, &__boltffi_envelope_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, options, (jlong)__boltffi_options_len, &__boltffi_options_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len, context, (const uint8_t *)__boltffi_options_ptr, (uintptr_t)__boltffi_options_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1warmup(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_warmup(receiver);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1unload(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_unload(receiver);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1release_1class_1xybrid_1bolt_1xybrid_1conversation_1context(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;

    (void)env;
    boltffi_release_class_xybrid_bolt_xybrid_conversation_context(handle);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1new(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    uint64_t __boltffi_result = boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new();

    return (jlong)__boltffi_result;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1with_1id(JNIEnv *env, jclass cls, jobject id, jint __boltffi_id_len) {
    (void)cls;

    void *__boltffi_id_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, id, (jlong)__boltffi_id_len, &__boltffi_id_ptr)) {
        goto __boltffi_error;
    }

    (void)env;
    uint64_t __boltffi_result = boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id((const uint8_t *)__boltffi_id_ptr, (uintptr_t)__boltffi_id_len);

    return (jlong)__boltffi_result;
__boltffi_error:
    return 0;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1push(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len) {
    (void)cls;

    void *__boltffi_envelope_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, envelope, (jlong)__boltffi_envelope_len, &__boltffi_envelope_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1set_1system(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len) {
    (void)cls;

    void *__boltffi_envelope_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, envelope, (jlong)__boltffi_envelope_len, &__boltffi_envelope_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1clear(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear(receiver);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1id(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1history_1len(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    uint32_t __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len(receiver);

    return (jint)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1history(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1has_1system(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1conversation_1context_1set_1max_1history_1len(JNIEnv *env, jclass cls, jlong receiver, jint len) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len(receiver, len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1release_1class_1xybrid_1bolt_1xybrid_1telemetry_1config(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;

    (void)env;
    boltffi_release_class_xybrid_bolt_xybrid_telemetry_config(handle);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1telemetry_1config_1new(JNIEnv *env, jclass cls, jobject api_key, jint __boltffi_api_key_len) {
    (void)cls;

    void *__boltffi_api_key_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, api_key, (jlong)__boltffi_api_key_len, &__boltffi_api_key_ptr)) {
        goto __boltffi_error;
    }

    (void)env;
    uint64_t __boltffi_result = boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new((const uint8_t *)__boltffi_api_key_ptr, (uintptr_t)__boltffi_api_key_len);

    return (jlong)__boltffi_result;
__boltffi_error:
    return 0;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1telemetry_1config_1set_1endpoint(JNIEnv *env, jclass cls, jlong receiver, jobject endpoint, jint __boltffi_endpoint_len) {
    (void)cls;

    void *__boltffi_endpoint_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, endpoint, (jlong)__boltffi_endpoint_len, &__boltffi_endpoint_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint(receiver, (const uint8_t *)__boltffi_endpoint_ptr, (uintptr_t)__boltffi_endpoint_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1telemetry_1config_1set_1app_1version(JNIEnv *env, jclass cls, jlong receiver, jobject version, jint __boltffi_version_len) {
    (void)cls;

    void *__boltffi_version_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, version, (jlong)__boltffi_version_len, &__boltffi_version_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version(receiver, (const uint8_t *)__boltffi_version_ptr, (uintptr_t)__boltffi_version_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1telemetry_1config_1set_1device_1label(JNIEnv *env, jclass cls, jlong receiver, jobject label, jint __boltffi_label_len) {
    (void)cls;

    void *__boltffi_label_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, label, (jlong)__boltffi_label_len, &__boltffi_label_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label(receiver, (const uint8_t *)__boltffi_label_ptr, (uintptr_t)__boltffi_label_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1telemetry_1config_1set_1device_1attribute(JNIEnv *env, jclass cls, jlong receiver, jobject key, jint __boltffi_key_len, jobject value, jint __boltffi_value_len) {
    (void)cls;

    void *__boltffi_key_ptr = NULL;
    void *__boltffi_value_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, key, (jlong)__boltffi_key_len, &__boltffi_key_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, value, (jlong)__boltffi_value_len, &__boltffi_value_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute(receiver, (const uint8_t *)__boltffi_key_ptr, (uintptr_t)__boltffi_key_len, (const uint8_t *)__boltffi_value_ptr, (uintptr_t)__boltffi_value_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1telemetry_1config_1set_1batch_1size(JNIEnv *env, jclass cls, jlong receiver, jint batch_size) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size(receiver, batch_size);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1telemetry_1config_1set_1flush_1interval_1secs(JNIEnv *env, jclass cls, jlong receiver, jint secs) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs(receiver, secs);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1telemetry_1config_1init(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init(receiver);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1release_1class_1xybrid_1bolt_1xybrid_1bundle(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;

    (void)env;
    boltffi_release_class_xybrid_bolt_xybrid_bundle(handle);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1bundle_1open(JNIEnv *env, jclass cls, jobject path, jint __boltffi_path_len) {
    (void)cls;

    void *__boltffi_path_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, path, (jlong)__boltffi_path_len, &__boltffi_path_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_bundle_open((const uint8_t *)__boltffi_path_ptr, (uintptr_t)__boltffi_path_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1model_1id(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1version(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_bundle_version(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1target(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_bundle_target(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1hash(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_bundle_hash(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1has_1metadata(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1file_1count(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    uint32_t __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count(receiver);

    return (jint)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1file_1name(JNIEnv *env, jclass cls, jlong receiver, jint index) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name(receiver, index);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1manifest_1json(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json(receiver, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1metadata_1json(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json(receiver, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1bundle_1extract(JNIEnv *env, jclass cls, jlong receiver, jobject output_dir, jint __boltffi_output_dir_len) {
    (void)cls;

    void *__boltffi_output_dir_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, output_dir, (jlong)__boltffi_output_dir_len, &__boltffi_output_dir_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_bundle_extract(receiver, (const uint8_t *)__boltffi_output_dir_ptr, (uintptr_t)__boltffi_output_dir_len);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1tool_1results_1envelope(JNIEnv *env, jclass cls, jobject user_text, jint __boltffi_user_text_len, jobject prior_assistant_text, jint __boltffi_prior_assistant_text_len, jobject results, jint __boltffi_results_len) {
    (void)cls;

    void *__boltffi_user_text_ptr = NULL;
    void *__boltffi_prior_assistant_text_ptr = NULL;
    void *__boltffi_results_ptr = NULL;
    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    if (!boltffi_jni_direct_buffer_address(env, user_text, (jlong)__boltffi_user_text_len, &__boltffi_user_text_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, prior_assistant_text, (jlong)__boltffi_prior_assistant_text_len, &__boltffi_prior_assistant_text_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, results, (jlong)__boltffi_results_len, &__boltffi_results_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_function_xybrid_bolt_tool_results_envelope((const uint8_t *)__boltffi_user_text_ptr, (uintptr_t)__boltffi_user_text_len, (const uint8_t *)__boltffi_prior_assistant_text_ptr, (uintptr_t)__boltffi_prior_assistant_text_len, (const uint8_t *)__boltffi_results_ptr, (uintptr_t)__boltffi_results_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
__boltffi_error:
    return NULL;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1json_1schema_1to_1gbnf(JNIEnv *env, jclass cls, jobject schema_json, jint __boltffi_schema_json_len) {
    (void)cls;

    void *__boltffi_schema_json_ptr = NULL;
    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    if (!boltffi_jni_direct_buffer_address(env, schema_json, (jlong)__boltffi_schema_json_len, &__boltffi_schema_json_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_function_xybrid_bolt_json_schema_to_gbnf((const uint8_t *)__boltffi_schema_json_ptr, (uintptr_t)__boltffi_schema_json_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
__boltffi_error:
    return NULL;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1set_1thermal_1state(JNIEnv *env, jclass cls, jint state) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_set_thermal_state((___XybridThermalState)state);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1clear_1thermal_1state(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    boltffi_function_xybrid_bolt_clear_thermal_state();

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1set_1battery_1level(JNIEnv *env, jclass cls, jbyte percent) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_set_battery_level(percent);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1clear_1battery_1level(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    boltffi_function_xybrid_bolt_clear_battery_level();

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1configure_1runtime(JNIEnv *env, jclass cls, jobject api_key, jint __boltffi_api_key_len, jobject gateway_url, jint __boltffi_gateway_url_len, jobject ingest_url, jint __boltffi_ingest_url_len) {
    (void)cls;

    void *__boltffi_api_key_ptr = NULL;
    void *__boltffi_gateway_url_ptr = NULL;
    void *__boltffi_ingest_url_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, api_key, (jlong)__boltffi_api_key_len, &__boltffi_api_key_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, gateway_url, (jlong)__boltffi_gateway_url_len, &__boltffi_gateway_url_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, ingest_url, (jlong)__boltffi_ingest_url_len, &__boltffi_ingest_url_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_configure_runtime((const uint8_t *)__boltffi_api_key_ptr, (uintptr_t)__boltffi_api_key_len, (const uint8_t *)__boltffi_gateway_url_ptr, (uintptr_t)__boltffi_gateway_url_len, (const uint8_t *)__boltffi_ingest_url_ptr, (uintptr_t)__boltffi_ingest_url_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1init_1sdk_1cache_1dir(JNIEnv *env, jclass cls, jobject cache_dir, jint __boltffi_cache_dir_len) {
    (void)cls;

    void *__boltffi_cache_dir_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, cache_dir, (jlong)__boltffi_cache_dir_len, &__boltffi_cache_dir_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_init_sdk_cache_dir((const uint8_t *)__boltffi_cache_dir_ptr, (uintptr_t)__boltffi_cache_dir_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1set_1binding(JNIEnv *env, jclass cls, jobject binding, jint __boltffi_binding_len) {
    (void)cls;

    void *__boltffi_binding_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, binding, (jlong)__boltffi_binding_len, &__boltffi_binding_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_set_binding((const uint8_t *)__boltffi_binding_ptr, (uintptr_t)__boltffi_binding_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1set_1api_1key(JNIEnv *env, jclass cls, jobject api_key, jint __boltffi_api_key_len) {
    (void)cls;

    void *__boltffi_api_key_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, api_key, (jlong)__boltffi_api_key_len, &__boltffi_api_key_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_set_api_key((const uint8_t *)__boltffi_api_key_ptr, (uintptr_t)__boltffi_api_key_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1set_1provider_1api_1key(JNIEnv *env, jclass cls, jobject provider, jint __boltffi_provider_len, jobject api_key, jint __boltffi_api_key_len) {
    (void)cls;

    void *__boltffi_provider_ptr = NULL;
    void *__boltffi_api_key_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, provider, (jlong)__boltffi_provider_len, &__boltffi_provider_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, api_key, (jlong)__boltffi_api_key_len, &__boltffi_api_key_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_set_provider_api_key((const uint8_t *)__boltffi_provider_ptr, (uintptr_t)__boltffi_provider_len, (const uint8_t *)__boltffi_api_key_ptr, (uintptr_t)__boltffi_api_key_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1set_1platform_1url(JNIEnv *env, jclass cls, jobject url, jint __boltffi_url_len) {
    (void)cls;

    void *__boltffi_url_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, url, (jlong)__boltffi_url_len, &__boltffi_url_ptr)) {
        goto __boltffi_error;
    }

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_set_platform_url((const uint8_t *)__boltffi_url_ptr, (uintptr_t)__boltffi_url_len);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
__boltffi_error:
    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1set_1speculative_1cloud(JNIEnv *env, jclass cls, jboolean enabled) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_set_speculative_cloud(enabled);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1has_1api_1key(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_function_xybrid_bolt_has_api_key();

    return (jboolean)__boltffi_result;
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1is_1speculative_1cloud_1enabled(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_function_xybrid_bolt_is_speculative_cloud_enabled();

    return (jboolean)__boltffi_result;
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1will_1speculate_1for_1model(JNIEnv *env, jclass cls, jobject model_id, jint __boltffi_model_id_len) {
    (void)cls;

    void *__boltffi_model_id_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, model_id, (jlong)__boltffi_model_id_len, &__boltffi_model_id_ptr)) {
        goto __boltffi_error;
    }

    (void)env;
    bool __boltffi_result = boltffi_function_xybrid_bolt_will_speculate_for_model((const uint8_t *)__boltffi_model_id_ptr, (uintptr_t)__boltffi_model_id_len);

    return (jboolean)__boltffi_result;
__boltffi_error:
    return JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1version(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_function_xybrid_bolt_version();

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1release_1memory(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    uint32_t __boltffi_result = boltffi_function_xybrid_bolt_release_memory();

    return (jint)__boltffi_result;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1set_1auto_1release(JNIEnv *env, jclass cls, jboolean enabled) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_function_xybrid_bolt_set_auto_release(enabled);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1is_1auto_1release_1enabled(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_function_xybrid_bolt_is_auto_release_enabled();

    return (jboolean)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1telemetry_1default_1endpoint(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_function_xybrid_bolt_telemetry_default_endpoint();

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1telemetry_1flush(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    boltffi_function_xybrid_bolt_telemetry_flush();

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1telemetry_1shutdown(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    boltffi_function_xybrid_bolt_telemetry_shutdown();

    return;
}
