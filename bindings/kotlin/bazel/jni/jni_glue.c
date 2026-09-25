#include <jni.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#if defined(__ANDROID__)
#include <pthread.h>
#endif

#include "xybrid_bolt.h"
static JavaVM *boltffi_jni_vm = NULL;
static jclass boltffi_jni_native_class = NULL;

#define BOLTFFI_JNI_LOCAL_FRAME_CAPACITY 64
static jint boltffi_jni_attach_current_thread(JavaVM *vm, JNIEnv **env) {
#if defined(__ANDROID__)
    return (*vm)->AttachCurrentThread(vm, env, NULL);
#else
    return (*vm)->AttachCurrentThread(vm, (void **)env, NULL);
#endif
}
#if defined(__ANDROID__)
static pthread_key_t boltffi_jni_env_key;
static pthread_once_t boltffi_jni_env_key_once = PTHREAD_ONCE_INIT;
static int boltffi_jni_env_key_status = 0;
static char boltffi_jni_tls_attached_marker;

static void boltffi_jni_android_env_destructor(void *value) {
    if (value != NULL && boltffi_jni_vm != NULL) {
        (*boltffi_jni_vm)->DetachCurrentThread(boltffi_jni_vm);
    }
}

static void boltffi_jni_android_env_key_init(void) {
    boltffi_jni_env_key_status =
        pthread_key_create(&boltffi_jni_env_key, boltffi_jni_android_env_destructor);
}

static jint boltffi_jni_android_attach_cached(JavaVM *vm, JNIEnv **env, int *attached) {
    *attached = 0;

    if (pthread_once(&boltffi_jni_env_key_once, boltffi_jni_android_env_key_init) != 0 ||
        boltffi_jni_env_key_status != 0) {
        jint result = boltffi_jni_attach_current_thread(vm, env);
        if (result == JNI_OK) {
            *attached = 1;
        }
        return result;
    }

    jint result = (*vm)->AttachCurrentThreadAsDaemon(vm, env, NULL);
    if (result != JNI_OK) {
        return result;
    }

    if (pthread_setspecific(boltffi_jni_env_key, &boltffi_jni_tls_attached_marker) != 0) {
        (*vm)->DetachCurrentThread(vm);
        *env = NULL;
        return JNI_ERR;
    }

    return JNI_OK;
}
#endif

static inline bool boltffi_jni_clear_exception(JNIEnv *env) {
    if (!(*env)->ExceptionCheck(env)) {
        return false;
    }
    (*env)->ExceptionClear(env);
    return true;
}

static void boltffi_jni_describe_load_exception(JNIEnv *env) {
    if ((*env)->ExceptionCheck(env)) {
        (*env)->ExceptionDescribe(env);
        (*env)->ExceptionClear(env);
    }
}

static bool boltffi_jni_report_class_load_failure(JNIEnv *env, const char *message, const char *diagnostic_class_name) {
    fprintf(stderr, "BoltFFI JNI_OnLoad failed: %s '%s'\n", message, diagnostic_class_name);
    boltffi_jni_describe_load_exception(env);
    return false;
}

static bool boltffi_jni_report_static_method_load_failure(JNIEnv *env, const char *diagnostic_class_name, const char *diagnostic_method_name, const char *diagnostic_signature) {
    fprintf(stderr, "BoltFFI JNI_OnLoad failed: could not resolve static method %s.%s%s\n", diagnostic_class_name, diagnostic_method_name, diagnostic_signature);
    boltffi_jni_describe_load_exception(env);
    return false;
}

static bool boltffi_jni_lookup_global_class_with_diagnostic(JNIEnv *env, const char *lookup_class_name, const char *diagnostic_class_name, jclass *out_class) {
    *out_class = NULL;
    jclass local_class = (*env)->FindClass(env, lookup_class_name);
    if (local_class == NULL) {
        return boltffi_jni_report_class_load_failure(env, "could not find JVM class", diagnostic_class_name);
    }
    jclass global_class = (*env)->NewGlobalRef(env, local_class);
    (*env)->DeleteLocalRef(env, local_class);
    if (global_class == NULL) {
        return boltffi_jni_report_class_load_failure(env, "could not create global reference for JVM class", diagnostic_class_name);
    }
    *out_class = global_class;
    return true;
}

static bool boltffi_jni_lookup_static_method_with_diagnostic(JNIEnv *env, jclass cls, const char *diagnostic_class_name, const char *lookup_method_name, const char *diagnostic_method_name, const char *lookup_signature, const char *diagnostic_signature, jmethodID *out_method) {
    *out_method = (*env)->GetStaticMethodID(env, cls, lookup_method_name, lookup_signature);
    if (*out_method == NULL) {
        return boltffi_jni_report_static_method_load_failure(env, diagnostic_class_name, diagnostic_method_name, diagnostic_signature);
    }
    return true;
}

static inline bool boltffi_jni_enter(JNIEnv **env, int *attached) {
    if (boltffi_jni_vm == NULL) {
        return false;
    }
    *env = NULL;
    *attached = 0;
    jint env_status = (*boltffi_jni_vm)->GetEnv(boltffi_jni_vm, (void **)env, JNI_VERSION_1_6);
    if (env_status == JNI_EDETACHED) {
#if defined(__ANDROID__)
        if (boltffi_jni_android_attach_cached(boltffi_jni_vm, env, attached) != JNI_OK) {
            return false;
        }
#else
        if (boltffi_jni_attach_current_thread(boltffi_jni_vm, env) != JNI_OK) {
            return false;
        }
        *attached = 1;
#endif
    } else if (env_status != JNI_OK) {
        return false;
    }

#if defined(__ANDROID__)
    JNIEnv *callback_env = *env;
    if ((*callback_env)->PushLocalFrame(callback_env, BOLTFFI_JNI_LOCAL_FRAME_CAPACITY) != JNI_OK) {
        boltffi_jni_clear_exception(callback_env);
        if (*attached) {
            (*boltffi_jni_vm)->DetachCurrentThread(boltffi_jni_vm);
            *attached = 0;
        }
        return false;
    }
#endif

    return true;
}

static inline void boltffi_jni_exit(JNIEnv *env, int attached) {
#if defined(__ANDROID__)
    if (env != NULL) {
        (*env)->PopLocalFrame(env, NULL);
        boltffi_jni_clear_exception(env);
    }
#else
    (void)env;
#endif
    if (attached) {
        (*boltffi_jni_vm)->DetachCurrentThread(boltffi_jni_vm);
    }
}

static jmethodID boltffi_jni_continuation_method = NULL;

static bool boltffi_jni_continuation_load(JNIEnv *env) {
    return boltffi_jni_lookup_static_method_with_diagnostic(env, boltffi_jni_native_class, "ai/xybrid/Native", "boltffiFutureContinuationCallback", "boltffiFutureContinuationCallback", "(JB)V", "(JB)V", &boltffi_jni_continuation_method);
}

static void boltffi_jni_continuation_unload(JNIEnv *env) {
    (void)env;
    boltffi_jni_continuation_method = NULL;
}

static void boltffi_jni_continuation_callback(uint64_t handle, int8_t poll_result) {
    if (boltffi_jni_vm == NULL || boltffi_jni_native_class == NULL || boltffi_jni_continuation_method == NULL) {
        return;
    }
    JNIEnv *env = NULL;
    int attached = 0;
    if (!boltffi_jni_enter(&env, &attached)) {
        return;
    }
    (*env)->CallStaticVoidMethod(env, boltffi_jni_native_class, boltffi_jni_continuation_method, (jlong)handle, (jbyte)poll_result);
    boltffi_jni_clear_exception(env);
    boltffi_jni_exit(env, attached);
}

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

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)reserved;
    JNIEnv *env = NULL;
    jint env_result = (*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6);
    if (env_result != JNI_OK) {
        fprintf(stderr, "BoltFFI JNI_OnLoad failed: GetEnv(JNI_VERSION_1_6) returned %d\n", (int)env_result);
        return JNI_ERR;
    }
    if (!boltffi_jni_lookup_global_class_with_diagnostic(env, "ai/xybrid/Native", "ai/xybrid/Native", &boltffi_jni_native_class)) {
        return JNI_ERR;
    }
    if (!boltffi_jni_continuation_load(env)) {
        (*env)->DeleteGlobalRef(env, boltffi_jni_native_class);
        boltffi_jni_native_class = NULL;
        return JNI_ERR;
    }
    boltffi_jni_vm = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    (void)reserved;
    JNIEnv *env = NULL;
    if ((*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6) == JNI_OK) {
        boltffi_jni_continuation_unload(env);
        if (boltffi_jni_native_class != NULL) {
            (*env)->DeleteGlobalRef(env, boltffi_jni_native_class);
        }
    }
    boltffi_jni_vm = NULL;
    boltffi_jni_native_class = NULL;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1release_1class_1xybrid_1bolt_1xybrid_1download(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;

    (void)env;
    boltffi_release_class_xybrid_bolt_xybrid_download(handle);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1download_1from_1registry(JNIEnv *env, jclass cls, jobject id, jint __boltffi_id_len) {
    (void)cls;

    void *__boltffi_id_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, id, (jlong)__boltffi_id_len, &__boltffi_id_ptr)) {
        goto __boltffi_error;
    }

    (void)env;
    uint64_t __boltffi_result = boltffi_init_class_xybrid_bolt_xybrid_download_from_registry((const uint8_t *)__boltffi_id_ptr, (uintptr_t)__boltffi_id_len);

    return (jlong)__boltffi_result;
__boltffi_error:
    return 0;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1download_1from_1registry_1with_1platform(JNIEnv *env, jclass cls, jobject id, jint __boltffi_id_len, jobject platform, jint __boltffi_platform_len) {
    (void)cls;

    void *__boltffi_id_ptr = NULL;
    void *__boltffi_platform_ptr = NULL;

    if (!boltffi_jni_direct_buffer_address(env, id, (jlong)__boltffi_id_len, &__boltffi_id_ptr)) {
        goto __boltffi_error;
    }
    if (!boltffi_jni_direct_buffer_address(env, platform, (jlong)__boltffi_platform_len, &__boltffi_platform_ptr)) {
        goto __boltffi_error;
    }

    (void)env;
    uint64_t __boltffi_result = boltffi_init_class_xybrid_bolt_xybrid_download_from_registry_with_platform((const uint8_t *)__boltffi_id_ptr, (uintptr_t)__boltffi_id_len, (const uint8_t *)__boltffi_platform_ptr, (uintptr_t)__boltffi_platform_len);

    return (jlong)__boltffi_result;
__boltffi_error:
    return 0;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1download_1status(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_download_status(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1download_1is_1finished(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_download_is_finished(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1download_1error(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_download_error(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1download_1cancel(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_download_cancel(receiver);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1release_1class_1xybrid_1bolt_1xybrid_1streaming_1session(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;

    (void)env;
    boltffi_release_class_xybrid_bolt_xybrid_streaming_session(handle);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1streaming_1session_1for_1model(JNIEnv *env, jclass cls, jlong model, jobject config, jint __boltffi_config_len) {
    (void)cls;

    void *__boltffi_config_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, config, (jlong)__boltffi_config_len, &__boltffi_config_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_streaming_session_for_model(model, (const uint8_t *)__boltffi_config_ptr, (uintptr_t)__boltffi_config_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1streaming_1session_1feed(JNIEnv *env, jclass cls, jlong receiver, jfloatArray samples) {
    (void)cls;

    jfloat *__boltffi_samples_ptr = NULL;
    jsize __boltffi_samples_len = 0;
    jfloat __boltffi_samples_stack[8];
    bool __boltffi_samples_needs_release = false;

    if (samples == NULL) {
        boltffi_jni_throw_illegal_argument(env, "BoltFFI array argument was null");
        goto __boltffi_error;
    }
    __boltffi_samples_len = (*env)->GetArrayLength(env, samples);
    if (__boltffi_samples_len <= (jsize)8) {
        (*env)->GetFloatArrayRegion(env, samples, 0, __boltffi_samples_len, __boltffi_samples_stack);
        if ((*env)->ExceptionCheck(env)) {
            goto __boltffi_error;
        }
        __boltffi_samples_ptr = __boltffi_samples_stack;
    } else {
        __boltffi_samples_ptr = (*env)->GetFloatArrayElements(env, samples, NULL);
        if (__boltffi_samples_ptr == NULL) {
            goto __boltffi_error;
        }
        __boltffi_samples_needs_release = true;
    }

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_streaming_session_feed(receiver, (const float *)__boltffi_samples_ptr, (uintptr_t)__boltffi_samples_len);

    if (__boltffi_samples_ptr != NULL) {
        if (__boltffi_samples_needs_release) {
            (*env)->ReleaseFloatArrayElements(env, samples, __boltffi_samples_ptr, JNI_ABORT);
        }
        __boltffi_samples_ptr = NULL;
    }
    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return;
    }

    return;
__boltffi_error:
    if (__boltffi_samples_ptr != NULL) {
        if (__boltffi_samples_needs_release) {
            (*env)->ReleaseFloatArrayElements(env, samples, __boltffi_samples_ptr, JNI_ABORT);
        }
        __boltffi_samples_ptr = NULL;
    }
    return;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1streaming_1session_1flush(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_streaming_session_flush(receiver, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1streaming_1session_1reset(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_streaming_session_reset(receiver);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return;
    }

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1streaming_1session_1cancel(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_streaming_session_cancel(receiver);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1streaming_1session_1is_1running(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_streaming_session_is_running(receiver);

    return (jboolean)__boltffi_result;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1release_1class_1xybrid_1bolt_1xybrid_1cancellation_1token(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;

    (void)env;
    boltffi_release_class_xybrid_bolt_xybrid_cancellation_token(handle);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1cancellation_1token_1new(JNIEnv *env, jclass cls) {
    (void)cls;

    (void)env;
    uint64_t __boltffi_result = boltffi_init_class_xybrid_bolt_xybrid_cancellation_token_new();

    return (jlong)__boltffi_result;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1cancellation_1token_1cancel(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    FfiStatus __boltffi_status = boltffi_method_class_xybrid_bolt_xybrid_cancellation_token_cancel(receiver);

    if (__boltffi_status.code != 0) {
        boltffi_jni_throw_status(env, __boltffi_status);
        return;
    }

    return;
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1cancellation_1token_1is_1cancelled(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    bool __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_cancellation_token_is_cancelled(receiver);

    return (jboolean)__boltffi_result;
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

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1run(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len, jobject options, jint __boltffi_options_len, jlong cancel) {
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

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_run(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len, (const uint8_t *)__boltffi_options_ptr, (uintptr_t)__boltffi_options_len, cancel, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
__boltffi_error:
    return NULL;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1run_1stream(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len, jobject options, jint __boltffi_options_len, jlong cancel) {
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

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_run_stream(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len, (const uint8_t *)__boltffi_options_ptr, (uintptr_t)__boltffi_options_len, cancel, &__boltffi_return);

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

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1run_1with_1context(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len, jlong context, jobject options, jint __boltffi_options_len, jlong cancel) {
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

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len, context, (const uint8_t *)__boltffi_options_ptr, (uintptr_t)__boltffi_options_len, cancel, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
__boltffi_error:
    return NULL;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1model_1run_1stream_1with_1context(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len, jlong context, jobject options, jint __boltffi_options_len, jlong cancel) {
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

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len, context, (const uint8_t *)__boltffi_options_ptr, (uintptr_t)__boltffi_options_len, cancel, &__boltffi_return);

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

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1release_1class_1xybrid_1bolt_1xybrid_1pipeline(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;

    (void)env;
    boltffi_release_class_xybrid_bolt_xybrid_pipeline(handle);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1pipeline_1from_1yaml(JNIEnv *env, jclass cls, jobject yaml, jint __boltffi_yaml_len) {
    (void)cls;

    void *__boltffi_yaml_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, yaml, (jlong)__boltffi_yaml_len, &__boltffi_yaml_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_pipeline_from_yaml((const uint8_t *)__boltffi_yaml_ptr, (uintptr_t)__boltffi_yaml_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1pipeline_1from_1file(JNIEnv *env, jclass cls, jobject path, jint __boltffi_path_len) {
    (void)cls;

    void *__boltffi_path_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, path, (jlong)__boltffi_path_len, &__boltffi_path_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_pipeline_from_file((const uint8_t *)__boltffi_path_ptr, (uintptr_t)__boltffi_path_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1init_1class_1xybrid_1bolt_1xybrid_1pipeline_1from_1bundle(JNIEnv *env, jclass cls, jobject path, jint __boltffi_path_len) {
    (void)cls;

    void *__boltffi_path_ptr = NULL;
    uint64_t __boltffi_return = (uint64_t){0};

    if (!boltffi_jni_direct_buffer_address(env, path, (jlong)__boltffi_path_len, &__boltffi_path_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_init_class_xybrid_bolt_xybrid_pipeline_from_bundle((const uint8_t *)__boltffi_path_ptr, (uintptr_t)__boltffi_path_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jlong)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1pipeline_1run(JNIEnv *env, jclass cls, jlong receiver, jobject envelope, jint __boltffi_envelope_len, jobject options, jint __boltffi_options_len) {
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

    FfiBuf_u8 error = boltffi_method_class_xybrid_bolt_xybrid_pipeline_run(receiver, (const uint8_t *)__boltffi_envelope_ptr, (uintptr_t)__boltffi_envelope_len, (const uint8_t *)__boltffi_options_ptr, (uintptr_t)__boltffi_options_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
__boltffi_error:
    return NULL;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1pipeline_1name(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_pipeline_name(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1pipeline_1stage_1names(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_pipeline_stage_names(receiver);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1method_1class_1xybrid_1bolt_1xybrid_1pipeline_1stage_1count(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    uint32_t __boltffi_result = boltffi_method_class_xybrid_bolt_xybrid_pipeline_stage_count(receiver);

    return (jint)__boltffi_result;
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

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1cache_1status(JNIEnv *env, jclass cls) {
    (void)cls;

    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    FfiBuf_u8 error = boltffi_function_xybrid_bolt_cache_status(&__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1cache_1entries(JNIEnv *env, jclass cls) {
    (void)cls;

    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    FfiBuf_u8 error = boltffi_function_xybrid_bolt_cache_entries(&__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
}

JNIEXPORT jboolean JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1cache_1is_1model_1cached(JNIEnv *env, jclass cls, jobject model_id, jint __boltffi_model_id_len) {
    (void)cls;

    void *__boltffi_model_id_ptr = NULL;
    bool __boltffi_return = (bool){0};

    if (!boltffi_jni_direct_buffer_address(env, model_id, (jlong)__boltffi_model_id_len, &__boltffi_model_id_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_function_xybrid_bolt_cache_is_model_cached((const uint8_t *)__boltffi_model_id_ptr, (uintptr_t)__boltffi_model_id_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return JNI_FALSE;
    }

    return (jboolean)__boltffi_return;
__boltffi_error:
    return JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1cache_1model_1path(JNIEnv *env, jclass cls, jobject model_id, jint __boltffi_model_id_len) {
    (void)cls;

    void *__boltffi_model_id_ptr = NULL;
    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    if (!boltffi_jni_direct_buffer_address(env, model_id, (jlong)__boltffi_model_id_len, &__boltffi_model_id_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_function_xybrid_bolt_cache_model_path((const uint8_t *)__boltffi_model_id_ptr, (uintptr_t)__boltffi_model_id_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
__boltffi_error:
    return NULL;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1cache_1list_1extracted_1model_1ids(JNIEnv *env, jclass cls) {
    (void)cls;

    FfiBuf_u8 __boltffi_return = (FfiBuf_u8){0};

    FfiBuf_u8 error = boltffi_function_xybrid_bolt_cache_list_extracted_model_ids(&__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return NULL;
    }

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_return);
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1cache_1remove_1model(JNIEnv *env, jclass cls, jobject model_id, jint __boltffi_model_id_len) {
    (void)cls;

    void *__boltffi_model_id_ptr = NULL;
    uint32_t __boltffi_return = (uint32_t){0};

    if (!boltffi_jni_direct_buffer_address(env, model_id, (jlong)__boltffi_model_id_len, &__boltffi_model_id_ptr)) {
        goto __boltffi_error;
    }

    FfiBuf_u8 error = boltffi_function_xybrid_bolt_cache_remove_model((const uint8_t *)__boltffi_model_id_ptr, (uintptr_t)__boltffi_model_id_len, &__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jint)__boltffi_return;
__boltffi_error:
    return 0;
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1function_1xybrid_1bolt_1cache_1clear(JNIEnv *env, jclass cls) {
    (void)cls;

    uint32_t __boltffi_return = (uint32_t){0};

    FfiBuf_u8 error = boltffi_function_xybrid_bolt_cache_clear(&__boltffi_return);

    if (error.ptr != NULL || error.len != 0) {
        boltffi_jni_throw_error_buffer(env, error);
        return 0;
    }

    return (jint)__boltffi_return;
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

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1download_1progress_1subscribe(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    uint64_t __boltffi_result = boltffi_stream_xybrid_bolt_xybrid_download_progress_subscribe(receiver);

    return (jlong)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1download_1progress_1pop_1batch(JNIEnv *env, jclass cls, jlong subscription, jlong max_count) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_stream_xybrid_bolt_xybrid_download_progress_pop_batch(subscription, max_count);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1download_1progress_1wait(JNIEnv *env, jclass cls, jlong subscription, jint timeout_milliseconds) {
    (void)cls;

    (void)env;
    WaitResult __boltffi_result = boltffi_stream_xybrid_bolt_xybrid_download_progress_wait(subscription, timeout_milliseconds);

    return (jint)__boltffi_result;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1download_1progress_1poll(JNIEnv *env, jclass cls, jlong subscription, jlong callback_data) {
    (void)cls;

    (void)env;
    boltffi_stream_xybrid_bolt_xybrid_download_progress_poll(subscription, callback_data, boltffi_jni_continuation_callback);

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1download_1progress_1unsubscribe(JNIEnv *env, jclass cls, jlong subscription) {
    (void)cls;

    (void)env;
    boltffi_stream_xybrid_bolt_xybrid_download_progress_unsubscribe(subscription);

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1download_1progress_1free(JNIEnv *env, jclass cls, jlong subscription) {
    (void)cls;

    (void)env;
    boltffi_stream_xybrid_bolt_xybrid_download_progress_free(subscription);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1streaming_1session_1partials_1subscribe(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    uint64_t __boltffi_result = boltffi_stream_xybrid_bolt_xybrid_streaming_session_partials_subscribe(receiver);

    return (jlong)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1streaming_1session_1partials_1pop_1batch(JNIEnv *env, jclass cls, jlong subscription, jlong max_count) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_stream_xybrid_bolt_xybrid_streaming_session_partials_pop_batch(subscription, max_count);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1streaming_1session_1partials_1wait(JNIEnv *env, jclass cls, jlong subscription, jint timeout_milliseconds) {
    (void)cls;

    (void)env;
    WaitResult __boltffi_result = boltffi_stream_xybrid_bolt_xybrid_streaming_session_partials_wait(subscription, timeout_milliseconds);

    return (jint)__boltffi_result;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1streaming_1session_1partials_1poll(JNIEnv *env, jclass cls, jlong subscription, jlong callback_data) {
    (void)cls;

    (void)env;
    boltffi_stream_xybrid_bolt_xybrid_streaming_session_partials_poll(subscription, callback_data, boltffi_jni_continuation_callback);

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1streaming_1session_1partials_1unsubscribe(JNIEnv *env, jclass cls, jlong subscription) {
    (void)cls;

    (void)env;
    boltffi_stream_xybrid_bolt_xybrid_streaming_session_partials_unsubscribe(subscription);

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1streaming_1session_1partials_1free(JNIEnv *env, jclass cls, jlong subscription) {
    (void)cls;

    (void)env;
    boltffi_stream_xybrid_bolt_xybrid_streaming_session_partials_free(subscription);

    return;
}

JNIEXPORT jlong JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1model_1download_1progress_1subscribe(JNIEnv *env, jclass cls, jlong receiver) {
    (void)cls;

    (void)env;
    uint64_t __boltffi_result = boltffi_stream_xybrid_bolt_xybrid_model_download_progress_subscribe(receiver);

    return (jlong)__boltffi_result;
}

JNIEXPORT jbyteArray JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1model_1download_1progress_1pop_1batch(JNIEnv *env, jclass cls, jlong subscription, jlong max_count) {
    (void)cls;

    (void)env;
    FfiBuf_u8 __boltffi_result = boltffi_stream_xybrid_bolt_xybrid_model_download_progress_pop_batch(subscription, max_count);

    return boltffi_jni_buffer_to_byte_array(env, __boltffi_result);
}

JNIEXPORT jint JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1model_1download_1progress_1wait(JNIEnv *env, jclass cls, jlong subscription, jint timeout_milliseconds) {
    (void)cls;

    (void)env;
    WaitResult __boltffi_result = boltffi_stream_xybrid_bolt_xybrid_model_download_progress_wait(subscription, timeout_milliseconds);

    return (jint)__boltffi_result;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1model_1download_1progress_1poll(JNIEnv *env, jclass cls, jlong subscription, jlong callback_data) {
    (void)cls;

    (void)env;
    boltffi_stream_xybrid_bolt_xybrid_model_download_progress_poll(subscription, callback_data, boltffi_jni_continuation_callback);

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1model_1download_1progress_1unsubscribe(JNIEnv *env, jclass cls, jlong subscription) {
    (void)cls;

    (void)env;
    boltffi_stream_xybrid_bolt_xybrid_model_download_progress_unsubscribe(subscription);

    return;
}

JNIEXPORT void JNICALL Java_ai_xybrid_Native_boltffi_1stream_1xybrid_1bolt_1xybrid_1model_1download_1progress_1free(JNIEnv *env, jclass cls, jlong subscription) {
    (void)cls;

    (void)env;
    boltffi_stream_xybrid_bolt_xybrid_model_download_progress_free(subscription);

    return;
}
