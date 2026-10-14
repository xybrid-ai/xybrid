// xybrid's Core ML encoder bridge for whisper.cpp (WHISPER_USE_COREML),
// replacing upstream's src/coreml/whisper-encoder.mm. Unlike upstream, it:
//  - checks every output against the loaded model before copying, so a
//    mismatched -encoder.mlmodelc cannot write past whisper.cpp's buffer;
//  - zero-fills the output and counts the failure when a run fails;
//  - drains temporaries in autorelease pools (Rust threads have none);
//  - calls MLModel directly, without upstream's app-wide MLModel category;
//  - declines below iOS 16. The Flutter pod targets iOS 13, so newer APIs sit
//    behind availability checks, enforced by -Werror=unguarded-availability-new.

#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <os/log.h>

#include "coreml/whisper-encoder.h"
#include "xybrid-whisper-coreml.h"

#include <cstring>
#include <deque>
#include <string>

namespace {

// Failure reasons go to unified logging; Rust sees only status and counts.
os_log_t coreml_log() {
    static os_log_t log = os_log_create("ai.xybrid.whisper", "coreml");
    return log;
}

// Feature names fixed by upstream's converter (models/convert-whisper-to-coreml.py).
NSString *const kInputName = @"logmel_data";
NSString *const kOutputName = @"output";

// Above any released Whisper; keeps every size product within 64 bits.
constexpr int32_t kMaxDimension = 1 << 16;

bool in_range(int32_t value) { return value > 0 && value <= kMaxDimension; }

// One begin_load / end_load bracket; at most one init for its path claims it.
struct Expectation {
    std::string encoder_path;
    int32_t n_mels;
    int32_t n_audio_ctx;
    int32_t n_audio_state;
    bool consumed = false;
    int32_t status = XYBRID_WHISPER_COREML_NOT_REQUESTED;
};

// A deque: push_back never moves the entry init holds across the load.
thread_local std::deque<Expectation> t_loads;
thread_local uint64_t t_failures = 0;

// The innermost open declaration for `path`, if init has not consumed it.
Expectation *claim(const char *path) {
    for (auto it = t_loads.rbegin(); it != t_loads.rend(); ++it) {
        if (!it->consumed && it->encoder_path == path) {
            it->consumed = true;
            return &*it;
        }
    }
    return nullptr;
}

// Input must be [1, n_mels, n_frames] Float32, the layout whisper.cpp sends.
bool input_matches(MLFeatureDescription *description, int64_t n_mels, int64_t n_frames) {
    if (description == nil || description.type != MLFeatureTypeMultiArray) {
        return false;
    }
    MLMultiArrayConstraint *constraint = description.multiArrayConstraint;
    if (constraint.dataType != MLMultiArrayDataTypeFloat32) {
        return false;
    }
    NSArray<NSNumber *> *shape = constraint.shape;
    return shape.count == 3 && shape[0].longLongValue == 1 && shape[1].longLongValue == n_mels &&
           shape[2].longLongValue == n_frames;
}

// Copies the output into whisper.cpp's buffer only if it fills it exactly:
// `dst_count` contiguous Float32/Float16 values, last dimension n_audio_state.
API_AVAILABLE(ios(16.0))
bool copy_output(MLMultiArray *output, float *out, size_t dst_count, int32_t n_audio_state) {
    NSArray<NSNumber *> *shape = output.shape;
    NSArray<NSNumber *> *strides = output.strides;
    if (shape.count == 0 || shape.count != strides.count ||
        shape.lastObject.longLongValue != n_audio_state) {
        return false;
    }
    size_t count = 1;
    for (NSInteger i = (NSInteger)shape.count - 1; i >= 0; --i) {
        const long long dim = shape[(NSUInteger)i].longLongValue;
        if (dim <= 0) {
            return false;
        }
        // Contiguous: each stride equals the size of the dimensions after it.
        if (dim > 1 && strides[(NSUInteger)i].longLongValue != static_cast<long long>(count)) {
            return false;
        }
        if (__builtin_mul_overflow(count, static_cast<size_t>(dim), &count) || count > dst_count) {
            return false;
        }
    }
    if (count != dst_count) {
        return false;
    }

    const MLMultiArrayDataType type = output.dataType;
    __block bool copied = false;
    [output getBytesWithHandler:^(const void *bytes, NSInteger size) {
        if (size < 0) {
            return;
        }
        const size_t available = static_cast<size_t>(size);
        if (type == MLMultiArrayDataTypeFloat32 && available / sizeof(float) >= count) {
            std::memcpy(out, bytes, count * sizeof(float));
            copied = true;
        } else if (type == MLMultiArrayDataTypeFloat16 && available / sizeof(__fp16) >= count) {
            const __fp16 *src = static_cast<const __fp16 *>(bytes);
            for (size_t i = 0; i < count; ++i) {
                out[i] = static_cast<float>(src[i]);
            }
            copied = true;
        }
    }];
    return copied;
}

}  // namespace

struct whisper_coreml_context {
    MLModel *model;
    int64_t n_mels;
    int64_t n_frames;       // mel frames per window the encoder accepts
    int32_t n_audio_state;  // encoder output width, from the loaded context
};

extern "C" {

void xybrid_whisper_coreml_begin_load(const char *encoder_path, int32_t n_mels, int32_t n_audio_ctx,
                                      int32_t n_audio_state) {
    // Bad input still pushes an entry, pre-claimed, so the bracket balances.
    t_loads.push_back(Expectation{encoder_path != nullptr ? encoder_path : "", n_mels, n_audio_ctx,
                                  n_audio_state, encoder_path == nullptr ||
                                      !(in_range(n_mels) && in_range(n_audio_ctx) && in_range(n_audio_state))});
}

int32_t xybrid_whisper_coreml_end_load(void) {
    if (t_loads.empty()) {
        return XYBRID_WHISPER_COREML_NOT_REQUESTED;
    }
    const int32_t status = t_loads.back().status;
    t_loads.pop_back();
    return status;
}

uint64_t xybrid_whisper_coreml_failure_count(void) { return t_failures; }

struct whisper_coreml_context *whisper_coreml_init(const char *path_model) {
    if (path_model == nullptr) {
        return nullptr;
    }
    // Undeclared loads keep the ggml encoder without touching Core ML.
    Expectation *expected = claim(path_model);
    if (expected == nullptr) {
        return nullptr;
    }
    if (@available(iOS 16.0, *)) {
    } else {
        expected->status = XYBRID_WHISPER_COREML_UNSUPPORTED_OS;
        return nullptr;
    }

    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:path_model];
        if (path == nil) {
            expected->status = XYBRID_WHISPER_COREML_LOAD_FAILED;
            return nullptr;
        }
        MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
        configuration.computeUnits = MLComputeUnitsAll;
        NSError *error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                           configuration:configuration
                                                   error:&error];
        if (model == nil) {
            os_log_error(coreml_log(), "Core ML could not load %{public}@: %{public}@", path,
                         error.localizedDescription);
            expected->status = XYBRID_WHISPER_COREML_LOAD_FAILED;
            return nullptr;
        }

        const int64_t n_frames = 2 * static_cast<int64_t>(expected->n_audio_ctx);
        MLModelDescription *description = model.modelDescription;
        MLFeatureDescription *output = description.outputDescriptionsByName[kOutputName];
        if (!input_matches(description.inputDescriptionsByName[kInputName], expected->n_mels, n_frames) ||
            output == nil || output.type != MLFeatureTypeMultiArray) {
            os_log_error(coreml_log(),
                         "%{public}@ does not match its model (expected input [1, %d, %lld] Float32)",
                         path, expected->n_mels, (long long)n_frames);
            expected->status = XYBRID_WHISPER_COREML_MISMATCH;
            return nullptr;
        }

        expected->status = XYBRID_WHISPER_COREML_ACTIVE;
        return new whisper_coreml_context{model, expected->n_mels, n_frames, expected->n_audio_state};
    }
}

void whisper_coreml_free(struct whisper_coreml_context *ctx) {
    @autoreleasepool {
        // Releases the model inside the pool along with anything it autoreleases.
        delete ctx;
    }
}

void whisper_coreml_encode(const whisper_coreml_context *ctx, int64_t n_ctx, int64_t n_mel,
                           float *mel, float *out) {
    // `n_ctx` is the window's mel frames; whisper.cpp sizes `out` as
    // (n_ctx / 2) * n_audio_state floats. Both factors are bounded.
    const size_t frames = n_ctx > 0 ? static_cast<size_t>(n_ctx / 2) : 0;
    const size_t dst_count = frames * static_cast<size_t>(ctx->n_audio_state);

    bool ok = false;
    @autoreleasepool {
        // A truncated audio_ctx sends fewer frames than the encoder takes.
        if (n_mel == ctx->n_mels && n_ctx == ctx->n_frames) {
            NSError *error = nil;
            MLMultiArray *input = [[MLMultiArray alloc]
                initWithDataPointer:mel
                              shape:@[ @1, @(n_mel), @(n_ctx) ]
                           dataType:MLMultiArrayDataTypeFloat32
                            strides:@[ @(n_ctx * n_mel), @(n_ctx), @1 ]
                        deallocator:nil
                              error:&error];
            MLFeatureValue *value = input == nil ? nil : [MLFeatureValue featureValueWithMultiArray:input];
            MLDictionaryFeatureProvider *features =
                value == nil ? nil
                             : [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{kInputName : value}
                                                                                 error:&error];
            id<MLFeatureProvider> result =
                features == nil ? nil : [ctx->model predictionFromFeatures:features error:&error];
            MLMultiArray *output = [result featureValueForName:kOutputName].multiArrayValue;
            // Always true here (init declines below iOS 16); the compiler needs it.
            if (@available(iOS 16.0, *)) {
                ok = output != nil && copy_output(output, out, dst_count, ctx->n_audio_state);
            }
            if (!ok) {
                os_log_error(coreml_log(), "Core ML encoder run failed: %{public}@ (output %{public}@)",
                             error.localizedDescription ?: @"output does not match the model",
                             output.shape);
            }
        } else {
            os_log_error(coreml_log(),
                         "Core ML encoder takes [1, %lld, %lld] mel frames, got [1, %lld, %lld]",
                         (long long)ctx->n_mels, (long long)ctx->n_frames, (long long)n_mel,
                         (long long)n_ctx);
        }
    }

    if (!ok) {
        std::memset(out, 0, dst_count * sizeof(float));
        ++t_failures;
    }
}

}  // extern "C"
