// xybrid's Core ML encoder bridge for whisper.cpp.
//
// Implements the C ABI whisper.cpp calls when built with WHISPER_USE_COREML
// (vendor/whisper-cpp/src/coreml/whisper-encoder.h), replacing upstream's
// src/coreml/whisper-encoder.mm, which this crate does not compile. Upstream's
// bridge ignores prediction errors and copies the encoder output into
// whisper.cpp's buffer without checking its size or element type. This one:
//
//  - vets the encoder against the context being initialised. Rust reads that
//    context's dimensions from the loaded model (not the file, which could be
//    replaced in between) and declares them, with the exact encoder path,
//    through xybrid_whisper_coreml_begin_load. Init declines an encoder whose
//    input differs, and every prediction's output is checked against those
//    dimensions before anything is copied. The encoder's output shape is not
//    part of its model description, so the per-prediction check is what stops
//    a mismatched `-encoder.mlmodelc` from writing past whisper.cpp's buffer.
//  - counts failed runs per thread and zero-fills the output, so a failure
//    never decodes a previous window's encoder state.
//  - drains every Objective-C temporary in an autorelease pool: the calling
//    thread is a Rust or tokio worker, which has no pool of its own.
//  - calls MLModel directly, so neither upstream's generated encoder class nor
//    its MLModel compat category (which replaces Core ML's async prediction
//    methods process-wide) is linked into the host app.
//  - requires iOS 16 (Float16 outputs, `getBytesWithHandler:`) and declines
//    below it, so the ggml encoder keeps working on older systems. Consumers
//    may target lower (the Flutter pod declares iOS 13), so every newer API is
//    behind an availability check; both builds compile this file with
//    -Werror=unguarded-availability-new to keep it that way.

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

// Reasons go to unified logging (Console, `log stream`); Rust only sees the
// status codes and failure counts.
os_log_t coreml_log() {
    static os_log_t log = os_log_create("ai.xybrid.whisper", "coreml");
    return log;
}

// Feature names fixed by upstream's converter (models/convert-whisper-to-coreml.py).
NSString *const kInputName = @"logmel_data";
NSString *const kOutputName = @"output";

// Far above any released Whisper (80/128 mels, 1500 frames, 1280 wide) and
// small enough that every size product below fits comfortably in 64 bits.
constexpr int32_t kMaxDimension = 1 << 16;

bool in_range(int32_t value) { return value > 0 && value <= kMaxDimension; }

// One begin_load / end_load bracket. Only an init for `encoder_path` may
// consume it, and only once.
struct Expectation {
    std::string encoder_path;
    int32_t n_mels;
    int32_t n_audio_ctx;
    int32_t n_audio_state;
    bool consumed = false;
    int32_t status = XYBRID_WHISPER_COREML_NOT_REQUESTED;
};

// A deque, not a vector: init holds a pointer to its claimed entry across the
// Core ML load, and push_back on a deque never moves existing elements, so a
// nested bracket opened meanwhile cannot leave that pointer dangling.
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

// Checks that `description` is a multi-array input of exactly
// [1, n_mels, n_frames] Float32, the only layout whisper.cpp hands over.
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

// Copies a prediction's output into whisper.cpp's buffer of `dst_count`
// floats, laid out frame by frame with `n_audio_state` values each. Rejects
// anything that does not fill it exactly: another size, a last dimension other
// than the model's state width, a non-contiguous layout, or an element type
// other than Float32 or Float16. Size products are checked as they grow, so
// no shape can overflow the arithmetic.
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
        // Row-major contiguous means each stride equals the element count of
        // the dimensions after it; a size-1 dimension's stride is irrelevant.
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
    // Out-of-range dimensions or no path leave an entry nothing can claim, so
    // the bracket still balances and the encoder simply stays off.
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
    // Only a declared load of exactly this encoder proceeds; anything else
    // keeps the ggml encoder without touching Core ML.
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
    // whisper.cpp passes the window's mel frame count as `n_ctx` and sizes
    // `out` as (n_ctx / 2) frames of the context's state width, which is the
    // width init accepted. Both factors are bounded (n_ctx by kMaxDimension
    // through the frame check below, or by whisper.cpp's own allocation), so
    // the product cannot overflow.
    const size_t frames = n_ctx > 0 ? static_cast<size_t>(n_ctx / 2) : 0;
    const size_t dst_count = frames * static_cast<size_t>(ctx->n_audio_state);

    bool ok = false;
    @autoreleasepool {
        // A truncated audio context arrives as fewer frames than the encoder
        // was converted for; Core ML would reject it, so refuse it here.
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
            // Init declined below iOS 16, so a context here implies it; the
            // check is what lets the compiler prove it.
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
