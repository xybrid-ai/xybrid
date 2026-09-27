// Registers the fastest ggml CPU backend this device supports, for the arm64
// AAR, which ships llama.cpp's CPU variants as separate modules
// (GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS — see //:llama_android_dl).
//
// The variants sit next to libxybrid_bolt.so inside the APK. ggml's own search
// (ggml_backend_load_all) lists directories on disk, which cannot see the
// libraries an app keeps uncompressed in its APK, so this asks the dynamic
// linker for each variant by name. ggml_backend_load() skips a variant whose
// ggml_backend_score() reports a CPU feature this device lacks, so trying them
// from most to least capable registers the best supported one.
//
// Runs once, when libxybrid_bolt.so is loaded, so llama.cpp, whisper.cpp and
// mtmd all find a CPU backend in ggml's registry. If none loads (a packaging
// mistake), model loading fails with ggml's "no backends" error rather than
// crashing.

#include "ggml-backend.h"

namespace {

// Most to least capable. Keep in sync with ANDROID_ARM64_CPU_VARIANTS in
// bazel/ggml/android.bzl, which decides what the AAR ships.
constexpr const char * kCpuVariants[] = {
    "libggml-cpu-android_armv8.6_1.so",  // + int8 matmul (Armv9-era cores)
    "libggml-cpu-android_armv8.2_2.so",  // dotprod + fp16 (Cortex-A55/A75 on)
    "libggml-cpu-android_armv8.0_1.so",  // baseline: every arm64 device
};

__attribute__((constructor)) void register_cpu_backend() {
    for (const char * variant : kCpuVariants) {
        if (ggml_backend_load(variant) != nullptr) {
            return;
        }
    }
}

}  // namespace
