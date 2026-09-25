"""The llama.cpp libraries the Android arm64 AAR ships (see //:llama_android_dl).

Shared between the //:llama_android_dl build (which declares them as outputs)
and //bindings/kotlin (which links and packages them).
"""

# The shared libraries libxybrid_bolt.so links against (DT_NEEDED).
ANDROID_ARM64_GGML_LIBS = [
    "libllama.so",
    "libggml.so",
    "libggml-base.so",
    "libmtmd.so",
]

# The CPU variants the AAR ships, of the seven ggml builds for Android: the
# baseline every arm64 device runs, dotprod+fp16 (Cortex-A55/A75 and later)
# and i8mm (Armv9-era cores). The armv9.0 (SVE2) build measured slower than
# armv8.6 on a Pixel 8 yet would outrank it, and the SME builds only pay off
# with KleidiAI kernels. Keep in sync with the list in
# bindings/kotlin/bazel/jni/ggml_cpu_backend.cpp.
ANDROID_ARM64_CPU_VARIANTS = [
    "libggml-cpu-android_armv8.0_1.so",
    "libggml-cpu-android_armv8.2_2.so",
    "libggml-cpu-android_armv8.6_1.so",
]
