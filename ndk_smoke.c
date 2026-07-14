// NDK toolchain smoke test — must compile for aarch64-linux-android via the
// rules_android_ndk cc_toolchain (proves the Android C/C++ foundation works
// before wiring llama.cpp).
int xybrid_ndk_answer(void) { return 42; }
