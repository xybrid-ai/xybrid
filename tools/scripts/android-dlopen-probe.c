// android-dlopen-probe — load a .so the way System.loadLibrary (dlopen,
// RTLD_NOW) does, print the dlerror() and exit non-zero on failure. Used by
// the build-android CI gate to prove the shipped libxybrid-bolt.so actually
// loads on a device/emulator — both as built and after an AGP-style strip.
//
// Why this exists: `llvm-readelf` reads the ELF via *section* headers, but the
// runtime loader reads via *program* headers, so a green readelf is not proof
// a lib loads. Only an on-device dlopen is. (This is the bug the bolt build
// works around: a patchelf'd .so passed readelf but bionic rejected the
// strip-corrupted copy with "empty/missing DT_HASH/DT_GNU_HASH".)
//
// Usage on device:
//   LD_LIBRARY_PATH=<dir> ./android-dlopen-probe <dir>/libxybrid-bolt.so
//
// RTLD_NOW forces eager symbol resolution, so an unresolved C++ ABI symbol
// (missing libc++_shared DT_NEEDED) fails here exactly as at app launch. The
// siblings (libc++_shared.so, libonnxruntime.so) must sit in LD_LIBRARY_PATH,
// mirroring the APK's per-ABI lib dir.
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path-to.so>\n", argv[0]);
        return 2;
    }
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (h == NULL) {
        // dlerror() can be NULL (no pending error / OOM); passing NULL to
        // %s is UB, so fall back to a literal.
        const char *err = dlerror();
        fprintf(stderr, "DLOPEN-FAIL: %s\n", err ? err : "unknown error");
        return 1;
    }
    fprintf(stdout, "DLOPEN-OK: %s\n", argv[1]);
    dlclose(h);
    return 0;
}
