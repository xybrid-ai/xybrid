// Smoke source for the Windows-MSVC toolchain (//bazel/toolchains/windows_msvc).
//
// Deliberately pulls the three header families that a real target needs and
// that a MinGW-flavoured toolchain would satisfy from the wrong place: the
// Windows SDK (windows.h), the MSVC C++ standard library (string/vector), and
// clang's own builtin resource-dir headers (immintrin.h).

#include <windows.h>

#include <immintrin.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

std::string ProcessIdText() {
    std::vector<char> buffer(32, '\0');
    const DWORD pid = GetCurrentProcessId();
    std::snprintf(buffer.data(), buffer.size(), "%lu", static_cast<unsigned long>(pid));
    return std::string(buffer.data());
}

}  // namespace

extern "C" __declspec(dllexport) unsigned int xybrid_msvc_smoke() {
    // Exercise the STL (heap allocation through the MSVC CRT) and an intrinsic
    // so a missing compiler-rt builtin or a mis-selected CRT fails here.
    const std::string text = ProcessIdText();
    const __m128i lanes = _mm_set1_epi32(static_cast<int>(text.size()));
    return static_cast<unsigned int>(_mm_cvtsi128_si32(lanes));
}
