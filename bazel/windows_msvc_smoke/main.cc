// Executable half of the Windows-MSVC smoke: proves the CRT startup path links,
// not just that objects compile.

#include <cstdio>

extern "C" unsigned int xybrid_msvc_smoke();

int main() {
    std::printf("xybrid_msvc_smoke=%u\n", xybrid_msvc_smoke());
    return 0;
}
