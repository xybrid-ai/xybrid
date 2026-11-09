/*
 * The SDK side of the isolation self-test: like xybrid's own code in
 * libxybrid_bolt.so, it calls libc, libm and the engine's C API. After
 * isolation memcpy, strlen and expf must come from bionic, not from the fake
 * engine linked into the same library. No libc header, as in fake_engine.c;
 * -fno-builtin keeps every call a call.
 */
#include <stddef.h>

void *memcpy(void *destination, const void *source, size_t size);
size_t strlen(const char *text);
float expf(float x);
unsigned zzz_embed_abi_version(void);

int selftest_host(char *destination, const char *source, float x) {
  memcpy(destination, source, strlen(source) + 1);
  return (int)expf(x) + (int)zzz_embed_abi_version();
}
