/*
 * A stand-in for the private zzz engine archive, for the isolation self-test
 * (BUILD.bazel here). Like the real one it defines the zzz_embed_* C API, its
 * own copies of libc and libm routines (memcpy weak, as a bundled runtime
 * often has it), a thread-local and a global variable and a constructor, and
 * imports one libc function (write). fake_engine_unused.c is its dead
 * second member. None of it is ever run.
 *
 * No libc header on purpose: bionic's fortified <string.h> defines memcpy
 * inline, which would clash with the definition below.
 */
#include <stddef.h>

extern long write(int fd, const void *buffer, size_t size);

__attribute__((weak)) void *memcpy(void *destination, const void *source, size_t size) {
  unsigned char *to = destination;
  const unsigned char *from = source;
  while (size--) *to++ = *from++;
  return destination;
}

size_t strlen(const char *text) {
  size_t length = 0;
  while (text[length]) length++;
  return length;
}

float expf(float x) { return 1.0f + x; }

_Thread_local int fake_engine_tls;
int fake_engine_counter = 1;

/* Not a constant store, or clang would fold the constructor away. */
__attribute__((constructor)) static void fake_engine_init(void) {
  fake_engine_counter += (int)write(2, "", 0);
}

unsigned zzz_embed_abi_version(void) { return (unsigned)fake_engine_counter; }

int zzz_embed_supports_model(const char *model) { return model != NULL && strlen(model) > 0; }

void zzz_embed_default_options(void *options) {
  static const char zero[16];
  memcpy(options, zero, sizeof zero);
}

int zzz_embed_open(const char *model, const char *path, const char *voice, const void *options,
                   void **session, void *error) {
  (void)model, (void)path, (void)voice, (void)options, (void)error;
  *session = NULL;
  return ++fake_engine_tls;
}

int zzz_embed_open_kitten(const void *options, void **session, void *error) {
  (void)options, (void)error;
  *session = NULL;
  return 0;
}

int zzz_embed_synthesize(void *session, const char *text, const char *language, void *callback,
                         void *user, void *result, void *error) {
  (void)session, (void)language, (void)callback, (void)user, (void)result, (void)error;
  return (int)expf((float)strlen(text));
}

void zzz_embed_cancel(void *session) { (void)session; }

void zzz_embed_close(void *session) {
  (void)session;
  fake_engine_tls = 0;
}
