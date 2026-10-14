// Hooks xybrid adds to whisper.cpp's Core ML encoder ABI (implemented in
// xybrid-whisper-coreml.mm). All state is per thread: whisper.cpp loads and
// runs the encoder on the calling thread.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Outcome of an encoder load bracketed by begin_load / end_load.
#define XYBRID_WHISPER_COREML_ACTIVE 0        // encoder loaded and matches the model
#define XYBRID_WHISPER_COREML_NOT_REQUESTED 1 // no init asked for this encoder
#define XYBRID_WHISPER_COREML_LOAD_FAILED 2   // Core ML could not load the encoder
#define XYBRID_WHISPER_COREML_MISMATCH 3      // encoder input does not match the model
#define XYBRID_WHISPER_COREML_UNSUPPORTED_OS 4 // below iOS 16

// Declares the encoder path and the loaded context's dimensions for the next
// whisper_init_state on this thread. Without it the bridge declines and
// whisper.cpp keeps its ggml encoder. Pair every call with end_load.
void xybrid_whisper_coreml_begin_load(const char *encoder_path, int32_t n_mels, int32_t n_audio_ctx,
                                      int32_t n_audio_state);

// Closes the innermost begin_load on this thread and returns its outcome.
int32_t xybrid_whisper_coreml_end_load(void);

// Failed encoder runs on this thread so far (never decreases). A failed run
// zero-fills the output, so discard any transcript produced across one.
uint64_t xybrid_whisper_coreml_failure_count(void);

#ifdef __cplusplus
}
#endif
