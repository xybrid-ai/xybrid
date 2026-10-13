// xybrid's additions to whisper.cpp's Core ML encoder ABI.
//
// whisper.cpp built with WHISPER_USE_COREML calls three functions declared in
// vendor/whisper-cpp/src/coreml/whisper-encoder.h. This crate implements them
// in xybrid-whisper-coreml.mm instead of compiling upstream's bridge, and adds
// the hooks below so Rust can tell the bridge what the loaded GGML model
// expects and learn what happened. Everything here is per calling thread:
// whisper.cpp loads the encoder in whisper_init_state and runs it inside
// whisper_full*, on the calling thread.

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
#define XYBRID_WHISPER_COREML_UNSUPPORTED_OS 4 // older than iOS 16, which the bridge requires

// Declare, for the next whisper_init_state on this thread, the encoder path
// whisper.cpp will ask for and the dimensions of the context being
// initialised (read from the loaded context, not the file). Only an init for
// exactly `encoder_path` consumes it; without a matching declaration the
// bridge declines and whisper.cpp keeps its ggml encoder. Declarations nest
// (a stack per thread); pair every call with xybrid_whisper_coreml_end_load.
void xybrid_whisper_coreml_begin_load(const char *encoder_path, int32_t n_mels, int32_t n_audio_ctx,
                                      int32_t n_audio_state);

// Close the innermost begin_load on this thread and return its
// XYBRID_WHISPER_COREML_* outcome.
int32_t xybrid_whisper_coreml_end_load(void);

// Total encoder runs that have failed on this thread. Monotonic: compare the
// value before and after a whisper_full* call. A failed run zero-fills the
// encoder output, so a transcript produced across a failure must be discarded.
uint64_t xybrid_whisper_coreml_failure_count(void);

#ifdef __cplusplus
}
#endif
