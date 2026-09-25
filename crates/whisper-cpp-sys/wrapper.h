/**
 * wrapper.h — umbrella header for bindgen.
 *
 * Unlike the llama.cpp side, whisper.cpp needs no first-party `_c` shim:
 * `whisper.h` is already plain C behind `extern "C"`, and every type our
 * callers touch is either an opaque struct pointer or a POD params struct.
 * This header exists only to give bindgen a single, stable entry point and to
 * fix the include order.
 *
 * The allowlist in `build.rs` (`whisper_.*` / `WHISPER_.*`) deliberately
 * excludes `ggml_*`: no `xybrid-whisper` or `xybrid-core` consumer references
 * a ggml symbol directly, and generating Rust definitions for ggml types here
 * would duplicate what `xybrid-llama-sys` already owns.
 */

#ifndef XYBRID_WHISPER_CPP_SYS_WRAPPER_H
#define XYBRID_WHISPER_CPP_SYS_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "whisper.h"

#endif /* XYBRID_WHISPER_CPP_SYS_WRAPPER_H */
