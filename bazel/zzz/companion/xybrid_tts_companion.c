#include "xybrid_tts_companion.h"
#include "companion_metadata.h"
#include <string.h>

/* This component alone uses API 29. Never satisfy this guard by overriding
 * macros against older stubs: its marked platform must select the API29 NDK. */
#if !defined(__ANDROID__) || !defined(__aarch64__) || !defined(__ANDROID_API__) || __ANDROID_API__ != 29
#error "The speech companion requires the separate Android ARM64 API29 toolchain"
#endif

_Static_assert(sizeof(void *) == 8, "ARM64 pointers required");
_Static_assert(ZZZ_EMBED_ABI_VERSION == 1, "engine ABI changed");
_Static_assert(XYBRID_TTS_ENGINE_ABI == 1, "metadata engine ABI changed");
_Static_assert(XYBRID_TTS_ANDROID_API == 29, "metadata OS floor changed");
_Static_assert(XYBRID_TTS_MODELS == 3, "selected model contract changed");
_Static_assert(offsetof(xybrid_tts_api_v1, contract_id) == 24, "identity offset");
_Static_assert(offsetof(xybrid_tts_api_v1, abi_version) == 56, "slot offset");
_Static_assert(sizeof(xybrid_tts_api_v1) == 120, "fixed ARM64 table size");
_Static_assert(sizeof(zzz_embed_error) == 196, "error layout");
_Static_assert(sizeof(zzz_embed_options) == 24, "options layout");
_Static_assert(sizeof(zzz_embed_result) == 24, "result layout");
_Static_assert(sizeof(zzz_embed_kitten_options) == 56, "Kitten layout");
_Static_assert(offsetof(zzz_embed_kitten_options, flags) == 12, "Kitten flags");
_Static_assert(offsetof(zzz_embed_kitten_options, seed) == 16, "Kitten seed");
_Static_assert(offsetof(zzz_embed_kitten_options, language_model_path) == 24, "Kitten paths");
_Static_assert(offsetof(zzz_embed_kitten_options, decoder_voice_path) == 48, "Kitten paths");

/* Constant relocations only, no constructor or load-time engine call. Exact
 * pointer assignments let the compiler check all eight engine signatures. */
static const xybrid_tts_api_v1 api = {
    .struct_size = sizeof(xybrid_tts_api_v1),
    .companion_abi = 1,
    .engine_abi = XYBRID_TTS_ENGINE_ABI,
    .android_api = XYBRID_TTS_ANDROID_API,
    .models = XYBRID_TTS_MODELS,
    .reserved = 0,
    .contract_id = XYBRID_TTS_CONTRACT_ID_BYTES,
    .abi_version = zzz_embed_abi_version,
    .supports_model = zzz_embed_supports_model,
    .default_options = zzz_embed_default_options,
    .open = zzz_embed_open,
    .open_kitten = zzz_embed_open_kitten,
    .synthesize = zzz_embed_synthesize,
    .cancel = zzz_embed_cancel,
    .close = zzz_embed_close,
};

int32_t xybrid_tts_get_api_v1(uint32_t requested_abi, uint32_t out_size,
    xybrid_tts_api_v1 *out) {
    if (out != NULL) {
        size_t clear_size = out_size < sizeof(*out) ? out_size : sizeof(*out);
        memset(out, 0, clear_size);
    }
    if (out == NULL || requested_abi != 1 || out_size != sizeof(*out)) {
        return -1;
    }
    memcpy(out, &api, sizeof(api));
    return 0;
}
