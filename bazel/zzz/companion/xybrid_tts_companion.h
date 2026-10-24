/* SDK-owned private transport. The verified engine header is a build input,
 * never a public companion payload. Keep this table identical to zzz-sys's
 * private repr(C) CompanionApiV1; no pointers may outlive the loaded library. */
#ifndef XYBRID_TTS_COMPANION_H
#define XYBRID_TTS_COMPANION_H
#include <stddef.h>
#include <stdint.h>
#include "zzz_embed.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xybrid_tts_api_v1 {
    uint32_t struct_size;
    uint32_t companion_abi;
    uint32_t engine_abi;
    uint32_t android_api;
    uint32_t models;
    uint32_t reserved;
    uint8_t contract_id[32];
    uint32_t (*abi_version)(void);
    int32_t (*supports_model)(const char *);
    void (*default_options)(zzz_embed_options *);
    int32_t (*open)(const char *, const char *, const char *,
        const zzz_embed_options *, zzz_embed_session **, zzz_embed_error *);
    int32_t (*open_kitten)(const zzz_embed_kitten_options *,
        zzz_embed_session **, zzz_embed_error *);
    int32_t (*synthesize)(zzz_embed_session *, const char *, const char *,
        zzz_embed_audio_callback, void *, zzz_embed_result *, zzz_embed_error *);
    void (*cancel)(zzz_embed_session *);
    void (*close)(zzz_embed_session *);
} xybrid_tts_api_v1;

/* Copies an exact ABI-1 table into aligned caller-owned writable storage.
 * Rejection zeroes min(out_size, sizeof(table)) bytes when out is nonnull;
 * no write exceeds the declared output size. No engine probe runs here.
 * Callers must validate all metadata/slots before calling any function. */
__attribute__((visibility("default")))
int32_t xybrid_tts_get_api_v1(uint32_t requested_abi, uint32_t out_size,
    xybrid_tts_api_v1 *out);

#ifdef __cplusplus
}
#endif
#endif
