# Phonon speech

Phonon runs locally through the optional `tts-zzz` backend on macOS ARM64,
Linux x86-64-v3 and Android ARM64. Standard prebuilt SDK artifacts do not
enable this feature; build a custom native SDK with the verified combined
engine archive described in [zzz-sys](../../crates/zzz-sys/README.md).
The engine has platform minimums of macOS 13, glibc 2.28 and Android API 29;
the final linked SDK can require a newer OS. Linux consumers need an
x86-64-v3 CPU. This backend does not support Windows, iOS or Metal.

Place `model_metadata.json` beside a converted Phonon GGUF:

```json
{
  "model_id": "phonon",
  "version": "1.0",
  "execution_template": {
    "type": "ZzzPhonon",
    "model_file": "phonon.gguf"
  },
  "files": ["phonon.gguf"],
  "metadata": { "task": "text-to-speech" }
}
```

The 202605 beta and Gradium `7e71a02d@200` bundles are supported. Omitting
`voice` uses the engine-selected bundled voice; the beta normally selects
`marlowe` and the newer bundle `freya`, with a bundled-voice fallback when a
conversion omits that speaker. Voice names must exist in the GGUF. External
voice files and voice cloning are outside this API.

Optional template fields are `voice`, `language`, `temperature`, `seed`,
`max_frames_per_chunk` and `max_chunk_tokens`. Omitted fields use native
defaults. Temperature `0` selects the bundle default (beta `0.4`, newer bundle
`0.3`); it does not request greedy sampling for this TTS backend. Positive
finite temperatures override it. Seed `0` is valid. Frame/token capacities
must be positive. Language defaults to English; Phonon also accepts `fr`,
`de`, `es` and `pt`.

```rust,ignore
use xybrid_sdk::{ModelLoader, RunOptions};
use xybrid_sdk::ir::{Envelope, EnvelopeKind};

let model = ModelLoader::from_directory("models/phonon")?.load()?;
let input = Envelope::new(EnvelopeKind::Text("Hello there.".into()));
let result = model.run(&input, None)?;
let pcm16_le = result.audio_bytes().expect("audio output");

let summary = model.run_tts_streaming(&input, &RunOptions::default(), |packet| {
    // Enqueue owned PCM16 LE audio for playback. first_sample is per channel.
    playback.enqueue(packet.pcm, packet.sample_rate, packet.channels);
    true
})?;
```

Request envelope metadata can override `voice_id`, `language`, `seed` and
`temperature`. An explicit `GenerationConfig.temperature` takes precedence
over temperature metadata, then the template, then native defaults. The same
rules apply to batch, `run_with_options`, `run_tts_streaming` and their async
forms. `GenerationConfig.max_tokens` is an LLM option, not a Phonon frame or
text-chunk capacity. These settings do not change Kitten's existing metadata.

Streaming delivers owned 24 kHz mono PCM16 packets during synthesis, without
packet normalization or fades. Returning `false` or cancelling the request's
`CancellationToken` stops delivery and inference. The summary distinguishes
`Completed`, `Cancelled` and `Limited`; limited audio is partial. Already
delivered packets cannot be retracted. Resource-abort and cloud-fallback
policies retain the existing SDK semantics.

Matching model, voice and open settings reuse one resident session. Language
is selected per utterance. A different voice or sampling configuration opens
a replacement session; later requests resolve their own settings again.
Returned errors and cancellation leave the model reusable. Keep mapped GGUF
files unchanged until the model is disposed. Bolt's existing `run_stream`
transport delivers the same packets and terminal metadata to language bindings.

Run local qualification against the existing model store with
`ZZZ_REQUIRE_FIXTURES=1` and the verified `XYBRID_ZZZ_PREBUILT_DIR`:

```sh
cargo test -p xybrid-zzz-sys --features bindings --test phonon -- --test-threads=1
cargo test -p xybrid-sdk --features tts-zzz --test zzz_phonon -- --test-threads=1
cargo test -p xybrid-ffi-facade --features xybrid-sdk/tts-zzz --test zzz_phonon -- --test-threads=1
```

Use `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0` and `CARGO_BUILD_JOBS=2` for bounded local builds.
Without local fixtures, these model tests print a skip reason; the required
fixture flag makes missing models fail qualification.
