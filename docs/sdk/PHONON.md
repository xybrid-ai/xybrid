# Phonon speech

Phonon runs locally through the optional `tts-zzz` backend on macOS ARM64,
Linux x86-64-v3 and Android ARM64. Standard prebuilt SDK artifacts do not
enable this feature; build a custom native SDK with the verified combined
engine archive described in [zzz-sys](../../crates/zzz-sys/README.md).
The engine has platform minimums of macOS 13, glibc 2.28 and Android API 29;
the final linked SDK can require a newer OS. Linux consumers need an
x86-64-v3 CPU. This backend does not support Windows, iOS or Metal.

## Checkpoints

Use **`phonon-31m-en`**, Gradium's public English checkpoint
([`gradium/phonon-31m-en`](https://huggingface.co/gradium/phonon-31m-en),
Apache-2.0). The two older bundles were earlier private previews; they still
load.

| Bundle | Status | Default voice | Default temperature |
|---|---|---|---|
| `phonon-31m-en` | Recommended; public, Apache-2.0 | `default` | 0.3 |
| Gradium `7e71a02d@200` | Earlier private preview | `freya` | 0.3 |
| 202605 beta | Earlier private preview | `marlowe` | 0.4 |

`phonon-31m-en` needs the zzz engine pinned by this SDK version,
`v0.1.0-rc.8` ([natives-manifest.json](../../crates/zzz-sys/natives-manifest.json)).
The SDK stages the `phonon-kitten-tts2` profile, which contains only Phonon
and Kitten TTS 2 and exposes the unchanged `zzz_embed` ABI. The
`v0.1.0-rc.6` engine pinned by xybrid 0.12.0 rejects the public checkpoint.
`tts-zzz` builds of this SDK version advertise the `zzz-phonon-31m` registry backend, which is
how the registry tells them apart from 0.12.0 builds (see
[registry telemetry](../telemetry/registry.md#backends-values)).

All three bundles produce 24 kHz mono audio. Gradium's own files
(`model.q8.gguf`, the tokenizer and the voice safetensors) do not load
directly: the engine reads one self-contained GGUF that holds the weights,
tokenizer and voices.

### Getting phonon-31m-en

- **From the xybrid registry**, once it is registered there, as model id
  `phonon-31m-en`. The registry serves it on the three platforms above, and
  only to builds that advertise `zzz-phonon-31m`; any other build gets a
  not-found error.

  ```rust,ignore
  let model = ModelLoader::from_registry("phonon-31m-en").load()?;
  ```

- **Bring your own converted bundle.** Place it in a directory beside a
  `model_metadata.json` (below) and load it with
  `ModelLoader::from_directory`. The reference conversion is
  `Phonon-31m-en.gguf`, 74,446,144 bytes, SHA-256
  `749244c33060d81af3ea5b66bef627bf9fdd05d6f2a2c86dfcb943dc4b63e2e6`, built
  from `gradium/phonon-31m-en` at revision
  `52433c46975bfa13144c56204c9f4ee1c5b148aa` and `gradium/phonon-voices` at
  revision `da6c9ec0d47be8df5c87c730a3aa97187f3229c3`.

```json
{
  "model_id": "phonon-31m-en",
  "version": "1.0",
  "execution_template": {
    "type": "ZzzPhonon",
    "model_file": "Phonon-31m-en.gguf"
  },
  "files": ["Phonon-31m-en.gguf"],
  "metadata": { "task": "text-to-speech" }
}
```

### Voices

`phonon-31m-en` carries 38 voices. `default` is the checkpoint's own default
voice, used when you omit `voice`. The other 37 are the English voices of
Gradium's [`gradium/phonon-voices`](https://huggingface.co/gradium/phonon-voices)
catalog (Apache-2.0). Their names are the catalog's display names folded to
ASCII and lowercased, with each run of other characters replaced by `_`:
"Captain Silas Blackwater" is `captain_silas_blackwater` and "Elodie-Rose" is
`elodie_rose`. Gradium's own runtime names voices differently: the
`en_us_Easton` of its quickstart is `easton` here.

| Catalog group | Voices |
|---|---|
| US English (`en-us`) | female: `delia`, `everly`, `hadley`, `harper`, `juniper`, `zoey`; male: `easton`, `micah`, `miles`, `preston`, `russell`, `trevor` |
| British English (`en-gb`) | female: `bryony`, `elodie_rose`, `freya`, `imogen`, `lottie`, `maeve`, `pippa`; male: `alfie`, `edwin`, `hugh`, `spencer`, `teddy`, `toby` |
| Characters | `azgorath`, `captain_silas_blackwater`, `fairy_godmother`, `fennick_the_young_elf`, `grub_the_ogre`, `master_aldric`, `morwenna_of_the_deep`, `old_mortlake`, `sir_bramwell`, `the_snow_queen`, `the_thing_under_the_stairs`, `vyrmathrax_the_ancient` |

For the previews, omitting `voice` selects `marlowe` (beta) or `freya`
(`7e71a02d@200`), with a bundled-voice fallback when a conversion omits that
speaker. Voice names must exist in the GGUF; the SDK cannot list them.
External voice files and voice cloning are outside this API.

## Options

Optional template fields are `voice`, `language`, `temperature`, `seed`,
`max_frames_per_chunk` and `max_chunk_tokens`. Omitted fields use native
defaults. Temperature `0` selects the bundle default (see the table above); it
does not request greedy sampling for this TTS backend. Positive finite
temperatures override it. Seed `0` is valid. Frame/token capacities must be
positive.

Language defaults to English; the engine also accepts `fr`, `de`, `es` and
`pt`. The language selects how text is normalized (numbers, symbols, dates),
not the pronunciation. `phonon-31m-en` speaks English only, so keep `en`:
text in other languages is read with English pronunciation.

The checkpoint's speaking-rate condition (xn-ptts' `duration_delta`) is not
exposed through the SDK; synthesis uses the bundle's value.

```rust,ignore
use xybrid_sdk::{ModelLoader, RunOptions};
use xybrid_sdk::ir::{Envelope, EnvelopeKind};

let model = ModelLoader::from_directory("models/phonon-31m-en")?.load()?;
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

## Streaming and sessions

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

## Attribution and responsible use

`phonon-31m-en` is by [Gradium](https://huggingface.co/gradium). Phonon
builds on Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
and the Mimi codec. Its weights, tokenizer and voices, and the
`gradium/phonon-voices` catalog, are released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The zzz
bundle repackages those files into one GGUF for the zzz engine. Gradium's
[model card](https://huggingface.co/gradium/phonon-31m-en) covers its
benchmarks, intended use and limitations.

Gradium asks that Phonon not be used to impersonate a real person, to deceive
listeners about who is speaking, or to clone a voice without the speaker's
consent. Tell listeners when speech is synthetic where that matters to them.

## Local qualification

Run local qualification against the existing model store with
`ZZZ_REQUIRE_FIXTURES=1` and the verified `XYBRID_ZZZ_PREBUILT_DIR`. The
tests read `Phonon-31m-en.gguf`, `Phonon-7e71a02d-200.gguf` and
`Phonon-202605-beta1.gguf` from `~/.zzz/models`:

```sh
cargo test -p xybrid-zzz-sys --features bindings --test phonon -- --test-threads=1
cargo test -p xybrid-sdk --features tts-zzz --test zzz_phonon -- --test-threads=1
cargo test -p xybrid-ffi-facade --features xybrid-sdk/tts-zzz --test zzz_phonon -- --test-threads=1
```

Use `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0` and `CARGO_BUILD_JOBS=2` for bounded local builds.
Without local fixtures, these model tests print a skip reason; the required
fixture flag makes missing models fail qualification.
