# KittenTTS 2 rc.4 local SDK integration

Kitten runs through `ModelLoader.FromDirectory`, the Bolt plugin, and the existing
Unity audio API. The plugin must be built with `tts-zzz`; installing these
managed sources alone does not enable Kitten in an existing plugin.

## Assets and metadata

Copy `model_metadata.json` from this folder beside these four files. Use
`accelerate: true` only on macOS. Six threads plus sentence mode is a demo
starting point; measure on the target device. Changing options requires retiring
the previous model/worker before loading a new directory/session.

| File | Bytes | SHA-256 |
|---|---:|---|
| `kitten-tts-2-q2_0.gguf` | 1051096928 | `66c8d5ad02b4dbb0c7b3af59de218ea6376b4e0943bf9791807c3c2400864dd6` |
| `kitten-s3-meanflow-f32.gguf` | 541920128 | `005a5c871d84f91fd222b03cc58eb05d9761d058df5740ce182625599267676b` |
| `bruno.lm.json` | | `e5810ec8720b79be598f43ccfbfac32ab06df070191e3afe17ee628640d5a319` |
| `bruno.s3.json` | | `69be324861eb751fa80f76288e73cd09fbf666522dc287fe2fd19826beae33a3` |

Language model: `KittenML/kitten-tts-2` revision
`baa41e5d2c5f64be0095365672a7858542261271`, converted Q2_0 with rc.4 scripts.
Decoder: `ResembleAI/chatterbox-turbo` revision
`749d1c1a46eb10492095d68fbcf55691ccf137cd`, converted F32 meanflow.
Voices: pinned zzz fixtures `bruno-short.json` and `bruno-voice.json`, renamed as
above. Keep all files unchanged until the model is disposed; the engine maps
weights and reuses resident voice prefixes. Live cloning is a follow-up.

`max_tokens: 0` uses available LM-window room with the engine's split-chunk
runaway guard. It does **not** mean 128. Explicit caps remain supported and can
return partial `Limited` audio. Pass expression markup intact in `Envelope.Text`.
Xybrid does not split or fade each native packet. Sentence mode computes an
entire sentence waveform before emitting it; packets are not incremental
waveform decoding of speech tokens.

## Build the local plugin

Native artifacts use Bazel. Private engine slices are verified by `zzz_pull.py`
against the rc.4 manifest, receipt, source revision, ABI, profile and target.
Keep engine archives, receipts and staging directories outside public artifacts
and caches. Engine source is
`xybrid-ai/zzz-research@56385c4a3fbab8bc503341947af37e96ebb0071d`.

```bash
# Python 3.11+, authenticated access to the pinned release as needed.
python3 tools/scripts/zzz_pull.py --target aarch64-apple-darwin \
  --archive /private/path/zzz-kitten-tts2-macos-arm64-v0.1.0-rc.4.tar.gz \
  > /tmp/xybrid-kitten-slice.txt
bazel build -c opt --config=macos --//:tts_zzz=true \
  --repo_env=XYBRID_ZZZ_PREBUILT_DIR="$(cat /tmp/xybrid-kitten-slice.txt)" \
  --remote_cache= --remote_executor= \
  //crates/xybrid-bolt:xybrid_bolt_cdylib
python3 tools/scripts/stage_unity_native.py \
  --lib bazel-bin/crates/xybrid-bolt/libxybrid_bolt.dylib \
  --target aarch64-apple-darwin \
  --plugins-root /private/path/KittenSdk/Plugins
```

For Linux use target `x86_64-unknown-linux-gnu`, its corresponding archive,
remove `--config=macos`, and stage `libxybrid_bolt.so`. If Python 3.11 is not the
host `python3`, pass `--repo_env=XYBRID_ZZZ_PYTHON=/usr/bin/python3.11`.
When cross-building from Linux, Bazel can resolve both host and target slices.
Point the repo environment variable at a private directory containing verified
slice directories named `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin` (and
`aarch64-linux-android` if resolved). Copy verified directories with private
permissions; symlinks inside staged slices are rejected. This is a grouping
root for Bazel; Cargo still takes one exact verified slice.

Linux also needs the pinned ORT library for the existing Kokoro route:

```bash
python3 tools/scripts/stage_unity_desktop_ort.py linux \
  /private/path/KittenSdk/Plugins/Linux
```

The macOS plugin links its pinned ONNX Runtime statically.

The engine slices support macOS ARM64 13+, Linux x86-64-v3/glibc 2.28+, and
Android ARM64 API 29+. The complete cross-built macOS SDK plugin currently has
`LC_BUILD_VERSION minos 14.0`; use macOS 14+ for that artifact and set the Unity
player minimum accordingly. Inspect the linked plugin rather than deriving its
minimum from the engine slice. Android's verified slice needs separate device
validation. This adds no Windows/iOS/Metal Kitten support.

## Install in a Unity project

Close Unity before replacing a loaded plugin. Point `Packages/manifest.json`
at this checkout's managed package (or a copied `bindings/unity` folder):

```json
"ai.xybrid.sdk": "file:/absolute/path/to/xybrid/bindings/unity"
```

Stage/copy the plugin tree into your project's `Assets/Xybrid/Plugins`.
Remove or park
old duplicate Xybrid plugins outside `Assets` and the package first; one copy
must be enabled for the target. Preserve the generated `.meta` files. Write
`Assets/Xybrid/Plugins/.xybrid-native-macos-version` containing `0.10.1` (the
managed package version), or the analogous `linux` marker. The existing editor
resolver then retains the local plugin. Its forced Download menu replaces it
with release natives, so restage the local plugin/marker afterward if used.
Do not commit local native binaries or model files.

Stage the two GGUFs and the two prepared Bruno JSONs into a readable model
directory with the provided metadata. Set `accelerate: true` on macOS.
Connect the provider to your application's dialogue and playback code;
buffering, avatar animation and provider UI remain application-owned.

## Provider contract

```csharp
TtsStreamResult Model.RunTtsStreaming(
    Envelope envelope, Action<TtsAudioChunk> onAudio = null,
    CancellationToken cancellationToken = default);
Task<TtsStreamResult> Model.RunTtsStreamingAsync(
    Envelope envelope, Action<TtsAudioChunk> onAudio = null,
    CancellationToken cancellationToken = default);
```

Packets expose `byte[] Pcm`, `uint SampleRate`, `uint Channels`,
`ulong FirstSample`. PCM is owned, raw signed PCM16 little-endian. Offsets start
at zero and are contiguous in samples **per channel**; count each packet as
`Pcm.Length / (2 * Channels)`. Consumers may retain packets after callback return.

Results expose `Status`, `SampleRate`, `Channels`, `Samples`, `Chunks`,
`LimitedChunks`, and `Error`. `TtsStatus` is `Completed`, `Cancelled`, `Limited`,
or `Failed`. Native synthesis counters describe progress; on cancellation,
queued packets can be discarded before the host reads them. `LimitedChunks`
counts engine chunks that hit the cap, not callback packets.

```csharp
// Execute this lifecycle on a worker; retain the model across lines.
using var loader = ModelLoader.FromDirectory(modelDirectory);
using var model = loader.Load();
using var cancel = new CancellationTokenSource();
using var input = Envelope.Text("Welcome, traveler. Take a seat by the fire.");
var outcome = model.RunTtsStreaming(input, packet =>
{
    // Queue owned packet; do not call Unity APIs here.
    // The game checks utterance generation IDs before enqueue/playback.
    playbackQueue.Enqueue(packet);
}, cancel.Token);
// Another thread may call cancel.Cancel() during native computation.
// Dispose/switch the model only after RunTtsStreaming returns.
```

Sync delivery runs on the calling thread. The async partner uses a worker and
also invokes callbacks there. A managed delegate never crosses the ABI, so no
AOT callback thunk/rooting is needed. The existing cancellation registration
keeps the native cancellation handle alive through the call. Host callback
exceptions cancel and drain the worker before propagating.

Cancellation checks queued requests before native entry, reaches active native
inference independently of the synthesis/model lock, suppresses stale queued
packets, and leaves the session reusable. An in-progress waveform stage can
delay cancellation. The game must invalidate its own generation IDs and clear
its bounded playback buffer on interruption. Create `AudioClip` on Unity's
main thread; the audio-thread reader copies buffered PCM only.

For nonstreaming mode use `model.Run(input)`: `AudioBytes` holds batch PCM16,
`SpeechStatus` and `LimitedChunks` retain Kitten terminal information, and
`Success` is false for limited/cancelled audio. `RunTts(text)` is the existing
byte-only convenience and throws on partial limits. Kitten batch `Run(input, cancellationToken: token)` also uses the controlled
stream internally to interrupt live synthesis and retain partial batch audio.
Streaming without a callback
drains/discards audio and still returns the summary.

## Validation

`tools/tts-stream-smoke` compiles the actual Unity API and generated Bolt
bindings against .NET 8 and exercises the native plugin. Run it with the normal
metadata directory, a copy with `max_tokens: 8`, and optionally a directory with
missing asset paths. On Linux point `LD_LIBRARY_PATH` at the staged `Linux`
plugin folder; on macOS use the corresponding `DYLD_LIBRARY_PATH`.

```bash
dotnet run --project tools/tts-stream-smoke -- \
  kitten /path/to/bundle /path/to/capped-bundle /path/to/missing-bundle
dotnet run --project tools/tts-stream-smoke -- \
  kokoro integration-tests/fixtures/models/kokoro-82m
dotnet build tools/unity-bolt-compile-check/UnityBoltCompileCheck.csproj
```

The harness checks retained PCM ownership, contiguous offsets, callback thread,
short/long speech, early packets, live/pre/final cancellation, resident reuse,
callback exception shutdown, explicit limits, partial batch results and missing
assets. It prints first audio, total time, largest packet gap and RSS. Native-host
and netstandard2.1/C#9 compilation do not replace Unity Editor, M1 performance,
audio playback or standalone Mono/IL2CPP validation. Those require the actual
project/backend and must be recorded separately.
