# Xybrid Unity SDK

On-device ML inference SDK for Unity - run TTS, ASR, and LLM models locally in your game.

## Installation

The Unity package is **managed-only**: it contains the C# API and, on import,
downloads the matching native libraries from the GitHub Release (SHA-256
verified) into `Assets/Xybrid/Plugins/`. No platform binaries are committed to
the package, so it installs cleanly from any source below.

### Option 1: OpenUPM (recommended)

Xybrid is published on [OpenUPM](https://openupm.com/packages/ai.xybrid.sdk/).

With the OpenUPM CLI:

```bash
openupm add ai.xybrid.sdk
```

Without the CLI, add the scoped registry in **Project Settings → Package
Manager → Scoped Registries → +**:

| Field    | Value                          |
|----------|--------------------------------|
| Name     | `OpenUPM`                      |
| URL      | `https://package.openupm.com`  |
| Scope(s) | `ai.xybrid`                    |

Then install **Xybrid SDK** from **Window → Package Manager → My Registries**.

Or edit `Packages/manifest.json` directly:

```json
{
  "scopedRegistries": [
    {
      "name": "OpenUPM",
      "url": "https://package.openupm.com",
      "scopes": ["ai.xybrid"]
    }
  ],
  "dependencies": {
    "ai.xybrid.sdk": "0.14.0"
  }
}
```

### Option 2: Git URL

UPM can install the package straight from the repository subfolder — no OpenUPM
registry required:

```
https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity
```

Pin a version by appending a tag:

```
https://github.com/xybrid-ai/xybrid.git?path=/bindings/unity#v0.14.0
```

(**Window → Package Manager → + → Add package from git URL**, or add it under
`dependencies` in `Packages/manifest.json`.)

### Option 3: Local development

If you've cloned the xybrid repository:

```json
{
  "dependencies": {
    "ai.xybrid.sdk": "file:../path/to/xybrid/bindings/unity"
  }
}
```

### Native libraries (fetched automatically)

The package ships **no native binaries**. The SDK downloads the platform
libraries for the installed version from the
[GitHub Release](https://github.com/xybrid-ai/xybrid/releases), verifies their
SHA-256, and installs them into `Assets/Xybrid/Plugins/`: the editor's own
platform on first import, and the build target's before a player build if they
aren't installed yet. This
keeps the package small and sidesteps Git/registry size limits — **including the
~326 MB iOS static library, which now installs automatically** (no manual step).
The Windows and Linux bundles include the matching ONNX Runtime shared library,
so desktop users do not need a system-wide ONNX Runtime installation.

To (re)download on demand, use the editor menu:

- **Xybrid → Native Libraries → Download for Current Editor** — the editor host
  platform, so Play mode works immediately.
- **Xybrid → Native Libraries → Download for Active Build Target** — the current
  build target's libraries.

Network access is required on first import; the libraries are cached in the
project afterward. Add `Assets/Xybrid/Plugins/` to your VCS ignore list if you
don't want the downloaded binaries committed.

## Quick Start

```csharp
using Xybrid;
using UnityEngine;

public class XybridExample : MonoBehaviour
{
    private Model model;

    void Start()
    {
        // Runs locally with no key. Pass an apiKey to light up the
        // dashboard: XybridClient.Initialize(apiKey: "xy_live_...")
        XybridClient.Initialize();

        // Load a model from the registry
        model = XybridClient.LoadModel("gemma-3-4b-it-qat-q4_0");
        Debug.Log($"Model loaded: {model.ModelId}");
    }

    public string Generate(string prompt)
    {
        // Run inference with a text prompt
        using var result = model.Run(Envelope.Text(prompt));
        result.ThrowIfFailed();
        return result.Text;
    }

    void OnDestroy()
    {
        model?.Dispose();
    }
}
```

### Text-to-Speech

```csharp
using Xybrid;

// Load a TTS model
using var model = XybridClient.LoadModel("kokoro-82m");

// Generate NPC dialogue audio
using var result = model.Run(Envelope.Text("Welcome, traveler. The road ahead is dangerous."));
result.ThrowIfFailed();

// result.Text contains the audio output
Debug.Log($"Inference completed in {result.LatencyMs}ms");
```

### Speech Recognition

```csharp
using System.IO;
using UnityEngine;
using Xybrid;

// English ASR; use "whisper-tiny-ggml" for multilingual recognition.
using var model = XybridClient.LoadModel("whisper-tiny-en-ggml");

// A complete WAV file: 16-bit PCM, mono, 16 kHz.
byte[] wavBytes = File.ReadAllBytes("voice-command.wav");
using var envelope = Envelope.Audio(wavBytes, sampleRate: 16000, channels: 1);
using var result = model.Run(envelope);
result.ThrowIfFailed();

Debug.Log($"Player said: {result.Text}");
```

These GGML models use the speech-recognition runtime included in the prebuilt
native libraries. For these models, `Envelope.Audio` and `Model.RunAudio` require
a complete WAV file, including its header. For Unity `Microphone` input, encode
the recorded samples as WAV first.

### Vision-Language Input

Vision-language envelopes are available in the v0.2.0 development surface and
require native libraries built with the vision feature.

```csharp
using Xybrid;
using UnityEngine;

// Load a VLM bundle built with a language GGUF plus mmproj sibling.
using var model = XybridClient.LoadModel("lfm2-vl-450m");

// Texture2D assets can be encoded before entering the SDK.
byte[] imageBytes = texture.EncodeToPNG();
using var image = Envelope.Image(imageBytes, "png");
using var prompt = Envelope.UserMessage("Describe this image", new[] { image });

using var result = model.Run(prompt);
result.ThrowIfFailed();

Debug.Log(result.Text);
```

### Inference Metrics

Every `InferenceResult` carries a typed `InferenceMetrics` with TTFT,
tok/s, per-stage latencies, and token counts. LLM-specific fields are
`null` for ASR / TTS / embedding runs.

```csharp
using Xybrid;

using var model = XybridClient.LoadModel("lfm2.5-350m");
using var result = model.Run(Envelope.Text("Tell me a joke."));
result.ThrowIfFailed();

var metrics = result.Metrics;
Debug.Log($"Total: {metrics.TotalMs} ms");
if (metrics.TtftMs.HasValue)
    Debug.Log($"TTFT: {metrics.TtftMs.Value} ms");
if (metrics.TokensPerSecond.HasValue)
    Debug.Log($"Throughput: {metrics.TokensPerSecond.Value:F1} tok/s");
if (metrics.TokensOut.HasValue)
    Debug.Log($"Tokens out: {metrics.TokensOut.Value}");

// For pipeline runs, per-stage latencies are populated.
// model.Run() leaves StageLatenciesMs empty.
foreach (var stage in metrics.StageLatenciesMs)
    Debug.Log($"  stage {stage.StageId}: {stage.LatencyMs} ms");
```

### Multi-Turn Conversation

```csharp
using Xybrid;

using var model = XybridClient.LoadModel("gemma-3-4b-it-qat-q4_0");
using var context = new ConversationContext();

// Set the NPC personality
context.SetSystem("You are a merchant in a medieval village. You sell potions and gear.");

// First turn
using var result1 = model.Run(Envelope.Text("What do you have for sale?", MessageRole.User), context);
Debug.Log(result1.Text);

// Second turn (conversation history is maintained)
using var result2 = model.Run(Envelope.Text("How much for the healing potion?", MessageRole.User), context);
Debug.Log(result2.Text);
```

## Available Models

| Model ID | Type | Size | Description |
|----------|------|------|-------------|
| `gemma-3-4b-it-qat-q4_0` | LLM | ~2.5GB | Conversational AI |
| `kokoro-82m` | TTS | ~330MB | Text-to-speech |
| `whisper-tiny-en-ggml` | ASR | ~32MB (~44MB on iOS) | English speech recognition (WAV input) |

Models are automatically downloaded from the Xybrid registry on first use.

## Supported Platforms

| Platform | Architecture | Status |
|----------|--------------|--------|
| macOS 14+ | Apple Silicon (arm64) | Supported |
| macOS | Intel (x86_64) | Not supported |
| Windows | x64 | Supported |
| Linux | x64 | Supported |
| iOS | arm64 (devices, iOS 16+) | Supported |
| Android | arm64-v8a, armeabi-v7a | Supported |
| Android | x86_64 | Supported up to Unity 6.4 |

**macOS:** the native library requires **macOS 14 or later** and **Apple Silicon
(arm64)** for both the editor and players. Set the player's minimum macOS version
to **14.0** or later. Use the Apple silicon Unity Editor: the Intel editor can't
load the library, on an Intel Mac or under Rosetta 2, which runs x86_64 code on
Apple Silicon and not the reverse. Build macOS players for **Apple Silicon**. An
**Intel 64-bit + Apple Silicon** player can load Xybrid only when it runs natively
on an Apple Silicon Mac, not on an Intel Mac or when opened using Rosetta.

**iOS:** the library is built for devices running iOS 16 or later, with no
Simulator build. Set **Target minimum iOS Version** to 16.0 or later.

**Android:** Unity 6.5 removed the x86_64 architecture
([announcement](https://discussions.unity.com/t/planned-breaking-changes-in-unity-6-5-updated-2026-03-27/1694205)),
and Unity 6.3 and 6.4 offer it only in projects that already target it
([Android Player settings](https://docs.unity3d.com/6000.4/Documentation/Manual/class-PlayerSettingsAndroid.html)).
armeabi-v7a ships without ONNX Runtime, so ONNX models such as `kokoro-82m`
don't run on it.

On Android, models are cached under `Application.persistentDataPath/xybrid`
(the SDK is given its `models` subfolder and keeps runtime-ready files beside it
in `extracted`). To use another folder, call `XybridBolt.XybridBolt.InitSdkCacheDir(path)` before
loading or listing any model; the first folder set wins.

Xybrid's Android library links the C++ runtime statically, so its Android
natives include no `libc++_shared.so` to collide with the one Unity's
GameActivity packages. Earlier releases installed a copy under
`Assets/Xybrid/Plugins/Android/<abi>/`; after you update the package, it is
removed the next time the Android natives are downloaded: before the next
Android build, or from **Xybrid > Native Libraries > Download for Active Build
Target** while Android is the active build target.

Device coverage: the Android natives have run end to end in a Unity app on an
arm64 phone, and the release workflow loads the x86_64 library on a standard
(4 KB-page) emulator. Loading on 16 KB-page devices and on armeabi-v7a, and
running a Unity app on an emulator, are tracked in
[#680](https://github.com/xybrid-ai/xybrid/issues/680).

## Building Native Libraries

If you need to build the native libraries yourself:

```bash
# Clone the repository
git clone https://github.com/xybrid-ai/xybrid.git
cd xybrid

# Build the native library + deploy into the Unity plugins tree
cargo xtask build-ffi --release --deploy-unity

# Output locations:
# - Native lib: target/release/libxybrid_bolt.dylib (macOS)
```

### Cross-platform builds

```bash
# macOS (from macOS)
cargo xtask build-ffi --release --deploy-unity

# Windows (from Windows)
cargo xtask build-ffi --release --deploy-unity

# iOS (from macOS)
cargo xtask build-ffi --release --target aarch64-apple-ios

# Android (Bazel, one build per ABI, staged as CI does in build-unity.yml;
# off a Linux x86_64 host, add --config=remote to bazel build and bazel-output.sh)
bazel build --config=android-arm64 //crates/xybrid-bolt:xybrid_bolt_cdylib
LIB=$(tools/scripts/bazel-output.sh //crates/xybrid-bolt:xybrid_bolt_cdylib --config=android-arm64)
python3 tools/scripts/stage_unity_native.py --lib "$LIB" --target aarch64-linux-android
# Likewise --config=android-armv7 with armv7-linux-androideabi,
# and --config=android-x86_64 with x86_64-linux-android
# Then run the check CI runs on the staged tree (no libc++_shared.so, static
# C++ runtime, right ELF machine per ABI, 16 KB alignment). macOS has no
# readelf: add --readelf "$(brew --prefix llvm)/bin/llvm-readelf"
python3 tools/scripts/verify_unity_android_natives.py --require-abis
```

## Package Structure

```
bindings/unity/
├── package.json                 # UPM package manifest
├── Runtime/
│   ├── Api/
│   │   ├── XybridClient.cs      # SDK entry point (Initialize, LoadModel)
│   │   ├── Model.cs             # Model inference (Run, RunText, RunAudio)
│   │   ├── ModelLoader.cs       # Model loading (FromRegistry, FromBundle)
│   │   ├── Envelope.cs          # Input data (Text, Audio, Image, UserMessage)
│   │   ├── InferenceResult.cs   # Output container (Text, Success, LatencyMs)
│   │   ├── ConversationContext.cs # Multi-turn LLM state
│   │   ├── MessageRole.cs       # Role enum (System, User, Assistant)
│   │   └── XybridException.cs   # Exception types
│   └── Plugins/                 # Empty in the package — native binaries and
│                                # their .meta import settings are fetched at
│                                # import into Assets/Xybrid/Plugins/ (see above)
├── Editor/                      # Native-library resolver (download + verify)
└── README.md
```

## Unity Version Compatibility

| Unity Version | Status |
|---------------|--------|
| 2021.3 LTS | Supported |
| 2022.3 LTS | Supported |
| 2023.x | Supported |
| 6000.x (Unity 6) | Supported |

## Troubleshooting

### "DllNotFoundException: xybrid_bolt"

1. Fetch the native library: **Xybrid → Native Libraries → Download for Current
   Editor** (the auto-download on import needs network access and the matching
   GitHub Release). Confirm it landed under `Assets/Xybrid/Plugins/`.
2. On macOS 14 or later, use the Apple silicon Unity Editor. The library is
   arm64 only, so the Intel editor can't load it, including under Rosetta 2.
3. On macOS, you may need to remove quarantine: `xattr -d com.apple.quarantine libxybrid_bolt.dylib`
4. Check the plugin import settings in Unity (select the .dylib and verify platform settings)

### "Model download failed"

1. Check your internet connection
2. Ensure the model ID is correct (see Available Models above)
3. Check `~/.xybrid/cache/` for partially downloaded files

## API Reference

See the [full API documentation](https://docs.xybrid.dev/en/docs/sdks/unity) for detailed reference.

## Telemetry

The Unity binding reports `binding=unity` in a small `X-Xybrid-Client` header attached to registry metadata calls. See [docs/telemetry/registry.md](../../docs/telemetry/registry.md) for the exact wire format and the opt-out switch (`XYBRID_TELEMETRY_OPTOUT=1`).

## License

Apache 2.0 - See [LICENSE](../../LICENSE) for details.
