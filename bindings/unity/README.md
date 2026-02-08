# Xybrid Unity SDK

On-device ML inference SDK for Unity - run TTS, ASR, and LLM models locally in your game.

## Installation

### Option 1: Git URL (Recommended)

1. Open your Unity project
2. Go to **Window → Package Manager**
3. Click **+ → Add package from git URL**
4. Enter:
   ```
   https://github.com/xybrid-ai/xybrid.git?path=bindings/unity
   ```

Or add directly to `Packages/manifest.json`:

```json
{
  "dependencies": {
    "ai.xybrid.sdk": "https://github.com/xybrid-ai/xybrid.git?path=bindings/unity"
  }
}
```

### Option 2: Local Development

If you've cloned the xybrid repository:

```json
{
  "dependencies": {
    "ai.xybrid.sdk": "file:../path/to/xybrid/bindings/unity"
  }
}
```

### Option 3: Tarball

Download the `.tgz` release from GitHub, then:
1. **Window → Package Manager**
2. Click **+ → Add package from tarball**
3. Select the downloaded `.tgz` file

## Quick Start

```csharp
using Xybrid.Native;
using UnityEngine;

public class XybridExample : MonoBehaviour
{
    private IntPtr loader;
    private IntPtr model;

    void Start()
    {
        // Initialize the SDK
        NativeMethods.xybrid_init();

        // Create a model loader
        loader = NativeMethods.xybrid_model_loader_new();

        // Load a model from the registry
        int result = NativeMethods.xybrid_model_loader_load(
            loader,
            "gemma-3-4b-it-qat-q4_0",  // model ID
            IntPtr.Zero,               // version (null = latest)
            out model
        );

        if (result != 0)
        {
            Debug.LogError($"Failed to load model: {NativeMethods.xybrid_last_error()}");
            return;
        }

        Debug.Log("Model loaded successfully!");
    }

    public string Generate(string prompt)
    {
        // Create input envelope with text
        IntPtr envelope = NativeMethods.xybrid_envelope_new_text(prompt);

        // Run inference
        int result = NativeMethods.xybrid_model_run(model, envelope, out IntPtr output);

        // Get the result text
        string response = "";
        if (result == 0)
        {
            IntPtr textPtr = NativeMethods.xybrid_result_get_text(output);
            response = Marshal.PtrToStringUTF8(textPtr);
            NativeMethods.xybrid_result_free(output);
        }

        NativeMethods.xybrid_envelope_free(envelope);
        return response;
    }

    void OnDestroy()
    {
        if (model != IntPtr.Zero)
            NativeMethods.xybrid_model_free(model);
        if (loader != IntPtr.Zero)
            NativeMethods.xybrid_model_loader_free(loader);
    }
}
```

## Available Models

| Model ID | Type | Size | Description |
|----------|------|------|-------------|
| `gemma-3-4b-it-qat-q4_0` | LLM | ~2.5GB | Conversational AI |
| `kokoro-82m` | TTS | ~330MB | Text-to-speech |
| `whisper-tiny` | ASR | ~75MB | Speech recognition |

Models are automatically downloaded from the Xybrid registry on first use.

## Supported Platforms

| Platform | Architecture | Status |
|----------|--------------|--------|
| macOS | Apple Silicon (arm64) | Supported |
| macOS | Intel (x86_64) | Supported |
| Windows | x64 | Planned |
| Linux | x64 | Planned |
| iOS | arm64 | Planned |
| Android | arm64-v8a | Planned |

## Building Native Libraries

If you need to build the native libraries yourself:

```bash
# Clone the repository
git clone https://github.com/xybrid-ai/xybrid.git
cd xybrid

# Build with C# bindings
cargo xtask build-ffi --release --csharp

# Output locations:
# - Native lib: target/release/libxybrid_ffi.dylib (macOS)
# - C# bindings: bindings/unity/Runtime/Native/NativeMethods.g.cs
```

### Cross-platform builds

```bash
# macOS (from macOS)
cargo xtask build-ffi --release --csharp

# Windows (from Windows)
cargo xtask build-ffi --release --csharp

# iOS (from macOS)
cargo xtask build-ffi --release --target aarch64-apple-ios

# Android (requires NDK)
cargo xtask build-ffi --release --target aarch64-linux-android
```

## Package Structure

```
bindings/unity/
├── package.json                 # UPM package manifest
├── Runtime/
│   ├── Native/
│   │   ├── NativeMethods.g.cs   # Auto-generated P/Invoke bindings
│   │   └── NativeHelpers.cs     # Helper utilities
│   └── Plugins/
│       ├── macOS/
│       │   └── libxybrid_ffi.dylib
│       ├── Windows/
│       │   └── xybrid_ffi.dll
│       ├── iOS/
│       │   └── libxybrid_ffi.a
│       └── Android/
│           └── libxybrid_ffi.so
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

### "DllNotFoundException: xybrid_ffi"

1. Ensure the native library is in the correct `Plugins/` subfolder for your platform
2. On macOS, you may need to remove quarantine: `xattr -d com.apple.quarantine libxybrid_ffi.dylib`
3. Check the plugin import settings in Unity (select the .dylib and verify platform settings)

### "Model download failed"

1. Check your internet connection
2. Ensure the model ID is correct (see Available Models above)
3. Check `~/.xybrid/cache/` for partially downloaded files

## API Reference

See the [full API documentation](https://docs.xybrid.ai/unity) for detailed reference.

## License

Apache 2.0 - See [LICENSE](../../LICENSE) for details.
