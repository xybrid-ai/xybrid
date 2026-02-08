# Xybrid Unity Starter Example

A minimal Unity project demonstrating Xybrid SDK integration for on-device ML inference in games.

> **Looking for more examples?** See the [Unity Examples Overview](../README.md) for full game examples.

## Requirements

- **Unity 2022.3 LTS** (or later)
- **Xcode** (for iOS builds)
- **Android Studio** (for Android builds)
- **Rust toolchain** (for building native libraries)

## Quick Start

### 1. Open in Unity

1. Open Unity Hub
2. Click "Open" and select the `examples/unity/starter` directory
3. Wait for Unity to import the project

### 2. Build Native Libraries

Before the SDK will work, you need to build the native libraries from the Rust source:

```bash
# From the repository root
cd repos/xybrid

# Build for iOS/macOS
cargo xtask build-xcframework

# Build for Android
cargo xtask build-android
```

Copy the built libraries to `Assets/Plugins/`:
- iOS: `libxybrid.a` → `Assets/Plugins/iOS/`
- Android: `libxybrid.so` → `Assets/Plugins/Android/{arm64-v8a,armeabi-v7a}/`
- macOS: `libxybrid.dylib` → `Assets/Plugins/macOS/`

### 3. Play in Editor

1. Open `Assets/Scenes/SampleScene`
2. Press Play to test in the Unity Editor

## Demo Scene

The `SampleScene` includes a complete inference demo with:

- **UI Canvas** with input field, button, and result display
- **XybridDemoController** script attached to the DemoPanel
- **Automatic SDK initialization** on scene load
- **Model loading** from the registry (kokoro-82m TTS model)
- **Inference execution** with latency measurement

### Scene Hierarchy

```
SampleScene
├── Main Camera
├── Directional Light
├── Canvas
│   └── DemoPanel (XybridDemoController)
│       ├── StatusText       # Shows SDK/model status
│       ├── InputField       # Text input for TTS
│       ├── RunInferenceButton
│       ├── ResultText       # Inference output
│       └── LatencyText      # Timing metrics
└── EventSystem
```

### How the Demo Works

1. **Start**: SDK initializes automatically
2. **Model Loading**: Loads kokoro-82m TTS model from registry
3. **Input**: Enter text in the input field
4. **Run Inference**: Click button to run TTS inference
5. **Result**: See output text and latency in milliseconds

> **Note**: Until native libraries are built, the demo simulates SDK calls.
> See "Build Native Libraries" section to enable real inference.

## Project Structure

```
examples/unity/starter/
├── Assets/
│   ├── Plugins/          # Native libraries (built from Rust)
│   │   ├── iOS/          # libxybrid.a
│   │   ├── Android/      # libxybrid.so (per architecture)
│   │   ├── macOS/        # libxybrid.dylib
│   │   └── Windows/      # xybrid.dll (future)
│   ├── Scenes/           # Unity scenes
│   │   └── SampleScene.unity
│   └── Scripts/          # C# demo scripts
│       └── XybridDemoController.cs
├── Packages/
│   └── manifest.json     # Package dependencies (includes Xybrid SDK)
└── ProjectSettings/      # Unity project configuration
```

## SDK Usage

The Xybrid SDK is included as a local Unity Package from `bindings/unity`.

### Basic Usage (when native libraries are available)

```csharp
using Xybrid;
using UnityEngine;

public class XybridDemo : MonoBehaviour
{
    async void Start()
    {
        // Load a model from the registry
        var loader = ModelLoader.FromRegistry("kokoro-82m");
        var model = loader.Load();

        // Create text input for TTS
        var input = Envelope.Text("Hello from Unity!");

        // Run inference
        var result = model.Run(input);

        if (result.Success)
        {
            Debug.Log($"Inference completed in {result.LatencyMs}ms");
            Debug.Log($"Result: {result.Text}");
        }
        else
        {
            Debug.LogError($"Inference failed: {result.Error}");
        }

        // Clean up
        result.Dispose();
        model.Dispose();
    }
}
```

### Available Classes

| Class | Description |
|-------|-------------|
| `ModelLoader` | Load models from registry or local bundles |
| `Model` | A loaded model ready for inference |
| `Envelope` | Input data (text or audio) for inference |
| `Result` | Inference output with latency metrics |

## Build Instructions

### iOS Build

**Prerequisites:**
- Xcode 14+ installed
- iOS Developer certificate configured
- Native libraries built (see "Build Native Libraries" section)

**Steps:**

1. **File > Build Settings**
2. Select **iOS** platform
3. Click **Switch Platform**
4. Configure in **Player Settings > Other Settings**:
   - Target minimum iOS Version: 15.0
   - Architecture: ARM64
   - Scripting Backend: IL2CPP
5. Click **Build** and select output folder
6. Open the generated Xcode project
7. In Xcode:
   - Select the target
   - Go to Signing & Capabilities
   - Configure your development team
   - Select provisioning profile
8. Build and run on device

**Native Library Location:**
```
Assets/Plugins/iOS/libxybrid.a
```

### Android Build

**Prerequisites:**
- Android SDK installed (via Unity Hub or Android Studio)
- Android NDK installed (via Unity Hub)
- Native libraries built (see "Build Native Libraries" section)

**Steps:**

1. **File > Build Settings**
2. Select **Android** platform
3. Click **Switch Platform**
4. Configure in **Player Settings > Other Settings**:
   - Minimum API Level: 28 (Android 9.0)
   - Target API Level: 33+ (Android 13+)
   - Scripting Backend: IL2CPP
   - Target Architectures: ARM64 (check box)
5. Configure in **Edit > Preferences > External Tools**:
   - Verify Android SDK path
   - Verify Android NDK path
6. Click **Build and Run** (with device connected)
   - Or click **Build** to create APK for later installation

**Native Library Locations:**
```
Assets/Plugins/Android/arm64-v8a/libxybrid.so
Assets/Plugins/Android/armeabi-v7a/libxybrid.so (optional, for older devices)
```

### macOS Build (Editor Testing)

For testing in the Unity Editor on macOS:

**Steps:**

1. Build the macOS native library:
   ```bash
   cd repos/xybrid
   cargo build --release --package xybrid-ffi
   ```
2. Copy the library:
   ```bash
   cp target/release/libxybrid_ffi.dylib examples/unity/starter/Assets/Plugins/macOS/libxybrid.dylib
   ```
3. In Unity, press **Play** to run the demo scene

**Native Library Location:**
```
Assets/Plugins/macOS/libxybrid.dylib
```

### Windows Build (Future)

Windows support is planned but not yet implemented. Native library builds for Windows will require:
- Visual Studio with C++ workload
- Windows SDK
- Rust toolchain for Windows targets

## Troubleshooting

### "DllNotFoundException: xybrid_ffi"

The native library hasn't been built or copied to the correct location. See **Build Native Libraries** section above.

### Project won't open in Unity

Make sure you're using Unity 2022.3 LTS or later. The project uses features that may not be available in older versions.

### iOS build fails with signing errors

1. Open the Xcode project generated by Unity
2. Select the target and go to Signing & Capabilities
3. Configure your development team and provisioning profile

### Android build fails with NDK errors

1. Ensure Android NDK is installed via Unity Hub
2. Check **Edit > Preferences > External Tools** for correct NDK path

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| iOS | Planned | Requires XCFramework build |
| Android | Planned | Requires AAR/SO build |
| macOS | Planned | Editor testing support |
| Windows | Future | Not yet implemented |

## API Reference

- **[SDK API Reference](../../../docs/sdk/API_REFERENCE.md)** - Complete API specification
- **[Unity SDK Docs](../../../docs/sdk/unity.md)** - Unity-specific documentation (planned)
- **[Online Documentation](https://docs.xybrid.dev/sdk/unity)** - Full guides and tutorials

## Related

- [Unity Examples Overview](../README.md) - All Unity examples
- [Main Examples README](../../README.md) - All platform examples
- [Xybrid Unity SDK](../../../bindings/unity/README.md)
- [Xybrid Repository](https://github.com/xybrid-ai/xybrid)

## License

MIT License - See [LICENSE](../../../LICENSE) for details.
