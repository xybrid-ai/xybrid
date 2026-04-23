# Xybrid Unity Telemetry Example

A minimal Unity project demonstrating end-to-end Xybrid telemetry wiring,
including the safe mobile pause/quit lifecycle required for production
iOS and Android integrations.

## What this example shows

The scene has **one GameObject** with a single MonoBehaviour
(`XybridTelemetryExample`) that:

1. Calls `XybridClient.Initialize()` at `Start()`.
2. Resolves the telemetry **API key** from env vars (or PlayerPrefs).
3. Builds a `TelemetryConfig` via the fluent builder (defaults to
   `https://ingest.xybrid.dev`) and hands it to `XybridClient.InitializeTelemetry`.
4. Runs one inference to emit at least one telemetry event.
5. Flushes the buffer on `OnApplicationPause(true)` so events reach the
   collector before the OS suspends or kills the process.
6. Flushes and then shuts the sender down on `OnApplicationQuit()`.

No API key is ever hardcoded, serialized into the scene, or committed to the
repository. The script reads credentials from runtime sources only.

## Requirements

- **Unity 2022.3 LTS** (or later)
- **Xybrid Unity SDK** (resolved via `Packages/manifest.json` as a local
  file reference to `bindings/unity`)
- **Native libraries** (see _Build Native Libraries_ below)
- A valid Xybrid telemetry API key

## Configuration

The only value you **must** supply is the API key. By default the example
sends events to the SDK's built-in ingest URL (`https://ingest.xybrid.dev`).
Override the endpoint only when self-hosting or pointing at staging.

The example reads configuration at runtime in this order:

| Priority | Source                    | API key (required)          | Endpoint (optional override)  |
| :------: | ------------------------- | --------------------------- | ----------------------------- |
|    1     | Environment variables     | `XYBRID_TELEMETRY_API_KEY`  | `XYBRID_TELEMETRY_ENDPOINT`   |
|    2     | `UnityEngine.PlayerPrefs` | `Xybrid.Telemetry.ApiKey`   | `Xybrid.Telemetry.Endpoint`   |

### Option A — environment variables (recommended for desktop / CI)

```bash
# Required
export XYBRID_TELEMETRY_API_KEY="xyb_live_..."
open -a "Unity" /path/to/examples/unity/Telemetry
```

Unity inherits the env vars from its parent shell; launching from the Unity
Hub GUI will **not** pick them up. Launch the Editor from a shell, or export
the vars globally.

### Option B — PlayerPrefs (for on-device runs)

Populate PlayerPrefs once from a setup script or an in-app settings screen:

```csharp
UnityEngine.PlayerPrefs.SetString("Xybrid.Telemetry.ApiKey", apiKey);
UnityEngine.PlayerPrefs.Save();
```

> Do **not** check in a scene or prefab that contains the key. PlayerPrefs
> writes are per-device and never end up in version control.

If no API key is supplied, the demo logs a warning and skips telemetry, but
still exercises the rest of the SDK so the scene runs clean in the editor.

### Optional — self-hosted or staging endpoint

If you're running your own collector, also set:

```bash
export XYBRID_TELEMETRY_ENDPOINT="https://telemetry.internal.example.com"
```

or the matching PlayerPrefs key `Xybrid.Telemetry.Endpoint`. The example then
calls `config.WithEndpoint(override)` before handing the config to
`InitializeTelemetry`. Omit both and the SDK default is used.

## Why both `OnApplicationPause` and `OnApplicationQuit`?

Unity invokes these lifecycle callbacks at different times:

- **`OnApplicationQuit`** fires when the player exits cleanly (desktop builds,
  editor play-mode stop). It is **not reliably delivered on mobile**: iOS and
  Android can terminate a backgrounded app at any moment, so depending on it
  alone means you lose the last batch of events.
- **`OnApplicationPause(true)`** fires every time the app loses focus:
  incoming call, home-button press, the user switching apps. This is the
  **only** hook that is guaranteed to run before the OS can freeze or kill
  the process.

Production mobile integrations therefore need **both**:

```csharp
private void OnApplicationPause(bool pauseStatus)
{
    if (pauseStatus) { XybridClient.FlushTelemetry(); }
}

private void OnApplicationQuit()
{
    XybridClient.FlushTelemetry();
    XybridClient.ShutdownTelemetry();
}
```

Flushing on pause guarantees delivery of anything buffered before the user
leaves the app. Flushing again on quit catches anything produced between the
last pause and a clean exit. `ShutdownTelemetry()` is safe to call multiple
times (the SDK treats repeats as no-ops), and the editor domain-reload guard
(`Editor/TelemetryDomainReloadGuard.cs` in the SDK) calls it for you between
script reloads so your iteration loop does not leak native workers.

## Build Native Libraries

Before the SDK links, build the native libraries for your target platform
and place them under `Assets/Plugins/`:

```bash
# From repository root
cd repos/xybrid

# macOS (editor + macOS builds)
cargo build --release -p xybrid-ffi --features platform-macos
cp target/release/libxybrid_ffi.dylib \
   examples/unity/Telemetry/Assets/Plugins/macOS/libxybrid.dylib

# iOS
cargo xtask build-xcframework
# Copy the static library per the xybrid README.

# Android
cargo xtask build-android
# Copy libxybrid.so into Assets/Plugins/Android/<abi>/
```

See `examples/unity/starter/README.md` for full platform-specific steps.

## Run the example

1. Open this project in Unity Hub (2022.3 LTS or later).
2. Let Unity import the package reference (`ai.xybrid.sdk`) from
   `bindings/unity`.
3. Open `Assets/Scenes/TelemetryScene.unity`.
4. Set your API key via env vars or PlayerPrefs (see above). Optionally set
   an endpoint override if you're self-hosting.
5. Press **Play**.

The Console should print, in order:

```
[XybridTelemetry] SDK initialized (version=...).
[XybridTelemetry] Telemetry started -> https://telemetry.example.com
[XybridTelemetry] Inference succeeded (...ms, output=...).
```

On Stop (or on quit from a standalone build) you should additionally see:

```
[XybridTelemetry] Telemetry shut down on quit.
```

## Project structure

```
examples/unity/Telemetry/
├── Assets/
│   ├── Plugins/                 # Native libs (built from Rust, .gitignored)
│   ├── Scenes/
│   │   └── TelemetryScene.unity # Single GameObject + MonoBehaviour
│   └── Scripts/
│       └── XybridTelemetryExample.cs
├── Packages/
│   └── manifest.json            # References ai.xybrid.sdk via file: URI
├── ProjectSettings/             # Unity editor + player defaults
├── .gitignore
└── README.md                    # This file
```

## Related

- [Starter Example](../starter/README.md) — Minimal inference demo
- [Unity SDK README](../../../bindings/unity/README.md)
- [SDK API Reference](../../../docs/sdk/API_REFERENCE.md)

## License

MIT License - See [LICENSE](../../../LICENSE) for details.
