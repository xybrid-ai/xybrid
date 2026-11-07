# Xybrid Unity Observability Example

A minimal Unity project demonstrating OpenTelemetry export from the Xybrid
Unity SDK: a traced inference, feedback linked to it, and the safe mobile
pause/quit lifecycle required for production iOS and Android integrations.

> Export is off until you configure it — an API key passed to
> `XybridClient.Initialize` does **not** enable it. You build an
> `ObservabilityConfig` and call `XybridClient.InitializeObservability`.

## What this example shows

The scene has **one GameObject** with a single MonoBehaviour
(`XybridObservabilityExample`) that:

1. Calls `XybridClient.Initialize()` at `Start()`.
2. Resolves an export target from env vars (or PlayerPrefs): an OTLP collector
   endpoint, or else a Xybrid API key.
3. Builds an `ObservabilityConfig` from the matching preset
   (`ObservabilityConfig.Otlp(endpoint)` or `ObservabilityConfig.Xybrid(apiKey)`),
   sets the service name/version and resource attributes, and hands it to
   `XybridClient.InitializeObservability`.
4. Runs one inference with `ObservationOptions` (a conversation id), logs the
   result's `Observation` trace/span ids, and records feedback on it with
   `XybridClient.RecordFeedback`.
5. Flushes on `OnApplicationPause(true)` so spans reach the backend before the
   OS suspends or kills the process.
6. Shuts export down (flush + stop) on `OnApplicationQuit()`.

No API key is ever hardcoded, serialized into the scene, or committed to the
repository. The script reads credentials from runtime sources only.

## Requirements

- **Unity 2022.3 LTS** (or later)
- **Xybrid Unity SDK** (resolved via `Packages/manifest.json` as a local
  file reference to `bindings/unity`)
- **Native libraries** (see _Build Native Libraries_ below)
- Either an OpenTelemetry Collector (or any OTLP/HTTP backend) or a Xybrid API key

## Configuration

The example picks the first target it finds:

| Priority | Target                                 | Environment variable          | PlayerPrefs key                     |
| :------: | -------------------------------------- | ----------------------------- | ----------------------------------- |
|    1     | OTLP collector (traces, metrics, logs) | `OTEL_EXPORTER_OTLP_ENDPOINT` | `Xybrid.Observability.OtlpEndpoint` |
|    2     | Xybrid dashboard (traces only)         | `XYBRID_API_KEY`              | `Xybrid.Observability.ApiKey`       |

### Option A — environment variables (recommended for desktop / CI)

```bash
# An OpenTelemetry Collector on this machine:
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
# ...or the Xybrid dashboard:
export XYBRID_API_KEY="xy_live_..."
open -a "Unity" /path/to/examples/unity/Observability
```

Unity inherits the env vars from its parent shell; launching from the Unity
Hub GUI will **not** pick them up. Launch the Editor from a shell, or export
the vars globally.

### Option B — PlayerPrefs (for on-device runs)

Populate PlayerPrefs once from a setup script or an in-app settings screen:

```csharp
UnityEngine.PlayerPrefs.SetString("Xybrid.Observability.ApiKey", apiKey);
UnityEngine.PlayerPrefs.Save();
```

> Do **not** check in a scene or prefab that contains the key. PlayerPrefs
> writes are per-device and never end up in version control.

If neither target is supplied, the demo logs a warning and leaves export off,
but still exercises the rest of the SDK so the scene runs clean in the editor.

### Feedback and the Xybrid preset

`RecordFeedback` and `RecordEvaluation` travel on the OTLP **logs** signal.
The Xybrid preset sends traces only, so with an API key the example logs
`Feedback recorded: False`; point it at a collector to see feedback land.

## Why both `OnApplicationPause` and `OnApplicationQuit`?

Unity invokes these lifecycle callbacks at different times:

- **`OnApplicationQuit`** fires when the player exits cleanly (desktop builds,
  editor play-mode stop). It is **not reliably delivered on mobile**: iOS and
  Android can terminate a backgrounded app at any moment, so depending on it
  alone means you lose the last spans.
- **`OnApplicationPause(true)`** fires every time the app loses focus:
  incoming call, home-button press, the user switching apps. This is the
  **only** hook that is guaranteed to run before the OS can freeze or kill
  the process.

Production mobile integrations therefore need **both**:

```csharp
private void OnApplicationPause(bool pauseStatus)
{
    if (pauseStatus) { XybridClient.FlushObservability(TimeSpan.FromSeconds(2)); }
}

private void OnApplicationQuit()
{
    XybridClient.ShutdownObservability(TimeSpan.FromSeconds(5));
}
```

Both calls wait at most the given timeout (5 seconds by default) and throw
`XybridException` when it is exceeded. `ShutdownObservability()` flushes
before stopping, is safe to call multiple times, and is a no-op when export is
off. The SDK's editor domain-reload guard
(`Editor/ObservabilityDomainReloadGuard.cs`) calls it for you between script
reloads so your iteration loop does not leak native export workers.

## Build Native Libraries

Before the SDK links, build the native libraries for your target platform
and place them under `Assets/Plugins/`:

```bash
# From repository root
cd repos/xybrid

# macOS (editor + macOS builds)
cargo build --release -p xybrid-bolt --features platform-macos
cp target/release/libxybrid_bolt.dylib \
   examples/unity/Observability/Assets/Plugins/macOS/libxybrid.dylib

# iOS / Android — mirror .github/workflows/build-unity.yml: build the bolt
# native with Bazel, then stage it into bindings/unity/Runtime/Plugins/:
bazel build --config=ios //crates/xybrid-bolt:xybrid_bolt_staticlib          # iOS
bazel build --config=android-arm64 //crates/xybrid-bolt:xybrid_bolt_cdylib   # Android (also android-armv7, android-x86_64)
python3 tools/scripts/stage_unity_native.py --lib <bazelisk cquery --output=files path> --target <triple>
```

See `examples/unity/starter/README.md` for full platform-specific steps.

## Run the example

1. Open this project in Unity Hub (2022.3 LTS or later).
2. Let Unity import the package reference (`ai.xybrid.sdk`) from
   `bindings/unity`.
3. Open `Assets/Scenes/ObservabilityScene.unity`.
4. Set a collector endpoint or API key via env vars or PlayerPrefs (see above).
5. Press **Play**.

The Console should print, in order:

```
[XybridObservability] SDK initialized (version=...).
[XybridObservability] Export started -> http://localhost:4318/v1/traces
[XybridObservability] Inference succeeded (...ms, output=...).
[XybridObservability] Trace 4bf92f3577b34da6a3ce929d0e0e4736, span 00f067aa0ba902b7.
[XybridObservability] Feedback recorded: True.
```

On Stop (or on quit from a standalone build) you should additionally see:

```
[XybridObservability] Shut down on quit.
```

## Project structure

```
examples/unity/Observability/
├── Assets/
│   ├── Plugins/                       # Native libs (built from Rust, .gitignored)
│   ├── Scenes/
│   │   └── ObservabilityScene.unity   # Single GameObject + MonoBehaviour
│   └── Scripts/
│       └── XybridObservabilityExample.cs
├── Packages/
│   └── manifest.json                  # References ai.xybrid.sdk via file: URI
├── ProjectSettings/                   # Unity editor + player defaults
├── .gitignore
└── README.md                          # This file
```

## Related

- [Starter Example](../starter/README.md) — Minimal inference demo
- [Unity SDK README](../../../bindings/unity/README.md)
- [SDK API Reference](../../../docs/sdk/API_REFERENCE.md)

## License

MIT License - See [LICENSE](../../../LICENSE) for details.
