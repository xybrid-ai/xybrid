# @xybrid/react-native consumer smoke

Standalone Expo app that qualifies the **published** `@xybrid/react-native`
package (npm + Maven Central + release XCFramework) on current Expo.

Pinned versions (update deliberately after new releases):

| Package | Version |
|---------|---------|
| Expo | 58.0.25 |
| React Native | 0.88.1 |
| React | 19.3.0 |
| `@xybrid/react-native` | 0.18.0 |

On launch the app loads FunctionGemma 270M from a staged documents path, runs
version / load / info / warmup / inference / streaming / cancel / conversation /
constrained JSON / release checks, and writes
`xybrid-consumer-smoke-report.json` into the app sandbox. CI host runners
launch a Release build (embedded Hermes bundle — no Metro), collect that JSON,
and fail on missing, stale, incomplete, or failed reports.

Android `minSdkVersion` is 29 via `expo-build-properties`. iOS keeps Expo’s
higher minimum (the published pod declares 16.0).

## Local setup

```sh
cd bindings/react-native/consumer-smoke
npm ci
npx expo prebuild
npx expo run:ios   # or run:android
```

Stage a model directory under the app documents path as
`models/functiongemma-270m-it/` (weights + `model_metadata.json`) before
expecting the LLM steps to pass. CI does this with the shared `fetch-model`
action and the host runners under `tools/scripts/rn_consumer_smoke/`.

`ios/` and `android/` are generated and gitignored.
