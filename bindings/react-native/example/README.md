# Xybrid RN example

A small RN 0.76 app that exercises `react-native-xybrid`'s init → load → run
path and reports per-step latency. Use it to smoke-test the binding locally.

## What's committed vs. generated

Committed:

- `App.tsx`, `index.js`, `app.json` — the actual demo
- `package.json`, `metro.config.js`, `babel.config.js`, `tsconfig.json` — wiring
- `bootstrap.sh` — generates the host shell

Gitignored (regenerated on demand):

- `ios/`, `android/` — produced by `bootstrap.sh` from the upstream RN template

This split keeps the diffable surface focused on the example's own code,
not on Xcode project blobs and gradle wrappers that come straight from
`@react-native-community/cli`.

## Running locally

Pre-req: the parent package must have its native artifacts staged.

```bash
# From the repo root, build the iOS XCFramework + Android .so files and
# stage them into bindings/react-native/{ios,android}/.
cargo xtask build-react-native --release
```

Then bootstrap and run the example:

```bash
cd bindings/react-native/example
./bootstrap.sh                 # one-time: scaffolds ios/ and android/
npm install
# Android:
npm run android
# iOS (macOS only):
cd ios && pod install && cd ..
npm run ios
```

Tap **Run smoke test** in the app. Each step (initialize, load, run) is
timed and printed; failures bubble up with the underlying error code.

## Changing the model

The default model ID (`whisper-tiny`) is editable in the input field.
Anything resolvable by the Xybrid registry works — try `kokoro-82m` for TTS
and the example switches to a voices probe instead of a text run.

## Updating the template

If `RN_VERSION` in `bootstrap.sh` is bumped, delete and re-bootstrap:

```bash
./bootstrap.sh --force
```
