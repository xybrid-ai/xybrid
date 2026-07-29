# Unity runtime smoke

This minimal project validates the committed Unity package against a real native
plugin. It is intentionally separate from the distributable package.

The validation has three rungs:

1. EditMode tests compile and execute the package inside the Unity Editor.
2. `Builder.PerformBuild` AOT-compiles the full package into an IL2CPP player
   with managed stripping disabled.
3. Running that player calls the native Bolt library through the public
   `Xybrid` API and exits non-zero on failure.

The project references `bindings/unity` through a relative local-package path.
Stage the platform's `xybrid_bolt` library and ONNX Runtime into that package
before running the smoke.

```bash
unity test tools/unity-runtime-smoke --mode EditMode \
  --output tools/unity-runtime-smoke/test-results.xml

unity build tools/unity-runtime-smoke \
  --target StandaloneLinux64 \
  --execute-method Builder.PerformBuild \
  --allow-dirty-build
```

## Linux — hosted IL2CPP validation

`.github/workflows/unity-editor.yml` runs all three rungs on relevant pull
requests and pushes to `master`. After the EditMode tests pass, GameCI builds
the smoke project with a Unity 6000.3.14f1 Linux IL2CPP image. The workflow
then launches this player headlessly:

```bash
./tools/unity-runtime-smoke/Build/linux-il2cpp/XybridSmoke.x86_64 \
  -batchmode -nographics \
  -logFile "${RUNNER_TEMP}/linux-il2cpp-player.log"
```

The gate requires both a zero process exit code and `[XybridSmoke] OK` in the
player log. The log is uploaded on success or failure.

## Windows — one-time manual validation

The Windows smoke is **run by hand on a Windows box that's already
Unity-licensed**, not in hosted CI. Unity licensing on an ephemeral
GitHub-hosted Windows runner is not viable: service accounts cannot activate
Editor licenses, and a locally activated `.ulf` is machine-bound
(`Machine bindings don't match` on a fresh runner image). The
`Bazel windows bolt DLL smoke (managed C# P/Invoke)` job in `bazel.yml`
already proves the Windows DLL loads and round-trips a real boltffi call under
the CLR on `windows-latest`; the rungs below add Unity's plugin import + IL2CPP
AOT path, which require a real licensed editor.

This is the last gate before flipping `build-unity.yml` from cargo/MSVC to
Bazel/MinGW for the Windows Unity native. Run it once, paste `[XybridSmoke] OK`,
then flip the producer.

### 1. Obtain the Bazel-built Windows DLL

Either download the `bazel-xybrid-bolt-windows-dll` artifact from a green
`bazel.yml` run on `master`, or build it locally:

```bash
bazelisk build --config=remote --config=windows -c opt \
  //crates/xybrid-bolt:xybrid_bolt_cdylib
```

### 2. Stage the native + ONNX Runtime into the Unity package

From the repo root (the smoke project references `bindings/unity` via a
relative local-package path):

```bash
python tools/scripts/stage_unity_native.py \
  --lib <path-to-xybrid_bolt.dll> \
  --target x86_64-pc-windows-gnu

python tools/scripts/stage_unity_desktop_ort.py windows \
  bindings/unity/Runtime/Plugins/Windows

# Suppress the package's download resolver so it can't swap the staged native
# for a release asset during the smoke.
VERSION=$(python -c "import json; print(json.load(open('bindings/unity/package.json'))['version'])")
mkdir -p tools/unity-runtime-smoke/Assets/Xybrid/Plugins
printf '%s\n' "$VERSION" \
  > tools/unity-runtime-smoke/Assets/Xybrid/Plugins/.xybrid-native-windows-version
```

### 3. Run EditMode, then build + run the IL2CPP player

```bash
unity test tools/unity-runtime-smoke --mode EditMode \
  --output tools/unity-runtime-smoke/editmode-results.xml

unity build tools/unity-runtime-smoke \
  --target StandaloneWindows64 \
  --execute-method Builder.PerformBuild \
  --allow-dirty-build
```

Then launch the built player and require `[XybridSmoke] OK`:

```bash
./tools/unity-runtime-smoke/Build/windows-il2cpp/XybridSmoke.exe \
  -batchmode -nographics -logFile tools/unity-runtime-smoke/Logs/player.log
```

The macOS equivalent (the same project, `--target StandaloneOSX`) was the
validation used during the Unity-on-bolt migration and is green: 24/24 EditMode
+ an IL2CPP player that called the native Bolt API and printed `[XybridSmoke] OK`.
