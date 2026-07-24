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
  --target StandaloneWindows64 \
  --execute-method Builder.PerformBuild \
  --allow-dirty-build
```

The Windows CI rung is a manually dispatched job in `bazel.yml`. It runs only
when the repository variable `UNITY_WINDOWS_SMOKE_ENABLED` is `true` and uses
the `UNITY_LICENSE` repository secret. Set it to the complete contents of the
`.ulf` file created by a locally activated Unity installation. On macOS, Unity
Hub normally writes that file to:

```text
/Library/Application Support/Unity/Unity_lic.ulf
```

The workflow writes the secret to a runner-temporary file only for activation,
then deletes that file. A Unity service account cannot activate an Editor
license and is not used by this smoke.
