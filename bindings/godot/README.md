# Xybrid Godot SDK

Godot 4.5+ GDExtension binding for Xybrid. The binding is implemented in Rust with the `godot` crate from godot-rust/gdext and routes all SDK translation through `xybrid-ffi-facade`.

## Requirements

- Godot 4.5 or newer.
- Rust stable new enough for `godot = 0.5.4`.
- Desktop targets are supported in v1: macOS, Linux, and Windows.
- macOS packaging uses an Apple framework, matching Godot's export layout.

Android, iOS, and Web are intentionally not part of this first binding because godot-rust still treats those paths as experimental or under-documented.

## Build

From the repository root:

```bash
tools/scripts/build-godot.sh debug
tools/scripts/build-godot.sh release
```

The script builds `xybrid-godot` and copies the native artifact into `bindings/godot/addons/xybrid/bin/<platform>/<profile>/`.

On macOS the artifact is not a bare `.dylib`. The script builds both `aarch64-apple-darwin` and `x86_64-apple-darwin`, combines them with `lipo`, and packages the result as:

```text
bindings/godot/addons/xybrid/bin/universal-apple-darwin/<profile>/Xybrid Godot.framework
```

Install the Rust targets first if needed:

```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin
```

For local single-architecture testing on macOS, override the target list:

```bash
XYBRID_GODOT_MACOS_TARGETS=aarch64-apple-darwin tools/scripts/build-godot.sh debug
```

Set `XYBRID_GODOT_SKIP_CODESIGN=1` to skip ad-hoc signing during local experiments.

For local development you can also run:

```bash
cargo build -p xybrid-godot --features platform-desktop
```

Use `platform-macos` on macOS when you want CoreML/Metal features linked.

## Install

Copy `bindings/godot/addons/xybrid` into your Godot project as `res://addons/xybrid`.

The included `xybrid.gdextension` file declares `compatibility_minimum = 4.5` and points to the packaged desktop libraries. On macOS it also declares the framework as an export dependency so Godot places it under `Contents/Frameworks` in exported apps.

## Quick Start

```gdscript
var runtime := XybridRuntime.new()
var init_result := runtime.init("", "", "", OS.get_user_data_dir().path_join("xybrid-cache"))
if not init_result.ok:
	push_error(init_result.message)
	return

var loader_result := runtime.model_from_registry("kokoro-82m")
if not loader_result.ok:
	push_error(loader_result.message)
	return

var loader: XybridModelLoader = loader_result.value
var model_result := loader.load()
if not model_result.ok:
	push_error(model_result.message)
	return

var model: XybridModel = model_result.value
var run_result := model.run({"kind": "text", "text": "Hello from Godot"}, {}, {})
if run_result.ok:
	print(run_result.value.text)
else:
	push_error(run_result.message)
```

`load`, `run`, `warmup`, and `unload` are blocking calls. Use `addons/xybrid/xybrid_async.gd` when calling them from gameplay UI.

## Dictionary Shapes

Fallible calls return:

```gdscript
{"ok": true, "value": value}
{"ok": false, "code": 6, "retryable": false, "message": "Inference failed: ..."}
```

Envelope input:

```gdscript
{"kind": "text", "text": "prompt", "role": "user"}
{"kind": "audio", "bytes": PackedByteArray(...)}
{"kind": "embedding", "values": PackedFloat32Array(...)}
{"kind": "image", "bytes": PackedByteArray(...), "format": "png"}
{"kind": "multipart", "parts": [text_envelope, image_envelope]}
```
