# @xybrid/web preview

`@xybrid/web` is the future umbrella browser package for Xybrid. Version `0.3.0` is a private preview that currently provides a low-level LiteRT tensor surface backed by `@litertjs/core` `2.5.2`.

## Supported now

- `XybridModel.load(metadataUrl, { wasmPath, accelerator })` for Xybrid metadata whose execution template is exactly `{ "type": "TfLite", "model_file": "..." }`.
- Explicit `wasm` or `webgpu` compilation, plus `auto` that uses wasm when WebGPU is unavailable.
- Positional arrays or name-keyed records of `Float32Array`, `Int32Array`, and `Uint8Array` inputs.
- Model-derived readonly tensor details, the selected compile path, and `isFullyAccelerated`.
- Copied typed-array outputs, WebGPU device-loss protection, rejected overlapping runs, and idempotent asynchronous disposal.
- Browser memory guards: 1 MiB metadata, 512 MiB models, and 256 MiB input/output tensors.

Metadata is parsed with a forward-compatible boundary: unrecognized fields are accepted, while malformed metadata, unsupported templates, and unavailable browser features become typed Xybrid errors. `model_file` must be a listed, bare relative filename in the same directory as the metadata document.

## Deliberately deferred

- Preprocessing and postprocessing pipelines.
- Voices and `vision_encoder` metadata.
- Tensor dtypes beyond float32, int32, and uint8.
- Threads, JSPI, WebNN, GPU buffer I/O, model registries, and higher-level multimodal APIs.
- LiteRT-LM and all token-generation surfaces.

## LiteRT wasm assets

LiteRT loads its wasm JavaScript and binary assets at runtime. Host the `@litertjs/core/wasm` directory and supply its URL through `wasmPath`. The example copies these files to `/litert` with `bun run example:assets`.

Metadata and model files may use another origin when that host permits browser CORS requests. LiteRT's wasm path is executable code and must remain on the page's own HTTP(S) origin.

LiteRT may execute unsupported WebGPU operations on CPU. `model.accelerator` reports the compile path selected by this wrapper; inspect `model.isFullyAccelerated` to distinguish a fully delegated graph from one with CPU fallback.

LiteRT initialization is a per-page singleton. Concurrent loads with the same wasm configuration share the one initialization; a later request with another wasm path fails with `RuntimeConfigurationError`. This preview always sets `threads: false` and `jspi: false`, so it has no SharedArrayBuffer or COOP/COEP requirement.

## Example

From `bindings/web`:

```sh
pnpm install
pnpm dev:example
```

The predev script downloads the pinned 708-byte LiteRT addition model, verifies SHA-256 `1317a76ceedc6e0a2b39c4ee2802f80b3b831b16ac96a99e48540113472aaee2`, and copies LiteRT wasm assets. No model binary is committed.

The demo page runs named model inputs `a` and `b` as `float32[10,10]` and displays a text PASS only when all 100 `Identity` values equal elementwise `a+b`. It walks through load, run, and verify with live timings, lets you choose the accelerator and the value of `b`, reports the compile path and delegation status, renders the 10 by 10 output grid, and shows the exact `@xybrid/web` calls it makes.

`pnpm test:browser` runs the same pinned model through the real LiteRT adapter in Chromium.

## API

```ts
import { XybridModel } from "@xybrid/web"

const model = await XybridModel.load("https://example.test/model_metadata.json", {
  wasmPath: "/litert",
  accelerator: "auto",
})

const result = await model.run({
  a: new Float32Array(100),
  b: new Float32Array(100),
})

console.log(result.byName["Identity"]?.data)
await model.dispose()
```
