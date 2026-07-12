# @xybrid/web preview

`@xybrid/web` is the future umbrella browser package for Xybrid. Version `0.3.0` is a private preview with two surfaces: a low-level LiteRT tensor surface backed by `@litertjs/core` `2.5.2`, and a text-generation surface backed by `@litert-lm/core` `0.14.0`.

## Supported now

### Tensor surface (`XybridModel`)

- `XybridModel.load(metadataUrl, { wasmPath, accelerator })` for Xybrid metadata whose execution template is exactly `{ "type": "TfLite", "model_file": "..." }`.
- Explicit `wasm` or `webgpu` compilation, plus `auto` that uses wasm when WebGPU is unavailable.
- Positional arrays or name-keyed records of `Float32Array`, `Int32Array`, and `Uint8Array` inputs.
- Model-derived readonly tensor details, the selected compile path, and `isFullyAccelerated`.
- Copied typed-array outputs, WebGPU device-loss protection, rejected overlapping runs, and idempotent asynchronous disposal.

### Text generation surface (`XybridLlm`)

- `XybridLlm.load(metadataUrl, { wasmPath, accelerator, onDownloadProgress? })` for metadata whose execution template is `{ "type": "LiteRtLm", "model_file": "...", "context_length"? }`.
- `generate(prompt, { maxOutputTokens? })` for one-shot replies and `generateStream(...)` for an async iterator of text deltas; each call is a fresh single-turn conversation and the model's embedded prompt template is applied by LiteRT-LM.
- `webgpu` maps to the LiteRT-LM GPU backend and `wasm` to its CPU backend; `auto` prefers WebGPU and falls back to CPU. The engine loads models through the wasm filesystem rather than the default streaming path, because streaming cannot read the zlib-compressed tokenizer sections used by current `litert-community` `.litertlm` files.
- Rejected overlapping generations, cancellation of in-flight decoding on early iterator exit or `dispose()`, and idempotent asynchronous disposal.

Shared behavior: browser memory guards (1 MiB metadata, 512 MiB models, and 256 MiB tensor I/O), and a forward-compatible metadata boundary — unrecognized fields are accepted, while malformed metadata, unsupported templates, and unavailable browser features become typed Xybrid errors. `model_file` must be a listed, bare relative filename in the same directory as the metadata document.

## Deliberately deferred

- Preprocessing and postprocessing pipelines.
- Voices and `vision_encoder` metadata.
- Tensor dtypes beyond float32, int32, and uint8.
- Threads, JSPI, WebNN, GPU buffer I/O, model registries, and higher-level multimodal APIs.
- Multi-turn conversations, system prompts, sampling controls, constrained decoding, and tool calling on the LiteRT-LM surface.

## Wasm assets

Both engines load their wasm JavaScript and binary assets at runtime, and each takes its own `wasmPath`: host `@litertjs/core/wasm` for the tensor surface and `@litert-lm/core/wasm` for the text surface. The example copies them to `/litert` and `/llm-runtime` with `bun run example:assets`. The underlying engines are an implementation detail of this preview — application code and UI copy should only speak in terms of `@xybrid/web`.

Metadata and model files may use another origin when that host permits browser CORS requests. Wasm paths are executable code and must remain on the page's own HTTP(S) origin.

LiteRT may execute unsupported WebGPU operations on CPU. `model.accelerator` reports the compile path selected by this wrapper; inspect `model.isFullyAccelerated` to distinguish a fully delegated graph from one with CPU fallback.

Each runtime's initialization is a per-page singleton. Concurrent loads with the same wasm configuration share the one initialization; a later request with another wasm path fails with `RuntimeConfigurationError`. This preview always sets `threads: false` and `jspi: false`, so it has no SharedArrayBuffer or COOP/COEP requirement.

## Example

From `bindings/web`:

```sh
pnpm install
pnpm dev:example
```

The predev script downloads two pinned models, verifies their SHA-256 checksums, and copies both runtimes' wasm assets. No model binary is committed.

- `/` runs SmolLM2-135M-Instruct, a 136 MB `.litertlm` language model from [litert-community](https://huggingface.co/litert-community/SmolLM2-135M-Instruct), streaming a reply on WebGPU (or the CPU engine) with live download progress, first-token latency, and generation timings, plus the exact `@xybrid/web` calls it makes.
- `/tensor.html` keeps the deterministic diagnostic: the pinned 708-byte LiteRT addition model runs named inputs `a` and `b` as `float32[10,10]` and shows PASS only when all 100 `Identity` outputs equal elementwise `a+b`.

`pnpm test:browser` runs both pages through the real adapters in Chromium.

## API

```ts
import { XybridLlm, XybridModel } from "@xybrid/web"

const llm = await XybridLlm.load("https://example.test/llm/model_metadata.json", {
  wasmPath: "/llm-runtime",
  accelerator: "auto",
  onDownloadProgress: ({ loadedBytes, totalBytes }) => render(loadedBytes, totalBytes),
})

for await (const delta of llm.generateStream("Hello!", { maxOutputTokens: 256 })) {
  append(delta)
}
await llm.dispose()

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
