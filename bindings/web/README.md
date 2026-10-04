# @xybrid/web

Private browser SDK preview for local GGUF text generation. It compiles Xybrid's
existing `xybrid-llama` Rust wrappers, C++ shim, and pinned llama.cpp into one
WebAssembly module. A Web Worker streams tokens while the page stays responsive.
The package includes CPU/WASM SIMD and optional WebGPU execution.

## Build and run

Install Bun, pnpm, and Bazelisk (`bazel` on PATH), then from the repository root:

```sh
git submodule update --init vendor/llama-cpp
cd bindings/web
pnpm install --frozen-lockfile
pnpm build
pnpm dev:example
```

`pnpm build` builds both runtime variants through Bazel and packages the ESM
client, declarations, worker, and runtime assets in `dist/`. Bazel downloads the
pinned Rust/Emscripten toolchains; no system Emscripten installation is needed.
`pnpm runtime:build` builds only the Rust/C++ artifacts.

The example downloads and verifies a pinned SmolLM2-135M-Instruct Q4_0 GGUF
(91,726,912 bytes). `pnpm build:example` creates a static site in `example/dist`.
`pnpm exec vite preview --config example/vite.config.ts` serves that build.
Serve over HTTPS or localhost. The single-thread CPU build needs no
SharedArrayBuffer or COOP/COEP headers. Generated assets and models are ignored;
`./clean.sh` lists build outputs and `./clean.sh --apply` removes them.

## Use in an application

Import `XybridLlm` from `@xybrid/web` in your bundler. Keep `dist/worker.js` with
the package; the client loads it relative to its own module. Copy the package's
`dist/runtime/` directory into your application's public `/xybrid/runtime/`
directory, or set `wasmPath` to another directory on the page's origin.
That directory must contain both `wasm/xybrid_runtime.{js,wasm}` and
`webgpu/xybrid_runtime.{js,wasm}` when using `auto` or WebGPU. Only the selected
backend is fetched. Serve WASM as `application/wasm` and enable gzip or Brotli.

The built ESM files bundle their JavaScript dependencies, so they can also be
hosted directly: copy all of `dist/` to `/sdk/` and import `/sdk/index.js`, with
`wasmPath: "/sdk/runtime"`. The production example and browser tests exercise
these packaged files.

```ts
import { XybridLlm } from "@xybrid/web";

const model = await XybridLlm.fromUrl("/models/model.gguf", {
  accelerator: "auto",
  wasmPath: "/xybrid/runtime",
  contextLength: 512,
  onDownloadProgress: ({ loadedBytes, totalBytes }) => {
    console.log(loadedBytes, totalBytes);
  },
});
try {
  for await (const delta of model.generateStream("Tell me a short story.", {
    maxOutputTokens: 64,
  })) {
    output.append(delta);
    // Breaking the loop cancels decoding and waits for native generation to settle.
  }
  console.log(model.loaded, model.lastRun);
} finally {
  await model.dispose();
  console.log(model.releasedMemory);
}
```

`generate(prompt, options?)` returns the complete string. `cancel()` requests a
stop and waits for native generation to settle. Concurrent generations on one
model are rejected. `dispose()` cancels active work, drops the Rust model/context,
and terminates the worker; it is idempotent.

Generation currently uses an embedded GGUF chat template, a fresh single-user
turn with a cleared KV cache, and greedy decoding. External templates,
metadata preprocessing/postprocessing, sampling metadata, vision, and speech are
rejected. `maxOutputTokens` defaults to 64 and must fit the context together with
the prompt. Direct loads default to a 512-token context; `contextLength` accepts
positive integers through 32,768. Cancellation is cooperative at token
boundaries; prompt prefill runs synchronously inside the worker.

## Model sources and verification

- `XybridLlm.fromUrl(ggufUrl, options?)`: direct HTTP(S) GGUF. Supply `sizeBytes`
  to enforce an exact byte count and optional `sha256` (64 lowercase hex digits)
  to verify integrity. A hash requires a declared size.
- `XybridLlm.load(metadataUrl, options?)`: Xybrid metadata with a `Gguf` template.
  `model_file` must be a listed, bare `.gguf` filename beside the metadata.
- `XybridLlm.fromRegistry(id, options?)`: requests `platform=web&format=gguf`,
  preserves the anonymous `X-Xybrid-Client` binding header, and requires an HTTPS
  download URL, exact size, SHA-256, and compatible `Gguf` metadata.
- `XybridLlm.fromHuggingFace(repo, options?)`: selects a top-level GGUF using
  `file` and optional `revision`. It uses repository metadata when present, or
  synthesizes `Gguf` metadata when the selection is unambiguous. It verifies the
  declared size and, when provided, a valid Git-LFS SHA-256 OID.

```json
{
  "model_id": "my-model",
  "version": "1",
  "execution_template": {
    "type": "Gguf",
    "model_file": "model.gguf",
    "context_length": 512
  },
  "files": ["model.gguf"],
  "preprocessing": [],
  "postprocessing": []
}
```

Load options include `signal` for aborting resolution, download, or worker
initialization. Executable runtime assets stay on the page's origin; cross-origin
model servers need CORS. Metadata is limited to 1 MiB and models to 512 MiB.
Public failures use the exported `XybridError` subclasses and `code` values.

## Measurements and CI

`loaded` reports actual Rust artifact version, chosen accelerator, model bytes,
download/checksum time, module initialization, load time, and WASM memory.
`lastRun` reports token counts, first-token latency, total time, decode tokens per
second, cancellation, and sampled peak memory. Decode speed excludes the first
token and includes worker yields. `releasedMemory` records allocations after
Rust destruction, before worker termination.

`Memory.heapBytes` is WASM linear-memory capacity; `allocatedBytes` is live
Rust/C++ allocator usage. Heap capacity does not shrink when models are freed.
These figures exclude JavaScript buffers, browser process memory, GPU memory,
and temporary allocation peaks during loading or prefill.

```sh
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm build:example
pnpm exec playwright install --with-deps chromium
XYBRID_WEB_REPORT=test-results/runtime-measurements.json pnpm test:browser
pnpm size
```

Required CI builds CPU and WebGPU artifacts, runs the production example and
packaged SDK in Chromium, and verifies streaming, iterator cancellation,
overlapping generation, reuse, checksum failure, model release, and GPU Rust
exports. It caches the GGUF and Bazel outputs and uploads `dist/`, measurements,
and raw/gzip sizes. Throughput is informational on shared runners.

Actual GPU inference requires an adapter with `shader-f16`. Run
`XYBRID_WEB_WEBGPU=1 pnpm test:browser --grep 'WebGPU generates'` on suitable
hardware, or dispatch `.github/workflows/web-webgpu.yml` with a self-hosted runner
labelled `webgpu`. The cloud VM's software adapter lacks this feature; actual GPU
inference and GPU speed remain unvalidated here.

## Migration from the LiteRT preview

The runtime dependencies and tensor `XybridModel` API were removed. Use
`XybridLlm` with GGUF and `Gguf` metadata in place of `.litertlm`/`LiteRtLm`.
`wasmPath` now defaults to `/xybrid/runtime`. Existing registry/Hugging Face,
streaming, generation, load cancellation, and typed-error surfaces remain.
The former `spike:*` commands are replaced by the normal build/example/test
commands above.

This is the browser binding of Xybrid's Rust llama inference path. The complete
`xybrid-core`/`xybrid-sdk`/BoltFFI API, pipelines, auth, cloud routing, and telemetry
export still need a browser platform layer. The package remains private and is
not published to npm.
