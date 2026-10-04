# Rust GGUF browser experiment

Load SmolLM2-135M-Instruct Q4_0 in a webpage, stream tokens, cancel decoding,
and measure the existing Rust runtime's memory and speed. The GGUF is used
directly, with the repository's pinned llama.cpp and `xybrid-llama` wrappers.

This experiment covers `xybrid-llama` and `xybrid-llama-sys`. The full
`xybrid-core`, `xybrid-sdk`, and BoltFFI surfaces still need a browser platform
layer. The client here is an experimental TypeScript handle under `spike/`.

## Run

From `bindings/web`, with Bun and Bazelisk (`bazel`) on PATH:

```sh
pnpm install --frozen-lockfile
pnpm spike:prepare
pnpm spike:dev --host 127.0.0.1 --port 4174
```

Open <http://127.0.0.1:4174>, load the model, and generate. Stop cancels at a
token boundary. Release drops the Rust context and model, then terminates the
worker. A released model can be loaded again.

Preparation builds `//bindings/web/spike:runtime` through Bazel's pinned
Emscripten and Rust toolchains, stages the JS/WASM pair, and downloads a pinned
91,726,912-byte GGUF. Both preparation and the browser check its size and
SHA-256; the artifact identity is in [model.ts](model.ts). Generated assets and
the model are ignored by git. `./clean.sh` lists them; `./clean.sh --apply`
removes them.

For a static build, run `pnpm spike:build` after preparation and serve
`spike/dist` over HTTPS or localhost. The demo currently uses origin-root
asset paths (`/model.gguf` and `/runtime/...`). The CPU path uses one worker,
one inference thread, and WASM SIMD, with Asyncify for cooperative yielding.
It requires no SharedArrayBuffer or COOP/COEP headers.

## Browser calls

Inside a Vite app that includes this experiment and hosts the prepared assets:

```ts
import { RustGgufModel } from "./client.ts";

const model = await RustGgufModel.load({ accelerator: "wasm" });
try {
  for await (const delta of model.generateStream("Tell me a story.", { maxTokens: 32 })) {
    appendText(delta);
    // Breaking the loop cancels generation and waits for Rust to settle.
  }
  console.log(model.loaded, model.lastRun);
} finally {
  console.log(await model.dispose());
}
```

`model.cancel()` also requests cancellation and waits for decoding to settle.
Cancellation runs at the next yielded token; prompt prefill is synchronous
inside the worker. Overlapping generations are rejected. Each run clears the
KV cache, applies the GGUF's embedded chat template, and uses greedy decoding.
The context is 512 tokens and the output limit is 1–256 tokens.

## What crosses into Rust

```text
webpage → TypeScript worker → Emscripten exports → Rust Engine
        → existing xybrid-llama → existing C++ shim → pinned llama.cpp
        ← JavaScript yield ← Rust streaming trampoline ← token callback
```

Bazel links Rust and C++ into one WASM module with one linear memory and
allocator. Rust owns the real `LlamaModel` and `LlamaContext` RAII handles and
uses `format_chat`, `tokenize_special`, and `generate_streaming`. The reported
Rust version comes from `CARGO_PKG_VERSION` in that artifact.

The worker serializes all native operations. At each token, the JS bridge
copies the text and yields to the worker event loop. Cancellation changes a
JS flag while the Rust/C++ stack is suspended; the resumed Rust callback
returns its existing callback-abort error. This avoids reentering a mutably
borrowed Rust engine. Iterator exit and disposal wait for generation before
freeing the context and model.

Two portability fixes are relevant: the link enables C++ exception
catching for embedded chat templates, and the shared shim disables mmap under
Emscripten. llama.cpp trims mappings with partial `munmap`, which Emscripten
does not support; owned tensor buffers can be freed correctly. Bazel's
`rules_rs` patch corrects Emscripten static archives to `.a`. The Emsdk patch
uses its hermetic Python for cache preparation and its compiler subprocesses,
declares Dawn's headers through relative virtual include paths, and makes the
port cache hash consistent between repository preparation and sandboxed links.

## Measurements and validation

```sh
# Uses Playwright's installed Chromium by default.
# Set CHROME_PATH=/usr/bin/google-chrome to use an existing Chrome.
XYBRID_SPIKE_REPORT=/absolute/path/measurements.json pnpm spike:test
pnpm lint
pnpm typecheck
pnpm test
```

The browser suite loads the real GGUF, records three 32-token runs, exits a
256-token stream after five deltas, runs again after cancellation, and disposes
a paused stream. It also checks overlapping generation, checksum rejection,
the page's Stop button, and unavailable WebGPU capability reporting.

The report distinguishes download/checksum time, module initialization, model
loading, first-token latency, total generation time, and decode throughput.
Decode throughput is `(tokens - 1) / (last-token-time - first-token-time)` and
includes the per-token worker yield. Cancellation time measures iterator exit
until the native generation settles.

Memory reports WASM heap capacity and live `dlmalloc` allocations across Rust
and C++. The generation peak is sampled at token callbacks; it does not capture
temporary allocations during model loading or prefill. Heap capacity stays
allocated until worker termination even after live model allocations fall.
The figures exclude JS download/MEMFS buffers, browser process memory, and GPU
allocations. Use them to compare runtime allocation and release, rather than
as a browser tab's total memory requirement.

Measured on 2026-10-04 in Chrome 154.0.8037.97 on a Linux x86_64 cloud VM
(Intel Xeon 2.90 GHz, eight logical CPUs). Inference used one thread, WASM SIMD,
Rust artifact version 0.10.1, Bazel's Rust 1.92.0, and Emscripten 6.0.11.
The prompt contained 22 tokens; each run generated 32 tokens with a cleared
KV cache.

| Run | First token | Decode speed | Total generation |
| --- | ---: | ---: | ---: |
| 1 | 642 ms | 24.07 tokens/s | 1,971 ms |
| 2 | 583 ms | 25.35 tokens/s | 1,846 ms |
| 3 | 567 ms | 22.27 tokens/s | 2,004 ms |

Local HTTP fetch plus checksum took 488 ms, module initialization 46 ms, and
model loading 856 ms. Live WASM allocations were 165.9 MiB after load, peaked
at 166.6 MiB at a token callback, and fell to 2.2 MiB after Rust disposal.
WASM capacity stayed at 170.8 MiB until worker termination. Iterator exit
settled cancellation in 35.4 ms after five consumed deltas (six native tokens,
including one in flight); disposing a paused stream took 45.2 ms.

Both CPU and GPU artifacts built successfully. Six browser tests passed;
actual GPU generation was skipped for lack of a compatible adapter. The
production Vite bundle was also exercised in Chrome for generation, Stop,
and memory release.
Workspace formatting, Clippy, and `cargo test --workspace --features
ort-download` passed, along with the existing 90 web unit tests.

## WebGPU

```sh
pnpm spike:prepare --webgpu
XYBRID_SPIKE_WEBGPU=1 pnpm spike:test --grep WebGPU
```

This builds both CPU and WebGPU assets. The GPU variant compiles the same
llama.cpp's `ggml-webgpu` backend, embeds its WGSL shaders, and uses the pinned
Emdawnwebgpu port with Asyncify. Bazel materializes the port and required system
libraries before freezing the compiler cache. Select WebGPU in the demo to
request GPU layers, or `auto` to select it when a compatible adapter exists.
Prepare both variants before using `auto` on a GPU-capable browser.

GPU execution requires HTTPS/localhost and a WebGPU adapter with `shader-f16`.
GPU generation is an opt-in browser test. The artifact test can instantiate
the GPU module and call its Rust exports without initializing a GPU backend.
The cloud VM used for this experiment exposes a SwiftShader software
adapter without `shader-f16`; GPU execution and GPU performance still require
validation on suitable hardware.

Both variants use Asyncify to match the exception ABI of the pinned Rust
Emscripten standard library. A JSPI build with native WASM exceptions would
require a compatible Rust standard library; mixing the two currently fails
to link. Asyncify also avoids requiring JSPI support from the browser.

## Next step toward the SDK

The proof establishes browser compilation and execution of the current Rust
llama path. Integrating the public SDK needs feature separation for native
dependencies, browser model storage/loading, worker-aware async execution in
place of native Tokio blocking/thread pools, and a binding over the existing
facade. Registry, routing, telemetry, and control sync need browser networking
and lifecycle handling alongside local inference. Larger models, CPU threads,
other browsers, and actual GPU inference need their own memory and performance
validation.
