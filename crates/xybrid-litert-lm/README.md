# xybrid-litert-lm

Safe Rust bindings to Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
runtime, through its C API. LiteRT-LM runs `.litertlm` language models, such as
Gemma 4, on CPU and GPU.

`xybrid-core` uses this crate for its `llm-litert` feature. It has no xybrid
dependencies and can be used on its own.

## Nothing is linked

The runtime is a shared library that this crate opens at run time. Building
with it needs no C++ toolchain and downloads nothing, and a program that
includes it still starts where the library is absent:
`Library::open_default()` then returns an error for which
`Error::is_runtime_unavailable()` is true.

The library is looked up in this order:

1. the file named by `XYBRID_LITERT_LM_LIBRARY`;
2. the platform's default name, through the system loader:
   `liblitert-lm.so` (Linux, Android), `liblitert-lm.dylib` (macOS, iOS),
   `litert-lm.dll` (Windows). On Android that resolves inside the app's own
   native library directory, so the app has to package the library.

## Pinned interface

The declarations in `src/sys.rs` are written by hand against **C API 1.0.0**,
which ships with **LiteRT-LM v0.19.1**. Upstream publishes the headers and
prebuilt libraries for Android, Linux, macOS and Windows as
`litert_lm_c_api-1.0.0.zip` on its
[release page](https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.19.1)
(SHA-256 `1d430d80eaf536f696a4922270ba55eb3d112189ac198c3066d487a45e3b4d4c`).

A library that reports another C API major version is refused. The C API
version says nothing about the runtime release behind it, and no call reports
the release, so which releases are trusted is the embedder's decision.

Behaviour of v0.19.1 that the bindings account for:

- Its CPU path implements one sampler, top-p. Greedy decoding is therefore
  expressed as top-p over a single candidate (`Sampling::greedy`).
- Thinking is on unless a conversation turns it off, so
  `ConversationConfig` always sets it explicitly (off by default).
- The API does not say when stream callbacks stop. Callback data is an id
  looked up in a table, not a pointer, so a callback that arrives after its
  stream is gone does nothing (see `src/stream.rs`).
- A final chunk that carries an error does not mean the runtime stopped: it
  marks every error final, including one it reports while still decoding. A
  stream that ended with an error is therefore cancelled before its
  conversation is deleted.
- Its wait has a deadline of its own and can fail. When it does, or when a
  cancelled stream never ends, nothing the runtime may still use is freed:
  the conversation and its engine are leaked, and the engine and every
  conversation on it refuse further work (`Engine::is_usable`).

## Streams

`Conversation::send_stream` returns a `Stream` that borrows the conversation.
The conversation, not that handle, owns the running stream: until the stream
is finished or dropped the conversation refuses other calls, and dropping the
conversation tears a running stream down. Nothing depends on the handle's
destructor, so `mem::forget` on it is safe.

After a cancel, or a stream that ended with an error, the conversation can
only be dropped, which is upstream's rule for a cancelled conversation.

## Tests

```bash
cargo test -p xybrid-litert-lm --features test-util
```

runs the unit tests and `tests/fake_runtime.rs`, which drives the stream
bridge against a scripted runtime inside the test process: cancels, errors,
a stream that never ends, a wait that fails, a forgotten stream handle,
callbacks after teardown. No model or native library is needed. Without the
feature the fake-runtime tests are not built, and the command still passes.

They also pass under Miri:

```bash
MIRIFLAGS=-Zmiri-ignore-leaks cargo +nightly miri test -p xybrid-litert-lm --features test-util --test fake_runtime
```

(`-Zmiri-ignore-leaks` because several tests check that a stream the runtime
never ended is leaked on purpose.)

### The scripted runtime

The `test-util` feature adds `xybrid_litert_lm::testing`. A `FakeRuntime`
implements every C function this crate calls and plays a `Script` chosen by
the message sent: a message that contains `script:<name>` runs the script
registered under that name. It records the settings and turns it was given
and the teardown calls it received, so a test can assert on both sides.

```rust
use xybrid_litert_lm::testing::{FakeRuntime, Script, Step};

let fake = FakeRuntime::new();
fake.script("hello", Script::new([Step::text("Hi"), Step::End]));
let engine = fake.engine();
```

`FakeRuntime::at(path)` registers a fake under a model path of the caller's
choosing, and `testing::library_for_model(path)` returns the fake library for
a registered path. An embedder that asks it before opening the real library
can run its own tests on the fake; `xybrid-core` does.

`tests/real_runtime.rs` runs against the real library and a real model, and
skips unless both are named:

```bash
XYBRID_LITERT_LM_LIBRARY=/path/to/liblitert-lm.dylib \
XYBRID_LITERT_LM_TEST_MODEL=/path/to/model.litertlm \
  cargo test -p xybrid-litert-lm --test real_runtime -- --test-threads=1
```

The runtime writes compiled-model caches beside the model file (about 1 GB
for Gemma 4 E2B) unless `EngineSettings::with_cache_dir` names another
directory; the test takes one from `XYBRID_LITERT_LM_CACHE_DIR`.

### Android emulator on Apple silicon

Upstream's arm64 Android library runs SME instructions when the CPU reports
SME. The emulator on an M4 Mac reports it and then faults on them
(`SIGILL` in `liblitert-lm.so`). Start the emulator with SME hidden from the
guest:

```bash
emulator -avd <name> -qemu -append arm64.nosme
```
