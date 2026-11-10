# Xybrid Python SDK

> **Status**: Active - generated BoltFFI bindings plus a Pythonic SDK wrapper.

This directory contains the Python package for Xybrid, providing local-first
model inference through the Rust `xybrid-bolt` native library. Requires
Python >= 3.10; `pytest` is used for tests.

## Installation

### Local Development

Build the native artifacts and stage them into the package:

```bash
# From the xybrid repo root
./tools/scripts/build-python-bolt.sh

# LLM (GGUF) support is not in the platform-* presets — opt in:
XYBRID_FEATURES=platform-macos,llm-llamacpp ./tools/scripts/build-python-bolt.sh
```

Install the package in editable mode:

```bash
python -m pip install -e bindings/python
```

That stages two binaries into `xybrid/_bolt/` — the `_native` CPython extension
and the `libxybrid_bolt` cdylib it loads. Both are build outputs and are
git-ignored; without them, importing `xybrid` raises `ImportError`.

## Quickstart

```python
import xybrid

xybrid.init()

model = xybrid.XybridModel.from_registry("your-model-id")
result = model.run(xybrid.XybridEnvelope.text("Hello from Python"))

print(result.text)
```

Pass an API key to configure cloud authentication:

```python
xybrid.init(api_key="sk_live_...")
```

Initialization is optional for local inference. An API key alone does not enable
observability export or replace an existing exporter. The first successful
`init()` applies its arguments; subsequent calls are no-ops.
Key prefixes and scopes, what gets exported, and provisional workspaces:
[API keys](https://docs.xybrid.dev/en/docs/concepts/api-keys).

Opt in to observability separately, using an export key independent of cloud
authentication:

```python
config = xybrid.XybridObservabilityConfig.xybrid("export-key")
config.set_metadata_only(True)  # omit prompts, outputs, and tool payloads
config.init()
# Before exiting, wait up to five seconds for queued exports and stop exporting.
xybrid.observability_shutdown(5000)
```

The Xybrid preset sends bearer-authenticated OTLP/HTTP protobuf traces. For a
Collector exporting traces, metrics, and logs, use
`XybridObservabilityConfig.otlp("http://localhost:4318")`. New explicit configs
capture content by default; call `set_metadata_only(True)` to exclude it.
`observability_flush(5000)` flushes without stopping export.

`observability_stats()` reports `accepted`, `exported`, `dropped`, `failed` and
`pending` counts per signal (`traces`, `logs`, `metrics`); after
`observability_shutdown` they are the session's final counters (what the
shutdown failed or dropped included) until the next `init()`. When a flush or
shutdown cannot deliver everything it raises `xybrid.TelemetryDelivery`: the
inference already succeeded, so do not rerun it.

Runs, pipelines and loads take an explicit W3C parent through `traceparent` on
`XybridRunOptions` or `XybridObservationOptions`, and return references
(`result.observation`, each pipeline stage's `observation`,
`model.load_observation_ref()`). For work Xybrid does not execute itself, start
a manual operation and pass its `context()` to children. As a context manager
it completes on success, fails with the label `application_error` (never the
exception text) on an exception, and cancels on `KeyboardInterrupt` or task
cancellation; closing or collecting an unfinished one cancels it without
flushing. The first terminal call wins.

```python
with xybrid.XybridWorkflowObservation("answer-question", None) as workflow:
    context = workflow.context()  # None while export is off
    child = None if context is None else xybrid.XybridObservationOptions(
        traceparent=context.traceparent,
        tracestate=context.tracestate,
        conversation_id=None,
        suppress_trace=False,
    )
    call = xybrid.XybridToolCallInfo("search", None, None, None, '{"q": "otel"}')
    with xybrid.XybridToolObservation(call, child) as tool:
        tool.record_result('{"hits": 3}')
```

[`examples/observability.py`](examples/observability.py) runs this against a
local Collector without downloading a model.

The retained `init(api_key="...", ingest_url="https://collector.example")`
compatibility form explicitly enables **metadata-only OTLP traces**, using that
API key for both cloud authentication and export. A base ingest URL maps to
`/v1/traces`; a full endpoint path is preserved. The destination must now accept
OTLP/HTTP protobuf, replacing the retired JSON event protocol. A blank ingest
URL leaves export settings alone. A nonblank URL requires a nonblank explicit
API key and an absolute HTTP(S) URL; invalid configuration raises `ConfigError`
and initialization can be retried. Use the separate config API for independent
export credentials or an unauthenticated Collector.

A runnable LLM example lives in [`examples/quickstart.py`](examples/quickstart.py):

```bash
python bindings/python/examples/quickstart.py
```

## Text to Speech and Voices

```python
import xybrid

xybrid.init()
model = xybrid.XybridModel.from_registry("kokoro-82m")

for voice in model.voices():
    label = "female" if voice.is_female else "male" if voice.is_male else "voice"
    print(voice.id, voice.name, label)

envelope = xybrid.XybridEnvelope.text(
    "This is generated on device.",
    voice="af_heart",
    speed=1.0,
)
result = model.run(envelope)

audio = result.audio_bytes
if audio is not None:
    Path("speech.wav").write_bytes(audio)
```

## Generation Config

```python
import xybrid

config = xybrid.GenerationConfigs.creative()
options = xybrid.XybridRunOptions(
    generation_config=config,
    abort_on=[],
    fallback_to_cloud=False,
    max_grace_tokens=0,
    correlation_id="example-run",
    traceparent=None,
    tracestate=None,
    conversation_id=None,
    suppress_trace=False,
)

result = model.run(xybrid.XybridEnvelope.text("Write a haiku"), options)
print(result.text)
```

Use `GenerationConfigs.greedy()` for deterministic decoding
(`temperature=0.0`, `top_p=1.0`, `top_k=0`) or
`GenerationConfigs.creative()` for a higher-temperature preset
(`temperature=0.9`, `top_p=0.95`, `top_k=50`).

## Multimodal Input

```python
image = xybrid.XybridEnvelope.image(image_bytes, format="jpg")
prompt = xybrid.XybridEnvelope.user_message(
    "Describe this image",
    images=[image],
)
result = model.run(prompt)
print(result.text)
```

Image formats are normalized up front (`jpg` becomes `jpeg`) and limited to
`png`, `jpeg`, and `webp`.

## Error Handling

```python
import xybrid

try:
    model = xybrid.XybridModel.from_registry("missing-model")
except xybrid.ModelNotFound as exc:
    print(f"Unknown model: {exc.id}")
except xybrid.XybridError as exc:
    print(f"Xybrid failed: {exc}")
```

All native fallible calls raise typed `XybridError` subclasses such as
`ModelNotFound`, `DirectoryNotFound`, `ConfigError`, `Timeout`, and
`InvalidImage`. Each carries the generated payload on `.error`, and reads its
fields directly (`exc.id`, `exc.message`).

Note that `xybrid.XybridError` is the *exception*. BoltFFI also generates a
same-named data class for the error union, which stays reachable as
`xybrid._bolt.XybridError`; its variants (`XybridErrorModelNotFound`, …) are
re-exported under their own names.

## How the Bindings Are Built

`xybrid/_bolt/` is generated by boltffi — a pure-Python wire layer plus a
CPython bridge (`_native.c`) compiled against the `xybrid-bolt` cdylib:

```bash
python3 tools/scripts/gen_python_bolt.py           # regenerate the sources
python3 tools/scripts/gen_python_bolt.py --check   # fail on drift (CI)
./tools/scripts/build-python-bolt.sh               # compile + stage binaries
```

Because that directory is byte-compared against fresh generator output, it
carries no hand-written code. The Pythonic surface the docs above use —
envelope factories, `result.text`, model properties, typed exceptions — is
attached to the generated classes at import by `xybrid/_sugar.py` and
`xybrid/_errors.py`. `tests/test_sdk.py` asserts each patched member exists, so
a generator change that drops one fails the suite instead of user code. Those
added members are not in the generated `.pyi` stubs yet, so type checkers do
not see them.

Wheels bundle both binaries, so they are tagged per interpreter ABI
(`cp312-cp312-macosx_11_0_arm64`) rather than `py3-none-any` — one wheel per
Python version and platform.

## Directory Structure

```
python/
├── pyproject.toml
├── setup.py            # per-ABI wheel tagging + staged-binary check
├── README.md
├── tests/
│   ├── test_bolt.py    # generated bindings load and export the surface
│   └── test_sdk.py     # the hand-written SDK layer
└── xybrid/
    ├── __init__.py     # init(), presets, public re-exports
    ├── _errors.py      # typed exceptions over the generated error union
    ├── _sugar.py       # factories/accessors attached to generated classes
    ├── py.typed
    └── _bolt/          # generated; binaries staged here, git-ignored
```
