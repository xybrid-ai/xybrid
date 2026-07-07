# Xybrid Python SDK

> **Status**: Active - low-level ctypes binding plus Pythonic SDK wrapper.

This directory contains the Python package for Xybrid, providing local-first
model inference through the Rust `xybrid-bolt` native library. Runtime code uses
only the Python standard library (`ctypes`); `pytest` is used for tests.

## Installation

### Local Development

Build the host native library and copy it into the Python package:

```bash
# From the xybrid repo root
./tools/scripts/build-python-bolt.sh
```

Install the package in editable mode:

```bash
python -m pip install -e bindings/python
```

The package can also run directly from the repo checkout during development.
When no bundled native library is present, the loader falls back to
`target/release/` and then `target/debug/`.

## Quickstart

```python
import xybrid

xybrid.init()

model = xybrid.XybridModel.from_registry("your-model-id")
result = model.run(xybrid.XybridEnvelope.text("Hello from Python"))

print(result.text)
```

Pass an API key to enable optional platform features such as telemetry:

```python
xybrid.init(api_key="xyb_...")
```

Anonymous initialization keeps inference local-only and telemetry disabled.

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
`InvalidImage`.

## Native Library Resolution

The ctypes layer resolves the native library in this order:

1. `XYBRID_BOLT_LIBRARY`, an absolute path to the native library.
2. Bundled package path: `xybrid/_native/libxybrid_bolt.dylib`,
   `libxybrid_bolt.so`, or `xybrid_bolt.dll`.
3. Development fallback from the repo root: `target/release/`, then
   `target/debug/`.

If resolution fails, build the host library:

```bash
./tools/scripts/build-python-bolt.sh
```

The wire layer in `xybrid/_bolt.py` is a hand-port of the BoltFFI 0.25.3 ABI,
using the committed Swift binding as its executable specification. It should be
replaced by generated Python output once the workspace migrates to
boltffi >= 0.26 and the generator can express handles and fallible functions.

## Directory Structure

```
python/
├── pyproject.toml
├── README.md
├── tests/
│   ├── test_bolt.py
│   └── test_sdk.py
└── xybrid/
    ├── __init__.py
    ├── _bolt.py
    ├── py.typed
    └── _native/
```
