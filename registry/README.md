# Xybrid Registry Server

HTTP registry server for serving Xybrid model bundles during local development and testing.

## Quick Start

Using the justfile (recommended):

```bash
# Start the registry server
just registry

# Or with cargo directly
cargo run --bin registry-server -p registry
```

Server runs on `http://localhost:8080` by default, serving from `registry/bundles/`.

## Usage with Justfile

```bash
# Start registry server (default: port 8080, registry/bundles/)
just registry

# Start on custom port
just registry-port 9000

# Serve from local cache (~/.xybrid/registry/)
just registry-local

# Serve from custom path
just registry-path /path/to/bundles

# Create all bundles
just bundles

# Create specific bundle
just bundle whisper-tiny

# List available bundles
just bundles-list

# Clean up bundles
just bundles-clean
```

## Directory Structure

```
registry/
├── Cargo.toml
├── README.md
├── src/
│   └── bin/
│       ├── server.rs          # Registry server
│       └── create_bundles.rs  # Bundle creation utility
└── bundles/
    ├── bundles.json           # Bundle definitions
    ├── whisper-tiny/
    │   └── 1.2/
    │       ├── macos-arm64/
    │       │   └── whisper-tiny.xyb
    │       ├── ios-aarch64/
    │       │   └── whisper-tiny.xyb
    │       └── android-arm64/
    │           └── whisper-tiny.xyb
    ├── wav2vec2-base-960h/
    │   └── 1.0/
    │       └── macos-arm64/
    │           └── wav2vec2-base-960h.xyb
    └── xtts-mini/
        └── 0.6/
            └── ios-aarch64/
                └── xtts-mini.xyb
```

## Bundle Storage Structure

Bundles are stored in a multi-platform format:
```
{model_id}/{version}/{platform}/{model_id}.xyb
```

Example:
```
bundles/
  whisper-tiny/
    1.2/
      macos-arm64/
        whisper-tiny.xyb
      ios-aarch64/
        whisper-tiny.xyb
      android-arm64/
        whisper-tiny.xyb
```

Supported platforms:
- `macos-arm64` - macOS on Apple Silicon
- `macos-x86_64` - macOS on Intel
- `ios-aarch64` - iOS devices
- `android-arm64` - Android 64-bit ARM
- `android-arm32` - Android 32-bit ARM

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /index` | Returns JSON array of `BundleDescriptor` |
| `GET /bundles/{path}` | Returns bundle file bytes |

### Index Response Format

```json
[
  {
    "id": "whisper-tiny",
    "version": "1.2",
    "target": "macos-arm64",
    "hash": "abc123...",
    "size_bytes": 12345678,
    "location": {
      "kind": "remote",
      "url": "/bundles/whisper-tiny/1.2/macos-arm64/whisper-tiny.xyb"
    }
  }
]
```

## Creating Bundles

### Using Justfile

```bash
just bundles          # Create all bundles
just bundle NAME      # Create specific bundle
just bundles-list     # List available bundles
just bundles-clean    # Clean up bundles
```

### Using Cargo Directly

```bash
cargo run --bin create-bundles -p registry                    # Create all
cargo run --bin create-bundles -p registry -- whisper-tiny    # Create specific
cargo run --bin create-bundles -p registry -- --list          # List bundles
cargo run --bin create-bundles -p registry -- --clean         # Clean up
```

### Bundle Configuration

Bundles are defined in `bundles/bundles.json`:

```json
{
  "bundles": {
    "my-model": {
      "model_id": "my-model",
      "version": "1.0",
      "targets": [
        {
          "platform": "macos-arm64",
          "model_file": "my-model.onnx",
          "model_type": "onnx",
          "source_path": "test_models/my-model.onnx",
          "fallback": "placeholder"
        }
      ],
      "description": "My model description"
    }
  }
}
```

Target options:
- **platform**: Target platform (`macos-arm64`, `ios-aarch64`, `android-arm64`, etc.)
- **model_file**: Filename for the model in the bundle
- **model_type**: Model format (`onnx`, `coreml`, `tflite`)
- **source_path**: Path to source model file (optional, for single file)
- **source_dir**: Path to directory containing all model files (preferred)
- **fallback**: Behavior if source file is not found:
  - `placeholder` - Create bundle with placeholder data (for testing/CI when model files aren't available)
  - `error` - Fail bundle creation if source file is missing (for production builds)

## Pipeline Integration

Use the registry in your pipeline config:

```yaml
name: "My Pipeline"
registry: http://localhost:8080
stages:
  - whisper-tiny@1.2
```

Or specify via CLI:

```bash
just run my_pipeline.yaml
# or
cargo run --bin xybrid -- run --config my_pipeline.yaml --registry http://localhost:8080
```

## Server Options

```bash
registry-server [PORT] [REGISTRY_PATH]

Arguments:
  PORT              Port to listen on (default: 8080)
  REGISTRY_PATH     Path to registry directory

Options:
  --local-cache, -l  Use local cache (~/.xybrid/registry/)
  --help, -h         Show help message

Examples:
  registry-server                           # port 8080, registry/bundles/
  registry-server 8080                      # port 8080, registry/bundles/
  registry-server 8080 ~/.xybrid/registry   # port 8080, local cache
  registry-server --local-cache             # port 8080, local cache
  registry-server 9000 -l                   # port 9000, local cache
```
