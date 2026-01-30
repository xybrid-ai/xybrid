# Xybrid CLI

Command-line interface for running hybrid cloud-edge AI inference pipelines.

## Installation

Build from source:

```bash
cargo build --release
```

The binary will be available at `target/release/xybrid`.

## Usage

### Run Command

Execute a pipeline from a YAML configuration file:

```bash
xybrid run --config <path-to-config.yaml>
```

Or use the short form:

```bash
xybrid run -c <path-to-config.yaml>
```

## Configuration File Format

The pipeline configuration is a YAML file with the following structure:

```yaml
# Optional pipeline name
name: "My Pipeline"

# List of stage names (executed in order)
stages:
  - "stage-name-1"
  - "stage-name-2"
  - "stage-name-3"

# Input envelope configuration
input:
  kind: "AudioRaw"  # or "Text", etc.

# Device metrics
metrics:
  network_rtt: 110      # Network round-trip time in milliseconds
  battery: 75           # Battery level (0-100)
  temperature: 24.0     # Device temperature in Celsius

# Model availability mapping (stage name -> available locally)
availability:
  "stage-name-1": true   # Available locally
  "stage-name-2": false  # Only available in cloud
  "stage-name-3": true   # Available locally
```

## Example

See `examples/hiiipe.yaml` for a complete example configuration:

```bash
xybrid run -c examples/hiiipe.yaml
```

This will execute the Hiiipe pipeline: Mic Input → ASR → Motivator → TTS.

## Output

The CLI provides:
- Configuration summary (stages, input, metrics, availability)
- Pipeline execution progress via telemetry logs
- Final results showing routing decisions, latency, and outputs for each stage

