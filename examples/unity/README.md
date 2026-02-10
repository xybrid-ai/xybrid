# Xybrid Unity Examples

This directory contains Unity example projects demonstrating Xybrid SDK integration for on-device ML inference in games.

## Available Examples

| Example | Description | Size | Location |
|---------|-------------|------|----------|
| **[Starter](./starter/)** | Minimal integration example with basic API usage | ~50MB | In this repo |
| **[Night Tavern](https://github.com/xybrid-ai/xybrid-unity-tavern)** | Full RPG with NPC dialogue and TTS audio | ~500MB | Separate repo |
| **[Otome 2](https://github.com/xybrid-ai/xybrid-unity-otome)** | Visual novel showcase with voice synthesis | ~300MB | Separate repo |

## Quick Start

### Starter Example (Recommended for beginners)

The starter example is included in this repository and provides a minimal working integration:

```bash
# From repository root
cd examples/unity/starter

# Open in Unity Hub or directly in Unity 2022.3 LTS
```

See [starter/README.md](./starter/README.md) for detailed instructions.

### Full Game Examples

For complete game implementations, download the example that matches your use case:

#### Night Tavern (RPG)

A fantasy tavern RPG demonstrating:
- NPC dialogue with LLM-powered responses
- Text-to-speech for character voices
- Real-time conversation memory

```bash
git clone https://github.com/xybrid-ai/xybrid-unity-tavern.git
```

**[Download Latest Release](https://github.com/xybrid-ai/xybrid-unity-tavern/releases)**

#### Otome 2 (Visual Novel)

A visual novel demonstrating:
- Branching dialogue systems
- Voice synthesis for multiple characters
- Save/load conversation state

```bash
git clone https://github.com/xybrid-ai/xybrid-unity-otome.git
```

**[Download Latest Release](https://github.com/xybrid-ai/xybrid-unity-otome/releases)**

## Version Compatibility

| Example | SDK Version | Unity Version | Status |
|---------|-------------|---------------|--------|
| Starter | v0.1.0-alpha2+ | 2022.3 LTS | Active |
| Night Tavern | v0.1.0-alpha2+ | 2022.3 LTS | Active |
| Otome 2 | v0.1.0+ | 2022.3 LTS | Planned |

## Requirements

All examples require:

- **Unity 2022.3 LTS** (or later)
- **Xybrid Unity SDK** (included via local package reference)
- **Native libraries** (built from Rust source)

Platform-specific requirements:
- **iOS**: Xcode 14+, iOS 15.0+ target
- **Android**: Android SDK, NDK (via Unity Hub), API 28+
- **macOS**: For editor testing

## Building Native Libraries

Before running any example, build the native libraries:

```bash
# From repository root
cd repos/xybrid

# Build for your target platform(s)
cargo xtask build-xcframework     # iOS + macOS
cargo xtask build-android         # Android
cargo xtask build-ffi             # Current platform (editor testing)
```

See individual example READMEs for library placement instructions.

## Project Structure

```
examples/unity/
├── README.md           # This file
├── starter/            # Minimal example (in repo)
│   ├── Assets/
│   ├── Packages/
│   ├── ProjectSettings/
│   └── README.md
│
# Separate repositories:
# github.com/xybrid-ai/xybrid-unity-tavern
# github.com/xybrid-ai/xybrid-unity-otome
```

## Related

- [Xybrid Unity SDK](../../bindings/unity/README.md) - SDK source code
- [SDK API Reference](../../docs/sdk/API_REFERENCE.md) - Complete API documentation
- [Main Examples README](../README.md) - All platform examples

## Contributing

To add a new Unity example:

1. Create a new repository: `xybrid-ai/xybrid-unity-<name>`
2. Use the starter example as a template
3. Add your example to the table above
4. Create tagged releases matching SDK versions

## License

MIT License - See [LICENSE](../../LICENSE) for details.
