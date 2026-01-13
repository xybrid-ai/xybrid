# Xybrid Core Benchmarks

This document describes the benchmarking infrastructure for comparing ONNX Runtime execution providers.

## Quick Start

```bash
# Download benchmark models
cd ../integration-tests
./download.sh mnist mobilenet kokoro-82m
chmod -R +r fixtures/models/

# Run benchmarks (from workspace root or xybrid repo root)
cargo bench -p xybrid-core                           # CPU only
cargo bench -p xybrid-core --features coreml-ep      # With CoreML (macOS)

# View HTML report
open target/criterion/report/index.html
```

## Execution Providers

| Provider | Feature Flag | Platforms | Hardware |
|----------|--------------|-----------|----------|
| CPU | default | All | CPU (XNNPACK optimized) |
| CoreML ANE | `coreml-ep` | macOS/iOS | Apple Neural Engine |
| CoreML GPU | `coreml-ep` | macOS/iOS | Apple GPU (Metal) |

## Benchmark Models

| Model | Type | Size | Input Shape | Best Provider |
|-------|------|------|-------------|---------------|
| `mnist` | Vision | 26KB | [1,1,28,28] | CPU (too small) |
| `mobilenet-v2` | Vision | 13MB | [1,3,224,224] | **CoreML ANE** |
| `kokoro-82m` | TTS | 170MB | dynamic | CPU |

## Results (M3 Mac)

| Model | CPU | CoreML ANE | CoreML GPU | Speedup |
|-------|-----|------------|------------|---------|
| MNIST | 60µs | 61µs | 107µs | 1.0x |
| MobileNetV2 | 8.1ms | **1.2ms** | ~9ms | **6.8x** |
| Kokoro-82M | 303ms | 356ms | 349ms | 0.85x |

## When CoreML Helps

### Good Candidates (5-10x speedup)
- **Vision models (CNNs)**: ResNet, MobileNet, YOLO
- **Fixed-shape transformers**: BERT encoder, sentence embeddings
- **Whisper encoder**: Static 30-second audio chunks

### Poor Candidates (CPU often faster)
- **TTS models**: Dynamic output length, transformer decoder
- **Tiny models (<1MB)**: Dispatch overhead exceeds compute
- **Variable sequence length**: Reshaping kills performance

## Technical Details

### CoreML Configuration

```rust
use xybrid_core::runtime_adapter::onnx::{
    ExecutionProviderKind, CoreMLConfig, CoreMLComputeUnits
};

// Use Neural Engine (best for supported models)
let provider = ExecutionProviderKind::CoreML(CoreMLConfig {
    compute_units: CoreMLComputeUnits::CpuAndNeuralEngine,
    use_subgraphs: true,
    require_static_shapes: false,
});

let session = ONNXSession::with_provider("model.onnx", provider)?;
```

### Compute Unit Options

| Option | Hardware | Use Case |
|--------|----------|----------|
| `CpuOnly` | CPU | Debugging, compatibility |
| `CpuAndGpu` | CPU + Metal GPU | General acceleration |
| `CpuAndNeuralEngine` | CPU + ANE | Best for supported ops |
| `All` | CPU + GPU + ANE | Maximum flexibility |

### Model Format Considerations

ONNX Runtime's CoreML EP supports two model formats:

| Format | Default | FP16 Behavior | Recommendation |
|--------|---------|---------------|----------------|
| NeuralNetwork | Yes | Silent cast to FP16 | Legacy, avoid |
| MLProgram | No | Respects precision | Use for accuracy |

To force MLProgram format (if precision matters):
```rust
// Currently requires custom session options
// See: https://ym2132.github.io/ONNX_MLProgram_NN_exploration
```

## Adding New Benchmarks

Edit `benches/inference_backends.rs`:

```rust
const BENCHMARK_MODELS: &[ModelConfig] = &[
    // Single tensor input (vision, embeddings)
    ModelConfig {
        model_dir: "your-model",
        model_file: "model.onnx",
        name: "your-model",
        inputs: ModelInputs::SingleFloat {
            name: "input",
            shape: &[1, 3, 224, 224],
        },
    },

    // Multi-tensor input (TTS)
    ModelConfig {
        model_dir: "tts-model",
        model_file: "model.onnx",
        name: "tts-model",
        inputs: ModelInputs::Tts {
            tokens_name: "tokens",
            style_name: "style",
            speed_name: "speed",
            num_tokens: 50,
            style_dim: 256,
        },
    },
];
```

## Performance Optimization Tips

### For ANE Acceleration

1. **Use static shapes**: Dynamic dimensions prevent ANE optimization
2. **Avoid constant reshaping**: Reshape operations force CPU fallback
3. **Prefer conv-heavy models**: ANE excels at convolutions
4. **Batch size of 1**: ANE optimized for single-sample inference

### For CPU Performance

1. **Enable XNNPACK**: Default in ONNX Runtime
2. **Use FP16 models**: 2x memory bandwidth reduction
3. **Tune thread count**: `ORT_NUM_THREADS` environment variable

## Known Limitations

1. **CoreML graph partitioning**: Complex models may be split across CPU/ANE, reducing benefit
2. **Dynamic shapes**: TTS models with variable output don't benefit from ANE
3. **FP16 precision**: CoreML NeuralNetwork format silently converts to FP16
4. **Debugging**: Hard to know which ops run on ANE vs CPU

## Future Work

- [ ] Benchmark Whisper encoder (expected 2-5x speedup)
- [ ] Benchmark wav2vec2 (ONNX, static audio chunks)
- [ ] Benchmark all-MiniLM embeddings (fixed sequence)
- [ ] iOS CoreML EP support
- [ ] Android NNAPI EP support
- [ ] Auto-selection based on model characteristics

## References

- [ONNX Runtime CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [CoreML FP16 Silent Conversion](https://ym2132.github.io/ONNX_MLProgram_NN_exploration)
- [Running Transformers on Mobile](https://huggingface.co/blog/tugrulkaya/running-large-transformer-models-on-mobile)
