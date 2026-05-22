# SDK Public API Reference

> **This file is the source of truth for all SDK public APIs.**
>
> **RULE**: Any change to public-facing APIs in Flutter, Kotlin, Swift, Unity, or future SDKs
> MUST first update this file. Implementation follows the spec, not the other way around.
>
> **Machine-readable companion**: See [`api-surface.yaml`](api-surface.yaml) for the structured
> contract used by CI validation and the `/sync-api` command.

## API Change Process

1. Propose API change in this file (via PR or discussion)
2. Get approval on the API design
3. Update this reference document AND `api-surface.yaml`
4. Implement in all SDKs (Dart, Kotlin, Swift, C#)

## Implementation Status

<!-- Keep this legend and the per-section matrices in sync with api-surface.yaml -->

| Symbol | Meaning |
|--------|---------|
| ✅ | Implemented and tested |
| 🚧 | Partially implemented or stub |
| — | Not yet implemented |

### SDK Naming Conventions

Each SDK uses idiomatic naming for its platform. The spec uses canonical names;
SDKs may prefix or adjust casing:

| Spec Name | Dart | Kotlin | Swift | C# (Unity) |
|-----------|------|--------|-------|-------------|
| `Envelope` | `XybridEnvelope` | `XybridEnvelope` | `XybridEnvelope` | `Envelope` |
| `InferenceResult` | `XybridResult` | `XybridResult` | `XybridResult` | `InferenceResult` |
| `OutputType` enum | `FfiOutputType` | `OutputType` | `OutputType` | `OutputType` |
| `PipelineInputType` | `FfiPipelineInputType` | `PipelineInputType` | — | — |

---

## Core Pattern: Loader → Model → Run

All SDKs follow the same three-step pattern:

```
1. Create a Loader  →  Xybrid.model(modelId: "whisper-tiny")
2. Load the Model   →  await loader.load()
3. Run Inference    →  await model.run(envelope: input)
```

---

## 1. Xybrid (Entry Point)

The main SDK entry point with static methods.

### Dart

```dart
class Xybrid {
  // Initialization
  static Future<void> init({
    String? apiKey,
    String? gatewayUrl,
    String? ingestUrl,
    String? resourceTelemetry,
  });

  // Runtime configuration
  static void setApiKey(String apiKey);
  static void setGatewayUrl(String gatewayUrl);

  // Model Loading (returns XybridModelLoader)
  static XybridModelLoader model({
    String? modelId,      // From registry
    String? platform,     // Optional platform override
    String? bundlePath,   // From local bundle
    String? modelDir,     // From local directory
  });

  // Pipeline Loading (returns XybridPipelineRef)
  static XybridPipelineRef pipeline({
    String? yaml,         // From YAML string
    String? filePath,     // From YAML file
  });

  // Cache
  static bool isModelCached(String modelId);
}
```

### Kotlin

```kotlin
object Xybrid {
  // Initialization (Android requires Context)
  fun init(context: Context)

  // API Key Management
  fun setApiKey(apiKey: String)

  // Model Loading
  fun model(
    modelId: String? = null,
    platform: String? = null,
    bundlePath: String? = null,
    modelDir: String? = null
  ): XybridModelLoader
}
```

### Implementation Status

| Method | Dart | Kotlin | Swift | C# |
|--------|------|--------|-------|----|
| `init()` | ✅ | ✅ | — | ✅ |
| `setApiKey()` | ✅ | — | — | — |
| `setGatewayUrl()` | ✅ | — | — | — |
| `model()` | ✅ | — | — | ✅ |
| `pipeline()` | ✅ | — | — | — |
| `isModelCached()` | ✅ | — | — | — |

---

## 2. XybridModelLoader

Creates model instances from various sources.

### Dart

```dart
class XybridModelLoader {
  // Factory methods
  factory XybridModelLoader.fromRegistry(String modelId);
  factory XybridModelLoader.fromBundle(String path);
  factory XybridModelLoader.fromDirectory(String path);

  // Load the model
  Future<XybridModel> load();

  // Load with progress events
  Stream<LoadEvent> loadWithProgress();
}
```

### Kotlin

```kotlin
class XybridModelLoader {
  companion object {
    fun fromRegistry(modelId: String): XybridModelLoader
    fun fromBundle(path: String): XybridModelLoader
    fun fromDirectory(path: String): XybridModelLoader
  }

  suspend fun load(): XybridModel
}
```

### C# (Unity)

```csharp
public class ModelLoader
{
    public static ModelLoader FromRegistry(string modelId);
    public static ModelLoader FromBundle(string bundlePath);
    public static ModelLoader FromDirectory(string directoryPath);
    public InferenceResult Load();
}
```

### `fromDirectory()`

Loads a model from a local directory containing a `model_metadata.json` and its referenced model files. Use this for models not in the xybrid registry — for example, custom-trained models, models downloaded from HuggingFace, or models bundled directly with your app.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `path` | `String` | Absolute path to a directory containing `model_metadata.json` and model files |

**Returns:** `XybridModelLoader` — call `.load()` to get a ready-to-use `XybridModel`.

**Errors:**

| Error | When |
|-------|------|
| `DirectoryNotFound` | The specified path does not exist or is not a directory |
| `MetadataNotFound` | No `model_metadata.json` file in the directory |
| `MetadataInvalid` | `model_metadata.json` exists but contains invalid JSON |

**Usage Example:**

```dart
// Dart / Flutter
final loader = XybridModelLoader.fromDirectory('/path/to/my-model');
final model = await loader.load();
final result = await model.run(envelope: XybridEnvelope.text("Hello!"));
```

```kotlin
// Kotlin / Android
val loader = XybridModelLoader.fromDirectory("/data/local/tmp/my-model")
val model = loader.load()
val result = model.run(XybridEnvelope.text("Hello!"))
```

```swift
// Swift / iOS
let loader = try XybridModelLoader.fromDirectory(path: modelPath)
let model = try await loader.load()
let result = try await model.run(envelope: .text("Hello!"))
```

```csharp
// C# / Unity
var loader = ModelLoader.FromDirectory(Path.Combine(Application.streamingAssetsPath, "my-model"));
using var result = loader.Load().Run(Envelope.Text("Hello!"));
```

### `fromHuggingFace()`

Downloads a model from a HuggingFace Hub repository and caches it locally. Model metadata (`model_metadata.json`) is auto-generated from the model card and file inspection if not present in the repository.

> **Note:** Requires the `huggingface` feature flag to be enabled at compile time.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `repo` | `String` | HuggingFace repository ID (e.g., `"xybrid-ai/kokoro-82m"`) |

**Returns:** `XybridModelLoader` — call `.load()` to download (if needed) and get a ready-to-use `XybridModel`.

**Usage Example:**

```dart
// Dart / Flutter
final loader = XybridModelLoader.fromHuggingFace('xybrid-ai/kokoro-82m');
final model = await loader.load();
final result = await model.run(envelope: XybridEnvelope.text("Hello!"));
```

```kotlin
// Kotlin / Android
val loader = XybridModelLoader.fromHuggingface(repo = "xybrid-ai/kokoro-82m")
val model = loader.load()
val result = model.run(XybridEnvelope.text("Hello!"))
```

```swift
// Swift / iOS
let loader = XybridModelLoader.fromHuggingface(repo: "xybrid-ai/kokoro-82m")
let model = try await loader.load()
let result = try await model.run(envelope: .text("Hello!"))
```

```csharp
// C# / Unity
var loader = ModelLoader.FromHuggingFace("xybrid-ai/kokoro-82m");
using var model = loader.Load();
var result = model.Run(Envelope.Text("Hello!"));
```

### Implementation Status

| Method | Dart | Kotlin | Swift | C# |
|--------|------|--------|-------|----|
| `fromRegistry()` | ✅ | ✅ | ✅ | ✅ |
| `fromBundle()` | ✅ | ✅ | ✅ | ✅ |
| `fromDirectory()` | ✅ | ✅ | ✅ | ✅ |
| `fromHuggingFace()` | ✅ | ✅ | ✅ | ✅ |
| `load()` | ✅ | ✅ | ✅ | ✅ |
| `loadWithProgress()` | ✅ | — | — | — |

---

## 3. XybridModel

Loaded model instance for running inference.

### Dart

```dart
class XybridModel {
  // Properties
  String get modelId;

  // Voice discovery (TTS models only)
  List<VoiceInfo>? get voices;
  VoiceInfo? get defaultVoice;
  bool get hasVoices;
  VoiceInfo? voice({required String voiceId});

  // Inference
  Future<XybridResult> run({
    required Envelope envelope,
    GenerationConfig? config,
  });
  Future<XybridResult> runWithContext({
    required Envelope envelope,
    required ConversationContext context,
    GenerationConfig? config,
  });

  // Streaming (LLM token-by-token)
  Stream<StreamToken> runStreaming({
    required Envelope envelope,
    GenerationConfig? config,
  });
  Stream<StreamToken> runStreamingWithFallback({
    required Envelope envelope,
    required RunOptions options,
    GenerationConfig? config,
  });
  Stream<StreamToken> runStreamingWithContext({
    required Envelope envelope,
    required ConversationContext context,
    GenerationConfig? config,
  });

  // Benchmarking
  Future<BenchmarkResult> benchmark({
    required Envelope envelope,
    required int iterations,
    required int warmupIterations,
  });

  // Lifecycle
  void unload();

  // Hardware info
  ExecutionProviderInfo executionProviderInfo();
}
```

### Kotlin

```kotlin
class XybridModel {
  val modelId: String

  // Voice discovery (TTS models only)
  val voices: List<VoiceInfo>?
  val defaultVoice: VoiceInfo?
  val hasVoices: Boolean
  fun voice(voiceId: String): VoiceInfo?

  suspend fun run(
    envelope: Envelope,
    config: GenerationConfig? = null
  ): XybridResult

  suspend fun runWithContext(
    envelope: Envelope,
    context: ConversationContext,
    config: GenerationConfig? = null
  ): XybridResult
}
```

### Rust

```rust
impl XybridModel {
    pub fn run_with_options(
        &self,
        envelope: &Envelope,
        options: &RunOptions,
    ) -> SdkResult<InferenceResult>;

    pub fn run_streaming_with_options<F>(
        &self,
        envelope: &Envelope,
        options: &RunOptions,
        on_token: F,
    ) -> SdkResult<InferenceResult>;
}
```

### Implementation Status

| Method | Dart | Kotlin | Swift | C# |
|--------|------|--------|-------|----|
| `modelId` | — | — | — | ✅ |
| `voices` | — | ✅ | ✅ | ✅ |
| `defaultVoice` | — | 🚧 | 🚧 | ✅ |
| `hasVoices` | — | ✅ | ✅ | ✅ |
| `voice()` | — | ✅ | ✅ | ✅ |
| `run()` | ✅ | ✅ | ✅ | ✅ |
| `runWithOptions()` / `run_with_options()` | Rust ✅ | planned | planned | planned |
| `runWithContext()` | ✅ | — | — | ✅ |
| `runWithContextOptions()` / `run_with_context_options()` | Rust ✅ | planned | planned | planned |
| `runStreaming()` | ✅ | — | — | ✅ |
| `runStreamingWithOptions()` / `run_streaming_with_options()` | Rust ✅ | planned | planned | planned |
| `runStreamingWithFallback()` | ✅ | planned | planned | planned |
| `runStreamingWithContext()` | ✅ | — | — | ✅ |
| `runStreamingWithContextOptions()` / `run_streaming_with_context_options()` | Rust ✅ | planned | planned | planned |
| `benchmark()` | — | — | — | — |
| `unload()` | — | — | — | — |
| `executionProviderInfo()` | — | — | — | — |

---

## 4. XybridPipelineRef / XybridPipeline

Multi-stage inference pipelines.

### Dart

```dart
class XybridPipeline {
  // Factory methods
  factory XybridPipeline.fromYaml(String yamlContent);
  factory XybridPipeline.fromFile(String path);
  factory XybridPipeline.fromBundle(String path);

  // Properties
  String? get name;
  bool get isReady;
  BigInt get stageCount;
  List<String> get stageNames;

  // Load models
  Future<void> load();

  // Execution
  Future<XybridResult> run({required Envelope envelope});
}
```

### Kotlin

```kotlin
class XybridPipelineRef {
  companion object {
    fun fromYaml(yamlContent: String): XybridPipelineRef
    fun fromFile(path: String): XybridPipelineRef
  }

  val name: String?
  val stageIds: List<String>

  suspend fun load(): XybridPipeline
}

class XybridPipeline {
  val name: String?
  val isReady: Boolean
  val stageCount: Long
  val stageNames: List<String>

  suspend fun run(envelope: Envelope): PipelineResult
}
```

### Rust

```rust
impl Pipeline {
    pub fn run_with_options(
        &self,
        envelope: &Envelope,
        options: &RunOptions,
    ) -> PipelineResult<PipelineExecutionResult>;

    pub async fn run_async_with_options(
        &self,
        envelope: &Envelope,
        options: &RunOptions,
    ) -> PipelineResult<PipelineExecutionResult>;
}

impl Xybrid {
    pub fn run_pipeline_with_options(
        yaml: &str,
        envelope: &Envelope,
        options: &RunOptions,
    ) -> PipelineResult<PipelineExecutionResult>;

    pub fn run_pipeline_streaming_with_options<F>(
        yaml: &str,
        envelope: &Envelope,
        options: &RunOptions,
        on_token: F,
    ) -> PipelineResult<PipelineExecutionResult>;
}
```

### Implementation Status

| Method | Dart | Kotlin | Swift | C# |
|--------|------|--------|-------|----|
| `fromYaml()` | ✅ | — | — | — |
| `fromFile()` | ✅ | — | — | — |
| `fromBundle()` | ✅ | — | — | — |
| `name` | ✅ | — | — | — |
| `isReady` | ✅ | — | — | — |
| `stageCount` | ✅ | — | — | — |
| `stageNames` | ✅ | — | — | — |
| `load()` | ✅ | — | — | — |
| `run()` | ✅ | — | — | — |
| `runWithOptions()` / `run_with_options()` | Rust ✅ | planned | planned | planned |
| `runPipelineStreamingWithOptions()` / `run_pipeline_streaming_with_options()` | Rust ✅ | planned | planned | planned |

> **Note**: The Dart SDK currently uses a single `XybridPipeline` class (no separate `PipelineRef`).
> The Kotlin spec shows the two-step `PipelineRef` → `Pipeline` pattern which is the target design.

---

## 5. Envelope

Universal input/output container.

**Design Note**: Envelope is task-agnostic. Optional metadata parameters (like `voiceId` and `speed`)
are passed through to pipeline stages that understand them. Stages that don't need these params
simply ignore them. This allows a single envelope type to flow through multi-stage pipelines
(e.g., ASR → LLM → TTS) without type changes.

Image envelopes carry raw bytes plus a `format` tag (`"png" | "jpeg" | "webp"`); preprocessing
pipelines decode and resize them. Multi-part user messages combine a text prompt with one or
more image envelopes for vision-language models — see `Envelope.userMessage` below.

### Dart

```dart
class XybridEnvelope {
  // Factory methods
  factory XybridEnvelope.text(
    String text, {
    String? voiceId,      // Optional: TTS voice selection
    double? speed,        // Optional: TTS speed multiplier
  });
  factory XybridEnvelope.audio(
    List<int> audioBytes, {
    int sampleRate = 16000,
    int channels = 1,
  });
  factory XybridEnvelope.embedding(List<double> embedding);

  // Vision (planned)
  factory XybridEnvelope.image(
    List<int> imageBytes,
    String format,        // "png" | "jpeg" | "webp"
  );
  factory XybridEnvelope.userMessage(
    String text, {
    List<XybridEnvelope> images = const [],
  });

  // Role support (for conversational context)
  factory XybridEnvelope.textWithRole(String text, MessageRole role);
  XybridEnvelope withRole(MessageRole role);
}
```

### Kotlin

```kotlin
sealed class XybridEnvelope {
  data class Text(
    val text: String,
    val voiceId: String? = null,
    val speed: Double? = null
  ) : XybridEnvelope()
  data class Audio(
    val audioBytes: ByteArray,
    val sampleRate: Int = 16000,
    val channels: Int = 1
  ) : XybridEnvelope()
  data class Embedding(val embedding: FloatArray) : XybridEnvelope()

  // Vision (planned)
  data class Image(
    val imageBytes: ByteArray,
    val format: String,          // "png" | "jpeg" | "webp"
  ) : XybridEnvelope()
  data class UserMessage(
    val text: String,
    val images: List<XybridEnvelope> = emptyList(),
  ) : XybridEnvelope()

  companion object {
    fun text(
      text: String,
      voiceId: String? = null,
      speed: Double? = null
    ): XybridEnvelope = Text(text, voiceId, speed)
    fun audio(
      audioBytes: ByteArray,
      sampleRate: Int = 16000,
      channels: Int = 1
    ): XybridEnvelope = Audio(audioBytes, sampleRate, channels)
    fun embedding(embedding: FloatArray): XybridEnvelope = Embedding(embedding)

    // Vision (planned)
    fun image(
      imageBytes: ByteArray,
      format: String,
    ): XybridEnvelope = Image(imageBytes, format)
    fun userMessage(
      text: String,
      images: List<XybridEnvelope> = emptyList(),
    ): XybridEnvelope = UserMessage(text, images)
  }
}
```

### Implementation Status

| Factory | Dart | Kotlin | Swift | C# |
|---------|------|--------|-------|----|
| `text()` | ✅ | ✅ | ✅ | ✅ |
| `text(voiceId, speed)` | ✅ | ✅ | ✅ | ✅ |
| `audio()` | ✅ | ✅ | ✅ | ✅ |
| `embedding()` | ✅ | ✅ | ✅ | — |
| `textWithRole()` | ✅ | — | — | ✅ |
| `withRole()` | ✅ | — | — | — |
| `image()` | 📋 | 📋 | 📋 | 📋 |
| `userMessage()` | 📋 | 📋 | 📋 | 📋 |

Legend: ✅ implemented · 📋 planned · — not applicable for this binding.

---

## 6. VoiceInfo (TTS Voice Metadata)

Voice metadata for TTS models. Available via `XybridModel.voices` for models with voice support.

### Dart

```dart
class VoiceInfo {
  final String id;           // Unique voice identifier (e.g., "af_bella")
  final String name;         // Display name (e.g., "Bella")
  final String? gender;      // "male", "female", "neutral"
  final String? language;    // BCP-47 language tag (e.g., "en-US")
  final String? style;       // Voice style (e.g., "cheerful", "professional")
  final bool isDefault;      // Whether this is the model's default voice
}
```

### Kotlin

```kotlin
data class VoiceInfo(
  val id: String,
  val name: String,
  val gender: String?,
  val language: String?,
  val style: String?,
  val isDefault: Boolean
)
```

### Implementation Status

| Property | Dart | Kotlin | Swift | C# |
|----------|------|--------|-------|----|
| `id` | — | ✅ | ✅ | ✅ |
| `name` | — | ✅ | ✅ | ✅ |
| `gender` | — | ✅ | ✅ | ✅ |
| `language` | — | ✅ | ✅ | ✅ |
| `style` | — | ✅ | ✅ | ✅ |
| `isDefault` | — | — | — | — |

### Usage Example

```dart
// Load TTS model
final loader = Xybrid.model(modelId: "kokoro-82m");
final model = await loader.load();

// Discover available voices
if (model.hasVoices) {
  print("Available voices:");
  for (final voice in model.voices!) {
    final marker = voice.isDefault ? " (default)" : "";
    print("  ${voice.id}: ${voice.name}$marker");
  }
}

// Run with specific voice
final result = await model.run(
  envelope: XybridEnvelope.text(
    "Hello world!",
    voiceId: "am_adam",
    speed: 1.0,
  ),
);

// Use default voice (omit voiceId)
final result2 = await model.run(
  envelope: XybridEnvelope.text("Hello!"),
);
```

### Pipeline Usage

The optional `voiceId` and `speed` parameters work seamlessly in pipelines:

```dart
// Pipeline: ASR → LLM → TTS
final pipeline = XybridPipeline.fromYaml('''
name: voice-assistant
stages:
  - model: whisper-tiny
  - model: llama-3-8b
  - model: kokoro-82m
''');
await pipeline.load();

final result = await pipeline.run(
  envelope: XybridEnvelope.audio(micInput),
);
```

For pipeline-level voice configuration, use stage config in YAML:

```yaml
name: voice-assistant
stages:
  - model: whisper-tiny
  - model: llama-3-8b
  - model: kokoro-82m
    config:
      voice_id: "am_adam"
      speed: 1.0
```

---

## 7. Result Types

> **Audio format**: TTS models produce raw PCM audio bytes (16-bit signed, little-endian).
> Typical sample rate is 24kHz mono (e.g., Kokoro TTS). The audio is returned as raw bytes,
> not base64-encoded. Convert to WAV or feed directly to platform audio APIs.

### Dart

```dart
class XybridResult {
  final bool success;
  final String? error;
  final String? text;
  final Uint8List? audioBytes;   // Raw PCM bytes (16-bit signed LE)
  final Float32List? embedding;
  final int latencyMs;

  // Convenience
  bool get isFailure;
  Uint8List? audioAsWav({int sampleRate = 24000, int channels = 1});
}
```

### Kotlin

```kotlin
data class XybridResult(
  val success: Boolean,
  val error: String?,
  val text: String?,
  val audioBytes: ByteArray?,    // Raw PCM bytes (16-bit signed LE)
  val embedding: FloatArray?,
  val outputType: OutputType,
  val latencyMs: Int,
  val modelId: String
) {
  val isFailure: Boolean
  val latencySeconds: Double
}

enum class OutputType { TEXT, AUDIO, EMBEDDING, UNKNOWN }
```

### C# (Unity)

```csharp
public sealed class InferenceResult : IDisposable
{
  public bool Success { get; }
  public string Error { get; }
  public string Text { get; }
  public byte[] AudioBytes { get; }     // Raw PCM bytes (16-bit signed LE)
  public float[] Embedding { get; }
  public OutputType OutputType { get; }
  public uint LatencyMs { get; }
  public bool HasAudio { get; }
  public bool HasEmbedding { get; }
}

public enum OutputType { Text, Audio, Embedding, Unknown }
```

### InferenceMetrics

Typed inference metrics surfaced on every `XybridResult`. LLM-specific fields
(`ttftMs`, `tokensPerSecond`, `prefillTps`, `decodeTps`, `tokensOut`) are
`null` when the model is ASR/TTS/embedding. `stageLatenciesMs` is empty for
`model.run()` and populated for `pipeline.run()`.

Population is best-effort: fields are parsed from the `Envelope.metadata`
string map written by `runtime_adapter::llm` and `execution::executor`.
Local LLM runs populate the LLM scalars; **cloud LLM runs currently surface
only `totalMs`** (the cloud adapter writes `backend` to envelope metadata
but not per-run scalars — those ride on span metadata today). Unparseable
values become `null`. Input-token counts (`promptTokens` / `tokensIn`) are
not on this surface yet — they exist as span metadata only and will be
added once an adapter writes the key to the envelope.

```dart
class XybridInferenceMetrics {
  final int totalMs;
  final int? ttftMs;
  final double? tokensPerSecond;
  final double? prefillTps;
  final double? decodeTps;
  final int? tokensOut;
  final List<XybridStageLatency> stageLatenciesMs;
}

class XybridStageLatency {
  final String stageId;
  final int latencyMs;
}
```

```kotlin
data class XybridInferenceMetrics(
  val totalMs: Int,
  val ttftMs: Int?,
  val tokensPerSecond: Double?,
  val prefillTps: Double?,
  val decodeTps: Double?,
  val tokensOut: Int?,
  val stageLatenciesMs: List<XybridStageLatency>
)

data class XybridStageLatency(val stageId: String, val latencyMs: Int)
```

```swift
public struct XybridInferenceMetrics {
  public let totalMs: Int
  public let ttftMs: Int?
  public let tokensPerSecond: Double?
  public let prefillTps: Double?
  public let decodeTps: Double?
  public let tokensOut: Int?
  public let stageLatenciesMs: [XybridStageLatency]
}

public struct XybridStageLatency {
  public let stageId: String
  public let latencyMs: Int
}
```

```csharp
public sealed class InferenceMetrics
{
  public uint TotalMs { get; }
  public uint? TtftMs { get; }
  public float? TokensPerSecond { get; }
  public float? PrefillTps { get; }
  public float? DecodeTps { get; }
  public uint? TokensOut { get; }
  public IReadOnlyList<StageLatency> StageLatenciesMs { get; }
}

public sealed class StageLatency
{
  public string StageId { get; }
  public uint LatencyMs { get; }
}
```

### Implementation Status

| Property | Dart | Kotlin | Swift | C# |
|----------|------|--------|-------|----|
| `success` | ✅ | ✅ | ✅ | ✅ |
| `error` | ✅ | — | — | ✅ |
| `text` | ✅ | ✅ | ✅ | ✅ |
| `audioBytes` | ✅ | ✅ | ✅ | ✅ |
| `embedding` | ✅ | ✅ | ✅ | ✅ |
| `outputType` | — | — | — | ✅ |
| `latencyMs` | ✅ | ✅ | ✅ | ✅ |
| `modelId` | — | — | — | ✅ |
| `isFailure` | ✅ | ✅ | ✅ | ✅ |
| `audioAsWav()` | ✅ | — | — | — |
| `metrics` | ✅ | ✅ | ✅ | ✅ |
| `metrics.ttftMs` | ✅ | ✅ | ✅ | ✅ |
| `metrics.tokensPerSecond` | ✅ | ✅ | ✅ | ✅ |
| `metrics.stageLatenciesMs` | ✅ | ✅ | ✅ | ✅ |

---

## 8. Supporting Types

### ConversationContext

Multi-turn LLM conversation support.

```dart
class ConversationContext {
  factory ConversationContext();
  ConversationContext withSystem(Envelope systemMessage);
  void push(Envelope message);
}
```

```kotlin
class ConversationContext {
  fun withSystem(systemMessage: Envelope): ConversationContext
  fun push(message: Envelope)
}
```

**Multi-turn vision** (planned): when a user message contains images
(built via `Envelope.userMessage(text, images: [...])`), the image-bearing envelope
stays attached to its turn in the conversation history. Vision-capable backends
re-prefill image tokens at each turn that references them, matching `llama.cpp`'s
`mtmd` defaults. There is no separate image-embedding cache in the planned initial implementation; image bytes
remain in memory for as long as the `ConversationContext` references them.

### MessageRole

```dart
enum MessageRole { system, user, assistant }
```

```kotlin
enum class MessageRole { SYSTEM, USER, ASSISTANT }
```

### GenerationConfig (LLM Generation Parameters)

Optional configuration for controlling LLM text generation. All fields are nullable —
when `null`, the model's defaults are used. Can be passed to **all inference methods**
(`run`, `runWithContext`, `runStreaming`, `runStreamingWithContext`), not just streaming.

#### Dart

```dart
class GenerationConfig {
  final int? maxTokens;            // Max tokens to generate (default: 2048)
  final double? temperature;       // Sampling temperature (default: 0.7)
  final double? topP;              // Nucleus sampling threshold (default: 0.9)
  final double? minP;              // Adaptive pruning threshold (default: 0.05)
  final int? topK;                 // Top-k sampling, 0 = disabled (default: 40)
  final double? repetitionPenalty; // Repetition penalty, 1.0 = off (default: 1.1)
  final List<String>? stopSequences;

  // Presets
  const GenerationConfig.greedy();    // temperature=0, topP=1, topK=0
  const GenerationConfig.creative();  // temperature=0.9, topP=0.95, topK=50
}
```

#### Kotlin

```kotlin
data class GenerationConfig(
  val maxTokens: UInt?,
  val temperature: Float?,
  val topP: Float?,
  val minP: Float?,
  val topK: UInt?,
  val repetitionPenalty: Float?,
  val stopSequences: List<String>?
)

// Presets
GenerationConfigs.greedy()    // temperature=0, topP=1, topK=0
GenerationConfigs.creative()  // temperature=0.9, topP=0.95, topK=50
```

#### Usage

```dart
// Custom parameters
final result = await model.run(
  envelope: XybridEnvelope.text("Hello!"),
  config: GenerationConfig(temperature: 0.3, topK: 20, maxTokens: 256),
);

// Preset
final stream = model.runStreaming(
  envelope: XybridEnvelope.text("Write a poem"),
  config: GenerationConfig.creative(),
);
```

### RunOptions and AbortPolicy

Per-run controls for cooperative cancellation and resource-driven local abort.
Rust SDK methods with options are available as model-level `run_with_options`,
`run_with_context_options`, `run_streaming_with_options`, and
`run_streaming_with_context_options`, plus pipeline-level `run_with_options`,
`run_async_with_options`, `Xybrid::run_pipeline_with_options`, and
`Xybrid::run_pipeline_streaming_with_options`.

```rust
let token = CancellationToken::new();
let options = RunOptions::new()
    .with_generation_config(GenerationConfig::greedy())
    .with_cancellation_token(token.clone())
    .with_abort_policy(
        AbortPolicy::default()
            .stop_on(AbortSignal::UserCancelled)
            .stop_on(AbortSignal::MemoryPressureCritical)
            .with_cloud_fallback(true)
            .with_max_grace_tokens(2),
    )
    .with_correlation_id("run-123");

let result = model.run_streaming_with_options(&envelope, &options, |token| {
    print!("{}", token.token);
    Ok(())
});
```

`fallback_to_cloud` is carried in policy and telemetry contracts so binding
layers and platform routing can restart on cloud where supported; local Rust
streaming abort is cooperative and checked before every emitted token.

`AbortSignal` is the shared policy enum. Rust currently supports
`UserCancelled`, `MemoryPressureWarn`, `MemoryPressureCritical`, `ThermalHot`,
and `ThermalCritical`; Flutter currently exposes the customer-facing fallback
signals `memoryPressureCritical` and `thermalCritical`.

Flutter exposes the customer opt-in surface as `RunOptions.cloudFallback(...)`
plus `AbortPolicy.cloudFallback(...)`; plain `RunOptions()` stays neutral and
does not opt into cloud retry. The Flutter FFI adapter maps that policy to the
Rust SDK abort policy, including the selected memory/thermal stop signals,
`fallbackToCloud`, and grace-token budget, and uses
`runStreamingWithFallback(...)` for the continuous token stream.

`cloudGatewayUrl` is an optional override for the Xybrid cloud gateway base URL.
Customer builds accept HTTPS Xybrid gateway hosts with a `/v1` base path. Debug
builds also accept localhost, private IP, and link-local gateways so apps can
exercise fallback against the local platform stack. URLs with embedded
credentials, query strings, fragments, unsupported schemes, or missing `/v1`
are rejected before the cloud retry starts.

Routing feedback is recorded inside the core orchestrator using low-cardinality
resource buckets. The SDK keeps `correlation_id` as an opaque string for
cross-binding compatibility; telemetry may include flat `routing_source`,
`routing_reason`, `outcome_category`, `abort_reason`, `fallback_target`,
`fallback_reason`, and `fallback_outcome` fields.

### Implementation Status

| Type | Dart | Kotlin | Swift | C# |
|------|------|--------|-------|----|
| `ConversationContext` | ✅ | — | — | ✅ |
| `MessageRole` | ✅ | — | — | ✅ |
| `GenerationConfig` | ✅ | ✅ | ✅ | ✅ |
| `RunOptions` | ✅ | planned | planned | planned |
| `AbortPolicy` | ✅ | planned | planned | planned |
| `AbortSignal` | ✅ | planned | planned | planned |
| `CancellationToken` | ✅ | planned | planned | planned |
| `StreamToken` | ✅ | — | — | ✅ |

---

## 9. Configuration Types

### Dart

```dart
class TelemetryConfiguration {
  final String serverUrl;  // default: "https://api.xybrid.ai"

  factory TelemetryConfiguration.local({int port = 8000});
}

class XybridConfiguration {
  final String? registry;
  final String? apiKey;
  final int timeoutMs;       // default: 30000
  final int retryAttempts;   // default: 3

  factory XybridConfiguration.local(String registryPath);
  const XybridConfiguration.defaults();
}
```

### Kotlin

```kotlin
data class TelemetryConfiguration(
  val serverUrl: String = "https://api.xybrid.ai"
) {
  companion object {
    fun local(port: Int = 8000): TelemetryConfiguration
  }
}

data class XybridConfiguration(
  val registry: String? = null,
  val apiKey: String? = null,
  val timeoutMs: Int = 30000,
  val retryAttempts: Int = 3
) {
  companion object {
    fun local(registryPath: String): XybridConfiguration
    fun defaults(): XybridConfiguration
  }
}
```

### C# (Unity)

The Unity SDK ships a fluent telemetry builder (`Xybrid.TelemetryConfig`) plus three
static lifecycle methods on `Xybrid.XybridClient` (`InitializeTelemetry`, `FlushTelemetry`,
`ShutdownTelemetry`). Construct a config, hand it to `XybridClient.InitializeTelemetry`
(which consumes the native handle), call `FlushTelemetry()` whenever you want a
synchronous drain, and `ShutdownTelemetry()` once at app exit.

```csharp
public sealed class TelemetryConfig : IDisposable
{
    public TelemetryConfig(string apiKey); // binds to the SDK default ingest URL
    public string Endpoint { get; }        // currently resolved ingest endpoint
    public TelemetryConfig WithEndpoint(string endpoint); // self-hosted override
    public TelemetryConfig WithAppVersion(string appVersion);
    public TelemetryConfig WithDeviceLabel(string deviceLabel);
    public TelemetryConfig WithDeviceAttribute(string key, string value);
    public TelemetryConfig WithBatchSize(uint batchSize);
    public TelemetryConfig WithFlushInterval(TimeSpan interval);
    public bool IsDisposed { get; }
    public void Dispose();
}

public static class XybridClient
{
    public static void InitializeTelemetry(TelemetryConfig config);
    public static void FlushTelemetry();
    public static void ShutdownTelemetry();
}
```

**Usage example — default ingest endpoint:**

```csharp
// C# / Unity
XybridClient.Initialize();

var config = new TelemetryConfig(
        apiKey: Environment.GetEnvironmentVariable("XYBRID_TELEMETRY_API_KEY"))
    .WithAppVersion("1.4.2")
    .WithDeviceLabel(SystemInfo.deviceModel)
    .WithDeviceAttribute("build", "release")
    .WithBatchSize(64)
    .WithFlushInterval(TimeSpan.FromSeconds(30));

// config.Endpoint reports "https://ingest.xybrid.dev" until overridden
XybridClient.InitializeTelemetry(config); // consumes config; subsequent Dispose is a no-op

// ... run inferences ...

XybridClient.FlushTelemetry();   // call from OnApplicationPause(true) on mobile
XybridClient.ShutdownTelemetry(); // call from OnApplicationQuit
```

**Self-hosted endpoint:**

```csharp
var config = new TelemetryConfig(apiKey)
    .WithEndpoint("https://telemetry.internal.example.com")
    .WithAppVersion("1.4.2");
// config.Endpoint now reports the override.
XybridClient.InitializeTelemetry(config);
```

### Rust — `SdkConfig`

The Rust SDK ships a small `SdkConfig` struct consumed by `init_sdk_cache_dir`.
It carries the `binding` identifier reported in the `X-Xybrid-Client` registry
telemetry header. Non-Rust bindings register their identifier through the
platform-specific entry points listed under "Setting `binding` per binding"
below — they do not expose `SdkConfig` directly.

```rust
pub struct SdkConfig {
    pub cache_dir: Option<std::path::PathBuf>,
    /// Reported in the `X-Xybrid-Client` registry header. Defaults to
    /// `DEFAULT_BINDING` ("rust") when unset.
    pub binding: Option<&'static str>,
}

impl SdkConfig {
    /// Override the binding identifier reported in the registry telemetry header.
    pub fn with_binding(self, binding: &'static str) -> Self;
    /// Resolve the configured binding identifier, falling back to `DEFAULT_BINDING`.
    pub fn binding(&self) -> &'static str;
}

pub const DEFAULT_BINDING: &str = "rust";
```

**Example — explicit Rust binding:**

```rust
use xybrid_sdk::{SdkConfig, DEFAULT_BINDING};

let config = SdkConfig::default().with_binding("my-tool");
assert_eq!(config.binding(), "my-tool");
```

**Setting `binding` per binding:**

| Binding | Resolves to | Set by |
|---------|-------------|--------|
| Rust SDK direct | `rust` (default) | `xybrid_sdk::DEFAULT_BINDING`, override with `SdkConfig::with_binding(...)` or `xybrid_sdk::set_binding(...)` |
| Flutter | `flutter` | Internal: `XybridSdkClient` calls `xybrid_sdk::set_binding("flutter")` from every FRB entry point |
| Kotlin (Android) | `kotlin` | Internal: `Xybrid.init(context)` calls UniFFI `setBinding("kotlin")` |
| Swift (iOS / macOS) | `swift` | Internal: `Xybrid.initialize()` calls UniFFI `setBinding(binding: "swift")` |
| Unity (C#) | `unity` | Internal: `XybridClient.Initialize()` calls native `xybrid_set_binding("unity")` |

The full wire format and the list of enum values for each header field is documented in [`docs/telemetry/registry.md`](../telemetry/registry.md).

### Implementation Status

| Type | Dart | Kotlin | Swift | C# | Rust |
|------|------|--------|-------|----|------|
| `TelemetryConfiguration` | 🚧 | — | — | ✅ | — |
| `XybridConfiguration` | — | — | — | — | — |
| `SdkConfig` | — | — | — | — | ✅ |

| Method (C# `XybridClient` / `TelemetryConfig`) | Dart | Kotlin | Swift | C# |
|-----------------------------------------------|------|--------|-------|----|
| `TelemetryConfig(apiKey)` ctor | — | — | — | ✅ |
| `Endpoint` property | — | — | — | ✅ |
| `WithEndpoint()` | — | — | — | ✅ |
| `WithAppVersion()` | — | — | — | ✅ |
| `WithDeviceLabel()` | — | — | — | ✅ |
| `WithDeviceAttribute()` | — | — | — | ✅ |
| `WithBatchSize()` | — | — | — | ✅ |
| `WithFlushInterval()` | — | — | — | ✅ |
| `XybridClient.InitializeTelemetry()` | ✅ | — | — | ✅ |
| `XybridClient.FlushTelemetry()` | — | — | — | ✅ |
| `XybridClient.ShutdownTelemetry()` | — | — | — | ✅ |

| Method (Rust `SdkConfig`) | Rust |
|---------------------------|------|
| `with_binding(binding)` | ✅ |
| `binding()` | ✅ |

> **Note**: Dart `Xybrid.initTelemetry(endpoint, apiKey)` ships in xybrid#97 —
> minimal surface matching the C# `InitializeTelemetry()` shape (endpoint + key).
> Flush / shutdown / batch configuration are not yet exposed on Dart; events flush
> on the Rust exporter's default 5 s interval. The C# (Unity) SDK remains the
> reference implementation of the wider telemetry-config surface; Kotlin and
> Swift telemetry init are still planned. `SdkConfig.binding` is Rust-only —
> non-Rust bindings register their identifier through the platform-specific
> entry points listed in [`docs/telemetry/registry.md`](../telemetry/registry.md).

---

## API Versioning

| Version | Status | Breaking Changes |
|---------|--------|------------------|
| 0.1.0   | Draft  | Initial API definition |

---

## Notes for Implementers

### Dart-specific

- Wrapper classes (`XybridModel`, `XybridEnvelope`, etc.) around FRB-generated FFI types
- `BigInt` for u64/usize values
- `Float32List` for embedding vectors
- Generated by `flutter_rust_bridge`

### Kotlin-specific

- Use `suspend` for async operations
- Use `sealed class` for sum types (Envelope)
- Use `data class` for value types
- Use `companion object` for factory methods
- Use Kotlin naming conventions (camelCase, enum UPPER_CASE)
- `init()` takes Android `Context` parameter (platform requirement)

### Swift-specific

- Currently uses raw UniFFI-generated types with type aliases
- No `Xybrid` singleton wrapper yet — uses `XybridModelLoader` directly

### Unity/C#-specific

- Synchronous API (no async/await — runs on Unity main thread)
- Uses C FFI layer (`xybrid-ffi`), not UniFFI
- `IDisposable` pattern for resource management

### Cross-SDK Consistency

1. Same method names (adjusted for language conventions)
2. Same parameter order
3. Same default values
4. Same error semantics (Result vs Exception)
