import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:piper_state/piper_state.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';
import 'package:xybrid_flutter/src/rust/api/sdk_bridge.dart' as sdk;
import 'package:xybrid_flutter/src/rust/api/benchmark.dart' as benchmark;

import 'package:xybrid_flutter_example/AppConstants.dart';
import 'package:xybrid_flutter_example/dart/string_extension.dart';

import 'audio/audio.dart';
import 'logger.dart';

// Selection types
enum SelectionType { pipeline, model }

// Input type detected from metadata
enum XybridInputType { audio, text, unknown }

// Output type detected from model
enum XybridOutputType { audio, text, unknown }

enum ModelTask { llm, tts, asr, unknown }

// Selection item
class SelectionItem {
  final String id;
  final String displayName;
  final SelectionType type;
  final String? description;
  final String? version;
  final ModelTask task;

  const SelectionItem({
    required this.id,
    required this.displayName,
    required this.type,
    required this.task,
    this.description,
    this.version,
  });

  bool get isAudioModel => task == ModelTask.asr;

  bool get isLlmModel => task == ModelTask.llm;

  bool get isTtsModel => task == ModelTask.tts;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SelectionItem &&
          runtimeType == other.runtimeType &&
          id == other.id;

  @override
  int get hashCode => id.hashCode;
}

// Loading state
enum LoadingState { idle, loading, loaded, error }

// Model status helper
class ModelStatus {
  final String name;
  final String status;

  ModelStatus({required this.name, required this.status});
}

// Recording state
class RecordingStatus {
  final bool isRecording;
  final double amplitude;
  final String? path;
  final String? message;

  const RecordingStatus({
    this.isRecording = false,
    this.amplitude = -60.0,
    this.path,
    this.message,
  });

  RecordingStatus copyWith({
    bool? isRecording,
    double? amplitude,
    String? path,
    String? message,
  }) {
    return RecordingStatus(
      isRecording: isRecording ?? this.isRecording,
      amplitude: amplitude ?? this.amplitude,
      path: path ?? this.path,
      message: message ?? this.message,
    );
  }
}

// Execution state
class ExecutionStatus {
  final bool isExecuting;
  final String log;
  final String result;

  const ExecutionStatus({
    this.isExecuting = false,
    this.log = '',
    this.result = '',
  });

  ExecutionStatus copyWith({bool? isExecuting, String? log, String? result}) {
    return ExecutionStatus(
      isExecuting: isExecuting ?? this.isExecuting,
      log: log ?? this.log,
      result: result ?? this.result,
    );
  }
}

// Streaming state
class StreamingStatus {
  final bool isStreaming;
  final String partialText;
  final int chunksProcessed;
  final int audioDurationMs;

  const StreamingStatus({
    this.isStreaming = false,
    this.partialText = '',
    this.chunksProcessed = 0,
    this.audioDurationMs = 0,
  });

  StreamingStatus copyWith({
    bool? isStreaming,
    String? partialText,
    int? chunksProcessed,
    int? audioDurationMs,
  }) {
    return StreamingStatus(
      isStreaming: isStreaming ?? this.isStreaming,
      partialText: partialText ?? this.partialText,
      chunksProcessed: chunksProcessed ?? this.chunksProcessed,
      audioDurationMs: audioDurationMs ?? this.audioDurationMs,
    );
  }
}

// Benchmark status
class BenchmarkStatus {
  final bool isRunning;
  final int currentIteration;
  final int totalIterations;
  final benchmark.BenchmarkResult? result;

  const BenchmarkStatus({
    this.isRunning = false,
    this.currentIteration = 0,
    this.totalIterations = 0,
    this.result,
  });

  BenchmarkStatus copyWith({
    bool? isRunning,
    int? currentIteration,
    int? totalIterations,
    benchmark.BenchmarkResult? result,
  }) {
    return BenchmarkStatus(
      isRunning: isRunning ?? this.isRunning,
      currentIteration: currentIteration ?? this.currentIteration,
      totalIterations: totalIterations ?? this.totalIterations,
      result: result ?? this.result,
    );
  }
}

class XybridViewModel extends ViewModel {
  final XybridRecorder _audioRecorder;

  XybridViewModel({required XybridRecorder audioRecorder})
    : _audioRecorder = audioRecorder;

  // State holders
  late final availableItems = asyncState<List<SelectionItem>>();
  late final selectedItem = state<SelectionItem?>(null);
  late final loadingState = state<LoadingState>(LoadingState.idle);
  late final statusMessage = state<String>('');
  late final errorMessage = state<String?>(null);
  late final inputType = state<XybridInputType>(XybridInputType.unknown);
  late final outputType = state<XybridOutputType>(XybridOutputType.unknown);
  late final modelStatuses = state<List<ModelStatus>>([]);
  late final textInput = state<String>('');

  // Recording state
  late final recording = state<RecordingStatus>(const RecordingStatus());

  // Execution state
  late final execution = state<ExecutionStatus>(const ExecutionStatus());

  // Streaming state
  late final streaming = state<StreamingStatus>(const StreamingStatus());
  late final streamingMode = state<bool>(false);

  // Benchmark state
  late final benchmarkMode = state<bool>(false);
  late final benchmarkStatus = state<BenchmarkStatus>(const BenchmarkStatus());
  late final benchmarkIterations = state<int>(10);
  late final warmupIterations = state<int>(3);

  // Voice selection state (for TTS models)
  late final availableVoices = state<List<FfiVoiceInfo>>([]);
  late final selectedVoice = state<FfiVoiceInfo?>(null);

  // Model state (new API: Xybrid.model() → loader.load() → model.run())
  sdk.XybridModel? _loadedModel;
  sdk.XybridPipeline? _loadedPipeline;
  XybridStreamer? _streamer;

  LlmClient? _llmClient;
  bool _ttsEnabled = false;

  // Getters for internal state
  XybridRecorder get audioRecorder => _audioRecorder;

  // Load available items
  void loadAvailableItems() {
    load(availableItems, () async {
      final items = <SelectionItem>[];

      // Load pipelines from assets
      final pipelineFiles = [
        'speech-to-text.yaml',
        'text-to-speech.yaml',
        'hiiipe.yaml',
        'llm-tts.yaml',
        'va-whisper-qwen-kokoro.yaml',
        // 'va-wav2vec2-openai-kokoro.yaml',
        // 'va-wav2vec2-openai-kitten.yaml',
        'va-whisper-openai-kitten.yaml',
        'va-whisper-openai-kokoro.yaml',
      ];

      const pipelineDir = 'assets/pipelines';

      String? extractPipelineDescription(String yaml) {
        final lines = yaml.split('\n');
        for (final line in lines) {
          if (line.startsWith('#') && !line.contains('---')) {
            return line.substring(1).trim();
          }
        }
        return null;
      }

      for (final file in pipelineFiles) {
        try {
          final content = await rootBundle.loadString('$pipelineDir/$file');
          final name = content.noExtension() ?? file.replaceAll('.yaml', '');
          items.add(
            SelectionItem(
              id: 'pipeline:$file',
              displayName: name,
              type: SelectionType.pipeline,
              description: extractPipelineDescription(content),
              task: ModelTask.unknown,
            ),
          );
        } catch (e, stack) {
          logger.e('Error loading pipeline $file: $e\n$stack');
          // File doesn't exist, skip
        }
      }

      // Add models from registry
      items.addAll([
        const SelectionItem(
          id: 'model:wav2vec2-base-960h',
          displayName: 'Wav2Vec2 ASR',
          type: SelectionType.model,
          task: ModelTask.asr,
          description: 'Speech recognition (English)',
          version: '1.0',
        ),
        const SelectionItem(
          id: 'model:whisper-tiny',
          displayName: 'Whisper Tiny',
          task: ModelTask.asr,
          type: SelectionType.model,
          description: 'Speech recognition (English)',
          version: '1.0',
        ),
        const SelectionItem(
          id: 'model:qwen2.5-0.5b-instruct',
          displayName: 'Qwen 2.5 LLM',
          task: ModelTask.llm,
          type: SelectionType.model,
          description: 'Text generation (Local LLM)',
          version: '1.0',
        ),
        const SelectionItem(
          id: 'model:gemma-3-1b',
          displayName: 'Gemma 3.1B LLM',
          task: ModelTask.llm,
          type: SelectionType.model,
          description: 'Text generation (Local LLM)',
          version: '1.0',
        ),
        const SelectionItem(
          id: 'model:smollm2-360m',
          displayName: 'Smollm2 360 LLM',
          type: SelectionType.model,
          task: ModelTask.llm,
          description: 'Text generation (Local LLM)',
          version: '1.0',
        ),
        const SelectionItem(
          id: 'model:kitten-tts-nano',
          displayName: 'KittenTTS Nano',
          type: SelectionType.model,
          task: ModelTask.tts,
          description: 'Text-to-speech (8 voices)',
          version: '0.1',
        ),
        const SelectionItem(
          id: 'model:kokoro-82m',
          displayName: 'Kokoro TTS',
          type: SelectionType.model,
          task: ModelTask.tts,
          description: 'Text-to-speech (24 voices)',
          version: '1.0',
        ),
      ]);

      return items;
    });
  }

  // Handle selection change
  Future<void> onSelectionChanged(SelectionItem? item) async {
    if (item == null || item == selectedItem.value) return;

    // Unload previous
    await _unloadCurrent();

    selectedItem.value = item;
    loadingState.value = LoadingState.loading;
    statusMessage.value = 'Loading ${item.displayName}...';
    errorMessage.value = null;
    execution.value = const ExecutionStatus();
    modelStatuses.value = [];

    try {
      if (item.type == SelectionType.pipeline) {
        await _loadPipeline(item);
      } else {
        await _loadModel(item);
      }
    } catch (e, s) {
      logger.e('Error loading item', stackTrace: s, error: e);
      loadingState.value = LoadingState.error;
      errorMessage.value = e.toString();
      statusMessage.value = 'Failed to load';
    }
  }

  Future<void> _unloadCurrent() async {
    final loadedModel = _loadedModel;
    if (loadedModel != null) {
      loadedModel.unload();
      _loadedModel = null;
    }

    // Pipelines don't have unload() - just clear the reference
    _loadedPipeline = null;

    _llmClient?.dispose();
    _llmClient = null;
    _ttsEnabled = false;
  }

  Future<void> _loadPipeline(SelectionItem item) async {
    final filename = item.id.replaceFirst('pipeline:', '');
    final yamlContent = await rootBundle.loadString(
      'assets/pipelines/$filename',
    );

    // For voice-assistant pipeline, we need LLM + models
    /*if (filename == 'voice-assistant.yaml') {
      await _loadVoiceAssistantPipeline(yamlContent);
    } else {*/
    // Use new API: Xybrid.pipeline() → loader.load()
    // final loader = Xybrid.pipeline( 'assets/pipelines/$filename', registry: 'http://localhost:8080');
    // Or
    logger.d("Loading pipeline $filename");
    final loader = Xybrid.pipeline(yaml: yamlContent);
    final pipeline = _loadedPipeline = await loader.load();

    inputType.value = pipeline.inputType.isText()
        ? XybridInputType.text
        : XybridInputType.audio;

    logger.d( pipeline.stageNames);

    // Determine output type from pipeline filename
    final isTextOutput =
        filename.contains('speech-to-text') || filename.contains('llm');
    outputType.value = isTextOutput
        ? XybridOutputType.text
        : XybridOutputType.audio;

    loadingState.value = LoadingState.loaded;
    statusMessage.value = 'Pipeline ready';
    modelStatuses.value = [
      for (final item in pipeline.stageNames)
        ModelStatus(name: item, status: 'ready'),
    ];
    //}
  }

  Future<void> _loadVoiceAssistantPipeline(String yamlContent) async {
    modelStatuses.value = [
      ModelStatus(name: 'ASR (wav2vec2)', status: 'loading...'),
      ModelStatus(name: 'LLM (gateway)', status: 'loading...'),
      ModelStatus(name: 'TTS (kitten)', status: 'loading...'),
    ];

    // Load ASR model using new API: Xybrid.model() → loader.load()
    try {
      final loader = Xybrid.model(modelId: 'wav2vec2-base-960h');
      _loadedModel = await loader.load();
      _updateModelStatus(0, 'loaded');
    } catch (e) {
      _updateModelStatus(0, 'failed: $e');
    }

    // Initialize LLM client
    // API key is configured via Xybrid.setApiKey() in main.dart
    try {
      final apiKey = Xybrid.getApiKey() ?? 'test-key';
      _llmClient = llmClientWithConfig(
        config: LlmClientConfig(
          backend: LlmBackend.gateway,
          gatewayUrl: AppConstants.gatewayUrl,
          apiKey: apiKey,
          defaultModel: 'gpt-4o-mini',
          timeoutMs: 60000,
          debug: true, // Enable debug for troubleshooting
        ),
      );
      _updateModelStatus(1, 'ready');
    } catch (e) {
      _updateModelStatus(1, 'failed: $e');
    }

    // TTS is lazy-loaded on first use
    _ttsEnabled = true;
    _updateModelStatus(2, 'ready (lazy)');

    loadingState.value = LoadingState.loaded;
    statusMessage.value = 'Voice Assistant ready';
    inputType.value = XybridInputType.audio;
  }

  void _updateModelStatus(int index, String status) {
    final current = modelStatuses.value;
    if (index < current.length) {
      final updated = List<ModelStatus>.from(current);
      updated[index] = ModelStatus(name: current[index].name, status: status);
      modelStatuses.value = updated;
    }
  }

  Future<void> _loadModel(SelectionItem item) async {
    final modelId = item.id.replaceFirst('model:', '');

    // Set input type
    inputType.value = item.isAudioModel
        ? XybridInputType.audio
        : XybridInputType.text;

    // Set output type
    if (item.isAudioModel || item.isLlmModel) {
      outputType.value = XybridOutputType.text;
    } else if (item.isTtsModel) {
      outputType.value = XybridOutputType.audio;
    } else {
      outputType.value = XybridOutputType.unknown;
    }

    // Clear voice state
    availableVoices.value = [];
    selectedVoice.value = null;

    try {
      // Use new API: Xybrid.model() → loader.load()
      final loader = Xybrid.model(modelId: modelId);
      _loadedModel = await loader.load();

      // For TTS models, discover available voices
      if (item.isTtsModel && _loadedModel!.hasVoices) {
        final voices = _loadedModel!.voices ?? [];
        availableVoices.value = voices;

        // Set default voice as selected
        final defaultVoice = _loadedModel!.defaultVoice;
        selectedVoice.value =
            defaultVoice ?? (voices.isNotEmpty ? voices.first : null);

        logger.d('TTS model loaded with ${voices.length} voices');
        if (defaultVoice != null) {
          logger.d('Default voice: ${defaultVoice.id} (${defaultVoice.name})');
        }
      }

      loadingState.value = LoadingState.loaded;
      statusMessage.value = '${item.displayName} loaded';
      modelStatuses.value = [ModelStatus(name: modelId, status: 'loaded')];
    } catch (e, s) {
      logger.e('Error loading model', stackTrace: s, error: e);
      loadingState.value = LoadingState.error;
      errorMessage.value = e.toString();
      modelStatuses.value = [ModelStatus(name: modelId, status: 'failed')];
    }
  }

  // Voice selection method
  void onVoiceChanged(FfiVoiceInfo? voice) {
    selectedVoice.value = voice;
    if (voice != null) {
      logger.d('Voice selected: ${voice.id} (${voice.name})');
    }
  }

  String _extractInputKind(String yaml) {
    final match = RegExp(r'kind:\s*"?(\w+)"?').firstMatch(yaml);
    return match?.group(1) ?? 'unknown';
  }

  // Recording methods
  Future<void> startRecording() async {
    try {
      _audioRecorder.onAmplitude = (db) {
        recording.update((r) => r.copyWith(amplitude: db));
      };

      _audioRecorder.onStateChanged = (state) {
        if (state == RecordingState.silenceDetected) {
          execution.update(
            (e) => e.copyWith(log: 'Recording... (silence detected)'),
          );
        } else if (state == RecordingState.voiceDetected) {
          execution.update(
            (e) => e.copyWith(log: 'Recording... (voice detected)'),
          );
        }
      };

      await _audioRecorder.start(config: RecordingConfig.asr);
      recording.update(
        (r) => r.copyWith(
          isRecording: true,
          message: 'Recording... (auto-stops on silence)',
        ),
      );
      execution.update(
        (e) => e.copyWith(log: 'Recording... (auto-stops on silence)'),
      );

      // Wait for auto-stop in background
      _waitForRecordingCompletion();
    } catch (e) {
      logger.e('Start recording error: $e');
      recording.update((r) => r.copyWith(isRecording: false));
      execution.update((ex) => ex.copyWith(log: 'Recording error: $e'));
    }
  }

  Future<void> _waitForRecordingCompletion() async {
    try {
      final result = await _audioRecorder.waitForCompletion();
      recording.update(
        (r) => r.copyWith(
          isRecording: false,
          path: result.path,
          message: result.autoStopped
              ? 'Auto-stopped after silence. Ready to execute.'
              : 'Recording saved (${result.durationMs}ms). Ready to execute.',
        ),
      );
      execution.update(
        (e) => e.copyWith(
          log: result.autoStopped
              ? 'Auto-stopped after silence. Ready to execute.'
              : 'Recording saved (${result.durationMs}ms). Ready to execute.',
        ),
      );
    } catch (e) {
      recording.update((r) => r.copyWith(isRecording: false));
    }
  }

  Future<void> stopRecording() async {
    try {
      final result = await _audioRecorder.stop();
      recording.update(
        (r) => r.copyWith(
          isRecording: false,
          path: result.path,
          message:
              'Recording saved (${result.durationMs}ms). Ready to execute.',
        ),
      );
      execution.update(
        (e) => e.copyWith(
          log: 'Recording saved (${result.durationMs}ms). Ready to execute.',
        ),
      );
    } catch (e, s) {
      logger.e("Stop recording error", error: e, stackTrace: s);
      recording.update((r) => r.copyWith(isRecording: false));
      execution.update((ex) => ex.copyWith(log: 'Stop recording error: $e'));
    }
  }

  // Toggle streaming mode
  void toggleStreamingMode() {
    streamingMode.value = !streamingMode.value;
    if (streamingMode.value) {
      benchmarkMode.value = false; // Disable benchmark when streaming enabled
      execution.update(
        (e) => e.copyWith(log: 'Streaming mode enabled. Tap mic to start.'),
      );
    } else {
      execution.update(
        (e) => e.copyWith(log: 'Batch mode enabled. Record then execute.'),
      );
    }
  }

  // Toggle benchmark mode
  void toggleBenchmarkMode() {
    benchmarkMode.value = !benchmarkMode.value;
    if (benchmarkMode.value) {
      streamingMode.value = false; // Disable streaming when benchmark enabled
      execution.update(
        (e) => e.copyWith(
          log:
              'Benchmark mode enabled. Will run ${benchmarkIterations.value} iterations '
              'with ${warmupIterations.value} warmup.',
        ),
      );
    } else {
      benchmarkStatus.value = const BenchmarkStatus();
      execution.update(
        (e) => e.copyWith(log: 'Normal mode. Ready to execute.'),
      );
    }
  }

  // Update benchmark iterations
  void setBenchmarkIterations(int iterations) {
    benchmarkIterations.value = iterations.clamp(1, 100);
  }

  // Update warmup iterations
  void setWarmupIterations(int iterations) {
    warmupIterations.value = iterations.clamp(0, 20);
  }

  // Start streaming ASR
  Future<void> startStreaming() async {
    final item = selectedItem.value;
    if (item == null) {
      execution.update((e) => e.copyWith(log: 'Select a model first'));
      return;
    }

    // Determine model ID for streaming
    String modelId;
    String version = '1.0';
    if (item.type == SelectionType.model) {
      modelId = item.id.replaceFirst('model:', '');
      // Only ASR models support streaming
      if (!modelId.contains('wav2vec') && !modelId.contains('whisper')) {
        execution.update(
          (e) => e.copyWith(log: 'Streaming only supported for ASR models'),
        );
        return;
      }
      // Use the version from the item if available
      version = item.version ?? '1.0';
    } else {
      // For pipelines, use whisper by default for streaming
      modelId = 'whisper-tiny';
      version = '1.0';
    }

    try {
      execution.update(
        (e) => e.copyWith(log: 'Downloading model from registry...'),
      );

      // Create streamer from registry (downloads and extracts the bundle)
      // Platform is auto-detected when not specified
      _streamer = await XybridStreamer.createFromRegistry(
        config: RegistryStreamingConfig(
          registryUrl: AppConstants.registryUrl,
          modelId: modelId,
          version: version,
          // platform is auto-detected
          enableVad: true,
          language: 'en',
        ),
      );

      // Listen for partial results
      _streamer!.onPartialResult.listen((partial) {
        streaming.update(
          (s) => s.copyWith(
            partialText: partial.text,
            chunksProcessed: partial.chunkIndex,
          ),
        );
        execution.update(
          (e) => e.copyWith(
            log: 'Streaming... (chunk ${partial.chunkIndex})',
            result: partial.text,
          ),
        );
      });

      streaming.value = const StreamingStatus(isStreaming: true);
      execution.update(
        (e) => e.copyWith(log: 'Streaming started. Speak now...'),
      );

      // Start streaming from microphone
      await _audioRecorder.startStreaming(
        onSamples: (samples) async {
          try {
            await _streamer?.feed(samples);
          } catch (e) {
            // Ignore feed errors during streaming
          }
        },
      );
    } catch (err, s) {
      logger.e("Streaming Error", error: err, stackTrace: s);
      streaming.value = const StreamingStatus();

      execution.update((e) => e.copyWith(log: 'Streaming error: $err'));
    }
  }

  // Stop streaming and get final result
  Future<void> stopStreaming() async {
    if (_streamer == null || !streaming.value.isStreaming) {
      return;
    }

    try {
      execution.update((e) => e.copyWith(log: 'Finalizing transcription...'));

      // Stop microphone
      final durationMs = await _audioRecorder.stopStreaming();

      // Flush streamer for final result
      final result = await _streamer!.flush();

      streaming.update(
        (s) => s.copyWith(
          isStreaming: false,
          partialText: result.text,
          audioDurationMs: result.durationMs,
          chunksProcessed: result.chunksProcessed,
        ),
      );

      execution.update(
        (e) => e.copyWith(
          log:
              'Streaming complete (${durationMs}ms, ${result.chunksProcessed} chunks)',
          result: result.text,
        ),
      );

      // Clean up streamer
      _streamer?.dispose();
      _streamer = null;
    } catch (e) {
      streaming.value = const StreamingStatus();
      execution.update((ex) => ex.copyWith(log: 'Stop streaming error: $e'));

      // Clean up
      _streamer?.dispose();
      _streamer = null;
    }
  }

  // Execute based on current selection
  Future<void> execute() async {
    final item = selectedItem.value;
    if (item == null) return;
    if (loadingState.value != LoadingState.loaded) return;

    // Check if benchmark mode is enabled
    if (benchmarkMode.value) {
      await executeBenchmark();
      return;
    }

    execution.value = const ExecutionStatus(
      isExecuting: true,
      log: 'Executing...',
    );

    try {
      /*if (item.id == 'pipeline:voice-assistant') {
        await _executeVoiceAssistant();
      } else*/
      if (item.type == SelectionType.pipeline) {
        await _executePipeline();
      } else {
        await _executeModel();
      }
    } catch (e, s) {
      logger.e("Model Execution error", error: e, stackTrace: s);
      if (kDebugMode) {
        print("Model Execution Error raw \n: $e");
      }
      execution.update((ex) => ex.copyWith(log: 'Error: $e'));
    } finally {
      execution.update((ex) => ex.copyWith(isExecuting: false));
    }
  }

  // Execute benchmark on current model
  Future<void> executeBenchmark() async {
    final item = selectedItem.value;
    if (item == null) return;
    if (loadingState.value != LoadingState.loaded) return;

    // Benchmark only works with models, not pipelines
    if (item.type == SelectionType.pipeline) {
      execution.update(
        (e) => e.copyWith(
          log: 'Benchmark mode only works with models, not pipelines.',
        ),
      );
      return;
    }

    if (_loadedModel == null) {
      execution.update((e) => e.copyWith(log: 'Model not loaded'));
      return;
    }

    final modelId = item.id.replaceFirst('model:', '');
    final isAudioModel =
        modelId.contains('wav2vec') || modelId.contains('whisper');

    // Prepare envelope for benchmark
    Envelope envelope;
    if (isAudioModel) {
      final recordingPath = recording.value.path;
      if (recordingPath == null) {
        execution.update(
          (e) => e.copyWith(log: 'Please record audio first for benchmark'),
        );
        return;
      }
      final audioBytes = await _audioRecorder.readAudioBytes(recordingPath);
      envelope = Envelope.audio(
        audioBytes: audioBytes,
        sampleRate: 16000,
        channels: 1,
      );
    } else {
      // TTS model - use text input with voice selection
      final text = textInput.value.trim();
      if (text.isEmpty) {
        execution.update(
          (e) => e.copyWith(log: 'Please enter text first for benchmark'),
        );
        return;
      }
      final voice = selectedVoice.value;
      envelope = Envelope.text(text: text, voiceId: voice?.id, speed: 1.0);
    }

    final iterations = benchmarkIterations.value;
    final warmup = warmupIterations.value;

    execution.value = const ExecutionStatus(isExecuting: true, log: '');
    benchmarkStatus.value = BenchmarkStatus(
      isRunning: true,
      totalIterations: iterations + warmup,
    );

    execution.update(
      (e) => e.copyWith(
        log:
            'Starting benchmark...\n'
            'Warmup: $warmup iterations\n'
            'Benchmark: $iterations iterations',
      ),
    );

    try {
      final result = await _loadedModel!.benchmark(
        envelope: envelope,
        iterations: iterations,
        warmupIterations: warmup,
      );

      benchmarkStatus.value = BenchmarkStatus(
        isRunning: false,
        totalIterations: iterations,
        result: result,
      );

      execution.update(
        (e) => e.copyWith(log: 'Benchmark complete!', result: result.summary()),
      );
    } catch (e, s) {
      logger.e("Benchmark error", error: e, stackTrace: s);
      benchmarkStatus.value = const BenchmarkStatus(isRunning: false);
      execution.update((ex) => ex.copyWith(log: 'Benchmark error: $e'));
    } finally {
      execution.update((ex) => ex.copyWith(isExecuting: false));
    }
  }

  Future<void> _executePipeline() async {
    if (_loadedPipeline == null) {
      execution.update((e) => e.copyWith(log: 'Pipeline not loaded'));
      return;
    }

    // TODO need api to check is first state text or audio
    // ex _loadedPipeline.firstStage.envelope type check
    final recordingPath = recording.value.path;
    if (recordingPath == null) {
      execution.update((e) => e.copyWith(log: 'Please record audio first'));
      return;
    }

    final audioBytes = await _audioRecorder.readAudioBytes(recordingPath);

    final envelope = Envelope.audio(
      audioBytes: audioBytes,
      sampleRate: 16000,
      channels: 1,
    );

    final result = await _loadedPipeline!.run(envelope: envelope);

    _handlePipelineResult(result);
  }

  Future<void> _executeModel() async {
    final item = selectedItem.value!;

    if (_loadedModel == null) {
      execution.update((e) => e.copyWith(log: 'Model not loaded'));
      return;
    }

    if (item.isAudioModel) {
      // ASR: audio → text
      final recordingPath = recording.value.path;
      if (recordingPath == null) {
        execution.update((e) => e.copyWith(log: 'Please record audio first'));
        return;
      }

      final audioBytes = await _audioRecorder.readAudioBytes(recordingPath);
      final envelope = Envelope.audio(
        audioBytes: audioBytes,
        sampleRate: 16000,
        channels: 1,
      );

      execution.update((e) => e.copyWith(log: 'Running ASR inference...'));

      // Use new API: model.run(envelope) → result.unwrapText()
      final result = await _loadedModel!.run(envelope: envelope);

      _handleNewApiModelResult(result);
    } else if (item.isLlmModel) {
      // LLM: text → text
      final text = textInput.value.trim();
      if (text.isEmpty) {
        execution.update((e) => e.copyWith(log: 'Please enter text first'));
        return;
      }

      execution.update((e) => e.copyWith(log: 'Running LLM inference...'));

      final envelope = Envelope.text(text: text);
      final result = await _loadedModel!.run(envelope: envelope);

      _handleNewApiModelResult(result);
    } else {
      // TTS: text → audio
      final text = textInput.value.trim();
      if (text.isEmpty) {
        execution.update((e) => e.copyWith(log: 'Please enter text first'));
        return;
      }

      // Get selected voice (if any)
      final voice = selectedVoice.value;
      final voiceInfo = voice != null ? ' with voice "${voice.name}"' : '';
      execution.update(
        (e) => e.copyWith(log: 'Running TTS inference$voiceInfo...'),
      );

      // Create envelope with optional voice selection
      final envelope = Envelope.text(
        text: text,
        voiceId: voice?.id,
        speed: 1.0,
      );
      // Use new API: model.run(envelope) → result.unwrapAudio()
      final result = await _loadedModel!.run(envelope: envelope);

      _handleNewApiTtsResult(result);
    }
  }

  Future<void> _executeVoiceAssistant() async {
    final recordingPath = recording.value.path;
    if (recordingPath == null) {
      execution.update((e) => e.copyWith(log: 'Please record audio first'));
      return;
    }

    if (_loadedModel == null) {
      execution.update((e) => e.copyWith(log: 'ASR model not loaded'));
      return;
    }

    // Stage 1: ASR using new API: model.run(envelope) → result.unwrapText()
    execution.update((e) => e.copyWith(log: 'Stage 1/3: Running ASR...'));

    final audioBytes = await _audioRecorder.readAudioBytes(recordingPath);
    final asrEnvelope = Envelope.audio(
      audioBytes: audioBytes,
      sampleRate: 16000,
      channels: 1,
    );

    final asrResult = await _loadedModel!.run(envelope: asrEnvelope);

    if (!asrResult.isText()) {
      execution.update(
        (e) => e.copyWith(
          log: 'ASR failed: No text output (got ${asrResult.outputType})',
        ),
      );
      return;
    }

    final transcription = asrResult.unwrapText();
    execution.update(
      (e) => e.copyWith(
        log:
            'Stage 1/3: ASR complete\n'
            '   You said: "$transcription"\n\n'
            'Stage 2/3: Sending to LLM...',
      ),
    );

    // Stage 2: LLM
    if (_llmClient == null) {
      execution.update((e) => e.copyWith(log: '${e.log}\nLLM not initialized'));
      return;
    }

    final llmResponse = await _llmClient!.chat(
      system:
          'You are a helpful voice assistant. Keep responses concise (2-3 sentences).',
      userMessage: transcription,
    );

    if (!llmResponse.success || llmResponse.text.isEmpty) {
      execution.update(
        (e) => e.copyWith(
          log: '${e.log}\nLLM failed: ${llmResponse.error ?? "No response"}',
        ),
      );
      return;
    }

    execution.update(
      (e) => e.copyWith(
        log:
            '${e.log}\nStage 2/3: LLM complete\n'
            '   Response: "${llmResponse.text}"\n\n'
            'Stage 3/3: Synthesizing speech...',
      ),
    );

    // Stage 3: TTS using new API
    if (_ttsEnabled) {
      // Load TTS model dynamically using new API
      final ttsLoader = Xybrid.model(modelId: 'kitten-tts-nano');

      try {
        final ttsModel = await ttsLoader.load();
        final ttsEnvelope = Envelope.text(text: llmResponse.text);
        final ttsResult = await ttsModel.run(envelope: ttsEnvelope);

        if (ttsResult.isAudio()) {
          final audioOutput = ttsResult.unwrapAudio();
          execution.update(
            (e) => e.copyWith(
              log:
                  '${e.log}\nStage 3/3: TTS complete\n'
                  '   Audio bytes: ${audioOutput.length}',
            ),
          );

          // Play audio
          await _playTtsAudio(Uint8List.fromList(audioOutput));
        } else {
          execution.update(
            (e) => e.copyWith(
              log:
                  '${e.log}\nTTS failed: No audio output (got ${ttsResult.outputType})',
            ),
          );
        }

        // Unload TTS model after use
        ttsModel.unload();
      } catch (e, s) {
        logger.e("TTS failed", error: e, stackTrace: s);
        execution.update((ex) => ex.copyWith(log: '${ex.log}\nTTS failed: $e'));
      }
    }

    execution.update(
      (e) => e.copyWith(
        result: 'You: "$transcription"\n\nAssistant: "${llmResponse.text}"',
      ),
    );
  }

  void _handlePipelineResult(sdk.FfiPipelineResult result) {
    if (result.isText()) {
      execution.update(
        (e) => e.copyWith(
          result: result.unwrapText(),
          log: 'Pipeline completed in (${result.totalLatencyMs}ms)',
        ),
      );
    } else if (result.isAudio()) {
      final audioBytes = result.unwrapAudio();
      execution.update(
        (e) => e.copyWith(
          log:
              'TTS complete (${audioBytes.length} bytes, ${result.totalLatencyMs}ms)',
        ),
      );
      _playTtsAudio(Uint8List.fromList(audioBytes));
    } else {
      execution.update(
        (e) => e.copyWith(
          log: 'Error: Unexpected output type ${result.outputType}',
        ),
      );
    }
  }

  // Result handlers using new API (FfiInferenceResult)
  void _handleNewApiModelResult(sdk.FfiInferenceResult result) {
    if (result.isText()) {
      execution.update(
        (e) => e.copyWith(
          result: result.unwrapText(),
          log: 'Inference complete (${result.latencyMs}ms)',
        ),
      );
    } else {
      execution.update(
        (e) => e.copyWith(
          log: 'Error: Unexpected output type ${result.outputType}',
        ),
      );
    }
  }

  void _handleNewApiTtsResult(sdk.FfiInferenceResult result) {
    if (result.isAudio()) {
      final audioBytes = result.unwrapAudio();
      execution.update(
        (e) => e.copyWith(
          log:
              'TTS complete (${audioBytes.length} bytes, ${result.latencyMs}ms)',
        ),
      );
      _playTtsAudio(Uint8List.fromList(audioBytes));
    } else {
      execution.update(
        (e) => e.copyWith(
          log: 'TTS error: Unexpected output type ${result.outputType}',
        ),
      );
    }
  }

  Future<void> _playTtsAudio(Uint8List audioBytes) async {
    try {
      final wavBytes = wrapInWavHeader(
        audioBytes,
        sampleRate: 24000,
        channels: 1,
      );
      final tempDir = await _audioRecorder.getTempDirectory();
      final wavPath = '$tempDir/tts_output.wav';
      await _audioRecorder.writeAudioFile(wavPath, wavBytes);

      execution.update((e) => e.copyWith(log: '${e.log}\nPlaying audio...'));

      await _audioRecorder.playAudioFile(wavPath);
    } catch (e, s) {
      logger.e("Playback err", error: e, stackTrace: s);
      execution.update(
        (ex) => ex.copyWith(log: '${ex.log}\nPlayback error: $e'),
      );
    }
  }

  @override
  void dispose() {
    _loadedModel?.unload();
    _llmClient?.dispose();
    _streamer?.dispose();
    super.dispose();
  }
}
