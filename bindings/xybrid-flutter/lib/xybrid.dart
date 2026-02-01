/// Xybrid - Hybrid cloud-edge ML inference orchestrator.
///
/// This library provides a clean Dart API for running ML models on-device
/// or in the cloud, with intelligent routing based on device capabilities.
///
/// ## Initialization
///
/// You must call [Xybrid.init] once before using any Xybrid functionality:
///
/// ```dart
/// void main() async {
///   await Xybrid.init();
///   runApp(MyApp());
/// }
/// ```
///
/// ## Quick Start
///
/// ```dart
/// // Initialize once at app startup
/// await Xybrid.init();
///
/// // Load a model from registry
/// final loader = XybridModelLoader.fromRegistry('kokoro-82m');
/// final model = await loader.load();
///
/// // Run text-to-speech
/// final envelope = XybridEnvelope.text('Hello, world!');
/// final result = await model.run(envelope);
/// final audioBytes = result.audioBytes;
/// ```
///
/// ## Pipelines
///
/// ```dart
/// // Load and run a pipeline
/// final pipeline = XybridPipeline.fromYaml('''
/// name: speech-to-text
/// stages:
///   - model: whisper-small
/// ''');
/// final envelope = XybridEnvelope.audio(bytes: audioData, sampleRate: 16000);
/// final result = await pipeline.run(envelope);
/// ```
///
/// ## Classes
///
/// - [XybridModelLoader] - Prepare and load models from registry or bundles
/// - [XybridModel] - A loaded model ready for inference
/// - [XybridEnvelope] - Input data wrapper for different modalities
/// - [XybridResult] - Inference output containing text, audio, or embeddings
/// - [XybridPipeline] - Multi-stage inference pipeline
library xybrid;

export 'src/envelope.dart' show XybridEnvelope;
export 'src/model_loader.dart' show XybridModelLoader, XybridModel, XybridException;
export 'src/pipeline.dart' show XybridPipeline;
export 'src/result.dart' show XybridResult;
export 'src/utils/utils.dart';
export 'src/xybrid.dart' show Xybrid;
