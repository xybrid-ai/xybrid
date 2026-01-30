/// Xybrid - Hybrid cloud-edge ML inference orchestrator.
///
/// This library provides a clean Dart API for running ML models on-device
/// or in the cloud, with intelligent routing based on device capabilities.
///
/// ## Quick Start
///
/// ```dart
/// // Load a model from registry
/// final loader = XybridModelLoader.fromRegistry('kokoro-82m');
/// final model = await loader.load();
///
/// // Run text-to-speech
/// final envelope = XybridEnvelope.text('Hello, world!');
/// final result = model.run(envelope);
/// final audioBytes = result.audioBytes;
/// ```
///
/// ## Classes
///
/// - [XybridModelLoader] - Prepare and load models from registry or bundles
/// - [XybridModel] - A loaded model ready for inference
/// - [XybridEnvelope] - Input data wrapper for different modalities
/// - [XybridResult] - Inference output containing text, audio, or embeddings
library xybrid;

export 'src/envelope.dart' show XybridEnvelope;
export 'src/model_loader.dart' show XybridModelLoader, XybridModel;
export 'src/result.dart' show XybridResult;
