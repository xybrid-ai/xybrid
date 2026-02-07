/// Pipeline loading and execution API for Xybrid.
///
/// This class wraps the FRB-generated [FfiPipeline] with a clean,
/// idiomatic Dart API.
library;

import 'envelope.dart';
import 'model_loader.dart';
import 'result.dart';
import 'rust/api/pipeline.dart';

/// A loaded pipeline ready for multi-stage inference.
///
/// Pipelines chain multiple models together, passing output from one stage
/// to the next. Create pipelines using the factory constructors:
/// - [XybridPipeline.fromYaml] for inline YAML definitions
/// - [XybridPipeline.fromFile] for YAML file paths
/// - [XybridPipeline.fromBundle] for pipeline bundles
///
/// ## Example
///
/// ```dart
/// // Load a pipeline from YAML
/// final pipeline = XybridPipeline.fromYaml('''
/// name: speech-to-text
/// stages:
///   - model: whisper-small
///     input: audio
/// ''');
///
/// // Run the pipeline
/// final envelope = XybridEnvelope.audio(bytes: audioData, sampleRate: 16000);
/// final result = await pipeline.run(envelope);
/// print(result.text);
/// ```
class XybridPipeline {
  /// The underlying FRB pipeline.
  final FfiPipeline _inner;

  XybridPipeline._(this._inner);

  /// Load a pipeline from a YAML string.
  ///
  /// Parses the YAML and resolves all model references via the registry.
  ///
  /// Throws [XybridException] if the YAML is invalid or models cannot be found.
  factory XybridPipeline.fromYaml(String yaml) {
    try {
      return XybridPipeline._(FfiPipeline.fromYaml(yaml: yaml));
    } catch (e) {
      throw XybridException('Failed to load pipeline from YAML: $e');
    }
  }

  /// Load a pipeline from a YAML file path.
  ///
  /// Reads the file, parses the YAML, and resolves all model references.
  ///
  /// Throws [XybridException] if the file cannot be read or is invalid.
  factory XybridPipeline.fromFile(String path) {
    try {
      return XybridPipeline._(FfiPipeline.fromFile(path: path));
    } catch (e) {
      throw XybridException('Failed to load pipeline from file: $e');
    }
  }

  /// Load a pipeline from a bundle path.
  ///
  /// Bundles may include the pipeline definition and associated model files.
  ///
  /// Throws [XybridException] if the bundle cannot be loaded.
  factory XybridPipeline.fromBundle(String path) {
    try {
      return XybridPipeline._(FfiPipeline.fromBundle(path: path));
    } catch (e) {
      throw XybridException('Failed to load pipeline from bundle: $e');
    }
  }

  /// Get the pipeline name (if specified in the YAML definition).
  String? get name => _inner.name();

  /// Get the number of stages in the pipeline.
  int get stageCount => _inner.stageCount().toInt();

  /// Get the stage names/identifiers in execution order.
  List<String> get stageNames => _inner.stageNames();

  /// Execute the pipeline with the given input envelope.
  ///
  /// Returns [XybridResult] containing output from the final stage.
  ///
  /// Throws [XybridException] if execution fails.
  Future<XybridResult> run(XybridEnvelope envelope) async {
    try {
      final ffiResult = await _inner.run(envelope: envelope.inner);
      return XybridResult.fromFfi(ffiResult);
    } catch (e) {
      throw XybridException('Pipeline execution failed: $e');
    }
  }
}
