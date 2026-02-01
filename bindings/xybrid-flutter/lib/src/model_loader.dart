/// Model loading API for Xybrid.
///
/// This class wraps the FRB-generated [FfiModelLoader] and [FfiModel]
/// with a clean, idiomatic Dart API.
library;

import 'envelope.dart';
import 'result.dart';
import 'rust/api/model.dart';

/// Exception thrown when Xybrid operations fail.
class XybridException implements Exception {
  /// The error message.
  final String message;

  /// Creates a new [XybridException] with the given [message].
  XybridException(this.message);

  @override
  String toString() => 'XybridException: $message';
}

/// Prepares a model for loading from registry or local bundle.
class XybridModelLoader {
  /// The underlying FRB model loader.
  final FfiModelLoader _inner;

  XybridModelLoader._(this._inner);

  /// Create a loader for a model from the Xybrid registry.
  ///
  /// The [modelId] should match a model ID in the registry (e.g., "kokoro-82m").
  factory XybridModelLoader.fromRegistry(String modelId) {
    return XybridModelLoader._(
      FfiModelLoader.fromRegistry(modelId: modelId),
    );
  }

  /// Create a loader for a model from a local bundle path.
  ///
  /// The [path] should point to a directory containing model_metadata.json.
  /// Throws if the bundle is invalid.
  factory XybridModelLoader.fromBundle(String path) {
    return XybridModelLoader._(
      FfiModelLoader.fromBundle(path: path),
    );
  }

  /// Load the model asynchronously.
  ///
  /// Downloads the model if loading from registry and not cached.
  /// Returns a ready-to-use [XybridModel] instance.
  Future<XybridModel> load() async {
    final ffiModel = await _inner.load();
    return XybridModel._(ffiModel);
  }
}

/// A loaded model ready for inference.
class XybridModel {
  /// The underlying FRB model.
  final FfiModel inner;

  XybridModel._(this.inner);

  /// Run inference with the given envelope.
  ///
  /// Returns [XybridResult] containing output text, audio, or embeddings
  /// depending on the model type.
  ///
  /// Throws [XybridException] if inference fails.
  Future<XybridResult> run(XybridEnvelope envelope) async {
    try {
      final ffiResult = await inner.run(envelope: envelope.inner);
      return XybridResult.fromFfi(ffiResult);
    } catch (e) {
      throw XybridException('Inference failed: $e');
    }
  }
}
