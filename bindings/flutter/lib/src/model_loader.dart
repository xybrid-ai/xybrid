/// Model loading API for Xybrid.
///
/// This class wraps the FRB-generated [FfiModelLoader] and [FfiModel]
/// with a clean, idiomatic Dart API.
library;

import 'envelope.dart';
import 'result.dart';

// TODO: Import FRB-generated bindings when available
// import '../src/rust/api/model.dart';

/// Prepares a model for loading from registry or local bundle.
class XybridModelLoader {
  // TODO: Replace with actual FRB type when generated
  // final FfiModelLoader _inner;

  XybridModelLoader._();

  /// Create a loader for a model from the Xybrid registry.
  ///
  /// The [modelId] should match a model ID in the registry (e.g., "kokoro-82m").
  factory XybridModelLoader.fromRegistry(String modelId) {
    // TODO: return XybridModelLoader._()..._inner = FfiModelLoader.fromRegistry(modelId);
    return XybridModelLoader._();
  }

  /// Create a loader for a model from a local bundle path.
  ///
  /// The [path] should point to a directory containing model_metadata.json.
  /// Throws if the bundle is invalid.
  factory XybridModelLoader.fromBundle(String path) {
    // TODO: Call FfiModelLoader.fromBundle(path) and handle errors
    return XybridModelLoader._();
  }

  /// Load the model asynchronously.
  ///
  /// Downloads the model if loading from registry and not cached.
  /// Returns a ready-to-use [XybridModel] instance.
  Future<XybridModel> load() async {
    // TODO: Call _inner.load() and wrap result
    return XybridModel._();
  }
}

/// A loaded model ready for inference.
class XybridModel {
  // TODO: Replace with actual FRB type when generated
  // final FfiModel _inner;

  XybridModel._();

  /// Run inference with the given envelope.
  ///
  /// Returns [XybridResult] containing output text, audio, or embeddings
  /// depending on the model type.
  XybridResult run(XybridEnvelope envelope) {
    // TODO: Call _inner.run(envelope._inner) and wrap result
    return XybridResult.internal();
  }
}
