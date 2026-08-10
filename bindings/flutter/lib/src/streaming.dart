/// Live (rolling-window) ASR for Xybrid.
///
/// Wraps the FRB-generated [FfiStreamSession] with a Dart-friendly API: open a
/// session from a loaded [XybridModel], listen to [partials], feed microphone
/// PCM, then [flush] for the final transcript.
library;

import 'dart:typed_data';

import 'model_loader.dart';
import 'rust/api/streaming.dart';

export 'rust/api/streaming.dart' show FfiPartialResult, FfiVadMode;

/// A live speech-to-text session that transcribes audio as it is fed.
///
/// Audio must be PCM **f32, mono, 16 kHz**. Use [pcm16ToFloat32] to convert
/// the 16-bit PCM most microphone plugins emit.
///
/// ```dart
/// final model = await XybridModelLoader.fromRegistry('whisper-small').load();
/// final asr = await XybridStreamSession.fromModel(model);
/// asr.partials.listen((p) => print(p.text)); // listen before feeding
/// asr.feed(samples);                          // call repeatedly from the mic
/// final transcript = await asr.flush();       // finalize
/// ```
class XybridStreamSession {
  final FfiStreamSession _inner;
  Stream<FfiPartialResult>? _partials;

  XybridStreamSession._(this._inner);

  /// Open a live ASR session for an already-loaded [model].
  ///
  /// The model's on-disk location is resolved from the loaded handle, so a
  /// model from the registry, Hugging Face, a bundle, or a directory all work
  /// the same way — no path needed. The backend (Whisper / Wav2Vec2) is
  /// auto-detected. Throws if the model does not support streaming.
  static Future<XybridStreamSession> fromModel(
    XybridModel model, {
    FfiVadMode vad = const FfiVadMode.off(),
    String? language,
    int? audioCtx,
  }) async {
    final inner = await model.inner.stream(
      config: FfiStreamingConfig(
        sampleRate: 16000,
        vad: vad,
        language: language,
        audioCtx: audioCtx,
      ),
    );
    return XybridStreamSession._(inner);
  }

  /// Partial transcripts, delivered as rolling-window chunks complete.
  ///
  /// Listen to this **before** calling [feed]; audio fed before the first
  /// listener attaches is not reported. The stream is single-subscription.
  Stream<FfiPartialResult> get partials => _partials ??= _inner.subscribe();

  /// Feed PCM f32 mono 16 kHz samples. Cheap and non-blocking — inference
  /// runs off the UI isolate, so calling this at microphone rate is fine.
  ///
  /// Accepts any `List<double>` in range -1.0..1.0; pass a [Float32List] for
  /// the fastest hand-off across the FFI boundary.
  void feed(List<double> samples) => _inner.feed(samples: samples);

  /// Finalize the stream and return the complete transcript.
  ///
  /// After this the session is finalized; further [feed] calls throw.
  Future<String> flush() => _inner.flush();

  /// Reset the session to transcribe fresh audio without reloading the model.
  Future<void> reset() => _inner.reset();

  /// Convert 16-bit little-endian PCM (the common microphone format) to the
  /// f32 samples [feed] expects.
  static Float32List pcm16ToFloat32(Uint8List bytes) {
    final view = ByteData.sublistView(bytes);
    final out = Float32List(bytes.length ~/ 2);
    for (var i = 0; i < out.length; i++) {
      out[i] = view.getInt16(i * 2, Endian.little) / 32768.0;
    }
    return out;
  }
}
