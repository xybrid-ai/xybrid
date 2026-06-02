/// Input envelope types for Xybrid inference.
///
/// This class wraps the FRB-generated [FfiEnvelope] with a clean,
/// idiomatic Dart API.
library;

import 'context.dart';
import 'rust/api/envelope.dart';

/// Raw pixel-buffer format accepted by [XybridEnvelope.imageRaw].
///
/// Mirrors the Rust `PixelFormat` (xybrid-core, `vision` feature) so camera or
/// canvas frames can be sent as raw pixels without JPEG re-encoding. Unsupported
/// raw formats (P010, 10-bit YUV, premultiplied alpha, opaque handles) are
/// rejected as `UnsupportedPixelFormat` before any pixel bytes are read.
enum PixelFormat {
  /// Packed RGB, 8 bits per channel.
  rgb8,

  /// Packed RGBA, 8 bits per channel.
  rgba8,

  /// Packed BGRA, 8 bits per channel.
  bgra8,

  /// Semi-planar YUV 4:2:0 with interleaved UV chroma.
  nv12,

  /// Semi-planar YUV 4:2:0 with interleaved VU chroma.
  nv21,

  /// Tri-planar YUV 4:2:0, also known as I420.
  i420;

  FfiPixelFormat _toFfi() => switch (this) {
        PixelFormat.rgb8 => FfiPixelFormat.rgb8,
        PixelFormat.rgba8 => FfiPixelFormat.rgba8,
        PixelFormat.bgra8 => FfiPixelFormat.bgra8,
        PixelFormat.nv12 => FfiPixelFormat.nv12,
        PixelFormat.nv21 => FfiPixelFormat.nv21,
        PixelFormat.i420 => FfiPixelFormat.i420,
      };
}

/// One memory plane inside a raw pixel image.
///
/// Mirrors the Rust `ImagePlane`. Packed RGB-family inputs carry a single
/// plane; NV12/NV21 carry two; I420 carries three. Plane validation in the
/// core constructor enforces row stride, pixel stride, and extents.
class ImagePlane {
  /// Byte offset into the raw pixel buffer where this plane begins.
  final int offset;

  /// Bytes between adjacent rows in this plane.
  final int rowStride;

  /// Bytes between adjacent samples in the same row.
  final int pixelStride;

  /// Plane width in samples. Chroma planes are usually subsampled.
  final int width;

  /// Plane height in samples. Chroma planes are usually subsampled.
  final int height;

  const ImagePlane({
    required this.offset,
    required this.rowStride,
    required this.pixelStride,
    required this.width,
    required this.height,
  });

  FfiImagePlane _toFfi() => FfiImagePlane(
        offset: BigInt.from(offset),
        rowStride: BigInt.from(rowStride),
        pixelStride: BigInt.from(pixelStride),
        width: width,
        height: height,
      );
}

/// YUV color conversion matrix for raw YUV camera frames.
///
/// Mirrors the Rust `YuvColorMatrix`.
enum YuvColorMatrix {
  /// ITU-R BT.601.
  bt601,

  /// ITU-R BT.709.
  bt709,

  /// ITU-R BT.2020.
  bt2020;

  FfiYuvColorMatrix _toFfi() => switch (this) {
        YuvColorMatrix.bt601 => FfiYuvColorMatrix.bt601,
        YuvColorMatrix.bt709 => FfiYuvColorMatrix.bt709,
        YuvColorMatrix.bt2020 => FfiYuvColorMatrix.bt2020,
      };
}

/// YUV luma/chroma numeric range.
///
/// Mirrors the Rust `YuvColorRange`.
enum YuvColorRange {
  /// Video/limited range.
  limited,

  /// Full range.
  full;

  FfiYuvColorRange _toFfi() => switch (this) {
        YuvColorRange.limited => FfiYuvColorRange.limited,
        YuvColorRange.full => FfiYuvColorRange.full,
      };
}

/// Color metadata required for raw YUV camera frames (`nv12`, `nv21`, `i420`).
///
/// Mirrors the Rust `YuvColorInfo`. RGB-family raw inputs must not carry it.
class YuvColorInfo {
  /// Conversion matrix.
  final YuvColorMatrix matrix;

  /// Numeric range.
  final YuvColorRange range;

  const YuvColorInfo({required this.matrix, required this.range});

  FfiYuvColorInfo _toFfi() => FfiYuvColorInfo(
        matrix: matrix._toFfi(),
        range: range._toFfi(),
      );
}

/// Envelope containing input data for model inference.
///
/// Create envelopes using the factory constructors for different input types:
/// - [XybridEnvelope.audio] for speech recognition
/// - [XybridEnvelope.text] for text-to-speech
/// - [XybridEnvelope.embedding] for embedding models
/// - [XybridEnvelope.image] for encoded image input
/// - [XybridEnvelope.imageRaw] for raw camera/canvas pixel frames
/// - [XybridEnvelope.userMessage] for vision-language prompts
class XybridEnvelope {
  /// The underlying FRB envelope.
  final FfiEnvelope inner;

  final _EnvelopeModality _modality;

  XybridEnvelope._(this.inner, this._modality);

  /// Create an audio envelope for speech recognition.
  ///
  /// [bytes] - Raw audio bytes (e.g., WAV file contents)
  /// [sampleRate] - Audio sample rate in Hz (typically 16000)
  /// [channels] - Number of audio channels (typically 1 for mono)
  factory XybridEnvelope.audio({
    required List<int> bytes,
    required int sampleRate,
    int channels = 1,
  }) {
    return XybridEnvelope._(
      FfiEnvelope.audio(
        bytes: bytes,
        sampleRate: sampleRate,
        channels: channels,
      ),
      _EnvelopeModality.audio,
    );
  }

  /// Create a text envelope for text-to-speech.
  ///
  /// [text] - The text to synthesize
  /// [voiceId] - Optional voice identifier (model-specific)
  /// [speed] - Optional speed multiplier (default 1.0)
  factory XybridEnvelope.text(String text, {String? voiceId, double? speed}) {
    return XybridEnvelope._(
      FfiEnvelope.text(text: text, voiceId: voiceId, speed: speed),
      _EnvelopeModality.text,
    );
  }

  /// Create an embedding envelope from float vector.
  ///
  /// [data] - The embedding vector
  factory XybridEnvelope.embedding(List<double> data) {
    return XybridEnvelope._(
      FfiEnvelope.embedding(data: data),
      _EnvelopeModality.embedding,
    );
  }

  /// Create an encoded image envelope for vision models.
  ///
  /// [bytes] - Encoded PNG, JPEG, or WebP image bytes
  /// [format] - Image format hint: `png`, `jpeg`, `jpg`, or `webp`
  factory XybridEnvelope.image({
    required List<int> bytes,
    required String format,
  }) {
    final normalizedFormat = _normalizeImageFormat(format);
    return XybridEnvelope._(
      FfiEnvelope.image(bytes: bytes, format: normalizedFormat),
      _EnvelopeModality.image,
    );
  }

  /// Create a raw pixel image envelope from a camera or canvas frame.
  ///
  /// Sends pre-decoded pixels straight through to vision models without JPEG
  /// re-encoding. The core constructor validates plane layout, dimensions, and
  /// color metadata, throwing on invalid input (e.g. unsupported [pixelFormat],
  /// plane/stride mismatch, or YUV frames missing [color]).
  ///
  /// [pixels] - Owned raw pixel bytes backing all planes.
  /// [pixelFormat] - Memory layout of [pixels].
  /// [width] / [height] - Image dimensions in pixels.
  /// [planes] - Per-plane descriptors into [pixels]. Packed RGB-family inputs
  ///   carry one plane; NV12/NV21 carry two; I420 carries three.
  /// [color] - Required for YUV formats (`nv12`, `nv21`, `i420`); must be null
  ///   for RGB-family formats.
  factory XybridEnvelope.imageRaw({
    required List<int> pixels,
    required PixelFormat pixelFormat,
    required int width,
    required int height,
    required List<ImagePlane> planes,
    YuvColorInfo? color,
  }) {
    return XybridEnvelope._(
      FfiEnvelope.imageRaw(
        pixels: pixels,
        pixelFormat: pixelFormat._toFfi(),
        width: width,
        height: height,
        planes: planes.map((plane) => plane._toFfi()).toList(growable: false),
        color: color?._toFfi(),
      ),
      _EnvelopeModality.image,
    );
  }

  /// Create a user-role multi-part message with image attachments.
  ///
  /// [images] must contain envelopes created by [XybridEnvelope.image] or
  /// [XybridEnvelope.imageRaw].
  factory XybridEnvelope.userMessage({
    required String text,
    List<XybridEnvelope> images = const [],
  }) {
    final nonImageIndex = images.indexWhere(
      (image) => image._modality != _EnvelopeModality.image,
    );
    if (nonImageIndex != -1) {
      throw ArgumentError.value(
        images,
        'images',
        'all attachments must be image envelopes',
      );
    }

    return XybridEnvelope._(
      FfiEnvelope.userMessage(
        text: text,
        images: images.map((image) => image.inner).toList(growable: false),
      ),
      _EnvelopeModality.multipart,
    );
  }

  /// Create a text envelope with a specific message role.
  ///
  /// This is used for building conversation context with proper role tagging.
  ///
  /// [text] - The message text
  /// [role] - The message role (system, user, or assistant)
  factory XybridEnvelope.textWithRole(String text, MessageRole role) {
    return XybridEnvelope._(
      FfiEnvelope.textWithRole(text: text, role: role.toFfi()),
      _EnvelopeModality.text,
    );
  }

  /// Set the message role on this envelope.
  ///
  /// Returns a new envelope with the role set.
  XybridEnvelope withRole(MessageRole role) {
    return XybridEnvelope._(inner.withRole(role: role.toFfi()), _modality);
  }

  /// Get the message role of this envelope, if set.
  MessageRole? get role {
    final ffiRole = inner.role();
    return ffiRole != null ? MessageRole.fromFfi(ffiRole) : null;
  }

  static String _normalizeImageFormat(String format) {
    switch (format.trim().toLowerCase()) {
      case 'png':
        return 'png';
      case 'jpg':
      case 'jpeg':
        return 'jpeg';
      case 'webp':
        return 'webp';
      default:
        throw ArgumentError.value(
          format,
          'format',
          'expected png, jpeg, jpg, or webp',
        );
    }
  }
}

enum _EnvelopeModality { audio, text, embedding, image, multipart }
