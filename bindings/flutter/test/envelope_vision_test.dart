import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

void main() {
  test('image envelope rejects unsupported formats before FFI', () {
    expect(
      () => XybridEnvelope.image(bytes: const [1, 2, 3], format: 'heic'),
      throwsArgumentError,
    );
  });

  test('multipart user message constructor is exposed', () {
    final makeUserMessage = XybridEnvelope.userMessage;

    expect(makeUserMessage, isA<Function>());
  });

  // Real raw-frame envelope construction goes through FFI (RustLib.init), so it
  // cannot run in the pure-Dart `flutter test` VM. End-to-end RGB8/NV21 mapping
  // to a core `ImageSource::Raw` envelope is covered by the Rust FFI unit tests
  // in `rust/src/api/envelope.rs`. These tests cover the reachable Dart surface:
  // the factory is exposed and the supporting types plumb coherently.
  test('imageRaw factory is exposed', () {
    final makeImageRaw = XybridEnvelope.imageRaw;

    expect(makeImageRaw, isA<Function>());
  });

  test('PixelFormat exposes the six raw layouts', () {
    expect(PixelFormat.values, hasLength(6));
    expect(
      PixelFormat.values,
      containsAll(<PixelFormat>[
        PixelFormat.rgb8,
        PixelFormat.rgba8,
        PixelFormat.bgra8,
        PixelFormat.nv12,
        PixelFormat.nv21,
        PixelFormat.i420,
      ]),
    );
  });

  test('ImagePlane carries the raw plane descriptor fields', () {
    const plane = ImagePlane(
      offset: 4,
      rowStride: 6,
      pixelStride: 3,
      width: 2,
      height: 2,
    );

    expect(plane.offset, 4);
    expect(plane.rowStride, 6);
    expect(plane.pixelStride, 3);
    expect(plane.width, 2);
    expect(plane.height, 2);
  });

  test('YuvColorInfo composes matrix and range', () {
    const color = YuvColorInfo(
      matrix: YuvColorMatrix.bt709,
      range: YuvColorRange.full,
    );

    expect(color.matrix, YuvColorMatrix.bt709);
    expect(color.range, YuvColorRange.full);
    expect(YuvColorMatrix.values, hasLength(3));
    expect(YuvColorRange.values, hasLength(2));
  });
}
