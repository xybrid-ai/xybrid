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
}
