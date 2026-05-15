import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

void main() {
  test('default run options do not opt into cloud fallback', () {
    const options = RunOptions();

    expect(options.abortPolicy.fallbackToCloud, isFalse);
    expect(options.toFfi().fallbackToCloud, isFalse);
  });

  test('cloud fallback run options expose abort policy controls', () {
    const policy = AbortPolicy.cloudFallback(maxGraceTokens: 2);
    const options = RunOptions.cloudFallback(
      cloudProvider: 'openai',
      cloudModel: 'gpt-4o-mini',
      cloudGatewayUrl: 'http://127.0.0.1:3001/v1',
      correlationId: 'run-123',
      abortPolicy: policy,
    );

    expect(options.abortPolicy.fallbackToCloud, isTrue);
    expect(
      options.abortPolicy.stopOn,
      containsAll({
        AbortSignal.memoryPressureCritical,
        AbortSignal.thermalCritical,
      }),
    );
    final ffi = options.toFfi();
    expect(ffi.abortOnMemoryPressureCritical, isTrue);
    expect(ffi.abortOnThermalCritical, isTrue);
    expect(ffi.fallbackToCloud, isTrue);
    expect(ffi.maxGraceTokens, 2);
  });

  test('custom abort policy maps stop signals and fallback permission to FFI',
      () {
    const options = RunOptions(
      abortPolicy: AbortPolicy(
        stopOn: {AbortSignal.memoryPressureCritical},
        fallbackToCloud: false,
        maxGraceTokens: 1,
      ),
    );

    final ffi = options.toFfi();

    expect(ffi.abortOnMemoryPressureCritical, isTrue);
    expect(ffi.abortOnThermalCritical, isFalse);
    expect(ffi.fallbackToCloud, isFalse);
    expect(ffi.maxGraceTokens, 1);
  });

  test('legacy maxGraceTokens still maps to fallback policy grace tokens', () {
    const options = RunOptions.cloudFallback(maxGraceTokens: 3);

    expect(options.abortPolicy.maxGraceTokens, isNull);
    expect(options.toFfi().maxGraceTokens, 3);
    expect(options.toFfi().fallbackToCloud, isTrue);
  });
}
