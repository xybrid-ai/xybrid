import 'dart:io';

import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/sdk_client.dart';
import 'package:xybrid_flutter/src/rust/frb_generated.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

/// Executes the real bridge when a compiled native library and pinned local
/// decision model are staged. A qualification lane must require them.
void main() {
  final library = Platform.environment['XYBRID_FLUTTER_NATIVE_LIBRARY'];
  final modelDirectory = Platform.environment['XYBRID_DECISION_MODEL_DIR'];
  final required = Platform.environment['XYBRID_REQUIRE_MODELS'] == '1';
  final available = library != null &&
      modelDirectory != null &&
      File(library).existsSync() &&
      Directory(modelDirectory).existsSync();

  test('decision requests cross the real Dart-to-Rust bridge', () async {
    expect(available, isTrue,
        reason:
            'stage XYBRID_FLUTTER_NATIVE_LIBRARY and XYBRID_DECISION_MODEL_DIR');
    await XybridRustLib.init(externalLibrary: ExternalLibrary.open(library!));
    final cache =
        await Directory.systemTemp.createTemp('xybrid-flutter-decision-');
    XybridSdkClient.initSdkCacheDir(cacheDir: cache.path);
    final model = await XybridModelLoader.fromDirectory(modelDirectory!).load();
    try {
      final request = DecisionRequest(
        state: DecisionContent.jsonText(
          '{"last":9223372036854775807,"first":9007199254740993,"zero":-0.0}',
        ),
        questions: [
          DecisionQuestion.choice(
            id: 'route',
            instructions:
                DecisionContent.text('Choose the only option.\u0000\u200b'),
            options: const [DecisionOption('only')],
          ),
          DecisionQuestion.score(
            id: 'level',
            instructions: DecisionContent.text('How much data is present?'),
            levels: [
              DecisionContent.text('little'),
              DecisionContent.text('much')
            ],
          ),
        ],
      );
      final first = await model.run(XybridEnvelope.decisionRequest(request));
      expect(first.executionTarget, FfiExecutionTarget.local);
      expect(first.decisions!.answers.map((answer) => answer.questionId),
          ['route', 'level']);
      expect(
          (first.decisions!.answers.first.value as ChoiceDecision).key, 'only');
      expect(
          first.decisions!.answers.first.distribution.single.probability, 1.0);
      expect(first.decisions!.answers.last.value, isA<ScoreDecision>());
      expect(first.tokensIn, greaterThan(0));
      expect(first.tokensOut, 0);
      expect(first.text, isNull);

      // The neutral IR preserves unsigned values, but Liquid's signed Jinja
      // profile must refuse them before inference without echoing content.
      final overflow = DecisionRequest(
        state: DecisionContent.jsonText(
          '{"private_canary":18446744073709551615}',
        ),
        questions: request.questions,
      );
      expect(
        () => XybridEnvelope.decisionRequest(overflow),
        throwsA(predicate<Object>((error) {
          final message = error.toString();
          return message.contains('number_out_of_range') &&
              !message.contains('private_canary') &&
              !message.contains('18446744073709551615');
        }, 'content-free number_out_of_range refusal')),
      );
      await expectLater(
        model.run(XybridEnvelope.decisionRequest(request),
            config: const GenerationConfig(maxTokens: 1)),
        throwsA(isA<XybridException>().having(
          (error) => error.message,
          'unsupported generation settings',
          contains('decision inference does not support generation settings'),
        )),
      );
      final cancel = CancellationToken()..cancel();
      await expectLater(
        model.run(XybridEnvelope.decisionRequest(request),
            cancellationToken: cancel),
        throwsA(isA<XybridException>()),
      );
      final duringRun = CancellationToken();
      final manyQuestions = DecisionRequest(
        state: request.state,
        questions: List.generate(
          16,
          (index) => DecisionQuestion.choice(
            id: 'q$index',
            instructions: DecisionContent.text('Choose the only option.'),
            options: const [DecisionOption('only')],
          ),
        ),
      );
      final pending = model.run(
        XybridEnvelope.decisionRequest(manyQuestions),
        cancellationToken: duringRun,
      );
      final cancelled = expectLater(
        pending,
        throwsA(isA<XybridException>().having(
          (error) => error.message.toLowerCase(),
          'cancellation category',
          contains('cancel'),
        )),
      );
      // A run transfers a fork. The original opaque handle must remain live
      // after dispatch; move:true on the original used to dispose it here.
      expect(duringRun.inner.isCancelled(), isFalse);
      await Future<void>.delayed(const Duration(milliseconds: 10));
      duringRun.cancel();
      expect(duringRun.inner.isCancelled(), isTrue);
      await cancelled;

      final recovered =
          await model.run(XybridEnvelope.decisionRequest(request));
      expect(recovered.tokensIn, first.tokensIn);
      expect((recovered.decisions!.answers.first.value as ChoiceDecision).key,
          'only');
    } finally {
      await model.unload();
      XybridRustLib.dispose();
      await cache.delete(recursive: true);
    }
  },
      skip: available || required
          ? false
          : 'native decision artifacts are not staged',
      timeout: const Timeout(Duration(minutes: 5)));
}
