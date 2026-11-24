import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/result.dart';
import 'package:xybrid_flutter/xybrid.dart';

void main() {
  test('builders preserve structured order, unsigned integers and signed zero',
      () {
    const state =
        '{"z":18446744073709551615,"a":9007199254740993,"f":1.0,"n":-0.0}';
    final request = DecisionRequest(
      state: DecisionContent.jsonText(state),
      questions: [
        DecisionQuestion.choice(
          id: 'second',
          instructions: DecisionContent.text('Select\u0000\u200b'),
          options: [
            DecisionOption('z',
                description: DecisionContent.jsonText('{"y":2,"a":1}')),
            const DecisionOption('a'),
          ],
        ),
        DecisionQuestion.score(
          id: 'first',
          instructions: DecisionContent.text('Rate'),
          levels: [null, DecisionContent.text('high')],
        ),
      ],
    );
    final text = request.toJsonText();
    expect(text, startsWith('{"state":$state,'));
    expect(text.indexOf('"second"'), lessThan(text.indexOf('"first"')));
    expect(text, contains('"criteria":{"z":{"y":2,"a":1},"a":null}'));
    expect(text, contains(r'Select\u0000'));
    expect(jsonDecode(text), isA<Map<String, dynamic>>());
  });

  test('invalid JSON errors do not disclose caller content', () {
    expect(
      () => DecisionContent.jsonText('{secret-canary'),
      throwsA(isA<ArgumentError>().having(
        (error) => error.toString(),
        'message',
        isNot(contains('secret-canary')),
      )),
    );
  });

  test('oversized content and transport refuse before native initialization',
      () {
    expect(
      () => DecisionContent.text('x' * (1024 * 1024 + 1)),
      throwsArgumentError,
    );
    expect(
      () => XybridEnvelope.decisionRequestJson(' ' * (8 * 1024 * 1024 + 1)),
      throwsArgumentError,
    );
  });

  test(
      'decision results preserve f64 and negative infinity without legacy data',
      () {
    const probability = 0.5000000000000001;
    final result = XybridResult.fromFfi(FfiRunResult.decision(FfiDecisionResult(
      answers: [
        FfiDecisionAnswer(
          questionId: 'q',
          distribution: const [
            FfiDecisionScore(
              key: 'first',
              rawScore: 1,
              probability: probability,
              origin: FfiDecisionOrigin.caller,
            ),
            FfiDecisionScore(
              key: 'second',
              rawScore: 0,
              probability: 1 - probability,
              origin: FfiDecisionOrigin.caller,
            ),
            FfiDecisionScore(
              key: 'last',
              rawScore: double.negativeInfinity,
              probability: 0,
              origin: FfiDecisionOrigin.caller,
            ),
          ],
          value: const FfiDecisionValue.choice(key: 'first'),
          confidence: probability,
        ),
      ],
      passes: 1,
      inputTokens: BigInt.from(4096),
      latencyMs: 11,
      executionTarget: FfiExecutionTarget.local,
      metrics: const FfiInferenceMetrics(
          totalMs: 11, tokensOut: 0, stageLatenciesMs: []),
    )));

    expect(result.success, isTrue);
    expect(result.executionTarget, FfiExecutionTarget.local);
    expect(result.text, isNull);
    expect(result.reasoningContent, isNull);
    expect(result.audioBytes, isNull);
    expect(result.embedding, isNull);
    expect(result.toolCalls, isEmpty);
    expect(result.latencyMs, 11);
    expect(result.tokensIn, 4096);
    expect(result.tokensOut, 0);
    final answer = result.decisions!.answers.single;
    expect((answer.value as ChoiceDecision).key, 'first');
    expect(answer.distribution.first.probability, probability);
    expect(answer.distribution.last.rawScore, double.negativeInfinity);
    expect(jsonEncode(answer.distribution.last),
        contains('"raw_score":"-Infinity"'));
  });
}
