// P4 spike: decision records round-trip through the FRB-generated Dart
// bindings and the real native library. Skips unless XYBRID_P4_NATIVE_LIB
// names a built libxybrid_flutter_ffi (cargo build -p xybrid_flutter).
// Mirrors tools/spikes/p4-decision-bindings/python/roundtrip.py.

import 'dart:io';

import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/decision.dart';
import 'package:xybrid_flutter/src/rust/frb_generated.dart';

const stateJson =
    '{"zeta":1,"alpha":{"2":true,"10":false,"1":null},'
    '"mid":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808,'
    '9223372036854775808,18446744073709551616,-0],'
    '"é":"\\u00e9\\n","nested":[[{"b":1,"a":2}]],"k":"v"}';
const stateJsonCanonical =
    '{"zeta":1,"alpha":{"2":true,"10":false,"1":null},'
    '"mid":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808,'
    '9223372036854775808,1.8446744073709552e19,0],'
    '"é":"é\\n","nested":[[{"b":1,"a":2}]],"k":"v"}';

FfiDecisionContent text(String value) => FfiDecisionContent.text(value: value);
FfiDecisionContent json(String value) => FfiDecisionContent.json(jsonText: value);

List<String> optionKeys() => [for (var i = 0; i < 30; i++) '${(i * 7) % 30}'];

FfiDecisionContent? optionDescription(int i) => switch (i % 4) {
      0 => text('option $i'),
      1 => json('{"z":$i,"a":[$i.0,"x"]}'),
      2 => null,
      _ => json('"a JSON string becomes text"'),
    };

FfiDecisionRequest request({String state = stateJson}) => FfiDecisionRequest(
      state: json(state),
      questions: [
        FfiDecisionQuestion(
          id: '10',
          instructions: FfiDecisionInstructions.content(
              value: json('{"task":"pick","why":[3,1,2]}')),
          kind: FfiDecisionQuestionKind.choice(options: [
            for (final (i, key) in optionKeys().indexed)
              FfiDecisionOption(key: key, description: optionDescription(i)),
          ]),
        ),
        FfiDecisionQuestion(
          id: '2',
          instructions:
              FfiDecisionInstructions.content(value: text('Is it urgent?')),
          kind: FfiDecisionQuestionKind.yesNo(
              yes: json('{"means":"urgent","rank":1}'), no: null),
        ),
        FfiDecisionQuestion(
          id: '1',
          instructions: null,
          kind: FfiDecisionQuestionKind.score(levels: [
            for (final d in [
              null,
              text('low'),
              json('[1,2.5]'),
              json('"mid"'),
              json('null'),
            ])
              FfiDecisionLevel(description: d),
          ]),
        ),
      ],
    );

List<String> expectedOutline() => [
      'state=json:$stateJsonCanonical',
      'q0:id=10:choice:instructions=json:{"task":"pick","why":[3,1,2]}',
      for (final (i, key) in optionKeys().indexed)
        'q0.o$i:key=$key:${switch (i % 4) {
          0 => 'text:option $i',
          1 => 'json:{"z":$i,"a":[$i.0,"x"]}',
          2 => 'none',
          _ => 'text:a JSON string becomes text',
        }}',
      'q1:id=2:yes_no:instructions=text:Is it urgent?',
      'q1.yes:json:{"means":"urgent","rank":1}',
      'q1.no:none',
      'q2:id=1:score:instructions=none',
      'q2.l0:none',
      'q2.l1:text:low',
      'q2.l2:json:[1,2.5]',
      'q2.l3:text:mid',
      'q2.l4:none',
    ];

String bits32(double value) {
  final data = ByteData(4)..setFloat32(0, value, Endian.little);
  return data.getUint32(0, Endian.little).toRadixString(16).padLeft(8, '0');
}

String bits64(double value) {
  final data = ByteData(8)..setFloat64(0, value, Endian.little);
  return data.getUint64(0, Endian.little).toRadixString(16).padLeft(16, '0');
}

double fromBits64(int bits) =>
    (ByteData(8)..setUint64(0, bits, Endian.little)).getFloat64(0, Endian.little);

List<String> decodedBits(FfiDecisions decisions) => [
      for (final (a, answer) in decisions.answers.indexed) ...[
        for (final (s, score) in answer.distribution.indexed) ...[
          'a$a.s$s.raw=${bits32(score.rawScore)}',
          'a$a.s$s.p=${bits64(score.probability)}',
        ],
        switch (answer.value) {
          FfiDecisionValue_Choice(:final key) => 'a$a.value=choice:$key',
          FfiDecisionValue_YesNo(:final pYes) => 'a$a.value=yes_no:${bits64(pYes)}',
          FfiDecisionValue_Score(:final value, :final statistic) =>
            'a$a.value=score:${bits64(value)}:'
                '${statistic == FfiScoreStatistic.median ? 'Median' : 'Expected'}',
        },
        answer.confidence == null
            ? 'a$a.confidence=none'
            : 'a$a.confidence=${bits64(answer.confidence!)}',
      ],
    ];

/// Structural rendering: generated records compare `List` fields by
/// identity, so equal requests are not `==`.
List<String> render(FfiDecisionRequest request) {
  String content(FfiDecisionContent? c) => switch (c) {
        null => 'none',
        FfiDecisionContent_Text(:final value) => 'text:$value',
        FfiDecisionContent_Json(:final jsonText) => 'json:$jsonText',
      };
  return [
    'state=${content(request.state)}',
    for (final q in request.questions) ...[
      '${q.id}|${switch (q.instructions) {
        FfiDecisionInstructions_Content(:final value) => content(value),
        null => 'none',
      }}',
      ...switch (q.kind) {
        FfiDecisionQuestionKind_Choice(:final options) => [
            for (final o in options) 'o:${o.key}:${content(o.description)}'
          ],
        FfiDecisionQuestionKind_YesNo(:final yes, :final no) => [
            'yes:${content(yes)}|no:${content(no)}'
          ],
        FfiDecisionQuestionKind_Score(:final levels) => [
            for (final l in levels) 'l:${content(l.description)}'
          ],
      },
    ],
  ];
}

void main() {
  final libraryPath = Platform.environment['XYBRID_P4_NATIVE_LIB'];

  setUpAll(() async {
    if (libraryPath == null) return;
    await XybridRustLib.init(externalLibrary: ExternalLibrary.open(libraryPath));
  });

  test('decision records round-trip through FRB', () {
    final req = request();

    expect(decisionSpikeRequestOutline(request: req), expectedOutline(),
        reason: 'Rust saw the state, 3 questions and 30 options in order');

    final normalized = render(request(state: stateJsonCanonical))
        .map((line) => line
            .replaceAll('json:"a JSON string becomes text"',
                'text:a JSON string becomes text')
            .replaceAll('l:json:"mid"', 'l:text:mid')
            .replaceAll('l:json:null', 'l:none'))
        .toList();
    expect(render(decisionSpikeEchoRequest(request: req)), normalized,
        reason: 'echo returns the normalized request');

    final decisions = decisionSpikeDecide(request: req);
    expect(decisions.answers.map((a) => a.questionId), ['10', '2', '1']);
    expect(decisions.answers[0].distribution.map((s) => s.key), optionKeys());
    expect(decisions.answers[1].distribution.map((s) => s.key), ['true', 'false']);
    expect(decisions.answers[1].confidence, isNull);
    expect(decisions.answers[2].distribution[0].rawScore, double.negativeInfinity);
    expect(decisions.usage.evaluatedTokens, BigInt.from(1000 + 1017 + 1034),
        reason: 'u64 crosses as BigInt');

    final expectedBits = decisionSpikeDecideBits(request: req);
    expect(decodedBits(decisions), expectedBits,
        reason: '${expectedBits.length} float bit patterns decode exactly');

    expect(decisionSpikeVerifyDecisions(request: req, decisions: decisions), isNull,
        reason: 'Dart re-encodes the result bit for bit');
    final first = decisions.answers[0].distribution[0];
    final firstBits = ByteData(8)..setFloat64(0, first.probability, Endian.little);
    final nudged = FfiDecisions(
      answers: [
        FfiDecisionAnswer(
          questionId: decisions.answers[0].questionId,
          distribution: [
            FfiDecisionScore(
              key: first.key,
              rawScore: first.rawScore,
              probability: fromBits64(firstBits.getUint64(0, Endian.little) + 1),
              origin: first.origin,
            ),
            ...decisions.answers[0].distribution.skip(1),
          ],
          value: decisions.answers[0].value,
          confidence: decisions.answers[0].confidence,
        ),
        ...decisions.answers.skip(1),
      ],
      usage: decisions.usage,
    );
    expect(decisionSpikeVerifyDecisions(request: req, decisions: nudged),
        'answers[0].distribution[0]',
        reason: 'a one-ULP change is detected');

    Object? error;
    try {
      decisionSpikeDecide(request: request(state: '{"secret":1,"secret":2}'));
    } catch (e) {
      error = e;
    }
    expect(error, isA<String>(), reason: 'FRB surfaces the error as a String');
    expect(error as String, allOf(contains('state'), isNot(contains('secret'))),
        reason: 'the refusal is content-free');
  }, skip: libraryPath == null ? 'set XYBRID_P4_NATIVE_LIB' : false);

  test('probe shapes survive FRB', () {
    final probe = FfiProbeOptionalList(
        levels: [null, text('x'), json('[1]'), null], labels: [null, 'a', null]);
    final probeBack = decisionSpikeProbeOptionalList(probe: probe);
    expect(probeBack.levels, probe.levels);
    expect(probeBack.labels, probe.labels);

    final nanPayload = fromBits64(0x7FF8000000001234);
    final specials = Float64List.fromList(
        [-0.0, 5e-324, double.maxFinite, double.negativeInfinity, nanPayload]);
    final nesting = FfiProbeNesting(
      rawScores: [
        Float32List.fromList([1.5, double.negativeInfinity]),
        Float32List(0),
        Float32List.fromList([-0.0]),
      ],
      probabilities: [specials, Float64List.fromList([0.1 + 0.2])],
      usage: FfiDecisionUsage(
          passes: 0xFFFFFFFF, evaluatedTokens: (BigInt.one << 64) - BigInt.one),
    );
    final back = decisionSpikeProbeNesting(probe: nesting);
    expect(back.probabilities[0].map(bits64), specials.map(bits64),
        reason: 'f64 bits (-0.0, subnormal, max, -inf, NaN payload)');
    expect(bits64(back.probabilities[1][0]), bits64(0.1 + 0.2));
    expect([for (final row in back.rawScores) row.map(bits32).toList()],
        [for (final row in nesting.rawScores) row.map(bits32).toList()]);
    expect(back.usage, nesting.usage, reason: 'Option<record> and u64::MAX survive');

    final tree = decisionSpikeProbeJson(
            jsonText: '{"2":1,"10":[1.0,-0,18446744073709551615],"1":{"z":null,"a":"s"}}')
        as FfiProbeJson_Obj;
    expect(tree.entries.map((e) => e.key), ['2', '10', '1'],
        reason: 'integer-like keys keep document order');
    final items = (tree.entries[1].value as FfiProbeJson_Seq).items;
    expect(items[0], isA<FfiProbeJson_Float>());
    expect(items[1], isA<FfiProbeJson_Int>());
    expect(items[2], isA<FfiProbeJson_Uint>());
    expect(decisionSpikeProbeJsonEcho(value: tree), tree,
        reason: 'recursive JSON tree echoes (freezed compares lists deeply)');

    final leaks = request().toString().contains('urgent') ||
        text('Is it urgent?').toString().contains('urgent');
    // ignore: avoid_print
    print('note toString() shows caller content: $leaks');
  }, skip: libraryPath == null ? 'set XYBRID_P4_NATIVE_LIB' : false);
}
