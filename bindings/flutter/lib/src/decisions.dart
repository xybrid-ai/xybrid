/// Ordered text/JSON requests and typed decision results.
library;

import 'dart:convert';

import 'rust/api/result.dart';

const _maxTransportBytes = 8 * 1024 * 1024;

String _wire(Iterable<String> parts) {
  final buffer = StringBuffer();
  var bytes = 0;
  for (final part in parts) {
    if (part.length > _maxTransportBytes - bytes) {
      throw ArgumentError('decision JSON exceeds the transport byte limit');
    }
    bytes += utf8.encode(part).length;
    if (bytes > _maxTransportBytes) {
      throw ArgumentError('decision JSON exceeds the transport byte limit');
    }
    buffer.write(part);
  }
  return buffer.toString();
}

String _jsonString(String value) {
  if (value.length > 1024 * 1024 || utf8.encode(value).length > 1024 * 1024) {
    throw ArgumentError('decision content exceeds the content byte limit');
  }
  return jsonEncode(value);
}

/// Content whose original JSON order and numeric spelling survive the bridge.
final class DecisionContent {
  DecisionContent._(this._json);

  /// Plain text content, including embedded NUL and zero-width spaces.
  factory DecisionContent.text(String text) =>
      DecisionContent._(_jsonString(text));

  /// Structured JSON preserved verbatim rather than converted through a map.
  ///
  /// Parsing here only validates syntax. The original text crosses the native
  /// boundary, where the shared protocol applies numeric, depth and byte limits.
  factory DecisionContent.jsonText(String text) {
    if (text.length > _maxTransportBytes ||
        utf8.encode(text).length > _maxTransportBytes) {
      throw ArgumentError('decision JSON exceeds the transport byte limit');
    }
    try {
      jsonDecode(text);
    } on FormatException {
      throw ArgumentError('invalid decision JSON content');
    }
    return DecisionContent._(text);
  }

  final String _json;
}

/// One choice option in caller-defined order.
final class DecisionOption {
  const DecisionOption(this.key, {this.description});
  final String key;
  final DecisionContent? description;
}

/// A typed question whose schema is validated by the shared native decoder.
final class DecisionQuestion {
  DecisionQuestion._(this.id, this._json);

  factory DecisionQuestion.choice({
    required String id,
    required DecisionContent instructions,
    required List<DecisionOption> options,
  }) {
    if (options.isEmpty || options.length > 255) {
      throw ArgumentError('decision choice requires 1–255 options');
    }
    final entries = options.map(
      (option) =>
          '${_jsonString(option.key)}:${option.description?._json ?? 'null'}',
    );
    return DecisionQuestion._(
      id,
      _wire([
        '{"type":"choice","instructions":',
        instructions._json,
        ',"criteria":{',
        _wire(_separated(entries)),
        '}}',
      ]),
    );
  }

  factory DecisionQuestion.yesNo({
    required String id,
    required DecisionContent instructions,
    DecisionContent? yesDescription,
    DecisionContent? noDescription,
  }) =>
      DecisionQuestion._(
        id,
        _wire([
          '{"type":"noul","instructions":',
          instructions._json,
          ',"criteria":{"true":',
          yesDescription?._json ?? 'null',
          ',"false":',
          noDescription?._json ?? 'null',
          '}}',
        ]),
      );

  /// Levels are zero-based and stay in the supplied order.
  factory DecisionQuestion.score({
    required String id,
    required DecisionContent instructions,
    required List<DecisionContent?> levels,
  }) {
    if (levels.length < 2 || levels.length > 10) {
      throw ArgumentError('decision score requires 2–10 levels');
    }
    return DecisionQuestion._(
      id,
      _wire([
        '{"type":"score","instructions":',
        instructions._json,
        ',"criteria":[',
        _wire(_separated(levels.map((level) => level?._json ?? 'null'))),
        ']}',
      ]),
    );
  }

  final String id;
  final String _json;
}

/// One local decision request containing 1–16 ordered questions.
final class DecisionRequest {
  DecisionRequest({this.state, required List<DecisionQuestion> questions})
      : questions = _questions(questions);

  static List<DecisionQuestion> _questions(List<DecisionQuestion> questions) {
    if (questions.isEmpty || questions.length > 16) {
      throw ArgumentError('decision request requires 1–16 questions');
    }
    return List.unmodifiable(questions);
  }

  final DecisionContent? state;
  final List<DecisionQuestion> questions;

  /// Produces Liquid SystemOne wire JSON without reserializing content.
  String toJsonText() => _wire([
        '{"state":',
        state?._json ?? 'null',
        ',"questions":{',
        _wire(_separated(
            questions.map((q) => '${_jsonString(q.id)}:${q._json}'))),
        '}}',
      ]);
}

Iterable<String> _separated(Iterable<String> entries) sync* {
  var first = true;
  for (final entry in entries) {
    if (!first) yield ',';
    first = false;
    yield entry;
  }
}

enum DecisionOrigin { caller, fixed }

enum ScoreStatistic { median, expected }

/// The canonical answer value. Probability outputs are not action policies.
sealed class DecisionValue {
  const DecisionValue();

  factory DecisionValue.fromFfi(FfiDecisionValue value) => switch (value) {
        FfiDecisionValue_Choice(:final key) => ChoiceDecision(key),
        FfiDecisionValue_YesNo(:final pYes) => YesNoDecision(pYes),
        FfiDecisionValue_Score(:final value, :final statistic) => ScoreDecision(
            value,
            switch (statistic) {
              FfiScoreStatistic.median => ScoreStatistic.median,
              FfiScoreStatistic.expected => ScoreStatistic.expected,
            },
          ),
      };
}

final class ChoiceDecision extends DecisionValue {
  const ChoiceDecision(this.key);
  final String key;
}

final class YesNoDecision extends DecisionValue {
  const YesNoDecision(this.pYes);
  final double pYes;
}

final class ScoreDecision extends DecisionValue {
  const ScoreDecision(this.value, this.statistic);
  final double value;
  final ScoreStatistic statistic;
}

/// One ordered probability entry; negative-infinite logits stay representable.
final class DecisionScore {
  DecisionScore.fromFfi(FfiDecisionScore entry)
      : key = entry.key,
        rawScore = entry.rawScore,
        probability = entry.probability,
        origin = switch (entry.origin) {
          FfiDecisionOrigin.caller => DecisionOrigin.caller,
          FfiDecisionOrigin.fixed => DecisionOrigin.fixed,
        };

  final String key;
  final double rawScore;
  final double probability;
  final DecisionOrigin origin;

  /// JSON uses the IR's special score spelling, never a null substitution.
  Map<String, Object> toJson() => {
        'key': key,
        'raw_score':
            rawScore == double.negativeInfinity ? '-Infinity' : rawScore,
        'probability': probability,
        'origin': origin.name,
      };
}

final class DecisionAnswer {
  DecisionAnswer.fromFfi(FfiDecisionAnswer answer)
      : questionId = answer.questionId,
        distribution = List.unmodifiable(
          answer.distribution.map(DecisionScore.fromFfi),
        ),
        value = DecisionValue.fromFfi(answer.value),
        confidence = answer.confidence;

  final String questionId;
  final List<DecisionScore> distribution;
  final DecisionValue value;
  final double? confidence;
}

/// Complete ordered decisions; partial results are never returned.
final class Decisions {
  Decisions.fromFfi(FfiDecisionResult result)
      : answers = List.unmodifiable(result.answers.map(DecisionAnswer.fromFfi)),
        passes = result.passes,
        inputTokens = result.inputTokens.toInt();

  final List<DecisionAnswer> answers;
  final int passes;
  final int inputTokens;
  int get outputTokens => 0;
}
