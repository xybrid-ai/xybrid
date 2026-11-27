// The step 6 staging scenario, run through the public Flutter API.
//
// Implements tools/scripts/observability_staging/README.md like the Python
// reference (bindings/python/tests/observability_staging.py): two sessions in
// one process, `direct` (the Xybrid preset, metadata only) and `collector`
// (OTLP to the pinned Collector, content captured with one literal
// redaction), each loading a real registry model under a remote parent,
// stopping one stream after three chunks, streaming a second to the end, and
// recording feedback and an evaluation on the run reference.
//
// A library, not a test: two lanes run it, and run_staging.py drives both.
// - test/observability_staging_test.dart runs it under `flutter test` on the
//   host (flutter-mac), against the compiled native library.
// - examples/flutter/integration_test/observability_staging_test.dart runs it
//   in the example app on a phone (flutter-pixel, flutter-iphone) through
//   `flutter drive`. It imports this file by a relative path: the file is not
//   part of the package's public API, and both lanes run the same code.
//
// The config holds the staging key. The lanes delete it as soon as it is
// read; nothing here prints it or copies it into the expectations.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

const _schema = 'xybrid-staging-scenario/1';

typedef _Json = Map<String, Object?>;

/// What a stream ended with: its outcome, the token events received, and the
/// last of them (the final token carries the metrics and the run reference).
typedef _Streamed = ({String outcome, int chunks, StreamToken? last});

_Json? _reference(ObservationRef? value) =>
    value == null ? null : {'trace_id': value.traceId, 'span_id': value.spanId};

/// One stats snapshot; the native counters are `BigInt`, written as integers.
_Json _snapshot() {
  final stats = XybridObservability.stats;
  return {
    'enabled': stats.enabled,
    for (final (signal, counts) in [
      ('traces', stats.traces),
      ('logs', stats.logs),
      ('metrics', stats.metrics),
    ])
      signal: {
        'accepted': counts.accepted.toInt(),
        'exported': counts.exported.toInt(),
        'dropped': counts.dropped.toInt(),
        'failed': counts.failed.toInt(),
        'pending': counts.pending.toInt(),
      },
  };
}

Future<String> _outcome(FutureOr<void> Function() call) async {
  try {
    await call();
  } catch (error) {
    return 'error: ${error.runtimeType}: $error';
  }
  return 'ok';
}

/// Fills a staging.json template (`{binding}`, `{lane}`, `{run}`, `{session}`).
String _fill(Object? template, Map<String, String> values) =>
    values.entries.fold(template! as String,
        (text, entry) => text.replaceAll('{${entry.key}}', entry.value));

/// Step 1: the session's export, identity and content rules, then init.
void _configure(_Json config, String session) {
  final constants = config['constants']! as _Json;
  final values = (config['sessions']! as _Json)[session]! as _Json;
  final perLane = constants['per_lane']! as _Json;
  final lane = config['lane']! as String;
  final names = {'binding': config['binding']! as String, 'lane': lane};
  final serviceName = _fill(perLane['service_name'], names);
  final serviceVersion = perLane['service_version']! as String;
  final deviceId = _fill(perLane['device_id'], names);
  final attributes = {
    for (final (key, value) in [
      ('run', config['run']),
      ('lane', lane),
      ('commit', config['commit']),
      ('session', session),
    ])
      'xybrid.acceptance.$key': value! as String,
  };
  XybridObservability.init(session == 'direct'
      ? ObservabilityConfig.xybrid(
          apiKey: values['api_key']! as String,
          tracesEndpoint: values['traces_endpoint']! as String,
          metadataOnly: true,
          serviceName: serviceName,
          serviceVersion: serviceVersion,
          deviceId: deviceId,
          resourceAttributes: attributes,
        )
      : ObservabilityConfig.otlp(
          endpoint: values['endpoint']! as String,
          serviceName: serviceName,
          serviceVersion: serviceVersion,
          deviceId: deviceId,
          resourceAttributes: attributes,
          redactionRules: [
            FfiRedactionRule.replaceLiteral(
                find: constants['secret']! as String,
                replacement: constants['replacement']! as String),
          ],
        ));
}

/// Steps 4 and 5: a push stream, cancelled after [stopAfter] chunks or read
/// to the end. flutter_rust_bridge posts every token to Dart as the model
/// produces it, so tokens can arrive ahead of this loop.
Future<_Streamed> _stream(
    XybridModel model, String prompt, int maxTokens, ObservationOptions parent,
    {int? stopAfter}) async {
  var chunks = 0;
  StreamToken? last;
  await for (final token in model.runStreaming(
    XybridEnvelope.text(prompt),
    config: GenerationConfig(maxTokens: maxTokens),
    options: RunOptions(
        abortPolicy: const AbortPolicy(stopOn: <AbortSignal>{}),
        observation: parent),
  )) {
    if (token.isError) {
      return (
        outcome: 'error: ${token.errorMessage}',
        chunks: chunks,
        last: last
      );
    }
    chunks++;
    last = token;
    if (stopAfter != null && !token.isFinal && chunks >= stopAfter) {
      // Leaving the loop cancels the subscription: the bridge closes its
      // port and the native callback refuses the next token, so the run ends
      // as a user cancellation (error.type cancelled) without a result.
      return (outcome: 'cancelled', chunks: chunks, last: null);
    }
  }
  if (last == null || !last.isFinal) {
    return (
      outcome: 'error: the stream closed without a final token',
      chunks: chunks,
      last: last
    );
  }
  return (
    outcome: stopAfter == null ? 'ok' : 'completed',
    chunks: chunks,
    last: last
  );
}

Future<void> _runSession(_Json config, String session, _Json record) async {
  final constants = config['constants']! as _Json;
  final values = (config['sessions']! as _Json)[session]! as _Json;
  final timeoutMs = constants['timeout_ms']! as int;
  final conversation = _fill(
      (constants['per_lane']! as _Json)['conversation_id'],
      {'run': config['run']! as String, 'session': session});
  final references = <String, Object?>{
    'workflow': null,
    'load': null,
    'run': null
  };
  record.addAll({
    'trace_id': values['trace_id'],
    'parent_span_id': values['parent_span_id'],
    'references': references,
  });
  record['init'] = await _outcome(() => _configure(config, session));
  if (record['init'] != 'ok') {
    throw StateError('$session init failed: ${record['init']}');
  }

  // Step 2: the workflow continues the driver's trace T under span P.
  final workflow = await XybridObservability.startWorkflow(
    constants['workflow_name']! as String,
    options: ObservationOptions(
        traceparent: '00-${values['trace_id']}-${values['parent_span_id']}-01'),
  );
  references['workflow'] = _reference(await workflow.observationRef());
  final context = await workflow.context();
  if (context == null) {
    throw StateError('the workflow returned no context to parent children');
  }
  final parent = ObservationOptions(
    traceparent: context.traceparent,
    tracestate: context.tracestate,
    conversationId: conversation,
  );

  // Step 3: a real registry model, loaded under the workflow.
  final model =
      await XybridModelLoader.fromRegistry(config['model_id']! as String)
          .withObservation(parent)
          .load();
  try {
    references['load'] = _reference(model.loadObservationRef);

    // Step 4: stopped after three chunks by cancelling the subscription.
    final stopped = constants['stopped']! as _Json;
    final halted = await _stream(model, stopped['prompt']! as String,
        stopped['max_tokens']! as int, parent,
        stopAfter: stopped['stop_after_chunks']! as int);
    record['stopped'] = {
      'outcome': halted.outcome,
      'chunks_consumed': halted.chunks,
      'delivery': 'push',
    };

    // Step 5: streamed to the end; the final token carries the run reference.
    final completed = constants['completed']! as _Json;
    final result = await _stream(model, completed['prompt']! as String,
        completed['max_tokens']! as int, parent);
    final last = result.last;
    record['completed'] = {
      'outcome': result.outcome,
      'chunks': result.chunks,
      'tokens_out': last?.metrics?.tokensOut,
      'text_chars': (last?.cumulativeText ?? '').runes.length,
      'finish_reason': last?.finishReason,
    };
    final run = last?.observation;
    references['run'] = _reference(run);
    if (run == null) {
      throw StateError('the completed stream returned no run reference '
          '(${result.outcome})');
    }

    // Step 6: quality results on the run reference.
    final feedback = constants['feedback']! as _Json;
    final evaluation = constants['evaluation']! as _Json;
    record['quality'] = {
      'feedback': XybridObservability.recordFeedback(run,
          score: (feedback['score']! as num).toDouble(),
          label: feedback['label']! as String,
          comment: feedback['comment']! as String),
      'evaluation': XybridObservability.recordEvaluation(run,
          name: evaluation['name']! as String,
          scoreValue: (evaluation['score_value']! as num).toDouble(),
          scoreLabel: evaluation['score_label']! as String,
          explanation: evaluation['explanation']! as String),
    };
  } finally {
    model.inner.dispose();
  }

  // Step 7: lifecycle.
  await workflow.complete();
  workflow.dispose();
  final lifecycle = record['lifecycle'] = <String, Object?>{
    'flush':
        await _outcome(() => XybridObservability.flush(timeoutMs: timeoutMs)),
  };
  final stats = record['stats'] = <String, Object?>{'after_flush': _snapshot()};
  lifecycle['shutdown'] =
      await _outcome(() => XybridObservability.shutdown(timeoutMs: timeoutMs));
  stats['after_shutdown'] = _snapshot();
}

/// Runs both sessions of [config] (the README's config.json, decoded) and
/// returns expectations.json as a map. Never throws: a failed step sets
/// `scenario_error`, and the lane then fails.
///
/// [initialize] loads the native library and calls `Xybrid.init`.
/// [sdkVersion] is the xybrid_flutter version the lane built, when it knows
/// it (the binding has no runtime version call).
Future<Map<String, Object?>> runStagingScenario(
  Map<String, Object?> config, {
  required Future<void> Function() initialize,
  String? sdkVersion,
}) async {
  final sessions = <String, _Json>{};
  final expectations = <String, Object?>{
    'schema': _schema,
    'binding': config['binding'],
    'lane': config['lane'],
    'run': config['run'],
    'sdk_version': sdkVersion,
    'model_id': config['model_id'],
    'device': {
      'system': Platform.operatingSystem,
      'release': Platform.operatingSystemVersion,
      'dart': Platform.version,
    },
    'sessions': sessions,
    'scenario_error': null,
  };
  try {
    await initialize();
    for (final session in const ['direct', 'collector']) {
      await _runSession(config, session, sessions[session] = {});
    }
  } catch (error, stack) {
    final frames = stack.toString().split('\n').take(12).join('\n');
    expectations['scenario_error'] = '$error\n$frames';
    // Stop exporting even when a step failed.
    await _outcome(() => XybridObservability.shutdown(
        timeoutMs: (config['constants']! as _Json)['timeout_ms']! as int));
  }
  debugPrint(const JsonEncoder.withIndent(' ').convert({
    for (final MapEntry(:key, :value) in sessions.entries)
      key: {
        for (final part in ['references', 'stopped', 'completed', 'quality'])
          part: value[part],
      },
  }));
  return expectations;
}
