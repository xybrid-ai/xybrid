// The shared observability scenario, run through the public Flutter API on
// the real native library.
//
// Implements tools/scripts/observability_capture/scenario.md step for step,
// like the Python reference (bindings/python/tests/observability_scenario.py):
// export to the capture receiver, then describe in expectations.json what the
// binding said it did, which check_capture.py compares with what arrived. No
// model weights are involved.
//
// Skipped unless XYBRID_SCENARIO_ENDPOINT is set, so `flutter test` stays
// native-free. run_scenario.py sets it, together with
// XYBRID_SCENARIO_EXPECTATIONS and XYBRID_SCENARIO_WORKDIR:
//
//   cargo build -p xybrid_flutter --features platform-desktop
//   cd bindings/flutter
//   XYBRID_FLUTTER_NATIVE_LIB=$PWD/../../target/debug/libxybrid_flutter_ffi.so \
//   python3 ../../tools/scripts/observability_capture/run_scenario.py \
//     --binding flutter --out-dir /tmp/flutter-scenario -- \
//     flutter test --no-pub test/observability_scenario_test.dart
//
// XYBRID_FLUTTER_NATIVE_LIB names the compiled libxybrid_flutter_ffi; without
// it `Xybrid.init()` uses flutter_rust_bridge's default loader (which honours
// FRB_DART_LOAD_EXTERNAL_LIBRARY_NATIVE_LIB_DIR). XYBRID_SCENARIO_SPEC
// overrides the scenario.json path.
@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';

import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart'
    show ExternalLibrary;
import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/observability.dart'
    show FfiSignalStats;
import 'package:xybrid_flutter/src/rust/frb_generated.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

const _binding = 'flutter';
const _schema = 'xybrid-observability-scenario/1';
const _counters = ['accepted', 'exported', 'dropped', 'failed', 'pending'];
const _referenceNames = [
  'workflow',
  'tool',
  'inference',
  'inference_failed',
  'inference_released',
  'load',
  'run',
];

typedef _Json = Map<String, Object?>;

String _hex(String text) => utf8
    .encode(text)
    .map((byte) => byte.toRadixString(16).padLeft(2, '0'))
    .join();

/// T and P: hex of the UTF-8 name, right-padded with zeros (scenario.md,
/// Identifiers).
(String, String) _scenarioIds(String binding) => (
      _hex('xy:$binding').padRight(32, '0').substring(0, 32),
      _hex(binding).padRight(16, '0').substring(0, 16),
    );

_Json? _reference(ObservationRef? value) =>
    value == null ? null : {'trace_id': value.traceId, 'span_id': value.spanId};

/// One stats snapshot; the native counters are `BigInt`, written as integers.
_Json _snapshot() {
  final stats = XybridObservability.stats;
  _Json signal(FfiSignalStats counts) {
    final values = [
      counts.accepted,
      counts.exported,
      counts.dropped,
      counts.failed,
      counts.pending,
    ];
    return {
      for (var i = 0; i < _counters.length; i++) _counters[i]: values[i].toInt()
    };
  }

  return {
    'enabled': stats.enabled,
    'traces': signal(stats.traces),
    'logs': signal(stats.logs),
    'metrics': signal(stats.metrics),
  };
}

Future<String> _outcome(Future<void> Function() call) async {
  try {
    await call();
  } catch (error) {
    return 'error: ${error.runtimeType}: $error';
  }
  return 'ok';
}

String _describe(XybridException error) =>
    '${error.runtimeType}: ${error.message}';

String _perBinding(_Json spec, String key, String binding) =>
    ((spec['per_binding']! as _Json)[key]! as String)
        .replaceAll('{binding}', binding);

/// Performs steps (a) to (e), filling [expectations] as it goes.
Future<void> _runScenario(
    _Json spec, String endpoint, Directory workdir, _Json expectations) async {
  final binding = expectations['binding']! as String;
  final traceId = expectations['trace_id']! as String;
  final parentSpanId = expectations['parent_span_id']! as String;
  final conversation = _perBinding(spec, 'conversation_id', binding);
  final timeoutMs = spec['timeout_ms']! as int;
  final references = expectations['references']! as _Json;
  final stats = expectations['stats']! as _Json;
  final lifecycle = expectations['lifecycle']! as _Json;
  final automatic = expectations['automatic']! as _Json;
  final secret = spec['secret']! as String;

  // (a) OTLP/JSON export to the receiver, with content rules and a device id.
  XybridObservability.init(ObservabilityConfig.otlp(
    endpoint: endpoint,
    jsonProtocol: true,
    serviceName: _perBinding(spec, 'service_name', binding),
    serviceVersion: _perBinding(spec, 'service_version', binding),
    deviceId: _perBinding(spec, 'device_id', binding),
    redactionRules: [
      FfiRedactionRule.replaceLiteral(
          find: secret, replacement: spec['replacement']! as String),
      FfiRedactionRule.omitAttribute(key: spec['omitted_attribute']! as String),
    ],
  ));

  // (b) Parent-in: the workflow continues the caller's trace T under span P.
  final workflow = await XybridObservability.startWorkflow(
    spec['workflow_name']! as String,
    options: ObservationOptions(traceparent: '00-$traceId-$parentSpanId-01'),
  );
  references['workflow'] = _reference(await workflow.observationRef());
  final context = await workflow.context();
  if (context == null) {
    throw StateError('the workflow returned no context to parent children');
  }
  final child = ObservationOptions(
      traceparent: context.traceparent, tracestate: context.tracestate);

  // (c) Manual children: a tool, a completed inference, a failed one and one
  // released unfinished.
  final toolSpec = spec['tool']! as _Json;
  final tool = await XybridObservability.startTool(
    ToolCallInfo(
      name: toolSpec['name']! as String,
      callId: toolSpec['call_id'] as String?,
      toolType: toolSpec['tool_type'] as String?,
      description: toolSpec['description'] as String?,
      argumentsJson: toolSpec['arguments_json'] as String?,
    ),
    options: child,
  );
  try {
    await tool.recordResult(resultJson: toolSpec['result_json']! as String);
    references['tool'] = _reference(await tool.observationRef());
    await tool.complete();
  } finally {
    tool.dispose();
  }

  final requestSpec = spec['inference_request']! as _Json;
  final request = InferenceRequest(
    operation: requestSpec['operation']! as String,
    provider: requestSpec['provider']! as String,
    model: requestSpec['model'] as String?,
    location: requestSpec['location'] == 'local'
        ? ExecutionLocation.local
        : ExecutionLocation.remote,
    runtime: requestSpec['runtime'] as String?,
    task: requestSpec['task'] as String?,
    stream: requestSpec['stream']! as bool,
    maxTokens: requestSpec['max_tokens'] as int?,
    temperature: (requestSpec['temperature'] as num?)?.toDouble(),
    topP: (requestSpec['top_p'] as num?)?.toDouble(),
    serverAddress: requestSpec['server_address'] as String?,
    serverPort: requestSpec['server_port'] as int?,
  );
  final content = spec['inference']! as _Json;
  final inference =
      await XybridObservability.startInference(request, options: child);
  try {
    await inference.recordSystemInstructions(
        partsJson: content['system_instructions_json']! as String);
    await inference.recordInputMessages(
        messagesJson: content['input_messages_json']! as String);
    await inference.recordOutputMessages(
        messagesJson: content['output_messages_json']! as String);
    references['inference'] = _reference(await inference.observationRef());
    await inference.complete();
  } finally {
    inference.dispose();
  }

  final failure = spec['failed_inference']! as _Json;
  final failed =
      await XybridObservability.startInference(request, options: child);
  references['inference_failed'] = _reference(await failed.observationRef());
  await failed.fail(
      errorType: failure['error_type']! as String,
      message: failure['message'] as String?);
  failed.dispose();

  final abandoned = spec['released_inference']! as _Json;
  final released =
      await XybridObservability.startInference(request, options: child);
  await released.recordInputMessages(
      messagesJson: abandoned['input_messages_json']! as String);
  references['inference_released'] =
      _reference(await released.observationRef());
  stats['before_release'] = _snapshot();
  // The last owner, never ended: disposal cancels and does not flush.
  released.dispose();
  stats['after_release'] = _snapshot();

  // (d) The automatic path: a weightless model loads lazily and fails to run,
  // both under the workflow with a conversation id.
  final modelSpec = spec['model']! as _Json;
  final modelDir =
      Directory('${workdir.path}/${modelSpec['directory_name']! as String}')
        ..createSync(recursive: true);
  File('${modelDir.path}/model_metadata.json')
      .writeAsStringSync(modelSpec['metadata_json']! as String);
  final automaticParent = ObservationOptions(
    traceparent: context.traceparent,
    tracestate: context.tracestate,
    conversationId: conversation,
  );
  XybridModel? model;
  try {
    model = await XybridModelLoader.fromDirectory(modelDir.path)
        .withObservation(automaticParent)
        .load();
  } on XybridException catch (error) {
    automatic['load'] = {'outcome': 'error', 'error': _describe(error)};
    automatic['run'] = {'outcome': 'skipped', 'error': null};
  }
  if (model != null) {
    automatic['load'] = {'outcome': 'ok', 'error': null};
    references['load'] = _reference(model.loadObservationRef);
    try {
      final result = await model.run(
        XybridEnvelope.text(modelSpec['run_input_text']! as String),
        options: RunOptions(observation: automaticParent),
      );
      automatic['run'] = {'outcome': 'ok', 'error': null};
      references['run'] = _reference(result.observation);
    } on XybridException catch (error) {
      automatic['run'] = {'outcome': 'error', 'error': _describe(error)};
    } finally {
      model.inner.dispose();
    }
  }

  await workflow.complete();
  workflow.dispose();

  // (e) Lifecycle: flush, stats, shutdown twice, then a no-op workflow.
  lifecycle['flush'] =
      await _outcome(() => XybridObservability.flush(timeoutMs: timeoutMs));
  stats['after_flush'] = _snapshot();
  lifecycle['shutdown'] = [
    await _outcome(() => XybridObservability.shutdown(timeoutMs: timeoutMs)),
    await _outcome(() => XybridObservability.shutdown(timeoutMs: timeoutMs)),
  ];
  stats['after_shutdown'] = _snapshot();
  final after = await XybridObservability.startWorkflow(
      spec['after_shutdown_workflow_name']! as String);
  final afterRef = await after.observationRef();
  final afterContext = await after.context();
  await after.complete();
  after.dispose();
  lifecycle['after_shutdown'] = {
    'workflow_ref': _reference(afterRef),
    'workflow_context': afterContext?.traceparent,
    'flush':
        await _outcome(() => XybridObservability.flush(timeoutMs: timeoutMs)),
  };
}

/// Loads the native library from XYBRID_FLUTTER_NATIVE_LIB when given, then
/// initializes the SDK as an application would.
Future<void> _initialize(Map<String, String> env) async {
  final library = env['XYBRID_FLUTTER_NATIVE_LIB'];
  if (library != null && library.isNotEmpty) {
    if (!File(library).existsSync()) {
      throw StateError('XYBRID_FLUTTER_NATIVE_LIB=$library does not exist');
    }
    await XybridRustLib.init(externalLibrary: ExternalLibrary.open(library));
  }
  await Xybrid.init();
}

String _packageVersion() {
  final pubspec = File('pubspec.yaml').readAsStringSync();
  return RegExp(r'^version:\s*(\S+)', multiLine: true)
          .firstMatch(pubspec)
          ?.group(1) ??
      'unknown';
}

void main() {
  final env = Platform.environment;
  final endpoint = env['XYBRID_SCENARIO_ENDPOINT'];

  test(
    'observability scenario exports through the native library',
    () async {
      final expectationsPath = env['XYBRID_SCENARIO_EXPECTATIONS'];
      if (expectationsPath == null || expectationsPath.isEmpty) {
        fail('XYBRID_SCENARIO_EXPECTATIONS is required with an endpoint');
      }
      final expectationsFile = File(expectationsPath);
      final workdirPath = env['XYBRID_SCENARIO_WORKDIR'];
      final workdir = workdirPath == null || workdirPath.isEmpty
          ? Directory('${expectationsFile.parent.path}/work')
          : Directory(workdirPath);
      final specPath = env['XYBRID_SCENARIO_SPEC'] ??
          '../../tools/scripts/observability_capture/scenario.json';
      final spec =
          jsonDecode(File(specPath).readAsStringSync()) as Map<String, Object?>;

      final (traceId, parentSpanId) = _scenarioIds(_binding);
      final expectations = <String, Object?>{
        'schema': _schema,
        'binding': _binding,
        'sdk_version': _packageVersion(),
        'trace_id': traceId,
        'parent_span_id': parentSpanId,
        'references': <String, Object?>{
          for (final name in _referenceNames) name: null
        },
        'automatic': <String, Object?>{},
        'stats': <String, Object?>{},
        'lifecycle': <String, Object?>{},
        'scenario_error': null,
      };

      Object? failure;
      try {
        await _initialize(env);
        await _runScenario(spec, endpoint!, workdir, expectations);
      } catch (error, stack) {
        failure = error;
        final frames = stack.toString().split('\n').take(12).join('\n');
        expectations['scenario_error'] = '$error\n$frames';
        // Stop exporting even when a step failed, so nothing keeps sending
        // to a receiver that is about to stop.
        if (XybridRustLib.instance.initialized) {
          await _outcome(() => XybridObservability.shutdown(
              timeoutMs: spec['timeout_ms']! as int));
        }
      }

      expectationsFile.parent.createSync(recursive: true);
      final temporary = File('${expectationsFile.path}.tmp')
        ..writeAsStringSync(
            '${const JsonEncoder.withIndent('  ').convert(expectations)}\n');
      temporary.renameSync(expectationsFile.path);
      if (failure != null) {
        fail('scenario step failed: ${expectations['scenario_error']}');
      }
    },
    skip: endpoint == null || endpoint.isEmpty
        ? 'set XYBRID_SCENARIO_ENDPOINT (run_scenario.py does) to run the '
            'scenario against the native library'
        : false,
    timeout: const Timeout(Duration(minutes: 5)),
  );
}
