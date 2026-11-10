import 'package:flutter_test/flutter_test.dart';
import 'package:xybrid_flutter/src/rust/api/envelope.dart';
import 'package:xybrid_flutter/src/rust/api/observability.dart';
import 'package:xybrid_flutter/src/rust/api/pipeline.dart';
import 'package:xybrid_flutter/src/rust/api/result.dart';
import 'package:xybrid_flutter/src/rust/frb_generated.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

const _workflow = ObservationRef(
    traceId: '4bf92f3577b34da6a3ce929d0e0e4736', spanId: '00f067aa0ba902b7');
const _stage = ObservationRef(
    traceId: '4bf92f3577b34da6a3ce929d0e0e4736', spanId: '11f067aa0ba902b7');

class _Envelope implements FfiEnvelope {
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Pipeline implements FfiPipeline {
  FfiObservationOptions? observation;
  Object? error;

  @override
  Future<FfiRunResult> run(
      {required FfiEnvelope envelope,
      FfiObservationOptions? observation}) async {
    this.observation = observation;
    if (error != null) throw error!;
    return const FfiRunResult.legacy(FfiResult(
      observation: _workflow,
      success: true,
      text: 'answer',
      latencyMs: 8,
      executionTarget: FfiExecutionTarget.cloud,
      metrics: FfiInferenceMetrics(totalMs: 8, stageLatenciesMs: [
        FfiStageLatency(stageId: 'draft', latencyMs: 8, observation: _stage),
      ]),
      toolCalls: [],
    ));
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Api implements XybridRustLibApi {
  final pipeline = _Pipeline();

  @override
  FfiPipeline crateApiPipelineFfiPipelineFromYaml({required String yaml}) =>
      pipeline;

  @override
  FfiEnvelope crateApiEnvelopeFfiEnvelopeText(
          {required String text, String? voiceId, double? speed}) =>
      _Envelope();

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  final api = _Api();
  setUpAll(() => XybridRustLib.initMock(api: api));
  tearDownAll(XybridRustLib.dispose);
  setUp(() => api.pipeline.error = null);

  test(
      'pipeline forwards parent controls and exposes distinct result references',
      () async {
    final pipeline = XybridPipeline.fromYaml('stages: []');
    const observation = ObservationOptions(
        traceparent: '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00',
        tracestate: 'vendor=value',
        conversationId: 'turn-7',
        suppressTrace: true);
    final result = await pipeline.run(XybridEnvelope.text('question'),
        observation: observation);
    expect(api.pipeline.observation!.traceparent, observation.traceparent);
    expect(api.pipeline.observation!.tracestate, observation.tracestate);
    expect(api.pipeline.observation!.conversationId, 'turn-7');
    expect(api.pipeline.observation!.suppressTrace, isTrue);
    expect(result.observation, _workflow);
    expect(result.metrics.stageLatenciesMs.single.stageId, 'draft');
    expect(result.metrics.stageLatenciesMs.single.observation, _stage);
    expect(result.text, 'answer');
    await pipeline.run(XybridEnvelope.text('question'));
    expect(api.pipeline.observation, isNull);
  });

  test('pipeline validation and delivery errors remain visible to the caller',
      () async {
    final pipeline = XybridPipeline.fromYaml('stages: []');
    api.pipeline.error = 'Invalid traceparent';
    await expectLater(
        pipeline.run(XybridEnvelope.text('question'),
            observation: const ObservationOptions(traceparent: 'bad')),
        throwsA(isA<XybridException>().having((e) => e.toString(), 'diagnostic',
            contains('Invalid traceparent'))));
  });

  test('manual request types are available from the public package', () {
    // All names below are exported by xybrid_flutter.dart, with no src/ import.
    const call =
        ToolCallInfo(name: 'lookup', argumentsJson: '{"city":"Paris"}');
    const inference = InferenceRequest(
        operation: 'chat',
        provider: 'app',
        location: ExecutionLocation.local,
        stream: false);
    const load = ModelLoadInfo(model: 'app-model');
    const usage = ObservationUsage();
    const context = ObservationContext(
        traceparent: '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01');
    expect(call.name, 'lookup');
    expect(inference.location, ExecutionLocation.local);
    expect(load.model, 'app-model');
    expect(usage.inputTokens, isNull);
    expect(context.traceparent, startsWith('00-'));
    // Compile-time coverage of all four public handle aliases.
    final handles = <Type>[
      WorkflowObservation,
      ToolObservation,
      InferenceObservation,
      ModelLoadObservation
    ];
    expect(handles, hasLength(4));
  });
}
