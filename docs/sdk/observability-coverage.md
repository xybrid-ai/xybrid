# Observability execution and binding coverage

Observability is optional and additive to offline execution. Configure export
explicitly. References are `{trace_id, span_id}`; a valid unsampled span still
has a reference, while explicit suppression has none. A reference does not
promise delivery. Default governed text capture applies once export is enabled;
select metadata-only mode or native redaction rules deliberately.

## Execution shapes

| Execution | Parent/control input | Reference output | Lifecycle |
|---|---|---|---|
| Model load: registry, path, reload, lazy materialization | Loader `.observation(...)` / existing options | Model `load_observation_ref()`; triggering run owns reload/lazy child | Load success/error; cancellation on abandoned work |
| Model inference, sync/async, token stream | `RunOptions` observation | Inference/terminal result | Logical totals once, attempts retain own usage |
| Pipeline | Pipeline observation options | Workflow on result; inference on stage results | Child stages and attempts share captured context |
| Streamed TTS | Existing run options | SDK summary plus inference reference; foreign terminal result | Completion/error/cancellation ends operation |
| Live ASR | Session observation options | Decoder reference on partial; workflow on terminal transcription | Reset starts a new session; cached partials add no inference; foreign flush ends worker |
| Manual workflow/tool/inference/load | Explicit normalized W3C options | Handle context/reference | First terminal transition wins; last owner cancels; no implicit flush |
| CLI direct-core calls and REPL | Operation-scoped capture adapter | Trace artifact / renderer | Each operation owns its local capture |
| Browser GGUF generation | Browser observation options carried through worker | Generation reference | Worker errors/disposal terminate work |

The ASR and streamed-TTS rows describe supported audio interfaces; they do not
add those interfaces to the browser GGUF preview. Third-party provider-child
spans require instrumentation in that provider. Xybrid hops carry context;
direct third-party origins require explicit allowlisting.

## Binding surfaces and validation

The matrix separates executed local checks from platform-host checks. Generated
output checks establish schema consistency; they do not establish device runtime
behavior. All native surfaces share the facade's parent validation, content
policy and first-terminal-wins implementation.

| Surface | Parent / result references | Manual operations and disposal | Content and lifecycle | Local validation |
|---|---|---|---|---|
| Rust SDK | Loader, model, pipeline, TTS, ASR | Typed handles retain session; drop cancels | Owned/borrowed sessions, explicit device ID, declarative/callback redaction | Session, routing, tool, load, pipeline, TTS and decoder regressions; real-model tests remain fixture-dependent |
| Swift | Existing options and generated result fields | `Xybrid.Observability.startWorkflow/startTool/startInference/startModelLoad`; `withWorkflow` | Config, rules, stats/flush/shutdown | Generation check executed. CI runtime lane: the `Observability runtime scenario (iOS Simulator)` job in [`build-apple.yml`](../../.github/workflows/build-apple.yml) runs the [shared scenario](../../tools/scripts/observability_capture/scenario.md) through the public Swift API on an iOS Simulator, linked against the xcframework CI builds, and checks what reaches a local OTLP receiver: parent-in, reference-out from the manual handles and the model load, manual operations, content redaction, the omitted attribute never reaching the receiver, release, the weightless model load and failed run, flush and shutdown. It has not run in CI yet. The same XCTest passed all 18 checks on Linux against the native library, which proves the test, not the Apple runtime |
| Kotlin | Existing options and generated result fields | `Xybrid.Observability.withWorkflow`; generated handles implement close/use | Same native config and lifecycle | Generation, wrapper compilation, three JUnit tests and RN codec checks. CI runtime lane: the `Observability scenario (x86_64 emulator)` job in [`build-android.yml`](../../.github/workflows/build-android.yml) runs the [shared scenario](../../tools/scripts/observability_capture/scenario.md) through the public Kotlin wrapper on an Android 15 x86_64 emulator, in the consumer APK built against the AAR CI builds, and checks what reaches a local OTLP receiver on the runner: parent-in, reference-out from the manual handles and the model load, manual operations, content redaction, the omitted attribute never reaching the receiver, release, the weightless model load and failed run, flush and shutdown. It has not run in CI yet. The same scenario code passed all 18 checks on a desktop JVM against the native library, which proves the test, not the Android runtime |
| Python | Generated options and result fields, Python properties | Four `Xybrid*Observation` classes; context managers and explicit `close()` | Generated config, native stats/flush/shutdown | Compiled-bridge tests cover load/run/cloud pipeline, parentage, content and handle lifecycle; real ASR/TTS runs require model fixtures. CI runtime lane: the `Python SDK (ubuntu-latest)` job in [`ci.yml`](../../.github/workflows/ci.yml) runs the [shared scenario](../../tools/scripts/observability_capture/scenario.md) through the reference implementation on the compiled bridge and checks what reaches a local OTLP receiver, with the same checks as the Swift, Kotlin, Unity and Flutter lanes. A second run without the omit rule must export the omitted content, which shows that the scenario records it. It has not run in CI with the control yet. Locally both runs pass on Linux against a debug build of the bridge |
| Unity C# | Wrapper options and result references | Four wrapper handles implement `IDisposable` | Same native config; `using` releases final owner | Generation and .NET compilation checks. CI runtime lane: the `Run the observability runtime scenario (EditMode)` step of the `Unity Editor (Linux, Mono + IL2CPP)` job in [`unity-editor.yml`](../../.github/workflows/unity-editor.yml) runs the [shared scenario](../../tools/scripts/observability_capture/scenario.md) through the public C# API in the Unity 6000.3 Editor on Linux (Mono), against the Bazel-built Linux native the job stages, and checks what reaches a local OTLP receiver on the runner: parent-in, reference-out from the manual handles and the model load, manual operations, content redaction, the omitted attribute never reaching the receiver, release, the weightless model load and failed run, flush and shutdown. It has not run in CI yet. Locally the same test passed all 18 checks under .NET 8 against a debug build of the Linux native, which proves the test and the C# wrapper on .NET, not the Unity Editor runtime. IL2CPP player, Android, iOS, macOS and Windows runtimes pending |
| Flutter | Existing load/run/pipeline/ASR options; stage references in stage latencies | `XybridObservability.start*`; generated opaque handles expose disposal | Native config/rules and asynchronous flush/shutdown | Reproducible FRB generation, Dart wrapper and Rust API tests. CI runtime lane: the `Observability runtime scenario (flutter test on Linux)` job in [`build-flutter.yml`](../../.github/workflows/build-flutter.yml) runs the [shared scenario](../../tools/scripts/observability_capture/scenario.md) through the public Dart API under `flutter test` on Linux, loading the release `libxybrid_flutter_ffi.so` the Linux build job produces, and checks what reaches a local OTLP receiver: parent-in, reference-out from the manual handles and the model load, manual operations, content redaction, the omitted attribute never reaching the receiver, release, the weightless model load and failed run, flush and shutdown. It has not run in CI yet. Locally the same test passed all 18 checks under `flutter test` on Linux against a debug build of the library. Android, iOS, macOS and Windows runtimes pending |
| React Native | TurboModule codec and both native shims | `WorkflowObservation`, `ToolObservation`, `InferenceObservation`, `ModelLoadObservation`; explicit `dispose()` and module invalidation | Native config/rules and lifecycle | Type, facade, wire, parity and Kotlin codec tests. No runtime lane: [`scenario.md`](../../tools/scripts/observability_capture/scenario.md) maps its API, but nothing runs the scenario through the TurboModule on iOS or Android yet, so this row is the one native surface the matrix does not demonstrate at runtime |
| Browser | Worker propagation and generation result reference | Portable JS observer handles; browser model dispose | Application-owned provider lifecycle | Worker/control tests and typecheck; browser/model acceptance is a separate lane |
| Portable JS | Explicit W3C controls / references | Workflow/tool/inference/load operations | Callback plus declarative policy, cancellable exporters | Shared fixtures, decoded wire and transport/lifecycle tests; real packed OTel 3.0/0.300 and 3.1/0.301 installations |

What the scenario lanes leave out:

- **Run-result references.** The scenario's model has no weights, so its run
  fails and returns no result; reference-out is shown from the manual handles
  and the model load (`load_observation_ref`). A run result's reference needs a
  model that runs.
- **Omission from the capture alone.** A value the omit rule drops is listed
  nowhere on the wire, so the receiver looks the same when the content was
  never recorded. The Python lane's control run without the rule, and a test
  that requires every lane's source to configure the rule and record the
  content, show that the rule removed it
  ([details](../../tools/scripts/observability_capture/scenario.md#what-the-checker-asserts)).

### Behavioral evidence

The following suites exercise the matrix. A generated-field comparison alone
does not establish a parent/reference round trip or a runtime lifecycle result.

| Surface | Regression source | Execution boundary |
|---|---|---|
| Rust | [SDK observation tests](../../crates/xybrid-sdk/src/observability.rs), [load attribute tests](../../crates/xybrid-sdk/tests/load_observability.rs), [core decoder tests](../../crates/xybrid-core/src/streaming/session.rs), [facade tests](../../crates/xybrid-ffi-facade/tests/observability.rs) | Synthetic runtime/HTTP fixtures and real host resource measurements exercise orchestration; downloaded-model tests skip when their assets are absent |
| Python | [Native observation tests](../../bindings/python/tests/test_observability.py), [runtime scenario](../../bindings/python/tests/test_observability_scenario.py) | Compiled native bridge and local OTLP/HTTP receiver; the runtime scenario runs the [reference implementation](../../bindings/python/tests/observability_scenario.py) through `run_scenario.py`, once as is and once without the omit rule |
| Flutter | [Observation wrapper tests](../../bindings/flutter/test/observability_test.dart), [pipeline tests](../../bindings/flutter/test/pipeline_observability_test.dart), [runtime scenario](../../bindings/flutter/test/observability_scenario_test.dart) | Dart wrapper and mocked FRB boundary; the runtime scenario loads the native Linux library under `flutter test` against a local OTLP receiver when `run_scenario.py` gives it an endpoint (its CI job does), and skips otherwise; mobile device execution remains separate |
| React Native | [Facade tests](../../bindings/react-native/tests/facade.test.mjs), [wire tests](../../bindings/react-native/tests/wire.test.mjs) | Mocked TurboModule; native codec tests and device execution add host coverage |
| Swift | [Observation tests](../../bindings/apple/Tests/XybridTests/ObservationTests.swift), [runtime scenario](../../bindings/apple/Tests/XybridTests/ObservabilityRuntimeTests.swift) | XCTest on an iOS Simulator in CI; the runtime scenario runs only in its CI job, against a local OTLP receiver, and skips elsewhere |
| Kotlin | [Observation tests](../../bindings/kotlin/src/test/kotlin/ai/xybrid/ObservationTest.kt), [runtime scenario](../../bindings/kotlin/consumer-smoke/app/src/androidTest/kotlin/ai/xybrid/smoke/ObservabilityScenarioTest.kt) | Executed wrapper/JVM tests; the runtime scenario is an instrumented test on an Android emulator that runs only in its CI job, against a local OTLP receiver, and skips elsewhere |
| Unity | [Observation tests](../../bindings/unity/Tests/Editor/ObservationTests.cs), [runtime scenario](../../bindings/unity/Tests/Editor/ObservabilityRuntimeScenarioTests.cs) | Unity Editor native test lane; .NET compile checks do not run these tests. The runtime scenario runs only in its CI step, in the Editor against a local OTLP receiver, and is ignored elsewhere |
| Portable JS / browser | [Portable tests](../../bindings/observability-js/test), [browser observation tests](../../bindings/web/test/observability.test.ts) | Local exporters, worker mocks and packed-package consumers; actual browser model execution is separate |

Native application examples below assume the usual SDK initialization has
completed. They describe the public wrapper shapes without adding model or
runtime dependencies to the portable package.

## Binding examples

Python closes the native owner deterministically. An exception ends the
operation with a classification; arbitrary exception text is not exported.

```python
import xybrid

with xybrid.XybridWorkflowObservation("checkout", None) as workflow:
    parent = workflow.context()
    options = xybrid.XybridObservationOptions(
        traceparent=parent.traceparent if parent else None,
        tracestate=parent.tracestate if parent else None,
        conversation_id=None,
        suppress_trace=False,
    )
    # Pass options through the existing loader/run/session APIs.
    reference = workflow.observation_ref()
# close() and later garbage collection release at most one owner.
```

React Native requires explicit release in `finally`; module invalidation also
releases handles still registered by the module.

```ts
import { WorkflowObservation } from '@xybrid/react-native';

const workflow = await WorkflowObservation.start('checkout');
try {
  const parent = await workflow.context();
  // Pass parent?.traceparent/tracestate as existing observation options.
  const reference = await workflow.observationRef();
  await workflow.complete();
} catch (error) {
  await workflow.fail('application_error');
  throw error;
} finally {
  await workflow.dispose();
}
```

Kotlin `use` and Unity `using` release their owner. Complete explicitly on
success; a release before completion cancels unfinished work.

```kotlin
Xybrid.Observability.withWorkflow("checkout") { operation ->
    val parent = operation.context()
    val reference = operation.observationRef()
    // Carry the parent into child options rather than relying on thread locals.
}
```

```csharp
using (var workflow = Xybrid.WorkflowObservation.Start("checkout")) {
    var parent = workflow.Context;
    var reference = workflow.Reference;
    workflow.Complete();
}
```

Swift offers `try await Xybrid.Observability.withWorkflow("checkout") { operation in ... }`.
Flutter offers `await XybridObservability.startWorkflow('checkout')`; read
`await operation.context()` / `observationRef()`, complete or fail explicitly,
and call `dispose()` in `finally` to release the FRB opaque owner. None of these
handle-disposal paths flushes a provider.

## Future runtime readiness

Transformers.js, LiteRT and Apple Foundation Models are adapter opportunities,
not shipped automatic instrumentation. Use the portable observer or SDK manual
handles around the runtime's existing load and generation calls. Record
`generate_content`, the actual runtime/model/location, governed messages and
only measured usage. Record a chunk when bytes/text arrive; record first token
only when the runtime supplies that measurement. Keep media as descriptors or
references. Do not add capability-specific `run*` methods or runtime dependencies
to `xybrid-observability` / `@xybrid/observability`.

The [standalone Rust adapter](../../crates/xybrid-observability/examples/standalone.rs),
[SDK session example](../../crates/xybrid-sdk/examples/observability_session.rs) and
[JS adapter example](../../bindings/observability-js/examples/node-basic.ts)
exercise those primitives without requiring a future runtime. Native adapters
can call the same typed inference/load handles through Swift/Kotlin/Flutter/
Python/C#; pass explicit context across their callbacks and async boundaries.
The [future runtime adapters](../../bindings/observability-js/examples/runtime-readiness.ts)
show callback and stream wrappers for Transformers.js, LiteRT and Foundation
Models without installing those runtimes or inferring missing measurements.

Hosted Collector, performance, staging and production acceptance are later
milestones. Local source and generation checks do not establish those outcomes.
