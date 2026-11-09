# Observability runtime scenario

One model-free scenario that every native binding runs identically through its
public API, against a local OTLP/HTTP receiver. The receiver records what the
binding exported; a checker compares it with what the binding says it did.
Passing it on a binding's real runtime (device, simulator, emulator, Editor)
is that binding's row of the parent-in / reference-out / manual operations /
content controls / lifecycle matrix in
[`docs/sdk/observability-coverage.md`](../../../docs/sdk/observability-coverage.md).
Semantics come from
[`crates/xybrid-observability/CONTRACT.md`](../../../crates/xybrid-observability/CONTRACT.md).

| File | Role |
|---|---|
| `capture_server.py` | Stdlib-only OTLP/HTTP receiver: `POST /v1/traces`, `/v1/logs`, `/v1/metrics`, answers `200 {}`, appends one JSON line per request, binds `0.0.0.0`, writes a ready file, exits 0 on SIGTERM. |
| `scenario.json` | Every constant below, machine-readable. The checker reads it. |
| `check_capture.py` | Asserts the capture against `expectations.json`; `PASS`/`FAIL` per check, exit 1 on any failure. `--ids BINDING` prints T and P; `--constants BINDING` prints every value as JSON. |
| `run_scenario.py` | Driver: starts the receiver, runs a lane's command, stops the receiver, runs the checker, writes `report.json`. |
| `fixtures/python/` | A recorded passing run (capture plus expectations), used by the unit tests. |
| [`bindings/python/tests/observability_scenario.py`](../../../bindings/python/tests/observability_scenario.py) | The reference implementation, run end to end by `bindings/python/tests/test_observability_scenario.py`, once as is and once with `--without-omit-rule` as the control for `content.omit`. |

Python 3.9 or newer, standard library only. Unit tests:
`tools/scripts/tests/test_observability_capture.py`.

## Running a lane

```bash
python3 tools/scripts/observability_capture/run_scenario.py \
  --binding python --out-dir out/python -- \
  python3 bindings/python/tests/observability_scenario.py
```

The driver binds a free port unless `--port` names one, and gives the command
`XYBRID_SCENARIO_BINDING`, `XYBRID_SCENARIO_ENDPOINT`,
`XYBRID_SCENARIO_EXPECTATIONS`, `XYBRID_SCENARIO_WORKDIR` and
`XYBRID_SCENARIO_PORT`, and removes every `OTEL_*` variable: the scenario
relies on the default content capture, which
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` would change. A device lane must
not set them either. A command that cannot forward environment variables
(an instrumented Android test, `xcodebuild test`, `flutter drive`) uses the
placeholders `{endpoint}`, `{port}`, `{expectations}`, `{workdir}`, `{binding}`
and `{out_dir}` in its arguments instead.

- Android emulator: `--endpoint-host 10.0.2.2`. The app needs cleartext HTTP
  to that host (a debug network security config) and the `INTERNET`
  permission.
- iOS simulator: `127.0.0.1` reaches the host. A physical device needs
  `--endpoint-host <host LAN address>` and an ATS exception for plain HTTP.
- A scenario that writes `expectations.json` on a device copies it back inside
  its command (`adb pull`, `xcrun simctl get_app_container`) to the path given
  by `--expectations`.

Without the driver:

```bash
python3 capture_server.py --port 4318 --out capture.jsonl --ready-file ready.json &
# ... run the scenario against http://<host>:4318 ...
kill -TERM %1 && wait
python3 check_capture.py --capture capture.jsonl --expectations expectations.json
```

## Identifiers

Each binding uses its own fixed trace, recognisable in any backend, and no
lane needs a hash function to derive it. Run one receiver per lane all the
same: the checker treats every recorded request as that lane's (resource
identity, counts).

- **T** (trace id) = hex of the UTF-8 bytes of `"xy:" + binding`, right-padded
  with `0` to 32 hex digits.
- **P** (the caller's span, the remote parent) = hex of the UTF-8 bytes of the
  binding name, right-padded with `0` and cut to 16 hex digits.
- The parent passed in is the W3C `traceparent` `00-T-P-01`.

| Binding | T | P |
|---|---|---|
| `python` | `78793a707974686f6e00000000000000` | `707974686f6e0000` |
| `swift` | `78793a73776966740000000000000000` | `7377696674000000` |
| `kotlin` | `78793a6b6f746c696e00000000000000` | `6b6f746c696e0000` |
| `unity` | `78793a756e6974790000000000000000` | `756e697479000000` |
| `flutter` | `78793a666c7574746572000000000000` | `666c757474657200` |
| `react-native` | `78793a72656163742d6e617469766500` | `72656163742d6e61` |
| `rust` | `78793a72757374000000000000000000` | `7275737400000000` |

The binding name is the one the SDK registers (`xybrid.sdk.binding` on the
exported resource, see `docs/telemetry/registry.md`). A lane whose host
legitimately reports another value, for example Kotlin on a desktop JVM without
`Xybrid.init(context)`, sets `"sdk_binding"` in `expectations.json` to it.

## Constants

Per binding (`{binding}` replaced by its name):

| Value | Template |
|---|---|
| service name, version | `xybrid-scenario-{binding}`, `1.0.0` |
| device id | `scenario-device-{binding}` |
| conversation id | `scenario-conversation-{binding}` |

Shared:

| Name | Value |
|---|---|
| secret (must never be exported) | `xybrid-scenario-secret-4c1d` |
| literal redaction replacement | `[scenario-redacted]` |
| omitted attribute | `gen_ai.system_instructions` |
| omitted content marker (must never be exported) | `xybrid-scenario-omitted-9b2e` |
| workflow name | `scenario-workflow` |
| workflow started after shutdown | `scenario-after-shutdown` |
| tool name, call id, type, description | `scenario_lookup`, `call-scenario-1`, `function`, none |
| inference request | operation `chat`, provider `scenario-provider`, model `scenario-model`, location local, runtime `scenario-runtime`, `stream` false, every other field none |
| failed inference error type | `scenario_error` |
| failed inference message | `upstream rejected xybrid-scenario-secret-4c1d` |
| model directory name, model id | `scenario-weightless`, `scenario-weightless` |
| run input text | `run input xybrid-scenario-secret-4c1d` |
| flush and shutdown timeout | `5000` ms |

JSON payloads, passed as these exact strings:

```text
tool arguments_json:
{"query":"weather","token":"xybrid-scenario-secret-4c1d"}
tool result_json:
{"answer":"sunny","echo":"xybrid-scenario-secret-4c1d"}
inference system instructions:
[{"type":"text","content":"xybrid-scenario-omitted-9b2e"}]
inference input messages:
[{"role":"user","parts":[{"type":"text","content":"my token is xybrid-scenario-secret-4c1d"}]}]
inference output messages:
[{"role":"assistant","parts":[{"type":"text","content":"I will not repeat xybrid-scenario-secret-4c1d"}],"finish_reason":"stop"}]
released inference input messages:
[{"role":"user","parts":[{"type":"text","content":"abandoned request"}]}]
<workdir>/scenario-weightless/model_metadata.json:
{"model_id":"scenario-weightless","version":"1.0","execution_template":{"type":"Onnx","model_file":"model.onnx"},"files":[]}
```

## Steps

Run the steps in order, in one process, with nothing else configuring
observability. Record each reference as `{"trace_id", "span_id"}` (lower-case
hex) or `null`, and each stats snapshot as
`{"enabled", "traces", "logs", "metrics"}` with the five counters
`accepted`, `exported`, `dropped`, `failed`, `pending` per signal.

**(a) Configure.** An OTLP config for the receiver endpoint with the JSON
protocol, service name and version, the device id, the literal redaction
(secret to replacement) and the omit rule for `gen_ai.system_instructions`;
then start it. Every binding offers OTLP/JSON, so the receiver never needs to
decode protobuf (it records a protobuf body and the checker fails it).

**(b) Parent-in.** Start workflow `scenario-workflow` with options carrying
`traceparent = 00-T-P-01` (no conversation id). Record its reference as
`workflow`. Read its context; children below get options built from that
context's `traceparent`/`tracestate`.

**(c) Manual children**, each with the workflow context as parent:

1. Tool `scenario_lookup` with the arguments JSON; record the result JSON;
   complete; reference `tool`.
2. Inference with the request above; record system instructions, input
   messages and output messages; complete; reference `inference`. The secret
   must come out as the replacement; `gen_ai.system_instructions` must be
   dropped entirely.
3. Inference that fails with error type `scenario_error` and the failure
   message; reference `inference_failed`. The span's status message must be
   the redacted message.
4. Inference that records the released input messages, then is released
   without any terminal call: read stats (`before_release`), release the last
   owner, read stats again (`after_release`); reference
   `inference_released`. It must export as `cancelled` with status ERROR, and
   `traces.accepted` must grow by exactly one across the release (the release
   ended it synchronously). The release must not flush; the checker notes how
   many spans were still pending.

**(d) Automatic path.** Create `<workdir>/scenario-weightless/` holding only
`model_metadata.json`. Load it from that directory with options
`traceparent`/`tracestate` = the workflow context and conversation id
`scenario-conversation-{binding}`. The load succeeds lazily: record
`load_observation_ref` as `load` and `automatic.load = {"outcome": "ok"}`.
Then run the run input text through the model with run options carrying the
same parent and conversation id. It fails (there is no `model.onnx`): record
`automatic.run = {"outcome": "error", "error": "<class>: <message>"}`;
`references.run` stays `null` because a failed run returns no result. Release
the model, then complete and release the workflow.

The scenario therefore shows reference-out from the manual handles and the
model load, not from a run result (`result.observation`,
`InferenceResult.observation`): that needs a model that runs, so weights. A
binding that does return a run reference records it as `references.run`, and
the checker requires it to name the run span.

If a platform rejects the directory at load time instead, record
`automatic.load = {"outcome": "error", ...}`,
`automatic.run = {"outcome": "skipped"}` and `references.load = null`; the
checker then requires one failed `load_model scenario-weightless` span under
the workflow carrying the conversation id.

**(e) Lifecycle.** Flush; record `lifecycle.flush` as `"ok"` or
`"error: ..."`; read stats (`after_flush`). Shut down twice and record both
results as `lifecycle.shutdown` (`["ok", "ok"]` expected); read stats
(`after_shutdown`, expected disabled and all zero). Then start workflow
`scenario-after-shutdown` with no options, read its reference and context
(both expected `null`), complete and release it, and flush again. Record
`lifecycle.after_shutdown = {"workflow_ref", "workflow_context", "flush"}`.
Nothing it does may reach the receiver.

**(f) Write `expectations.json`** to the path the driver gave, even when a
step threw: then set `scenario_error` to the error text and exit non-zero.

```json
{
  "schema": "xybrid-observability-scenario/1",
  "binding": "python",
  "sdk_version": "0.17.0",
  "trace_id": "78793a707974686f6e00000000000000",
  "parent_span_id": "707974686f6e0000",
  "references": {
    "workflow": {"trace_id": "...", "span_id": "..."},
    "tool": {}, "inference": {}, "inference_failed": {}, "inference_released": {},
    "load": {}, "run": null
  },
  "automatic": {
    "load": {"outcome": "ok", "error": null},
    "run": {"outcome": "error", "error": "InferenceError: ..."}
  },
  "stats": {"before_release": {}, "after_release": {}, "after_flush": {}, "after_shutdown": {}},
  "lifecycle": {
    "flush": "ok",
    "shutdown": ["ok", "ok"],
    "after_shutdown": {"workflow_ref": null, "workflow_context": null, "flush": "ok"}
  },
  "scenario_error": null
}
```

`fixtures/python/expectations.json` is a complete real example; the optional
`sdk_binding` is described under Identifiers.

## What the checker asserts

| Check | Passes when |
|---|---|
| `capture.decoded` | Something arrived, every request went to an OTLP signal path and was JSON. |
| `expectations.valid` | Schema, T and P match the binding, required sections present, no `scenario_error`. |
| `references.exported` | Every non-null reference is an exported span with the same trace and span id; manual references are in T. |
| `workflow.parent_in` | `invoke_workflow scenario-workflow` is in T, its parent is P, no error. |
| `manual.children` | Tool and the three inferences are children of the workflow, named `execute_tool scenario_lookup` / `chat scenario-model`. |
| `tool.content` | Arguments and result equal the payloads with the secret replaced; call id kept; no error. |
| `inference.content` | Input and output messages equal the payloads with the secret replaced; the span carries no `gen_ai.system_instructions`; no error. |
| `content.redaction` | The secret appears in no request of any signal; the replacement appears. |
| `content.omit` | No span or log carries `gen_ai.system_instructions`; the omitted marker appears nowhere. Absence only: see below. |
| `resource.identity` | Every resource of every signal has the device id, service name and version, and `xybrid.sdk.binding`. |
| `inference.failed` | `error.type = scenario_error`, status ERROR, status message = the redacted message, and a `gen_ai.client.operation.exception` log for that span. |
| `inference.released` | `error.type = cancelled`, status ERROR, no exception log, `traces.accepted` grew by one across the release. |
| `automatic.load` | The load span is a child of the workflow with the conversation id; its error matches the recorded outcome. |
| `automatic.run` | Exactly one workflow child besides the manual operations and the load, the run span: conversation id, model id, `error.type`, status ERROR; a non-null `references.run` names it. |
| `trace.tree` | Every span in T has an exported parent; only the workflow hangs under P. Spans outside T are noted. |
| `stats.consistent` | `accepted = exported + dropped + failed + pending` in every snapshot; after the flush nothing pending, dropped or failed; `traces.exported` at least the spans asserted; the receiver holds at least as many spans, log records and metric points as were reported exported; metrics were exported. |
| `lifecycle.shutdown` | Flush ok, both shutdowns ok, stats afterwards disabled with all five counters of every signal present and zero, the post-shutdown workflow has no reference or context, the post-shutdown flush is ok. |
| `lifecycle.nothing_after_shutdown` | The post-shutdown workflow was never exported. |

`content.omit` and the omit clause of `inference.content` can only show that
the omitted content reached the receiver nowhere. A rule-omitted value is not
listed anywhere (CONTRACT.md, "Content pipeline"), so the capture looks the
same when nobody recorded the system instructions. Two other checks show that
the rule removed them:

- the Python control run: `observability_scenario.py --without-omit-rule`
  configures no omit rule, and the checker must then fail exactly
  `content.omit` and `inference.content` because the marker reached the
  receiver (`bindings/python/tests/test_observability_scenario.py`);
- `tools/scripts/tests/test_observability_capture.py` requires every lane's
  source to configure the omit rule and record the system instructions.

## API per binding

Names below are the public wrappers each lane calls (generated names where the
wrapper adds none). Release means dropping the last owner without a terminal
call.

| Step | Python | Swift | Kotlin | Unity C# | Flutter | React Native |
|---|---|---|---|---|---|---|
| Config | `XybridObservabilityConfig.otlp(e)`, `set_json_protocol(True)`, `set_service`, `set_device_id`, `add_literal_redaction`, `omit_content_attribute`, `init()` | `Xybrid.Observability.otlp(endpoint:)` `.jsonProtocol()` `.service(_:version:)` `.deviceId(_:)` `.redactLiteral(_:replacement:)` `.omitContent(_:)` `.start()` | `Xybrid.Observability.otlp(e)` `.jsonProtocol()` `.service(n, v)` `.deviceId(id)` `.redactLiteral(f, r)` `.omitContent(k)` `.start()` | `ObservabilityConfig.Otlp(e)` `.WithJsonProtocol()` `.WithService(n, v)` `.WithDeviceId(id)` `.WithLiteralRedaction(f, r)` `.WithoutContentAttribute(k)`, `XybridClient.InitializeObservability(config)` | `XybridObservability.init(ObservabilityConfig.otlp(endpoint:, jsonProtocol: true, serviceName:, serviceVersion:, deviceId:, redactionRules: [FfiRedactionRule.replaceLiteral(...), FfiRedactionRule.omitAttribute(key:)]))` | `ObservabilityConfig.otlp(e)` `.setJsonProtocol(true)` `.setService(n, v)` `.setDeviceId(id)` `.addLiteralRedaction(f, r)` `.omitContentAttribute(k)` `.init()` |
| Workflow with parent | `XybridWorkflowObservation(name, XybridObservationOptions(traceparent=...))` | `Xybrid.Observability.startWorkflow(_:options: .make(traceparent:))` | `Xybrid.Observability.startWorkflow(name, observationOptions(traceparent = ...))` | `WorkflowObservation.Start(name, new ObservationOptions { Traceparent = ... })` | `XybridObservability.startWorkflow(name, options: ObservationOptions(traceparent: ...))` | `WorkflowObservation.start(name, { traceparent })` |
| Reference, child options | `observation_ref()`, `context()` | `observationRef()`, `context()?.childOptions` | `observationRef()`, `context()?.childOptions` | `Reference`, `Context` | `observationRef()`, `context()` | `observationRef()`, `context()` |
| Tool | `XybridToolObservation(XybridToolCallInfo(...), child)`, `record_result`, `complete` | `startTool(XybridToolCallInfo(name:callId:toolType:description:argumentsJson:), options:)`, `recordResult(resultJson:)`, `complete()` | `startTool(XybridToolCallInfo(...), child)`, `recordResult`, `complete` | `ToolObservation.Start(new XybridBolt.XybridToolCallInfo(...), o)`, `RecordResult`, `Complete` | `startTool(ToolCallInfo(...), options:)`, `recordResult(resultJson:)`, `complete()` | `ToolObservation.start({ name, callId, toolType, argumentsJson }, child)`, `recordResult`, `complete` |
| Inference content | `record_system_instructions`, `record_input_messages`, `record_output_messages` | `recordSystemInstructions(partsJson:)`, `recordInputMessages(messagesJson:)`, `recordOutputMessages(messagesJson:)` | `recordSystemInstructions`, `recordInputMessages`, `recordOutputMessages` | `RecordSystemInstructions`, `RecordInputMessages`, `RecordOutputMessages` | `recordSystemInstructions(partsJson:)`, `recordInputMessages(messagesJson:)`, `recordOutputMessages(messagesJson:)` | `recordSystemInstructions`, `recordInputMessages`, `recordOutputMessages` |
| Fail | `fail(error_type=, message=)` | `fail(errorType:message:)` | `fail(errorType, message)` | `Fail(errorType, message)` | `fail(errorType:, message:)` | `fail(errorType, message)` |
| Release | `close()` | drop the last strong reference (scope end or `= nil`) | `close()` | `Dispose()` | `dispose()`; the model: `model.inner.dispose()` (`XybridModel` has no `dispose`) | `dispose()` |
| Load with parent | `XybridModel.load(XybridModelSourceDirectory(p), options)`, `load_observation_ref()` | `XybridModel(load: .directory(path:), observation:)`, `loadObservationRef()` | `XybridModel.load(XybridModelSource.Directory(p), options)`, `loadObservationRef()` | `ModelLoader.FromDirectory(p).Load(observation)`, `LoadObservation` | `XybridModelLoader.fromDirectory(p).withObservation(o).load()`, `loadObservationRef` | `ModelLoader.fromDirectory(p).withObservation(o).load()`, `(await model.info()).loadObservationRef` |
| Run with parent | `model.run(XybridEnvelope.text(t), XybridRunOptions(traceparent=, tracestate=, conversation_id=, ...))` | `model.run(envelope: .text(t), options: .make(traceparent:tracestate:conversationId:))` | `model.run(Envelope.text(t), runOptions(traceparent =, tracestate =, conversationId =))` | `model.Run(Envelope.Text(t), observation: new ObservationOptions { ... })` | `model.run(XybridEnvelope.text(t), options: RunOptions(observation: ObservationOptions(...)))` | `model.run(Envelope.text(t), { traceparent, tracestate, conversationId })` |
| Flush, stats, shutdown | `observability_flush(ms)`, `observability_stats()`, `observability_shutdown(ms)` | `Xybrid.Observability.flush(timeoutMs:)`, `.stats`, `.shutdown(timeoutMs:)` | `Xybrid.Observability.flush(ms)`, `.stats`, `.shutdown(ms)` | `XybridClient.FlushObservability(t)`, `XybridClient.ObservabilityStats()`, `XybridClient.ShutdownObservability(t)` | `XybridObservability.flush(timeoutMs:)`, `.stats`, `.shutdown(timeoutMs:)` | `Observability.flush(ms)`, `Observability.stats()`, `Observability.shutdown(ms)` |

Lane notes:

- React Native has no file API of its own: the lane writes the model directory
  from its native test host or a test-only dependency, under the app's cache
  directory.
- Flutter stats counters are `BigInt`; write them as JSON integers.
- Unity Editor tests share the Editor process with everything else that
  touches the native library: shut observability down in teardown even when
  the scenario fails, so later tests do not keep exporting to a stopped
  receiver.
- Swift and Kotlin `withWorkflow` helpers complete on return; the scenario
  uses explicit `startWorkflow` so the release step stays explicit.

## What only a device or host lane proves

The Python reference runs the compiled bridge on Linux, which proves the
harness and the shared native implementation. Each other binding's row is
proven only when its own lane runs this scenario on that runtime: the Swift
package on an Apple host or simulator, the Kotlin AAR on an Android emulator,
Unity in the Editor or a player, Flutter on a device or desktop embedder, and
React Native through its TurboModule on iOS and Android.
