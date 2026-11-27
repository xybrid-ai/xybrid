# Decision requests in native SDKs

The `lfm3-d1/text/v1` implementation uses the existing `run` methods. Load a local directory containing a qualified d1-2B GGUF and decision execution metadata. Registry publication and packaged-device qualification are separate release gates; these examples do not claim that every native package or accelerator has passed them.

Save this request as `request.json`. The request is a JSON string at each foreign boundary, so object order, `1` versus `1.0`, signed zero, embedded NUL and integers above JavaScript's exact-number range reach the shared decoder unchanged. Do not parse and re-serialize it through a dictionary first. The neutral carrier preserves unsigned integers; this pinned Liquid/Jinja profile rejects values above `i64::MAX` (`9223372036854775807`) before rendering because its template integers are signed.

```json
{"state":{"ticket":"The order arrived damaged","order_id":9007199254740993},"questions":{"action":{"type":"choice","instructions":"Choose the next support action.","criteria":{"refund":"Refund the order","replace":"Send a replacement"}}}}
```

The loaded model is authoritative. A `model` field in Liquid JSON cannot change it. Requests are local-only and return all answers atomically. Streaming, history, chat tools, generation options, cloud fallback and pipelines are refused. Cancellation uses each SDK's existing cancellation handle or signal.

Swift:

```swift
let input = try Envelope.decisionRequestJson(requestJson)
let result = try model.run(envelope: input)
let answers = result.decisions?.answers
```

Kotlin:

```kotlin
val input = Envelope.decisionRequestJson(requestJson)
val result = model.run(input)
val answers = result.decisions?.answers
```

Python:

```python
request_json = pathlib.Path("request.json").read_text()
input = xybrid.XybridEnvelope.decision_request_json(request_json)
result = model.run(input)
answers = result.decisions.answers
```

Unity/C#:

```csharp
using var input = Xybrid.Envelope.DecisionRequestJson(requestJson);
using var result = model.Run(input);
var answers = result.Decisions?.Answers;
```

Flutter:

```dart
final input = XybridEnvelope.decisionRequestJson(requestJson);
final result = await model.run(input);
final answers = result.decisions?.answers;
```

React Native:

```typescript
const input = Envelope.decisionRequestJson(requestJson);
const result = await model.run(input);
const answers = result.decisions?.answers;
```

Swift, Kotlin, Python and Unity also expose generated `XybridDecisionRequest`, `XybridDecisionQuestion`, `XybridDecisionOption` and content variants for ordered typed construction. `decisionRequestEnvelope` / `decision_request_envelope` validates these records before constructing the input. React Native offers `Envelope.decisionRequest(request)`; structured content uses `{kind: 'json', jsonText: '...'}` rather than a JavaScript object.

Each answer carries its original question ID, typed choice/yes-no/score value, an ordered distribution, and optional confidence. Probabilities remain double precision and are protocol outputs, not a claim of calibration on a quantized device. A yes/no answer reports `pYes` and no confidence; an ordinal score reports a zero-based median. `passes` and `inputTokens` describe total work across questions; generated output tokens are zero. JSON bridges spell a negative-infinite raw score `"-Infinity"`, never `null`.

The limits are 1–16 questions, 1–255 choice options, 2–10 score levels, 8 MiB raw transport, 1 MiB aggregate content and 4,096 input tokens per question. Protocol errors are non-retryable and do not include request content. Unsupported newer foreign variants are refused instead of silently dropped. Existing result record fields and existing enum ordinals remain unchanged; the decision payload is an appended envelope variant.
