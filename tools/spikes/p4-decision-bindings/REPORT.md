# P4 — Binding feasibility spike: decision types across xybrid's FFI layers

Branch `conductor/decisions-p4-binding-feasibility-spike` (base `master` @ `899352e`).
Contract: framework v2 + v2.1 corrections (C3, C4, C7). Toolchains: boltffi CLI
0.30.1 (the repo pin), flutter_rust_bridge_codegen 2.11.1 with Flutter 3.35.7 /
Dart 3.9.2 (the repo pins), Rust 1.99.1, Kotlin 1.9.22 + coroutines 1.7.3 (the
repo pins), JDK 17, .NET SDK 8.0.131, Swift 6.2 (swift.org UBI 9 build on Amazon
Linux 2023), CPython 3.11, Node 24.

## 0. Verdict

**Go for contract 1**, with the changes in §9. None of them blocks P1's Rust API.
Most are rules for P9's FFI naming plus three IR clarifications: `DecisionContent`
must be opaque; the number parser must not follow serde_json; conversions to FFI
must be fallible.

- Every contract-1 shape crosses every generator **without flattening**: records,
  data enums (including nested, optional and recursive ones), `Vec<Record>`,
  `Vec<Option<enum>>`, nested lists, `Option<record>`, `Option<f64>`, `f32`, `f64`,
  `u32`, `u64`. That covers Bolt → Swift, Kotlin, Java, C# (Unity), Python and
  TypeScript, and FRB → Dart. Two naming rules apply (§4.1, §4.2).
- The round trip is **proven at runtime in seven languages**: Rust (facade + Bolt
  records), Python, C#, Kotlin/JNI, Java/JNI, Swift (on Linux) and Dart/FRB. Each
  run goes through the real native library with the generated codecs. A request
  with ordered JSON state, three question kinds and 30 options keeps its order. A
  result's 80 `f32`/`f64` values come back bit for bit, `-inf` included, and a
  one-ULP change is detected.
- **JSON as text** is the right FFI form for `OrderedJson`. Every host JSON library
  tested loses order, integer/float identity, `-0`, big integers or `-inf` in at
  least one way (§5). A recursive JSON tree also crosses fine (probe), but it pushes
  number classification onto every host, costs code in every language, and hits a
  Kotlin naming trap. Not recommended.

## 1. What the spike built (all committed on this branch)

| Piece | Where | Notes |
|---|---|---|
| Contract-1 stand-in | `crates/xybrid-ffi-facade/src/decision/stand_in.rs` | Sealed types and an opaque `OrderedJson`, built only by `parse` (I64-first per C3; refuses duplicate keys, non-finite numbers and depth > 128) and read through `JsonView`. Null → `None` at optional fields; redacted `Debug`; invariant-checked `DecisionAnswer::new`. A stub engine applies the d1 rule (`exp((s-max)/T)`, f64) to synthetic scores, giving a `-inf` raw score at one level. |
| Facade translation | `crates/xybrid-ffi-facade/src/decision/mod.rs` | FFI records plus the one translation: `DecisionRequest::into_ir` (parse JSON text, normalize, structural checks; content-free `InferenceError`), `from_ir`, `Decisions::from_ir` / `into_ir`, and `first_bit_difference`. |
| Spike hooks | `crates/xybrid-ffi-facade/src/decision/spike.rs` | Outline of what the IR saw, the expected float bit patterns, bit-exact verify. Shared by Bolt and Flutter. |
| Bolt mirrors | `crates/xybrid-bolt/src/lib.rs` | 14 `#[data]` records/enums; `decision_spike_*` exports (echo, outline, decide, bits, verify); 5 `XybridProbe*` records isolating one shape each. |
| Generated bindings | `bindings/{apple,kotlin,python,unity}` | Written by the repo's own scripts; all four `--check` modes pass. The Unity record-struct tripwire moves 21 → 32. |
| Flutter FRB | `bindings/flutter/rust/src/api/decision.rs`, `lib/src/rust/**`, `test/spike/` | FRB 2.11.1 codegen (a run on the unmodified tree reproduced the committed output exactly), plus a Dart round-trip test that skips unless `XYBRID_P4_NATIVE_LIB` is set. |
| Harnesses | `tools/spikes/p4-decision-bindings/` | Python, C#, Kotlin, Java and Swift harnesses, the host-JSON probes behind §5 (`host-json/`), a README with the exact commands, and a `clean.sh` for the C# outputs. |
| RN parity | `bindings/react-native/parity.json` | Explicit exclusions with reasons for the spike exports and records; the parity guard failed until they were added. |

## 2. Results

Same request in every language: state
`{"zeta":1,"alpha":{"2":true,"10":false,"1":null},"mid":[1.0,-0.0,1e300,5e-324,u64::MAX,i64::MIN,2^63,2^64,-0],"é":"é\n",…}`;
a choice question with 30 options keyed `0,7,14,21,28,5,…` (integer-like and
scrambled); descriptions mixing text, JSON objects, absent values and JSON
strings; a yes/no question; a score question with levels
`[None, text, [1,2.5], "mid", null]`.

| Check | Rust | Python | C# (Unity bindings, C# 9) | Kotlin + JNI | Java + JNI | Swift (Linux) | Dart (FRB) |
|---|---|---|---|---|---|---|---|
| Rust IR saw state, 3 questions, 30 options in order | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Echo = normalized request (null → absent, JSON string → text, compact JSON) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Answers in request order, distribution in option order, yes/no = true,false | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 80 float bit patterns decode exactly (incl. `-inf` raw score) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Host re-encodes the result bit-identically; one-ULP change detected | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Invalid JSON → typed, content-free error | ✅ `InferenceError` | ✅ `XybridErrorInferenceError` | ✅ `XybridError.InferenceError` | ✅ `XybridError.InferenceError` | ✅ | ✅ `.inferenceError` | ✅ (FRB: `String`) |
| Probes: `Vec<Option<enum>>`, nested `f32`/`f64` lists (`-0.0`, subnormal, max, `-inf`, NaN payload), `Option<record>`, `u32::MAX`/`u64::MAX`, recursive JSON tree, records inside variants | ✅ | ✅ | ✅ | ✅ | not run (core checks only) | ✅ | ✅ (record-in-variant not probed; the shipped `FfiStreamEvent::Complete(FfiResult)` already uses it) |
| Checks | 12 unit tests | 21 | 19 | 20 | 11 | 20 | 2 tests |

TypeScript (WASM) was generated with `--deny-skipped` but not compiled or run.
`bindings/web` does not ship Bolt's TS output.

## 3. Representability, per shape

All entries are `boltffi generate <lang> --deny-skipped` = clean, plus a
compile/run where marked in §2. **Nothing needed flattening.** The existing
comment on `XybridStreamEvent` ("the pinned C# generator cannot lower that enum
shape reliably") no longer holds for these shapes at 0.30.1: records directly
inside data-enum variants round-trip in C#.

| Rust shape | Swift | Kotlin | Java | C# | Python | TS | Dart (FRB) |
|---|---|---|---|---|---|---|---|
| record | `struct` (Hashable) | `data class` | final class | `readonly record struct` (Unity: plain `readonly struct`) | frozen dataclass | `interface` | class (`==` per field) |
| data enum, struct variants | `enum` + payloads | sealed class | abstract class + static subclasses | `abstract record` + `sealed record`s | base + subclass per variant | tagged union | freezed sealed class |
| C-like enum | `enum: Int32` | `enum class` | enum | `enum` | `IntEnum` | enum | enum |
| `Option<T>` | `T?` | `T?` | `java.util.Optional<T>` | `T?` | `T \| None` | `T \| null` | `T?` |
| `Vec<Record>` | `[T]` | `List<T>` | `List<T>` | `T[]` | `list[T]` | `T[]` | `List<T>` |
| `Vec<Option<enum>>` | `[T?]` | `List<T?>` | `List<Optional<T>>` | `T?[]` | `list[T \| None]` | ✓ | `List<T?>` |
| `Vec<Vec<f32>>` | `[[Float]]` | `List<FloatArray>` | `List<float[]>` | `float[][]` | `list[list[float]]` | ✓ | `List<Float32List>` |
| recursive enum (via `Vec`) | ✓ (no `indirect`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `f32` / `f64` | bit-exact | bit-exact | bit-exact | bit-exact | bit-exact (`struct`) | — | bit-exact |
| `u32` / `u64` | `UInt32`/`UInt64` | `UInt`/`ULong` | **signed** `int`/`long` | `uint`/`ulong` | `int` | `number`/`bigint` | `int`/**`BigInt`** |
| `i64` | `Int64` | `Long` | `long` | `long` | `int` | `bigint` | `PlatformInt64` |

## 4. Constraints found (generator rules P9 must follow)

1. **C# rejects a variant field named like its variant.** `Text { text }` becomes
   `sealed record Text(string Text)`; .NET 8 fails with CS8866. Same for
   `Content { content }`, `Request { request }` and `Decisions { decisions }`.
   The existing `XybridEnvelopeKind::Text { text }` already hits this; the Unity
   script patches it by hand (transform (i)). Rule: the FFI field must differ
   (`Text { value }`, `Content { value }`). **For P9's envelope variants use
   `DecisionRequest { request }` and `DecisionResult { decisions }`;
   `Decisions { decisions }` does not compile in C#.** Python/Swift/Kotlin are
   unaffected.
2. **Kotlin: don't name a variant `List`** (or another stdlib collection). Inside
   the sealed class, `data class List(val items: List<…>)` resolves `List` to
   the variant, and kotlinc 1.9.22 fails with "no type arguments expected for
   class List". boltffi qualifies primitives (`kotlin.Int`) in that position but
   not collections. The JSON-tree probe now uses `Seq`/`Obj`.
3. **Unknown tags crash Swift.** Decoding an unknown data-enum tag calls
   `fatalError`, and an unknown C-like value force-unwraps `rawValue:)!`. Kotlin
   and Java throw `IllegalArgumentException`/`NoSuchElementException`, C# throws
   for data enums and silently casts for C-like ones, Python raises `ValueError`.
   Old glue cannot read new output variants. This is fine while glue and native
   lib ship in one artifact (XCFramework/AAR/wheel/Unity package). It means
   **appending a result variant is never compatible with older glue**. The repo
   already carries append-only-tail hacks for `XybridResult.reasoning_content` in
   three generator scripts, so skew does happen.
4. **The wire is positional.** Appending a field to an *input* record breaks
   old-glue → new-native calls; appending to an *output* record needs the
   `has_remaining` shim the scripts add for `reasoning_content`. So freeze the
   decision records' field sets before first release. Use records (not bare
   `Option`s) where growth is likely; that's why levels are `DecisionLevel`
   records.
5. **Default string forms print caller content in every generated language.**
   Swift `String(describing:)` reflects every field; Kotlin data-class
   `toString` and Python dataclass `repr` print everything. Java record
   `toString` prints the string fields (ids, keys), though its data-enum
   subclasses don't override `toString`. C# data-enum variant records print
   their payload, while the Unity down-levelled structs print only their type
   name. Dart freezed variants print their payload, while plain FRB classes
   print `Instance of …`. Verified in each harness (the `note` lines). Rust
   `Debug` is redacted, with canary tests in the facade. Contract 7's canary
   rule can only cover Rust and the hand-written wrappers (§9 C1-e).
6. **Equality.** Swift, Kotlin, Python and Java compare structurally (Kotlin
   `FloatArray` fields by reference). Unity's down-levelled structs fall back to
   reflection and compare arrays by reference. Dart plain classes compare `List`
   fields by identity, so two equal requests are not `==`; freezed variants
   compare deeply.
7. **`#[non_exhaustive]` IR enums force wildcard arms outside `xybrid-core`**
   (seen in Bolt). With the real IR, the facade's IR → FFI conversion has to be
   fallible (`Decisions::try_from_ir`), refusing an unknown variant as
   `UnsupportedModelCapability` like `Envelope::try_from_sdk` does.
8. **React Native's parity guard reads only `crates/xybrid-bolt/src/lib.rs`.**
   Keep the decision records there, or extend
   `tests/support/bolt-surface.mjs`. Otherwise the guard silently misses them.
9. **The Unity tripwire** (`EXPECTED_RECORD_STRUCTS`) must be bumped
   deliberately: 21 → 32 on this branch (8 decision record structs + 3 probes);
   the production set alone makes it 29.
10. **Java target**: disabled in `boltffi.toml`; without config it generates
    package `com.example.xybrid_bolt`, and `javac` needs `-encoding UTF-8`
    (non-ASCII doc comments). If Java is to ship, add a `[targets.java.jvm]`
    package and a compile gate. M6's "include Java" holds only if it ships.
11. **Diff noise**: adding types makes boltffi re-emit existing helpers in a
    different order (Kotlin, Python `_native.c`), so the committed bindings diff
    (+7,234 / −1,454 across four bindings, probes and spike hooks included)
    overstates the change. §7 has clean per-language numbers.

## 5. Ordering and number pitfalls (host side)

Same input everywhere:
`{"questions":{"10":1,"2":2,"b":3,"1":4},"state":{"x":1.0,"big":18446744073709551615,"neg0":-0,"negf":-0.0,"exp":1e300}}`

| Host library (tested here) | Object order | `1.0` | `u64::MAX` | `-0` / `-0.0` | `-inf` out |
|---|---|---|---|---|---|
| JS `JSON.parse`/`stringify` (Node 24/V8; key order and number parsing are ECMAScript rules, so Hermes/JSC behave the same) | **`1,2,10,b`** (integer-like keys first, ascending) | **`1`** | **`18446744073709552000`** | **`0` / `0`** | **`null`** |
| JS object insertion `b,2,a,1` | **`1,2,b,a`** | | | | |
| Swift `JSONSerialization` (Foundation, Linux) | **arbitrary** (`b,1,10,2`) | **`1`** | exact | `0` / `-0` | throws |
| Python `json` | kept | kept | exact | `0` / `-0.0` | **`-Infinity`** (bare, invalid JSON) |
| Dart `jsonDecode`/`jsonEncode` (VM) | kept | kept | **`18446744073709552000.0`** | `0` / `-0.0` | throws |
| .NET `JsonNode` | kept | kept (raw text) | exact | kept | — |
| .NET typed `double` | — | **`1`** | — | `-0` | throws |
| serde_json `Value` (workspace: no `preserve_order`) | **sorted** (`1,10,2,b`) | kept | exact | **`-0` → `F64(-0.0)`** (C3 wants `I64(0)`) | — |

Consequences:

- **Ordered collections cross as arrays only.** Questions, options, levels,
  answers and distributions are all arrays in the FFI form. RN's TurboModule
  `Object` arrives as `NSDictionary` / `ReadableMap` (unordered; RN's Swift
  codec even sorts metadata keys), so order is gone on both platforms before
  Bolt sees it, whatever JS did.
- **JSON content crosses as text and is parsed only in Rust**
  (`OrderedJson::parse`). A host must never build it from its own
  dictionaries. Python and Dart callers can serialize with their own
  libraries; JS, Swift and .NET-typed callers cannot without losing order or
  `1.0`. Binding docs and convenience helpers should accept JSON *text*.
  Helpers that take host objects should be refused or warned about.
- **SystemOne `questions` is an object.** A JS caller that builds SystemOne
  JSON with `JSON.stringify` from an object keyed by `"10","2","1"` sends the
  wrong question order before Rust is ever involved. The binding APIs should
  take question arrays, not SystemOne objects.
- **C4 (`-inf` raw scores):** the FFI form is fine everywhere. Bolt carries IEEE
  bits, FRB carries IEEE bits, and RN carries a `Double` through JSI.
  `-inf` survived in all seven runtimes. Any *JSON* form built by a host is not
  fine: `null` in JS, throws in Swift/Dart/.NET, a bare token in Python. If a
  binding needs a JSON result (SystemOne output, logs), it must get the C4
  `"-Infinity"` string form from Rust serde, never from the host serializer.
- **`u64`** (`evaluated_tokens`) is exact in RN only below 2^53, is `BigInt` in
  Dart, and is a signed `long` in Java. All acceptable for token counts.

## 6. Recommended FFI representation per contract-1 type

| Contract-1 type | Bolt (`Xybrid*`) / FRB (`Ffi*`) | RN wire (TS) |
|---|---|---|
| `OrderedJson`, `JsonNumber` | never crosses as a value: JSON **text** inside `DecisionContent::Json { json_text }`, parsed by `OrderedJson::parse` in the facade | `jsonText: string` |
| `DecisionContent` | data enum `Text { value } \| Json { json_text }` (field not `text`: §4.1) | `{ text } \| { jsonText }` |
| `DecisionRequest` | record `{ state: Option<Content>, questions: Vec<Question> }` | `{ state?, questions: [] }` |
| `DecisionState` | flattened into `request.state` (v1 has no media) | — |
| `DecisionMedia` | absent in FFI v1; O2 appends a `media` field (input wire change; fine because glue ships with native) | — |
| `DecisionQuestion` | record `{ id, instructions: Option<Instructions>, kind }` | `{ id, instructions?, kind }` |
| `DecisionInstructions` | data enum `Content { value }` (O2 appends `Spoken`) | `{ text } \| { jsonText }` |
| `DecisionQuestionKind` | data enum `Choice { options } \| YesNo { yes, no } \| Score { levels }`, no flattening | tagged union |
| `DecisionOption` | record `{ key, description: Option<Content> }` | |
| score levels (`Vec<Option<Content>>`) | `Vec<DecisionLevel { description: Option<Content> }>`; `Vec<Option<enum>>` also works everywhere, but the record can grow | `[{ description? }]` |
| `Decisions` | record `{ answers: Vec<Answer>, usage }` | |
| `DecisionUsage` | record `{ passes: u32, evaluated_tokens: u64 }` | numbers |
| `DecisionAnswer` | record `{ question_id, distribution, value, confidence: Option<f64> }`; public-field POD, invariants enforced by Rust on the way out, and no production entry accepts it inbound | |
| `DecisionScore` | record `{ key, raw_score: f32 (may be -inf), probability: f64, origin }` | numbers; never `JSON.stringify` |
| `DecisionOrigin`, `ScoreStatistic` | C-like enums | string literals |
| `DecisionValue` | data enum `Choice { key } \| YesNo { p_yes } \| Score { value, statistic }` | tagged union |
| envelope integration | append `XybridEnvelopeKind::DecisionRequest { request }`, `::DecisionResult { decisions }`, `XybridOutputType::Decisions` at the tails; no new `run*` functions (run-surface rule) | `kind: 'decisionRequest' / 'decisions'`, `outputType: 'decisions'` |
| errors | existing variants: invalid request → `InferenceError`, refusal → `UnsupportedModelCapability`; no new `XybridError` variant for v1 | existing codes |

## 7. Generated-code size and ergonomics

Production-shaped surface (14 decision records/enums, 2 envelope variants, 1
output-type variant, no spike hooks or probes) vs master, raw `boltffi generate`
output:

| Target | Lines before → after | +lines | +KiB | New files |
|---|---|---|---|---|
| Swift (+ C header) | 3,521 → 3,906 | +385 (+10.9%) | +14.3 | 0 |
| Kotlin (+ JNI glue) | 7,050 → 7,762 | +712 (+10.1%) | +22.2 | 0 |
| Java (+ JNI glue, not shipped) | 10,473 → 11,707 | +1,234 (+11.8%) | +39.7 | 14 |
| C# | 5,565 → 6,211 | +646 (+11.6%) | +19.2 | 14 |
| Python (`.py` + `.pyi` + `_native.c`) | 15,244 → 17,143 | +1,899 (+12.5%) | +64.5 | 0 |
| TypeScript | 5,509 → 6,421 | +912 (+16.6%) | +28.3 | 0 |

- Hand-written Rust in this spike: the facade module is 645 lines (records,
  translation, bit-diff helper and redacted `Debug`); the Bolt mirrors + `From`
  impls are 372 lines (plus 211 of spike exports and probes, 118 of tests); the
  FRB mirrors + `From` impls are 349 of 491 lines. The stand-in IR (1,372 lines)
  is P1's job and will be replaced.
- Dart (FRB, spike incl. probes and 9 functions): `decision.dart` 428 lines,
  `decision.freezed.dart` 3,261 (freezed dominates), and `frb_generated.*`
  +~4.6k lines across Dart and Rust.
- Native binary size: **not measured** (would need two release builds;
  spike-only code would also inflate it).
- Ergonomics are natural everywhere. Swift: `.text(value:)`,
  `.choice(options:)`, exhaustive `switch`. Kotlin: sealed classes and `copy`,
  but `UInt`/`ULong`. Python: one class per variant (`XybridDecisionContentText`),
  which `_sugar.py` should wrap in factories. C#: `new
  XybridDecisionContent.Text("…")` with pattern matching. Java: `Optional`
  everywhere. Dart: freezed factories, but `BigInt` for `u64` and identity
  `==` on lists.

## 8. React Native (analysis; no implementation)

RN is hand-bridged (Codegen TurboModule over the Swift and Kotlin SDKs), so
nothing arrives for free. A decision surface needs:

- **Codegen spec (`src/NativeXybrid.ts`)**: no new method. Decisions ride
  `run(model, envelope: Object, options)` (run-surface rule), with a new wire
  envelope kind.
- **TS facade (`src/types.ts`, `src/wire.ts`, `src/envelope.ts`)**:
  `DecisionRequest` with **arrays** for questions, options and levels.
  `DecisionContent = { text } | { jsonText }`, where JSON content is a string
  passed through untouched: never `JSON.parse`/`stringify` it (§5). Add
  `WireEnvelope` `{ kind: 'decisionRequest', request }` and the result
  `{ kind: 'decisions', decisions }`, `OutputType 'decisions'`,
  `InferenceResult.decisions`, and an `Envelope.decisionRequest(...)` factory.
  Don't offer a SystemOne-object helper in JS (integer-like question ids reorder).
- **Native codecs**: `ios/XybridCodec.swift` (`decodeEnvelope`,
  `encodeEnvelope`, `encodeOutputType`) and
  `android/.../XybridCodec.kt` (`when` on kind / output type) each need
  decode-request and encode-decisions functions over arrays
  (`[Any]`/`ReadableArray`). Raw scores go out as `Double(Float)` / `putDouble`
  (`-inf` survives JSI), and `u64` as `Double`.
- **The two codecs switch exhaustively** (`XybridCodec.swift:189`, `:301`;
  `XybridCodec.kt:190`, `:309`; Kotlin ≥ 1.7 makes non-exhaustive `when` on
  sealed/enum types an error). **P9 appending `XybridEnvelopeKind` /
  `XybridOutputType` variants breaks the RN build on both platforms** until those
  arms exist. P9 must add refusal arms (or the full codec) in the same change, or
  land together with P10a.
- **Errors**: no new codes (`xybrid_inference_error`,
  `xybrid_unsupported_model_capability` cover it).
- **`parity.json`**: map the 14 records' fields/variants, the 2 envelope
  variants and the output type; `tests/parity.test.mjs` enforces it. It caught
  the spike's additions at once (`bolt has #[data] XybridDecisionContent but
  parity.json doesn't map it`). A staged PR keeps explicit exclusions with
  reasons *and* keeps the load refusal.
- **Tests**: `tests/wire.test.mjs` (order through `toWire`/`fromWire` with
  integer-like ids, `-Infinity` and `u64` boundaries),
  `tests/facade.test.mjs` (fake native), `tests/swift/main.swift` and
  `tests/kotlin/CodecTest.kt` (codec order and `-inf`).

## 9. Contract-1 changes recommended before freeze (go/no-go)

**GO**, with:

- **C1-a `DecisionContent` must be opaque.** C3 makes `Some(Json(null))`
  unconstructible, which a public `enum DecisionContent { Text(String),
  Json(OrderedJson) }` cannot guarantee. The stand-in shows a working shape:
  `DecisionContent::text(..)`, `DecisionContent::from_json(OrderedJson) ->
  Option<Self>` (null → `None`, string → `Text`) and a `view()` enum. The
  contract text should say so.
- **C1-b Number parsing must not follow serde_json.** serde_json maps `-0` to
  `F64(-0.0)` (C3: `I64(0)`), `Value` sorts keys, and `float_roundtrip` is off in
  the workspace (default float parsing is best effort). The stand-in's
  hand-written parser (~400 lines, `str::parse::<f64>` = correctly rounded,
  I64 → U64 → F64) passes the classification tests. If P1 uses a serde visitor,
  it needs explicit `-0` and float handling.
- **C1-c Fallible IR → FFI conversion.** Because the IR enums are
  `#[non_exhaustive]`, the contract should state that the facade refuses unknown
  variants (`UnsupportedModelCapability`, "unrepresentable") rather than
  panicking or dropping them.
- **C1-d (optional, cheap now) `DecisionLevel { description }`** in the IR
  instead of `Vec<Option<DecisionContent>>`. The FFI uses a record either way,
  and per-level fields later (labels, numeric values for `Expected`) would
  otherwise be a breaking IR change.
- **C1-e Redaction scope.** Contract 1/7 canaries hold for Rust `Debug` (tested)
  and hand-written wrappers. Generated foreign records print content via
  default `toString`/`repr`/`description` in every language (§4.5), so the
  contract should scope the rule, or P9/P10 must add wrapper types.
- **C1-f Contract 4 naming.** `DecisionContent` FFI = `Text { value } |
  Json { json_text }`, not `{text}` (CS8866); RN's TS may still say `{text}`. The
  envelope result variant must not be `Decisions { decisions }` (use
  `DecisionResult { decisions }`). No variant named `List` (or other Kotlin
  collection names).
- **C1-g C4 confirmed** for every FFI form. Add: bindings never render decision
  results to JSON with host serializers.
- **C1-h Wire compatibility (P9's, per contract).** State it explicitly: no
  cross-version glue/native compatibility for decision records; enums
  append-only; record field sets frozen at first release (only output tails with
  `has_remaining` shims). No FFI entry point accepts `Decisions` inbound (the
  spike's `verify` hook excepted).

No-go items: none.

## 10. Minimal production plan

**P9 (facade + Bolt + generated outputs)**, after P7:
1. Facade: keep `decision/mod.rs` (records, `into_ir`, redacted `Debug`, the 9
   tests), swap `stand_in` for `xybrid_sdk::ir` decision types, make IR → FFI
   `try_from_ir`. Add `EnvelopeKind::{DecisionRequest, DecisionResult}` +
   `OutputType::Decisions` with `into_sdk`/`try_from_sdk`/`of_envelope` arms over
   P7's SDK variants. Check question/option counts and the total input bytes
   *before* parsing any JSON text (the spike parses first and applies its byte
   limit per text), so an oversized request is refused cheaply (contract 7).
2. Load gate: today `ensure_ffi_supported` refuses scorers for every binding.
   Make it capability-aware, e.g. allow decision models only for bindings that
   registered and support them (`facade::get_binding()` already names the
   caller; mind the `swift`/`kotlin` fallback before registration). Each
   P10 then lifts its own gate with complete input **and** output support
   (contract 4 / M6).
3. Bolt: the 14 mirrors from this spike (drop `decision_spike_*` and probes),
   envelope + output-type variants at the tails, `From` impls. No new exports.
   Run the four scripts, bump the Unity tripwire, keep Java disabled unless it
   ships.
4. Same change, so nothing stops compiling: RN codec arms (refuse until P10a),
   Swift/Kotlin wrapper accessors (`result.decisions`, envelope factory), Unity
   `InferenceResult.MapOutputType` arm, and `parity.json` exclusions with reasons.
5. Tests worth promoting from the spike: the facade unit tests; a Python
   round-trip test in `bindings/python/tests`, which the `python-sdk` CI job
   already builds natives for, so it is a real cross-boundary check on every
   PR; optionally running the C# harness under `tools/unity-bolt-compile-check`
   on .NET 8 with the relinked cdylib.

**P10a React Native**: §8.

**P10b Flutter**: the spike's FRB mirrors (minus hooks and probes);
`FfiEnvelope::decision_request(FfiDecisionRequest)`; `FfiResult.decisions:
Option<FfiDecisions>` + an `ensure_ffi_payload` arm; Dart wrappers in `lib/src`;
FRB 2.11.1 regeneration (the CI drift check); lift Flutter's own load guard;
mock tests plus a native round-trip test gated like the spike's.

**P10c Python + Unity**: `_sugar.py` factories, `XybridResult.decisions` and
envelope helpers (no `_errors.py` change). Unity `Runtime/Api` (`Envelope`
factory, `InferenceResult.Decisions`, output type), the compile gate, and a
runtime smoke as in the C# harness.

## 11. Not verified

- Real mobile runtimes: Swift ran on Linux (swift.org 6.2), not iOS/macOS;
  Kotlin/Java ran on a desktop JVM, not Android ART; C# on .NET 8 CoreCLR at
  C# 9, not Unity Mono/IL2CPP (the netstandard2.1 compile gate passed). The wire
  is the same little-endian byte layout everywhere, so I expect the same results
  there, but they did not run.
- Bazel builds (`//bindings/kotlin:xybrid_kotlin_aar`,
  `//bindings/apple:XybridFFI`) and the Swift wrapper compile on macOS
  (`build-apple`).
- TypeScript/WASM output: generated only.
- React Native: analysis only (Node test suite run; no device build).
- Native binary size delta; performance of large requests.
- Uninhabited `DecisionMedia` as an FFI type (not attempted; v1 omits it).

## 12. Verification (commands and results)

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass (4m26s; only llama-cpp-sys build-script notices) |
| `cargo test -p xybrid-ffi-facade -p xybrid_bolt -p xybrid_flutter` | facade 81 + 2 passed, bolt 19 passed, flutter 41 passed |
| `cargo test -p xybrid-ffi-facade --lib decision` / `-p xybrid_bolt --lib decision` | 9 / 3 passed |
| `cargo test --workspace --features ort-download` | pass: 91 suites, 2,696 passed, 0 failed, 30 ignored (7m58s) |
| `python3 tools/scripts/gen_python_bolt.py --check`, `gen_kotlin_bolt.py --check`, `gen_unity_bolt_csharp.py --check`, `bindings/apple/scripts/gen-bolt-bindings.sh --check` | all "up to date" |
| `boltffi generate {swift,kotlin,python,csharp,typescript,java} --deny-skipped` | all succeed (Java with `targets.java.jvm` enabled locally) |
| `dotnet build tools/unity-bolt-compile-check/UnityBoltCompileCheck.csproj -c Release` + native import check | build succeeded (netstandard2.1 + net8.0, C# 9); 148 imports, 0 failures |
| raw generated C# at net8.0 (before the rename) | CS8866 on `Text(string Text)`, `Content(... Content)`, `Request(... Request)`, `Decisions(... Decisions)` (and the pre-existing `XybridEnvelopeKind.Text`) |
| `kotlinc` 1.9.22 on generated `XybridBolt.kt` (with the `List` probe variant) | "no type arguments expected for class List" ×2; passes after the rename |
| `swiftc -typecheck` (Swift 6.2, Linux) on generated `xybrid_bolt.swift` | clean |
| `javac -encoding UTF-8` on generated Java | clean (without `-encoding`: unmappable-character errors) |
| Python harness (wheel and `xybrid._bolt`) | 21/21 |
| C# harness (Unity bindings, C# 9, .NET 8) | 19/19 |
| Kotlin harness (JVM + generated JNI glue) | 20/20 |
| Java harness (`boltffi pack java`) | 11/11 |
| Swift harness (Linux) | 20/20 |
| `flutter_rust_bridge_codegen generate` (2.11.1) on unmodified tree | no diff (byte-identical to committed) |
| `flutter test test/spike/decision_roundtrip_test.dart` with `XYBRID_P4_NATIVE_LIB` | 2/2 |
| `flutter test` (whole package, no native lib) | 47 passed, 2 skipped (the spike tests) |
| `dart analyze` on generated `decision.dart` + test | clean (one info fixed) |
| `/tmp/p4venv/bin/python -m pytest bindings/python/tests -q` (regenerated `_bolt`) | 64 passed |
| `node tests/run.mjs` (React Native) | before exclusions: parity fails on the new export and record; after: 42 passed, 1 skipped |
| `python3.11 -m unittest discover -s tools/scripts/tests -p 'test_*.py'` | 555 OK, 1 skipped (same on master). With AL2023's default python3 3.9: 6 failures + 51 errors on branch *and* master (interpreter, not this change) |
| host JSON probes (`tools/spikes/p4-decision-bindings/host-json`) | §5 table |
