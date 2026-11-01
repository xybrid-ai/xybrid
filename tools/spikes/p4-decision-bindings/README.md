# P4 decision-binding spike: round-trip harnesses

Throwaway evidence for the P4 binding feasibility spike (decision-model
framework, contract v2.1). Each harness sends the same decision request
through generated bindings into the real native library and checks:

- order: Rust reports what its IR saw (`decision_spike_request_outline`) and
  the harness compares it with what it sent — an ordered JSON state with
  integer-like keys, three question kinds, 30 options with scrambled
  integer-like keys;
- normalization: JSON `null` becomes absence, a JSON string becomes text, JSON
  becomes compact text (contract v2.1 C3);
- exactness: every `f32`/`f64` of a stub result decodes bit for bit (Rust
  sends the expected bit patterns separately), including a `-inf` raw score
  (C4), and the result the harness re-encodes is bit-identical in Rust; a
  one-ULP change is detected;
- typed, content-free errors; the probe shapes (`Vec<Option<enum>>`, nested
  lists, a recursive JSON tree, records inside variants).

Nothing here ships. See `.context/p4-binding-spike.md` on the spike branch
for the report.

## Native library

All Bolt harnesses use the boltffi-relinked cdylib from the Python wheel (a
plain `cargo build` omits the generated export shims):

```bash
cd crates/xybrid-bolt
boltffi pack python --python python3.11          # boltffi 0.30.1
mkdir -p /tmp/p4wheel && cd /tmp/p4wheel && python3 -c \
  "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" \
  "$OLDPWD"/dist/python/wheelhouse/xybrid_bolt-*-cp311-*.whl
```

## Python (generated wheel, CPython bridge)

```bash
python3.11 -m venv /tmp/p4venv
/tmp/p4venv/bin/pip install crates/xybrid-bolt/dist/python/wheelhouse/*.whl
cd /tmp && /tmp/p4venv/bin/python -I "$OLDPWD"/tools/spikes/p4-decision-bindings/python/roundtrip.py
```

## C# (the Unity-down-leveled bindings, C# 9, on .NET 8)

```bash
dotnet build tools/spikes/p4-decision-bindings/csharp/P4DecisionRoundTrip.csproj -c Release
LD_LIBRARY_PATH=/tmp/p4wheel/xybrid_bolt \
  dotnet tools/spikes/p4-decision-bindings/csharp/bin/Release/net8.0/P4DecisionRoundTrip.dll
```

## Kotlin (generated Kotlin + JNI glue, desktop JVM)

Kotlin 1.9.22 and kotlinx-coroutines 1.7.3, the versions bindings/kotlin pins:

```bash
kotlinc bindings/kotlin/src/main/kotlin/ai/xybrid/XybridBolt.kt \
  -cp kotlinx-coroutines-core-jvm-1.7.3.jar -d /tmp/p4kt/generated.jar
gcc -shared -fPIC -o /tmp/p4kt/libxybrid_bolt_jni.so bindings/kotlin/bazel/jni/jni_glue.c \
  -Ibindings/kotlin/bazel/jni -I"$JAVA_HOME"/include -I"$JAVA_HOME"/include/linux \
  -L/tmp/p4wheel/xybrid_bolt -lxybrid_bolt -Wl,-rpath,/tmp/p4wheel/xybrid_bolt
kotlinc tools/spikes/p4-decision-bindings/kotlin/Roundtrip.kt \
  -cp /tmp/p4kt/generated.jar:kotlinx-coroutines-core-jvm-1.7.3.jar -d /tmp/p4kt/roundtrip.jar
java -Djava.library.path=/tmp/p4kt -cp \
  /tmp/p4kt/roundtrip.jar:/tmp/p4kt/generated.jar:kotlinx-coroutines-core-jvm-1.7.3.jar:"$KOTLIN_HOME"/lib/kotlin-stdlib.jar \
  RoundtripKt
```

## Java (boltffi's Java target, disabled in boltffi.toml)

Switch `[targets.java.jvm] enabled = true` locally (do not commit it), then:

```bash
(cd crates/xybrid-bolt && boltffi pack java)
cd /tmp && javac -encoding UTF-8 -d /tmp/p4java $(find "$OLDPWD"/crates/xybrid-bolt/dist/java/com -name '*.java') \
  "$OLDPWD"/tools/spikes/p4-decision-bindings/java/Roundtrip.java
java -Djava.library.path="$OLDPWD"/crates/xybrid-bolt/dist/java -cp /tmp/p4java Roundtrip
```

## Swift (committed generated Swift, swift.org toolchain on Linux)

```bash
mkdir -p /tmp/p4swift/mod && cp bindings/apple/include/xybrid-bolt.h /tmp/p4swift/mod/
printf 'module XybridFFI {\n  header "xybrid-bolt.h"\n  export *\n}\n' > /tmp/p4swift/mod/module.modulemap
swiftc -swift-version 5 -module-name XybridSpike -I /tmp/p4swift/mod \
  tools/spikes/p4-decision-bindings/swift/main.swift bindings/apple/Sources/Xybrid/xybrid_bolt.swift \
  -L/tmp/p4wheel/xybrid_bolt -lxybrid_bolt -Xlinker -rpath -Xlinker /tmp/p4wheel/xybrid_bolt -o /tmp/p4swift/roundtrip
/tmp/p4swift/roundtrip
```

## Dart (FRB 2.11.1, Flutter 3.35.7)

The Dart harness lives in the Flutter package so it can import the generated
bindings: `bindings/flutter/test/spike/decision_roundtrip_test.dart`. It skips
unless a native library is named:

```bash
cargo build -p xybrid_flutter
cd bindings/flutter && XYBRID_P4_NATIVE_LIB="$PWD"/../../target/debug/libxybrid_flutter_ffi.so \
  flutter test test/spike/decision_roundtrip_test.dart
```
