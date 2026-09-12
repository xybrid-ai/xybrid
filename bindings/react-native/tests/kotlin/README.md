# Kotlin metrics conversion fixture

Run `npm test` from `bindings/react-native`, with the Kotlin CLI (`kotlinc`,
1.9 or newer) and a JDK (`java`) on `PATH`. To run just the Android checks:

```sh
node --test --test-name-pattern='Android' tests/metrics-encoders.test.mjs
```

The Node test compiles the **production** `XybridMetricsEncoder.kt` and runs
`main.kt` on the JVM. Like the Swift fixture, the canonical metrics records are
synthetic; the React Native map/array doubles record exactly what the converter
writes. No converter implementation is copied into the tests. These files are
outside Android's source sets and are never included in the native module.

The fixture covers distinct populated values, every stage's ID and latency in
order, absent optionals, an empty stage list, reported zeroes, each optional
field independently, and unsigned values above `Int.MAX_VALUE`. Swapping
`tokensPerSecond` and `prefillTps` in the production converter fails it.

The existing Build React Native `npm test` step runs this fixture; its Ubuntu
runner provides Kotlin and Java. A missing compiler, compilation error, or
failed assertion fails the test rather than silently skipping it. Temporary
build output is isolated per run and removed on both success and failure.

This validates Kotlin conversion logic, not the React Native/JNI runtime,
native model execution, or packaged Android linking; the Android example build
remains the integration gate.
