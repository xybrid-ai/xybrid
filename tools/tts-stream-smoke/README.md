# Managed TTS native-host check

Compiles the committed Unity managed API and generated Bolt C# into a .NET 8
console host. This checks actual packet decoding, cancellation, resident reuse
and batch outcomes with real local models. It does not run Unity or IL2CPP.

Build the plugin with Bazel and stage native dependencies as described in
`bindings/unity/Samples~/KittenTts/README.md`. Use identical assets/options when
comparing timings with a raw engine host. Run after other builds/tests finish to
avoid CPU contention. Kitten loader construction is lazy: `cold-short` includes
native engine open; `loader` measures only SDK metadata setup.

```bash
LD_LIBRARY_PATH=/path/to/Plugins/Linux:/path/to/Plugins \
  dotnet run --project tools/tts-stream-smoke -- \
  kitten /path/to/bundle /path/to/capped-bundle /path/to/missing-bundle
LD_LIBRARY_PATH=/path/to/Plugins/Linux:/path/to/Plugins \
  dotnet run --project tools/tts-stream-smoke -- \
  kokoro integration-tests/fixtures/models/kokoro-82m
# Target only first-packet cancellation, stale-queue suppression and reuse.
LD_LIBRARY_PATH=/path/to/Plugins/Linux:/path/to/Plugins \
  dotnet run --project tools/tts-stream-smoke -- delivery /path/to/bundle
```

The normal bundle uses six threads, seed/token-cap zero, sentence mode, and
Accelerate only on macOS. The capped directory uses the same four assets with
`max_tokens: 8`. The missing directory has valid metadata pointing at absent
files. Dispose each model only after its active worker returns. JSON output
includes first audio, total time, speech duration, packets, engine chunk/limit
counts, RSS, largest packet gap and gaps over 10 ms. On this transport, short
packet gaps measure delivery while longer gaps include the next waveform pass;
they are not promises about playback latency.
