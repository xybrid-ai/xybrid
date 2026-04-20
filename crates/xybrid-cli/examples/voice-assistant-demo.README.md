# Voice Assistant Demo — 3-stage tracing showcase

A 3-stage, fully-on-device YAML pipeline meant to produce a rich
flame-graph on the Xybrid Studio `/traces` dashboard:

| Stage | Model | What the span shows |
|-------|----------------------------|----------------------------------------------------------|
| `asr` | `wav2vec2-base-960h` | audio preprocess + CTC decode timing |
| `llm` | `qwen2.5-0.5b-instruct` | prefill + decode bars, `ttft_ms`, `tokens_in/out`, finish reason |
| `tts` | `kokoro-82m` | phonemize + model inference |

YAML lives at [`voice-assistant-demo.yaml`](./voice-assistant-demo.yaml).

---

## One-time setup

All three models need to be extracted on disk so the orchestrator finds
them by id. Warm each with a single-model run (each downloads the `.xyb`
and extracts it to `~/.xybrid/cache/extracted/<id>/`):

```bash
cd repos/xybrid   # workstation layout

./target/release/xybrid -q run -m wav2vec2-base-960h \
    --input-audio integration-tests/fixtures/input/jfk.wav

./target/release/xybrid -q run -m qwen2.5-0.5b-instruct \
    --input-text "hi" >/dev/null

./target/release/xybrid -q run -m kokoro-82m \
    --input-text "hi" -o /tmp/warm.wav
```

Confirm all three show up:

```bash
ls ~/.xybrid/cache/extracted/ | grep -E 'wav2vec2|qwen2.5|kokoro'
```

---

## Running the pipeline

### A) SDK path (recommended for the tracing demo)

This path goes through `PipelineRef::from_yaml().load().run()`, which
handles bundle extraction + emits a proper span tree via
`xybrid_core::tracing`. It's the same path the production SDK uses, so
what you see in `/traces` after this run is exactly what a real
customer integration produces.

```bash
cd repos/xybrid

# Local platform (requires backend on :3001 + ingest on :8081)
XYBRID_API_KEY=sk_test_u9Y5WQeT6SfMmvdj2iwJ6dYdzehclUIs \
XYBRID_PLATFORM_URL=http://localhost:8081 \
  cargo run --release --example voice_assistant_demo \
    -p xybrid-sdk --features platform-macos -- \
    integration-tests/fixtures/input/jfk.wav \
    /tmp/voice-assistant-reply.wav
```

Open <http://localhost:5173/traces> — the newest row is a
`PipelineComplete` event with three nested spans and populated
Token / TTFT / Decode / Prefill metadata. Click it for the flamegraph.

Leave `XYBRID_API_KEY` unset to run locally without shipping telemetry
(useful while developing the pipeline).

### B) CLI path (works, but no spans yet)

```bash
./target/release/xybrid run \
  -c crates/xybrid-cli/examples/voice-assistant-demo.yaml \
  --input-audio integration-tests/fixtures/input/jfk.wav \
  --output /tmp/voice-assistant-reply.wav \
  --trace
```

The pipeline executes end-to-end but the CLI currently prints
`No spans recorded` and routes the LLM stage through the cloud-stub
availability function instead of the extracted local model. Use path
(A) until that gap is closed in the CLI pipeline runner
(`crates/xybrid-cli/src/commands/run.rs::execute_pipeline`).

---

## Tuning the trace shape

Adjust the YAML for different flamegraphs — the shape of the spans
reflects the stage config directly.

| Knob                     | Effect                                              |
|--------------------------|-----------------------------------------------------|
| `llm.max_tokens: 80`     | Longer / shorter decode bar; more token metadata    |
| `llm.streaming: true`    | Emits the partial-token ladder the flame-graph uses |
| `tts.voice: af_bella`    | Swap to any voice from `xybrid models info kokoro-82m` |
| Add a `policy.yaml` stage| Shows the policy-engine pre-stage span              |
| Split into named IDs     | Each `id:` becomes the span label — prefer short names (`asr`, `llm`, `tts`) |

---

## Troubleshooting

**`BundleNotExtracted` panic**  
Skip straight to "One-time setup" above — the pipeline expects
extracted dirs under `~/.xybrid/cache/extracted/<id>/`. Single-model
runs (`xybrid -m <id>`) populate that directory as a side-effect.

**LLM stage routes to `cloud` and emits `cloud-output-...`**  
Only happens with the CLI pipeline path (B). Known gap: the CLI's
`Orchestrator::execute_pipeline` doesn't register the SDK's extracted
bundles with the `availability_fn`, so everything hits the cloud
availability stub. SDK path (A) wires this correctly.

**No row shows up on `/traces`**  
Check the backend is up (`lsof -nP -iTCP:3001`) and its analytics
backend is running. The SDK path hits the **ingest** service on :8081,
not the API on :3001.
