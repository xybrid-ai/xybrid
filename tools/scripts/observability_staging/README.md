# Staging acceptance scenario (step 6)

One real-model scenario that every binding runs through its own public API,
on its real runtime, exporting to Xybrid **staging** two ways in one launch:

| | `direct` session | `collector` session |
|---|---|---|
| Export | the Xybrid preset (`xybrid(api_key)`) with its traces endpoint set to staging: bearer key, protobuf, TLS from the device | `otlp(<collector endpoint>)` to the pinned Collector on the host, which forwards traces and evaluation results to staging and keeps metrics and other logs in a local sink |
| Content | metadata only | captured, with one literal redaction |
| Key on the device | yes, for the run (read from `config.json`, then deleted) | no |

The driver (`run_staging.py`) then reads every row back from the staging API
by trace id and checks it against what the binding said it did
(`expectations.json`), against the Collector's copy, and against the
dashboard aggregate. It is the per-binding proof that complements the
model-free capture scenario in
[`../observability_capture/scenario.md`](../observability_capture/scenario.md).

| File | Role |
|---|---|
| `staging.json` | Every constant below; lanes, prompts, the secret, feedback and evaluation values. |
| `run_staging.py` | Driver: `run` (one lane: Collector, config, lane command, readback, checks, report), `isolation` (cross-tenant reads, run where the other tenants' keys live), `recheck`, `summary`. |
| `staging_checks.py` | Every check, as pure functions of the saved answers. |
| `lanes.sh` | One subcommand per lane: build or stage the candidate, copy the config to the runtime, launch, copy `expectations.json` back. |
| `react-native-lane.sh` | The React Native lanes' device steps (Metro, config copy, launch without tapping, polling for `expectations.json`), called by `lanes.sh`. |
| `collector-lan.yaml` | Collector overlay: bind the OTLP receiver to the host's LAN address (iPhone lanes). |
| `console.mjs` | Read-only console check: after you sign in on a temporary Chrome profile, every launch trace's page and the overview KPIs against the staging API. Set `XYBRID_CONSOLE_PROFILE` to a private directory outside the evidence (e.g. `~/.xybrid-staging-console`) to keep the profile, and so the sign-in, between runs; delete it with the step's keys. `XYBRID_CONSOLE_WORKSPACE=/o/<org>/<project>` opens the launch project directly; the check still verifies it. |
| `fixtures/` | A recorded passing run (Python lane), used by the unit tests. |

Python 3.9+ standard library only. Unit tests:
`python3 -m unittest tools/scripts/tests/test_observability_staging.py`.

## Never production, never the key in evidence

Both staging hosts are checked by `run_interop.check_xybrid_target` before any
request, and no redirect is followed. The key is read from a file
(`--key-file`, default `~/.xybrid-step6/key`, mode 0600) and reaches exactly
two places: the Collector's environment and the lane's `config.json`
(mode 0600, in a private directory outside the evidence), which the lane copies
to the runtime and the scenario deletes after reading it. The driver deletes
its copy when the lane command returns, removes `OTEL_*` and `XYBRID_*KEY`
variables from the lane's environment, and scans every file it wrote for the
key and for `Bearer ` before it finishes (`evidence.secrets`).

## config.json (input)

Written by the driver, one per launch:

```json
{
  "schema": "xybrid-staging-scenario/1",
  "binding": "python", "lane": "python-mac",
  "run": "s6-20261111t153000z-python-mac",
  "commit": "00e617102bc77ca0602115669cf62a68e56a92bb",
  "model_id": "lfm2.5-230m",
  "sessions": {
    "direct": {"traces_endpoint": "https://staging.ingest.xybrid.dev/v1/traces", "api_key": "<key>",
               "trace_id": "<32 hex>", "parent_span_id": "<16 hex>"},
    "collector": {"endpoint": "http://127.0.0.1:4318",
                  "trace_id": "<32 hex>", "parent_span_id": "<16 hex>"}
  },
  "constants": { "...": "staging.json, without the lanes table" }
}
```

T and P are random per session and per launch, so every row the checks read
belongs to this launch. A lane that cannot read files (the browser) receives
the same object from its test runner.

## Steps (each session, `direct` then `collector`, one process)

1. **Configure and start.**
   - `direct`: `xybrid(api_key)`, traces endpoint = `sessions.direct.traces_endpoint`,
     metadata only.
   - `collector`: `otlp(sessions.collector.endpoint)`, default content capture,
     literal redaction `secret` → `replacement`.
   - Both: service `xybrid-staging-{binding}` / `1.0.0`, device id
     `staging-device-{lane}`, resource attributes
     `xybrid.acceptance.run` = run, `.lane` = lane, `.commit` = commit,
     `.session` = the session name.
2. **Workflow** `staging-workflow` with `traceparent = 00-T-P-01`; record its
   reference as `workflow`; its context's `traceparent`/`tracestate` are the
   parent of everything below.
3. **Load** `model_id` from the registry with the workflow as parent and
   conversation id `staging-conversation-{run}-{session}`; record
   `load_observation_ref` as `load`.
4. **Stopped stream.** Stream `stopped.prompt` with `max_tokens =
   stopped.max_tokens`, the workflow parent and the conversation id. After
   `stop_after_chunks` chunks, stop it the binding's own way (close the
   stream, cancel the token, return an error from the callback, cancel the
   subscription). Record `stopped = {"outcome", "chunks_consumed", "delivery"}`:
   `outcome` is `cancelled` when the stream ended without a result,
   `completed` if it finished anyway, `error: ...` otherwise; `delivery` is
   `callback` when the binding's callback refused the next chunk itself
   (the span then counts exactly the consumed chunks), `pull` or `push` when
   chunks were buffered ahead of the consumer (the span counts what the SDK
   handed over, at least the consumed chunks).
5. **Completed stream.** Stream `completed.prompt` with `max_tokens =
   completed.max_tokens`, same parent and conversation id, to the end. Record
   `completed = {"outcome": "ok", "chunks", "tokens_out", "text_chars",
   "finish_reason"}`: `chunks` counts the token events received,
   `tokens_out` is the result's output token count (`metrics.tokens_out`, or
   the binding's equivalent; `null` when it has none), and the run's
   reference is `run`.
6. **Quality.** `record_feedback(run, score=1.0, label="thumbs_up",
   comment=feedback.comment)` and `record_evaluation(run, name, 0.75, "pass",
   explanation)`; record both return values as `quality = {"feedback",
   "evaluation"}`.
7. **Lifecycle.** Release the model, complete the workflow, flush (record
   `"ok"` or `"error: ..."`), stats `after_flush`, shut down (record), stats
   `after_shutdown`.

Then write `expectations.json` (even when a step threw: set
`scenario_error` and exit non-zero):

```json
{
  "schema": "xybrid-staging-scenario/1",
  "binding": "python", "lane": "python-mac", "run": "...",
  "sdk_version": "0.17.0", "model_id": "lfm2.5-230m",
  "sessions": {
    "direct": {
      "trace_id": "...", "parent_span_id": "...", "init": "ok",
      "references": {"workflow": {"trace_id": "...", "span_id": "..."}, "load": {}, "run": {}},
      "stopped": {"outcome": "cancelled", "chunks_consumed": 3, "delivery": "pull"},
      "completed": {"outcome": "ok", "chunks": 9, "tokens_out": 10, "text_chars": 41, "finish_reason": "stop"},
      "quality": {"feedback": false, "evaluation": false},
      "lifecycle": {"flush": "ok", "shutdown": "ok"},
      "stats": {"after_flush": {"enabled": true, "traces": {"accepted": 5, "exported": 5, "dropped": 0,
                                                            "failed": 0, "pending": 0}, "logs": {}, "metrics": {}},
                "after_shutdown": {}}
    },
    "collector": {}
  },
  "scenario_error": null
}
```

## Checks

Per session (`<lane>/<session>` in the report):

| Check | Passes when |
|---|---|
| `expectations.valid` | Schema, binding, lane, run, T and P match the config; init ok; nothing missing; no `scenario_error`. |
| `refs.stored` | `workflow`, `load` and `run` are stored spans with the reported trace and span ids, in T. |
| `trace.rooted` | The workflow's parent is P; P is not stored; it is the only span under P; every other span in T has a stored parent. |
| `identity` | Every stored span's resource has `xybrid.sdk.binding`, service name and version, device id and the four `xybrid.acceptance.*` values of this launch. |
| `load.span` | The load span is a `load_model` child of the workflow, without error. |
| `run.mapping` | The run span is a counted `genai.request` child of the workflow: model, provider, local execution, conversation id, no error; output tokens and chunk count equal what the binding reported; time to first chunk present. |
| `stream.cancelled` | Exactly one other request under the workflow: `error.type = cancelled`, status ERROR, conversation id, and a chunk count equal to the consumed chunks (`callback`) or between them and `max_tokens` (`pull`, `push`); the binding saw it end without a result. |
| `lifecycle.final` | Flush and shutdown ok; after shutdown nothing pending, dropped or failed, `accepted = exported` for every signal. |
| `direct.no_content` | No stored span in T has content (no `content.values`, no content keys) and no answer for T contains the secret or a prompt. |
| `direct.quality` | Feedback and evaluation returned `false`; staging stores no evaluation for T. |
| `direct.delivery` | Only traces were exported (logs and metrics zero), and the exported span count equals the stored span count of T. |
| Interop checks | `collector.*`, `sinks.routing`, `ids.*`, `mapping.*`, `content`, `content.redaction`, `timing.*`, `evaluations` from `observability_interop/expectations.py`, with the Collector's tee as ground truth. |
| `collector.quality` | Both calls returned `true`; staging links exactly the two records to the run span with the sent names, scores and labels; the feedback comment is redacted. |
| `collector.metric_ledger` | The sink holds one `gen_ai.client.inference.duration` measurement per run of the session, one of them `error.type = cancelled`. |

Browser (`web-mac`): no Collector; its second session is `direct-logs`
(traces and logs straight to staging, content captured with the literal
redaction), checked by `content.redacted`, `quality.linked` and
`direct_logs.delivery` instead of the interop and metric-ledger checks.

### Rust-only cases (`rust-mac`)

Five more sessions, each exporting to the Collector under its own T and P,
cover what the foreign surface has no entry for. The shared sessions'
stopped stream is already the token callback returning an error (delivery
`callback`: the span counts exactly the accepted chunks).

| Session | What runs | Check |
|---|---|---|
| `case-pipeline` | a one-stage streaming LLM pipeline (no `target:`, no provider) whose `on_token` fails after two chunks | `case.pipeline_on_token_failure`: the call fails; the stage's request is `cancelled` with exactly two chunks |
| `case-tts` | `kitten-tts-nano-0.8`, a three-chunk text streamed to completion, then stopped by the chunk callback after one | `case.tts_streaming`: two `text_to_speech` requests, completed with the delivered chunk count, stopped `cancelled` after one |
| `case-asr` | `whisper-tiny-ggml`: `integration-tests/fixtures/input/jfk.wav` as one batch run, then fed second by second to a live stream and flushed | `case.asr_batch_and_live`: a `speech_to_text` request, and one `invoke_workflow asr_session` with decoder runs, all without error |
| `case-frames` | five runs of one frame session with a 10-minute live trace interval | `case.frame_session`: one traced run (the only returned reference), five `gen_ai.client.inference.duration` measurements in the sink |
| `case-fallback` | a local stream aborted by a staged memory-pressure signal and finished by the staging gateway (`gpt-4o-mini`) | `case.cloud_fallback`: one counted request whose `attempt 0` is local and ended `aborted_for_cloud_fallback`, whose `attempt 1` is remote with reason `fallback`, and a gateway span under that remote attempt |

### CLI lanes

`cli-mac` runs the candidate CLI as customers do (key in `XYBRID_API_KEY`,
`--platform-url https://staging.api.xybrid.dev --observability`): one local
LLM run, one local pipeline and one cloud stage, each with `--trace-export`.
`run_staging.py export-check` reads every exported trace back from staging
with the export as the ground truth (the interop `ids.*`, `mapping.*`,
`content`, `timing.*` checks), checks the `cli` binding and service, and the
aggregate. `cli-legacy-mac` runs the released v0.17.0 CLI, which has no OTLP
export and posts legacy events to `<platform-url>/v1/events/batch`; staging
serves that route on the ingest host only, so that host is its platform URL.

Per launch: `aggregate.stats` (the change in `GET /v1/telemetry/stats`
`total_executions_24h` equals the counted request spans stored in this
launch's traces; exact because the project receives nothing else),
`deployment.unchanged` (the platform's last staging deployments are the same
before and after), `evidence.secrets`.

Cross-tenant (`run_staging.py isolation`, run where tenant A's and B's keys
live): every T read with another organization's key (A) and with another
project's key in the same organization (B) answers 404 for spans,
evaluations and content; a span B writes under the same T stays visible only
to B, and `recheck` on the host shows the launch's span set unchanged.

## Running a lane

```bash
python3 tools/scripts/observability_staging/run_staging.py run --lane python-mac -- \
  python3 bindings/python/tests/observability_staging.py --config {config} --expectations {expectations}
```

The command's arguments may use `{config}`, `{expectations}`, `{out_dir}`,
`{lane}`, `{run}`, `{collector_port}` and `{repo}`; its environment carries
the same as `XYBRID_STAGING_*`. `lanes.sh <lane>` wraps each lane's build,
copy and launch. Reports go to
`~/xybrid-acceptance-evidence/<UTC date>/step6/lanes/<lane>/<run>/`
(`--out-root`): `report.json`, `report.md`, `expectations.json`, every staging
answer under `raw/`, the Collector's tee and sink captures, its log and
metrics, and the lane's own log.
