# Observability interop harness

Runs real producers through the pinned OpenTelemetry Collector into Langfuse,
Phoenix or Xybrid, reads each destination back through its own API, and checks
it against what the Collector received. This is the acceptance for the
Collector profiles in
[`crates/xybrid-observability/collector/`](../../../crates/xybrid-observability/collector/).
It covers IDs and parents, model and provider, usage, errors, content and
redaction, first chunk versus first token, and linked quality results.

```text
 rust example ──┐                         ┌─ destination profile ──► Langfuse / Phoenix / Xybrid staging
 js example ────┼─► otelcol-contrib 0.165 ─┼─ sinks.yaml ──────────► capture server (sinks.jsonl)
 js no_content ─┤    127.0.0.1:4318        └─ harness tee ─────────► capture server (ground-truth.jsonl)
 python bridge ─┘                                                     │
                     destination API readback ◄── compared with ──────┘
```

The tee copies every span, log record and metric point before any mapping, so
the ground truth is exactly what the producers sent. Every destination check
compares the backend's answer with it, span by span, under the canonical IDs.

## Files

| File | Role |
|---|---|
| `run_interop.py` | The runner: Collector, capture servers, workloads, readback, checks, `report.json` + `report.md`. |
| `otelcol.py` | Pinned Collector: download, sha256 check, cache, `validate`, start/stop, its own metrics. |
| `collector.lock.json` | otelcol-contrib version and per-platform archive sha256 (kept equal to `collector/README.md` by a test). |
| `collector-harness.yaml` | Overlay added after the profile: loopback receiver, the ground-truth tee, Collector self-metrics on `127.0.0.1:18888`. |
| `collector-sinks-json.yaml` | Overlay for `sinks.yaml`: its two exporters send OTLP/JSON, so the harness's capture server can be the sink. |
| `workloads.py` | The producers and how each reports its references. |
| `python_quality.py` | The Python workload's second process: evaluation + feedback on the scenario's stored inference reference. |
| `destinations.py` | API clients and normalizers (Langfuse, Phoenix, Xybrid, and `--wire` output). |
| `expectations.py` | `DESTINATIONS`, the per-backend table, and every check. |
| `compose.yaml`, `interop.env` | Pinned self-hosted Langfuse and Phoenix, fixed loopback ports, local-only credentials. |
| `up.sh`, `down.sh` | Start (and wait for) / stop the destinations. Bash 3.2. |
| `fixtures/` | Recorded ground truth and profile output, and API responses for the unit tests (see Fixtures). |

Python 3.9+ standard library only (the Python *workload* needs the SDK's
interpreter, 3.10+). Unit tests:
`python3 -m unittest tools/scripts/tests/test_observability_interop.py`.

## Workloads

| Name | Program | Export | Content | Quality results |
|---|---|---|---|---|
| `rust` | `crates/xybrid-observability/examples/standalone.rs` | http/protobuf (`OTEL_EXPORTER_OTLP_ENDPOINT`) | span + event; `sk-demo-123` → `[KEY]` | evaluation + feedback |
| `js` | `bindings/observability-js/examples/node-basic.ts` (Node) | http/protobuf | default (span only); `Paris` → `[CITY]` | evaluation + feedback |
| `js-metadata` | the same with `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT=no_content` | http/protobuf | none | evaluation + feedback (no comment) |
| `python` | `bindings/python/tests/observability_scenario.py` on the compiled bridge (the shared [runtime scenario](../observability_capture/scenario.md)), then `python_quality.py` | http/json, then http/protobuf | literal rule + omit rule | evaluation + feedback |

Each prints an `observation refs: {...}` line (the Python scenario writes its
references to `expectations.json`); the harness checks that every reported
reference reached the Collector and can be looked up at the destination. Each
workload's resource carries `xybrid.interop.workload=<name>` (through
`OTEL_RESOURCE_ATTRIBUTES`), which splits the ground truth per workload. The
Python workload's capture also runs the scenario's own checker
(`check_capture.py`), so the native producer's export is checked after the
Collector as well as at the destination. Its trace id is the scenario's fixed
`78793a707974686f6e00000000000000`: a backend that kept earlier runs returns
their spans too, and the checks ignore every span the run did not produce.

## Running on a Mac

Once per checkout (producers):

```bash
cargo build -p xybrid-observability --example standalone
(cd bindings/observability-js && pnpm install --frozen-lockfile && pnpm exec tsdown)
cargo install boltffi_cli --version 0.30.1 --locked     # if `boltffi` is missing
./tools/scripts/build-python-bolt.sh                     # the Python bridge, for the python3 on PATH
python3 -c 'import sys; print(sys.executable)'           # pass this interpreter as --python
```

The JS example runs on Node with type stripping (22.18+ or 23.6+ as is,
22.6+ with the flag the harness adds); `--js-runtime bun` uses Bun instead.

Check every Collector profile combination against the pinned release (the
first run downloads `otelcol-contrib_0.165.0_darwin_arm64.tar.gz`, verifies
sha256 `f9a0ed31…98f6` and caches the binary in `~/.cache/xybrid/otelcol`):

```bash
python3 tools/scripts/observability_interop/run_interop.py --validate-profiles
```

Phoenix and Langfuse (self-hosted, OrbStack):

```bash
orbctl start
tools/scripts/observability_interop/up.sh phoenix
python3 tools/scripts/observability_interop/run_interop.py --destination phoenix --python "$(which python3)"

tools/scripts/observability_interop/up.sh langfuse      # first start migrates; up to ~5 min
python3 tools/scripts/observability_interop/run_interop.py --destination langfuse --python "$(which python3)"

tools/scripts/observability_interop/down.sh --volumes   # between acceptance runs
```

Xybrid staging (hosted; read-only API calls plus the Collector's export):

```bash
XYBRID_STAGING_API_KEY=sk_... python3 tools/scripts/observability_interop/run_interop.py \
  --destination xybrid --python "$(which python3)"
```

`up.sh` is `docker compose -f compose.yaml --env-file interop.env --profile
<phoenix|langfuse> up -d --wait` followed by a wait for the HTTP API;
`down.sh` is the matching `down --remove-orphans [--volumes]`.

`--destination all` runs the three in turn (Xybrid only when the key is
set). `--workloads rust,js` selects producers. Reports go to
`$TMPDIR/xybrid-interop/<UTC time>/<destination>/` unless `--out-dir` says
otherwise. Exit status: 0 all checks passed, 1 a check failed, 2 the run could
not start (the message says what is missing).

Without any backend, `--wire` points the profile's exporter at a local capture
server (OTLP/JSON instead of the profile's protobuf) and runs the same table
against what the profile would send; this needs no Docker and no key:

```bash
python3 tools/scripts/observability_interop/run_interop.py --destination langfuse --wire
```

### Never production

The harness only talks to loopback hosts and, for `--destination xybrid`,
to `staging.*.xybrid.dev`: `XYBRID_STAGING_INGEST_URL` and
`XYBRID_STAGING_API_URL` (from `interop.env` or the environment) are refused
before any request unless they are a staging or loopback host. `xybrid.yaml`'s
own defaults (production ingest) are never used: the harness always sets both
endpoints. Without `XYBRID_STAGING_API_KEY` the Xybrid destination does not
run. The producers never see a hosted key: each workload step runs without
`OTEL_*`, `XYBRID_API_KEY`, `XYBRID_STAGING_API_KEY` and
`XYBRID_PLATFORM_API_KEY`, in its own process group, so a
`--workload-timeout` kills what it started too (`cargo run` and its example).

## Destinations

`compose.yaml` pins every image by tag and index digest:

| Service | Image |
|---|---|
| Phoenix | `arizephoenix/phoenix:version-21.0.0@sha256:1f9a144a…baad`, `PHOENIX_ENABLE_AUTH=false` |
| Langfuse web / worker | `langfuse/langfuse:4.77.0@sha256:2b47d0ae…b6a0`, `langfuse/langfuse-worker:4.77.0@sha256:1bc3597f…9438` |
| ClickHouse | `clickhouse/clickhouse-server:26.3.62.3@sha256:de3e47c6…e40a` |
| Postgres | `postgres:17.11@sha256:ff871389…b4f` |
| Redis | `redis:7.4.12@sha256:2e3eb114…2a90` |
| MinIO | `cgr.dev/chainguard/minio:latest@sha256:616286de…ce76` |

Langfuse is the six-service self-hosted stack of Langfuse's own compose file,
on the current release line (v4: upstream's compose defaults to it, and
`langfuse.yaml`'s `x-langfuse-ingestion-version: 4` header selects v4's
real-time OTel path). The comment at the top of `compose.yaml` gives the
`LANGFUSE_WEB_IMAGE` / `LANGFUSE_WORKER_IMAGE` digests of the last v3 release
(3.227.0); the readback falls back to v3's APIs on a 404. Langfuse needs about
4 GB of OrbStack's memory, Phoenix well under 1 GB.

| Setting (`interop.env`) | Value |
|---|---|
| Collector OTLP/HTTP | `127.0.0.1:4318` (`--collector-port` overrides) |
| Collector self-metrics | `127.0.0.1:18888` |
| Phoenix UI + API + OTLP | `http://127.0.0.1:16006`, project `default` |
| Langfuse UI + API + OTLP | `http://127.0.0.1:13000`, login `interop@xybrid.invalid` / `xybrid-interop-local` |
| Langfuse project keys | `pk-lf-xybrid-interop-local` / `sk-lf-xybrid-interop-local` (headless init) |

Every port is published on loopback only; databases are not published.

## What each destination keeps

The table is `DESTINATIONS` in `expectations.py`; `report.md` prints the one
that applied. Roles are the platform's own classification (request operations,
then attempts, then workflow/tool/load).

| | Langfuse | Phoenix | Xybrid |
|---|---|---|---|
| IDs | observation id = span id, `parentObservationId` | `context.trace_id/span_id`, `parent_id` | `trace_id/span_id/parent_span_id` |
| Logical inference | `GENERATION` | `LLM` | `genai.request`, counted |
| Attempt | `SPAN` (one generation per request), `usageDetails` `{}` | `CHAIN`, no `llm.token_count` | `genai.attempt`, own usage, not counted |
| Tool / workflow / load | `TOOL` / `CHAIN` / `SPAN` | `TOOL` / `CHAIN` / `CHAIN` | `genai.tool` / `genai.workflow` / `genai.load_model` |
| Model | response model first (Langfuse's own order) | response model first | both, typed |
| Provider | `metadata.provider` (the profile's key only) | `llm.provider` | `genai.provider_name` |
| Usage | `usageDetails` on the generation; Langfuse sums every observation's, so attempts carry none | `llm.token_count.*` on LLM only | typed tokens |
| Error | `level=ERROR`, `statusMessage` | `status_code`, `status_message` | status + `genai.error_type` |
| Content | `input` / `output` | `input.value` / `output.value` | `content.values` or the `/content` route |
| First chunk / first token | two `metadata.attributes`, `completionStartTime` unset | two attributes | two typed columns |
| Evaluations, feedback | **dropped** (no OTLP logs) | **dropped** (no OTLP logs) | linked via `/v1/logs` → `GET …/evaluations` |
| Metrics | dropped (nop) unless `sinks.yaml` | same | same |

Every run also adds `sinks.yaml` (unless `--no-sinks`), pointed at a capture
server, and checks it received every metric point and every log record the
profile does not send to its destination (for Xybrid: everything but the
evaluation results).

### Checks

Collector stage, per workload: `producer.exit`, `producer.refs_exported`,
`collector.signals` (spans, the two evaluation results, metric points),
`collector.redacted_at_source`, `sinks.routing`, and for Python the 18
`scenario.*` checks of `check_capture.py`. Per run: `collector.config_valid`,
`collector.version`, `collector.export` (the destination exporter sent every
span, nothing failed, no export error in `collector.log`; Xybrid: exactly the
evaluation results went to `/v1/logs`, and unreadable self-metrics fail
instead of skipping, since nothing else shows that in API mode).

Destination stage, per workload: `readback` (every read's last attempt
succeeded: a refused, reset or timed-out read, or an answer that is not JSON,
fails here, and a failed quality-result read marks `evaluations` as not read
rather than "dropped, as expected"), `ids.spans`, `ids.parents`,
`ids.remote_parent` (the Python workflow's caller span P, never exported),
`ids.producer_refs`, `mapping.kind`, `mapping.model_provider`,
`mapping.usage` (equal, absent where the table says so, Xybrid's counted
flag, which must be present), `mapping.errors`, `content` (present and equal
to the producer's, or absent for `js-metadata`), `content.redaction` (no
redacted or metadata-only string in what the destination answered for this
run's own spans; earlier runs under the same trace id are ignored), `timing.first_chunk_vs_first_token` (each value
under its own field, equal to the producer's; a completion start equal to the
first chunk fails), `evaluations` (linked with name, score, label, source (required
where the table sets `evaluation_source`) and comment, or dropped as the
table says). `SKIP` means the workload has nothing
to check there (no remote parent, no usage, no streaming timing).

### To confirm on the first run against each backend

The normalizers follow each API's published schema; the fixtures were shaped
from those schemas, not recorded. Points a first live run settles (a
mismatch fails a named check, and `raw/` holds the answer to fix the
normalizer or the table with):

- Langfuse v4 `metadata` layout (the timing attributes are searched flat,
  nested, prefixed or JSON-encoded) and whether it keeps a
  `parentObservationId` that names a span it never received (`ids.remote_parent`).
- Langfuse stores the explicit empty `langfuse.observation.usage_details` as
  an empty `usageDetails` on attempts (so the trace's usage total counts the
  request once) and names the Rust generation after its response model
  `tiny-chat-q4` (`extractModelName` in 4.77.0 checks the response model
  before the request model).
- Phoenix 21 attribute form in the REST answer (flat or nested: both read).
- Xybrid staging stores a feedback comment exactly as sent (the platform
  scrubs secrets; the comments used contain none).

## Reports

`<out>/<destination>/`:

| File | Content |
|---|---|
| `report.json` | Schema `xybrid-observability-interop/1`: producers (git sha, dirty flag, crate and package versions, runtimes), Collector (version, platform, archive sha256, configs, exporter counters), destination (product, version, container image ids), the expectation table, every check with `pass`/`fail`/`skip`, failures and notes. |
| `report.md` | The same for people. |
| `ground-truth.jsonl`, `<workload>/ground-truth.jsonl` | What the Collector received, all and per workload. |
| `sinks.jsonl`, `wire.jsonl` | What `sinks.yaml` (and, with `--wire`, the profile) sent. |
| `raw/` | Every API answer, by label (for fixture updates). |
| `collector.log`, `collector-metrics.txt`, `<workload>/step*.log` | Process output. |

## Fixtures

- `fixtures/ground-truth/`: a recorded run (2026-11-09, Linux, otelcol-contrib
  0.165.0, `--destination xybrid --wire`): each workload's capture as the
  Collector received it, the Python scenario's `expectations.json` and the
  reported references.
- `fixtures/wire/`: what the profiles sent in that run (`xybrid.jsonl`) and in
  the matching Langfuse and Phoenix runs (Rust workload, with its own ground truth).
- `fixtures/api/`: Langfuse v2 observations / v3 scores, Phoenix spans /
  span annotations and Xybrid spans / content / evaluations for the Rust and
  Python workloads, built from the ground truth above in each API's published
  shape. Replace them with `raw/` answers from a live run when one differs.
