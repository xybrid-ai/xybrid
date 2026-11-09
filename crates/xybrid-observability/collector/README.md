# Collector profiles

OpenTelemetry Collector configurations that take the canonical telemetry the
Xybrid packages and SDKs export (`gen_ai.*` plus `xybrid.genai.*`, see
[`CONTRACT.md`](../CONTRACT.md)) and deliver it to a destination. Instrumentation
stays canonical; any destination dialect is applied here, at export time.

| File | Use |
|---|---|
| [`xybrid.yaml`](xybrid.yaml) | Traces to Xybrid, and `gen_ai.evaluation.result` records (evaluations, user feedback) to Xybrid's `/v1/logs`. Other log events and metrics are discarded unless `sinks.yaml` is added. |
| [`langfuse.yaml`](langfuse.yaml) | Traces to Langfuse, mapped to Langfuse observation types, input/output, level and session. |
| [`phoenix.yaml`](phoenix.yaml) | Traces to Arize Phoenix, mapped to OpenInference attributes. No `Authorization` header. |
| [`phoenix-auth.yaml`](phoenix-auth.yaml) | Overlay for `phoenix.yaml`: adds `PHOENIX_API_KEY` for Phoenix Cloud or an authenticated server. |
| [`sinks.yaml`](sinks.yaml) | Overlay for any profile: sends metrics and the remaining log events to your own OTLP stores (`METRICS_OTLP_ENDPOINT`, `LOGS_OTLP_ENDPOINT`). |

Each profile lists its environment variables at the top. Overlays are added
with more `--config` flags; the Collector merges them:

```bash
otelcol-contrib --config xybrid.yaml
otelcol-contrib --config langfuse.yaml --config sinks.yaml
otelcol-contrib --config phoenix.yaml --config phoenix-auth.yaml
```

Applications export OTLP/HTTP to the Collector (`http://<collector>:4318`), for
example with `OtlpConfig::collector` (Rust) or `createOtlpPipeline` (JS), or the
`otlp(...)` configuration of a native SDK.

## Pinned Collector

The profiles need the contrib distribution (the `transform` and `filter`
processors) and are validated with:

| | |
|---|---|
| Distribution | `otelcol-contrib` |
| Version | [v0.165.0](https://github.com/open-telemetry/opentelemetry-collector-releases/releases/tag/v0.165.0) (2026-11-07) |
| `otelcol-contrib_0.165.0_darwin_arm64.tar.gz` | sha256 `f9a0ed310e9beeb0fabca9d9c24bd0d3467f6339e3207710dbd66a6adf1298f6` |
| `otelcol-contrib_0.165.0_linux_amd64.tar.gz` | sha256 `bd6e73d94bf3359ad6fac25529a8ee308adb9b0b8a0e8af034ed3891fbe5aa33` |

The checksums are the release's own `.sha256` assets. Check every profile and
overlay combination before deploying one:

```bash
otelcol-contrib validate --config xybrid.yaml --config sinks.yaml
```

Moving to a newer Collector means re-running that validation for every
combination in the table above and updating this pin.
