// Configuration boundaries: header isolation and merging, the environment constructor, resource
// variables, limit validation before any timer (R-CFG-1..5), and the runtime SDK version check.
import { afterEach, describe, expect, spyOn, test } from "bun:test";
import { fileURLToPath } from "node:url";
import { SDK_INFO } from "@opentelemetry/core";
import { DEFAULTS } from "../src/catalog.ts";
import { resolveOtlpOptions } from "../src/otlp/config.ts";
import {
  createOtlpExporters,
  createOtlpPipeline,
  OtlpLogExporter,
  OtlpMetricExporter,
  type OtlpPipeline,
  type OtlpPipelineOptions,
  OtlpTraceExporter,
  otlpConfigFromEnv,
  xybridTracesPreset,
} from "../src/otlp/index.ts";
import { attribute, decodeRequest } from "./support/otlp-decode.ts";
import { startServer, type TestServer } from "./support/otlp-server.ts";

const request = { operation: "chat", provider: "p", model: "m" } as const;

const cleanups: (() => Promise<unknown>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0)) await cleanup();
});
const track = (pipeline: OtlpPipeline, server?: TestServer): OtlpPipeline => {
  cleanups.push(async () => {
    await pipeline.shutdown(1000);
    await server?.close();
  });
  return pipeline;
};

/** Sets process.env variables for the duration of `run`. */
const withProcessEnv = async <T>(
  values: Readonly<Record<string, string>>,
  run: () => Promise<T>,
): Promise<T> => {
  const previous = new Map(Object.keys(values).map((key) => [key, process.env[key]]));
  Object.assign(process.env, values);
  try {
    return await run();
  } finally {
    for (const [key, value] of previous) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
};

/** Headers `fetch` itself adds; everything else came from the pipeline. */
const FETCH_OWNED = new Set([
  "host",
  "content-length",
  "connection",
  "accept-encoding",
  "accept-language",
  "user-agent",
  "sec-fetch-mode",
]);
const sentHeaders = (headers: Readonly<Record<string, string>>): Record<string, string> =>
  Object.fromEntries(
    Object.entries(headers).filter(
      ([name, value]) => !FETCH_OWNED.has(name) && !(name === "accept" && value === "*/*"),
    ),
  );

describe("header isolation and merging (R-CFG-1, R-CFG-2)", () => {
  test("the Xybrid preset sends only its bearer header whatever the environment says", async () => {
    const server = await startServer();
    await withProcessEnv(
      {
        OTEL_EXPORTER_OTLP_HEADERS: "Authorization=Basic%20x,x-honeycomb-team=k",
        OTEL_EXPORTER_OTLP_TRACES_HEADERS: "x-other=y,Accept=application%2Fprivate",
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: "http://127.0.0.1:1/elsewhere",
        OTEL_EXPORTER_OTLP_PROTOCOL: "http/json",
        OTEL_EXPORTER_OTLP_TIMEOUT: "1",
      },
      async () => {
        const pipeline = track(
          createOtlpPipeline(
            xybridTracesPreset("xk_live", { endpoint: `${server.url}/v1/traces` }),
          ),
          server,
        );
        pipeline.observer().startInference(request).end();
        expect(await pipeline.forceFlush(3000)).toBe(true);
      },
    );
    expect(server.requests).toHaveLength(1);
    expect(sentHeaders(server.requests[0]?.headers ?? {})).toEqual({
      authorization: "Bearer xk_live",
      "content-type": "application/x-protobuf",
    });
  });

  test("generic endpoints merge env, signal env and programmatic headers case-insensitively", async () => {
    const server = await startServer();
    const environment = {
      OTEL_EXPORTER_OTLP_HEADERS: "a=1,b=2,Shared=env,content-type=text%2Fplain",
      OTEL_EXPORTER_OTLP_TRACES_HEADERS: "B=3,c=4",
      OTEL_EXPORTER_OTLP_LOGS_HEADERS: "logs-only=1",
    };
    const resolved = resolveOtlpOptions({
      traces: { endpoint: `${server.url}/v1/traces`, headers: { SHARED: "prog", d: "5" } },
      metrics: {
        endpoint: `${server.url}/v1/metrics`,
        headers: { only: "me" },
        headerPolicy: "isolated",
      },
      environment,
    });
    expect(resolved.traces?.headers).toEqual({
      a: "1",
      B: "3",
      SHARED: "prog",
      c: "4",
      d: "5",
      "Content-Type": "application/x-protobuf",
    });
    expect(resolved.metrics?.headers).toEqual({
      only: "me",
      "Content-Type": "application/x-protobuf",
    });
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, headers: { SHARED: "prog", d: "5" } },
        environment,
      }),
      server,
    );
    pipeline.observer().startInference(request).end();
    expect(await pipeline.forceFlush(3000)).toBe(true);
    expect(sentHeaders(server.requests[0]?.headers ?? {})).toEqual({
      a: "1",
      b: "3",
      shared: "prog",
      c: "4",
      d: "5",
      "content-type": "application/x-protobuf",
    });
  });

  test("environment: null ignores the process environment; process.env is read when undefined", async () => {
    await withProcessEnv({ OTEL_EXPORTER_OTLP_HEADERS: "from-env=1" }, async () => {
      const endpoint = "http://127.0.0.1:1/v1/traces";
      expect(resolveOtlpOptions({ traces: { endpoint } }).traces?.headers["from-env"]).toBe("1");
      expect(
        resolveOtlpOptions({ traces: { endpoint }, environment: null }).traces?.headers["from-env"],
      ).toBeUndefined();
      const exporters = createOtlpExporters({ traces: { endpoint }, environment: null });
      expect(exporters.traces).toBeInstanceOf(OtlpTraceExporter);
    });
  });
});

describe("otlpConfigFromEnv (R-CFG-3)", () => {
  const shared = {
    OTEL_EXPORTER_OTLP_ENDPOINT: "http://collector:4318/",
    OTEL_EXPORTER_OTLP_HEADERS: "api-key=abc%20def,bad,=x,k=%zz,ok=1",
    OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: " http://logs:4318/custom ",
    OTEL_EXPORTER_OTLP_METRICS_PROTOCOL: " HTTP/JSON ",
    OTEL_EXPORTER_OTLP_PROTOCOL: "grpc",
    OTEL_EXPORTER_OTLP_TRACES_TIMEOUT: "2500",
    OTEL_EXPORTER_OTLP_TIMEOUT: "abc",
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: "",
    OTEL_SERVICE_NAME: "svc",
    OTEL_RESOURCE_ATTRIBUTES: "deployment.environment=dev",
    OTEL_BSP_MAX_QUEUE_SIZE: "100",
    OTEL_BSP_MAX_EXPORT_BATCH_SIZE: "0",
    OTEL_BSP_SCHEDULE_DELAY: "250",
    OTEL_BSP_EXPORT_TIMEOUT: "   ",
    OTEL_METRIC_EXPORT_INTERVAL: "5000",
  };

  test("endpoints, protocols, timeouts, batching and the metric interval", () => {
    const config = otlpConfigFromEnv(shared);
    expect(config.traces).toEqual({
      endpoint: "http://collector:4318/v1/traces",
      protocol: "http/protobuf",
      timeoutMs: 2500,
    });
    // An empty signal endpoint is unset, so the base applies; protocol is case-insensitive.
    expect(config.metrics).toEqual({
      endpoint: "http://collector:4318/v1/metrics",
      protocol: "http/json",
    });
    expect(config.logs).toEqual({ endpoint: "http://logs:4318/custom", protocol: "http/protobuf" });
    expect(config.batch).toEqual({ maxQueueSize: 100, scheduledDelayMs: 250 });
    expect(config.metricExportIntervalMs).toBe(5000);
    expect(config.environment).toBe(shared);
  });

  test("headers and resource variables apply at build, percent-decoded, malformed pairs skipped", () => {
    const resolved = resolveOtlpOptions(otlpConfigFromEnv(shared));
    expect(resolved.traces?.headers).toEqual({
      "api-key": "abc def",
      ok: "1",
      "Content-Type": "application/x-protobuf",
    });
    expect(resolved.metrics?.headers["Content-Type"]).toBe("application/json");
    expect(resolved.resourceAttributes).toEqual({
      "deployment.environment": "dev",
      "service.name": "svc",
    });
    expect(resolved.limits.maxQueueSize).toBe(100);
    expect(resolved.limits.metricExportTimeoutMs).toBe(5000);
  });

  test("OTEL_{SIGNAL}_EXPORTER containing none skips a signal", () => {
    const config = otlpConfigFromEnv({
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://collector:4318",
      OTEL_METRICS_EXPORTER: "none",
      OTEL_LOGS_EXPORTER: "otlp, NONE",
      OTEL_TRACES_EXPORTER: "otlp",
    });
    expect(Object.keys(config)).toEqual(["traces", "environment"]);
  });

  test("nothing configured, or OTEL_SDK_DISABLED=true in any case, means no signals", () => {
    expect(otlpConfigFromEnv({})).toEqual({ environment: {} });
    for (const disabled of ["true", "TRUE", " true "]) {
      expect(
        otlpConfigFromEnv({
          OTEL_SDK_DISABLED: disabled,
          OTEL_EXPORTER_OTLP_ENDPOINT: "http://collector:4318",
        }),
      ).toEqual({});
    }
    const enabled = otlpConfigFromEnv({
      OTEL_SDK_DISABLED: "false",
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://collector:4318",
    });
    expect(Object.keys(enabled).sort()).toEqual(["environment", "logs", "metrics", "traces"]);
  });

  test("a pipeline from otlpConfigFromEnv({}) has no providers", () => {
    const pipeline = track(createOtlpPipeline(otlpConfigFromEnv({ OTEL_SDK_DISABLED: "true" })));
    expect(pipeline.tracerProvider).toBeUndefined();
    expect(pipeline.meterProvider).toBeUndefined();
    expect(pipeline.loggerProvider).toBeUndefined();
  });
});

describe("resource variables (R-CFG-4)", () => {
  const resourceOf = (server: TestServer) => {
    const received = server.received("/v1/traces")[0];
    if (received === undefined) throw new Error("no trace request");
    const decoded = decodeRequest("traces", received.headers["content-type"], received.body);
    return decoded.resourceSpans[0].resource.attributes;
  };

  test("programmatic resource values win over OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES", async () => {
    const server = await startServer();
    const pipeline = track(
      createOtlpPipeline({
        traces: { endpoint: `${server.url}/v1/traces`, protocol: "http/json" },
        resource: { serviceName: "prog", attributes: { team: "prog-team" } },
        environment: {
          OTEL_SERVICE_NAME: "env-svc",
          OTEL_RESOURCE_ATTRIBUTES: "deployment.environment=dev,service.name=attr,team=a%20b,bad",
        },
      }),
      server,
    );
    pipeline.observer().startInference(request).end();
    expect(await pipeline.forceFlush(3000)).toBe(true);
    const attributes = resourceOf(server);
    expect(attribute(attributes, "service.name")).toEqual({ stringValue: "prog" });
    expect(attribute(attributes, "team")).toEqual({ stringValue: "prog-team" });
    expect(attribute(attributes, "deployment.environment")).toEqual({ stringValue: "dev" });
  });

  test("OTEL_SERVICE_NAME beats service.name in OTEL_RESOURCE_ATTRIBUTES", () => {
    const endpoint = "http://127.0.0.1:1/v1/traces";
    expect(
      resolveOtlpOptions({
        traces: { endpoint },
        environment: { OTEL_SERVICE_NAME: "env", OTEL_RESOURCE_ATTRIBUTES: "service.name=attr" },
      }).resourceAttributes,
    ).toEqual({ "service.name": "env" });
    expect(
      resolveOtlpOptions({
        traces: { endpoint },
        environment: { OTEL_RESOURCE_ATTRIBUTES: "service.name=attr" },
      }).resourceAttributes,
    ).toEqual({ "service.name": "attr" });
  });

  test("the Xybrid preset applies resource variables from the process environment", async () => {
    const server = await startServer();
    await withProcessEnv(
      { OTEL_SERVICE_NAME: "preset-svc", OTEL_RESOURCE_ATTRIBUTES: "region=eu" },
      async () => {
        const pipeline = track(
          createOtlpPipeline(xybridTracesPreset("k", { endpoint: `${server.url}/v1/traces` })),
          server,
        );
        pipeline.observer().startInference(request).end();
        expect(await pipeline.forceFlush(3000)).toBe(true);
      },
    );
    const attributes = resourceOf(server);
    expect(attribute(attributes, "service.name")).toEqual({ stringValue: "preset-svc" });
    expect(attribute(attributes, "region")).toEqual({ stringValue: "eu" });
  });
});

describe("limits are validated before any timer (R-CFG-5)", () => {
  const endpoint = "http://127.0.0.1:1/v1/traces";
  const bad = [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, "10", null] as const;
  /** Durations must also fit a timer. */
  const badMs = [...bad, 2 ** 31] as const;
  const cases: [string, (value: unknown) => OtlpPipelineOptions, readonly unknown[]][] = [
    ["batch.maxQueueSize", (value) => ({ batch: { maxQueueSize: value as number } }), bad],
    [
      "batch.maxExportBatchSize",
      (value) => ({ batch: { maxExportBatchSize: value as number } }),
      bad,
    ],
    [
      "batch.scheduledDelayMs",
      (value) => ({ batch: { scheduledDelayMs: value as number } }),
      badMs,
    ],
    ["batch.exportTimeoutMs", (value) => ({ batch: { exportTimeoutMs: value as number } }), badMs],
    ["batch.maxBufferedBytes", (value) => ({ batch: { maxBufferedBytes: value as number } }), bad],
    ["traces.timeoutMs", (value) => ({ traces: { endpoint, timeoutMs: value as number } }), badMs],
    [
      "retry.maxRetries",
      (value) => ({ retry: { maxRetries: value as number } }),
      [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, "1", null],
    ],
    [
      "retry.initialDelayMs",
      (value) => ({ retry: { initialDelayMs: value as number } }),
      [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, null],
    ],
    [
      "retry.maxDelayMs",
      (value) => ({ retry: { maxDelayMs: value as number } }),
      [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, null, 100],
    ],
    ["maxRequestBodyBytes", (value) => ({ maxRequestBodyBytes: value as number }), bad],
    ["metricExportIntervalMs", (value) => ({ metricExportIntervalMs: value as number }), badMs],
  ];

  test("every invalid numeric option throws RangeError naming it, creating no timer or controller", () => {
    const timeouts = spyOn(globalThis, "setTimeout");
    const intervals = spyOn(globalThis, "setInterval");
    const OriginalAbortController = globalThis.AbortController;
    let controllers = 0;
    globalThis.AbortController = class extends OriginalAbortController {
      constructor() {
        super();
        controllers += 1;
      }
    };
    try {
      for (const [name, build, values] of cases) {
        for (const value of values) {
          const options: OtlpPipelineOptions = {
            traces: { endpoint },
            metrics: { endpoint: "http://127.0.0.1:1/v1/metrics" },
            logs: { endpoint: "http://127.0.0.1:1/v1/logs" },
            ...(name === "retry.maxDelayMs" && value === 100
              ? { retry: { initialDelayMs: 200, maxDelayMs: 100 } }
              : build(value)),
          };
          for (const construct of [
            () => createOtlpPipeline(options),
            () => createOtlpExporters(options),
            () => new OtlpTraceExporter(options),
          ]) {
            let error: unknown;
            try {
              construct();
            } catch (caught) {
              error = caught;
            }
            expect(error).toBeInstanceOf(RangeError);
            expect((error as Error).message).toContain(name);
          }
        }
      }
      expect(timeouts).not.toHaveBeenCalled();
      expect(intervals).not.toHaveBeenCalled();
      expect(controllers).toBe(0);
    } finally {
      timeouts.mockRestore();
      intervals.mockRestore();
      globalThis.AbortController = OriginalAbortController;
    }
  });

  test("bad endpoints, protocols, header policies and headers throw TypeError", () => {
    const invalid: OtlpPipelineOptions[] = [
      { traces: { endpoint: "not a url" } },
      { traces: { endpoint: "/v1/traces" } },
      { traces: { endpoint: "ftp://collector/v1/traces" } },
      { traces: { endpoint: "http://user:secret@collector/v1/traces" } },
      { traces: { endpoint: 42 as unknown as string } },
      { traces: { endpoint, protocol: "grpc" as "http/json" } },
      { traces: { endpoint, headerPolicy: "merge" as "isolated" } },
      { traces: { endpoint, headers: { "bad header": "x" } } },
      { traces: { endpoint, headers: { ok: "line\nbreak" } } },
      { traces: { endpoint, headers: { ok: 1 as unknown as string } } },
      { resource: { serviceName: 1 as unknown as string } },
      { environment: "env" as unknown as Record<string, string> },
    ];
    for (const options of invalid) {
      expect(() => createOtlpPipeline(options)).toThrow(TypeError);
    }
    expect(() => new OtlpTraceExporter({ logs: { endpoint } })).toThrow(TypeError);
  });

  test("valid edge values are accepted; the batch size is clamped to the queue size", () => {
    const resolved = resolveOtlpOptions({
      traces: { endpoint },
      batch: { maxQueueSize: 4, maxExportBatchSize: 100 },
      retry: { maxRetries: 0, initialDelayMs: 0, maxDelayMs: 0 },
    });
    expect(resolved.limits.maxExportBatchSize).toBe(4);
    expect(resolved.limits.maxRetries).toBe(0);
    expect(resolved.limits.exportTimeoutMs).toBe(DEFAULTS.batch.export_timeout_ms);
    expect(resolved.limits.maxBufferedBytes).toBe(DEFAULTS.max_buffered_bytes_per_signal);
    expect(resolved.limits.maxRequestBodyBytes).toBe(DEFAULTS.max_request_body_bytes);
  });

  test("a process with an invalid batch size exits instead of hanging (otlp-8)", async () => {
    const source = fileURLToPath(new URL("../src/otlp/index.ts", import.meta.url));
    const script = `
      import { createOtlpPipeline } from ${JSON.stringify(source)};
      for (const maxExportBatchSize of [0, Number.NaN, -1]) {
        try {
          createOtlpPipeline({ traces: { endpoint: "http://127.0.0.1:1/v1/traces" }, batch: { maxExportBatchSize } });
          process.exit(2);
        } catch (error) {
          if (!(error instanceof RangeError)) process.exit(3);
        }
      }
    `;
    const child = Bun.spawn([process.execPath, "-e", script], { stdout: "pipe", stderr: "pipe" });
    const exited = await Promise.race([
      child.exited,
      new Promise<"hung">((resolve) => setTimeout(() => resolve("hung"), 10_000)),
    ]);
    if (exited === "hung") child.kill();
    expect(exited).toBe(0);
  });

  test("lifecycle timeouts that are not finite non-negative numbers use the defaults", async () => {
    const pipeline = track(createOtlpPipeline({ traces: { endpoint } }));
    expect(await pipeline.forceFlush(Number.NaN)).toBe(true);
    expect(await pipeline.forceFlush(-5)).toBe(true);
    expect(await pipeline.shutdown(Number.POSITIVE_INFINITY)).toBe(true);
  });
});

describe("an unsupported OpenTelemetry SDK is rejected at runtime (D33)", () => {
  // npm can accept conflicting optional peers, so the OTLP boundary checks the SDK that
  // actually loaded. `SDK_INFO` is what `@opentelemetry/core` reports for that SDK.
  const VERSION_KEY = "telemetry.sdk.version";
  const options: OtlpPipelineOptions = {
    environment: null,
    traces: { endpoint: "http://127.0.0.1:1/v1/traces" },
    metrics: { endpoint: "http://127.0.0.1:1/v1/metrics" },
    logs: { endpoint: "http://127.0.0.1:1/v1/logs" },
  };
  const withSdkVersion = <T>(version: string, run: () => T): T => {
    const previous = SDK_INFO[VERSION_KEY];
    SDK_INFO[VERSION_KEY] = version;
    try {
      return run();
    } finally {
      SDK_INFO[VERSION_KEY] = previous;
    }
  };

  test("every OTLP entry point throws before any timer or controller exists", () => {
    const timeouts = spyOn(globalThis, "setTimeout");
    const intervals = spyOn(globalThis, "setInterval");
    const OriginalAbortController = globalThis.AbortController;
    let controllers = 0;
    globalThis.AbortController = class extends OriginalAbortController {
      constructor() {
        super();
        controllers += 1;
      }
    };
    try {
      for (const version of ["2.12.0", "2.6.0", "3.2.0", "3.10.0", "4.0.0", "30.0.0", ""]) {
        for (const construct of [
          () => resolveOtlpOptions(options),
          () => createOtlpPipeline(options),
          () => createOtlpExporters(options),
          () => new OtlpTraceExporter(options),
          () => new OtlpLogExporter(options),
          () => new OtlpMetricExporter(options),
        ]) {
          let error: unknown;
          withSdkVersion(version, () => {
            try {
              construct();
            } catch (caught) {
              error = caught;
            }
          });
          expect(error).toBeInstanceOf(TypeError);
          expect((error as Error).message).toBe(
            "@xybrid/observability/otlp requires OpenTelemetry SDK 3.0 or 3.1.",
          );
        }
      }
      expect(timeouts).not.toHaveBeenCalled();
      expect(intervals).not.toHaveBeenCalled();
      expect(controllers).toBe(0);
    } finally {
      timeouts.mockRestore();
      intervals.mockRestore();
      globalThis.AbortController = OriginalAbortController;
    }
  });

  test("SDK 3.0.x and 3.1.x pass the check", () => {
    for (const version of ["3.0.0", "3.0.7", "3.1.0", "3.1.12"]) {
      const resolved = withSdkVersion(version, () => resolveOtlpOptions(options));
      expect(resolved.traces?.url).toBe("http://127.0.0.1:1/v1/traces");
    }
  });
});
