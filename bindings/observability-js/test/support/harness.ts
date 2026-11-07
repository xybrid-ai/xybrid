import { SpanKind, SpanStatusCode } from "@opentelemetry/api";
import {
  InMemoryLogRecordExporter,
  LoggerProvider,
  type ReadableLogRecord,
  SimpleLogRecordProcessor,
} from "@opentelemetry/sdk-logs";
import {
  DataPointType,
  MeterProvider,
  MetricReader,
  type ResourceMetrics,
} from "@opentelemetry/sdk-metrics";
import {
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
  TracerProvider,
} from "@opentelemetry/sdk-trace";
import {
  type ContentOptions,
  INSTRUMENTATION_SCOPE_NAME,
  type LoggerProviderLike,
  Observer,
} from "../../src/index.ts";

export const CLOCK_START_MS = 1_700_000_000_000;

/** A manual epoch-millisecond clock. */
export class ManualClock {
  constructor(public nowMs: number = CLOCK_START_MS) {}
  readonly now = (): number => this.nowMs;
  set(ms: number): void {
    this.nowMs = ms;
  }
  advance(ms: number): void {
    this.nowMs += ms;
  }
}

class CollectingReader extends MetricReader {
  protected override async onForceFlush(): Promise<void> {}
  protected override async onShutdown(): Promise<void> {}
}

export type Harness = {
  readonly observer: Observer;
  readonly clock: ManualClock;
  readonly spans: InMemorySpanExporter;
  readonly logs: InMemoryLogRecordExporter;
  readonly tracerProvider: TracerProvider;
  readonly loggerProvider: LoggerProvider;
  collectMetrics(): Promise<ResourceMetrics>;
};

/** In-memory providers owned by the test plus an observer over them. */
export const createHarness = (
  options: { content?: ContentOptions; logs?: boolean; traces?: boolean; metrics?: boolean } = {},
): Harness => {
  const clock = new ManualClock();
  const spans = new InMemorySpanExporter();
  const logs = new InMemoryLogRecordExporter();
  const reader = new CollectingReader();
  const tracerProvider = new TracerProvider({
    spanProcessors: [new SimpleSpanProcessor({ exporter: spans })],
  });
  const meterProvider = new MeterProvider({ readers: [reader] });
  const loggerProvider = new LoggerProvider({
    processors: [new SimpleLogRecordProcessor({ exporter: logs })],
  });
  const observer = new Observer({
    tracerProvider: options.traces === false ? null : tracerProvider,
    meterProvider: options.metrics === false ? null : meterProvider,
    loggerProvider: options.logs === false ? null : (loggerProvider as LoggerProviderLike),
    clock: clock.now,
    ...(options.content === undefined ? {} : { content: options.content }),
  });
  return {
    observer,
    clock,
    spans,
    logs,
    tracerProvider,
    loggerProvider,
    collectMetrics: async () => (await reader.collect()).resourceMetrics,
  };
};

const hrToMs = ([seconds, nanos]: readonly [number, number]): number =>
  seconds * 1000 + nanos / 1e6;
const round = (value: number): number => Math.round(value * 1e6) / 1e6;

export const spanKindName = (kind: SpanKind): string => SpanKind[kind] ?? String(kind);
export const statusCodeName = (code: SpanStatusCode): string =>
  ({ [SpanStatusCode.UNSET]: "UNSET", [SpanStatusCode.OK]: "OK", [SpanStatusCode.ERROR]: "ERROR" })[
    code
  ] ?? String(code);

export const relativeMs = (time: readonly [number, number], origin = CLOCK_START_MS): number =>
  round(hrToMs(time) - origin);

/** Sorts object keys recursively so JSON output is canonical. */
export const canonical = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(canonical);
  if (value !== null && typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const key of Object.keys(value).sort()) {
      result[key] = canonical((value as Record<string, unknown>)[key]);
    }
    return result;
  }
  return value;
};

export type NormalizedMetric = {
  name: string;
  unit: string;
  points: (
    | { attributes: Record<string, unknown>; count: number; sum: number }
    | { attributes: Record<string, unknown>; value: number }
  )[];
};

export const normalizeMetrics = (resourceMetrics: ResourceMetrics): NormalizedMetric[] => {
  const result: NormalizedMetric[] = [];
  for (const scope of resourceMetrics.scopeMetrics) {
    if (scope.scope.name !== INSTRUMENTATION_SCOPE_NAME) continue;
    for (const metric of scope.metrics) {
      if (metric.dataPoints.length === 0) continue;
      const points = metric.dataPoints.map((point) => {
        const attributes = canonical(point.attributes) as Record<string, unknown>;
        if (metric.dataPointType === DataPointType.HISTOGRAM) {
          const histogram = point.value as { count: number; sum?: number };
          return { attributes, count: histogram.count, sum: round(histogram.sum ?? 0) };
        }
        return { attributes, value: point.value as number };
      });
      points.sort((a, b) => {
        const left = JSON.stringify(a.attributes);
        const right = JSON.stringify(b.attributes);
        return left < right ? -1 : left > right ? 1 : 0;
      });
      result.push({ name: metric.descriptor.name, unit: metric.descriptor.unit, points });
    }
  }
  return result.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
};

export type { ReadableLogRecord, ReadableSpan };
