import NativeXybrid from './NativeXybrid';
import type { ObservabilityStats, ObservationRef } from './types';

/** The plain object {@link ObservabilityConfig.init} sends across the bridge. */
export type WireObservabilityConfig = (
  | { kind: 'xybrid'; apiKey: string }
  | { kind: 'otlp'; endpoint: string }
) & {
  tracesEndpoint?: string;
  headers?: Record<string, string>;
  jsonProtocol?: boolean;
  service?: { name: string; version?: string };
  resourceAttributes?: Record<string, string>;
  deviceId?: string;
  providerOrigins?: string[];
  redactionRules?: Array<{ kind: 'replaceLiteral'; find: string; replacement: string } | { kind: 'omitAttribute'; key: string }>;
  metadataOnly?: boolean;
  maxContentBytes?: number;
  liveTraceIntervalMs?: number;
};

/**
 * OpenTelemetry export settings. Export is off until you build one and call
 * {@link init}; an API key passed to `Xybrid.initialize` does not enable it.
 * Once enabled, prompts and outputs are captured unless
 * {@link setMetadataOnly} says otherwise.
 *
 * ```ts
 * await ObservabilityConfig.xybrid(XYBRID_API_KEY).setService('my-app', '1.4.0').init();
 * // or, to your own OpenTelemetry Collector:
 * await ObservabilityConfig.otlp('http://collector:4318').setMetadataOnly(true).init();
 * ```
 *
 * Setters return the config, so calls chain. Nothing reaches native code
 * until {@link init}.
 */
export class ObservabilityConfig {
  private readonly wire: WireObservabilityConfig;

  private constructor(preset: WireObservabilityConfig) {
    this.wire = preset;
  }

  /** Traces only, to the Xybrid dashboard, authenticated with `apiKey`. */
  static xybrid(apiKey: string): ObservabilityConfig {
    return new ObservabilityConfig({ kind: 'xybrid', apiKey });
  }

  /** Traces, metrics and logs to an OTLP/HTTP endpoint such as `http://host:4318`. */
  static otlp(endpoint: string): ObservabilityConfig {
    return new ObservabilityConfig({ kind: 'otlp', endpoint });
  }

  /** Override the full trace endpoint URL (staging, self-hosted). */
  setTracesEndpoint(endpoint: string): this {
    this.wire.tracesEndpoint = endpoint;
    return this;
  }

  /** Add an HTTP header sent with every export request. */
  setHeader(name: string, value: string): this {
    this.wire.headers = { ...this.wire.headers, [name]: value };
    return this;
  }

  /** Send OTLP JSON instead of protobuf. */
  setJsonProtocol(json: boolean): this {
    this.wire.jsonProtocol = json;
    return this;
  }

  /** The reported service (app) name and version. */
  setService(name: string, version?: string): this {
    this.wire.service = version === undefined ? { name } : { name, version };
    return this;
  }

  /** Attach a resource attribute (for example `deployment.environment`). */
  setResourceAttribute(key: string, value: string): this {
    this.wire.resourceAttributes = { ...this.wire.resourceAttributes, [key]: value };
    return this;
  }

  /** Permit W3C propagation to this exact provider origin. */
  allowProviderContext(origin: string): this {
    (this.wire.providerOrigins ??= []).push(origin);
    return this;
  }

  /** Caller-supplied stable device identity; never detected automatically. */
  setDeviceId(id: string): this {
    this.wire.deviceId = id;
    return this;
  }

  /** Replace literal text in governed string values, in insertion order. */
  addLiteralRedaction(find: string, replacement: string): this {
    if (find.length === 0) throw new TypeError('redaction find must be nonempty');
    (this.wire.redactionRules ??= []).push({ kind: 'replaceLiteral', find, replacement });
    return this;
  }

  /** Omit one exact governed content attribute. */
  omitContentAttribute(key: string): this {
    if (key.length === 0) throw new TypeError('redaction key must be nonempty');
    (this.wire.redactionRules ??= []).push({ kind: 'omitAttribute', key });
    return this;
  }

  /** Record metadata only: no prompts, outputs or tool payloads. */
  setMetadataOnly(metadataOnly: boolean): this {
    this.wire.metadataOnly = metadataOnly;
    return this;
  }

  /** Maximum bytes per captured content attribute. */
  setMaxContentBytes(bytes: number): this {
    this.wire.maxContentBytes = bytes;
    return this;
  }

  /** Minimum interval between traced frames of one live session. */
  setLiveTraceIntervalMs(ms: number): this {
    this.wire.liveTraceIntervalMs = ms;
    return this;
  }

  /**
   * Start exporting with this configuration, replacing any previous one
   * (a JS reload can call it again). Rejects with `xybrid_config_error` when
   * the native side rejects the settings.
   */
  async init(): Promise<void> {
    await NativeXybrid.observabilityInit(this.wire);
  }
}

/** End-user feedback on a run. Set at least one field. */
export interface Feedback {
  score?: number;
  label?: string;
  comment?: string;
}

/** An evaluation result for a run (an LLM judge, a heuristic, a test). */
export interface Evaluation {
  name: string;
  scoreValue?: number;
  scoreLabel?: string;
  explanation?: string;
}

const DEFAULT_TIMEOUT_MS = 5000;

/** Process-wide observability controls. Configure export with {@link ObservabilityConfig}. */
export const Observability = {
  /** The Xybrid dashboard's trace endpoint, which {@link ObservabilityConfig.xybrid} uses. */
  defaultTracesEndpoint(): Promise<string> {
    return NativeXybrid.observabilityDefaultTracesEndpoint();
  },

  /**
   * Delivery counters: live while export runs; after `shutdown`, the final ones
   * (what the shutdown failed or dropped included) until the next `init`; all
   * zero before any.
   */
  async stats(): Promise<ObservabilityStats> {
    return (await NativeXybrid.observabilityStats()) as ObservabilityStats;
  },

  /**
   * Export what is queued, waiting at most `timeoutMs`. Call before the app
   * backgrounds. Rejects when the bound is exceeded.
   */
  flush(timeoutMs: number = DEFAULT_TIMEOUT_MS): Promise<void> {
    return NativeXybrid.observabilityFlush(timeoutMs);
  },

  /** Flush and stop exporting, waiting at most `timeoutMs`. Idempotent. */
  shutdown(timeoutMs: number = DEFAULT_TIMEOUT_MS): Promise<void> {
    return NativeXybrid.observabilityShutdown(timeoutMs);
  },

  /**
   * Record end-user feedback on a run (`result.observation`). Resolves
   * `false` when no logs signal is configured — the Xybrid preset sends
   * traces only.
   */
  recordFeedback(target: ObservationRef, feedback: Feedback): Promise<boolean> {
    return NativeXybrid.observabilityRecordFeedback(target, feedback);
  },

  /** Record an evaluation result on a run. Resolves `false` without a logs signal. */
  recordEvaluation(target: ObservationRef, evaluation: Evaluation): Promise<boolean> {
    return NativeXybrid.observabilityRecordEvaluation(target, evaluation);
  },
};
