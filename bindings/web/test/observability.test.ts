import { afterEach, describe, expect, test } from "bun:test";
import { LoggerProvider } from "@opentelemetry/sdk-logs";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  TracerProvider,
} from "@opentelemetry/sdk-trace";
import { type ContentOptions, ObservationContext, Observer } from "@xybrid/observability";
import {
  InferenceError,
  InputValidationError,
  RuntimeConfigurationError,
  RuntimeInitializationError,
} from "../src/errors.ts";
import { RuntimeInitializer } from "../src/internal/initialization.ts";
import type { LoadMetrics, Request, Response, RunMetrics } from "../src/internal/protocol.ts";
import type {
  EngineGenerateOptions,
  LlmEngine,
  LlmGeneration,
  LlmRuntime,
} from "../src/internal/runtime.ts";
import { rustLlmRuntime } from "../src/internal/rust-llm-runtime.ts";
import { loadLlm, XybridLlm } from "../src/llm.ts";
import { parseMetadata } from "../src/metadata.ts";
import type { LlmLoadOptions } from "../src/types.ts";
import { ggufMetadata } from "./helpers.ts";

const metadataUrl = new URL("https://models.example/smollm2/model_metadata.json");
const PARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
const memory = { heapBytes: 4096, allocatedBytes: 2048 };
const loadedMetrics: LoadMetrics = {
  accelerator: "wasm",
  rustVersion: "0.16.0",
  adapter: null,
  modelBytes: 1234,
  downloadMs: 1,
  initializationMs: 1,
  loadMs: 1,
  memory,
};

const observe = (content?: ContentOptions) => {
  const spans = new InMemorySpanExporter();
  const tracerProvider = new TracerProvider({
    spanProcessors: [new SimpleSpanProcessor({ exporter: spans })],
  });
  const observer = new Observer({
    tracerProvider,
    meterProvider: null,
    loggerProvider: new LoggerProvider(),
    ...(content === undefined ? {} : { content }),
  });
  return { observer, spans };
};

/** An arbitrary observer can fail even when the library's own hooks are guarded. */
const throwingObserver = (observer: Observer, hook: string, hits: string[]): Observer => {
  const wrap = <T extends object>(target: T): T =>
    new Proxy(target, {
      get(object, key) {
        const value: unknown = Reflect.get(object, key, object);
        const fail = () => {
          hits.push(hook);
          throw new Error(`telemetry failed: ${hook}`);
        };
        if (key === hook && typeof value !== "function") fail();
        if (typeof value !== "function") return value;
        return (...args: unknown[]) => {
          if (key === hook) fail();
          const result: unknown = Reflect.apply(value, object, args);
          return (key === "startInference" || key === "startModelLoad") &&
            typeof result === "object" &&
            result !== null
            ? wrap(result)
            : result;
        };
      },
    });
  return wrap(observer);
};

type Script = {
  /** Deltas yielded to the caller. */
  readonly deltas: readonly string[];
  /** Worker-reported tokens: [epoch ms, text]; empty text is an incomplete character. */
  readonly tokens?: readonly (readonly [number, string])[];
  readonly completedAtMs?: number;
  readonly run?: RunMetrics;
  readonly fail?: boolean;
};

const runMetrics = (overrides: Partial<RunMetrics> = {}): RunMetrics => ({
  cancelled: false,
  promptTokens: 11,
  generatedTokens: 3,
  firstTokenMs: 5,
  totalMs: 20,
  decodeTokensPerSecond: 50,
  peakMemory: memory,
  memory,
  ...overrides,
});

/** A runtime whose engine replays `script`, recording what each generate call received. */
const scriptedRuntime = (script: Script, reportsTiming: boolean) => {
  const calls: EngineGenerateOptions[] = [];
  let failLoad = false;
  let deleted = 0;
  let disposed = 0;
  const engine: LlmEngine & { lastRun: RunMetrics | undefined } = {
    loaded: loadedMetrics,
    lastRun: undefined,
    reportsTiming,
    generate: async (_prompt, options): Promise<LlmGeneration> => {
      calls.push(options);
      let stopped = false;
      let finished = false;
      const stream = (async function* () {
        for (const [index, delta] of script.deltas.entries()) {
          if (stopped) return;
          for (const [atMs, text] of script.tokens?.filter((_, i) => i === index) ?? []) {
            options.telemetry?.onToken(atMs, text);
          }
          if (script.fail === true && index === 1) throw new Error("decode failed");
          yield delta;
        }
        if (script.completedAtMs !== undefined) options.telemetry?.onComplete(script.completedAtMs);
        finished = true;
        engine.lastRun = script.run;
      })();
      return {
        stream,
        // Like the worker: cancelling a finished stream is a no-op.
        cancel: () => {
          if (finished || stopped) return;
          stopped = true;
          engine.lastRun = runMetrics({ cancelled: true, generatedTokens: 1 });
        },
        dispose: async () => {
          disposed += 1;
        },
      };
    },
    delete: async () => {
      deleted += 1;
    },
  };
  const runtime: LlmRuntime<number> = {
    initialize: async () => undefined,
    probeAccelerator: async () => undefined,
    fetchModel: async () => {
      if (failLoad) throw new DOMException("network down", "NetworkError");
      return 1;
    },
    modelFromChunks: async () => 1,
    createEngine: async () => engine,
  };
  return {
    runtime,
    calls,
    failLoad: () => (failLoad = true),
    deletedEngines: () => deleted,
    disposedGenerations: () => disposed,
  };
};

const load = (runtime: LlmRuntime<number>, options: LlmLoadOptions) =>
  loadLlm(
    metadataUrl,
    { wasmPath: "/runtime", accelerator: "wasm", ...options },
    runtime,
    async () => parseMetadata(ggufMetadata()),
    new RuntimeInitializer(),
  );

const drain = async (stream: AsyncGenerator<string, void, void>): Promise<string> => {
  let text = "";
  for await (const delta of stream) text += delta;
  return text;
};

describe("observed loads", () => {
  test("a load becomes a load_model operation with the model's identity", async () => {
    const { observer, spans } = observe();
    const { runtime } = scriptedRuntime({ deltas: [] }, true);
    const session = await load(runtime, { observer, observation: { parent: PARENT } });
    const [span] = spans.getFinishedSpans();
    expect(span?.name).toBe("load_model browser-smollm2");
    expect(span?.attributes).toMatchObject({
      "gen_ai.operation.name": "load_model",
      "gen_ai.request.model": "browser-smollm2",
      "gen_ai.provider.name": "llama.cpp",
      "xybrid.genai.runtime.name": "llama.cpp",
      "xybrid.genai.model.format": "gguf",
      "xybrid.genai.model.size_bytes": 1234,
      "xybrid.genai.accelerator": "wasm",
      "xybrid.genai.resource.memory.used_bytes": 2048,
    });
    expect(span?.spanContext().traceId).toBe("4bf92f3577b34da6a3ce929d0e0e4736");
    expect(session.loaded.observation?.spanId).toBe(span?.spanContext().spanId ?? "");
  });

  test("a failed load ends with its error type", async () => {
    const { observer, spans } = observe();
    const control = scriptedRuntime({ deltas: [] }, true);
    control.failLoad();
    await expect(load(control.runtime, { observer })).rejects.toThrow();
    const [span] = spans.getFinishedSpans();
    expect(span?.name).toBe("load_model smollm2");
    expect(span?.status.code).toBe(2);
    // The error class name is the low-cardinality error type.
    expect(span?.attributes["error.type"]).toBe("NetworkError");
  });

  test("rejects an observer that is not an Observer", async () => {
    const { runtime } = scriptedRuntime({ deltas: [] }, true);
    await expect(load(runtime, { observer: {} as unknown as Observer })).rejects.toBeInstanceOf(
      RuntimeConfigurationError,
    );
    await expect(
      Reflect.apply(XybridLlm.load, XybridLlm, [
        metadataUrl,
        { wasmPath: "/runtime", observer: observe().observer, observation: { trace: "never" } },
      ]),
    ).rejects.toBeInstanceOf(RuntimeConfigurationError);
  });
});

describe("observed generation", () => {
  test("worker-measured times, usage, content and the observation ref", async () => {
    const { observer, spans } = observe();
    const t0 = performance.timeOrigin + performance.now();
    const script: Script = {
      deltas: ["Hel", "lo", "!"],
      // The first token decodes to an incomplete character: first token precedes first chunk.
      tokens: [
        [t0 + 40, ""],
        [t0 + 55, "lo"],
        [t0 + 70, "!"],
      ],
      completedAtMs: t0 + 90,
      run: runMetrics({ generatedTokens: 3 }),
    };
    const { runtime, calls } = scriptedRuntime(script, true);
    const session = await load(runtime, { observer });
    const text = await drain(
      session.generateStream("Say hello", {
        maxOutputTokens: 3,
        observation: { parent: PARENT, conversationId: "conv-7" },
      }),
    );
    expect(text).toBe("Hello!");
    const inference = spans.getFinishedSpans().find((span) => span.name.startsWith("chat"));
    expect(inference?.name).toBe("chat browser-smollm2");
    expect(inference?.kind).toBe(0);
    expect(inference?.parentSpanContext?.spanId).toBe("00f067aa0ba902b7");
    const attributes = inference?.attributes ?? {};
    expect(attributes).toMatchObject({
      "gen_ai.operation.name": "chat",
      "gen_ai.provider.name": "llama.cpp",
      "gen_ai.request.model": "browser-smollm2",
      "gen_ai.request.max_tokens": 3,
      "gen_ai.request.stream": true,
      "gen_ai.conversation.id": "conv-7",
      "xybrid.genai.execution.location": "local",
      "xybrid.genai.runtime.name": "llama.cpp",
      "xybrid.genai.accelerator": "wasm",
      "gen_ai.usage.input_tokens": 11,
      "gen_ai.usage.output_tokens": 3,
      "gen_ai.response.finish_reasons": ["length"],
      "xybrid.genai.response.chunk_count": 2,
      "xybrid.genai.response.time_per_output_token": 0.02,
    });
    const started = (inference?.startTime[0] ?? 0) * 1000 + (inference?.startTime[1] ?? 0) / 1e6;
    expect(Number(attributes["xybrid.genai.response.time_to_first_token"])).toBeCloseTo(
      (t0 + 40 - started) / 1000,
      5,
    );
    expect(Number(attributes["gen_ai.response.time_to_first_chunk"])).toBeCloseTo(
      (t0 + 55 - started) / 1000,
      5,
    );
    const ended = (inference?.endTime[0] ?? 0) * 1000 + (inference?.endTime[1] ?? 0) / 1e6;
    expect(ended).toBeCloseTo(t0 + 90, 2);
    expect(JSON.parse(String(attributes["gen_ai.input.messages"]))).toEqual([
      { role: "user", parts: [{ type: "text", content: "Say hello" }] },
    ]);
    expect(JSON.parse(String(attributes["gen_ai.output.messages"]))).toEqual([
      { role: "assistant", parts: [{ type: "text", content: "Hello!" }], finish_reason: "length" },
    ]);
    // The context crosses to the engine (the worker) as a traceparent of this span.
    const traceparent = calls[0]?.telemetry?.context.traceparent;
    expect(ObservationContext.fromTraceparent(traceparent ?? "")?.spanId).toBe(
      inference?.spanContext().spanId ?? "",
    );
    expect(session.lastRun?.observation).toEqual({
      traceId: inference?.spanContext().traceId ?? "",
      spanId: inference?.spanContext().spanId ?? "",
    });
  });

  test("generate() is not a streaming request; no_content keeps text off the span", async () => {
    const { observer, spans } = observe({ capture: "no_content" });
    const { runtime } = scriptedRuntime({ deltas: ["a", "b"], run: runMetrics() }, true);
    const session = await load(runtime, { observer });
    expect(await session.generate("hi")).toBe("ab");
    const inference = spans.getFinishedSpans().find((span) => span.name.startsWith("chat"));
    expect(inference?.attributes["gen_ai.request.stream"]).toBeUndefined();
    expect(inference?.attributes["gen_ai.request.max_tokens"]).toBe(64);
    expect(inference?.attributes["gen_ai.input.messages"]).toBeUndefined();
    expect(inference?.attributes["gen_ai.output.messages"]).toBeUndefined();
    expect(inference?.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  });

  test("engines without their own timing are timed per received chunk", async () => {
    const { observer, spans } = observe();
    const { runtime } = scriptedRuntime({ deltas: ["a", "", "b"] }, false);
    const session = await load(runtime, { observer });
    await drain(session.generateStream("hi"));
    const inference = spans.getFinishedSpans().find((span) => span.name.startsWith("chat"));
    expect(inference?.attributes["xybrid.genai.response.chunk_count"]).toBe(2);
    expect(inference?.attributes["xybrid.genai.response.time_to_first_token"]).toBeUndefined();
    // No runtime metrics: usage is never fabricated.
    expect(inference?.attributes["gen_ai.usage.input_tokens"]).toBeUndefined();
  });

  test("abandoning the stream records a cancellation", async () => {
    const { observer, spans } = observe();
    const { runtime } = scriptedRuntime({ deltas: ["a", "b", "c"] }, true);
    const session = await load(runtime, { observer });
    for await (const _delta of session.generateStream("hi")) break;
    const inference = spans.getFinishedSpans().find((span) => span.name.startsWith("chat"));
    expect(inference?.attributes["error.type"]).toBe("cancelled");
    expect(inference?.status.message).toBe("cancelled");
    expect(inference?.attributes["gen_ai.usage.output_tokens"]).toBe(1);
  });

  test("a runtime failure records the error", async () => {
    const { observer, spans } = observe();
    const { runtime } = scriptedRuntime({ deltas: ["a", "b"], fail: true }, true);
    const session = await load(runtime, { observer });
    await expect(drain(session.generateStream("hi"))).rejects.toThrow();
    const inference = spans.getFinishedSpans().find((span) => span.name.startsWith("chat"));
    expect(inference?.attributes["error.type"]).toBe("InferenceError");
    expect(inference?.status.code).toBe(2);
  });

  test("suppressed generations return no observation", async () => {
    const { observer, spans } = observe();
    const { runtime } = scriptedRuntime({ deltas: ["a"], run: runMetrics() }, true);
    const session = await load(runtime, { observer });
    await drain(session.generateStream("hi", { observation: { trace: "suppress" } }));
    expect(spans.getFinishedSpans().filter((span) => span.name.startsWith("chat"))).toHaveLength(0);
    expect(session.lastRun?.observation).toBeUndefined();
  });

  test("malformed per-call observation options are input errors", async () => {
    const { observer } = observe();
    const { runtime } = scriptedRuntime({ deltas: [] }, true);
    const session = await load(runtime, { observer });
    expect(() => session.generateStream("hi", { observation: { parent: 5 as never } })).toThrow(
      InputValidationError,
    );
  });

  test("without an observer the engine gets no telemetry and results carry no observation", async () => {
    const { runtime, calls } = scriptedRuntime({ deltas: ["a"], run: runMetrics() }, true);
    const session = await load(runtime, {});
    await drain(session.generateStream("hi", { observation: { conversationId: "ignored" } }));
    expect(calls[0]).toEqual({});
    expect(session.lastRun?.observation).toBeUndefined();
    expect(session.loaded.observation).toBeUndefined();
  });
});

describe("load error privacy and URL identity", () => {
  const failedDownload = async (check: (requests: string[]) => Promise<void>): Promise<void> => {
    const originalFetch = globalThis.fetch;
    const originalLocation = Object.getOwnPropertyDescriptor(globalThis, "location");
    const requests: string[] = [];
    Object.defineProperty(globalThis, "location", {
      configurable: true,
      value: { href: "https://app.example/demo/" },
    });
    globalThis.fetch = Object.assign(
      async (input: Parameters<typeof fetch>[0]) => {
        requests.push(input instanceof globalThis.Request ? input.url : String(input));
        return new globalThis.Response("expired", { status: 403, statusText: "Forbidden" });
      },
      { preconnect: originalFetch.preconnect },
    );
    try {
      await check(requests);
    } finally {
      globalThis.fetch = originalFetch;
      if (originalLocation === undefined) Reflect.deleteProperty(globalThis, "location");
      else Object.defineProperty(globalThis, "location", originalLocation);
    }
  };

  for (const capture of ["no_content", "span_only", "event_only", "span_and_event"] as const) {
    test(`failed signed URL never exports credentials in ${capture} mode`, async () => {
      await failedDownload(async (requests) => {
        const { observer, spans } = observe({ capture });
        const url =
          "https://bucket.example/models/qwen.gguf?X-Amz-Credential=AKIASECRET&X-Amz-Signature=secret-signature&token=secret-token#secret-fragment";
        // The caller retains the original diagnostic; the adapter only exports its class.
        await expect(XybridLlm.fromUrl(url, { observer, accelerator: "wasm" })).rejects.toThrow(
          "secret-signature",
        );
        expect(requests).toHaveLength(1);
        const [span] = spans.getFinishedSpans();
        expect(span?.attributes["gen_ai.request.model"]).toBe("qwen");
        expect(span?.attributes["error.type"]).toBe("RuntimeInitializationError");
        expect(span?.status).toEqual({ code: 2, message: "RuntimeInitializationError" });
        const exported = JSON.stringify({
          attributes: span?.attributes,
          status: span?.status,
          events: span?.events,
        });
        for (const secret of [
          "AKIASECRET",
          "secret-signature",
          "secret-token",
          "secret-fragment",
          "X-Amz-",
        ]) {
          expect(exported).not.toContain(secret);
        }
      });
    });
  }

  test("metadata load errors do not forward URL userinfo or query credentials to the observer", async () => {
    const { observer, spans } = observe({ capture: "no_content" });
    const { runtime } = scriptedRuntime({ deltas: [] }, true);
    const original = new RuntimeInitializationError(
      new Error(
        "GET https://user-secret:password-secret@models.example/model.json?token=query-secret#fragment-secret",
      ),
    );
    await expect(
      loadLlm(
        metadataUrl,
        { wasmPath: "/runtime", observer },
        runtime,
        async () => {
          throw original;
        },
        new RuntimeInitializer(),
      ),
    ).rejects.toBe(original);
    const [span] = spans.getFinishedSpans();
    expect(span?.status).toEqual({ code: 2, message: "RuntimeInitializationError" });
    expect(JSON.stringify(span?.attributes)).not.toContain("secret");
  });

  for (const observed of [false, true]) {
    for (const [path, model] of [
      ["q4_k_m-100%.gguf", "q4_k_m-100%"],
      ["q4-100%25%.gguf", "q4-100%25%"],
      ["%E0%A4.gguf", "%E0%A4"],
      ["ok%20name.gguf", "ok name"],
      ["100%25.gguf", "100%"],
      ["caf%C3%A9.GGUF", "café"],
    ]) {
      test(`${observed ? "observed" : "plain"} direct URL reaches fetch for ${path}`, async () => {
        await failedDownload(async (requests) => {
          const { observer, spans } = observe();
          const url = new URL(`https://models.example/${path}?token=secret`);
          await expect(
            XybridLlm.fromUrl(url, {
              accelerator: "wasm",
              ...(observed ? { observer } : {}),
            }),
          ).rejects.toBeInstanceOf(RuntimeInitializationError);
          expect(requests).toEqual([url.href]);
          if (observed)
            expect(spans.getFinishedSpans()[0]?.attributes["gen_ai.request.model"]).toBe(model);
        });
      });
    }

    test(`${observed ? "observed" : "plain"} browser-relative URL keeps valid escapes`, async () => {
      await failedDownload(async (requests) => {
        const { observer, spans } = observe();
        await expect(
          XybridLlm.fromUrl("../models/100%25.gguf", {
            accelerator: "wasm",
            ...(observed ? { observer } : {}),
          }),
        ).rejects.toBeInstanceOf(RuntimeInitializationError);
        expect(requests).toEqual(["https://app.example/models/100%25.gguf"]);
        if (observed)
          expect(spans.getFinishedSpans()[0]?.attributes["gen_ai.request.model"]).toBe("100%");
      });
    });

    test(`${observed ? "observed" : "plain"} metadata URL with invalid UTF-8 still loads successfully`, async () => {
      const { observer } = observe();
      const control = scriptedRuntime({ deltas: ["ok"] }, true);
      const urls: string[] = [];
      control.runtime.fetchModel = async (url) => {
        urls.push(url.href);
        return 1;
      };
      const session = await loadLlm(
        new URL("https://models.example/%E0%A4/model_metadata.json"),
        { wasmPath: "/runtime", accelerator: "wasm", ...(observed ? { observer } : {}) },
        control.runtime,
        async () => parseMetadata(ggufMetadata()),
        new RuntimeInitializer(),
      );
      expect(urls).toEqual(["https://models.example/%E0%A4/model.gguf"]);
      expect(await session.generate("hi")).toBe("ok");
      await session.dispose();
      expect(control.deletedEngines()).toBe(1);
    });
  }
});

describe("telemetry failure isolation", () => {
  test("throwing provider accessors disable telemetry without losing the loaded engine", async () => {
    const observer = new Proxy(observe().observer, {
      get() {
        throw new Error("provider accessor failed");
      },
    });
    const control = scriptedRuntime({ deltas: ["ok"] }, true);
    const session = await load(control.runtime, { observer });
    expect(await session.generate("hi")).toBe("ok");
    await session.dispose();
    expect(control.deletedEngines()).toBe(1);
  }, 1_000);

  test("load cancellation survives a throwing observation cancel", async () => {
    const hits: string[] = [];
    const control = scriptedRuntime({ deltas: [] }, true);
    const original = new DOMException("load cancelled", "AbortError");
    control.runtime.fetchModel = async () => {
      throw original;
    };
    await expect(
      load(control.runtime, {
        observer: throwingObserver(observe().observer, "cancel", hits),
      }),
    ).rejects.toBe(original);
    expect(hits).toEqual(["cancel"]);
  });

  for (const hook of [
    "startModelLoad",
    "setAttribute",
    "recordResource",
    "end",
    "observationRef",
  ]) {
    test(`load survives ${hook} failures and retains a disposable engine`, async () => {
      const { observer } = observe();
      const hits: string[] = [];
      const control = scriptedRuntime({ deltas: ["ok"] }, true);
      const session = await load(control.runtime, {
        observer: throwingObserver(observer, hook, hits),
      });
      expect(hits.length).toBeGreaterThan(0);
      expect(await session.generate("hi")).toBe("ok");
      await session.dispose();
      await session.dispose();
      expect(control.deletedEngines()).toBe(1);
    }, 1_000);
  }

  for (const hook of ["startModelLoad", "endWithError", "observationRef"]) {
    test(`failed load preserves its original error when ${hook} fails`, async () => {
      const { observer } = observe();
      const hits: string[] = [];
      const original = new RuntimeInitializationError(new Error("download failed"));
      const { runtime } = scriptedRuntime({ deltas: [] }, true);
      runtime.fetchModel = async () => {
        throw original;
      };
      await expect(
        load(runtime, { observer: throwingObserver(observer, hook, hits) }),
      ).rejects.toBe(original);
      expect(hits.length).toBeGreaterThan(0);
    });
  }

  for (const hook of [
    "startInference",
    "contentCapture",
    "recordInputMessages",
    "context",
    "recordFirstToken",
    "recordChunk",
    "recordUsage",
    "recordTokenTiming",
    "recordOutputMessages",
    "recordFinishReasons",
    "end",
    "observationRef",
  ]) {
    test(`generation survives ${hook} failures, runs again and disposes`, async () => {
      const { observer, spans } = observe();
      const hits: string[] = [];
      const control = scriptedRuntime(
        {
          deltas: ["hello", " world"],
          tokens: [
            [1_000, "hello"],
            [1_001, " world"],
          ],
          completedAtMs: 1_002,
          run: runMetrics(),
        },
        true,
      );
      const session = await load(control.runtime, {
        observer: throwingObserver(observer, hook, hits),
      });
      expect(await drain(session.generateStream("hi"))).toBe("hello world");
      expect(await session.generate("again")).toBe("hello world");
      expect(hits.length).toBeGreaterThan(0);
      // Failure to record an attribute must not skip the operation's end.
      if (hook === "recordInputMessages" || hook === "recordOutputMessages") {
        expect(
          spans.getFinishedSpans().filter((span) => span.name.startsWith("chat")),
        ).toHaveLength(2);
      }
      await session.dispose();
      expect(control.disposedGenerations()).toBe(2);
      expect(control.deletedEngines()).toBe(1);
    }, 1_000);
  }

  test("chunk timing failures for engines without token timing do not drop output", async () => {
    const hits: string[] = [];
    const control = scriptedRuntime({ deltas: ["a", "b"] }, false);
    const session = await load(control.runtime, {
      observer: throwingObserver(observe().observer, "recordChunk", hits),
    });
    expect(await session.generate("hi")).toBe("ab");
    expect(hits).toHaveLength(2);
    await session.dispose();
    expect(control.deletedEngines()).toBe(1);
  });

  test("error finalization failures preserve inference errors and release the run", async () => {
    const hits: string[] = [];
    const script = { deltas: ["a", "b"], fail: true };
    const control = scriptedRuntime(script, true);
    const session = await load(control.runtime, {
      observer: throwingObserver(observe().observer, "endWithError", hits),
    });
    await expect(session.generate("hi")).rejects.toBeInstanceOf(InferenceError);
    expect(hits).toEqual(["endWithError"]);
    script.fail = false;
    expect(await session.generate("again")).toBe("ab");
    await session.dispose();
    expect(control.disposedGenerations()).toBe(2);
    expect(control.deletedEngines()).toBe(1);
  }, 1_000);

  test("cancellation telemetry failures cannot hang disposal of an active stream", async () => {
    const hits: string[] = [];
    const control = scriptedRuntime({ deltas: ["a", "b"] }, true);
    const session = await load(control.runtime, {
      observer: throwingObserver(observe().observer, "cancel", hits),
    });
    const stream = session.generateStream("hi");
    expect((await stream.next()).value).toBe("a");
    await session.dispose();
    expect(hits).toEqual(["cancel"]);
    expect(control.disposedGenerations()).toBe(1);
    expect(control.deletedEngines()).toBe(1);
  }, 1_000);

  for (const hook of ["onStart", "onEnd"] as const) {
    test(`throwing span processor ${hook} cannot break loads or later generations`, async () => {
      let hits = 0;
      const processor = new SimpleSpanProcessor({ exporter: new InMemorySpanExporter() });
      processor[hook] = () => {
        hits += 1;
        throw new Error("processor failed");
      };
      const observer = new Observer({
        tracerProvider: new TracerProvider({ spanProcessors: [processor] }),
        meterProvider: null,
        loggerProvider: null,
      });
      const control = scriptedRuntime({ deltas: ["ok"] }, true);
      const session = await load(control.runtime, { observer });
      expect(await session.generate("hi")).toBe("ok");
      expect(await session.generate("again")).toBe("ok");
      await session.dispose();
      expect(hits).toBe(3);
      expect(control.deletedEngines()).toBe(1);
    }, 1_000);
  }

  for (const kind of ["gen_ai.input.messages", "gen_ai.output.messages"]) {
    test(`throwing ${kind} redactor cannot break inference or cleanup`, async () => {
      let hits = 0;
      const { observer } = observe({
        redact: (key, value) => {
          if (key === kind) {
            hits += 1;
            throw new Error("redactor failed");
          }
          return value;
        },
      });
      const control = scriptedRuntime({ deltas: ["ok"] }, true);
      const session = await load(control.runtime, { observer });
      expect(await session.generate("hi")).toBe("ok");
      expect(await session.generate("again")).toBe("ok");
      await session.dispose();
      expect(hits).toBe(2);
      expect(control.deletedEngines()).toBe(1);
    }, 1_000);
  }
});

describe("worker protocol", () => {
  const OriginalWorker = globalThis.Worker;
  afterEach(() => {
    globalThis.Worker = OriginalWorker;
  });

  /** Stands in for worker.ts: replies like the runtime worker, with times when observed. */
  class FakeWorker {
    static requests: Request[] = [];
    onmessage: ((event: MessageEvent<Response>) => void) | null = null;
    onerror: ((event: ErrorEvent) => void) | null = null;
    onmessageerror: (() => void) | null = null;
    postMessage(request: Request): void {
      FakeWorker.requests.push(request);
      const reply = (message: Response) =>
        queueMicrotask(() => this.onmessage?.({ data: message } as MessageEvent<Response>));
      if (request.type === "load")
        reply({ type: "loaded", id: request.id, metrics: loadedMetrics });
      if (request.type === "generate") {
        const observed = request.observation !== undefined;
        const at = (offset: number) => (observed ? { atMs: 1_000 + offset } : {});
        reply({ type: "token", id: request.id, tokenId: 1, text: "", ...at(1) });
        reply({ type: "token", id: request.id, tokenId: 2, text: "hé", ...at(2) });
        reply({
          type: "complete",
          id: request.id,
          metrics: runMetrics({ generatedTokens: 2 }),
          ...(observed ? { completedAtMs: 1_010 } : {}),
        });
      }
      if (request.type === "dispose") reply({ type: "disposed", id: request.id, memory });
    }
    terminate(): void {}
  }

  test("observed requests carry the traceparent and get epoch times back", async () => {
    FakeWorker.requests = [];
    globalThis.Worker = FakeWorker as unknown as typeof Worker;
    await rustLlmRuntime.initialize({
      wasmPath: "https://app.example/runtime/",
      threads: false,
      jspi: false,
    });
    const engine = await rustLlmRuntime.createEngine(
      { bytes: new Uint8Array(4), downloadMs: 0 },
      "wasm",
      512,
    );
    expect(engine.reportsTiming).toBe(true);
    const tokens: [number, string][] = [];
    let completedAt: number | undefined;
    const observed = await engine.generate("hi", {
      maxOutputTokens: 4,
      telemetry: {
        context: { traceparent: PARENT },
        onToken: (atMs, text) => tokens.push([atMs, text]),
        onComplete: (atMs) => {
          completedAt = atMs;
        },
      },
    });
    expect(await drain(observed.stream)).toBe("hé");
    await observed.dispose();
    expect(tokens).toEqual([
      [1_001, ""],
      [1_002, "hé"],
    ]);
    expect(completedAt).toBe(1_010);
    const plain = await engine.generate("hi", {});
    await drain(plain.stream);
    await plain.dispose();
    const generates = FakeWorker.requests.filter((request) => request.type === "generate");
    expect(generates[0]).toMatchObject({ observation: { traceparent: PARENT }, maxTokens: 4 });
    expect("observation" in (generates[1] ?? {})).toBe(false);
    await engine.delete();
  });

  test("throwing token and completion callbacks cannot drop tokens or strand worker requests", async () => {
    FakeWorker.requests = [];
    globalThis.Worker = FakeWorker as unknown as typeof Worker;
    await rustLlmRuntime.initialize({
      wasmPath: "https://app.example/runtime/",
      threads: false,
      jspi: false,
    });
    const engine = await rustLlmRuntime.createEngine(
      { bytes: new Uint8Array(4), downloadMs: 0 },
      "wasm",
      512,
    );
    let tokens = 0;
    let completions = 0;
    const generation = await engine.generate("hi", {
      telemetry: {
        context: {},
        onToken: () => {
          tokens += 1;
          throw new Error("token telemetry failed");
        },
        onComplete: () => {
          completions += 1;
          throw new Error("completion telemetry failed");
        },
      },
    });
    const deltas: string[] = [];
    for await (const delta of generation.stream) deltas.push(delta);
    expect(deltas).toEqual(["", "hé"]);
    await generation.dispose();
    expect(tokens).toBe(2);
    expect(completions).toBe(1);
    expect(engine.lastRun?.generatedTokens).toBe(2);
    const next = await engine.generate("again", {});
    expect(await drain(next.stream)).toBe("hé");
    await next.dispose();
    await engine.delete();
    expect(engine.releasedMemory).toEqual(memory);
  }, 1_000);
});
