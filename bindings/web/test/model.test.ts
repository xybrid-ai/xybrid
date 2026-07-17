import { describe, expect, test } from "bun:test";

import {
  ConcurrentRunError,
  DeviceLostError,
  DisposedError,
  InferenceError,
  InputValidationError,
  InvalidMetadataError,
  RuntimeConfigurationError,
  RuntimeInitializationError,
  XybridError,
} from "../src/errors.ts";
import { RuntimeInitializer } from "../src/internal/initialization.ts";
import type { loadModelBytes } from "../src/internal/model-download.ts";
import { XybridModel } from "../src/model.ts";
import type { LoadOptions } from "../src/types.ts";
import { createRuntime, details, loadWithDependencies, tensor, tfliteMetadata } from "./helpers.ts";

const metadataUrl = new URL("https://models.example/add/model_metadata.json");
const options: LoadOptions = { wasmPath: "/wasm", accelerator: "auto" };

const load = async (
  runtime = createRuntime(),
  accelerator: LoadOptions = options,
  loadBytes: typeof loadModelBytes = async () => new Uint8Array([0]),
) => ({
  runtime,
  model: await loadWithDependencies(
    metadataUrl,
    accelerator,
    runtime.runtime,
    async () => tfliteMetadata(),
    new RuntimeInitializer(),
    loadBytes,
  ),
});

describe("XybridModel runtime lifecycle", () => {
  test("rejects reflective construction without the module-private token", () => {
    expect(() => Reflect.construct(XybridModel, [{}])).toThrow(RuntimeConfigurationError);
  });

  test("rejects malformed public load options with typed errors", async () => {
    for (const malformed of [null, undefined, {}, { wasmPath: "/wasm", accelerator: "invalid" }]) {
      await expect(
        Reflect.apply(XybridModel.load, XybridModel, [metadataUrl, malformed]),
      ).rejects.toBeInstanceOf(RuntimeConfigurationError);
    }
    await expect(XybridModel.load("data:application/json,%7B%7D", options)).rejects.toBeInstanceOf(
      InvalidMetadataError,
    );
  });

  test("falls back from WebGPU failures to wasm only for auto", async () => {
    const runtime = createRuntime();
    runtime.failWebGpu();
    const { model } = await load(runtime);
    expect(model.accelerator).toBe("wasm");
    expect(runtime.compiled).toEqual(["wasm"]);

    const explicit = createRuntime();
    explicit.failWebGpu();
    await expect(
      load(explicit, { wasmPath: "/wasm", accelerator: "webgpu" }),
    ).rejects.toBeInstanceOf(RuntimeInitializationError);
    expect(explicit.compiled).toEqual([]);

    const wasm = createRuntime();
    await load(wasm, { wasmPath: "/wasm", accelerator: "wasm" });
    expect(wasm.compiled).toEqual(["wasm"]);

    const compilationFailure = createRuntime();
    compilationFailure.failWebGpuCompilation();
    const { model: recovered } = await load(compilationFailure);
    expect(recovered.accelerator).toBe("wasm");
    expect(compilationFailure.compiled).toEqual(["webgpu", "wasm"]);
  });

  test("fetches direct-URL model bytes once across an auto fallback", async () => {
    const runtime = createRuntime();
    runtime.failWebGpuCompilation();
    let fetches = 0;
    await load(runtime, options, async () => {
      fetches += 1;
      return new Uint8Array([0]);
    });
    expect(fetches).toBe(1);
    expect(runtime.compiled).toEqual(["webgpu", "wasm"]);
  });

  test("does not download model bytes before an explicit WebGPU preflight", async () => {
    const runtime = createRuntime();
    runtime.failWebGpu();
    let fetches = 0;
    await expect(
      load(runtime, { wasmPath: "/wasm", accelerator: "webgpu" }, async () => {
        fetches += 1;
        return new Uint8Array([0]);
      }),
    ).rejects.toBeInstanceOf(RuntimeInitializationError);
    expect(fetches).toBe(0);
    expect(runtime.compiled).toEqual([]);
  });

  test("surfaces a caller abort from a direct model download", async () => {
    const controller = new AbortController();
    const runtime = createRuntime();
    let observedSignal: AbortSignal | undefined;
    const pending = load(
      runtime,
      { wasmPath: "/wasm", accelerator: "wasm", signal: controller.signal },
      (_url, signal) => {
        observedSignal = signal;
        return new Promise<Uint8Array<ArrayBuffer>>((_resolve, reject) => {
          signal?.addEventListener(
            "abort",
            () =>
              reject(signal.reason ?? new DOMException("The operation was aborted.", "AbortError")),
            { once: true },
          );
        });
      },
    );
    await Bun.sleep(0);
    expect(observedSignal).toBeDefined();
    controller.abort();
    let caught: unknown;
    try {
      await pending;
    } catch (error: unknown) {
      caught = error;
    }
    expect(caught).toMatchObject({ name: "AbortError" });
    expect(caught).not.toBeInstanceOf(RuntimeInitializationError);
    expect(observedSignal?.aborted).toBe(true);
  });

  test("does not fetch direct model bytes for a pre-aborted caller", async () => {
    const controller = new AbortController();
    controller.abort();
    let fetches = 0;
    await expect(
      load(
        createRuntime(),
        { wasmPath: "/wasm", accelerator: "wasm", signal: controller.signal },
        async () => {
          fetches += 1;
          return new Uint8Array([0]);
        },
      ),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(fetches).toBe(0);
  });

  test("aborts an in-flight model download when initialization fails", async () => {
    const initializationError = new Error("initialization failed");
    const control = createRuntime();
    const runtime = {
      ...control.runtime,
      initialize: async () => {
        throw initializationError;
      },
    };
    let observedSignal: AbortSignal | undefined;
    const pending = loadWithDependencies(
      metadataUrl,
      { wasmPath: "/wasm", accelerator: "wasm" },
      runtime,
      async () => tfliteMetadata(),
      new RuntimeInitializer(),
      (_url, signal) => {
        observedSignal = signal;
        return new Promise<Uint8Array<ArrayBuffer>>((_resolve, reject) => {
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
        });
      },
    );
    let caught: unknown;
    try {
      await pending;
    } catch (error: unknown) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(RuntimeInitializationError);
    expect((caught as RuntimeInitializationError).causeValue).toBe(initializationError);
    expect(observedSignal?.aborted).toBe(true);
  });

  test("preserves both non-availability failures when auto fallback also fails", async () => {
    const runtime = createRuntime();
    runtime.failWebGpuCompilation();
    runtime.failWasmCompilation();
    let caught: unknown;
    try {
      await load(runtime);
    } catch (error: unknown) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(RuntimeInitializationError);
    if (!(caught instanceof RuntimeInitializationError)) {
      return;
    }
    expect(caught.causeValue).toBeInstanceOf(AggregateError);
    const aggregate = caught.causeValue as AggregateError;
    expect(aggregate.errors).toHaveLength(2);
    expect(aggregate.errors[0]).toBeInstanceOf(RuntimeInitializationError);
    expect(aggregate.errors[1]).toBeInstanceOf(RuntimeInitializationError);
  });

  test("shares same-config initialization and rejects conflicting wasm configuration", async () => {
    const runtime = createRuntime();
    const initializer = new RuntimeInitializer();
    await Promise.all([
      loadWithDependencies(
        metadataUrl,
        options,
        runtime.runtime,
        async () => tfliteMetadata(),
        initializer,
      ),
      loadWithDependencies(
        metadataUrl,
        options,
        runtime.runtime,
        async () => tfliteMetadata(),
        initializer,
      ),
    ]);
    expect(runtime.initialized).toEqual(["/wasm"]);
    await expect(
      loadWithDependencies(
        metadataUrl,
        { wasmPath: "/another-wasm", accelerator: "wasm" },
        runtime.runtime,
        async () => tfliteMetadata(),
        initializer,
      ),
    ).rejects.toBeInstanceOf(RuntimeConfigurationError);
  });

  test("overlaps metadata loading and initialization before compiling", async () => {
    const runtime = createRuntime();
    runtime.holdInitialization();
    const initializer = new RuntimeInitializer();
    let metadataStarted = false;
    let resolveMetadata: ((metadata: Record<string, unknown>) => void) | undefined;
    const metadata = new Promise<Record<string, unknown>>((resolve) => {
      resolveMetadata = resolve;
    });
    const loading = loadWithDependencies(
      metadataUrl,
      options,
      runtime.runtime,
      async () => {
        metadataStarted = true;
        return metadata;
      },
      initializer,
    );

    await Bun.sleep(0);
    expect(metadataStarted).toBe(true);
    expect(runtime.initialized).toEqual([]);
    expect(runtime.compiled).toHaveLength(0);

    resolveMetadata?.(tfliteMetadata());
    await Bun.sleep(0);
    expect(runtime.initialized).toEqual(["/wasm"]);
    runtime.releaseInitialization();
    await loading;
    expect(runtime.compiled).toEqual(["webgpu"]);
  });

  test("does not poison a shared initializer when metadata validation fails", async () => {
    const initializer = new RuntimeInitializer();
    const invalidRuntime = createRuntime();
    await expect(
      loadWithDependencies(
        metadataUrl,
        { wasmPath: "/wasm-a", accelerator: "wasm" },
        invalidRuntime.runtime,
        async () => tfliteMetadata({ preprocessing: [{ type: "Normalize" }] }),
        initializer,
      ),
    ).rejects.toBeInstanceOf(XybridError);
    expect(invalidRuntime.initialized).toHaveLength(0);

    const validRuntime = createRuntime();
    await expect(
      loadWithDependencies(
        metadataUrl,
        { wasmPath: "/wasm-b", accelerator: "wasm" },
        validRuntime.runtime,
        async () => tfliteMetadata(),
        initializer,
      ),
    ).resolves.toBeDefined();
    expect(validRuntime.initialized).toEqual(["/wasm-b"]);
  });

  test("validates positional and named typed tensor inputs before allocating LiteRT tensors", async () => {
    const { model, runtime } = await load();
    await expect(model.run([tensor([1, 2]), tensor([3, 4, 5])])).resolves.toMatchObject({
      byName: { Identity: { data: new Float32Array([1, 2]) } },
    });
    await expect(model.run({ a: tensor([1, 2]), b: tensor([3, 4, 5]) })).resolves.toBeDefined();
    const allocatedBeforeInvalidRuns = runtime.tensors.length;
    await expect(model.run({ a: tensor([1, 2]) })).rejects.toBeInstanceOf(InputValidationError);
    await expect(model.run([new Int32Array([1, 2]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      InputValidationError,
    );
    await expect(model.run([tensor([1]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      InputValidationError,
    );
    expect(runtime.tensors).toHaveLength(allocatedBeforeInvalidRuns);
    expect(runtime.tensors.filter((value) => value.isDeleted())).not.toHaveLength(0);
  });

  test("deep-freezes metadata snapshots without freezing runtime-owned descriptors", async () => {
    const { model } = await load();
    const input = model.inputs[0];
    if (input === undefined) {
      throw new Error("test runtime must expose an input descriptor");
    }

    expect(Object.isFrozen(details.inputs[0])).toBe(false);
    expect(Object.isFrozen(details.inputs[0]?.shape)).toBe(false);
    expect(Object.isFrozen(model.inputs)).toBe(true);
    expect(Object.isFrozen(input)).toBe(true);
    expect(Object.isFrozen(input.shape)).toBe(true);

    const mutableInput = input as unknown as { name: string; shape: number[] };
    expect(() => {
      mutableInput.name = "mutated";
    }).toThrow(TypeError);
    expect(() => {
      mutableInput.shape[0] = 99;
    }).toThrow(TypeError);

    await expect(model.run([tensor([1, 2]), tensor([3, 4, 5])])).resolves.toBeDefined();
    expect(input.name).toBe("a");
    expect(input.shape).toEqual([1, 2]);
  });

  test("returns typed validation errors for malformed JavaScript inputs", async () => {
    const { model } = await load();
    await expect(Reflect.apply(model.run, model, [null])).rejects.toBeInstanceOf(
      InputValidationError,
    );
    await expect(
      Reflect.apply(model.run, model, [[new DataView(new ArrayBuffer(8)), tensor([3, 4, 5])]]),
    ).rejects.toBeInstanceOf(InputValidationError);
    await expect(
      model.run({ a: tensor([1, 2]), b: { data: new Float32Array(0), shape: [1, 2 ** 53] } }),
    ).rejects.toThrow("non-negative integer dimensions");
  });

  test("copies outputs and deletes every tensor on success and backend failure", async () => {
    const { model, runtime } = await load();
    const result = await model.run([tensor([1, 2]), tensor([3, 4, 5])]);
    expect(result.outputs[0]?.data).toEqual(new Float32Array([1, 2]));
    expect(runtime.tensors.every((value) => value.isDeleted())).toBe(true);

    runtime.failRun();
    await expect(model.run([tensor([1, 2]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      InferenceError,
    );
    expect(runtime.tensors.every((value) => value.isDeleted())).toBe(true);
  });

  test("returns model-controlled output names as own properties", async () => {
    const runtime = createRuntime();
    runtime.setOutputName("__proto__");
    const { model } = await load(runtime);
    const result = await model.run([tensor([1, 2]), tensor([3, 4, 5])]);
    expect(Object.hasOwn(result.byName, "__proto__")).toBe(true);
    expect(Object.getPrototypeOf(result.byName)).toBe(Object.prototype);
  });

  test("uses runtime output shapes and rejects oversized outputs before reading", async () => {
    const dynamic = createRuntime();
    dynamic.setOutputShape([1, 1]);
    const { model: dynamicModel } = await load(dynamic);
    const result = await dynamicModel.run([tensor([1, 2]), tensor([3, 4, 5])]);
    expect(result.outputs[0]?.shape).toEqual([1, 1]);

    const oversized = createRuntime();
    oversized.setOutputShape([1, 67_108_865]);
    const { model: oversizedModel } = await load(oversized);
    await expect(oversizedModel.run([tensor([1, 2]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      InferenceError,
    );
    expect(oversized.tensors.at(-1)?.reads()).toBe(0);
  });

  test("cleans partial allocations and rejects incomplete runtime outputs", async () => {
    const partial = createRuntime();
    const { model: partialModel } = await load(partial);
    partial.failTensorCreationAt(2);
    await expect(partialModel.run([tensor([1, 2]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      InferenceError,
    );
    expect(partial.tensors).toHaveLength(1);
    expect(partial.tensors[0]?.isDeleted()).toBe(true);

    const incomplete = createRuntime();
    const { model: incompleteModel } = await load(incomplete);
    incomplete.omitOutputs();
    await expect(incompleteModel.run([tensor([1, 2]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      InferenceError,
    );
    expect(incomplete.tensors.every((value) => value.isDeleted())).toBe(true);
  });

  test("rejects overlapping calls, device loss, and runs after disposal", async () => {
    const { model, runtime } = await load();
    const first = model.run([tensor([1, 2]), tensor([3, 4, 5])]);
    await expect(model.run([tensor([1, 2]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      ConcurrentRunError,
    );
    await first;

    runtime.loseDevice();
    await expect(model.run([tensor([1, 2]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      DeviceLostError,
    );
    await model.dispose();
    await model.dispose();
    expect(runtime.deletedModels()).toBe(1);
    await expect(model.run([tensor([1, 2]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      DisposedError,
    );
  });

  test("waits for an in-flight run before deleting once and rejects new work", async () => {
    const { model, runtime } = await load();
    runtime.holdRun();
    const running = model.run([tensor([1, 2]), tensor([3, 4, 5])]);
    const disposing = model.dispose();
    await expect(model.run([tensor([1, 2]), tensor([3, 4, 5])])).rejects.toBeInstanceOf(
      DisposedError,
    );
    expect(runtime.deletedModels()).toBe(0);
    runtime.releaseRun();
    await running;
    await disposing;
    expect(runtime.deletedModels()).toBe(1);
  });

  test("disposes the model after an in-flight run rejects", async () => {
    const { model, runtime } = await load();
    runtime.holdRun();
    runtime.failRun();
    const running = model.run([tensor([1, 2]), tensor([3, 4, 5])]);
    const disposing = model.dispose();
    runtime.releaseRun();
    await expect(running).rejects.toBeInstanceOf(InferenceError);
    await expect(disposing).resolves.toBeUndefined();
    expect(runtime.deletedModels()).toBe(1);
  });
});
