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
} from "../src/errors.ts";
import { RuntimeInitializer } from "../src/internal/initialization.ts";
import { XybridModel } from "../src/model.ts";
import type { LoadOptions } from "../src/types.ts";
import { createRuntime, loadWithDependencies, tensor, tfliteMetadata } from "./helpers.ts";

const metadataUrl = new URL("https://models.example/add/model_metadata.json");
const options: LoadOptions = { wasmPath: "/wasm", accelerator: "auto" };

const load = async (runtime = createRuntime(), accelerator: LoadOptions = options) => ({
  runtime,
  model: await loadWithDependencies(metadataUrl, accelerator, runtime.runtime, async () =>
    tfliteMetadata(),
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

  test("falls back from WebGPU to wasm only for auto", async () => {
    const runtime = createRuntime();
    runtime.failWebGpu();
    const { model } = await load(runtime);
    expect(model.accelerator).toBe("wasm");
    expect(runtime.compiled).toEqual(["webgpu", "wasm"]);

    const explicit = createRuntime();
    explicit.failWebGpu();
    await expect(
      load(explicit, { wasmPath: "/wasm", accelerator: "webgpu" }),
    ).rejects.toBeInstanceOf(RuntimeInitializationError);
    expect(explicit.compiled).toEqual(["webgpu"]);

    const wasm = createRuntime();
    await load(wasm, { wasmPath: "/wasm", accelerator: "wasm" });
    expect(wasm.compiled).toEqual(["wasm"]);

    const compilationFailure = createRuntime();
    compilationFailure.failWebGpuCompilation();
    await expect(load(compilationFailure)).rejects.toBeInstanceOf(RuntimeInitializationError);
    expect(compilationFailure.compiled).toEqual(["webgpu"]);
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

  test("returns typed validation errors for malformed JavaScript inputs", async () => {
    const { model } = await load();
    await expect(Reflect.apply(model.run, model, [null])).rejects.toBeInstanceOf(
      InputValidationError,
    );
    await expect(
      Reflect.apply(model.run, model, [[new DataView(new ArrayBuffer(8)), tensor([3, 4, 5])]]),
    ).rejects.toBeInstanceOf(InputValidationError);
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
