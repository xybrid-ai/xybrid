import { type AcceleratorPreference, XybridModel } from "../../src/index.ts";

import "./style.css";

class ExampleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ExampleError";
  }
}

const requireElement = <T extends HTMLElement>(selector: string, kind: abstract new () => T): T => {
  const element = document.querySelector(selector);
  if (element instanceof kind) {
    return element;
  }
  throw new ExampleError(`The demo page is missing its ${selector} element.`);
};

const form = requireElement("#run-form", HTMLFormElement);
const accelerator = requireElement("#accelerator", HTMLSelectElement);
const acceleratorHint = requireElement("#accelerator-hint", HTMLParagraphElement);
const addend = requireElement("#addend", HTMLInputElement);
const button = requireElement("#run-button", HTMLButtonElement);
const status = requireElement("#status", HTMLOutputElement);
const steps = requireElement("#steps", HTMLOListElement);
const facts = requireElement("#run-facts", HTMLDivElement);
const factAccelerator = requireElement("#fact-accelerator", HTMLElement);
const factDelegated = requireElement("#fact-delegated", HTMLElement);
const factLoad = requireElement("#fact-load", HTMLElement);
const factRun = requireElement("#fact-run", HTMLElement);
const outputView = requireElement("#output-view", HTMLDivElement);
const codeSample = requireElement("#code-sample", HTMLElement);

let currentModel: XybridModel | undefined;
let currentPreference: AcceleratorPreference | undefined;

const setStatus = (
  state: "ready" | "loading" | "pass" | "error",
  title: string,
  detail: string,
): void => {
  status.dataset["state"] = state;
  status.replaceChildren(
    Object.assign(document.createElement("strong"), { textContent: title }),
    document.createTextNode(detail),
  );
};

type StepName = "load" | "run" | "verify";
type StepState = "idle" | "active" | "done" | "error";
const stepNames: readonly StepName[] = ["load", "run", "verify"];

const setStep = (name: StepName, state: StepState, note = ""): void => {
  const step = steps.querySelector(`[data-step="${name}"]`);
  if (!(step instanceof HTMLLIElement)) {
    return;
  }
  step.dataset["state"] = state;
  const noteTarget = step.querySelector("[data-note]");
  if (noteTarget !== null) {
    noteTarget.textContent = note;
  }
};

const resetSteps = (): void => {
  for (const name of stepNames) {
    setStep(name, "idle");
  }
};

const markActiveStepFailed = (): void => {
  for (const name of stepNames) {
    const step = steps.querySelector(`[data-step="${name}"]`);
    if (step instanceof HTMLLIElement && step.dataset["state"] === "active") {
      setStep(name, "error", "failed here");
    }
  }
};

const formatMs = (milliseconds: number): string => `${Math.max(1, Math.round(milliseconds))} ms`;

const preferredAccelerator = (): AcceleratorPreference => {
  switch (accelerator.value) {
    case "auto":
    case "wasm":
    case "webgpu":
      return accelerator.value;
    default:
      throw new ExampleError("The selected accelerator is not supported by this preview.");
  }
};

const addendValue = (): number => {
  const value = Number(addend.value);
  if (!Number.isInteger(value) || value < -1000 || value > 1000) {
    throw new ExampleError("b must be a whole number between -1000 and 1000.");
  }
  return value;
};

const getModel = async (
  preference: AcceleratorPreference,
): Promise<{ model: XybridModel; cached: boolean }> => {
  if (currentModel !== undefined && currentPreference === preference) {
    return { model: currentModel, cached: true };
  }
  await currentModel?.dispose();
  currentModel = undefined;
  currentPreference = undefined;
  const loaded = await XybridModel.load("/model_metadata.json", {
    wasmPath: "/litert",
    accelerator: preference,
  });
  currentModel = loaded;
  currentPreference = preference;
  return { model: loaded, cached: false };
};

const isExpectedSum = (actual: Float32Array, expected: Float32Array): boolean =>
  actual.length === expected.length && actual.every((value, index) => value === expected[index]);

const showEmptyOutput = (message: string): void => {
  const empty = document.createElement("p");
  empty.className = "output-empty";
  empty.textContent = message;
  outputView.replaceChildren(empty);
};

const showOutputGrid = (data: Float32Array, b: number): void => {
  const table = document.createElement("table");
  table.className = "grid";
  const caption = document.createElement("caption");
  caption.textContent = `Identity output, float32[10, 10] row-major. Each cell is its a value plus ${b}.`;
  table.append(caption);
  const body = document.createElement("tbody");
  for (let row = 0; row < 10; row += 1) {
    const tableRow = document.createElement("tr");
    for (let column = 0; column < 10; column += 1) {
      const index = row * 10 + column;
      const cell = document.createElement("td");
      cell.textContent = String(data[index] ?? "?");
      cell.title = `output[${index}] = a[${index}] (${index}) + b (${b})`;
      tableRow.append(cell);
    }
    body.append(tableRow);
  }
  table.append(body);
  const scroller = document.createElement("div");
  scroller.className = "grid-scroll";
  scroller.append(table);
  outputView.replaceChildren(scroller);
};

const showFacts = (model: XybridModel, cached: boolean, loadMs: number, runMs: number): void => {
  factAccelerator.textContent = model.accelerator;
  factDelegated.textContent = model.isFullyAccelerated
    ? `yes, every operation ran on ${model.accelerator}`
    : "partial, some operations fell back to the CPU";
  factLoad.textContent = cached ? "cached from the previous run" : formatMs(loadMs);
  factRun.textContent = formatMs(runMs);
  facts.hidden = false;
};

const updateCodeSample = (): void => {
  const b = addend.value === "" ? "10" : addend.value;
  codeSample.textContent = `import { XybridModel } from "@xybrid/web";

const model = await XybridModel.load("/model_metadata.json", {
  wasmPath: "/litert", // LiteRT wasm assets served by this site
  accelerator: "${accelerator.value}",
});

const a = Float32Array.from({ length: 100 }, (_, i) => i);
const b = new Float32Array(100).fill(${b});

const { byName } = await model.run({ a, b });
console.log(byName["Identity"]?.data); // Float32Array(100): a[i] + ${b}

await model.dispose();`;
};

if (!("gpu" in navigator)) {
  const webgpuOption = accelerator.querySelector('option[value="webgpu"]');
  if (webgpuOption instanceof HTMLOptionElement) {
    webgpuOption.textContent = "webgpu (not detected in this browser)";
  }
  acceleratorHint.textContent =
    "This browser reports no WebGPU support, so auto will compile for wasm.";
}

accelerator.addEventListener("change", () => {
  updateCodeSample();
  if (accelerator.value !== currentPreference) {
    button.textContent = "Load model and run";
  }
});
addend.addEventListener("input", updateCodeSample);
updateCodeSample();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  accelerator.disabled = true;
  addend.disabled = true;
  facts.hidden = true;
  resetSteps();
  showEmptyOutput("Running: the output values will appear here.");
  try {
    const preference = preferredAccelerator();
    const b = addendValue();
    setStatus(
      "loading",
      "RUNNING",
      `Loading the model and executing with the ${preference} preference.`,
    );

    setStep("load", "active");
    const loadStart = performance.now();
    const { model, cached } = await getModel(preference);
    const loadMs = performance.now() - loadStart;
    setStep("load", "done", cached ? "cached" : formatMs(loadMs));

    setStep("run", "active");
    const a = Float32Array.from({ length: 100 }, (_, index) => index);
    const bTensor = new Float32Array(100).fill(b);
    const runStart = performance.now();
    const result = await model.run({ a, b: bTensor });
    const runMs = performance.now() - runStart;
    setStep("run", "done", formatMs(runMs));

    setStep("verify", "active");
    const output = result.byName["Identity"];
    if (output === undefined || !(output.data instanceof Float32Array)) {
      throw new ExampleError(
        "The deterministic model did not return the expected float32 Identity output.",
      );
    }
    const expected = Float32Array.from(a, (value) => value + b);
    if (!isExpectedSum(output.data, expected)) {
      throw new ExampleError("Model output did not equal elementwise a+b.");
    }
    setStep("verify", "done", "100 of 100 exact");

    showFacts(model, cached, loadMs, runMs);
    showOutputGrid(output.data, b);
    setStatus(
      "pass",
      "PASS",
      `100 float32 values matched a+b via the ${model.accelerator} compile path.`,
    );
    button.textContent = "Run again";
  } catch (error: unknown) {
    markActiveStepFailed();
    showEmptyOutput("The run stopped before producing outputs.");
    const detail = error instanceof Error ? error.message : "Unknown failure.";
    setStatus("error", "ERROR", detail);
  } finally {
    button.disabled = false;
    accelerator.disabled = false;
    addend.disabled = false;
  }
});
