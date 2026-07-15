import { type AcceleratorPreference, XybridLlm } from "../../src/index.ts";

import "./style.css";

const MAX_OUTPUT_TOKENS = 256;
const MODEL_METADATA_URL = "/llm/model_metadata.json";
const WASM_PATH = "/llm-runtime";

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
const prompt = requireElement("#prompt", HTMLTextAreaElement);
const button = requireElement("#run-button", HTMLButtonElement);
const status = requireElement("#status", HTMLOutputElement);
const steps = requireElement("#steps", HTMLOListElement);
const facts = requireElement("#run-facts", HTMLDivElement);
const factAccelerator = requireElement("#fact-accelerator", HTMLElement);
const factLoad = requireElement("#fact-load", HTMLElement);
const factFirstToken = requireElement("#fact-first-token", HTMLElement);
const factRun = requireElement("#fact-run", HTMLElement);
const outputView = requireElement("#output-view", HTMLDivElement);
const codeSample = requireElement("#code-sample", HTMLElement);

let currentLlm: XybridLlm | undefined;
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

type StepName = "load" | "generate";
type StepState = "idle" | "active" | "done" | "error";
const stepNames: readonly StepName[] = ["load", "generate"];

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

const formatSeconds = (milliseconds: number): string => `${(milliseconds / 1000).toFixed(1)} s`;

const formatMegabytes = (bytes: number): string => `${Math.round(bytes / (1024 * 1024))} MB`;

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

const promptValue = (): string => {
  const value = prompt.value.trim();
  if (value.length === 0) {
    throw new ExampleError("Enter a prompt before running the model.");
  }
  return value;
};

const getLlm = async (
  preference: AcceleratorPreference,
): Promise<{ llm: XybridLlm; cached: boolean }> => {
  if (currentLlm !== undefined && currentPreference === preference) {
    return { llm: currentLlm, cached: true };
  }
  await currentLlm?.dispose();
  currentLlm = undefined;
  currentPreference = undefined;
  const loaded = await XybridLlm.load(MODEL_METADATA_URL, {
    wasmPath: WASM_PATH,
    accelerator: preference,
    onDownloadProgress: ({ loadedBytes, totalBytes }) => {
      const total = totalBytes === undefined ? "" : ` of ${formatMegabytes(totalBytes)}`;
      setStep("load", "active", `${formatMegabytes(loadedBytes)}${total}`);
    },
  });
  currentLlm = loaded;
  currentPreference = preference;
  return { llm: loaded, cached: false };
};

const showEmptyOutput = (message: string): void => {
  const empty = document.createElement("p");
  empty.className = "output-empty";
  empty.textContent = message;
  outputView.replaceChildren(empty);
};

const createOutputStream = (): HTMLParagraphElement => {
  const stream = document.createElement("p");
  stream.className = "output-stream";
  outputView.replaceChildren(stream);
  return stream;
};

const showFacts = (
  llm: XybridLlm,
  cached: boolean,
  loadMs: number,
  firstTokenMs: number | undefined,
  runMs: number,
): void => {
  factAccelerator.textContent =
    llm.accelerator === "webgpu" ? "webgpu, on-device GPU engine" : "wasm, on-device CPU engine";
  factLoad.textContent = cached ? "cached from the previous run" : formatSeconds(loadMs);
  factFirstToken.textContent = firstTokenMs === undefined ? "-" : formatMs(firstTokenMs);
  factRun.textContent = formatSeconds(runMs);
  facts.hidden = false;
};

const updateCodeSample = (): void => {
  const promptForSample = (prompt.value.trim() || "Hello!").replace(/`/g, "'");
  codeSample.textContent = `import { XybridLlm } from "@xybrid/web";

const llm = await XybridLlm.load("${MODEL_METADATA_URL}", {
  wasmPath: "${WASM_PATH}", // engine wasm assets served by this site
  accelerator: "${accelerator.value}",
});

const stream = llm.generateStream(
  \`${promptForSample}\`,
  { maxOutputTokens: ${MAX_OUTPUT_TOKENS} },
);
for await (const delta of stream) {
  process(delta); // tokens arrive as they decode
}

await llm.dispose();`;
};

if (!("gpu" in navigator)) {
  const webgpuOption = accelerator.querySelector('option[value="webgpu"]');
  if (webgpuOption instanceof HTMLOptionElement) {
    webgpuOption.textContent = "webgpu (not detected in this browser)";
  }
  acceleratorHint.textContent =
    "This browser reports no WebGPU support, so auto will run on the CPU engine.";
}

accelerator.addEventListener("change", () => {
  updateCodeSample();
  if (accelerator.value !== currentPreference) {
    button.textContent = "Load model and generate";
  }
});
prompt.addEventListener("input", updateCodeSample);
updateCodeSample();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  accelerator.disabled = true;
  prompt.disabled = true;
  facts.hidden = true;
  resetSteps();
  showEmptyOutput("Running: the streamed reply will appear here.");
  try {
    const preference = preferredAccelerator();
    const userPrompt = promptValue();
    setStatus(
      "loading",
      "RUNNING",
      `Loading the model and generating with the ${preference} preference.`,
    );

    setStep("load", "active");
    const loadStart = performance.now();
    const { llm, cached } = await getLlm(preference);
    const loadMs = performance.now() - loadStart;
    setStep("load", "done", cached ? "cached" : formatSeconds(loadMs));

    setStep("generate", "active");
    const outputStream = createOutputStream();
    const generateStart = performance.now();
    let firstTokenMs: number | undefined;
    let text = "";
    for await (const delta of llm.generateStream(userPrompt, {
      maxOutputTokens: MAX_OUTPUT_TOKENS,
    })) {
      if (firstTokenMs === undefined) {
        firstTokenMs = performance.now() - generateStart;
      }
      text += delta;
      outputStream.textContent = text;
      setStep("generate", "active", `${text.length} chars`);
    }
    const runMs = performance.now() - generateStart;
    if (text.length === 0) {
      throw new ExampleError("The model returned no text; try a different prompt.");
    }
    setStep("generate", "done", formatSeconds(runMs));

    showFacts(llm, cached, loadMs, firstTokenMs, runMs);
    setStatus(
      "pass",
      "DONE",
      `Streamed ${text.length} characters from SmolLM2-135M on the ${llm.accelerator} engine.`,
    );
    button.textContent = "Generate again";
  } catch (error: unknown) {
    markActiveStepFailed();
    showEmptyOutput("The run stopped before producing a reply.");
    const detail = error instanceof Error ? error.message : "Unknown failure.";
    setStatus("error", "ERROR", detail);
  } finally {
    button.disabled = false;
    accelerator.disabled = false;
    prompt.disabled = false;
  }
});
