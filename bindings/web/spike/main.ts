import { RustGgufModel } from "./client.ts";
import { MODEL } from "./model.ts";
import type { Accelerator, Memory } from "./protocol.ts";

const element = <T extends HTMLElement>(id: string): T => {
  const found = document.getElementById(id);
  if (found === null) throw new Error(`Missing #${id}`);
  return found as T;
};
const loadButton = element<HTMLButtonElement>("load");
const runButton = element<HTMLButtonElement>("run");
const stopButton = element<HTMLButtonElement>("stop");
const disposeButton = element<HTMLButtonElement>("dispose");
const status = element("status");
const output = element("output");
const progress = element<HTMLProgressElement>("progress");
let model: RustGgufModel | undefined;
const report: Record<string, unknown> = { model: MODEL, userAgent: navigator.userAgent };
const showReport = (): void => {
  element("report").textContent = JSON.stringify(report, null, 2);
};
const showMemory = (memory: Memory): void => {
  element("heap").textContent = `${(memory.heapBytes / 1048576).toFixed(1)} MiB`;
  element("allocated").textContent = `${(memory.allocatedBytes / 1048576).toFixed(1)} MiB`;
};
const setStatus = (state: string, text: string): void => {
  status.dataset["state"] = state;
  status.textContent = text;
};

loadButton.onclick = () => {
  loadButton.disabled = true;
  progress.hidden = false;
  setStatus("loading", "Downloading and verifying the GGUF…");
  void RustGgufModel.load({
    accelerator: element<HTMLSelectElement>("accelerator").value as Accelerator | "auto",
    onDownloadProgress: (loaded, total) => {
      progress.value = (loaded / total) * 100;
    },
  })
    .then((loaded) => {
      model = loaded;
      report["load"] = loaded.loaded;
      showReport();
      showMemory(loaded.loaded.memory);
      setStatus(
        "ready",
        `Ready · ${loaded.loaded.accelerator} · Rust ${loaded.loaded.rustVersion}${loaded.loaded.adapter ? ` · ${loaded.loaded.adapter}` : ""}`,
      );
      runButton.disabled = false;
      disposeButton.disabled = false;
    })
    .catch((error: unknown) => {
      setStatus("error", error instanceof Error ? error.message : String(error));
      loadButton.disabled = false;
    })
    .finally(() => {
      progress.hidden = true;
    });
};

runButton.onclick = () => {
  const current = model;
  if (current === undefined) return;
  output.textContent = "";
  runButton.disabled = true;
  disposeButton.disabled = true;
  stopButton.disabled = false;
  setStatus("generating", "Generating locally…");
  void (async () => {
    try {
      for await (const delta of current.generateStream(
        element<HTMLTextAreaElement>("prompt").value,
        {
          maxTokens: Number(element<HTMLInputElement>("max-tokens").value),
        },
      ))
        output.textContent += delta;
      const metrics = current.lastRun;
      if (metrics !== undefined) {
        report["run"] = metrics;
        showReport();
        showMemory(metrics.memory);
        element("first-token").textContent =
          metrics.firstTokenMs === null ? "—" : `${metrics.firstTokenMs.toFixed(0)} ms`;
        element("speed").textContent =
          metrics.decodeTokensPerSecond === null
            ? "—"
            : `${metrics.decodeTokensPerSecond.toFixed(1)} tok/s`;
        setStatus(
          metrics.cancelled ? "cancelled" : "done",
          `${metrics.cancelled ? "Stopped" : "Done"} · ${metrics.generatedTokens} tokens · ${metrics.totalMs.toFixed(0)} ms · Rust ${current.loaded.rustVersion}`,
        );
      }
    } catch (error) {
      setStatus("error", error instanceof Error ? error.message : String(error));
    } finally {
      runButton.disabled = false;
      disposeButton.disabled = false;
      stopButton.disabled = true;
    }
  })();
};

stopButton.onclick = () => {
  stopButton.disabled = true;
  setStatus("stopping", "Stopping generation…");
  void model?.cancel().catch((error: unknown) => {
    setStatus("error", String(error));
  });
};

disposeButton.onclick = () => {
  const current = model;
  if (current === undefined) return;
  disposeButton.disabled = true;
  runButton.disabled = true;
  void current
    .dispose()
    .then((memory) => {
      report["disposed"] = memory;
      showReport();
      showMemory(memory);
      model = undefined;
      loadButton.disabled = false;
      setStatus("disposed", "Model released through Rust RAII. Ready to load again.");
    })
    .catch((error: unknown) => {
      setStatus("error", String(error));
    });
};
