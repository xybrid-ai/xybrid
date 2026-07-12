import { mkdir, rm } from "node:fs/promises";
import ky from "ky";

const publicDirectory = new URL("../public/", import.meta.url);

type PinnedModel = {
  readonly url: string;
  readonly sha256: string;
  readonly bytes: number;
  readonly target: URL;
  readonly label: string;
};

const models: readonly PinnedModel[] = [
  {
    url: "https://raw.githubusercontent.com/google-ai-edge/LiteRT/9011640bbcbda69ec761b5c2d6ae2cdc8270b61d/litert/js/packages/core/testdata/add_10x10.tflite",
    sha256: "1317a76ceedc6e0a2b39c4ee2802f80b3b831b16ac96a99e48540113472aaee2",
    bytes: 708,
    target: new URL("model.tflite", publicDirectory),
    label: "LiteRT addition fixture",
  },
  {
    url: "https://huggingface.co/litert-community/SmolLM2-135M-Instruct/resolve/main/SmolLM2_135M_Instruct.litertlm",
    sha256: "ccdc5c85735743f081b7d44ca309cab569f76c0f2f0e8e163449a63721969c37",
    bytes: 142_819_328,
    target: new URL("llm/SmolLM2_135M_Instruct.litertlm", publicDirectory),
    label: "SmolLM2-135M-Instruct LiteRT-LM model (136 MB)",
  },
];

type WasmAssetSet = {
  readonly source: URL;
  readonly target: URL;
  readonly assets: readonly string[];
};

const wasmAssetSets: readonly WasmAssetSet[] = [
  {
    source: new URL("../../node_modules/@litertjs/core/wasm/", import.meta.url),
    target: new URL("litert/", publicDirectory),
    assets: [
      "litert_wasm_compat_internal.js",
      "litert_wasm_compat_internal.wasm",
      "litert_wasm_internal.js",
      "litert_wasm_internal.wasm",
    ],
  },
  {
    source: new URL("../../node_modules/@litert-lm/core/wasm/", import.meta.url),
    target: new URL("llm-runtime/", publicDirectory),
    assets: [
      "litertlm_wasm_asyncify_internal.js",
      "litertlm_wasm_asyncify_internal.wasm",
      "litertlm_wasm_compat_asyncify_internal.js",
      "litertlm_wasm_compat_asyncify_internal.wasm",
      "litertlm_wasm_compat_internal.js",
      "litertlm_wasm_compat_internal.wasm",
      "litertlm_wasm_internal.js",
      "litertlm_wasm_internal.wasm",
    ],
  },
];

class AssetPreparationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AssetPreparationError";
  }
}

const verifyModel = (model: PinnedModel, bytes: Uint8Array): void => {
  if (bytes.byteLength !== model.bytes) {
    throw new AssetPreparationError(
      `Expected ${model.bytes} bytes for the ${model.label}, received ${bytes.byteLength}.`,
    );
  }
  const digest = new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
  if (digest !== model.sha256) {
    throw new AssetPreparationError(
      `Downloaded ${model.label} checksum does not match the pinned artifact.`,
    );
  }
};

const downloadModel = async (model: PinnedModel): Promise<Uint8Array> => {
  console.log(`Downloading the ${model.label}...`);
  return new Uint8Array(await ky.get(model.url, { timeout: 600_000 }).arrayBuffer());
};

const modelBytes = async (model: PinnedModel): Promise<Uint8Array | undefined> => {
  const file = Bun.file(model.target);
  if (await file.exists()) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    try {
      verifyModel(model, bytes);
      return undefined;
    } catch (error: unknown) {
      if (!(error instanceof AssetPreparationError)) {
        throw error;
      }
    }
  }
  const bytes = await downloadModel(model);
  verifyModel(model, bytes);
  return bytes;
};

await mkdir(publicDirectory, { recursive: true });
await mkdir(new URL("llm/", publicDirectory), { recursive: true });
for (const model of models) {
  const bytes = await modelBytes(model);
  if (bytes !== undefined) {
    await Bun.write(model.target, bytes);
  }
}
for (const assetSet of wasmAssetSets) {
  await rm(assetSet.target, { recursive: true, force: true });
  await mkdir(assetSet.target, { recursive: true });
  await Promise.all(
    assetSet.assets.map((asset) =>
      Bun.write(new URL(asset, assetSet.target), Bun.file(new URL(asset, assetSet.source))),
    ),
  );
}
