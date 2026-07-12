import { mkdir, rm } from "node:fs/promises";
import ky from "ky";

const modelUrl =
  "https://raw.githubusercontent.com/google-ai-edge/LiteRT/9011640bbcbda69ec761b5c2d6ae2cdc8270b61d/litert/js/packages/core/testdata/add_10x10.tflite";
const expectedSha256 = "1317a76ceedc6e0a2b39c4ee2802f80b3b831b16ac96a99e48540113472aaee2";
const expectedBytes = 708;
const publicDirectory = new URL("../public/", import.meta.url);
const modelPath = new URL("model.tflite", publicDirectory);
const wasmSource = new URL("../../node_modules/@litertjs/core/wasm/", import.meta.url);
const wasmTarget = new URL("litert/", publicDirectory);
const wasmAssets = [
  "litert_wasm_compat_internal.js",
  "litert_wasm_compat_internal.wasm",
  "litert_wasm_internal.js",
  "litert_wasm_internal.wasm",
];

class AssetPreparationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AssetPreparationError";
  }
}

const verifyModel = (bytes: Uint8Array): void => {
  if (bytes.byteLength !== expectedBytes) {
    throw new AssetPreparationError(
      `Expected ${expectedBytes} model bytes, received ${bytes.byteLength}.`,
    );
  }
  const digest = new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
  if (digest !== expectedSha256) {
    throw new AssetPreparationError(
      "Downloaded model checksum does not match the pinned LiteRT fixture.",
    );
  }
};

const downloadModel = async (): Promise<Uint8Array> => {
  return new Uint8Array(await ky.get(modelUrl).arrayBuffer());
};

const modelBytes = async (): Promise<Uint8Array> => {
  const file = Bun.file(modelPath);
  if (await file.exists()) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    try {
      verifyModel(bytes);
      return bytes;
    } catch (error: unknown) {
      if (!(error instanceof AssetPreparationError)) {
        throw error;
      }
    }
  }
  const bytes = await downloadModel();
  verifyModel(bytes);
  return bytes;
};

await mkdir(publicDirectory, { recursive: true });
await rm(wasmTarget, { recursive: true, force: true });
await mkdir(wasmTarget, { recursive: true });
await Bun.write(modelPath, await modelBytes());
await Promise.all(
  wasmAssets.map((asset) =>
    Bun.write(new URL(asset, wasmTarget), Bun.file(new URL(asset, wasmSource))),
  ),
);
