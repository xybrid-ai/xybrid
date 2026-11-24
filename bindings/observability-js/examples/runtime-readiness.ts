// Future integration examples: callbacks belong to the application. No runtime is installed
// or initialized here. These adapters work with an application-owned Observer and providers.
import type { Observer, StartOptions } from "@xybrid/observability";

type GeneratedText = {
  readonly text: string;
  readonly inputTokens?: number;
  readonly outputTokens?: number;
  readonly finishReason?: string;
};

/** Wrap a Transformers.js pipeline call; token counts must come from the runtime. */
export const observeTransformersGeneration = async (
  observer: Observer,
  model: string,
  prompt: string,
  generate: (prompt: string) => Promise<GeneratedText>,
  options?: StartOptions,
): Promise<GeneratedText> => {
  const op = observer.startInference(
    {
      operation: "text_completion",
      provider: "transformers.js",
      runtime: "transformers.js",
      model,
      location: "local",
    },
    options,
  );
  op.recordInputMessages([{ role: "user", parts: [{ type: "text", content: prompt }] }]);
  try {
    const result = await generate(prompt);
    op.recordUsage({ inputTokens: result.inputTokens, outputTokens: result.outputTokens });
    // Preserve unknown finish reasons instead of guessing from the returned text.
    if (result.finishReason !== undefined) {
      op.recordOutputMessages([
        {
          role: "assistant",
          parts: [{ type: "text", content: result.text }],
          finish_reason: result.finishReason,
        },
      ]);
    }
    op.end();
    return result;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") op.cancel();
    else op.endWithError({ errorType: "runtime_error" });
    throw error;
  }
};

/** Wrap LiteRT embedding inference; neither token usage nor token timing is inferred. */
export const observeLiteRtEmbeddings = async (
  observer: Observer,
  model: string,
  embed: () => Promise<readonly number[]>,
  options?: StartOptions,
): Promise<readonly number[]> => {
  const op = observer.startInference(
    {
      operation: "embeddings",
      provider: "litert",
      runtime: "litert",
      model,
      location: "local",
    },
    options,
  );
  try {
    const vector = await embed();
    op.end();
    return vector;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") op.cancel();
    else op.endWithError({ errorType: "runtime_error" });
    throw error;
  }
};

/** Wrap a Foundation Models text stream. Chunks are not measured model tokens. */
export async function* observeFoundationModelsStream(
  observer: Observer,
  model: string,
  stream: () => AsyncIterable<string>,
  options?: StartOptions,
): AsyncGenerator<string> {
  const op = observer.startInference(
    {
      operation: "chat",
      provider: "apple",
      runtime: "foundation_models",
      model,
      location: "local",
      stream: true,
    },
    options,
  );
  try {
    for await (const chunk of stream()) {
      op.recordChunk();
      yield chunk;
    }
    op.end();
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") op.cancel();
    else op.endWithError({ errorType: "runtime_error" });
    throw error;
  } finally {
    // Early iterator return (consumer abandonment) cancels; a terminal call is idempotent.
    op.cancel();
  }
}
