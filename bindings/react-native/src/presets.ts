import type { GenerationConfig, ToolDefinition } from './types';

// Generation-config presets with the same values as the Swift and Kotlin
// `GenerationConfigs`. Pure TS — no native call.

/**
 * Greedy decoding (deterministic, temperature 0): the usual choice for
 * extraction and tool calling.
 */
export function greedy(
  options: { maxTokens?: number; grammar?: string; tools?: ToolDefinition[] } = {},
): GenerationConfig {
  return {
    temperature: 0.0,
    topP: 1.0,
    topK: 0,
    stopSequences: [],
    ...options,
  };
}

/** Higher temperature, for more varied output. */
export function creative(
  options: { maxTokens?: number; tools?: ToolDefinition[] } = {},
): GenerationConfig {
  return {
    temperature: 0.9,
    topP: 0.95,
    topK: 50,
    stopSequences: [],
    ...options,
  };
}

/** Preset factories for {@link GenerationConfig}. */
export const GenerationConfigs = { greedy, creative };
