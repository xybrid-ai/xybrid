import type { GenerationConfig } from './types';

// Generation-config presets, mirroring `ai.xybrid.GenerationConfigs` in the
// Kotlin SDK 1:1 so a React Native caller reaches for the same named defaults
// as every other binding. Pure TS — no bridge hop.

/** Greedy decoding preset (deterministic, temperature 0). */
export function greedy(): GenerationConfig {
  return {
    temperature: 0.0,
    topP: 1.0,
    topK: 0,
    stopSequences: [],
  };
}

/** Creative generation preset (higher temperature). */
export function creative(): GenerationConfig {
  return {
    temperature: 0.9,
    topP: 0.95,
    topK: 50,
    stopSequences: [],
  };
}

/** Preset factory functions for {@link GenerationConfig}. */
export const GenerationConfigs = { greedy, creative };
