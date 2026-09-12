import type { ModelInfo, OutputType } from './types';

const OUTPUT_TYPES: ReadonlySet<string> = new Set([
  'text',
  'audio',
  'embedding',
  'unknown',
]);

function record(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError('Invalid model info: expected an object');
  }
  return value as Record<string, unknown>;
}

function stringField(value: unknown, name: string): string {
  if (typeof value !== 'string') {
    throw new TypeError(`Invalid model info: invalid ${name}`);
  }
  return value;
}

function booleanField(value: unknown, name: string): boolean {
  if (typeof value !== 'boolean') {
    throw new TypeError(`Invalid model info: invalid ${name}`);
  }
  return value;
}

/** Validate and decode the complete native model metadata snapshot. */
export function decodeModelInfo(value: unknown): ModelInfo {
  const valueRecord = record(value);
  const outputType = stringField(valueRecord.outputType, 'outputType');
  if (!OUTPUT_TYPES.has(outputType)) {
    throw new TypeError('Invalid model info: invalid outputType');
  }

  return {
    modelId: stringField(valueRecord.modelId, 'modelId'),
    version: stringField(valueRecord.version, 'version'),
    outputType: outputType as OutputType,
    isLoaded: booleanField(valueRecord.isLoaded, 'isLoaded'),
    supportsStreaming: booleanField(
      valueRecord.supportsStreaming,
      'supportsStreaming',
    ),
    supportsTokenStreaming: booleanField(
      valueRecord.supportsTokenStreaming,
      'supportsTokenStreaming',
    ),
    isLlm: booleanField(valueRecord.isLlm, 'isLlm'),
  };
}
