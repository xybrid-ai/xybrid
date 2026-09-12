import type { ModelInfo, OutputType } from '../src/types';

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends
  (<T>() => T extends B ? 1 : 2) ? true : false;
type Expect<T extends true> = T;

type OutputTypeContract = Expect<
  Equal<OutputType, 'text' | 'audio' | 'embedding' | 'unknown'>
>;

const everyOutputType: OutputType[] = [
  'text',
  'audio',
  'embedding',
  'unknown',
];

const completeSnapshot: ModelInfo = {
  modelId: 'fixture-model',
  version: '1.2.3',
  outputType: everyOutputType[0],
  isLoaded: true,
  supportsStreaming: true,
  supportsTokenStreaming: true,
  isLlm: true,
};

void (null as unknown as OutputTypeContract);
void completeSnapshot;
