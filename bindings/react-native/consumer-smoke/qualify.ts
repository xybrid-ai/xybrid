/**
 * Published-package qualification: load FunctionGemma from the staged sandbox
 * directory, exercise the public SDK surface, and write a JSON report.
 *
 * Assertions use constrained outputs and structural checks — never prose
 * quality. Success is recorded only after model/conversation handles release.
 */

import { File, Paths } from 'expo-file-system';

import {
  ConversationContext,
  Envelope,
  GenerationConfigs,
  ModelLoader,
  Xybrid,
  isXybridError,
  jsonSchemaToGbnf,
  type Model,
} from '@xybrid/react-native';

import {
  REQUIRED_STEP_IDS,
  REPORT_SCHEMA_VERSION,
  newRunId,
  writeReport,
  type SmokeReport,
  type StepId,
  type StepResult,
} from './report';

// Host runners stage weights + model_metadata.json under this documents path.
export const MODEL_DIR_NAME = 'models/functiongemma-270m-it';

const NPM_PACKAGE_VERSION: string = require('@xybrid/react-native/package.json').version;
const REACT_NATIVE_VERSION: string = require('react-native/package.json').version;
const EXPO_VERSION: string = require('expo/package.json').version;

function modelDirectoryPath(): string {
  // Native loaders want a filesystem path; File.uri is file://…
  return new File(Paths.document, MODEL_DIR_NAME).uri.replace(/^file:\/\//, '');
}

export async function runQualification(): Promise<SmokeReport> {
  const startedAt = new Date().toISOString();
  const runId = newRunId();
  const results: StepResult[] = [];
  let nativeSdk = '';
  let model: Model | null = null;
  let fatal: string | undefined;

  const record = (id: StepId, ok: boolean, detail?: string, durationMs?: number) => {
    results.push({ id, ok, detail, durationMs });
  };

  try {
    nativeSdk = await Xybrid.version();
    record(
      'version_match',
      nativeSdk === NPM_PACKAGE_VERSION,
      `npm=${NPM_PACKAGE_VERSION} native=${nativeSdk}`,
    );
    if (nativeSdk !== NPM_PACKAGE_VERSION) {
      throw new Error(`native SDK ${nativeSdk} != npm package ${NPM_PACKAGE_VERSION}`);
    }

    const modelPath = modelDirectoryPath();
    const loadStarted = Date.now();
    model = await ModelLoader.fromDirectory(modelPath).load();
    record('model_load', true, modelPath, Date.now() - loadStarted);

    const infoStarted = Date.now();
    const info = await model.info();
    record(
      'model_info',
      Boolean(info.modelId) && info.isLlm === true,
      `${info.modelId} (${info.outputType})`,
      Date.now() - infoStarted,
    );
    if (!info.isLlm) throw new Error('staged model is not an LLM');

    const warmStarted = Date.now();
    await model.warmup();
    record('warmup', true, undefined, Date.now() - warmStarted);

    const inferStarted = Date.now();
    const batch = await model.run(Envelope.text('Reply with the single word: ok'), {
      generationConfig: GenerationConfigs.greedy({ maxTokens: 16 }),
    });
    const text = (batch.text ?? '').trim();
    record('inference', text.length > 0, text.slice(0, 80), Date.now() - inferStarted);
    if (!text) throw new Error('batch inference returned empty text');

    const streamStarted = Date.now();
    const stream = model.runStreaming(Envelope.text('Reply with the single word: ok'), {
      generationConfig: GenerationConfigs.greedy({ maxTokens: 16 }),
    });
    let streamed = '';
    let next = await stream.next();
    while (!next.done) {
      streamed += next.value.token;
      next = await stream.next();
    }
    const finalText = (next.value?.text ?? '').trim();
    const streamOk = streamed.trim().length > 0 && finalText.length > 0;
    record(
      'streaming',
      streamOk,
      `token text=${streamed.length} chars, final=${finalText.length} chars`,
      Date.now() - streamStarted,
    );
    if (!streamOk) throw new Error('streaming did not produce both token text and a final result');

    const streamController = new AbortController();
    const cancelStarted = Date.now();
    let cancelled = false;
    let cancelDetail = 'stream ended before it yielded a token';
    try {
      for await (const token of model.runStreaming(Envelope.text('Repeat the word one.'), {
        generationConfig: {
          grammar: `root ::= "${'one '.repeat(256)}one"`,
          maxTokens: 128,
        },
        signal: streamController.signal,
      })) {
        if (token.token && !streamController.signal.aborted) {
          streamController.abort();
          cancelDetail = 'stream completed normally after abort';
        }
      }
    } catch (error) {
      cancelled =
        streamController.signal.aborted && isXybridError(error) && error.code === 'xybrid_cancelled';
      if (!cancelled) throw error;
      cancelDetail = 'xybrid_cancelled';
    }
    record('cancel_settles', cancelled, cancelDetail, Date.now() - cancelStarted);
    if (!cancelled) throw new Error(cancelDetail);

    const postStarted = Date.now();
    const afterCancel = await model.run(Envelope.text('Reply with the single word: ok'), {
      generationConfig: GenerationConfigs.greedy({ maxTokens: 16 }),
    });
    const afterText = (afterCancel.text ?? '').trim();
    record(
      'post_cancel_inference',
      afterText.length > 0,
      afterText.slice(0, 80),
      Date.now() - postStarted,
    );
    if (!afterText) throw new Error('inference after cancel returned empty text');

    // Conversation: verify the context retains turns. Do not judge recall prose.
    const chat = await ConversationContext.create();
    try {
      const turn1 = Envelope.user('My name is Ada. Reply with one word: ok.');
      const first = await model.run(turn1, {
        context: chat,
        generationConfig: { maxTokens: 16 },
      });
      await chat.push(turn1);
      await chat.push(first.envelope);
      await model.run(Envelope.user('What is my name?'), {
        context: chat,
        generationConfig: { maxTokens: 16 },
      });
      const history = await chat.info();
      const conversationOk = history.historyLength === 2;
      record('conversation', conversationOk, `history=${history.historyLength}`);
      if (!conversationOk) throw new Error(`expected history length 2, got ${history.historyLength}`);
    } finally {
      await chat.release();
    }

    const grammar = await jsonSchemaToGbnf({
      type: 'object',
      properties: { colour: { type: 'string' } },
      required: ['colour'],
      additionalProperties: false,
    });
    const structured = await model.run(Envelope.text('Give a colour as JSON.'), {
      generationConfig: { grammar, maxTokens: 32 },
    });
    let colour: string | undefined;
    try {
      const parsed = JSON.parse(structured.text ?? '');
      colour = typeof parsed.colour === 'string' ? parsed.colour : undefined;
    } catch {
      colour = undefined;
    }
    const jsonOk = typeof colour === 'string' && colour.length > 0;
    record('constrained_json', jsonOk, structured.text?.slice(0, 80));
    if (!jsonOk) throw new Error('constrained JSON output missing colour string');
  } catch (error) {
    fatal = isXybridError(error) ? `${error.code}: ${error.message}` : String(error);
    const seen = new Set(results.map((r) => r.id));
    for (const id of REQUIRED_STEP_IDS) {
      // release is recorded in finally after real cleanup.
      if (id === 'release' || seen.has(id)) continue;
      record(id, false, fatal);
    }
  } finally {
    const releaseStarted = Date.now();
    try {
      await model?.release();
      record('release', true, model ? 'released' : 'nothing to release', Date.now() - releaseStarted);
    } catch (error) {
      const detail = isXybridError(error) ? `${error.code}: ${error.message}` : String(error);
      record('release', false, detail, Date.now() - releaseStarted);
      fatal = fatal ?? detail;
    }
  }

  const allRequiredOk =
    REQUIRED_STEP_IDS.every((id) => results.some((r) => r.id === id && r.ok)) && fatal === undefined;

  const report: SmokeReport = {
    schemaVersion: REPORT_SCHEMA_VERSION,
    runId,
    startedAt,
    finishedAt: new Date().toISOString(),
    // success only after cleanup (release recorded above).
    success: allRequiredOk && results.some((r) => r.id === 'release' && r.ok),
    versions: {
      npmPackage: NPM_PACKAGE_VERSION,
      nativeSdk: nativeSdk || 'unknown',
      expo: EXPO_VERSION,
      reactNative: REACT_NATIVE_VERSION,
    },
    results,
    ...(fatal ? { error: fatal } : {}),
  };

  await writeReport(report);
  return report;
}
