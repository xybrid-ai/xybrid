// The step 6 staging scenario (tools/scripts/observability_staging/README.md)
// through the public @xybrid/react-native API, on the TurboModule of a real
// device: two sessions in one launch, `direct` (the Xybrid preset, metadata
// only) and `collector` (OTLP to the pinned Collector, content captured with
// one literal redaction), each loading a registry model under a remote parent,
// stopping one stream after three chunks, streaming a second to the end, and
// recording feedback and an evaluation on the run reference. It mirrors
// bindings/python/tests/observability_staging.py.
//
// A lane copies config.json (it holds the staging key) to
// <documents>/xybrid-staging/ and launches the app, which then runs this
// instead of the smoke test (App.tsx). The scenario deletes the config as soon
// as it has read it, never logs or writes the key, and writes
// expectations.json next to it, also when a step fails (`scenario_error`).

import { Directory, File, Paths } from 'expo-file-system';
import { Platform, Settings } from 'react-native';

import {
  Envelope,
  ModelLoader,
  Observability,
  ObservabilityConfig,
  WorkflowObservation,
  Xybrid,
  isXybridError,
  type InferenceResult,
  type Model,
  type ObservationContext,
  type ObservationRef,
  type RunOptions,
  type SignalStats,
} from '@xybrid/react-native';

const SCHEMA = 'xybrid-staging-scenario/1';
const SESSIONS = ['direct', 'collector'] as const;

type Session = (typeof SESSIONS)[number];
type Json = { [key: string]: unknown };
type Reference = { trace_id: string; span_id: string } | null;

/** What the scenario reads from config.json (written by run_staging.py). */
interface StagingConfig {
  binding: string;
  lane: string;
  run: string;
  commit: string;
  model_id: string;
  sessions: {
    direct: { traces_endpoint: string; api_key: string; trace_id: string; parent_span_id: string };
    collector: { endpoint: string; trace_id: string; parent_span_id: string };
  };
  constants: {
    per_lane: { service_name: string; service_version: string; device_id: string; conversation_id: string };
    workflow_name: string;
    stopped: { prompt: string; max_tokens: number; stop_after_chunks: number };
    completed: { prompt: string; max_tokens: number };
    secret: string;
    replacement: string;
    feedback: { score: number; label: string; comment: string };
    evaluation: { name: string; score_value: number; score_label: string; explanation: string };
    timeout_ms: number;
  };
}

function reference(value: ObservationRef | null | undefined): Reference {
  return value ? { trace_id: value.trace_id, span_id: value.span_id } : null;
}

function counters({ accepted, exported, dropped, failed, pending }: SignalStats): Json {
  return { accepted, exported, dropped, failed, pending };
}

async function snapshot(): Promise<Json> {
  const stats = await Observability.stats();
  return {
    enabled: stats.enabled,
    traces: counters(stats.traces),
    logs: counters(stats.logs),
    metrics: counters(stats.metrics),
  };
}

function describe(error: unknown): string {
  if (isXybridError(error)) return `${error.code}: ${error.message}`;
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

/** `ok`, or `error: <code>: <message>` when the SDK rejects the call. */
async function outcome(call: () => Promise<unknown>): Promise<string> {
  try {
    await call();
    return 'ok';
  } catch (error) {
    if (!isXybridError(error)) throw error;
    return `error: ${describe(error)}`;
  }
}

/** Python's `str.format` for staging.json's `{name}` templates. */
function format(template: string, values: { [name: string]: string }): string {
  return template.replace(/\{(\w+)\}/g, (field, name: string) => values[name] ?? field);
}

function runOptions(maxTokens: number, context: ObservationContext, conversationId: string): RunOptions {
  return {
    generationConfig: { maxTokens },
    traceparent: context.traceparent,
    tracestate: context.tracestate,
    conversationId,
  };
}

/** Step 1: the session's export, identity and content rules, then init. */
async function configure(config: StagingConfig, session: Session): Promise<void> {
  const { binding, lane, constants } = config;
  let observability: ObservabilityConfig;
  if (session === 'direct') {
    const { api_key, traces_endpoint } = config.sessions.direct;
    observability = ObservabilityConfig.xybrid(api_key)
      .setTracesEndpoint(traces_endpoint)
      .setMetadataOnly(true);
  } else {
    observability = ObservabilityConfig.otlp(config.sessions.collector.endpoint)
      .addLiteralRedaction(constants.secret, constants.replacement);
  }
  const perLane = constants.per_lane;
  observability
    .setService(format(perLane.service_name, { binding }), perLane.service_version)
    .setDeviceId(format(perLane.device_id, { lane }));
  const acceptance = { run: config.run, lane, commit: config.commit, session };
  for (const [key, value] of Object.entries(acceptance)) {
    observability.setResourceAttribute(`xybrid.acceptance.${key}`, value);
  }
  await observability.init();
}

interface Streamed {
  outcome: string;
  chunks: number;
  result: InferenceResult | null;
  finishReason: string | null;
}

/**
 * Steps 4 and 5: a pull stream (the native SDK buffers tokens ahead of each
 * `next()`), stopped after `stopAfter` chunks or read to the end.
 */
async function stream(
  model: Model,
  prompt: string,
  options: RunOptions,
  stopAfter: number | null,
): Promise<Streamed> {
  const tokens = model.runStreaming(Envelope.text(prompt), options);
  let chunks = 0;
  let finishReason: string | null = null;
  for (;;) {
    const step = await tokens.next();
    if (step.done) {
      // `undefined` only when the native stream was disposed underneath us.
      if (step.value === undefined) throw new Error('the stream ended without a result');
      return { outcome: stopAfter === null ? 'ok' : 'completed', chunks, result: step.value, finishReason };
    }
    chunks += 1;
    finishReason = step.value.finishReason ?? finishReason;
    if (stopAfter !== null && chunks >= stopAfter) {
      // Ending the iteration (what `break` does) disposes the native stream
      // before its terminal event, which cancels the run at its next token (a
      // user cancellation, error.type cancelled).
      await tokens.return(undefined);
      return { outcome: 'cancelled', chunks, result: null, finishReason: null };
    }
  }
}

async function runSession(config: StagingConfig, session: Session, record: Json): Promise<void> {
  const { constants } = config;
  const values = config.sessions[session];
  const timeoutMs = constants.timeout_ms;
  const conversationId = format(constants.per_lane.conversation_id, { run: config.run, session });
  const references: { [name: string]: Reference } = { workflow: null, load: null, run: null };
  Object.assign(record, { trace_id: values.trace_id, parent_span_id: values.parent_span_id, references });
  record.init = await outcome(() => configure(config, session));
  if (record.init !== 'ok') throw new Error(`${session} init failed: ${record.init}`);

  // Step 2: the workflow continues the driver's trace T under span P.
  const workflow = await WorkflowObservation.start(constants.workflow_name, {
    traceparent: `00-${values.trace_id}-${values.parent_span_id}-01`,
  });
  try {
    references.workflow = reference(await workflow.observationRef());
    const context = await workflow.context();
    if (!context) throw new Error('the workflow has no context');

    // Step 3: a real registry model, loaded under the workflow.
    const model = await ModelLoader.fromRegistry(config.model_id)
      .withObservation({ traceparent: context.traceparent, tracestate: context.tracestate, conversationId })
      .load();
    try {
      references.load = reference((await model.info()).loadObservationRef);

      // Step 4: stopped after three chunks by ending the iteration.
      const { stopped, completed } = constants;
      let result = await stream(
        model,
        stopped.prompt,
        runOptions(stopped.max_tokens, context, conversationId),
        stopped.stop_after_chunks,
      );
      record.stopped = { outcome: result.outcome, chunks_consumed: result.chunks, delivery: 'pull' };

      // Step 5: streamed to the end; the result carries the run reference.
      result = await stream(
        model,
        completed.prompt,
        runOptions(completed.max_tokens, context, conversationId),
        null,
      );
      const final = result.result;
      if (final === null) throw new Error('the completed stream returned no result');
      record.completed = {
        outcome: result.outcome,
        chunks: result.chunks,
        tokens_out: final.metrics.tokensOut ?? null,
        text_chars: [...(final.text ?? '')].length,
        finish_reason: result.finishReason,
      };
      const run = final.observation;
      references.run = reference(run);
      if (!run) throw new Error('the completed run returned no observation reference');

      // Step 6: quality results on the run reference.
      const { feedback, evaluation } = constants;
      record.quality = {
        feedback: await Observability.recordFeedback(run, {
          score: feedback.score,
          label: feedback.label,
          comment: feedback.comment,
        }),
        evaluation: await Observability.recordEvaluation(run, {
          name: evaluation.name,
          scoreValue: evaluation.score_value,
          scoreLabel: evaluation.score_label,
          explanation: evaluation.explanation,
        }),
      };
    } finally {
      await model.release();
    }

    // Step 7: lifecycle.
    await workflow.complete();
  } finally {
    await workflow.dispose();
  }
  const lifecycle: Json = { flush: await outcome(() => Observability.flush(timeoutMs)) };
  const stats: Json = { after_flush: await snapshot() };
  lifecycle.shutdown = await outcome(() => Observability.shutdown(timeoutMs));
  stats.after_shutdown = await snapshot();
  Object.assign(record, { lifecycle, stats });
}

async function runStaging(directory: Directory, source: File, report: (status: string) => void) {
  const sessions: { [name: string]: Json } = {};
  const expectations: Json = {
    schema: SCHEMA,
    binding: null,
    lane: null,
    run: null,
    sdk_version: null,
    model_id: null,
    device: { system: Platform.OS, release: String(Platform.Version) },
    sessions,
    scenario_error: null,
  };
  let timeoutMs = 5000;
  try {
    let text: string;
    try {
      text = await source.text();
    } finally {
      source.delete(); // it holds the staging key; nothing reads it again
    }
    let config: StagingConfig;
    try {
      config = JSON.parse(text) as StagingConfig;
    } catch {
      throw new Error('config.json is not valid JSON'); // never echo its text
    }
    timeoutMs = config.constants.timeout_ms;
    Object.assign(expectations, {
      binding: config.binding,
      lane: config.lane,
      run: config.run,
      sdk_version: await Xybrid.version(),
      model_id: config.model_id,
    });
    for (const session of SESSIONS) {
      report(`${config.run}: ${session} session`);
      sessions[session] = {};
      await runSession(config, session, sessions[session]);
    }
  } catch (error) {
    // The error, then its stack frames (the stack's first line repeats the message).
    const frames = error instanceof Error ? (error.stack ?? '').split('\n').slice(1, 9) : [];
    expectations.scenario_error = [describe(error), ...frames].join('\n');
    await Observability.shutdown(timeoutMs).catch(() => {});
  }
  try {
    const temporary = new File(directory, 'expectations.json.tmp');
    temporary.write(`${JSON.stringify(expectations, null, 2)}\n`);
    temporary.moveSync(new File(directory, 'expectations.json'), { overwrite: true });
  } catch (error) {
    report(`could not write expectations.json: ${describe(error)}`);
    return;
  }
  const summary = Object.fromEntries(
    Object.entries(sessions).map(([name, { references, stopped, completed, quality }]) => [
      name,
      { references, stopped, completed, quality },
    ]),
  );
  console.log(`[xybrid-staging] ${JSON.stringify(summary)}`);
  const error = expectations.scenario_error as string | null;
  report(`${error ? `failed: ${error.split('\n')[0]}` : 'done'}; expectations.json written`);
}

/** iOS launch argument `-XybridStagingPrepare YES` (NSUserDefaults' argument domain). */
function prepareRequested(): boolean {
  try {
    return Platform.OS === 'ios' && Boolean(Settings.get('XybridStagingPrepare'));
  } catch {
    return false;
  }
}

/**
 * Runs the staging scenario when a lane left
 * `<documents>/xybrid-staging/config.json`, reporting a one-line status
 * through `onStatus`. Returns false, touching nothing, when there is no config
 * and the app was not launched to prepare the folder.
 */
export function runStagingIfRequested(onStatus: (status: string) => void): boolean {
  const directory = new Directory(Paths.document, 'xybrid-staging');
  const source = new File(directory, 'config.json');
  const report = (status: string) => {
    console.log(`[xybrid-staging] ${status}`);
    onStatus(`staging: ${status}`);
  };
  if (!source.exists) {
    // devicectl may not create the folder on a fresh install: the iPhone lane
    // then launches the app once with the prepare argument to create it.
    if (!prepareRequested()) return false;
    try {
      directory.create({ intermediates: true, idempotent: true });
      report('folder ready for a config');
    } catch (error) {
      report(`could not create the folder: ${describe(error)}`);
    }
    return true;
  }
  report('starting');
  runStaging(directory, source, report).catch((error) => report(`failed: ${describe(error)}`));
  return true;
}
