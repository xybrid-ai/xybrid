import { File } from 'expo-file-system';
import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Button,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import {
  ConversationContext,
  Envelope,
  GenerationConfigs,
  ModelLoader,
  Xybrid,
  bytesToBase64,
  isXybridError,
  jsonSchemaToGbnf,
  type Model,
} from 'react-native-xybrid';

import { decodeAudio, resample, toWavBase64, transcribeLive } from './speech';

// Smoke test for react-native-xybrid: runs once on launch (and again on the
// button) through the SDK surface an app touches — version, load, model info,
// warmup, run, streaming with a stop button, a two-turn conversation,
// structured output, cache status — timing every step. Each step is also
// logged with a `[xybrid-smoke]` prefix so a device run can be checked from
// `adb logcat` / the simulator log without tapping anything.
//
// Needs an Expo development build (`expo run:ios` / `expo run:android`), not
// Expo Go: the package ships native code.

// A registry id, a local model directory / .gguf file (plain path or file://
// URL), or `none` to skip the LLM steps. Set EXPO_PUBLIC_XYBRID_SMOKE_MODEL
// when starting Metro, e.g. to a model pushed over adb to an offline device.
const DEFAULT_MODEL = process.env.EXPO_PUBLIC_XYBRID_SMOKE_MODEL || 'lfm2.5-230m';

// Optional speech round trip after the LLM steps: "<audio source>,<asr>" where
// the source is a TTS model (registry id or local path) or a .wav file on the
// device, e.g. EXPO_PUBLIC_XYBRID_SMOKE_SPEECH=kitten-tts-nano-0.8,whisper-tiny-ggml
const SPEECH_MODELS = process.env.EXPO_PUBLIC_XYBRID_SMOKE_SPEECH?.split(',').map((part: string) => part.trim());

function loaderFor(model: string): ModelLoader {
  if (!model.startsWith('/') && !model.startsWith('file://')) return ModelLoader.fromRegistry(model);
  return model.endsWith('.gguf') ? ModelLoader.fromModelFile(model) : ModelLoader.fromDirectory(model);
}

type Step = { label: string; durationMs: number; ok: boolean; detail?: string };

export default function App() {
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [steps, setSteps] = useState<Step[]>([]);
  const [busy, setBusy] = useState(false);
  const started = useRef(false);

  const push = useCallback((step: Step) => {
    console.log(
      `[xybrid-smoke] ${step.ok ? 'ok' : 'FAIL'} ${step.label}` +
        (step.durationMs ? ` (${step.durationMs} ms)` : '') +
        (step.detail ? ` — ${step.detail}` : ''),
    );
    setSteps((previous) => [...previous, step]);
  }, []);

  const run = useCallback(async () => {
    setBusy(true);
    setSteps([]);
    let model: Model | null = null;
    try {
      const version = await timed('Xybrid.version()', () => Xybrid.version(), push);
      push({ label: `→ SDK ${version}`, durationMs: 0, ok: true });

      // "none" skips the LLM steps (e.g. to iterate on the speech round trip).
      if (modelId === 'none') {
        if (SPEECH_MODELS?.length === 2) await speechRoundTrip(SPEECH_MODELS[0], SPEECH_MODELS[1], push);
        return;
      }

      model = await timed(`load ${modelId}`, () => loaderFor(modelId).load(), push);
      const info = await timed('model.info()', () => model!.info(), push);
      push({
        label: `→ ${info.modelId} (${info.outputType}${info.isLlm ? ', LLM' : ''})`,
        durationMs: 0,
        ok: true,
      });
      if (!info.isLlm) {
        push({ label: '→ not an LLM: skipping the text steps', durationMs: 0, ok: true });
        return;
      }

      await timed('model.warmup()', () => model!.warmup(), push);

      const result = await timed(
        'model.run()',
        () =>
          model!.run(Envelope.text('Name one colour of the sea.'), {
            generationConfig: GenerationConfigs.greedy({ maxTokens: 24 }),
          }),
        push,
      );
      const { metrics } = result;
      const decode = metrics.decodeTps === undefined ? '?' : metrics.decodeTps.toFixed(1);
      const stages = metrics.stageLatenciesMs.map((s) => `${s.stageId} ${s.latencyMs}`).join(', ');
      push({
        label:
          `→ ${metrics.tokensOut ?? '?'} tokens, ttft ${metrics.ttftMs ?? '?'} ms, ` +
          `decode ${decode} tok/s, native ${metrics.totalMs} ms` +
          (stages ? ` [${stages}]` : ''),
        durationMs: 0,
        ok: true,
        detail: result.text,
      });

      // Stop button: abort the stream after five tokens.
      const controller = new AbortController();
      let streamed = '';
      let tokens = 0;
      const startedAt = Date.now();
      try {
        for await (const token of model.runStreaming(Envelope.text('Count from one to fifty.'), {
          generationConfig: { maxTokens: 128 },
          signal: controller.signal,
        })) {
          streamed += token.token;
          tokens += 1;
          if (tokens === 5) controller.abort();
        }
      } catch (error) {
        // Cancelling mid-stream may surface as `xybrid_cancelled`.
        if (!(isXybridError(error) && error.code === 'xybrid_cancelled')) throw error;
      }
      push({
        label: `model.runStreaming() stopped after ${tokens} tokens`,
        durationMs: Date.now() - startedAt,
        ok: tokens >= 5 && tokens < 20,
        detail: streamed,
      });

      const chat = await ConversationContext.create();
      try {
        const question = Envelope.user('My name is Ada. Reply with one word: ok.');
        const first = await timed(
          'conversation turn 1',
          () => model!.run(question, { context: chat, generationConfig: { maxTokens: 16 } }),
          push,
        );
        await chat.push(question);
        await chat.push(first.envelope);
        const second = await timed(
          'conversation turn 2',
          () =>
            model!.run(Envelope.user('What is my name?'), {
              context: chat,
              generationConfig: { maxTokens: 16 },
            }),
          push,
        );
        const history = await chat.info();
        push({
          label: `→ history ${history.historyLength} turns`,
          durationMs: 0,
          ok: history.historyLength === 2,
          detail: second.text,
        });
      } finally {
        await chat.release();
      }

      const grammar = await timed(
        'jsonSchemaToGbnf()',
        () =>
          jsonSchemaToGbnf({
            type: 'object',
            properties: { colour: { type: 'string' } },
            required: ['colour'],
          }),
        push,
      );
      const structured = await timed(
        'structured output',
        () =>
          model!.run(Envelope.text('Give a colour as JSON.'), {
            generationConfig: { grammar, maxTokens: 32 },
          }),
        push,
      );
      let parsed = false;
      try {
        JSON.parse(structured.text ?? '');
        parsed = true;
      } catch {
        // reported below
      }
      push({ label: '→ output parses as JSON', durationMs: 0, ok: parsed, detail: structured.text });

      if (SPEECH_MODELS?.length === 2) await speechRoundTrip(SPEECH_MODELS[0], SPEECH_MODELS[1], push);

      const cache = await timed('Xybrid.modelCacheStatus()', () => Xybrid.modelCacheStatus(), push);
      push({
        label: `→ ${cache.modelCount} models, ${(cache.totalSizeBytes / 1e6).toFixed(0)} MB`,
        durationMs: 0,
        ok: true,
      });
    } catch (error) {
      push({
        label: 'error',
        durationMs: 0,
        ok: false,
        detail: isXybridError(error) ? `${error.code}: ${error.message}` : String(error),
      });
    } finally {
      // Weights live in the native heap: always release.
      await model?.release().catch(() => {});
      console.log('[xybrid-smoke] done');
      setBusy(false);
    }
  }, [modelId, push]);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    void run();
  }, [run]);

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={styles.title}>react-native-xybrid</Text>
        <Text style={styles.subtitle}>Smoke test (runs on launch).</Text>

        <View style={styles.row}>
          <Text style={styles.label}>Model ID</Text>
          <TextInput
            style={styles.input}
            value={modelId}
            onChangeText={setModelId}
            autoCapitalize="none"
            autoCorrect={false}
            editable={!busy}
          />
        </View>

        <Button title={busy ? 'Running…' : 'Run again'} onPress={run} disabled={busy} />

        <View style={styles.steps}>
          {steps.map((step, index) => (
            <View key={index} style={styles.step}>
              <Text style={[styles.stepLabel, !step.ok && styles.stepError]}>
                {step.ok ? '✓' : '✗'} {step.label}
                {step.durationMs > 0 ? `  (${step.durationMs.toFixed(0)} ms)` : ''}
              </Text>
              {step.detail ? <Text style={styles.stepDetail}>{step.detail}</Text> : null}
            </View>
          ))}
        </View>
      </ScrollView>
      <StatusBar style="light" />
    </SafeAreaView>
  );
}

async function speechRoundTrip(source: string, asrSource: string, push: (step: Step) => void) {
  const asr = await timed(`load ${asrSource}`, () => loaderFor(asrSource).load(), push);
  try {
    const wavBase64 = source.endsWith('.wav')
      ? await timed(`read ${source}`, async () => bytesToBase64(await new File(source.startsWith('/') ? `file://${source}` : source).bytes()), push)
      : await synthesize(source, push);
    const audio = decodeAudio(wavBase64, 16000);

    const batch = await timed('asr.run()', () => asr.run(Envelope.audio(wavBase64)), push);
    push({ label: '→ batch transcript', durationMs: 0, ok: /hello/i.test(batch.text ?? ''), detail: batch.text });

    const live = await timed(
      'live ASR session',
      () => transcribeLive(asr, resample(audio.samples, audio.sampleRate, 16000)),
      push,
    );
    push({
      label: `→ live transcript (${live.partials} partials)`,
      durationMs: 0,
      ok: /hello/i.test(live.transcript),
      detail: live.transcript,
    });
  } finally {
    await asr.release();
  }
}

/** Speak a sentence and return it as a WAV (base64). */
async function synthesize(ttsSource: string, push: (step: Step) => void): Promise<string> {
  const tts = await timed(`load ${ttsSource}`, () => loaderFor(ttsSource).load(), push);
  try {
    const voice = await timed('tts.defaultVoice()', () => tts.defaultVoice(), push);
    const speech = await timed(
      'tts.run()',
      () => tts.run(Envelope.text('Hello from React Native.', { voiceId: voice?.id })),
      push,
    );
    // ONNX TTS returns headerless 16-bit PCM without a sample-rate tag; Kitten
    // and Kokoro produce 24 kHz.
    const audio = decodeAudio(speech.audioBytesBase64 ?? '', 24000);
    push({
      label: `→ ${(audio.samples.length / audio.sampleRate).toFixed(2)} s of ${audio.wav ? 'WAV' : 'PCM16'} at ${audio.sampleRate} Hz`,
      durationMs: 0,
      ok: audio.samples.length > audio.sampleRate / 10,
    });
    return toWavBase64(audio.samples, audio.sampleRate);
  } finally {
    await tts.release();
  }
}

async function timed<T>(label: string, fn: () => Promise<T>, push: (step: Step) => void): Promise<T> {
  const startedAt = Date.now();
  try {
    const value = await fn();
    push({ label, durationMs: Date.now() - startedAt, ok: true });
    return value;
  } catch (error) {
    push({
      label,
      durationMs: Date.now() - startedAt,
      ok: false,
      detail: isXybridError(error) ? `${error.code}: ${error.message}` : String(error),
    });
    throw error;
  }
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0f172a' },
  body: { padding: 20, gap: 16 },
  title: { color: '#f8fafc', fontSize: 22, fontWeight: '700' },
  subtitle: { color: '#94a3b8', fontSize: 14 },
  row: { gap: 6 },
  label: { color: '#cbd5e1', fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.5 },
  input: {
    backgroundColor: '#1e293b',
    color: '#f8fafc',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 8,
    fontFamily: 'Menlo',
    fontSize: 14,
  },
  steps: { gap: 8, marginTop: 12 },
  step: { gap: 2 },
  stepLabel: { color: '#a7f3d0', fontFamily: 'Menlo', fontSize: 13 },
  stepError: { color: '#fca5a5' },
  stepDetail: { color: '#cbd5e1', fontFamily: 'Menlo', fontSize: 12, paddingLeft: 16 },
});
