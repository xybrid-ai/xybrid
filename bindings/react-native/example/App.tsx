import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useState } from 'react';
import {
  Button,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { Model, ModelLoader, Xybrid } from 'react-native-xybrid';

// Smoke test for the react-native-xybrid TurboModule: initialize → load a
// model from the registry → run one inference, timing each step. A failure
// in any layer (JS bridge, UniFFI marshal, Rust load/run) surfaces here with
// the underlying error code rather than a silent crash.
//
// NOTE: this requires an Expo *development build* (`expo run:ios` /
// `expo run:android`), NOT Expo Go — the module ships custom native code.

const DEFAULT_MODEL = 'whisper-tiny';

type Step = { label: string; durationMs: number; ok: boolean; detail?: string };

export default function App() {
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [steps, setSteps] = useState<Step[]>([]);
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState<Model | null>(null);

  useEffect(() => {
    // Free the native handle on unmount — loaded models hold weights in the
    // native heap.
    return () => {
      model?.release().catch(() => {});
    };
  }, [model]);

  const push = useCallback((s: Step) => setSteps((prev) => [...prev, s]), []);

  const run = useCallback(async () => {
    setBusy(true);
    setSteps([]);
    try {
      // Release the previously-loaded model BEFORE loading another. Models
      // hold hundreds of MB in the native heap; loading a second while the
      // first is still resident can OOM the device on repeat runs.
      if (model) {
        await model.release().catch(() => {});
        setModel(null);
      }

      await timed('Xybrid.initialize()', () => Xybrid.initialize(), push);

      const loaded = await timed(
        `ModelLoader.fromRegistry(${modelId}).load()`,
        () => ModelLoader.fromRegistry(modelId).load(),
        push,
      );
      setModel(loaded);

      if (await loaded.hasVoices()) {
        const voices = await timed('model.voices()', () => loaded.voices(), push);
        push({ label: `→ ${voices?.length ?? 0} voices`, durationMs: 0, ok: true });
      } else {
        const result = await timed(
          'model.run(text envelope)',
          () => loaded.run({ kind: 'text', text: 'hello from expo' }),
          push,
        );
        push({
          label: `→ success=${result.success}, latency=${result.latencyMs}ms`,
          durationMs: 0,
          ok: true,
          detail: result.text,
        });
      }
    } catch (err) {
      push({
        label: 'error',
        durationMs: 0,
        ok: false,
        detail: err instanceof Error ? `${err.name}: ${err.message}` : String(err),
      });
    } finally {
      setBusy(false);
    }
  }, [modelId, model, push]);

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={styles.title}>react-native-xybrid</Text>
        <Text style={styles.subtitle}>
          Expo dev-build smoke test (init → load → run).
        </Text>

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

        <Button title={busy ? 'Running…' : 'Run smoke test'} onPress={run} disabled={busy} />

        <View style={styles.steps}>
          {steps.map((s, i) => (
            <View key={i} style={styles.step}>
              <Text style={[styles.stepLabel, !s.ok && styles.stepError]}>
                {s.ok ? '✓' : '✗'} {s.label}
                {s.durationMs > 0 ? `  (${s.durationMs.toFixed(0)} ms)` : ''}
              </Text>
              {s.detail ? <Text style={styles.stepDetail}>{s.detail}</Text> : null}
            </View>
          ))}
        </View>
      </ScrollView>
      <StatusBar style="light" />
    </SafeAreaView>
  );
}

async function timed<T>(
  label: string,
  fn: () => Promise<T>,
  push: (s: Step) => void,
): Promise<T> {
  const t0 = Date.now();
  try {
    const out = await fn();
    push({ label, durationMs: Date.now() - t0, ok: true });
    return out;
  } catch (err) {
    push({
      label,
      durationMs: Date.now() - t0,
      ok: false,
      detail: err instanceof Error ? err.message : String(err),
    });
    throw err;
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
