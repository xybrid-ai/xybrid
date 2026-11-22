import { StatusBar } from 'expo-status-bar';
import { useEffect, useRef, useState } from 'react';
import { SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';

import { runQualification } from './qualify';
import { reportPath, type SmokeReport } from './report';

/**
 * Headless qualification UI: runs once on launch, shows the report path and
 * per-step results. CI never reads the screen — the host pulls the JSON file.
 */
export default function App() {
  const [report, setReport] = useState<SmokeReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    void (async () => {
      try {
        const result = await runQualification();
        setReport(result);
        console.log(
          `[xybrid-consumer-smoke] ${result.success ? 'ok' : 'FAIL'} runId=${result.runId} report=${reportPath()}`,
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        console.log(`[xybrid-consumer-smoke] FAIL ${message}`);
      }
    })();
  }, []);

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.body}>
        <Text style={styles.title}>@xybrid/react-native</Text>
        <Text style={styles.subtitle}>Published-package consumer smoke</Text>
        <Text style={styles.mono}>report: {reportPath()}</Text>
        {error ? <Text style={styles.fail}>{error}</Text> : null}
        {report ? (
          <View style={styles.steps}>
            <Text style={report.success ? styles.ok : styles.fail}>
              {report.success ? 'success' : 'failed'} · {report.runId}
            </Text>
            {report.results.map((step) => (
              <Text key={step.id} style={step.ok ? styles.ok : styles.fail}>
                {step.ok ? '✓' : '✗'} {step.id}
                {step.detail ? ` — ${step.detail}` : ''}
              </Text>
            ))}
          </View>
        ) : (
          <Text style={styles.mono}>{error ? 'stopped' : 'running…'}</Text>
        )}
      </ScrollView>
      <StatusBar style="light" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0f172a' },
  body: { padding: 20, gap: 12 },
  title: { color: '#f8fafc', fontSize: 22, fontWeight: '700' },
  subtitle: { color: '#94a3b8', fontSize: 14 },
  mono: { color: '#cbd5e1', fontFamily: 'Menlo', fontSize: 12 },
  steps: { gap: 6, marginTop: 8 },
  ok: { color: '#a7f3d0', fontFamily: 'Menlo', fontSize: 13 },
  fail: { color: '#fca5a5', fontFamily: 'Menlo', fontSize: 13 },
});
