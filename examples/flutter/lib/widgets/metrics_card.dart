// Reusable card that renders the typed metrics surfaced on every
// [XybridResult]. The card auto-hides when no metric has a value —
// ASR/TTS/embedding runs (which don't emit LLM-specific fields and
// don't carry stage latencies) produce an empty section, so showing
// nothing is the intended UX.

import 'package:flutter/material.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

class MetricsCard extends StatelessWidget {
  const MetricsCard({super.key, required this.metrics});

  final XybridInferenceMetrics metrics;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final rows = <_MetricRow>[];

    final ttft = metrics.ttftMs;
    if (ttft != null) {
      rows.add(_MetricRow('TTFT', '$ttft ms'));
    }
    final tps = metrics.tokensPerSecond;
    if (tps != null) {
      rows.add(_MetricRow('Throughput', '${tps.toStringAsFixed(1)} tok/s'));
    }
    final prefill = metrics.prefillTps;
    if (prefill != null) {
      rows.add(_MetricRow('Prefill', '${prefill.toStringAsFixed(1)} tok/s'));
    }
    final decode = metrics.decodeTps;
    if (decode != null) {
      rows.add(_MetricRow('Decode', '${decode.toStringAsFixed(1)} tok/s'));
    }
    final tokensOut = metrics.tokensOut;
    if (tokensOut != null) {
      rows.add(_MetricRow('Tokens out', '$tokensOut'));
    }
    if (metrics.stageLatenciesMs.isNotEmpty) {
      final stages = metrics.stageLatenciesMs
          .map((s) => '${s.stageId}=${s.latencyMs}ms')
          .join(', ');
      rows.add(_MetricRow('Stages', stages));
    }

    if (rows.isEmpty) {
      return const SizedBox.shrink();
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withAlpha(120),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Metrics',
            style: theme.textTheme.labelLarge?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 6),
          for (final row in rows)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                children: [
                  Text(
                    row.label,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    row.value,
                    style: theme.textTheme.bodySmall?.copyWith(
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class _MetricRow {
  const _MetricRow(this.label, this.value);
  final String label;
  final String value;
}
