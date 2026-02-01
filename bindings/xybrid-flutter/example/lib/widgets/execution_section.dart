import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_piper/flutter_piper.dart';
import 'package:xybrid_flutter_example/xybrid_view_model.dart';

void _copyToClipboard(BuildContext context, String text) {
  Clipboard.setData(ClipboardData(text: text));
  ScaffoldMessenger.of(context).showSnackBar(
    const SnackBar(
      content: Text('Copied to clipboard'),
      duration: Duration(seconds: 1),
    ),
  );
}

class ExecutionSection extends StatelessWidget {
  final XybridViewModel vm;

  const ExecutionSection({super.key, required this.vm});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text(
                  'Execution',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const Spacer(),
                // Benchmark mode toggle
                vm.benchmarkMode.build((isBenchmark) {
                  return Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.speed,
                        size: 16,
                        color: isBenchmark ? Colors.orange : Colors.grey,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        'Benchmark',
                        style: TextStyle(
                          fontSize: 12,
                          color: isBenchmark ? Colors.orange : Colors.grey,
                        ),
                      ),
                      Switch(
                        value: isBenchmark,
                        onChanged: (_) => vm.toggleBenchmarkMode(),
                        activeColor: Colors.orange,
                      ),
                    ],
                  );
                }),
              ],
            ),
            // Benchmark settings (when enabled)
            vm.benchmarkMode.build((isBenchmark) {
              if (!isBenchmark) return const SizedBox.shrink();
              return _BenchmarkSettings(vm: vm);
            }),
            const SizedBox(height: 12),
            // Execute button
            Row(
              children: [
                Expanded(
                  child: vm.execution.build((exec) {
                    return vm.benchmarkMode.build((isBenchmark) {
                      return ElevatedButton.icon(
                        onPressed: exec.isExecuting ? null : () => vm.execute(),
                        icon: exec.isExecuting
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : Icon(isBenchmark ? Icons.speed : Icons.play_arrow),
                        label: Text(isBenchmark ? 'Run Benchmark' : 'Execute'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor:
                              isBenchmark ? Colors.orange : Colors.green,
                          foregroundColor: Colors.white,
                        ),
                      );
                    });
                  }),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _ExecutionLog(vm: vm),
            _ExecutionResult(vm: vm),
            // Benchmark results (when available)
            vm.benchmarkStatus.build((status) {
              if (status.result == null) return const SizedBox.shrink();
              return _BenchmarkResultCard(status: status);
            }),
          ],
        ),
      ),
    );
  }
}

class _BenchmarkSettings extends StatelessWidget {
  final XybridViewModel vm;

  const _BenchmarkSettings({required this.vm});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.orange.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.orange.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Benchmark Settings',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.bold,
              color: Colors.orange,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: vm.benchmarkIterations.build((iterations) {
                  return _IterationField(
                    label: 'Iterations',
                    value: iterations,
                    onChanged: vm.setBenchmarkIterations,
                    min: 1,
                    max: 100,
                  );
                }),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: vm.warmupIterations.build((warmup) {
                  return _IterationField(
                    label: 'Warmup',
                    value: warmup,
                    onChanged: vm.setWarmupIterations,
                    min: 0,
                    max: 20,
                  );
                }),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _IterationField extends StatelessWidget {
  final String label;
  final int value;
  final ValueChanged<int> onChanged;
  final int min;
  final int max;

  const _IterationField({
    required this.label,
    required this.value,
    required this.onChanged,
    required this.min,
    required this.max,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          '$label: ',
          style: const TextStyle(fontSize: 12),
        ),
        IconButton(
          icon: const Icon(Icons.remove, size: 16),
          onPressed: value > min ? () => onChanged(value - 1) : null,
          padding: EdgeInsets.zero,
          constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
        ),
        Container(
          width: 36,
          alignment: Alignment.center,
          child: Text(
            '$value',
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ),
        IconButton(
          icon: const Icon(Icons.add, size: 16),
          onPressed: value < max ? () => onChanged(value + 1) : null,
          padding: EdgeInsets.zero,
          constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
        ),
      ],
    );
  }
}

class _BenchmarkResultCard extends StatelessWidget {
  final BenchmarkStatus status;

  const _BenchmarkResultCard({required this.status});

  @override
  Widget build(BuildContext context) {
    final result = status.result;
    if (result == null) return const SizedBox.shrink();

    return Container(
      margin: const EdgeInsets.only(top: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.orange.withOpacity(0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.orange.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.analytics, size: 16, color: Colors.orange),
              const SizedBox(width: 8),
              Text(
                'Benchmark Results: ${result.modelId}',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                  color: Colors.orange,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'Execution Provider: ${result.executionProvider}',
            style: TextStyle(fontSize: 11, color: Colors.grey[600]),
          ),
          const SizedBox(height: 12),
          // Stats grid
          Row(
            children: [
              Expanded(
                child: _StatItem(
                  label: 'Mean',
                  value: '${result.meanMs.toStringAsFixed(2)}ms',
                  highlight: true,
                ),
              ),
              Expanded(
                child: _StatItem(
                  label: 'Median',
                  value: '${result.medianMs.toStringAsFixed(2)}ms',
                ),
              ),
              Expanded(
                child: _StatItem(
                  label: 'Std Dev',
                  value: '${result.stdDevMs.toStringAsFixed(2)}ms',
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _StatItem(
                  label: 'Min',
                  value: '${result.minMs.toStringAsFixed(2)}ms',
                ),
              ),
              Expanded(
                child: _StatItem(
                  label: 'Max',
                  value: '${result.maxMs.toStringAsFixed(2)}ms',
                ),
              ),
              Expanded(
                child: _StatItem(
                  label: 'P95',
                  value: '${result.p95Ms.toStringAsFixed(2)}ms',
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _StatItem(
                  label: 'Iterations',
                  value: '${result.iterations}',
                ),
              ),
              Expanded(
                child: _StatItem(
                  label: 'Total Time',
                  value: '${(result.totalTimeMs / 1000).toStringAsFixed(2)}s',
                ),
              ),
              const Expanded(child: SizedBox()),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final String label;
  final String value;
  final bool highlight;

  const _StatItem({
    required this.label,
    required this.value,
    this.highlight = false,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(fontSize: 10, color: Colors.grey[600]),
        ),
        Text(
          value,
          style: TextStyle(
            fontSize: 13,
            fontWeight: highlight ? FontWeight.bold : FontWeight.normal,
            color: highlight ? Colors.orange[800] : null,
          ),
        ),
      ],
    );
  }
}

class _ExecutionLog extends StatelessWidget {
  final XybridViewModel vm;

  const _ExecutionLog({required this.vm});

  @override
  Widget build(BuildContext context) {
    return vm.execution.build((exec) {
      return GestureDetector(
        onTap: () => _copyToClipboard(context, exec.log),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.grey[100],
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            exec.log.isEmpty ? 'Ready to execute' : exec.log,
            style: TextStyle(
              fontFamily: 'monospace',
              fontSize: 12,
              color: exec.log.contains('Error')
                  ? Colors.red[700]
                  : Colors.grey[800],
            ),
          ),
        ),
      );
    });
  }
}

class _ExecutionResult extends StatelessWidget {
  final XybridViewModel vm;

  const _ExecutionResult({required this.vm});

  @override
  Widget build(BuildContext context) {
    return vm.execution.build((exec) {
      if (exec.result.isEmpty) return const SizedBox.shrink();
      return Padding(
        padding: const EdgeInsets.only(top: 12),
        child: GestureDetector(
          onTap: () => _copyToClipboard(context, exec.result),
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.indigo[50],
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.indigo.shade200),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Text(
                      'Result',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: Colors.indigo,
                      ),
                    ),
                    const Spacer(),
                    Icon(Icons.copy, size: 14, color: Colors.indigo[300]),
                    const SizedBox(width: 4),
                    Text(
                      'Tap to copy',
                      style: TextStyle(fontSize: 10, color: Colors.indigo[300]),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                SelectableText(
                  exec.result,
                  style: const TextStyle(fontSize: 14),
                ),
              ],
            ),
          ),
        ),
      );
    });
  }
}
