import 'package:flutter/material.dart';
import 'package:flutter_piper/flutter_piper.dart';
import 'package:xybrid_flutter_example/xybrid_view_model.dart';

class StatusSection extends StatelessWidget {
  final XybridViewModel vm;

  const StatusSection({super.key, required this.vm});

  @override
  Widget build(BuildContext context) {
    return vm.loadingState.build((state) {
      final isLoading = state == LoadingState.loading;
      final isError = state == LoadingState.error;
      final isLoaded = state == LoadingState.loaded;

      return Card(
        color: isError
            ? Colors.red[50]
            : isLoaded
                ? Colors.green[50]
                : null,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (isLoading)
                const Padding(
                  padding: EdgeInsets.only(bottom: 12),
                  child: LinearProgressIndicator(),
                ),
              Row(
                children: [
                  if (isError)
                    const Icon(Icons.error, color: Colors.red, size: 20)
                  else if (isLoaded)
                    const Icon(
                      Icons.check_circle,
                      color: Colors.green,
                      size: 20,
                    ),
                  if (isError || isLoaded) const SizedBox(width: 8),
                  Expanded(
                    child: vm.statusMessage.build(
                      (msg) => Text(
                        msg,
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: isError ? Colors.red[700] : null,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
              vm.errorMessage.build((error) {
                if (error == null) return const SizedBox.shrink();
                return Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    error,
                    style: TextStyle(fontSize: 12, color: Colors.red[700]),
                  ),
                );
              }),
              vm.modelStatuses.build((statuses) {
                if (statuses.isEmpty) return const SizedBox.shrink();
                return Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Column(
                    children: statuses
                        .map((m) => _ModelStatusRow(status: m))
                        .toList(),
                  ),
                );
              }),
            ],
          ),
        ),
      );
    });
  }
}

class _ModelStatusRow extends StatelessWidget {
  final ModelStatus status;

  const _ModelStatusRow({required this.status});

  @override
  Widget build(BuildContext context) {
    final m = status;
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          Icon(
            m.status.contains('loaded') || m.status.contains('ready')
                ? Icons.check_circle_outline
                : m.status.contains('failed')
                    ? Icons.error_outline
                    : Icons.hourglass_empty,
            size: 16,
            color: m.status.contains('loaded') || m.status.contains('ready')
                ? Colors.green
                : m.status.contains('failed')
                    ? Colors.red
                    : Colors.orange,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              '${m.name}: ${m.status}',
              style: const TextStyle(fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}
