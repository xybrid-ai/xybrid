import 'package:flutter/material.dart';
import 'package:flutter_piper/flutter_piper.dart';
import 'package:xybrid_flutter_example/xybrid_view_model.dart';

class AudioInput extends StatelessWidget {
  final XybridViewModel vm;

  const AudioInput({super.key, required this.vm});

  @override
  Widget build(BuildContext context) {
    return vm.streamingMode.build((isStreamingMode) {
      return Column(
        children: [
          _ModeSelector(vm: vm, isStreamingMode: isStreamingMode),
          const SizedBox(height: 12),
          if (isStreamingMode)
            _StreamingInput(vm: vm)
          else
            _BatchInput(vm: vm),
        ],
      );
    });
  }
}

class _ModeSelector extends StatelessWidget {
  final XybridViewModel vm;
  final bool isStreamingMode;

  const _ModeSelector({required this.vm, required this.isStreamingMode});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const Text('Mode:'),
        const SizedBox(width: 8),
        ChoiceChip(
          label: const Text('Batch'),
          selected: !isStreamingMode,
          onSelected: (_) {
            if (isStreamingMode) vm.toggleStreamingMode();
          },
        ),
        const SizedBox(width: 8),
        ChoiceChip(
          label: const Text('Streaming'),
          selected: isStreamingMode,
          onSelected: (_) {
            if (!isStreamingMode) vm.toggleStreamingMode();
          },
          avatar: isStreamingMode ? const Icon(Icons.stream, size: 16) : null,
        ),
      ],
    );
  }
}

class _BatchInput extends StatelessWidget {
  final XybridViewModel vm;

  const _BatchInput({required this.vm});

  @override
  Widget build(BuildContext context) {
    return vm.recording.build((rec) {
      return Column(
        children: [
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: rec.isRecording ? null : () => vm.startRecording(),
                  icon: const Icon(Icons.mic),
                  label: const Text('Record'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    foregroundColor: Colors.white,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: rec.isRecording ? () => vm.stopRecording() : null,
                  icon: const Icon(Icons.stop),
                  label: const Text('Stop'),
                ),
              ),
            ],
          ),
          if (rec.isRecording) ...[
            const SizedBox(height: 12),
            Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: const BoxDecoration(
                    color: Colors.red,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(child: _AmplitudeMeter(amplitude: rec.amplitude)),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              'Auto-stops on silence',
              style: TextStyle(fontSize: 11, color: Colors.grey[600]),
            ),
          ],
          if (rec.path != null && !rec.isRecording)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Row(
                children: const [
                  Icon(Icons.check, color: Colors.green, size: 16),
                  SizedBox(width: 4),
                  Text(
                    'Recording ready',
                    style: TextStyle(color: Colors.green),
                  ),
                ],
              ),
            ),
        ],
      );
    });
  }
}

class _StreamingInput extends StatelessWidget {
  final XybridViewModel vm;

  const _StreamingInput({required this.vm});

  @override
  Widget build(BuildContext context) {
    return vm.streaming.build((streamState) {
      final isStreaming = streamState.isStreaming;

      return Column(
        children: [
          _StreamingControls(vm: vm, isStreaming: isStreaming),
          if (isStreaming) ...[
            const SizedBox(height: 12),
            _StreamingIndicator(chunksProcessed: streamState.chunksProcessed),
          ],
          if (streamState.partialText.isNotEmpty) ...[
            const SizedBox(height: 12),
            _PartialTranscription(
              isStreaming: isStreaming,
              partialText: streamState.partialText,
            ),
          ],
        ],
      );
    });
  }
}

class _StreamingControls extends StatelessWidget {
  final XybridViewModel vm;
  final bool isStreaming;

  const _StreamingControls({required this.vm, required this.isStreaming});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: isStreaming ? null : () => vm.startStreaming(),
            icon: Icon(isStreaming ? Icons.mic : Icons.stream),
            label: Text(isStreaming ? 'Streaming...' : 'Start Streaming'),
            style: ElevatedButton.styleFrom(
              backgroundColor: isStreaming ? Colors.orange : Colors.blue,
              foregroundColor: Colors.white,
            ),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: isStreaming ? () => vm.stopStreaming() : null,
            icon: const Icon(Icons.stop),
            label: const Text('Stop'),
          ),
        ),
      ],
    );
  }
}

class _StreamingIndicator extends StatelessWidget {
  final int chunksProcessed;

  const _StreamingIndicator({required this.chunksProcessed});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: Colors.blue,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Colors.blue.withValues(alpha: 0.5),
                    blurRadius: 4,
                    spreadRadius: 2,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            const Text(
              'Real-time transcription active',
              style: TextStyle(color: Colors.blue),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Text(
          'Chunks: $chunksProcessed',
          style: TextStyle(fontSize: 11, color: Colors.grey[600]),
        ),
      ],
    );
  }
}

class _PartialTranscription extends StatelessWidget {
  final bool isStreaming;
  final String partialText;

  const _PartialTranscription({
    required this.isStreaming,
    required this.partialText,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.blue[50],
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.blue.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                isStreaming ? Icons.hearing : Icons.check_circle,
                size: 16,
                color: Colors.blue,
              ),
              const SizedBox(width: 4),
              Text(
                isStreaming ? 'Partial result:' : 'Final result:',
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          SelectableText(
            partialText,
            style: const TextStyle(fontSize: 14),
          ),
        ],
      ),
    );
  }
}

class _AmplitudeMeter extends StatelessWidget {
  final double amplitude;

  const _AmplitudeMeter({required this.amplitude});

  @override
  Widget build(BuildContext context) {
    // Normalize amplitude from dB (-60 to 0) to 0.0-1.0
    final normalized = ((amplitude + 60) / 60).clamp(0.0, 1.0);
    final isSilent = amplitude <= -40; // Silence threshold

    return Container(
      height: 20,
      decoration: BoxDecoration(
        color: Colors.grey[200],
        borderRadius: BorderRadius.circular(10),
      ),
      child: Stack(
        children: [
          FractionallySizedBox(
            widthFactor: normalized,
            child: Container(
              decoration: BoxDecoration(
                color: isSilent ? Colors.orange : Colors.green,
                borderRadius: BorderRadius.circular(10),
              ),
            ),
          ),
          Center(
            child: Text(
              isSilent ? 'Silence' : 'Speaking',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.bold,
                color: Colors.grey[700],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
