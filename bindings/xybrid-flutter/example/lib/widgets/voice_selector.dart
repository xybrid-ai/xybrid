import 'package:flutter/material.dart';
import 'package:flutter_piper/flutter_piper.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart' show FfiVoiceInfo;
import 'package:xybrid_flutter_example/xybrid_view_model.dart';

/// Voice selector widget for TTS models
///
/// Displays a dropdown of available voices when a TTS model is loaded.
/// Only visible when the loaded model has voices.
class VoiceSelector extends StatelessWidget {
  final XybridViewModel vm;

  const VoiceSelector({super.key, required this.vm});

  String _buildVoiceLabel(FfiVoiceInfo voice) {
    final parts = <String>[voice.name];

    if (voice.gender != null) {
      parts.add('(${voice.gender})');
    }

    if (voice.isDefault) {
      parts.add('[default]');
    }

    return parts.join(' ');
  }

  String _buildVoiceSubtitle(FfiVoiceInfo voice) {
    final parts = <String>[];

    if (voice.language != null) {
      parts.add(voice.language!);
    }

    if (voice.style != null) {
      parts.add(voice.style!);
    }

    parts.add('ID: ${voice.id}');

    return parts.join(' | ');
  }

  IconData _getGenderIcon(String? gender) {
    switch (gender?.toLowerCase()) {
      case 'male':
        return Icons.male;
      case 'female':
        return Icons.female;
      default:
        return Icons.record_voice_over;
    }
  }

  Color _getGenderColor(String? gender) {
    switch (gender?.toLowerCase()) {
      case 'male':
        return Colors.blue;
      case 'female':
        return Colors.pink;
      default:
        return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return vm.availableVoices.build((voices) {
      // Only show if voices are available
      if (voices.isEmpty) {
        return const SizedBox.shrink();
      }

      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.record_voice_over, size: 20),
                  const SizedBox(width: 8),
                  Text(
                    'Voice Selection (${voices.length} voices)',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              vm.selectedVoice.build((selected) {
                return DropdownButtonFormField<FfiVoiceInfo>(
                  initialValue: selected,
                  decoration: InputDecoration(
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                    prefixIcon: Icon(
                      _getGenderIcon(selected?.gender),
                      color: _getGenderColor(selected?.gender),
                    ),
                  ),
                  isExpanded: true,
                  items: voices.map((voice) {
                    return DropdownMenuItem<FfiVoiceInfo>(
                      value: voice,
                      child: Row(
                        children: [
                          Icon(
                            _getGenderIcon(voice.gender),
                            size: 18,
                            color: _getGenderColor(voice.gender),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Text(
                                  _buildVoiceLabel(voice),
                                  style: TextStyle(
                                    fontWeight: voice.isDefault
                                        ? FontWeight.bold
                                        : FontWeight.normal,
                                  ),
                                ),
                                Text(
                                  _buildVoiceSubtitle(voice),
                                  style: TextStyle(
                                    fontSize: 11,
                                    color: Colors.grey[600],
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    );
                  }).toList(),
                  onChanged: (voice) => vm.onVoiceChanged(voice),
                );
              }),
            ],
          ),
        ),
      );
    });
  }
}
