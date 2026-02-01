/// Audio utilities for Xybrid.
///
/// This module provides helpers for audio recording and processing.
library;

import 'dart:typed_data';

/// Wraps raw PCM audio bytes in a WAV header for playback.
///
/// TTS models typically output raw PCM bytes (16-bit signed integers).
/// This function adds the required WAV header for audio players.
///
/// Parameters:
/// - [pcmBytes]: Raw PCM audio data (16-bit signed, little-endian)
/// - [sampleRate]: Sample rate in Hz (e.g., 24000 for Kokoro TTS)
/// - [channels]: Number of audio channels (typically 1 for mono)
///
/// Example:
/// ```dart
/// final result = await model.run(envelope);
/// final pcmBytes = result.audioBytes!;
/// final wavBytes = wrapInWavHeader(pcmBytes, sampleRate: 24000, channels: 1);
/// // Now wavBytes can be played with just_audio or saved as a .wav file
/// ```
Uint8List wrapInWavHeader(
  Uint8List pcmBytes, {
  required int sampleRate,
  required int channels,
}) {
  const bitsPerSample = 16;
  final byteRate = sampleRate * channels * bitsPerSample ~/ 8;
  final blockAlign = channels * bitsPerSample ~/ 8;
  final dataSize = pcmBytes.length;
  final fileSize = 36 + dataSize;

  final header = BytesBuilder();
  header.add('RIFF'.codeUnits);
  header.add(_int32ToBytes(fileSize));
  header.add('WAVE'.codeUnits);
  header.add('fmt '.codeUnits);
  header.add(_int32ToBytes(16)); // Subchunk1Size for PCM
  header.add(_int16ToBytes(1)); // AudioFormat: 1 = PCM
  header.add(_int16ToBytes(channels));
  header.add(_int32ToBytes(sampleRate));
  header.add(_int32ToBytes(byteRate));
  header.add(_int16ToBytes(blockAlign));
  header.add(_int16ToBytes(bitsPerSample));
  header.add('data'.codeUnits);
  header.add(_int32ToBytes(dataSize));

  final result = BytesBuilder();
  result.add(header.toBytes());
  result.add(pcmBytes);
  return result.toBytes();
}

Uint8List _int32ToBytes(int value) {
  return Uint8List(4)..buffer.asByteData().setInt32(0, value, Endian.little);
}

Uint8List _int16ToBytes(int value) {
  return Uint8List(2)..buffer.asByteData().setInt16(0, value, Endian.little);
}
