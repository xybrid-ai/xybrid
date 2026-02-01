import 'dart:typed_data';

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
  header.add(_int32ToBytes(16));
  header.add(_int16ToBytes(1));
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
