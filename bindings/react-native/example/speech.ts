// Speech round trip for the smoke test: synthesize a sentence, transcribe the
// result in one batch run, then transcribe it again through a live ASR session
// fed in 100 ms chunks — the path a dictation UI takes.

import type { Model } from 'react-native-xybrid';

/**
 * Decode TTS output into Float32 samples. ONNX TTS models return raw 16-bit
 * mono PCM at the model's rate (`fallbackRate`); codec models return a WAV.
 */
export function decodeAudio(
  base64: string,
  fallbackRate: number,
): { samples: Float32Array; sampleRate: number; wav: boolean } {
  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  if (String.fromCharCode(...bytes.slice(0, 4)) !== 'RIFF') {
    const samples = new Float32Array(Math.floor(bytes.length / 2));
    for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
    return { samples, sampleRate: fallbackRate, wav: false };
  }

  let offset = 12;
  let sampleRate = 16000;
  let channels = 1;
  while (offset + 8 <= bytes.length) {
    const id = String.fromCharCode(...bytes.slice(offset, offset + 4));
    const size = view.getUint32(offset + 4, true);
    if (id === 'fmt ') {
      channels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
    } else if (id === 'data') {
      const frames = Math.floor(size / 2 / channels);
      const samples = new Float32Array(frames);
      for (let i = 0; i < frames; i++) {
        samples[i] = view.getInt16(offset + 8 + i * 2 * channels, true) / 32768;
      }
      return { samples, sampleRate, wav: true };
    }
    offset += 8 + size + (size % 2);
  }
  throw new Error('WAV has no data chunk');
}

/**
 * Wrap Float32 samples as a 16-bit mono WAV (base64). Batch ASR only accepts
 * WAV input today, while ONNX TTS models return headerless PCM.
 */
export function toWavBase64(samples: Float32Array, sampleRate: number): string {
  const bytes = new Uint8Array(44 + samples.length * 2);
  const view = new DataView(bytes.buffer);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) bytes[offset + i] = text.charCodeAt(i);
  };
  ascii(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  }
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return globalThis.btoa(binary);
}

/** Linear resampling — plenty for a smoke test. */
export function resample(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return samples;
  const out = new Float32Array(Math.floor((samples.length * to) / from));
  for (let i = 0; i < out.length; i++) {
    const position = (i * from) / to;
    const index = Math.floor(position);
    const next = Math.min(index + 1, samples.length - 1);
    out[i] = samples[index] + (samples[next] - samples[index]) * (position - index);
  }
  return out;
}

/**
 * Feed `samples` (16 kHz) through a live session, collecting partials, and
 * return the flushed transcript plus how many partials arrived.
 */
export async function transcribeLive(
  asr: Model,
  samples: Float32Array,
): Promise<{ transcript: string; partials: number }> {
  const session = await asr.stream();
  let partials = 0;
  const reader = (async () => {
    for await (const _ of session.partials()) partials += 1;
  })();
  try {
    const chunk = 1600; // 100 ms at 16 kHz
    for (let offset = 0; offset < samples.length; offset += chunk) {
      await session.feed(samples.subarray(offset, offset + chunk));
    }
    // A second of silence lets the chunker close the last window.
    await session.feed(new Float32Array(16000));
    const transcript = await session.flush();
    await reader;
    return { transcript, partials };
  } finally {
    await session.release();
  }
}
