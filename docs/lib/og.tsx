import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { ImageResponseOptions } from 'next/dist/compiled/@vercel/og/types';

/**
 * The options of the share images, like fumadocs.dev's `getImageResponseOptions`.
 * `next/og` falls back to Google's Noto fonts for scripts its default font
 * (Geist) lacks, but tries Noto Sans JP before Noto Sans SC, so a Chinese card
 * mixes Japanese and Chinese glyph forms; it also only logs a failed download
 * and renders boxes. Pinning the font of each translation fixes both.
 */

/** the font of each translation that needs one */
const localeFonts: Record<string, string> = { zh: 'Noto Sans SC' };

/**
 * A Google Font as TTF (what `next/og` reads, served to clients without a
 * browser User-Agent), subset to the characters of `text`.
 */
async function loadGoogleFont(family: string, weight: number, text: string): Promise<ArrayBuffer> {
  const query = `family=${encodeURIComponent(family)}:wght@${weight}&text=${encodeURIComponent(text)}`;
  const css = await fetch(`https://fonts.googleapis.com/css2?${query}`).then((res) => res.text());
  const src = /src: url\((.+?)\) format\('(?:opentype|truetype)'\)/.exec(css)?.[1];
  // fail the build rather than ship an image of boxes
  if (!src) throw new Error(`OG image: Google Fonts served no ${family} ${weight}`);

  const res = await fetch(src);
  if (!res.ok) throw new Error(`OG image: ${family} ${weight} failed: HTTP ${res.status}`);
  return res.arrayBuffer();
}

export async function getImageResponseOptions(
  locale: string | undefined,
  text: string,
): Promise<ImageResponseOptions> {
  const name = locale && localeFonts[locale];
  if (!name) return {};

  const [regular, bold] = await Promise.all([
    loadGoogleFont(name, 400, text),
    loadGoogleFont(name, 700, text),
  ]);
  return {
    fonts: [
      { name, data: regular, weight: 400, style: 'normal' },
      { name, data: bold, weight: 700, style: 'normal' },
    ],
  };
}

let logo: Promise<string> | undefined;

/** `docs/logo.jpg` as a data URL, the icon of the images; they render at build time, from `docs/`. */
export function getLogo(): Promise<string> {
  logo ??= readFile(join(process.cwd(), 'logo.jpg')).then(
    (data) => `data:image/jpeg;base64,${data.toString('base64')}`,
  );
  return logo;
}
