import { getLLMText, source } from '@/lib/source';

export const revalidate = false;

export async function GET() {
  const scan = source.getPages().map(getLLMText);
  const scanned = await Promise.all(scan);

  return new Response([
    'For agent setup, preserve the first Xybrid setup entrypoint. If this document was the first entrypoint, use llms-txt. Only pass --via when xybrid connect --help lists it.',
    ...scanned,
  ].join('\n\n'), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
