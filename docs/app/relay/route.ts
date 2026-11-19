import { ANALYTICS_MAX_BYTES, DOCS_ORIGIN, analyticsAllowed, capturePayload } from '@/lib/analytics';
import { localizedPagePaths } from '@/lib/source';

const paths = new Set(localizedPagePaths());
const endpoint = 'https://us.i.posthog.com/i/v0/e/';

async function readEvent(request: Request): Promise<unknown> {
  if (request.headers.get('content-type') !== 'application/json' || !request.body) return;
  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0;
  let body = '';
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    bytes += chunk.value.byteLength;
    if (bytes > ANALYTICS_MAX_BYTES) {
      await reader.cancel();
      return;
    }
    body += decoder.decode(chunk.value, { stream: true });
  }
  return JSON.parse(body + decoder.decode());
}

export async function POST(request: Request) {
  const empty = () => new Response(null, { status: 204, headers: { 'Cache-Control': 'no-store' } });
  const key = process.env.POSTHOG_DOCS_KEY?.trim();
  if (!key || new URL(request.url).origin !== DOCS_ORIGIN || request.headers.get('origin') !== DOCS_ORIGIN ||
    !analyticsAllowed(process.env.NEXT_PUBLIC_ANALYTICS_ENABLED, process.env.NEXT_PUBLIC_ANALYTICS_NOTICE_EFFECTIVE_AT)) return empty();

  try {
    const event = capturePayload(await readEvent(request), paths);
    if (event) {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...event, api_key: key }),
        signal: AbortSignal.timeout(1500),
        redirect: 'error',
        cache: 'no-store',
      });
      await response.body?.cancel();
    }
  } catch {
    return empty();
  }
  return empty();
}
