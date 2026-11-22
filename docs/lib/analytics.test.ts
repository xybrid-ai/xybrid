import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DOCS_ORIGIN, analyticsAllowed, capturePayload } from './analytics.ts';
import { relayEvent } from './analytics-relay.ts';

const notice = '2025-06-01T12:00:00Z';
const now = Date.parse(notice);
const pageview = {
  id: '12345678-1234-4123-8123-123456789abc',
  path: '/en/docs',
  event: '$pageview',
  timestamp: new Date(now).toISOString(),
};
const paths = new Set([pageview.path]);

test('analytics requires an explicit flag and an effective, canonical notice date', () => {
  assert.equal(analyticsAllowed('true', notice, now), true);
  assert.equal(analyticsAllowed(undefined, notice, now), false);
  assert.equal(analyticsAllowed('true', undefined, now), false);
  assert.equal(analyticsAllowed('true', notice, now - 1), false);
  assert.equal(analyticsAllowed('true', '2025-02-30T12:00:00Z', now), false);
});

test('capture rejects unlisted URLs and events and invalid timestamps', () => {
  for (const overrides of [
    { path: '/en/docs?private=yes' },
    { path: '/en/private' },
    { event: '$identify' },
    { event: 'outbound_navigation', destination: 'elsewhere' },
    { timestamp: undefined },
    { timestamp: '2025-02-30T12:00:00.000Z' },
    { timestamp: '2025-06-01T12:06:00.000Z' },
  ]) {
    assert.equal(capturePayload({ ...pageview, ...overrides }, paths, now), undefined);
  }
});

test('capture rebuilds allowlisted properties and preserves each event timestamp', () => {
  const event = capturePayload({ ...pageview, campaign: 'ChatGPT.com', referrerDomain: 'example.com',
    api_key: 'untrusted', properties: { email: 'private' }, destination: 'github' }, paths, now);
  assert.deepEqual(event, {
    distinct_id: pageview.id,
    event: '$pageview',
    timestamp: pageview.timestamp,
    properties: {
      site: 'docs',
      $current_url: `${DOCS_ORIGIN}${pageview.path}`,
      $host: 'docs.xybrid.dev',
      $pathname: pageview.path,
      $geoip_disable: true,
      $process_person_profile: false,
      campaign_source: 'chatgpt',
      $referring_domain: 'example.com',
    },
  });
  const leaveAt = '2025-06-01T12:00:01.000Z';
  const leave = capturePayload({ ...pageview, event: '$pageleave', timestamp: leaveAt,
    campaign: 'private', referrerDomain: 'example.com' }, paths, now);
  assert.equal(leave?.timestamp, leaveAt);
  assert.equal(leave?.properties.$referring_domain, undefined);
  assert.equal(leave?.properties.campaign_source, undefined);
});

test('relay does not forward before notice, without a key, or from another origin', async (context) => {
  const fetch = context.mock.method(globalThis, 'fetch', async (..._args: Parameters<typeof globalThis.fetch>) => new Response(null));
  context.mock.method(Date, 'now', () => now);
  const original = process.env;
  process.env = { ...original, NEXT_PUBLIC_ANALYTICS_ENABLED: 'true', POSTHOG_DOCS_KEY: 'test-key' };
  context.after(() => { process.env = original; });
  delete process.env.NEXT_PUBLIC_ANALYTICS_NOTICE_EFFECTIVE_AT;
  const relay = async (site = DOCS_ORIGIN, origin = DOCS_ORIGIN) => {
    const response = await relayEvent(new Request(`${site}/relay`, {
      method: 'POST',
      headers: { origin, 'content-type': 'application/json', cookie: 'private' },
      body: JSON.stringify({ ...pageview, api_key: 'forged' }),
    }), paths);
    assert.equal(response.status, 204);
    assert.equal(response.headers.get('cache-control'), 'no-store');
  };
  await relay();
  process.env.NEXT_PUBLIC_ANALYTICS_NOTICE_EFFECTIVE_AT = '2025-06-01T12:00:01Z';
  await relay();
  process.env.NEXT_PUBLIC_ANALYTICS_NOTICE_EFFECTIVE_AT = notice;
  delete process.env.POSTHOG_DOCS_KEY;
  await relay();
  process.env.POSTHOG_DOCS_KEY = 'test-key';
  process.env.NEXT_PUBLIC_ANALYTICS_ENABLED = 'false';
  await relay();
  process.env.NEXT_PUBLIC_ANALYTICS_ENABLED = 'true';
  await relay('https://preview.example.com');
  await relay(DOCS_ORIGIN, 'https://other.example.com');
  assert.equal(fetch.mock.callCount(), 0);

  await relay();
  assert.equal(fetch.mock.callCount(), 1);
  const [, init] = fetch.mock.calls[0].arguments;
  assert.deepEqual(init?.headers, { 'Content-Type': 'application/json' });
  assert.deepEqual(JSON.parse(String(init?.body)), { ...capturePayload(pageview, paths, now), api_key: 'test-key' });
});
