export const DOCS_ORIGIN = 'https://docs.xybrid.dev';
export const ANALYTICS_RELAY = '/relay';
export const ANALYTICS_MAX_BYTES = 1024;
const MAX_EVENT_CLOCK_SKEW_MS = 5 * 60 * 1000;

const campaignSources = new Map([
  ['chatgpt', 'chatgpt'],
  ['chatgpt.com', 'chatgpt'],
  ['claude', 'claude'],
  ['perplexity', 'perplexity'],
  ['github', 'github'],
  ['docs', 'docs'],
  ['newsletter', 'newsletter'],
]);

const destinations = ['console', 'github'] as const;
type Destination = (typeof destinations)[number];

export type DocsEvent = {
  id: string;
  path: string;
  timestamp: string;
  campaign?: string;
} & (
  | { event: '$pageview'; referrerDomain?: string }
  | { event: '$pageleave' }
  | { event: 'outbound_navigation'; destination: Destination }
);

export function analyticsAllowed(enabled: string | undefined, effectiveAt: string | undefined, now = Date.now()) {
  if (enabled !== 'true' || !effectiveAt || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/.test(effectiveAt)) return false;
  const effective = new Date(effectiveAt);
  return Number.isFinite(effective.getTime()) && effective.toISOString() === effectiveAt.replace('Z', '.000Z') &&
    effective.getTime() <= now;
}

export function campaignSource(value: string | null) {
  return value && value.length <= 32 ? campaignSources.get(value.toLowerCase()) : undefined;
}

function externalDomain(domain: string) {
  if (domain === new URL(DOCS_ORIGIN).hostname || domain.length > 253 ||
    /\.(?:localhost|local|internal|test|onion|invalid|example)$/.test(domain)) return;
  if (/^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(domain)) return domain;
}

export function referrerDomain(referrer: string) {
  try {
    const url = new URL(referrer);
    if ((url.protocol === 'https:' || url.protocol === 'http:') && !url.username && !url.password) {
      return externalDomain(url.hostname.replace(/\.$/, ''));
    }
  } catch {}
}

export function outboundDestination(url: URL, currentOrigin: string): Destination | undefined {
  if (url.protocol !== 'https:' || url.origin === currentOrigin || url.username || url.password) return;
  switch (url.origin) {
    case 'https://console.xybrid.dev': return 'console';
    case 'https://github.com': return 'github';
  }
}

export function capturePayload(input: unknown, paths: ReadonlySet<string>, now = Date.now()) {
  if (!input || typeof input !== 'object') return;
  const { id, path, event, timestamp, campaign, destination, referrerDomain } = input as Record<string, unknown>;
  if (typeof id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(id)) return;
  if (typeof path !== 'string' || !paths.has(path)) return;
  if (event !== '$pageview' && event !== '$pageleave' && event !== 'outbound_navigation') return;
  if (event === 'outbound_navigation' && !destinations.includes(destination as Destination)) return;
  if (typeof timestamp !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(timestamp)) return;
  const occurredAt = new Date(timestamp);
  if (!Number.isFinite(occurredAt.getTime()) || occurredAt.toISOString() !== timestamp ||
    Math.abs(occurredAt.getTime() - now) > MAX_EVENT_CLOCK_SKEW_MS) return;
  const campaignValue = typeof campaign === 'string' ? campaignSource(campaign) : undefined;
  const domain = event === '$pageview' && typeof referrerDomain === 'string' ? externalDomain(referrerDomain) : undefined;

  return {
    distinct_id: id,
    event,
    timestamp,
    properties: {
      site: 'docs',
      $current_url: `${DOCS_ORIGIN}${path}`,
      $host: new URL(DOCS_ORIGIN).hostname,
      $pathname: path,
      $geoip_disable: true,
      $process_person_profile: false,
      ...(campaignValue ? { campaign_source: campaignValue } : {}),
      ...(domain ? { $referring_domain: domain } : {}),
      ...(event === 'outbound_navigation' ? { destination } : {}),
    },
  };
}
