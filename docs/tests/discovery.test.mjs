import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';

const origin = process.env.DOCS_TEST_ORIGIN ?? 'http://localhost:3000';
const site = 'https://docs.xybrid.dev';

async function get(path, type, headers) {
  const response = await fetch(new URL(path, origin), { headers, redirect: 'manual' });
  assert.equal(response.status, 200, path);
  assert.equal(response.headers.get('content-type')?.split(';')[0], type, path);
  return response;
}

test('root crawl controls and sitemap are public and list localized HTML', async () => {
  const robots = await (await get('/robots.txt', 'text/plain')).text();
  assert.match(robots, /User-agent: OAI-SearchBot[\s\S]*?Allow: \//);
  assert.match(robots, /User-agent: GPTBot[\s\S]*?Disallow: \//);
  assert.match(robots, /Content-Signal: ai-train=no, search=yes, ai-input=yes/);
  assert.ok(robots.includes(`Sitemap: ${site}/sitemap.xml`));

  const sitemap = await fetch(new URL('/sitemap.xml', origin), { redirect: 'manual' });
  assert.equal(sitemap.status, 200);
  assert.match(sitemap.headers.get('content-type'), /xml/);
  const xml = await sitemap.text();
  const urls = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(([, url]) => url);
  assert.equal(new Set(urls).size, urls.length);
  for (const lang of ['en', 'zh']) {
    assert.ok(urls.includes(`${site}/${lang}`));
    assert.ok(urls.includes(`${site}/${lang}/docs/quickstart`));
  }
  for (const url of urls) {
    assert.equal(new URL(url).origin, site);
    assert.match(new URL(url).pathname, /^\/(en|zh)(\/docs(?:\/[^.]+)?)?$/);
  }
});

test('home and docs select Markdown without changing default HTML', async () => {
  for (const lang of ['en', 'zh']) {
    const root = await fetch(new URL('/', origin), {
      headers: { 'Accept-Language': lang }, redirect: 'manual',
    });
    assert.equal(root.status, 307);
    assert.equal(new URL(root.headers.get('location'), origin).pathname, `/${lang}`);
    assert.match(root.headers.get('link'), /rel="api-catalog"/);
    const negotiatedRoot = await fetch(new URL('/', origin), {
      headers: { Accept: 'text/markdown', 'Accept-Language': lang },
    });
    assert.equal(new URL(negotiatedRoot.url).pathname, `/${lang}`);
    assert.match(negotiatedRoot.headers.get('content-type'), /^text\/markdown/);

    for (const [page, explicit] of [
      [`/${lang}`, `/${lang}/index.md`],
      [`/${lang}/docs/quickstart`, `/${lang}/docs/quickstart.md`],
    ]) {
      const html = await get(page, 'text/html');
      const md = await get(page, 'text/markdown', { Accept: 'text/markdown' });
      for (const response of [html, md]) {
        assert.match(response.headers.get('cache-control'), /\bno-store\b/);
      }
      if (new URL(origin).hostname === '127.0.0.1' || new URL(origin).hostname === 'localhost') {
        for (const response of [html, md]) {
          assert.equal(response.headers.get('cdn-cache-control'), 'no-store');
        }
      }
      assert.equal(await md.text(), await (await get(explicit, 'text/markdown')).text());
      await get(page, 'text/html', { Accept: 'text/html, text/markdown;q=0.2' });
      if (page === `/${lang}`) {
        for (const response of [html, md]) {
          assert.match(response.headers.get('link'), /rel="api-catalog"/);
          assert.match(response.headers.get('link'), /rel="service-doc"/);
          assert.ok(response.headers.get('link').includes(`<${explicit}>`));
        }
      }
    }
  }
});

test('catalogs resolve to the live MCP server and byte-verified skill', async () => {
  const catalog = await (await get('/.well-known/api-catalog', 'application/linkset+json')).json();
  const endpoint = '/api/mcp';
  const service = catalog.linkset.find(({ anchor }) => anchor === `${site}${endpoint}`);
  const [description] = service['service-desc'];
  const card = await (await get(new URL(description.href).pathname, description.type)).json();
  const [remote] = card.remotes;
  assert.equal(remote.url, `${site}${endpoint}`);

  const compatibility = await (await get('/.well-known/mcp/server-card.json', 'application/json')).json();
  const protocolVersion = '2025-03-26';
  const live = await fetch(new URL(endpoint, origin), {
    method: 'POST',
    headers: { Accept: 'application/json, text/event-stream', 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0', id: 1, method: 'initialize',
      params: { protocolVersion, capabilities: {}, clientInfo: { name: 'discovery-test', version: '1.0.0' } },
    }),
  });
  assert.equal(live.status, 200);
  const body = await live.text();
  const { result } = JSON.parse(body.match(/^data: (.+)$/m)?.[1] ?? body);
  assert.deepEqual(compatibility.serverInfo, result.serverInfo);
  assert.deepEqual(compatibility.capabilities, result.capabilities);
  assert.ok(remote.supportedProtocolVersions.includes(result.protocolVersion));

  const ai = await get('/.well-known/ai-catalog.json', 'application/json');
  assert.equal(ai.headers.get('access-control-allow-origin'), '*');
  const { entries } = await ai.json();
  assert.ok(entries.some(({ url }) => url === description.href));
  for (const entry of entries) {
    assert.match(entry.identifier, /^urn:air:docs\.xybrid\.dev:[^:]+:[^:]+$/);
    assert.ok(entry.url && !('data' in entry));
    assert.ok(entry.representativeQueries.length >= 2 && entry.representativeQueries.length <= 5);
    await get(new URL(entry.url).pathname, entry.type);
  }

  const { skills } = await (await get('/.well-known/agent-skills/index.json', 'application/json')).json();
  assert.ok(skills.length > 0);
  for (const skill of skills) {
    const artifact = await fetch(new URL(new URL(skill.url).pathname, origin), { redirect: 'manual' });
    assert.equal(artifact.status, 200);
    const digest = createHash('sha256').update(Buffer.from(await artifact.arrayBuffer())).digest('hex');
    assert.equal(skill.digest, `sha256:${digest}`);
  }
});
