import { AI_CATALOG, SITE } from '@/lib/discovery';

export const dynamic = 'force-static';

export function GET() {
  return new Response(`User-agent: *
Allow: /
Disallow: /api/
Allow: /api/mcp/server-card
Content-Signal: ai-train=no, search=yes, ai-input=yes

User-agent: OAI-SearchBot
User-agent: Claude-SearchBot
User-agent: PerplexityBot
Allow: /
Disallow: /api/
Allow: /api/mcp/server-card

User-agent: GPTBot
User-agent: ClaudeBot
User-agent: Google-Extended
Disallow: /

Sitemap: ${SITE}/sitemap.xml
Agentmap: ${SITE}${AI_CATALOG}
`, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
