import { API_CATALOG, MCP_CARD, MCP_CARD_TYPE, MCP_PATH, SITE } from '@/lib/discovery';

export const dynamic = 'force-static';

export function GET() {
  return Response.json({
    linkset: [
      { anchor: `${SITE}${API_CATALOG}`, item: [{ href: `${SITE}${MCP_PATH}` }] },
      {
        anchor: `${SITE}${MCP_PATH}`,
        'service-desc': [{ href: `${SITE}${MCP_CARD}`, type: MCP_CARD_TYPE }],
        'service-doc': [{ href: `${SITE}/llms.txt`, type: 'text/plain' }],
      },
    ],
  }, { headers: { 'Content-Type': 'application/linkset+json' } });
}
