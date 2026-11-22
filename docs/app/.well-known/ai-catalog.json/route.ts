import { API_CATALOG, MCP_CARD, MCP_CARD_TYPE, SITE, SKILLS_INDEX } from '@/lib/discovery';

export const dynamic = 'force-static';

const id = `urn:air:${new URL(SITE).hostname}`;

export function GET() {
  return Response.json({
    specVersion: '1.0',
    host: { identifier: new URL(SITE).hostname, displayName: 'Xybrid Documentation' },
    entries: [
      {
        identifier: `${id}:mcp:docs`,
        displayName: 'Xybrid Docs MCP',
        type: MCP_CARD_TYPE,
        url: `${SITE}${MCP_CARD}`,
        representativeQueries: ['Find the Xybrid quickstart', 'Search Xybrid SDK documentation'],
      },
      {
        identifier: `${id}:api:catalog`,
        displayName: 'Xybrid Docs API Catalog',
        type: 'application/linkset+json',
        url: `${SITE}${API_CATALOG}`,
        representativeQueries: ['Discover the public docs APIs', 'Find the Xybrid MCP endpoint'],
      },
      {
        identifier: `${id}:skills:docs`,
        displayName: 'Xybrid Agent Skills',
        type: 'application/json',
        url: `${SITE}${SKILLS_INDEX}`,
        representativeQueries: ['How should an agent read Xybrid docs?', 'Find the Xybrid docs skill'],
      },
    ],
  }, { headers: { 'Access-Control-Allow-Origin': '*' } });
}
