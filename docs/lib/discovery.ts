export const SITE = 'https://docs.xybrid.dev';

export const API_CATALOG = '/.well-known/api-catalog';
export const AI_CATALOG = '/.well-known/ai-catalog.json';
export const SKILLS_INDEX = '/.well-known/agent-skills/index.json';
export const MCP_PATH = '/api/mcp';
export const MCP_CARD = `${MCP_PATH}/server-card`;
export const MCP_WELL_KNOWN_CARD = '/.well-known/mcp/server-card.json';
export const MCP_CARD_TYPE = 'application/mcp-server-card+json';

export const mcpIdentity = { name: 'xybrid-docs', version: '1.0.0' };
export const mcpCapabilities = { tools: { listChanged: false } };

export function homeDiscoveryLinks(lang: string) {
  return [
    `<${API_CATALOG}>; rel="api-catalog"; type="application/linkset+json"`,
    `</${lang}/docs/introduction>; rel="service-doc"`,
    `<${MCP_CARD}>; rel="service-desc"; type="${MCP_CARD_TYPE}"`,
    `<${AI_CATALOG}>; rel="describedby"; type="application/json"`,
    `<${SKILLS_INDEX}>; rel="describedby"; type="application/json"`,
    `</${lang}/index.md>; rel="alternate"; type="text/markdown"`,
    `</${lang}/llms.txt>; rel="alternate"; type="text/plain"`,
  ].join(', ');
}
