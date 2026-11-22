import { SUPPORTED_PROTOCOL_VERSIONS } from '@modelcontextprotocol/server';
import { MCP_CARD_TYPE, MCP_PATH, SITE, mcpIdentity } from '@/lib/discovery';

export const dynamic = 'force-static';

export function GET() {
  return Response.json({
    $schema: 'https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json',
    name: `dev.xybrid/${mcpIdentity.name}`,
    version: mcpIdentity.version,
    description: 'Search and read the public Xybrid SDK documentation.',
    websiteUrl: SITE,
    remotes: [{
      type: 'streamable-http',
      url: `${SITE}${MCP_PATH}`,
      supportedProtocolVersions: SUPPORTED_PROTOCOL_VERSIONS,
    }],
  }, { headers: { 'Content-Type': MCP_CARD_TYPE, 'Access-Control-Allow-Origin': '*' } });
}
