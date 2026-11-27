import { MCP_PATH, SITE, mcpCapabilities, mcpIdentity } from '@/lib/discovery';

export const dynamic = 'force-static';

export function GET() {
  return Response.json({
    serverInfo: mcpIdentity,
    transport: { type: 'streamable-http', endpoint: `${SITE}${MCP_PATH}` },
    capabilities: mcpCapabilities,
  });
}
