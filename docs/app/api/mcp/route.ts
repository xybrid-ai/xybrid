import { createMcpHandler, McpServer } from '@modelcontextprotocol/server';
import { registerSearchTool, registerSourceTools } from 'fumadocs-core/mcp';
import { i18n } from '@/lib/i18n';
import { search } from '@/lib/search';
import { docsLlms, source } from '@/lib/source';

// The MCP server of the docs, see https://fumadocs.dev/docs/next/integrations/llms#mcp-server.
// Connect an agent with `{ "mcpServers": { "xybrid-docs": { "url": "https://docs.xybrid.dev/api/mcp" } } }`.
const handler = createMcpHandler(() => {
  const mcp = new McpServer(
    {
      name: 'xybrid-docs',
      version: '1.0.0',
    },
    // the tools never change, a stream to watch them would hold a serverless function open
    { capabilities: { tools: { listChanged: false } } },
  );

  // the pages of the default language, like `llms.txt` (as fumadocs.dev does)
  registerSourceTools(mcp, source, {
    ...docsLlms,
    index: () => docsLlms.index(i18n.defaultLanguage),
  });
  registerSearchTool(mcp, search);

  return mcp;
});

export async function GET(request: Request) {
  return handler.fetch(request);
}

export async function POST(request: Request) {
  return handler.fetch(request);
}

export async function DELETE(request: Request) {
  return handler.fetch(request);
}
