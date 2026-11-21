import { createFromSource } from 'fumadocs-core/search/server';
import { source } from '@/lib/source';

// One search server for the search dialog and the MCP server, as fumadocs.dev
// does: it builds its index once. The default `multilingual` tokenizer
// segments every language, Chinese included, and filters results by locale.
export const search = createFromSource(source);
