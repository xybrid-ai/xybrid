'use client';
import { useEffect } from 'react';
import { useRouter } from 'fumadocs-core/framework';
import { fetchClient } from 'fumadocs-core/search/client/fetch';
import { useI18n } from 'fumadocs-ui/contexts/i18n';
import { i18n } from '@/lib/i18n';

interface ModelContextTool<Input> {
  name: string;
  description: string;
  inputSchema?: object;
  execute: (input: Input) => Promise<unknown>;
}

declare global {
  interface Document {
    /** WebMCP, see https://webmachinelearning.github.io/webmcp/ */
    modelContext?: {
      registerTool: <Input>(
        tool: ModelContextTool<Input>,
        options?: { signal?: AbortSignal },
      ) => Promise<void>;
    };
  }
}

function docsDestination(input: string) {
  const origin = window.location.origin;
  const url = new URL(input, origin);
  const path = url.pathname.replace(/\/+$/, '');
  const isDocs = i18n.languages.some((lang) =>
    path === `/${lang}/docs` || path.startsWith(`/${lang}/docs/`),
  );

  if (url.origin !== origin || !/^https?:$/.test(url.protocol) || url.username || url.password ||
      !isDocs || /%(?:2f|5c|25)/i.test(path)) {
    throw new Error('Expected a same-origin documentation page URL');
  }

  return { path, search: url.search, hash: url.hash };
}

/** expose the docs to AI agents in the browser */
export function WebMCP() {
  const router = useRouter();
  const { locale } = useI18n();

  useEffect(() => {
    const context = document.modelContext;
    if (!context) return;
    const controller = new AbortController();
    const { signal } = controller;
    const client = fetchClient({ locale });

    async function registerTool<Input>(tool: ModelContextTool<Input>) {
      try {
        await context?.registerTool(tool, { signal });
      } catch (error) {
        if (!signal.aborted || error !== signal.reason) {
          console.error(`WebMCP registration failed for ${tool.name}:`, error);
        }
      }
    }

    void registerTool(
      {
        name: 'search_docs',
        description: 'Search the docs, returns matching pages and headings with their URLs',
        inputSchema: {
          type: 'object',
          properties: { query: { type: 'string' } },
          required: ['query'],
        },
        async execute({ query }: { query: string }) {
          return client.search(query);
        },
      },
    );
    void registerTool(
      {
        name: 'read_page',
        description: 'Read a docs page as Markdown by its URL',
        inputSchema: {
          type: 'object',
          properties: { url: { type: 'string' } },
          required: ['url'],
        },
        async execute({ url }: { url: string }) {
          const { path } = docsDestination(url);
          const res = await fetch(`${path}.md`, { signal, redirect: 'error' });
          if (!res.ok) throw new Error(`page not found: ${url}`);
          return res.text();
        },
      },
    );
    void registerTool(
      {
        name: 'open_page',
        description: 'Navigate to a docs page by its URL',
        inputSchema: {
          type: 'object',
          properties: { url: { type: 'string' } },
          required: ['url'],
        },
        async execute({ url }: { url: string }) {
          const { path, search, hash } = docsDestination(url);
          const destination = `${path}${search}${hash}`;
          await router.push(destination);
          return `opened ${destination}`;
        },
      },
    );

    return () => controller.abort();
  }, [router, locale]);

  return null;
}
