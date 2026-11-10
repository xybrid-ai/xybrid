import { flattenTree, type Node } from 'fumadocs-core/page-tree';
import { i18n } from '@/lib/i18n';
import { source } from '@/lib/source';

export const revalidate = false;

const SITE = 'https://docs.xybrid.dev';

interface Section {
  title: string;
  urls: string[];
}

// The sidebar order of the default language: each separator starts a
// section, and folders (SDKs, CLI reference, ...) list their pages inline.
function sections(nodes: Node[]): Section[] {
  const result: Section[] = [{ title: 'Docs', urls: [] }];
  for (const node of nodes) {
    if (node.type === 'separator') {
      if (typeof node.name === 'string' && node.name.trim()) {
        result.push({ title: node.name.trim(), urls: [] });
      }
    } else if (node.type === 'page') {
      result[result.length - 1].urls.push(node.url);
    } else {
      const pages = flattenTree(node.children);
      if (node.index) pages.unshift(node.index);
      result[result.length - 1].urls.push(...pages.map((page) => page.url));
    }
  }
  return result.filter((section) => section.urls.length > 0);
}

// The short index of https://llmstxt.org: one line per page, with the full
// text at /llms-full.txt.
export function GET() {
  const language = i18n.defaultLanguage;
  const pages = new Map(source.getPages(language).map((page) => [page.url, page]));
  const lines = [
    '# Xybrid',
    '',
    '> Xybrid is an on-device AI SDK: speech-to-text, text-to-speech, LLMs, vision and embeddings run locally in mobile, desktop and game apps (Flutter, Swift, Kotlin, React Native, Unity, Rust, a CLI, and a browser preview). Local inference needs no account or key; cloud fallback and trace export are opt-in.',
    '',
    `The full text of every page is at ${SITE}/llms-full.txt. The pages are also available in Chinese under ${SITE}/zh/docs.`,
  ];
  for (const section of sections(source.getPageTree(language).children)) {
    lines.push('', `## ${section.title}`, '');
    for (const url of section.urls) {
      const page = pages.get(url);
      if (!page) continue;
      const description = page.data.description ? `: ${page.data.description}` : '';
      lines.push(`- [${page.data.title}](${SITE}${url})${description}`);
    }
  }

  return new Response(`${lines.join('\n')}\n`, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
