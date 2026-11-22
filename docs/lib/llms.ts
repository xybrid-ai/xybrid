import { flattenTree } from 'fumadocs-core/page-tree';
import { i18n } from '@/lib/i18n';
import { docsLlms, SITE, source } from '@/lib/source';

// URLs are `<...>` autolinks: a bare URL would run into the punctuation after
// it, `，` and `。` included.
const summary: Record<string, string> = {
  en: `> Xybrid is an on-device AI SDK: speech-to-text, text-to-speech, LLMs, vision and embeddings run locally in mobile, desktop and game apps (Flutter, Swift, Kotlin, React Native, Unity, Rust, a CLI, and a browser preview). Local inference needs no account or key; cloud fallback and trace export are opt-in.

The full text of every page is at <${SITE}/llms-full.txt>, and each link below is the Markdown of one page. Most pages are also available in Chinese: <${SITE}/zh/llms.txt>. Agents can also search and read the docs over MCP at <${SITE}/api/mcp>.`,
  zh: `> Xybrid 是一个本地 AI SDK：语音转文字、文字转语音、大语言模型、视觉模型和嵌入模型直接在移动端、桌面端和游戏应用中本地运行（Flutter、Swift、Kotlin、React Native、Unity、Rust、命令行，以及浏览器预览版）。本地推理无需账号或密钥；云端回退和轨迹导出需另行显式开启。

所有页面的全文位于 <${SITE}/zh/llms-full.txt>，下方每个链接都是一个页面的 Markdown。部分页面尚未翻译，以英文提供。英文索引：<${SITE}/llms.txt>。AI 代理也可以通过 MCP 搜索和阅读文档：<${SITE}/api/mcp>。`,
};

/**
 * The `llms.txt` of a language, in the format of https://llmstxt.org: an H1, a
 * summary, then H2 sections that list one link per line.
 *
 * `llms()` titles the index after the page tree root (`# Docs`), lists sidebar
 * separators as `- **Guides**`, and folders (`SDKs`) as an item with no link
 * whose pages are indented under it. So the title and summary are ours, each
 * separator becomes a section, and the pages of a folder join its section.
 */
export async function llmsIndex(lang: string = i18n.defaultLanguage): Promise<Response> {
  const file = await docsLlms.serve('/llms.txt', lang);
  if (!file) throw new Error('llms() serves `/llms.txt` for every language');

  const index = (await file.text())
    .replace(/^# .*\n+/, '')
    .replace(/^- \*\*(.+)\*\*$/gm, '## $1\n')
    .replace(/^ *- (?!\[).*\n/gm, '')
    .replace(/^ +- \[/gm, '- [');

  return new Response(`# Xybrid\n\n${summary[lang] ?? summary.en}\n\n${index.trim()}\n`, {
    headers: file.headers,
  });
}

/**
 * The `llms-full.txt` of a language: every page, in sidebar order, so the
 * overview and quick start come first. `llms()` would join them in file order
 * (`sdks/android` first). Pages outside the sidebar, like the English pages a
 * language falls back to, come last.
 */
export async function llmsFull(lang: string = i18n.defaultLanguage): Promise<Response> {
  const order = new Map(
    flattenTree(source.getPageTree(lang).children).map((node, i) => [node.url, i]),
  );
  const rank = (url: string) => order.get(url) ?? order.size;
  const pages = source.getPages(lang).sort((a, b) => rank(a.url) - rank(b.url));
  const text = await Promise.all(pages.map((page) => docsLlms.page(page)));

  return new Response(text.join('\n\n'), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
