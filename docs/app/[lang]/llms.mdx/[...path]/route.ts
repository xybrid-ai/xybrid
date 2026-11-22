import { i18n } from '@/lib/i18n';
import { llmsFull, llmsIndex } from '@/lib/llms';
import { docsLlms } from '@/lib/source';

export const revalidate = false;
// other paths are 404s, never cached responses
export const dynamicParams = false;

const markdownHeaders = {
  'Content-Type': 'text/markdown; charset=utf-8',
  'Cache-Control': 'private, no-store',
};

/**
 * The files for agents of a language. `next.config.mjs` rewrites their public
 * paths here: `/<lang>/index.md`, `/<lang>/docs/<page>.md`, `/<lang>/llms.txt` and
 * `/<lang>/llms-full.txt`.
 */
export async function GET(
  _req: Request,
  { params }: RouteContext<'/[lang]/llms.mdx/[...path]'>,
) {
  const { lang, path } = await params;
  const file = path.join('/');
  if (file === 'index.md') {
    const home = homeContent(lang);
    return new Response([
      '# Xybrid', '', home.tagline, '', home.description, '',
      `- [${home.quickstart}](/${lang}/docs/quickstart.md)`,
      `- [${home.index}](/${lang}/llms.txt)`,
      `- [${home.full}](/${lang}/llms-full.txt)`, '',
    ].join('\n'), { headers: markdownHeaders });
  }
  if (file === 'llms.txt') return llmsIndex(lang);
  if (file === 'llms-full.txt') return llmsFull(lang);

  // the Markdown of a page, whose URL carries the locale: `/en/docs/x.md`
  const response = await docsLlms.serve(`/${lang}/${file}`, lang);

  if (!response) return new Response(null, { status: 404 });
  response.headers.set('Cache-Control', markdownHeaders['Cache-Control']);
  return response;
}

export async function generateStaticParams() {
  const params: { lang: string; path: string[] }[] = [];
  for (const lang of i18n.languages) {
    params.push({ lang, path: ['index.md'] });
    for (const file of await docsLlms.files(lang)) {
      // `/llms.txt` -> ['llms.txt'], `/zh/docs/quickstart.md` -> ['docs', 'quickstart.md']
      const path = decodeURI(file).slice(1).split('/');
      if (path[0] === lang) path.shift();
      params.push({ lang, path });
    }
  }
  return params;
}
import { homeContent } from '@/lib/home';
