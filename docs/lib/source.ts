import { docs } from 'fumadocs-mdx:collections/server';
import { type InferPageType, llms, loader } from 'fumadocs-core/source';
import { lucideIconsPlugin } from 'fumadocs-core/source/lucide-icons';
import { i18n } from '@/lib/i18n';

export { SITE } from '@/lib/discovery';

// See https://fumadocs.dev/docs/headless/source-api for more info
export const source = loader({
  baseUrl: '/docs',
  source: docs.toFumadocsSource(),
  plugins: [lucideIconsPlugin()],
  i18n,
});

/** The share image of a page, built by `app/[lang]/og/docs/[...slug]`. */
export function getPageImage(page: InferPageType<typeof source>) {
  const segments = [...page.slugs, 'image.png'];

  return {
    segments,
    url: '/' + [page.locale, 'og', 'docs', ...segments].filter(Boolean).join('/'),
  };
}

/**
 * The docs for agents, see https://fumadocs.dev/docs/next/integrations/llms:
 * `llms.txt`, `llms-full.txt` and the Markdown of each page. Every URL carries
 * the locale, so a page is `/<lang>/docs/<slug>.md`, and a language's files are
 * `/<lang>/llms.txt` and `/<lang>/llms-full.txt`.
 */
export const docsLlms = llms(source, {
  renderPage: async (page) => `# ${page.data.title} (${page.url})

${await page.data.getText('processed')}`,
});
