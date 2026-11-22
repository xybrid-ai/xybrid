import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

// The languages of `lib/i18n.ts`, keep them equal. Constraining `:lang` keeps
// `/robots.txt` and `/llms.txt` out of the rewrites below.
const lang = ':lang(en|zh)';

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  async rewrites() {
    // the files for agents, served by `app/[lang]/llms.mdx/[...path]`
    return [
      // `/en/docs/quickstart.md`, the Markdown of a page
      {
        source: `/${lang}/docs/:path*.md`,
        destination: '/:lang/llms.mdx/docs/:path*.md',
      },
      // `/zh/llms.txt` and `/zh/llms-full.txt`, the index and full text of a language
      {
        source: `/${lang}/:path*.txt`,
        destination: '/:lang/llms.mdx/:path*.txt',
      },
    ];
  },
};

export default withMDX(config);
