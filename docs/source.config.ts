import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import { remarkMdxMermaid } from 'fumadocs-core/mdx-plugins';

// Pages and `meta.json` use the default schemas of `fumadocs-core/source/schema`;
// extend them here, see https://fumadocs.dev/docs/mdx/collections
export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    postprocess: {
      // the Markdown of each page, for `llms.txt`, `llms-full.txt` and `<page>.md`
      includeProcessedMarkdown: true,
    },
  },
});

export default defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMdxMermaid],
  },
});
