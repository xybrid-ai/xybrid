import { type NextFetchEvent, type NextRequest, NextResponse } from 'next/server';
import { createI18nMiddleware } from 'fumadocs-core/i18n/middleware';
import { isMarkdownPreferred, rewritePath } from 'fumadocs-core/negotiation';
import { i18n } from '@/lib/i18n';

const i18nProxy = createI18nMiddleware(i18n);

// `/en/docs/quickstart` -> `/en/llms.mdx/docs/quickstart.md`
const { rewrite: rewriteDocs } = rewritePath(
  '/:lang/docs{/*path}',
  '/:lang/llms.mdx/docs{/*path}.md',
);

export default function proxy(request: NextRequest, event: NextFetchEvent) {
  const { pathname } = request.nextUrl;
  const locale = pathname.split('/')[1];
  // An agent that prefers Markdown (`Accept: text/markdown`) gets the `.md` of
  // a docs page. `.md` and `.txt` paths are rewritten by `next.config.mjs`,
  // and a path without a locale is first redirected to `/<lang>/...` below.
  const markdown =
    // only Markdown types: `text/plain`, which the default also counts, is in
    // the `Accept` of HTTP clients like axios that want the HTML page
    isMarkdownPreferred(request, { markdownMediaTypes: ['text/markdown', 'text/x-markdown'] }) &&
    !/\.(md|txt)$/.test(pathname) &&
    (i18n.languages as string[]).includes(locale) &&
    rewriteDocs(pathname);

  if (markdown) {
    return NextResponse.rewrite(new URL(markdown, request.nextUrl), {
      // the URL has two representations, selected by `Accept`
      headers: { Vary: 'Accept' },
    });
  }

  return i18nProxy(request, event);
}

export const config = {
  // Skip routes that live outside `app/[lang]`. `/llms.txt` and
  // `/llms-full.txt` must be served as is: the i18n proxy would redirect them
  // to `/en/llms.txt`.
  matcher: [
    '/((?!api|_next/static|_next/image|favicon.ico|og|llms.txt|llms-full.txt).*)',
  ],
};
