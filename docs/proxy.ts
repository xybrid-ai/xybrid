import { type NextFetchEvent, type NextRequest, NextResponse } from 'next/server';
import { createI18nMiddleware } from 'fumadocs-core/i18n/middleware';
import { isMarkdownPreferred, rewritePath } from 'fumadocs-core/negotiation';
import { i18n } from '@/lib/i18n';
import { homeDiscoveryLinks } from '@/lib/discovery';

const i18nProxy = createI18nMiddleware(i18n);

// `/en/docs/quickstart` -> `/en/llms.mdx/docs/quickstart.md`
const { rewrite: rewriteDocs } = rewritePath(
  '/:lang/docs{/*path}',
  '/:lang/llms.mdx/docs{/*path}.md',
);

export default async function proxy(request: NextRequest, event: NextFetchEvent) {
  const { pathname } = request.nextUrl;
  const prefix = pathname.split('/')[1];
  const locale = i18n.languages.find((lang) => prefix === lang);
  const home = locale !== undefined && pathname === `/${locale}`;
  const markdown = home ? `/${locale}/llms.mdx/index.md`
    : locale && !/\.(md|txt)$/.test(pathname) && rewriteDocs(pathname);

  if (markdown) {
    const prefersMarkdown = !request.headers.has('rsc') &&
      isMarkdownPreferred(request, { markdownMediaTypes: ['text/markdown', 'text/x-markdown'] });
    const response = prefersMarkdown
      ? NextResponse.rewrite(new URL(markdown, request.nextUrl))
      : (await i18nProxy(request, event)) ?? NextResponse.next();
    response.headers.append('Vary', 'Accept');
    response.headers.set('Cache-Control', 'private, no-store');
    response.headers.set('CDN-Cache-Control', 'no-store');
    if (home) response.headers.append('Link', homeDiscoveryLinks(locale));
    return response;
  }

  const response = (await i18nProxy(request, event)) ?? NextResponse.next();
  if (pathname === '/') {
    const location = response.headers.get('Location');
    const target = location ? new URL(location, request.nextUrl) : null;
    const lang = target?.pathname.split('/')[1];
    if (target) {
      target.pathname = target.pathname.replace(/\/$/, '');
      response.headers.set('Location', target.toString());
    }
    response.headers.append('Vary', 'Accept-Language');
    response.headers.append('Link', homeDiscoveryLinks(
      i18n.languages.find((candidate) => candidate === lang) ?? i18n.defaultLanguage,
    ));
  }
  return response;
}

export const config = {
  // Skip routes that live outside `app/[lang]`. `/llms.txt` and
  // `/llms-full.txt` must be served as is: the i18n proxy would redirect them
  // to `/en/llms.txt`.
  matcher: [
    '/((?!api/|_next/|favicon\\.ico$|og/|llms\\.txt$|llms-full\\.txt$|robots\\.txt$|sitemap\\.xml$|relay$|\\.well-known(?:/|$)).*)',
  ],
};
