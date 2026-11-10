import { createI18nMiddleware } from 'fumadocs-core/i18n/middleware';
import { i18n } from '@/lib/i18n';

export default createI18nMiddleware(i18n);

export const config = {
  // Skip routes that live outside `app/[lang]`. `/llms.txt` and
  // `/llms-full.txt` must be served as is: the i18n middleware would redirect
  // them to `/en/llms.txt`, which does not exist.
  matcher: [
    '/((?!api|_next/static|_next/image|favicon.ico|og|llms.txt|llms-full.txt).*)',
  ],
};
