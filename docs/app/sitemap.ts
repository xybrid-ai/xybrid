import type { MetadataRoute } from 'next';
import { i18n } from '@/lib/i18n';
import { SITE, source } from '@/lib/source';

export default function sitemap(): MetadataRoute.Sitemap {
  const paths = new Set(i18n.languages.flatMap((lang) => [
    `/${lang}`,
    ...source.getPages(lang).map((page) => page.url),
  ]));
  return [...paths].map((path) => ({ url: `${SITE}${path}` }));
}
