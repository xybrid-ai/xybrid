import type { MetadataRoute } from 'next';
import { DOCS_ORIGIN } from '@/lib/analytics';
import { localizedPagePaths } from '@/lib/source';

export default function sitemap(): MetadataRoute.Sitemap {
  return Array.from(new Set(localizedPagePaths()), (path) => ({ url: `${DOCS_ORIGIN}${path}` }));
}
