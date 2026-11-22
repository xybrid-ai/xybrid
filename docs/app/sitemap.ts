import type { MetadataRoute } from 'next';
import { localizedPagePaths, SITE } from '@/lib/source';

export default function sitemap(): MetadataRoute.Sitemap {
  return Array.from(new Set(localizedPagePaths()), (path) => ({ url: `${SITE}${path}` }));
}
