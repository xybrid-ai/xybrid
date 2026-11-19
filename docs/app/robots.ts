import type { MetadataRoute } from 'next';
import { DOCS_ORIGIN } from '@/lib/analytics';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: '*', allow: '/', disallow: ['/api/', '/relay'] },
    sitemap: `${DOCS_ORIGIN}/sitemap.xml`,
  };
}
