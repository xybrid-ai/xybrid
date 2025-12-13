import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

// Note: Orama doesn't support Chinese (zh) directly, so we use 'english' as fallback
// Chinese search will still work but won't benefit from language-specific tokenization
export const { GET } = createFromSource(source, {
  localeMap: {
    en: 'english',
    zh: 'english', // Fallback - Orama doesn't support Chinese
  },
});
