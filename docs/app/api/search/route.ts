import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

// The default `multilingual` tokenizer segments every language, Chinese
// included, and filters results by the locale of the search dialog.
export const { GET } = createFromSource(source);
