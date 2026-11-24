import { llmsFull } from '@/lib/llms';

export const revalidate = false;

// Every page of the default language, `/zh/llms-full.txt` has the Chinese ones.
export function GET() {
  return llmsFull();
}
