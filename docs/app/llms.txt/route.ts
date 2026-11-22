import { llmsIndex } from '@/lib/llms';

export const revalidate = false;

// The index in the default language, `/zh/llms.txt` has the Chinese one.
export function GET() {
  return llmsIndex();
}
