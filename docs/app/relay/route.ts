import { relayEvent } from '@/lib/analytics-relay';
import { localizedPagePaths } from '@/lib/source';

const paths = new Set(localizedPagePaths());

export function POST(request: Request) {
  return relayEvent(request, paths);
}
