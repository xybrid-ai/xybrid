import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { SITE } from '@/lib/discovery';

export const dynamic = 'force-static';

const path = '/.well-known/agent-skills/xybrid-docs/SKILL.md';

export async function GET() {
  const artifact = await readFile(join(process.cwd(), 'public', path));
  return Response.json({
    $schema: 'https://schemas.agentskills.io/discovery/0.2.0/schema.json',
    skills: [{
      name: 'xybrid-docs',
      type: 'skill-md',
      description: 'Find and read authoritative Xybrid SDK documentation for the current task.',
      url: `${SITE}${path}`,
      digest: `sha256:${createHash('sha256').update(artifact).digest('hex')}`,
    }],
  }, { headers: { 'Access-Control-Allow-Origin': '*' } });
}
