import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions } from '@/lib/layout.shared';

export default async function Layout({
  params,
  children,
}: {
  params: Promise<{ lang: string }>;
  children: React.ReactNode;
}) {
  const { lang } = await params;

  return (
    <DocsLayout
      tabs={false}
      tree={source.getPageTree(lang)}
      {...baseOptions(lang)}
    >
      {children}
    </DocsLayout>
  );
}
