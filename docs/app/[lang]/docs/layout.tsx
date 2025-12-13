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
      sidebar={{ tabs: false }}
      tree={source.pageTree[lang]}
      {...baseOptions(lang)}
    >
      {children}
    </DocsLayout>
  );
}
