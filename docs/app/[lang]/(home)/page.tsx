import Link from 'next/link';
import { homeContent } from '@/lib/home';
import { i18n } from '@/lib/i18n';

export function generateStaticParams() {
  return i18n.languages.map((lang) => ({ lang }));
}

export default async function HomePage({
  params,
}: {
  params: Promise<{ lang: string }>;
}) {
  const { lang } = await params;
  const t = homeContent(lang);

  return (
    <div className="flex flex-col justify-center text-center flex-1 px-4">
      <h1 className="text-4xl font-bold mb-4">Xybrid</h1>
      <p className="text-xl text-muted-foreground mb-8">{t.tagline}</p>
      <p className="text-lg mb-8 max-w-2xl mx-auto">{t.description}</p>
      <div className="flex gap-4 justify-center">
        <Link
          href={`/${lang}/docs/quickstart`}
          className="inline-flex items-center justify-center rounded-md border border-input bg-background px-6 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
        >
          {t.quickstart}
        </Link>
      </div>
    </div>
  );
}
