import { RootProvider } from 'fumadocs-ui/provider/next';
import { i18nProvider } from 'fumadocs-ui/i18n';
import { Inter } from 'next/font/google';
import type { Metadata } from 'next';
import { DocsAnalytics } from '@/components/docs-analytics';
import { WebMCP } from '@/components/webmcp';
import { i18n } from '@/lib/i18n';
import { translations } from '@/lib/layout.shared';
import { baseUrl, SITE_NAME } from '@/lib/metadata';
import { getPageImage, source } from '@/lib/source';
import '../global.css';

const inter = Inter({
  subsets: ['latin'],
});

// The defaults of every page; a page that sets `openGraph` replaces this one.
export async function generateMetadata({
  params,
}: {
  params: Promise<{ lang: string }>;
}): Promise<Metadata> {
  const { lang } = await params;
  const introduction = source.getPage(['introduction'], lang);

  return {
    title: { template: `%s | ${SITE_NAME}`, default: SITE_NAME },
    description: introduction?.data.description,
    metadataBase: baseUrl,
    openGraph: {
      siteName: SITE_NAME,
      // the home page shares the card of the introduction
      images: introduction && getPageImage(introduction).url,
    },
  };
}

export default async function RootLayout({
  params,
  children,
}: {
  params: Promise<{ lang: string }>;
  children: React.ReactNode;
}) {
  const { lang } = await params;
  const analyticsConfigured = process.env.NEXT_PUBLIC_ANALYTICS_ENABLED === 'true' &&
    Boolean(process.env.POSTHOG_DOCS_KEY?.trim()) && i18n.languages.some((language) => language === lang);

  return (
    <html lang={lang} className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider i18n={i18nProvider(translations, lang)}>
          {children}
          <WebMCP />
        </RootProvider>
        {analyticsConfigured && (
          <DocsAnalytics paths={[`/${lang}`, ...source.getPages(lang).map((page) => page.url)]} />
        )}
      </body>
    </html>
  );
}
