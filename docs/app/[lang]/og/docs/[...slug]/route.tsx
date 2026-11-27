import { notFound } from 'next/navigation';
import { generateOGImage } from 'fumadocs-ui/og';
import { SITE_NAME } from '@/lib/metadata';
import { getImageResponseOptions, getLogo } from '@/lib/og';
import { source } from '@/lib/source';

export const revalidate = false;

export async function GET(_req: Request, { params }: RouteContext<'/[lang]/og/docs/[...slug]'>) {
  const { slug, lang } = await params;
  const page = source.getPage(slug.slice(0, -1), lang);
  if (!page) notFound();

  const { title, description = '' } = page.data;
  return generateOGImage({
    title,
    description,
    site: SITE_NAME,
    icon: <img src={await getLogo()} alt="" width={72} height={72} style={{ borderRadius: 16 }} />,
    // the magenta of the logo
    primaryColor: 'rgba(217, 70, 239, 0.55)',
    primaryTextColor: 'rgb(232, 121, 249)',
    ...(await getImageResponseOptions(page.locale, `${title}${description}${SITE_NAME}`)),
  });
}

export function generateStaticParams() {
  return source.generateParams().map((item) => ({ ...item, slug: [...item.slug, 'image.png'] }));
}
