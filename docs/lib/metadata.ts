/** The name of the site, in titles and share images. */
export const SITE_NAME = 'Xybrid';

/** The origin that metadata URLs, like share images, resolve against. */
export const baseUrl =
  process.env.NODE_ENV === 'development' || !process.env.VERCEL_PROJECT_PRODUCTION_URL
    ? new URL('http://localhost:3000')
    : new URL(`https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`);
