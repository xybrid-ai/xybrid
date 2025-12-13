import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: 'Xybrid',
    },
    links: [
      {
        text: 'Documentation',
        url: '/docs',
      },
      {
        text: 'GitHub',
        url: 'https://github.com/xybrid-ai/xybrid',
        external: true,
      },
    ],
    githubUrl: 'https://github.com/xybrid-ai/xybrid',
  };
}
