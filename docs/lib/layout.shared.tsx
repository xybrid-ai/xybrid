import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { GlobeIcon } from 'lucide-react';
import { i18n } from '@/lib/i18n';

export function baseOptions(locale: string): BaseLayoutProps {
  return {
    nav: {
      title: 'Xybrid',
    },
    i18n,
    githubUrl: 'https://github.com/xybrid-ai/xybrid',
    links: [
      {
        icon: <GlobeIcon />,
        text: 'Website',
        url: 'https://xybrid.dev',
        external: true,
        // secondary items will be displayed differently on navbar
        secondary: true,
      },
      {
        icon: (
          <img
            height="16"
            width="16"
            src="https://cdn.simpleicons.org/discord?viewbox=auto"
            alt="Discord"
          />
        ),
        text: 'Community',
        url: 'https://discord.gg/cgd3tbFPWx',
        external: true,
        secondary: false,
      },
      {
        icon: (
          <img
            height="16"
            width="16"
            src="https://cdn.simpleicons.org/github/gray?viewbox=auto"
            alt="GitHub"
          />
        ),
        text: 'GitHub',
        url: 'https://github.com/xybrid-ai/xybrid',
        external: true,
      },
    ],
  };
}
