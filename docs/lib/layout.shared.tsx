import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { uiTranslations } from 'fumadocs-ui/i18n';
import { zhCN } from '@fumadocs/language/zh-cn';
import { GlobeIcon } from 'lucide-react';
import { i18n } from '@/lib/i18n';

// The UI strings of every language: the zh-CN pack of Fumadocs, then our own
// wording on top. Keys are the English label plus its context, like
// `Search(search dialog)`; `add()` type-checks them.
export const translations = i18n
  .translations()
  .extend(uiTranslations())
  .preset('zh', zhCN())
  .add({
    en: {
      displayName: 'English',
    },
    zh: {
      displayName: '中文',
      'Search(search trigger)': '搜索文档',
      'Search(search dialog)': '搜索文档',
      'No results found(search dialog)': '没有找到结果',
      'On this page(table of contents)': '目录',
      'Last updated on(page footer)': '最后更新',
      // not in the pack yet; the menu button of the home page on mobile
      'Toggle Menu(mobile menu)(aria-label)': '切换菜单',
    },
  });

export function baseOptions(locale: string): BaseLayoutProps {
  return {
    nav: {
      title: 'Xybrid',
      // the home page of the current language, `/` would redirect by Accept-Language
      url: `/${locale}`,
    },
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
        url: 'https://discord.gg/YhFHHkhbad',
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
