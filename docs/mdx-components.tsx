import defaultMdxComponents from 'fumadocs-ui/mdx';
import * as TabsComponents from 'fumadocs-ui/components/tabs';
import type { MDXComponents } from 'mdx/types';
import type { ComponentProps } from 'react';
import { Mermaid } from '@/components/mdx/mermaid';

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    ...TabsComponents,
    // Keep every tab panel in the HTML, hidden until selected, so search
    // engines and agents reading the page see all of them. Fumadocs UI 16.12
    // stopped mounting inactive panels; a tab can still opt out.
    Tab: (props: ComponentProps<typeof TabsComponents.Tab>) => (
      <TabsComponents.Tab forceMount {...props} />
    ),
    Mermaid,
    ...components,
  };
}
