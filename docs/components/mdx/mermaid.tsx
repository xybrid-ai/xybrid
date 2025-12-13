'use client';

import { useTheme } from 'next-themes';
import { useEffect, useId, useState } from 'react';

let importPromise: Promise<typeof import('mermaid')> | null = null;

function importMermaid() {
  if (!importPromise) {
    importPromise = import('mermaid');
  }
  return importPromise;
}

export function Mermaid({ chart }: { chart: string }) {
  const id = useId();
  const { resolvedTheme } = useTheme();
  const [svg, setSvg] = useState<string>('');

  useEffect(() => {
    const render = async () => {
      const mermaid = (await importMermaid()).default;

      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        fontFamily: 'inherit',
        theme: resolvedTheme === 'dark' ? 'dark' : 'default',
      });

      try {
        const result = await mermaid.render(
          `mermaid-${id.replace(/:/g, '')}`,
          chart
        );
        setSvg(result.svg);
      } catch (error) {
        console.error('Mermaid rendering error:', error);
      }
    };

    render();
  }, [chart, id, resolvedTheme]);

  return (
    <div
      className="my-4 flex justify-center [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
