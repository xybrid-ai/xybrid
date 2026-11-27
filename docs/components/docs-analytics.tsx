'use client';

import { usePathname } from 'next/navigation';
import { useEffect, useRef } from 'react';
import {
  ANALYTICS_RELAY,
  DOCS_ORIGIN,
  analyticsAllowed,
  campaignSource,
  outboundDestination,
  referrerDomain,
  type DocsEvent,
} from '@/lib/analytics';

function send(event: DocsEvent) {
  void fetch(ANALYTICS_RELAY, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(event),
    keepalive: true,
    credentials: 'omit',
    redirect: 'error',
    referrerPolicy: 'no-referrer',
  }).catch(() => {});
}

export function DocsAnalytics({ paths }: { paths: string[] }) {
  const pathname = usePathname();
  const sentInitialView = useRef(false);

  useEffect(() => {
    if (!pathname || !paths.includes(pathname) || window.location.origin !== DOCS_ORIGIN ||
      !analyticsAllowed(process.env.NEXT_PUBLIC_ANALYTICS_ENABLED, process.env.NEXT_PUBLIC_ANALYTICS_NOTICE_EFFECTIVE_AT)) return;

    let visit: Pick<DocsEvent, 'id' | 'path' | 'campaign'> | undefined;
    function enter() {
      if (visit || !pathname) return;
      visit = {
        id: crypto.randomUUID(),
        path: pathname,
        campaign: campaignSource(new URLSearchParams(window.location.search).get('utm_source')),
      };
      const domain = !sentInitialView.current ? referrerDomain(document.referrer) : undefined;
      sentInitialView.current = true;
      send({ ...visit, event: '$pageview', timestamp: new Date().toISOString(), referrerDomain: domain });
    }
    function leave() {
      if (!visit) return;
      send({ ...visit, event: '$pageleave', timestamp: new Date().toISOString() });
      visit = undefined;
    }
    function onLink(event: MouseEvent) {
      if (!visit || !event.isTrusted || event.defaultPrevented || event.button !== (event.type === 'auxclick' ? 1 : 0)) return;
      const link = event.composedPath().find((node) => node instanceof HTMLAnchorElement);
      if (!(link instanceof HTMLAnchorElement) || !link.href || link.hasAttribute('download')) return;
      try {
        const destination = outboundDestination(new URL(link.href), window.location.origin);
        if (destination) send({ ...visit, event: 'outbound_navigation', timestamp: new Date().toISOString(), destination });
      } catch {}
    }

    enter();
    window.addEventListener('pageshow', enter);
    window.addEventListener('pagehide', leave);
    document.addEventListener('click', onLink);
    document.addEventListener('auxclick', onLink);
    return () => {
      window.removeEventListener('pageshow', enter);
      window.removeEventListener('pagehide', leave);
      document.removeEventListener('click', onLink);
      document.removeEventListener('auxclick', onLink);
      leave();
    };
  }, [pathname, paths]);

  return null;
}
