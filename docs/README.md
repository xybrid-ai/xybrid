# xybrid-doc

This is a Next.js application generated with
[Create Fumadocs](https://github.com/fuma-nama/fumadocs).

Run development server:

```bash
pnpm dev
```

Open http://localhost:3000 with your browser to see the result.

## Explore

In the project, you can see:

- `lib/source.ts`: Code for content source adapter, [`loader()`](https://fumadocs.dev/docs/headless/source-api) provides the interface to access your content.
- `lib/layout.shared.tsx`: Shared options for layouts, optional but preferred to keep.

| Route                     | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `app/(home)`              | The route group for your landing page and other pages. |
| `app/docs`                | The documentation layout and pages.                    |
| `app/api/search/route.ts` | The Route Handler for search.                          |
| `app/robots.ts`, `app/sitemap.ts` | Root robots and sitemap endpoints, outside language redirects. |

## Production analytics

Docs analytics is **off by default**. Only after the actual published notice
has completed its 30-day period, configure `NEXT_PUBLIC_ANALYTICS_ENABLED=true`,
`NEXT_PUBLIC_ANALYTICS_NOTICE_EFFECTIVE_AT` (UTC `YYYY-MM-DDTHH:mm:ssZ`) and
the server-only `POSTHOG_DOCS_KEY` for the isolated US Xybrid analytics project.
Set the effective timestamp from the actual notice and do not backdate it.
Enable **Discard client IP data** on that project before turning collection on.
Build and deploy with the same gate values; `NEXT_PUBLIC_*` is inlined at build time.

Only HTTPS `docs.xybrid.dev` sends events. The first-party `/relay` forwards
allowlisted pageviews, exits and explicit outbound navigation to Docs, Console
or GitHub without forwarding browser headers, URLs with queries or fragments,
or arbitrary campaign values. Only the initial pageview may include the external
referring hostname; no referrer URL, path, port, query, or fragment is sent.
No docs page has a setup-prompt copy control to
instrument. A random in-memory ID is replaced on each page visit; it is never
stored or used to identify a person. Heatmaps, replay, autocapture, dead-click
and performance capture are off by construction: there is no PostHog browser
SDK or remote configuration, only explicit events. Treat exits as best effort.

### Fumadocs MDX

A `source.config.ts` config file has been included, you can customise different options like frontmatter schema.

Read the [Introduction](https://fumadocs.dev/docs/mdx) for further details.

## Learn More

To learn more about Next.js and Fumadocs, take a look at the following
resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js
  features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
- [Fumadocs](https://fumadocs.dev) - learn about Fumadocs
