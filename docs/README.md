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
| `app/robots.txt/route.ts`, `app/sitemap.ts` | Root robots and sitemap endpoints, outside language redirects. |

## Production analytics

Docs analytics is **off by default**. Only after the actual published notice
has completed its 30-day period, configure `NEXT_PUBLIC_ANALYTICS_ENABLED=true`,
`NEXT_PUBLIC_ANALYTICS_NOTICE_EFFECTIVE_AT` (UTC `YYYY-MM-DDTHH:mm:ssZ`) and
the server-only `POSTHOG_DOCS_KEY` for the isolated US Xybrid analytics project.
Set the effective timestamp from the actual notice and do not backdate it.
Enable **Discard client IP data** on that project before turning collection on.
Build and deploy with the same gate values; `NEXT_PUBLIC_*` is inlined at build time.
Before enabling collection, configure a Vercel Firewall SDK rate limit with ID
`docs-analytics-relay` and a threshold appropriate for the analytics budget.
The relay uses one shared bucket across visitors and forwards nothing if the
limit is exceeded, missing, or unavailable. No browser headers are sent to
the rate-limit check. Optionally add a separate per-IP edge limit on `POST /relay`.

Only HTTPS `docs.xybrid.dev` sends events. The first-party `/relay` forwards
allowlisted pageviews, exits and explicit outbound navigation to Console or
GitHub without forwarding browser headers, URLs with queries or fragments,
or arbitrary campaign values. Only the initial pageview may include the external
referring hostname; no referrer URL, path, port, query, or fragment is sent.
No docs page has a setup-prompt copy control to
instrument. A random in-memory ID is replaced on each page visit; it is never
stored or used to identify a person. Heatmaps, replay, autocapture, dead-click
and performance capture are off by construction: there is no PostHog browser
SDK or remote configuration, only explicit events. The relay preserves each
event's occurrence time and drops timestamps more than five minutes from its
clock. Treat exits as best effort. Run `pnpm test` for the analytics checks;
the docs build runs them too.

### Fumadocs MDX

A `source.config.ts` config file has been included, you can customise different options like frontmatter schema.

Read the [Introduction](https://fumadocs.dev/docs/mdx) for further details.

## Agent Discovery

- HTML is the default. Send `Accept: text/markdown` to localized home and docs URLs, or use `/<lang>/index.md`, individual docs `.md` URLs, and `/<lang>/llms.txt` for explicit representations. Preserve the CDN no-store policy for negotiated URLs: Next.js can replace the proxy's `Vary: Accept` with its App Router headers.
- Root `robots.txt` allows search and AI-assisted reading, opts out of training, and directs crawlers to the generated sitemap. The live `/api/` handlers are not crawl targets; direct MCP connections do not require authentication.
- The API catalog links to the public docs MCP server. Its Server Card lives at `/api/mcp/server-card`; the well-known compatibility card also advertises the live tool capability. `/.well-known/ai-catalog.json` links to both the MCP discovery surface and the public skills index. The build hashes the exact skill bytes served by the site.
- The docs site is not an OAuth issuer or protected OAuth resource and does not publish `auth.md`. The separate Xybrid platform API key flow is not advertised as OAuth.
- DNS-AID requires an authoritative DNS change outside this build. After deployment, configure the index binding at `_index._agents.docs.xybrid.dev` with the DNS provider. For clients that use HTTPS and the well-known AI Catalog path at the index host, the proposed zone record is `_index._agents.docs.xybrid.dev. 3600 IN SVCB 1 docs.xybrid.dev. alpn=h2 port=443`. Confirm client support, publish the zone's DNSSEC delegation, and verify the SVCB record and chain through a validating resolver. Do not advertise an A2A service for this MCP-only site.

To check discovery against a production build, run `pnpm build && pnpm start` and, in another terminal, `node --test tests/discovery.test.mjs`. Set `DOCS_TEST_ORIGIN` to check another deployed build instead.

## Learn More

To learn more about Next.js and Fumadocs, take a look at the following
resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js
  features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
- [Fumadocs](https://fumadocs.dev) - learn about Fumadocs
