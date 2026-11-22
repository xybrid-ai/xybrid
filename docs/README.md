# xybrid-doc

This is a Next.js application generated with
[Create Fumadocs](https://github.com/fuma-nama/fumadocs).

Run development server:

```bash
npm run dev
# or
pnpm dev
# or
yarn dev
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
