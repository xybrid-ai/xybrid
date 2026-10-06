# Releasing @xybrid/web

The browser GGUF preview ships with the workspace version. `just bump-version`
updates `package.json`, `src/version.ts`, and the Rust workspace together. Start
releases through a `release/v<version>` branch; the release pipeline creates the
release PR.

## What ships

The npm archive `xybrid-web-<version>.tgz` includes:

- ESM client, shared JavaScript chunks, source maps and TypeScript declarations.
- `dist/worker.js`, kept beside the client entry point.
- `dist/runtime/wasm/xybrid_runtime.{js,wasm}` for CPU/WASM SIMD.
- `dist/runtime/webgpu/xybrid_runtime.{js,wasm}` for experimental WebGPU.
- README, Apache license, copyright notice and bundled dependency license notices.

Models, examples, tests, source build scripts and toolchains stay out of the
archive. Applications serve the runtime directory on their own origin as
described in [README.md](./README.md). Consumers need no native build tools.

## Per release

1. `release-prep.yml` builds both runtimes with the pinned Bazel toolchains at
   the release version and bundles the client. It packs the archive, checks its
   contents, imports the ESM in an isolated consumer, and typechecks the exported
   API. Chromium then runs the extracted package and verifies CPU generation,
   cancellation and both runtimes' actual Rust versions. Hardware WebGPU
   generation remains an opt-in test.
2. The draft GitHub Release includes the tested archive and
   `xybrid-web-<version>.tgz.sha256`. Failure of the web build or tests blocks
   creation of the draft release and release PR. CLI-only dry-run rehearsals
   continue to skip this job.
3. After the release PR merges, `release-publish.yml` downloads that same
   archive, verifies its checksum, contents and version, and stages it on npm
   with provenance. It uses `next` for prereleases and `latest` for stable
   versions. A maintainer approves the staged package on npmjs.com with 2FA.
   Reruns skip a version already public on npm.

The npm job is off until repository variable `NPM_WEB_PUBLISH_ENABLED` is `true`.
GitHub release packaging runs regardless of that variable. The web npm job has
no dependency on React Native, Maven Central or Apple checksums.

## One-time npm setup

1. Ensure the publishing account can publish public packages in the `@xybrid`
   scope. npm cannot stage a package that does not exist yet: bootstrap the first
   approved version of `@xybrid/web` using the tested GitHub release tarball and
   the appropriate `next` or `latest` tag.
2. On npmjs.com, configure `@xybrid/web` → Settings → Trusted publishing with
   GitHub Actions repository `xybrid-ai/xybrid`, workflow `release-publish.yml`,
   and stage-only permission. The workflow requests OIDC and attaches provenance.
   Until trusted publishing is configured, use a stage-only granular token
   scoped to `@xybrid/web` in repository secret `NPM_WEB_TOKEN`.
3. Set `NPM_WEB_PUBLISH_ENABLED=true`. The existing React Native flag and token
   remain separate. Each staged version becomes public only after npm approval.

Staging needs Node >= 22.14 and npm >= 11.15; the workflow uses Node 22 and npm
11. If a merge-triggered OIDC request is rejected, recover with a full
`release-publish.yml` dispatch on the release tag (`flutter_only=false`).

## Verify locally

```sh
cd bindings/web
pnpm install --frozen-lockfile
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm pack:release
XYBRID_WEB_PACKAGE=test-results/npm/package pnpm build:example
pnpm exec playwright install --with-deps chromium
pnpm test:browser
```

To check a downloaded archive without rebuilding it, install the development
dependencies and run:

```sh
node scripts/check-pack.mjs /path/to/xybrid-web-<version>.tgz
```

The checkout must have the same package version as the archive. The verifier
also leaves an extracted copy in `test-results/npm/package` for browser tests.
`./clean.sh --apply` removes these generated outputs and the staged license/notices.
