# Releasing @xybrid/web

The browser GGUF preview ships with the workspace version. `just bump-version`
updates both JS package manifests, their telemetry version sources, the exact
observability dependency, and the Rust workspace together. Start
releases through a `release/v<version>` branch; the release pipeline creates the
release PR.

## What ships

The npm archive `xybrid-web-<version>.tgz` includes:

- ESM client, shared JavaScript chunks, source maps and TypeScript declarations.
- `dist/worker.js`, kept beside the client entry point.
- `dist/runtime/wasm/xybrid_runtime.{js,wasm}` for CPU/WASM SIMD.
- `dist/runtime/webgpu/xybrid_runtime.{js,wasm}` for experimental WebGPU.
- README, Apache license, copyright notice and bundled dependency license notices.

`@xybrid/web` depends on `@xybrid/observability` at the same version (its
declarations reference the observer types; the bundle contains none of its
code). In this repository `pnpm-workspace.yaml` links the sibling
`bindings/observability-js`, and `scripts/prepare-observability.mjs` (run before
`test`, `typecheck` and `build`) installs and builds it. `npm pack` never sees
that link, so the archive names the plain version; `pack:release` fails if it
does not. Source packing also prepares the sibling when called directly.
Release verification consumes the checksummed observability archive instead of
building a sibling package. `@xybrid/observability` must be public on npm before
web can be staged; a staged dependency still awaiting approval cannot be installed.

Models, examples, tests, source build scripts and toolchains stay out of the
archive. Applications serve the runtime directory on their own origin as
described in [README.md](./README.md). Consumers need no native build tools.

## Per release

1. `release-prep.yml` tests, builds, packs and verifies `@xybrid/observability`,
   then builds both runtimes with the pinned Bazel toolchains at
   the release version and bundles the client. It packs the archive, checks its
   contents, imports the ESM in an isolated consumer, and typechecks the exported
   API. Chromium then runs the extracted package and verifies CPU generation,
   cancellation and both runtimes' actual Rust versions. Hardware WebGPU
   generation remains an opt-in test.
2. The draft GitHub Release includes both tested archives and their
   `xybrid-web-<version>.tgz.sha256` /
   `xybrid-observability-<version>.tgz.sha256` checksums. Failure of the web build or tests blocks
   creation of the draft release and release PR. To install before npm approval,
   verify both checksums and run
   `npm install ./xybrid-observability-<version>.tgz ./xybrid-web-<version>.tgz @opentelemetry/api`.
   Installing only the web archive requires observability to be public on npm.
   CLI-only dry-run rehearsals
   continue to skip this job.
3. After the release PR merges, `release-publish.yml` downloads that same
   archive, verifies its checksum, contents and version, and stages it on npm
   with provenance. It uses `next` for prereleases and `latest` for stable
   versions. A maintainer approves the staged package on npmjs.com with 2FA.
   Reruns skip a version already public on npm. The observability job stages its
   archive first. Approve observability on npm, then web; the web job waits up
   to 30 minutes for the exact dependency version to be public, and otherwise
   fails with a rerun command without staging web.

The npm job is off until repository variable `NPM_WEB_PUBLISH_ENABLED` is `true`.
Enabling web also enables `publish-observability-npm`, which uses its own trusted
publisher or `NPM_OBSERVABILITY_TOKEN`; see
[the observability release guide](../observability-js/RELEASING.md).
GitHub release packaging runs regardless of those variables. The web npm job has
no dependency on React Native or Maven Central. Like every package publisher, it
waits for `verify-spm-checksum`, so a release whose XCFramework fails its
checksum check stages nothing on npm.

## One-time npm setup

1. Ensure the publishing account can publish public packages in the `@xybrid`
   scope. npm cannot stage a package that does not exist yet: bootstrap the first
   approved version of `@xybrid/observability` first, then `@xybrid/web`, using
   their tested GitHub release tarballs and the appropriate `next` or `latest`
   tag. Complete observability's trusted-publisher setup as well.
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

Routine PR CI runs `pnpm test:release` before building the Rust/WASM runtimes.
Changes to either JS package's scripts, manifests, lockfiles or pnpm workspace
configuration also select the tooling release/version regression tests.

## Verify locally

To try an already published GitHub release in the browser without compiling the
runtime or publishing to npm, run `pnpm demo:release 0.13.0`. It verifies the
release archive and launches the example; see [README.md](./README.md).

```sh
cd bindings/web
pnpm install --frozen-lockfile
pnpm lint
pnpm typecheck
pnpm test
pnpm test:release
pnpm build
pnpm pack:release
XYBRID_WEB_PACKAGE=test-results/npm/package pnpm build:example
pnpm exec playwright install --with-deps chromium
pnpm test:browser
```

To check a downloaded archive without rebuilding it, install the development
dependencies and run:

```sh
node scripts/check-pack.mjs /path/to/xybrid-web-<version>.tgz <version> /path/to/xybrid-observability-<version>.tgz
```

The optional second argument selects the expected archive version (otherwise
the checkout's version is used). The third argument supplies the exact
observability archive; when omitted during downloaded-archive verification,
the verifier fetches that exact version from npm. No sibling install or build
is needed in archive mode. The verifier also leaves an extracted copy in `test-results/npm/package` for browser tests.
`./clean.sh --apply` removes these generated outputs and the staged license/notices.
