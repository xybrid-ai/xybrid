# Releasing @xybrid/observability

Releases start by pushing a `release/v<version>` branch, following the root
AGENTS.md release process. Do not publish from a feature branch.
`just bump-version` synchronizes the package manifest, `src/version.ts`, the
Rust workspace, and `@xybrid/web`'s exact observability dependency. The local
pnpm link is an override and never appears in the published manifest.

`release-prep.yml` tests and builds this package in the Web SDK job, validates
the actual npm archive in an isolated consumer, and uses that archive to check
web's exported declarations. Core imports are checked with only the API and
logs API installed; the OTLP declarations are checked with the pinned SDK peers.
The package's exported telemetry `VERSION` must match the archive manifest.
The draft release includes `xybrid-observability-<version>.tgz` and its SHA-256
checksum alongside the web assets. A failure blocks the draft and release PR.

After the release PR merges, `publish-observability-npm` checks out the release
tag, downloads the tested archive, checks its checksum and package contents,
and stages those exact bytes with npm provenance. It never rebuilds the
release package. Stable versions use `latest`; prereleases use `next`. Reruns
skip versions already public on npm.

## One-time npm setup

1. Configure a publisher for public packages in the `@xybrid` scope. npm staging
   requires an existing package, so bootstrap the first approved observability
   version from its tested GitHub release archive before enabling staging.
2. Configure `@xybrid/observability`'s trusted publishing on npmjs.com for
   repository `xybrid-ai/xybrid`, workflow `release-publish.yml`, with stage-only
   permission. The workflow requests GitHub OIDC and provenance. Until that is
   configured, supply a stage-only granular token scoped to this package in
   secret `NPM_OBSERVABILITY_TOKEN`.
3. Set `NPM_OBSERVABILITY_PUBLISH_ENABLED=true` for standalone publishing.
   `NPM_WEB_PUBLISH_ENABLED=true` also enables this dependency publisher.
4. Approve the staged observability version with 2FA on npmjs.com before web.
   Web staging waits for the exact version to be public for up to 30 minutes,
   then fails without staging if it is unavailable. After approving the
   dependency, rerun failed jobs using the command printed in the workflow.

Both npm publishers wait for the shared `verify-spm-checksum` release gate.
Staging requires Node >=22.14 and npm >=11.15; the workflow uses Node 22 and
npm 11. A full dispatch of `release-publish.yml` on the release tag
(`flutter_only=false`) supports recovery if merge-triggered OIDC is rejected.

## Local verification

```sh
pnpm install --frozen-lockfile
pnpm lint
pnpm typecheck
pnpm test
pnpm pack:release
```

Source `pack:release` builds before packing, including when invoked directly
with `node scripts/check-pack.mjs`. To verify a downloaded asset without
building or rewriting it, install development dependencies and run:

```sh
node scripts/check-pack.mjs /path/to/xybrid-observability-<version>.tgz <version>
```

Generated bundles, archives and staged license files are removed by
`./clean.sh --apply`. SDK 2.x peer compatibility is outside this release;
the supported pinned peer versions are listed in README.md.
