# Releasing @xybrid/observability

The package and telemetry source versions follow the Cargo workspace. Releases
use the repository's `release/v<version>` process; do not publish from a feature
branch. `just bump-version` synchronizes the package and source versions.

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
