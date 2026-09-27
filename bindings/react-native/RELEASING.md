# Releasing @xybrid/react-native

The npm package ships with every Xybrid release; there is no separate
React Native release. This page covers what the pipeline does for it and the
one-time setup it needs.

## What ships

| Piece | Where consumers get it |
|---|---|
| TypeScript facade, Codegen spec, iOS/Android shims | the npm tarball (~120 KiB) |
| Swift SDK sources (`Xybrid.swift`, `xybrid_bolt.swift`) | the npm tarball (`ios/XybridSwift/`, staged by `scripts/prepack.sh`) |
| iOS Rust core (`XybridFFI.xcframework`) | the GitHub Release asset `XybridFFI-v<version>.xcframework.zip`, downloaded at `pod install` and checked against `package.json` → `xybrid.iosXcframeworkSha256` |
| Android Kotlin SDK + native libraries | `ai.xybrid:xybrid-kotlin:<version>` on Maven Central |

All three are the same version: `just bump-version` rewrites the package
version and the Kotlin SDK pin (`tools/scripts/version-sync.sh`).

## Per release — automatic

1. **`release-prep.yml`** builds the XCFramework, then the *Patch checksums*
   job writes its SHA-256 into both `Package.swift` and
   `bindings/react-native/package.json` (`scripts/pin-ios-checksum.sh`) and
   pushes that to the release branch.
2. **`release-publish.yml`**, after the release PR merges:
   - `verify-spm-checksum` downloads the now-public zip and checks both pins
     (`pin-ios-checksum.sh --check`);
   - `publish-npm` runs the test suite and the tarball check at the tag,
     waits (up to 30 minutes) until Maven Central serves
     `ai.xybrid:xybrid-kotlin:<version>` — without it the package's Android
     build fails — then runs `npm stage publish --provenance`. A maintainer
     approves the staged version on npmjs.com with 2FA. Versions with a
     prerelease suffix (`0.10.0-rc1`) go to the `next` dist-tag, others to
     `latest` after approval. Re-runs skip a version already live on npm.

If the wait times out, the Kotlin deployment is usually sitting unreleased in
the Central Portal (central.sonatype.com → Deployments). Release it, wait for
the sync, then re-run the failed job (`gh run rerun <run-id> --failed`).

`publish-npm` is **off** until the repository variable `NPM_PUBLISH_ENABLED`
is `true`.

## One-time setup

1. **Own the `@xybrid` scope.** The package is `@xybrid/react-native`, so the
   `xybrid` organization must exist on npmjs.com (Add Organization, free for
   public packages) and the publishing account must be able to publish in it.
   The CocoaPods pod stays `react-native-xybrid` (pod names cannot contain
   `@` or `/`).
2. **Stage-only token.** Until trusted publishing is configured, save a
   stage-only granular token with write access to `@xybrid/react-native` as the
   repository secret `NPM_TOKEN`. Set `NPM_PUBLISH_ENABLED=true`. The first
   package version was published separately because npm cannot stage a package
   that does not exist yet.
3. **Trusted publishing (optional).** On npmjs.com → package → Settings →
   Trusted publishing, add GitHub Actions: repository `xybrid-ai/xybrid`,
   workflow `release-publish.yml`, with **stage-only** permission. Then delete
   the `NPM_TOKEN` secret. Every later staging request authenticates with a
   short-lived OIDC token and carries provenance.

For each release, review the staged version in npmjs.com's **Staged Packages**
tab and approve it with 2FA. The version and its `next` or `latest` tag become
public only after approval. Staging requires npm CLI >= 11.15.0 and Node.js >=
22.14.0; the release workflow installs npm 11 on Node 22.

If npm ever rejects the OIDC token of the merge (`pull_request`) run, use the
same pattern as pub.dev: stage from a `workflow_dispatch` of
`release-publish.yml` on the tag.

## Checking a release candidate by hand

```sh
cd bindings/react-native
npm ci && npm test && node scripts/check-pack.mjs
npm pack                                   # xybrid-react-native-<version>.tgz
# In a fresh app (Expo: npx create-expo-app, then a dev build):
npm install /path/to/xybrid-react-native-<version>.tgz
XYBRID_XCFRAMEWORK_PATH=/path/to/XybridFFI.xcframework.zip npx pod-install
```

Before the release is public, `pod install` cannot download the XCFramework,
so point `XYBRID_XCFRAMEWORK_PATH` at the draft release's zip or a local
Bazel build (`bazel build --config=ios //bindings/apple:XybridFFI`).
