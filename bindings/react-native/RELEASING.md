# Releasing react-native-xybrid

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
     build fails — then runs `npm publish --provenance`. Versions with a
     prerelease suffix (`0.10.0-rc1`) go to the `next` dist-tag, others to
     `latest`. Re-runs skip a version already on npm.

If the wait times out, the Kotlin deployment is usually sitting unreleased in
the Central Portal (central.sonatype.com → Deployments). Release it, wait for
the sync, then re-run the failed job (`gh run rerun <run-id> --failed`).

`publish-npm` is **off** until the repository variable `NPM_PUBLISH_ENABLED`
is `true`.

## One-time setup

1. **Pick the package name.** The code uses `react-native-xybrid`. If you
   prefer the scope the web package uses (`@xybrid/react-native`), rename it in
   `package.json`, `tools/scripts/version-sync.sh` (nothing else keys on the
   name) and the docs, and create the `xybrid` npm organization.
2. **Bootstrap publish.** npm only lets you configure a trusted publisher on a
   package that exists, so the first version needs a token: create a granular
   access token (publish scope, short expiry), save it as the repository secret
   `NPM_TOKEN`, set `NPM_PUBLISH_ENABLED=true`, and let the next release publish.
   (Or publish that version by hand from a clean checkout of the tag:
   `cd bindings/react-native && npm ci && npm publish --access public`.)
3. **Trusted publishing.** On npmjs.com → package → Settings → Trusted
   publishing, add GitHub Actions: repository `xybrid-ai/xybrid`, workflow
   `release-publish.yml`. Then delete the `NPM_TOKEN` secret and, under
   Publishing access, require 2FA and disallow tokens. Every later publish
   authenticates with a short-lived OIDC token and carries provenance.

If npm ever rejects the OIDC token of the merge (`pull_request`) run, use the
same pattern as pub.dev: publish from a `workflow_dispatch` of
`release-publish.yml` on the tag.

## Checking a release candidate by hand

```sh
cd bindings/react-native
npm ci && npm test && node scripts/check-pack.mjs
npm pack                                   # react-native-xybrid-<version>.tgz
# In a fresh app (Expo: npx create-expo-app, then a dev build):
npm install /path/to/react-native-xybrid-<version>.tgz
XYBRID_XCFRAMEWORK_PATH=/path/to/XybridFFI.xcframework.zip npx pod-install
```

Before the release is public, `pod install` cannot download the XCFramework,
so point `XYBRID_XCFRAMEWORK_PATH` at the draft release's zip or a local
Bazel build (`bazel build --config=ios //bindings/apple:XybridFFI`).
