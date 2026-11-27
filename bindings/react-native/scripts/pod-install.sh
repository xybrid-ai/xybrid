#!/usr/bin/env bash
set -euo pipefail

report() {
    if [[ "${GITHUB_ACTIONS:-}" == true ]]; then
        printf '::%s::%s\n' "$1" "$2" >&2
    else
        printf '%s: %s\n' "$1" "$2" >&2
    fi
}

die() {
    report error "$1"
    exit 1
}

here="$(cd "$(dirname "$0")" && pwd)"
app="$(cd "${1:-$here/../example}" && pwd)"
properties="$app/node_modules/react-native/sdks/hermes-engine/version.properties"
[[ -f "$properties" ]] || die "missing $properties; run npm ci in $app first"
[[ -f "$app/ios/Podfile" ]] || die "missing $app/ios/Podfile; run Expo prebuild first"

key=HERMES_V1_VERSION_NAME
if [[ "${RCT_HERMES_V1_ENABLED:-}" == 0 ]]; then
    key=HERMES_VERSION_NAME
fi
version="$(sed -n "s/^$key=//p" "$properties" | tr -d '\r')"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "invalid $key in $properties: $version"

fetch() {
    curl --fail --location --silent --show-error --retry 5 --retry-all-errors \
        --connect-timeout 15 --max-time 300 "$@"
}

sha256() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    else
        sha256sum "$1" | cut -d ' ' -f 1
    fi
}

artifact="hermes-ios-$version-hermes-ios-debug.tar.gz"
repository="${ENTERPRISE_REPOSITORY:-https://repo1.maven.org/maven2}"
url="${repository%/}/com/facebook/hermes/hermes-ios/$version/$artifact"
expected="$(fetch "$url.sha256")" || die "could not fetch the Hermes $version SHA-256"
[[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] || die "invalid SHA-256 for Hermes $version"
expected="$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')"

cache="${XYBRID_HERMES_DIR:-$HOME/.xybrid/cache/hermes}/$expected"
mkdir -p "$cache"
cache="$(cd "$cache" && pwd)"
tarball="$cache/$artifact"
scratch="$(mktemp -d "$cache/.pod-install.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

if [[ ! -f "$tarball" || "$(sha256 "$tarball")" != "$expected" ]]; then
    fetch --output "$scratch/$artifact" "$url" || die "could not download Hermes $version"
    [[ "$(sha256 "$scratch/$artifact")" == "$expected" ]] || die "Hermes $version SHA-256 mismatch"
    mv -f "$scratch/$artifact" "$tarball"
fi

cd "$app/ios"
log="$scratch/pod.log"
pin='Using pre-built binary from local path defined by HERMES_ENGINE_TARBALL_PATH'
for attempt in 1 2 3; do
    status=0
    HERMES_ENGINE_TARBALL_PATH="$tarball" pod install 2>&1 | tee "$log" || status=$?
    if [[ "$status" != 0 ]]; then
        reason="pod install exited with $status"
    elif grep -Eq 'No prebuilt artifacts found|Using the latest commit from|Using release tarball from URL' "$log"; then
        reason="CocoaPods fell back from the required prebuilt artifacts"
    elif ! grep -Fq "$pin" "$log"; then
        reason="CocoaPods did not report using the pinned Hermes tarball"
    elif ! grep -Fq "hermes-engine ($version)" Podfile.lock; then
        reason="Podfile.lock did not resolve Hermes $version"
    else
        echo "==> pod install: Hermes $version pinned from $tarball"
        exit 0
    fi
    rm -rf Pods Podfile.lock build
    if [[ "$attempt" == 3 ]]; then
        die "pod install failed after 3 attempts: $reason"
    fi
    report warning "pod install attempt $attempt: $reason; retrying"
    sleep "${XYBRID_POD_RETRY_DELAY:-20}"
done
