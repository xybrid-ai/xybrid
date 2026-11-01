#!/usr/bin/env bash
# Fetch the pinned LiquidAI/d1-2B snapshot the HF reference runner reads.
#
#   tools/conformance/decision_hf/download.sh
#
# Every file pins.json lists is fetched from its immutable URL
# (https://huggingface.co/<repo>/resolve/<revision>/<file>) into
# $XYBRID_DECISION_HF_CACHE/snapshots/<revision>/d1-2B/ (default:
# <repo>/target/decision-hf) and kept only if its size and sha256 match the
# pin. A file already there that matches is not fetched again. Needs curl and
# jq. Prints the snapshot directory. About 3.9 GB, almost all of it the
# bfloat16 weights. See README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PINS="$HERE/pins.json"
CACHE="${XYBRID_DECISION_HF_CACHE:-$REPO_ROOT/target/decision-hf}"

die() {
    echo "download: $*" >&2
    exit 1
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

size_file() {
    wc -c <"$1" | tr -d ' '
}

command -v curl >/dev/null || die "curl is required"
command -v jq >/dev/null || die "jq is required"

repo="$(jq -er .repo "$PINS")"
revision="$(jq -er .revision "$PINS")"
case "$revision" in
    *[!0-9a-f]* | "") die "pins.json revision is not a commit: $revision" ;;
esac
dest="$CACHE/snapshots/$revision/d1-2B"
mkdir -p "$dest"

fetched=0
while IFS=$'\t' read -r name bytes sha; do
    case "$name" in
        "" | */* | .*) die "unsafe file name in pins.json: $name" ;;
    esac
    path="$dest/$name"
    if [ -f "$path" ] && [ "$(size_file "$path")" = "$bytes" ] && [ "$(sha256_file "$path")" = "$sha" ]; then
        continue
    fi
    url="https://huggingface.co/$repo/resolve/$revision/$name"
    echo "  fetching $url"
    # HTTPS only, redirects included (the LFS files redirect to a CDN); the
    # size and sha256 below are the integrity check.
    curl -fL --proto '=https' --proto-redir '=https' --connect-timeout 30 --retry 3 -sS \
        -o "$path.part" "$url" || die "download failed: $url"
    actual="$(sha256_file "$path.part")"
    if [ "$(size_file "$path.part")" != "$bytes" ] || [ "$actual" != "$sha" ]; then
        rm -f "$path.part"
        die "$name does not match its pin (sha256 $actual, pinned $sha)"
    fi
    mv -f "$path.part" "$path"
    fetched=$((fetched + 1))
done < <(jq -r '.files | to_entries[] | [.key, (.value.bytes | tostring), .value.sha256] | @tsv' "$PINS")

echo "download: $repo@$revision verified ($fetched fetched)" >&2
echo "$dest"
