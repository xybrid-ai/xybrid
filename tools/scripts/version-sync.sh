#!/usr/bin/env bash
# version-sync.sh — Sync version across all packages from Cargo workspace
#
# Usage:
#   ./tools/scripts/version-sync.sh              # Read from Cargo.toml, update non-Rust files
#   ./tools/scripts/version-sync.sh 0.2.0        # Set version everywhere
#   ./tools/scripts/version-sync.sh --check      # Verify all versions match (for CI)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Files that hold version declarations
CARGO_WORKSPACE="$REPO_ROOT/Cargo.toml"
FLUTTER_PUBSPEC="$REPO_ROOT/bindings/flutter/pubspec.yaml"
UNITY_PACKAGE="$REPO_ROOT/bindings/unity/package.json"
KOTLIN_GRADLE="$REPO_ROOT/bindings/kotlin/build.gradle.kts"

# Extract current workspace version from Cargo.toml
get_cargo_version() {
    grep -A5 '\[workspace\.package\]' "$CARGO_WORKSPACE" \
        | grep '^version' \
        | head -1 \
        | sed 's/.*= *"\(.*\)"/\1/'
}

# Extract version from Flutter pubspec.yaml
get_flutter_version() {
    grep '^version:' "$FLUTTER_PUBSPEC" | sed 's/version: *//'
}

# Extract version from Unity package.json
get_unity_version() {
    python3 -c "import json; print(json.load(open('$UNITY_PACKAGE'))['version'])"
}

# Extract version from Kotlin build.gradle.kts
get_kotlin_version() {
    grep '^version = ' "$KOTLIN_GRADLE" | sed 's/version = "\(.*\)"/\1/'
}

# Set version in Cargo workspace (all Rust crates inherit via version.workspace = true)
set_cargo_version() {
    local version="$1"
    sed -i.bak "s/^version = \".*\"/version = \"$version\"/" "$CARGO_WORKSPACE"
    rm -f "$CARGO_WORKSPACE.bak"
}

# Set version in Flutter pubspec.yaml
set_flutter_version() {
    local version="$1"
    sed -i.bak "s/^version: .*/version: $version/" "$FLUTTER_PUBSPEC"
    rm -f "$FLUTTER_PUBSPEC.bak"
}

# Set version in Unity package.json
set_unity_version() {
    local version="$1"
    python3 -c "
import json
with open('$UNITY_PACKAGE', 'r') as f:
    data = json.load(f)
data['version'] = '$version'
with open('$UNITY_PACKAGE', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
}

# Set version in Kotlin build.gradle.kts
set_kotlin_version() {
    local version="$1"
    sed -i.bak "s/^version = \".*\"/version = \"$version\"/" "$KOTLIN_GRADLE"
    rm -f "$KOTLIN_GRADLE.bak"
}

# Check mode: verify all versions match
check_versions() {
    local cargo_version
    cargo_version="$(get_cargo_version)"
    local exit_code=0

    echo "Cargo workspace version: $cargo_version"
    echo ""

    for name_func in "Flutter:get_flutter_version" "Unity:get_unity_version" "Kotlin:get_kotlin_version"; do
        local name="${name_func%%:*}"
        local func="${name_func##*:}"
        local version
        version="$($func 2>/dev/null || echo "NOT FOUND")"

        if [ "$version" = "$cargo_version" ]; then
            echo "  $name: $version ✓"
        else
            echo "  $name: $version ✗ (expected $cargo_version)"
            exit_code=1
        fi
    done

    echo ""
    if [ "$exit_code" -eq 0 ]; then
        echo "All versions in sync."
    else
        echo "Version mismatch detected!"
    fi
    return $exit_code
}

# Main
case "${1:-}" in
    --check)
        check_versions
        ;;
    "")
        # Sync non-Rust files to match Cargo workspace version
        VERSION="$(get_cargo_version)"
        echo "Syncing all packages to version: $VERSION"
        set_flutter_version "$VERSION"
        set_unity_version "$VERSION"
        set_kotlin_version "$VERSION"
        echo "Done. Run '$0 --check' to verify."
        ;;
    --help|-h)
        echo "Usage: $0 [VERSION|--check|--help]"
        echo ""
        echo "  (no args)    Sync non-Rust packages to Cargo workspace version"
        echo "  VERSION      Set version everywhere (Cargo + all bindings)"
        echo "  --check      Verify all versions match (for CI)"
        ;;
    *)
        # Set version everywhere
        VERSION="$1"
        echo "Setting all packages to version: $VERSION"
        set_cargo_version "$VERSION"
        set_flutter_version "$VERSION"
        set_unity_version "$VERSION"
        set_kotlin_version "$VERSION"
        echo "Done. Rust crates inherit via version.workspace = true."
        echo "Run 'cargo metadata --no-deps' to verify Rust crates."
        ;;
esac
