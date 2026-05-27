#!/usr/bin/env bash
# =============================================================================
# verify-release.sh
# =============================================================================
# End-to-end consumer-side verification for a published xybrid release.
#
# Usage:
#   ./tools/scripts/verify-release.sh                    # uses workspace version
#   ./tools/scripts/verify-release.sh 0.1.0-rc4
#   ./tools/scripts/verify-release.sh 0.1.0-rc4 --skip-ios
#   ./tools/scripts/verify-release.sh 0.1.0-rc4 --only spm,cargo
#
# What it does, per registry, in an isolated tmp workspace:
#
#   spm     — generates a minimal SwiftPM consumer that depends on
#             `from: "<version>"`, runs `swift package resolve`, confirms
#             the xcframework zip downloads + its SHA-256 matches the
#             checksum in Package.swift at the tag.
#   ios     — runs `xcodebuild build` on examples/ios/XybridExample (iOS
#             Simulator destination, no codesign) against the current
#             repo's Package.swift. With useLocalNatives=false this
#             exercises the published xcframework end-to-end in a real
#             iOS target. macOS-only.
#   cargo   — generates a minimal Rust binary that depends on
#             `xybrid = "=<version>"`, runs `cargo fetch`, confirms the
#             crate + its three workspace deps resolve from crates.io.
#   flutter — generates a minimal Flutter app that depends on
#             `xybrid_flutter: <version>`, runs `flutter pub get`,
#             confirms it resolves from pub.dev.
#   maven   — generates a minimal Gradle project that depends on
#             `ai.xybrid:xybrid-kotlin:<version>`, runs `gradle
#             dependencies`, confirms it resolves from Maven Central.
#             Maven Central's mirror sync after publish is async (~10–60
#             min); if the artifact isn't there yet, the test fails
#             with a clear "not yet on Central" message rather than
#             a generic resolve error.
#
# Toolchains not installed are skipped (not counted as failures). Exits
# non-zero iff one or more attempted tests fail.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default to current workspace version if none given.
default_version() {
    grep -A5 '\[workspace\.package\]' "${REPO_ROOT}/Cargo.toml" \
        | grep '^version' | head -1 | sed 's/.*= *"\(.*\)"/\1/'
}

VERSION=""
SKIP_IOS=false
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-ios)  SKIP_IOS=true; shift ;;
        --only)      ONLY="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            exit 2
            ;;
        *)
            if [ -z "$VERSION" ]; then VERSION="$1"; shift
            else echo "ERROR: extra positional arg: $1" >&2; exit 2
            fi
            ;;
    esac
done

if [ -z "$VERSION" ]; then
    VERSION="$(default_version)"
fi
TAG="v${VERSION}"
echo "Verifying release: ${VERSION} (tag ${TAG})"
echo

# Per-target enable flags (default: all on).
RUN_SPM=true ; RUN_IOS=true ; RUN_CARGO=true ; RUN_FLUTTER=true ; RUN_MAVEN=true
if [ -n "$ONLY" ]; then
    RUN_SPM=false ; RUN_IOS=false ; RUN_CARGO=false ; RUN_FLUTTER=false ; RUN_MAVEN=false
    IFS=',' read -ra TARGETS <<< "$ONLY"
    for t in "${TARGETS[@]}"; do
        case "$t" in
            spm)     RUN_SPM=true ;;
            ios)     RUN_IOS=true ;;
            cargo)   RUN_CARGO=true ;;
            flutter) RUN_FLUTTER=true ;;
            maven)   RUN_MAVEN=true ;;
            *) echo "ERROR: unknown target in --only: $t" >&2; exit 2 ;;
        esac
    done
fi
[ "$SKIP_IOS" = true ] && RUN_IOS=false

WORK="$(mktemp -d -t xybrid-verify-XXXX)"
trap 'rm -rf "$WORK"' EXIT
echo "Working dir: $WORK"
echo

# Per-target outcome: PASS / FAIL / SKIP (toolchain missing or disabled)
declare -A RESULT
declare -A DETAIL
results_order=(spm ios cargo flutter maven)
for k in "${results_order[@]}"; do RESULT[$k]="SKIP"; DETAIL[$k]="not attempted"; done

run_or_skip() {
    local target=$1 enabled=$2 toolcheck=$3 fn=$4
    if [ "$enabled" != true ]; then
        DETAIL[$target]="disabled"; return 0
    fi
    if ! eval "$toolcheck" >/dev/null 2>&1; then
        DETAIL[$target]="toolchain missing"; return 0
    fi
    echo "──────── $target ────────"
    if "$fn"; then RESULT[$target]="PASS"
    else RESULT[$target]="FAIL"
    fi
    echo
}

# =============================================================================
# SPM — swift package resolve against the published tag
# =============================================================================
test_spm() {
    local dir="$WORK/spm"
    mkdir -p "$dir/Sources/Verify"
    cat > "$dir/Package.swift" <<EOF
// swift-tools-version:5.7
import PackageDescription
let package = Package(
    name: "Verify",
    platforms: [.iOS(.v13), .macOS(.v10_15)],
    products: [.executable(name: "Verify", targets: ["Verify"])],
    dependencies: [
        .package(url: "https://github.com/xybrid-ai/xybrid", from: "${VERSION}")
    ],
    targets: [.executableTarget(name: "Verify", dependencies: [.product(name: "Xybrid", package: "xybrid")])]
)
EOF
    cat > "$dir/Sources/Verify/main.swift" <<'EOF'
import Xybrid
print("ok")
EOF
    (cd "$dir" && swift package resolve 2>&1) || { DETAIL[spm]="swift package resolve failed"; return 1; }
    # Sanity: did we actually pin to the requested version?
    local pinned
    pinned=$(grep -A4 '"identity" : "xybrid"' "$dir/Package.resolved" | grep '"version"' | head -1 | sed 's/.*: *"\(.*\)".*/\1/')
    if [ "$pinned" != "$VERSION" ]; then
        DETAIL[spm]="resolved to '$pinned', expected '$VERSION'"; return 1
    fi
    DETAIL[spm]="resolved $VERSION, xcframework SHA-256 verified"
}

# =============================================================================
# iOS — xcodebuild on examples/ios/XybridExample (iOS Simulator, no codesign)
# =============================================================================
test_ios() {
    local proj="$REPO_ROOT/examples/ios/XybridExample.xcodeproj"
    if [ ! -d "$proj" ]; then
        DETAIL[ios]="examples/ios/XybridExample.xcodeproj not found"; return 1
    fi
    local derived="$WORK/ios-derived"
    # Use a generic iOS Simulator destination so we don't depend on a
    # specific simulator runtime being installed.
    if xcodebuild \
            -project "$proj" \
            -scheme XybridExample \
            -destination 'generic/platform=iOS Simulator' \
            -derivedDataPath "$derived" \
            CODE_SIGNING_ALLOWED=NO \
            build > "$WORK/ios.log" 2>&1; then
        DETAIL[ios]="built examples/ios/XybridExample against Package.swift @ workspace"
    else
        DETAIL[ios]="xcodebuild failed (see $WORK/ios.log)"
        tail -40 "$WORK/ios.log" | sed 's/^/  | /'
        return 1
    fi
}

# =============================================================================
# Cargo — cargo fetch against crates.io
# =============================================================================
test_cargo() {
    local dir="$WORK/cargo"
    mkdir -p "$dir/src"
    cat > "$dir/Cargo.toml" <<EOF
[package]
name = "verify"
version = "0.1.0"
edition = "2021"

[dependencies]
xybrid = "=${VERSION}"
EOF
    cat > "$dir/src/main.rs" <<'EOF'
fn main() {}
EOF
    (cd "$dir" && cargo fetch 2>&1) || { DETAIL[cargo]="cargo fetch failed"; return 1; }
    # Sanity: lockfile pins the right version.
    local pinned
    pinned=$(grep -A 1 '^name = "xybrid"$' "$dir/Cargo.lock" | grep '^version' | head -1 | sed 's/version = "\(.*\)"/\1/')
    if [ "$pinned" != "$VERSION" ]; then
        DETAIL[cargo]="resolved to '$pinned', expected '$VERSION'"; return 1
    fi
    DETAIL[cargo]="resolved $VERSION (+ xybrid-{sdk,core,macros})"
}

# =============================================================================
# Flutter — flutter pub get against pub.dev
# =============================================================================
test_flutter() {
    local dir="$WORK/flutter"
    mkdir -p "$dir"
    flutter create --template=app --org com.xybrid.verify --suppress-analytics "$dir/app" \
        > "$WORK/flutter-create.log" 2>&1 || { DETAIL[flutter]="flutter create failed"; return 1; }
    cat > "$dir/app/pubspec.yaml" <<EOF
name: xybrid_verify_app
description: rc verification
version: 0.0.1
publish_to: 'none'

environment:
  sdk: '>=3.0.0 <4.0.0'
  flutter: '>=3.0.0'

dependencies:
  flutter:
    sdk: flutter
  xybrid_flutter: ${VERSION}

flutter:
  uses-material-design: true
EOF
    (cd "$dir/app" && flutter pub get --suppress-analytics 2>&1) \
        | tee "$WORK/flutter-get.log" | tail -5
    local pinned
    pinned=$(awk '/^  xybrid_flutter:/{p=1; next} p && /^    version:/{print; exit}' "$dir/app/pubspec.lock" \
              | sed 's/.*"\(.*\)".*/\1/')
    if [ "$pinned" != "$VERSION" ]; then
        DETAIL[flutter]="resolved to '$pinned', expected '$VERSION'"; return 1
    fi
    DETAIL[flutter]="resolved $VERSION from pub.dev"
}

# =============================================================================
# Maven Central — gradle dependencies against repo1.maven.org
# =============================================================================
test_maven() {
    # Fast pre-check: is the POM live on Maven Central yet?
    local pom_url="https://repo1.maven.org/maven2/ai/xybrid/xybrid-kotlin/${VERSION}/xybrid-kotlin-${VERSION}.pom"
    if ! curl -sf -o /dev/null "$pom_url"; then
        DETAIL[maven]="ai.xybrid:xybrid-kotlin:${VERSION} not yet on Maven Central (Sonatype mirror sync pending)"
        return 1
    fi
    local dir="$WORK/maven"
    mkdir -p "$dir/src/main/kotlin"
    cat > "$dir/settings.gradle.kts" <<'EOF'
rootProject.name = "verify"
EOF
    cat > "$dir/build.gradle.kts" <<EOF
plugins {
    application
    kotlin("jvm") version "1.9.23"
}
repositories { mavenCentral() }
dependencies {
    implementation("ai.xybrid:xybrid-kotlin:${VERSION}")
}
application { mainClass.set("AppKt") }
EOF
    cat > "$dir/src/main/kotlin/App.kt" <<'EOF'
fun main() {}
EOF
    if (cd "$dir" && gradle dependencies --configuration runtimeClasspath --console=plain) \
            > "$WORK/maven.log" 2>&1; then
        if grep -q "ai.xybrid:xybrid-kotlin:${VERSION}" "$WORK/maven.log"; then
            DETAIL[maven]="resolved ${VERSION} from Maven Central"
        else
            DETAIL[maven]="gradle ran but ${VERSION} not in runtimeClasspath"
            return 1
        fi
    else
        DETAIL[maven]="gradle dependencies failed (see $WORK/maven.log)"
        tail -30 "$WORK/maven.log" | sed 's/^/  | /'
        return 1
    fi
}

run_or_skip spm     "$RUN_SPM"     "command -v swift"      test_spm
run_or_skip ios     "$RUN_IOS"     "command -v xcodebuild" test_ios
run_or_skip cargo   "$RUN_CARGO"   "command -v cargo"      test_cargo
run_or_skip flutter "$RUN_FLUTTER" "command -v flutter"    test_flutter
run_or_skip maven   "$RUN_MAVEN"   "command -v gradle"     test_maven

echo "════════════════════════════════════════════════════════════════════"
echo "  Release verification summary — ${VERSION}"
echo "════════════════════════════════════════════════════════════════════"
printf "  %-8s  %-5s  %s\n" "target" "state" "detail"
printf "  %-8s  %-5s  %s\n" "──────" "─────" "──────"
fail_count=0
for k in "${results_order[@]}"; do
    case "${RESULT[$k]}" in
        PASS) marker="✅" ;;
        FAIL) marker="❌"; fail_count=$((fail_count+1)) ;;
        SKIP) marker="⊝"  ;;
    esac
    printf "  %-8s  %s    %s\n" "$k" "$marker" "${DETAIL[$k]}"
done
echo

if [ "$fail_count" -gt 0 ]; then
    echo "FAILED: ${fail_count} target(s)."
    exit 1
fi
echo "All attempted targets passed."
