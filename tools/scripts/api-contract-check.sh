#!/usr/bin/env bash
#
# api-contract-check.sh — Validate SDK implementations against api-surface.yaml
#
# Usage:
#   ./tools/scripts/api-contract-check.sh          # Report mode (warnings only)
#   ./tools/scripts/api-contract-check.sh --check   # CI mode (exit 1 on drift — disabled until baseline resolved)
#   ./tools/scripts/api-contract-check.sh --strict   # Strict CI mode (exit 1 on drift)
#
# Requirements: yq (https://github.com/mikefarah/yq)
#
# This script validates that methods marked as "implemented" in api-surface.yaml
# actually exist in the corresponding SDK source code.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Files live in-repo at docs/sdk/
if [ -f "$REPO_ROOT/docs/sdk/api-surface.yaml" ]; then
    API_SURFACE="$REPO_ROOT/docs/sdk/api-surface.yaml"
else
    echo "ERROR: Cannot find docs/sdk/api-surface.yaml (expected at $REPO_ROOT/docs/sdk/)"
    exit 1
fi

API_REFERENCE="$(dirname "$API_SURFACE")/API_REFERENCE.md"

# SDK source paths
DART_SRC="$REPO_ROOT/bindings/flutter/lib"
KOTLIN_SRC="$REPO_ROOT/bindings/kotlin"
SWIFT_SRC="$REPO_ROOT/bindings/apple"
UNITY_SRC="$REPO_ROOT/bindings/unity"

MODE="report"
if [ "${1:-}" = "--check" ]; then
    MODE="check"
elif [ "${1:-}" = "--strict" ]; then
    MODE="strict"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

WARNINGS=0
ERRORS=0

warn() {
    echo -e "${YELLOW}WARNING${NC}: $1"
    WARNINGS=$((WARNINGS + 1))
}

error() {
    echo -e "${RED}ERROR${NC}: $1"
    ERRORS=$((ERRORS + 1))
}

ok() {
    echo -e "${GREEN}OK${NC}: $1"
}

info() {
    echo -e "${BLUE}INFO${NC}: $1"
}

# ─── Prerequisite check ────────────────────────────────────────────────
check_yq() {
    if ! command -v yq &> /dev/null; then
        echo "ERROR: yq is required but not installed."
        echo "Install: brew install yq  (or see https://github.com/mikefarah/yq)"
        exit 1
    fi
}

# ─── Check api-surface.yaml is valid ────────────────────────────────────
check_yaml_valid() {
    info "Checking api-surface.yaml is valid YAML..."
    if yq '.' "$API_SURFACE" > /dev/null 2>&1; then
        ok "api-surface.yaml is valid YAML"
    else
        error "api-surface.yaml is not valid YAML"
        return 1
    fi
}

# ─── Check every class has a matching section in API_REFERENCE.md ────────
check_sections_exist() {
    info "Checking API_REFERENCE.md has sections for all classes..."
    local classes
    classes=$(yq '.classes | keys | .[]' "$API_SURFACE")

    for class in $classes; do
        local section
        section=$(yq ".classes.${class}.section // \"\"" "$API_SURFACE")
        if [ -n "$section" ] && [ "$section" != "null" ]; then
            if grep -q "## ${section}" "$API_REFERENCE" 2>/dev/null; then
                ok "Section found: $section"
            else
                warn "Missing section in API_REFERENCE.md: '$section' (for class $class)"
            fi
        fi
    done
}

# ─── Check a method exists in SDK source ─────────────────────────────────
check_method_in_sdk() {
    local method_name="$1"
    local sdk="$2"
    local class_name="$3"
    local src_dir=""
    local pattern=""

    case "$sdk" in
        dart)
            src_dir="$DART_SRC"
            [ ! -d "$src_dir" ] && return 2
            # Search for method name in Dart files
            pattern="(${method_name}|${method_name}\()"
            ;;
        kotlin)
            src_dir="$KOTLIN_SRC"
            [ ! -d "$src_dir" ] && return 2
            # Match both plain (val foo) and backticked (var `foo`) — UniFFI
            # generates the backticked form for record fields.
            pattern="(fun ${method_name}|val ${method_name}|var ${method_name}|val \`${method_name}\`|var \`${method_name}\`|${method_name}\()"
            ;;
        swift)
            src_dir="$SWIFT_SRC"
            [ ! -d "$src_dir" ] && return 2
            # Match plain declarations and record-field forms produced by
            # UniFFI's Swift generator (`public var foo`, `public let foo`).
            pattern="(func ${method_name}|var ${method_name}|let ${method_name}|public var ${method_name}|public let ${method_name}|${method_name}\()"
            ;;
        csharp)
            src_dir="$UNITY_SRC"
            [ ! -d "$src_dir" ] && return 2
            # C# uses PascalCase — match case-insensitively
            pattern="(${method_name}|${method_name}\()"
            if grep -rqi -E "$pattern" "$src_dir" 2>/dev/null; then
                return 0
            else
                return 1
            fi
            ;;
        *)
            return 2
            ;;
    esac

    if grep -rq -E "$pattern" "$src_dir" 2>/dev/null; then
        return 0  # Found
    else
        return 1  # Not found
    fi
}

# ─── Validate implemented status matches reality ─────────────────────────
validate_implementations() {
    info "Validating 'implemented' status matches SDK source code..."

    local classes
    classes=$(yq '.classes | keys | .[]' "$API_SURFACE")

    for class in $classes; do
        echo ""
        info "Checking class: $class"

        # Check methods
        local methods
        methods=$(yq ".classes.${class}.methods // {} | keys | .[]" "$API_SURFACE" 2>/dev/null || true)

        for method in $methods; do
            for sdk in dart kotlin swift csharp; do
                local status
                status=$(yq ".classes.${class}.methods.${method}.status.${sdk} // \"\"" "$API_SURFACE" 2>/dev/null || true)

                if [ "$status" = "implemented" ]; then
                    if check_method_in_sdk "$method" "$sdk" "$class"; then
                        ok "$class.$method — $sdk: found"
                    else
                        warn "$class.$method — $sdk: marked 'implemented' but not found in source"
                    fi
                fi
            done
        done

        # Check properties
        local properties
        properties=$(yq ".classes.${class}.properties // {} | keys | .[]" "$API_SURFACE" 2>/dev/null || true)

        for prop in $properties; do
            for sdk in dart kotlin swift csharp; do
                local status
                status=$(yq ".classes.${class}.properties.${prop}.status.${sdk} // \"\"" "$API_SURFACE" 2>/dev/null || true)

                if [ "$status" = "implemented" ]; then
                    if check_method_in_sdk "$prop" "$sdk" "$class"; then
                        ok "$class.$prop — $sdk: found"
                    else
                        warn "$class.$prop — $sdk: marked 'implemented' but not found in source"
                    fi
                fi
            done
        done

        # Check factories
        local factories
        factories=$(yq ".classes.${class}.factories // {} | keys | .[]" "$API_SURFACE" 2>/dev/null || true)

        for factory in $factories; do
            for sdk in dart kotlin swift csharp; do
                local status
                status=$(yq ".classes.${class}.factories.${factory}.status.${sdk} // \"\"" "$API_SURFACE" 2>/dev/null || true)

                if [ "$status" = "implemented" ]; then
                    if check_method_in_sdk "$factory" "$sdk" "$class"; then
                        ok "$class.$factory — $sdk: found"
                    else
                        warn "$class.$factory() — $sdk: marked 'implemented' but not found in source"
                    fi
                fi
            done
        done
    done
}

# ─── Check for valid status values ────────────────────────────────────────
check_status_values() {
    info "Checking status values are valid..."
    local valid_statuses="implemented partial stub planned"

    # Extract all status values
    local statuses
    statuses=$(yq '.. | select(has("status")) | .status | to_entries | .[].value' "$API_SURFACE" 2>/dev/null | sort -u || true)

    for status in $statuses; do
        if echo "$valid_statuses" | grep -qw "$status"; then
            ok "Valid status: $status"
        else
            error "Invalid status value: '$status' (must be one of: $valid_statuses)"
        fi
    done
}

# ─── Summary ──────────────────────────────────────────────────────────────
summary() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  API Contract Check Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "  Warnings: ${YELLOW}${WARNINGS}${NC}"
    echo -e "  Errors:   ${RED}${ERRORS}${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ "$ERRORS" -gt 0 ]; then
        echo -e "${RED}FAILED${NC}: $ERRORS error(s) found"
        if [ "$MODE" = "strict" ]; then
            exit 1
        fi
    fi

    if [ "$WARNINGS" -gt 0 ]; then
        echo -e "${YELLOW}DRIFT DETECTED${NC}: $WARNINGS warning(s)"
        if [ "$MODE" = "strict" ]; then
            exit 1
        fi
    fi

    if [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
        echo -e "${GREEN}PASSED${NC}: API contract is in sync"
    fi

    # --check mode: always exit 0 (soft warning) until we flip to strict
    # --strict mode: exit 1 on any drift
    exit 0
}

# ─── Main ────────────────────────────────────────────────────────────────
main() {
    echo "╔═══════════════════════════════════════════════╗"
    echo "║       Xybrid API Contract Validator           ║"
    echo "╚═══════════════════════════════════════════════╝"
    echo ""
    echo "Mode: $MODE"
    echo "Contract: $API_SURFACE"
    echo ""

    check_yq
    check_yaml_valid
    check_status_values
    check_sections_exist
    validate_implementations
    summary
}

main
