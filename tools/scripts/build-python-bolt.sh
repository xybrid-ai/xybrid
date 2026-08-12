#!/usr/bin/env bash
# build-python-bolt.sh
#
# Build the xybrid BoltFFI cdylib for the host and refresh the Python package's
# bundled native library copy.
#
# Why this exists:
#
# - Since boltffi 0.29 the wire layer is generated, not hand-ported: the
#   package under `bindings/python/xybrid/_bolt/` comes from
#   `tools/scripts/gen_python_bolt.py`, and it imports a compiled CPython
#   bridge (`_native`) that dlopens the cdylib sitting beside it.
# - So this script runs `boltffi pack python`, which builds the cdylib, compiles
#   that bridge against it, and emits a wheel; both artifacts are then staged
#   next to the generated sources.
#
# Usage: ./tools/scripts/build-python-bolt.sh
# Optional env overrides:
#   XYBRID_FEATURES  Cargo features to enable (defaults per host OS below;
#                    add llm-llamacpp for GGUF/LLM support)
#   DEBUG=1          Build target/debug instead of target/release
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

case "$(uname -s)" in
    Darwin)
        DEFAULT_FEATURES="platform-macos"
        LIB_NAME="libxybrid_bolt.dylib"
        ;;
    Linux)
        DEFAULT_FEATURES="platform-desktop"
        LIB_NAME="libxybrid_bolt.so"
        ;;
    *)
        echo "error: unsupported host OS '$(uname -s)'; expected Darwin or Linux" >&2
        exit 1
        ;;
esac

FEATURES="${XYBRID_FEATURES:-$DEFAULT_FEATURES}"
PROFILE="release"
PROFILE_FLAG="--release"
if [ "${DEBUG:-0}" = "1" ]; then
    PROFILE="debug"
    PROFILE_FLAG=""
fi

echo "==> Building Python bolt native library"
echo "    Features: $FEATURES"
echo "    Profile:  $PROFILE"

PY_BOLT_DIR="$REPO_ROOT/bindings/python/xybrid/_bolt"
mkdir -p "$PY_BOLT_DIR"

# `pack` owns the cargo build: it compiles the cdylib, then the CPython bridge
# against it, then wheels both. Features must reach *that* build — a separate
# `cargo build --features ...` would only warm the cache, since the artifact
# staged below comes from the wheel.
# shellcheck disable=SC2086  # deliberate word-split: empty PROFILE_FLAG = debug
(cd "$REPO_ROOT/crates/xybrid-bolt" && boltffi pack python $PROFILE_FLAG \
    --cargo-arg=--features --cargo-arg="$FEATURES")

# `pack` leaves the compiled bridge only inside the wheel, so take it from
# there rather than from dist/, whose package dir holds just the sources.
WHEEL="$(ls -t "$REPO_ROOT"/crates/xybrid-bolt/dist/python/wheelhouse/*.whl 2>/dev/null | head -1)"
if [ -z "$WHEEL" ]; then
    echo "error: boltffi pack python produced no wheel" >&2
    exit 1
fi
# Stage BOTH artifacts out of the wheel. The cdylib must be boltffi's relinked
# one: `pack` rebuilds target/ with a plain cargo build that omits the generated
# export shims (boltffi_release_class_*, boltffi_init_class_*), so copying from
# target/ yields a library the bridge cannot resolve against.
BRIDGE_NAME="$(python3 -c "
import sys, zipfile
names = [n for n in zipfile.ZipFile(sys.argv[1]).namelist()
         if '/_native' in n and n.rsplit('.', 1)[-1] in ('so', 'pyd')]
print(names[0] if names else '')
" "$WHEEL")"
if [ -z "$BRIDGE_NAME" ]; then
    echo "error: no _native extension inside $WHEEL" >&2
    exit 1
fi
python3 -c "
import sys, zipfile, pathlib, shutil
wheel, name, dest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
with zipfile.ZipFile(wheel) as z, z.open(name) as src:
    target = dest / pathlib.PurePosixPath(name).name
    with target.open('wb') as out:
        shutil.copyfileobj(src, out)
    target.chmod(0o755)
" "$WHEEL" "$BRIDGE_NAME" "$PY_BOLT_DIR"
python3 -c "
import sys, zipfile, pathlib, shutil
wheel, name, dest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
with zipfile.ZipFile(wheel) as z, z.open(name) as src:
    target = dest / pathlib.PurePosixPath(name).name
    with target.open('wb') as out:
        shutil.copyfileobj(src, out)
" "$WHEEL" "xybrid_bolt/$LIB_NAME" "$PY_BOLT_DIR"

echo "==> Done. Staged $LIB_NAME + $(basename "$BRIDGE_NAME") in $PY_BOLT_DIR"
