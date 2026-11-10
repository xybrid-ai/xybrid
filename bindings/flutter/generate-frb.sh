#!/usr/bin/env bash
# Generate with the crate's pinned FRB and normalize its Rust formatter gaps.
set -euo pipefail
cd "$(dirname "$0")"
expected=$(sed -n 's/^flutter_rust_bridge = "=\([^"]*\)"/\1/p' rust/Cargo.toml)
actual=$(flutter_rust_bridge_codegen --version)
if [[ "$actual" != "flutter_rust_bridge_codegen $expected" ]]; then
  echo "Expected flutter_rust_bridge_codegen $expected; got $actual" >&2
  exit 1
fi
flutter_rust_bridge_codegen generate
# FRB 2.11.1 emits trailing spaces in opaque methods whose closure rustfmt
# leaves verbatim. Normalize them after generation, never by hand.
python3 - <<'PY'
from pathlib import Path
path = Path('rust/src/frb_generated.rs')
path.write_text(''.join(line.rstrip() + '\n' for line in path.read_text().splitlines()))
PY
rustfmt --edition 2021 rust/src/frb_generated.rs
