"""Expose the workspace version from Cargo.toml as a Bazel constant.

Cargo sets `CARGO_PKG_VERSION` — what `env!("CARGO_PKG_VERSION")` reads, e.g.
`xybrid_sdk::SDK_VERSION` — from Cargo.toml. rules_rust sets it from each Rust
target's `version` attribute instead, which defaults to "0.0.0". Every artifact
we ship is Bazel-built, so without this they all reported SDK version 0.0.0: in
`version()`, in the registry's `X-Xybrid-Client` header and on telemetry.

This repository rule reads `[workspace.package] version` from the root
Cargo.toml and writes it to `@xybrid_version//:version.bzl` as
`XYBRID_VERSION`, which every first-party Rust target passes as `version`.
Cargo.toml stays the single place a release bumps (`just bump-version`): the
file is watched, so a bump regenerates the constant.
"""

def _workspace_version(cargo_toml):
    """Returns `version` from Cargo.toml's `[workspace.package]` table, or None."""
    in_table = False
    for line in cargo_toml.splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("["):
            in_table = line == "[workspace.package]"
        elif in_table:
            key, _, value = line.partition("=")
            if key.strip() == "version":
                return value.strip().strip("\"")
    return None

def _cargo_version_impl(rctx):
    version = _workspace_version(rctx.read(rctx.attr.cargo_toml, watch = "yes"))
    if not version:
        fail("no [workspace.package] version in %s" % rctx.attr.cargo_toml)
    rctx.file("BUILD.bazel", "")
    rctx.file("version.bzl", "XYBRID_VERSION = \"%s\"\n" % version)

cargo_version_repository = repository_rule(
    implementation = _cargo_version_impl,
    attrs = {
        "cargo_toml": attr.label(
            doc = "The workspace Cargo.toml.",
            mandatory = True,
        ),
    },
    doc = "Writes Cargo.toml's [workspace.package] version to version.bzl as XYBRID_VERSION.",
)
