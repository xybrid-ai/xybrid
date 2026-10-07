"""Private prebuilt input, verified by the same tool used for Cargo builds."""

def _zzz_prebuilt_impl(ctx):
    staged = ctx.os.environ.get("XYBRID_ZZZ_PREBUILT_DIR", "")
    if not staged:
        fail("Stage the pinned slice with tools/scripts/zzz_pull.py and set --repo_env=XYBRID_ZZZ_PREBUILT_DIR=<verified-slice>")
    # Cross builds may resolve more than one slice while loading select arms.
    # Accept a private root containing verified directories named by target.
    child = ctx.path(staged).get_child(ctx.attr.target)
    if child.exists:
        staged = str(child)
    verifier = ctx.path(ctx.attr.verifier)
    # Reading the manifest makes pin changes invalidate repository verification.
    ctx.read(ctx.attr.manifest)
    python = ctx.os.environ.get("XYBRID_ZZZ_PYTHON", "python3")
    verified = ctx.execute([python, str(verifier), "--target", ctx.attr.target, "--verify-only"], environment = {"XYBRID_ZZZ_PREBUILT_DIR": staged}, quiet = True)
    if verified.return_code:
        fail("Private zzz slice verification failed: " + verified.stderr)
    root = ctx.path(verified.stdout.strip())
    ctx.symlink(root.get_child("lib/libzzz_embed.a"), "libzzz_embed.a")
    ctx.symlink(root.get_child("include/zzz_embed.h"), "zzz_embed.h")
    libraries = {"aarch64-apple-darwin": ["-lSystem"], "aarch64-linux-android": ["-lc", "-lm", "-ldl"], "x86_64-unknown-linux-gnu": ["-lc", "-lm", "-lpthread", "-ldl"]}[ctx.attr.target]
    ctx.file("BUILD.bazel", """load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")
cc_import(name = "archive", static_library = "libzzz_embed.a", hdrs = ["zzz_embed.h"])
cc_library(name = "zzz", deps = [":archive"], linkopts = %s, visibility = ["//visibility:public"])
""" % repr(libraries))

zzz_prebuilt_repository = repository_rule(
    implementation = _zzz_prebuilt_impl,
    attrs = {
        "target": attr.string(mandatory = True),
        "verifier": attr.label(default = "//tools/scripts:zzz_pull.py", allow_single_file = True),
        "manifest": attr.label(default = "//crates/zzz-sys:natives-manifest.json", allow_single_file = True),
    },
    environ = ["XYBRID_ZZZ_PREBUILT_DIR", "XYBRID_ZZZ_PYTHON"],
    local = True,
)
