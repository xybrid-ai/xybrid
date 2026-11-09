"""Private prebuilt input, verified by the same tool used for Cargo builds."""

# Targets added with schema 5 (pinned from rc.9). Only these may lack a row,
# which defers a fixed missing-pin failure to analysis. The original three
# targets retain strict verification even if their row is accidentally removed.
_UNADOPTED_TARGETS = ["aarch64-apple-ios", "aarch64-apple-ios-sim", "aarch64-unknown-linux-gnu"]

_SYSTEM_LIBRARIES = {
    "aarch64-apple-darwin": ["-lSystem"],
    "aarch64-apple-ios": ["-lSystem"],
    "aarch64-apple-ios-sim": ["-lSystem"],
    "aarch64-linux-android": ["-lc", "-lm", "-ldl"],
    "aarch64-unknown-linux-gnu": ["-lc", "-lm", "-lpthread", "-ldl"],
    "x86_64-unknown-linux-gnu": ["-lc", "-lm", "-lpthread", "-ldl"],
}

def _missing_pin_repository(ctx):
    # Bazel loads packages from unselected select arms. Defer this fixed failure
    # to analysis of the selected target, without staging or private fetches.
    ctx.file("unavailable.bzl", """def _unavailable_impl(ctx):
    fail("zzz-sys: no pinned zzz engine slice for this target; adopt a qualified schema-5 pin before enabling bindings")
unavailable = rule(implementation = _unavailable_impl)
""")
    ctx.file("BUILD.bazel", """load(":unavailable.bzl", "unavailable")
unavailable(name = "zzz", visibility = ["//visibility:public"])
unavailable(name = "notices", visibility = ["//visibility:public"])
""")

def _zzz_prebuilt_impl(ctx):
    # Reading the manifest makes pin changes invalidate repository verification.
    manifest = json.decode(ctx.read(ctx.attr.manifest))
    # A row for the target with a malformed/wrong profile is still an attempted
    # pin: the verifier must reject it rather than treating it as unadopted.
    matches = [entry for entry in manifest["archives"] if entry.get("target") == ctx.attr.target]
    if ctx.attr.target in _UNADOPTED_TARGETS and not matches:
        _missing_pin_repository(ctx)
        return
    staged = ctx.os.environ.get("XYBRID_ZZZ_PREBUILT_DIR", "")
    if not staged:
        fail("Stage the pinned slice with tools/scripts/zzz_pull.py and set --repo_env=XYBRID_ZZZ_PREBUILT_DIR=<verified-slice>")
    # Cross builds may resolve more than one slice while loading select arms.
    # Accept a private root containing verified directories named by target.
    child = ctx.path(staged).get_child(ctx.attr.target)
    if child.exists:
        staged = str(child)
    verifier = ctx.path(ctx.attr.verifier)
    python = ctx.os.environ.get("XYBRID_ZZZ_PYTHON", "python3")
    verified = ctx.execute([python, str(verifier), "--target", ctx.attr.target, "--verify-only"], environment = {"XYBRID_ZZZ_PREBUILT_DIR": staged}, quiet = True)
    if verified.return_code:
        fail("Private zzz slice verification failed: " + verified.stderr)
    root = ctx.path(verified.stdout.strip())
    ctx.symlink(root.get_child("lib/libzzz_embed.a"), "libzzz_embed.a")
    ctx.symlink(root.get_child("include/zzz_embed.h"), "zzz_embed.h")
    # The verifier independently pins the exact schema-5 inventory and checks
    # every staged byte. Export only that verified inventory, never a glob or
    # receipt-derived map. Frozen schema-4 native inputs remain usable as before.
    notice_load = ""
    if len(matches) == 1 and matches[0].get("receipt_schema") == 5:
        notice_paths = ["notices/" + name for name in sorted(matches[0]["notice_files"])]
        for name in sorted(matches[0]["notice_files"]):
            ctx.symlink(root.get_child(name), "notices/" + name)
        notices = "filegroup(name = \"notices\", srcs = %s, visibility = [\"//visibility:public\"])\n" % repr(notice_paths)
    else:
        ctx.file("notices_unavailable.bzl", """def _unavailable_impl(ctx):
    fail("zzz-sys: final-package notices require a qualified schema-5 pin; frozen schema-4 inputs have no notice export")
unavailable = rule(implementation = _unavailable_impl)
""")
        notice_load = "load(\":notices_unavailable.bzl\", \"unavailable\")\n"
        notices = "unavailable(name = \"notices\", visibility = [\"//visibility:public\"])\n"
    libraries = _SYSTEM_LIBRARIES[ctx.attr.target]
    ctx.file("BUILD.bazel", """load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")
%scc_import(name = "archive", static_library = "libzzz_embed.a", hdrs = ["zzz_embed.h"])
cc_library(name = "zzz", deps = [":archive"], linkopts = %s, visibility = ["//visibility:public"])
%s""" % (notice_load, repr(libraries), notices))

zzz_prebuilt_repository = repository_rule(
    implementation = _zzz_prebuilt_impl,
    attrs = {
        "target": attr.string(mandatory = True),
        "verifier": attr.label(default = "//crates/zzz-sys:zzz_pull.py", allow_single_file = True),
        "manifest": attr.label(default = "//crates/zzz-sys:natives-manifest.json", allow_single_file = True),
    },
    environ = ["XYBRID_ZZZ_PREBUILT_DIR", "XYBRID_ZZZ_PYTHON"],
    local = True,
)
