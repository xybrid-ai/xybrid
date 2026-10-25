"""Explicit final-package wrapper; the standard Android AAR target is unchanged."""

def _notice_arguments(files):
    # Names come from the repository's notices/ layout. The packager checks the
    # exact fourteen names and hashes against the packaged zzz_pull.py contract,
    # so this rule keeps no second copy of that inventory.
    names = {}
    for file in files:
        parts = file.path.split("/notices/")
        if len(parts) != 2 or not parts[1] or parts[1] in names:
            fail("Android zzz packaging requires exact canonical notice exports")
        names[parts[1]] = file
    return ["%s=%s" % (name, names[name].path) for name in sorted(names)]

def _zzz_aar_impl(ctx):
    if not ctx.attr.enabled:
        fail("xybrid_kotlin_zzz_aar requires --//:tts_zzz_android_companion=true with --//:tts_zzz=false")
    output = ctx.actions.declare_file(ctx.attr.out)
    arguments = ctx.actions.args()
    arguments.add("pack")
    arguments.add("--archive", ctx.file.base)
    arguments.add("--companion", ctx.file.companion)
    arguments.add_all(ctx.attr.base_files, before_each = "--base-file")
    arguments.add_all(_notice_arguments(ctx.files.notices), before_each = "--notice")
    arguments.add("--output", output)
    ctx.actions.run(
        executable = ctx.executable._packager,
        tools = [ctx.attr._packager[DefaultInfo].files_to_run],
        inputs = [ctx.file.base, ctx.file.companion] + ctx.files.notices,
        outputs = [output],
        arguments = [arguments],
        mnemonic = "ZzzAndroidPackage",
        progress_message = "Adding the zzz companion and notices to the Android AAR",
    )
    return [DefaultInfo(files = depset([output]))]

zzz_aar = rule(
    implementation = _zzz_aar_impl,
    attrs = {
        "base": attr.label(mandatory = True, allow_single_file = [".aar"]),
        "base_files": attr.string_list(mandatory = True),
        "companion": attr.label(mandatory = True, allow_single_file = [".so"]),
        "notices": attr.label(mandatory = True, allow_files = True),
        "enabled": attr.bool(mandatory = True),
        "out": attr.string(mandatory = True),
        "_packager": attr.label(default = ":pack_zzz_aar", executable = True, cfg = "exec"),
    },
)
