"""Retain complete production native licenses in the public XCFramework ZIP."""

def _native_notices_xcframework_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".xcframework.zip")
    arguments = ctx.actions.args()
    arguments.add("--archive", ctx.file.archive)
    arguments.add("--notice", ctx.file.notice)
    arguments.add("--output", output)
    ctx.actions.run(
        executable = ctx.executable._packager,
        tools = [ctx.attr._packager[DefaultInfo].files_to_run],
        inputs = [ctx.file.archive, ctx.file.notice],
        outputs = [output],
        arguments = [arguments],
        mnemonic = "NativeAppleNotices",
        progress_message = "Retaining native licenses in Apple final package",
    )
    return [DefaultInfo(files = depset([output]))]

native_notices_xcframework = rule(
    implementation = _native_notices_xcframework_impl,
    attrs = {
        "archive": attr.label(mandatory = True, allow_single_file = [".zip"]),
        "notice": attr.label(mandatory = True, allow_single_file = True),
        "_packager": attr.label(default = ":pack_native_notices", executable = True, cfg = "exec"),
    },
)
