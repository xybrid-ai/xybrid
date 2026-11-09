"""Explicit final-package wrapper; standard Apple release target is unchanged."""

# Path-only projection of the immutable contract in packaged zzz_pull.py. The
# action independently checks its hashes; analysis rejects accidental extra,
# missing or ambiguous repository exports before running the tool.
_NOTICE_PATHS = [
    "LICENSES/Go-BSD-3-Clause.txt",
    "LICENSES/KittenTTS-APACHE-2.0.txt",
    "LICENSES/LLVM-compiler-rt.txt",
    "LICENSES/PrismML-llama-MIT.txt",
    "LICENSES/Zig-MIT.txt",
    "LICENSES/compiler-rt-release_80.txt",
    "LICENSES/musl-COPYRIGHT.txt",
    "LICENSES/musl-math-NOTICES.txt",
    "LICENSES/xn-APACHE-2.0.txt",
    "LICENSES/xn-MIT.txt",
    "LICENSES/xn-ptts-APACHE-2.0.txt",
    "LICENSES/xn-ptts-MIT.txt",
    "RUNTIME_TERMS.md",
    "THIRD_PARTY_NOTICES.md",
]

def _notice_inputs(files):
    result = {}
    for file in files:
        matches = [name for name in _NOTICE_PATHS if file.path.endswith("/notices/" + name)]
        if len(matches) != 1 or matches[0] in result:
            fail("Apple zzz packaging requires exact canonical notice exports")
        result[matches[0]] = file
    if sorted(result) != _NOTICE_PATHS:
        fail("Apple zzz packaging requires all fourteen notices from each slice")
    return result

def _zzz_xcframework_impl(ctx):
    if not ctx.attr.enabled:
        fail("XybridFFI_zzz requires explicit --//:tts_zzz=true")
    device = _notice_inputs(ctx.files.device_notices)
    simulator = _notice_inputs(ctx.files.simulator_notices)
    output = ctx.actions.declare_file(ctx.label.name + ".xcframework.zip")
    arguments = ctx.actions.args()
    arguments.add("pack")
    arguments.add("--archive", ctx.file.archive)
    for name in _NOTICE_PATHS:
        arguments.add("--device-notice", name + "=" + device[name].path)
        arguments.add("--simulator-notice", name + "=" + simulator[name].path)
    arguments.add("--output", output)
    ctx.actions.run(
        executable = ctx.executable._packager,
        tools = [ctx.attr._packager[DefaultInfo].files_to_run],
        inputs = [ctx.file.archive] + ctx.files.device_notices + ctx.files.simulator_notices,
        outputs = [output],
        arguments = [arguments],
        mnemonic = "ZzzAppleNotices",
        progress_message = "Verifying both zzz notice inventories in Apple final package",
    )
    return [DefaultInfo(files = depset([output]))]

zzz_xcframework = rule(
    implementation = _zzz_xcframework_impl,
    attrs = {
        "archive": attr.label(mandatory = True, allow_single_file = [".zip"]),
        "device_notices": attr.label(mandatory = True, allow_files = True),
        "simulator_notices": attr.label(mandatory = True, allow_files = True),
        "enabled": attr.bool(mandatory = True),
        "_packager": attr.label(default = ":pack_zzz_notices", executable = True, cfg = "exec"),
    },
)
