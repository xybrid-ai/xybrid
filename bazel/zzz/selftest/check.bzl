"""Build-time assertions of the engine isolation self-test (see BUILD.bazel)."""

load("@rules_cc//cc:find_cc_toolchain.bzl", "CC_TOOLCHAIN_TYPE", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//bazel/zzz:isolate.bzl", "zzz_engine_library")

def _names(ctx, kind, names):
    file = ctx.actions.declare_file("%s/%s.txt" % (ctx.label.name, kind))
    ctx.actions.write(file, "".join([name + "\n" for name in names]))
    return file

def _isolation_selftest_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    raw = zzz_engine_library(ctx.attr.raw[CcInfo].linking_context.linker_inputs.to_list()).archive
    isolated = zzz_engine_library(ctx.attr.engine[CcInfo].linking_context.linker_inputs.to_list()).archive
    lists = [
        _names(ctx, "api", ctx.attr.api),
        _names(ctx, "local", ctx.attr.local),
        _names(ctx, "undefined", ctx.attr.undefined),
        _names(ctx, "collected", ctx.attr.collected),
        _names(ctx, "imports", ctx.attr.imports),
    ]
    report = ctx.actions.declare_file(ctx.label.name + ".txt")
    arguments = ctx.actions.args()
    arguments.add_all([cc_toolchain.nm_executable, cc_toolchain.objdump_executable])
    arguments.add_all([raw, isolated, ctx.file.library] + lists + [report])
    ctx.actions.run(
        executable = ctx.executable._check,
        arguments = [arguments],
        inputs = depset([raw, isolated, ctx.file.library] + lists, transitive = [cc_toolchain.all_files]),
        outputs = [report],
        use_default_shell_env = True,
        mnemonic = "ZzzIsolationSelftest",
        progress_message = "Checking the engine isolation on a fake engine %{label}",
        toolchain = CC_TOOLCHAIN_TYPE,
    )
    return [DefaultInfo(files = depset([report]))]

isolation_selftest = rule(
    implementation = _isolation_selftest_impl,
    doc = "Fails the build unless `engine` isolates `raw` and `library` links it without interposition.",
    attrs = {
        "api": attr.string_list(mandatory = True, doc = "The only symbols `engine` may define globally."),
        "collected": attr.string_list(
            mandatory = True,
            doc = "Imports of an unreferenced `raw` member only: `library` must not reference them.",
        ),
        "engine": attr.label(mandatory = True, providers = [CcInfo], doc = "The isolated fake engine."),
        "imports": attr.string_list(mandatory = True, doc = "Undefined dynamic symbols `library` must have."),
        "library": attr.label(mandatory = True, allow_single_file = [".so"], doc = "Linked from the host and `engine`."),
        "local": attr.string_list(mandatory = True, doc = "Global in `raw`, local in `engine`, never exported by `library`."),
        "raw": attr.label(mandatory = True, providers = [CcInfo], doc = "The fake engine before isolation."),
        "undefined": attr.string_list(mandatory = True, doc = "Imports of the fake engine that must stay undefined in `engine`."),
        "_check": attr.label(
            default = Label(":check.sh"),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = use_cc_toolchain(),
)
