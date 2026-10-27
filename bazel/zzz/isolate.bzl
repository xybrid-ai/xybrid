"""Isolates the static zzz speech engine from the code it is linked beside.

The engine archive carries its own runtime (string and math routines,
compiler-rt helpers) and defines those names GLOBAL or WEAK: memcpy, memset,
strlen, expf, logf, __stack_chk_fail, __emutls_get_address and more. The
Android arm64 libxybrid_bolt.so takes the whole archive (through the alwayslink
Rust staticlib), so linked as is, xybrid's OWN Rust and C code in that library
resolved those names to the engine's copies instead of bionic's.

`zzz_isolated_engine` takes the engine's one static archive from a CcInfo dep
and, with the target C++ toolchain's own tools, runs isolate_engine.sh:

  1. `ld -r --whole-archive`: prelink every member into one relocatable
     object. The engine's references to its own runtime resolve inside it;
     its TLS, .init_array and .eh_frame sections pass through unchanged.
  2. `objcopy --keep-global-symbols`: keep only the C API global (the
     functions crates/zzz-sys/src/api.rs declares, ZZZ_EMBED_API) and make
     every other definition local. Undefined symbols (the engine's libc, libm
     and libdl imports) stay undefined and global. A COMDAT group whose
     signature becomes local stops being a COMDAT group, so the final link
     cannot fold it into a same-named one outside the engine either.
  3. Fail the action unless the object defines exactly the API as its
     global/weak symbols, defines every name it did before, and still imports
     (undefined and global) exactly what it did before; and unless step 1
     left nothing undefined that the archive does not import itself (a
     definition of the engine's left undefined would bind outside it).
     Refuse common symbols, which a relocatable link cannot allocate.
  4. `ar` the object into a static archive.

The result is returned as a CcInfo with the dep's link flags and headers, so it
drops in where the raw archive was. Only counts and the public API names are
ever printed: tool diagnostics can name private engine symbols, so the script
withholds them.
"""

load("@rules_cc//cc:find_cc_toolchain.bzl", "CC_TOOLCHAIN_TYPE", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

# The engine's C API, exactly the extern "C" functions of `static_functions`
# in crates/zzz-sys/src/api.rs (tools/scripts/tests/test_zzz_isolation.py
# keeps the two equal). Nothing else in the engine may stay global.
ZZZ_EMBED_API = [
    "zzz_embed_abi_version",
    "zzz_embed_supports_model",
    "zzz_embed_default_options",
    "zzz_embed_open",
    "zzz_embed_open_kitten",
    "zzz_embed_synthesize",
    "zzz_embed_cancel",
    "zzz_embed_close",
]

def zzz_engine_library(linker_inputs):
    """The engine's single static archive and what its link carries.

    Fails analysis unless the linker inputs hold exactly one library, and that
    library is a static archive only: two archives, or a shared one, are not
    an engine this rule knows how to isolate.

    Args:
      linker_inputs: the dep's `CcInfo.linking_context.linker_inputs`, as a list.

    Returns:
      A struct with `archive` (File; the PIC archive when there is one), `pic`,
      `alwayslink`, `user_link_flags` and `additional_inputs`, the last two in
      link order.
    """
    libraries = []
    user_link_flags = []
    additional_inputs = []
    for linker_input in linker_inputs:
        libraries.extend(linker_input.libraries)
        user_link_flags.extend(linker_input.user_link_flags)
        additional_inputs.extend(linker_input.additional_inputs)
    if len(libraries) != 1:
        fail("zzz_isolated_engine: the engine must provide exactly one library, found %d" % len(libraries))
    library = libraries[0]
    if library.dynamic_library or library.interface_library:
        fail("zzz_isolated_engine: the engine must be a static archive, not a shared library")
    archive = library.pic_static_library or library.static_library
    if not archive:
        fail("zzz_isolated_engine: the engine provides no static archive")
    return struct(
        archive = archive,
        pic = library.pic_static_library != None,
        alwayslink = library.alwayslink,
        user_link_flags = user_link_flags,
        additional_inputs = additional_inputs,
    )

def _zzz_isolated_engine_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    engine_info = ctx.attr.engine[CcInfo]
    engine = zzz_engine_library(engine_info.linking_context.linker_inputs.to_list())
    if len(ctx.attr.api) != len({name: None for name in ctx.attr.api}):
        fail("zzz_isolated_engine: api lists a symbol twice")

    # No analysis-time check that the toolchain declares these tools:
    # rules_rust also configures every dep of a rust_library in the exec
    # configuration (proc_macro_deps), where this target must analyze with the
    # host toolchain but is never built. The script rejects a missing tool.

    api = ctx.actions.declare_file(ctx.label.name + "/api.txt")
    ctx.actions.write(api, "".join([name + "\n" for name in ctx.attr.api]))
    # Named apart from the private input (libzzz_embed.a), so an action graph
    # shows which of the two each link consumes.
    output = ctx.actions.declare_file(ctx.label.name + "/libzzz_embed_isolated.a")

    arguments = ctx.actions.args()
    arguments.add_all([
        cc_toolchain.ld_executable,
        cc_toolchain.objcopy_executable,
        cc_toolchain.nm_executable,
        cc_toolchain.ar_executable,
    ])
    arguments.add_all([engine.archive, api, output])
    ctx.actions.run(
        executable = ctx.executable._isolate,
        arguments = [arguments],
        # all_files: the tools must reach a remote executor as action inputs.
        inputs = depset([engine.archive, api], transitive = [cc_toolchain.all_files]),
        outputs = [output],
        use_default_shell_env = True,
        mnemonic = "ZzzIsolateEngine",
        progress_message = "Isolating the static speech engine %{label}",
        toolchain = CC_TOOLCHAIN_TYPE,
    )

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    library = cc_common.create_library_to_link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        static_library = output,
        pic_static_library = output if engine.pic else None,
        alwayslink = engine.alwayslink,
    )
    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset([library]),
        user_link_flags = engine.user_link_flags,
        additional_inputs = depset(engine.additional_inputs),
    )
    return [
        DefaultInfo(files = depset([output])),
        CcInfo(
            compilation_context = engine_info.compilation_context,
            linking_context = cc_common.create_linking_context(linker_inputs = depset([linker_input])),
        ),
    ]

zzz_isolated_engine = rule(
    implementation = _zzz_isolated_engine_impl,
    doc = "A static engine archive with every symbol but its C API made local (see the module docstring).",
    attrs = {
        "api": attr.string_list(
            doc = "The engine's C API: the only symbols that stay global. Pass ZZZ_EMBED_API.",
            mandatory = True,
            allow_empty = False,
        ),
        "engine": attr.label(
            doc = "A CcInfo target carrying the engine's single static archive (and its link flags).",
            mandatory = True,
            providers = [CcInfo],
        ),
        "_isolate": attr.label(
            default = Label("//bazel/zzz:isolate_engine.sh"),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
    provides = [CcInfo],
)
