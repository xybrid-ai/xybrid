"""Declares the hermetic-LLVM-to-Windows-MSVC C++ toolchains.

Mirrors `@llvm//toolchain:declare_toolchains.bzl`: one `cc_toolchain` per exec
platform (Bazel cannot bind one `cc_toolchain` to several `toolchain()`s with
different `exec_compatible_with`), each registered for Windows targets carrying
`@llvm//constraints/windows/abi:msvc`.

`@llvm`'s own Windows toolchain is declared with no ABI constraint, so it also
matches an MSVC platform. These toolchains must therefore be registered BEFORE
`@llvm//toolchain:all` in MODULE.bazel — first match wins.
"""

load("@llvm//platforms:common.bzl", "SUPPORTED_EXECS")
load("@llvm//toolchain:selects.bzl", "resource_dir_arg")
load("@rules_cc//cc/toolchains:feature_set.bzl", "cc_feature_set")
load("@rules_cc//cc/toolchains:toolchain.bzl", "cc_toolchain")

# x86_64 only: MODULE.bazel configures both sysroot repositories with
# `architectures = ["x64"]`, so no arm64 import libraries are fetched. Adding
# aarch64 here means adding it there too — the `:msvc_lib` / `:winsdk_*_lib`
# aliases already select the right subdirectory.
_TARGET_CPUS = ["x86_64"]

def declare_toolchains():
    """Declares one cc_toolchain per exec platform plus its toolchain() bindings."""
    for (exec_os, exec_cpu) in SUPPORTED_EXECS:
        name = "{}_{}_cc_toolchain".format(exec_os, exec_cpu)

        cc_feature_set(
            name = name + "_known_features",
            all_of = [
                ":msvc_def_file",
                "@llvm//toolchain/features:external_include_paths",
                "@llvm//toolchain/features:generate_pdb_file",
                "@llvm//toolchain/features:targets_windows",
                # Enabled internally by the --compilation_mode flag family:
                # known, but not enabled here.
                "@llvm//toolchain/features:all_non_legacy_builtin_features",
                "@llvm//toolchain/features/legacy:all_legacy_builtin_features",
                # Always last: carries user_compile_flags / user_link_flags.
                "//bazel/toolchains/windows_msvc/link:features",
            ],
        )

        cc_feature_set(
            name = name + "_enabled_features",
            all_of = [
                # Bazel hands the linker a generated .def file for DLLs.
                ":msvc_def_file",
                "@llvm//toolchain/features:targets_windows",
                "@llvm//toolchain/features:opt",
                "@llvm//toolchain/features:dbg",
                "@llvm//toolchain/features:archive_param_file",
                "@llvm//toolchain/features/legacy:all_legacy_builtin_features",
                "//bazel/toolchains/windows_msvc/link:features",
            ],
        )

        cc_toolchain(
            name = name,
            # ORDER-SENSITIVE: the resource dir emits `-Xclang
            # -internal-isystem` for clang's builtin headers and must precede
            # the Microsoft include paths, which use the same mechanism — that
            # list is searched in command-line order. See the
            # `headers_include_search_paths` comment in BUILD.bazel.
            args = [
                resource_dir_arg(exec_os, exec_cpu),
                ":toolchain_args",
            ],
            artifact_name_patterns = [
                ":executable_pattern",
                ":dynamic_library_pattern",
                ":interface_library_pattern",
                ":static_library_pattern",
            ],
            compiler = "clang",
            enabled_features = [name + "_enabled_features"],
            known_features = [name + "_known_features"],
            supports_param_files = True,
            target_system_name = "x86_64-pc-windows-msvc",
            tool_map = ":tools",
        )

        for target_cpu in _TARGET_CPUS:
            native.toolchain(
                name = "{}_{}_to_windows_msvc_{}".format(exec_os, exec_cpu, target_cpu),
                exec_compatible_with = [
                    "@platforms//cpu:{}".format(exec_cpu),
                    "@platforms//os:{}".format(exec_os),
                ],
                target_compatible_with = [
                    "@platforms//cpu:{}".format(target_cpu),
                    "@platforms//os:windows",
                    "@llvm//constraints/windows/abi:msvc",
                ],
                toolchain = name,
                toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
                visibility = ["//visibility:public"],
            )
