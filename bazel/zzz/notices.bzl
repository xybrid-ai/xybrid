"""The zzz engine's fourteen notices, packaged into a final SDK artifact.

`ZZZ_NOTICE_PATHS` is the path-only projection of `SCHEMA5_NOTICE_FILES` in
crates/zzz-sys/zzz_pull.py. For a schema-5 pin the repository rule
(repositories.bzl) exports exactly these files, each verified byte for byte,
as `@zzz_<target>//:notices` under `notices/<path>`. bindings/apple/zzz_notices.bzl
keeps its own private copy of the list so the Apple rule stays untouched;
tools/scripts/tests/test_zzz_aar_notices.py asserts that all three agree.
"""

load("@rules_pkg//pkg:providers.bzl", "PackageFilesInfo")

ZZZ_NOTICE_PATHS = [
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

def zzz_notice_map(files, prefix):
    """Maps each canonical notice to `<prefix>/<path>`.

    Fails analysis unless `files` is exactly the fourteen canonical notices,
    each once, so a missing, extra or ambiguous export never reaches a package.

    Args:
      files: the File objects of a `@zzz_<target>//:notices` export.
      prefix: the package directory that receives them.

    Returns:
      A dict from package path to File.
    """
    root = prefix.strip("/")
    result = {}
    for file in files:
        matches = [name for name in ZZZ_NOTICE_PATHS if file.path.endswith("/notices/" + name)]
        if file.is_directory or len(matches) != 1:
            fail("zzz notices: %s is not one of the fourteen canonical notices" % file.path)
        destination = root + "/" + matches[0]
        if destination in result:
            fail("zzz notices: %s is exported more than once" % matches[0])
        result[destination] = file
    missing = [name for name in ZZZ_NOTICE_PATHS if root + "/" + name not in result]
    if missing:
        fail("zzz notices: the export lacks " + ", ".join(missing))
    return result

def _zzz_notice_files_impl(ctx):
    return [
        PackageFilesInfo(
            dest_src_map = zzz_notice_map(ctx.files.notices, ctx.attr.prefix),
            attributes = {"mode": "0644"},
        ),
        DefaultInfo(files = depset(ctx.files.notices)),
    ]

zzz_notice_files = rule(
    implementation = _zzz_notice_files_impl,
    doc = "Places a verified zzz notice export under `prefix` in a rules_pkg package.",
    attrs = {
        "notices": attr.label(mandatory = True, allow_files = True),
        "prefix": attr.string(mandatory = True),
    },
    provides = [PackageFilesInfo],
)
