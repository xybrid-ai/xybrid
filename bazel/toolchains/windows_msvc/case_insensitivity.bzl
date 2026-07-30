"""Makes Microsoft's case-insensitive filenames work on a case-sensitive host.

Windows SDK payloads mix casings freely — `Windows.h` is included as
`<windows.h>`, while `kernelspecs.h` includes `"DriverSpecs.h"` for a file named
`driverspecs.h`, and `uuid.lib` is on disk as `Uuid.Lib`. That is invisible on
Windows and on a default macOS APFS volume, and fatal on Linux:

    ggml/src/ggml.c:51:10: fatal error: 'windows.h' file not found
    lld-link: error: could not open 'uuid.lib': No such file or directory

`@windows_support`'s `transformations` attribute cannot fix this for us, because
repository rules run on the Bazel *client*. Fetch on a Mac and the alias names
collapse onto the originals; that repository is then uploaded verbatim to a
case-sensitive Linux RBE worker, still missing them. The rule below derives its
aliases from the file list instead, so it is correct whichever host fetched the
payload.

Real files, not a clang `-ivfsoverlay`. An overlay is cheaper and needs no
enumeration, but it only works for compiler invocations Bazel spawns itself:
rules_foreign_cc absolutises include paths (`absolutize_path_in_str` prefixes
any `external/` or `bazel-out/` token) while the overlay names its roots
execroot-relative, so the roots stop matching and the overlay silently goes
inert. Nor can the overlay carry absolute paths — the generating action's
sandbox is not the consuming action's. A directory of real files survives every
consumer: Bazel actions, cc-rs build scripts, and CMake.

Upstream tracks a first-class overlay at hermeticbuild/windows_support#3.
"""

load("@bazel_skylib//rules/directory:providers.bzl", "DirectoryInfo")

def _lowercase_aliases_impl(ctx):
    # alias basename -> real path. Only files whose name is not already
    # lowercase produce one, so the copy set stays a fraction of the payload.
    aliases = {}
    sources = []

    def add(alias, source):
        if alias in aliases:
            fail("{} and {} both map to {}".format(aliases[alias], source.path, alias))
        aliases[alias] = source.path
        sources.append(source)

    # `extra_aliases` covers the other direction, which cannot be derived: a
    # file that IS lowercase on disk but gets included under a mixed-case name
    # (`kernelspecs.h` includes "DriverSpecs.h" for `driverspecs.h`). The set is
    # fixed per SDK version, and a missing entry fails the build loudly rather
    # than silently mis-resolving.
    wanted = ctx.attr.extra_aliases
    for target in ctx.attr.directories:
        for source in target[DirectoryInfo].transitive_files.to_list():
            lowercased = source.basename.lower()
            if lowercased != source.basename:
                add(lowercased, source)
            if source.basename in wanted:
                add(wanted[source.basename], source)

    for real, alias in ctx.attr.extra_aliases.items():
        if alias not in aliases:
            fail("extra_aliases: no file named {} to alias as {}".format(real, alias))

    manifest = ctx.actions.declare_file(ctx.label.name + ".manifest")
    ctx.actions.write(
        manifest,
        "".join([
            "{}\t{}\n".format(aliases[alias], alias)
            for alias in sorted(aliases)
        ]),
    )

    directory = ctx.actions.declare_directory(ctx.label.name)
    ctx.actions.run_shell(
        arguments = [manifest.path, directory.path],
        # Copies rather than symlinks: a symlink out of a tree artifact does not
        # survive remote execution. The copies share content digests with the
        # originals, so the CAS stores them once.
        command = """
set -euo pipefail
mkdir -p "$2"
while IFS=$'\\t' read -r source alias; do
    cp -f "${source}" "$2/${alias}"
done < "$1"
""",
        inputs = depset(direct = [manifest] + sources),
        outputs = [directory],
        mnemonic = "WindowsSdkCaseAliases",
        progress_message = "Aliasing %{label}",
    )
    return [DefaultInfo(files = depset([directory]))]

lowercase_aliases = rule(
    implementation = _lowercase_aliases_impl,
    doc = "Collects case-variant copies of Microsoft's mixed-case files into one directory.",
    attrs = {
        "directories": attr.label_list(
            doc = "Directories to scan.",
            mandatory = True,
            providers = [DirectoryInfo],
        ),
        "extra_aliases": attr.string_dict(
            doc = "Real basename -> additional alias basename, for files that are lowercase on disk.",
        ),
    },
)
