"""Makes Microsoft's case-insensitive filenames work on a case-sensitive host.

Windows SDK payloads mix casings freely — `Windows.h` is included as
`<windows.h>`, while `kernelspecs.h` includes `"DriverSpecs.h"` for a file named
`driverspecs.h`, and `uuid.lib` is on disk as `Uuid.Lib`. That is invisible on
Windows and on a default macOS APFS volume, and fatal on Linux:

    bazel/windows_msvc/smoke.cc:8:10: fatal error: 'windows.h' file not found
    lld-link: error: could not open 'uuid.lib': No such file or directory

`@windows_support`'s `transformations` attribute cannot fix this for us, because
repository rules run on the Bazel *client*. Fetch on a Mac and the alias names
collapse onto the originals; that repository is then uploaded verbatim to a
case-sensitive Linux RBE worker, still missing them. Both rules below derive
their output from the file list instead, so they are correct whichever host
fetched the payload.

Headers and libraries need different mechanisms because `-ivfsoverlay` is a
clang flag and lld-link never sees it.

Upstream plans to expose an overlay directly — hermeticbuild/windows_support#3.
"""

load("@bazel_skylib//rules/directory:providers.bzl", "DirectoryInfo")

def _vfs_overlay_impl(ctx):
    # Containing directory -> [(basename, path)]. Each key becomes one overlay
    # root. EVERY file is listed, not just the mixed-case ones: the miscasing
    # goes both ways, so there is no single canonical casing to normalise
    # towards. `case-sensitive: false` makes every listed entry match any
    # casing, which covers both directions at once.
    entries = {}
    for target in ctx.attr.directories:
        for source in target[DirectoryInfo].transitive_files.to_list():
            slash = source.path.rfind("/")
            entries.setdefault(source.path[:slash], []).append(
                (source.path[slash + 1:], source.path),
            )

    overlay = ctx.actions.declare_file(ctx.label.name + ".yaml")
    ctx.actions.write(
        overlay,
        # JSON is valid YAML, and json.encode handles the escaping. Paths stay
        # execution-root-relative: clang resolves them against the compiler's
        # working directory, which Bazel sets to the execution root, and
        # absolute paths would bake in the generating action's sandbox.
        json.encode({
            "version": 0,
            "case-sensitive": False,
            "roots": [
                {
                    "name": directory,
                    "type": "directory",
                    "contents": [
                        {
                            "name": basename,
                            "type": "file",
                            "external-contents": path,
                        }
                        for basename, path in sorted(entries[directory])
                    ],
                }
                for directory in sorted(entries)
            ],
        }),
    )
    return [DefaultInfo(files = depset([overlay]))]

vfs_overlay = rule(
    implementation = _vfs_overlay_impl,
    doc = "Emits a clang -ivfsoverlay file making the given header directories case-insensitive.",
    attrs = {
        "directories": attr.label_list(
            doc = "Header directories to make case-insensitive.",
            mandatory = True,
            providers = [DirectoryInfo],
        ),
    },
)

def _lowercase_library_aliases_impl(ctx):
    # Only the mixed-case libraries are copied — 179 files / ~40 MB of the
    # SDK's 400 MB — so the action's input set stays small. Autolink pragmas
    # (`#pragma comment(lib, "uuid.lib")`) always spell the name lowercase, so
    # one direction is enough here, unlike the header overlay.
    sources = []
    aliases = {}
    for target in ctx.attr.directories:
        for source in target[DirectoryInfo].transitive_files.to_list():
            lowercased = source.basename.lower()
            if lowercased == source.basename:
                continue
            if lowercased in aliases:
                fail("{} and {} both lowercase to {}".format(
                    aliases[lowercased],
                    source.path,
                    lowercased,
                ))
            aliases[lowercased] = source.path
            sources.append(source)

    manifest = ctx.actions.declare_file(ctx.label.name + ".manifest")
    ctx.actions.write(
        manifest,
        "".join([
            "{}\t{}\n".format(aliases[lowercased], lowercased)
            for lowercased in sorted(aliases)
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
        mnemonic = "WindowsSdkLibraryAliases",
        progress_message = "Aliasing %{label}",
    )
    return [DefaultInfo(files = depset([directory]))]

lowercase_library_aliases = rule(
    implementation = _lowercase_library_aliases_impl,
    doc = "Collects lowercase copies of every mixed-case import library into one directory.",
    attrs = {
        "directories": attr.label_list(
            doc = "Library directories to alias.",
            mandatory = True,
            providers = [DirectoryInfo],
        ),
    },
)
