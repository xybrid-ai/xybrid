# Optional Android speech component (unconfigured)

This SDK-owned C adapter is an opt-in build prerequisite for
`libxybrid_zzz_tts.so`. It exports only `xybrid_tts_get_api_v1` and links only
the independently verified selected Phonon + Kitten archive. **Current rc.8
pins cannot build its metadata or notice delivery inputs.** Rust's production
`EXPECTED_CONTRACT` remains `None`; no SDK feature, default, AAR, Flutter or
Unity package is activated here.

The fixed ARM64 table matches `crates/zzz-sys/src/companion.rs`: 24 header
bytes, 32 identity bytes and eight function pointers, totaling 120 bytes. The
getter copies into caller-owned storage only for exact ABI 1/size 120. Rejection
clears at most the declared output size, bounded by 120. It performs no engine
probe. Static initialization contains only fixed metadata/function pointers;
actual constructors in the eventual linked engine still require inspection.
The loader owns the library/table lifetime, validates every slot and field,
and checks actual device API 29 before opening this fixed SONAME. Static Rust
`bindings` retain precedence when both transport features are enabled.

`metadata.py` runs the packaged `zzz_pull.py` against the actual committed
Android pin and additionally requires schema 5, baseline CPU, API 29 and exactly
`c,m,dl`. The verifier independently checks the selected profile/models,
embed-only ABI 1, archive SHA256 and immutable 14-file notice map. Identity is:

```text
SHA256(b"xybrid.android-tts-companion.v1\0" +
       json.dumps({"release_tag": pin.release_tag, "entry": pin.entry},
                  sort_keys=True, separators=(",", ":"),
                  ensure_ascii=True).encode("ascii"))
```

The entire actual selected entry, including archive hash and notice map,
participates. The generated header contains this identity and fixed ABI/API/
model constants. No invented engine hash, runtime-supplied expected identity,
private receipt or source path is accepted. This identifies a reviewed SDK
component contract; it does not authenticate arbitrary code in a compromised
native namespace. Future adoption must independently compile the expected ID
into Rust only after final package qualification.

After a genuine notice-bearing release is published, reviewed pins are adopted,
and matching private inputs are staged, the trusted producer's concrete target
is:

```sh
bazel build -c opt \
  --extra_toolchains=//bazel/zzz/companion:api29_toolchain \
  --repo_env=XYBRID_ZZZ_PREBUILT_DIR \
  //bazel/zzz/companion:delivery_inputs
```

Run this on a trusted Linux x86_64 executor with `XYBRID_ZZZ_PREBUILT_DIR`
pointing to its verified private slice. It uses the repository's pinned Linux-
host NDK. Credentials, execution and private-cache restrictions belong to the
separately reviewed trusted producer, not this target. `:companion` returns only the fixed-name shared
object. `:delivery_inputs` adds `@zzz_android//:notices`, the shared exact 14-file
input under `notices/<canonical-relative-path>`. It is an input set for a final
SDK package, not a standalone engine distribution or a complete public bundle.
It contains no raw archive, engine header or receipt. The notice export is a
separate coordinated prerequisite; missing/schema 4 notice targets fail analysis.

The separate named NDK repository uses real `api_level=29`, as supported by the
[inspected repository rule](https://github.com/bazelbuild/rules_android_ndk/blob/v0.1.5/rules.bzl).
Its [extension](https://github.com/bazelbuild/rules_android_ndk/blob/v0.1.5/extension.bzl)
only configures the original hardcoded repository. The API 29 toolchain is **not
globally registered**. The explicit extra-toolchain option takes priority over
the original registered API 28 toolchain, which also matches the marked platform;
without it, the adapter's actual ARM64/API==29 compile guard rejects the build.
This order follows [Bazel's toolchain resolution contract](https://bazel.build/extending/toolchains#toolchain-resolution);
if multiple extra toolchains are passed, the last one has highest priority.
The transition changes only the companion child. Its custom constraint cannot
match any existing base platform, so SDK siblings retain API 28 and their current
minima. Do not replace this seam with macro overrides against API 28 stubs.

The anonymous version script plus hidden visibility and archive localization
hide every engine/runtime symbol. No C++ code or runtime dependency is requested.
A future actual binary must pass:

```sh
python3 bazel/zzz/companion/check_exports.py --readelf <pinned-ndk-llvm-readelf> \
  <exact-final-libxybrid_zzz_tts.so>
```

The audit rejects any `zzz_*` dynamic import/export, any extra export, wrong
endianness/machine/SONAME, unexpected DT_NEEDED, load alignment below 16 KiB,
or non-congruent LOAD offsets/addresses for the declared alignment. It must run
again after final packaging/strip. It does not establish actual API 29 import
availability, old-OS base loading, selected model correctness, absence of private
strings/debug information, Android linker namespaces, APK alignment or installed
application behavior. Real Bazel toolchain resolution/linking, actual ELF exports,
16 KiB device loading, API 21/24/28 negative and API 29+ positive installed SDK apps,
capability/ownership/runtime tests and exact final notice delivery remain
unqualified. Offline tests use synthetic metadata/headers and readelf reports;
no real engine component has been built or distributed by this preparation.
