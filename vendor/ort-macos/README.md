# Vendored ONNX Runtime — macOS (static)

`aarch64-apple-darwin/libonnxruntime.a` is the exact artifact `ort-sys` 2.0.0-rc.11
would download at build time (pyke CDN, `ms@1.23.2`, flavor `none`/CPU+CoreML):

  https://cdn.pyke.io/0/pyke:ort-rs/ms@1.23.2/aarch64-apple-darwin.tar.lzma2
  sha256: 0897a0e1b840566a97e5a49497b02cbc204be2d006815174b639bc99731840f9

Vendored because a build-time download is incompatible with remote execution
(the fetched file is not a tracked action output, so it does not propagate to
the next action). Fed to ort-sys via ORT_LIB_LOCATION — the same mechanism as
`vendor/ort-ios` (see MODULE.bazel ort-sys annotation + //:ort_macos_arm64).
