#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint xybrid_flutter.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'xybrid_flutter'
  s.version          = '0.1.0'
  s.summary          = 'Xybrid Flutter SDK for hybrid cloud-edge ML inference.'
  s.description      = <<-DESC
Xybrid Flutter SDK enables running ML models on-device (edge) or in the cloud,
with intelligent routing based on device capabilities. Supports ASR, TTS, and LLM pipelines.
                       DESC
  s.homepage         = 'https://github.com/xybrid-ai/bindings/flutter'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Xybrid' => 'support@xybrid.dev' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'

  # ONNX Runtime 1.23.2 with CoreML EP. NOT vendored in the pod (too large for
  # pub.dev), and not needed separately by a pub.dev install: the precompiled
  # Rust library already contains it. Only a source build resolves it, in
  # build_pod.sh, for ort-sys to link:
  # - Monorepo dev: symlink at Frameworks/onnxruntime.xcframework -> vendor/ort-ios/
  # - otherwise: downloaded from HuggingFace to ~/.xybrid/cache/ort-ios/
  # See cargokit/build_pod.sh for the full resolution logic.

  # iOS 13.0 minimum for modern APIs (Metal 2, Combine, CoreML 3, etc.)
  s.platform = :ios, '13.0'
  s.swift_version = '5.0'

  # System frameworks required for ML inference
  # - Metal: GPU compute for Candle models
  # - MetalPerformanceShaders: Optimized ML primitives
  # - Accelerate: BLAS/LAPACK for CPU inference (ONNX Runtime)
  # - CoreML: Apple Neural Engine acceleration (CoreML EP)
  s.frameworks = 'Metal', 'MetalPerformanceShaders', 'Accelerate', 'CoreML'

  s.script_phase = {
    :name => 'Build Rust library',
    # First argument is relative path to the `rust` folder, second is name of rust library
    :script => 'sh "$PODS_TARGET_SRCROOT/../cargokit/build_pod.sh" ../rust xybrid_flutter_ffi',
    :execution_position => :before_compile,
    :input_files => ['${BUILT_PRODUCTS_DIR}/cargokit_phony'],
    # Let XCode know that the static library referenced in -force_load below is
    # created by this build step.
    :output_files => ["${BUILT_PRODUCTS_DIR}/libxybrid_flutter_ffi.a"],
  }

  # ORT library/header search paths include BOTH possible locations:
  # 1. Vendored path (monorepo dev with symlink)
  # 2. Downloaded cache path (pub.dev installs)
  # Xcode silently ignores non-existent search paths.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    # Exclude i386 and x86_64 from simulator builds
    # Modern Macs (M1/M2/M3) use arm64, and ONNX Runtime static library has issues with fat binaries
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64',
    # Force load Rust static library and link C++ standard library
    'OTHER_LDFLAGS' => '-force_load ${BUILT_PRODUCTS_DIR}/libxybrid_flutter_ffi.a -lc++',
    # Library search paths for ORT — vendored (monorepo) + cached download (pub.dev)
    'LIBRARY_SEARCH_PATHS' => '"$(inherited)" "$(PODS_TARGET_SRCROOT)/Frameworks/onnxruntime.xcframework/ios-arm64" "$(HOME)/.xybrid/cache/ort-ios/1.23.2/onnxruntime.xcframework/ios-arm64"',
    # Header search paths for ORT C headers
    'HEADER_SEARCH_PATHS' => '"$(inherited)" "$(PODS_TARGET_SRCROOT)/Frameworks/onnxruntime.xcframework/ios-arm64/Headers" "$(HOME)/.xybrid/cache/ort-ios/1.23.2/onnxruntime.xcframework/ios-arm64/Headers"',
    # Enable Metal API validation in debug builds
    'MTL_ENABLE_DEBUG_INFO' => 'INCLUDE_SOURCE',
  }

  # user_target_xcconfig propagates settings to the app target (Runner)
  # This is CRITICAL for flutter_rust_bridge which uses DynamicLibrary.process()
  # to lookup FFI symbols from statically linked libraries
  s.user_target_xcconfig = {
    # Force load the Rust static library (NOT the framework) into the app binary
    # This ensures all FFI symbols are available for DynamicLibrary.process()
    'OTHER_LDFLAGS' => '-force_load ${BUILT_PRODUCTS_DIR}/xybrid_flutter/libxybrid_flutter_ffi.a -lc++',
    # Library search paths for ORT — vendored (monorepo) + cached download (pub.dev)
    'LIBRARY_SEARCH_PATHS' => '"$(inherited)" "${PODS_ROOT}/xybrid_flutter/Frameworks/onnxruntime.xcframework/ios-arm64" "$(HOME)/.xybrid/cache/ort-ios/1.23.2/onnxruntime.xcframework/ios-arm64"',
  }
end