#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint xybrid_flutter.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'xybrid_flutter'
  s.version          = '0.0.1'
  s.summary          = 'Xybrid Flutter SDK for hybrid cloud-edge ML inference.'
  s.description      = <<-DESC
Xybrid Flutter SDK enables running ML models on-device (edge) or in the cloud,
with intelligent routing based on device capabilities. Supports ASR, TTS, and LLM pipelines.
                       DESC
  s.homepage         = 'https://github.com/xybrid-ai/xybrid-flutter'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Xybrid' => 'support@xybrid.dev' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'

  # Custom ONNX Runtime 1.23.2 xcframework with CoreML EP
  # Built from source to match ort crate 2.0.0-rc.11 (API v23)
  # NOTE: We use a custom build instead of onnxruntime-c CocoaPod because:
  # - CocoaPod is stuck at 1.20.0 (API v20), incompatible with ort 2.0.0-rc.11
  # - Our build includes CoreML EP for Apple Neural Engine acceleration
  s.vendored_frameworks = 'Frameworks/onnxruntime.xcframework'

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

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    # Exclude i386 and x86_64 from simulator builds
    # Modern Macs (M1/M2/M3) use arm64, and ONNX Runtime static library has issues with fat binaries
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64',
    # Force load Rust static library and link C++ standard library
    # Also force load the custom ORT xcframework
    'OTHER_LDFLAGS' => '-force_load ${BUILT_PRODUCTS_DIR}/libxybrid_flutter_ffi.a -lc++',
    # Add library search path for custom ONNX Runtime xcframework
    'LIBRARY_SEARCH_PATHS' => '"$(inherited)" "$(PODS_TARGET_SRCROOT)/Frameworks/onnxruntime.xcframework/ios-arm64"',
    # Header search path for ORT headers
    'HEADER_SEARCH_PATHS' => '"$(inherited)" "$(PODS_TARGET_SRCROOT)/Frameworks/onnxruntime.xcframework/ios-arm64/Headers"',
    # Enable Metal API validation in debug builds
    'MTL_ENABLE_DEBUG_INFO' => 'INCLUDE_SOURCE',
  }

  # user_target_xcconfig propagates settings to the app target (Runner)
  # This is CRITICAL for flutter_rust_bridge which uses DynamicLibrary.process()
  # to lookup FFI symbols from statically linked libraries
  s.user_target_xcconfig = {
    # Force load the Rust static library (NOT the framework) into the app binary
    # This ensures all FFI symbols are available for DynamicLibrary.process()
    # The Rust library is built to: ${BUILT_PRODUCTS_DIR}/xybrid_flutter/libxybrid_flutter.a
    # Note: The framework only contains ObjC glue code, Rust lib is separate
    'OTHER_LDFLAGS' => '-force_load ${BUILT_PRODUCTS_DIR}/xybrid_flutter/libxybrid_flutter_ffi.a -lc++',
    # Library search paths for custom ONNX Runtime xcframework
    'LIBRARY_SEARCH_PATHS' => '"$(inherited)" "${PODS_ROOT}/xybrid_flutter/Frameworks/onnxruntime.xcframework/ios-arm64"',
  }
end