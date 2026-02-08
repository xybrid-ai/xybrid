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
  s.homepage         = 'https://github.com/xybrid-ai/bindings/flutter'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Xybrid' => 'support@xybrid.dev' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'FlutterMacOS'

  s.platform = :osx, '10.15'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'

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
    # Flutter.framework does not contain a i386 slice.
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    # ONNX Runtime (ort-sys) does not provide prebuilt x86_64-apple-darwin binaries.
    # Intel Macs are not supported. Apple Silicon Macs run arm64 natively.
    'EXCLUDED_ARCHS[sdk=macosx*]' => 'x86_64',
    'OTHER_LDFLAGS' => '-force_load ${BUILT_PRODUCTS_DIR}/libxybrid_flutter_ffi.a -lc++ -framework SystemConfiguration -framework Security -framework CoreFoundation -framework Metal -framework MetalPerformanceShaders -framework Accelerate',
  }

  # Automatically configure user's app to exclude x86_64 on macOS.
  # This is required because ONNX Runtime does not provide Intel Mac binaries.
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=macosx*]' => 'x86_64',
  }

  # Required frameworks for reqwest HTTP client, TLS, and Candle Metal/Accelerate acceleration
  s.frameworks = ['SystemConfiguration', 'Security', 'CoreFoundation', 'Metal', 'MetalPerformanceShaders', 'Accelerate']
end