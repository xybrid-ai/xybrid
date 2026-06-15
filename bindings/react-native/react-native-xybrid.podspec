require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "react-native-xybrid"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "13.0" }
  s.source       = { :git => "https://github.com/xybrid-ai/xybrid.git", :tag => "v#{s.version}" }

  # Swift wrapper sources (`Xybrid.swift`, `xybrid_bolt.swift`) ride along
  # with this pod rather than being pulled from SPM. This keeps consumer
  # setup to a single `npm install` + `pod install` and avoids resolving two
  # parallel package managers for the same Rust core. The files are copied
  # in by `cargo xtask stage-react-native`; they live under
  # `ios/XybridSwift/` so the `.swift` files are discovered alongside the
  # TurboModule glue.
  s.source_files = "ios/**/*.{h,m,mm,swift}"
  s.requires_arc = true
  s.swift_version = "5.0"

  # Pre-built bolt static lib bundled as an XCFramework. Copied in from
  # `bindings/apple/XCFrameworks/XybridFFI.xcframework` by
  # `cargo xtask stage-react-native`. (Android pulls its natives from the
  # Maven AAR instead — see android/build.gradle.)
  s.vendored_frameworks = "ios/Frameworks/XybridFFI.xcframework"

  # System frameworks the Rust core links against (mirrors Package.swift).
  s.frameworks = "Metal", "MetalPerformanceShaders", "MetalPerformanceShadersGraph",
                 "CoreML", "Accelerate", "Security"
  s.libraries  = "c++"

  s.pod_target_xcconfig = {
    "DEFINES_MODULE" => "YES",
    "SWIFT_OBJC_INTEROP_MODE" => "objcxx",
    # Codegen-emitted headers live under Pods/Headers/Public/RNXybridSpec
    # once the New Architecture is enabled in the host app.
    "HEADER_SEARCH_PATHS" => '"$(PODS_TARGET_SRCROOT)/ios" "$(PODS_ROOT)/Headers/Public/RNXybridSpec"',
    # Apple Silicon required. The staged XCFramework does not contain
    # `ios-x86_64-simulator` or `macos-x86_64` slices — xtask intentionally
    # drops those targets because ort-sys (v2.0.0-rc.11) ships no prebuilt
    # ONNX Runtime for Intel Mac / Intel iOS Simulator. Excluding x86_64
    # here turns "missing library at link time" into "Xcode picks arm64
    # automatically" on Apple Silicon hosts. Intel Mac and Rosetta-mode
    # builds are unsupported by design — see README.md.
    "EXCLUDED_ARCHS[sdk=iphonesimulator*]" => "i386 x86_64"
  }

  # Wire up React Native's New Architecture (TurboModules + codegen).
  # Mirrors the boilerplate in react-native-mmkv / react-native-screens.
  install_modules_dependencies(s) if respond_to?(:install_modules_dependencies)
end
