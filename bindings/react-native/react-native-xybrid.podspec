require "json"
require_relative "ios/xybrid_natives"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  # The npm package is @xybrid/react-native; pod names cannot contain `@` or
  # `/`, so the pod keeps this name (it only shows up in Podfile.lock).
  s.name         = "react-native-xybrid"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  # Matches the XCFramework's minimum (bindings/apple/BUILD.bazel) and
  # Package.swift; React Native 0.76+ already needs iOS 15.1.
  s.platforms    = { :ios => "16.0" }
  s.source       = { :git => "https://github.com/xybrid-ai/xybrid.git", :tag => "v#{s.version}" }

  # The TurboModule (XybridModule.mm + the Swift implementation) and the
  # Swift SDK it calls (ios/XybridSwift: Xybrid.swift + xybrid_bolt.swift, the
  # same sources the standalone Apple SDK ships) compile into this one pod.
  s.source_files = "ios/*.{mm,swift}", "ios/XybridSwift/*.swift"
  s.swift_version = "5.0"

  # The Rust core, resolved at `pod install` time — downloaded from the
  # GitHub Release and checked against package.json's pinned SHA-256, or a
  # local build. See ios/xybrid_natives.rb for the order and overrides.
  s.vendored_frameworks = XybridNatives.prepare!(__dir__)

  # System frameworks the Rust core links against (mirrors Package.swift).
  s.frameworks = "Metal", "MetalPerformanceShaders", "MetalPerformanceShadersGraph",
                 "CoreML", "Accelerate", "Security"
  s.libraries  = "c++"

  s.pod_target_xcconfig = {
    "DEFINES_MODULE" => "YES",
    # A method missing from the Codegen protocol only warns — and then
    # crashes at the first JS call, because the TurboModule looks methods up
    # by the protocol's selectors. Make it a build error instead.
    "WARNING_CFLAGS" => "$(inherited) -Werror=protocol",
    # The XCFramework has no x86_64 simulator slice (no prebuilt ONNX Runtime
    # for Intel simulators), so build simulator targets for arm64 only.
    "EXCLUDED_ARCHS[sdk=iphonesimulator*]" => "i386 x86_64",
  }

  # React Native's New Architecture wiring (TurboModule headers + Codegen).
  install_modules_dependencies(s)
end
