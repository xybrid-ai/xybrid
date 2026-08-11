// swift-tools-version:5.7
import PackageDescription

// =============================================================================
// Xybrid Swift SDK — Swift Package Manager Distribution
// =============================================================================
//
// This is the SINGLE Package.swift for both local development and SPM
// consumption.
//
// FOR EXTERNAL USERS (consuming via GitHub):
//
//   .package(url: "https://github.com/xybrid-ai/xybrid", exact: "0.1.0-beta13")
//
// FOR LOCAL DEVELOPMENT:
//
//   1. Build the xcframework:  bazel build --config=ios //bindings/apple:XybridFFI
//   2. Unzip it locally:       unzip -o bazel-bin/bindings/apple/XybridFFI.xcframework.zip -d bindings/apple/XCFrameworks
//   3. Toggle to local mode:   ./bindings/apple/scripts/set-natives-mode.sh --set-local
//   4. Open in Xcode or run:   swift build
//
//   Toggle back to remote mode before committing:
//     ./bindings/apple/scripts/set-natives-mode.sh --set-remote
//
// =============================================================================
//
// useLocalNatives = true  → Use the local xcframework at
//                            bindings/apple/XCFrameworks/XybridFFI.xcframework
//                            (unzipped from the Bazel build above).
//
// useLocalNatives = false → Download the xcframework zip from the GitHub
//                            release for `sdkVersion`. This is the mode
//                            external SPM consumers resolve.
//
// =============================================================================
let useLocalNatives = false

// Version for remote XybridFFI download (used when useLocalNatives = false).
// Updated by the release workflow at tag time.
let sdkVersion = "0.5.0"

// SHA-256 of XybridFFI-v<sdkVersion>.xcframework.zip on the GitHub release.
// Updated by `bindings/apple/scripts/sync-spm-checksum.sh` (or the release
// workflow) so the manifest at the tagged commit matches the published asset.
let xybridFFIChecksum = "86e7c6bbe26b3d3605e10cfa5733417bf5eb03ce5822cd649196be761ce01c72"

let package = Package(
    name: "Xybrid",
    platforms: [
        // iOS-only, min iOS 16 — matches what the shipped
        // XybridFFI.xcframework actually is (boltffi.toml: include_macos =
        // false, so only ios-arm64 + ios-arm64-simulator ship;
        // deployment_target = "16.0", so the binary cannot link below iOS 16).
        // A lower floor (.v13) or a .macOS declaration let SPM resolve the
        // package for apps it can't actually link — re-add either only after
        // the xcframework is rebuilt to match (each a separate feature).
        .iOS(.v16),
    ],
    products: [
        .library(
            name: "Xybrid",
            targets: ["Xybrid"]
        ),
    ],
    targets: [
        .target(
            name: "Xybrid",
            dependencies: ["XybridFFI"],
            path: "bindings/apple/Sources/Xybrid",
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
                .linkedFramework("CoreML"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Security"),
            ]
        ),
        .testTarget(
            name: "XybridTests",
            dependencies: ["Xybrid"],
            path: "bindings/apple/Tests/XybridTests"
        ),
        xybridFFITarget(),
    ]
)

func xybridFFITarget() -> Target {
    if useLocalNatives {
        return .binaryTarget(
            name: "XybridFFI",
            path: "bindings/apple/XCFrameworks/XybridFFI.xcframework"
        )
    } else {
        return .binaryTarget(
            name: "XybridFFI",
            url: "https://github.com/xybrid-ai/xybrid/releases/download/v\(sdkVersion)/XybridFFI-v\(sdkVersion).xcframework.zip",
            checksum: xybridFFIChecksum
        )
    }
}
