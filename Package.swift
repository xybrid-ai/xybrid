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
//   1. Build the xcframework:  cargo xtask build-xcframework
//   2. Toggle to local mode:   ./bindings/apple/scripts/set-natives-mode.sh --set-local
//   3. Open in Xcode or run:   swift build
//
//   Toggle back to remote mode before committing:
//     ./bindings/apple/scripts/set-natives-mode.sh --set-remote
//
// =============================================================================
//
// useLocalNatives = true  → Use the local xcframework at
//                            bindings/apple/XCFrameworks/XybridFFI.xcframework
//                            (built by `cargo xtask build-xcframework`).
//
// useLocalNatives = false → Download the xcframework zip from the GitHub
//                            release for `sdkVersion`. This is the mode
//                            external SPM consumers resolve.
//
// =============================================================================
let useLocalNatives = false

// Version for remote XybridFFI download (used when useLocalNatives = false).
// Updated by the release workflow at tag time.
let sdkVersion = "0.1.0-beta13"

// SHA-256 of XybridFFI-v<sdkVersion>.xcframework.zip on the GitHub release.
// Updated by `bindings/apple/scripts/sync-spm-checksum.sh` (or the release
// workflow) so the manifest at the tagged commit matches the published asset.
let xybridFFIChecksum = "0000000000000000000000000000000000000000000000000000000000000000"

let package = Package(
    name: "Xybrid",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15),
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
