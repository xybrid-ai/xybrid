// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "Xybrid",
    platforms: [
        // Match the published iOS-only XCFramework and its deployment floor.
        .iOS(.v16),
    ],
    products: [
        .library(
            name: "Xybrid",
            targets: ["Xybrid"]
        )
    ],
    targets: [
        .target(
            name: "Xybrid",
            dependencies: ["XybridFFI"],
            path: "Sources/Xybrid",
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
        .binaryTarget(
            name: "XybridFFI",
            url: "https://github.com/xybrid-ai/xybrid/releases/download/v0.17.0/XybridFFI-v0.17.0.xcframework.zip",
            checksum: "a3f27a52d88e9d05179c2ccf537f54352c6263630000f493c8292af1e476e9bc"
        )
    ]
)
