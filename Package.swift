// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "Xybrid",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15)
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
            url: "https://github.com/xybrid-ai/xybrid/releases/download/v0.10.0-rc1/XybridFFI-v0.10.0-rc1.xcframework.zip",
            checksum: "801b4dac0bdf7b199276d8366615141ddd60e3108f6a602f94270b2730ae3a78"
        )
    ]
)
