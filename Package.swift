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
            url: "https://github.com/xybrid-ai/xybrid/releases/download/v0.7.0/XybridFFI-v0.7.0.xcframework.zip",
            checksum: "61bb868f3904715d6779ce467d623396e014ad21e981d600d26f0e79fba82cd0"
        )
    ]
)
