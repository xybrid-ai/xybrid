// swift-tools-version:5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

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
        // Main Swift target with public API
        .target(
            name: "Xybrid",
            dependencies: ["xybrid_uniffiFFI"],
            path: "Sources/Xybrid"
        ),
        // FFI target for UniFFI-generated C bindings
        // The module name must match what the generated Swift code imports
        .target(
            name: "xybrid_uniffiFFI",
            path: "Sources/xybrid_uniffiFFI",
            publicHeadersPath: "include"
        )
    ]
)
