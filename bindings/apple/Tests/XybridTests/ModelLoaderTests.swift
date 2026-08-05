import Foundation
import XCTest
@testable import Xybrid

final class ModelLoaderTests: XCTestCase {
    func testRegistryShortcutCreatesUnloadedRegistryReference() {
        let loader = Xybrid.model("kokoro-82m")

        XCTAssertEqual(loader.source, .registry("kokoro-82m"))
    }

    func testTypedSourceCreatesUnloadedBundleReference() {
        let url = URL(fileURLWithPath: "/models/kokoro.xyb")
        let source = ModelSource.bundle(url)

        let loader = Xybrid.model(source)

        XCTAssertEqual(loader.source, source)
    }
}
