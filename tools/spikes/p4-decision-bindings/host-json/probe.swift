// Host JSON probe (Swift Foundation): key order and number identity. Usage: see README.md.
import Foundation
let t = CommandLine.arguments[1]
let o = try! JSONSerialization.jsonObject(with: t.data(using: .utf8)!) as! [String: Any]
print("order", (o["questions"] as! [String: Any]).keys.map { $0 })
print("state", String(data: try! JSONSerialization.data(withJSONObject: o["state"]!, options: [.sortedKeys]), encoding: .utf8)!)
do { _ = try JSONSerialization.data(withJSONObject: [-Double.infinity]); print("-inf ok") } catch { print("-inf throws") }
