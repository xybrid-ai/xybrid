//! Prints `detect_capabilities()` output. Useful for sanity-checking the
//! probes return sensible values on the host machine.

fn main() {
    let caps = xybrid_core::device::capabilities::detect_capabilities();
    println!("{caps:#?}");
}
