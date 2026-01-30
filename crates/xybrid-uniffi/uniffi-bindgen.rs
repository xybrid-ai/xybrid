//! UniFFI bindgen binary for generating Swift and Kotlin bindings.
//!
//! This binary is used to generate platform-specific bindings from the
//! UniFFI library. Run with:
//!
//! ```bash
//! cargo run -p xybrid-uniffi --bin uniffi-bindgen -- \
//!     generate --library target/release/libxybrid_uniffi.dylib \
//!     --language swift --out-dir ./bindings/swift
//! ```

fn main() {
    uniffi::uniffi_bindgen_main()
}
