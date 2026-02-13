//! Platform detection for consistent platform identification across SDK consumers.
//!
//! This module provides compile-time platform detection that returns
//! standardized platform strings used by the Xybrid model registry
//! and bundle system.
//!
//! # Platform Strings
//!
//! The following platform strings are recognized:
//!
//! | Platform | Architecture | String |
//! |----------|--------------|--------|
//! | macOS | ARM64 (M1/M2/M3) | `macos-arm64` |
//! | macOS | x86_64 (Intel) | `macos-x86_64` |
//! | iOS | ARM64 | `ios-arm64` |
//! | Android | ARM64 | `android-arm64` |
//! | Android | ARMv7 | `android-arm` |
//! | Linux | x86_64 | `linux-x86_64` |
//! | Windows | x86_64 | `windows-x86_64` |
//!
//! # Example
//!
//! ```rust
//! use xybrid_sdk::current_platform;
//!
//! let platform = current_platform();
//! println!("Running on: {}", platform);
//! ```

/// Detect the current platform at compile time.
///
/// Returns a standardized platform string that matches the Xybrid
/// model registry's platform identifiers.
///
/// # Platform Detection Logic
///
/// 1. First checks for iOS (target_os = "ios")
/// 2. Then checks for Android (target_os = "android")
/// 3. Then checks for macOS (target_os = "macos")
/// 4. Then checks for Linux (target_os = "linux")
/// 5. Then checks for Windows (target_os = "windows")
/// 6. Falls back to "unknown" if none match
///
/// For each OS, the architecture is detected:
/// - `aarch64` → ARM64 variants
/// - `arm` → ARMv7 (Android only)
/// - `x86_64` → Intel/AMD 64-bit
fn detect_platform() -> &'static str {
    // iOS detection (must come before macOS since both are Apple platforms)
    #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
    {
        "ios-arm64"
    }

    // Android detection
    #[cfg(all(target_os = "android", target_arch = "aarch64"))]
    {
        "android-arm64"
    }

    #[cfg(all(target_os = "android", target_arch = "arm"))]
    {
        "android-arm"
    }

    // macOS detection
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "macos-arm64"
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "macos-x86_64"
    }

    // Linux detection
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "linux-x86_64"
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "linux-arm64"
    }

    // Windows detection
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "windows-x86_64"
    }

    // Fallback for unknown platforms
    #[cfg(not(any(
        all(target_os = "ios", target_arch = "aarch64"),
        all(target_os = "android", target_arch = "aarch64"),
        all(target_os = "android", target_arch = "arm"),
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "windows", target_arch = "x86_64"),
    )))]
    {
        "unknown"
    }
}

/// Get the current platform identifier.
///
/// This is the primary function to use for platform detection.
/// It returns a standardized platform string that can be used
/// with the Xybrid model registry.
///
/// # Example
///
/// ```rust
/// use xybrid_sdk::current_platform;
///
/// let platform = current_platform();
/// assert!(!platform.is_empty());
///
/// // Use with registry
/// // let bundle = registry.fetch("whisper-tiny", platform)?;
/// ```
///
/// # Returns
///
/// A static string identifying the platform (e.g., `"macos-arm64"`, `"ios-arm64"`).
pub fn current_platform() -> &'static str {
    detect_platform()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_string_not_empty() {
        let platform = current_platform();
        assert!(!platform.is_empty(), "Platform string should not be empty");
    }

    #[test]
    fn test_platform_format() {
        let platform = current_platform();
        // Platform should be in format "os-arch" or "unknown"
        if platform != "unknown" {
            assert!(
                platform.contains('-'),
                "Platform string should contain a hyphen: {}",
                platform
            );
        }
    }

    #[test]
    fn test_platform_consistency() {
        // Calling multiple times should return the same value
        let platform1 = current_platform();
        let platform2 = current_platform();
        assert_eq!(platform1, platform2);
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_macos_arm64() {
        assert_eq!(current_platform(), "macos-arm64");
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    fn test_macos_x86_64() {
        assert_eq!(current_platform(), "macos-x86_64");
    }

    #[test]
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn test_linux_x86_64() {
        assert_eq!(current_platform(), "linux-x86_64");
    }

    #[test]
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    fn test_windows_x86_64() {
        assert_eq!(current_platform(), "windows-x86_64");
    }
}
