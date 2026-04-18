//! Cross-platform device introspection for telemetry.
//!
//! The [`Device`] struct describes the host the SDK is running on. Today it
//! exposes a stable hashed [`Device::id`] and a [`Device::platform`] string;
//! more fields (OS version, locale, hardware capabilities, ...) can be added
//! over time without breaking callers.
//!
//! # Device ID
//!
//! [`Device::id`] is a SHA-256 hash of a platform-specific machine identifier
//! combined with a salt. It is stable across process restarts, anonymous
//! (one-way), and computed once per process.
//!
//! ## Platform Strategies
//!
//! | Platform | Source |
//! |----------|--------|
//! | macOS | `kern.uuid` via sysctl (IOPlatformUUID) |
//! | Linux | `/etc/machine-id` or `/var/lib/dbus/machine-id` |
//! | Windows | `HKLM\SOFTWARE\Microsoft\Cryptography\MachineGuid` |
//! | iOS / Android | UUID persisted in the SDK cache directory |
//! | Other | UUID v5 derived from the platform string (deterministic fallback) |
//!
//! # Privacy
//!
//! The raw machine identifier is never stored or transmitted. Only the
//! salted SHA-256 hash leaves the process, so the original hardware
//! identity cannot be recovered.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Compile-time salt mixed into the device-ID hash. Not a secret — its
/// purpose is to namespace our hashes so they cannot be looked up against
/// rainbow tables of bare machine IDs published by other tools.
const DEVICE_ID_SALT: &str = "xybrid-device-id-v1";

/// Cached, lazily-computed [`Device`] singleton for the process lifetime.
static DEVICE: OnceLock<Device> = OnceLock::new();

/// Describes the host the SDK is running on.
///
/// Use [`Device::current()`] to obtain the cached instance for this process.
/// Fields are public so callers can read them directly; new fields will be
/// added over time as we surface more host context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Device {
    /// Stable, anonymous device identifier (64-char hex SHA-256).
    /// See module docs for the per-platform machine-ID source.
    pub id: String,

    /// Platform triple, e.g. `"macos-arm64"`, `"ios-arm64"`, `"linux-x86_64"`.
    /// Matches [`crate::platform::current_platform()`].
    pub platform: String,
}

impl Device {
    /// Return the cached [`Device`] for this process.
    ///
    /// The first call resolves the machine ID and computes the hash; later
    /// calls are zero-cost. The returned reference lives for the program
    /// lifetime — clone it if you need an owned value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_sdk::Device;
    ///
    /// let d = Device::current();
    /// assert_eq!(d.id.len(), 64);
    /// assert!(!d.platform.is_empty());
    /// ```
    pub fn current() -> &'static Device {
        DEVICE.get_or_init(Device::detect)
    }

    /// Compute a fresh [`Device`] without consulting the cache.
    ///
    /// Mostly useful in tests; production code should use [`Device::current`].
    pub fn detect() -> Device {
        Device {
            id: compute_device_id(),
            platform: crate::platform::current_platform().to_string(),
        }
    }
}

/// Convenience accessor for the device ID string.
///
/// Equivalent to `&Device::current().id`.
pub fn device_id() -> &'static str {
    &Device::current().id
}

fn compute_device_id() -> String {
    let raw_id = get_machine_id();
    let mut hasher = Sha256::new();
    hasher.update(raw_id.as_bytes());
    hasher.update(DEVICE_ID_SALT.as_bytes());
    format!("{:x}", hasher.finalize())
}

// ============================================================================
// Platform-specific machine ID retrieval
// ============================================================================

/// macOS: read IOPlatformUUID via sysctl kern.uuid
#[cfg(target_os = "macos")]
fn get_machine_id() -> String {
    get_macos_machine_id().unwrap_or_else(fallback_machine_id)
}

#[cfg(target_os = "macos")]
fn get_macos_machine_id() -> Option<String> {
    // kern.uuid returns the IOPlatformUUID — stable across reboots
    let output = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("kern.uuid")
        .output()
        .ok()?;
    if output.status.success() {
        let uuid = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !uuid.is_empty() {
            return Some(uuid);
        }
    }
    None
}

/// Linux: read /etc/machine-id (systemd) or /var/lib/dbus/machine-id
#[cfg(target_os = "linux")]
fn get_machine_id() -> String {
    get_linux_machine_id().unwrap_or_else(fallback_machine_id)
}

#[cfg(target_os = "linux")]
fn get_linux_machine_id() -> Option<String> {
    for path in &["/etc/machine-id", "/var/lib/dbus/machine-id"] {
        if let Ok(content) = std::fs::read_to_string(path) {
            let id = content.trim().to_string();
            if !id.is_empty() {
                return Some(id);
            }
        }
    }
    None
}

/// Windows: read MachineGuid from the registry
#[cfg(target_os = "windows")]
fn get_machine_id() -> String {
    get_windows_machine_id().unwrap_or_else(fallback_machine_id)
}

#[cfg(target_os = "windows")]
fn get_windows_machine_id() -> Option<String> {
    // Use reg.exe to avoid a winreg dependency
    let output = std::process::Command::new("reg")
        .args([
            "query",
            r"HKLM\SOFTWARE\Microsoft\Cryptography",
            "/v",
            "MachineGuid",
        ])
        .output()
        .ok()?;
    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout);
        // Output format: "    MachineGuid    REG_SZ    <guid>"
        for line in text.lines() {
            if line.contains("MachineGuid") {
                if let Some(guid) = line.split_whitespace().last() {
                    if !guid.is_empty() {
                        return Some(guid.to_string());
                    }
                }
            }
        }
    }
    None
}

/// iOS: persist a UUID in the app sandbox. Stable across app restarts but not
/// across reinstalls (matching Apple's `identifierForVendor` semantics).
#[cfg(target_os = "ios")]
fn get_machine_id() -> String {
    get_persisted_id().unwrap_or_else(fallback_machine_id)
}

/// Android: same persisted-UUID approach as iOS. Avoids a JNI dependency
/// for `Settings.Secure.ANDROID_ID`.
#[cfg(target_os = "android")]
fn get_machine_id() -> String {
    get_persisted_id().unwrap_or_else(fallback_machine_id)
}

/// For iOS and Android: persist a UUID in the SDK cache directory.
/// Reads an existing ID or generates + writes a new one.
#[cfg(any(target_os = "ios", target_os = "android"))]
fn get_persisted_id() -> Option<String> {
    let cache_dir = crate::get_sdk_cache_dir()
        .or_else(dirs::cache_dir)
        .or_else(dirs::data_local_dir)?;

    let id_file = cache_dir.join(".xybrid-device-id");

    // Try reading an existing ID
    if let Ok(content) = std::fs::read_to_string(&id_file) {
        let id = content.trim().to_string();
        if !id.is_empty() {
            return Some(id);
        }
    }

    // Generate and persist a new one
    let new_id = uuid::Uuid::new_v4().to_string();
    let _ = std::fs::create_dir_all(&cache_dir);
    let _ = std::fs::write(&id_file, &new_id);
    Some(new_id)
}

/// Catch-all for platforms not explicitly handled above (e.g. WASM).
#[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows",
    target_os = "ios",
    target_os = "android",
)))]
fn get_machine_id() -> String {
    fallback_machine_id()
}

/// Fallback: deterministic UUID v5 from the platform string. Used only when
/// the platform-specific source above fails — same on every device of the
/// same platform, so it's strictly worse than a real machine ID.
fn fallback_machine_id() -> String {
    let platform = crate::platform::current_platform();
    let namespace = uuid::Uuid::NAMESPACE_DNS;
    uuid::Uuid::new_v5(&namespace, format!("xybrid-fallback-{platform}").as_bytes()).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_returns_cached_instance() {
        let a = Device::current();
        let b = Device::current();
        assert!(
            std::ptr::eq(a, b),
            "Device::current must return the same singleton"
        );
    }

    #[test]
    fn id_is_64_hex_chars() {
        let d = Device::current();
        assert_eq!(d.id.len(), 64, "SHA-256 hex should be 64 chars");
        assert!(
            d.id.chars().all(|c| c.is_ascii_hexdigit()),
            "id must be hex"
        );
    }

    #[test]
    fn platform_matches_current_platform() {
        assert_eq!(
            Device::current().platform,
            crate::platform::current_platform()
        );
    }

    #[test]
    fn detect_is_deterministic() {
        let a = Device::detect();
        let b = Device::detect();
        assert_eq!(a, b, "detect() must produce the same Device every time");
    }

    #[test]
    fn salt_changes_hash_output() {
        let raw = get_machine_id();
        let mut hasher = Sha256::new();
        hasher.update(raw.as_bytes());
        let unsalted = format!("{:x}", hasher.finalize());

        let salted = compute_device_id();
        assert_ne!(unsalted, salted, "Salt must change the hash output");
    }

    #[test]
    fn machine_id_not_empty() {
        assert!(
            !get_machine_id().is_empty(),
            "Machine ID must never be empty"
        );
    }

    #[test]
    fn device_id_helper_matches_struct() {
        assert_eq!(device_id(), Device::current().id.as_str());
    }
}
