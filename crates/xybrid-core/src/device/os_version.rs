//! Runtime OS versions for registry compatibility checks.

/// Detect the running system's registry OS compatibility token.
///
/// Returns `android-api-N` from Android's runtime SDK property, `ios-X.Y` or
/// `macos-X.Y` from Foundation, or `glibc-X.Y` from the GNU C library itself.
/// Apple patch versions are omitted; an iOS simulator reports its simulated
/// iOS version through Foundation, rather than the host's macOS version.
///
/// Unsupported platforms (including non-glibc Linux) and unknown versions
/// return `None`. This does not use build targets, environment overrides,
/// distribution release strings, or child processes to infer a version.
pub fn runtime_os_version() -> Option<String> {
    #[cfg(target_os = "android")]
    {
        use std::ffi::{c_char, c_int};

        extern "C" {
            fn __system_property_get(name: *const c_char, value: *mut c_char) -> c_int;
        }

        // Bionic's PROP_VALUE_MAX includes the terminating NUL. This property
        // is the device's SDK_INT, not its Android release or our build floor.
        let mut value = [0_u8; 92];
        // SAFETY: the property name is NUL-terminated and value has the
        // PROP_VALUE_MAX bytes required by __system_property_get. Bionic
        // writes at most that many bytes and retains neither pointer.
        let length = unsafe {
            __system_property_get(c"ro.build.version.sdk".as_ptr(), value.as_mut_ptr().cast())
        };
        let length = usize::try_from(length).ok()?;
        if length == 0 || length >= value.len() {
            return None;
        }
        android_os_token(std::str::from_utf8(&value[..length]).ok()?)
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    {
        let version = objc2_foundation::NSProcessInfo::processInfo().operatingSystemVersion();
        let platform = if cfg!(target_os = "ios") {
            "ios"
        } else {
            "macos"
        };
        apple_os_token(platform, version.majorVersion, version.minorVersion)
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        use std::ffi::{c_char, CStr};

        extern "C" {
            fn gnu_get_libc_version() -> *const c_char;
        }

        // SAFETY: glibc's getter has no preconditions and returns a pointer
        // to a static NUL-terminated string. Only GNU Linux links this symbol.
        let version = unsafe { gnu_get_libc_version() };
        if version.is_null() {
            return None;
        }
        // SAFETY: the non-null pointer refers to glibc's static string above.
        glibc_os_token(unsafe { CStr::from_ptr(version) }.to_str().ok()?)
    }

    #[cfg(not(any(
        target_os = "android",
        target_os = "ios",
        target_os = "macos",
        all(target_os = "linux", target_env = "gnu")
    )))]
    {
        None
    }
}

#[cfg(any(
    target_os = "android",
    all(target_os = "linux", target_env = "gnu"),
    test
))]
fn decimal_component(value: &str) -> Option<u32> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

#[cfg(any(target_os = "android", test))]
fn android_os_token(sdk: &str) -> Option<String> {
    let sdk = decimal_component(sdk)?;
    (sdk > 0).then(|| format!("android-api-{sdk}"))
}

#[cfg(any(target_os = "ios", target_os = "macos", test))]
fn apple_os_token(platform: &str, major: isize, minor: isize) -> Option<String> {
    if !matches!(platform, "ios" | "macos") || major <= 0 || minor < 0 {
        return None;
    }
    Some(format!("{platform}-{major}.{minor}"))
}

#[cfg(any(all(target_os = "linux", target_env = "gnu"), test))]
fn glibc_os_token(version: &str) -> Option<String> {
    let (major, minor) = version.split_once('.')?;
    let major = decimal_component(major)?;
    let minor = decimal_component(minor)?;
    (major > 0).then(|| format!("glibc-{major}.{minor}"))
}

#[cfg(test)]
mod tests {
    use super::{android_os_token, apple_os_token, glibc_os_token, runtime_os_version};

    #[test]
    fn android_formats_sdk_api_levels() {
        assert_eq!(android_os_token("29").as_deref(), Some("android-api-29"));
        assert_eq!(android_os_token("36").as_deref(), Some("android-api-36"));
        assert_eq!(android_os_token("034").as_deref(), Some("android-api-34"));
    }

    #[test]
    fn android_rejects_unknown_and_non_api_values() {
        for value in [
            "",
            "0",
            "-1",
            "+34",
            " 34",
            "34\n",
            "14.0",
            "REL",
            "4294967296",
        ] {
            assert_eq!(android_os_token(value), None, "accepted {value:?}");
        }
    }

    #[test]
    fn apple_formats_major_and_minor_versions() {
        assert_eq!(apple_os_token("ios", 18, 0).as_deref(), Some("ios-18.0"));
        assert_eq!(apple_os_token("ios", 26, 1).as_deref(), Some("ios-26.1"));
        assert_eq!(
            apple_os_token("macos", 15, 2).as_deref(),
            Some("macos-15.2")
        );
        assert_eq!(
            apple_os_token("macos", 10, 15).as_deref(),
            Some("macos-10.15")
        );
    }

    #[test]
    fn apple_rejects_unknown_versions_and_other_platform_names() {
        for (major, minor) in [(0, 0), (-1, 0), (18, -1)] {
            assert_eq!(apple_os_token("ios", major, minor), None);
            assert_eq!(apple_os_token("macos", major, minor), None);
        }
        for platform in [
            "", "darwin", "tvos", "watchos", "visionos", "linux", "windows",
        ] {
            assert_eq!(apple_os_token(platform, 18, 0), None);
        }
    }

    #[test]
    fn glibc_formats_major_and_minor_versions() {
        assert_eq!(glibc_os_token("2.34").as_deref(), Some("glibc-2.34"));
        assert_eq!(glibc_os_token("2.40").as_deref(), Some("glibc-2.40"));
        assert_eq!(glibc_os_token("02.09").as_deref(), Some("glibc-2.9"));
    }

    #[test]
    fn glibc_rejects_missing_malformed_and_release_strings() {
        for value in [
            "",
            "2",
            "2.",
            ".34",
            "0.34",
            "2.-1",
            "+2.34",
            " 2.34",
            "2.34\n",
            "2.34-dev",
            "2.34.1",
            "6.1.17",
            "Ubuntu 24.04",
            "musl-1.2.5",
            "2.4294967296",
        ] {
            assert_eq!(glibc_os_token(value), None, "accepted {value:?}");
        }
    }

    #[test]
    fn native_runtime_token_matches_target() {
        let expected_prefix = if cfg!(target_os = "android") {
            Some("android-api-")
        } else if cfg!(target_os = "ios") {
            Some("ios-")
        } else if cfg!(target_os = "macos") {
            Some("macos-")
        } else if cfg!(all(target_os = "linux", target_env = "gnu")) {
            Some("glibc-")
        } else {
            None
        };

        let Some(prefix) = expected_prefix else {
            assert_eq!(runtime_os_version(), None);
            return;
        };
        let token = runtime_os_version().expect("supported native runtime must report its version");
        let version = token
            .strip_prefix(prefix)
            .expect("token must match target OS");
        let components: Vec<_> = version.split('.').collect();
        assert_eq!(
            components.len(),
            if prefix == "android-api-" { 1 } else { 2 }
        );
        for component in &components {
            assert!(!component.is_empty());
            assert!(component.bytes().all(|byte| byte.is_ascii_digit()));
        }
        assert!(components[0].parse::<u32>().unwrap() > 0);
    }
}
