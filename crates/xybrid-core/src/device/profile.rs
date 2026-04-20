use std::collections::BTreeMap;

use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DeviceProfile {
    pub chip_family: Option<String>,
    pub ram_gb: Option<u32>,
    pub os: Option<String>,
    pub os_version: Option<String>,
    pub kernel_version: Option<String>,
    pub arch: Option<String>,
    pub hostname: Option<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub custom: BTreeMap<String, String>,
}

impl DeviceProfile {
    pub fn detect() -> Self {
        let mut sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::new().with_frequency())
                .with_memory(MemoryRefreshKind::new().with_ram()),
        );
        sys.refresh_cpu_all();
        sys.refresh_memory();

        let chip_family = sys
            .cpus()
            .first()
            .and_then(|cpu| non_empty(cpu.brand().trim()));

        let total_memory = sys.total_memory();
        let ram_gb = if total_memory > 0 {
            let rounded = ((total_memory as f64) / 1_073_741_824.0).round() as u32;
            (rounded > 0).then_some(rounded)
        } else {
            None
        };

        Self {
            chip_family,
            ram_gb,
            os: System::name(),
            os_version: System::os_version(),
            kernel_version: System::kernel_version(),
            arch: Some(std::env::consts::ARCH.to_string()),
            hostname: None,
            custom: BTreeMap::new(),
        }
    }

    pub fn merged_with(mut self, other: Self) -> Self {
        if other.chip_family.is_some() {
            self.chip_family = other.chip_family;
        }
        if other.ram_gb.is_some() {
            self.ram_gb = other.ram_gb;
        }
        if other.os.is_some() {
            self.os = other.os;
        }
        if other.os_version.is_some() {
            self.os_version = other.os_version;
        }
        if other.kernel_version.is_some() {
            self.kernel_version = other.kernel_version;
        }
        if other.arch.is_some() {
            self.arch = other.arch;
        }
        if other.hostname.is_some() {
            self.hostname = other.hostname;
        }
        self.custom.extend(other.custom);
        self
    }

    /// True when every field is `None` and `custom` is empty. Used by the
    /// SDK to decide whether to emit a `device` object at all.
    pub fn is_empty(&self) -> bool {
        self.chip_family.is_none()
            && self.ram_gb.is_none()
            && self.os.is_none()
            && self.os_version.is_none()
            && self.kernel_version.is_none()
            && self.arch.is_none()
            && self.hostname.is_none()
            && self.custom.is_empty()
    }
}

fn non_empty(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceProfile;

    #[test]
    fn detect_returns_arch_and_os() {
        let detected = DeviceProfile::detect();

        assert!(detected.arch.is_some());
        assert!(detected.os.is_some());
        assert!(detected.hostname.is_none());
    }

    #[test]
    fn merged_with_applies_patch_then_override() {
        let auto = DeviceProfile {
            chip_family: Some("A".to_string()),
            ram_gb: Some(8),
            ..Default::default()
        };
        let patch = DeviceProfile {
            chip_family: Some("B".to_string()),
            ..Default::default()
        };
        let override_ = DeviceProfile {
            chip_family: Some("C".to_string()),
            os: Some("D".to_string()),
            ..Default::default()
        };

        let merged = auto.merged_with(patch).merged_with(override_);

        assert_eq!(merged.chip_family.as_deref(), Some("C"));
        assert_eq!(merged.ram_gb, Some(8));
        assert_eq!(merged.os.as_deref(), Some("D"));
    }
}
