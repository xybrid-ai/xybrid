//! Keep the x86-64-v3 archive behind a baseline-safe CPU/OS check.
//!
//! The x86_64 OS ABI supplies v1. The masks below cover every additional
//! v2/v3 requirement in the x86-64 psABI, including OS-enabled XMM/YMM state.
//! Source: <https://gitlab.com/x86-psABIs/x86-64-ABI/-/blob/master/x86-64-ABI/low-level-sys-info.tex>
//! Use raw CPUID rather than feature-detection macros: those macros can return
//! compile-time `true` when Rust itself is built with a raised target CPU.

const LEAF1_REQUIRED: u32 = (1 << 0) // SSE3
    | (1 << 9) // SSSE3
    | (1 << 12) // FMA
    | (1 << 13) // CMPXCHG16B
    | (1 << 19) // SSE4.1
    | (1 << 20) // SSE4.2
    | (1 << 22) // MOVBE
    | (1 << 23) // POPCNT
    | (1 << 26) // XSAVE
    | (1 << 27) // OSXSAVE
    | (1 << 28) // AVX
    | (1 << 29); // F16C
const LEAF7_REQUIRED: u32 = (1 << 3) | (1 << 5) | (1 << 8); // BMI1, AVX2, BMI2
const EXTENDED1_REQUIRED: u32 = (1 << 0) | (1 << 5); // LAHF/SAHF, LZCNT
#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
const XSAVE_ENABLED: u32 = (1 << 26) | (1 << 27);
const AVX_STATE: u64 = (1 << 1) | (1 << 2); // XMM, YMM

#[derive(Clone, Copy, Default)]
struct X86Capabilities {
    leaf1_ecx: u32,
    leaf7_ebx: u32,
    extended1_ecx: u32,
    xcr0: u64,
}

impl X86Capabilities {
    fn supports_v3(self) -> bool {
        self.leaf1_ecx & LEAF1_REQUIRED == LEAF1_REQUIRED
            && self.leaf7_ebx & LEAF7_REQUIRED == LEAF7_REQUIRED
            && self.extended1_ecx & EXTENDED1_REQUIRED == EXTENDED1_REQUIRED
            && self.xcr0 & AVX_STATE == AVX_STATE
    }
}

#[derive(Clone, Copy, Default)]
struct CpuFacts {
    requires_v3: bool,
    x86: X86Capabilities,
}

impl CpuFacts {
    fn available(self) -> bool {
        !self.requires_v3 || self.x86.supports_v3()
    }
}

// XGETBV itself faults unless both the CPU and OS have enabled XSAVE.
#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn read_avx_state(leaf1_ecx: u32, read_xcr0: impl FnOnce() -> u64) -> u64 {
    if leaf1_ecx & XSAVE_ENABLED == XSAVE_ENABLED {
        read_xcr0()
    } else {
        0
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn detect() -> CpuFacts {
    use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    let max_basic = __cpuid(0).eax;
    let leaf1_ecx = if max_basic >= 1 { __cpuid(1).ecx } else { 0 };
    let leaf7_ebx = if max_basic >= 7 {
        __cpuid_count(7, 0).ebx
    } else {
        0
    };
    let extended1_ecx = if __cpuid(0x8000_0000).eax >= 0x8000_0001 {
        __cpuid(0x8000_0001).ecx
    } else {
        0
    };
    let xcr0 = read_avx_state(leaf1_ecx, || {
        // SAFETY: read_avx_state checks XSAVE and OSXSAVE before executing
        // XGETBV. Index 0 reads the architecturally defined XCR0 register.
        unsafe { _xgetbv(0) }
    });
    CpuFacts {
        requires_v3: true,
        x86: X86Capabilities {
            leaf1_ecx,
            leaf7_ebx,
            extended1_ecx,
            xcr0,
        },
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn detect() -> CpuFacts {
    CpuFacts::default()
}

pub(crate) fn available() -> bool {
    #[cfg(test)]
    if let Some(facts) = OVERRIDE.with(std::cell::Cell::get) {
        return facts.available();
    }
    detect().available()
}

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<CpuFacts>> = const { std::cell::Cell::new(None) };
}

/// A scoped per-thread override also exercises the real FFI entry guards on
/// non-x86 development hosts, without changing production detection.
#[cfg(test)]
pub(crate) fn without_v3<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(Option<CpuFacts>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let previous = OVERRIDE.with(|value| {
        value.replace(Some(CpuFacts {
            requires_v3: true,
            x86: X86Capabilities::default(),
        }))
    });
    let _restore = Restore(previous);
    run()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supported() -> X86Capabilities {
        X86Capabilities {
            leaf1_ecx: LEAF1_REQUIRED,
            leaf7_ebx: LEAF7_REQUIRED,
            extended1_ecx: EXTENDED1_REQUIRED,
            xcr0: AVX_STATE,
        }
    }

    #[test]
    fn exact_v3_floor_and_supersets_are_supported() {
        assert!(supported().supports_v3());
        assert!(X86Capabilities {
            leaf1_ecx: u32::MAX,
            leaf7_ebx: u32::MAX,
            extended1_ecx: u32::MAX,
            xcr0: u64::MAX,
        }
        .supports_v3());
    }

    #[test]
    fn each_missing_cpu_requirement_is_rejected() {
        for bit in 0..32 {
            for mask in 0..3 {
                let mut facts = supported();
                let (value, required) = match mask {
                    0 => (&mut facts.leaf1_ecx, LEAF1_REQUIRED),
                    1 => (&mut facts.leaf7_ebx, LEAF7_REQUIRED),
                    _ => (&mut facts.extended1_ecx, EXTENDED1_REQUIRED),
                };
                if required & (1 << bit) != 0 {
                    *value &= !(1 << bit);
                    assert!(!facts.supports_v3(), "mask {mask}, missing bit {bit}");
                }
            }
        }
    }

    #[test]
    fn avx_requires_both_xmm_and_ymm_os_state() {
        for xcr0 in [0, 1, 2, 4] {
            assert!(!X86Capabilities {
                xcr0,
                ..supported()
            }
            .supports_v3());
        }
    }

    #[test]
    fn xgetbv_is_never_executed_without_cpu_and_os_xsave() {
        for leaf1_ecx in [0, 1 << 26, 1 << 27] {
            assert_eq!(read_avx_state(leaf1_ecx, || panic!("unsafe XGETBV")), 0);
        }
        assert_eq!(read_avx_state(XSAVE_ENABLED, || AVX_STATE), AVX_STATE);
    }

    #[test]
    fn other_platforms_do_not_require_the_x86_floor() {
        assert!(CpuFacts::default().available());
        assert!(!CpuFacts {
            requires_v3: true,
            ..CpuFacts::default()
        }
        .available());
    }

    #[test]
    fn entrypoint_override_is_scoped() {
        let before = available();
        without_v3(|| {
            assert!(!available());
            without_v3(|| assert!(!available()));
            assert!(!available());
        });
        assert_eq!(available(), before);
    }
}
