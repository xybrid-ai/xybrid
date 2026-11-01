//! How `finish`'s `f64`s are held to the fixtures'.
//!
//! The fixtures were observed in one immutable reference environment, the
//! oracle's (`oracle.json`'s `platform`): Linux on x86-64 with glibc 2.34, on
//! a CPU with FMA and AVX2, where glibc's `exp` runs its FMA code. `finish`
//! does upstream's operations in upstream's order, and Rust's `f64::exp` is
//! glibc's `exp` there, so there every `f64` must equal the fixtures' bit for
//! bit. Another `exp` (another libm, another glibc, glibc's SSE2 code) can
//! move last bits: elsewhere the `f64`s are measured and reported, never
//! asserted, until a tolerance measured on that platform is reviewed and
//! committed.
//!
//! The environment is detected at run time. `XYBRID_DECISION_F64_REFERENCE`
//! overrides the detection: `1` asserts bit equality (a pinned reference
//! container sets it), `0` only measures.
//!
//! What does not depend on `exp`'s last bits is exact everywhere: whether each
//! value is zero, subnormal or normal, and how the values of one answer
//! compare (ties and order); the callers hold keys, their order, the choice,
//! the score level, raw scores and usage exactly.

use std::fmt;
use std::io::Write as _;

use super::cases::{self, Fixture};

/// The reference environment: `oracle.json`'s `platform`, and the CPU
/// features glibc's `exp` dispatch reads.
const REFERENCE_OS: &str = "linux";
const REFERENCE_ARCH: &str = "x86_64";
const REFERENCE_LIBC: &str = "glibc 2.34";

/// The variable that overrides the detection.
const OVERRIDE: &str = "XYBRID_DECISION_F64_REFERENCE";

/// What `exp` depends on in this process.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Platform {
    os: &'static str,
    arch: &'static str,
    /// `"glibc <version>"` of the glibc loaded, on GNU/Linux.
    libc: Option<String>,
    /// FMA and AVX2 are usable: glibc's `exp` runs `__exp_fma`
    /// (`sysdeps/x86_64/fpu/multiarch/ifunc-fma4.h`).
    fma_avx2: bool,
}

impl Platform {
    fn detect() -> Self {
        Self {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            libc: glibc_version().map(|version| format!("glibc {version}")),
            fma_avx2: fma_avx2(),
        }
    }

    fn is_reference(&self) -> bool {
        self.os == REFERENCE_OS
            && self.arch == REFERENCE_ARCH
            && self.libc.as_deref() == Some(REFERENCE_LIBC)
            && self.fma_avx2
    }
}

impl fmt::Display for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.os, self.arch)?;
        if let Some(libc) = &self.libc {
            write!(f, ", {libc}")?;
        }
        if self.arch == "x86_64" {
            let fma = if self.fma_avx2 { "with" } else { "without" };
            write!(f, ", {fma} FMA and AVX2")?;
        }
        Ok(())
    }
}

/// The version of the glibc this process loaded.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn glibc_version() -> Option<String> {
    extern "C" {
        fn gnu_get_libc_version() -> *const std::ffi::c_char;
    }
    // SAFETY: glibc returns a pointer to a static, NUL-terminated string.
    let version = unsafe { std::ffi::CStr::from_ptr(gnu_get_libc_version()) };
    version.to_str().ok().map(str::to_owned)
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn glibc_version() -> Option<String> {
    None
}

#[cfg(target_arch = "x86_64")]
fn fma_avx2() -> bool {
    std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn fma_avx2() -> bool {
    false
}

/// Whether `f64`s are asserted bit for bit, and why.
fn mode(platform: &Platform, value: Option<&str>) -> (bool, &'static str) {
    match value {
        Some("1") => (true, "XYBRID_DECISION_F64_REFERENCE=1"),
        Some("0") => (false, "XYBRID_DECISION_F64_REFERENCE=0"),
        None | Some("") if platform.is_reference() => (true, "detected"),
        None | Some("") => (false, "not the reference environment"),
        Some(other) => panic!("{OVERRIDE} is {other:?}: set it to 1 or 0, or leave it unset"),
    }
}

/// Holds one test's answers to the fixtures' `f64`s, and reports.
pub(super) struct F64Parity {
    platform: Platform,
    exact: bool,
    why: &'static str,
    compared: usize,
    moved: usize,
    max_ulps: u64,
    max_error: f64,
}

impl F64Parity {
    /// The comparison for this environment.
    pub(super) fn new() -> Self {
        let platform = Platform::detect();
        let value = match std::env::var(OVERRIDE) {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => panic!("{OVERRIDE} is not Unicode"),
        };
        let (exact, why) = mode(&platform, value.as_deref());
        Self::with(platform, exact, why)
    }

    fn with(platform: Platform, exact: bool, why: &'static str) -> Self {
        Self {
            platform,
            exact,
            why,
            compared: 0,
            moved: 0,
            max_ulps: 0,
            max_error: 0.0,
        }
    }

    /// Holds one answer's `f64`s, each `(name, upstream's, ours)`: their
    /// classes, ties and order exactly, then each value bit for bit in the
    /// reference environment, or measured elsewhere.
    pub(super) fn answer(&mut self, what: &str, values: &[(String, f64, f64)]) {
        for (name, expected, actual) in values {
            assert!(actual.is_finite(), "{what}: {name} is {actual}");
            assert_eq!(
                class(*expected),
                class(*actual),
                "{what}: {name}: expected {expected:?}, got {actual:?}"
            );
        }
        for (index, (a, expected_a, actual_a)) in values.iter().enumerate() {
            for (b, expected_b, actual_b) in &values[index + 1..] {
                assert_eq!(
                    expected_a.partial_cmp(expected_b),
                    actual_a.partial_cmp(actual_b),
                    "{what}: {a} against {b}: expected {expected_a:?} and {expected_b:?}, \
                     got {actual_a:?} and {actual_b:?}"
                );
            }
        }
        for (name, expected, actual) in values {
            self.compared += 1;
            if expected.to_bits() == actual.to_bits() {
                continue;
            }
            assert!(
                !self.exact,
                "{what}: {name}: expected {expected:?}, got {actual:?} ({}; {})",
                self.platform, self.why
            );
            self.moved += 1;
            self.max_ulps = self.max_ulps.max(ulps(*expected, *actual));
            self.max_error = self.max_error.max((expected - actual).abs());
        }
    }

    /// The line [`report`](Self::report) prints.
    fn summary(&self, test: &str) -> String {
        if self.exact {
            format!(
                "{test}: f64 parity: {} values equal the fixtures bit for bit ({}; {})",
                self.compared, self.platform, self.why
            )
        } else {
            format!(
                "{test}: f64 parity UNMEASURED on {} ({}): not asserted until a reviewed \
                 tolerance measured on this platform is committed. Observed: {} of {} \
                 values moved, at most {} ulp and {:e} apart. Zeros, subnormals, ties and \
                 order, keys, choices, score levels, raw scores and usage were held exactly.",
                self.platform, self.why, self.moved, self.compared, self.max_ulps, self.max_error
            )
        }
    }

    /// Prints what `test` compared, past the test harness's capture (which
    /// drops a passing test's `eprintln!`): bit equality in the reference
    /// environment; elsewhere, that `f64` parity is unmeasured, and what
    /// moved.
    pub(super) fn report(&self, test: &str) {
        let _ = writeln!(std::io::stderr(), "{}", self.summary(test));
    }
}

/// Sign and class: zero, subnormal or normal.
fn class(value: f64) -> (bool, std::num::FpCategory) {
    (value.is_sign_negative(), value.classify())
}

/// How many `f64`s apart two finite `f64`s are (adjacent ones: 1).
fn ulps(a: f64, b: f64) -> u64 {
    fn ordinal(value: f64) -> i128 {
        let bits = value.to_bits();
        let magnitude = i128::from(bits & !(1 << 63));
        if bits >> 63 == 1 {
            -magnitude
        } else {
            magnitude
        }
    }
    u64::try_from((ordinal(a) - ordinal(b)).unsigned_abs())
        .expect("two finite f64s are fewer than 2^64 ulps apart")
}

#[test]
fn the_reference_is_the_oracles_platform() {
    let oracle = cases::fixture_file("oracle.json");
    let platform = |key| {
        cases::at(&oracle, &["platform", key])
            .and_then(Fixture::as_str)
            .unwrap_or_else(|| panic!("oracle.json has no platform.{key}"))
            .to_string()
    };
    assert_eq!(platform("system").to_lowercase(), REFERENCE_OS);
    assert_eq!(platform("machine"), REFERENCE_ARCH);
    assert_eq!(platform("libc"), REFERENCE_LIBC);
}

#[test]
fn only_the_reference_environment_or_the_override_asserts_bits() {
    let reference = Platform {
        os: "linux",
        arch: "x86_64",
        libc: Some("glibc 2.34".into()),
        fma_avx2: true,
    };
    assert!(reference.is_reference());
    let others = [
        Platform {
            libc: Some("glibc 2.39".into()),
            ..reference.clone()
        },
        Platform {
            fma_avx2: false,
            ..reference.clone()
        },
        Platform {
            libc: None,
            ..reference.clone()
        },
        Platform {
            os: "macos",
            arch: "aarch64",
            libc: None,
            fma_avx2: false,
        },
    ];
    for platform in &others {
        assert!(!platform.is_reference(), "{platform}");
    }
    assert_eq!(mode(&reference, None), (true, "detected"));
    assert!(mode(&reference, Some("")).0);
    assert!(!mode(&reference, Some("0")).0);
    for platform in &others {
        assert!(!mode(platform, None).0, "{platform}");
        assert!(mode(platform, Some("1")).0, "{platform}");
    }
    let typo = std::panic::catch_unwind(|| mode(&reference, Some("yes")));
    assert!(typo.is_err(), "an unknown value is refused");
}

#[test]
fn elsewhere_moved_values_are_measured_not_asserted() {
    let mut parity = F64Parity::with(
        Platform {
            os: "macos",
            arch: "aarch64",
            libc: None,
            fma_avx2: false,
        },
        false,
        "not the reference environment",
    );
    let next = |x: f64| f64::from_bits(x.to_bits() + 1);
    let values = [
        ("p(a)".to_string(), 0.75, next(next(0.75))),
        ("p(b)".to_string(), 0.25, 0.25),
        ("confidence".to_string(), 0.75, next(next(0.75))),
    ];
    parity.answer("case", &values);
    assert_eq!((parity.compared, parity.moved, parity.max_ulps), (3, 2, 2));
    assert_eq!(parity.max_error, next(next(0.75)) - 0.75);
    let summary = parity.summary("test");
    assert!(
        summary.contains("f64 parity UNMEASURED on macos aarch64"),
        "{summary}"
    );
    assert!(
        summary.contains("2 of 3 values moved, at most 2 ulp"),
        "{summary}"
    );
}

#[test]
fn in_the_reference_environment_one_moved_bit_fails() {
    let mut parity = F64Parity::with(
        Platform {
            os: "linux",
            arch: "x86_64",
            libc: Some("glibc 2.34".into()),
            fma_avx2: true,
        },
        true,
        "detected",
    );
    let values = [("p(a)".to_string(), 0.5, 0.5)];
    parity.answer("case", &values);
    assert!(parity
        .summary("test")
        .contains("1 values equal the fixtures bit for bit"));
    let moved = [(
        "p(a)".to_string(),
        0.5,
        f64::from_bits(0.5f64.to_bits() + 1),
    )];
    let failed = std::panic::catch_unwind(move || parity.answer("case", &moved));
    assert!(failed.is_err());
}

/// The checks the old tolerance waved through (any two values below
/// `1e-290`) fail on every platform, measured or not.
#[test]
fn zeros_subnormals_ties_and_order_are_exact_everywhere() {
    let measured = || {
        F64Parity::with(
            Platform {
                os: "windows",
                arch: "x86_64",
                libc: None,
                fma_avx2: true,
            },
            false,
            "not the reference environment",
        )
    };
    let fails = |values: Vec<(String, f64, f64)>| {
        let mut parity = measured();
        std::panic::catch_unwind(move || parity.answer("case", &values)).is_err()
    };
    let next = |x: f64| f64::from_bits(x.to_bits() + 1);
    let named = |values: &[(f64, f64)]| -> Vec<(String, f64, f64)> {
        values
            .iter()
            .enumerate()
            .map(|(index, &(expected, actual))| (format!("v{index}"), expected, actual))
            .collect()
    };
    // Underflow: zero against a tiny value, either way, and a subnormal
    // against a normal.
    assert!(fails(named(&[(0.0, 1e-291)])));
    assert!(fails(named(&[(1e-300, 0.0)])));
    assert!(fails(named(&[(0.0, -0.0)])));
    assert!(fails(named(&[(5e-324, 2.3e-308)])));
    // A tie that breaks, and an order that flips or collapses.
    assert!(fails(named(&[(0.5, 0.5), (0.5, next(0.5))])));
    assert!(fails(named(&[(0.25, 0.3), (0.3, 0.25)])));
    assert!(fails(named(&[(0.25, 0.3), (0.3, 0.3)])));
    // Moved last bits that keep all of that are only measured.
    assert!(!fails(named(&[(1e-300, next(1e-300)), (0.5, 0.5)])));
    // Non-finite never passes.
    assert!(fails(named(&[(0.5, f64::NAN)])));
}

#[test]
fn ulps_count_steps_across_zero() {
    let one = f64::from_bits(1);
    assert_eq!(ulps(1.0, 1.0), 0);
    assert_eq!(ulps(1.0, f64::from_bits(1.0f64.to_bits() + 3)), 3);
    assert_eq!(ulps(one, -one), 2);
    assert_eq!(ulps(f64::MAX, -f64::MAX), 2 * f64::MAX.to_bits());
}
