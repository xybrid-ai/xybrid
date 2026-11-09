//! The `/v1/agent/connect` proof of work.
//!
//! Find a nonce, written as a canonical decimal `u64` (no sign, no leading
//! zeros), such that `SHA256(UTF8(challenge) || ":" || ASCII(nonce))` starts
//! with at least `difficulty` zero bits. The challenge is opaque to the client.
//! The platform names the rule `sha256-leading-zero-bits`.

use sha2::{Digest, Sha256};

/// The algorithm name the platform returns with each challenge.
pub(crate) const ALGORITHM: &str = "sha256-leading-zero-bits";
/// Refuse to grind harder than this: about 2^28 hashes, under a minute in a
/// release build. It is the platform's own upper bound (its settings clamp
/// the difficulty to 18..=28 bits); anything above is a misconfiguration or
/// a host trying to burn the caller's CPU.
pub(crate) const MAX_DIFFICULTY: u8 = 28;

/// Leading zero bits of `SHA256(challenge ":" nonce)`.
#[cfg(test)]
fn work_bits(challenge: &str, nonce: u64) -> u32 {
    let mut hasher = prefix(challenge);
    let mut buffer = [0u8; 20];
    hasher.update(decimal(nonce, &mut buffer));
    leading_zero_bits(&hasher.finalize())
}

/// The smallest nonce meeting `difficulty`, or `None` above [`MAX_DIFFICULTY`].
pub(crate) fn solve(challenge: &str, difficulty: u8) -> Option<u64> {
    if difficulty > MAX_DIFFICULTY {
        return None;
    }
    let prefix = prefix(challenge);
    let mut buffer = [0u8; 20];
    (0..u64::MAX).find(|nonce| {
        let mut hasher = prefix.clone();
        hasher.update(decimal(*nonce, &mut buffer));
        leading_zero_bits(&hasher.finalize()) >= u32::from(difficulty)
    })
}

fn prefix(challenge: &str) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(challenge.as_bytes());
    hasher.update(b":");
    hasher
}

/// `nonce` as canonical ASCII decimal, written into the tail of `buffer`.
fn decimal(mut nonce: u64, buffer: &mut [u8; 20]) -> &[u8] {
    let mut start = buffer.len();
    loop {
        start -= 1;
        buffer[start] = b'0' + (nonce % 10) as u8;
        nonce /= 10;
        if nonce == 0 {
            break;
        }
    }
    &buffer[start..]
}

fn leading_zero_bits(digest: &[u8]) -> u32 {
    let mut bits = 0;
    for byte in digest {
        if *byte == 0 {
            bits += 8;
        } else {
            return bits + byte.leading_zeros();
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smallest nonces and their exact bit counts, computed independently with
    /// Python's `hashlib` (not this code). 5, 11, 13 and 17 are not
    /// byte-aligned; 8, 13 and the `v1.` challenge match the vectors the
    /// platform publishes in `backend/src/agent/challenge.rs`.
    const VECTORS: &[(&str, u8, u64, u32)] = &[
        ("xybrid-pow-test-vector", 0, 0, 2),
        ("xybrid-pow-test-vector", 1, 0, 2),
        ("xybrid-pow-test-vector", 5, 14, 5),
        ("xybrid-pow-test-vector", 8, 47, 8),
        ("xybrid-pow-test-vector", 11, 307, 11),
        ("xybrid-pow-test-vector", 13, 2_900, 13),
        ("xybrid-pow-test-vector", 17, 92_240, 17),
        (
            "v1.1767225600.20.AAAAAAAAAAAAAAAAAAAAAA.vector",
            13,
            3_285,
            13,
        ),
        ("caf\u{e9}-\u{2603}", 9, 3_423, 11),
    ];

    #[test]
    fn matches_independent_vectors() {
        for &(challenge, difficulty, nonce, bits) in VECTORS {
            assert_eq!(
                solve(challenge, difficulty),
                Some(nonce),
                "{challenge} @ {difficulty}"
            );
            assert_eq!(work_bits(challenge, nonce), bits, "{challenge} @ {nonce}");
        }
    }

    #[test]
    fn difficulty_twenty_vector() {
        // Python: smallest nonce 440423 with 21 leading zero bits (also a
        // published platform vector). Checked, not solved, to keep debug
        // test runs fast.
        assert_eq!(work_bits("xybrid-pow-test-vector", 440_423), 21);
        assert!(work_bits("xybrid-pow-test-vector", 440_422) < 20);
    }

    #[test]
    fn nonces_are_canonical_decimal() {
        let mut buffer = [0u8; 20];
        assert_eq!(decimal(0, &mut buffer), b"0");
        assert_eq!(decimal(47, &mut buffer), b"47");
        assert_eq!(decimal(u64::MAX, &mut buffer), b"18446744073709551615");
        // Python: sha256("xybrid-pow-test-vector:18446744073709551615") has 4 zero bits.
        assert_eq!(work_bits("xybrid-pow-test-vector", u64::MAX), 4);
    }

    #[test]
    fn refuses_unreasonable_difficulty() {
        assert_eq!(solve("anything", MAX_DIFFICULTY + 1), None);
        assert_eq!(solve("anything", u8::MAX), None);
    }

    #[test]
    fn counts_bits_across_bytes() {
        assert_eq!(leading_zero_bits(&[0x00, 0x00, 0x80]), 16);
        assert_eq!(leading_zero_bits(&[0x00, 0x01]), 15);
        assert_eq!(leading_zero_bits(&[0x0f]), 4);
        assert_eq!(leading_zero_bits(&[0x00; 4]), 32);
    }
}
