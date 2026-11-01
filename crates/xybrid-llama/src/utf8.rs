//! UTF-8 text from token pieces.
//!
//! A token's piece is raw bytes, and a byte-level vocabulary can split one
//! character across tokens: qwen3 writes "🙂" as several tokens, none of them
//! UTF-8 on its own. Decoding each piece by itself dropped such characters;
//! [`Utf8Decoder`] carries the leading bytes of an unfinished character over
//! to the next piece instead. The streaming callback and
//! [`crate::LlamaModel::detokenize`] decode the same way, so they see the same
//! text for every real vocabulary. They do not read the same bytes by
//! construction, though: the stream gets each piece through a 1023-byte,
//! NUL-terminated callback buffer (see `streaming_trampoline`), and
//! `detokenize` reads pieces whole.

use std::borrow::Cow;

/// Incremental UTF-8 decoder for a stream of token pieces.
///
/// [`Self::push`] returns every character the bytes so far complete. Bytes
/// that cannot be UTF-8, whatever follows them, become U+FFFD exactly as
/// [`String::from_utf8_lossy`] replaces them. The leading bytes of a
/// character the next piece may finish wait inside the decoder, and are
/// dropped by [`Self::discard_unfinished`] when the stream ends: an unfinished
/// character has no text. Pushing a byte string piece by piece therefore
/// yields the same text as pushing it whole.
#[derive(Debug, Default)]
pub(crate) struct Utf8Decoder {
    /// Leading bytes of a character whose remaining bytes have not arrived
    /// (at most three).
    unfinished: Vec<u8>,
}

impl Utf8Decoder {
    /// Decode `piece` after whatever an earlier piece left unfinished.
    pub(crate) fn push<'p>(&mut self, piece: &'p [u8]) -> Cow<'p, str> {
        if self.unfinished.is_empty() {
            if let Ok(text) = std::str::from_utf8(piece) {
                return Cow::Borrowed(text);
            }
        }

        self.unfinished.extend_from_slice(piece);
        let mut text = String::with_capacity(self.unfinished.len());
        let mut rest = self.unfinished.as_slice();
        loop {
            match std::str::from_utf8(rest) {
                Ok(valid) => {
                    text.push_str(valid);
                    rest = &[];
                    break;
                }
                Err(error) => {
                    let (valid, after) = rest.split_at(error.valid_up_to());
                    text.push_str(
                        std::str::from_utf8(valid).expect("bytes before `valid_up_to` are UTF-8"),
                    );
                    match error.error_len() {
                        // Bytes that cannot be UTF-8, whatever follows them.
                        Some(invalid) => {
                            text.push(char::REPLACEMENT_CHARACTER);
                            rest = &after[invalid..];
                        }
                        // The leading bytes of a character the next piece may
                        // finish: they wait for it.
                        None => {
                            rest = after;
                            break;
                        }
                    }
                }
            }
        }
        let decoded = self.unfinished.len() - rest.len();
        self.unfinished.drain(..decoded);
        Cow::Owned(text)
    }

    /// Drop the leading bytes of an unfinished character at the end of a
    /// stream, returning how many there were.
    pub(crate) fn discard_unfinished(&mut self) -> usize {
        let count = self.unfinished.len();
        self.unfinished.clear();
        count
    }
}

#[cfg(test)]
mod tests {
    use super::Utf8Decoder;
    use std::borrow::Cow;

    /// What each push returns when `pieces` arrive in order.
    fn decode(pieces: &[&[u8]]) -> Vec<String> {
        let mut decoder = Utf8Decoder::default();
        pieces
            .iter()
            .map(|piece| decoder.push(piece).into_owned())
            .collect()
    }

    #[test]
    fn a_complete_piece_is_borrowed_as_is() {
        let mut decoder = Utf8Decoder::default();
        assert!(matches!(
            decoder.push("ok 🙂".as_bytes()),
            Cow::Borrowed("ok 🙂")
        ));
    }

    /// A byte-level vocabulary splits "🙂" (four bytes) and each of "日本語"
    /// (three bytes apiece) across tokens; no part is UTF-8 on its own.
    #[test]
    fn a_character_split_across_pieces_arrives_with_its_last_byte() {
        assert_eq!(decode(&[b"\xF0\x9F", b"\x99\x82 ok"]), ["", "🙂 ok"]);
        assert_eq!(
            decode(&[b"\xF0", b"\x9F", b"\x99", b"\x82"]),
            ["", "", "", "🙂"]
        );
        assert_eq!(
            decode(&[b"\xE6", b"\x97\xA5\xE6", b"\x9C\xAC\xE8\xAA", b"\x9E"]),
            ["", "日", "本", "語"]
        );
        assert_eq!(decode(&[b"ok \xF0\x9F", b"\x99\x82!"]), ["ok ", "🙂!"]);
    }

    #[test]
    fn bytes_that_cannot_be_utf8_become_replacement_characters() {
        assert_eq!(decode(&[b"a\x80b"]), ["a\u{FFFD}b"]);
        // Leading bytes the next piece does not continue.
        assert_eq!(decode(&[b"\xF0\x9F", b"x"]), ["", "\u{FFFD}x"]);
    }

    /// The streamed text adds up to the text decoded at once, wherever the
    /// token boundaries fall.
    #[test]
    fn piece_boundaries_never_change_the_text() {
        let mut bytes = "a🙂日本é".as_bytes().to_vec();
        // An invalid byte, then the start of a character that never ends.
        bytes.extend_from_slice(b"\x80z\xF0\x9F");
        let whole = Utf8Decoder::default().push(&bytes).into_owned();
        assert_eq!(whole, "a🙂日本é\u{FFFD}z");

        for first in 0..=bytes.len() {
            for second in first..=bytes.len() {
                let pieces = [&bytes[..first], &bytes[first..second], &bytes[second..]];
                assert_eq!(
                    decode(&pieces).concat(),
                    whole,
                    "split at {first} and {second}"
                );
            }
        }
    }

    #[test]
    fn an_unfinished_character_at_the_end_has_no_text() {
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.push(b"ok \xE6\x97"), "ok ");
        assert_eq!(decoder.discard_unfinished(), 2);
        // Nothing carries over: a lone continuation byte is not UTF-8.
        assert_eq!(decoder.push(b"\xA5"), "\u{FFFD}");
        assert_eq!(decoder.discard_unfinished(), 0);
    }
}
