pub(super) struct ParsedCall {
    pub(super) name: String,
    pub(super) arguments: String,
}

#[derive(Debug)]
pub(super) struct ParseError;

pub(super) struct Cursor<'a> {
    pub(super) input: &'a str,
    pub(super) pos: usize,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    pub(super) fn bump(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    pub(super) fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    pub(super) fn skip_ws(&mut self) {
        while matches!(self.peek_char(), Some(ch) if ch.is_whitespace()) {
            self.bump();
        }
    }

    pub(super) fn consume_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub(super) fn expect_char(&mut self, expected: char) -> Result<(), ParseError> {
        if self.consume_char(expected) {
            Ok(())
        } else {
            Err(ParseError)
        }
    }

    pub(super) fn expect_str(&mut self, expected: &str) -> Result<(), ParseError> {
        if self.starts_with(expected) {
            self.pos += expected.len();
            Ok(())
        } else {
            Err(ParseError)
        }
    }

    pub(super) fn starts_with(&self, expected: &str) -> bool {
        self.input[self.pos..].starts_with(expected)
    }

    pub(super) fn consume_digits(&mut self) -> usize {
        let mut count = 0;
        while matches!(self.peek_char(), Some(ch) if ch.is_ascii_digit()) {
            self.bump();
            count += 1;
        }
        count
    }

    pub(super) fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    pub(super) fn parse_identifier(&mut self, allow_dots: bool) -> Result<String, ParseError> {
        let mut ident = String::new();
        self.parse_identifier_segment(&mut ident)?;

        if allow_dots {
            while self.consume_char('.') {
                ident.push('.');
                self.parse_identifier_segment(&mut ident)?;
            }
        }

        Ok(ident)
    }

    pub(super) fn parse_identifier_segment(
        &mut self,
        ident: &mut String,
    ) -> Result<(), ParseError> {
        let Some(first) = self.peek_char() else {
            return Err(ParseError);
        };
        if !is_identifier_start(first) {
            return Err(ParseError);
        }
        ident.push(self.bump().ok_or(ParseError)?);

        while let Some(ch) = self.peek_char() {
            if !is_identifier_continue(ch) {
                break;
            }
            ident.push(self.bump().ok_or(ParseError)?);
        }
        Ok(())
    }
}

pub(super) fn is_identifier_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

pub(super) fn is_identifier_continue(ch: char) -> bool {
    is_identifier_start(ch) || ch.is_ascii_digit()
}
