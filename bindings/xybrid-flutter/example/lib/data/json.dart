Map<String, dynamic> _parseJson(String json) {
  // Simple JSON parsing - in production, use dart:convert
  return Map<String, dynamic>.from((const JsonCodec()).decode(json) as Map);
}

extension JsonStringExtension on String {
  Map<String, dynamic> tryJsonParse() => _parseJson(this);
}

// JSON codec for parsing
class JsonCodec {
  const JsonCodec();

  dynamic decode(String source) {
    return _parseValue(source.trim(), 0).$1;
  }

  (dynamic, int) _parseValue(String s, int i) {
    i = _skipWhitespace(s, i);
    if (i >= s.length) return (null, i);

    final c = s[i];
    if (c == '{') return _parseObject(s, i);
    if (c == '[') return _parseArray(s, i);
    if (c == '"') return _parseString(s, i);
    if (c == 't' || c == 'f') return _parseBool(s, i);
    if (c == 'n') return _parseNull(s, i);
    return _parseNumber(s, i);
  }

  (Map<String, dynamic>, int) _parseObject(String s, int i) {
    final map = <String, dynamic>{};
    i++; // skip '{'
    i = _skipWhitespace(s, i);
    if (s[i] == '}') return (map, i + 1);

    while (true) {
      i = _skipWhitespace(s, i);
      final (key, ni) = _parseString(s, i);
      i = _skipWhitespace(s, ni);
      i++; // skip ':'
      final (value, vi) = _parseValue(s, i);
      map[key] = value;
      i = _skipWhitespace(s, vi);
      if (s[i] == '}') return (map, i + 1);
      i++; // skip ','
    }
  }

  (List<dynamic>, int) _parseArray(String s, int i) {
    final list = <dynamic>[];
    i++; // skip '['
    i = _skipWhitespace(s, i);
    if (s[i] == ']') return (list, i + 1);

    while (true) {
      final (value, vi) = _parseValue(s, i);
      list.add(value);
      i = _skipWhitespace(s, vi);
      if (s[i] == ']') return (list, i + 1);
      i++; // skip ','
    }
  }

  (String, int) _parseString(String s, int i) {
    i++; // skip opening '"'
    final buf = StringBuffer();
    while (s[i] != '"') {
      if (s[i] == '\\') {
        i++;
        switch (s[i]) {
          case 'n':
            buf.write('\n');
            break;
          case 't':
            buf.write('\t');
            break;
          case 'r':
            buf.write('\r');
            break;
          case '"':
            buf.write('"');
            break;
          case '\\':
            buf.write('\\');
            break;
          default:
            buf.write(s[i]);
        }
      } else {
        buf.write(s[i]);
      }
      i++;
    }
    return (buf.toString(), i + 1);
  }

  (num, int) _parseNumber(String s, int i) {
    final start = i;
    if (s[i] == '-') i++;
    while (i < s.length &&
        (s[i].codeUnitAt(0) >= 48 && s[i].codeUnitAt(0) <= 57)) {
      i++;
    }
    if (i < s.length && s[i] == '.') {
      i++;
      while (i < s.length &&
          (s[i].codeUnitAt(0) >= 48 && s[i].codeUnitAt(0) <= 57)) {
        i++;
      }
    }
    if (i < s.length && (s[i] == 'e' || s[i] == 'E')) {
      i++;
      if (s[i] == '+' || s[i] == '-') i++;
      while (i < s.length &&
          (s[i].codeUnitAt(0) >= 48 && s[i].codeUnitAt(0) <= 57)) {
        i++;
      }
    }
    final str = s.substring(start, i);
    return (str.contains('.') ? double.parse(str) : int.parse(str), i);
  }

  (bool, int) _parseBool(String s, int i) {
    if (s.substring(i, i + 4) == 'true') return (true, i + 4);
    return (false, i + 5);
  }

  (dynamic, int) _parseNull(String s, int i) {
    return (null, i + 4);
  }

  int _skipWhitespace(String s, int i) {
    while (i < s.length &&
        (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) {
      i++;
    }
    return i;
  }
}
