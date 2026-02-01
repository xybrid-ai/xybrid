String? _extractFileName(String fullFile) {
  final match = RegExp(r'name:\s*"([^"]+)"').firstMatch(fullFile);
  return match?.group(1);
}

extension StringFileNameExtension on String {
  String? noExtension() {
    return _extractFileName(this);
  }
}
