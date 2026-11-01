// Host JSON probe (Dart): key order and number identity. Usage: dart run probe.dart "$(cat input.json)"
import 'dart:convert';
void main(List<String> a) {
  final o = jsonDecode(a[0]) as Map<String, dynamic>;
  print('order ${(o['questions'] as Map).keys.toList()}');
  print('state ${jsonEncode(o['state'])}');
  try { print('-inf ${jsonEncode(double.negativeInfinity)}'); } catch (e) { print('-inf throws ${e.runtimeType}'); }
}
