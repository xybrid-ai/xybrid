// Widget tests for the Xybrid SDK example app.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:xybrid_example/main.dart';

void main() {
  testWidgets('SDK initialization screen shows loading state',
      (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const XybridExampleApp());

    // Verify that the loading indicator is shown.
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
    expect(find.text('Initializing Xybrid SDK...'), findsOneWidget);
  });

  testWidgets('App bar shows correct title', (WidgetTester tester) async {
    await tester.pumpWidget(const XybridExampleApp());

    expect(find.text('Xybrid SDK Example'), findsOneWidget);
  });
}
