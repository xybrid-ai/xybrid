# Xybrid Examples

This directory contains platform-specific example applications demonstrating how to use the Xybrid SDK.

## Directory Structure

```
examples/
├── flutter/    # Flutter example app (iOS, Android, macOS)
├── ios/        # Native iOS example (Swift)
├── android/    # Native Android example (Kotlin)
└── unity/      # Unity example project
```

## Flutter

The Flutter example demonstrates cross-platform usage of the Xybrid SDK using the Flutter binding.

**Requirements:**
- Flutter 3.19.0 or later
- Dart SDK 3.3.0 or later
- For iOS/macOS: Xcode with valid developer account
- For Android: Android Studio with NDK installed

**Getting Started:**
```bash
cd flutter
flutter pub get
flutter run
```

## iOS

The native iOS example shows how to integrate Xybrid directly into a Swift project using the Swift bindings.

**Requirements:**
- Xcode 15.0 or later
- iOS 15.0+ deployment target
- Valid Apple Developer account for device testing

## Android

The native Android example demonstrates Kotlin integration with the Xybrid SDK.

**Requirements:**
- Android Studio Hedgehog or later
- Minimum SDK: API 28 (Android 9.0)
- NDK installed via SDK Manager

## Unity

The Unity example shows how to use Xybrid in game development scenarios.

**Requirements:**
- Unity 2022.3 LTS or later
- Platform-specific build support modules installed

## Building Examples

Each example includes its own build instructions. See the README in each subdirectory for platform-specific details.

## Contributing Examples

When adding new examples:

1. Create the example in the appropriate platform directory
2. Include a README.md with setup and run instructions
3. Keep dependencies minimal and well-documented
4. Test on actual devices when possible
5. Follow the existing code style and patterns
