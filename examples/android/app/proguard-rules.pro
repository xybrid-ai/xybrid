# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.kts.

# Keep Xybrid SDK classes
-keep class ai.xybrid.** { *; }

# Keep JNA classes used by UniFFI
-keep class com.sun.jna.** { *; }
-keep class * implements com.sun.jna.** { *; }
