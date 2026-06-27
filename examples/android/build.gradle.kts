// Top-level build file for Xybrid Example Android app
plugins {
    id("com.android.application") version "8.13.2" apply false
    id("com.android.library") version "8.13.2" apply false
    // Bumped from 1.8.22 to 1.9.22 so bolt-emitted Kotlin (uses the
    // `enum entries` API, stabilized in 1.9) compiles. Matches the
    // version already pinned in bindings/kotlin/settings.gradle.kts.
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
    id("com.vanniktech.maven.publish") version "0.30.0" apply false
}
