// Consumer smoke for the BAZEL-built AAR: a minimal real AGP app that consumes
// bindings/kotlin's Bazel output (copied into app/libs/ by the CI step) the way
// a Maven consumer would — proves the manifest parses, classes.jar dexes, and
// all three ABIs' jniLibs land in the APK. Versions mirror ../settings.gradle.kts.
pluginManagement {
    plugins {
        id("com.android.application") version "8.2.2"
        kotlin("android") version "1.9.22"
    }
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "xybrid-aar-consumer-smoke"
include(":app")
