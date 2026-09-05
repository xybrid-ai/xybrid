plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "ai.xybrid.smoke"
    compileSdk = 34

    defaultConfig {
        applicationId = "ai.xybrid.smoke"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "0.0.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // The Bazel-built AAR, copied here by the smoke step (gitignored).
    implementation(files("libs/xybrid-kotlin.aar"))
    // The AAR's transitive runtime dep. A Maven consumer gets this from the
    // POM; a local-file AAR carries no metadata, so declare it explicitly.
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    // Match the binding's existing Android test framework/version.
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
}
