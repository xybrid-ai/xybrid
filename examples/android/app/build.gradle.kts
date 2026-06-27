import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Resolve Xybrid config without committing it to the repo. Precedence:
//   1. -P<name>=... on the Gradle command line (CI / one-liner)
//   2. <name> environment variable
//   3. <localPropKey> in local.properties (gitignored — best for Android Studio)
// Unset resolves to "", which the SDK treats as anonymous / default platform.
val localProperties = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

fun resolveConfig(name: String, localPropKey: String): String =
    project.findProperty(name)?.toString()
        ?: System.getenv(name)
        ?: localProperties.getProperty(localPropKey)
        ?: ""

val xybridApiKey = resolveConfig("XYBRID_API_KEY", "xybrid.apiKey")
val xybridPlatformUrl = resolveConfig("XYBRID_PLATFORM_URL", "xybrid.platformUrl")

android {
    namespace = "ai.xybrid.example"
    compileSdk = 34

    defaultConfig {
        applicationId = "ai.xybrid.example"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Surfaced to the app as BuildConfig.XYBRID_API_KEY / XYBRID_PLATFORM_URL.
        buildConfigField("String", "XYBRID_API_KEY", "\"$xybridApiKey\"")
        buildConfigField("String", "XYBRID_PLATFORM_URL", "\"$xybridPlatformUrl\"")

        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    composeOptions {
        // Bumped from 1.4.8 → 1.5.10 to match Kotlin 1.9.22 (the version
        // bolt-generated bindings require for the `enum entries` API).
        // Compose Compiler/Kotlin compatibility table:
        // https://developer.android.com/jetpack/androidx/releases/compose-kotlin
        kotlinCompilerExtensionVersion = "1.5.10"
    }

    packagingOptions {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // Xybrid SDK (local project dependency)
    implementation(project(":xybrid"))

    // Kotlin coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // AndroidX Core
    implementation("androidx.core:core-ktx:1.10.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.1")
    implementation("androidx.activity:activity-compose:1.7.2")

    // Compose BOM (Bill of Materials) for consistent versioning
    implementation(platform("androidx.compose:compose-bom:2023.06.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation(platform("androidx.compose:compose-bom:2023.06.01"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
