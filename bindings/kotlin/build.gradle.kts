plugins {
    id("com.android.library")
    kotlin("android")
    id("com.vanniktech.maven.publish")
}

group = "ai.xybrid"
version = "0.2.0"

android {
    namespace = "ai.xybrid"
    compileSdk = 34

    defaultConfig {
        minSdk = 24

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            abiFilters += setOf("armeabi-v7a", "arm64-v8a", "x86_64")
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
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("libs")
        }
    }

}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("net.java.dev.jna:jna:5.13.0@aar")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}

mavenPublishing {
    publishToMavenCentral(com.vanniktech.maven.publish.SonatypeHost.CENTRAL_PORTAL)
    // Sign only when a GPG key is present. The release pipeline sets
    // ORG_GRADLE_PROJECT_signingInMemoryKey (→ the `signingInMemoryKey`
    // project property) from secrets.GPG_PRIVATE_KEY, so Maven Central
    // publishes stay signed. Local `publishToMavenLocal` (used to consume
    // the binding from working-tree source in the RN/Android examples) has
    // no key configured, so it skips signing instead of failing on a
    // missing .asc artifact.
    if (project.findProperty("signingInMemoryKey") != null) {
        signAllPublications()
    }

    coordinates("ai.xybrid", "xybrid-kotlin", version.toString())

    pom {
        name.set("Xybrid Kotlin SDK")
        description.set("On-device ML inference for Android")
        url.set("https://github.com/xybrid-ai/xybrid")

        licenses {
            license {
                name.set("Apache-2.0")
                url.set("https://www.apache.org/licenses/LICENSE-2.0")
            }
        }
        scm {
            url.set("https://github.com/xybrid-ai/xybrid")
            connection.set("scm:git:git://github.com/xybrid-ai/xybrid.git")
            developerConnection.set("scm:git:ssh://git@github.com/xybrid-ai/xybrid.git")
        }
        developers {
            developer {
                id.set("xybrid-ai")
                name.set("Xybrid AI")
            }
        }
    }
}
