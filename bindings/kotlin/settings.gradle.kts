pluginManagement {
    plugins {
        id("com.android.library") version "8.2.2"
        kotlin("android") version "1.9.22"
        id("com.vanniktech.maven.publish") version "0.30.0"
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

rootProject.name = "xybrid-kotlin"
