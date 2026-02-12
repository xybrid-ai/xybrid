pluginManagement {
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

rootProject.name = "XybridExample"

// Include the app module
include(":app")

// Include Xybrid SDK as a local project module
include(":xybrid")
project(":xybrid").projectDir = file("../../bindings/kotlin")
