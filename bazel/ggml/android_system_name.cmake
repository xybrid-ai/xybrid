# Included right after ggml's `project()` call (CMAKE_PROJECT_ggml_INCLUDE)
# by the Android arm64 arm of //:llama.
#
# rules_foreign_cc configures Android cross builds as CMAKE_SYSTEM_NAME=Linux
# with ANDROID=YES. ggml picks its GGML_CPU_ALL_VARIANTS list (and links
# libdl) by the system name, so a Linux name would build the desktop-Linux
# variants, whose int8-matmul builds all require SVE, which most Android SoCs
# do not expose. Inside ggml's scope only, name the platform the way the NDK
# toolchain file would, so ggml builds its Android variant set.
if(ANDROID)
    set(CMAKE_SYSTEM_NAME Android)
endif()
