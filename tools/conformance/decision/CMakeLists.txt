# Upstream decision oracle: builds the pinned llama.cpp (vendor/llama-cpp) with
# its own CMake, then links the oracle against upstream's `server-context`
# library, which holds tools/server/server-decision.cpp compiled exactly as
# llama-server compiles it. The stock llama-server comes from the same tree.
#
#   cmake -S tools/conformance/decision -B tools/conformance/decision/build -G Ninja
#   cmake --build tools/conformance/decision/build --target decision-oracle llama-server
#
# build.sh runs exactly that. Nothing here is part of xybrid's cargo or Bazel build.
cmake_minimum_required(VERSION 3.14...3.28)
project(xybrid_decision_oracle C CXX)

set(XYBRID_LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/llama-cpp"
    CACHE PATH "llama.cpp checkout to build (the pinned vendor/llama-cpp submodule)")
if (NOT EXISTS "${XYBRID_LLAMA_CPP_DIR}/tools/server/server-decision.cpp")
    message(FATAL_ERROR "no llama.cpp checkout at ${XYBRID_LLAMA_CPP_DIR}: run `git submodule update --init vendor/llama-cpp`")
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Upstream options: llama-server and the libraries it is made of, nothing else.
# No network at build time: the server is built without its web UI.
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI        OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_IS_DEV    OFF CACHE BOOL "" FORCE)

# The CPU kernels of xybrid's own natives (crates/llama-cpp-sys/build.rs): no
# -march=native, no OpenMP, and the x86-64-v3 baseline on x86_64.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    foreach (flag GGML_SSE42 GGML_AVX GGML_AVX2 GGML_BMI2 GGML_FMA GGML_F16C)
        set(${flag} ON CACHE BOOL "" FORCE)
    endforeach()
endif()

add_subdirectory("${XYBRID_LLAMA_CPP_DIR}" llama.cpp EXCLUDE_FROM_ALL)

add_executable(decision-oracle oracle.cpp)
set_target_properties(decision-oracle PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_features(decision-oracle PRIVATE cxx_std_17)
target_include_directories(decision-oracle PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${XYBRID_LLAMA_CPP_DIR}/tools/server"
    "${XYBRID_LLAMA_CPP_DIR}/tools/mtmd"
    "${XYBRID_LLAMA_CPP_DIR}")
target_link_libraries(decision-oracle PRIVATE server-context)
