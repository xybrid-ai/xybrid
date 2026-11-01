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
get_filename_component(xybrid_llama_dir "${XYBRID_LLAMA_CPP_DIR}" REALPATH)
if (NOT EXISTS "${xybrid_llama_dir}/tools/server/server-decision.cpp")
    message(FATAL_ERROR "no llama.cpp checkout at ${xybrid_llama_dir}: run `git submodule update --init vendor/llama-cpp`")
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

# The cpu-* settings mean the ggml CPU backend and nothing else: every other
# backend off, and the CPU backend's alternative matmul libraries (BLAS,
# Accelerate, KleidiAI) off too, since they change the numerics. The CPU
# kernels are those of xybrid's own natives (crates/llama-cpp-sys/build.rs): no
# -march=native, no OpenMP, the x86-64-v3 baseline on x86_64. A value set
# otherwise, on the command line or in a reused cache, stops the configuration
# rather than being overridden in silence.
set(xybrid_oracle_off
    GGML_BLAS GGML_ACCELERATE GGML_CPU_KLEIDIAI GGML_CANN GGML_CUDA GGML_ET GGML_HIP GGML_METAL GGML_MUSA
    GGML_RPC GGML_VIRTGPU GGML_VIRTGPU_BACKEND GGML_SYCL GGML_VULKAN GGML_WEBGPU GGML_ZDNN GGML_OPENCL
    GGML_HEXAGON GGML_ZENDNN GGML_OPENVINO GGML_BACKEND_DL GGML_CPU_ALL_VARIANTS GGML_NATIVE GGML_OPENMP)
set(xybrid_oracle_on GGML_CPU)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    list(APPEND xybrid_oracle_on GGML_SSE42 GGML_AVX GGML_AVX2 GGML_BMI2 GGML_FMA GGML_F16C)
endif()
foreach (opt IN LISTS xybrid_oracle_off)
    if (DEFINED ${opt} AND ${opt})
        message(FATAL_ERROR "${opt}=${${opt}}: the decision oracle builds the ggml CPU backend only. "
                            "Reconfigure with -D${opt}=OFF or start from an empty build directory.")
    endif()
    set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
foreach (opt IN LISTS xybrid_oracle_on)
    if (DEFINED ${opt} AND NOT ${opt})
        message(FATAL_ERROR "${opt}=${${opt}}: the decision oracle builds the CPU kernels of xybrid's natives. "
                            "Reconfigure with -D${opt}=ON or start from an empty build directory.")
    endif()
    set(${opt} ON CACHE BOOL "" FORCE)
endforeach()

add_subdirectory("${xybrid_llama_dir}" llama.cpp EXCLUDE_FROM_ALL)

# Whatever upstream's defaults do, no backend but the CPU one may exist.
foreach (backend blas cann cuda et hip metal musa rpc virtgpu sycl vulkan webgpu zdnn opencl hexagon zendnn openvino)
    if (TARGET ggml-${backend})
        message(FATAL_ERROR "ggml-${backend} is configured: the decision oracle builds the ggml CPU backend only")
    endif()
endforeach()
if (NOT TARGET ggml-cpu)
    message(FATAL_ERROR "ggml-cpu is not configured")
endif()

# The build profile: everything here that changes the code the compiler emits.
# identity.cmake folds it into the identity stamped into both executables.
function(xybrid_json_string out value)
    string(REPLACE "\\" "\\\\" value "${value}")
    string(REPLACE "\"" "\\\"" value "${value}")
    string(REPLACE "\n" "\\n" value "${value}")
    string(REPLACE "\t" "\\t" value "${value}")
    set(${out} "\"${value}\"" PARENT_SCOPE)
endfunction()
set(xybrid_profile_options ${xybrid_oracle_off} ${xybrid_oracle_on}
    BUILD_SHARED_LIBS GGML_LLAMAFILE GGML_CPU_REPACK GGML_CPU_HBM GGML_CPU_ARM_ARCH GGML_AVX_VNNI GGML_AVX512
    GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16 GGML_LTO
    GGML_STATIC GGML_GPROF GGML_SCHED_NO_REALLOC GGML_SANITIZE_THREAD GGML_SANITIZE_ADDRESS GGML_SANITIZE_UNDEFINED
    LLAMA_SANITIZE_THREAD LLAMA_SANITIZE_ADDRESS LLAMA_SANITIZE_UNDEFINED LLAMA_LLGUIDANCE)
list(SORT xybrid_profile_options)
list(REMOVE_DUPLICATES xybrid_profile_options)
set(options_json "")
foreach (opt IN LISTS xybrid_profile_options)
    xybrid_json_string(key "${opt}")
    xybrid_json_string(value "${${opt}}")
    if (options_json)
        string(APPEND options_json ",")
    endif()
    string(APPEND options_json "${key}:${value}")
endforeach()
string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_upper)
xybrid_json_string(j_system "${CMAKE_SYSTEM_NAME}")
xybrid_json_string(j_processor "${CMAKE_SYSTEM_PROCESSOR}")
xybrid_json_string(j_build_type "${CMAKE_BUILD_TYPE}")
xybrid_json_string(j_c_compiler "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
xybrid_json_string(j_cxx_compiler "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
xybrid_json_string(j_c_flags "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${build_type_upper}}")
xybrid_json_string(j_cxx_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_type_upper}}")
xybrid_json_string(j_linker_flags "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${build_type_upper}}")
set(profile_json "{\"system\":${j_system},\"processor\":${j_processor},\"build_type\":${j_build_type},\"c_compiler\":${j_c_compiler},\"cxx_compiler\":${j_cxx_compiler},\"c_flags\":${j_c_flags},\"cxx_flags\":${j_cxx_flags},\"exe_linker_flags\":${j_linker_flags},\"backends\":[\"cpu\"],\"options\":{${options_json}}}")
set(xybrid_profile "${CMAKE_BINARY_DIR}/oracle-build-profile.json")
file(WRITE "${xybrid_profile}.tmp" "${profile_json}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${xybrid_profile}.tmp" "${xybrid_profile}")

# The identity of a build: the llama.cpp commit it compiles (and whether that
# checkout is clean), the hashes of the oracle's sources and of the upstream
# files it runs or mirrors, and the build profile. identity.cmake recomputes it
# on every build and stamps it into decision-oracle and llama-server alike, so
# oracle.py reads what the executables were built from, not what the checkout
# holds now. `decision-oracle --identity` prints it.
find_package(Git QUIET)
set(xybrid_identity_c "${CMAKE_BINARY_DIR}/oracle_identity.c")
if (NOT EXISTS "${xybrid_identity_c}")
    file(WRITE "${xybrid_identity_c}" "const char xybrid_oracle_identity[] = \"\";\n")
endif()
add_custom_target(oracle-identity
    COMMAND "${CMAKE_COMMAND}"
        "-DLLAMA_DIR=${xybrid_llama_dir}"
        "-DTOOL_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DGIT=${GIT_EXECUTABLE}"
        "-DPROFILE=${xybrid_profile}"
        "-DOUT_C=${xybrid_identity_c}"
        "-DOUT_JSON=${CMAKE_BINARY_DIR}/bin/oracle-identity.json"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/identity.cmake"
    BYPRODUCTS "${xybrid_identity_c}" "${CMAKE_BINARY_DIR}/bin/oracle-identity.json"
    COMMENT "Stamping the oracle identity"
    VERBATIM)

add_executable(decision-oracle oracle.cpp "${xybrid_identity_c}")
add_dependencies(decision-oracle oracle-identity)
set_target_properties(decision-oracle PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_features(decision-oracle PRIVATE cxx_std_17)
target_include_directories(decision-oracle PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${xybrid_llama_dir}/tools/server"
    "${xybrid_llama_dir}/tools/mtmd"
    "${xybrid_llama_dir}")
target_link_libraries(decision-oracle PRIVATE server-context)

# The stock llama-server carries the same identity: an inert data object linked
# into it, so the server that answers the black-box check names its own build.
target_sources(llama-server PRIVATE "${xybrid_identity_c}")
add_dependencies(llama-server oracle-identity)
