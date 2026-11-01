# The identity of a decision-oracle build, recomputed on every build (cmake -P).
#
# In:  LLAMA_DIR  the configured llama.cpp source directory (absolute, resolved)
#      TOOL_DIR   tools/conformance/decision
#      GIT        the git executable, or empty
#      PROFILE    oracle-build-profile.json, written at configure time
# Out: OUT_C      a C file defining xybrid_oracle_identity, linked into
#                 decision-oracle and llama-server
#      OUT_JSON   the same JSON, next to the executables
# Both are rewritten only when the identity changes.
#
# The commit is read from LLAMA_DIR only if LLAMA_DIR is the top of its own git
# checkout: an uninitialized submodule directory would otherwise report the
# enclosing repository's commit. The fingerprint hashes everything but the
# directory path, so two checkouts of the same sources built the same way
# share it.
cmake_minimum_required(VERSION 3.14)

function(json_string out value)
    string(REPLACE "\\" "\\\\" value "${value}")
    string(REPLACE "\"" "\\\"" value "${value}")
    string(REPLACE "\n" "\\n" value "${value}")
    string(REPLACE "\t" "\\t" value "${value}")
    set(${out} "\"${value}\"" PARENT_SCOPE)
endfunction()

# {"name": sha256 or null} for files under a directory, in the given order
function(hash_files out dir)
    set(json "")
    foreach (rel IN LISTS ARGN)
        if (EXISTS "${dir}/${rel}")
            file(SHA256 "${dir}/${rel}" digest)
            set(value "\"${digest}\"")
        else()
            set(value "null")
        endif()
        if (json)
            string(APPEND json ",")
        endif()
        string(APPEND json "\"${rel}\":${value}")
    endforeach()
    set(${out} "{${json}}" PARENT_SCOPE)
endfunction()

file(READ "${PROFILE}" profile)

set(checkout "none")
set(commit "null")
set(dirty "null")
if (GIT)
    execute_process(COMMAND "${GIT}" -C "${LLAMA_DIR}" rev-parse --show-toplevel
        OUTPUT_VARIABLE top RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if (rc EQUAL 0)
        get_filename_component(top "${top}" REALPATH)
        if (top STREQUAL LLAMA_DIR)
            execute_process(COMMAND "${GIT}" -C "${LLAMA_DIR}" rev-parse HEAD
                OUTPUT_VARIABLE head RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            execute_process(COMMAND "${GIT}" --no-optional-locks -C "${LLAMA_DIR}" status --porcelain=v1 --untracked-files=normal
                OUTPUT_VARIABLE status RESULT_VARIABLE rc_status ERROR_QUIET)
            if (rc EQUAL 0 AND rc_status EQUAL 0)
                set(checkout "git")
                set(commit "\"${head}\"")
                if (status STREQUAL "")
                    set(dirty "false")
                else()
                    set(dirty "true")
                endif()
            endif()
        else()
            set(checkout "inside another repository")
        endif()
    endif()
else()
    set(checkout "git unavailable")
endif()

hash_files(tool_files "${TOOL_DIR}" CMakeLists.txt identity.cmake oracle.cpp test_access.h)
hash_files(upstream_files "${LLAMA_DIR}"
    tools/server/server-decision.cpp tools/server/server-decision.h tools/server/server-context.cpp
    tools/server/server.cpp tools/server/server-common.cpp)

set(source "{\"checkout\":\"${checkout}\",\"commit\":${commit},\"dirty\":${dirty}}")
set(stable "{\"llama_cpp\":${source},\"tool_files\":${tool_files},\"upstream_files\":${upstream_files},\"build\":${profile}}")
string(SHA256 fingerprint "${stable}")
json_string(dir "${LLAMA_DIR}")
set(identity "{\"identity_version\":1,\"fingerprint\":\"${fingerprint}\",\"llama_cpp_dir\":${dir},\"llama_cpp\":${source},\"tool_files\":${tool_files},\"upstream_files\":${upstream_files},\"build\":${profile}}")

# a C string: escape what C reads specially
string(REPLACE "\\" "\\\\" literal "${identity}")
string(REPLACE "\"" "\\\"" literal "${literal}")
set(c_source "/* Written by tools/conformance/decision/identity.cmake: what this build was made from. */\n")
string(APPEND c_source "#if defined(__GNUC__) || defined(__clang__)\n__attribute__((used))\n#endif\n")
string(APPEND c_source "const char xybrid_oracle_identity[] = \"XYBRID_ORACLE_IDENTITY ${literal}\";\n")

foreach (pair "OUT_C;c_source" "OUT_JSON;identity")
    list(GET pair 0 path_var)
    list(GET pair 1 content_var)
    set(path "${${path_var}}")
    get_filename_component(parent "${path}" DIRECTORY)
    file(MAKE_DIRECTORY "${parent}")
    file(WRITE "${path}.tmp" "${${content_var}}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${path}.tmp" "${path}")
    file(REMOVE "${path}.tmp")
endforeach()
