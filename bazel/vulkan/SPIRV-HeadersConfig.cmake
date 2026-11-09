# Keep SPIR-V's header-only target on the staged SDK include tree. Importing
# the host /usr package would expose its libc headers to the hermetic compiler.
if(NOT TARGET SPIRV-Headers::SPIRV-Headers)
    add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)
    set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "/opt/xybrid-vulkan-include")
endif()
