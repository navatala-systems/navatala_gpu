cmake_minimum_required(VERSION 3.18)
project(navatala_ffi_installed_consumer LANGUAGES C)

find_package(gpu_runtime CONFIG REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_BINARY_DIR}/bin")
endforeach()

add_executable(navatala_ffi_installed_consumer main.c)
target_link_libraries(navatala_ffi_installed_consumer PRIVATE gpu_runtime::gpu_runtime)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(navatala_ffi_installed_consumer PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()
