# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Build-time examples. Each example is a small, self-contained driver
# program that exercises one piece of the runtime + kernel corpus. Examples
# gracefully skip (exit 0) when the necessary backend isn't available so
# they're safe to register as CTest cases on CI hosts without a GPU.

if(NOT TARGET gpu_runtime)
    message(FATAL_ERROR
        "examples/ must be built alongside the runtime — set "
        "NAVATALA_GPU_BUILD_RUNTIME=ON.")
endif()

add_executable(axpy_example axpy_example.cpp)
target_link_libraries(axpy_example PRIVATE gpu_runtime)

if(GPU_RUNTIME_NAVATALA_WRAPPER)
    add_executable(wrapper_axpy_example wrapper_axpy_example.cpp)
    target_link_libraries(wrapper_axpy_example PRIVATE gpu_runtime)

    add_executable(wrapper_spmv_example wrapper_spmv_example.cpp)
    target_link_libraries(wrapper_spmv_example PRIVATE gpu_runtime)
endif()

set(NAVATALA_GPU_CUDA_SAMPLES_REGISTRY
    "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/src/internal/_registry_strings_cuda_samples.cpp")
if(EXISTS "${NAVATALA_GPU_CUDA_SAMPLES_REGISTRY}")
    add_executable(registry_lookup_example
        registry_lookup_example.cpp
        "${NAVATALA_GPU_CUDA_SAMPLES_REGISTRY}")
    target_include_directories(registry_lookup_example PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/src/internal")
    target_link_libraries(registry_lookup_example PRIVATE gpu_runtime)
endif()

# Register with CTest only when tests are enabled — keeps `ctest` from
# discovering an example case when the consumer asked for tests off.
# The example exits 0 when no real backend context is available, and it
# exercises the stub FFI in the default no-GPU build.
if(BUILD_TESTING OR GPU_RUNTIME_BUILD_TESTS)
    add_test(NAME axpy_example COMMAND axpy_example)
    if(GPU_RUNTIME_NAVATALA_WRAPPER)
        add_test(NAME wrapper_axpy_example COMMAND wrapper_axpy_example)
        add_test(NAME wrapper_spmv_example COMMAND wrapper_spmv_example)
    endif()
    if(TARGET registry_lookup_example)
        add_test(NAME registry_lookup_example COMMAND registry_lookup_example)
    endif()
endif()
