# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Navatala GPU — runtime library build.
#
# Builds the cross-platform GPU runtime against any subset of the supported
# backends. Each backend is opt-out via cache variable; missing toolkits are
# silently skipped at configure time.

cmake_minimum_required(VERSION 3.18)
project(gpu_runtime VERSION 0.1.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
endif()

# ============================================================================
# Options
# ============================================================================

option(GPU_RUNTIME_USE_CUDA   "Enable CUDA backend"              ON)
option(GPU_RUNTIME_USE_HIP    "Enable HIP backend"               ON)
option(GPU_RUNTIME_USE_VULKAN "Enable Vulkan compute backend"    ON)
option(GPU_RUNTIME_USE_OPENCL "Enable OpenCL backend"            ON)
option(GPU_RUNTIME_USE_METAL  "Enable Metal backend (macOS/iOS)" OFF)
option(GPU_RUNTIME_NAVATALA_WRAPPER "Build the public Navatala C FFI wrapper" ON)
option(GPU_RUNTIME_NAVATALA_FFI_STUB "Use Navatala stub FFI implementation (no GPU required)" ON)
option(GPU_RUNTIME_HIDE_INTERNAL_SYMBOLS "Hide internal C++ runtime symbols on ELF shared-library builds" ON)
option(GPU_RUNTIME_INSTALL_STANDALONE_ARTIFACTS
       "Install runtime headers/library for standalone CMake consumers" ON)

# Test compilation is gated on GPU_RUNTIME_BUILD_TESTS. When this library is
# built as part of the navatala_gpu top-level project, the parent CMake
# forwards NAVATALA_GPU_BUILD_TESTS into this variable so a single
# `-DNAVATALA_GPU_BUILD_TESTS=OFF` switch suppresses tests across the tree.
# Default ON to match BUILD_TESTING's default behaviour for standalone builds.
option(GPU_RUNTIME_BUILD_TESTS "Build the runtime test suite" ON)

# ============================================================================
# Backend probing
# ============================================================================

set(GPU_RUNTIME_HAVE_CUDA   0)
set(GPU_RUNTIME_HAVE_HIP    0)
set(GPU_RUNTIME_HAVE_ROCBLAS 0)
set(GPU_RUNTIME_HAVE_VULKAN 0)
set(GPU_RUNTIME_HAVE_OPENCL 0)
set(GPU_RUNTIME_HAVE_METAL  0)

set(BACKEND_SOURCES "")
set(BACKEND_LIBS    "")
set(BACKEND_INCLUDE_DIRS "")
set(GPU_LIBRARY_OPS_SOURCES "")

if(GPU_RUNTIME_USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(GPU_RUNTIME_HAVE_CUDA 1)
        list(APPEND BACKEND_SOURCES src/backend_cuda.cpp)
        list(APPEND BACKEND_LIBS CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
        message(STATUS "[Navatala GPU] CUDA backend enabled")
    else()
        message(STATUS "[Navatala GPU] CUDA backend requested but CUDAToolkit not found — skipping")
    endif()
endif()

if(GPU_RUNTIME_USE_HIP)
    find_package(hip QUIET)
    if(hip_FOUND)
        set(GPU_RUNTIME_HAVE_HIP 1)
        list(APPEND BACKEND_SOURCES src/backend_hip.cpp)
        list(APPEND BACKEND_LIBS hip::host hip::amdhip64)
        find_library(HIPRTC_LIBRARY NAMES hiprtc HINTS ${hip_INCLUDE_DIRS}/../lib)
        if(HIPRTC_LIBRARY)
            list(APPEND BACKEND_LIBS ${HIPRTC_LIBRARY})
        endif()
        message(STATUS "[Navatala GPU] HIP backend enabled")

        find_package(rocblas CONFIG QUIET HINTS /opt/rocm PATHS /opt/rocm)
        if(TARGET roc::rocblas)
            set(GPU_RUNTIME_HAVE_ROCBLAS 1)
            list(APPEND GPU_LIBRARY_OPS_SOURCES src/gpu_library_hip_blas.cpp)
            list(APPEND BACKEND_LIBS roc::rocblas)
            message(STATUS "[Navatala GPU] rocBLAS LibraryOps enabled")
        else()
            find_path(ROCBLAS_INCLUDE_DIR
                NAMES rocblas/rocblas.h rocblas.h
                HINTS /opt/rocm/include)
            find_library(ROCBLAS_LIBRARY
                NAMES rocblas
                HINTS /opt/rocm/lib /opt/rocm/lib64)
            if(ROCBLAS_LIBRARY AND ROCBLAS_INCLUDE_DIR)
                set(GPU_RUNTIME_HAVE_ROCBLAS 1)
                list(APPEND GPU_LIBRARY_OPS_SOURCES src/gpu_library_hip_blas.cpp)
                list(APPEND BACKEND_LIBS "${ROCBLAS_LIBRARY}")
                list(APPEND BACKEND_INCLUDE_DIRS "${ROCBLAS_INCLUDE_DIR}")
                message(STATUS "[Navatala GPU] rocBLAS LibraryOps enabled (manual probe)")
            else()
                message(STATUS "[Navatala GPU] rocBLAS not found; HIP LibraryOps BLAS dispatch disabled")
            endif()
        endif()
    else()
        message(STATUS "[Navatala GPU] HIP backend requested but ROCm not found — skipping")
    endif()
endif()

if(GPU_RUNTIME_USE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        set(GPU_RUNTIME_HAVE_VULKAN 1)
        list(APPEND BACKEND_SOURCES src/backend_vulkan.cpp)
        list(APPEND BACKEND_LIBS Vulkan::Vulkan)
        message(STATUS "[Navatala GPU] Vulkan backend enabled")
    else()
        message(STATUS "[Navatala GPU] Vulkan backend requested but Vulkan SDK not found — skipping")
    endif()
endif()

if(GPU_RUNTIME_USE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(GPU_RUNTIME_HAVE_OPENCL 1)
        list(APPEND BACKEND_SOURCES src/backend_opencl.cpp)
        list(APPEND BACKEND_LIBS OpenCL::OpenCL)
        message(STATUS "[Navatala GPU] OpenCL backend enabled")
    else()
        message(STATUS "[Navatala GPU] OpenCL backend requested but OpenCL not found — skipping")
    endif()
endif()

if(GPU_RUNTIME_USE_METAL AND APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        set(GPU_RUNTIME_HAVE_METAL 1)
        list(APPEND BACKEND_SOURCES src/backend_metal.mm)
        list(APPEND BACKEND_LIBS ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
        message(STATUS "[Navatala GPU] Metal backend enabled")
    endif()
endif()

# ============================================================================
# Core sources
# ============================================================================

set(CORE_SOURCES
    src/device_factory.cpp
    src/memory_resource.cpp
    src/pool_memory_resource.cpp
    src/binning_memory_resource.cpp
    src/program_cache.cpp
    src/stream_pool.cpp
    src/resource_adaptors.cpp
    src/public_runtime_stubs.cpp
    src/runtime_profile.cpp
    ${GPU_LIBRARY_OPS_SOURCES}
)

set(NAVATALA_FFI_SOURCES "")
set(NAVATALA_CPP_WRAPPER_SOURCES
    src/navatala_backend.cpp
    src/navatala_resource.cpp
    src/navatala_memory.cpp
    src/navatala_buffer.cpp
    src/navatala_index.cpp
    src/navatala_neural_operators.cpp
    src/navatala_profiling.cpp
)
if(GPU_RUNTIME_NAVATALA_WRAPPER)
    if(GPU_RUNTIME_NAVATALA_FFI_STUB)
        set(NAVATALA_FFI_SOURCES src/navatala_ffi_stub.cpp)
        message(STATUS "[Navatala GPU] public FFI enabled (stub mode)")
    else()
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/navatala_ffi_real.cpp")
            message(FATAL_ERROR
                "[Navatala GPU] public FFI real mode requested "
                "(GPU_RUNTIME_NAVATALA_FFI_STUB=OFF), but "
                "src/navatala_ffi_real.cpp does not exist")
        endif()
        set(NAVATALA_FFI_SOURCES src/navatala_ffi_real.cpp)
        message(STATUS "[Navatala GPU] public FFI enabled (real mode)")
    endif()
else()
    message(STATUS "[Navatala GPU] public FFI disabled")
endif()
if(GPU_RUNTIME_NAVATALA_WRAPPER)
    foreach(wrapper_src IN LISTS NAVATALA_CPP_WRAPPER_SOURCES)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${wrapper_src}")
            list(APPEND NAVATALA_FFI_SOURCES ${wrapper_src})
        endif()
    endforeach()
endif()

set(GPU_RUNTIME_HAVE_HIP_TRANSFORMER_REGISTRY 0)
if(GPU_RUNTIME_NAVATALA_WRAPPER AND NOT GPU_RUNTIME_NAVATALA_FFI_STUB AND GPU_RUNTIME_HAVE_HIP)
    set(NAVATALA_HIP_TRANSFORMER_REGISTRY_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/internal/_registry_strings_hip_transformer.cpp")
    if(EXISTS "${NAVATALA_HIP_TRANSFORMER_REGISTRY_SOURCE}")
        list(APPEND NAVATALA_FFI_SOURCES src/internal/_registry_strings_hip_transformer.cpp)
        set(GPU_RUNTIME_HAVE_HIP_TRANSFORMER_REGISTRY 1)
        message(STATUS "[Navatala GPU] HIP transformer registry shard enabled for MFMA wrapper dispatch")
    else()
        message(STATUS "[Navatala GPU] HIP transformer registry shard not found; MFMA wrapper dispatch disabled")
    endif()
endif()

# ============================================================================
# Library target
# ============================================================================

add_library(gpu_runtime SHARED ${CORE_SOURCES} ${BACKEND_SOURCES} ${NAVATALA_FFI_SOURCES})

if(GPU_RUNTIME_NAVATALA_WRAPPER)
    target_compile_definitions(gpu_runtime PUBLIC NAVATALA_ENABLE_EXCEPTIONS=1)
endif()

target_include_directories(gpu_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${BACKEND_INCLUDE_DIRS}
)

target_compile_definitions(gpu_runtime PRIVATE
    GPU_RUNTIME_HAVE_CUDA=${GPU_RUNTIME_HAVE_CUDA}
    GPU_RUNTIME_HAVE_HIP=${GPU_RUNTIME_HAVE_HIP}
    GPU_RUNTIME_HAVE_ROCBLAS=${GPU_RUNTIME_HAVE_ROCBLAS}
    GPU_RUNTIME_HAVE_VULKAN=${GPU_RUNTIME_HAVE_VULKAN}
    GPU_RUNTIME_HAVE_OPENCL=${GPU_RUNTIME_HAVE_OPENCL}
    GPU_RUNTIME_HAVE_METAL=${GPU_RUNTIME_HAVE_METAL}
    NAVATALA_GPU_HAVE_HIP_TRANSFORMER_REGISTRY=${GPU_RUNTIME_HAVE_HIP_TRANSFORMER_REGISTRY}
)

target_link_libraries(gpu_runtime PRIVATE ${BACKEND_LIBS})

if(UNIX AND NOT APPLE AND GPU_RUNTIME_HIDE_INTERNAL_SYMBOLS AND NOT GPU_RUNTIME_BUILD_TESTS)
    set(GPU_RUNTIME_EXPORTS_MAP "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu_runtime_exports.map")
    if(EXISTS "${GPU_RUNTIME_EXPORTS_MAP}")
        target_link_options(gpu_runtime PRIVATE
            "LINKER:--version-script=${GPU_RUNTIME_EXPORTS_MAP}"
        )
    endif()
elseif(UNIX AND NOT APPLE AND GPU_RUNTIME_HIDE_INTERNAL_SYMBOLS AND GPU_RUNTIME_BUILD_TESTS)
    message(STATUS "[Navatala GPU] Internal symbol hiding disabled while runtime tests are enabled")
endif()

# ============================================================================
# Install
# ============================================================================

include(GNUInstallDirs)
if(GPU_RUNTIME_INSTALL_STANDALONE_ARTIFACTS)
    install(TARGETS gpu_runtime
        EXPORT  gpu_runtime-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# ============================================================================
# Tests
# ============================================================================

if(GPU_RUNTIME_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
