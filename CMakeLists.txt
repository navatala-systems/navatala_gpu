# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
#
# Navatala GPU — top-level build.
#
# Adds the runtime library and (optionally) the kernel host registry and the
# example orchestrator. Each sub-component has its own CMakeLists.txt that
# this file delegates to.

cmake_minimum_required(VERSION 3.18)
project(navatala_gpu VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Top-level build options
# ============================================================================

option(NAVATALA_GPU_BUILD_RUNTIME  "Build the cross-platform GPU runtime library" ON)
option(NAVATALA_GPU_BUILD_EXAMPLES "Build worked examples that launch kernels"   ON)
option(NAVATALA_GPU_BUILD_TESTS    "Build runtime test suite"                    ON)
option(NAVATALA_GPU_USE_CUDA       "Enable CUDA runtime backend"                 ON)
option(NAVATALA_GPU_USE_HIP        "Enable HIP/ROCm runtime backend"             ON)
option(NAVATALA_GPU_USE_VULKAN     "Enable Vulkan runtime backend"               ON)
option(NAVATALA_GPU_USE_OPENCL     "Enable OpenCL runtime backend"               ON)
option(NAVATALA_GPU_USE_METAL      "Enable Metal runtime backend"                OFF)

# Note: the host registry sources under runtime/src/internal/ and the
# orchestrator/ source tree ship as code but don't carry CMakeLists.txt
# entries here — they have non-trivial external dependencies (a host CFD
# stack for the orchestrator, a kernel registry data structure for the
# host) that we don't want to advertise as a turnkey build. Consumers who
# want to build them can add their own CMake glue; see
# orchestrator/README.md.

# Forward NAVATALA_GPU_BUILD_TESTS into the runtime's named variable AND
# into CTest's standard BUILD_TESTING. The redundancy is deliberate:
# runtime/CMakeLists.txt reads GPU_RUNTIME_BUILD_TESTS so its source is
# self-explanatory when grepped; BUILD_TESTING keeps `-DBUILD_TESTING=OFF`
# working as a standard CTest convention.
if(NAVATALA_GPU_BUILD_TESTS)
    set(GPU_RUNTIME_BUILD_TESTS ON  CACHE BOOL "Build runtime test suite" FORCE)
    set(BUILD_TESTING           ON  CACHE BOOL "Build the testing tree"   FORCE)
else()
    set(GPU_RUNTIME_BUILD_TESTS OFF CACHE BOOL "Build runtime test suite" FORCE)
    set(BUILD_TESTING           OFF CACHE BOOL "Build the testing tree"   FORCE)
endif()
include(CTest)

# Public backend switches are forwarded to the runtime's standalone option
# names. This keeps the top-level package predictable for downstream builds
# while preserving the runtime subproject's existing cache variables.
set(GPU_RUNTIME_USE_CUDA   ${NAVATALA_GPU_USE_CUDA}   CACHE BOOL "Enable CUDA backend"           FORCE)
set(GPU_RUNTIME_USE_HIP    ${NAVATALA_GPU_USE_HIP}    CACHE BOOL "Enable HIP backend"            FORCE)
set(GPU_RUNTIME_USE_VULKAN ${NAVATALA_GPU_USE_VULKAN} CACHE BOOL "Enable Vulkan compute backend" FORCE)
set(GPU_RUNTIME_USE_OPENCL ${NAVATALA_GPU_USE_OPENCL} CACHE BOOL "Enable OpenCL backend"         FORCE)
set(GPU_RUNTIME_USE_METAL  ${NAVATALA_GPU_USE_METAL}  CACHE BOOL "Enable Metal backend"          FORCE)

# ============================================================================
# Sub-components
# ============================================================================

if(NAVATALA_GPU_BUILD_RUNTIME)
    add_subdirectory(runtime)
endif()

if(NAVATALA_GPU_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

# ============================================================================
# Convenience: print build summary
# ============================================================================

message(STATUS "")
message(STATUS "Navatala GPU build configuration")
message(STATUS "  Runtime library:    ${NAVATALA_GPU_BUILD_RUNTIME}")
message(STATUS "  Examples:           ${NAVATALA_GPU_BUILD_EXAMPLES}")
message(STATUS "  Runtime tests:      ${NAVATALA_GPU_BUILD_TESTS}")
message(STATUS "  CUDA backend:       ${NAVATALA_GPU_USE_CUDA}")
message(STATUS "  HIP backend:        ${NAVATALA_GPU_USE_HIP}")
message(STATUS "  Vulkan backend:     ${NAVATALA_GPU_USE_VULKAN}")
message(STATUS "  OpenCL backend:     ${NAVATALA_GPU_USE_OPENCL}")
message(STATUS "  Metal backend:      ${NAVATALA_GPU_USE_METAL}")
message(STATUS "")
