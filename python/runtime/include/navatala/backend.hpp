// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Navatala C++ Wrapper Layer - Backend Abstraction
// Part of the Navatala GPU Runtime Library
#pragma once

#include "config.hpp"
#include "error.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// @file navatala/backend.hpp
/// @brief GPU backend enumeration and capability queries.
///
/// This header provides:
/// - backend enum for supported GPU compute backends
/// - backend_capabilities struct with detailed device information
/// - Functions to query available backends and their capabilities

namespace navatala {

// ============================================================================
// Backend Enumeration
// ============================================================================

/// @brief Supported GPU compute backends.
///
/// The numeric values are stable and correspond to the NAVATALA_BACKEND_*
/// macros defined in config.hpp.
enum class backend : uint32_t {
    cuda        = NAVATALA_BACKEND_CUDA,    ///< NVIDIA CUDA
    hip         = NAVATALA_BACKEND_HIP,     ///< AMD HIP/ROCm
    metal       = NAVATALA_BACKEND_METAL,   ///< Apple Metal
    opencl      = NAVATALA_BACKEND_OPENCL,  ///< OpenCL
    vulkan      = NAVATALA_BACKEND_VULKAN,  ///< Vulkan Compute
    auto_select = NAVATALA_BACKEND_AUTO     ///< Auto-select best available
};

/// @brief Returns a string name for a backend.
[[nodiscard]] constexpr const char* backend_name(backend b) noexcept {
    switch (b) {
        case backend::cuda:        return "CUDA";
        case backend::hip:         return "HIP";
        case backend::metal:       return "Metal";
        case backend::opencl:      return "OpenCL";
        case backend::vulkan:      return "Vulkan";
        case backend::auto_select: return "auto";
        default:                   return "unknown";
    }
}

// ============================================================================
// Backend Capabilities
// ============================================================================

/// @brief Detailed capabilities of a GPU backend/device.
///
/// This structure provides comprehensive information about a GPU device,
/// including compute capabilities, memory limits, and supported features.
struct backend_capabilities {
    backend type;                       ///< The backend type

    // Device identification
    std::string name;                   ///< Backend name, e.g., "NVIDIA CUDA 12.2"
    std::string device_name;            ///< Device name, e.g., "NVIDIA RTX 4090"
    std::string driver_version;         ///< Driver version string

    // Version information
    int major_version;                  ///< Major version number
    int minor_version;                  ///< Minor version number
    int patch_version;                  ///< Patch version number

    // Memory capabilities
    bool supports_managed_memory;       ///< Unified/managed memory support
    bool supports_unified_memory;       ///< Unified virtual addressing
    bool supports_virtual_memory;       ///< Virtual memory management
    bool supports_memory_pools;         ///< Memory pool allocation

    // Precision/compute capabilities
    bool supports_fp16;                 ///< Half-precision float support
    bool supports_bf16;                 ///< BFloat16 support
    bool supports_tf32;                 ///< TensorFloat-32 support
    bool supports_fp64;                 ///< Double-precision support
    bool supports_int8;                 ///< INT8 compute support
    bool supports_tensor_cores;         ///< Tensor Cores / Matrix Cores

    // Async capabilities
    bool supports_streams;              ///< Multiple execution streams
    bool supports_events;               ///< Event synchronization
    bool supports_graphs;               ///< Graph capture and execution
    bool supports_cooperative_groups;   ///< Cooperative kernel launch

    // Memory limits
    size_t max_global_memory_bytes;     ///< Total global memory
    size_t max_shared_memory_bytes;     ///< Max shared memory per block
    size_t max_constant_memory_bytes;   ///< Max constant memory
    size_t max_register_bytes;          ///< Max registers per thread
    size_t max_allocation_bytes;        ///< Max single allocation size

    // Compute limits
    uint32_t max_thread_block_size;     ///< Max threads per block
    uint32_t max_grid_dim_x;            ///< Max grid dimension X
    uint32_t max_grid_dim_y;            ///< Max grid dimension Y
    uint32_t max_grid_dim_z;            ///< Max grid dimension Z
    int max_thread_blocks;              ///< Max concurrent blocks
    int multiprocessor_count;           ///< Number of SMs/CUs
    int threads_per_warp;               ///< Threads per warp/wavefront
    int max_warps_per_multiprocessor;   ///< Max warps per SM/CU

    // Default construction with safe defaults
    backend_capabilities() noexcept
        : type(backend::auto_select)
        , name()
        , device_name()
        , driver_version()
        , major_version(0)
        , minor_version(0)
        , patch_version(0)
        , supports_managed_memory(false)
        , supports_unified_memory(false)
        , supports_virtual_memory(false)
        , supports_memory_pools(false)
        , supports_fp16(false)
        , supports_bf16(false)
        , supports_tf32(false)
        , supports_fp64(false)
        , supports_int8(false)
        , supports_tensor_cores(false)
        , supports_streams(false)
        , supports_events(false)
        , supports_graphs(false)
        , supports_cooperative_groups(false)
        , max_global_memory_bytes(0)
        , max_shared_memory_bytes(0)
        , max_constant_memory_bytes(0)
        , max_register_bytes(0)
        , max_allocation_bytes(0)
        , max_thread_block_size(0)
        , max_grid_dim_x(0)
        , max_grid_dim_y(0)
        , max_grid_dim_z(0)
        , max_thread_blocks(0)
        , multiprocessor_count(0)
        , threads_per_warp(0)
        , max_warps_per_multiprocessor(0)
    {}
};

// ============================================================================
// Backend Query Functions (Declarations)
// ============================================================================

/// @brief Queries and returns a list of all currently available GPU backends.
///
/// This function probes the system for available GPU compute backends
/// and returns those that can be used.
///
/// @return A vector of available backend types.
/// @note This function caches its result after the first call.
[[nodiscard]] NAVATALA_API std::vector<backend> available_backends() noexcept;

/// @brief Checks if a specific GPU backend is available on the system.
///
/// @param b The backend type to check.
/// @return True if the backend is available, false otherwise.
[[nodiscard]] NAVATALA_API bool is_backend_available(backend b) noexcept;

/// @brief Retrieves the capabilities for a specific GPU backend.
///
/// This function queries the default device for the specified backend
/// and returns its capabilities.
///
/// @param b The backend type to query.
/// @return A backend_capabilities struct containing device details.
///         If the backend is not available, returns a default-constructed
///         capabilities struct with is_backend_available(b) returning false.
[[nodiscard]] NAVATALA_API backend_capabilities get_capabilities(backend b) noexcept;

/// @brief Retrieves capabilities for a specific device on a backend.
///
/// @param b The backend type to query.
/// @param device_id The device ID to query.
/// @return A backend_capabilities struct containing device details.
[[nodiscard]] NAVATALA_API backend_capabilities get_capabilities(backend b, int device_id) noexcept;

/// @brief Returns the currently selected backend.
///
/// This returns the backend that will be used when auto_select is specified.
/// The selection is based on:
/// 1. NAVATALA_DEFAULT_BACKEND if not auto
/// 2. Environment variable GPU_RUNTIME_BACKEND if set
/// 3. First available backend in priority order: CUDA > HIP > Metal > Vulkan > OpenCL
///
/// @return The currently selected backend type.
[[nodiscard]] NAVATALA_API backend current_backend() noexcept;

/// @brief Returns the number of devices available for a backend.
///
/// @param b The backend to query.
/// @return The number of available devices, or 0 if the backend is unavailable.
[[nodiscard]] NAVATALA_API int device_count(backend b) noexcept;

// ============================================================================
// Compile-Time Backend Checks
// ============================================================================

/// @brief Compile-time check for backend support.
///
/// This is a constexpr function that checks if a backend is one of the
/// known supported types. It does not check runtime availability.
///
/// @param b The backend to check.
/// @return True if the backend is a known supported type.
[[nodiscard]] constexpr bool is_backend_supported_compile_time(backend b) noexcept {
    return b == backend::cuda ||
           b == backend::hip ||
           b == backend::metal ||
           b == backend::opencl ||
           b == backend::vulkan;
}

/// @brief Returns the default backend preference order.
///
/// This defines the priority order for auto-selecting a backend.
/// Higher priority backends are tried first.
///
/// @return A constexpr array of backends in priority order.
[[nodiscard]] constexpr std::array<backend, 5> backend_priority_order() noexcept {
    return {{
        backend::cuda,   // Highest priority
        backend::hip,
        backend::metal,
        backend::vulkan,
        backend::opencl  // Lowest priority
    }};
}

} // namespace navatala
