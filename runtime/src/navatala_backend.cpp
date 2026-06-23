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

// Navatala C++ Wrapper Layer - Backend Implementation
// Part of the Navatala GPU Runtime Library
#include "../include/navatala/backend.hpp"
#include "../include/navatala/navatala_ffi.h"

#include <cstring>
#include <algorithm>

namespace navatala {

// Convert FFI backend to C++ backend enum
static backend ffi_to_backend(NavatalaBackend b) {
    switch (b) {
        case NAVATALA_BACKEND_CUDA_FFI:   return backend::cuda;
        case NAVATALA_BACKEND_HIP_FFI:    return backend::hip;
        case NAVATALA_BACKEND_METAL_FFI:  return backend::metal;
        case NAVATALA_BACKEND_OPENCL_FFI: return backend::opencl;
        case NAVATALA_BACKEND_VULKAN_FFI: return backend::vulkan;
        case NAVATALA_BACKEND_AUTO_FFI:   return backend::auto_select;
        default:                          return backend::auto_select;
    }
}

// Convert C++ backend enum to FFI backend
static NavatalaBackend backend_to_ffi(backend b) {
    switch (b) {
        case backend::cuda:        return NAVATALA_BACKEND_CUDA_FFI;
        case backend::hip:         return NAVATALA_BACKEND_HIP_FFI;
        case backend::metal:       return NAVATALA_BACKEND_METAL_FFI;
        case backend::opencl:      return NAVATALA_BACKEND_OPENCL_FFI;
        case backend::vulkan:      return NAVATALA_BACKEND_VULKAN_FFI;
        case backend::auto_select: return NAVATALA_BACKEND_AUTO_FFI;
        default:                   return NAVATALA_BACKEND_AUTO_FFI;
    }
}

std::vector<backend> available_backends() noexcept {
    std::vector<backend> result;

    int count = navatala_get_available_backend_count();
    if (count <= 0) {
        return result;
    }

    std::vector<NavatalaBackend> ffi_backends(static_cast<size_t>(count));
    int actual = navatala_get_available_backends(ffi_backends.data(), count);

    result.reserve(static_cast<size_t>(actual));
    for (int i = 0; i < actual; ++i) {
        result.push_back(ffi_to_backend(ffi_backends[static_cast<size_t>(i)]));
    }

    return result;
}

bool is_backend_available(backend b) noexcept {
    return navatala_is_backend_available(backend_to_ffi(b)) != 0;
}

backend_capabilities get_capabilities(backend b) noexcept {
    return get_capabilities(b, 0);
}

backend_capabilities get_capabilities(backend b, int device_id) noexcept {
    backend_capabilities caps;

    NavatalaBackendCapabilities ffi_caps;
    NavatalaErrorCode err = navatala_get_backend_capabilities(
        backend_to_ffi(b), device_id, &ffi_caps);

    if (err != NAVATALA_SUCCESS) {
        return caps;  // Return default-constructed caps on error
    }

    // Copy fields
    caps.type = ffi_to_backend(ffi_caps.type);
    caps.name = ffi_caps.name;
    caps.device_name = ffi_caps.device_name;
    caps.driver_version = ffi_caps.driver_version;

    caps.major_version = ffi_caps.major_version;
    caps.minor_version = ffi_caps.minor_version;
    caps.patch_version = ffi_caps.patch_version;

    caps.supports_managed_memory = ffi_caps.supports_managed_memory != 0;
    caps.supports_unified_memory = ffi_caps.supports_unified_memory != 0;
    caps.supports_virtual_memory = ffi_caps.supports_virtual_memory != 0;
    caps.supports_memory_pools = ffi_caps.supports_memory_pools != 0;

    caps.supports_fp16 = ffi_caps.supports_fp16 != 0;
    caps.supports_bf16 = ffi_caps.supports_bf16 != 0;
    caps.supports_tf32 = ffi_caps.supports_tf32 != 0;
    caps.supports_fp64 = ffi_caps.supports_fp64 != 0;
    caps.supports_int8 = ffi_caps.supports_int8 != 0;
    caps.supports_tensor_cores = ffi_caps.supports_tensor_cores != 0;

    caps.supports_streams = ffi_caps.supports_streams != 0;
    caps.supports_events = ffi_caps.supports_events != 0;
    caps.supports_graphs = ffi_caps.supports_graphs != 0;
    caps.supports_cooperative_groups = ffi_caps.supports_cooperative_groups != 0;

    caps.max_global_memory_bytes = ffi_caps.max_global_memory_bytes;
    caps.max_shared_memory_bytes = ffi_caps.max_shared_memory_bytes;
    caps.max_constant_memory_bytes = ffi_caps.max_constant_memory_bytes;
    caps.max_register_bytes = ffi_caps.max_register_bytes;
    caps.max_allocation_bytes = ffi_caps.max_allocation_bytes;

    caps.max_thread_block_size = ffi_caps.max_thread_block_size;
    caps.max_grid_dim_x = ffi_caps.max_grid_dim_x;
    caps.max_grid_dim_y = ffi_caps.max_grid_dim_y;
    caps.max_grid_dim_z = ffi_caps.max_grid_dim_z;
    caps.max_thread_blocks = ffi_caps.max_thread_blocks;
    caps.multiprocessor_count = ffi_caps.multiprocessor_count;
    caps.threads_per_warp = ffi_caps.threads_per_warp;
    caps.max_warps_per_multiprocessor = ffi_caps.max_warps_per_multiprocessor;

    return caps;
}

backend current_backend() noexcept {
    return ffi_to_backend(navatala_get_current_backend());
}

int device_count(backend b) noexcept {
    return navatala_get_device_count(backend_to_ffi(b));
}

} // namespace navatala
