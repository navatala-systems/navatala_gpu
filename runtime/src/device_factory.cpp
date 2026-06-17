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

// GPU Runtime Device Factory
// Backend selection and device creation

// Standard library headers first to avoid namespace/linkage interference
#include <memory>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <stdexcept>

// Project headers after standard library
#include <gpu_runtime.h>
#include "memory_resource.h"
#include "pool_memory_resource.h"
#include "binning_memory_resource.h"
#include "stream_pool.h"

namespace GpuRuntime {

// Forward declarations of backend-specific device implementations
// Only declare if the backend is enabled
#if GPU_RUNTIME_HAVE_CUDA
std::unique_ptr<Device> CudaDevice_create(int device_id);
#endif

#if GPU_RUNTIME_HAVE_HIP
std::unique_ptr<Device> HipDevice_create(int device_id);
#endif

#if GPU_RUNTIME_HAVE_VULKAN
std::unique_ptr<Device> VulkanDevice_tryCreate(int device_id);
#endif

#if GPU_RUNTIME_HAVE_OPENCL
std::unique_ptr<Device> OpenCLDevice_tryCreate(int device_id);
#endif

#if GPU_RUNTIME_HAVE_METAL
std::unique_ptr<Device> MetalDevice_tryCreate(int device_id);
#endif

std::unique_ptr<Device> Device::create(int device_id) {
    const char* backend_env = std::getenv("GPU_RUNTIME_BACKEND");
    std::string backend = backend_env ? backend_env : "auto";

    auto tryCreate = [&](auto&& fn) -> std::unique_ptr<Device> {
        try {
            return fn();
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Device::create(" << device_id << "): " << e.what() << "\n";
            return nullptr;
        } catch (...) {
            std::cerr << "[ERROR] Device::create(" << device_id << "): unknown exception\n";
            return nullptr;
        }
    };

    if (backend == "auto") {
        // Platform probing order
        // Use CMake-defined macros for proper optional compilation
#if GPU_RUNTIME_HAVE_METAL
        if (auto dev = tryCreate([&] { return MetalDevice_tryCreate(device_id); })) {
            std::cerr << "[INFO] Using Metal backend (auto-detected on macOS)\n";
            return dev;
        }
#endif
#if GPU_RUNTIME_HAVE_HIP
        if (auto dev = tryCreate([&] { return HipDevice_create(device_id); })) {
            std::cerr << "[INFO] Using HIP backend (auto-detected on AMD platform)\n";
            return dev;
        }
#endif
#if GPU_RUNTIME_HAVE_CUDA
        if (auto dev = tryCreate([&] { return CudaDevice_create(device_id); })) {
            std::cerr << "[INFO] Using CUDA backend (auto-detected on NVIDIA platform)\n";
            return dev;
        }
#endif
#if GPU_RUNTIME_HAVE_VULKAN
        if (auto dev = tryCreate([&] { return VulkanDevice_tryCreate(device_id); })) {
            std::cerr << "[INFO] Using Vulkan backend (fallback)\n";
            return dev;
        }
#endif
#if GPU_RUNTIME_HAVE_OPENCL
        if (auto dev = tryCreate([&] { return OpenCLDevice_tryCreate(device_id); })) {
            std::cerr << "[INFO] Using OpenCL backend (fallback)\n";
            return dev;
        }
#endif

        std::cerr << "[ERROR] Failed to create GPU device: no suitable backend found\n";
        std::cerr << "   Set GPU_RUNTIME_BACKEND environment variable to specify a backend\n";
        std::cerr << "   Available backends: cuda, hip, vulkan, opencl, metal\n";
        return nullptr;
    }

    // Explicit backend selection
    if (backend == "cuda") {
#if GPU_RUNTIME_HAVE_CUDA
        return tryCreate([&] { return CudaDevice_create(device_id); });
#else
        std::cerr << "[ERROR] CUDA backend requested but CUDA not available at compile time\n";
        return nullptr;
#endif
    }
    if (backend == "hip") {
#if GPU_RUNTIME_HAVE_HIP
        return tryCreate([&] { return HipDevice_create(device_id); });
#else
        std::cerr << "[ERROR] HIP backend requested but HIP not available at compile time\n";
        return nullptr;
#endif
    }
    if (backend == "vulkan") {
#if GPU_RUNTIME_HAVE_VULKAN
        return tryCreate([&] { return VulkanDevice_tryCreate(device_id); });
#else
        std::cerr << "[ERROR] Vulkan backend requested but Vulkan not available at compile time\n";
        return nullptr;
#endif
    }
    if (backend == "opencl") {
#if GPU_RUNTIME_HAVE_OPENCL
        return tryCreate([&] { return OpenCLDevice_tryCreate(device_id); });
#else
        std::cerr << "[ERROR] OpenCL backend requested but OpenCL not available at compile time\n";
        return nullptr;
#endif
    }
    if (backend == "metal") {
#if GPU_RUNTIME_HAVE_METAL
        return tryCreate([&] { return MetalDevice_tryCreate(device_id); });
#else
        std::cerr << "[ERROR] Metal backend requested but not available on this platform\n";
        return nullptr;
#endif
    }

    std::cerr << "[ERROR] Unknown backend: " << backend << "\n";
    std::cerr << "   Available backends: auto, cuda, hip, vulkan, opencl, metal\n";
    return nullptr;
}

// ============================================================================
// FFI Bridge for Runtime Capability Probing ()
// ============================================================================

extern "C" {

GpuRuntimeDeviceHandle GpuRuntime_createDevice(int device_id) {
    try {
        auto device = Device::create(device_id);
        if (device) {
            // Release ownership to the caller (they must call GpuRuntime_destroyDevice)
            return reinterpret_cast<GpuRuntimeDeviceHandle>(device.release());
        }
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_createDevice: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_createDevice: unknown exception" << std::endl;
        return nullptr;
    }
}

void GpuRuntime_queryCapabilities(
    GpuRuntimeDeviceHandle device,
    int* out_version_major,
    int* out_version_minor,
    int* out_version_patch,
    char* out_extensions,
    int extensions_buf_size,
    uint32_t* out_subgroup_size,
    size_t* out_max_shared_memory,
    uint32_t* out_max_workgroup_size,
    uint8_t* out_atomic_float64,
    uint8_t* out_atomic_float32,
    uint8_t* out_atomic_float16,
    uint8_t* out_atomic_bfloat16,
    uint8_t* out_atomic_int64,
    uint8_t* out_atomic_int32,
    uint8_t* out_atomic_int16,
    uint8_t* out_atomic_int8
) {
    if (!device) return;

    try {
        auto* dev = reinterpret_cast<Device*>(device);

        // Query version
        if (out_version_major || out_version_minor || out_version_patch) {
            std::string versionStr = dev->getComputeCapability();
            // Try to parse version string (format varies by backend)
            // Default to 1,0,0 if parsing fails
            int major = 1, minor = 0, patch = 0;
            if (sscanf(versionStr.c_str(), "%d.%d", &major, &minor) == 2) {
                // Successfully parsed major.minor
            }
            if (out_version_major) *out_version_major = major;
            if (out_version_minor) *out_version_minor = minor;
            if (out_version_patch) *out_version_patch = patch;
        }

        // Query extensions (comma-separated list)
        if (out_extensions && extensions_buf_size > 0) {
            // Build a comma-separated list of supported features/extensions
            std::string extensions;

            if (dev->supportsFeature("float64")) extensions += "float64,";
            if (dev->supportsFeature("float16")) extensions += "float16,";
            if (dev->supportsFeature("bfloat16")) extensions += "bfloat16,";
            if (dev->supportsFeature("atomicFloat64")) extensions += "atomicFloat64,";
            if (dev->supportsFeature("atomicFloat32")) extensions += "atomicFloat32,";
            if (dev->supportsFeature("managedMemory")) extensions += "managedMemory,";
            if (dev->supportsFeature("sharedMem")) extensions += "sharedMem,";
            if (dev->supportsFeature("graphs")) extensions += "graphs,";

            // Remove trailing comma if present
            if (!extensions.empty() && extensions.back() == ',') {
                extensions.pop_back();
            }

            // Copy to output buffer (with null termination)
            std::strncpy(out_extensions, extensions.c_str(), extensions_buf_size - 1);
            out_extensions[extensions_buf_size - 1] = '\0';
        }

        // Query subgroup size
        if (out_subgroup_size) {
            *out_subgroup_size = dev->getSubgroupSize();
        }

        // Query memory limits
        if (out_max_shared_memory) {
            *out_max_shared_memory = dev->getMaxSharedMemoryPerWorkgroup();
        }

        if (out_max_workgroup_size) {
            // Default value if getWorkgroupSize() not available
            *out_max_workgroup_size = 1024;
        }

        // Query atomic support
        if (out_atomic_float64) {
            *out_atomic_float64 = dev->supportsAtomicOp(AtomicOpType::Add, DataType::Float64) ? 1 : 0;
        }
        if (out_atomic_float32) {
            *out_atomic_float32 = dev->supportsAtomicOp(AtomicOpType::Add, DataType::Float32) ? 1 : 0;
        }
        if (out_atomic_float16) {
            *out_atomic_float16 = dev->supportsAtomicOp(AtomicOpType::Add, DataType::Float16) ? 1 : 0;
        }
        if (out_atomic_bfloat16) {
            *out_atomic_bfloat16 = dev->supportsAtomicOp(AtomicOpType::Add, DataType::BFloat16) ? 1 : 0;
        }
        if (out_atomic_int64) {
            *out_atomic_int64 = dev->supportsAtomicOp(AtomicOpType::Add, DataType::Int64) ? 1 : 0;
        }
        if (out_atomic_int32) {
            *out_atomic_int32 = dev->supportsAtomicOp(AtomicOpType::Add, DataType::Int32) ? 1 : 0;
        }
        if (out_atomic_int16) {
            *out_atomic_int16 = dev->supportsAtomicOp(AtomicOpType::Add, DataType::Int16) ? 1 : 0;
        }
        if (out_atomic_int8) {
            *out_atomic_int8 = dev->supportsAtomicOp(AtomicOpType::Add, DataType::Int8) ? 1 : 0;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_queryCapabilities: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_queryCapabilities: unknown exception" << std::endl;
    }
}

void GpuRuntime_destroyDevice(GpuRuntimeDeviceHandle device) {
    if (device) {
        try {
            // Reclaim ownership and delete the device
            delete reinterpret_cast<Device*>(device);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] GpuRuntime_destroyDevice: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[ERROR] GpuRuntime_destroyDevice: unknown exception" << std::endl;
        }
    }
}

// ============================================================================
// Memory Resource FFI Implementation (RMM Formalization)
// ============================================================================

GpuRuntimeMemoryResourceHandle GpuRuntime_createPoolResource(
    GpuRuntimeDeviceHandle device,
    size_t initial_size,
    size_t max_size
) {
    if (!device) return nullptr;
    try {
        Device* dev = reinterpret_cast<Device*>(device);
        PoolConfig config;
        config.initial_pool_size = initial_size;
        config.maximum_pool_size = max_size;
        auto* resource = new PoolMemoryResource(*dev, config);
        return reinterpret_cast<GpuRuntimeMemoryResourceHandle>(resource);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_createPoolResource: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_createPoolResource: unknown exception" << std::endl;
        return nullptr;
    }
}

GpuRuntimeMemoryResourceHandle GpuRuntime_createBinningResource(
    GpuRuntimeDeviceHandle device,
    size_t min_bin,
    size_t max_bin
) {
    if (!device) return nullptr;
    try {
        Device* dev = reinterpret_cast<Device*>(device);
        auto* resource = new BinningMemoryResource(*dev, min_bin, max_bin);
        return reinterpret_cast<GpuRuntimeMemoryResourceHandle>(resource);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_createBinningResource: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_createBinningResource: unknown exception" << std::endl;
        return nullptr;
    }
}

void* GpuRuntime_resourceAllocate(
    GpuRuntimeMemoryResourceHandle resource,
    size_t bytes,
    size_t alignment
) {
    if (!resource) return nullptr;
    try {
        auto* mr = reinterpret_cast<MemoryResource*>(resource);
        return mr->allocate(bytes, alignment);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_resourceAllocate: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_resourceAllocate: unknown exception" << std::endl;
        return nullptr;
    }
}

void GpuRuntime_resourceDeallocate(
    GpuRuntimeMemoryResourceHandle resource,
    void* ptr,
    size_t bytes,
    size_t alignment
) {
    if (!resource || !ptr) return;
    try {
        auto* mr = reinterpret_cast<MemoryResource*>(resource);
        mr->deallocate(ptr, bytes, alignment);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_resourceDeallocate: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_resourceDeallocate: unknown exception" << std::endl;
    }
}

void GpuRuntime_destroyResource(GpuRuntimeMemoryResourceHandle resource) {
    if (resource) {
        try {
            delete reinterpret_cast<MemoryResource*>(resource);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] GpuRuntime_destroyResource: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[ERROR] GpuRuntime_destroyResource: unknown exception" << std::endl;
        }
    }
}

void GpuRuntime_getResourceStats(
    GpuRuntimeMemoryResourceHandle resource,
    size_t* out_allocated,
    size_t* out_peak,
    size_t* out_count
) {
    if (!resource) return;
    try {
        // Use polymorphic get_stats() - works for all resource types
        auto* mr = reinterpret_cast<MemoryResource*>(resource);
        auto stats = mr->get_stats();
        if (out_allocated) *out_allocated = stats.allocated_bytes;
        if (out_peak) *out_peak = stats.peak_allocated_bytes;
        if (out_count) *out_count = stats.allocation_count;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_getResourceStats: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_getResourceStats: unknown exception" << std::endl;
    }
}

// ============================================================================
// Stream Pool FFI Implementation (RMM Formalization)
// ============================================================================

GpuRuntimeStreamPoolHandle GpuRuntime_createStreamPool(
    GpuRuntimeDeviceHandle device,
    size_t initial_size
) {
    if (!device) return nullptr;
    try {
        Device* dev = reinterpret_cast<Device*>(device);
        auto* pool = new StreamPool(*dev, initial_size);
        return reinterpret_cast<GpuRuntimeStreamPoolHandle>(pool);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_createStreamPool: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_createStreamPool: unknown exception" << std::endl;
        return nullptr;
    }
}

GpuRuntimeQueueHandle GpuRuntime_poolAcquireQueue(GpuRuntimeStreamPoolHandle pool) {
    if (!pool) return nullptr;
    try {
        auto* streamPool = reinterpret_cast<StreamPool*>(pool);
        auto queue = streamPool->acquire();
        // Release ownership to FFI caller
        return reinterpret_cast<GpuRuntimeQueueHandle>(queue.release());
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_poolAcquireQueue: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_poolAcquireQueue: unknown exception" << std::endl;
        return nullptr;
    }
}

void GpuRuntime_poolReleaseQueue(GpuRuntimeStreamPoolHandle pool, GpuRuntimeQueueHandle queue) {
    if (!pool || !queue) return;
    try {
        auto* streamPool = reinterpret_cast<StreamPool*>(pool);
        // Take ownership back from FFI caller
        std::unique_ptr<Queue> q(reinterpret_cast<Queue*>(queue));
        streamPool->release(std::move(q));
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_poolReleaseQueue: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_poolReleaseQueue: unknown exception" << std::endl;
    }
}

void GpuRuntime_destroyStreamPool(GpuRuntimeStreamPoolHandle pool) {
    if (pool) {
        try {
            delete reinterpret_cast<StreamPool*>(pool);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] GpuRuntime_destroyStreamPool: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[ERROR] GpuRuntime_destroyStreamPool: unknown exception" << std::endl;
        }
    }
}

void GpuRuntime_queueSynchronize(GpuRuntimeQueueHandle queue) {
    if (!queue) return;
    try {
        auto* q = reinterpret_cast<Queue*>(queue);
        q->synchronize();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] GpuRuntime_queueSynchronize: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[ERROR] GpuRuntime_queueSynchronize: unknown exception" << std::endl;
    }
}

} // extern "C"

} // namespace GpuRuntime
