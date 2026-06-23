// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd

#include "gpu_library_ops.h"
#include "gpu_runtime.h"

namespace GpuRuntime {

DeviceMemTelemetry deviceMemTelemetry()
{
    return {};
}

std::unique_ptr<LibraryOps> createLibraryOps()
{
#if defined(GPU_RUNTIME_HAVE_ROCBLAS) && GPU_RUNTIME_HAVE_ROCBLAS
    auto hip = createHipLibraryOps();
    if (hip) {
        return hip;
    }
#endif
    return nullptr;
}

std::unique_ptr<LibraryOps> createLibraryOpsForBackend(const std::string& backend)
{
#if defined(GPU_RUNTIME_HAVE_ROCBLAS) && GPU_RUNTIME_HAVE_ROCBLAS
    if (backend == "hip" || backend == "rocm") {
        return createHipLibraryOps();
    }
#else
    (void)backend;
#endif
    return nullptr;
}

std::unique_ptr<LibraryOps> createCudaLibraryOps()
{
    return nullptr;
}

#if !(defined(GPU_RUNTIME_HAVE_ROCBLAS) && GPU_RUNTIME_HAVE_ROCBLAS)
std::unique_ptr<LibraryOps> createHipLibraryOps()
{
    return nullptr;
}
#endif

std::unique_ptr<LibraryOps> createMetalLibraryOps()
{
    return nullptr;
}

std::unique_ptr<LibraryOps> createGinkgoLibraryOps()
{
    return nullptr;
}

std::unique_ptr<LibraryOps> createAmgXLibraryOps()
{
    return nullptr;
}

} // namespace GpuRuntime
