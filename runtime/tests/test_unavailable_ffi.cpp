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

#include <navatala/navatala_ffi.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
    assert(argc == 2);
    assert(std::strcmp(navatala_get_runtime_mode(), argv[1]) == 0);
    const bool stub = std::strcmp(argv[1], "stub") == 0;
    if (navatala_get_available_backend_count() != 0) {
        std::cout << "Requires a GPU-less environment\n";
        return 77;
    }
    assert(navatala_get_available_backend_count() == 0);
    assert(navatala_get_current_backend() == NAVATALA_BACKEND_AUTO_FFI);
    uint8_t ready = 23;
    assert(navatala_gpu_queue_query(nullptr, &ready) == NAVATALA_NOT_IMPLEMENTED);
    assert(ready == 23 && navatala_gpu_queue_is_ready(nullptr) == 0);
    NavatalaBackend output = NAVATALA_BACKEND_AUTO_FFI;
    assert(navatala_get_available_backends(&output, 1) == 0);
    assert(output == NAVATALA_BACKEND_AUTO_FFI);
    assert(navatala_get_available_backends(nullptr, 1) == 0);
    assert(navatala_get_available_backends(&output, -1) == 0);
    for (auto backend : {NAVATALA_BACKEND_CUDA_FFI, NAVATALA_BACKEND_HIP_FFI,
                         NAVATALA_BACKEND_METAL_FFI, NAVATALA_BACKEND_OPENCL_FFI,
                         NAVATALA_BACKEND_VULKAN_FFI, NAVATALA_BACKEND_AUTO_FFI}) {
        assert(navatala_is_backend_available(backend) == 0);
        if (stub) assert(navatala_is_backend_compiled(backend) == 0);
        assert(navatala_is_backend_initialized(backend) == 0);
        assert(navatala_get_device_count(backend) == 0);
        NavatalaBackendCapabilities caps{};
        auto status = navatala_get_backend_capabilities(backend, 0, &caps);
        assert(status != NAVATALA_SUCCESS);
        assert(caps.device_name[0] == '\0');
        assert(caps.max_global_memory_bytes == 0);
        size_t free_bytes = 123, total_bytes = 456;
        uint8_t supported = 1;
        status = navatala_get_backend_memory_info(
            backend, 0, &free_bytes, &total_bytes, &supported);
        assert(status != NAVATALA_SUCCESS);
        assert(!supported && free_bytes == 0 && total_bytes == 0);
        if (stub) assert(status == NAVATALA_NOT_IMPLEMENTED);
    }
    std::cout << argv[1] << " unavailable FFI controls passed\n";
}
