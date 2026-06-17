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

// Memory Resource Implementation
// Part of the RMM formalization effort

#include "memory_resource.h"
#include <unordered_map>
#include <mutex>

namespace GpuRuntime {

// ============================================================================
// DefaultMemoryResource Implementation
// ============================================================================

struct DefaultMemoryResource::Impl {
    std::mutex mutex;
    // Map from pointer to Buffer for proper deallocation
    std::unordered_map<void*, std::unique_ptr<Buffer>> allocations;
};

DefaultMemoryResource::DefaultMemoryResource(Device& device, MemoryKind kind)
    : device_(device)
    , kind_(kind)
    , impl_(std::make_unique<Impl>())
{
}

DefaultMemoryResource::~DefaultMemoryResource() {
    // All buffers are automatically destroyed by unique_ptr
}

void* DefaultMemoryResource::do_allocate(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;

    // Round up to alignment
    size_t aligned_bytes = align_up(bytes, alignment);

    // Create buffer
    auto buffer = device_.createBuffer(aligned_bytes, kind_);
    if (!buffer) {
        throw AllocationError(bytes, alignment);
    }

    // Not all backends expose raw device pointers (e.g. Vulkan without buffer-device-address).
    // For those backends, treat the backend-native handle as the allocation token.
    // This is sufficient for deallocation and for higher-level resources that only need
    // an opaque identifier (e.g. stubbed RMM layers on non-CUDA/HIP backends).
    void* ptr = buffer->getDevicePointer();
    if (!ptr) {
        ptr = buffer->nativeHandle();
    }
    if (!ptr) {
        throw AllocationError(bytes, alignment);
    }

    // Store buffer for later deallocation
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->allocations[ptr] = std::move(buffer);

    return ptr;
}

void DefaultMemoryResource::do_deallocate(void* ptr, size_t /*bytes*/, size_t /*alignment*/) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->allocations.find(ptr);
    if (it != impl_->allocations.end()) {
        impl_->allocations.erase(it);
    }
    // Note: If ptr not found, we silently ignore (might be from a different resource)
}

bool DefaultMemoryResource::do_is_equal(const MemoryResource& other) const noexcept {
    auto* other_default = dynamic_cast<const DefaultMemoryResource*>(&other);
    if (!other_default) return false;
    return &device_ == &other_default->device_ && kind_ == other_default->kind_;
}

// ============================================================================
// Default Resource Management
// ============================================================================

namespace {
    std::mutex g_default_resource_mutex;
    std::unordered_map<Device*, MemoryResource*> g_default_resources;
    std::unordered_map<Device*, std::unique_ptr<DefaultMemoryResource>> g_owned_defaults;
}

MemoryResource* get_default_resource(Device& device) {
    std::lock_guard<std::mutex> lock(g_default_resource_mutex);

    auto it = g_default_resources.find(&device);
    if (it != g_default_resources.end()) {
        return it->second;
    }

    // Create default resource for this device
    auto default_mr = std::make_unique<DefaultMemoryResource>(device);
    MemoryResource* result = default_mr.get();
    g_default_resources[&device] = result;
    g_owned_defaults[&device] = std::move(default_mr);

    return result;
}

MemoryResource* set_default_resource(Device& device, MemoryResource* resource) {
    std::lock_guard<std::mutex> lock(g_default_resource_mutex);

    MemoryResource* old = nullptr;
    auto it = g_default_resources.find(&device);
    if (it != g_default_resources.end()) {
        old = it->second;
    }

    if (resource) {
        g_default_resources[&device] = resource;
    } else {
        g_default_resources.erase(&device);
    }

    return old;
}

} // namespace GpuRuntime
