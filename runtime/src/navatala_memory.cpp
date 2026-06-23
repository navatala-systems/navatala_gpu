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

// Navatala C++ Wrapper Layer - Memory Implementation
// Part of the Navatala GPU Runtime Library
#include "../include/navatala/memory.hpp"
#include "../include/navatala/resource.hpp"
#include "../include/navatala/navatala_ffi.h"

namespace navatala::memory {

// ============================================================================
// Pool Implementation
// ============================================================================

result<pool> pool::create(resources& res, size_t initial_size, size_t max_size) {
    if (!res.handle()) {
        return result<pool>(error_code::invalid_handle, "pool::create() with invalid resources");
    }

    NavatalaGpuMemoryResource* pool_handle = nullptr;
    NavatalaErrorCode err = navatala_gpu_create_memory_pool(
        res.handle(), initial_size, max_size, &pool_handle);

    if (err != NAVATALA_SUCCESS) {
        return result<pool>(translate_error(err), "pool::create()");
    }

    return pool(pool_handle, res.handle(), initial_size, max_size);
}

pool::~pool() noexcept {
    if (handle_) {
        navatala_gpu_destroy_memory_resource(handle_);
        handle_ = nullptr;
    }
}

result<void*> pool::allocate(size_t bytes, size_t alignment) {
    if (!handle_) {
        return result<void*>(error_code::invalid_handle, "allocate() on invalid pool");
    }

    if (bytes == 0) {
        return result<void*>(error_code::invalid_param, "allocate() with zero bytes");
    }

    void* ptr = nullptr;
    NavatalaErrorCode err = navatala_gpu_resource_allocate(handle_, bytes, alignment, &ptr);

    if (err != NAVATALA_SUCCESS) {
        return result<void*>(translate_error(err), "pool::allocate()");
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.allocated_bytes += bytes;
        stats_.total_allocations++;
        stats_.active_allocations++;
        if (stats_.allocated_bytes > stats_.peak_allocated_bytes) {
            stats_.peak_allocated_bytes = stats_.allocated_bytes;
        }
        stats_.free_bytes = stats_.capacity_bytes - stats_.allocated_bytes;
    }

    return ptr;
}

void pool::deallocate(void* ptr, size_t bytes, size_t /*alignment*/) noexcept {
    if (!handle_ || !ptr) {
        return;
    }

    navatala_gpu_resource_deallocate(handle_, ptr, bytes);

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        if (bytes <= stats_.allocated_bytes) {
            stats_.allocated_bytes -= bytes;
        } else {
            stats_.allocated_bytes = 0;  // Shouldn't happen, but be safe
        }
        stats_.total_deallocations++;
        if (stats_.active_allocations > 0) {
            stats_.active_allocations--;
        }
        stats_.free_bytes = stats_.capacity_bytes - stats_.allocated_bytes;
    }
}

memory_stats pool::get_stats() const noexcept {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

size_t pool::allocated_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.allocated_bytes;
}

size_t pool::capacity_bytes() const noexcept {
    // Query from FFI if available, otherwise return our tracked value
    if (!handle_) {
        return 0;
    }

    size_t allocated = 0, capacity = 0, free = 0;
    navatala_gpu_resource_stats(handle_, &allocated, &capacity, &free);

    // Update our stats if FFI provides capacity
    if (capacity > 0) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.capacity_bytes = capacity;
    }

    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.capacity_bytes;
}

size_t pool::free_bytes() const noexcept {
    if (!handle_) {
        return 0;
    }

    size_t allocated = 0, capacity = 0, free = 0;
    navatala_gpu_resource_stats(handle_, &allocated, &capacity, &free);

    return free;
}

bool pool::owns(void* ptr) const noexcept {
    if (!handle_ || !ptr) {
        return false;
    }

    return navatala_gpu_resource_owns(handle_, ptr) != 0;
}

void pool::reset() {
    // Reset is a no-op for the current implementation
    // A real implementation would release all memory back to the pool
    // without destroying the pool itself

    if (!handle_) {
        return;
    }

    // For now, just reset statistics
    // Real implementation would need FFI support for pool reset
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.allocated_bytes = 0;
    stats_.active_allocations = 0;
    stats_.free_bytes = stats_.capacity_bytes;
}

} // namespace navatala::memory
