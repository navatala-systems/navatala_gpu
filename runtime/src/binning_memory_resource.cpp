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

// Binning Memory Resource Implementation
// Part of the RMM formalization effort

#include "binning_memory_resource.h"
#include <algorithm>
#include <mutex>
#include <map>
#include <unordered_map>
#include <cmath>

namespace GpuRuntime {

struct BinningMemoryResource::Impl {
    Device* device = nullptr;
    MemoryResource* upstream = nullptr;
    bool owns_upstream = false;

    size_t min_bin_size;
    size_t max_bin_size;

    std::mutex mutex;

    // Bins: size -> list of free pointers
    std::map<size_t, std::vector<void*>> bins;

    // Track allocation sizes for proper deallocation
    std::unordered_map<void*, size_t> alloc_sizes;

    // Statistics
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t peak_allocated = 0;

    ~Impl() {
        // Free all cached blocks
        for (auto& [size, ptrs] : bins) {
            for (void* ptr : ptrs) {
                if (upstream) {
                    upstream->deallocate(ptr, size, 256);
                }
            }
        }

        if (owns_upstream && upstream) {
            delete upstream;
        }
    }

    size_t compute_bin_size(size_t bytes) const {
        if (bytes <= min_bin_size) return min_bin_size;
        if (bytes > max_bin_size) return 0;  // Too large for binning

        // Round up to next power of 2
        size_t bin = min_bin_size;
        while (bin < bytes) {
            bin *= 2;
        }
        return (bin <= max_bin_size) ? bin : 0;
    }
};

BinningMemoryResource::BinningMemoryResource(Device& device,
                                             size_t min_bin_size,
                                             size_t max_bin_size)
    : impl_(std::make_unique<Impl>())
{
    impl_->device = &device;
    impl_->min_bin_size = min_bin_size;
    impl_->max_bin_size = max_bin_size;
    impl_->upstream = new DefaultMemoryResource(device);
    impl_->owns_upstream = true;
}

BinningMemoryResource::BinningMemoryResource(MemoryResource& upstream,
                                             size_t min_bin_size,
                                             size_t max_bin_size)
    : impl_(std::make_unique<Impl>())
{
    impl_->upstream = &upstream;
    impl_->owns_upstream = false;
    impl_->min_bin_size = min_bin_size;
    impl_->max_bin_size = max_bin_size;
}

BinningMemoryResource::~BinningMemoryResource() = default;

BinningMemoryResource::BinningMemoryResource(BinningMemoryResource&& other) noexcept = default;
BinningMemoryResource& BinningMemoryResource::operator=(BinningMemoryResource&& other) noexcept = default;

void* BinningMemoryResource::do_allocate(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;

    size_t bin_size = impl_->compute_bin_size(bytes);

    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (bin_size > 0) {
        // Try to get from bin
        auto it = impl_->bins.find(bin_size);
        if (it != impl_->bins.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            impl_->alloc_sizes[ptr] = bin_size;
            ++impl_->cache_hits;
            return ptr;
        }
    }

    // Allocate from upstream
    ++impl_->cache_misses;
    size_t alloc_size = (bin_size > 0) ? bin_size : bytes;
    void* ptr = impl_->upstream->allocate(alloc_size, alignment);
    if (ptr) {
        impl_->alloc_sizes[ptr] = alloc_size;
        // Update peak tracking
        size_t total = 0;
        for (const auto& [p, s] : impl_->alloc_sizes) {
            total += s;
        }
        impl_->peak_allocated = std::max(impl_->peak_allocated, total);
    } else {
        throw AllocationError(bytes, alignment);
    }
    return ptr;
}

void BinningMemoryResource::do_deallocate(void* ptr, size_t bytes, size_t alignment) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto it = impl_->alloc_sizes.find(ptr);
    if (it == impl_->alloc_sizes.end()) {
        // Not our allocation
        return;
    }

    size_t actual_size = it->second;
    impl_->alloc_sizes.erase(it);

    size_t bin_size = impl_->compute_bin_size(actual_size);
    if (bin_size > 0) {
        // Return to bin for reuse
        impl_->bins[bin_size].push_back(ptr);
    } else {
        // Too large, return to upstream
        impl_->upstream->deallocate(ptr, actual_size, alignment);
    }
}

bool BinningMemoryResource::do_is_equal(const MemoryResource& other) const noexcept {
    return this == &other;
}

BinningMemoryResource::Statistics BinningMemoryResource::get_statistics() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Statistics stats;
    stats.num_bins = impl_->bins.size();
    stats.cache_hits = impl_->cache_hits;
    stats.cache_misses = impl_->cache_misses;
    stats.cached_bytes = 0;
    for (const auto& [size, ptrs] : impl_->bins) {
        stats.cached_bytes += size * ptrs.size();
    }
    return stats;
}

ResourceStats BinningMemoryResource::get_stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    ResourceStats result;
    result.allocation_count = impl_->alloc_sizes.size();
    result.allocated_bytes = 0;
    for (const auto& [ptr, size] : impl_->alloc_sizes) {
        result.allocated_bytes += size;
    }
    result.peak_allocated_bytes = impl_->peak_allocated;
    return result;
}

size_t BinningMemoryResource::min_bin_size() const {
    return impl_->min_bin_size;
}

size_t BinningMemoryResource::max_bin_size() const {
    return impl_->max_bin_size;
}

size_t BinningMemoryResource::bin_size_for(size_t bytes) const {
    return impl_->compute_bin_size(bytes);
}

void BinningMemoryResource::clear_cache() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto& [size, ptrs] : impl_->bins) {
        for (void* ptr : ptrs) {
            impl_->upstream->deallocate(ptr, size, 256);
        }
        ptrs.clear();
    }
    impl_->bins.clear();
}

} // namespace GpuRuntime
