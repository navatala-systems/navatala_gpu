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

// Pool Memory Resource Implementation
// Part of the RMM formalization effort

#include "pool_memory_resource.h"
#include <algorithm>
#include <mutex>
#include <vector>
#include <unordered_map>

namespace GpuRuntime {

struct PoolMemoryResource::Impl {
    Device* device = nullptr;
    MemoryResource* upstream = nullptr;
    bool owns_upstream = false;
    PoolConfig config;

    std::mutex mutex;

    // Free list: blocks available for reuse
    struct Block {
        void* ptr;
        size_t size;
    };
    std::vector<Block> free_blocks;

    // Allocated blocks: tracking for deallocation
    std::unordered_map<void*, size_t> allocated_blocks;

    // Statistics
    size_t total_allocated = 0;
    size_t peak_allocated = 0;
    size_t upstream_bytes = 0;

    ~Impl() {
        // Free all blocks back to upstream
        for (const auto& block : free_blocks) {
            if (upstream) {
                upstream->deallocate(block.ptr, block.size, config.alignment);
            }
        }
        // Note: allocated_blocks should be empty if all deallocations were made

        if (owns_upstream && upstream) {
            delete upstream;
        }
    }

    void* allocate_from_free_list(size_t bytes, size_t alignment) {
        // Best-fit allocation from free list
        auto best_it = free_blocks.end();
        size_t best_waste = SIZE_MAX;

        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
            if (it->size >= bytes) {
                size_t waste = it->size - bytes;
                if (waste < best_waste) {
                    best_waste = waste;
                    best_it = it;
                }
            }
        }

        if (best_it != free_blocks.end()) {
            void* ptr = best_it->ptr;
            size_t block_size = best_it->size;

            // Remove from free list
            free_blocks.erase(best_it);

            // If block is significantly larger, split it
            if (config.coalesce_on_free && block_size > bytes + config.alignment) {
                // Return excess to free list
                void* excess_ptr = static_cast<char*>(ptr) + bytes;
                size_t excess_size = block_size - bytes;
                free_blocks.push_back({excess_ptr, excess_size});
                block_size = bytes;
            }

            allocated_blocks[ptr] = block_size;
            total_allocated += block_size;  // FIX M2: Use actual block size, not requested
            peak_allocated = std::max(peak_allocated, total_allocated);

            return ptr;
        }

        return nullptr;
    }

    void* allocate_from_upstream(size_t bytes, size_t alignment) {
        if (!upstream) return nullptr;

        // Allocate from upstream
        void* ptr = upstream->allocate(bytes, alignment);
        if (ptr) {
            allocated_blocks[ptr] = bytes;
            total_allocated += bytes;
            peak_allocated = std::max(peak_allocated, total_allocated);
            upstream_bytes += bytes;
        }
        return ptr;
    }

    void coalesce_free_blocks() {
        if (!config.coalesce_on_free || free_blocks.size() < 2) return;

        // Sort by pointer address
        std::sort(free_blocks.begin(), free_blocks.end(),
                  [](const Block& a, const Block& b) { return a.ptr < b.ptr; });

        // Merge adjacent blocks
        std::vector<Block> merged;
        merged.reserve(free_blocks.size());

        for (const auto& block : free_blocks) {
            if (merged.empty()) {
                merged.push_back(block);
            } else {
                Block& last = merged.back();
                void* end_of_last = static_cast<char*>(last.ptr) + last.size;
                if (end_of_last == block.ptr) {
                    // Adjacent: merge
                    last.size += block.size;
                } else {
                    merged.push_back(block);
                }
            }
        }

        free_blocks = std::move(merged);
    }
};

PoolMemoryResource::PoolMemoryResource(Device& device)
    : PoolMemoryResource(device, PoolConfig{})
{
}

PoolMemoryResource::PoolMemoryResource(Device& device, const PoolConfig& config)
    : impl_(std::make_unique<Impl>())
{
    impl_->device = &device;
    impl_->config = config;
    impl_->upstream = new DefaultMemoryResource(device);
    impl_->owns_upstream = true;

    // Pre-allocate initial pool
    if (config.initial_pool_size > 0) {
        void* initial = impl_->upstream->allocate(config.initial_pool_size, config.alignment);
        if (initial) {
            impl_->free_blocks.push_back({initial, config.initial_pool_size});
            impl_->upstream_bytes = config.initial_pool_size;
        }
    }
}

PoolMemoryResource::PoolMemoryResource(MemoryResource& upstream, const PoolConfig& config)
    : impl_(std::make_unique<Impl>())
{
    impl_->upstream = &upstream;
    impl_->owns_upstream = false;
    impl_->config = config;

    // Pre-allocate initial pool
    if (config.initial_pool_size > 0) {
        void* initial = impl_->upstream->allocate(config.initial_pool_size, config.alignment);
        if (initial) {
            impl_->free_blocks.push_back({initial, config.initial_pool_size});
            impl_->upstream_bytes = config.initial_pool_size;
        }
    }
}

PoolMemoryResource::~PoolMemoryResource() = default;

PoolMemoryResource::PoolMemoryResource(PoolMemoryResource&& other) noexcept = default;
PoolMemoryResource& PoolMemoryResource::operator=(PoolMemoryResource&& other) noexcept = default;

void* PoolMemoryResource::do_allocate(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;

    // Round up to minimum alignment
    size_t actual_alignment = std::max(alignment, impl_->config.alignment);
    size_t aligned_bytes = align_up(bytes, actual_alignment);

    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Try free list first
    void* ptr = impl_->allocate_from_free_list(aligned_bytes, actual_alignment);
    if (ptr) return ptr;

    // Check if we'd exceed maximum pool size
    if (impl_->config.maximum_pool_size > 0 &&
        impl_->upstream_bytes + aligned_bytes > impl_->config.maximum_pool_size) {
        throw AllocationError(bytes, alignment);
    }

    // Allocate from upstream
    ptr = impl_->allocate_from_upstream(aligned_bytes, actual_alignment);
    if (!ptr) {
        throw AllocationError(bytes, alignment);
    }
    return ptr;
}

void PoolMemoryResource::do_deallocate(void* ptr, size_t bytes, size_t alignment) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto it = impl_->allocated_blocks.find(ptr);
    if (it == impl_->allocated_blocks.end()) {
        // Not our allocation, ignore
        return;
    }

    size_t actual_size = it->second;
    impl_->allocated_blocks.erase(it);
    impl_->total_allocated -= actual_size;  // FIX M2: Use stored actual size, not caller's bytes

    // Return to free list
    impl_->free_blocks.push_back({ptr, actual_size});

    // Coalesce if enabled
    impl_->coalesce_free_blocks();
}

bool PoolMemoryResource::do_is_equal(const MemoryResource& other) const noexcept {
    return this == &other;
}

PoolMemoryResource::Statistics PoolMemoryResource::get_statistics() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Statistics stats;
    stats.allocated_bytes = impl_->total_allocated;
    stats.peak_allocated_bytes = impl_->peak_allocated;
    stats.upstream_bytes = impl_->upstream_bytes;
    stats.allocation_count = impl_->allocated_blocks.size();
    stats.free_block_count = impl_->free_blocks.size();

    stats.free_bytes = 0;
    for (const auto& block : impl_->free_blocks) {
        stats.free_bytes += block.size;
    }

    return stats;
}

ResourceStats PoolMemoryResource::get_stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    ResourceStats result;
    result.allocation_count = impl_->allocated_blocks.size();
    result.allocated_bytes = impl_->total_allocated;
    result.peak_allocated_bytes = impl_->peak_allocated;
    return result;
}

size_t PoolMemoryResource::allocated_bytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->total_allocated;
}

size_t PoolMemoryResource::free_bytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    size_t total = 0;
    for (const auto& block : impl_->free_blocks) {
        total += block.size;
    }
    return total;
}

size_t PoolMemoryResource::peak_allocated_bytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->peak_allocated;
}

const PoolConfig& PoolMemoryResource::config() const {
    return impl_->config;
}

} // namespace GpuRuntime
