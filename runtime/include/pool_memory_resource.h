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

// Pool Memory Resource - Pool-based memory allocation
// Part of the RMM formalization effort
//
// Maintains a free list of memory blocks to reduce allocation overhead.
// Coalesces adjacent free blocks to reduce fragmentation.
//
// ## Bytes Tracking Convention
//
// This resource tracks **actual allocated bytes** after alignment rounding:
// - Allocation sizes are rounded up to alignment boundaries
// - Statistics reflect real memory consumption, not requested sizes
// - Deallocation uses stored actual size, ignoring caller's bytes parameter
//
// This ensures accurate memory accounting for the physical pool.

#pragma once

#include "memory_resource.h"
#include <memory>
#include <cstddef>

namespace GpuRuntime {

/// Configuration for pool memory resource.
struct PoolConfig {
    /// Initial pool size (default: 64 MB)
    size_t initial_pool_size = 1ULL << 26;

    /// Maximum pool size (default: 1 GB)
    size_t maximum_pool_size = 1ULL << 30;

    /// Block alignment (default: 256 bytes)
    size_t alignment = 256;

    /// Whether to coalesce adjacent free blocks (default: true)
    bool coalesce_on_free = true;

    /// Maximum block size for pool management (larger allocations go to upstream)
    /// Default: 0 means no limit
    size_t max_block_size = 0;
};

/// Pool-based memory resource.
///
/// Maintains a free list of memory blocks. When an allocation is requested,
/// it first checks the free list. If no suitable block is found, it allocates
/// from the upstream resource.
///
/// When memory is deallocated, it is returned to the free list and optionally
/// coalesced with adjacent free blocks.
///
/// Thread-safe: Yes (protected by internal mutex).
///
/// Example usage:
///   PoolConfig config;
///   config.initial_pool_size = 128 * 1024 * 1024;  // 128 MB
///   PoolMemoryResource pool(device, config);
///   void* ptr = pool.allocate(1024);
///   // ... use ptr ...
///   pool.deallocate(ptr, 1024);
class PoolMemoryResource : public MemoryResource {
public:
    /// Construct with default configuration.
    explicit PoolMemoryResource(Device& device);

    /// Construct with custom configuration.
    PoolMemoryResource(Device& device, const PoolConfig& config);

    /// Construct with custom upstream resource.
    PoolMemoryResource(MemoryResource& upstream, const PoolConfig& config);

    ~PoolMemoryResource() override;

    // Non-copyable
    PoolMemoryResource(const PoolMemoryResource&) = delete;
    PoolMemoryResource& operator=(const PoolMemoryResource&) = delete;

    // Movable
    PoolMemoryResource(PoolMemoryResource&& other) noexcept;
    PoolMemoryResource& operator=(PoolMemoryResource&& other) noexcept;

    /// Get statistics
    struct Statistics {
        /// Current bytes allocated to user
        size_t allocated_bytes;
        /// Current bytes in free list
        size_t free_bytes;
        /// Peak bytes ever allocated
        size_t peak_allocated_bytes;
        /// Total bytes allocated from upstream
        size_t upstream_bytes;
        /// Number of active allocations
        size_t allocation_count;
        /// Number of free blocks
        size_t free_block_count;
    };

    Statistics get_statistics() const;

    /// Get base resource statistics (for polymorphic access)
    ResourceStats get_stats() const override;

    /// Get current allocated bytes
    size_t allocated_bytes() const;

    /// Get current free bytes in pool
    size_t free_bytes() const;

    /// Get peak allocated bytes
    size_t peak_allocated_bytes() const;

    /// Get configuration
    const PoolConfig& config() const;

protected:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const MemoryResource& other) const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace GpuRuntime
