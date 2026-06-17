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

// Resource Adaptors - Wrappers that add behavior to memory resources
// Part of the RMM formalization effort
//
// Adaptors wrap an upstream resource to add functionality:
// - AlignedResourceAdaptor: Enforces minimum alignment
// - TrackingResourceAdaptor: Tracks all allocations
// - LimitingResourceAdaptor: Enforces size limit
// - StatisticsResourceAdaptor: Collects statistics
//
// ## Bytes Tracking Convention
//
// Adaptors track **requested bytes** (what the caller passed), not actual/aligned bytes.
// This is intentional:
// - Adaptors wrap upstream resources and see only what callers request
// - Actual alignment/rounding happens in concrete implementations (Pool, Binning)
// - Statistics from adaptors reflect "requested" workload, not physical memory
//
// When composing resources, understand that byte counts may differ at each layer:
// - Caller requests 1000 bytes
// - TrackingAdaptor records 1000 bytes
// - PoolMemoryResource allocates 1024 bytes (aligned)

#pragma once

#include "memory_resource.h"
#include <memory>
#include <cstddef>
#include <atomic>
#include <mutex>

namespace GpuRuntime {

/// Wraps a resource to enforce minimum alignment.
///
/// All allocations are rounded up to the specified alignment.
/// This is useful when the upstream resource doesn't guarantee
/// sufficient alignment for certain operations (e.g., vectorized loads).
///
/// Thread-safe: Same as upstream resource.
class AlignedResourceAdaptor : public MemoryResource {
public:
    /// Construct adaptor.
    /// @param upstream Upstream resource (must outlive this adaptor)
    /// @param alignment Minimum alignment to enforce (must be power of 2)
    AlignedResourceAdaptor(MemoryResource& upstream, size_t alignment);

    ~AlignedResourceAdaptor() override;

    /// Get the enforced alignment
    size_t alignment() const noexcept { return alignment_; }

    /// Get the upstream resource
    MemoryResource& upstream() noexcept { return upstream_; }
    const MemoryResource& upstream() const noexcept { return upstream_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const MemoryResource& other) const noexcept override;

private:
    MemoryResource& upstream_;
    size_t alignment_;
};

/// Wraps a resource to track all allocations.
///
/// Maintains a record of all active allocations, including size and
/// optional debug information. Useful for debugging memory leaks.
///
/// Thread-safe: Yes (protected by internal mutex).
class TrackingResourceAdaptor : public MemoryResource {
public:
    /// Construct adaptor.
    /// @param upstream Upstream resource (must outlive this adaptor)
    explicit TrackingResourceAdaptor(MemoryResource& upstream);

    ~TrackingResourceAdaptor() override;

    // Non-copyable
    TrackingResourceAdaptor(const TrackingResourceAdaptor&) = delete;
    TrackingResourceAdaptor& operator=(const TrackingResourceAdaptor&) = delete;

    /// Get number of active allocations
    size_t allocation_count() const;

    /// Get total bytes currently allocated
    size_t total_allocated_bytes() const;

    /// Get peak bytes ever allocated
    size_t peak_allocated_bytes() const;

    /// Get base resource statistics (for polymorphic access)
    ResourceStats get_stats() const override;

    /// Log all active allocations (for debugging)
    void log_allocations() const;

    /// Check if any allocations are still active
    bool has_allocations() const { return allocation_count() > 0; }

    /// Get the upstream resource
    MemoryResource& upstream() noexcept { return upstream_; }
    const MemoryResource& upstream() const noexcept { return upstream_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const MemoryResource& other) const noexcept override;

private:
    struct Impl;
    MemoryResource& upstream_;
    std::unique_ptr<Impl> impl_;
};

/// Wraps a resource to enforce a size limit.
///
/// Allocations that would exceed the limit fail with AllocationError.
///
/// Thread-safe: Yes (uses atomic counters).
class LimitingResourceAdaptor : public MemoryResource {
public:
    /// Construct adaptor.
    /// @param upstream Upstream resource (must outlive this adaptor)
    /// @param limit Maximum total bytes that can be allocated
    LimitingResourceAdaptor(MemoryResource& upstream, size_t limit);

    ~LimitingResourceAdaptor() override;

    /// Get remaining allocation capacity
    size_t remaining() const noexcept;

    /// Get the limit
    size_t limit() const noexcept { return limit_; }

    /// Get current allocation
    size_t allocated() const noexcept { return allocated_.load(); }

    /// Get the upstream resource
    MemoryResource& upstream() noexcept { return upstream_; }
    const MemoryResource& upstream() const noexcept { return upstream_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const MemoryResource& other) const noexcept override;

private:
    MemoryResource& upstream_;
    size_t limit_;
    std::atomic<size_t> allocated_{0};
};

/// Wraps a resource to collect allocation statistics.
///
/// Tracks total allocations, deallocations, bytes allocated/deallocated,
/// peak usage, and current usage.
///
/// Thread-safe: Yes (uses atomic counters and mutex for complex stats).
class StatisticsResourceAdaptor : public MemoryResource {
public:
    /// Construct adaptor.
    /// @param upstream Upstream resource (must outlive this adaptor)
    explicit StatisticsResourceAdaptor(MemoryResource& upstream);

    ~StatisticsResourceAdaptor() override;

    /// Detailed statistics structure
    struct DetailedStats {
        /// Total number of allocations made
        size_t total_allocations;
        /// Total number of deallocations made
        size_t total_deallocations;
        /// Total bytes allocated (cumulative)
        size_t bytes_allocated;
        /// Total bytes deallocated (cumulative)
        size_t bytes_deallocated;
        /// Peak bytes allocated at any time
        size_t peak_bytes;
        /// Current bytes allocated
        size_t current_bytes;
    };

    /// Get detailed statistics
    DetailedStats get_detailed_stats() const;

    /// Get base resource statistics (for polymorphic access)
    ResourceStats get_stats() const override;

    /// Reset statistics (current_bytes is preserved)
    void reset_stats();

    /// Get the upstream resource
    MemoryResource& upstream() noexcept { return upstream_; }
    const MemoryResource& upstream() const noexcept { return upstream_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const MemoryResource& other) const noexcept override;

private:
    MemoryResource& upstream_;
    mutable std::mutex mutex_;
    DetailedStats stats_{};
};

} // namespace GpuRuntime
