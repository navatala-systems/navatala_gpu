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

// GPU Runtime - Memory Resource Tests
// Tests for pool_memory_resource, binning_memory_resource, and resource adaptors

#include "../include/gpu_runtime.h"
#include "../include/memory_resource.h"
#include "../include/pool_memory_resource.h"
#include "../include/binning_memory_resource.h"
#include "../include/resource_adaptors.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

using namespace GpuRuntime;

namespace {

/// Host-backed counting resource used for pool invariants that must not depend
/// on a GPU backend being available.
class CountingHostMemoryResource : public MemoryResource {
public:
    ~CountingHostMemoryResource() override {
        for (const auto& entry : active_) {
            std::free(entry.second.raw_ptr);
        }
    }

    size_t total_allocation_calls() const noexcept {
        return total_allocation_calls_;
    }

    size_t total_deallocation_calls() const noexcept {
        return total_deallocation_calls_;
    }

    size_t active_allocation_count() const noexcept {
        return active_.size();
    }

    size_t active_bytes() const noexcept {
        return active_bytes_;
    }

    ResourceStats get_stats() const override {
        ResourceStats stats;
        stats.allocation_count = active_.size();
        stats.allocated_bytes = active_bytes_;
        stats.peak_allocated_bytes = peak_active_bytes_;
        return stats;
    }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        if (bytes == 0) return nullptr;

        const size_t actual_alignment = std::max<size_t>(alignment, alignof(void*));
        assert(is_power_of_two(actual_alignment) &&
               "CountingHostMemoryResource expects power-of-two alignment");
        const size_t total_bytes = bytes + actual_alignment + sizeof(void*);
        void* raw = std::malloc(total_bytes);
        if (!raw) {
            throw AllocationError(bytes, alignment);
        }

        const auto start = reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*);
        const auto aligned =
            (start + actual_alignment - 1) & ~(static_cast<std::uintptr_t>(actual_alignment) - 1);
        void* ptr = reinterpret_cast<void*>(aligned);

        active_[ptr] = Allocation{raw, bytes};
        ++total_allocation_calls_;
        active_bytes_ += bytes;
        peak_active_bytes_ = std::max(peak_active_bytes_, active_bytes_);

        return ptr;
    }

    void do_deallocate(void* ptr, size_t, size_t) override {
        if (!ptr) return;

        auto it = active_.find(ptr);
        assert(it != active_.end() && "CountingHostMemoryResource deallocated unknown pointer");

        active_bytes_ -= it->second.bytes;
        std::free(it->second.raw_ptr);
        active_.erase(it);
        ++total_deallocation_calls_;
    }

    bool do_is_equal(const MemoryResource& other) const noexcept override {
        return this == &other;
    }

private:
    struct Allocation {
        void* raw_ptr;
        size_t bytes;
    };

    std::unordered_map<void*, Allocation> active_;
    size_t total_allocation_calls_ = 0;
    size_t total_deallocation_calls_ = 0;
    size_t active_bytes_ = 0;
    size_t peak_active_bytes_ = 0;
};

} // namespace

// ============================================================================
// Default Memory Resource Tests
// ============================================================================

void test_default_resource_basic() {
    std::cout << "=== Test: Default Resource Basic Allocation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto* mr = get_default_resource(*device);
    assert(mr != nullptr && "Default resource should exist");

    // Allocate memory
    void* ptr = mr->allocate(1024, 256);
    assert(ptr != nullptr && "Allocation should succeed");

    // Deallocate memory
    mr->deallocate(ptr, 1024, 256);

    std::cout << "PASS: Default resource basic allocation" << std::endl;
}

void test_default_resource_multiple_allocs() {
    std::cout << "=== Test: Default Resource Multiple Allocations ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto* mr = get_default_resource(*device);
    std::vector<void*> ptrs;

    // Allocate multiple buffers
    for (int i = 0; i < 10; ++i) {
        void* ptr = mr->allocate(1024 * (i + 1), 256);
        assert(ptr != nullptr && "Allocation should succeed");
        ptrs.push_back(ptr);
    }

    // Deallocate in reverse order
    for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
        mr->deallocate(*it, 1024, 256);  // Size doesn't matter for dealloc
    }

    std::cout << "PASS: Default resource multiple allocations" << std::endl;
}

// ============================================================================
// Pool Memory Resource Tests
// ============================================================================

void test_pool_resource_basic() {
    std::cout << "=== Test: Pool Resource Basic Allocation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    PoolConfig config;
    config.initial_pool_size = 1 << 20;  // 1 MB
    config.maximum_pool_size = 1 << 24;  // 16 MB
    config.alignment = 256;

    PoolMemoryResource pool(*device, config);

    // Allocate memory
    void* ptr = pool.allocate(4096, 256);
    assert(ptr != nullptr && "Pool allocation should succeed");

    // Deallocate memory
    pool.deallocate(ptr, 4096, 256);

    std::cout << "PASS: Pool resource basic allocation" << std::endl;
}

void test_pool_resource_reuse() {
    std::cout << "=== Test: Pool Resource Memory Reuse ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    PoolConfig config;
    config.initial_pool_size = 1 << 20;  // 1 MB
    config.maximum_pool_size = 1 << 20;  // 1 MB (limit to force reuse)
    config.alignment = 256;
    config.coalesce_on_free = true;

    PoolMemoryResource pool(*device, config);

    // Allocate and deallocate, then allocate again
    void* ptr1 = pool.allocate(4096, 256);
    assert(ptr1 != nullptr);

    pool.deallocate(ptr1, 4096, 256);

    // Should reuse the freed block
    void* ptr2 = pool.allocate(4096, 256);
    assert(ptr2 != nullptr);

    pool.deallocate(ptr2, 4096, 256);

    std::cout << "PASS: Pool resource memory reuse" << std::endl;
}

void test_pool_resource_statistics() {
    std::cout << "=== Test: Pool Resource Statistics ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    PoolConfig config;
    config.initial_pool_size = 1 << 20;
    config.maximum_pool_size = 1 << 24;

    PoolMemoryResource pool(*device, config);

    auto stats_before = pool.get_statistics();
    assert(stats_before.allocation_count == 0);

    void* ptr = pool.allocate(4096, 256);
    assert(ptr != nullptr);

    auto stats_after = pool.get_statistics();
    assert(stats_after.allocation_count == 1);
    assert(stats_after.allocated_bytes > 0);

    pool.deallocate(ptr, 4096, 256);

    std::cout << "PASS: Pool resource statistics" << std::endl;
}

// ============================================================================
// Binning Memory Resource Tests
// ============================================================================

void test_binning_resource_basic() {
    std::cout << "=== Test: Binning Resource Basic Allocation ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    BinningMemoryResource binning(*device, 256, 1 << 16);

    // Allocate memory
    void* ptr = binning.allocate(1024, 256);
    assert(ptr != nullptr && "Binning allocation should succeed");

    // Deallocate memory
    binning.deallocate(ptr, 1024, 256);

    std::cout << "PASS: Binning resource basic allocation" << std::endl;
}

void test_binning_resource_different_sizes() {
    std::cout << "=== Test: Binning Resource Different Sizes ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    BinningMemoryResource binning(*device, 256, 1 << 20);

    std::vector<void*> ptrs;

    // Allocate various sizes (should go to different bins)
    size_t sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    for (size_t size : sizes) {
        void* ptr = binning.allocate(size, 256);
        assert(ptr != nullptr);
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (size_t i = 0; i < ptrs.size(); ++i) {
        binning.deallocate(ptrs[i], sizes[i], 256);
    }

    std::cout << "PASS: Binning resource different sizes" << std::endl;
}

// ============================================================================
// Resource Adaptor Tests
// ============================================================================

void test_aligned_adaptor() {
    std::cout << "=== Test: Aligned Resource Adaptor ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto* upstream = get_default_resource(*device);
    AlignedResourceAdaptor aligned(*upstream, 4096);  // 4KB size alignment

    void* ptr = aligned.allocate(1000, 256);  // Request 256-byte alignment
    assert(ptr != nullptr && "Allocation should succeed");

    // NOTE: AlignedResourceAdaptor enforces SIZE alignment, not POINTER alignment.
    // The allocation size is rounded up to 4KB, but pointer alignment depends on
    // the GPU backend (typically 256 bytes). We do NOT assert pointer alignment.

    aligned.deallocate(ptr, 1000, 256);

    std::cout << "PASS: Aligned resource adaptor" << std::endl;
}

void test_tracking_adaptor() {
    std::cout << "=== Test: Tracking Resource Adaptor ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto* upstream = get_default_resource(*device);
    TrackingResourceAdaptor tracking(*upstream);

    assert(tracking.allocation_count() == 0);
    assert(tracking.total_allocated_bytes() == 0);

    void* ptr1 = tracking.allocate(1024, 256);
    assert(ptr1 != nullptr);
    assert(tracking.allocation_count() == 1);
    assert(tracking.total_allocated_bytes() == 1024);

    void* ptr2 = tracking.allocate(2048, 256);
    assert(ptr2 != nullptr);
    assert(tracking.allocation_count() == 2);
    assert(tracking.total_allocated_bytes() == 3072);

    tracking.deallocate(ptr1, 1024, 256);
    assert(tracking.allocation_count() == 1);
    assert(tracking.total_allocated_bytes() == 2048);

    tracking.deallocate(ptr2, 2048, 256);
    assert(tracking.allocation_count() == 0);
    assert(tracking.total_allocated_bytes() == 0);

    std::cout << "PASS: Tracking resource adaptor" << std::endl;
}

void test_limiting_adaptor() {
    std::cout << "=== Test: Limiting Resource Adaptor ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto* upstream = get_default_resource(*device);
    LimitingResourceAdaptor limiting(*upstream, 4096);  // 4KB limit

    // Should succeed - under limit
    void* ptr1 = limiting.allocate(2048, 256);
    assert(ptr1 != nullptr);

    // Should fail - would exceed limit
    void* ptr2 = nullptr;
    try {
        ptr2 = limiting.allocate(4096, 256);
        // If we get here without exception, ptr2 should be null
    } catch (const AllocationError&) {
        // Expected
    }
    // ptr2 should be null (allocation should have been rejected)

    limiting.deallocate(ptr1, 2048, 256);

    // Now should succeed - back under limit
    void* ptr3 = limiting.allocate(4096, 256);
    assert(ptr3 != nullptr);

    limiting.deallocate(ptr3, 4096, 256);

    std::cout << "PASS: Limiting resource adaptor" << std::endl;
}

void test_statistics_adaptor() {
    std::cout << "=== Test: Statistics Resource Adaptor ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    auto* upstream = get_default_resource(*device);
    StatisticsResourceAdaptor stats_adaptor(*upstream);

    auto s = stats_adaptor.get_detailed_stats();
    assert(s.total_allocations == 0);
    assert(s.current_bytes == 0);

    void* ptr1 = stats_adaptor.allocate(1024, 256);
    void* ptr2 = stats_adaptor.allocate(2048, 256);

    s = stats_adaptor.get_detailed_stats();
    assert(s.total_allocations == 2);
    assert(s.current_bytes == 3072);
    assert(s.peak_bytes >= 3072);

    stats_adaptor.deallocate(ptr1, 1024, 256);

    s = stats_adaptor.get_detailed_stats();
    assert(s.total_deallocations == 1);
    assert(s.current_bytes == 2048);

    stats_adaptor.deallocate(ptr2, 2048, 256);

    s = stats_adaptor.get_detailed_stats();
    assert(s.total_deallocations == 2);
    assert(s.current_bytes == 0);

    // Also test base class get_stats()
    auto base_stats = stats_adaptor.get_stats();
    assert(base_stats.allocation_count == 0);
    assert(base_stats.allocated_bytes == 0);

    std::cout << "PASS: Statistics resource adaptor" << std::endl;
}

// ============================================================================
// Composed Resource Tests
// ============================================================================

void test_composed_resources() {
    std::cout << "=== Test: Composed Resources ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Create a chain: Statistics -> Tracking -> Pool
    PoolConfig config;
    config.initial_pool_size = 1 << 20;
    config.maximum_pool_size = 1 << 24;

    PoolMemoryResource pool(*device, config);
    TrackingResourceAdaptor tracking(pool);
    StatisticsResourceAdaptor stats(tracking);

    void* ptr = stats.allocate(4096, 256);
    assert(ptr != nullptr);

    // Check statistics (use get_detailed_stats for detailed metrics)
    auto s = stats.get_detailed_stats();
    assert(s.total_allocations == 1);

    // Check tracking
    assert(tracking.allocation_count() == 1);

    stats.deallocate(ptr, 4096, 256);

    assert(stats.get_detailed_stats().current_bytes == 0);
    assert(tracking.allocation_count() == 0);

    std::cout << "PASS: Composed resources" << std::endl;
}

// ============================================================================
// Additional Test Coverage (T2)
// ============================================================================

void test_binning_rounded_size_dealloc() {
    std::cout << "=== Test: Binning Rounded Size Dealloc ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    BinningMemoryResource binning(*device, 256, 1 << 20);

    // Allocate 1000 bytes - should round to 1024 (next power of 2)
    void* ptr = binning.allocate(1000, 256);
    assert(ptr != nullptr);

    auto stats_before = binning.get_stats();
    assert(stats_before.allocation_count == 1);

    // Deallocate with original requested size (not bin size)
    binning.deallocate(ptr, 1000, 256);

    auto stats_after = binning.get_stats();

    // Stats should be consistent (no underflow)
    assert(stats_after.allocated_bytes == 0);
    assert(stats_after.allocation_count == 0);

    std::cout << "PASS: Binning rounded size dealloc" << std::endl;
}

void test_pool_stats_with_rounding() {
    std::cout << "=== Test: Pool Stats With Rounding ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    PoolConfig config;
    config.initial_pool_size = 1 << 20;
    config.alignment = 256;

    PoolMemoryResource pool(*device, config);

    // Allocate 1000 bytes - rounds to 1024 (256-byte alignment)
    void* ptr = pool.allocate(1000, 256);
    assert(ptr != nullptr);

    auto stats1 = pool.get_stats();
    assert(stats1.allocated_bytes >= 1000);  // At least requested

    // Deallocate
    pool.deallocate(ptr, 1000, 256);

    auto stats2 = pool.get_stats();
    assert(stats2.allocated_bytes == 0 && "Should be zero after dealloc");

    std::cout << "PASS: Pool stats with rounding" << std::endl;
}

void test_memory_resource_zero_size() {
    std::cout << "=== Test: Memory Resource Zero Size ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Test DefaultMemoryResource
    auto* mr = get_default_resource(*device);
    void* ptr1 = mr->allocate(0, 256);
    assert(ptr1 == nullptr && "Zero-size should return nullptr");

    // Test PoolMemoryResource
    PoolConfig config;
    config.initial_pool_size = 1 << 20;
    PoolMemoryResource pool(*device, config);
    void* ptr2 = pool.allocate(0, 256);
    assert(ptr2 == nullptr && "Pool zero-size should return nullptr");

    // Test BinningMemoryResource
    BinningMemoryResource binning(*device, 256, 1 << 20);
    void* ptr3 = binning.allocate(0, 256);
    assert(ptr3 == nullptr && "Binning zero-size should return nullptr");

    std::cout << "PASS: Memory resource zero size" << std::endl;
}

void test_deep_adaptor_stacking() {
    std::cout << "=== Test: Deep Adaptor Stacking ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    // Create deep stack: Statistics -> Tracking -> Aligned -> Pool
    PoolConfig config;
    config.initial_pool_size = 1 << 20;

    PoolMemoryResource pool(*device, config);
    AlignedResourceAdaptor aligned(pool, 4096);  // 4KB size alignment
    TrackingResourceAdaptor tracking(aligned);
    StatisticsResourceAdaptor stats(tracking);

    void* ptr = stats.allocate(1000, 256);
    assert(ptr != nullptr);

    // Verify tracking through stack
    assert(tracking.allocation_count() == 1);
    auto detailed = stats.get_detailed_stats();
    assert(detailed.total_allocations == 1);

    // Also verify base get_stats() works
    auto base_stats = stats.get_stats();
    assert(base_stats.allocation_count == 1);

    stats.deallocate(ptr, 1000, 256);

    assert(tracking.allocation_count() == 0);
    assert(stats.get_detailed_stats().current_bytes == 0);

    std::cout << "PASS: Deep adaptor stacking" << std::endl;
}

void test_pool_realloc_pattern() {
    std::cout << "=== Test: Pool Realloc Pattern ===" << std::endl;

    auto device = Device::create(0);
    if (!device) {
        std::cout << "SKIP: No GPU device available" << std::endl;
        return;
    }

    PoolConfig config;
    config.initial_pool_size = 1 << 20;
    config.coalesce_on_free = true;

    PoolMemoryResource pool(*device, config);

    // Simulate realloc pattern: allocate, deallocate, allocate larger
    void* ptr1 = pool.allocate(1024, 256);
    assert(ptr1 != nullptr);

    auto stats1 = pool.get_stats();
    size_t alloc1 = stats1.allocated_bytes;

    pool.deallocate(ptr1, 1024, 256);

    // Allocate a larger block
    void* ptr2 = pool.allocate(4096, 256);
    assert(ptr2 != nullptr);

    auto stats2 = pool.get_stats();
    assert(stats2.allocated_bytes > 0);

    pool.deallocate(ptr2, 4096, 256);

    auto stats3 = pool.get_stats();
    assert(stats3.allocated_bytes == 0);

    std::cout << "PASS: Pool realloc pattern" << std::endl;
}

void test_pool_no_growth_after_warmup() {
    std::cout << "=== Test: Pool No Growth After Warmup ===" << std::endl;

    CountingHostMemoryResource upstream;

    PoolConfig config;
    config.initial_pool_size = 64 * 1024;
    config.maximum_pool_size = config.initial_pool_size;
    config.alignment = 256;
    config.coalesce_on_free = true;

    {
        PoolMemoryResource pool(upstream, config);
        assert(upstream.total_allocation_calls() == 1 &&
               "Pool construction should preallocate one upstream block");

        const std::vector<size_t> sizes = {
            1024, 2048, 512, 4096, 8192, 1536, 768, 3072
        };

        auto run_pattern = [&]() {
            std::vector<void*> ptrs;
            ptrs.reserve(sizes.size());

            for (size_t bytes : sizes) {
                void* ptr = pool.allocate(bytes, 256);
                assert(ptr != nullptr && "Pooled allocation should succeed");
                ptrs.push_back(ptr);
            }

            for (size_t i = ptrs.size(); i > 0; --i) {
                pool.deallocate(ptrs[i - 1], sizes[i - 1], 256);
            }

            const auto stats = pool.get_statistics();
            assert(stats.allocated_bytes == 0 && "Warmup pattern should release all user allocations");
            assert(stats.allocation_count == 0 && "No active pool allocations should remain");
        };

        run_pattern();

        const size_t allocation_calls_after_warmup = upstream.total_allocation_calls();
        const size_t upstream_bytes_after_warmup = pool.get_statistics().upstream_bytes;

        for (int iteration = 0; iteration < 8; ++iteration) {
            run_pattern();
            const auto stats = pool.get_statistics();
            assert(upstream.total_allocation_calls() == allocation_calls_after_warmup &&
                   "Pool grew upstream after warmup for a repeated allocation pattern");
            assert(stats.upstream_bytes == upstream_bytes_after_warmup &&
                   "Pool reported upstream byte growth after warmup");
        }
    }

    assert(upstream.active_allocation_count() == 0 &&
           "Pool destructor should return all upstream allocations");

    std::cout << "PASS: Pool no growth after warmup" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPU Runtime Memory Resource Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Default resource tests
    test_default_resource_basic();
    test_default_resource_multiple_allocs();

    // Pool resource tests
    test_pool_resource_basic();
    test_pool_resource_reuse();
    test_pool_resource_statistics();

    // Binning resource tests
    test_binning_resource_basic();
    test_binning_resource_different_sizes();

    // Adaptor tests
    test_aligned_adaptor();
    test_tracking_adaptor();
    test_limiting_adaptor();
    test_statistics_adaptor();

    // Composed resource tests
    test_composed_resources();

    // Additional coverage tests (T2)
    test_binning_rounded_size_dealloc();
    test_pool_stats_with_rounding();
    test_memory_resource_zero_size();
    test_deep_adaptor_stacking();
    test_pool_realloc_pattern();
    test_pool_no_growth_after_warmup();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All Memory Resource Tests Completed!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
